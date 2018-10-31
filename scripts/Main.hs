import Control.Monad
import Control.Monad.Extra
import Data.Maybe
import Data.Monoid
import Distribution.Simple.Program.Find
import Distribution.Simple.Utils
import Distribution.System
import Distribution.Verbosity
import System.Directory
import System.Environment
import System.FilePath
import System.FilePath.Glob
import System.IO.Temp
import System.Process

import Program
import Utils

import qualified Program.CMake    as CMake
import qualified Program.Curl     as Curl
import qualified Program.Ldd      as Ldd
import qualified Program.Patchelf as Patchelf
import qualified Program.MsBuild  as MsBuild
import qualified Program.SevenZip as SevenZip

depsArchiveUrl, packageBaseUrl :: String
depsArchiveUrl = "https://packages.luna-lang.org/dataframes/libs-dev-v140.7z"
packageBaseUrl = "https://packages.luna-lang.org/dataframes/windows-package-base.7z"

-- Function downloads archive from given URL and extracts it to the target dir.
-- The archive is placed in temp folder, so function doesn't leave any trash behind.
downloadAndUnpack7z :: FilePath -> FilePath -> IO ()
downloadAndUnpack7z archiveUrl targetDirectory = do
    withSystemTempDirectory "" $ \tmpDir -> do
        let archiveLocalPath = tmpDir </> takeFileName archiveUrl
        Curl.download archiveUrl archiveLocalPath
        SevenZip.unpack archiveLocalPath targetDirectory

-- We need to extract the package with dev libraries and set the environment
-- variable DATAFRAMES_DEPS_DIR so the MSBuild project recognizes it.
--
-- The package contains all dependencies except for Python (with numpy).
-- Python needs to be provided by CI environment and pointed to by `PythonDir`
-- environment variable.
prepareEnvironment :: FilePath -> IO ()
prepareEnvironment tempDir = do
    case buildOS of
        Windows -> do
            let depsDirLocal = tempDir </> "deps"
            downloadAndUnpack7z depsArchiveUrl depsDirLocal
            setEnv "DATAFRAMES_DEPS_DIR" depsDirLocal

        _       -> return ()

installBinary outputDirectory dependenciesDirectory sourcePath = do
    newBinaryPath <- copyToDir outputDirectory sourcePath
    Patchelf.setRelativeRpath newBinaryPath [dependenciesDirectory, outputDirectory]

installDependencyTo targetDirectory sourcePath = installBinary targetDirectory targetDirectory sourcePath

repoDir :: IO FilePath
repoDir = do
    case buildOS of
        Windows -> getEnvDefault "APPVEYOR_BUILD_FOLDER" "C:\\dev\\Dataframes"
        Linux   -> getEnvDefault "DATAFRAMES_REPO_PATH" "/root/project"
        _       -> error $ "not implemented: repoDir for buildOS == " <> show buildOS

data DataframesBuildArtifacts = DataframesBuildArtifacts
    { dataframesBinaries :: [FilePath]
    , dataframesTests :: [FilePath]
    }
data DataframesPackageArtifacts = DataframesPackageArtifacts
    { dataframesPackageArchive :: FilePath
    , dataframesPackageDirectory :: FilePath
    }

dynamicLibraryExtension = case buildOS of
    Windows -> "dll"
    Linux -> "so"
    _ -> error $ "dynamicLibraryExtension: not implemented: " <> show buildOS

nativeLibsOsDir = case buildOS of
    Windows -> "windows"
    Linux -> "linux"
    _ -> error $ "nativeLibsOsDir: not implemented: " <> show buildOS

dataframesPackageName = case buildOS of
    Windows -> "Dataframes-Win-x64-v141" <.> "7z"
    Linux -> "Dataframes-Linux-x64" <.> "7z"
    _ -> error $ "dataframesPackageName: not implemented: " <> show buildOS

pythonPrefix = case buildOS of
    Linux -> "/python-dist"
    _ -> error $ "pythonPrefix: not implemented: " <> show buildOS


dependenciesToPackage :: [FilePath] -> IO [FilePath]
dependenciesToPackage binaries = do
    let libraryBlacklist = [
            "libX11", "libXext", "libXau", "libXdamage", "libXfixes", "libX11-xcb",
            "libXxf86vm", "libXdmcp", "libGL", "libdl", "libc", "librt", "libm", "libpthread",
            "libXcomposite",
            "libnvidia-tls", "libnvidia-glcore", "libnvidia-glsi",
            "libXrender", "libXi",
            "libdrm",
            "libutil",
            "libgbm", "libdbus-1",
            "libselinux",
            "ld-linux-x86-64"
            ]
    let isDependencyToPack path = notElem (dropExtensions $ takeFileName path) libraryBlacklist
    dependencies <- Ldd.sharedDependenciesOfBinaries binaries
    pure $ filter isDependencyToPack dependencies

buildProject :: FilePath -> FilePath -> IO DataframesBuildArtifacts
buildProject repoDir stagingDir = do
    let dataframesLibPath = repoDir </> "native_libs" </> "src"
    case buildOS of
        Windows -> do
            MsBuild.build $ dataframesLibPath </> "DataframeHelper.sln"
        Linux -> do
            let buildDir = stagingDir </> "build"
            let cmakeVariables =  [ ("PYTHON_LIBRARY", pythonPrefix </> "lib/libpython3.7m.so")
                                  , ("PYTHON_NUMPY_INCLUDE_DIR", pythonPrefix </>  "lib/python3.7/site-packages/numpy/core/include")]
            let options = CMake.OptionBuildType CMake.ReleaseWithDebInfo : (CMake.OptionSetVariable <$> cmakeVariables)
            CMake.build buildDir dataframesLibPath options

        _ -> undefined

    let builtBinariesDir = repoDir </> "native_libs" </> nativeLibsOsDir
    builtDlls <- glob $ builtBinariesDir </> "*" <.> dynamicLibraryExtension
    pure $ DataframesBuildArtifacts
        { dataframesBinaries = builtDlls
        , dataframesTests = [builtBinariesDir </> "DataframeHelperTests" <.> exeExtension]
        }

package :: FilePath -> FilePath -> DataframesBuildArtifacts -> IO DataframesPackageArtifacts
package repoDir stagingDir buildArtifacts = do
    let packageRoot = stagingDir </> "Dataframes"
    let packageBinariesDir = packageRoot </> "native_libs" </> nativeLibsOsDir

    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    mapM (copyDirectory repoDir packageRoot) dirsToCopy

    let builtDlls = dataframesBinaries buildArtifacts
    when (null builtDlls) $ error "Build action have not build any binaries despite declaring success!"

    case buildOS of
        Windows -> do
            downloadAndUnpack7z packageBaseUrl packageBinariesDir
            when (null builtDlls) $ error "failed to found built .dll files"
            mapM (copyToDir packageBinariesDir) builtDlls
            return ()
        Linux -> do
            dependencies <- dependenciesToPackage builtDlls
            let libsDirectory = packageRoot </> "lib"
            mapM (installDependencyTo libsDirectory) dependencies
            mapM (installBinary packageBinariesDir libsDirectory) builtDlls

            copyDirectoryRecursive silent (pythonPrefix </> "lib/python3.7") (packageRoot </> "python-libs")
            pyCacheDirs <- glob $ packageRoot </> "python-libs" </> "**" </> "__pycache__"
            mapM removePathForcibly pyCacheDirs
            removePathForcibly $ packageRoot </> "python-libs" </> "config-3.7m-x86_64-linux-gnu"
            removePathForcibly $ packageRoot </> "python-libs" </> "test"

    SevenZip.pack [packageRoot] dataframesPackageName
    putStrLn $ "Packaging done, file saved to: " <> dataframesPackageName
    pure $ DataframesPackageArtifacts
        { dataframesPackageArchive = dataframesPackageName
        , dataframesPackageDirectory = packageRoot
        }

runTests repoDir buildArtifacts packageArtifacts = do
    -- Run tests
    -- The test executable must be placed in the package directory
    -- so all the dependencies are properly visible.
    -- The CWD must be repository though for test to properly find
    -- the data files.
    let packageDirBinaries = dataframesPackageDirectory packageArtifacts </> "native_libs" </> nativeLibsOsDir
    tests <- mapM (copyToDir packageDirBinaries) (dataframesTests buildArtifacts)
    withCurrentDirectory repoDir $ do
        mapM (flip callProcess ["--report_level=detailed"]) tests

main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        prepareEnvironment stagingDir
        repoDir <- repoDir
        buildArtifacts <- buildProject repoDir stagingDir
        packageArtifacts <- package repoDir stagingDir buildArtifacts
        runTests repoDir buildArtifacts packageArtifacts
