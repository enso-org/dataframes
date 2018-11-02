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
import qualified Program.Tar      as Tar

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

-- This function purpose is to transform environment from its initial state
-- to the state where build step can be executed - so all dependencies and
-- other global state (like environment variable) that build script is using
-- must be set.
prepareEnvironment :: FilePath -> IO ()
prepareEnvironment tempDir = do
    case buildOS of
        Windows -> do
            -- We need to extract the package with dev libraries and set the environment
            -- variable DATAFRAMES_DEPS_DIR so the MSBuild project recognizes it.
            --
            -- The package contains all dependencies except for Python (with numpy).
            -- Python needs to be provided by CI environment and pointed to by `PythonDir`
            -- environment variable.
            let depsDirLocal = tempDir </> "deps"
            downloadAndUnpack7z depsArchiveUrl depsDirLocal
            setEnv "DATAFRAMES_DEPS_DIR" depsDirLocal
        Linux ->
            -- On Linux all dependencies are assumed to be already installed. Such is the case
            -- with the Docker image used to run Dataframes CI, and should be similarly with
            -- developer machines.
            return ()
        _     ->
            error $ "not implemented: prepareEnvironment for buildOS == " <> show buildOS

-- Gets path to the local copy of the Dataframes repo
repoDir :: IO FilePath
repoDir = getEnvRequired "DATAFRAMES_REPO_PATH" -- TODO: should be able to deduce from this packaging executable location

pack :: [FilePath] -> FilePath -> IO ()
pack pathsToPack outputArchive = case takeExtension outputArchive of
    "7z"   -> SevenZip.pack pathsToPack outputArchive
    "gz"   -> Tar.pack      pathsToPack outputArchive Tar.GZIP
    "bz2"  -> Tar.pack      pathsToPack outputArchive Tar.BZIP2
    "xz"   -> Tar.pack      pathsToPack outputArchive Tar.XZ
    "lzma" -> Tar.pack      pathsToPack outputArchive Tar.LZMA
    _      -> fail $ "cannot deduce compression algorithm from extension: " <> takeExtension outputArchive

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
    Linux   -> "so"
    _       -> error $ "dynamicLibraryExtension: not implemented: " <> show buildOS

nativeLibsOsDir = case buildOS of
    Windows -> "windows"
    Linux   -> "linux"
    _       -> error $ "nativeLibsOsDir: not implemented: " <> show buildOS

dataframesPackageName = case buildOS of
    Windows -> "Dataframes-Win-x64-v141.7z"
    Linux   -> "Dataframes-Linux-x64.tar.gz"
    _       -> error $ "dataframesPackageName: not implemented: " <> show buildOS

-- Path to directory with Python installation, should not contain other things
-- (that was passed as --prefix to Python's configure script)
pythonPrefix :: IO FilePath
pythonPrefix = getEnvRequired "PYTHON_PREFIX_PATH"

-- Linux-specific function.
-- The function takes paths to shared libraries and returns the ones that
-- should be distributed as the part of package. Libraries that are filtered
-- out should be assumed to be present at end-user's machine.
--
-- It is based on a list of libraries that should not be shipped, that was
-- written down based on previous experience. It mostly consists of low-level
-- libraries that are "just present" on "all" systems or libraries that
-- presumably would not work even if we shipped them.
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
    filter isDependencyToPack <$> Ldd.dependenciesOfBinaries binaries

buildProject :: FilePath -> FilePath -> IO DataframesBuildArtifacts
buildProject repoDir stagingDir = do
    let dataframesLibPath = repoDir </> "native_libs" </> "src"
    case buildOS of
        Windows -> do
            MsBuild.build $ dataframesLibPath </> "DataframeHelper.sln"
        Linux -> do
            pythonPrefix <- pythonPrefix
            let buildDir = stagingDir </> "build"
            let cmakeVariables =  [ ("PYTHON_LIBRARY",           pythonPrefix </> "lib/libpython3.7m.so")
                                  , ("PYTHON_NUMPY_INCLUDE_DIR", pythonPrefix </> "lib/python3.7/site-packages/numpy/core/include")]
            let options = CMake.OptionBuildType CMake.ReleaseWithDebInfo : (CMake.OptionSetVariable <$> cmakeVariables)
            CMake.build buildDir dataframesLibPath options
        _ -> error $ "buildProject: not implemented: " <> show buildOS

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
            mapM (Patchelf.installDependencyTo libsDirectory) dependencies
            mapM (Patchelf.installBinary packageBinariesDir libsDirectory) builtDlls

            -- Copy Python installation to the package and remove some parts that are heavy and not needed.
            pythonPrefix <- pythonPrefix
            copyDirectoryRecursive silent (pythonPrefix </> "lib/python3.7") (packageRoot </> "python-libs")
            pyCacheDirs <- glob $ packageRoot </> "python-libs" </> "**" </> "__pycache__"
            mapM removePathForcibly pyCacheDirs
            removePathForcibly $ packageRoot </> "python-libs" </> "config-3.7m-x86_64-linux-gnu"
            removePathForcibly $ packageRoot </> "python-libs" </> "test"

    pack [packageRoot] dataframesPackageName
    putStrLn $ "Packaging done, file saved to: " <> dataframesPackageName
    pure $ DataframesPackageArtifacts
        { dataframesPackageArchive = dataframesPackageName
        , dataframesPackageDirectory = packageRoot
        }

runTests :: FilePath -> DataframesBuildArtifacts -> DataframesPackageArtifacts -> IO ()
runTests repoDir buildArtifacts packageArtifacts = do
    -- Run tests
    -- The test executable must be placed in the package directory
    -- so all the dependencies are properly visible.
    -- The CWD must be repository though for test to properly find
    -- the data files.
    let packageDirBinaries = dataframesPackageDirectory packageArtifacts </> "native_libs" </> nativeLibsOsDir
    tests <- mapM (copyToDir packageDirBinaries) (dataframesTests buildArtifacts)
    withCurrentDirectory repoDir $ do
        mapM_ (flip callProcess ["--report_level=detailed"]) tests

main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        prepareEnvironment stagingDir
        repoDir <- repoDir
        buildArtifacts <- buildProject repoDir stagingDir
        packageArtifacts <- package repoDir stagingDir buildArtifacts
        runTests repoDir buildArtifacts packageArtifacts
