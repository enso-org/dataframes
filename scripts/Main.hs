import Control.Monad
import Control.Monad.Extra
import Data.List
import Data.Maybe
import Data.Monoid
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
import qualified Program.Otool    as Otool
import qualified Program.InstallNameTool as INT

import qualified Platform.OSX as OSX

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

-- Gets path to the local copy of the Dataframes repo
repoDir :: IO FilePath
repoDir = getEnvRequired "DATAFRAMES_REPO_PATH" -- TODO: should be able to deduce from this packaging executable location

-- Helper that does two things:
-- 1) use file extension to deduce compression method
-- 2) switch CWD so tar shall pack the folder at archive's root
--    (without maintaining directory's absolute path in archive)
packDirectory :: FilePath -> FilePath -> IO ()
packDirectory pathToPack outputArchive = do
    -- As we switch cwd, relative path to output might get affected.
    -- Let's store it as absolute path first.
    outputArchiveAbs <- makeAbsolute outputArchive
    withCurrentDirectory (takeDirectory pathToPack) $ do
        -- Input path must be relative though.
        pathToPackRel <- makeRelativeToCurrentDirectory pathToPack
        let tarPack =  Tar.pack [pathToPackRel] outputArchiveAbs
        case takeExtension outputArchive of
            ".7z"   -> SevenZip.pack [pathToPack] outputArchiveAbs
            ".gz"   -> tarPack Tar.GZIP
            ".bz2"  -> tarPack Tar.BZIP2
            ".xz"   -> tarPack Tar.XZ
            ".lzma" -> tarPack Tar.LZMA
            _       -> fail $ "packDirectory: cannot deduce compression algorithm from extension: " <> takeExtension outputArchive

dynamicLibraryExtension :: String
dynamicLibraryExtension = case buildOS of
    Windows -> "dll"
    Linux   -> "so"
    OSX     -> "dylib"
    _       -> error $ "dynamicLibraryExtension: not implemented: " <> show buildOS

nativeLibsOsDir :: String
nativeLibsOsDir = case buildOS of
    Windows -> "windows"
    Linux   -> "linux"
    OSX     -> "macos"
    _       -> error $ "nativeLibsOsDir: not implemented: " <> show buildOS

dataframesPackageName :: String
dataframesPackageName = case buildOS of
    Windows -> "Dataframes-Win-x64.7z"
    Linux   -> "Dataframes-Linux-x64.tar.gz"
    OSX     -> "Dataframes-macOS-x64.tar.gz"
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
        OSX  -> return ()
        _     ->
            error $ "not implemented: prepareEnvironment for buildOS == " <> show buildOS


data DataframesBuildArtifacts = DataframesBuildArtifacts
    { dataframesBinaries :: [FilePath]
    , dataframesTests :: [FilePath]
    } deriving (Eq, Show)

-- This function should be called only in a properly prepared build environment.
-- It builds the project and produces build artifacts.
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
        OSX -> do
            let buildDir = stagingDir </> "build"
            CMake.build buildDir dataframesLibPath [CMake.OptionBuildType CMake.ReleaseWithDebInfo]
        _ -> error $ "buildProject: not implemented: " <> show buildOS

    let builtBinariesDir = repoDir </> "native_libs" </> nativeLibsOsDir
    builtDlls <- glob $ builtBinariesDir </> "*" <.> dynamicLibraryExtension
    pure $ DataframesBuildArtifacts
        { dataframesBinaries = builtDlls
        , dataframesTests = [builtBinariesDir </> "DataframeHelperTests" <.> exeExtension]
        }

data DataframesPackageArtifacts = DataframesPackageArtifacts
    { dataframesPackageArchive :: FilePath
    , dataframesPackageDirectory :: FilePath
    } deriving (Eq, Show)

copyInPythonLibs :: FilePath -> FilePath -> IO ()
copyInPythonLibs pythonPrefix packageRoot = do
    pythonInPackage <- copyDirectory (pythonPrefix </> "lib") (packageRoot) "python3.7"
    pyCacheDirs <- glob $ pythonInPackage </> "**" </> "__pycache__"
    mapM removePathForcibly pyCacheDirs
    removePathForcibly $ pythonInPackage </> "config-3.7m-x86_64-linux-gnu"
    removePathForcibly $ pythonInPackage </> "test"

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
        _ -> do
            dependencies <- dependenciesToPackage builtDlls
            let libsDirectory = packageRoot </> "lib"
            mapM (Patchelf.installDependencyTo libsDirectory) dependencies
            mapM (Patchelf.installBinary packageBinariesDir libsDirectory) builtDlls

            -- Copy Python installation to the package and remove some parts that are heavy and not needed.
            pythonPrefix <- pythonPrefix
            copyInPythonLibs pythonPrefix packageRoot

    packDirectory packageRoot dataframesPackageName
    putStrLn $ "Packaging done, file saved to: " <> dataframesPackageName
    pure $ DataframesPackageArtifacts
        { dataframesPackageArchive = dataframesPackageName
        , dataframesPackageDirectory = packageRoot
        }

data OsxDependencyCategory = Local | Python | System deriving (Eq, Show)

categorizeDependency dependencyFullPath =
    if isInfixOf "/lib/python3.7/" dependencyFullPath
        then Python
        else if (isPrefixOf "/usr/lib/system/" dependencyFullPath) || (isPrefixOf "/System" dependencyFullPath)
            then System
            else if takeDirectory dependencyFullPath == "/usr/lib"
                then System
                else Local

isLocalDep dep = categorizeDependency dep == Local

-- Function installs Mach-O binary in a target binary folder.
-- install name shall be rewritten to contain only a filename
-- install names of direct local dependencies shall be rewritten, assuming they are in the same dir
installBinary :: FilePath -> FilePath -> IO ()
installBinary targetBinariesDir sourcePath = do
    -- putStrLn $ "installing " <> takeFileName sourcePath <> " to " <> targetBinariesDir
    destinationPath <- copyToDir targetBinariesDir sourcePath
    callProcess "chmod" ["777", destinationPath]
    INT.setInstallName destinationPath $ takeFileName destinationPath
    (filter isLocalDep -> directDeps) <- Otool.usedLibraries destinationPath
    flip mapM directDeps $ \installName -> do
        when (isLocalDep installName) $ do
            -- local dependencies of local dependencies are in the same folder as the current binary
            -- NOTE: in future, in multi-package world, there might be more folders
            INT.change destinationPath installName $ "@loader_path" </> takeFileName installName
    callProcess "chmod" ["555", destinationPath]

osxpack repoDir stagingDir buildArtifacts = do
    removeDirectoryRecursiveIfExists stagingDir
    createDirectoryIfMissing True stagingDir
    let packageRoot = stagingDir </> "Dataframes"
    let packageBinariesDir = packageRoot </> "native_libs" </> nativeLibsOsDir

    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    mapM (copyDirectory repoDir packageRoot) dirsToCopy

    let buildArtifactBinaries = dataframesBinaries buildArtifacts <>  dataframesTests buildArtifacts
    when (null buildArtifactBinaries) $ error "Build action have not build any binaries despite declaring success!"

    let testsBinary = head $ dataframesTests buildArtifacts

    allDeps1 <- OSX.getDependenciesOfDylibs $ dataframesBinaries buildArtifacts
    allDeps2 <- OSX.getDependenciesOfExecutable testsBinary ["--help"]
    let allDeps = nub $ allDeps1 <> allDeps2
    let resolveInstallName installName = find (\dep -> takeFileName installName == takeFileName dep) allDeps

    -- Place all "local depenencies" into the lib directory
    let trulyLocalDependency path = isLocalDep path && (not $ elem path buildArtifactBinaries)
    flip mapM (filter trulyLocalDependency allDeps) $ installBinary packageBinariesDir
    -- Place all artifact binaries in the destination directory
    flip mapM buildArtifactBinaries $ installBinary packageBinariesDir

    -- Copy Python installation to the package and remove some parts that are heavy and not needed.
    pythonPrefix <- pythonPrefix
    copyInPythonLibs pythonPrefix packageRoot

runTests :: FilePath -> DataframesBuildArtifacts -> DataframesPackageArtifacts -> IO ()
runTests repoDir buildArtifacts packageArtifacts = do
    -- The test executable must be placed in the package directory
    -- so all the dependencies are properly visible.
    -- The CWD must be repository though for test to properly find
    -- the data files.
    let packageDirBinaries = dataframesPackageDirectory packageArtifacts </> "native_libs" </> nativeLibsOsDir
    tests <- mapM (copyToDir packageDirBinaries) (dataframesTests buildArtifacts)
    withCurrentDirectory repoDir $ do
        mapM_ (flip callProcess ["--report_level=detailed"]) tests


main :: IO ()
main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        prepareEnvironment stagingDir
        repoDir <- repoDir
        buildArtifacts <- buildProject repoDir stagingDir
        packageArtifacts <- package repoDir stagingDir buildArtifacts
        runTests repoDir buildArtifacts packageArtifacts
