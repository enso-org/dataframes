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
import qualified Program.CMake    as CMake
import qualified Program.Curl     as Curl
import qualified Program.Ldd      as Ldd
import qualified Program.Patchelf as Patchelf
import qualified Program.MsBuild  as MsBuild
import qualified Program.SevenZip as SevenZip

depsArchiveUrl, packageBaseUrl :: String
depsArchiveUrl = "https://packages.luna-lang.org/dataframes/libs-dev-v140.7z"
packageBaseUrl = "https://packages.luna-lang.org/dataframes/windows-package-base.7z"

-- Retrieves a value of environment variable, returning the provided default
-- if the requested variable was not set.
getEnvDefault :: String -> String -> IO String
getEnvDefault variableName defaultValue =
    fromMaybe defaultValue <$> lookupEnv variableName

-- Copies to the given directory file under given path. Returns the copied-to path.
copyToDir :: FilePath -> FilePath -> IO FilePath
copyToDir destDir sourcePath = do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copy " ++ sourcePath ++ " to " ++ destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath
    return destPath

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
    let depsDirLocal = tempDir </> "deps"
    downloadAndUnpack7z depsArchiveUrl depsDirLocal
    setEnv "DATAFRAMES_DEPS_DIR" depsDirLocal

-- Copies subdirectory with all its contents between two directories
copyDirectory :: FilePath -> FilePath -> FilePath -> IO ()
copyDirectory sourceDirectory targetDirectory subdirectoryFilename = do
    let from = sourceDirectory </> subdirectoryFilename
    let to = targetDirectory </> subdirectoryFilename
    copyDirectoryRecursive silent from to

-- shortRelativePath requires normalised paths to work correctly.
-- this is helper function because we don't want to bother with checking
-- whether path is normalised everywhere else
relativeNormalisedPath :: FilePath -> FilePath -> FilePath
relativeNormalisedPath (normalise -> p1) (normalise -> p2) = shortRelativePath p1 p2

relativeRpath :: FilePath -> FilePath -> String
relativeRpath binaryPath dependenciesDir = "$ORIGIN" </> relativeNormalisedPath binaryPath dependenciesDir

setRpath :: FilePath -> [FilePath] -> IO ()
setRpath binaryPath depsDirectories = Patchelf.setRpath binaryPath $ relativeRpath (takeDirectory binaryPath) <$> depsDirectories

installBinary outputDirectory dependenciesDirectory sourcePath = do
    newBinaryPath <- copyToDir outputDirectory sourcePath
    setRpath newBinaryPath [dependenciesDirectory, outputDirectory]

installDependencyTo targetDirectory sourcePath = installBinary targetDirectory targetDirectory sourcePath

makePackage repoDir stagingDir = do
    -- Package
    let builtBinariesDir = repoDir </> "native_libs" </> "linux"
    let packageFile = "Dataframes-Linux-x64-v141" <.> "7z"
    let packageRoot = stagingDir </> "Dataframes"
    let packageBinaries = packageRoot </> "native_libs" </> "linux"
    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    mapM (copyDirectory repoDir packageRoot) dirsToCopy

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
    let testBuiltExe = stagingDir </> "build/DataframeHelperTests"
    builtDlls <- glob (builtBinariesDir </> "*.so")
    let builtBinaries = testBuiltExe : builtDlls

    when (null builtDlls) $ error "failed to found built .dll files"
    let libsDirectory = packageRoot </> "lib"
    dependencies <- Ldd.sharedDependenciesOfBinaries builtBinaries
    mapM (installDependencyTo libsDirectory) (filter isDependencyToPack dependencies)
    mapM (installBinary packageBinaries libsDirectory) builtBinaries

    copyDirectoryRecursive silent (pythonPrefix </> "lib/python3.7") (packageRoot </> "python-libs")
    pyCacheDirs <- glob $ packageRoot </> "python-libs" </> "**" </> "__pycache__"
    mapM removePathForcibly pyCacheDirs
    removePathForcibly $ packageRoot </> "python-libs" </> "config-3.7m-x86_64-linux-gnu"
    removePathForcibly $ packageRoot </> "python-libs" </> "test"

    SevenZip.pack [packageRoot] $ packageFile
    putStrLn $ "Packaging done, file saved to: " <> packageFile

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
    _ -> error $ "dataframesPackageName: not implemented: " <> show buildOS

buildProject :: FilePath -> FilePath -> IO DataframesBuildArtifacts
buildProject repoDir stagingDir = do
    let dataframesLibPath = repoDir </> "native_libs" </> "src"
    case buildOS of
        Windows -> do
            MsBuild.build $ dataframesLibPath </> "DataframeHelper.sln"

            let builtBinariesDir = dataframesLibPath </> "x64" </> "Release"
            builtDlls <- glob $ builtBinariesDir </> "*.dll"
            pure $ DataframesBuildArtifacts
                { dataframesBinaries = builtDlls
                , dataframesTests = [builtBinariesDir </> "DataframeHelperTests.exe"]
                }
        Linux -> do
            let buildDir = stagingDir </> "build"
            let cmakeVariables = CMake.OptionSetVariable <$> [ ("PYTHON_LIBRARY", pythonPrefix </> "lib/libpython3.7m.so")
                                                             , ("PYTHON_NUMPY_INCLUDE_DIR", pythonPrefix </>  "lib/python3.7/site-packages/numpy/core/include")]
            let options = CMake.OptionBuildType CMake.ReleaseWithDebInfo : cmakeVariables
            CMake.cmake buildDir dataframesLibPath options
            callProcessCwd buildDir "make" ["-j", "2"]

            let builtBinariesDir = repoDir </> "native_libs" </> nativeLibsOsDir
            builtDlls <- glob (builtBinariesDir </> "*.so")
            pure $ DataframesBuildArtifacts
                { dataframesBinaries = builtDlls
                , dataframesTests = [buildDir </> "build/DataframeHelperTests"]
                }

        _ -> undefined

main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        when (buildOS == Windows) (prepareEnvironment stagingDir)
        repoDir <- repoDir
        buildArtifacts <- buildProject repoDir stagingDir

        let packageRoot = stagingDir </> "Dataframes"
        let packageBinariesDir = packageRoot </> "native_libs" </> nativeLibsOsDir

        case buildOS of
            Windows -> do
                let builtDlls = dataframesBinaries buildArtifacts
                downloadAndUnpack7z packageBaseUrl packageBinariesDir
                when (null builtDlls) $ error "failed to found built .dll files"
                mapM (copyToDir packageBinariesDir) builtDlls
                let dirsToCopy = ["src", "visualizers", ".luna-package"]
                mapM (copyDirectory repoDir packageRoot) builtDlls
                SevenZip.pack [packageRoot] $ dataframesPackageName
                putStrLn $ "Packaging done, file saved to: " <> dataframesPackageName
            Linux -> do
                makePackage repoDir stagingDir

        -- Run tests
        -- The test executable must be placed in the package directory
        -- so all the dependencies are properly visible.
        -- The CWD must be repository though for test to properly find
        -- the data files.
        mapM (copyToDir packageBinariesDir) (dataframesTests buildArtifacts)
        withCurrentDirectory repoDir $ do
            mapM (flip callProcess ["--report_level=detailed"]) (dataframesTests buildArtifacts)


mainLinux :: IO ()
mainLinux = do
    withSystemTempDirectory "" $ \stagingDir -> do
        inCircleCI <- (==) (Just "true") <$> lookupEnv "CIRCLECI"
        let repoDir = if inCircleCI then "/root/project" else "/Dataframes"

        putStrLn $ "Repository path: " <> repoDir
        putStrLn $ "Staging path: " <> stagingDir

        let cmakeProjectDir = repoDir </> "native_libs" </> "src"
        let buildDir = stagingDir </> "build"

        -- Build
        let cmakeVariables = CMake.OptionSetVariable <$> [ ("PYTHON_LIBRARY", pythonPrefix </> "lib/libpython3.7m.so")
                                                         , ("PYTHON_NUMPY_INCLUDE_DIR", pythonPrefix </>  "lib/python3.7/site-packages/numpy/core/include")]
        let options = CMake.OptionBuildType CMake.ReleaseWithDebInfo : cmakeVariables
        CMake.cmake buildDir cmakeProjectDir options
        callProcessCwd buildDir "make" ["-j", "2"]
        -- callProcessCwd repoDir (buildDir </> "DataframeHelperTests") []

        -- Package
        makePackage repoDir stagingDir

        return ()

mainWin :: IO ()
mainWin = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        prepareEnvironment stagingDir

        repoDir <- getEnvDefault "APPVEYOR_BUILD_FOLDER" "C:\\dev\\Dataframes"
        let dataframesSolutionPath = repoDir </> "native_libs" </> "src" </> "DataframeHelper.sln"
        MsBuild.build dataframesSolutionPath

        let packageRoot = stagingDir </> "Dataframes"
        let packageBinaries = packageRoot </> "native_libs" </> "windows"
        let builtBinariesDir = repoDir </> "native_libs" </> "src" </> "x64" </> "Release"
        let packageFile = "Dataframes-Win-x64-v141" <.> "7z"
        downloadAndUnpack7z packageBaseUrl packageBinaries
        builtDlls <- glob (builtBinariesDir </> "*.dll")
        when (null builtDlls) $ error "failed to found built .dll files"
        mapM (copyToDir packageBinaries) builtDlls
        let dirsToCopy = ["src", "visualizers", ".luna-package"]
        mapM (copyDirectory repoDir packageRoot) dirsToCopy
        SevenZip.pack [packageRoot] $ packageFile
        putStrLn $ "Packaging done, file saved to: " <> packageFile

        -- Run tests
        -- The test executable must be placed in the package directory
        -- so all the dependencies are properly visible.
        -- The CWD must be repository though for test to properly find
        -- the data files.
        let testsExeSrc = builtBinariesDir </> "DataframeHelperTests.exe"
        let testsExeDst = packageBinaries </> takeFileName testsExeSrc
        copyFile testsExeSrc testsExeDst
        withCurrentDirectory repoDir $ do
            callProcess testsExeDst ["--report_level=detailed"]
