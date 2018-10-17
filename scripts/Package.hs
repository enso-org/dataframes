#!/usr/bin/env stack
-- stack --resolver lts-12.12 script --package Cabal,directory,extra,filepath,Glob,process,temporary

{-# LANGUAGE LambdaCase #-}

import Control.Monad
import Control.Monad.Extra
import Data.Maybe
import Data.Monoid
import Distribution.Simple.Program.Find
import Distribution.Simple.Utils
import Distribution.Verbosity
import System.Directory
import System.Environment
import System.FilePath
import System.FilePath.Glob
import System.IO.Temp
import System.Process

depsArchiveUrl :: String
depsArchiveUrl = "https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/libs-dev-v140.7z"

packageBaseUrl :: String
packageBaseUrl = "https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/windows-package-base.7z"

getEnvDefault :: String -> String -> IO String
getEnvDefault variableName defaultValue =
    fromMaybe defaultValue <$> lookupEnv variableName

findProgram :: ProgramSearchPath -> String -> IO (Maybe FilePath)
findProgram whereToSearch name = do
    fmap fst <$> findProgramOnSearchPath silent whereToSearch name

download :: String -> FilePath -> IO ()
download url destPath = callProcess "curl" ["-fSL", "-o", destPath, url]

find7z :: IO (Maybe FilePath)
find7z = do
    let default7zPath = "C:\\Program Files\\7-Zip"
    findProgram [ProgramSearchPathDefault, ProgramSearchPathDir default7zPath] "7z"

get7zPath :: IO FilePath
get7zPath = find7z >>= \case
    Just programPath -> return programPath
    Nothing          -> error errorMsg
    where errorMsg = "cannot find 7z, please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"

unpack7z :: FilePath -> FilePath -> IO ()
unpack7z archive outputDirectory = do
    programPath <- get7zPath
    callProcess programPath ["x", "-y", "-o" <> outputDirectory, archive]

pack7z :: [FilePath] -> FilePath -> IO ()
pack7z packedPaths outputArchivePath = do
    programPath <- get7zPath
    callProcess programPath $ ["a", "-y", outputArchivePath] <> packedPaths

buildWithMsBuild :: FilePath -> IO ()
buildWithMsBuild solutionPath = do
    let msbuildPath = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\amd64\\MSBuild.exe"
    callProcess msbuildPath ["/property:Configuration=Release", solutionPath]

copyToDir :: FilePath -> FilePath -> IO ()
copyToDir destDir sourcePath = do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copy " ++ sourcePath ++ " to " ++ destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath

pushArtifact :: FilePath -> IO ()
pushArtifact path = do
    callProcess "appveyor" ["PushArtifact", path]

-- Function downloads 7z to temp folder, so it doesn't leave any trash behind.
downloadAndUnpack7z :: FilePath -> FilePath -> IO ()
downloadAndUnpack7z archiveUrl targetDirectory = do
    withSystemTempDirectory "" $ \tmpDir -> do
        let archiveLocalPath = tmpDir </> takeFileName archiveUrl
        download archiveUrl archiveLocalPath
        unpack7z archiveLocalPath targetDirectory

-- We need to extract the package with dev libraries and set the environment
-- variable DATAFRAMES_DEPS_DIR so the MSBuild project recognizes it.
--
-- The package contains all dependencies except for Python (with numpy).
-- Python needs to be provided by CI environment and pointed to by `PythonDir`
-- environemt variable.
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

main :: IO ()
main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        -- let stagingDir = "C:\\Users\\mwurb\\AppData\\Local\\Temp\\-777f232250ff9e9c"
        prepareEnvironment stagingDir

        repoDir <- getEnvDefault "APPVEYOR_BUILD_FOLDER" "C:\\dev\\Dataframes"
        let dataframesSolutionPath = repoDir </> "native_libs" </> "src" </> "DataframeHelper.sln"
        buildWithMsBuild dataframesSolutionPath

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
        pack7z [packageRoot] $ packageFile
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
