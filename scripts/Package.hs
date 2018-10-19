#!/usr/bin/env stack
-- stack --resolver lts-12.12 script --package Cabal,directory,extra,filepath,Glob,process,temporary

{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

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

import Program
import qualified Program.Curl     as Curl
import qualified Program.MsBuild  as MsBuild
import qualified Program.SevenZip as SevenZip

depsArchiveUrl, packageBaseUrl :: String
depsArchiveUrl = "https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/libs-dev-v140.7z"
packageBaseUrl = "https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/windows-package-base.7z"

getEnvDefault :: String -> String -> IO String
getEnvDefault variableName defaultValue =
    fromMaybe defaultValue <$> lookupEnv variableName

copyToDir :: FilePath -> FilePath -> IO ()
copyToDir destDir sourcePath = do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copy " ++ sourcePath ++ " to " ++ destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath

-- Function downloads 7z to temp folder, so it doesn't leave any trash behind.
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
