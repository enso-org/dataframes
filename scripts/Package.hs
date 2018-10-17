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
import System.Process
import System.FilePath.Glob
import System.IO.Temp

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

buildWithMsBuild solutionPath = do
    let msbuildPath = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\amd64\\MSBuild.exe"
    callProcess msbuildPath ["/property:Configuration=Release", solutionPath]

copyToDir destDir sourcePath = do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copy " ++ sourcePath ++ " to " ++ destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath

pushArtifact path = do
    callProcess "appveyor" ["PushArtifact", path]

-- We need to extract the package with dev libraries and set the environment
-- variable DATAFRAMES_DEPS_DIR so the MSBuild project recognizes it.
--
-- The package contains all dependencies except for Python (with numpy).
-- Python needs to be provided by CI environment and pointed to by `PythonDir`
-- environemt variable.
prepareEnvironment :: FilePath -> IO ()
prepareEnvironment tempDir = do
    let depsArchiveUrl = "https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/libs-dev-v140.7z"
    let depsArchiveLocal = tempDir </> "libs-dev-v140.7z"
    let depsDirLocal = tempDir </> "deps"
    download depsArchiveUrl depsArchiveLocal
    unpack7z depsArchiveLocal depsDirLocal
    setEnv "DATAFRAMES_DEPS_DIR" depsDirLocal

-- Copies subdirectory with all its contents between two directories
copyDirectory sourceDirectory targetDirectory subdirectoryFilename = do
    let from = sourceDirectory </> subdirectoryFilename
    let to = targetDirectory </> subdirectoryFilename
    copyDirectoryRecursive silent from to

main = do
    withSystemTempDirectory "" $ \stagingDir -> do
        prepareEnvironment stagingDir

        repoDir <- getEnvDefault "APPVEYOR_BUILD_FOLDER" "C:\\dev\\Dataframes"
        let dataframesSolutionPath = repoDir </> "native_libs" </> "src" </> "DataframeHelper.sln"
        buildWithMsBuild dataframesSolutionPath

        let packageRoot = stagingDir </> "Dataframes"
        let packageBinaries = packageRoot </> "native_libs" </> "windows"
        let packageArchive = stagingDir </> "package.7z"
        let builtBinariesDir = repoDir </> "native_libs" </> "src" </> "x64" </> "Release"
        let packageFile = "Dataframes-Win-x64-v141" <.> "7z"
        download packageBaseUrl packageArchive
        unpack7z packageArchive packageBinaries
        builtDlls <- glob (builtBinariesDir </> "*.dll")
        builtExes <- glob (builtBinariesDir </> "*.exe")
        when (null builtDlls) $ error "Missing built DLL!"
        when (null builtExes) $ error "Missing built EXE!"
        sequence $ copyToDir packageBinaries <$> builtDlls
        mapM (copyToDir packageBinaries) (builtDlls <> builtExes)
        let dirsToCopy = ["src", "visualizers", ".luna-package"]
        mapM (copyDirectory repoDir packageRoot) dirsToCopy
        pack7z [packageRoot] $ packageFile
        putStrLn $ "Packaging done, file saved to: " <> packageFile
        -- getLine
