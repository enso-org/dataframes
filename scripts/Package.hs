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
getEnvDefault variableName defaultValue = fromMaybe defaultValue <$> lookupEnv variableName

findProgram :: ProgramSearchPath -> String -> IO (Maybe FilePath)
findProgram whereToSearch name = do
    fmap fst <$> findProgramOnSearchPath silent whereToSearch name

download :: String -> FilePath -> IO ()
download url destPath = callProcess "curl" ["-fSL", "-o", destPath, url]

find7z :: IO (Maybe FilePath)
find7z = findProgram [ProgramSearchPathDefault, ProgramSearchPathDir default7zPath] "7z"
    where default7zPath = "C:\\Program Files\\7-Zip"

get7z :: IO FilePath
get7z = find7z >>= \case
    Just programPath -> return programPath
    Nothing          -> error "cannot find 7z, please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"

unpack7z :: FilePath -> FilePath -> IO ()
unpack7z archive outputDirectory = do
    programPath <- get7z
    callProcess programPath ["x", "-y", "-o" <> outputDirectory, archive]

pack7z :: [FilePath] -> FilePath -> IO ()
pack7z packedPaths outputArchivePath = do
    programPath <- get7z
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

main = do
    repoDir <- getEnvDefault "APPVEYOR_BUILD_FOLDER" "C:\\dev\\Dataframes"
    buildWithMsBuild (repoDir </> "native_libs" </> "src" </> "DataframeHelper.sln")
    withSystemTempDirectory "" $ \stagingDir -> do
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
        sequence $ copyToDir packageBinaries <$> builtExes
        copyDirectoryRecursive silent (repoDir </> "src") (packageRoot </> "src")
        copyDirectoryRecursive silent (repoDir </> "visualizers") (packageRoot </> "visualizers")
        pack7z [packageRoot] $ packageFile
        putStrLn $ "Packaging done, file saved to: " <> packageFile
        -- getLine