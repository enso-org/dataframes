module Paths where

import Prologue

import qualified Program.Git as Git

import Distribution.System (OS (Linux, OSX, Windows), buildOS)
import System.Directory   (getCurrentDirectory)
import System.Environment (getExecutablePath, lookupEnv)
import System.FilePath    (takeDirectory)
import Text.Printf        (printf)

-- | Gets path to the repository root. Basically this involves:
--
--   1. check @BUILD_SOURCESDIRECTORY@ environment variable.
--
--   2. check current executable path
--
--   3. check working directory
--
--   If nothing of the above is available, an error is raised.
repoDir :: MonadIO m => m FilePath
repoDir = liftIO $ lookupEnv "BUILD_SOURCESDIRECTORY" >>= \case
    Just path -> pure path
    Nothing -> do
        exePath <- getExecutablePath
        Git.repositoryRoot (takeDirectory exePath) >>= \case
            Just path -> pure path
            Nothing   -> do
                cwdDir <- getCurrentDirectory
                Git.repositoryRoot cwdDir >>= \case
                    Just path -> pure path
                    Nothing   -> error $ "cannot deduce repository root path, please define BUILD_SOURCESDIRECTORY"

defaultArchiveFormat :: String
defaultArchiveFormat = case buildOS of
    Windows -> "7z"
    Linux   -> "tar.gz"
    OSX     -> "tar.gz"
    _       -> error $ "defaultArchiveFormat: not implemented: " <> show buildOS

packageFileName :: String -> FilePath
packageFileName projectName = printf "%s-%s-%s.%s" projectName (systemName :: String) (archName :: String) defaultArchiveFormat where
    archName = "x64" 
    systemName = case buildOS of
        Windows -> "Win"
        Linux   -> "Linux"
        OSX     -> "macOS"
        _       -> error $ "packageFileName: not implemented: " <> show buildOS
