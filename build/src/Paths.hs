module Paths where

import Prologue

import qualified Program.Git as Git

import Distribution.System (OS (Linux, OSX, Windows), buildOS)
import System.Directory   (getCurrentDirectory)
import System.Environment (getExecutablePath, lookupEnv)
import System.FilePath    (takeDirectory)
import Text.Printf        (printf)

-- | Attempts to deduce path to the repository root. Basically this involves
--   checking, if any of the following locations belongs to a git repository:
--
--   1. @BUILD_SOURCESDIRECTORY@ environment variable (automatically set
--      on Azure Pipelines).
--   2. current executable path
--   3. working directory
--
--   If none of the above is available, an error is raised.
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

-- | Default archive format for current platform. On Linux and macOS we prefer
--   @tar.gz@, because @tar@ program can typically be assumed to be present. On
--   Windows we prefer 7z, as it is easier to place on the end-user's machine.
defaultArchiveFormat :: String
defaultArchiveFormat = case buildOS of
    Windows -> "7z"
    Linux   -> "tar.gz"
    OSX     -> "tar.gz"
    _       -> error $ "defaultArchiveFormat: not implemented: " <> show buildOS

-- | Returns a platform-specific name for an archive with a package.
packageArchiveDefaultName :: String -> FilePath
packageArchiveDefaultName projectName = 
    let archName = "x64" 
        systemName = case buildOS of
            Windows -> "Win"
            Linux   -> "Linux"
            OSX     -> "macOS"
            _       -> error $ "packageFileName: not implemented: "
                             <> show buildOS
    in printf "%s-%s-%s.%s" 
        projectName 
        (systemName :: String) 
        (archName   :: String) 
        defaultArchiveFormat     where
