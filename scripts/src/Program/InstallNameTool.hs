module Program.InstallNameTool where

import Control.Monad
import Control.Monad.IO.Class (MonadIO)
import Data.List
import System.FilePath

import Program
import Utils

data InstallNameTool
instance Program InstallNameTool where
    executableName = "install_name_tool"

relativeLoaderPath :: FilePath -> FilePath -> String
relativeLoaderPath binaryPath dependency = "@loader_path" </> relativeNormalisedPath (takeDirectory binaryPath) dependency

-- Changes the shared library dependency path (install name) in a Mach-O executable
change :: (MonadIO m) => FilePath -> FilePath -> FilePath -> m ()
change input old new = call @InstallNameTool ["-change", old, new, input]

changeRpath :: (MonadIO m) => FilePath -> FilePath -> FilePath -> m ()
changeRpath input old new = call @InstallNameTool ["-rpath", old, new, input]

addRpath :: (MonadIO m) => FilePath -> FilePath -> m ()
addRpath input new = call @InstallNameTool ["-add_rpath", new, input]

deleteRpath :: (MonadIO m) => FilePath -> FilePath -> m ()
deleteRpath input old = call @InstallNameTool ["-delete_rpath", old, input]

-- Sets the identification install name of the given Mach-O shared library.
setInstallName :: (MonadIO m) => FilePath -> FilePath -> m ()
setInstallName input new = call @InstallNameTool ["-id", new, input]
