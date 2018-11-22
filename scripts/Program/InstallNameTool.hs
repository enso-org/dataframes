module Program.InstallNameTool where

import Control.Monad
import Data.List

import Program

data InstallNameTool
instance Program InstallNameTool where
    executableName = "install_name_tool"

-- Changes the shared library dependency path (install name) in a Mach-O executable
change :: FilePath -> FilePath -> FilePath -> IO ()
change input old new = call @InstallNameTool ["-change", old, new, input]

changeRpath :: FilePath -> FilePath -> FilePath -> IO ()
changeRpath input old new = call @InstallNameTool ["-rpath", old, new, input]

addRpath :: FilePath -> FilePath -> IO ()
addRpath input new = call @InstallNameTool ["-add_rpath", new, input]

deleteRpath :: FilePath -> FilePath -> IO ()
deleteRpath input old = call @InstallNameTool ["-delete_rpath", old, input]

-- Sets the identification install name of the given Mach-O shared library.
setInstallName :: FilePath -> FilePath -> IO ()
setInstallName input new = call @InstallNameTool ["-id", new, input]
