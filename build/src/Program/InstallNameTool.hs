module Program.InstallNameTool where

import Prologue

import qualified Program as Program
import qualified Utils   as Utils

import Program         (Program)
import System.FilePath (takeDirectory, (</>))

-------------------------------
-- === Install Name Tool === --
-------------------------------

-- === Definition === --

data InstallNameTool
instance Program InstallNameTool where
    executableName = "install_name_tool"


-- === API === --

-- | Generates rpath entry being a relative path (@loader_path-based) between
-- binary and given dependency. 
relativeLoaderPath 
    :: FilePath -- ^ Binary path
    -> FilePath -- ^ Dependency path
    -> String
relativeLoaderPath binaryPath dependency = loaderPathPlaceholder </> binaryToDep
    where
        loaderPathPlaceholder = "@loader_path"
        binaryDir = takeDirectory binaryPath
        binaryToDep = Utils.relativeNormalisedPath binaryDir dependency

-- | Changes the shared library dependency path (install name) in a Mach-O
-- executable
change 
    :: (MonadIO m) 
    => FilePath  -- ^ Patched binary
    -> FilePath  -- ^ Current dependency install name
    -> FilePath  -- ^ New dependency instlal name to be set
    -> m ()
change input = call input .: ChangeDependencyInstallName

changeRpath 
    :: (MonadIO m) 
    => FilePath  -- ^ Patched binary
    -> FilePath  -- ^ Current rpath value to be changed
    -> FilePath  -- ^ New rpath value to be set
    -> m ()
changeRpath input =  call input .: ChangeRpath

addRpath 
    :: (MonadIO m) 
    => FilePath -- ^ Patched binary
    -> FilePath -- ^ rpath to be added binary (must not be already present)
    -> m ()
addRpath input = call input . AddRpath

deleteRpath 
    :: (MonadIO m) 
    => FilePath  -- ^ Patched binary
    -> FilePath  -- ^ rpath value to be removed
    -> m ()
deleteRpath input = call input . DeleteRpath

-- | Sets the identification install name of the given Mach-O shared library.
setInstallName 
    :: (MonadIO m) 
    => FilePath -- ^ Patched binary
    -> FilePath -- ^ install name to be set
    -> m ()
setInstallName input new = Program.call @InstallNameTool ["-id", new, input]



---------------------
-- === Command === --
---------------------

-- === Definition === --
data Command
    = ChangeDependencyInstallName 
        { _oldName :: FilePath,  _newName :: FilePath }
    | ChangeRpath 
        { _oldRpath :: FilePath, _newRpath :: FilePath }
    | AddRpath 
        { _newRpath :: FilePath }
    | DeleteRpath 
        { _oldRpath :: FilePath }
    | SetInstallName 
        { _newName :: FilePath }

instance Program.Argument Command where
    format = \case 
        ChangeDependencyInstallName old new -> ["-change", old, new]
        ChangeRpath old new                 -> ["-rpath", old, new]
        AddRpath new                        -> ["-add_rpath", new]
        DeleteRpath old                     -> ["-delete_rpath", old]
        SetInstallName new                  -> ["-id", new]


-- === API === --

call :: MonadIO m => FilePath -> Command -> m ()
call targetFile cmd = Program.call @InstallNameTool args
    where args = Program.format cmd <> [targetFile]