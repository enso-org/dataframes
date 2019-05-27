module Platform.Utils.Windows where

import Prologue

import qualified Utils as Utils

import System.FilePath ((</>))

-- | Returns program files directory for 32-bit programs, typically @C:\\Program
--   Files (x86)@.
programFiles86 :: MonadIO m => m FilePath
programFiles86 = Utils.getEnvDefault "ProgramFiles(x86)" "C:\\Program Files (x86)"

-- | Obtains path to Windows directory, typically @C:\Windows@ 
systemRoot :: MonadIO m => m FilePath
systemRoot = Utils.getEnvDefault "SystemRoot" "C:\\Windows"

-- | A directory in the Windows operating system containing system executables
--   and libraries. Despite its name, it is native also on 64-bit Windows.
system32 :: MonadIO m => m FilePath
system32 = systemRoot <&> (</> "System32")
