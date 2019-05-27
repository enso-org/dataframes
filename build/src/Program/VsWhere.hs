module Program.VsWhere where

    
import Prologue

import qualified Program as Program
import qualified Utils as Utils

import Program (Program)
import System.FilePath ((</>))

data VsWhere
instance Program VsWhere where
    -- vswhere is places by VS installer at default location, as specified by
    -- readme here: https://github.com/Microsoft/vswhere
    defaultLocations = do
        programFiles <- Utils.getEnvDefault "ProgramFiles(x86)" "C:\\Program Files (x86)"
        pure $ [programFiles </> "Microsoft Visual Studio\\Installer"]
    executableName = "vswhere"
