module Program.Curl where

import Prologue

import qualified Program as Program

import Program (Program)



------------------
-- === Curl === --
------------------

-- === Definition === --

data Curl
instance Program Curl where
    executableName = "curl"
    
-- === API === --

download 
    :: MonadIO m 
    => String  -- ^ URL to be fetched
    -> FilePath -- ^ Output file
    -> m ()
download url destPath = Program.call @Curl $ Program.format switches <> [url] 
    where
        switches = [FollowRedirect, OutputFile destPath] 
    

-- === Switches === --

data Switch
    = FollowRedirect
    | OutputFile FilePath 
    | Silent -- ^ Suppress progress indicator

instance Program.Argument Switch where
    format = \case
        FollowRedirect  -> ["-f"]
        OutputFile path -> ["-o", path]
        Silent          -> ["-s"]
