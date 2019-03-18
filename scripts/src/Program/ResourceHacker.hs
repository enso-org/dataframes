module Program.ResourceHacker where

import Prologue

import qualified System.Process.Typed as Process
import qualified Program as Program

import Data.List (isInfixOf)

import Distribution.System (OS (Windows), buildOS)

data ResourceHacker
instance Program.Program ResourceHacker where
    defaultLocations = ["C:\\Program Files (x86)\\Resource Hacker" | buildOS == Windows]
    executableName = "ResourceHacker"
    
    -- | All calls shall use custom 'formatShellCommand' and use shell.
    proc program args = Process.shell $ formatShellCommand program args
    
-- Note: Resource Hacker seems to be extremely fragile in handling command 
-- arguments. For some reason wrapping its arguments in quotes (as does process
-- library) causes it to fail. Only paths can be quoted and it is necessary
-- when path contains spaces.
-- Because of that we call it by shell command formatted by function below.
formatShellCommand :: FilePath -> [String] -> String
formatShellCommand resHackerPath args = intercalate " " partsQuoted where
    partsQuoted = quoteIfNeeded <$> resHackerPath : args
    quoteIfNeeded p = if  " " `isInfixOf` p
        then "\"" <> p <> "\""
        else p
    
compileCmd :: (MonadIO m) => FilePath -> FilePath -> m ()
compileCmd rcPath resPath = Program.call @ResourceHacker $
    [ "-open", rcPath
    , "-save", resPath
    , "-action", "compile"
    , "-log", "CONSOLE"
    ]

addoverwriteCmd :: (MonadIO m) => FilePath -> FilePath -> m ()
addoverwriteCmd exePath resPath = Program.call @ResourceHacker $
    [ "-open", exePath
    , "-save", exePath
    , "-action", "addoverwrite"
    , "-resource", resPath
    , "-log", "CONSOLE"
    ]