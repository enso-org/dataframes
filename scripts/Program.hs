-- FIXME: DO NOT USE String, Use Text. Always.

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Program where

import Data.Maybe
import Data.Monoid
import Data.List
import Distribution.Simple.Program.Find
import Distribution.Verbosity
import System.Exit
import System.Process
import Text.Printf
import Control.Lens

-- Show (Argument a) => 
class Program a where
    {-# MINIMAL executableName | executableNames #-}

    data family Argument a
    
    type family Argument2 a
    type Argument2 a = String

    -- Path to directory where program executable might be found.
    defaultLocations :: [FilePath]
    defaultLocations = []

    executableName :: FilePath
    executableName = head $ executableNames @a

    executableNames :: [FilePath]
    executableNames = [executableName @a]

    lookupProgram :: IO (Maybe FilePath)
    lookupProgram = lookupExecutable (executableNames @a) (defaultLocations @a)

    -- FIXME: is there a reason anyone wants to override it?
    notFoundError :: String
    notFoundError = "failed to find program " <> prettyNames <> ", " <> notFoundFixSuggestion @a
        where prettyNames = intercalate " nor " $ executableNames @a

    notFoundFixSuggestion :: String
    notFoundFixSuggestion = "please make sure it is visible in PATH"

    -- Returns absolute path to the program, throws if not found
    getProgram :: IO FilePath
    getProgram = fromMaybe (error $ notFoundError @a) <$> lookupProgram @a

    callCwd :: FilePath -> [String] -> IO ()
    callCwd cwd args = do
        programPath <- getProgram @a
        callProcessCwd cwd programPath args

    readProgram :: [String] -> IO String
    readProgram args = do
        programPath <- getProgram @a
        readProcess programPath args ""


call :: forall a. Program a => [String] -> IO ()
call args = do
    programPath <- getProgram @a
    callProcess programPath args
    
lookupExecutable :: [FilePath] -> [FilePath] -> IO (Maybe FilePath)
lookupExecutable [] _ = pure Nothing
lookupExecutable (exeName : exeNamesTail)  additionalDirs = do
    let locations = ProgramSearchPathDefault : (ProgramSearchPathDir <$> additionalDirs)
    fmap fst <$> findProgramOnSearchPath silent locations exeName >>= \case
        Just path -> pure $ Just path
        Nothing -> lookupExecutable exeNamesTail additionalDirs

runProcessWait :: CreateProcess -> IO ()
runProcessWait p = do
    handle   <- view _4 <$> createProcess p
    exitCode <- waitForProcess handle
    case exitCode of
        ExitSuccess -> return ()
        ExitFailure codeValue ->
            fail $ printf "runProcessWait failed: %s: exit code %d" (show $ cmdspec p) (codeValue)

callProcessCwd :: FilePath -> FilePath -> [String] -> IO ()
callProcessCwd cwd program args = runProcessWait $ (proc program args) {cwd = Just cwd}
