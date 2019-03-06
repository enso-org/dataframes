{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Program where

import Control.Monad.IO.Class
import Data.Maybe
import Data.Monoid
import Data.List
import Distribution.Simple.Program.Find
import Distribution.Verbosity
import System.Exit
import System.Process
import Text.Printf

class Program a where
    {-# MINIMAL executableName | executableNames #-}

    -- Path to directory where program executable might be found.
    defaultLocations :: [FilePath]
    defaultLocations = []

    executableName :: FilePath
    executableName = head $ executableNames @a

    executableNames :: [FilePath]
    executableNames = [executableName @a]

    lookupProgram :: (MonadIO m) => m (Maybe FilePath)
    lookupProgram = lookupExecutable (executableNames @a) (defaultLocations @a)

    notFoundError :: String
    notFoundError = "failed to find program " <> prettyNames <> ", " <> notFoundFixSuggestion @a
        where prettyNames = intercalate " nor " $ executableNames @a

    notFoundFixSuggestion :: String
    notFoundFixSuggestion = "please make sure it is visible in PATH"

    -- Returns absolute path to the program, throws if not found
    getProgram :: (MonadIO m) => m FilePath
    getProgram = liftIO $ fromMaybe (error $ notFoundError @a) <$> lookupProgram @a

    call :: (MonadIO m) => [String] -> m ()
    call args = do
        programPath <- getProgram @a
        liftIO $ callProcess programPath args

    callCwd :: (MonadIO m) => FilePath -> [String] -> m ()
    callCwd cwd args = do
        programPath <- getProgram @a
        callProcessCwd cwd programPath args

    readProgram :: (MonadIO m) => [String] -> m String
    readProgram args = do
        programPath <- getProgram @a
        liftIO $ readProcess programPath args ""

    -- Equivalent of System.Process `proc` function.
    prog :: (MonadIO m) => [String] -> m CreateProcess
    prog args = do 
        programPath <- getProgram @a
        pure $ proc programPath args
    
    -- Just like `prog` but also sets custom working directory.
    progCwd :: (MonadIO m) => FilePath -> [String] -> m CreateProcess
    progCwd cwdToUse args = do 
        programPath <- getProgram @a
        pure $ (proc programPath args) { cwd = Just cwdToUse }
    

readCreateProgram :: (MonadIO m) => CreateProcess -> m String
readCreateProgram args = do
    liftIO $ readCreateProcess args ""

readCreateProgramWithExitCode :: (MonadIO m) => CreateProcess -> m (ExitCode, String, String)
readCreateProgramWithExitCode args = do
    -- putStrLn $ show args
    liftIO $ readCreateProcessWithExitCode args ""

lookupExecutable :: (MonadIO m) => [FilePath] -> [FilePath] -> m (Maybe FilePath)
lookupExecutable [] _ = pure Nothing
lookupExecutable (exeName : exeNamesTail)  additionalDirs = do
    let locations = ProgramSearchPathDefault : (ProgramSearchPathDir <$> additionalDirs)
    fmap fst <$> (liftIO $ findProgramOnSearchPath silent locations exeName) >>= \case
        Just path -> pure $ Just path
        Nothing -> lookupExecutable exeNamesTail additionalDirs

runProcessWait :: (MonadIO m) => CreateProcess -> m ()
runProcessWait p = do
    (_, _, _, handle) <- liftIO $ createProcess p
    exitCode <- liftIO $ waitForProcess handle
    case exitCode of
        ExitSuccess -> return ()
        ExitFailure codeValue ->
            fail $ printf "runProcessWait failed: %s: exit code %d" (show $ cmdspec p) (codeValue)

callProcessCwd :: (MonadIO m) => FilePath -> FilePath -> [String] -> m ()
callProcessCwd cwd program args = liftIO $ runProcessWait $ (proc program args) {cwd = Just cwd}
