{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Description : Common interface for finding and calling programs.

The module defines abstractions common for all programs that can be called through this library.
-}

module Program where

import Control.Monad.IO.Class
import Data.Maybe
import Data.Monoid
import Data.List
import Distribution.Simple.Program.Find
import Distribution.Simple.Utils (fromUTF8LBS)
import Distribution.Verbosity
import System.Exit
import System.Process.Typed
import Text.Printf

-- |A class defining an abstraction over a program. The purpose is mainly to provide some shared code (e.g. for looking up the program in @PATH@) and provide a number of convienence functions.
-- The whole thing can be seen as mostly a wrapper over typed-process library.
class Program a where
    {-# MINIMAL executableName | executableNames #-}

    -- |A set of paths where program might be found. Used only as a fallback (unless instance overwrites 'lookupProgram'), when program is not visible in standard system-spefici locations (typically the @PATH@ environment variable). Default implementation returns just an empty list.
    -- This function can be useful when the program has a well-known install location but usually is not added to @PATH@.
    defaultLocations :: [FilePath]
    defaultLocations = []

    -- |Name of the executable with this program. It is not necessary to include @.exe@ extension on Windows.
    executableName :: FilePath
    executableName = head $ executableNames @a

    -- |Names of executables running this program — this function should be used if the program can be called using several different names and it is not possible to rely reliably on single one being present. It is not necessary to include @.exe@ extension on Windows.
    executableNames :: [FilePath]
    executableNames = [executableName @a]

    -- |Returns an absolute path to the program executable. It is searching in @PATH@ environment variable, system-specific default locations and program specific locations (see 'defaultLocations').
    -- If the program cannot be found, silently returns 'Nothing'.
    lookupProgram :: (MonadIO m) => m (Maybe FilePath)
    lookupProgram = lookupExecutable (executableNames @a) (defaultLocations @a)

    -- |Error message that shall be raised on failure to find the program. 
    notFoundError :: String
    notFoundError = "failed to find program " <> prettyNames <> ", " <> notFoundFixSuggestion @a
        where prettyNames = intercalate " nor " $ executableNames @a

    -- |Text that should contain some actionable suggestion to the user on how to make program visible (e.g. program or system-specific installation guides). Will be included by default as part of 'notFoundError'.
    notFoundFixSuggestion :: String
    notFoundFixSuggestion = "please make sure it is visible in PATH"

    -- |Throwing variant of 'lookupProgram'.
    getProgram :: (MonadIO m) => m FilePath
    getProgram = liftIO $ fromMaybe (error $ notFoundError @a) <$> lookupProgram @a

    -- |Calls the program with given argument set. Waits for the process to finish. Throws if program cannot be started or if it returned non-zero exit code.
    call :: (MonadIO m) 
         => [String] -- ^Program arguments
         -> m ()
    call args = prog @a args >>= runProcess_

    -- |Just like 'call' but allows for setting a different working directory.
    callCwd :: (MonadIO m)
            => FilePath -- ^Working directory. NOTE: must point to existing directory, or the call will fail.
            -> [String] -- ^Program arguments.
            -> m ()
    callCwd cwd args = progCwd @a cwd args >>= runProcess_

    -- |Just like 'call' but returns the program's standard output.
    readProgram :: (MonadIO m) => [String] -> m String
    readProgram args = do
        -- TODO: no input?
        fromUTF8LBS <$> (readProcessStdout_ =<< prog @a args)

    -- |Equivalent of "System.Process.Typed"'s 'proc' function. Throws, if the program cannot be found.
    prog :: (MonadIO m) 
         => [String] -- ^Program arguments.
         -> m (ProcessConfig () () ())
    prog args = do 
        programPath <- getProgram @a
        pure $ proc programPath args
    
    -- | Just like 'prog' but also sets custom working directory.
    progCwd :: (MonadIO m) 
            => FilePath -- ^Working directory. NOTE: must point to existing directory, or the call will fail.
            -> [String] -- ^Program arguments.
            -> m (ProcessConfig () () ())
    progCwd cwdToUse args = do 
        (setWorkingDir cwdToUse) <$> prog @a args

-- |Function return an absolute path to the first executable name from the list that can be found.
lookupExecutable :: (MonadIO m) 
                 => [FilePath] -- ^List of executable names.
                 -> [FilePath] -- ^List of additional locations to be checked in addition to default ones.
                 -> m (Maybe FilePath)
lookupExecutable [] _ = pure Nothing
lookupExecutable (exeName : exeNamesTail)  additionalDirs = do
    let locations = ProgramSearchPathDefault : (ProgramSearchPathDir <$> additionalDirs)
    fmap fst <$> (liftIO $ findProgramOnSearchPath silent locations exeName) >>= \case
        Just path -> pure $ Just path
        Nothing -> lookupExecutable exeNamesTail additionalDirs

