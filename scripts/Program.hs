{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Program where

import Data.Maybe
import Data.Monoid
import Distribution.Simple.Program.Find
import Distribution.Verbosity
import System.Exit
import System.Process
import Text.Printf

lookupExecutable :: FilePath -> [FilePath] -> IO (Maybe FilePath)
lookupExecutable exeName additionalDirs = do
    let locations = ProgramSearchPathDefault : (ProgramSearchPathDir <$> additionalDirs)
    fmap fst <$> findProgramOnSearchPath silent locations exeName

class Program a where
    defaultLocations :: [FilePath]
    defaultLocations = []

    executableName :: FilePath

    lookupProgram :: IO (Maybe FilePath)
    lookupProgram = lookupExecutable (executableName @a) (defaultLocations @a)

    notFoundError :: String
    notFoundError = "failed to found program " <> executableName @a <> ", please make sure it is visible in PATH"

    getProgram :: IO FilePath
    getProgram = fromMaybe (error $ notFoundError @a) <$> lookupProgram @a

    call :: [String] -> IO ()
    call args = do
        programPath <- getProgram @a
        callProcess programPath args

    callCwd :: FilePath -> [String] -> IO ()
    callCwd cwd args = do
        programPath <- getProgram @a
        callProcessCwd cwd programPath args

    readProgram :: [String] -> IO String
    readProgram args = readProcess (executableName @a) args ""


runProcessWait :: CreateProcess -> IO ()
runProcessWait p = do
    -- print p
    (_, _, _, handle) <- createProcess p
    exitCode <- waitForProcess handle
    case exitCode of
        ExitSuccess -> return ()
        ExitFailure codeValue ->
            fail $ printf "runProcessWait failed: %s: exit code %d" (show $ cmdspec p) (codeValue)

callProcessCwd :: FilePath -> FilePath -> [String] -> IO ()
callProcessCwd cwd program args = runProcessWait $ (proc program args) {cwd = Just cwd}