{-|
Description : Common interface for finding and calling programs.

The module defines abstractions common for all programs that can be called
through this library.
-}

module Program where

import Prologue hiding (log)

import qualified Distribution.Simple.Program.Find as Cabal
import qualified Distribution.Simple.Utils        as Cabal
import qualified Distribution.Verbosity           as Cabal
import qualified System.Process                   as ProcessUntyped
import qualified System.Process.Typed             as Process

import Data.ByteString.Lazy (ByteString)
import System.Process.Typed (ProcessConfig)
import Text.Printf          (printf)


type IOProgram m p = (Program p, MonadIO m)

-- | Values that can be used as an arguments when calling a process.
class Argument arg where
    {-# MINIMAL format | format' #-}
    -- | If datatype yields more than a single string as arguments, this method
    -- should be implemented.
    format :: arg -> [String]
    format arg = [format' arg]

    -- | If datatype yields a single string as an argument, this method should
    -- be implemented.
    --
    -- There is no other reason to call it ever, except for the default 'format'
    -- implementation.
    format' :: arg -> String
    format' arg = case format arg of
        head : _ -> head
        _        -> ""

instance Argument String where
    format arg = [arg]

instance {-# OVERLAPPABLE #-} Argument nested
      => Argument [nested] where
    format = concat . fmap format


-- | A class defining an abstraction over a program. The purpose is mainly to
--   provide some shared code (e.g. for looking up the program in @PATH@) and
--   provide a number of convienence functions. The whole thing can be seen as
--   mostly a wrapper over typed-process library.
class Program p where
    {-# MINIMAL executableName | executableNames #-}

    -- | A set of paths where program might be found. Used only as a fallback
    --   (unless instance overwrites 'lookupProgram'), when program is not
    --   visible in standard system-spefici locations (typically the @PATH@
    --   environment variable). Default implementation returns just an empty
    --   list. This function can be useful when the program has a well-known
    --   install location but usually is not added to @PATH@.
    defaultLocations :: MonadIO m => m [FilePath]
    defaultLocations = pure []

    -- | Name of the executable with this program. It is not necessary to
    --   include @.exe@ extension on Windows.
    executableName :: FilePath
    executableName = case executableNames @p of
        headName : _ -> headName
        []           -> error "`executableNames` must provide a non-empty list"

    -- |Names of executables running this program — this function should be used
    -- if the program can be called using several different names and it is not
    -- possible to rely reliably on single one being present. It is not
    -- necessary to include @.exe@ extension on Windows.
    executableNames :: [FilePath]
    executableNames = [executableName @p]

    -- |Returns an absolute path to the program executable. It is searching in
    -- @PATH@ environment variable, system-specific default locations and
    -- program specific locations (see 'defaultLocations'). If the program
    -- cannot be found, silently returns 'Nothing'.
    lookupProgram :: MonadIO m => m (Maybe FilePath)
    lookupProgram = do
        additionalLocationsToCheck <- defaultLocations @p
        lookupExecutable (executableNames @p) additionalLocationsToCheck

    -- |Error message that shall be raised on failure to find the program.
    notFoundError :: String
    notFoundError = printf "failed to find program %s, %s"
                        prettyNames suggestion where
        prettyNames = intercalate " nor " $ executableNames @p
        suggestion  = notFoundFixSuggestion @p

    -- |Text that should contain some actionable suggestion to the user on how
    -- to make program visible (e.g. program or system-specific installation
    -- guides). Will be included by default as part of 'notFoundError'.
    notFoundFixSuggestion :: String
    notFoundFixSuggestion = "please make sure it is visible in PATH"

    -- | Equivalent of "System.Process.Typed"'s 'System.Process.Typed.proc'
    -- function.
    proc :: FilePath -- ^Path to program
         -> [String] -- ^Program arguments.
         -> (ProcessConfig () () ())
    proc = Process.proc

    -- | Equivalent of "System.Process"'s 'System.Process.proc'
    -- function.
    proc' :: FilePath -- ^Path to program
         -> [String] -- ^Program arguments.
         -> ProcessUntyped.CreateProcess
    proc' = ProcessUntyped.proc


-- | Throwing variant of 'lookupProgram'.
get :: forall p m. IOProgram m p => m FilePath
get = liftIO $ fromJust (error $ notFoundError @p) <$> lookupProgram @p

-- | Finds program and prepares 'ProcessConfig'.
prog :: forall p m. IOProgram m p
        => [String] -- ^Program arguments.
        -> m (ProcessConfig () () ())
prog args = do
    programPath <- get @p
    pure $ proc @p programPath args

-- | Finds program and prepares 'System.Process.CreateProcess'.
prog' :: forall p m. IOProgram m p
    => [String] -- ^Program arguments.
    -> m ProcessUntyped.CreateProcess
prog' args = do
    programPath <- get @p
    pure $ proc' @p programPath args

-- |Calls the program with given argument set. Waits for the process to
-- finish. Throws if program cannot be started or if it returned non-zero
-- exit code.
call :: forall p m. IOProgram m p
    => [String] -- ^Program arguments
    -> m ()
call args = prog @p args >>= Process.runProcess_

-- |Just like 'call' but allows for setting a different working directory.
callCwd
    :: forall p m. IOProgram m p
    => FilePath -- ^ Working directory. NOTE: must point to existing directory,
                --   or the call will fail.
    -> [String] -- ^ Program arguments.
    -> m ()
callCwd cwd args = progCwd @p cwd args >>= Process.runProcess_

-- |Just like 'call' but returns the program's standard output.
read :: forall p m. IOProgram m p => [String] -> m String
read args = do
    -- TODO: no input?
    Cabal.fromUTF8LBS <$> (Process.readProcessStdout_ =<< prog @p args)

-- |Just like 'call' but returns the program's standard output and error output.
read' :: forall p m. IOProgram m p => [String] -> m (ByteString, ByteString)
read' args = Process.readProcess_ =<< prog @p args

-- | Just like 'prog' but also sets custom working directory.
progCwd
    :: forall p m. IOProgram m p
    => FilePath -- ^ Working directory. NOTE: must point to existing directory,
                --   or the call will fail.
    -> [String] -- ^ Program arguments.
    -> m (ProcessConfig () () ())
progCwd cwdToUse args = do
    (Process.setWorkingDir cwdToUse) <$> prog @p args

-- | Function return an absolute path to the first executable name from the list
--   that can be found.
lookupExecutable :: (MonadIO m)
                 => [FilePath] -- ^ List of executable names.
                 -> [FilePath] -- ^ List of fallback locations
                 -> m (Maybe FilePath)
lookupExecutable [] _ = pure Nothing
lookupExecutable (exeName : exeNamesTail) fallbackDirs = do
    let locations = Cabal.ProgramSearchPathDefault
                  : (Cabal.ProgramSearchPathDir <$> fallbackDirs)
    mLookupResult <- liftIO
        $ Cabal.findProgramOnSearchPath Cabal.silent locations exeName
    case fst <$> mLookupResult of
        Just path -> pure $ Just path
        Nothing   -> lookupExecutable exeNamesTail fallbackDirs
