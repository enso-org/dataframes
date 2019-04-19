module Program.Git where

import Prologue

import qualified Program as Program
import qualified Data.ByteString.Lazy.UTF8 as UTF8
import qualified System.Process.Typed             as Process

import System.Directory (doesDirectoryExist)
import System.FilePath (takeDirectory, normalise, isRelative, (</>))
import System.Exit (ExitCode(ExitSuccess, ExitFailure))

------------------
-- === Curl === --
------------------

-- === Definition === --

data Git

instance Program.Program Git where
    executableName = "git"

-- === Utils === --


-- | A few git query-like commands return a text line output on success.
-- Output line may be followed by an empty line.
expectSingleLineOut :: MonadIO m => Process.ProcessConfig () () () -> m (Maybe String)
expectSingleLineOut cmd = runMaybeT $ do
    let cmd' = Process.setStderr Process.closed cmd
    (result, stdout) <- liftIO $ Process.readProcessStdout cmd'
    -- Note: in theory the call might fail due to unexpected reasons.
    guard (result == ExitSuccess)
    case lines (UTF8.toString stdout) of
        []                  -> fail "expected non-empty output"
        firstLine : []      -> pure $ firstLine
        firstLine : "" : [] -> pure $ firstLine
        _                   -> fail "output may be followed by a single blank line only"
    
-- === API === --

-- | Returns true if given path points to git repository or its subtree.
-- WARNING: behavior is defined only for directories that exist.
isRepositorySubtree :: MonadIO m => FilePath -> m Bool
isRepositorySubtree path = do
    cmd <- Program.progCwd @Git path ["rev-parse", "--is-inside-work-tree"]
    maybeLine <- expectSingleLineOut cmd
    pure $ maybeLine == Just "true"

-- Returns repository root (without trailing .git directory) when given a path
-- to repository or its subtree.
-- Returns Nothing if given path does not belong to repository.
repositoryRoot :: FilePath -> IO (Maybe FilePath)
repositoryRoot path = do
    cmd <- Program.progCwd @Git path ["rev-parse", "--git-dir"]
    maybeLine <- expectSingleLineOut cmd
    pure $ do
        line <- maybeLine
        pure $ takeDirectory $ normalise $ 
            -- For some reson git returns relative path ".git" when called in 
            -- repo root, and absolute path to .git when called in its strict
            -- subtree. Well then... let's make sure it remains absolute.
            if isRelative line 
                then path </> line
                else line