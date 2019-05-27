module Program.Make where

import Prologue

import qualified Program as Program
import qualified Utils   as Utils

import GHC.Conc (getNumProcessors)
import Program  (Program)

data Make
instance Program Make where
    executableName = "make"

make :: (MonadIO m) => FilePath -> m ()
make location = do
    cpuCount <- liftIO $ getNumProcessors
    jobCount <- Utils.getEnvDefault "JOB_COUNT" (show cpuCount)
    Program.callCwd @Make location ["-j", jobCount]
