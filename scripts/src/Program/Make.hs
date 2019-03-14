module Program.Make where

import Control.Monad.IO.Class (MonadIO, liftIO)
import GHC.Conc (getNumProcessors)
import Program
import Utils

data Make
instance Program Make where
    executableName = "make"

make :: (MonadIO m) => FilePath -> m ()
make location = do
    cpuCount <- liftIO $ getNumProcessors
    jobCount <- getEnvDefault "JOB_COUNT" (show cpuCount)
    callCwd @Make location ["-j", jobCount]