module Program.CMake where

import Control.Monad.IO.Class
import GHC.Conc
import System.Directory
import Text.Printf

import Program
import Utils

data CMake
instance Program CMake where
    executableName = "cmake"

type VariableName = String
type VariableValue = String

data BuildType = Debug | ReleaseWithDebInfo

data Option = OptionSetVariable (VariableName, VariableValue)
            | OptionBuildType BuildType

formatOption :: Option -> [String]
formatOption (OptionSetVariable (name, value)) = [printf "-D%s=%s" name value]
formatOption (OptionBuildType ReleaseWithDebInfo) = formatOption $ OptionSetVariable ("CMAKE_BUILD_TYPE", "RelWithDebInfo")

formatOptions :: [Option] -> [String]
formatOptions opts = concat $ formatOption <$> opts

cmake :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
cmake whereToRun whatToBuild options = do
    liftIO $ createDirectoryIfMissing True whereToRun
    let varOptions = formatOptions options
    callCwd @CMake whereToRun (varOptions <> [whatToBuild])

build :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
build whereToRun whatToBuild options = do
    cmake whereToRun whatToBuild options
    make whereToRun

data Make
instance Program Make where
    executableName = "make"

make :: (MonadIO m) => FilePath -> m ()
make location = do
    cpuCount <- liftIO $ getNumProcessors
    jobCount <- getEnvDefault "JOB_COUNT" (show cpuCount)
    callCwd @Make location ["-j", jobCount]