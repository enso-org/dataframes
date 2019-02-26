module Program.CMake where

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

cmake :: FilePath -> FilePath -> [Option] -> IO ()
cmake whereToRun whatToBuild options = do
    createDirectoryIfMissing True whereToRun
    let varOptions = formatOptions options
    callCwd @CMake whereToRun (varOptions <> [whatToBuild])

build :: FilePath -> FilePath -> [Option] -> IO ()
build whereToRun whatToBuild options = do
    cmake whereToRun whatToBuild options
    make whereToRun

data Make
instance Program Make where
    executableName = "make"

make :: FilePath -> IO ()
make location = do
    cpuCount <- getNumProcessors
    jobCount <- getEnvDefault "JOB_COUNT" (show cpuCount)
    callCwd @Make location ["-j", jobCount]