module Program.CMake where


import Prologue

import qualified Program      as Program
import qualified Program.Make as Make

import Program          (Program)
import System.Directory (createDirectoryIfMissing)
import Text.Printf      (printf)


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
formatOption (OptionBuildType Debug) = formatOption $ OptionSetVariable ("CMAKE_BUILD_TYPE", "Debug")
formatOption (OptionBuildType ReleaseWithDebInfo) = formatOption $ OptionSetVariable ("CMAKE_BUILD_TYPE", "RelWithDebInfo")

formatOptions :: [Option] -> [String]
formatOptions opts = concat $ formatOption <$> opts

cmake :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
cmake whereToRun whatToBuild options = do
    liftIO $ createDirectoryIfMissing True whereToRun
    let varOptions = formatOptions options
    Program.callCwd @CMake whereToRun (varOptions <> [whatToBuild])

-- FIXME: drop dependency on make, use --build
build :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
build whereToRun whatToBuild options = do
    cmake whereToRun whatToBuild options
    Make.make whereToRun
