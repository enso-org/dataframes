module Program.CMake where

import Program
import System.Directory
import Text.Printf

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
cmake whereToRun whatToBuild opts = do
    createDirectoryIfMissing True whereToRun
    let varOptions = formatOptions opts
    callCwd @CMake whereToRun (varOptions <> [whatToBuild])
