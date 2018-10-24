module Program.CMake where

import Program
import System.Directory

data CMake
instance Program CMake where
    executableName = "cmake"

type VariableName = String
type VariableValue = String

data Option = Variable (VariableName, VariableValue)

formatOption :: Option -> [String]
formatOption (Variable (name, value)) = ["-D" <> name <> "=" <> value]

formatOptions :: [Option] -> [String]
formatOptions opts = concat $ formatOption <$> opts

cmake :: FilePath -> FilePath -> [Option] -> IO ()
cmake whereToRun whatToBuild opts = do
    createDirectoryIfMissing True whereToRun
    let varOptions = formatOptions opts
    callCwd @CMake whereToRun (varOptions <> [whatToBuild])
