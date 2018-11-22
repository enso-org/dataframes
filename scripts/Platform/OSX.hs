module Platform.OSX where

import Data.FileEmbed
import Data.List
import Data.Monoid
import System.FilePath
import System.IO.Temp
import System.Process
import System.Exit
import Text.Printf

import qualified Data.ByteString as BS

dlopenProgram :: BS.ByteString
dlopenProgram = $(embedFile "helpers/main.cpp")

-- Creates a process, observing what dynamic libraries get loaded.
-- Returns a list of absolute paths to loaded libraries, as provided by dyld.
-- NOTE: will wait for process to finish, should not be used with proceses that need input or wait for sth
-- NOTE: creates a process that may do pretty much anything (be careful of side effects)
getDependenciesOfExecutable :: FilePath -> [String] -> IO [FilePath]
getDependenciesOfExecutable exePath args = do
    let spawnInfo = (proc exePath args) { env = Just [("DYLD_PRINT_LIBRARIES", "1")] }
    result@(code, out, err) <- readCreateProcessWithExitCode spawnInfo ""
    case code of
        ExitFailure code -> fail $ printf  "call failed: %s:\nout: %s\nerr: %s\nreturn code %d" (show spawnInfo) out err code
        _ -> return ()

    -- we filter only lines beggining with he dyld prefix
    let dyldPrefix = "dyld: loaded: "
    let (relevant, irrelevant) = partition (isPrefixOf dyldPrefix) (lines err)
    let loadedPaths =(drop $ length dyldPrefix) <$> relevant
    pure $ delete exePath loadedPaths


getDependenciesOfDylibs :: [FilePath] -> IO [FilePath]
getDependenciesOfDylibs targets = withSystemTempDirectory "" $ \tempDir -> do
    let programSrcPath = tempDir </> "main.cpp"
    let programExePath = tempDir </> "moje"
    BS.writeFile programSrcPath dlopenProgram
    callProcess "clang++" [programSrcPath, "-o" <> programExePath]
    getDependenciesOfExecutable programExePath targets