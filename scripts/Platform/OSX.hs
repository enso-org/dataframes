module Platform.OSX where

import Control.Monad
import Data.FileEmbed
import Data.List
import Data.Monoid
import System.FilePath
import System.FilePath.Glob
import System.IO.Temp
import System.Process
import System.Exit
import Text.Printf

import qualified Data.ByteString as BS

import qualified Program.InstallNameTool as INT
import qualified Program.Otool as Otool

import Utils

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

-- Pattern for telling whether a path points to something belonging to the
-- internal python libraries directory. We want to match paths like
-- /Users/mwu/python-dist/lib/python3.7/lib-dynload/_asyncio.cpython-37m-darwin.so
internalPythonPath :: Pattern
internalPythonPath = compile "**/lib/python*.*/**/*"

-- System - a dependency that is assumed to be present out-of-the box on all
--          targeted OSX systems. Should not be distributed.
-- Python - a dependency belonging to the Python installation. Should be
--          distributed as part of Python package (special rule).
-- Local  - all other dependencies, typically "plain" local libraries, should
--          be distributed using typical scenario (installBinary).
data DependencyCategory = Local | Python | System
    deriving (Eq, Show)

categorizeDependency :: FilePath -> DependencyCategory
categorizeDependency dependencyFullPath =
    if match internalPythonPath dependencyFullPath
        then Python
        else if (isPrefixOf "/usr/lib/system/" dependencyFullPath) || (isPrefixOf "/System" dependencyFullPath)
            then System
            else if takeDirectory dependencyFullPath == "/usr/lib"
                then System
                else Local

isLocalDep :: FilePath -> Bool
isLocalDep dep = categorizeDependency dep == Local

-- Function installs Mach-O binary in a target binary folder.
-- install name shall be rewritten to contain only a filename
-- install names of direct local dependencies shall be rewritten, assuming they are in the same dir
installBinary :: FilePath -> FilePath -> IO ()
installBinary targetBinariesDir sourcePath = do
    -- putStrLn $ "installing " <> takeFileName sourcePath <> " to " <> targetBinariesDir
    destinationPath <- copyToDir targetBinariesDir sourcePath
    callProcess "chmod" ["777", destinationPath]
    INT.setInstallName destinationPath $ takeFileName destinationPath
    (filter isLocalDep -> directDeps) <- Otool.usedLibraries destinationPath
    flip mapM directDeps $ \installName -> do
        when (isLocalDep installName) $ do
            -- local dependencies of local dependencies are in the same folder as the current binary
            -- NOTE: in future, in multi-package world, there might be more folders
            INT.change destinationPath installName $ "@loader_path" </> takeFileName installName
    callProcess "chmod" ["555", destinationPath]