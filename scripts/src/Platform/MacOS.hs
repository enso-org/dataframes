module Platform.MacOS where


import Prologue

import qualified Data.ByteString         as BS
import qualified Program.InstallNameTool as INT
import qualified Program.Otool           as Otool
import qualified System.Process.Typed    as Process
import qualified Utils                   as Utils

import Control.Monad             (filterM)
import Data.FileEmbed            (embedFile)
import Data.List                 (delete, elemIndex, find, isPrefixOf,
                                  partition)
import Distribution.Simple.Utils (fromUTF8LBS)
import System.Directory          (doesPathExist)
import System.Exit               (ExitCode (ExitFailure))
import System.FilePath           (replaceFileName, takeDirectory, takeFileName,
                                  (</>))
import System.FilePath.Glob      (Pattern, compile, match)
import System.IO.Temp            (withSystemTempDirectory)
import Text.Printf               (printf)



dlopenProgram :: BS.ByteString
dlopenProgram = $(embedFile "helpers/main.cpp")

-- Creates a process, observing what dynamic libraries get loaded.
-- Returns a list of absolute paths to loaded libraries, as provided by dyld.
-- NOTE: will wait for process to finish, should not be used with proceses that need input or wait for sth
-- NOTE: creates a process that may do pretty much anything (be careful of side effects)
getDependenciesOfExecutable :: (MonadIO m) => FilePath -> [String] -> m [FilePath]
getDependenciesOfExecutable exePath args = do
    let envToAdd = [("DYLD_PRINT_LIBRARIES", "1")]
    let spawnInfo = (Process.setEnv envToAdd $ Process.proc exePath args)
    (code, fromUTF8LBS -> out, fromUTF8LBS -> err) <- Process.readProcess spawnInfo
    case code of
        ExitFailure code -> liftIO $ fail $ printf  "call failed: %s:\nout: %s\nerr: %s\nreturn code %d" (show spawnInfo) out err code
        _ -> pure ()

    -- we filter only lines beggining with he dyld prefix
    let dyldPrefix = "dyld: loaded: "
    let (relevant, _) = partition (isPrefixOf dyldPrefix) (lines err)
    let loadedPaths = drop (length dyldPrefix) <$> relevant
    pure $ delete exePath loadedPaths


getDependenciesOfDylibs :: (MonadIO m) => [FilePath] -> m [FilePath]
getDependenciesOfDylibs targets = liftIO $ withSystemTempDirectory "" $ \tempDir -> do
    let programSrcPath = tempDir </> "main.cpp"
    let programExePath = tempDir </> "moje"
    BS.writeFile programSrcPath dlopenProgram
    Process.runProcess_ $ Process.proc "clang++" [programSrcPath, "-o" <> programExePath]
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
-- returns the path to the installed dependency (in the target dir)
installBinary :: (MonadIO m) => FilePath -> FilePath -> m FilePath
installBinary targetBinariesDir sourcePath = do
    -- putStrLn $ "installing " <> takeFileName sourcePath <> " to " <> targetBinariesDir
    destinationPath <- Utils.copyToDir targetBinariesDir sourcePath
    Process.runProcess_ $ Process.proc "chmod" ["777", destinationPath]
    INT.setInstallName destinationPath $ takeFileName destinationPath
    directDeps <- filter isLocalDep <$> Otool.usedLibraries destinationPath
    flip mapM directDeps $ \installName -> do
        when (isLocalDep installName) $ do
            -- local dependencies of local dependencies are in the same folder as the current binary
            -- NOTE: in future, in multi-package world, there might be more folders
            INT.change destinationPath installName $ "@loader_path" </> takeFileName installName
    pure destinationPath

-- If installed libraries list contains library with the same name (until dot)
-- then it will used instead of current install name.
-- See workaroundSymlinkedDeps for a full explanation.
fixUnresolvedDependency :: (MonadIO m) => [FilePath] -> FilePath -> String -> m ()
fixUnresolvedDependency installedBinaries binary dependency = do
    let depName = takeFileName dependency
    let dotPosition = Utils.fromJustVerbose ("dependency install name " <> dependency <> " is expected to contain a dot character") (elemIndex '.' depName)
    let namePrefix = take dotPosition depName
    let matchesPrefix binaryPath = isPrefixOf namePrefix $ takeFileName binaryPath
    let match = find matchesPrefix installedBinaries
    case match of
        Nothing -> error $ printf "installed binary: %s: cannot resolve dependency: %s" binary dependency
        Just matchingPath -> do
            let adjustedDependency = replaceFileName dependency $ takeFileName matchingPath
            liftIO $ putStrLn $ printf "\tpatching %s -> %s" dependency adjustedDependency 
            INT.change binary dependency adjustedDependency

-- Checks if this is a dependency expected to be next to the loaded binary
-- but not actually present.
unresolvedLocalDependency :: (MonadIO m) => FilePath -> FilePath -> m Bool
unresolvedLocalDependency installedBinary dependencyInstallName = do
    if isPrefixOf "@loader_path" dependencyInstallName then do
        let expectedPath = replaceFileName installedBinary (takeFileName dependencyInstallName)
        not <$> (liftIO <$> doesPathExist) expectedPath
    else pure False

-- This procedure is to workaround issues when install names of dependencies
-- point to symbolic links. Currently we just copy all actual dependencies 
-- (and know their "real paths") but this might be not enough.
-- For example: libicui18n.63.dylib depends on libicudata.63.dylib
-- and libicudata.63.dylib is a symlink to libicudata.63.1.dylib.
-- We place just libicudata.63.1.dylib in the package and dyld will fail to
-- find libicudata.63.dylib.
-- This procedure looks for unresolved local libraries (ie. the library install
-- names that are not present next to installed binary, and tries to patch the 
-- install name so it refers to the library name as placed in the package.
-- This affects only libraries with same name prefix (that is part of name 
-- until first dot).
--
-- In the long-term this solution should be abandoned, dependencies should be
-- described by packages, just like binaries to install.
workaroundSymlinkedDeps :: (MonadIO m) => [FilePath] -> m ()
workaroundSymlinkedDeps installedBinaries = do
    let handleBinary target = liftIO $ do
            putStrLn $ "Looking into " <> target
            deps <- Otool.usedLibraries target
            let localDeps = filter (isPrefixOf "@loader_path") deps
            missingDeps <- filterM (unresolvedLocalDependency target) localDeps
            unless (null missingDeps) $ do
                putStrLn $ "Will try patching paths for dependencies of " <> target
                mapM_ (fixUnresolvedDependency installedBinaries target) missingDeps
           
    mapM_ handleBinary installedBinaries
