{-| Description : Utilities specific to macOS. -}

module Platform.MacOS where

import Prologue

import qualified Data.ByteString         as BS
import qualified Platform.Unix           as Unix
import qualified Program.InstallNameTool as INT
import qualified Program.Otool           as Otool
import qualified System.Process.Typed    as Process
import qualified Utils                   as Utils

import Control.Monad                  (filterM)
import Data.FileEmbed                 (embedFile)
import Data.List                      (delete, elemIndex, find, isPrefixOf,
                                       partition)
import Distribution.Simple.BuildPaths (dllExtension)
import Distribution.Simple.Utils      (fromUTF8LBS)
import Distribution.System            (buildPlatform)
import System.Directory               (doesPathExist)
import System.Exit                    (ExitCode (ExitFailure))
import System.FilePath                (replaceFileName, takeDirectory,
                                       takeExtension, takeFileName, (</>))
import System.FilePath.Glob           (Pattern, compile, match)
import System.IO.Temp                 (withSystemTempDirectory)
import Text.Printf                    (printf)


-- | Source code of C++ program that loads dybamic libraries passed as its
--   arguments. See 'getDependenciesOfExecutable'.
dlopenProgram :: BS.ByteString
dlopenProgram = $(embedFile "helpers/main.cpp")

-- | Creates a process, observing what dynamic libraries get loaded. Returns a
--   list of absolute paths to loaded libraries, as provided by dyld.
--
--   NOTE: will wait for process to finish, should not be used with proceses
--   that need input or wait for sth
--
--   NOTE: creates a process that may do pretty much anything (be careful of
--   side effects)
--
--   NOTE: when dependency is accessed through symlink, the target location will
--   be returned (see 'workaroundSymlinkedDeps')
getDependenciesOfExecutable
    :: (MonadIO m)
    => FilePath -- ^ Executable path
    -> [String] -- ^ Arguments to be used to call executable.
    -> m [FilePath]
getDependenciesOfExecutable exePath args = do
    let envToAdd = [("DYLD_PRINT_LIBRARIES", "1")]
    let spawnInfo = (Process.setEnv envToAdd $ Process.proc exePath args)
    (code, fromUTF8LBS -> out, fromUTF8LBS -> err)
        <- Process.readProcess spawnInfo
    case code of
        ExitFailure code ->
            liftIO $ fail $ printf
                "call failed: %s:\nout: %s\nerr: %s\nreturn code %d"
                (show spawnInfo) out err code
        _ -> pure ()

    -- we filter only lines beggining with he dyld prefix
    let dyldPrefix = "dyld: loaded: "
    let (relevant, _) = partition (isPrefixOf dyldPrefix) (lines err)
    let loadedPaths = drop (length dyldPrefix) <$> relevant
    pure $ delete exePath loadedPaths

-- | Tries to collect all dynamic libraries loaded when given set of dynamic
--   libraries is loaded. Internally builds program that loads them and runs it.
--   Requires @clang++@ compiler to be available.
getDependenciesOfDylibs :: (MonadIO m) => [FilePath] -> m [FilePath]
getDependenciesOfDylibs targets
    = liftIO $ withSystemTempDirectory "" $ \tempDir -> do
    let programSrcPath = tempDir </> "main.cpp"
    let programExePath = tempDir </> "moje"
    BS.writeFile programSrcPath dlopenProgram
    Process.runProcess_
        $ Process.proc "clang++" [programSrcPath, "-o" <> programExePath]
    getDependenciesOfExecutable programExePath targets

-- | Pattern for telling whether a path points to something belonging to the
--   internal python libraries directory. We want to match paths like
--   /Users/mwu/python-dist/lib/python3.7/lib-dynload/_asyncio.cpython-37m-darwin.so
internalPythonPath :: Pattern
internalPythonPath = compile "**/lib/python*.*/**/*"

-- | All dynamic library dependencies on macOS can be split to the following
--   categories, depending on how we want to package and redistribute them.
data DependencyCategory =
      System -- ^ a dependency that is assumed to be present out-of-the box on
             --   all targeted OSX systems. Should not be distributed.
    | Python -- ^ a dependency belonging to the Python installation. Should be
             --   distributed as part of Python package (special rule).
    | Local  -- ^ all other dependencies, typically "plain" local libraries,
             --   should be distributed using typical scenario (installBinary).
    deriving (Eq, Show)

-- | To what category given dynamic library belongs.
categorizeDependency :: FilePath -- ^ An absolute path to dynamic library.
                     -> DependencyCategory
categorizeDependency dependencyFullPath =
    if match internalPythonPath dependencyFullPath
        then Python
        else if (isPrefixOf "/usr/lib/system/" dependencyFullPath)
             || (isPrefixOf "/System" dependencyFullPath)
            then System
            else if takeDirectory dependencyFullPath == "/usr/lib"
                then System
                else Local

-- | Checks if given library is a 'Local' dependency.
isLocalDep :: FilePath -> Bool
isLocalDep dep = categorizeDependency dep == Local

-- | Function installs Mach-O binary in a target binary folder. install name
--   shall be rewritten to contain only a filename install names of direct local
--   dependencies shall be rewritten, assuming they are in the same dir returns
--   the path to the installed dependency (in the target dir)
installBinary :: (MonadIO m) => FilePath -> FilePath -> m FilePath
installBinary targetBinariesDir sourcePath = do
    -- putStrLn $ "installing " <> takeFileName sourcePath <> " to " <> targetBinariesDir
    destinationPath <- Utils.copyToDir targetBinariesDir sourcePath
    Unix.withWritableFile destinationPath $ do
        INT.setInstallName destinationPath $ takeFileName destinationPath
        directDeps <- filter isLocalDep <$> Otool.usedLibraries destinationPath
        for_ directDeps $ \installName -> do
            when (isLocalDep installName) $ do
                -- local dependencies of local dependencies are in the same
                -- folder as the current binary
                --
                -- NOTE: in future, in multi-package world, there might be more
                -- folders
                INT.change destinationPath installName
                    $ "@loader_path" </> takeFileName installName
    pure destinationPath

-- | Returns file name prefix until the first dot character
--   e.g. @/foo/bar/libicudata.63.dylib@ -> @libicudata@
takeNamePrefix :: FilePath -> FilePath
takeNamePrefix = takeWhile (/= '.') . takeFileName

-- | Returns a predicate that checks if given file has the same name as the
--   first argument up to the first dot.
--   E.g. @libicudata.63.dylib@ matches @libicudata.63.1.dylib@
matchesPrefix
    :: FilePath -- ^ Path to compare against
    -> (FilePath -> Bool) -- ^ Resulting predicate
matchesPrefix lhs rhs = takeNamePrefix lhs == takeNamePrefix rhs

-- | Tries to resolve unresolved dependency by using a library with the same
--   name but different version. If such library is found among installed
--   dependencies, its install name shall be used. See 'workaroundSymlinkedDeps'
--   for a full explanation.
--   Fails when the library cannot be resolved.
fixUnresolvedDependency
    :: (MonadIO m)
    => [FilePath] -- ^ All installed dependencies
    -> FilePath  -- ^ Binary to be patched
    -> String  -- ^ Dependency of the bianry
    -> m ()
fixUnresolvedDependency installedBinaries binary dependency = do
    let match = find (matchesPrefix dependency) installedBinaries
    case match of
        Nothing -> error
            $ printf "for binary %s: cannot resolve dependency: %s"
              binary dependency
        Just matchingPath -> do
            let matchedFileName = takeFileName matchingPath
            let adjustedDependency = replaceFileName dependency matchedFileName
            liftIO $ putStrLn
                   $ printf "\tpatching %s -> %s" dependency adjustedDependency
            INT.change binary dependency adjustedDependency

-- | Checks if this is a dependency expected to be next to the loaded binary
--   but not actually present.
unresolvedLocalDependency :: (MonadIO m) => FilePath -> FilePath -> m Bool
unresolvedLocalDependency installedBinary dependencyInstallName = do
    if isPrefixOf "@loader_path" dependencyInstallName then do
        let dependencyInstallFileName = takeFileName dependencyInstallName
        let expectedPath = replaceFileName installedBinary dependencyInstallFileName
        not <$> (liftIO <$> doesPathExist) expectedPath
    else pure False

-- | This procedure is to workaround issues when install names of dependencies
--   point to symbolic links. Currently we just copy all actual dependencies
--   (and know their "real paths") but this might be not enough. For example:
--   @libicui18n.63.dylib@ depends on @libicudata.63.dylib@ and
--   @libicudata.63.dylib@ is a symlink to @libicudata.63.1.dylib@. We place
--   just @libicudata.63.1.dylib@ in the package and dyld will fail to find
--   @libicudata.63.dylib@.
--
--   This procedure looks for unresolved local libraries (ie. the library
--   install names that are not present next to installed binary, and tries to
--   patch the install name so it refers to the library name as placed in the
--   package. This affects only libraries with same name prefix (that is part of
--   name until first dot).
--
--   In the long-term this solution should be abandoned, dependencies should be
--   described by packages, just like binaries to install.
workaroundSymlinkedDeps :: (MonadIO m) => [FilePath] -> m ()
workaroundSymlinkedDeps installedBinaries =
    for_ installedBinaries $ \target -> do
        putStrLn $ "Looking into " <> target
        deps <- Otool.usedLibraries target
        let localDeps = filter (isPrefixOf "@loader_path") deps
        missingDeps <- filterM (unresolvedLocalDependency target) localDeps
        unless (null missingDeps) $ do
            putStrLn $ "Will try patching paths for dependencies of " <> target
            for_ missingDeps $ fixUnresolvedDependency installedBinaries target

-- | Checks filename extension.
isDylib :: FilePath -> Bool
isDylib path = takeExtension path == '.' : dllExtension buildPlatform

-- | The target binaries and their shared library dependencies get copied into
--   target directory.
packageBinaries
    :: MonadIO m
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m [FilePath] -- ^ List of installed binaries (their target path).
packageBinaries targetDir binaries additionalLocations = do
    unless (null additionalLocations)
        (error "additional library location not supported on macOS" :: m ())

    let (dylibBinaries, exeBinaries) = partition isDylib binaries
    -- Note: This code assumed that all redistributable build artifacts are dylibs.
    --       It might need generalization in future.
    dylibDeps <- getDependenciesOfDylibs dylibBinaries
    unless (null exeBinaries)
        (error $ "packaging arbtitrary executables not supported on macOS, encountered: " <> show exeBinaries :: m ())

    let allDeps = dylibDeps <> [] -- FIXME: exe deps

    let trulyLocalDependency path = isLocalDep path && (not $ elem path binaries)
    let localDependencies = filter trulyLocalDependency allDeps
    let binariesToInstall = localDependencies <> binaries
    putStrLn $ "Binaries to install: " <> show binariesToInstall

    -- Place all artifact binaries and their local dependencies in the destination directory
    binariesInstalled <- for binariesToInstall $ installBinary targetDir

    -- Workaround possible issues caused by deps install names referring to symlinks
    workaroundSymlinkedDeps binariesInstalled

    pure binariesInstalled
