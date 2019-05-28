-- | General utilities for packaging Luna Libraries.
module Package.Library where

-- TODO: deduplicate with https://github.com/luna/luna/blob/master/package/
-- there should be a single, definitive code unit for luna library package
-- layout

import Prologue

import qualified Archive        as Archive
import qualified Logger         as Logger
import qualified Paths          as Paths
import qualified System.IO.Temp as Temporary
import qualified Utils          as Utils

import Control.Monad        (filterM)
import Distribution.System
import System.Directory     (createDirectoryIfMissing, doesFileExist,
                             removePathForcibly)
import System.FilePath      (takeFileName, (</>))
import System.FilePath.Glob

nativeLibs    = "native_libs"
visualizers   = "visualizers"
sources       = "src"

nativeLibsBin = nativeLibs </> osBinDirName

rootDirectories :: [FilePath]
rootDirectories = [sources, nativeLibs, visualizers, ".luna-package"]

-- | Luna package layout has different binary directory name for each of 3
-- supported platforms.
osBinDirName :: String
osBinDirName = case buildOS of
    Windows -> "windows"
    Linux   -> "linux"
    OSX     -> "macos"
    _       -> error $ "osBinDirName: not implemented: " <> show buildOS


type Hook input output m = input -> m output

defaultHook :: (Applicative m, Default output) => Hook input output m
defaultHook = const $ pure def

instance (Applicative m, Default output)
    => Default (Hook input output m) where
    def = const $ pure def
    -- def = Hook { _action = const $ pure def }

-- | Function prepares build by deducing some information and creating the
--   temporary directory.
prepareBuild 
    :: (Logger.Logger m, MonadIO m) 
    => String  -- ^ Project name.
    -> m BuildInformation
prepareBuild name = do
    repoDir  <- Paths.repoDir
    let outputArchive = Paths.packageArchiveDefaultName name
    tmpDirRoot <- liftIO Temporary.getCanonicalTemporaryDirectory
    tmpDir <- liftIO $ Temporary.createTempDirectory tmpDirRoot "luna-packaging"
    let outDir = tmpDir </> name
    Logger.logS $ "Created temporary directory: " <> tmpDir
    pure $ BuildInformation
        { _rootDir           = repoDir
        , _libraryName       = name
        , _outputDirectory   = outDir
        , _outputArchiveName = outputArchive
        , _tempDirectory     = tmpDir
        , _keepTemp          = False
        , _testsEnabled      = True
        }

data BuildInformation = BuildInformation
    { _rootDir           :: FilePath -- ^ Root directory of the packaged library sources.
    , _libraryName       :: String   -- ^ Name of the packaged library.
    , _outputDirectory   :: FilePath -- ^ Output directory where the package contents wil be placed. NOTE: the original directory contents will be rmeoved.
    , _outputArchiveName :: FilePath -- ^ Output archive file with the library package.
    , _tempDirectory     :: FilePath
    , _keepTemp          :: Bool     -- ^ Encourages temporary files to be kept, so e.g. they can be investigated later.
    , _testsEnabled      :: Bool
    } deriving (Show)
makeLenses ''BuildInformation

data InitInput = InitInput
    { _buildInformation1 :: BuildInformation }
makeLenses ''InitInput

data BuildInput inited = BuildInput
    { _buildInformation2 :: BuildInformation
    , _initData2         :: inited
    }
makeLenses ''BuildInput

data InstallInput inited built = InstallInput
    { _buildInformation3 :: BuildInformation
    , _initData3         :: inited
    , _builtData3        :: built
    , _nativeLibsBinDir  :: FilePath
    }
makeLenses ''InstallInput

data TestInput inited built packaged = TestInput
    { _buildInformation4 :: BuildInformation
    , _initData4         :: inited
    , _builtData4        :: built
    , _packagedData4     :: packaged
    }
makeLenses ''TestInput


data Hooks m inited built packaged = Hooks
    { _initialize        :: Hook InitInput inited m
    , _buildNativeLibs   :: Hook (BuildInput inited) built m
    , _installNativeLibs :: Hook (InstallInput inited built) packaged m
    , _runTests          :: Hook (TestInput inited built packaged) () m
    } deriving (Show)
makeLenses ''Hooks

instance (Applicative m, Default built, Default packaged, Default inited)
    => Default (Hooks m inited built packaged) where
    def = Hooks
        { _initialize        = defaultHook
        , _buildNativeLibs   = defaultHook
        , _installNativeLibs = defaultHook
        , _runTests          = defaultHook
        }

runHook :: Hook input output m -> input -> m output
runHook hook input = hook input

-- | Creates a redistributable package in the given output directory. Its
--   earlier contents will be deleted.
package 
    :: (Logger.Logger m, MonadIO m) 
    => BuildInformation 
    -> Hooks m inited built packaged 
    -> m ()
package build pack = do
    let sourceDirectory = build ^. rootDir
    let outDir = build ^. outputDirectory
    Logger.logS $ "Building " <> show build
    Logger.logS $ "Will create package at " <> outDir

    Logger.logS $ "Cleaning the target directory"
    Utils.prepareEmptyDirectory outDir
    Logger.logS $ "Initialization..."
    inited <- (pack ^. initialize) (InitInput build)

    -- TODO: decide which ones are optional and appropriately handle them
    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    for dirsToCopy $ Utils.copyDirectory sourceDirectory outDir

    -- optional files, i.e. no error if not present
    let optionalFilesToCopy = (sourceDirectory </>) <$> ["snippets.yaml"]
    for optionalFilesToCopy $ flip Utils.copyFileToDirIfExists outDir

    Logger.logS $ "Building native libraries"
    built <- (pack ^. buildNativeLibs) (BuildInput build inited)
    Logger.logS $ "Installing native libraries"
    package <- runHook 
        (pack ^. installNativeLibs) 
        (InstallInput build inited built $ outDir </> nativeLibsBin)

    Logger.logS $ "Package structure ready at " <> outDir

    Logger.logS $ "Packing output directory into archive..."
    let archiveName = build ^. outputArchiveName
    Utils.removeFileIfExists archiveName
    Archive.packDirectory outDir archiveName

    when (build ^. testsEnabled) $ do
        Logger.logS $ "Testing"
        runHook (pack ^. runTests) (TestInput build inited built package)

    unless (build ^. keepTemp) $ do
        Logger.logS $ "Removing " <> build ^. tempDirectory
        liftIO $ removePathForcibly (build ^. tempDirectory)

    pure ()
