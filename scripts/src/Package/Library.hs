module Package.Library where

import Prologue

import Distribution.System
import Logger
import System.Directory (createDirectoryIfMissing)
import System.FilePath
import System.FilePath.Glob

import qualified Paths as Paths
import qualified Utils as Utils

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
-- data Hook input output m = Hook { _action :: input -> m output } deriving (Show)
-- makeLenses ''Hook

defaultHook :: (Applicative m, Default output) => Hook input output m
defaultHook = const $ pure def

instance (Applicative m, Default output)
    => Default (Hook input output m) where
    def = const $ pure def
    -- def = Hook { _action = const $ pure def }
    
deduceTarget :: MonadIO m => FilePath -> m BuildInformation
deduceTarget out = do
    repoDir <- Paths.repoDir
    let name = takeFileName repoDir
    pure $ BuildInformation repoDir name out

data BuildInformation = BuildInformation
    { _rootDir :: FilePath
    , _libraryName :: String
    , _outputDirectory :: FilePath
    } deriving (Show)
makeLenses ''BuildInformation

data InitInput = InitInput { _buildInformation1 :: BuildInformation }
makeLenses ''InitInput

data BuildInput inited = BuildInput { _buildInformation2 :: BuildInformation, _initData2 :: inited }
makeLenses ''BuildInput

data InstallInput inited built = InstallInput { _buildInformation3 :: BuildInformation, _initData3 :: inited, _builtData3 :: built }
makeLenses ''InstallInput

data TestInput inited built packaged = TestInput { _buildInformation4 :: BuildInformation, _initData4 :: inited, _builtData4 :: built, _packagedData4 :: packaged }
makeLenses ''TestInput


data Hooks m inited built packaged = Hooks
    { _initialize :: Hook InitInput inited m
    , _buildNativeLibs :: Hook  (BuildInput inited) built m 
    , _installNativeLibs :: Hook (InstallInput inited built) packaged m
    , _runTests :: Hook (TestInput inited built packaged) () m
    } deriving (Show)
makeLenses ''Hooks

instance (Applicative m, Default built, Default packaged, Default inited)
    => Default (Hooks m inited built packaged) where
    def = Hooks
        { _initialize = defaultHook
        , _buildNativeLibs = defaultHook
        , _installNativeLibs = defaultHook
        , _runTests = defaultHook
        }

runHook :: Hook input output m -> input -> m output
runHook hook input = hook input
-- runHook hook input = (hook ^. action) input



-- | Creates a redistributable package in the given output directory. Its
--   earlier contents will be deleted.
package :: (Logger m, MonadIO m) => BuildInformation -> Hooks m inited built packaged -> m ()
package build pack = do
    let sourceDirectory = build ^. rootDir
    let outDir = build ^. outputDirectory
    logS $ "Building " <> show build
    logS $ "Will create package at " <> outDir

    logS $ "Cleaning the target directory"
    Utils.prepareEmptyDirectory outDir
    logS $ "Initialization..."
    inited <- (pack ^. initialize) (InitInput build)
    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    for dirsToCopy $ Utils.copyDirectory sourceDirectory outDir
    logS $ "Building native libraries"
    built <- (pack ^. buildNativeLibs) (BuildInput build inited)
    logS $ "Installing native libraries"
    package <- runHook (pack ^. installNativeLibs) (InstallInput build inited built) -- , outDir </> nativeLibsBin)
    logS $ "Testing"
    package <- runHook (pack ^. runTests) (TestInput build inited built package) -- (inited, built, package)
    logS $ "Package structure ready at " <> outDir
    pure ()
