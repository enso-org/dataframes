module Package.Library where

import Prologue

import Distribution.System
import System.Directory (createDirectoryIfMissing)
import System.FilePath
import System.FilePath.Glob

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

-- data ActionTriple input m = ActionTriple
--     { _before :: input -> m ()
--     , _action :: input -> m ()
--     , _after  :: input -> m ()
--     } deriving (Show)
-- makeLenses ''ActionTriple

-- instance (Applicative m)
--     => Default (ActionTriple input m) where
--     def = ActionTriple
--         { _before = const $ pure ()
--         , _action = const $ pure ()
--         , _after  = const $ pure ()
--         }

-- simpleAction a = ActionTriple def a def

-- executeAction :: (Monad m) => ActionTriple input m -> input -> m ()
-- executeAction triple input = do
--     triple ^. before $ input
--     triple ^. action $ input
--     triple ^. after  $ input

data Hook input m = Hook { _action :: input -> m () } deriving (Show)
makeLenses ''Hook

instance (Applicative m)
    => Default (Hook input m) where
    def = Hook { _action = const $ pure def }

data Hooks m = Hooks
    { _installNativeLibs :: Hook FilePath m
    } deriving (Show)
makeLenses ''Hooks

instance Applicative m 
    => Default (Hooks m) where
    def = Hooks
        { _installNativeLibs = def
        }

runHook :: Hook input m -> input -> m ()
runHook hook input = (hook ^. action) input

-- | Creates a redistributable package in the given output directory. Its
--   earlier contents will be deleted.
package :: MonadIO m => FilePath -> FilePath -> Hooks m -> m ()
package sourceDirectory outputDirectory pack = do
    putStrLn $ "Creating package at " <> outputDirectory
    Utils.prepareEmptyDirectory outputDirectory

    let dirsToCopy = ["src", "visualizers", ".luna-package"]
    for dirsToCopy $ Utils.copyDirectory sourceDirectory outputDirectory
    putStrLn $ "Packaging native libraries"
    runHook (pack ^. installNativeLibs) (outputDirectory </> nativeLibsBin)
    pure ()
