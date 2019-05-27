-- | Module for high-level platform-specific utilities that can be feasibly
-- exposed under unified interface.
module Platform where

import Prologue

import Distribution.System (OS (..), buildOS, buildPlatform)
import System.Directory    (exeExtension)
import System.FilePath     ((-<.>), (<.>))

import qualified Distribution.Simple.BuildPaths as Cabal
import qualified Platform.Linux   as Linux
import qualified Platform.MacOS   as MacOS
import qualified Platform.Windows as Windows

dynamicLibraryPrefix :: String
dynamicLibraryPrefix = case buildOS of
    Windows -> ""
    _       -> "lib"

dynamicLibraryExtension :: String
dynamicLibraryExtension = Cabal.dllExtension buildPlatform

-- | Converts root name into os-specific library filename, e.g.:
--   @DataframeHelper@ becomes @DataframeHelper.dll@ on Windows or
--   @libDataframeHelper.so@ on Linux.
libraryFilename :: FilePath -> FilePath
libraryFilename name = dynamicLibraryPrefix <> name <.> dynamicLibraryExtension

-- | Adds, if needed, a platform-appropriate executable extensions.
executableFilename :: FilePath -> FilePath
executableFilename = (-<.> exeExtension)

-- | Places the binaries and their dependencies (shared libraries) in the target
--   directory. Binaries might get patched to properly discover their packaged
--   dependencies.
packageBinaries 
    :: MonadIO m 
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m [FilePath] -- ^ List of installed binaries (their target path).
packageBinaries = case buildOS of
    Windows -> Windows.packageBinaries
    Linux   -> Linux.packageBinaries
    OSX     -> MacOS.packageBinaries