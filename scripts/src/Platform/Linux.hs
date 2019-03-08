{-|
Description : Utilities specific to Linux.

The module defines utilities specific to Linux. Note that while this module can be compiled on other platforms, functions are not expected to properly work in such environment.
-}

module Platform.Linux where

import Control.Monad.IO.Class
import Data.List
import System.FilePath

import qualified Program.Ldd as Ldd

-- |Filenames (without extension) of libraries that shouldn't be redistributed in the package. Typically low level system libraries or driver-specific modules.
--
-- The list is mostly based on previous experience. It is likely incomplete, not entirely reliable and will be subject to future updates.
librariesNotToBeDistributed :: [String]
librariesNotToBeDistributed =  [
    "libX11", "libXext", "libXau", "libXdamage", "libXfixes", "libX11-xcb",
    "libXxf86vm", "libXdmcp", "libGL", "libdl", "libc", "librt", "libm", "libpthread",
    "libXcomposite",
    "libnvidia-tls", "libnvidia-glcore", "libnvidia-glsi",
    "libXrender", "libXi",
    "libdrm",
    "libutil",
    "libgbm", "libdbus-1",
    "libselinux",
    "ld-linux-x86-64"
    ]

-- | Checks if the given library should be distributed as part of relocatable package. Relies on 'librariesNotToBeDistributed'.
isDistributable 
    :: FilePath  -- ^ Shared library path — can be absolute path, relative path or just a filename.
    -> Bool
isDistributable libraryPath = notElem (dropExtensions $ takeFileName libraryPath) librariesNotToBeDistributed

-- | Function collects the list of shared library dependencies that should package along the given set of binaries.
--
-- Note that when packaging, all the binaries (including dependencies) should be also patched to properly prefer shipped libraries over the ones installed locally on the end-user's machine. See 'Program.Patchelf.installBinary'.
dependenciesToPackage 
    :: (MonadIO m) 
    => [FilePath] -- ^ Binaries to be packaged that will be checked for dependencies.
    -> m [FilePath] -- ^ Shared library depencies (including transitive ones) that need to be shipped along. 
dependenciesToPackage binaries = filter isDistributable <$> Ldd.dependenciesOfBinaries binaries