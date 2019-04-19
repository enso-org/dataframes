{-|
Description : Utilities specific to Linux.

The module defines utilities specific to Linux. Note that while this module can
be compiled on other platforms, functions are not expected to properly work in
such environment.
-}

module Platform.Linux where

import Prologue

import qualified Platform.Unix    as Unix
import qualified Program.Ldd      as Ldd
import qualified Program.Patchelf as Patchelf
import qualified Utils            as Utils

import System.FilePath          (dropExtensions, takeFileName)

-- | Filenames (without extension) of libraries that shouldn't be redistributed
--   in the package. Typically low level system libraries or driver-specific
--   modules.
--
--   The list is mostly based on previous experience. It is likely incomplete,
--   not entirely reliable and will be subject to future updates.
librariesNotToBeDistributed :: [String]
librariesNotToBeDistributed =  
    [ "libX11", "libXext", "libXau", "libXdamage", "libXfixes", "libX11-xcb"
    , "libXxf86vm", "libXdmcp", "libGL", "libdl", "libc", "librt", "libm"
    , "libpthread", "libXcomposite", "libnvidia-tls", "libnvidia-glcore"
    , "libnvidia-glsi", "libXrender", "libXi", "libdrm", "libutil", "libgbm"
    , "libdbus-1", "libselinux", "ld-linux-x86-64"
    ]

-- | Checks if the given library should be distributed as part of relocatable
--   package. Relies on 'librariesNotToBeDistributed'.
isDistributable
    :: FilePath  -- ^ Shared library path — can be absolute path, relative path
                 --   or just a filename.
    -> Bool
isDistributable libraryPath 
    = notElem 
        (dropExtensions $ takeFileName libraryPath) 
        librariesNotToBeDistributed

-- | Function collects the list of shared library dependencies that should
--   package along the given set of binaries. This is implemented on top of
--   @ldd@ command.
--
--   Note that when packaging, all the binaries (including dependencies) should
--   be also patched to properly prefer shipped libraries over the ones
--   installed locally on the end-user's machine. See
--   'Program.Patchelf.installBinary'.
dependenciesToPackage
    :: (MonadIO m)
    => [FilePath]   -- ^ Binaries to be packaged that will be checked for 
                    ---  dependencies.
    -> m [FilePath] -- ^ Shared library depencies (including transitive ones) 
                    --   that need to be shipped along.
dependenciesToPackage binaries 
    = filter isDistributable <$> Ldd.dependenciesOfBinaries binaries

-- | Copies the binary to the given directory and sets rpath relative path to
--   another directory. (the dependencies directory will be treated as relative
--   to the output directory)
installBinary
    :: (MonadIO m)
    => FilePath   -- ^ Output directory where the binary will be copied into.
    -> FilePath   -- ^ Directory containing library dependencies (in the same
                  --   package structure as output directory).
    -> FilePath   -- ^ Binary to be copied and patched.
    -> m FilePath -- ^ Path to the packaged binary.
installBinary outputDirectory dependenciesDirectory sourcePath = do
    newBinaryPath <- Utils.copyToDir outputDirectory sourcePath
    Unix.withWritableFile newBinaryPath $
        Patchelf.setRelativeRpath 
            newBinaryPath 
            [dependenciesDirectory, outputDirectory]
    pure newBinaryPath

-- | Installs binary to the folder and sets this folder as rpath. Typically used
--   with dependencies (when install-to directory and dependencies directory are
--   same)
installDependencyTo :: (MonadIO m) => FilePath -> FilePath -> m FilePath
installDependencyTo targetDirectory 
    = installBinary targetDirectory targetDirectory

    
-- | The target binaries and their shared library dependencies get copied into
--   target directory. Fails if there are unresolved dependencies.
packageBinaries 
    :: MonadIO m 
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m [FilePath] -- ^ List of installed binaries (their target path).
packageBinaries targetDir binaries additionalLocations = do
    unless (null additionalLocations)
        (error "additional library location not supported on Linux" :: m ())
    dependencies <- dependenciesToPackage binaries
    for (dependencies <> binaries) $ installDependencyTo targetDir
