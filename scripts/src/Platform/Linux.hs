module Platform.Linux where

import Data.List
import System.FilePath

import qualified Program.Ldd as Ldd
-- List of filenames of libraries that should not be distributed along with
-- our package but rather should be assumed to be present on end-user's machine
--
-- The list is mostly based on previous experience.
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

canBeDistributed :: FilePath -> Bool
canBeDistributed libraryPath = notElem (dropExtensions $ takeFileName libraryPath) librariesNotToBeDistributed

-- Linux-specific function.
-- The function takes paths to shared libraries and returns the ones that
-- should be distributed as the part of package. Libraries that are filtered
-- out should be assumed to be present at end-user's machine.
dependenciesToPackage :: [FilePath] -> IO [FilePath]
dependenciesToPackage binaries = filter canBeDistributed <$> Ldd.dependenciesOfBinaries binaries