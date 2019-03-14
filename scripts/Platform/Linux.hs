module Platform.Linux where

import Control.Exception (bracket)
import Data.List
import System.FilePath
import System.PosixCompat.Files (fileMode, getFileStatus, ownerWriteMode, setFileMode, unionFileModes)

import qualified Program.Ldd as Ldd
import qualified Program.Patchelf as Patchelf
import Utils (copyToDir)

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

-- Executes action while temporarily changing file mode to make it writable to owner.
-- Restores file mode before returning.
-- Will fail if the file is not owned.
withWritableFile :: FilePath -> IO () -> IO ()
withWritableFile path action = bracket makeWritable restoreMode (const action) where
    makeWritable = do
        oldStatus <- getFileStatus path
        setFileMode path $ unionFileModes (fileMode oldStatus) ownerWriteMode
        return oldStatus
    restoreMode oldStatus = setFileMode path (fileMode oldStatus)
    

-- Copies the binary to the given directory and sets rpath relative path to another directory.
-- (the dependencies directory will be treated as relative to the output directory)
installBinary :: FilePath -> FilePath -> FilePath -> IO ()
installBinary outputDirectory dependenciesDirectory sourcePath = do
    newBinaryPath <- copyToDir outputDirectory sourcePath
    withWritableFile newBinaryPath $ 
        Patchelf.setRelativeRpath newBinaryPath [dependenciesDirectory, outputDirectory]

-- Installs binary to the folder and sets this folder as rpath.
-- Typically used with dependencies (when install-to directory and dependencies directory are same)
installDependencyTo :: FilePath -> FilePath -> IO ()
installDependencyTo targetDirectory sourcePath = installBinary targetDirectory targetDirectory sourcePath
