module Program.Patchelf where

import Data.List
import Data.String.Utils
import System.FilePath
import Text.Printf

import Program
import Utils

data Patchelf
instance Program Patchelf where
    executableName = "patchelf"

-- Given a path to a binary image and path to a dependencies folder, returns
-- a relative rpath entry using the $ORIGIN syntax.
relativeRpath :: FilePath -> FilePath -> String
relativeRpath binaryPath dependenciesDir = "$ORIGIN" </> relativeNormalisedPath (takeDirectory binaryPath) dependenciesDir

-- Given a path to binary image, sets rpath on the binary so the directories
-- given as second argument are added as paths relative to the binary.
setRelativeRpath :: FilePath -> [FilePath] -> IO ()
setRelativeRpath binaryPath depsDirectories = setRpath binaryPath $ relativeRpath binaryPath <$> depsDirectories

-- returns runpath if present, or secondarily rpath if present, or fails if neither present
getRpath :: FilePath -> IO String
getRpath binaryPath =
    strip <$> readProgram @Patchelf ["--print-rpath", binaryPath]

-- sets runpath on executable (overwriting if present)
setRpath :: FilePath -> [String] -> IO ()
setRpath binaryPath rpath = do
    let rpathFormatted = intercalate ":" rpath
    putStrLn $ printf "Setting rpath on %s to %s" binaryPath rpathFormatted
    call @Patchelf ["--set-rpath", rpathFormatted, "--force-rpath", binaryPath]

-- NOTE: requires relatively new version of patchelf (likely 0.9), otherwise fail with stupid message "stat: No such file or directory"
removeRpath :: FilePath -> IO ()
removeRpath binaryPath = call @Patchelf ["--remove-rpath", binaryPath]

--------------------------------------------------------------------------------------

-- Copies the binary to the given directory and sets rpath relative path to another directory.
-- (the dependencies directory will be treated as relative to the output directory)
installBinary :: FilePath -> FilePath -> FilePath -> IO ()
installBinary outputDirectory dependenciesDirectory sourcePath = do
    newBinaryPath <- copyToDir outputDirectory sourcePath
    setRelativeRpath newBinaryPath [dependenciesDirectory, outputDirectory]

-- Installs binary to the folder and sets this folder as rpath.
-- Typically used with dependencies (when install-to directory and dependencies directory are same)
installDependencyTo :: FilePath -> FilePath -> IO ()
installDependencyTo targetDirectory sourcePath = installBinary targetDirectory targetDirectory sourcePath
