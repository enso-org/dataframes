module Program.Patchelf where

import Control.Monad.IO.Class (MonadIO, liftIO)
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
setRelativeRpath :: (MonadIO m) => FilePath -> [FilePath] -> m ()
setRelativeRpath binaryPath depsDirectories = setRpath binaryPath $ relativeRpath binaryPath <$> depsDirectories

-- returns runpath if present, or secondarily rpath if present, or fails if neither present
getRpath :: (MonadIO m) => FilePath -> m String
getRpath binaryPath =
    strip <$> readProgram @Patchelf ["--print-rpath", binaryPath]

-- sets runpath on executable (overwriting if present)
setRpath :: (MonadIO m) => FilePath -> [String] -> m ()
setRpath binaryPath rpath = do
    let rpathFormatted = intercalate ":" rpath
    liftIO $ putStrLn $ printf "Setting rpath on %s to %s" binaryPath rpathFormatted
    call @Patchelf ["--set-rpath", rpathFormatted, "--force-rpath", binaryPath]

-- NOTE: requires relatively new version of patchelf (likely 0.9), otherwise fail with stupid message "stat: No such file or directory"
removeRpath :: (MonadIO m) => FilePath -> m ()
removeRpath binaryPath = call @Patchelf ["--remove-rpath", binaryPath]

