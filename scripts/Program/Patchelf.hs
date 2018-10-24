module Program.Patchelf where

import Data.String.Utils
import Text.Printf

import Program

data Patchelf
instance Program Patchelf where
    executableName = "patchelf"

-- returns runpath if present, or secondarily rpath if present, or fails if neither present
getRpath :: FilePath -> IO String
getRpath binaryPath =
    strip <$> readProgram @Patchelf ["--print-rpath", binaryPath]

-- sets runpath on executable (overwriting if present)
setRpath :: FilePath -> String -> IO ()
setRpath binaryPath rpath = do
    putStrLn $ printf "Setting rpath on %s to %s" binaryPath rpath
    call @Patchelf ["--set-rpath", rpath, "--force-rpath", binaryPath]

-- NOTE: requires relatively new version of patchelf (likely 0.9), otherwise fail with stupid message "stat: No such file or directory"
removeRpath :: FilePath -> IO ()
removeRpath binaryPath = call @Patchelf ["--remove-rpath", binaryPath]
