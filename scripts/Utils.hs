module Utils where

import Data.Maybe
import Distribution.Simple.Utils
import Distribution.Verbosity
import System.Directory
import System.Environment
import System.FilePath

fromJustVerbose :: String -> Maybe a -> a
fromJustVerbose msg maybeA = case maybeA of
    Just a -> a
    Nothing -> error msg

-- Copies subdirectory with all its contents between two directories
copyDirectory :: FilePath -> FilePath -> FilePath -> IO ()
copyDirectory sourceDirectory targetDirectory subdirectoryFilename = do
    let from = sourceDirectory </> subdirectoryFilename
    let to = targetDirectory </> subdirectoryFilename
    copyDirectoryRecursive silent from to

-- Copies to the given directory file under given path. Returns the copied-to path.
copyToDir :: FilePath -> FilePath -> IO FilePath
copyToDir destDir sourcePath = do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copy " ++ sourcePath ++ " to " ++ destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath
    pure destPath

-- Retrieves the value of an environment variable, returning the provided
-- default if the requested variable was not set.
getEnvDefault :: String -> String -> IO String
getEnvDefault variableName defaultValue =
    fromMaybe defaultValue <$> lookupEnv variableName

-- Retrieves the value of an environment variable, throwing an exception if the
-- variable was not set.
getEnvRequired :: String -> IO String
getEnvRequired variableName = lookupEnv variableName >>= \case
    Just value -> pure value
    Nothing    -> error $ "required environment variable `" <> variableName <> "` is not set!"

-- shortRelativePath requires normalised paths to work correctly.
-- this is helper function because we don't want to bother with checking
-- whether path is normalised everywhere else
relativeNormalisedPath :: FilePath -> FilePath -> FilePath
relativeNormalisedPath (normalise -> p1) (normalise -> p2) = shortRelativePath p1 p2
