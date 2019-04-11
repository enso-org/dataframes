module Utils where

import Prologue

import Distribution.Simple.Utils (copyDirectoryRecursive, shortRelativePath)
import Distribution.Verbosity    (silent)
import System.Directory          (copyFile, createDirectoryIfMissing,
                                  removeDirectoryRecursive)
import System.Environment        (lookupEnv)
import System.FilePath           (normalise, takeFileName, (</>))
import System.IO.Error           (isDoesNotExistError)
import Conduit                   (ConduitM, await, yield)

-- | If True returns Just value, else Nothing
toMaybe :: Bool -> a -> Maybe a
toMaybe = \case
    True -> Just
    False -> const Nothing

-- | Ensures: directory under given path exists and is empty
prepareEmptyDirectory :: MonadIO m => FilePath -> m ()
prepareEmptyDirectory path = liftIO $ do
    removeDirectoryRecursiveIfExists path
    createDirectoryIfMissing True path

-- | As 'removeDirectoryRecursive' but doesn't fail when the path does not
-- exist.
removeDirectoryRecursiveIfExists :: (MonadIO m) => FilePath -> m ()
removeDirectoryRecursiveIfExists path = liftIO $ catchJust
    (\e -> if isDoesNotExistError e then Just () else Nothing)
    (removeDirectoryRecursive path)
    (const $ pure ())

-- | As fromJust but provides an error message if called on Nothing
fromJustVerbose :: String -> Maybe a -> a
fromJustVerbose msg maybeA = case maybeA of
    Just a  -> a
    Nothing -> error msg

-- | Copies subdirectory with all its contents between two directories
copyDirectory :: (MonadIO m) => FilePath -> FilePath -> FilePath -> m FilePath
copyDirectory sourceDirectory targetDirectory subdirectoryFilename = liftIO $ do
    let from = sourceDirectory </> subdirectoryFilename
    let to = targetDirectory </> subdirectoryFilename
    putStrLn $ "Copying " <> from <> " to " <> targetDirectory
    copyDirectoryRecursive silent from to
    pure to

-- | Copies to the given directory file under given path. Returns the copied-to
-- path.
copyToDir :: (MonadIO m) => FilePath -> FilePath -> m FilePath
copyToDir destDir sourcePath = liftIO $ do
    createDirectoryIfMissing True destDir
    putStrLn $ "Copying " <> sourcePath <> " to " <> destDir
    let destPath = destDir </> takeFileName sourcePath
    copyFile sourcePath destPath
    pure destPath

-- | Retrieves the value of an environment variable, returning the provided
-- default if the requested variable was not set.
getEnvDefault :: (MonadIO m) => String -> String -> m String
getEnvDefault variableName defaultValue = liftIO $
    fromJust defaultValue <$> lookupEnv variableName

-- | Retrieves the value of an environment variable, throwing an exception if
-- the variable was not set.
getEnvRequired :: (MonadIO m) => String -> m String
getEnvRequired variableName = liftIO $ lookupEnv variableName >>= \case
    Just value -> pure value
    Nothing    -> error $ "required environment variable `" <> variableName <> "` is not set!"

-- | 'shortRelativePath' requires normalised paths to work correctly. this is
-- helper function because we don't want to bother with checking whether path is
-- normalised everywhere else
relativeNormalisedPath :: FilePath -> FilePath -> FilePath
relativeNormalisedPath (normalise -> p1) (normalise -> p2) = shortRelativePath p1 p2

-- | Conduit that passess data through, while applying on stateful, monadic
-- processor.
processChunk
    :: MonadIO m
    => s -- ^ initial state
    -> (s -> a -> IO s) -- ^ processor applied at chunk to obtain the new state
    -> ConduitM a a m ()
processChunk state processor = await >>= \case
    Nothing    -> pure ()
    Just chunk -> do
        newState <- liftIO $ processor state chunk
        yield chunk
        processChunk newState processor