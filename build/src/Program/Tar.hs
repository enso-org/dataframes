module Program.Tar where

import Prologue

import qualified Program as Program
import qualified System.Process       as Process
import qualified Data.ByteString.Lazy.Char8 as BS8

import Conduit              (linesUnboundedAsciiC, sinkNull, (.|), ConduitM, sinkList)
import Data.Conduit.Process (sourceProcessWithStreams)
import Program              (Program)

import Data.ByteString (ByteString)
import System.Exit          (ExitCode (ExitFailure, ExitSuccess))

import qualified Progress as Progress
import qualified Utils as Utils


data Tar
instance Program Tar where
    executableName = "tar"


----------------------
-- === Command  === --
----------------------
    
data Format = GZIP | BZIP2 | XZ | LZMA deriving (Show, Eq)

instance Program.Argument Format where
    format = \case
        GZIP  -> ["-z"]
        BZIP2 -> ["-j"]
        XZ    -> ["-J"]
        LZMA  -> ["--lzma"]

data Command = Create | Append | List | Extract deriving (Show, Eq)
instance Program.Argument Command where
    format = \case
        Create  -> ["-c"]
        Append  -> ["-r"]
        List    -> ["-t"]
        Extract -> ["-x"]

data Switch 
    = TargetFile FilePath
    | Verbose
    | UseFormat Format
instance Program.Argument Switch where
    format = \case
        TargetFile path -> ["-f", path]
        Verbose         -> ["-v"]
        UseFormat f     -> Program.format f

callArgs :: Command -> [Switch] -> FilePath -> [FilePath] -> [String]
callArgs cmd switches archivePath targets = 
           Program.format cmd
        <> (Program.format $ TargetFile archivePath)
        <> Program.format switches
        <> targets

call :: MonadIO m => Command -> [Switch] -> FilePath -> [FilePath] -> m ()
call = Program.call @Tar .:: callArgs



-----------------------
-- === Progress  === --
-----------------------

data UnpackProgressInfo = UnpackProgressInfo
    { doneFiles :: Int
    , allFiles  :: Int
    } deriving (Show)


updateChunk 
    :: MonadIO m 
    => (UnpackProgressInfo -> IO ()) -- ^ Callback.
    -> Int -- ^ Total file count.
    -> Int -- ^ Files processed so far.
    -> ConduitM ByteString ByteString m ()
updateChunk callback totalCount currentCount = 
    Utils.processChunk 1 $ \currentCount _ -> do
        liftIO $ callback $ UnpackProgressInfo currentCount totalCount
        pure $ currentCount + 1

instance Progress.Progress UnpackProgressInfo where
    ratio UnpackProgressInfo{..} = Just $ fromIntegral doneFiles / fromIntegral allFiles


------------------
-- === API  === --
-------------------

pack :: (MonadIO m) => [FilePath] -> FilePath -> Format -> m ()
pack pathsToPack archivePath format = 
    call Create [UseFormat format] archivePath pathsToPack

-- | Returns the number of file in the archive
fileCount :: (MonadIO m) => FilePath -> m Int
fileCount path = do
    (result, _) <- Program.read' @Tar $ callArgs List [] path []
    pure $ length $ BS8.lines result

-- | Unpack all files from given archive into directory.
unpack 
    :: (MonadIO m) 
    => FilePath -- ^ Output directory
    -> FilePath -- ^ Archive
    -> m ()
unpack outputDirectory archivePath = 
    call Extract [] archivePath []

-- | Unpack all files from given archive into directory.
unpackWithProgress
    :: (MonadIO m, MonadMask m)
    => Progress.Observer UnpackProgressInfo -- ^ callback
    -> FilePath -- ^ Output directory
    -> FilePath -- ^ Archive
    -> m ()
unpackWithProgress callback = Progress.runProgressible callback .: unpackWithProgress'

-- | Unpack all files from given archive into directory.
unpackWithProgress'
    :: (MonadIO m) 
    => FilePath -- ^ Output directory
    -> FilePath -- ^ Archive
    -> (UnpackProgressInfo -> IO ()) -- ^ callback
    -> m ()
unpackWithProgress' outputDirectory archivePath callback = do
    count    <- fileCount archivePath
    procCfg1 <- Program.prog' @Tar $ callArgs Extract [Verbose] archivePath []
    let procCfg = procCfg1 { Process.cwd = Just outputDirectory }
    let inputConduit = pure ()
    let outConduit = sinkNull
    let errConduit = linesUnboundedAsciiC .| updateChunk callback count 1 .| sinkList
    
    (result, out, err) <- liftIO
        $ sourceProcessWithStreams procCfg inputConduit outConduit errConduit
    print err
    case result of
        ExitSuccess      -> pure ()
        -- TODO report proper error, including "err" in payload (it is where actual error message will be placed)
        ExitFailure code -> error $ "failed to unpack " <> archivePath <> " to " <> outputDirectory