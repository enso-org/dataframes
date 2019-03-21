module Network where


import Prologue

import qualified Data.Attoparsec.ByteString.Char8 as Attoparsec
import qualified Data.ByteString                  as ByteString
import qualified Data.ByteString.Char8            as ByteStringChar8
import qualified Progress                         as Progress

import Conduit                   (ConduitM, await, runConduit, runResourceT,
                                  sinkFile, yield, (.|))
import Data.ByteString           (ByteString)
import Network.HTTP.Conduit      (Response, http, newManager, parseRequest,
                                  responseBody, responseHeaders,
                                  tlsManagerSettings)
import Network.HTTP.Types.Header (hContentDisposition, hContentLength)



----------------------
-- === Progress === --
----------------------

-- === Definition === --

data DownloadProgress = DownloadProgress
    { _bytesCompleted :: Int
    , _bytesTotal     :: Maybe Int
    } deriving (Show)
makeLenses ''DownloadProgress

type ProgressCallback = DownloadProgress -> IO ()


-- === Instances === --

instance Progress.Progress DownloadProgress where
    ratio :: DownloadProgress -> Maybe Float
    ratio p = do
        total        <- fromIntegral <$> p ^. bytesTotal
        let completed = fromIntegral  $  p ^. bytesCompleted
        pure $ completed / total


-- === API === --

advanceProgress :: ByteString -> DownloadProgress -> DownloadProgress
advanceProgress chunk = bytesCompleted %~ (+ ByteString.length chunk)

updateProgress
    :: MonadIO m
    => ProgressCallback
    -> DownloadProgress
    -> ConduitM ByteString ByteString m ()
updateProgress callback previousProgress = await >>= \case
    Nothing    -> pure ()
    Just chunk -> do
        let newProgress = advanceProgress chunk previousProgress
        liftIO $ callback newProgress
        yield chunk
        updateProgress callback newProgress

parseLength :: ByteString -> Maybe Int
parseLength bs = do
    let parser = Attoparsec.decimal <* Attoparsec.endOfInput
    let result = Attoparsec.parseOnly parser bs
    hush result

contentLength :: Response body -> Maybe Int
contentLength response = do
    field <- lookup hContentLength (responseHeaders response)
    parseLength field

downloadFileTo
    :: (MonadMask m, MonadIO m)
    => String   -- ^ URL to download
    -> FilePath -- ^ Location to store the file
    -> Progress.Observer DownloadProgress
    -> m ()
downloadFileTo url targetPath
    = Progress.runProgressible $ downloadFileTo' url targetPath

downloadFileTo'
    :: MonadIO m
    => String
    -> FilePath
    -> (DownloadProgress -> IO ())
    -> m ()
downloadFileTo' url targetPath cb = liftIO $ do
    request <- parseRequest url
    manager <- newManager tlsManagerSettings
    runResourceT $ do
        response <- http request manager
        let initialProgress = DownloadProgress 0 (contentLength response)
        runConduit $ responseBody response
                  .| updateProgress cb initialProgress
                  .| sinkFile targetPath
