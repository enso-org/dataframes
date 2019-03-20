module Network where

    
import Prologue

import qualified Progress as Progress
-- import Text.Printf
-- import Data.Maybe
-- import Data.IORef
import qualified Data.ByteString as ByteString
import qualified Data.ByteString.Char8 as ByteStringChar8
import Data.ByteString (ByteString)
-- import Text.Read

-- import Data.Conduit.Binary (sinkFile) -- Exported from the package conduit-extra
import Network.HTTP.Conduit (Response, responseHeaders, http, newManager, parseRequest, responseBody, tlsManagerSettings)
-- import Network.HTTP.Types
import Conduit (ConduitM, (.|), await, yield, runConduit, runResourceT, sinkFile)
-- import Control.Monad.Trans.Resource
import Network.HTTP.Types.Header (hContentLength, hContentDisposition)
-- import qualified Control.Monad.State.Layered as State
-- import Data.Conduit.Process

data DownloadProgress = DownloadProgress 
    { _bytesCompleted :: Int
    , _bytesTotal :: Maybe Int 
    } deriving (Show)
makeLenses ''DownloadProgress

instance Progress.Progress DownloadProgress where
    -- progressRatio :: DownloadProgress -> Maybe Float
    ratio p = do
        total        <- fromIntegral <$> p ^. bytesTotal
        let completed = fromIntegral  $  p ^. bytesCompleted
        pure $ completed / total

type ProgressCallback = DownloadProgress -> IO ()

advanceProgress :: DownloadProgress -> ByteString -> DownloadProgress
advanceProgress p chunk = p & bytesCompleted %~ (+ ByteString.length chunk)

updateProgress :: MonadIO m => ProgressCallback -> DownloadProgress -> ConduitM ByteString ByteString m ()
updateProgress cb previousProgress = await >>= maybe (pure ()) (\chunk -> do
    let newProgress = advanceProgress previousProgress chunk
    liftIO $ cb newProgress
    yield chunk
    updateProgress cb newProgress)

contentLength :: Response body -> Maybe Int
contentLength response = do
    field <- lookup hContentLength (responseHeaders response)
    let parts = unsafeRead (ByteStringChar8.unpack field) 
    pure parts

-- contentFilename :: Response body -> _
contentFilename response = do
    lookup hContentDisposition (responseHeaders response)

downloadFileTo :: MonadIO m => ProgressCallback -> String -> FilePath -> m ()
downloadFileTo cb url targetPath = liftIO $ do
    request <- parseRequest url
    manager <- newManager tlsManagerSettings
    -- putStrLn $ "downloading " <> url
    runResourceT $ do
        response <- http request manager
        liftIO $ print $ responseHeaders response
        liftIO $ print $ contentFilename response
        let initialProgress = DownloadProgress 0 (contentLength response)
        runConduit $ responseBody response 
                  .| updateProgress cb initialProgress
                  .| sinkFile targetPath
    -- putStrLn $ "downloading done"
    