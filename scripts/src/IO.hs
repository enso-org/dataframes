{-# OPTIONS -cpp #-}

-- | This module provides custom IO functions that support outputting non-ascii
--   characters on Windows console.
module IO where

import Prologue


import System.IO          (Handle, hPutStr, stdout)

#ifdef mingw32_HOST_OS
--------------------------------------------------------------------------------
import Data.Text          (Text)
import Data.Text.Foreign  (useAsPtr)
import Foreign.Ptr        (Ptr)
import System.Win32.Types (HANDLE, withHandleToHANDLE)

foreign import ccall unsafe "writeText" writeTextC :: Ptr Word16 -> Int32 -> HANDLE -> IO Int64

hPutText :: (MonadIO m) => Handle -> Text -> m ()
hPutText handle text = liftIO $ void $ useAsPtr text $ \ptr length -> do
    withHandleToHANDLE handle $
        writeTextC ptr (fromIntegral length)
--------------------------------------------------------------------------------
#else
--------------------------------------------------------------------------------
hPutText :: (MonadIO m) => Handle -> Text -> m ()
hPutText handle text = liftIO $ hPutStr handle $ convert text
--------------------------------------------------------------------------------
#endif

hPutTextLn :: (MonadIO m) => Handle -> Text -> m ()
hPutTextLn handle text = do
    hPutText handle text
    hPutText handle "\n"

putText :: (MonadIO m) => Text -> m ()
putText = hPutText stdout

putTextLn :: (MonadIO m) => Text -> m ()
putTextLn = hPutTextLn stdout

