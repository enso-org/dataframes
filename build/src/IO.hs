{-# OPTIONS -cpp #-}

-- | This module provides custom IO functions that support outputting non-ascii
--   characters on Windows console. When writing text to handles to the console,
--   functions from this module should be used to prevent crashes on non-ascii
--   characters.
--
--   The 'hPutStr' and 'hPutText' take arbitrary handle and if it points to a
--   Windows console buffer use special 'writeConsole' function, otherwise
--   falling back to base's `System.IO.hPutStr'.
module IO where

import Prologue


import System.IO          (Handle, hPutStr, stdout)

#ifdef mingw32_HOST_OS
--------------------------------------------------------------------------------

import qualified Data.Text.IO as Text
import Data.Text          (Text)
import Data.Text.Foreign  (useAsPtr)
import Foreign.Ptr        (Ptr)
import System.Win32.Types (HANDLE, withHandleToHANDLE)

-- The two functions below are defined in cbits/Console.cpp. They are simple
-- wrappers over Winapi.
foreign import ccall unsafe "writeConsole" writeConsoleC :: Ptr Word16 -> Int32 -> HANDLE -> IO Int64
foreign import ccall unsafe "isConsole" isConsoleC :: HANDLE -> IO Bool

-- | Checks if this is a handle to the Windows console screen buffer.
isConsole :: (MonadIO m) => Handle -> m Bool
isConsole handle = liftIO <$> withHandleToHANDLE handle $ isConsoleC

-- | Writes given text to the given handle. The handle must point to a console
--   screen buffer. This function allows reliably outputting non-ascii
--   characters onto Windows terminal.
--
-- Returns the number of written characters on success or -1 on failure.
writeConsole :: (MonadIO m) => Handle -> Text -> m Int64
writeConsole handle text = liftIO $
    -- possible improvement here: useAsPtr copies the text, though we do
    -- read-only operation, so copy should not be needed
    useAsPtr text $ \textPtr textLength -> do
        withHandleToHANDLE handle $ do
            writeConsoleC textPtr (fromIntegral textLength)

hPutText :: (MonadIO m) => Handle -> Text -> m ()
hPutText handle text = isConsole handle >>= \case
    True  -> void $ writeConsole handle text
    False -> liftIO $ Text.hPutStr handle text

hPutStr :: (MonadIO m) => Handle -> String -> m ()
hPutStr handle text = hPutText handle $ convert text

--------------------------------------------------------------------------------
#else
--------------------------------------------------------------------------------
hPutText :: (MonadIO m) => Handle -> Text -> m ()
hPutText handle text = IO.hPutStr handle $ convert text

hPutStr :: (MonadIO m) => Handle -> String -> m ()
hPutStr = fmap liftIO <$> System.IO.hPutStr
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
