{-|
Description : Utilities specific to Unix-like systems, like Linux od macOS.
-}

module Platform.Unix where

import Prologue

import Control.Exception        (bracket)
import System.PosixCompat.Files (fileMode, getFileStatus, ownerWriteMode,
                                    setFileMode, unionFileModes)


-- === API === --

-- | Executes action while temporarily changing file mode to make it writable to
--   owner. Restores file mode before returning. Will fail if the file is not
--   owned.
withWritableFile :: MonadIO m => FilePath -> IO () -> m ()
withWritableFile path action 
    = liftIO $ bracket makeWritable restoreMode (const action) where
    makeWritable = do
        oldStatus <- getFileStatus path
        setFileMode path $ unionFileModes (fileMode oldStatus) ownerWriteMode
        pure oldStatus
    restoreMode oldStatus = setFileMode path (fileMode oldStatus)
