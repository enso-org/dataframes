module Program.Curl where

import Control.Monad.IO.Class
import Program

data Curl
instance Program Curl where
    executableName = "curl"

download :: (MonadIO m) => String -> FilePath -> m ()
download url destPath = call @Curl ["-fSL", "-o", destPath, url]