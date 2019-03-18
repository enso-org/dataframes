module Program.Curl where

import Prologue

import qualified Program as Program

import Program (Program)

data Curl
instance Program Curl where
    executableName = "curl"

download :: (MonadIO m) => String -> FilePath -> m ()
download url destPath = Program.call @Curl ["-fSL", "-o", destPath, url]
