module Program.Curl where

import Program

data Curl
instance Program Curl where
    executableName = "curl"

download :: String -> FilePath -> IO ()
download url destPath = call @Curl ["-fSL", "-o", destPath, url]