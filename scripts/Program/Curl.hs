{-# LANGUAGE TypeApplications #-}

module Program.Curl where

import Program

data Curl
instance Program Curl where
    defaultLocations = ["C:\\Program Files\\7-Zip"]
    executableName = "curl"

download :: String -> FilePath -> IO ()
download url destPath = call @Curl ["-fSL", "-o", destPath, url]