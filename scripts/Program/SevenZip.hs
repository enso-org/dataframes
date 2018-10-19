{-# LANGUAGE TypeApplications #-}

module Program.SevenZip where

import Program

data SevenZip
instance Program SevenZip where
    defaultLocations = ["C:\\Program Files\\7-Zip"]
    executableName = "7z"
    notFoundError = "cannot find 7z, please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"

unpack :: FilePath -> FilePath -> IO ()
unpack archive outputDirectory =
    call @SevenZip ["x", "-y", "-o" <> outputDirectory, archive]

pack :: [FilePath] -> FilePath -> IO ()
pack packedPaths outputArchivePath =
    call @SevenZip $ ["a", "-y", outputArchivePath] <> packedPaths
