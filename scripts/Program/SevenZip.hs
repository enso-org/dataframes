module Program.SevenZip where

import Distribution.System

import Program

data SevenZip
instance Program SevenZip where
    defaultLocations = ["C:\\Program Files\\7-Zip" | buildOS == Windows] -- Installer by default does not add 7zip to path, with this it "just works"
    executableNames = ["7z", "7za"] -- Any of these is fine but usually only one is available
    notFoundFixSuggestion = "please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"

unpack :: FilePath -> FilePath -> IO ()
unpack archive outputDirectory =
    call @SevenZip ["x", "-y", "-o" <> outputDirectory, archive]

pack :: [FilePath] -> FilePath -> IO ()
pack packedPaths outputArchivePath =
    call @SevenZip $ ["a", "-y", outputArchivePath] <> packedPaths
