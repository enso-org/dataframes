module Program.SevenZip where

import Prologue

import qualified Program as Program

import Distribution.System (OS (Windows), buildOS)
import Program             (Program)

data SevenZip
instance Program SevenZip where
    -- On Windows installer by default does not add 7zip to PATH
    defaultLocations = ["C:\\Program Files\\7-Zip" | buildOS == Windows]

    -- Any of these is fine but usually only one is available
    executableNames = ["7z", "7za"] 
    notFoundFixSuggestion = "please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"

-- https://sevenzip.osdn.jp/chm/cmdline/commands/index.htm
data Command = 
      Add 
    | ExtractWithFullPaths
instance Program.Argument Command where
    format' = \case
        Add                  -> "a"
        ExtractWithFullPaths -> "x"

-- https://sevenzip.osdn.jp/chm/cmdline/switches/index.htm
data Switch = 
      OutputDirectory FilePath
    | AssumeYes
instance Program.Argument Switch where
    format' = \case
        OutputDirectory dir -> "-o" <> dir
        AssumeYes           -> "-y"

-- | Extract archive contents preserving full paths.
unpack :: (MonadIO m) 
       => FilePath  -- ^ Archive to unpack
       -> FilePath  -- ^ Output directory
       -> m ()
unpack archive outputDirectory = 
    call 
        ExtractWithFullPaths 
        [AssumeYes, OutputDirectory outputDirectory] 
        archive 
        []

-- | Adds files to archive, creating it if there was none.
pack :: (MonadIO m)
     => [FilePath] -- ^ Files to pack
     -> FilePath -- ^ Output archive
     -> m ()
pack packedPaths outputArchivePath =
    call 
        Add
        [AssumeYes]
        outputArchivePath
        packedPaths

-- | Helper function to call to Seven Zip with its strongly typed switches
call :: (MonadIO m) => Command -> [Switch] -> FilePath -> [String] -> m ()
call command switches baseArchive arguments = 
    Program.call @SevenZip 
        $  Program.format command
        <> Program.format switches
        <> [baseArchive] 
        <> arguments