{-# LANGUAGE RecordWildCards #-}
module Program.SevenZip.Internal where

import Prologue

import qualified Data.Attoparsec.Text as Attoparsec
import qualified Data.Text            as Text
import qualified Program              as Program
import qualified Progress             as Progress
import qualified System.Process       as Process

import Conduit              (decodeUtf8LenientC, foldC, mapM_C, (.|))
import Data.Attoparsec.Text (Parser)
import Data.Conduit.Process (sourceProcessWithStreams)
import Distribution.System  (OS (Windows), buildOS)
import Program              (Program)
import System.Exit          (ExitCode (ExitFailure, ExitSuccess))
import Text.Printf          (printf)



-----------------------
-- === Unpacking === --
-----------------------

-- === Definition === --

data UnpackProgressInfo = UnpackProgressInfo
    { percentProgress :: Int
    , currentName     :: Text
    } deriving (Show)

instance Progress.Progress UnpackProgressInfo where
    ratio p = Just $ convert (percentProgress p) / 100



------------------
-- === 7zip === --
------------------

-- === Definition === --

data SevenZip
instance Program SevenZip where
    -- On Windows installer by default does not add 7zip to PATH
    defaultLocations = pure ["C:\\Program Files\\7-Zip" | buildOS == Windows]

    -- Any of these is fine but usually only one is available
    executableNames = ["7z", "7za"]
    notFoundFixSuggestion = "please install from https://7-zip.org.pl/ or make sure that program is visible in PATH"



-- === Output parsing === --

-- | Parses "line" of 7z output, see 'parseProgress'
progressParser :: Parser UnpackProgressInfo
progressParser = UnpackProgressInfo
    <$> Attoparsec.decimal
    <*  Attoparsec.string "% "
    <*> Attoparsec.takeText
    <*  Attoparsec.endOfInput

-- | parses progress from text output by 7z. Expected chunks of texts are like:
-- "65% file.txt"
parseProgress :: Text -> Maybe UnpackProgressInfo
parseProgress chunk = hush $ parseChunk where
    parseChunk = Attoparsec.parseOnly progressParser chunk

retrieveProgressFromOutput :: Text -> [UnpackProgressInfo]
retrieveProgressFromOutput text = infos where
    -- Progress updates are not real lines, they are separated only by '\r'
    parts    = Text.split (flip elem ("\r\n" :: String)) text
    stripped = Text.strip <$> parts
    infos    = catMaybes $ parseProgress <$> stripped

processOutputChunk :: (UnpackProgressInfo -> IO ()) -> Text -> IO ()
processOutputChunk cb chunk = mapM_ cb infos
    where infos = retrieveProgressFromOutput chunk



-- === API === --

-- | Extract archive contents preserving full paths.
unpack :: (MonadIO m)
       => FilePath  -- ^ Archive to unpack
       -> FilePath  -- ^ Output directory
       -> m ()
unpack archive outputDirectory =
    call ExtractWithFullPaths
         [AssumeYes, OutputDirectory outputDirectory]
         archive
         []

-- | Adds files to archive, creating it if there was none.
pack :: (MonadIO m)
     => [FilePath] -- ^ Files to pack
     -> FilePath -- ^ Output archive
     -> m ()
pack packedPaths outputArchivePath =
    call Add [AssumeYes] outputArchivePath packedPaths


type Callback = Progress.Observer UnpackProgressInfo

-- | Similar to 'unpack' but allows tracking progress. All stdout and stderr
-- output is intercepted.
unpackWithProgress
    :: (MonadMask m, MonadThrow m, MonadIO m)
    => Callback -- ^ Callback
    -> FilePath -- ^ Archive to unpack
    -> FilePath -- ^ Output directory
    -> m ()
unpackWithProgress callback
    = flip Progress.runProgressible callback .: unpackWithProgress'

unpackWithProgress' :: (MonadThrow m, MonadIO m)
    => FilePath -> FilePath -> (UnpackProgressInfo -> IO ()) -> m ()
unpackWithProgress' archivePath outputDirectory callback = do
    program <- Program.get @SevenZip
    let command  =  ExtractWithFullPaths
    let switches = [ OverwriteMode OverwriteAll
                   , RedirectStream ProgressInformation RedirectToStdout
                   , SetCharset UTF8
                   , OutputDirectory outputDirectory
                   ]
    let procCfg  = Process.proc program
                 $ Program.format command
                <> Program.format switches
                <> [archivePath]

    let inputConduit = pure ()
    -- Note: even though we set the flag, 7z does not handle very well
    -- outputting UTF-8 text, at least on Windows. Might be worth to closer
    -- investigate in future. For now - we just need to be lenient and assume
    -- that filepaths we get in the progress update might be broken.
    let outConduit = decodeUtf8LenientC .| mapM_C (processOutputChunk callback)
    let errConduit = decodeUtf8LenientC .| foldC
    (result, out, err) <- liftIO
        $ sourceProcessWithStreams procCfg inputConduit outConduit errConduit
    case result of
        ExitSuccess      -> pure ()
        ExitFailure code -> throwM $ UnpackingException
            { _archivePath     = archivePath
            , _outputDirectory = outputDirectory
            , _stderr          = err
            , _exitCode        = code
            }



---------------------
-- === Command === --
---------------------

-- === Definition === --

-- Command list: https://sevenzip.osdn.jp/chm/cmdline/commands/index.htm
data Command
    = Add
    | ExtractWithFullPaths

instance Program.Argument Command where
    format' = \case
        Add                  -> "a"
        ExtractWithFullPaths -> "x"

-- https://sevenzip.osdn.jp/chm/cmdline/switches/index.htm
data Switch
    = OutputDirectory FilePath
    | AssumeYes
    | OverwriteMode OverwriteModeSwitch
    | RedirectStream StreamType StreamDestination
    | SetCharset Charset
    deriving (Show)

data OverwriteModeSwitch
    = OverwriteAll
    | SkipExisting
    | AutoRenameExtracted
    | AutoRenameExisting
    deriving (Show)


instance Program.Argument Switch where
    format = pure . \case
        OutputDirectory dir               -> "-o" <> dir
        AssumeYes                         -> "-y"
        OverwriteMode OverwriteAll        -> "-aoa"
        OverwriteMode SkipExisting        -> "-aos"
        OverwriteMode AutoRenameExtracted -> "-aou"
        OverwriteMode AutoRenameExisting  -> "-aot"
        RedirectStream str dest           -> "-bs" <> strFmt <> destFmt
            where
                strFmt = case str of
                    StandardOutput      -> "o"
                    ErrorOutput         -> "e"
                    ProgressInformation -> "p"
                destFmt = case dest of
                    DisableStream    -> "0"
                    RedirectToStdout -> "1"
                    RedirectToStderr -> "2"
        SetCharset charset                -> "-scc" <> charsetFmt
            where
                charsetFmt = case charset of
                    UTF8 -> "UTF-8"
                    WIN  -> "WIN"
                    DOS  -> "DOS"

data StreamType
    = StandardOutput
    | ErrorOutput
    | ProgressInformation
    deriving (Show)

data StreamDestination
    = DisableStream
    | RedirectToStdout
    | RedirectToStderr
    deriving (Show)

data Charset = UTF8 | WIN | DOS deriving (Show)



-- === API === --

-- | Helper function to call to Seven Zip with its strongly typed switches
call :: (MonadIO m) => Command -> [Switch] -> FilePath -> [String] -> m ()
call command switches baseArchive arguments =
    Program.call @SevenZip
        $  Program.format command
        <> Program.format switches
        <> [baseArchive]
        <> arguments

------------------------
-- === Exceptions === --
------------------------

data UnpackingException = UnpackingException
    { _archivePath     :: FilePath
    , _outputDirectory :: FilePath
    , _stderr          :: Text
    , _exitCode        :: Int
    } deriving (Show)
instance Exception UnpackingException where
    displayException UnpackingException{..} =
        printf "failed to unpack %s to %s: process returned code %d, stderr: %s"
               _archivePath _outputDirectory _stderr _exitCode
