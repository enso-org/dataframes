{- 
Wrapper over Resource Hacker program — a Windows-only, freeware program allowing editing resources in executables. We use it to set version info.
-}

{-# LANGUAGE RecordWildCards #-}

module Program.ResourceHacker where

import Prologue

import qualified Data.Text            as Text
import qualified Program              as Program
import qualified System.Process.Typed as Process
import qualified Utils                as Utils

import Data.List           (isInfixOf)
import Distribution.System (OS (Windows), buildOS)
import NeatInterpolation   (text)
import System.FilePath     ((</>), replaceExtension, takeFileName)
import System.IO.Temp      (withSystemTempDirectory)

data ResourceHacker
instance Program.Program ResourceHacker where
    defaultLocations = ["C:\\Program Files (x86)\\Resource Hacker" | buildOS == Windows]
    executableName = "ResourceHacker"
    -- | All calls shall use custom 'formatShellCommand' and use shell-style calls, see 'formatShellCommand'.
    proc program args = Process.shell $ formatShellCommand program args
    
-- | Compiles .rc resource script file into binary resource file (.res).
compile :: (MonadIO m) 
        => FilePath -- ^ Input resource script file
        -> FilePath -- ^ Output binary resource file
        -> m ()
compile rcPath resPath = call rcPath resPath ["-action", "compile"]

-- | Adds resource to an executable file, overwriting if one already exists.
addoverwrite :: (MonadIO m) 
             => FilePath -- ^ Executable file to add resource to. 
             -> FilePath -- ^ Output executable file (can be the same path).
             -> FilePath -- ^ Binary resource file (.res).
             -> m ()
addoverwrite srcExe dstExe resPath = call srcExe dstExe ["-action", "addoverwrite", "-resource", resPath]

-- | Writes given version information on executable.
setVersion :: (MonadIO m, MonadMask m) 
           => FilePath -- ^ Executable to patch.
           -> FilePath -- ^ Output path for patched executable
           -> VersionInfo  -- ^ Information to be written.
           -> m ()
setVersion srcExe dstExe version = withSystemTempDirectory "" $ \tmpDir -> do
    let rcFile = tmpDir </> "version.rc"
    let resFile = replaceExtension rcFile "res"
    writeFile rcFile (convert $ versionInfoRC version)
    compile rcFile resFile
    addoverwrite srcExe dstExe resFile
    
-- | Note: Resource Hacker seems to be extremely fragile in handling command 
-- arguments. For some reason wrapping its arguments in quotes (as does process
-- library) causes it to fail. Only paths can be quoted and it is necessary
-- when path contains spaces.
-- Because of that we call it by shell command formatted by function below.
formatShellCommand :: FilePath -- ^ Path to resource hacker program,
                   -> [String] -- ^ Arguments to call with.
                   -> String
formatShellCommand resHackerPath args = intercalate " " partsQuoted where
    partsQuoted = quoteIfNeeded <$> resHackerPath : args
    quoteIfNeeded p = if  " " `isInfixOf` p
        then "\"" <> p <> "\""
        else p

-- | Helper function that runs specified command with given input and output settings, logging output to console.
call :: MonadIO m
     => FilePath -- ^ Input file
     -> FilePath -- ^ Output file
     -> [String] -- ^ Command-specific arguments
     -> m ()
call openFile saveFile command = Program.call @ResourceHacker $
       [ "-open", openFile, "-save", saveFile]
    <> command
    <> ["-log", "CONSOLE"]

-- | Generate resource script .rc file contents for given version information.
versionInfoRC :: VersionInfo -> Text
versionInfoRC VersionInfo{..} = [text|
1 VERSIONINFO
    FILEVERSION    $commaVersion
    PRODUCTVERSION $commaVersion
{
    BLOCK "StringFileInfo"
    {
        BLOCK "040904b0"
        {
            VALUE "CompanyName",        "$(_companyName)"
            VALUE "FileDescription",    "$_fileDescription"
            VALUE "FileVersion",        "$dotVersion"
            VALUE "LegalCopyright",     "$_legalCopyright"
            VALUE "OriginalFilename",   "$_originalFilename"
            VALUE "ProductName",        "$_productName"
            VALUE "ProductVersion",     "$dotVersion"
        }
    }
    BLOCK "VarFileInfo"
    {
        VALUE "Translation", 0x409, 1200
    }
}
|] where 
    commaVersion = prettyPrintVersion 4 ","
    dotVersion   = prettyPrintVersion 4 "."
    prettyPrintVersion count separator = Text.intercalate separator 
                                       $ show' <$> versionParts count
    -- If too few version components were given, fill with zeroes.
    versionParts count = take count $ _version <> repeat 0
-- * Version info

versionInfo :: [Int] -> FilePath -> Text -> Text -> Text -> VersionInfo
versionInfo version exeFile name fileDescription companyName = VersionInfo
    { _version = version
    , _companyName = companyName
    , _fileDescription =  fileDescription
    , _legalCopyright = "Copyright (c) 2019 " <> companyName
    , _originalFilename = convert $ takeFileName exeFile
    , _productName = name
    }
    
data VersionInfo = VersionInfo 
    { _version :: [Int] -- ^ Currently both product and component version (assumed to be the same). Up to 4 numbers are used. Each value should be representible as 16-bit integer.
    , _companyName :: Text
    , _fileDescription :: Text
    , _legalCopyright :: Text
    , _originalFilename :: Text
    , _productName :: Text
    } deriving (Generic, Show, Eq, Ord)

makeLenses ''VersionInfo
-- ttt :: IO ()
-- ttt = do
--     let stack = "C:\\Users\\mwu\\AppData\\Roaming\\local\\bin\\stack.exe"
--     let stackCopy = "C:\\temp\\stack.exe"
--     let version = versionInfo [1,8] stackCopy "stack" "tool for building things" "stack authors"
--     setVersion stack stackCopy version
--     pure ()

    