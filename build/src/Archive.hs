
-- | Common interface for dealing with archives using @tar@ and @7z@ programs.
module Archive where

import Prologue

import qualified Program.SevenZip as SevenZip
import qualified Program.Tar      as Tar
import qualified Progress         as Progress

import System.Directory (makeAbsolute, makeRelativeToCurrentDirectory,
                         withCurrentDirectory)
import System.FilePath  (takeDirectory, takeExtension)

-- | Possible archive formats recognized by this module.
data ArchiveFormat
    = SevenZipFormat
    | TarFormat Tar.Format
    | UnknownFormat
    deriving (Show, Eq)

-- | Deduces archive format from its filename extension. Supported extensions:
--
--   * 7z
--   * gz
--   * bz2
--   * xz
--   * lzma
deduceFormat :: FilePath -> ArchiveFormat
deduceFormat path = case takeExtension path of
    ".7z"   -> SevenZipFormat
    ".gz"   -> TarFormat Tar.GZIP
    ".bz2"  -> TarFormat Tar.BZIP2
    ".xz"   -> TarFormat Tar.XZ
    ".lzma" -> TarFormat Tar.LZMA
    _       -> UnknownFormat

-- | Packs given directory to the archive. Compression method shall be deduced
--   from the output file extension (see 'deduceFormat'). Depends on `Tar.Tar`
--   or `SevenZip.SevenZip` programs being available (and their relevant
--   dependencies).
--
--   The packed folder will be treated as an archive root (absolute paths won't
--   be preserved).
packDirectory :: MonadIO m
    => FilePath -- ^ Directory to be packed
    -> FilePath -- ^ Path where to create archive
    -> m ()
packDirectory pathToPack outputArchive = liftIO $ do
    -- We do some trickery with cwd to avoid storing absolute paths in the
    -- archive. It is needed only for @tar@. @7z@ does not really need it, but
    -- for simplicity we treat them the same way.
    --
    -- As we switch cwd, relative path to output might get affected. Let's store
    -- it as absolute path first.
    outputArchiveAbs <- makeAbsolute outputArchive
    withCurrentDirectory (takeDirectory pathToPack) $ do
        -- Input path must be relative.
        pathToPackRel <- makeRelativeToCurrentDirectory pathToPack
        let tarPack =  Tar.pack [pathToPackRel] outputArchiveAbs
        case deduceFormat outputArchive of
            SevenZipFormat -> SevenZip.pack [pathToPack] outputArchiveAbs
            TarFormat format -> tarPack Tar.GZIP
            UnknownFormat    -> fail $
                "packDirectory: cannot deduce compression algorithm from extension: "
             <> takeExtension outputArchive

-- | Extracts archive contents into given output directory.
unpack :: MonadIO m
    => FilePath -- ^ Archive to be unpacked
    -> FilePath -- ^ Output directory
    -> m ()
unpack archivePath outputDirectory = case deduceFormat archivePath of
        SevenZipFormat   -> SevenZip.unpack archivePath outputDirectory
        -- Well, actually @7z@ can extract tar.* archives as well. However, it
        -- cannot do it in a single step (separate decompress / untar steps), so
        -- we prefer using the actual @tar@ tool. In future we might consider
        -- using @7z@ as a fallback when @tar@ is not available on the machine.
        TarFormat format -> Tar.unpack outputDirectory archivePath
        UnknownFormat    -> liftIO $ fail $
            "unpack: cannot deduce compression algorithm from extension: "
         <> takeExtension archivePath

-- TODO: wrapper for unpacking with progress tracking