module Archive where

import Prologue

import qualified Program.SevenZip as SevenZip
import qualified Program.Tar      as Tar

import System.Directory (makeAbsolute, makeRelativeToCurrentDirectory, withCurrentDirectory)
import System.FilePath  (takeDirectory, takeExtension)


-- | Helper that does two things:
-- 
-- 1. use file extension to deduce compression method
-- 
-- 2. switch CWD so tar shall pack the folder at archive's root
--    (without maintaining directory's absolute path in archive)
packDirectory :: MonadIO m => FilePath -> FilePath -> m ()
packDirectory pathToPack outputArchive = liftIO $ do
    -- As we switch cwd, relative path to output might get affected.
    -- Let's store it as absolute path first.
    outputArchiveAbs <- makeAbsolute outputArchive
    withCurrentDirectory (takeDirectory pathToPack) $ do
        -- Input path must be relative though.
        pathToPackRel <- makeRelativeToCurrentDirectory pathToPack
        let tarPack =  Tar.pack [pathToPackRel] outputArchiveAbs
        case takeExtension outputArchive of
            ".7z"   -> SevenZip.pack [pathToPack] outputArchiveAbs
            ".gz"   -> tarPack Tar.GZIP
            ".bz2"  -> tarPack Tar.BZIP2
            ".xz"   -> tarPack Tar.XZ
            ".lzma" -> tarPack Tar.LZMA
            _       -> fail $ "packDirectory: cannot deduce compression algorithm from extension: " <> takeExtension outputArchive
