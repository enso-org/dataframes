module Program.Tar where

import Prologue

import qualified Program as Program

import Program (Program)
    

data Tar
instance Program Tar where
    executableName = "tar"

data Format = GZIP | BZIP2 | XZ | LZMA

pack :: (MonadIO m) => [FilePath] -> FilePath -> Format -> m ()
pack pathsToPack outputArchivePath format = Program.call @Tar $ ["-c", formatArg, "-f", outputArchivePath] <> pathsToPack
    where formatArg = case format of
            GZIP  -> "-z"
            BZIP2 -> "-j"
            XZ    -> "-J"
            LZMA  -> "--lzma"