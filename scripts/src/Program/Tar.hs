module Program.Tar where

import Program

data Tar
instance Program Tar where
    executableName = "tar"

data Format = GZIP | BZIP2 | XZ | LZMA

pack pathsToPack outputArchivePath format = call @Tar $ ["-c", formatArg, "-f", outputArchivePath] <> pathsToPack
    where formatArg = case format of
            GZIP  -> "-z"
            BZIP2 -> "-j"
            XZ    -> "-J"
            LZMA  -> "--lzma"