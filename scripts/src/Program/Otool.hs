module Program.Otool where

import Control.Monad
import Data.List
import Data.String.Utils (strip)

import Program

data Otool
instance Program Otool where
    executableName = "otool"

usedLibraries :: FilePath -> IO [FilePath]
usedLibraries path = do
    (lines -> output) <- readProgram @Otool ["-L", path]
    when (null output) $ error "otool must output at least one line"

    -- Lines have a form like below:
    -- @rpath/libDataframeHelper.dylib (compatibility version 0.0.0, current version 0.0.0)
    -- heuristics is that we assume that name is before the '(' 
    let extractName = strip .  takeWhile (/= '(')
    let extractedNames = extractName <$> tail output -- Note: tail, because the first line contains the name of our executable
    -- print extractedNames/
    return extractedNames