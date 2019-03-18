module Program.Otool where

import Prologue

import qualified Program as Program

import Data.String.Utils (strip)
import Program           (Program)


data Otool
instance Program Otool where
    executableName = "otool"

usedLibraries :: (MonadIO m) => FilePath -> m [FilePath]
usedLibraries path = do
    (lines -> output :: [String]) <- Program.read @Otool ["-L", path]
    let outputTail = fromJust (error "otool must output at least one line") $ tail output

    -- Lines have a form like below:
    -- @rpath/libDataframeHelper.dylib (compatibility version 0.0.0, current version 0.0.0)
    -- heuristics is that we assume that name is before the '('
    let extractName = strip . takeWhile (/= '(')
    let extractedNames = extractName <$> outputTail -- Note: tail, because the first line contains the name of our executable
    -- print extractedNames/
    pure extractedNames
