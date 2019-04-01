module Program.Ldd where


import Prologue

import qualified Data.Set  as Set
import qualified Data.Text as T
import qualified Program   as Program

import Control.Monad    (filterM)
import Program          (Program)
import System.Directory (doesFileExist)

data Ldd
instance Program Ldd where
    executableName = "ldd"

-- import qualified Data.ByteString.Lazy as BSL
type Entry = (String, String, String)

-- String that separates .so name and its absolute path
arrow :: Text
arrow = " => "

-- eg. called with "libgraphite2.so.3 => /lib64/libgraphite2.so.3"
-- returns ("libgraphite2.so.3", "/lib64/libgraphite2.so.3")
-- but if no " => " is present, then single path is assumed and returned
-- as first (relative path) or second element (absolute path)
-- Note: the point is not to get tricked with irregular vdso and ld.so entries.
parseNameAndPath :: Text -> (Text, Text)
parseNameAndPath (T.strip -> input) =
        if T.null afterArrow then fromSingle else (beforeArrow, afterArrow)
    where
        (beforeArrow, (T.drop (T.length arrow) -> afterArrow)) = T.breakOn arrow input
        fromSingle = if T.head input == '/' then ("", input) else (input, "") -- FIXME unsafe

-- eg. called with "libc.so.6 => /lib64/libc.so.6 (0x00007f87e4e92000)"
-- returns: ("libc.so.6","/lib64/libc.so.6","(0x00007f87e4e92000)")
parseNamePathAddress :: Text -> (Text, Text, Text)
parseNamePathAddress (T.strip -> input) =
        if hasAddress then
            (p1, p2, afterLastSpace)
        else
            (p1, p2, "")
    where
        (beforeLastSpace, afterLastSpace) = T.breakOnEnd " " input
        hasAddress = T.isPrefixOf "(0x" afterLastSpace -- heuristics, in theory a file path can contain " (0x" — but if there is a path, there should be an address later anyway
        (p1, p2) = parseNameAndPath 
            $ if hasAddress 
                then beforeLastSpace 
                else input

-- Returns list of shared libraries that are necessary to load given binary
-- executable image.
-- Note: this are not necessarily all the dependencies.
dependenciesOfBinary :: (MonadIO m) => FilePath -> m [FilePath]
dependenciesOfBinary binary = do
    lddOutput <- T.pack <$> Program.read @Ldd [binary]
    let parsedLibraryInfo = parseNamePathAddress <$> T.lines lddOutput
    pure $ (\(_,b,_) -> T.unpack b) <$> parsedLibraryInfo

-- Returns list of shared libraries that are necessary to load given binary
-- executable images.
-- Note: this are not necessarily all the dependencies.
dependenciesOfBinaries :: (MonadIO m) => [FilePath] -> m [FilePath]
dependenciesOfBinaries binaries = do
    listOfListOfDeps <- mapM dependenciesOfBinary binaries
    -- get rid of duplicates by passing through set
    let listOfDeps = Set.toList $ Set.unions $ Set.fromList <$> listOfListOfDeps
    filterM (liftIO <$> doesFileExist) listOfDeps
