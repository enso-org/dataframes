module Program.Ldd where

import Control.Applicative
import Control.Monad
import Data.Maybe
import Data.Monoid
-- import Data.Text.Encoding (decodeUtf8)
import Data.String.Utils
import System.FilePath
import System.Directory
import System.Process
import Text.Printf

import Program

import qualified Data.Text.Lazy as TL
import Data.Text.Lazy.Encoding (decodeUtf8)

import Data.Set (Set)
import qualified Data.Set as Set

data Ldd
instance Program Ldd where
    executableName = "ldd"

-- import qualified Data.ByteString.Lazy as BSL
type Entry = (String, String, String)

-- String that separates .so name and its absolute path
arrow = " => " :: TL.Text

-- eg. called with "libgraphite2.so.3 => /lib64/libgraphite2.so.3"
-- returns ("libgraphite2.so.3", "/lib64/libgraphite2.so.3")
-- but if no " => " is present, then single path is assumed and returned
-- as first (relative path) or second element (absolute path)
-- Note: the point is not to get tricked with irregular vdso and ld.so entries.
parseNameAndPath :: TL.Text -> (TL.Text, TL.Text)
parseNameAndPath (TL.strip -> input) =
        if TL.null afterArrow then fromSingle else (beforeArrow, afterArrow)
    where
        (beforeArrow, (TL.drop (TL.length arrow) -> afterArrow)) = TL.breakOn arrow input
        fromSingle = if TL.head input == '/' then ("", input) else (input, "")

-- eg. called with "libc.so.6 => /lib64/libc.so.6 (0x00007f87e4e92000)"
-- returns: ("libc.so.6","/lib64/libc.so.6","(0x00007f87e4e92000)")
parseNamePathAddress :: TL.Text -> (TL.Text, TL.Text, TL.Text)
parseNamePathAddress (TL.strip -> input) =
        if hasAddress then
            (p1, p2, afterLastSpace)
        else
            (p1, p2, "")
    where
        (beforeLastSpace, afterLastSpace) = TL.breakOnEnd " " input
        hasAddress = TL.isPrefixOf "(0x" afterLastSpace -- heuristics, in theory a file path can contain " (0x" — but if there is a path, there should be an address later anyway
        (p1, p2) = parseNameAndPath $ if hasAddress then beforeLastSpace else input

-- Returns list of shared libraries that are necessary to load given binary
-- executable image.
-- Note: this are not necessarily all the dependencies.
dependenciesOfBinary :: FilePath -> IO [FilePath]
dependenciesOfBinary binary = do
    lddOutput <- TL.pack <$> readProgram @Ldd [binary]
    let parsedLibraryInfo = parseNamePathAddress <$> TL.lines lddOutput
    pure $ (\(_,b,_) -> TL.unpack b) <$> parsedLibraryInfo

-- Returns list of shared libraries that are necessary to load given binary
-- executable images.
-- Note: this are not necessarily all the dependencies.
dependenciesOfBinaries :: [FilePath] -> IO [FilePath]
dependenciesOfBinaries binaries = do
    listOfListOfDeps <- mapM dependenciesOfBinary binaries
    -- get rid of duplicates by passing through set
    let listOfDeps = Set.toList $ Set.unions $ Set.fromList <$> listOfListOfDeps
    filterM doesFileExist listOfDeps
