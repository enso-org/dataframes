import Prologue 

import System.Exit

import Control.Monad (liftM2)
import qualified Data.ByteString as ByteString
import qualified Program.SevenZip as SevenZip

import Data.ByteString (ByteString)
import System.IO.Temp (withSystemTempDirectory, withTempDirectory, writeTempFile)
import System.Directory (listDirectory)

fileContents = "qwertyuiop"

compareFiles :: FilePath -> FilePath -> IO Bool
compareFiles file1 file2 = (liftM2 (==)) (readFile file1) (readFile file2)

test = testM . pure
testM condition msg = condition >>= \case
    True  -> putStrLn $ "success: " <> msg
    False -> die $ "failure: " <> msg


testPack :: IO ()
testPack = do
    let toPackPath = "toPack"
    let packedPath = "packed.7z"
    ByteString.writeFile "topack" fileContents
    SevenZip.pack [toPackPath] packedPath
    withTempDirectory "." "unpacked" $ \tempDir -> do
        SevenZip.unpack packedPath tempDir
        unpackedFiles <- listDirectory tempDir
        test (length unpackedFiles == 1) "single unpacked file"
        testM (compareFiles toPackPath (unsafeHead unpackedFiles)) "unpacked file contents"

-- runTestCase name 

main = do
    testPack
    pure ()
