import Prologue 

import System.Exit

import Control.Monad (liftM2)
import qualified Data.ByteString as ByteString
import qualified Program.SevenZip as SevenZip
import qualified Progress as Progress

import Data.ByteString (ByteString)
import System.IO.Temp (withSystemTempDirectory, withTempDirectory, writeTempFile)
import System.Directory (listDirectory, withCurrentDirectory)

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

cb arg = print arg

testPackProgressive :: IO ()
testPackProgressive = do
    let src = "C:\\Users\\mwu\\Downloads\\qt-everywhere-src-5.12.1.zip"
    -- let src = "C:\\Users\\mwu\\Downloads\\gui.zip"
    SevenZip.unpackWithProgress (Progress.withTextProgressBar 80) src "."

runTestCase :: String -> IO () -> IO ()
runTestCase name action = withSystemTempDirectory name $ \dir -> do
    putStrLn $ "Running `" <> name <> "` in " <> dir
    withCurrentDirectory dir action

main = do
    runTestCase "pack unpack" testPackProgressive
    -- runTestCase "pack unpack" testPack    
    pure ()
