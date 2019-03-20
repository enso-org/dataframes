{-# LANGUAGE PartialTypeSignatures #-}
module Progress where

import Prologue

import Control.Exception.Safe (bracket)
import System.IO (hFlush, stdout)

class Progress progress where
    ratio :: progress -> Maybe Float

    isDone :: progress -> Maybe Bool
    isDone p = (== 1.0) <$> ratio p 

-- | Represents information about changed progressible action status.
data Notification p =
      Started
    | Ongoing p
    | Finished
    deriving (Show, Eq)


type Progressible p m = (p -> IO ()) -> m ()

type Observer p = Notification p -> IO ()

instance Progress p => Progress (Notification p) where
    ratio Started     = Just 0.0
    ratio (Ongoing p) = ratio p
    ratio Finished    = Just 1.0

formatProgressBar :: Progress p => Int -> p -> String
formatProgressBar width p = let markerCount = width - 2 in case ratio p of
    Just ratio -> "[" <> replicate completedMarkers '=' <> replicate inProgressMarkers '>' <> replicate leftMarkers '.' <> "]" where
        completedMarkers = floor $ ratio * (fromIntegral markerCount)
        inProgressMarkers = if ratio == 1 then 0 else 1
        leftMarkers = markerCount - completedMarkers - inProgressMarkers
    Nothing -> "[" <> replicate markerCount '?' <> "]"
    

withTextProgressBar :: (Progress p) => Int -> Observer p
withTextProgressBar width progress = do
    let line = formatProgressBar width progress
    case progress of
        Finished -> putStrLn line
        _        -> do
            putStr $ line <> "\r"
            hFlush stdout

-- | Guarantees to emit Started and Finished notifications. 
runProgressible :: (MonadMask m, MonadIO m) => Progressible p m -> Observer p -> m ()
runProgressible action cb = bracket 
    (liftIO $ cb Started)
    (const $ liftIO $ cb Finished)
    (const $ action $ cb . Ongoing)

-- data PPP = PPP Float
-- instance Progress PPP where
--     ratio (PPP f) = Just f

-- progressibleAction :: MonadIO m => Progressible PPP m
-- progressibleAction cb = liftIO $ cb $ PPP 0.5

-- fff :: Progress p => Observer p -> IO ()
-- fff cb = do
--     cb $ Started
--     cb $ Finished

-- aaa :: IO ()
-- aaa = runProgressible progressibleAction (withTextProgressBar 80)