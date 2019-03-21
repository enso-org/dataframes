{-# LANGUAGE PartialTypeSignatures #-}
module Progress where

import Prologue

import Control.Exception.Safe (bracket)
import System.IO              (hFlush, stdout)



----------------------
-- === Progress === --
----------------------

-- === Definition == --
class Progress progress where
    ratio  :: progress -> Maybe Float
    isDone :: progress -> Maybe Bool
    isDone = fmap (== 1.0) . ratio

-- | Represents information about changed progressible action status.
data Notification p
    = Started
    | Ongoing p
    | Finished
    deriving (Show, Eq)

type Progressible p m a = (p -> IO ()) -> m a
type Observer     p     = Notification p -> IO ()


-- === Instances === --

instance Progress p
      => Progress (Notification p) where
    ratio Started     = Just 0.0
    ratio Finished    = Just 1.0
    ratio (Ongoing p) = ratio p


-- === API === --

formatProgressBar :: Progress p => Int -> p -> String
formatProgressBar width p =
    let markerCount  = width - bracketCount
        bracketed s  = "[" <> s <> "]"
        bracketCount = 2
    in case ratio p of
        Nothing    -> bracketed $ replicate markerCount       '?'
        Just ratio -> bracketed $ replicate completedMarkers  '='
                               <> replicate inProgressMarkers '>'
                               <> replicate leftMarkers       '.'
            where completedMarkers  = floor $ ratio * (convert markerCount)
                  inProgressMarkers = if ratio == 1 then 0 else 1
                  leftMarkers       = markerCount - completedMarkers
                                                  - inProgressMarkers

withTextProgressBar :: Progress p => Int -> Observer p
withTextProgressBar width progress = do
    let line = formatProgressBar width progress
    case progress of
        Finished -> putStrLn line
        _        -> do
            putStr $ line <> "\r"
            hFlush stdout

-- | Guarantees to emit Started and Finished notifications.
runProgressible :: (MonadMask m, MonadIO m)
    => Progressible p m a -> Observer p -> m a
runProgressible action cb = bracket
    (liftIO $ cb Started)
    (const . liftIO $ cb Finished)
    (const . action $ cb . Ongoing)
