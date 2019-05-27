{-# LANGUAGE PartialTypeSignatures #-}
module Progress where

import Prologue

import Control.Exception.Safe (bracket)
import System.IO              (hFlush, stdout)



----------------------
-- === Progress === --
----------------------

-- === Definition == --

-- | Represents progressible operation state. 
class Progress progress where
    -- | Progress value represented as floating point value. Should be in range
    --   [0.0, 1.0], where @0.0@ represent initial state (no work done yet) and
    --   @1.0@ represents complete operation (all work already done).
    --
    --   Progressible operation are not required to be able to provide such
    --   information (e.g. they might not know the total amount of work to be
    --   done), in such case 'Nothing' is used.
    ratio  :: progress -> Maybe Float

    -- | 'True' iff operation is completed (no more work to be done).
    isDone :: progress -> Maybe Bool
    isDone = fmap (== 1.0) . ratio

-- | Represents information about progressible action status.
data Notification p
    = Started
    | Ongoing p
    | Finished
    deriving (Show, Eq)

-- | Represents a progressible action that during its run emits a sequence of
--   progress states 'p' by calling the provided callback function (being an IO
--   action).
type Progressible p m a = (p -> IO ()) -> m a

-- | Action that represents callback receiving notifications about the opration
--   progress.
type Observer     p     = Notification p -> IO ()


-- === Instances === --

instance Progress p
      => Progress (Notification p) where
    ratio Started     = Just 0.0
    ratio Finished    = Just 1.0
    ratio (Ongoing p) = ratio p


-- === API === --

-- | Function pretty-prints simple ascii progress bar of given length for given
--   progress state.
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

-- | Obtain an 'Observer' that outputs progress onto stdout as ascii progress
--   bar. As @\r@ is used to update its state, it should not be used when there
--   is any other input during the operation.
withTextProgressBar :: Progress p => Int -> Observer p
withTextProgressBar width progress = do
    let line = formatProgressBar width progress
    case progress of
        Finished -> putStrLn line
        _        -> do
            putStr $ line <> "\r"
            hFlush stdout

-- | Wraps progressible action that reports its intermediary progress state 'p'
--   into an action that sends a stream of `Notification's. Emits 'Started'
--   before running the action and 'Finished' after the action ends.
runProgressible :: (MonadMask m, MonadIO m)
    => Observer p 
    -> Progressible p m a
    -> m a
runProgressible cb action = bracket
    (liftIO $ cb Started)
    (const . liftIO $ cb Finished)
    (const . action $ cb . Ongoing)
