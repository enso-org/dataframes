{-# LANGUAGE UndecidableInstances #-}

module Logger where

import Prologue

import qualified Data.Text      as Text
import qualified Data.Text.Lazy as TextLazy
import qualified IO             as IO

class Logger m where
    logText :: Text -> m ()

    log :: (ToText s) => s -> m ()
    log = logText . convert

    logS :: String -> m ()
    logS = logText . convert


instance Logger IO where
    logText = IO.putTextLn

instance {-# OVERLAPPABLE #-}
    ( Monad m
    , Monad (t m)
    , MonadTrans t
    , Logger m
    ) => Logger (t m) where
    logText = lift . Logger.logText
    {-# INLINE logText #-}


newtype SilentT m a = SilentT (IdentityT m a)
    deriving (Functor, Applicative, Monad, MonadTrans)
makeLenses ''SilentT

runSilent :: SilentT m a -> m a
runSilent = runIdentityT . unwrap
{-# INLINE runSilent #-}

instance Applicative m => Logger (SilentT m) where
    logText = const $ pure ()
    {-# INLINE logText #-}
