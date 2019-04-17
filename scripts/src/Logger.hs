{-# LANGUAGE UndecidableInstances #-}

module Logger where

import Prologue

import qualified Data.Text as Text
import qualified Data.Text.Lazy as TextLazy

class Logger m where
    log :: (ToString s) => s -> m ()

instance Logger IO where
    log = putStrLn . toString

instance {-# OVERLAPPABLE #-}
    ( Monad m
    , Monad (t m)
    , MonadTrans t
    , Logger m
    ) => Logger (t m) where
    log = lift . Logger.log
    {-# INLINE log #-}


newtype SilentT m a = SilentT (IdentityT m a)
    deriving (Functor, Applicative, Monad, MonadTrans)
makeLenses ''SilentT

runSilent :: SilentT m a -> m a
runSilent = runIdentityT . unwrap
{-# INLINE runSilent #-}

instance Applicative m => Logger (SilentT m) where
    log = const $ pure ()
    {-# INLINE log #-}