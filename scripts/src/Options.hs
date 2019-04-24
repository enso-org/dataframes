module Options where

import Prologue

data BuilderConfiguration = BuilderConfiguration
    { _projectRoot :: FilePath
    , _projectName :: String
    }

data Command 
    = Package 
        { outputArchive :: Maybe FilePath
        , outputDirectory :: Maybe FilePath
        }