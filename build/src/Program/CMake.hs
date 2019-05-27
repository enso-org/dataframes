module Program.CMake where


import Prologue

import qualified Program      as Program
import qualified Program.Make as Make

import Program          (Program)
import System.Directory (createDirectoryIfMissing)
import Text.Printf      (printf)



-------------------
-- === CMake === --
-------------------

-- === Definition === --

data CMake

instance Program CMake where
    executableName = "cmake"

-- === Switches === --

data Option = OptionSetVariable SetVariable
            | OptionBuildType BuildType

data BuildType = Debug | ReleaseWithDebInfo deriving (Show)

data SetVariable = SetVariable
    { _name :: String
    , _value :: String
    } deriving (Show)
makeLenses ''SetVariable


formatVariable :: SetVariable -> String
formatVariable var = printf "-D%s=%s" (var ^. name) (var ^. value)

formatBuildType :: BuildType -> String
formatBuildType = \case 
    Debug              -> "Debug"
    ReleaseWithDebInfo -> "RelWithDebInfo"

instance Program.Argument Option where
    format = pure . \case 
        OptionSetVariable var     -> formatVariable var
        OptionBuildType buildType -> formatVariable var 
            where var = SetVariable "CMAKE_BUILD_TYPE" buildTypeFmt
                  buildTypeFmt = formatBuildType buildType

                  
-- === API === --

cmake :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
cmake whereToRun whatToBuild options = do
    liftIO $ createDirectoryIfMissing True whereToRun
    let varOptions = Program.format options
    Program.callCwd @CMake whereToRun (varOptions <> [whatToBuild])

-- FIXME: drop dependency on make, use --build
build :: (MonadIO m) => FilePath -> FilePath -> [Option] -> m ()
build whereToRun whatToBuild options = do
    cmake whereToRun whatToBuild options
    Make.make whereToRun
