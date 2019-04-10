module Program.MsBuild where

import Prologue

import qualified Data.Text.IO as Text
import qualified Program      as Program

import Control.Error     (atMay)
import Data.List         (elemIndex)
import Data.String.Utils (strip)
import NeatInterpolation (text)
import Program           (Program)
import System.FilePath   ((</>))
import System.IO         (hClose)
import System.IO.Temp    (withSystemTempFile)
import Text.Printf       (printf)


----------------------
-- === MS BUILD === --
----------------------

-- === Definition === --

data MsBuild

instance Program MsBuild where
    -- | We currently detect MS Build that comes with VS 2017 / 2019
    defaultLocations = do
        -- TODO? could likely use vswhere
        -- not sure if they finally allow installation in arbitrary location
        let editions = ["Community", "Enterprise"]
        let vsDirRoot = "C:\\Program Files (x86)\\Microsoft Visual Studio"
        let vsToMsBuild msBuildVer = "MSBuild" </> msBuildVer </> "Bin\\amd64"
        pure 
            $ [vsDirRoot </> "2019" </> edition </> vsToMsBuild "Current" | edition <- editions]
           <> [vsDirRoot </> "2017" </> edition </> vsToMsBuild "15.0"    | edition <- editions]
    executableName   = "MSBuild"


-- === CLI Switches === --

type PropertyName = String
type PropertyValue = String

data Switch
    = SetProperty { propertyName :: String, propertyValue :: String} -- ^ Set or override the specified project-level property
    | Target      { target :: String } -- ^ Specify target to build.
    | NoLogo -- ^ Don't display the startup banner or the copyright message.
    deriving (Eq, Show)

instance Program.Argument Switch where
    format = \case
        SetProperty{..} -> [printf "-property:%s=%s" propertyName propertyValue]
        Target     {..} -> [printf "-target:%s"      target]
        NoLogo          -> ["-nologo"]

-- | Class for various convenience datatypes that can be covnerted to switches
class Switchlike s where
    toSwitches :: s -> [Switch]


-- === Helper Switch-like types === --

-- | Default project templates provide 'Debug' and 'Release' configurations,
-- however user can create custom ones. Also, generated projects (e.g. by CMake)
-- can contain other configurations.
data Configuration
    = Debug
    | Release
    | CustomConfiguration String
    deriving (Eq, Show)

instance Switchlike Configuration where
    toSwitches c = pure $ SetProperty "Configuration" $ case c of
        Debug                 -> "Debug"
        Release               -> "Release"
        CustomConfiguration c -> c

-- | Sometimes instead of "Win32" a "x86" platform name is used. In such case use 'CustomPlatform' constructor.
data Platform
    = Win32
    | X64
    | CustomPlatform String
    deriving (Eq, Show)

instance Switchlike Platform where
    toSwitches c = pure $ SetProperty "Platform" $ case c of
        Win32            -> "Win32"
        X64              -> "x64"
        CustomPlatform c -> c

-- | 'Configuration' and 'Platform' are almost always used together.
data BuildConfiguration = BuildConfiguration Configuration Platform

instance Switchlike BuildConfiguration where
    toSwitches (BuildConfiguration c p) = toSwitches c <> toSwitches p

instance Default BuildConfiguration where
    def = BuildConfiguration Release X64

-- === Helper API (using switches) === --

call :: MonadIO m => [Switch] -> FilePath -> m ()
call switches projectFile = Program.call @MsBuild args
    where args = Program.format switches <> [projectFile]

readOut :: MonadIO m => [Switch] -> FilePath -> m String
readOut switches projectFile = Program.read @MsBuild args
    where args = Program.format switches <> [projectFile]

-- === Querying utils === --

-- | Internal name used as target name for the query
queryTargetName :: IsString s => s
queryTargetName = "queryValue"

-- | Project template for property querying.
queryProjectCode 
    :: Text -- ^ Absolute path to project that is to be queried.
    -> Text -- ^ Name of the queried property.
    -> Text -- ^ MS Build project file (XML) that when built shall output query result
queryProjectCode targetProject valueName = [text|
<?xml version="1.0" encoding="utf-8"?>
<Project InitialTargets="setup" DefaultTargets="$queryTargetName" ToolsVersion="14.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <Import Project="$targetProject"/>
    <Target Name="$queryTargetName">
        <Message Text="$(Query)"/>
    </Target>
    <Target Name="setup">
        <ItemGroup>
            <_Query Include="$valueName"/>
        </ItemGroup>
        <PropertyGroup>
            <Query>$(%(_Query.Identity))</Query>
        </PropertyGroup>
    </Target>
</Project>
|]

-- | Retrieve 
parseQueryResult :: String -> Maybe String
parseQueryResult output = do
    let strippedLines = strip <$> lines output
    queryHeaderIndex <- elemIndex (queryTargetName <> ":") strippedLines
    atMay strippedLines $ queryHeaderIndex + 1


-- === API === --

-- | Builds the targets in the given project file.
build 
    :: (MonadIO m) 
    => 
    BuildConfiguration 
    -> FilePath  -- ^ Project or solution file.
    -> m ()
build config solutionPath =
    call (toSwitches config) solutionPath

queryProperty 
    :: (MonadMask m, MonadIO m) 
    => BuildConfiguration
    -> FilePath -- ^ Target project
    -> String  -- ^ Property that we query about
    -> m (Maybe String)
queryProperty conf targetProject valueName = do
    let text = queryProjectCode (convert targetProject) (convert valueName)
    out <- withSystemTempFile "" $ \tmpProjectPath tmpHandle -> do
        liftIO $ Text.hPutStr tmpHandle text
        liftIO $ hClose tmpHandle
        let switches = NoLogo : toSwitches conf
        readOut switches tmpProjectPath
    pure $ parseQueryResult out


    