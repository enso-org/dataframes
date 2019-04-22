module Program.PowerShell where

import Prologue

import qualified Data.Text as Text
import qualified Program as Program
import qualified Platform.Utils.Windows as WinUtils

import Distribution.System  (OS (Windows), buildOS)
import NeatInterpolation (text)
import System.FilePath      ((</>))

-------------------------
-- === Power Shell === --
-------------------------

-- === Definition === --

data PowerShell
instance Program.Program PowerShell where
    executableName = "powershell"
    defaultLocations = case buildOS of
        Windows -> do
            system32 <- WinUtils.system32
            -- Despite this looking like a hardcoded version, on newer Windows
            -- this location contains newer version of Power Shell.
            pure $ [system32 </> "WindowsPowerShell" </> "v1.0"]
        _ -> pure []

-- === API === --

runCommand :: MonadIO m => String -> m ()
runCommand script = do
    putStrLn $ "Running script: " <> script
    Program.call @PowerShell ["-Command", script]

-- === Helpers === --

-- | Power Shell requires us to escape quotes by reduplicating them.
escape :: Text.Text -> Text.Text
escape = Text.replace "\"" "\"\""
   
-----------------------
-- === Utilities === --
-----------------------

data EnvironmentVariableTarget = Machine | Process | User deriving (Show)

-- | Sets environment. If 'User' or 'Machine' is passed as a target, the effect
--   will be permanent.
--
--   Using 'Machine' target typically requires administrator privileges,
--   otherwise it will fail.
setEnv :: EnvironmentVariableTarget -> String -> String -> IO ()
setEnv (Text.pack . show -> target) (escape . Text.pack -> name) (escape . Text.pack -> value) = do
    let script = Text.unpack $ [text|[Environment]::SetEnvironmentVariable("$name", "$value", [EnvironmentVariableTarget]::$target)|]
    runCommand script

-- | Gets environment variable value. Returns empty string if the variable does
--   not exist. (or if its value is actually an empty string)
getEnv :: EnvironmentVariableTarget -> String -> IO String
getEnv target name = do
    let script = "[Environment]::GetEnvironmentVariable(\"" <> name <> "\", [EnvironmentVariableTarget]::" <> show target <> ")"
    Program.read @PowerShell ["-Command", script]