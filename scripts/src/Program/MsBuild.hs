module Program.MsBuild where

import Control.Monad.IO.Class (MonadIO)

import Program

data MsBuild
instance Program MsBuild where
    defaultLocations = [ "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\amd64"
                       , "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Enterprise\\MSBuild\\15.0\\Bin\\amd64"]
    executableName   = "MSBuild.exe"

build :: (MonadIO m) => FilePath -> m ()
build solutionPath =
    call @MsBuild ["/property:Configuration=Release", solutionPath]
