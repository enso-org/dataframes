module Program.MsBuild where

import Prologue

import qualified Program as Program

import Program (Program)
import System.FilePath ((</>))

data MsBuild
instance Program MsBuild where
    defaultLocations = do
        let editions = ["Community", "Enterprise"]
        let vsDirRoot = "C:\\Program Files (x86)\\Microsoft Visual Studio"
        let vsToMsBuild msBuildVer = "MSBuild" </> msBuildVer </> "Bin\\amd64"
        pure $
               [vsDirRoot </> "2019" </> edition </> vsToMsBuild "Current" | edition <- editions]
            <> [vsDirRoot </> "2017" </> edition </> vsToMsBuild "15.0"    | edition <- editions]
    executableName   = "MSBuild.exe"

build :: (MonadIO m) => FilePath -> m ()
build solutionPath =
    Program.call @MsBuild ["/property:Configuration=Release", solutionPath]
