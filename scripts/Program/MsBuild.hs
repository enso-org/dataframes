module Program.MsBuild where

import Program

data MsBuild
instance Program MsBuild where
    defaultLocations = ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\amd64"]
    executableName   = "MSBuild.exe"

build :: FilePath -> IO ()
build solutionPath =
    call @MsBuild ["/property:Configuration=Release", solutionPath]
