module Platform.Windows where

import qualified Data.Pecoff as Pecoff
import qualified Data.Pecoff.Imports as Pecoff

import Prologue

-- | As per registry entries at:
-- HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\KnownDLLs
knownSystemLibraries :: [FilePath]
knownSystemLibraries = 
    ["advapi32.dll", "clbcatq.dll", "combase.dll", "COMDLG32.dll", "coml2.dll"
    , "difxapi.dll", "gdi32.dll", "gdiplus.dll", "IMAGEHLP.dll", "IMM32.dll"
    , "kernel32.dll", "MSCTF.dll", "MSVCRT.dll", "NORMALIZ.dll", "NSI.dll"
    , "ole32.dll", "OLEAUT32.dll", "PSAPI.DLL", "rpcrt4.dll", "sechost.dll"
    , "Setupapi.dll", "SHCORE.dll", "SHELL32.dll", "SHLWAPI.dll", "user32.dll"
    , "WLDAP32.dll", "wow64.dll", "wow64cpu.dll", "wow64win.dll", "wowarmhw.dll"
    , "WS2_32.dll", "xtajit.dll"
    ]

-- | Return filenames of imported DLLs. That includes only direct dependencies.
dependenciesOfBinary :: MonadIO m => FilePath -> m [FilePath]
dependenciesOfBinary path = do
    pecoff <- liftIO $ Pecoff.readPecoff path
    let libnames = Pecoff.libraryName <$> (Pecoff.imports pecoff)
    pure libnames



    
lookupDependency :: MonadIO m => FilePath -> m (FilePath)
lookupDependency path = do
    liftIO $ putStrLn "bar"
    pure $ []
