module Platform.Windows where

import Prologue

import qualified Data.Pecoff                      as Pecoff
import qualified Data.Pecoff.Imports              as Pecoff
import qualified Data.Set                         as Set
import qualified Distribution.Simple.Program.Find as Cabal
import qualified Program                          as Program
import qualified Utils                            as Utils

import Control.Monad        (filterM)
import Data.Char            (toUpper)
import Data.List            (isInfixOf)
import Data.Set             (Set, (\\))
import System.Directory     (doesDirectoryExist, doesFileExist)
import System.FilePath      (takeFileName, (</>))
import System.FilePath.Glob (glob)

data DependencyType = System | Local | UCRT | MSVCRT deriving (Eq, Show)

-- | Libraries that should not be packaged and distributed.
knownSystemLibraries :: [FilePath]
knownSystemLibraries =
    -- Entries as observed under registry key:
    -- HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\KnownDLLs
    [ "advapi32.dll", "clbcatq.dll", "combase.dll", "COMDLG32.dll", "coml2.dll"
    , "difxapi.dll", "gdi32.dll", "gdiplus.dll", "IMAGEHLP.dll", "IMM32.dll"
    , "kernel32.dll", "MSCTF.dll", "MSVCRT.dll", "NORMALIZ.dll", "NSI.dll"
    , "ole32.dll", "OLEAUT32.dll", "PSAPI.DLL", "rpcrt4.dll", "sechost.dll"
    , "Setupapi.dll", "SHCORE.dll", "SHELL32.dll", "SHLWAPI.dll", "user32.dll"
    , "WLDAP32.dll", "wow64.dll", "wow64cpu.dll", "wow64win.dll", "wowarmhw.dll"
    , "WS2_32.dll", "xtajit.dll"
    ]
    <> -- The entries below come from experience. The list is incomplete.
    [ "version.dll", "KernelBase.dll" ]

-- | We compare library names only after normalizing them - eg. KERNEL32.dll and
-- Kernel32.dll are the same.
normalizeBinaryName :: FilePath -> FilePath
normalizeBinaryName = fmap toUpper . takeFileName

-- | Returns directory containing MSVC runtime redistributable DLLs. Depends on
-- finding the MSVC installation.
lookupMSVCRedist :: MonadIO m => m (Maybe FilePath)
lookupMSVCRedist = do
    -- eg. C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Redist\MSVC\14.16.27012\x64\Microsoft.VC141.CRT
    let redistPathPattern = "C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\*\\VC\\Redist\\MSVC\\*\\x64\\Microsoft.VC*.CRT"
    matchingLocations <- liftIO $ glob redistPathPattern
    -- we don't want to use "Preview" VS installations
    let matchingLocationsStable = filter (not . isInfixOf "Preview") matchingLocations
    pure $ maximum matchingLocationsStable
    -- ^ maximum, because we want to use latest redist, as they are
    -- compatible only this way (you can use newer in the place of older)

-- | Returns directory containing Universal CRT redistributable DLLs. Depends on
-- finding Windows SDK kit.
lookupUCRTRedist :: MonadIO m => m (Maybe FilePath)
lookupUCRTRedist = do
    -- eg. C:\Program Files (x86)\Windows Kits\10\Redist\10.0.17763.0\ucrt\DLLs\x64
    let ucrtPathPattern = "C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\10.*\\ucrt\\DLLs\\x64"
    matchingLocations <- liftIO $ glob ucrtPathPattern
    pure $ maximum matchingLocations

-- | If file by given name exists in directory, return path to this file
lookupFileInDir :: MonadIO m => FilePath -> FilePath -> m (Maybe FilePath)
lookupFileInDir file dir = do
    -- putStrLn $ "Looking for " <> file <> " in " <> dir
    let path = dir </> file
    exists <- liftIO $ doesFileExist path
    pure $ Utils.toMaybe exists path

lookupInPATH :: MonadIO m => [FilePath] -> FilePath -> m (Maybe FilePath)
lookupInPATH additionalPaths dll = do
    -- putStrLn $ "Looking for " <> dll
    systemPaths <- liftIO $ Cabal.getSystemSearchPath -- ^ includes PATH env
    let paths = additionalPaths <> systemPaths
    let testPaths (head : tail) = lookupFileInDir dll head >>= \case
            Just dir -> pure $ Just dir
            Nothing  -> testPaths tail
        testPaths [] = pure Nothing
    testPaths paths

installBinariesInternal
    :: MonadIO m
    => FilePath  -- ^ Target directory, where binaries are to be placed
    -> [FilePath] -- ^ Additional DLL lookup directories
    -> Set FilePath -- ^ Binaries to be placed (arbitrary paths)
    -> Set FilePath -- ^ Dependencies to be resolved and placed (normalized filenames)
    -> Set FilePath -- ^ Binaries already placed (normalized filenames)
    -> Set FilePath -- ^ Dependencies that were failed to be resolved (normalized filenames)
    -> m (Either [FilePath] [FilePath])
installBinariesInternal targetDir lookupDirs binariesToInstall dependenciesToInstall installed unresolvedBinaries
    | (not . Set.null) binariesToInstall = do
        -- If there is binary to install, install it and add its dependencies to
        -- the list
        let systemLibs = Set.fromList $ normalizeBinaryName <$> knownSystemLibraries
        let (head, binariesToInstall') = Set.deleteFindMin binariesToInstall
        placedBinary <- Utils.copyToDir targetDir head
        deps <- fmap normalizeBinaryName <$> Pecoff.dependenciesOfBinary placedBinary

        let newDeps = Set.fromList deps
                        \\ installed
                        \\ unresolvedBinaries
                        \\ systemLibs
        let installed' = Set.insert (normalizeBinaryName placedBinary) installed
        installBinariesInternal targetDir lookupDirs (binariesToInstall') (Set.union newDeps dependenciesToInstall) installed' unresolvedBinaries
    | (not . Set.null) dependenciesToInstall = do
        let (head, dependenciesToInstall') = Set.deleteFindMin dependenciesToInstall
        resolvedDependency <- lookupInPATH lookupDirs head
        case resolvedDependency of
            -- If dependency can be resolved, treat it as additional binary to
            -- install. Otherwise, add to unresolved list.
            Just dep -> installBinariesInternal targetDir lookupDirs (Set.singleton dep) dependenciesToInstall' installed unresolvedBinaries
            Nothing  -> installBinariesInternal targetDir lookupDirs (Set.empty)         dependenciesToInstall' installed (Set.insert head unresolvedBinaries)
    | otherwise = do
        pure $ if Set.null unresolvedBinaries
            then Right $ Set.toList installed
            else Left  $ Set.toList unresolvedBinaries

installBinaries
    :: MonadIO m
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m (Either [FilePath] [FilePath])
installBinaries targetDir binaries additionalLocations = do
    -- liftIO $ putStrLn $ "Installing " <> show binaries
    maybeRedistDir <- lookupMSVCRedist
    maybeUCRTDir <- lookupUCRTRedist
    let redistDir = Utils.fromJustVerbose "cannot find MSVC redist" maybeRedistDir
    let ucrtDir = Utils.fromJustVerbose "cannot find UCRT redist" maybeUCRTDir
    -- get rid of not existing directories (usually there are some) to save checking
    binLocations <- liftIO $ filterM doesDirectoryExist $ [redistDir, ucrtDir] <> additionalLocations
    installBinariesInternal targetDir binLocations (Set.fromList binaries) def def def
