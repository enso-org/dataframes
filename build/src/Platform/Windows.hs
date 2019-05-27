module Platform.Windows where

import Prologue

import qualified Data.Pecoff                      as Pecoff
import qualified Data.Pecoff.Imports              as Pecoff
import qualified Data.Set                         as Set
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

-- | Path being a toupper mapped filename. 
newtype NormalizedFilename = NormalizedFilename { filename :: FilePath }
    deriving (Eq, Ord, Show)

getProgramFiles86 :: MonadIO m => m FilePath
getProgramFiles86 = Utils.getEnvDefault "ProgramFiles(x86)" "C:\\Program Files (x86)"

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
normalizeBinaryName :: FilePath -> NormalizedFilename
normalizeBinaryName = NormalizedFilename . fmap toUpper . takeFileName

-- | Returns directory containing MSVC runtime redistributable DLLs. Depends on
-- finding the MSVC installation.
lookupMSVCRedist :: MonadIO m => m (Maybe FilePath)
lookupMSVCRedist = do
    -- eg. C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Redist\MSVC\14.16.27012\x64\Microsoft.VC141.CRT
    vsDir <- getProgramFiles86 <&> (</> "Microsoft Visual Studio")
    let redistPathPattern = vsDir </> "*\\*\\VC\\Redist\\MSVC\\*\\x64\\Microsoft.VC*.CRT"
    matchingLocations <- liftIO $ glob redistPathPattern
    -- we don't want to use "Preview" VS installations
    let matchingLocationsStable = filter (not . isInfixOf "Preview") matchingLocations
    pure $ maximum matchingLocationsStable
    -- maximum, because we want to use latest redist, as they are
    -- compatible only this way (you can use newer in the place of older)

-- | Returns directory containing Universal CRT redistributable DLLs. Depends on
-- finding Windows SDK kit. UCRT redist, unlike MSVC redist, is an OS component.
lookupUCRTRedist :: MonadIO m => m (Maybe FilePath)
lookupUCRTRedist = do
    -- eg. C:\Program Files (x86)\Windows Kits\10\Redist\10.0.17763.0\ucrt\DLLs\x64
    winSdkDir <- getProgramFiles86 <&> (</> "Windows Kits")
    let ucrtPathPattern = winSdkDir </> "10\\Redist\\10.*\\ucrt\\DLLs\\x64"
    matchingLocations <- liftIO $ glob ucrtPathPattern
    pure $ maximum matchingLocations

installBinariesInternal
    :: MonadIO m
    => FilePath  -- ^ Target directory, where binaries are to be placed
    -> [FilePath] -- ^ Additional DLL lookup directories
    -> Set FilePath -- ^ Binaries to be placed
    -> Set NormalizedFilename -- ^ Dependencies to be resolved and placed
    -> Set NormalizedFilename -- ^ Binaries already placed
    -> Set NormalizedFilename -- ^ Dependencies that were failed to be resolved
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
        resolvedDependency <- Utils.lookupInPATH lookupDirs $ filename head
        case resolvedDependency of
            -- If dependency can be resolved, treat it as additional binary to
            -- install. Otherwise, add to unresolved list.
            Just dep -> installBinariesInternal targetDir lookupDirs (Set.singleton dep) dependenciesToInstall' installed unresolvedBinaries
            Nothing  -> installBinariesInternal targetDir lookupDirs (Set.empty)         dependenciesToInstall' installed (Set.insert head unresolvedBinaries)
    | otherwise = do
        pure $ if Set.null unresolvedBinaries
            then Right $ filename <$> Set.toList installed
            else Left  $ filename <$> Set.toList unresolvedBinaries

-- | The target binaries and their DLL dependencies get copied into target
--   directory. Even if some dependency cannot be resolved, function tries to
--   place as much dependencies and binaries and possible.
installBinaries
    :: MonadIO m
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m (Either [FilePath] [FilePath]) -- ^ On success: list of installed binaries (their target path). On failure: list of unresolved dependencies.
installBinaries targetDir binaries additionalLocations = do
    -- liftIO $ putStrLn $ "Installing " <> show binaries
    
    maybeRedistDir <- lookupMSVCRedist
    maybeUCRTDir <- lookupUCRTRedist
    let redistDir = Utils.fromJustVerbose "cannot find MSVC redist" maybeRedistDir
    let ucrtDir = Utils.fromJustVerbose "cannot find UCRT redist" maybeUCRTDir
    let allPotentialLocations = [redistDir, ucrtDir] <> additionalLocations
    -- get rid of not existing directories (usually there are some) to save checking
    binLocations <- liftIO $ filterM doesDirectoryExist allPotentialLocations
    installBinariesInternal targetDir binLocations (Set.fromList binaries) def def def

-- | The target binaries and their DLL dependencies get copied into target
--   directory. Fails if there are unresolved dependencies.
packageBinaries 
    :: MonadIO m 
    => FilePath  -- ^ Target directory to place binaries within
    -> [FilePath] -- ^ Binaries to be installed
    -> [FilePath] -- ^ Additional locations with binaries
    -> m [FilePath] -- ^ List of installed binaries (their target path).
packageBinaries targetDir binaries additionalLocations = do
    installBinaries targetDir binaries additionalLocations >>= \case
        Left unresolved -> error $ "Failed to package binaries, unresolved dependencies: " <> show unresolved
        Right paths -> pure paths