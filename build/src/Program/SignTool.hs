module Program.SignTool where

import Prologue

import qualified Program as Program
import qualified Utils   as Utils

import Distribution.System (OS(Windows), buildOS)

data SignTool
instance Program.Program SignTool where
    defaultLocations = pure $ case buildOS of 
        Windows ->
            [ "C:\\Program Files (x86)\\Windows Kits\\10\\bin\\x64"
            , "C:\\Program Files (x86)\\Windows Kits\\10\\App Certification Kit"
            ]
        _       -> 
            []
    executableName = "signtool"

-- | Signs binary using a password-protected certificate.
sign :: MonadIO m 
     => FilePath -- ^ File with certificate
     -> String  -- ^ Password protecting the certificate
     -> FilePath -- ^ Binary to be signedz
     -> m ()
sign certificateFile certificatePassword binaryToSign = do
    let timestampUrl = "http://timestamp.digicert.com"
    Program.call @SignTool 
        ["sign", "/v"
        , "/f", certificateFile
        , "/p", certificatePassword
        , "/t", timestampUrl
        , binaryToSign
        ]

-- | Signs given binary using a password-protected certificate. Certifacte
--   location shall be read from @CERT_PATH@ environment variable, and the
--   password from @CERT_PASS@. Both variables are required to be ste.
signEnv 
    :: MonadIO m 
    => FilePath -- ^ Binary to be signed
    -> m ()
signEnv binaryToSign = do
    cert <- Utils.getEnvRequired "CERT_PATH"
    certPass <- Utils.getEnvRequired "CERT_PASS"
    sign cert certPass binaryToSign
