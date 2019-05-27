This is the `build-support` library. What started as a simple script for building Dataframes library, has been extended, generalized and eventually made into a library. 

It is meant to contain utilities supporting building and packaging for our packages. 

# Packaging
Each project being a single sources subtree should be packagable. Packaging should be either fully trivial or adher to unified interface.

Packaging is considered trivial when there are no custom steps and all configuration can be implied from our conventions. When there are native applications/libraries, packaging cannot be trivial, currently it is limited to pure Luna libraries. 

When packaging is non-trivial, a project should contain `build/` subdirectory under its root. Running `stack run` in the build directory should build and package the project. In future, the build program should take command-line arguments according to the future specification.

Trivial packaging should be possible to run through the tool that will be created in future. (likely it will be part of `luna-manager`). The tool will also allow hiding the `stack run` detail.

# Library
## Program wrappers
* `Program` module provides common interface for spawning processes with known programs. It is basically a wrapper over `typed-process` and `process` libraries, so the library user doesn't have to deal with them directly and repeat the same boilerplate pattern.
* `Program.*` submodules provides a number of wrappers for well-known programs, exposing their functionalities as a Haskell API — so the end user can use meaningful Haskell functions rather than pretty print shell commands.

## Platform-specific helpers
* `Platform.Utils.*` provide low-level platform-specific utilities
* `Platform.*` provides high-level platform-specific utilities, often implemented on top of Programs.

## Root-level modules
* `Archive` providies unified interface for dealing with various archives.
* `IO` provides (relatively) safe functions for IO that support outputting non-ascii characters on Windows

# Future work
Library still needs a number of improvement in the following areas:
* Error handling — should be expanded, unified and rebuilt on top of our package
* IO and Logger — likely should be split and included into prologue library
* other minor refactorings as per our coding guidelines

Packaging needs some further design and implementation work:
* redesign an application package layout
* design a CLI interface for common tasks / customizations
* provide a cli tool that can build and package projects following our specification