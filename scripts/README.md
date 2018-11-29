This directory contains the `dataframes-package` Haskell program.

The program is provisional. It supports all our targeted platforms (Windows, Linux, macOS) but requires specific environment having been prepared beforehand.  The aim is to iteratively improve and generalize it, in compliance with our long-term libraries package & distribution vision.

Running this program builds the Dataframes library and creates a relocatable package with binary artifacts.

Program can be run by calling:
```
stack run
```

The program can be called from any location, the artifacts will appear in current working directory.

## General workings
### Windows
Runs in AppVeyor environment. Dependencies are provided by the pre-built archive. Build Dataframes. Extract them onto pre-built package skeleton.

### Linux
Runs in Docker environment with all dependencies pre-built. Build Dataframes. Build package from them, its ldd-found dependencies and Python distribution.

### macOS
Runs in VM environment, dependencies are obtained through brew or build from sources as part of CI script. Packaging uses automatic dependency detection (based on dyld).

## Input
* local environment:
  * Dataframes repository copy under the path stored in environment variable `DATAFRAMES_REPO_PATH`
  * `7z` program either available in `PATH` or installed in `C:\Program Files\7-Zip` (the default installation path)
  * `curl` program available in `PATH` (present by default on newer Windows)
  * Python 3 with numpy package installed under a path stored in `PythonDir` environment variable.
  * On Windows:
    * Visual Studio C++ toolset (2017.8)
    * MS Build binary under the path `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\amd64\MSBuild.exe` (part of Visual Studio installation)
  * On Linux:
    * CMake and GCC-7 or newer
    * ldd
    * patchelf
  * On macOS:
    * CMake
    * XCode >= 10
* external hand-made resources (needed on Windows only):
  * archive with build-time dependencies (import libraries, headers, binaries), currently assumed at: `https://packages.luna-lang.org/dataframes/libs-dev-v140.7z`
    * it contains .props file (property sheet) — this file will be included by MSBuild script. It is required that this sheet provides all the build-time dependencies of C++ parts of Dataframes
  * archive with package skeleton (i. e. package sans Dataframes library itself), currently assumed at: `https://packages.luna-lang.org/dataframes/windows-package-base.7z`
    * this archive will be a skeleton for `Dataframes\native_libs\windows` directory in the redistributable package
* also, obviously, to build this program `stack` is needed.

## Output
In the current working directory the archive will appear, depending on platform:
* `Dataframes-Win-x64.7z` file — relocatable Dataframe library package (and all its dependencies) for 64-bit Windows.
* `Dataframes-Linux-x64.7z` file — relocatable Dataframe library package (and all its dependencies) for 64-bit Linux.
* `Dataframes-macOS-x64.7z` file — relocatable Dataframe library package (and all its dependencies) for 64-bit macOS.

## How the input packages are built
### Build-time dependencies
Contains the following libraries:
* Apache Arrow
* Boost
* {fmt}
* xlnt
* date
* rapidjson
* pybind11

Each library comes with its own MS Build project property sheet that makes library visible to build system (by adjusting include/library dirs). The archive provides all-in-one .props file in its root path. It is required that Dataframes can be built with MSVC when the file is included.

Dependencies were built manually and appropriately structured. The process should eventually become fully automated.

There is an additional property sheet for Python, that adds Python installation that is placed under the location stored in `PythonDir` environment variable. Python and numpy is not part of this package, but if the variable is set, this package exposes Python and numpy to MSBuild.

All binaries are built with latest Visual Studio C++ compiler (as of writing, 15.8.7). They are 64-bit and use dynamic threaded runtime. Both Debug and Relase binaries are provided.

### Package skeleton
It basically consists of three parts:
* Python and its packages
* Other C/C++ library dependencies binaries
* runtime binaries

It is expected that Dataframes binaries placed in the directory with package skeleton can be run by system without any additional environment requirements.

#### Python
Python parts can packaged using the following batch script:
```
curl -fsL -o python-3.7.0-embed-amd64.zip https://www.python.org/ftp/python/3.7.0/python-3.7.0-embed-amd64.zip
"C:\\Program Files\\7-Zip\\7z" x -y  python-3.7.0-embed-amd64.zip
del python-3.7.0-embed-amd64.zip
ECHO .\lib\site-packages>>"python37._pth"
curl -fsL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
.\python.exe get-pip.py
del get-pip.py
.\python.exe -mpip install numpy seaborn matplotlib
```

#### C and C++ dependencies
C and C++ dependencies are packaged just by copying their dependencies. In typical scenario it involves copying:
* Arrow.dll
* xlnt.dll

from the build-time dependencies package. That's assuming all other libraries are either static or header only. In some setups libraries like Boost, date and {fmt} may have their .DLL files as well.

#### Runtime libraries
Placing runtime libraries is just basically copying all DLL files from two folders:
* Universal C Runtime (UCRT) comes with Windows SDK. It has all the libraries named like `api-ms-win-*-*-l1-1-0.dll` and `ucrtbase.dll`. The source location depends on Windows SDK version installed on the building machine, following schema like `C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64`.
* Visual C++ Redistributable — the exact path depends on VS installation path, it is like: `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Redist\MSVC\14.15.26706\x64\Microsoft.VC141.CRT`.