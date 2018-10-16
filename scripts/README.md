This directory contains the `Package.hs` script.

The script is provisional, Windows-only and relies on archives prepared beforehand. The aim is to iteratively improve and generalize it, in compliance with our long-term libraries package & distribution vision.

## Package.hs
Running this script builds the Dataframes library and creates a relocatable package with binary artifacts.

Script can be run by calling:
```
stack repo\scripts\Package.hs
```

The script can be called from any location, the artifacts will appear in current working directory.

### Input
* local environment:
  * Dataframes repository copy under the path stored in environment variable `APPVEYOR_BUILD_FOLDER`
  * MS Build binary under the path `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\amd64\MSBuild.exe` (part of Visual Studio installation)
  * `7z` program either available in `PATH` or installed in `C:\Program Files\7-Zip` (the default installation path)
  * `curl` program available in `PATH` (present by default on newer Windows)
  * Python 3 with numpy package installed under a path stored in `PythonDir` environment variable.
* hand-made resources:
  * archive with build-time dependencies (import libraries, headers, binaries), currently assumed at: `https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/libs-dev-v140.7z`
  * archive with package skeleton (i. e. package sans Dataframes library itself), currently assumed at: `https://s3-us-west-2.amazonaws.com/packages-luna/dataframes/windows-package-base.7z`

### Output
In the current working directory:
* `Dataframes-Win-x64-v141.7z` file — relocatable Dataframe library package (and all its dependencies) for 64-bit Windows.

### How the input packages are built
#### Build-time dependencies
Contains the following libraries:
* Apache Arrow
* Boost
* {fmt}
* xlnt
* date
* rapidjson
* pybind11

Each library comes with its own MS Build project property sheet that makes library visible to build system (by adjusting include/library dirs).

Dependencies were built manually and appropriately structured. The process should eventually become fully automated.

There is an additional property sheet for Python, that adds Python installation that is placed under the location stored in `PythonDir` environment variable.

#### Package skeleton
It basically consists of three parts:
* Python and its packages
* Other C/C++ library dependencies binaries
* runtime binaries

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

C and C++ dependencies are packaged just by copying their dependencies. In typical scenario it involves copying:
* Arrow.dll
* xlnt.dll

from the build-time dependencies package. That's assuming all other libraries are either static or header only. In some setups libraries like Boost, date and {fmt} may have their .DLL files as well.

Placing runtime libraries is just basically copying all DLL files from two folders:
* Universal C Runtime (UCRT) comes with Windows SDK. It has all the libraries named like `api-ms-win-*-*-l1-1-0.dll` and `ucrtbase.dll`. The source location depends on Windows SDK version installed on the building machine, following schema like `C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64`.
* Visual C++ Redistributable — the exact path depends on VS installation path, it is like: `C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Redist\MSVC\14.15.26706\x64\Microsoft.VC141.CRT`.