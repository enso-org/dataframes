# Dataframes implementation in Luna

## Purpose
This project is a library with dataframes implementation. Dataframes are structures allowing more comfortable work with big datasets.

## Third-party dependencies
Required dependencies:
* [Apache Arrow](https://arrow.apache.org/) — from it [C++ library](https://github.com/apache/arrow/tree/master/cpp) component must be installed
* C++ build tools:
    * [CMake](https://cmake.org/)
    * A mostly C++17-compliant compiler. The tested ones are Visual Studio 2017.7 and GCC 7.3.0. Anything newer is expected to work as well.

Optional dependencies:
These dependencies are not required to compile the helper library, however without them certain functionalities shall be disabled.
* [{fmt}](http://fmtlib.net/) C++ library — needed if C++ library logs are enabled (meant only for debugging purposes)
* [xlnt library](https://github.com/mwu-tow/xlnt) (C++/CMake) — needed for .xlsx file format support. NOTE: On MacOS mwu-tow's fork is needed to fix the compilation issue. On other platforms, [official library repo](https://github.com/tfussell/xlnt) can be used.

## Build & Install
* make sure that dependecies are all installed
 build the helper C++ library — CMake will automatically place the built binary in the native_libs/platform directory, so `luna` should out-of-the-box be able to find it.
    * on Windows start Visual Studio x64 Tools Command Prompt  and type:
      ```
      cd Dataframes\native_libs
      mkdir build
      cd build
      cmake -G"NMake Makefiles" ..\src
      nmake
      ``` 
    * on other platforms:
      ```
      cd Dataframes/native_libs
      mkdir build
      cd build
      cmake ../src
      make
      ``` 
* happily use the dataframes libray

## Tutorial
TBD
