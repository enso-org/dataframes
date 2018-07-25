# Dataframes implementation in Luna

## Purpose
This project is a library with dataframes implementation. Dataframes are structures allowing more comfortable work with big datasets.

## Third-party dependencies
Required dependencies:
* [Apache Arrow](https://arrow.apache.org/) — from it [C++ library](https://github.com/apache/arrow/tree/master/cpp) component must be installed

Optional dependencies:
These dependencies are not required to compile the helper library, however without them certain functionalities shall be disabled.
* [{fmt}](http://fmtlib.net/) C++ library — needed if C++ library logs are enabled (meant only for debugging purposes)
* [xlnt library](https://github.com/mwu-tow/xlnt) (C++/CMake) — needed for .xlsx file format support. NOTE: On MacOS mwu-tow's fork is needed to fix the compilation issue. On other platforms, [official library repo](https://github.com/tfussell/xlnt) can be used.

## Build & Install
1 make sure that dependecies are all installed
1 build the helper C++ library — CMake will automatically place the built binary in the native_libs/platform directory, so `luna` should out-of-the-box be able to find it.
* happily use dataframes

## Tutorial
TBD
