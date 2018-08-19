# Dataframes implementation in Luna

## Purpose
This project is a library with dataframes implementation. Dataframes are structures allowing more comfortable work with big datasets.

## Third-party dependencies
Required dependencies:
* [Apache Arrow](https://arrow.apache.org/) — from it [C++ library](https://github.com/apache/arrow/tree/master/cpp) component must be installed
* [Boost C++ Libraries](https://www.boost.org/) — also required by Apache Arrow.
* [RapidJSON](https://github.com/Tencent/rapidjson) — needed for LQuery processing
* C++ build tools:
    * [CMake](https://cmake.org/) — cross-platform build tool for C++ used by the C++ helper and all its dependencies.
    * A mostly C++17-compliant compiler. The tested ones are Visual Studio 2017.7 on Windows and GCC 7.3.0 on Ubuntu. Anything newer is expected to work as well.

Optional dependencies:
These dependencies are not required to compile the helper library, however without them certain functionalities shall be disabled.
* [{fmt}](http://fmtlib.net/) C++ library — needed if C++ library logs are enabled (meant only for debugging purposes)
* [xlnt library](https://github.com/mwu-tow/xlnt) C++ library — needed for .xlsx file format support. NOTE: On MacOS mwu-tow's fork is needed to fix the compilation issue. On other platforms, [official library repo](https://github.com/tfussell/xlnt) can be used.

## Build & Install
* make sure that dependecies are all installed
 build the helper C++ library — CMake will automatically place the built binary in the native_libs/platform directory, so `luna` should out-of-the-box be able to find it.
    * on Windows start *Visual Studio x64 Tools Command Prompt* and type:
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
    where `Dataframes` refer to the local copy of this repo.
* happily use the dataframes libray

## Overview
The library currently provides wrappers for Apache Arrow structures. 

### Storage types
* `ArrayData` — type-erased storage for `Array` consisting of several contiguous memory buffers. The buffer count depends on stored type. Typically there are two buffers: one for values and one for masking nulls. More comples types (union, lists) will use more.
* `Array tag` — data array with strongly typed accessors. See section below for supported `tag` types.
* `ChunkedArray tag` — a list of `Array`s of the same type viewed as a single large array. Allows storing large sequences of data (that could not be feasably stored in a single memory block) and efficient slice / concat operations.
* `Column` — type erased accessor for a named `ChunkedArray`. Stored type is represented by using one of its constructors.
* `Table` — ordered sequence of `Column`s.

### Type tag types
These types are provided by the library to identify types that can be stored by `Array` and their mapping to Luna types. Currently provided type tags are listed in the table below.

| Tag type        | Luna value type | Apache Arrow type   | Memory per element                          |
|-----------------|-----------------|---------------------|---------------------------------------------|
| StringType      | Text            | utf8 non-nullable   | 4 bytes + 1 byte per character + 1 bit mask |
| MaybeStringType | Maybe Text      | utf8 nullable       | as above                                    |
| Int64Type       | Int             | int64 non-nullable  | 8 bytes + 1 bit mask                        |
| MaybeInt64Type  | Maybe Int       | int64 nullable      | as above                                    |
| DoubleType      | Real            | double non-nullable | 8 bytes + 1 bit mask                        |
| MaybeDoubleType | Maybe Real      | double nullable     | as above                                    |

Note: Arrow's `utf8` type is a list of non-nullable bytes.

### IO types
#### Parsers
CSV files are supported. XLSX files are supported if the helper C++ library was built with XLNT third-part library enabled.

* `CSVParser` allows reading `Table` from [CSV file](https://tools.ietf.org/html/rfc4180).
* `CSVGenerator` allows storing `Table` to CSV.
* `XLSXParser` allows reading Microsoft Excel Open XML Spreadsheet File files (.xlsx).
* `XLSXGenerator` allows storing `Table`s to Microsoft Excel Open XML Spreadsheet File files (.xlsx).

### Other types
* `DataType` represents the type of values being stored in a `ArrayData`. Note that this type does not contain information whether it is nullable — being nullable is a property of `Field`, not `Datatype`.
* `Field` is a named `DataType` with an additional information whether values are nullable. Describes contents of the `Column`.
* `Schema` is a sequence of `Field`s describing the `Table`.

## Tutorial
TBD
