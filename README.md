# Dataframes implementation in Luna
## Purpose
This project is a library with dataframes implementation. Dataframes are structures allowing more comfortable work with big datasets.

## Build status

| Environment                 | Build status |
|-----------------------------|--------------|
| Linux, GCC-7 & GCC-8        | [![CircleCI](https://circleci.com/gh/luna/Dataframes.svg?style=svg)](https://circleci.com/gh/luna/Dataframes)            |
| Windows, Visual Studio 2017 | [![Build status](https://ci.appveyor.com/api/projects/status/vna6martcp4nlb77/branch/master?svg=true)](https://ci.appveyor.com/project/lunalangCI/dataframes/branch/master) |
## Third-party dependencies
Required dependencies:
* C++ build tools:
    * [CMake](https://cmake.org/) — cross-platform build tool for C++ used by the C++ helper and all its dependencies.
    * A mostly C++17-compliant compiler. The tested ones are Visual Studio 2017.8 on Windows and GCC 7.3.0 on Ubuntu. Anything newer is expected to work as well.
* Libraries:
  * [Apache Arrow](https://arrow.apache.org/) — from it [C++ library](https://github.com/apache/arrow/tree/master/cpp) component must be installed
  * [Boost C++ Libraries](https://www.boost.org/) — also required by Apache Arrow.
  * [date library](https://github.com/HowardHinnant/date) — for calendar support for timestamps.
  * [pybind11](https://github.com/pybind/pybind11) — C++ Python bindings
  * [Python 3.6+](https://www.python.org/) with some packages:
    * `matplotlib`
    * `seaborn`
  * [RapidJSON](https://github.com/Tencent/rapidjson) — needed for LQuery processing
  * [{fmt}](http://fmtlib.net/) - **version 5.2.0 does not work, use 5.2.1** -  C++ library string formatting

Optional dependencies:
These dependencies are not required to compile the helper library, however without them certain functionalities shall be disabled.
* [xlnt library](https://github.com/mwu-tow/xlnt) C++ library — needed for .xlsx file format support. NOTE: On MacOS mwu-tow's fork is needed to fix the compilation issue. On other platforms, [official library repo](https://github.com/tfussell/xlnt) can be used.

## Build & Install
* make sure that dependecies are all installed.
    * On Mac it is easily done with Anaconda (https://www.anaconda.com/download/).
    * Once you have installed it, you can run the following commands to install Arrow:
        ```bash
        conda create -n dataframes python=3.6
        conda activate dataframes
        conda install arrow-cpp=0.10.* -c conda-forge
        conda install pyarrow=0.10.* -c conda-forge
        conda install rapidjson
        ```
    * With that in place, you need to instruct CMake where to find the libraries you've just installed. Add the following lines to `native_libs/src/CMakeLists.txt`:
        ```cmake
        set(CMAKE_LIBRARY_PATH "/anaconda3/envs/dataframes/lib")
        set(CMAKE_INCLUDE_PATH "/anaconda3/envs/dataframes/include")
        ```
    And you should be all set.
* build the helper C++ library — CMake will automatically place the built binary in the native_libs/platform directory, so `luna` should out-of-the-box be able to find it.
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
* happily use the dataframes library

## Overview
The library currently provides wrappers for Apache Arrow structures.

### Storage types
* `ArrayData` — type-erased storage for `Array` consisting of several contiguous memory buffers. The buffer count depends on stored type. Typically there are two buffers: one for values and one for masking nulls. More comples types (union, lists) will use more.
* `Array tag` — data array with strongly typed accessors. See section below for supported `tag` types.
* `ChunkedArray tag` — a list of `Array`s of the same type viewed as a single large array. Allows storing large sequences of data (that could not be feasably stored in a single memory block) and efficient slice / concat operations.
* `Column` — type erased accessor for a named `ChunkedArray`. Stored type is represented by using one of its constructors. Described by `Field`.
* `Table` — ordered sequence of `Column`s. Described by `Schema`.

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
CSV and Feather files are supported. XLSX files are supported if the helper C++ library was built with XLNT third-part library enabled.

| Format                                          | Parser Type     | Generator Type     | Remarks                                                       |
|-------------------------------------------------|-----------------|--------------------|---------------------------------------------------------------|
| [CSV file](https://tools.ietf.org/html/rfc4180) | `CSVParser`     | `CSVGenerator`     |                                                               |
| XLSX                                            | `XLSXParser`    | `XLSXGenerator`    | Requires optional XLNT library                                |
| Feather                                         | `FeatherParser` | `FeatherGenerator` | Best performance, not all value types are currently supported |


#### Methods
Parser type shall provide the following method:
* `readFile path :: Text -> Table`
Generator type shall provide the following method:
* `writeFile path table :: Text -> Table -> IO None`

Column names are by default read from the file. CSV and XLSX parsers can also work with files that do not contain the reader row. In such case one of the methods below should be called:
* `useCustomNames names` where `names :: [Text]` are user-provided list of desired column names. If there are more columns in file than names count, then more names will be generated.
* `useGeneratedColumnNames` — all column names will be automatically generated (and the first row will be treated as containing values).

Similarly, the CSV and XLSX generators can be configured whether to output a heading row with names.
* `setHeaderPolicy WriteHeaderLine` or `setHeaderPolicy SkipHeaderLine`

The CSV generator can be also configured whether the fields should be always enclosed within quotes or whether this should be done only when necessary (the latter being the default):
* `setQuotingPolicy QuoteWhenNeeded` or `setQuotingPolicy QuoteAllFields`


### Other types
* `DataType` represents the type of values being stored in a `ArrayData`. Note that this type does not contain information whether it is nullable — being nullable is a property of `Field`, not `Datatype`.
* `Field` is a named `DataType` with an additional information whether values are nullable. Describes contents of the `Column`.
* `Schema` is a sequence of `Field`s describing the `Table`.

## Data processing API


## Data description API
### Table
* `corr` — calculates correlation matrix, with Pearson correlation coefficient for each column pair.
* `corrWith columnName` — calculates Pearson correlation coefficient between given column in a table and its other columns.
* `countValues columnName` — returns table with pairs (value, count).
* `describeNa` — calculates count of null values and their ratio to the total row count.
* `describe columnName` — calculates a number of statistics for a given column (mean, std, min, quartiles, max).

### Column
* `countMissing` — returns the number of null values in the column.
* `countValues` — counts occurences of each unique value and returns pairs (value, count).
* stats:
  * `min` — minimum of values.
  * `max` — maximum of values.
  * `mean` — mean of values.
  * `median` — median of values (interpolated, if value count is even)
  * `std` — standard deviation of values.
  * `var` — variation of values.
  * `sum` — sum of values.
  * `quantile q` — value at given quantile, q belongs to <0,1>.
* `describe` — calculates a number of statistics for a given column (mean, std, min, quartiles, max).

## Tutorial
TBD
