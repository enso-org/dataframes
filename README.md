# Dataframes implementation in Luna

## Purpose
This project is a library with dataframes implementation. Dataframes are structures allowing more comfortable work with big datasets.

## Build & Install


## Tutorial
Luna Dataframes are structures consisting of a header, type and data. The header is type of `Maybe(List Text)`, the type is a list of `FieldType` and data has 2 dimensional array of `CStrings`. The `FieldType` can be `Text`, `Real` or custom category. At this point by default the type is set to `Text`.
Current implementation of Dataframes is providing functionalities like reading and writing csv files, copying rows and columns, joining, finding empty fields and `map`. To see the example usage please see `Main.luna` file.
