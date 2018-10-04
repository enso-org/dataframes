#pragma once

#include <Core/Common.h>
#include "Python/IncludePython.h"

namespace arrow
{
    class ChunkedArray;
    class Column;
    class Table;
}

EXPORT pybind11::array_t<double> columnToNpArr(const arrow::Column &col);
EXPORT std::shared_ptr<arrow::Column> npArrayToColumn(pybind11::array_t<double> arr, std::string name);
EXPORT pybind11::array tableToNpMatrix(const arrow::Table& table);

namespace sklearn
{
    EXPORT void fit(pybind11::object model, const arrow::Table &xs, const arrow::Column &y);
    EXPORT double score(pybind11::object model, const arrow::Table &xs, const arrow::Column &y);
    EXPORT std::shared_ptr<arrow::Column> predict(pybind11::object model, const arrow::Table &xs);
    EXPORT std::shared_ptr<arrow::Table> confusionMatrix(const arrow::Column &ytrue, const arrow::Column &ypred);
}

extern "C"
{
    EXPORT void toNpArr(const arrow::Table* tb, const char **outError) noexcept;
    EXPORT void freePyObj(PyObject* o, const char **outError) noexcept;
    EXPORT PyObject* newLogisticRegression(double C, const char **outError) noexcept;
    EXPORT PyObject* newLinearRegression(const char **outError) noexcept;
    EXPORT void fit(PyObject* model, const arrow::Table *xs, const arrow::Column *y, const char **outError) noexcept;
    EXPORT double score(PyObject* model, const arrow::Table* xs, const arrow::Column* y, const char **outError) noexcept;
    EXPORT arrow::Column* predict(PyObject* model, const arrow::Table* xs, const char **outError) noexcept;
    EXPORT arrow::Table* confusionMatrix(const arrow::Column* ytrue, const arrow::Column* ypred, const char **outError) noexcept;
    EXPORT arrow::Table* oneHotEncode(const arrow::Column* col, const char **outError) noexcept;
}