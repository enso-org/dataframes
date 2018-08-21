#pragma once

#include <memory>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"

namespace arrow
{
    class Array;
    class Buffer;
    class Column;
    class Table;
}


EXPORT std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table, const std::vector<int> &columnIndices);
EXPORT std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table);

EXPORT std::shared_ptr<arrow::Array> fillNA(std::shared_ptr<arrow::Array> column, DynamicField value);
EXPORT std::shared_ptr<arrow::ChunkedArray> fillNA(std::shared_ptr<arrow::ChunkedArray> column, DynamicField value);
EXPORT std::shared_ptr<arrow::Column> fillNA(std::shared_ptr<arrow::Column> column, DynamicField value);
EXPORT std::shared_ptr<arrow::Table> fillNA(std::shared_ptr<arrow::Table> table, const std::unordered_map<std::string, DynamicField> &valuesPerColumn);

EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText);
EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const arrow::Buffer &maskBuffer);
EXPORT std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText);

EXPORT DynamicField adjustTypeForFilling(DynamicField valueGivenByUser, const arrow::DataType &type);