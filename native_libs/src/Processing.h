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

DFH_EXPORT std::shared_ptr<arrow::Buffer> slice(std::shared_ptr<arrow::Buffer> buffer, int64_t startAt, int64_t length);
DFH_EXPORT std::shared_ptr<arrow::Array> slice(std::shared_ptr<arrow::Array> array, int64_t startAt, int64_t length);
DFH_EXPORT std::shared_ptr<arrow::Column> slice(std::shared_ptr<arrow::Column> column, int64_t startAt, int64_t length);
DFH_EXPORT std::shared_ptr<arrow::Table> slice(std::shared_ptr<arrow::Table> table, int64_t startAt, int64_t length);

DFH_EXPORT std::shared_ptr<arrow::Column> interpolateNA(std::shared_ptr<arrow::Column> column);
DFH_EXPORT std::shared_ptr<arrow::Table> interpolateNA(std::shared_ptr<arrow::Table> table);

DFH_EXPORT std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table, const std::vector<int> &columnIndices);
DFH_EXPORT std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table);

DFH_EXPORT std::shared_ptr<arrow::Array> fillNA(std::shared_ptr<arrow::Array> column, DynamicField value);
DFH_EXPORT std::shared_ptr<arrow::ChunkedArray> fillNA(std::shared_ptr<arrow::ChunkedArray> column, DynamicField value);
DFH_EXPORT std::shared_ptr<arrow::Column> fillNA(std::shared_ptr<arrow::Column> column, DynamicField value);
DFH_EXPORT std::shared_ptr<arrow::Table> fillNA(std::shared_ptr<arrow::Table> table, const std::unordered_map<std::string, DynamicField> &valuesPerColumn);

DFH_EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText);
DFH_EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const arrow::Buffer &maskBuffer);
DFH_EXPORT std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText);
DFH_EXPORT std::shared_ptr<arrow::Column> shift(std::shared_ptr<arrow::Column> column, int64_t offset);

DFH_EXPORT DynamicField adjustTypeForFilling(DynamicField valueGivenByUser, const arrow::DataType &type);
DFH_EXPORT std::shared_ptr<arrow::Table> groupBy(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Column> keyColumn);
