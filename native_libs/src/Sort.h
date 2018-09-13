#pragma once

#include <memory>
#include <vector>
#include "Core/Common.h"

namespace arrow
{
    class Array;
    class Column;
    class Table;
}


enum class SortOrder : uint8_t
{
    Ascending, Descending
};

enum class NullPosition : uint8_t
{
    Before, After
};

using Permutation = std::vector<int64_t>; // [new index] -> old index

DFH_EXPORT std::shared_ptr<arrow::Array> permuteToArray(const std::shared_ptr<arrow::Column> &column, const Permutation &indices);
DFH_EXPORT std::shared_ptr<arrow::Column> permute(const std::shared_ptr<arrow::Column> &column, const Permutation &indices);
DFH_EXPORT std::shared_ptr<arrow::Table> permute(const std::shared_ptr<arrow::Table> &table, const Permutation &indices);

struct SortBy
{
    std::shared_ptr<arrow::Column> column;
    SortOrder order;
    NullPosition nulls;

    SortBy(std::shared_ptr<arrow::Column> column, SortOrder order = SortOrder::Ascending, NullPosition nulls = NullPosition::Before)
        : column(column), order(order), nulls(nulls)
    {}
};

DFH_EXPORT std::shared_ptr<arrow::Table> sortTable(const std::shared_ptr<arrow::Table> &table, const std::vector<SortBy> &sortBy);

