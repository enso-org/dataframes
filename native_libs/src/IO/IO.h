#pragma once

#include <functional>
#include <string>
#include <iosfwd>
#include <stdexcept>
#include <vector>

#include <arrow/type.h>

#include "variant.h"

#include "Core/Common.h"

namespace arrow
{
    class Array;
    class ArrayBuilder;
    class DataType;
}

struct TakeFirstRowAsHeaders {};
struct GenerateColumnNames {};

using HeaderPolicy = std::variant<TakeFirstRowAsHeaders, GenerateColumnNames, std::vector<std::string>>;

struct DFH_EXPORT ColumnType
{
    std::shared_ptr<arrow::DataType> type;
    bool nullable;
    bool deduced; // deduced types allow fallback

    ColumnType(std::shared_ptr<arrow::DataType> type, bool nullable, bool deduced);
    ColumnType(const arrow::Column &column, bool deduced);
};

enum class GeneratorHeaderPolicy : int8_t
{
    GenerateHeaderLine,
    SkipHeaderLine

};

enum class GeneratorQuotingPolicy : int8_t
{
    QuoteWhenNeeded,
    QueteAllFields
};

template<arrow::Type::type type>
inline constexpr auto defaultValue()
{
    if constexpr(type == arrow::Type::STRING)
    {
        return "";
    }
    else if constexpr(type == arrow::Type::INT64 || type == arrow::Type::TIMESTAMP)
    {
        return std::int64_t(0);
    }
    else if constexpr(type == arrow::Type::DOUBLE)
    {
        return 0.0;
    }
    else
        throw std::runtime_error(__FUNCTION__ + std::string(" : type not supported ") + std::to_string(type));
}

std::vector<std::string> decideColumnNames(int count, const HeaderPolicy &policy, std::function<std::string(int)> readHeaderCell);
std::shared_ptr<arrow::Table> buildTable(std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> arrays, std::vector<ColumnType> columnTypes);

DFH_EXPORT std::ofstream openFileToWrite(const char *filepath);
DFH_EXPORT void writeFile(const char *filepath, std::string_view contents);
DFH_EXPORT std::ifstream openFileToRead(const char *filepath);
DFH_EXPORT std::string getFileContents(const char *filepath);
