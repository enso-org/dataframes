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

using HeaderPolicy = variant<TakeFirstRowAsHeaders, GenerateColumnNames, std::vector<std::string>>;

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

DFH_EXPORT std::shared_ptr<arrow::Table> readTableFromFile(std::string_view filepath);
DFH_EXPORT void writeTableToFile(std::string_view filepath, const arrow::Table &table);
DFH_EXPORT std::ofstream openFileToWrite(std::string_view filepath);
DFH_EXPORT void writeFile(std::string_view, std::string_view contents);
DFH_EXPORT std::ifstream openFileToRead(std::string_view filepath);
DFH_EXPORT std::string getFileContents(std::string_view filepath);

// Basic interface for classes that perform table IO for specific file formats
struct TableFileHandler
{
    virtual ~TableFileHandler() = default;

    virtual std::string fileSignature() const = 0;
    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath) const = 0;  // throws on failure
    virtual void write(std::string_view filePath, const arrow::Table &table) const = 0;
    virtual std::vector<std::string> fileExtensions() const = 0;

    std::shared_ptr<arrow::Table> tryReading(std::string_view filePath) const; // returns nullptr on failure
    bool fileMightBeCompatible(std::string_view filePath) const; // might give false positive (just checks signature)
    bool filePathExtensionMatches(std::string_view filePath) const;
};

template<typename ReadOptions, typename WriteOptions>
struct TableFileHandlerWithOptions : TableFileHandler
{
    using TableFileHandler::read;
    using TableFileHandler::write;

    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath) const
    {
        return read(filePath, ReadOptions{});
    }
    virtual void write(std::string_view filePath, const arrow::Table &table) const
    {
        return write(filePath, table, WriteOptions{});
    }

    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath, const ReadOptions &options) const = 0;
    virtual void write(std::string_view filePath, const arrow::Table &table, const WriteOptions &options) const = 0;
};
