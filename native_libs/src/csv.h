#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <nonstd/variant.hpp>
#include <vector>

#include <arrow/type.h>

#include "Core/Common.h"

namespace arrow
{
    class Table;
}

struct EXPORT NaiveStringView
{
    char *text{};
    std::ptrdiff_t length{};
    NaiveStringView(char *text,  std::ptrdiff_t length)
        : text(text), length(length)
    {}

    NaiveStringView(NaiveStringView &&) = default;

    friend auto &operator<<(std::ostream &out, const NaiveStringView &nsv)
    {
        out.write(nsv.text, nsv.length);
        return out;
    }

    auto str() const
    {
        return std::string{ text, text + length };
    }
    friend bool operator==(const NaiveStringView &lhs, const NaiveStringView &rhs)
    {
        return lhs.str() == rhs.str();
    }
};

struct ParsedCsv
{
    using Field = NaiveStringView;
    using Record = std::vector<Field>;
    using Table = std::vector<Record>;

    std::unique_ptr<std::string> buffer; // note: due to SSO we can't keep buffer by value, as we want data memory address to be constant
    Table records;

    size_t fieldCount{};
    size_t recordCount{};

    ParsedCsv(std::unique_ptr<std::string> buffer, Table records);
    ParsedCsv(const ParsedCsv &) = delete;
    ParsedCsv(ParsedCsv &&) = default;
};

struct TakeFirstRowAsHeaders {};
struct GenerateColumnNames {};
using HeaderPolicy = nonstd::variant<TakeFirstRowAsHeaders, GenerateColumnNames, std::vector<std::string>>;

struct ColumnType
{
    std::shared_ptr<arrow::DataType> type;
    bool nullable;

    ColumnType(std::shared_ptr<arrow::DataType> type, bool nullable)
        : type(type), nullable(nullable)
    {}
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

EXPORT NaiveStringView parseField(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT std::vector<NaiveStringView> parseRecord(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT std::vector<std::vector<NaiveStringView>> parseCsvTable(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT ParsedCsv parseCsvFile(const char *filepath, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
EXPORT ParsedCsv parseCsvData(std::string data, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
EXPORT std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes);

EXPORT void generateCsv(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
