#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

#include <arrow/type.h>

#include "Core/Common.h"
#include "IO.h"

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

struct EXPORT CsvParser
{
    char *bufferStart{};
    char *bufferIterator{};
    char *bufferEnd{};

    size_t lastColumnCount = 0;

    char fieldSeparator{};
    char recordSeparator{};
    char quote{};

    CsvParser(char *bufferStart, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote)
        : bufferStart(bufferStart), bufferIterator(bufferStart)
        , bufferEnd(bufferEnd), fieldSeparator(fieldSeparator)
        , recordSeparator(recordSeparator), quote(quote)
    {}

    explicit CsvParser(std::string &s)
        : CsvParser(s.data(), s.data() + s.length(), ',', '\n', '"')
    {}


    NaiveStringView parseField(); // sets buffer Iterator to the next separator
    std::vector<NaiveStringView> parseRecord();
    std::vector<std::vector<NaiveStringView>> parseCsvTable();
};

EXPORT ParsedCsv parseCsvFile(const char *filepath, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
EXPORT ParsedCsv parseCsvData(std::string data, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
EXPORT std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes);

EXPORT void generateCsv(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
