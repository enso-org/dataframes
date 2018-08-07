#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

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

    std::string buffer;
    Table records;

    size_t fieldCount{};
    size_t recordCount{};

    ParsedCsv(std::string buffer, Table records);
    ParsedCsv(const ParsedCsv &) = delete;
    ParsedCsv(ParsedCsv &&) = default;
};

EXPORT NaiveStringView parseField(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT std::vector<NaiveStringView> parseRecord(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT std::vector<std::vector<NaiveStringView>> parseCsvTable(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote);
EXPORT ParsedCsv parseCsvFile(const char *filepath, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
EXPORT std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv);
