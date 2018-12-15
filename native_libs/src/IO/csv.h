#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string_view>
#include <vector>

#include <arrow/type.h>

#include "Core/Common.h"
#include "IO.h"

namespace arrow
{
    class Table;
}


DFH_EXPORT arrow::Type::type deduceType(std::string_view text);

struct ParsedCsv
{
    using Field = std::string_view;
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

struct DFH_EXPORT CsvParser
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


    std::string_view parseField(); // sets buffer Iterator to the next separator
    std::vector<std::string_view> parseRecord();
    std::vector<std::vector<std::string_view>> parseCsvTable();
};

DFH_EXPORT ParsedCsv parseCsvData(std::string data, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');
DFH_EXPORT std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes, int typeDeductionDepth);

DFH_EXPORT void generateCsv(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"');

struct CsvCommonOptions
{
    char fieldSeparator = ',';
    char recordSeparator = '\n';
    char quote = '"';
};

struct CsvReadOptions : CsvCommonOptions
{
    HeaderPolicy header = TakeFirstRowAsHeaders{};
    std::vector<ColumnType> columnTypes = {};
    int typeDeductionDepth = 50;
};

struct CsvWriteOptions : CsvCommonOptions
{
    GeneratorHeaderPolicy headerPolicy = GeneratorHeaderPolicy::GenerateHeaderLine;
    GeneratorQuotingPolicy quotingPolicy;    
};

struct DFH_EXPORT FormatCSV : TableFileHandlerWithOptions<CsvReadOptions, CsvWriteOptions>
{
    using TableFileHandler::read;
    using TableFileHandler::write;

    std::shared_ptr<arrow::Table> readString(std::string data, const CsvReadOptions &options) const;
    std::string writeToString(const arrow::Table &table, const CsvWriteOptions &options) const;

    virtual std::string fileSignature() const override;
    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath, const CsvReadOptions &options) const override;
    virtual void write(std::string_view filePath, const arrow::Table &table, const CsvWriteOptions &options) const override;
    virtual std::vector<std::string> fileExtensions() const override;
};
