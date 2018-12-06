#pragma once
#include "Core/Common.h"
#include "IO.h"

namespace arrow
{
    class Table;
}

struct XlsxReadOptions
{
    HeaderPolicy header = TakeFirstRowAsHeaders{};
    std::vector<ColumnType> columnTypes = {};
};

struct XlsxWriteOptions
{
    GeneratorHeaderPolicy headerPolicy = GeneratorHeaderPolicy::GenerateHeaderLine;
};

struct DFH_EXPORT FormatXLSX : TableFileHandlerWithOptions<XlsxReadOptions, XlsxWriteOptions>
{
    using TableFileHandler::read;
    using TableFileHandler::write;

    virtual std::string fileSignature() const override;
    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath, const XlsxReadOptions &options) const override;
    virtual void write(std::string_view filePath, const arrow::Table &table, const XlsxWriteOptions &options) const override;
    virtual std::vector<std::string> fileExtensions() const override;
};
