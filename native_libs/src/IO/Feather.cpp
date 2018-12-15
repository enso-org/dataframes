#include "Feather.h"
#include "IO.h"

#include <arrow/io/file.h>
#include <arrow/ipc/feather.h>
#include <arrow/table.h>

#include "Core/ArrowUtilities.h"

std::string FormatFeather::fileSignature() const
{
    return "FEA1";
}

std::shared_ptr<arrow::Table> FormatFeather::read(std::string_view filePath) const
{
    std::shared_ptr<arrow::io::ReadableFile> out;
    checkStatus(arrow::io::ReadableFile::Open((std::string)filePath, &out));

    std::unique_ptr<arrow::ipc::feather::TableReader> reader;
    checkStatus(arrow::ipc::feather::TableReader::Open(out, &reader));

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Column>> columns;
    fields.resize(reader->num_columns());
    columns.resize(reader->num_columns());

    for(int columnIndex = 0; columnIndex < reader->num_columns(); columnIndex++)
    {
        auto &columnTarget = columns.at(columnIndex);
        checkStatus(reader->GetColumn(columnIndex, &columnTarget));
        fields.at(columnIndex) = columnTarget->field();
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, columns);
    return table;
}

void FormatFeather::write(std::string_view filePath, const arrow::Table &table) const
{
    std::shared_ptr<arrow::io::FileOutputStream> out;
    checkStatus(arrow::io::FileOutputStream::Open((std::string)filePath, &out));

    std::unique_ptr<arrow::ipc::feather::TableWriter> writer;
    checkStatus(arrow::ipc::feather::TableWriter::Open(out, &writer));

    writer->SetNumRows(table.num_rows());
    for(auto columnIndex = 0; columnIndex < table.num_columns(); columnIndex++)
    {
        const auto column = table.column(columnIndex);
        for(auto &chunk : column->data()->chunks())
            checkStatus(writer->Append(column->name(), *chunk));
    }

    checkStatus(writer->Finalize());
}

std::vector<std::string> FormatFeather::fileExtensions() const
{
    return { "feather" };
}
