#include "IO.h"
#include "XLSX.h"
#include "csv.h"
#include "Feather.h"

#if __cpp_lib_filesystem >= 201703
#include <filesystem>
#endif
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>

#include <boost/algorithm/string/predicate.hpp>

namespace
{
auto supportedFormatHandlers()
{
    std::vector<std::unique_ptr<TableFileHandler>> handlers;
    handlers.push_back(std::make_unique<FormatXLSX>());
    handlers.push_back(std::make_unique<FormatFeather>());
    handlers.push_back(std::make_unique<FormatCSV>());
    return handlers;
}

std::vector<std::string> defaultColumnNames(int count)
{
    std::vector<std::string> ret;
    for(int column = 0; column < count; column++)
        ret.push_back("col" + std::to_string(column));
    return ret;
}
}


std::vector<std::string> decideColumnNames(int count, const HeaderPolicy &policy, std::function<std::string(int)> readHeaderCell)
{    
    const auto suppliedNames = [&] () -> std::vector<std::string>
    {
        if(auto names = get_if<std::vector<std::string>>(&policy))
            return *names;
        return {};
    }();

    std::vector<std::string> ret;
    for(int column = 0; column < count; column++)
    {
        if(holds_alternative<TakeFirstRowAsHeaders>(policy))
        {
            const auto cellText = readHeaderCell(column);
            if(cellText.size())
                ret.push_back(cellText);
            else
                ret.push_back("MISSING_" + std::to_string(column));
        }
        else if(column >= suppliedNames.size())
            ret.push_back("col" + std::to_string(column));
        else
            ret.push_back(suppliedNames.at(column));
    }
    return ret;
}

std::shared_ptr<arrow::Table> buildTable(std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> arrays, std::vector<ColumnType> columnTypes)
{
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for(int column = 0; column < arrays.size(); column++)
    {
        const auto array = arrays.at(column);
        const auto nullable = array->null_count() > 0 || columnTypes.at(column).nullable;
        fields.push_back(std::make_shared<arrow::Field>(names.at(column), columnTypes.at(column).type, nullable));
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, arrays);
    return table;
}

std::shared_ptr<arrow::Table> readTableFromFile(std::string_view filepath)
{
    for(auto &&handler : supportedFormatHandlers())
        if(auto table = handler->tryReading(filepath))
            return table;

    THROW("Failed to load file {}: it doesn't parse with default settings as any of the supported formats");
}

void writeTableToFile(std::string_view filepath, const arrow::Table &table)
{
    for(auto &&handler : supportedFormatHandlers())
        if(handler->filePathExtensionMatches(filepath))
            return handler->write(filepath, table);
    
    THROW("cannot write table to {}: cannot deduce format type from extension", filepath);
}

std::ofstream openFileToWrite(std::string_view filepath)
{
    // what we care about is mostly MSVC because on Windows paths are not utf-8 by default
    // and fortunately MSVC implements C++17 filesystem library
#if __cpp_lib_filesystem >= 201703
    std::ofstream out{ std::filesystem::u8path(filepath), std::ios::binary };
#else
    std::ofstream out{ (std::string)filepath, std::ios::binary };
#endif

    if(!out)
        THROW("Cannot open the file to write: {}", filepath);

    return out;
}

void writeFile(std::string_view filepath, std::string_view contents)
{
    auto out = openFileToWrite(filepath);
    out.write(contents.data(), contents.size());
    if(!out)
        THROW("Failed while writing file `{}`", filepath);
}

std::ifstream openFileToRead(std::string_view filepath)
{
    // see comment in the function above
#if __cpp_lib_filesystem >= 201703
    std::ifstream in{ std::filesystem::u8path(filepath), std::ios::binary };
#else
    std::ifstream in{ (std::string)filepath, std::ios::binary };
#endif

    if(!in)
        THROW("Cannot open the file to read: {}", filepath);

    return in;
}

std::string getFileContents(std::string_view filepath)
{
    try
    {
        auto input = openFileToRead(filepath);

        input.seekg(0, std::ios::end);
        auto length = input.tellg();
        if(length == -1)
            THROW("failed to tell the file's length");

        std::string contents;
        contents.resize(input.tellg());
        input.seekg(0, std::ios::beg);
        input.read(&contents[0], contents.size());
        if(!input)
            THROW("failed reading stream");

        return contents;
    }
    catch(std::exception &e)
    {
        THROW("Failed to load file {}: {}", filepath, e.what());
    }
}

ColumnType::ColumnType(std::shared_ptr<arrow::DataType> type, bool nullable, bool deduced) 
    : type(type), nullable(nullable), deduced(deduced)
{
}

ColumnType::ColumnType(const arrow::Column &column, bool deduced) 
    : type(column.type()), nullable(column.field()->nullable()), deduced(deduced)
{
}

std::shared_ptr<arrow::Table> TableFileHandler::tryReading(std::string_view filePath) const
{
    try
    {
        // short path: if the file header is wrong, don't bother reading the whole thing and parsing
        if(!fileMightBeCompatible(filePath))
            return nullptr;
        
        return read(filePath);
    }
    catch(...)
    {
        return nullptr;
    }
}

bool TableFileHandler::fileMightBeCompatible(std::string_view filePath) const
{
    auto expectedSignature = fileSignature();
    auto input = openFileToRead(filePath);

    std::string buffer(expectedSignature.size(), '\0');
    input.read(buffer.data(), buffer.size());
    bool readOk = !!input;

    // restore pristine input state
    input.clear();
    input.seekg(0);
    return readOk && expectedSignature == buffer;
}

bool TableFileHandler::filePathExtensionMatches(std::string_view filePath) const
{
    for(auto &&extension : fileExtensions())
    {
        const auto extensionSize = extension.size();
        if(filePath.size() < extensionSize + 1)
            continue;
        if(*(filePath.rbegin() + extensionSize) != '.')
            continue;
        if(boost::iends_with(filePath, extension))
            return true;
    }

    return false;
}
