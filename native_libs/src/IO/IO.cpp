#include "IO.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>

std::vector<std::string> defaultColumnNames(int count)
{
    std::vector<std::string> ret;
    for(int column = 0; column < count; column++)
        ret.push_back("col" + std::to_string(column));
    return ret;
}

std::vector<std::string> decideColumnNames(int count, const HeaderPolicy &policy, std::function<std::string(int)> readHeaderCell)
{    
    const auto suppliedNames = [&] () -> std::vector<std::string>
    {
        if(auto names = nonstd::get_if<std::vector<std::string>>(&policy))
            return *names;
        return {};
    }();

    std::vector<std::string> ret;
    for(int column = 0; column < count; column++)
    {
        if(nonstd::holds_alternative<TakeFirstRowAsHeaders>(policy))
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

std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder &builder)
{
    std::shared_ptr<arrow::Array> ret;
    auto status = builder.Finish(&ret);
    if(!status.ok())
        throw std::runtime_error(status.ToString());

    return ret;
}

std::string getFileContents(const char *filepath)
{
    try
    {
        std::ifstream input{filepath};
        if(!input)
            throw std::runtime_error("Failed to open the file");

        std::string contents;
        input.seekg(0, std::ios::end);
        contents.resize(input.tellg());
        input.seekg(0, std::ios::beg);
        input.read(&contents[0], contents.size());
        input.close();
        return(contents);
    }
    catch(std::exception &e)
    {
        // make sure that filename is recorded in the error message
        std::stringstream errMsg;
        errMsg << "Failed to load file `" << filepath << "`: " << e.what();
        throw std::runtime_error(errMsg.str());
    }
}