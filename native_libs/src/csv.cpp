#include "csv.h"
#include "IO.h"

#include <algorithm>
#include <sstream>

#include <arrow/table.h>
#include <arrow/builder.h>

    // returns field, sets buffer Iterator to the next separator
NaiveStringView parseField(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote)
{
    if(bufferIterator == bufferEnd)
        NaiveStringView{bufferIterator, 0};

    char firstChar = *bufferIterator;
    bool quoted = firstChar == quote;

    // Fast path for unquoted fields
    // Just consume text until separator is encountered
    if(!quoted)
    {
        const auto start = bufferIterator;
        for(; bufferIterator != bufferEnd; ++bufferIterator)
        {
            char c = *bufferIterator;
            if(c == fieldSeparator || c == recordSeparator)
                break;
        }
        return { start, std::distance(start, bufferIterator) };
    }

    // Quoted fields
    ++bufferIterator; // consume initial quote - won't be needed
    const auto start = bufferIterator;

    int rewriteOffset = 0;

    for(; bufferIterator != bufferEnd; ++bufferIterator)
    {
        char c = *bufferIterator;

        if(c == quote)
        {
            // Either field terminator or nested double quote
            const auto nextIterator = bufferIterator+1;
            const auto nextIsAlsoQuote = nextIterator!=bufferEnd && *nextIterator == quote;
            if(nextIsAlsoQuote)
            {
                ++rewriteOffset;
                ++bufferIterator;
            }
            else
            {
                auto length = std::distance(start, bufferIterator++);
                return { start, length - rewriteOffset };
            }
        }

        if(rewriteOffset)
            *(bufferIterator - rewriteOffset) = c;
    }

    throw std::runtime_error("reached the end of the file with an unmatched quote character");
}

std::vector<NaiveStringView> parseRecord(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote)
{
    std::vector<NaiveStringView> ret;

    while(true)
    {
        char initialCharacter = *bufferIterator;

        ret.push_back(parseField(bufferIterator, bufferEnd, fieldSeparator, recordSeparator, quote));

        if(bufferIterator >= bufferEnd)
            return ret;

        if(*bufferIterator++ == recordSeparator)
            return ret;
    }
}

std::vector<std::vector<NaiveStringView>> parseCsvTable(char *&bufferIterator, char *bufferEnd, char fieldSeparator, char recordSeparator, char quote)
{
    std::vector<std::vector<NaiveStringView>> ret;

    for( ; bufferIterator < bufferEnd; )
    {
        ret.push_back(parseRecord(bufferIterator, bufferEnd, fieldSeparator, recordSeparator, quote));
    }

    return ret;
}

ParsedCsv parseCsvFile(const char *filepath, char fieldSeparator, char recordSeparator, char quote)
{
    auto buffer = getFileContents(filepath);
    auto itr = buffer.data();
    auto table = parseCsvTable(itr, itr + buffer.size(), fieldSeparator, recordSeparator, quote);
    return { std::move(buffer), std::move(table) };
}

template <typename Builder>
void parse(const NaiveStringView &field, Builder &builder);

template<>
void parse<arrow::Int64Builder>(const NaiveStringView &field, arrow::Int64Builder &builder)
{
    std::istringstream fieldText{std::string{field.text, field.text + field.length}};
    int64_t value = -1;
    fieldText >> value;
    if(!fieldText.fail() && !fieldText.bad() && fieldText.eof())
    {
        builder.Append(value);
    }
    else
    {
        builder.AppendNull();
    }
}

std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv)
{
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.resize(csv.fieldCount);

    for(int column = 0; column < csv.fieldCount; column++)
    {
        arrow::Int64Builder builder;
        for(int row = 0; row < csv.recordCount; row++)
        {
            const auto &record = csv.records[row];
            if(column >= record.size())
            {
                builder.AppendNull();
            }
            else
            {
                const auto &field = csv.records[row][column];
                parse(field, builder);
            }
        }

        auto status = builder.Finish(&arrays[column]);
        if(!status.ok())
            throw std::runtime_error("Failed to finish building column " + std::to_string(column) + ": " + status.ToString());
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for(int column = 0; column < csv.fieldCount; column++)
    {
        fields.push_back(std::make_shared<arrow::Field>("column" + std::to_string(column), std::make_shared<arrow::Int64Type>()));
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, arrays);
    return table;
}

bool needsEscaping(const std::string &record, char seperator)
{
    if(record.empty())
        return false;

    if(record.front() == ' ' || record.back() == ' ')
        return true;

    if(record.find(seperator) != std::string::npos)
        return true;

    return false;
}

ParsedCsv::ParsedCsv(std::string buffer, Table records_)
    : buffer(std::move(buffer)), records(std::move(records_))
{
    recordCount = records.size();

    const auto biggestRecord = std::max_element(records.begin(), records.end(), 
        [] (auto &record1, auto &record2) 
            { return record1.size() < record2.size(); });
    
    if(biggestRecord != records.end())
        fieldCount = biggestRecord->size();
}
