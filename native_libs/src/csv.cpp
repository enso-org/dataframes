#include "csv.h"
#include "IO.h"

#include <algorithm>
#include <iostream>
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

enum class MissingField
{
    AsNull, AsZeroValue
};

struct ColumnPolicy
{
    arrow::Type::type type;
    MissingField missing;
};

template<arrow::Type::type type> struct BuilderFor_                      {};
template<>                       struct BuilderFor_<arrow::Type::INT64>  { using Builder = arrow::Int64Builder;  };
template<>                       struct BuilderFor_<arrow::Type::DOUBLE> { using Builder = arrow::DoubleBuilder; };
template<>                       struct BuilderFor_<arrow::Type::STRING> { using Builder = arrow::StringBuilder; };
template<arrow::Type::type type> using  BuilderFor = typename BuilderFor_<type>::Builder;


template<typename Builder>
std::shared_ptr<arrow::Array> finish(Builder &builder)
{
    std::shared_ptr<arrow::Array> ret;
    auto status = builder.Finish(&ret);
    if(!status.ok())
        throw std::runtime_error(status.ToString());

    return ret;
}

template<arrow::Type::type type>
struct ColumnBuilder
{
    MissingField missingField;
    BuilderFor<type> builder;

    ColumnBuilder(MissingField missingField) : missingField(missingField) {}

    void addFromString(const NaiveStringView &field)
    {
        const auto fieldEnd = field.text + field.length;
        *fieldEnd = '\0';
        char *next = nullptr;
        if constexpr(type == arrow::Type::INT64)
        {
            auto v = std::strtoll(field.text, &next, 10);
            if(next == fieldEnd)
            {
                builder.Append(v);
            }
            else
            {
                addMissing();
            }
        }
        else if constexpr(type == arrow::Type::DOUBLE)
        {
            auto v = std::strtod(field.text, &next);
            if(next == fieldEnd)
            {
                builder.Append(v);
            }
            else
            {
                addMissing();
            }
        }
        else if constexpr(type == arrow::Type::STRING)
        {
            builder.Append(field.text, field.length);
        }
        else
            throw std::runtime_error("wrong type");
    }
    void addMissing()
    {
        builder.AppendNull();
    }
};

std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes)
{
    // empty table
    if(csv.recordCount == 0 || csv.fieldCount == 0)
    {
        auto schema = std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
        return arrow::Table::Make(schema, std::vector<std::shared_ptr<arrow::Array>>{});
    }

    if(columnTypes.size() < csv.fieldCount)
        columnTypes.resize(csv.fieldCount, ColumnType{std::make_shared<arrow::StringType>(), false});

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(csv.fieldCount);

    const bool takeFirstRowAsNames = std::holds_alternative<TakeFirstRowAsHeaders>(header);
    const auto suppliedNames = [&] () -> std::vector<std::string>
    {
        if(auto names = std::get_if<std::vector<std::string>>(&header))
            return *names;
        return {};
    }();

    const int startRow = takeFirstRowAsNames ? 1 : 0;
    const auto missingFieldsPolicy = MissingField::AsNull;

    for(int column = 0; column < csv.fieldCount; column++)
    {
        auto processColumn = [&] (auto &&builder)
        {
            for(int row = startRow; row < csv.recordCount; row++)
            {
                const auto &record = csv.records[row];
                if(column >= record.size())
                {
                    builder.addMissing();
                }
                else
                {
                    const auto &field = csv.records[row][column];
                    builder.addFromString(field);
                }
            }
            arrays.push_back(finish(builder.builder));
        };

        switch(columnTypes.at(column).type->id())
        {
        case arrow::Type::INT64:
            processColumn(ColumnBuilder<arrow::Type::INT64>{missingFieldsPolicy});
            break;
        case arrow::Type::DOUBLE:
            processColumn(ColumnBuilder<arrow::Type::DOUBLE>{missingFieldsPolicy});
            break;
        case arrow::Type::STRING:
            processColumn(ColumnBuilder<arrow::Type::STRING>{missingFieldsPolicy});
            break;
        }
    }


    const auto names = [&]
    {
        std::vector<std::string> ret;
        for(int column = 0; column < csv.fieldCount; column++)
        {
            if(takeFirstRowAsNames)
            {
                const auto &headerRow = csv.records[0];
                if(column < headerRow.size())
                    ret.push_back(headerRow[column].str());
                else
                    ret.push_back("MISSING_" + std::to_string(column));
            }
            else if(column >= suppliedNames.size())
                ret.push_back("col" + std::to_string(column));
            else
                ret.push_back(suppliedNames.at(column));
        }
        return ret;
    }();


    std::vector<std::shared_ptr<arrow::Field>> fields;
    for(int column = 0; column < csv.fieldCount; column++)
    {
        const auto array = arrays.at(column);
        std::cout << column << " : " << array->null_count() << " " << array->length() << std::endl;
        const auto nullable = arrays.at(column)->null_count();
        fields.push_back(std::make_shared<arrow::Field>(names[column], columnTypes[column].type, nullable));
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
