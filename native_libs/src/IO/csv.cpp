#include "csv.h"
#include "IO.h"
#include "Core/ArrowUtilities.h"
#include "Core/Logger.h"
#include "Core/Utils.h"


#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>

#include <arrow/table.h>
#include <arrow/builder.h>

using namespace std::literals;

arrow::Type::type deduceType(std::string_view text)
{
    if(text.empty())
        return arrow::Type::NA;
    if(parseAs<int64_t>(text))
        return arrow::Type::INT64;
    if(parseAs<double>(text))
        return arrow::Type::DOUBLE;
    return arrow::Type::STRING;
}

ParsedCsv parseCsvFile(const char *filepath, char fieldSeparator, char recordSeparator, char quote)
{
    auto buffer = getFileContents(filepath);
    return parseCsvData(std::move(buffer), fieldSeparator, recordSeparator, quote);
}

ParsedCsv parseCsvData(std::string data, char fieldSeparator /*= ','*/, char recordSeparator /*= '\n'*/, char quote /*= '"'*/)
{
    auto bufferPtr = std::make_unique<std::string>(std::move(data));
    CsvParser parser{bufferPtr->data(), bufferPtr->data() + bufferPtr->size(), fieldSeparator, recordSeparator, quote};
    auto table = parser.parseCsvTable();
    return { std::move(bufferPtr), std::move(table) };
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

template<arrow::Type::type type>
struct ColumnBuilder
{
    MissingField missingField;
    BuilderFor<type> builder;

    ColumnBuilder(MissingField missingField) : missingField(missingField) {}

    void addFromString(const std::string_view &field)
    {
        if constexpr(type == arrow::Type::STRING)
        {
            if(field.size())
                checkStatus(builder.Append(field.data(), (int32_t)field.size()));
            else
               addMissing(); 
        }
        else
        {
            if(field.size() != 0)
            {
                const auto fieldEnd = const_cast<char *>(field.data() + field.size());
                *fieldEnd = '\0';
                char *next = nullptr;
                if constexpr(type == arrow::Type::INT64)
                {
                    if(auto v = parseAs<int64_t>(field))
                    {
                        checkStatus(builder.Append(*v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else if constexpr(type == arrow::Type::DOUBLE)
                {
                    if(auto v = parseAs<double>(field))
                    {
                        checkStatus(builder.Append(*v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else
                    throw std::runtime_error("wrong type");
            }
            else
            {
                addMissing();
            }
        }
    }
    void addMissing()
    {
        if(missingField == MissingField::AsNull)
            checkStatus(builder.AppendNull());
        else
            checkStatus(builder.Append(defaultValue<type>()));
    }
    void reserve(int64_t count)
    {
        checkStatus(builder.Reserve(count));
    }
};

ColumnType deduceType(const ParsedCsv &csv, size_t columnIndex, size_t startRow, size_t lookupDepth)
{
    lookupDepth = std::min<int>(lookupDepth, csv.records.size());

    std::unordered_set<arrow::Type::type> encounteredTypes;
    for(int i = startRow; i < lookupDepth; i++)
    {
        const auto &record = csv.records.at(i);
        if(columnIndex < record.size())
        {
            const auto field = record.at(columnIndex);
            encounteredTypes.insert(deduceType(field));
        }
    }

    auto typePtr = [&]
    {
        if(encounteredTypes.count(arrow::Type::STRING))
            return arrow::TypeTraits<arrow::StringType>::type_singleton();
        if(encounteredTypes.count(arrow::Type::DOUBLE))
            return arrow::TypeTraits<arrow::DoubleType>::type_singleton();
        if(encounteredTypes.count(arrow::Type::INT64))
            return arrow::TypeTraits<arrow::Int64Type>::type_singleton();

        return arrow::TypeTraits<arrow::StringType>::type_singleton();
    }();

    return ColumnType{typePtr, encounteredTypes.count(arrow::Type::NA) > 0, true};
}

std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes, int typeDeductionDepth)
{
    // empty table
    if(csv.recordCount == 0 || csv.fieldCount == 0)
    {
        auto schema = std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
        return arrow::Table::Make(schema, std::vector<std::shared_ptr<arrow::Array>>{});
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(csv.fieldCount);

    const bool takeFirstRowAsNames = std::holds_alternative<TakeFirstRowAsHeaders>(header);
    const int startRow = takeFirstRowAsNames ? 1 : 0;

    // Attempt to deduce all non-specified types
    for(size_t i = columnTypes.size(); i < csv.fieldCount; i++)
    {
        columnTypes.push_back(deduceType(csv, i, startRow, typeDeductionDepth));
    }

    for(int column = 0; column < csv.fieldCount; column++)
    {
        const auto typeInfo = columnTypes.at(column);
        const auto missingFieldsPolicy = (typeInfo.deduced || typeInfo.nullable) ? MissingField::AsNull : MissingField::AsZeroValue;
        auto processColumn = [&] (auto &&builder)
        {
            builder.reserve(csv.recordCount);
            for(int row = startRow; row < csv.recordCount; row++)
            {
                const auto &record = csv.records[row];
                if(column < record.size())
                {
                    const auto &field = csv.records[row][column];
                    builder.addFromString(field);
                }
                else
                {
                    builder.addMissing();
                }
            }
            arrays.push_back(finish(builder.builder));
        };

        visitType(*typeInfo.type, [&] (auto id)
        {
            processColumn(ColumnBuilder<id.value>{missingFieldsPolicy});
        });
    }

    const auto names = decideColumnNames((int)csv.fieldCount, header, [&] (int column)
    {
        const auto &headerRow = csv.records[0];
        if(column < (int)headerRow.size())
            return std::string(headerRow[column]);
        else
            return ""s;
    });
    
    return buildTable(names, arrays, columnTypes);
}

ParsedCsv::ParsedCsv(std::unique_ptr<std::string> buffer, Table records_)
    : buffer(std::move(buffer))
    , records(std::move(records_))
{
    recordCount = records.size();

    const auto biggestRecord = std::max_element(records.begin(), records.end(), 
        [] (auto &record1, auto &record2) 
            { return record1.size() < record2.size(); });
    
    if(biggestRecord != records.end())
        fieldCount = biggestRecord->size();
}

struct CsvGenerator
{
    std::ostream &out;
    GeneratorQuotingPolicy quotingPolicy;
    char fieldSeparator = ',';
    char recordSeparator = '\n';
    char quote = '"';

    explicit CsvGenerator(std::ostream &out, GeneratorQuotingPolicy quotingPolicy, char fieldSeparator = ',', char recordSeparator = '\n', char quote = '"') 
        : out(out)
        , quotingPolicy(quotingPolicy)
        , fieldSeparator(fieldSeparator)
        , recordSeparator(recordSeparator)
        , quote(quote)
    {}

    bool needsEscaping(const char *data, int32_t length) const
    {
        if(length == 0)
            return false;

        if(data[0] == ' ' || data[length-1] == ' ')
            return true;

        for(int i = 0; i < length; i++)
        {
            char c = data[i];
            if(c == fieldSeparator || c == recordSeparator || c == quote)
                return true;
        }

        return false;
    };

    void writeField(const char *data, int32_t length)
    {
        if(quotingPolicy == GeneratorQuotingPolicy::QueteAllFields || needsEscaping(data, length))
        {
            // TODO workaround for mac
        #ifdef _MSC_VER
            out << std::quoted(std::string_view(data, length), quote, quote);            
        #else
            out << std::quoted(std::string{data, data + length}, quote, quote);
        #endif
        }
        else
            out.write(data, length);
    };
};

struct ColumnWriter
{
    char buffer[256];

    std::vector<std::shared_ptr<arrow::Array>> chunks;

    int currentChunk = 0;
    int usedFromChunk = 0;

    int32_t helper;

    ColumnWriter(const std::shared_ptr<arrow::ChunkedArray> &chunkedArray)
        : chunks(chunkedArray->chunks())
    {}

    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator) = 0;

    void consumeField(CsvGenerator &generator)
    {
        const auto &chunk = chunks[currentChunk];
        if(!chunk->IsNull(usedFromChunk))
        {
            consumeFromChunk(*chunk, generator);
        }
        
        if(++usedFromChunk >= chunk->length())
            currentChunk++;
    }
};

// Note: type below is meant to be specialzied for supported types
template<arrow::Type::type id>
struct ColumnWriterFor 
{
};

template<>
struct ColumnWriterFor<arrow::Type::STRING> : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        auto ptr = static_cast<const arrow::StringArray&>(chunk).GetValue(usedFromChunk, &helper);
        generator.writeField(reinterpret_cast<const char*>(ptr), helper);

    }
};

template<>
struct ColumnWriterFor<arrow::Type::INT64> : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        auto value = static_cast<const arrow::Int64Array&>(chunk).Value(usedFromChunk);
        auto n = std::snprintf(buffer, std::size(buffer), "%" PRId64, value);
        generator.writeField(buffer, n);

    }
};

template<>
struct ColumnWriterFor<arrow::Type::DOUBLE> : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        auto value = static_cast<const arrow::DoubleArray&>(chunk).Value(usedFromChunk);
        auto n = std::snprintf(buffer, std::size(buffer), "%lf", value);
        generator.writeField(buffer, n);

    }
};

void generateCsv(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, char fieldSeparator /*= ','*/, char recordSeparator /*= '\n'*/, char quote /*= '"'*/)
{
    CsvGenerator generator{out, quotingPolicy, fieldSeparator, recordSeparator, quote};

    std::vector<std::unique_ptr<ColumnWriter>> writers;
    writers.reserve(table.num_columns());

    if(headerPolicy == GeneratorHeaderPolicy::GenerateHeaderLine)
    {
        for(int column = 0; column < table.num_columns(); column++)
        {
            if(column)
                out << fieldSeparator;

            const auto c = table.column(column);
            const auto &name = c->name();
            generator.writeField(name.data(), (int)name.size());
        }

        out << recordSeparator;
    }

    for(int column = 0; column < table.num_columns(); column++)
    {
        const auto c = table.column(column);
        visitType(*c->type(), [&] (auto id)
        {
            writers.push_back(std::make_unique<ColumnWriterFor<id.value>>(c->data()));
        });
    }

    // write records
    for(int row = 0; row < table.num_rows(); row++)
    {
        if(row)
            out << recordSeparator;

        for(int column = 0; column < table.num_columns(); column++)
        {
            if(column)
                out << fieldSeparator;

            writers[column]->consumeField(generator);
        }
    }
}

std::string_view CsvParser::parseField()
{
    if(bufferIterator == bufferEnd)
        std::string_view{bufferIterator, 0};

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
        return std::string_view(start, std::distance(start, bufferIterator));
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
                return std::string_view(start, length - rewriteOffset);
            }
        }

        if(rewriteOffset)
            *(bufferIterator - rewriteOffset) = c;
    }

    throw std::runtime_error("reached the end of the file with an unmatched quote character");
}

std::vector<std::string_view> CsvParser::parseRecord()
{
    std::vector<std::string_view> ret;
    ret.reserve(lastColumnCount);
    while(true)
    {
        char initialCharacter = *bufferIterator;

        ret.push_back(parseField());

        if(bufferIterator >= bufferEnd)
            return ret;

        if(*bufferIterator++ == recordSeparator)
            return ret;
    }
}

std::vector<std::vector<std::string_view>> CsvParser::parseCsvTable()
{
    std::vector<std::vector<std::string_view>> ret;

    for( ; bufferIterator < bufferEnd; )
    {
        auto parsedRecord = parseRecord();
        lastColumnCount = parsedRecord.size();
        ret.push_back(std::move(parsedRecord));
    }

    return ret;
}
