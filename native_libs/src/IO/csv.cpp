#include "csv.h"
#include "IO.h"
#include "Core/Logger.h"
#include "Core/ArrowUtilities.h"


#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include <arrow/table.h>
#include <arrow/builder.h>

using namespace std::literals;

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

    void addFromString(const NaiveStringView &field)
    {
        if constexpr(type == arrow::Type::STRING)
        {
            checkStatus(builder.Append(field.text, field.length));
        }
        else
        {
            if(!field.length)
            {
                addMissing();
            }
            else
            {
                const auto fieldEnd = field.text + field.length;
                *fieldEnd = '\0';
                char *next = nullptr;
                if constexpr(type == arrow::Type::INT64)
                {
                    auto v = std::strtoll(field.text, &next, 10);
                    if(next == fieldEnd)
                    {
                        checkStatus(builder.Append(v));
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
                        checkStatus(builder.Append(v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else
                    throw std::runtime_error("wrong type");
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

std::shared_ptr<arrow::Table> csvToArrowTable(const ParsedCsv &csv, HeaderPolicy header, std::vector<ColumnType> columnTypes)
{
    // empty table
    if(csv.recordCount == 0 || csv.fieldCount == 0)
    {
        auto schema = std::make_shared<arrow::Schema>(std::vector<std::shared_ptr<arrow::Field>>{});
        return arrow::Table::Make(schema, std::vector<std::shared_ptr<arrow::Array>>{});
    }

    // If there is no type info for column, default to non-nullable Text (it always works)
    if(columnTypes.size() < csv.fieldCount)
    {
        const ColumnType nonNullableText{ std::make_shared<arrow::StringType>(), false };
        columnTypes.resize(csv.fieldCount, nonNullableText);
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(csv.fieldCount);

    const bool takeFirstRowAsNames = std::holds_alternative<TakeFirstRowAsHeaders>(header);


    const int startRow = takeFirstRowAsNames ? 1 : 0;

    for(int column = 0; column < csv.fieldCount; column++)
    {
        const auto typeInfo = columnTypes.at(column);
        const auto missingFieldsPolicy = typeInfo.nullable ? MissingField::AsNull : MissingField::AsZeroValue;
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

        switch(typeInfo.type->id())
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
        default:
            throw std::runtime_error("type not supported " + typeInfo.type->ToString());
        }
    }

    const auto names = decideColumnNames(csv.fieldCount, header, [&] (int column)
    {
        const auto &headerRow = csv.records[0];
        if(column < (int)headerRow.size())
            return headerRow[column].str();
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

struct StringColumnWriter : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        auto ptr = static_cast<const arrow::StringArray&>(chunk).GetValue(usedFromChunk, &helper);
        generator.writeField(reinterpret_cast<const char*>(ptr), helper);

    }
};

struct Int64ColumnWriter : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        auto value = static_cast<const arrow::Int64Array&>(chunk).Value(usedFromChunk);
        auto n = std::snprintf(buffer, std::size(buffer), "%" PRId64, value);
        generator.writeField(buffer, n);

    }
};

struct DoubleColumnWriter : ColumnWriter
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
        switch(c->field()->type()->id())
        {
        case arrow::Type::INT64:
            writers.push_back(std::make_unique<Int64ColumnWriter>(c->data()));
            break;
        case arrow::Type::DOUBLE:
            writers.push_back(std::make_unique<DoubleColumnWriter>(c->data()));
            break;
        case arrow::Type::STRING:
            writers.push_back(std::make_unique<StringColumnWriter>(c->data()));
            break;
        default:
            throw std::runtime_error("type not supported " + c->field()->type()->ToString());
        }
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

NaiveStringView CsvParser::parseField()
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

std::vector<NaiveStringView> CsvParser::parseRecord()
{
    std::vector<NaiveStringView> ret;
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

std::vector<std::vector<NaiveStringView>> CsvParser::parseCsvTable()
{
    std::vector<std::vector<NaiveStringView>> ret;

    for( ; bufferIterator < bufferEnd; )
    {
        auto parsedRecord = parseRecord();
        lastColumnCount = parsedRecord.size();
        ret.push_back(std::move(parsedRecord));
    }

    return ret;
}
