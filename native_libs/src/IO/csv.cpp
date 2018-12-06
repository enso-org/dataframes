#include "csv.h"
#include "IO.h"
#include "Core/ArrowUtilities.h"
#include "Core/Logger.h"
#include "Core/Utils.h"


#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <arrow/table.h>
#include <arrow/builder.h>

using namespace std::literals;



// to allow conditional static_asserts 
template<arrow::Type::type id> struct always_false2 : std::false_type {};
template<arrow::Type::type id> constexpr bool always_false2_v = always_false2<id>::value;

arrow::Type::type deduceType(std::string_view text)
{
    if(text.empty())
        return arrow::Type::NA;
    if(Parser::as<Timestamp>(text))
        return arrow::Type::TIMESTAMP;
    if(Parser::as<int64_t>(text))
        return arrow::Type::INT64;
    if(Parser::as<double>(text))
        return arrow::Type::DOUBLE;
    return arrow::Type::STRING;
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

template<arrow::Type::type id>
struct ColumnBuilder
{
    using ArrowType = typename TypeDescription<id>::ArrowType;

    MissingField missingField;
    std::shared_ptr<BuilderFor<id>> builder;

    ColumnBuilder(MissingField missingField, const std::shared_ptr<ArrowType> &type)
        : missingField(missingField) 
        , builder(makeBuilder(type))
    {}

    NO_INLINE void addFromString(const std::string_view &field)
    {
        if constexpr(id == arrow::Type::STRING)
        {
            if(field.size())
                checkStatus(builder->Append(field.data(), (int32_t)field.size()));
            else
               addMissing(); 
        }
        else
        {
            if(field.size() != 0)
            {
                // some number parsers require strings to be null terminated
                // we can safely overwrite data with null, as it can be only junk or parsed separatror
                if constexpr(Parser::requiresNull)
                {
                    const auto fieldEnd = const_cast<char *>(field.data() + field.size());
                    *fieldEnd = '\0';
                }

                if constexpr(id == arrow::Type::INT64)
                {
                    if(auto v = Parser::as<int64_t>(field))
                    {
                        checkStatus(builder->Append(*v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else if constexpr(id == arrow::Type::DOUBLE)
                {
                    if(auto v = Parser::as<double>(field))
                    {
                        checkStatus(builder->Append(*v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else if constexpr(id == arrow::Type::TIMESTAMP)
                {
                    if(auto v = Parser::as<Timestamp>(field))
                    {
                        checkStatus(append(*builder, *v));
                    }
                    else
                    {
                        addMissing();
                    }
                }
                else
                    static_assert(always_false2_v<id>, "wrong type");
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
            checkStatus(builder->AppendNull());
        else
            checkStatus(builder->Append(defaultValue<id>()));
    }
    void reserve(int64_t count)
    {
        checkStatus(builder->Reserve(count));
    }
};

ColumnType deduceType(const ParsedCsv &csv, size_t columnIndex, size_t startRow, size_t lookupDepth)
{
    lookupDepth = std::min(lookupDepth, csv.records.size());

    std::unordered_set<arrow::Type::type> encounteredTypes;
    for(size_t i = startRow; i < lookupDepth; i++)
    {
        const auto &record = csv.records.at(i);
        if(columnIndex < record.size())
        {
            const auto field = record.at(columnIndex);
            encounteredTypes.insert(deduceType(field));
        }
    }

    auto typePtr = [&] () -> TypePtr
    {
        if(encounteredTypes.count(arrow::Type::TIMESTAMP))
        {
            // string if there are timestamps and types other than timestamps (excluding nulls)
            if(encounteredTypes.size() > 1 + encounteredTypes.count(arrow::Type::NA))
                return arrow::TypeTraits<arrow::StringType>::type_singleton();
            return getTypeSingleton<arrow::Type::TIMESTAMP>();
        }
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

    const bool takeFirstRowAsNames = holds_alternative<TakeFirstRowAsHeaders>(header);
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
            arrays.push_back(finish(*builder.builder));
        };

        visitDataType(typeInfo.type, [&] (auto type)
        {
            constexpr auto id = idFromDataPointer<decltype(type)>;
            if constexpr(id != arrow::Type::LIST)
                processColumn(ColumnBuilder<id>{missingFieldsPolicy, type});
            else
                throw std::runtime_error("not supported: list embedded within CSV field");
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
    virtual ~ColumnWriter() {}

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

template<>
struct ColumnWriterFor<arrow::Type::TIMESTAMP> : ColumnWriter
{
    using ColumnWriter::ColumnWriter;
    virtual void consumeFromChunk(const arrow::Array &chunk, CsvGenerator &generator)
    {
        // TODO support other timestamp units 
        assert(std::dynamic_pointer_cast<arrow::TimestampType>(chunk.type())->unit() == arrow::TimeUnit::NANO);
        auto ticksCount = static_cast<const arrow::TimestampArray&>(chunk).Value(usedFromChunk);
        TimestampDuration nanosecondTicks(ticksCount);
        Timestamp timestamp(nanosecondTicks);
        auto timet = timestamp.toTimeT();
        auto tm = std::gmtime(&timet);
        auto n = std::strftime(buffer, std::size(buffer), "%F", tm);
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
            if(recordSeparator == '\n' && c == '\r') // treat CRLF as if LF
            {
                auto nextItr = bufferIterator+1;
                if(nextItr != bufferEnd  &&  *nextItr == '\n')
                    break;
            }
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

        const auto next = *bufferIterator++;
        if(next == recordSeparator)
            return ret;
        if(next == '\r' && recordSeparator == '\n')
        {
            if(bufferIterator >= bufferEnd)
                return ret;

            const auto next = *bufferIterator++;
            if(next == '\n')
                return ret;
        }
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

std::shared_ptr<arrow::Table> FormatCSV::readString(std::string data, const CsvReadOptions &options) const
{
    auto csv = parseCsvData(std::move(data), options.fieldSeparator, options.recordSeparator, options.quote);
    return csvToArrowTable(csv, options.header, options.columnTypes, options.typeDeductionDepth);
}

std::string FormatCSV::writeToString(const arrow::Table &table, const CsvWriteOptions &options) const
{
    std::ostringstream out;
    generateCsv(out, table, options.headerPolicy, options.quotingPolicy, options.fieldSeparator, options.recordSeparator, options.quote);
    return out.str();
}

std::string FormatCSV::fileSignature() const
{
    // return empty string
    return {};
}

std::shared_ptr<arrow::Table> FormatCSV::read(std::string_view filePath, const CsvReadOptions &options) const
{
    auto buffer = getFileContents(filePath);
    return readString(std::move(buffer), options);
}

void FormatCSV::write(std::string_view filePath, const arrow::Table &table, const CsvWriteOptions &options) const
{
    auto out = openFileToWrite(filePath);
    generateCsv(out, table, options.headerPolicy, options.quotingPolicy, options.fieldSeparator, options.recordSeparator, options.quote);
}

std::vector<std::string> FormatCSV::fileExtensions() const
{
    return { "csv", "txt" };
}
