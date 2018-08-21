#include "ArrowUtilities.h"

using namespace std::literals;

std::vector<std::shared_ptr<arrow::Column>> getColumns(const arrow::Table &table)
{
    std::vector<std::shared_ptr<arrow::Column>> columns;
    for(int i = 0; i < table.num_columns(); i++)
    {
        columns.push_back(table.column(i));
    }
    return columns;
}

std::unordered_map<std::string, std::shared_ptr<arrow::Column>> getColumnMap(const arrow::Table &table)
{
    std::unordered_map<std::string, std::shared_ptr<arrow::Column>> ret;

    for(auto &&column : getColumns(table))
        ret[column->name()] = column;

    return ret;
}

std::shared_ptr<arrow::Field> setNullable(bool nullable, std::shared_ptr<arrow::Field> field)
{
    if(field->nullable() == nullable)
        return field;

    return arrow::field(field->name(), field->type(), nullable, field->metadata());
}

std::shared_ptr<arrow::Schema> setNullable(bool nullable, std::shared_ptr<arrow::Schema> schema)
{
    std::vector<std::shared_ptr<arrow::Field>> newFields;
    for(auto &&field : schema->fields())
        newFields.push_back(setNullable(nullable, field));

    return arrow::schema(newFields, schema->metadata());
}

std::shared_ptr<arrow::Table> tableFromArrays(std::vector<PossiblyChunkedArray> arrays, std::vector<std::string> names, std::vector<bool> nullables)
{
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunkedArrays;
    for(auto &someArray: arrays)
    {
        auto chunk = std::visit(overloaded {
            [&] (std::shared_ptr<arrow::Array> array) { return std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{array}); },
            [&] (std::shared_ptr<arrow::ChunkedArray> array) { return array; }
            }, someArray);
        chunkedArrays.push_back(chunk);
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for(int i = 0; i < chunkedArrays.size(); i++)
    {
        const auto arr = chunkedArrays.at(i);
        if(names.size() <= i)
            names.push_back("col" + std::to_string(i));
        if(nullables.size() <= i)
            nullables.push_back(arr->null_count());

        fields.push_back(arrow::field(names.at(i), arr->type(), nullables.at(i)));
    }

    std::vector<std::shared_ptr<arrow::Column>> columns;
    for(int i = 0; i < chunkedArrays.size(); i++)
    {
        auto field = fields.at(i);
        auto chunks = chunkedArrays.at(i);
        auto column = std::make_shared<arrow::Column>(field, chunks);
        columns.push_back(column);
    }


    auto schema = arrow::schema(std::move(fields));
    return arrow::Table::Make(schema, std::move(columns));
}

DynamicField arrayAt(const arrow::Array &array, int64_t index)
{
    return visitArray(array, [&](auto *array) -> DynamicField
    {
        if(array->IsValid(index))
            return arrayValueAt(*array, index);
        else
            return std::nullopt;
    });
}

DynamicField arrayAt(const arrow::ChunkedArray &array, int64_t index)
{
    auto [chunk, indexInChunk] = locateChunk(array, index);
    return arrayAt(*chunk, indexInChunk);
}

DynamicField arrayAt(const arrow::Column &column, int64_t index)
{
    return arrayAt(*column.data(), index);
}

std::pair<std::shared_ptr<arrow::Array>, int64_t> locateChunk(const arrow::ChunkedArray &chunkedArray, int64_t index)
{
    validateIndex(chunkedArray, index);

    int64_t i = index;
    for(auto &chunk : chunkedArray.chunks())
    {
        if(i < chunk->length())
            return {chunk, i};

        i -= chunk->length();
    }

    throw std::runtime_error(__FUNCTION__ + ": unexpected failure"s);
}

std::vector<DynamicField> rowAt(const arrow::Table &table, int64_t index)
{
    std::vector<DynamicField> ret;
    for(int i = 0; i < table.num_columns(); i++)
    {
        auto column = table.column(i);
        ret.push_back(arrayAt(*column, index));
    }
    return ret;
}

void validateIndex(const arrow::Array &array, int64_t index)
{
    validateIndex(array.length(), index);
}

void validateIndex(const arrow::ChunkedArray &array, int64_t index)
{
    validateIndex(array.length(), index);
}

void validateIndex(const arrow::Column &column, int64_t index)
{
    validateIndex(column.length(), index);
}

BitmaskGenerator::BitmaskGenerator(int64_t length, bool initialValue) : length(length)
{
    auto bytes = arrow::BitUtil::BytesForBits(length);
    std::tie(buffer, data) = allocateBuffer<uint8_t>(bytes);
    std::memset(data, initialValue ? 0xFF : 0, bytes);
    // TODO: above sets by bytes, the last byte should have only part of bits set
}

bool BitmaskGenerator::get(int64_t index)
{
    return arrow::BitUtil::GetBit(data, index);
}

void BitmaskGenerator::set(int64_t index)
{
    arrow::BitUtil::SetBit(data, index);
}

void BitmaskGenerator::clear(int64_t index)
{
    arrow::BitUtil::ClearBit(data, index);
}
