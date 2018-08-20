#include "ArrowUtilities.h"


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

BitmaskGenerator::BitmaskGenerator(int64_t length, bool initialValue) : length(length)
{
    auto bytes = arrow::BitUtil::BytesForBits(length);
    buffer = allocateBuffer<uint8_t>(bytes);
    data = buffer->mutable_data();
    std::memset(data, initialValue ? 0xFF : 0, bytes);
    // TODO: above sets by bytes, the last byte should have only part of bits set
}

void BitmaskGenerator::set(int64_t index)
{
    arrow::BitUtil::SetBit(data, index);
}

void BitmaskGenerator::clear(int64_t index)
{
    arrow::BitUtil::ClearBit(data, index);
}
