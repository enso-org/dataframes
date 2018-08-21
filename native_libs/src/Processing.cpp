#include "Processing.h"

#include <bitset>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <arrow/table.h>
#include <arrow/type_traits.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include "Core/ArrowUtilities.h"
#include "LQuery/AST.h"
#include "LQuery/Interpreter.h"

using namespace std::literals;

template<arrow::Type::type id_>
struct FilteredArrayBuilder
{
    static constexpr arrow::Type::type id = id_;

    using T = typename TypeDescription<id>::StorageValueType;
    using Array = typename TypeDescription<id>::Array;

    int64_t currentOffset = 0; // current position in value buffer for var-sized elements (currently only strings)
    int64_t addedCount = 0; // needed to know where to write next elements
    int64_t processedCount = 0; // needed to know which bit to read from mask
    int64_t length = 0;
    const unsigned char * const mask{};

    uint8_t *nullData{};
    T *valueData{};
    int32_t *offsetsData{};

    std::shared_ptr<arrow::Buffer> bitmask;
    std::shared_ptr<arrow::Buffer> offsets;
    std::shared_ptr<arrow::Buffer> values;

    FilteredArrayBuilder(const unsigned char * const mask, int64_t length, const arrow::ChunkedArray &array)
        : length(length), mask(mask)
    {
        if(length == 0)
            return;

        const auto maskByteCount = arrow::BitUtil::BytesForBits(length);
        std::tie(bitmask, nullData) = allocateBuffer<unsigned char>(maskByteCount);
        std::memset(nullData, 0xFF, maskByteCount);
        const auto lastByte = 0xFF >> (8 - length % 8);
        if(lastByte) // exception would be 0, if length is multiple of 8 - then we don't touch the mask
            *(nullData + maskByteCount - 1) = lastByte;

        if constexpr(id == arrow::Type::STRING)
        {
            std::tie(offsets, offsetsData) = allocateBuffer<int32_t>(length + 1);
            offsetsData[0] = 0;

            int64_t totalStringLength = 0;
            for(auto &chunk : array.chunks())
            {
                const auto &array = dynamic_cast<const arrow::StringArray&>(*chunk);
                const auto dataLength = array.value_offset(array.length()) - array.value_offset(0);
                totalStringLength += dataLength;
            }

            std::tie(values, valueData) = allocateBuffer<uint8_t>(totalStringLength);
        }
        else
        {
            std::tie(values, valueData) = allocateBuffer<T>(length);
        }
    }

//     template<unsigned char maskCode>
//     static constexpr unsigned char onesBy(int position)
//     {
//         int ret = 0;
//         for(int i = 0; i < position; i++)
//             if(maskCode & (1 << i))
//                 ret++;
//         return ret;
//     }

    template<bool nullable>
    FORCE_INLINE void addElem(const Array &array, const T *arrayValues, int arrayIndex)
    {
        if(!nullable || array.IsValid(arrayIndex))
        {
            valueData[addedCount++] = arrayValues[arrayIndex];
        }
        else
        {
            arrow::BitUtil::ClearBit(nullData, addedCount++);
        }
    };

    template<unsigned char maskCode, bool nullable>
    void addStatic8(const Array &array, const T *arrayValues, int arrayIndex)
    {
        // Note: it will be fast even if we call dynamic variant - compiler can easily propagate const
//          for(int bit = 0; bit < 8; ++bit)
//              addDynamic1<nullable>(maskCode, array, arrayValues, arrayIndex + bit, bit);
        if constexpr((maskCode & (1 << 0)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 0);
        if constexpr((maskCode & (1 << 1)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 1);
        if constexpr((maskCode & (1 << 2)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 2);
        if constexpr((maskCode & (1 << 3)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 3);
        if constexpr((maskCode & (1 << 4)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 4);
        if constexpr((maskCode & (1 << 5)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 5);
        if constexpr((maskCode & (1 << 6)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 6);
        if constexpr((maskCode & (1 << 7)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 7);
    }

    template<bool nullable>
    FORCE_INLINE void addDynamic1(unsigned char maskCode, const Array &array, const T *arrayValues, int arrayIndex, int bitIndex)
    {
        if((maskCode & (1 << bitIndex)) != 0) 
            addElem<nullable>(array, arrayValues, arrayIndex);
    }

    template<bool nullable>
    void addDynamic8(unsigned char maskCode, const Array &array, const T *arrayValues, int arrayIndex)
    {
        for(int bit = 0; bit < 8; ++bit)
            addDynamic1<nullable>(maskCode, array, arrayValues, arrayIndex + bit, bit);
    }


    template<bool nullable>
    void addInternal(const arrow::Array &array_)
    {
        if(array_.length() == 0)
            return;

        const auto &array = dynamic_cast<const Array&>(array_);
        const auto N = array.length();

        if constexpr(id == arrow::Type::STRING)
        {
            const auto sourceOffsets = array.raw_value_offsets();

            for(int i = 0; i < N; i++)
            {
                if(arrow::BitUtil::GetBit(mask, processedCount))
                {
                    if(array.IsValid(i))
                    {
                        int32_t elemLength;
                        auto elemPtr = array.GetValue(i, &elemLength);
                        std::memcpy(valueData + currentOffset, elemPtr, elemLength);
                        currentOffset += elemLength;
                    }
                    else
                    {
                        arrow::BitUtil::ClearBit(nullData, addedCount);
                    }
                    offsetsData[++addedCount] = currentOffset;
                }
                ++processedCount;
            }
        }
        else
        {
            const auto arrayValues = array.raw_values();
            const auto initiallyProcessed = processedCount;
            const auto sourceIndex = [&] { return processedCount - initiallyProcessed; };

            // Generally we process mask byte-by-byte. That means consuming elements in batches of 8.
            // However an array (chunk) is not necessarily aligned to start or end at the multiple of 8.
            // So we need to take care for leading and trailing elements.
            const auto initialBitIndex = initiallyProcessed % 8;
            const auto leadingElementCount = std::min(N, (8 - initialBitIndex)) % 8;
            const auto fullBytesOfMask = (N - leadingElementCount) / 8;
            const auto fullByteEncodedElementCount = fullBytesOfMask * 8;
            const auto trailingElementCount = N - leadingElementCount - fullByteEncodedElementCount;
            
            // Start processing with leading elements
            {
                const auto leadingMaskCode = mask[processedCount / 8];
                const auto initialBitIndex = initiallyProcessed % 8;
                const auto endBitIndex = initialBitIndex + leadingElementCount;
                for(int bit = initialBitIndex; bit < endBitIndex; ++bit)
                {
                    addDynamic1<nullable>(leadingMaskCode, array, arrayValues, sourceIndex(), bit);
                    ++processedCount;
                }
            }


            // Consume elements encoded on full mask bytes
            if(fullByteEncodedElementCount)
            {
                assert(processedCount % 8 == 0);
                assert(fullByteEncodedElementCount % 8 == 0);
            }

            auto alignedMask = mask + processedCount/8;
            for(auto i = sourceIndex(); i < fullByteEncodedElementCount; i += 8)
            {
                const auto maskCode = *alignedMask++;

                // We force generating code for each possible mask byte value.
                switch(maskCode)
                {
#define CASE(a, code, offset) \
                    case (code+offset): addStatic8<(code+offset), nullable>(array, arrayValues, i); break;
                    BOOST_PP_REPEAT_FROM_TO(0, 128, CASE, 0)
                    BOOST_PP_REPEAT_FROM_TO(0, 128, CASE, 128)
                    // Note: ugly MSVC workaround: repeating doesn't work with higher index ranges (like 256)
                    // so we keep indices 0-128 and add 0/128 as offset.
#undef CASE
                }
            }
            processedCount += fullByteEncodedElementCount;

            // consume trailing elements after last full byte
            {
                const auto trailingMaskCode = mask[processedCount / 8];
                for(int bit = 0; bit < trailingElementCount; ++bit)
                {
                    addDynamic1<nullable>(trailingMaskCode, array, arrayValues, sourceIndex(), bit);
                    ++processedCount;
                }
            }
        }
    }

    void addInternal(const arrow::Array &array_)
    {
         if(array_.null_count())
             addInternal<true>(array_);
         else
             addInternal<false>(array_);
    }
    void addInternal(const arrow::ChunkedArray &chunkedArray)
    {
        for(auto &&chunk : chunkedArray.chunks())
            addInternal(*chunk);
    }
    void addInternal(const arrow::Column &column)
    {
        addInternal(*column.data());
    }

    std::shared_ptr<arrow::Array> finish()
    {
        if constexpr(id == arrow::Type::STRING)
            return std::make_shared<Array>(length, offsets, values, bitmask, arrow::kUnknownNullCount);
        else
            return std::make_shared<Array>(length, values, bitmask, arrow::kUnknownNullCount);
    }

    static std::shared_ptr<arrow::Column> makeFiltered(const unsigned char * const mask, int64_t length, const arrow::Column &column)
    {
        FilteredArrayBuilder fab{mask, length, *column.data()};
        fab.addInternal(column);
        auto retArr = fab.finish();
        return std::make_shared<arrow::Column>(column.field(), retArr);
    }
};

std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table, const std::vector<int> &columnIndices)
{
    BitmaskGenerator bitmask{table->num_rows(), true};
    for(auto columnIndex : columnIndices)
    {
        auto column = table->column(columnIndex);
        if(column->null_count() == 0)
            continue;

        int64_t row = 0;
        iterateOverGeneric(*column,
            [&] (auto&&) { row++;                },
            [&] ()       { bitmask.clear(row++); });
    }

    return filter(table, *bitmask.buffer);
}

std::shared_ptr<arrow::Table> dropNA(std::shared_ptr<arrow::Table> table)
{
    std::vector<int> columnIndices;
    for(int i = 0; i < table->num_columns(); i++)
        columnIndices.push_back(i);

    return dropNA(table, columnIndices);
}

template<typename Array>
std::shared_ptr<arrow::Array> fillNATyped(const Array &array, DynamicField value)
{
    if constexpr(std::is_same_v<Array, arrow::StringArray>)
    {
        std::string_view valueToFill = std::visit(overloaded{
            [] (const std::string_view &sv) { return sv; },
            [] (const std::string &s) { return std::string_view(s); },
            [] (const auto &v) -> std::string_view { throw std::runtime_error("cannot fill string array with value of type "s + typeid(v).name()); }
            }, value);
        
        arrow::StringBuilder builder;
        iterateOver<arrow::Type::STRING>(array, 
            [&] (auto &&s) { builder.Append(s.data(), s.size()); },
            [&] () { builder.Append(valueToFill.data(), valueToFill.size()); });
        return finish(builder);

    }
    else
    {
        using T = typename Array::value_type;
        auto valueToFill = std::get<T>(value);

        auto [buffer, data] = allocateBuffer<T>(array.length());
        int row = 0;
        std::memcpy(data, array.raw_values(), buffer->size());
        iterateOver<ArrayTypeDescription<Array>::id>(array, 
            [&] (auto &&) { ++row; },
            [&] () { data[row++] = valueToFill; });

        return std::make_shared<Array>(array.length(), buffer, nullptr);
    }
}

std::shared_ptr<arrow::Array> fillNA(std::shared_ptr<arrow::Array> array, DynamicField value)
{
    if(array->null_count() == 0)
        return array;

    return visitArray(*array, [&] (auto *array) 
    {
        return std::visit([&] (auto value)
        {
            return fillNATyped(*array, value);
        }, value);
    });
}

std::shared_ptr<arrow::ChunkedArray> fillNA(std::shared_ptr<arrow::ChunkedArray> array, DynamicField value)
{
    if(array->null_count() == 0)
        return array;

    arrow::ArrayVector newArrays;
    for(auto &&chunk : array->chunks())
        newArrays.push_back(fillNA(chunk, value));

    return std::make_shared<arrow::ChunkedArray>(newArrays);
}

std::shared_ptr<arrow::Column> fillNA(std::shared_ptr<arrow::Column> column, DynamicField value)
{
    if(column->null_count() == 0)
        return column;

    auto newField = setNullable(false, column->field());
    auto newChunks = fillNA(column->data(), value);
    return std::make_shared<arrow::Column>(newField, newChunks);
}

std::shared_ptr<arrow::Table> fillNA(std::shared_ptr<arrow::Table> table, const std::unordered_map<std::string, DynamicField> &valuesPerColumn)
{
    const auto nonNullable = [](auto &&column) { return column->field()->nullable() == false; };
    const auto oldColumns = getColumns(*table);
    if(std::all_of(oldColumns.begin(), oldColumns.end(), nonNullable))
        return table;

    auto newColumns = transformToVector(oldColumns, [&] (auto &&column)
    {
        if(auto itr = valuesPerColumn.find(column->name()); itr != valuesPerColumn.end())
        {
            const auto value = itr->second;
            return fillNA(column, value);
        }
        else
            return column;
    });
    
    auto newFields = transformToVector(newColumns, 
        [&](auto &&column) { return column->field(); });
    auto newSchema = arrow::schema(newFields, table->schema()->metadata());
    return arrow::Table::Make(newSchema, newColumns);
}

std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, predicate] = ast::parsePredicate(*table, dslJsonText);
    const auto maskBuffer = execute(*table, predicate, mapping);
    return filter(table, *maskBuffer);
}

std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const arrow::Buffer &maskBuffer)
{
    const unsigned char * const maskData = maskBuffer.data();
    const auto oldRowCount = table->num_rows();

    int newRowCount = 0;
    for(int i = 0; i < oldRowCount; i++)
        newRowCount += arrow::BitUtil::GetBit(maskData, i);

    std::vector<std::shared_ptr<arrow::Column>> newColumns;

    for(int columnIndex = 0; columnIndex < table->num_columns(); columnIndex++)
    {
        const auto column = table->column(columnIndex);
        const auto filteredColumn = [&] () -> std::shared_ptr<arrow::Column>
        {
            const auto t = column->type();
            switch(t->id())
            {
            case arrow::Type::INT64 : return FilteredArrayBuilder<arrow::Type::INT64 >::makeFiltered(maskData, newRowCount, *column);
            case arrow::Type::DOUBLE: return FilteredArrayBuilder<arrow::Type::DOUBLE>::makeFiltered(maskData, newRowCount, *column);
            case arrow::Type::STRING: return FilteredArrayBuilder<arrow::Type::STRING>::makeFiltered(maskData, newRowCount, *column);
            default                 : throw  std::runtime_error(__FUNCTION__ + std::string(": not supported array type ") + t->ToString());
            }
        }();
        newColumns.push_back(filteredColumn);
    }

    return arrow::Table::Make(table->schema(), newColumns);
}

std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, v] = ast::parseValue(*table, dslJsonText);
    return execute(*table, v, mapping);
}

struct ToString
{
    std::string operator() (int64_t value)          const { return std::to_string(value); }
    std::string operator() (double value)           const { return std::to_string(value); }
    std::string operator() (std::string value)      const { return std::move(value); }
    std::string operator() (std::string_view value) const { return std::string(value); }
    template<typename T>
    std::string operator() (T)                      const { throw std::runtime_error(__FUNCTION__ + ": invalid conversion"s); }
};
struct ToInt64
{
    int64_t operator() (int64_t value)            const { return value; }
    int64_t operator() (double value)             const { return value; }
    int64_t operator() (const std::string &value) const { return std::stoll(value); }
    int64_t operator() (std::string_view value)   const { return std::stoll(std::string(value)); }
    template<typename T>
    int64_t operator() (T)                        const { throw std::runtime_error(__FUNCTION__ + ": invalid conversion"s); }
};
struct ToDouble
{
    double operator() (int64_t value)            const { return value; }
    double operator() (double value)             const { return value; }
    double operator() (const std::string &value) const { return std::stod(value); }
    double operator() (std::string_view value)   const { return std::stod(std::string(value)); }
    template<typename T>
    double operator() (T)                        const { throw std::runtime_error(__FUNCTION__ + ": invalid conversion"s); }
};

DynamicField adjustTypeForFilling(DynamicField valueGivenByUser, const arrow::DataType &type)
{
    switch(type.id())
    {
    case arrow::Type::STRING: return std::visit(ToString{}, valueGivenByUser);
    case arrow::Type::INT64: return std::visit(ToInt64{}, valueGivenByUser);
    case arrow::Type::DOUBLE: return std::visit(ToDouble{}, valueGivenByUser);
    default: throw std::runtime_error("Not supported column type to fill: "s + type.ToString());
    }
}
