#include "Processing.h"

#include <bitset>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <arrow/table.h>
#include <arrow/type_traits.h>

#define RAPIDJSON_NOMEMBERITERATORCLASS 
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

    int64_t currentOffset = 0;
    int64_t addedCount = 0;
    int64_t processedCount = 0;
    int64_t length;
    const unsigned char * const mask{};

    uint8_t *nullData{};
    T *valueData{};
    int32_t *offsetsData{};

    std::shared_ptr<arrow::Buffer> bitmask;
    std::shared_ptr<arrow::Buffer> offsets;
    std::shared_ptr<arrow::Buffer> values;

    template<typename Elem>
    auto allocateBuffer(int64_t length)
    {
        auto buffer = ::allocateBuffer<Elem>(length);
        auto data = reinterpret_cast<Elem*>(buffer->mutable_data());
        return std::make_pair(buffer, data);
    }

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
    void addElem(const Array &array, const T *arrayValues, int arrayIndex)
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
         for(int bit = 0; bit < 8; ++bit)
             addDynamic1<nullable>(maskCode, array, arrayValues, arrayIndex, bit);
//         if constexpr((maskCode & (1 << 0)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 0);
//         if constexpr((maskCode & (1 << 1)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 1);
//         if constexpr((maskCode & (1 << 2)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 2);
//         if constexpr((maskCode & (1 << 3)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 3);
//         if constexpr((maskCode & (1 << 4)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 4);
//         if constexpr((maskCode & (1 << 5)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 5);
//         if constexpr((maskCode & (1 << 6)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 6);
//         if constexpr((maskCode & (1 << 7)) != 0) addElem<nullable>(array, arrayValues, arrayIndex + 7);
    }

    template<bool nullable>
    void addDynamic1(unsigned char maskCode, const Array &array, const T *arrayValues, int arrayIndexBase, int bitPosition)
    {
        if((maskCode & (1 << bitPosition)) != 0) 
            addElem<nullable>(array, arrayValues, arrayIndexBase + bitPosition);
    }

    template<bool nullable>
    void addDynamic8(unsigned char maskCode, const Array &array, const T *arrayValues, int arrayIndex)
    {
        for(int bit = 0; bit < 8; ++bit)
            addDynamic1<nullable>(maskCode, array, arrayValues, arrayIndex, bit);
    }


    template<bool nullable>
    void addInternal(const arrow::Array &array_)
    {
        if(array_.length() == 0)
            return;

        // TODO fix the chunks
        if(processedCount % 8)
            throw std::runtime_error("not implemented: chunked array with elem count not being multple of 8");


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
                        arrow::BitUtil::ClearBit(nullData, addedCount++);
                    }
                    offsetsData[++addedCount] = currentOffset;
                }
                ++processedCount;
            }
        }
        else
        {
            const auto arrayValues = array.raw_values();

            static_assert(sizeof(T) <= 8);

            auto maskAdjustedToI = mask + (processedCount / 8);

            const auto remainderCount = N % 8;
            const auto bigIterationsCount = remainderCount ? (N - 8) : N;
            int i = 0;

            // This loop consumes single mask byte, i. e. 8 source array entries.
            for(; i < bigIterationsCount; i += 8)
            {
                const auto maskCode = maskAdjustedToI[i / 8];

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

            // If source array length is not multiple of 8, we consume remainder.
            const auto maskCode = maskAdjustedToI[i / 8];
            for(int bit = 0; bit < remainderCount; bit++)
                addDynamic1<nullable>(maskCode, array, arrayValues, i, bit);

            processedCount += array.length();
        }
    }

    void addInternal(const arrow::Array &array_)
    {
         if(array_.null_count())
             addInternal<true>(array_);
         else
             addInternal<false>(array_);

//         const auto &array = dynamic_cast<const Array&>(array_);
//         const auto N = (int32_t)array.length();
//         const auto arrayValues = array.raw_values();
// 
//         for(int i = 0; i < N; ++i)
//         {
//             if(mask[processedCount++])
//             {
//                 if(array.IsValid(i))
//                 {
//                     valueData[addedCount++] = arrayValues[i];
//                 }
//                 else
//                 {
//                     arrow::BitUtil::ClearBit(nullData, addedCount++);
//                 };
//             }
//         }

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

std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, predicate] = ast::parsePredicate(*table, dslJsonText);
    const auto mask = execute(*table, predicate, mapping);
    const unsigned char * const maskBuffer = mask->data();

    const auto oldRowCount = table->num_rows();
    int newRowCount = 0;
    for(int i = 0; i < oldRowCount; i++)
        newRowCount += arrow::BitUtil::GetBit(maskBuffer, i);
    //const int64_t newRowCount = std::accumulate(maskBuffer, maskBuffer + oldRowCount, std::int64_t{});
    
    std::vector<std::shared_ptr<arrow::Column>> newColumns;


//     auto bufferN = arrow::BitUtil::BytesForBits(oldRowCount);
//     auto buffer = allocateBuffer<uint8_t>(bufferN);
//     auto bd = buffer->mutable_data();
//     std::memset(bd, 0, bufferN);
//     for(int i = 0; i < oldRowCount; i++)
//     {
//         if(maskBuffer[i])
//             arrow::BitUtil::SetBit(buffer->mutable_data(), i);
//     }

    for(int columnIndex = 0; columnIndex < table->num_columns(); columnIndex++)
    {
        const auto column = table->column(columnIndex);
        const auto filteredColumn = [&] () -> std::shared_ptr<arrow::Column>
        {
            const auto t = column->type();
            switch(t->id())
            {
            case arrow::Type::INT64 : return FilteredArrayBuilder<arrow::Type::INT64>::makeFiltered(maskBuffer, newRowCount, *column);
            case arrow::Type::DOUBLE: return FilteredArrayBuilder<arrow::Type::DOUBLE>::makeFiltered(maskBuffer, newRowCount, *column);
            case arrow::Type::STRING: return FilteredArrayBuilder<arrow::Type::STRING>::makeFiltered(maskBuffer, newRowCount, *column);
            default                 : throw  std::runtime_error(__FUNCTION__ + std::string(": not supported array type ") + t->ToString());
            }
        }();
        newColumns.push_back(filteredColumn);

// 
//         // TODO handle zero chunks?
//         // TODO handle more chunks
//         const auto chunk = column->data()->chunk(0);
// 
// 
//         visitArray(chunk.get(), [&](auto *array) 
//         {
//             using TD = ArrayTypeDescription<std::remove_pointer_t<decltype(array)>>;
//             using T = typename TD::ValueType;
// 
//             if constexpr(std::is_scalar_v<T>)
//             {
// //                   measure("inner", 100000, [&]
// //                   {
//                     constexpr auto idd = std::is_same_v<T, int64_t> ? arrow::Type::INT64 : arrow::Type::DOUBLE;
//                     FilteredArrayBuilder<idd> fab{newRowCount, mask->data()};
//                     if(newRowCount)
//                         fab.add(*chunk);
//                     newColumns.push_back(fab.finish());
// //                  });
//             }
//             else
//             {
//                 const auto stringSource = static_cast<const arrow::StringArray *>(array);
//                 arrow::StringBuilder builder;
// 
//                 int32_t lengthBuffer;
// 
//                 for(int64_t sourceItr = 0; sourceItr < oldRowCount; sourceItr++)
//                 {
//                     if(arrow::BitUtil::GetBit(maskBuffer, sourceItr))
//                     {
//                         auto ptr = stringSource->GetValue(sourceItr, &lengthBuffer);
//                         if(array->IsNull(sourceItr))
//                             builder.AppendNull();
//                         else
//                             builder.Append(ptr, lengthBuffer);
//                     }
//                 }
// 
//                 newColumns.push_back(finish(builder));
//             }
//         });
// 
// 
     }

    return arrow::Table::Make(table->schema(), newColumns);
}

std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, v] = ast::parseValue(*table, dslJsonText);
    return execute(*table, v, mapping);
}