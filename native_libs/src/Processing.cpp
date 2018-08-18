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

#include "Core/ArrowUtilities.h"
#include "LQuery/AST.h"
#include "LQuery/Interpreter.h"

using namespace std::literals;

template<arrow::Type::type id_>
struct FilteredArrayBuilder
{
    static constexpr arrow::Type::type id = id_;

    using T = typename TypeDescription<id>::ValueType;
    using Array = typename TypeDescription<id>::Array;

    int64_t addedCount = 0;
    int64_t processedCount = 0;
    int64_t length;
    const unsigned char * const mask{};

    uint8_t *nullData{};
    T * valueData{};

    std::shared_ptr<arrow::Buffer> bitmask;
    std::shared_ptr<arrow::Buffer> offsets;
    std::shared_ptr<arrow::Buffer> values;

    FilteredArrayBuilder(int64_t length, const unsigned char * const mask)
        : length(length), mask(mask)
    {
        if(length == 0)
            return;

        const auto maskByteCount = arrow::BitUtil::BytesForBits(length);
        bitmask = allocateBuffer<unsigned char>(maskByteCount);
        values = allocateBuffer<T>(length);

        nullData = bitmask->mutable_data();
        valueData = reinterpret_cast<T*>(values->mutable_data());

        std::memset(valueData, 0xFF, maskByteCount);
        const auto lastByte = 0xFF >> (length % 8);
        if(lastByte) // exception would be 0, if length is multiple of 8 - then we don't touch the mask
            *(nullData + maskByteCount - 1) = lastByte;
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
    void add(const Array &array, const T *arrayValues, int i)
    {
        if constexpr((maskCode & (1 << 0)) != 0) addElem<nullable>(array, arrayValues, i + 0);
        if constexpr((maskCode & (1 << 1)) != 0) addElem<nullable>(array, arrayValues, i + 1);
        if constexpr((maskCode & (1 << 2)) != 0) addElem<nullable>(array, arrayValues, i + 2);
        if constexpr((maskCode & (1 << 3)) != 0) addElem<nullable>(array, arrayValues, i + 3);
        if constexpr((maskCode & (1 << 4)) != 0) addElem<nullable>(array, arrayValues, i + 4);
        if constexpr((maskCode & (1 << 5)) != 0) addElem<nullable>(array, arrayValues, i + 5);
        if constexpr((maskCode & (1 << 6)) != 0) addElem<nullable>(array, arrayValues, i + 6);
        if constexpr((maskCode & (1 << 7)) != 0) addElem<nullable>(array, arrayValues, i + 7);
    }

    template<bool nullable>
    void addDynamic(unsigned char maskCode, const Array &array, const T *arrayValues, int i)
    {
        if((maskCode & (1 << 0)) != 0) addElem<nullable>(array, arrayValues, i + 0);
        if((maskCode & (1 << 1)) != 0) addElem<nullable>(array, arrayValues, i + 1);
        if((maskCode & (1 << 2)) != 0) addElem<nullable>(array, arrayValues, i + 2);
        if((maskCode & (1 << 3)) != 0) addElem<nullable>(array, arrayValues, i + 3);
        if((maskCode & (1 << 4)) != 0) addElem<nullable>(array, arrayValues, i + 4);
        if((maskCode & (1 << 5)) != 0) addElem<nullable>(array, arrayValues, i + 5);
        if((maskCode & (1 << 6)) != 0) addElem<nullable>(array, arrayValues, i + 6);
        if((maskCode & (1 << 7)) != 0) addElem<nullable>(array, arrayValues, i + 7);
    }


    template<bool nullable>
    void addInternal(const arrow::Array &array_)
    {
        if(array_.length() == 0)
            return;

        // Note: we unroll loop 8x and don't care about consuming a few entries too much
        // buffers are always padded to multiple of 64 bytes and stored value is at most 8 bytes
        // so even in worst case we won't crash
        // and additional elements will be ignored anyway, because we set array length
        static_assert(sizeof(T) <= 8);

        const auto &array = dynamic_cast<const Array&>(array_);
        const auto N = array.length();
        const auto arrayValues = array.raw_values();

        auto maskAdjustedToI = mask + (processedCount / 8);

        for(int i = 0; i < N; i += 8)
        {
            const auto maskCode = maskAdjustedToI[i / 8];
            //processedCount += 8;
            switch(maskCode)
            {
            case 0: add<0, nullable>(array, arrayValues, i); break;
            case 1: add<1, nullable>(array, arrayValues, i); break;
            case 2: add<2, nullable>(array, arrayValues, i); break;
            case 3: add<3, nullable>(array, arrayValues, i); break;
            case 4: add<4, nullable>(array, arrayValues, i); break;
            case 5: add<5, nullable>(array, arrayValues, i); break;
            case 6: add<6, nullable>(array, arrayValues, i); break;
            case 7: add<7, nullable>(array, arrayValues, i); break;
            case 8: add<8, nullable>(array, arrayValues, i); break;
            case 9: add<9, nullable>(array, arrayValues, i); break;
            case 10: add<10, nullable>(array, arrayValues, i); break;
            case 11: add<11, nullable>(array, arrayValues, i); break;
            case 12: add<12, nullable>(array, arrayValues, i); break;
            case 13: add<13, nullable>(array, arrayValues, i); break;
            case 14: add<14, nullable>(array, arrayValues, i); break;
            case 15: add<15, nullable>(array, arrayValues, i); break;
            case 16: add<16, nullable>(array, arrayValues, i); break;
            case 17: add<17, nullable>(array, arrayValues, i); break;
            case 18: add<18, nullable>(array, arrayValues, i); break;
            case 19: add<19, nullable>(array, arrayValues, i); break;
            case 20: add<20, nullable>(array, arrayValues, i); break;
            case 21: add<21, nullable>(array, arrayValues, i); break;
            case 22: add<22, nullable>(array, arrayValues, i); break;
            case 23: add<23, nullable>(array, arrayValues, i); break;
            case 24: add<24, nullable>(array, arrayValues, i); break;
            case 25: add<25, nullable>(array, arrayValues, i); break;
            case 26: add<26, nullable>(array, arrayValues, i); break;
            case 27: add<27, nullable>(array, arrayValues, i); break;
            case 28: add<28, nullable>(array, arrayValues, i); break;
            case 29: add<29, nullable>(array, arrayValues, i); break;
            case 30: add<30, nullable>(array, arrayValues, i); break;
            case 31: add<31, nullable>(array, arrayValues, i); break;
            case 32: add<32, nullable>(array, arrayValues, i); break;
            case 33: add<33, nullable>(array, arrayValues, i); break;
            case 34: add<34, nullable>(array, arrayValues, i); break;
            case 35: add<35, nullable>(array, arrayValues, i); break;
            case 36: add<36, nullable>(array, arrayValues, i); break;
            case 37: add<37, nullable>(array, arrayValues, i); break;
            case 38: add<38, nullable>(array, arrayValues, i); break;
            case 39: add<39, nullable>(array, arrayValues, i); break;
            case 40: add<40, nullable>(array, arrayValues, i); break;
            case 41: add<41, nullable>(array, arrayValues, i); break;
            case 42: add<42, nullable>(array, arrayValues, i); break;
            case 43: add<43, nullable>(array, arrayValues, i); break;
            case 44: add<44, nullable>(array, arrayValues, i); break;
            case 45: add<45, nullable>(array, arrayValues, i); break;
            case 46: add<46, nullable>(array, arrayValues, i); break;
            case 47: add<47, nullable>(array, arrayValues, i); break;
            case 48: add<48, nullable>(array, arrayValues, i); break;
            case 49: add<49, nullable>(array, arrayValues, i); break;
            case 50: add<50, nullable>(array, arrayValues, i); break;
            case 51: add<51, nullable>(array, arrayValues, i); break;
            case 52: add<52, nullable>(array, arrayValues, i); break;
            case 53: add<53, nullable>(array, arrayValues, i); break;
            case 54: add<54, nullable>(array, arrayValues, i); break;
            case 55: add<55, nullable>(array, arrayValues, i); break;
            case 56: add<56, nullable>(array, arrayValues, i); break;
            case 57: add<57, nullable>(array, arrayValues, i); break;
            case 58: add<58, nullable>(array, arrayValues, i); break;
            case 59: add<59, nullable>(array, arrayValues, i); break;
            case 60: add<60, nullable>(array, arrayValues, i); break;
            case 61: add<61, nullable>(array, arrayValues, i); break;
            case 62: add<62, nullable>(array, arrayValues, i); break;
            case 63: add<63, nullable>(array, arrayValues, i); break;
            case 64: add<64, nullable>(array, arrayValues, i); break;
            case 65: add<65, nullable>(array, arrayValues, i); break;
            case 66: add<66, nullable>(array, arrayValues, i); break;
            case 67: add<67, nullable>(array, arrayValues, i); break;
            case 68: add<68, nullable>(array, arrayValues, i); break;
            case 69: add<69, nullable>(array, arrayValues, i); break;
            case 70: add<70, nullable>(array, arrayValues, i); break;
            case 71: add<71, nullable>(array, arrayValues, i); break;
            case 72: add<72, nullable>(array, arrayValues, i); break;
            case 73: add<73, nullable>(array, arrayValues, i); break;
            case 74: add<74, nullable>(array, arrayValues, i); break;
            case 75: add<75, nullable>(array, arrayValues, i); break;
            case 76: add<76, nullable>(array, arrayValues, i); break;
            case 77: add<77, nullable>(array, arrayValues, i); break;
            case 78: add<78, nullable>(array, arrayValues, i); break;
            case 79: add<79, nullable>(array, arrayValues, i); break;
            case 80: add<80, nullable>(array, arrayValues, i); break;
            case 81: add<81, nullable>(array, arrayValues, i); break;
            case 82: add<82, nullable>(array, arrayValues, i); break;
            case 83: add<83, nullable>(array, arrayValues, i); break;
            case 84: add<84, nullable>(array, arrayValues, i); break;
            case 85: add<85, nullable>(array, arrayValues, i); break;
            case 86: add<86, nullable>(array, arrayValues, i); break;
            case 87: add<87, nullable>(array, arrayValues, i); break;
            case 88: add<88, nullable>(array, arrayValues, i); break;
            case 89: add<89, nullable>(array, arrayValues, i); break;
            case 90: add<90, nullable>(array, arrayValues, i); break;
            case 91: add<91, nullable>(array, arrayValues, i); break;
            case 92: add<92, nullable>(array, arrayValues, i); break;
            case 93: add<93, nullable>(array, arrayValues, i); break;
            case 94: add<94, nullable>(array, arrayValues, i); break;
            case 95: add<95, nullable>(array, arrayValues, i); break;
            case 96: add<96, nullable>(array, arrayValues, i); break;
            case 97: add<97, nullable>(array, arrayValues, i); break;
            case 98: add<98, nullable>(array, arrayValues, i); break;
            case 99: add<99, nullable>(array, arrayValues, i); break;
            case 100: add<100, nullable>(array, arrayValues, i); break;
            case 101: add<101, nullable>(array, arrayValues, i); break;
            case 102: add<102, nullable>(array, arrayValues, i); break;
            case 103: add<103, nullable>(array, arrayValues, i); break;
            case 104: add<104, nullable>(array, arrayValues, i); break;
            case 105: add<105, nullable>(array, arrayValues, i); break;
            case 106: add<106, nullable>(array, arrayValues, i); break;
            case 107: add<107, nullable>(array, arrayValues, i); break;
            case 108: add<108, nullable>(array, arrayValues, i); break;
            case 109: add<109, nullable>(array, arrayValues, i); break;
            case 110: add<110, nullable>(array, arrayValues, i); break;
            case 111: add<111, nullable>(array, arrayValues, i); break;
            case 112: add<112, nullable>(array, arrayValues, i); break;
            case 113: add<113, nullable>(array, arrayValues, i); break;
            case 114: add<114, nullable>(array, arrayValues, i); break;
            case 115: add<115, nullable>(array, arrayValues, i); break;
            case 116: add<116, nullable>(array, arrayValues, i); break;
            case 117: add<117, nullable>(array, arrayValues, i); break;
            case 118: add<118, nullable>(array, arrayValues, i); break;
            case 119: add<119, nullable>(array, arrayValues, i); break;
            case 120: add<120, nullable>(array, arrayValues, i); break;
            case 121: add<121, nullable>(array, arrayValues, i); break;
            case 122: add<122, nullable>(array, arrayValues, i); break;
            case 123: add<123, nullable>(array, arrayValues, i); break;
            case 124: add<124, nullable>(array, arrayValues, i); break;
            case 125: add<125, nullable>(array, arrayValues, i); break;
            case 126: add<126, nullable>(array, arrayValues, i); break;
            case 127: add<127, nullable>(array, arrayValues, i); break;
            case 128: add<128, nullable>(array, arrayValues, i); break;
            case 129: add<129, nullable>(array, arrayValues, i); break;
            case 130: add<130, nullable>(array, arrayValues, i); break;
            case 131: add<131, nullable>(array, arrayValues, i); break;
            case 132: add<132, nullable>(array, arrayValues, i); break;
            case 133: add<133, nullable>(array, arrayValues, i); break;
            case 134: add<134, nullable>(array, arrayValues, i); break;
            case 135: add<135, nullable>(array, arrayValues, i); break;
            case 136: add<136, nullable>(array, arrayValues, i); break;
            case 137: add<137, nullable>(array, arrayValues, i); break;
            case 138: add<138, nullable>(array, arrayValues, i); break;
            case 139: add<139, nullable>(array, arrayValues, i); break;
            case 140: add<140, nullable>(array, arrayValues, i); break;
            case 141: add<141, nullable>(array, arrayValues, i); break;
            case 142: add<142, nullable>(array, arrayValues, i); break;
            case 143: add<143, nullable>(array, arrayValues, i); break;
            case 144: add<144, nullable>(array, arrayValues, i); break;
            case 145: add<145, nullable>(array, arrayValues, i); break;
            case 146: add<146, nullable>(array, arrayValues, i); break;
            case 147: add<147, nullable>(array, arrayValues, i); break;
            case 148: add<148, nullable>(array, arrayValues, i); break;
            case 149: add<149, nullable>(array, arrayValues, i); break;
            case 150: add<150, nullable>(array, arrayValues, i); break;
            case 151: add<151, nullable>(array, arrayValues, i); break;
            case 152: add<152, nullable>(array, arrayValues, i); break;
            case 153: add<153, nullable>(array, arrayValues, i); break;
            case 154: add<154, nullable>(array, arrayValues, i); break;
            case 155: add<155, nullable>(array, arrayValues, i); break;
            case 156: add<156, nullable>(array, arrayValues, i); break;
            case 157: add<157, nullable>(array, arrayValues, i); break;
            case 158: add<158, nullable>(array, arrayValues, i); break;
            case 159: add<159, nullable>(array, arrayValues, i); break;
            case 160: add<160, nullable>(array, arrayValues, i); break;
            case 161: add<161, nullable>(array, arrayValues, i); break;
            case 162: add<162, nullable>(array, arrayValues, i); break;
            case 163: add<163, nullable>(array, arrayValues, i); break;
            case 164: add<164, nullable>(array, arrayValues, i); break;
            case 165: add<165, nullable>(array, arrayValues, i); break;
            case 166: add<166, nullable>(array, arrayValues, i); break;
            case 167: add<167, nullable>(array, arrayValues, i); break;
            case 168: add<168, nullable>(array, arrayValues, i); break;
            case 169: add<169, nullable>(array, arrayValues, i); break;
            case 170: add<170, nullable>(array, arrayValues, i); break;
            case 171: add<171, nullable>(array, arrayValues, i); break;
            case 172: add<172, nullable>(array, arrayValues, i); break;
            case 173: add<173, nullable>(array, arrayValues, i); break;
            case 174: add<174, nullable>(array, arrayValues, i); break;
            case 175: add<175, nullable>(array, arrayValues, i); break;
            case 176: add<176, nullable>(array, arrayValues, i); break;
            case 177: add<177, nullable>(array, arrayValues, i); break;
            case 178: add<178, nullable>(array, arrayValues, i); break;
            case 179: add<179, nullable>(array, arrayValues, i); break;
            case 180: add<180, nullable>(array, arrayValues, i); break;
            case 181: add<181, nullable>(array, arrayValues, i); break;
            case 182: add<182, nullable>(array, arrayValues, i); break;
            case 183: add<183, nullable>(array, arrayValues, i); break;
            case 184: add<184, nullable>(array, arrayValues, i); break;
            case 185: add<185, nullable>(array, arrayValues, i); break;
            case 186: add<186, nullable>(array, arrayValues, i); break;
            case 187: add<187, nullable>(array, arrayValues, i); break;
            case 188: add<188, nullable>(array, arrayValues, i); break;
            case 189: add<189, nullable>(array, arrayValues, i); break;
            case 190: add<190, nullable>(array, arrayValues, i); break;
            case 191: add<191, nullable>(array, arrayValues, i); break;
            case 192: add<192, nullable>(array, arrayValues, i); break;
            case 193: add<193, nullable>(array, arrayValues, i); break;
            case 194: add<194, nullable>(array, arrayValues, i); break;
            case 195: add<195, nullable>(array, arrayValues, i); break;
            case 196: add<196, nullable>(array, arrayValues, i); break;
            case 197: add<197, nullable>(array, arrayValues, i); break;
            case 198: add<198, nullable>(array, arrayValues, i); break;
            case 199: add<199, nullable>(array, arrayValues, i); break;
            case 200: add<200, nullable>(array, arrayValues, i); break;
            case 201: add<201, nullable>(array, arrayValues, i); break;
            case 202: add<202, nullable>(array, arrayValues, i); break;
            case 203: add<203, nullable>(array, arrayValues, i); break;
            case 204: add<204, nullable>(array, arrayValues, i); break;
            case 205: add<205, nullable>(array, arrayValues, i); break;
            case 206: add<206, nullable>(array, arrayValues, i); break;
            case 207: add<207, nullable>(array, arrayValues, i); break;
            case 208: add<208, nullable>(array, arrayValues, i); break;
            case 209: add<209, nullable>(array, arrayValues, i); break;
            case 210: add<210, nullable>(array, arrayValues, i); break;
            case 211: add<211, nullable>(array, arrayValues, i); break;
            case 212: add<212, nullable>(array, arrayValues, i); break;
            case 213: add<213, nullable>(array, arrayValues, i); break;
            case 214: add<214, nullable>(array, arrayValues, i); break;
            case 215: add<215, nullable>(array, arrayValues, i); break;
            case 216: add<216, nullable>(array, arrayValues, i); break;
            case 217: add<217, nullable>(array, arrayValues, i); break;
            case 218: add<218, nullable>(array, arrayValues, i); break;
            case 219: add<219, nullable>(array, arrayValues, i); break;
            case 220: add<220, nullable>(array, arrayValues, i); break;
            case 221: add<221, nullable>(array, arrayValues, i); break;
            case 222: add<222, nullable>(array, arrayValues, i); break;
            case 223: add<223, nullable>(array, arrayValues, i); break;
            case 224: add<224, nullable>(array, arrayValues, i); break;
            case 225: add<225, nullable>(array, arrayValues, i); break;
            case 226: add<226, nullable>(array, arrayValues, i); break;
            case 227: add<227, nullable>(array, arrayValues, i); break;
            case 228: add<228, nullable>(array, arrayValues, i); break;
            case 229: add<229, nullable>(array, arrayValues, i); break;
            case 230: add<230, nullable>(array, arrayValues, i); break;
            case 231: add<231, nullable>(array, arrayValues, i); break;
            case 232: add<232, nullable>(array, arrayValues, i); break;
            case 233: add<233, nullable>(array, arrayValues, i); break;
            case 234: add<234, nullable>(array, arrayValues, i); break;
            case 235: add<235, nullable>(array, arrayValues, i); break;
            case 236: add<236, nullable>(array, arrayValues, i); break;
            case 237: add<237, nullable>(array, arrayValues, i); break;
            case 238: add<238, nullable>(array, arrayValues, i); break;
            case 239: add<239, nullable>(array, arrayValues, i); break;
            case 240: add<240, nullable>(array, arrayValues, i); break;
            case 241: add<241, nullable>(array, arrayValues, i); break;
            case 242: add<242, nullable>(array, arrayValues, i); break;
            case 243: add<243, nullable>(array, arrayValues, i); break;
            case 244: add<244, nullable>(array, arrayValues, i); break;
            case 245: add<245, nullable>(array, arrayValues, i); break;
            case 246: add<246, nullable>(array, arrayValues, i); break;
            case 247: add<247, nullable>(array, arrayValues, i); break;
            case 248: add<248, nullable>(array, arrayValues, i); break;
            case 249: add<249, nullable>(array, arrayValues, i); break;
            case 250: add<250, nullable>(array, arrayValues, i); break;
            case 251: add<251, nullable>(array, arrayValues, i); break;
            case 252: add<252, nullable>(array, arrayValues, i); break;
            case 253: add<253, nullable>(array, arrayValues, i); break;
            case 254: add<254, nullable>(array, arrayValues, i); break;
            case 255: add<255, nullable>(array, arrayValues, i); break;
            }
        }

        processedCount += array.length();
    }

    _declspec(noinline) void add(const arrow::Array &array_)
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
    void add(const arrow::ChunkedArray &chunkedArray)
    {
        for(auto &&chunk : chunkedArray.chunks())
            add(*chunk);
    }
    void add(const arrow::Column &column)
    {
        add(*column.data());
    }
    
    std::shared_ptr<arrow::Array> finish()
    {
        return std::make_shared<Array>(length, values, bitmask, arrow::kUnknownNullCount);
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
    
    std::vector<std::shared_ptr<arrow::Array>> newColumns;


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

        // TODO handle zero chunks?
        // TODO handle more chunks
        const auto chunk = column->data()->chunk(0);


        visitArray(chunk.get(), [&](auto *array) 
        {
            using TD = ArrayTypeDescription<std::remove_pointer_t<decltype(array)>>;
            using T = typename TD::ValueType;

            if constexpr(std::is_scalar_v<T>)
            {
//                  measure("inner", 100000, [&]
//                  {
                    constexpr auto idd = std::is_same_v<T, int64_t> ? arrow::Type::INT64 : arrow::Type::DOUBLE;
                    FilteredArrayBuilder<idd> fab{newRowCount, mask->data()};
                    if(newRowCount)
                        fab.add(*chunk);
                    newColumns.push_back(fab.finish());
//                });
            }
            else
            {
                const auto stringSource = static_cast<const arrow::StringArray *>(array);
                arrow::StringBuilder builder;

                int32_t lengthBuffer;

                for(int64_t sourceItr = 0; sourceItr < oldRowCount; sourceItr++)
                {
                    if(arrow::BitUtil::GetBit(maskBuffer, sourceItr))
                    {
                        auto ptr = stringSource->GetValue(sourceItr, &lengthBuffer);
                        if(array->IsNull(sourceItr))
                            builder.AppendNull();
                        else
                            builder.Append(ptr, lengthBuffer);
                    }
                }

                newColumns.push_back(finish(builder));
            }
        });


    }

    return arrow::Table::Make(table->schema(), newColumns);
}

std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, v] = ast::parseValue(*table, dslJsonText);
    return execute(*table, v, mapping);
}