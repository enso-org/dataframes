#include <chrono>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "Core/ArrowUtilities.h"
#include "Core/Common.h"
#include "Core/Error.h"
#include "Core/Logger.h"
#include "Analysis.h"
#include "Processing.h"
#include "LifetimeManager.h"
#include "IO/csv.h"
#include "IO/Feather.h"
#include "IO/IO.h"
#include "IO/JSON.h"
#include "IO/XLSX.h"


#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/util/checked_cast.h>
#include <arrow/builder.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#ifdef _DEBUG
#pragma comment(lib, "arrowd.lib")
#else
#pragma comment(lib, "arrow.lib")
#endif

using namespace std;

template<typename ...Ts> struct TypeList {};

using Types = TypeList<int32_t, int64_t, float, double>; 

struct UInt8  { static constexpr arrow::Type::type id = arrow::Type::UINT8 ; };
struct UInt16 { static constexpr arrow::Type::type id = arrow::Type::UINT16; };
struct UInt32 { static constexpr arrow::Type::type id = arrow::Type::UINT32; };
struct UInt64 { static constexpr arrow::Type::type id = arrow::Type::UINT64; };
struct Int8   { static constexpr arrow::Type::type id = arrow::Type::INT8 ;  };
struct Int16  { static constexpr arrow::Type::type id = arrow::Type::INT16;  };
struct Int32  { static constexpr arrow::Type::type id = arrow::Type::INT32;  };
struct Int64  { static constexpr arrow::Type::type id = arrow::Type::INT64;  };
struct Float  { static constexpr arrow::Type::type id = arrow::Type::FLOAT;  };
struct Double { static constexpr arrow::Type::type id = arrow::Type::DOUBLE; };
struct String { static constexpr arrow::Type::type id = arrow::Type::STRING; };

template<typename Tag>
using TypeDescriptionForTag = TypeDescription<Tag::id>;

template<typename TypeTag>
auto asSpecificArray(arrow::Array *array)
{
    return throwingDowncastArray<TypeTag::id>(array);
}

template<typename T>
void validateIndex(const std::vector<T> &array, int64_t index)
{
    if(index < 0 || index >= (int64_t)array.size())
    {
        std::ostringstream out;
        out << "wrong index " << index << " when array length is " << array.size();
        throw std::out_of_range{ out.str() };
    }
}

std::shared_ptr<arrow::DataType> idToDataType(arrow::Type::type id)
{
    switch(id)
    {
    case arrow::Type::INT32:
        return std::make_shared<arrow::Int32Type>();
    case arrow::Type::INT64:
        return std::make_shared<arrow::Int64Type>();
    case arrow::Type::DOUBLE:
        return std::make_shared<arrow::DoubleType>();
    case arrow::Type::STRING:
        return std::make_shared<arrow::StringType>();
    default:
    {
        std::ostringstream out;
        out << "Not yet supported datatype id: " << id;
        throw std::runtime_error(out.str());
    }
    }
}

namespace
{
    // NOTE: we need an object that will keep returned strings alive in memory
    // it must be thread local, as two luna threads can concurrently call out methods
    thread_local std::string returnedStringBuffer;

    const char *returnString(std::string s)
    {
        returnedStringBuffer = std::move(s);
        return returnedStringBuffer.c_str();
    }

    HeaderPolicy headerPolicyFromC(int columnNamesPolicy, const char **columnNames)
    {
        if(columnNamesPolicy < 0)
        {
            std::vector<std::string> names;
            for(int i = 0; i < -columnNamesPolicy; i++)
                names.push_back(columnNames[i]);
            return names;
        }
        else if(columnNamesPolicy == 0)
        {
            static_assert(std::is_same_v<TakeFirstRowAsHeaders, std::variant_alternative_t<0, HeaderPolicy>>);
            return TakeFirstRowAsHeaders{};
        }
        else if(columnNamesPolicy == 1)
        {
            static_assert(std::is_same_v<GenerateColumnNames, std::variant_alternative_t<1, HeaderPolicy>>);
            return GenerateColumnNames{};
        }
        else
            throw std::runtime_error("Invalid column name policy code " + std::to_string(columnNamesPolicy));
    };
    std::vector<ColumnType> columnTypesFromC(int columnTypeInfoCount, const int8_t *columnTypes, const int8_t *columnIsNullableTypes)
    {
         std::vector<ColumnType> types;
         for(int i = 0; i < columnTypeInfoCount; i++)
         {
             auto typeId = (arrow::Type::type) columnTypes[i];
             bool nullable = columnIsNullableTypes[i];
             auto type = idToDataType(typeId);
             types.emplace_back(type, nullable, true);
         }
         return types;
    };
    template<typename T>
    std::vector<T> vectorFromC(const T *vals, int32_t size)
    {
        std::vector<T> ret;
        for(int i = 0; i < size; i++)
            ret.push_back(vals[i]);
        return ret;
    }
}

extern "C"
{
    EXPORT void setVerbosity(bool verbose)
    {
        Logger::instance().enabled.store(verbose);
    }
}

// DATATYPE
extern "C"
{
    // NOTE: needs release
    EXPORT arrow::DataType *dataTypeNew(int8_t id, const char **outError)
    {
        LOG("{}", (int)id);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = idToDataType((arrow::Type::type)id);
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }
    EXPORT const char *dataTypeName(arrow::DataType *datatype, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(datatype->name());
        };
    }
    EXPORT const char *dataTypeToString(arrow::DataType *datatype, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(datatype->ToString());
        };
    }
    EXPORT std::int8_t dataTypeId(arrow::DataType *datatype, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return datatype->id();
        };
    }
    EXPORT std::int32_t dataTypeFixedWidthTypeBitWidth(const arrow::DataType *datatype, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return throwingCast<const arrow::FixedWidthType*>(datatype)->bit_width();
        };
    }
}


extern "C"
{
#define COMMON_BUILDER(TYPENAME) \
    EXPORT TypeDescriptionForTag<TYPENAME>::BuilderType *builder##TYPENAME##New(const char **outError) noexcept                                                               \
    {                                                                                                                                                       \
        LOG(""); \
        /* NOTE: needs release */                                                                                                                           \
        return TRANSLATE_EXCEPTION(outError)                                                                                                               \
        {                                                                                                                                                   \
            return LifetimeManager::instance().addOwnership(std::make_shared<TypeDescriptionForTag<TYPENAME>::BuilderType>());                                       \
        };                                                                                                                                                   \
    }                                                                                                                                                       \
    EXPORT void builder##TYPENAME##Reserve(TypeDescriptionForTag<TYPENAME>::BuilderType *builder, int64_t count, const char **outError) noexcept                           \
    {                                                                                                                                                       \
        LOG("@{}: {}", (void*)builder, count); \
        TRANSLATE_EXCEPTION(outError)                                                                                                                       \
        {                                                                                                                                                   \
            checkStatus(builder->Reserve(count));                                                                                                           \
        };                                                                                                                                                   \
    }                                                                                                                                                       \
    EXPORT void builder##TYPENAME##Resize(TypeDescriptionForTag<TYPENAME>::BuilderType *builder, int64_t count, const char **outError) noexcept                               \
    {                                                                                                                                                       \
        LOG("@{}: {}", (void*)builder, count); \
        return TRANSLATE_EXCEPTION(outError)                                                                                                               \
        {                                                                                                                                                   \
            checkStatus(builder->Resize(count));                                                                                                           \
        };                                                                                                                                                   \
    }                                                                                                                                                       \
    EXPORT void builder##TYPENAME##AppendValue(TypeDescriptionForTag<TYPENAME>::BuilderType *builder, TypeDescriptionForTag<TYPENAME>::CType value, const char **outError) noexcept \
    {                                                                                                                                                       \
        LOG("@{}: {} :: {}", (void*)builder, value, #TYPENAME); \
        return TRANSLATE_EXCEPTION(outError)                                                                                                               \
        {                                                                                                                                                   \
            checkStatus(builder->Append(value));                                                                                                           \
        };                                                                                                                                                   \
    }                                                                                                                                                       \
    EXPORT void builder##TYPENAME##AppendNull(TypeDescriptionForTag<TYPENAME>::BuilderType *builder, const char **outError) noexcept                                       \
    {                                                                                                                                                       \
        LOG("@{}", (void*)builder); \
        return TRANSLATE_EXCEPTION(outError)                                                                                                               \
        {                                                                                                                                                   \
            checkStatus(builder->AppendNull());                                                                                                               \
        };                                                                                                                                                   \
    }                                                                                                                                                       \
    EXPORT arrow::Array *builder##TYPENAME##Finish(TypeDescriptionForTag<TYPENAME>::BuilderType *builder, const char **outError) noexcept                                   \
    {                                                                                                                                                       \
        LOG("@{}", (void*)builder); \
        return TRANSLATE_EXCEPTION(outError)                                                                                                               \
        {                                                                                                                                                   \
            std::shared_ptr<arrow::Array> resultArray = nullptr;                                                                                           \
            checkStatus(builder->Finish(&resultArray));                                                                                                       \
            /* Note: owner must release ownership by arrayRelease */                                                                                       \
            return LifetimeManager::instance().addOwnership(resultArray);                                                                                   \
        };                                                                                                                                                   \
    }

    COMMON_BUILDER(UInt8);
    COMMON_BUILDER(UInt16);
    COMMON_BUILDER(UInt32);
    COMMON_BUILDER(UInt64);
    COMMON_BUILDER(Int8);
    COMMON_BUILDER(Int16);
    COMMON_BUILDER(Int32);
    COMMON_BUILDER(Int64);
    COMMON_BUILDER(Float);
    COMMON_BUILDER(Double);
    COMMON_BUILDER(String);

    // TODO current string append needlessly allocates std::string for BinaryBuilder::Append argument

    EXPORT int64_t builderLength(arrow::ArrayBuilder *builder) noexcept
    {
        LOG("@{}", (void*)builder);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return builder->length();
        };
    }
    EXPORT int64_t builderCapacity(arrow::ArrayBuilder *builder) noexcept
    {
        LOG("@{}", (void*)builder);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return builder->capacity();
        };
    }
    EXPORT int64_t builderNullCount(arrow::ArrayBuilder *builder) noexcept
    {
        LOG("@{}", (void*)builder);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return builder->null_count();
        };
    }
//     EXPORT arrow::ResizableBuffer *builderObtainNullBuffer(arrow::ArrayBuilder *builder, const char **outError) noexcept
//     {
//         LOG("@{}", (void*)builder);
//         return TRANSLATE_EXCEPTION(outError)
//         {
//             return LifetimeManager::instance().addOwnership(builder->null_bitmap());
//         };
//     }
    // needs release
    EXPORT arrow::DataType *builderObtainType(arrow::ArrayBuilder *builder, const char **outError) noexcept
    {
        LOG("@{}", (void*)builder);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(builder->type());
        };
    }
}

// BUFFER
extern "C"
{
    EXPORT int64_t bufferSize(arrow::Buffer *buffer, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return buffer->size();
        };
    }

    EXPORT const void *bufferData(arrow::Buffer *buffer, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            return buffer->data();
        };
    }

    // NOTE: needs release 
    // (shallow view)
    EXPORT arrow::Buffer *bufferSlice(arrow::Buffer *buffer, int64_t start, int64_t byteCount, const char **outError) noexcept
    {
        LOG("@{} beginning from {} will copy {} bytes", (void*)buffer, start, byteCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            // TODO check bounds
            auto src = LifetimeManager::instance().accessOwned<arrow::Buffer>(buffer);
            auto ret = arrow::SliceBuffer(src, start, byteCount);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }

    // NOTE: needs release 
    // (deep copy)
    EXPORT arrow::Buffer *bufferCopy(arrow::Buffer *buffer, int64_t start, int64_t byteCount, const char **outError) noexcept
    {
        LOG("@{} beginning from {} will copy {} bytes", (void*)buffer, start, byteCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            // TODO check if needs checking bounds
            std::shared_ptr<arrow::Buffer> ret = nullptr;
            checkStatus(buffer->Copy(start, byteCount, &ret));
            return LifetimeManager::instance().addOwnership(ret);
        };
    }

    EXPORT int64_t bufferCapacity(arrow::Buffer *buffer, const char **outError) noexcept
    {
        LOG("@{}", (void*)buffer);
        return TRANSLATE_EXCEPTION(outError)
        {
            return buffer->capacity();
        };
    }
}

// ARRAY DATA
extern "C"
{
    EXPORT int64_t arrayDataLength(arrow::ArrayData *arrayData) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return arrayData->length;
        };
    }
    EXPORT int64_t arrayDataNullCount(arrow::ArrayData *arrayData) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return arrayData->null_count;
        };
    }
    EXPORT int64_t arrayDataOffset(arrow::ArrayData *arrayData) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return arrayData->offset;
        };
    }

    // NOTE: needs release
    EXPORT arrow::DataType *arrayDataType(arrow::ArrayData *arrayData, const char **outError) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return LifetimeManager::instance().addOwnership(arrayData->type);
        };
    }

    EXPORT int64_t arrayDataBufferCount(arrow::ArrayData *arrayData) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return arrayData->buffers.size();
        };
    }

    // NOTE: needs release
    EXPORT arrow::Buffer *arrayDataBufferAt(arrow::ArrayData *arrayData, size_t bufferIndex, const char **outError) noexcept
    {
        LOG("@{}", (void*)arrayData);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(arrayData->buffers.at(bufferIndex));
        };
    }
}

// ARRAY
extern "C"
{
    EXPORT int64_t arrayBufferCount(arrow::Array *array) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return array->data()->length;
        };
    }

    // arrow::Array subclasses are needed to obtain specific elements or access
    // raw values buffer. We use macro to generate method for each supported type.
    // 
    // This is likely the best we can do when building C language API.

#define NUMERIC_ARRAY_METHODS(TYPENAME)                                                                                                  \
    EXPORT  TypeDescriptionForTag<TYPENAME>::CType array##TYPENAME##ValueAt(arrow::Array *array, int64_t index, const char **outError) noexcept\
    {                                                                                                                                    \
        LOG("[{}]", index);                                                                                                              \
        /* NOTE: needs release */                                                                                                        \
        return TRANSLATE_EXCEPTION(outError)                                                                                             \
        {                                                                                                                                \
            validateIndex(*array, index);                                                                                                 \
            return asSpecificArray<TYPENAME>(array)->Value(index);                                                                       \
        };                                                                                                                               \
    }                                                                                                                                    \
    EXPORT const TypeDescriptionForTag<TYPENAME>::CType *array##TYPENAME##RawValues(arrow::Array *array, const char **outError) noexcept       \
    {                                                                                                                                    \
        LOG("@{}", (void*)array);                                                                                                        \
        /* NOTE: needs release */                                                                                                        \
        return TRANSLATE_EXCEPTION(outError)                                                                                             \
        {                                                                                                                                \
            return asSpecificArray<TYPENAME>(array)->raw_values();                                                                       \
        };                                                                                                                               \
    }

    NUMERIC_ARRAY_METHODS(UInt8);
    NUMERIC_ARRAY_METHODS(UInt16);
    NUMERIC_ARRAY_METHODS(UInt32);
    NUMERIC_ARRAY_METHODS(UInt64);
    NUMERIC_ARRAY_METHODS(Int8);
    NUMERIC_ARRAY_METHODS(Int16);
    NUMERIC_ARRAY_METHODS(Int32);
    NUMERIC_ARRAY_METHODS(Int64);
    NUMERIC_ARRAY_METHODS(Float);
    NUMERIC_ARRAY_METHODS(Double);

    // string array uses somewhat different interface than numeric -- and needs a special method thereof
    // TODO should an actual array subtype be required? generally speaking having right data in array should be enough
    EXPORT const char * arrayStringValueAt(arrow::Array *array, int64_t index, const char **outError) noexcept
    {                                                                                                         
        LOG("[{}]", index);                                                                                   
        /* NOTE: needs release */                                                                             
        return TRANSLATE_EXCEPTION(outError)                                                                  
        {                                                                                                     
            validateIndex(*array, index);
            return returnString(asSpecificArray<String>(array)->GetString(index));
        };
    }


    // NOTE: needs release
    EXPORT arrow::Buffer *primitiveArrayValueBuffer(arrow::Array *array, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            if(auto primitiveArray = dynamic_cast<arrow::PrimitiveArray*>(array))
                return LifetimeManager::instance().addOwnership(primitiveArray->values());

            std::ostringstream out;
            out << "Failed to cast " << array << " to specific array!";
            throw std::runtime_error(out.str());
        };
    }

    // NOTE: needs release
    EXPORT arrow::Buffer *arrayNullBitmapBuffer(arrow::Array *array, size_t bufferIndex, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(array->null_bitmap());
        };
    }

    // NOTE: needs release
    EXPORT arrow::ArrayData *arrayData(arrow::Array *array, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(array->data());
        };
    }

    // NOTE: needs release
    EXPORT arrow::Array *arraySlice(arrow::Array *array, int64_t offset, int64_t length, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(array->Slice(offset, length));
        };
    }

    EXPORT bool arrayIsNullAt(arrow::Array *array, int64_t index, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            validateIndex(*array, index);
            return array->IsNull(index);
        };
    }

    EXPORT int64_t arrayLength(arrow::Array *array) noexcept
    {
        LOG("@{}", (void*)array);
        return array->length();
    }

    EXPORT int64_t nullCount(arrow::Array *array) noexcept
    {
        LOG("@{}", (void*)array);
        return array->null_count();
    }
}

// CHUNKED ARRAY
extern "C"
{
    EXPORT arrow::ChunkedArray *chunkedArrayNewSingleton(arrow::Array *array, const char **outError)
    {
        LOG("{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedArray = LifetimeManager::instance().accessOwned(array);
            auto ptr = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{managedArray});
            return LifetimeManager::instance().addOwnership(std::move(ptr));
        };
    }
    EXPORT arrow::ChunkedArray *chunkedArrayNewChunks(const arrow::Array **arrays, int32_t chunkCount, const char **outError)
    {
        LOG("{} {}", (void*)arrays, chunkCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            arrow::ArrayVector chunksCollected = LifetimeManager::instance().accessOwnedArray(arrays, chunkCount);

            // All arrays in chunk must be of the same type
            for(int32_t i = 1; i < chunkCount; i++)
            {
                if(i != 0)
                {
                    if(!chunksCollected.front()->type()->Equals(chunksCollected.back()->type()))
                    {
                        std::ostringstream msg;
                        auto writeTypeAt = [&] (int index) 
                        {
                            msg << "at " << index << " the chunk type is " << chunksCollected.at(index)->type()->ToString();
                        };
                        msg << "Type mismatch between given array chunks: ";
                        writeTypeAt(0);
                        msg << "; while ";
                        writeTypeAt(i);
                        throw std::runtime_error(msg.str());
                    }
                }
            }

            auto ptr = std::make_shared<arrow::ChunkedArray>(std::move(chunksCollected));
            return LifetimeManager::instance().addOwnership(std::move(ptr));
        };
    }

    EXPORT int64_t chunkedArrayLength(arrow::ChunkedArray *array)
    {
        LOG("@{}", (void*)array);
        return array->length();
    }
    EXPORT int64_t chunkedArrayNullCount(arrow::ChunkedArray *array)
    {
        LOG("@{}", (void*)array);
        return array->null_count();
    }
    EXPORT int64_t chunkedArrayChunkCount(arrow::ChunkedArray *array)
    {
        LOG("@{}", (void*)array);
        return array->num_chunks();
    }

    // NOTE: needs release
    EXPORT arrow::Array *chunkedArrayChunkAt(arrow::ChunkedArray *array, int32_t index, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
			validateIndex(array->chunks(), index);
            return LifetimeManager::instance().addOwnership(array->chunk(index));
        };
    }

    // NOTE: needs release
    EXPORT arrow::DataType *chunkedArrayDataType(arrow::ChunkedArray *array, const char **outError) noexcept
    {
        LOG("@{}", (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(array->type());
        };
    }
}

// FIELD
extern "C"
{
    EXPORT arrow::Field *fieldNew(const char *name, const arrow::DataType *type, bool nullable, const char **outError) noexcept
    {
        LOG("{} {} {}", name, type->ToString(), nullable);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedType = LifetimeManager::instance().accessOwned(type);
            auto ret = std::make_shared<arrow::Field>(name, std::move(managedType), nullable);
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }
    EXPORT const char *fieldName(arrow::Field *field, const char **outError) noexcept
    {
        LOG("@{}", (void*)field);
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(field->name());
        };
    }
    EXPORT arrow::DataType *fieldType(arrow::Field *field, const char **outError) noexcept
    {
        LOG("@{}", (void*)field);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(field->type());
        };
    }
    EXPORT bool fieldNullable(arrow::Field *field) noexcept
    {
        LOG("@{}", (void*)field);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return field->nullable();
        };
    }
    EXPORT const char *fieldToString(arrow::Field *field, const char **outError) noexcept
    {
        LOG("@{}", (void*)field);
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(field->ToString());
        };
    }
    EXPORT bool fieldEquals(arrow::Field *lhs, arrow::Field *rhs, const char **outError) noexcept
    {
        LOG("@{} @{}", (void*)lhs, (void*)rhs);
        return TRANSLATE_EXCEPTION(outError)
        {
            return lhs->Equals(*rhs);
        };
    }
}

// COLUMN
extern "C"
{
    EXPORT arrow::Column *columnNew(arrow::Field *field, arrow::ChunkedArray *array, const char **outError) noexcept
    {
        LOG("{} {}", (void*)field, (void*)array);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto fieldManaged = LifetimeManager::instance().accessOwned(field);
            auto arrayManaged = LifetimeManager::instance().accessOwned(array);
            auto ret = std::make_shared<arrow::Column>(fieldManaged, arrayManaged);
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }
    EXPORT int64_t columnLength(arrow::Column *column) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return column->length();
        };
    }
    EXPORT int64_t columnNullCount(arrow::Column *column) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return column->null_count();
        };
    }
    EXPORT arrow::Field *columnField(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(column->field());
        };
    }
    EXPORT arrow::DataType *columnType(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(column->type());
        };
    }
    EXPORT arrow::ChunkedArray *columnData(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(column->data());
        };
    }
    EXPORT const char *columnName(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(column->name());
        };
    }
    EXPORT arrow::Column *columnSlice(arrow::Column *column, int64_t fromIndex, int64_t count, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(column->Slice(fromIndex, count));
        };
    }
    EXPORT arrow::Table *columnCountValues(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = countValues(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnMin(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateMin(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnMax(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateMax(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnMean(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateMean(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnMedian(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateMedian(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnStd(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateStandardDeviation(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnVar(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateVariance(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnSum(arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateSum(*column);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *columnQuantile(arrow::Column *column, double q, const char **outError) noexcept
    {
        LOG("@{}", (void*)column);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateQuantile(*column, q);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
}

// SCHEMA
extern "C"
{
    EXPORT arrow::Schema *schemaNew(const arrow::Field **fields, int32_t fieldCount, const char **outError) noexcept
    {
        LOG("{} {}", (void*)fields, fieldCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto fieldsManaged = LifetimeManager::instance().accessOwnedArray(fields, fieldCount);
            auto ret = std::make_shared<arrow::Schema>(std::move(fieldsManaged));
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }

    EXPORT int32_t schemaFieldCount(arrow::Schema *schema) noexcept
    {
        LOG("@{}", (void*)schema);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return schema->num_fields();
        };
    }

    EXPORT arrow::Field *schemaFieldAt(arrow::Schema *schema, int32_t index, const char **outError) noexcept
    {
        LOG("@{}, {}", (void*)schema, index);
        return TRANSLATE_EXCEPTION(outError)
        {
            validateIndex(schema->fields(), index);
            return LifetimeManager::instance().addOwnership(schema->field(index));
        };
    }

    EXPORT arrow::Field *schemaFieldByName(arrow::Schema *schema, const char *name, const char **outError) noexcept
    {
        LOG("@{}, {}", (void*)schema, name);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(schema->GetFieldByName(name));
        };
    }

    EXPORT int32_t schemaFieldIndexByName(arrow::Schema *schema, const char *name, const char **outError) noexcept
    {
        LOG("@{}, {}", (void*)schema, name);
        return TRANSLATE_EXCEPTION(outError)
        {
            return (int32_t)schema->GetFieldIndex(name);
        };
    }

    EXPORT const char *schemaToString(arrow::Schema *schema, const char **outError) noexcept
    {
        LOG("@{}", (void*)schema);
        return TRANSLATE_EXCEPTION(outError)
        {
            return returnString(schema->ToString());
        };
    }

    EXPORT arrow::Schema *schemaAddField(arrow::Schema *schema, int32_t index, arrow::Field *field, const char **outError) noexcept
    {
        LOG("@{} {} {}", (void*)schema, index, (void*)field);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto fieldManaged = LifetimeManager::instance().accessOwned(field);

            std::shared_ptr<arrow::Schema> ret;
            checkStatus(schema->AddField(index, fieldManaged, &ret));
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }
    EXPORT arrow::Schema *schemaRemoveField(arrow::Schema *schema, int32_t index, const char **outError) noexcept
    {
        LOG("@{} {}", (void*)schema, index);
        return TRANSLATE_EXCEPTION(outError)
        {
            std::shared_ptr<arrow::Schema> ret;
            checkStatus(schema->RemoveField(index, &ret));
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }
}

// TABLE
extern "C"
{
    EXPORT arrow::Table *tableNewFromSchamColumns(arrow::Schema *schema, const arrow::Column **columns, int32_t columnCount, const char **outError) noexcept
    {
        LOG("{} {} {}", (void*)schema, (void*)columns, columnCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedSchema = LifetimeManager::instance().accessOwned(schema);
            auto managedColumns = LifetimeManager::instance().accessOwnedArray(columns, columnCount);
            auto ret = arrow::Table::Make(managedSchema, managedColumns);
            return LifetimeManager::instance().addOwnership(std::move(ret));
        };
    }

    EXPORT arrow::Schema *tableSchema(arrow::Table *table, const char **outError) noexcept
    {
        LOG("@{}", (void*)table);
        return TRANSLATE_EXCEPTION(outError)
        {
            return LifetimeManager::instance().addOwnership(table->schema());
        };
    }

    EXPORT std::int64_t tableRowCount(arrow::Table *table) noexcept
    {
        LOG("@{}", (void*)table);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return table->num_rows();
        };
    }
    EXPORT std::int32_t tableColumnCount(arrow::Table *table) noexcept
    {
        LOG("@{}", (void*)table);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            return table->num_columns();
        };
    }
    EXPORT arrow::Column *tableColumnAt(arrow::Table *table, int32_t index, const char **outError) noexcept
    {
        LOG("@{} {}", (void*)table, index);
        return TRANSLATE_EXCEPTION(outError)
        {
            validateIndex(table->num_columns(), index);
            return LifetimeManager::instance().addOwnership(table->column(index));
        };
    }
    EXPORT arrow::Table *tableAddColumn(arrow::Table *table, int32_t index, arrow::Column *column, const char **outError) noexcept
    {
        LOG("@{} {}", (void*)table, index);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedColumn = LifetimeManager::instance().accessOwned(column);

            std::shared_ptr<arrow::Table> ret;
            checkStatus(table->AddColumn(index, managedColumn, &ret));
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Table *tableRemoveColumn(arrow::Table *table, int32_t index, const char **outError) noexcept
    {
        LOG("@{} {}", (void*)table, index);
        return TRANSLATE_EXCEPTION(outError)
        {
            std::shared_ptr<arrow::Table> ret;
            checkStatus(table->RemoveColumn(index, &ret));
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT bool tableEquals(arrow::Table *lhs, arrow::Table *rhs, const char **outError) noexcept
    {
        LOG("@{} @{}", (void*)lhs, (void*)rhs);
        return TRANSLATE_EXCEPTION(outError)
        {
            return lhs->Equals(*rhs);
        };
    }
    EXPORT arrow::Table *tableFilter(arrow::Table *table, const char *lqueryJSON, const char **outError) noexcept
    {
        LOG("@{} @{}", (void*)table, (void*)lqueryJSON);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);
            auto ret = filter(managedTable, lqueryJSON);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::ChunkedArray *tableMapToChunkedArray(arrow::Table *table, const char *lqueryJSON, const char **outError) noexcept
    {
        LOG("@{} @{}", (void*)table, (void*)lqueryJSON);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);
            auto chunk = each(managedTable, lqueryJSON);
            auto ret = std::make_shared<arrow::ChunkedArray>(std::vector<std::shared_ptr<arrow::Array>>{chunk});
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Column *tableMapToColumn(arrow::Table *table, const char *retName, const char *lqueryJSON, const char **outError) noexcept
    {
        LOG("@{} @{}", (void*)table, (void*)lqueryJSON);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);
            auto chunk = each(managedTable, lqueryJSON);
            auto field = arrow::field(retName, chunk->type(), chunk->null_count());
            auto ret = std::make_shared<arrow::Column>(field, chunk);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
//     EXPORT arrow::Table *tableDropNABy(arrow::Table *table, const int32_t *indices, int32_t columnCount, const char **outError) noexcept
//     {
//         LOG("@{} column count={}", (void*)table, columnCount);
//         return TRANSLATE_EXCEPTION(outError)
//         {
//             auto managedTable = LifetimeManager::instance().accessOwned(table);
//             std::vector<int> indicesManaged = vectorFromC(indices, columnCount);
//             auto ret = dropNA(managedTable, indicesManaged);
//             return LifetimeManager::instance().addOwnership(ret);
//         };
//     }
    EXPORT arrow::Table *tableDropNA(arrow::Table *table, const char **outError) noexcept
    {
        LOG("@{}", (void*)table);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);
            auto ret = dropNA(managedTable);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
    EXPORT arrow::Table *tableDropNAByName(arrow::Table *table, const char *columnName, const char **outError) noexcept
    {
        LOG("@{} column name={}", (void*)table, columnName);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);
            for(int i = 0; i < table->num_columns(); i++)
            {
                auto column = table->column(i);
                if(column->name() == columnName)
                {
                    auto ret = dropNA(managedTable, {i});
                    return LifetimeManager::instance().addOwnership(ret);
                }
            }

            throw std::runtime_error("Failed to find column by name: "s + columnName);
        };
    }
    EXPORT arrow::Table *tableFillNA(arrow::Table *table, const char *valueJSON, const char **outError) noexcept
    {
        LOG("@{} value={}", (void*)table, valueJSON);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto managedTable = LifetimeManager::instance().accessOwned(table);

            std::unordered_map<std::string, DynamicField> fillValuePerColumn;

            auto doc = parseJSON(valueJSON);
            if(doc.IsObject())
            {
                const auto columnMap = getColumnMap(*table);
                for(auto itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr)
                {
                    const auto name = itr->name.GetString();
                    if(auto colItr = columnMap.find(name); colItr != columnMap.end())
                    {
                        const auto value = parseAsField(itr->value);
                        fillValuePerColumn[name] = adjustTypeForFilling(value, *colItr->second->type());
                    }
                    else
                        throw std::runtime_error("cannot find column by name: "s + name);
                }
            }
            else
            {
                auto value = parseAsField(valueJSON);
                for(auto &column : getColumns(*table))
                {
                    try
                    {
                        const auto typeId = column->type()->id();
                        fillValuePerColumn[column->name()] = adjustTypeForFilling(value, *column->type());
                    }
                    catch(std::exception &e)
                    {
                        throw std::runtime_error("cannot fill column " + column->name() + " of type " 
                            + column->type()->ToString() + " with value " + valueJSON + ": " + e.what());
                    }
                }
            }

            auto ret = fillNA(managedTable, fillValuePerColumn);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }

    EXPORT arrow::Table *tableCorrelationMatrix(arrow::Table *table, const char **outError) noexcept
    {
        LOG("@{} value={}", (void*)table);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto ret = calculateCorrelationMatrix(*table);
            return LifetimeManager::instance().addOwnership(ret);
        };
    }
}

arrow::Table *readTableFromCSVFileContentsHelper(std::string data, const char **columnNames, int32_t columnNamesPolicy, int8_t *columnTypes, int8_t *columnIsNullableTypes, int32_t columnTypeInfoCount)
{
    const auto headerPolicy = headerPolicyFromC(columnNamesPolicy, columnNames);
    const auto types = columnTypesFromC(columnTypeInfoCount, columnTypes, columnIsNullableTypes);

    auto csv = parseCsvData(std::move(data)); 
    auto table = csvToArrowTable(csv, headerPolicy, types);
    LOG("table has size {}x{}", table->num_columns(), table->num_rows());
    return LifetimeManager::instance().addOwnership(table);
}


// IO
extern "C"
{
    EXPORT arrow::Table *readTableFromFile(const char *filename, const char **outError)
    {
        LOG("@{}", filename);
        return TRANSLATE_EXCEPTION(outError)
        {
            // assume csv
            // TODO deduce separators
            auto csv = parseCsvFile(filename); 
            auto table = csvToArrowTable(csv, TakeFirstRowAsHeaders{}, {});
            return LifetimeManager::instance().addOwnership(table);
        };
    }

    EXPORT arrow::Table *readTableFromCSVFileContents(const char *data, const char **columnNames, int32_t columnNamesPolicy, int8_t *columnTypes, int8_t *columnIsNullableTypes, int32_t columnTypeInfoCount, const char **outError)
    {
        LOG("size={} names={}, namesPolicyCode={}, typeInfoCount={}", std::strlen(data), (void*)columnNames, columnNamesPolicy, columnTypeInfoCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            std::string buffer{ data };
            return readTableFromCSVFileContentsHelper(std::move(data), columnNames, columnNamesPolicy, columnTypes, columnIsNullableTypes, columnTypeInfoCount);
        };
    }

    EXPORT arrow::Table *readTableFromCSVFile(const char *filename, const char **columnNames, int32_t columnNamesPolicy, int8_t *columnTypes, int8_t *columnIsNullableTypes, int32_t columnTypeInfoCount, const char **outError)
    {
        LOG("@{} names={}, namesPolicyCode={}, typeInfoCount={}", filename, (void*)columnNames, columnNamesPolicy, columnTypeInfoCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto buffer = getFileContents(filename);
            return readTableFromCSVFileContentsHelper(std::move(buffer ), columnNames, columnNamesPolicy, columnTypes, columnIsNullableTypes, columnTypeInfoCount);
        };
    }

    EXPORT const char *writeTableToCsvString(arrow::Table *table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, const char **outError)
    {
        LOG("table={}", (void*)table);
        return TRANSLATE_EXCEPTION(outError)
        {
            std::ostringstream out;
            generateCsv(out, *table, headerPolicy, quotingPolicy);
            return returnString(out.str());
        };
    }

    EXPORT void writeTableToCsvFile(const char *filename, arrow::Table *table, GeneratorHeaderPolicy headerPolicy, GeneratorQuotingPolicy quotingPolicy, const char **outError)
    {
        LOG("table={}, filepath={}", (void*)table, filename);
        return TRANSLATE_EXCEPTION(outError)
        {
            std::ofstream out{filename};
            if(!out)
                throw std::runtime_error("Cannot write to file "s + filename);

            generateCsv(out, *table, headerPolicy, quotingPolicy);
        };
    }

    EXPORT arrow::Table *readTableFromXLSXFile(const char *filename, const char **columnNames, int32_t columnNamesPolicy, int8_t *columnTypes, int8_t *columnIsNullableTypes, int32_t columnTypeInfoCount, const char **outError)
    {
        LOG("@{} names={}, namesPolicyCode={}, typeInfoCount={}", filename, (void*)columnNames, columnNamesPolicy, columnTypeInfoCount);
        return TRANSLATE_EXCEPTION(outError)
        {
            const auto headerPolicy = headerPolicyFromC(columnNamesPolicy, columnNames);
            const auto columnTypesPolicy = columnTypesFromC(columnTypeInfoCount, columnTypes, columnIsNullableTypes);
            auto table = readXlsxFile(filename, headerPolicy, columnTypesPolicy);
            return LifetimeManager::instance().addOwnership(std::move(table));
        };
    }

    EXPORT void writeTableToXLSXFile(const char *filename, arrow::Table *table, GeneratorHeaderPolicy headerPolicy, const char **outError)
    {
        LOG("table={}, filepath={}", (void*)table, filename);
        return TRANSLATE_EXCEPTION(outError)
        {
            // NOTE: this will silently fail if the target directory does not exist
            std::ofstream out{filename, std::ios::binary};
            if(!out)
                throw std::runtime_error("Cannot write to file "s + filename);

            writeXlsx(out, *table, headerPolicy);
        };
    }

    EXPORT arrow::Table *readTableFromFeatherFile(const char *filename, const char **outError)
    {
        LOG("{}", filename);
        return TRANSLATE_EXCEPTION(outError)
        {
            auto table = loadTableFromFeatherFile(filename);
            return LifetimeManager::instance().addOwnership(std::move(table));
        };
    }

    EXPORT void writeTableToFeatherFile(const char *filename, arrow::Table *table, const char **outError)
    {
        LOG("table={}, filepath={}", (void*)table, filename);
        return TRANSLATE_EXCEPTION(outError)
        {
            saveTableToFeatherFile(filename, *table);
        };
    }
}

// RESOURCE MANAGEMENT
extern "C"
{
    EXPORT void release(void *handle) noexcept
    {
        LOG("@{}", (void*)handle);
        return TRANSLATE_EXCEPTION(nullptr)
        {
            LifetimeManager::instance().releaseOwnership(handle);
        };
    }
}
