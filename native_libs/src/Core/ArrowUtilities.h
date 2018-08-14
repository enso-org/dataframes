#pragma once

#include <stdexcept>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/type.h>

template<arrow::Type::type type>
struct TypeDescription
{};

template<typename T>
struct NumericTypeDescription
{
    using BuilderType = arrow::NumericBuilder<T>;
    using ValueType = typename BuilderType::value_type;
    using CType = typename BuilderType::value_type;
    using Array = arrow::NumericArray<T>;
};

template<> struct TypeDescription<arrow::Type::UINT8 > : NumericTypeDescription<arrow::UInt8Type>  {};
template<> struct TypeDescription<arrow::Type::UINT16> : NumericTypeDescription<arrow::UInt16Type> {};
template<> struct TypeDescription<arrow::Type::UINT32> : NumericTypeDescription<arrow::UInt32Type> {};
template<> struct TypeDescription<arrow::Type::UINT64> : NumericTypeDescription<arrow::UInt64Type> {};
template<> struct TypeDescription<arrow::Type::INT8  > : NumericTypeDescription<arrow::Int8Type>   {};
template<> struct TypeDescription<arrow::Type::INT16 > : NumericTypeDescription<arrow::Int16Type>  {};
template<> struct TypeDescription<arrow::Type::INT32 > : NumericTypeDescription<arrow::Int32Type>  {};
template<> struct TypeDescription<arrow::Type::INT64 > : NumericTypeDescription<arrow::Int64Type>  {};
template<> struct TypeDescription<arrow::Type::FLOAT > : NumericTypeDescription<arrow::FloatType>  {};
template<> struct TypeDescription<arrow::Type::DOUBLE> : NumericTypeDescription<arrow::DoubleType> {};

template<> struct TypeDescription<arrow::Type::STRING>
{
    using BuilderType = arrow::StringBuilder;
    using ValueType = std::string;
    using CType = const char *;
    using Array = arrow::StringArray;
};

template <arrow::Type::type type>
auto arrayAt(const arrow::Array &array, int64_t index)
{
    const auto &arr = static_cast<const typename TypeDescription<type>::Array &>(array);
    if constexpr(type == arrow::Type::STRING)
        return arr.GetString(index);
    else
        return arr.Value(index);
}

template <arrow::Type::type type, typename ElementF, typename NullF>
void iterateOver(const arrow::Array &array, ElementF &&handleElem, NullF &&handleNull)
{
    for(int64_t row = 0; row < array.length(); row++)
    {
        if(!array.IsNull(row))
        {
            handleElem(arrayAt<type>(array, row));
        }
        else
        {
            handleNull();
        }
    }
}

template <arrow::Type::type type, typename ElementF, typename NullF>
void iterateOver(const arrow::ChunkedArray &arrays, ElementF &&handleElem, NullF &&handleNull)
{
    for(auto &chunk : arrays.chunks())
    {
        iterateOver<type>(*chunk, handleElem, handleNull);
    }
}

template <typename ElementF, typename NullF>
void iterateOverGeneric(const arrow::Column &column, ElementF &&handleElem, NullF &&handleNull)
{
    const auto t = column.field()->type();
    switch(t->id())
    {
    case arrow::Type::INT64 : return iterateOver<arrow::Type::INT64 >(*column.data(), handleElem, handleNull);
    case arrow::Type::DOUBLE: return iterateOver<arrow::Type::DOUBLE>(*column.data(), handleElem, handleNull);
    case arrow::Type::STRING: return iterateOver<arrow::Type::STRING>(*column.data(), handleElem, handleNull);
    default                 : throw  std::runtime_error(__FUNCTION__ + std::string(": not supported array type ") + t->ToString());
    }
}

inline void checkStatus(const arrow::Status &status)
{
    if(!status.ok())
        throw std::runtime_error(status.ToString());
}

template<typename To, typename From>
To throwingCast(From *from)
{
    if(auto ret = dynamic_cast<To>(from))
        return ret;

    std::ostringstream out;    
    out << "Failed to cast " << from;
    if(from) // we can obtain RTTI typename for non-null pointers
        out << " being " << typeid(*from).name();

    out << " to " << typeid(std::remove_pointer_t<To>).name();

    throw std::runtime_error(out.str());
}

// downcasts array to the relevant type, throwing if type mismatch
template<arrow::Type::type type>
auto throwingDowncastArray(arrow::Array *array)
{
    return throwingCast<typename TypeDescription<type>::Array *>(array);
}
