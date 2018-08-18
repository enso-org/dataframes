#pragma once

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include "Common.h"

template<typename T>
constexpr auto ValueTypeToId()
{
    if constexpr(std::is_same_v<T, int64_t>)
        return arrow::Type::INT64;
    else if constexpr(std::is_same_v<T, double>)
        return arrow::Type::DOUBLE;
    else if constexpr(std::is_same_v<T, std::string>)
        return arrow::Type::STRING;
    else
        static_assert(always_false_v<T>);
}

template<arrow::Type::type type>
struct TypeDescription
{};

template<typename T>
struct NumericTypeDescription
{
    using ArrowType = T;
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
    using ArrowType = arrow::StringType;
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
    const auto N = array.length();
    const auto nullCount = array.null_count();
    //const auto nullBitmapData = array.null_bitmap_data();

    // special fast paths when there are no nulls or array is all nulls
    if(nullCount == 0)
    {
        for(int64_t row = 0; row < N; row++)
            handleElem(arrayAt<type>(array, row));
    }
    else if(nullCount == N)
    {
        for(int64_t row = 0; row < N; row++)
            handleNull();
    }
    else
    {
        for(int64_t row = 0; row < N; row++)
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
}

template <arrow::Type::type type, typename ElementF, typename NullF>
void iterateOver(const arrow::ChunkedArray &arrays, ElementF &&handleElem, NullF &&handleNull)
{
    assert(type == arrays.type()->id());
    for(auto &chunk : arrays.chunks())
    {
        iterateOver<type>(*chunk, handleElem, handleNull);
    }
}

template <typename ElementF, typename NullF>
void iterateOverGeneric(const arrow::Array &array, ElementF &&handleElem, NullF &&handleNull)
{
    const auto t = array.type();
    switch(t->id())
    {
    case arrow::Type::INT64 : return iterateOver<arrow::Type::INT64 >(array, handleElem, handleNull);
    case arrow::Type::DOUBLE: return iterateOver<arrow::Type::DOUBLE>(array, handleElem, handleNull);
    case arrow::Type::STRING: return iterateOver<arrow::Type::STRING>(array, handleElem, handleNull);
    default                 : throw  std::runtime_error(__FUNCTION__ + std::string(": not supported array type ") + t->ToString());
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

// downcasts array to the relevant type, throwing if type mismatch
template<arrow::Type::type type>
auto staticDowncastArray(const arrow::Array *array)
{
    return static_cast<const typename TypeDescription<type>::Array *>(array);
}

template<typename Function>
auto visitArray(const arrow::Array *array, Function &&f)
{
    assert(array);
    switch(array->type_id())
    {
    case arrow::Type::INT64 : return f(staticDowncastArray<arrow::Type::INT64 >(array));
    case arrow::Type::DOUBLE: return f(staticDowncastArray<arrow::Type::DOUBLE>(array));
    case arrow::Type::STRING: return f(staticDowncastArray<arrow::Type::STRING>(array));
    default: throw std::runtime_error("array type not supported to downcast: " + array->type()->ToString());
    }
}

inline std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder &builder)
{
    std::shared_ptr<arrow::Array> ret;
    auto status = builder.Finish(&ret);
    if(!status.ok())
        throw std::runtime_error(status.ToString());

    return ret;
}

template<typename T>
std::shared_ptr<arrow::Buffer> allocateBuffer(size_t length)
{
    std::shared_ptr<arrow::Buffer> ret{};
    checkStatus(arrow::AllocateBuffer(length * sizeof(T), &ret));
    return ret;
}

template<typename Array>
using ArrayTypeDescription = TypeDescription<std::remove_pointer_t<Array>::TypeClass::type_id>;

template<typename T>
void toVector(std::vector<T> &out, const arrow::Array &array)
{
    iterateOverGeneric
    (
        array, 
        [&] (auto &&elem)
        {
            if constexpr(std::is_convertible_v<decltype(elem), T>)
                out.push_back(std::forward<decltype(elem)>(elem));
            else
                throw std::runtime_error(std::string("Type mismatch: expected ") + typeid(T).name() + " got " + typeid(elem).name());
        },
        [&] { out.push_back({}); }
    );
}

template<typename T>
std::vector<T> toVector(const arrow::Array &array)
{
    std::vector<T> ret;
    toVector(ret, array);
    return ret;
}

template<typename T>
std::vector<T> toVector(const arrow::ChunkedArray &array)
{
    std::vector<T> ret;
    for(auto &&chunk : array.chunks())
        toVector(ret, *chunk);
    return ret;
}

template<typename T>
std::vector<T> toVector(const arrow::Column &array)
{
    return toVector<T>(*array.data());
}

template<typename T>
auto toArray(const std::vector<T> &elems)
{
    using BuilderT = typename TypeDescription<ValueTypeToId<T>()>::BuilderType;
    BuilderT builder;
    for(auto &&elem : elems)
        builder.Append(elem);

    return finish(builder);
}

namespace detail
{
    template<typename T, std::size_t N>
    auto nthColumnToVector(const arrow::Table &table)
    {
        return toVector<T>(*table.column(N));
    }

    template<typename ...Ts, std::size_t... N>
    auto toVectorsHlp(const arrow::Table &table, std::index_sequence<N...>)
    {
        using Ret = std::tuple<Ts...>;
        return std::make_tuple(detail::nthColumnToVector<std::tuple_element_t<N, Ret>, N>(table)...);
    }
}

template<typename ...Ts>
std::tuple<std::vector<Ts>...> toVectors(const arrow::Table &table)
{
    if(sizeof...(Ts) > table.num_columns())
        throw std::runtime_error("Table does not contain required column count!");

    return detail::toVectorsHlp<Ts...>(table, std::index_sequence_for<Ts...>{});
}

struct EXPORT BitmaskGenerator
{
    uint8_t *data;
    std::shared_ptr<arrow::Buffer> buffer;
    int64_t length;

    BitmaskGenerator(int64_t length, bool initialValue);

    void set(int64_t index);
    void clear(int64_t index);
};

EXPORT std::shared_ptr<arrow::Table> tableFromArrays(std::vector<std::shared_ptr<arrow::Array>> arrays, std::vector<std::string> names = {}, std::vector<bool> nullables = {});
