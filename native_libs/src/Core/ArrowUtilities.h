#pragma once

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include "variant.h"

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
    using ObservedType = ValueType;
    using CType = typename BuilderType::value_type;
    using Array = arrow::NumericArray<T>;
    using StorageValueType = ValueType;
    using OffsetType = void;
    static constexpr arrow::Type::type id = ArrowType::type_id;
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
    using ObservedType = std::string_view;
    using CType = const char *;
    using Array = arrow::StringArray;
    using StorageValueType = uint8_t;
    using OffsetType = int32_t;
    static constexpr arrow::Type::type id = ArrowType::type_id;
};

template<typename Array>
using ArrayTypeDescription = TypeDescription<std::decay_t<std::remove_pointer_t<Array>>::TypeClass::type_id>;

template <typename Array>
auto arrayValueAtTyped(const Array &array, int64_t index)
{
    if constexpr(std::is_same_v<arrow::StringArray, Array>)
    {
        int32_t length = 0;
        auto ptr = array.GetValue(index, &length);
        return std::string_view(reinterpret_cast<const char *>(ptr), length);
    }
    else
        return array.Value(index);
}

template <arrow::Type::type type>
auto arrayValueAt(const arrow::Array &array, int64_t index)
{
    using Array = typename TypeDescription<type>::Array;
    return arrayValueAtTyped(static_cast<const Array &>(array), index);
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
            handleElem(arrayValueAt<type>(array, row));
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
                handleElem(arrayValueAt<type>(array, row));
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

template <arrow::Type::type type, typename ElementF, typename NullF>
void iterateOver(const arrow::Column &column, ElementF &&handleElem, NullF &&handleNull)
{
    return iterateOver<type>(*column.data(), handleElem, handleNull);
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
void iterateOverGeneric(const arrow::ChunkedArray &array, ElementF &&handleElem, NullF &&handleNull)
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
    return iterateOverGeneric(*column.data(), handleElem, handleNull);
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
auto visitArray(const arrow::Array &array, Function &&f)
{
    switch(array.type_id())
    {
    case arrow::Type::INT64 : return f(staticDowncastArray<arrow::Type::INT64 >(&array));
    case arrow::Type::DOUBLE: return f(staticDowncastArray<arrow::Type::DOUBLE>(&array));
    case arrow::Type::STRING: return f(staticDowncastArray<arrow::Type::STRING>(&array));
    default: throw std::runtime_error("array type not supported to downcast: " + array.type()->ToString());
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
std::pair<std::shared_ptr<arrow::Buffer>, T*> allocateBuffer(size_t length)
{
    std::shared_ptr<arrow::Buffer> ret{};
    checkStatus(arrow::AllocateBuffer(length * sizeof(T), &ret));
    return { ret, reinterpret_cast<T*>(ret->mutable_data()) } ;
}

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

template<typename T> struct strip_optional                                    { using type = T; };
template<typename T> struct strip_optional<std::optional<T>> : std::true_type { using type = T; };
template<typename T>
using strip_optional_t = typename strip_optional<T>::type;



template<typename T>
auto toArray(const std::vector<T> &elems)
{
    using ValueT = strip_optional_t<T>;

    using BuilderT = typename TypeDescription<ValueTypeToId<ValueT>()>::BuilderType;
    BuilderT builder;
    for(auto &&elem : elems)
    {
        if constexpr(is_optional_v<T>)
        {
            if(elem)
                builder.Append(*elem);
            else
                builder.AppendNull();
        }
        else
            builder.Append(elem);
    }

    return finish(builder);
}

template<typename T>
auto toColumn(const std::vector<T> &elems, std::string name = "col")
{
    auto array = toArray(elems);
    auto field = arrow::field(name, array->type(), is_optional_v<T>);
    return std::make_shared<arrow::Column>(field, array);
}

template<typename T>
auto scalarToColumn(const T &elem, std::string name = "col")
{
    return toColumn(std::vector<T>{elem}, std::move(name));
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

EXPORT std::vector<std::shared_ptr<arrow::Column>> getColumns(const arrow::Table &table);
EXPORT std::unordered_map<std::string, std::shared_ptr<arrow::Column>> getColumnMap(const arrow::Table &table);

struct EXPORT BitmaskGenerator
{
    uint8_t *data;
    std::shared_ptr<arrow::Buffer> buffer;
    int64_t length;

    BitmaskGenerator(int64_t length, bool initialValue);

    bool get(int64_t index);
    void set(int64_t index);
    void clear(int64_t index);
};

std::shared_ptr<arrow::Field> setNullable(bool nullable, std::shared_ptr<arrow::Field> field);
std::shared_ptr<arrow::Schema> setNullable(bool nullable, std::shared_ptr<arrow::Schema> field);

using PossiblyChunkedArray = std::variant<std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::ChunkedArray>>;

EXPORT std::shared_ptr<arrow::Table> tableFromArrays(std::vector<PossiblyChunkedArray> arrays, std::vector<std::string> names = {}, std::vector<bool> nullables = {});

using DynamicField = std::variant<int64_t, double, std::string_view, std::string, std::nullopt_t>;

using DynamicJustVector = std::variant<std::vector<int64_t>, std::vector<double>, std::vector<std::string_view>>;
EXPORT DynamicJustVector toJustVector(const arrow::ChunkedArray &chunkedArray);
EXPORT DynamicJustVector toJustVector(const arrow::Column &column);

EXPORT DynamicField arrayAt(const arrow::Array &array, int64_t index);
EXPORT DynamicField arrayAt(const arrow::ChunkedArray &array, int64_t index);
EXPORT DynamicField arrayAt(const arrow::Column &column, int64_t index);

EXPORT std::pair<std::shared_ptr<arrow::Array>, int64_t> locateChunk(const arrow::ChunkedArray &chunkedArray, int64_t index);

EXPORT std::vector<DynamicField> rowAt(const arrow::Table &table, int64_t index);

EXPORT void validateIndex(const arrow::Array &array, int64_t index);
EXPORT void validateIndex(const arrow::ChunkedArray &array, int64_t index);
EXPORT void validateIndex(const arrow::Column &column, int64_t index);


template<typename F>
auto visitType(const arrow::DataType &type, F &&f)
{
    switch(type.id())
    {
    case arrow::Type::INT64 : return f(std::integral_constant<arrow::Type::type, arrow::Type::INT64 >{});
    case arrow::Type::DOUBLE: return f(std::integral_constant<arrow::Type::type, arrow::Type::DOUBLE>{});
    case arrow::Type::STRING: return f(std::integral_constant<arrow::Type::type, arrow::Type::STRING>{});
    default: throw std::runtime_error("array type not supported to downcast: " + type.ToString());
    }
}

inline void append(arrow::StringBuilder &sb, std::string_view sv)
{
    sb.Append(sv.data(), static_cast<int32_t>(sv.length()));
}

template<typename Builder, typename T>
void append(Builder &sb, T v)
{
    sb.Append(v);
}
