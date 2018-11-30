#pragma once

#include <cassert>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include "variant.h"

#include <date/date.h>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include "Common.h"

using TypePtr = std::shared_ptr<arrow::DataType>;

namespace date
{
    class year_month_day;
}

using TimestampDuration = std::chrono::duration<int64_t, std::nano>; // nanoseconds

struct Timestamp;

namespace std
{
    DFH_EXPORT std::string to_string(const Timestamp &t);
}

// timestamp represents nanoseconds since unix epoch
struct DFH_EXPORT Timestamp : std::chrono::time_point<std::chrono::system_clock, TimestampDuration>
{
    using Base = std::chrono::time_point<std::chrono::system_clock, TimestampDuration>;
    explicit Timestamp(int64_t nanoticks) : Base(TimestampDuration(nanoticks)) {}
    Timestamp(date::year_month_day ymd);
    using Base::time_point;

    int64_t toStorage() const { return time_since_epoch().count(); }
    time_t toTimeT() const 
    {
        using namespace std::chrono;
        return system_clock::to_time_t(time_point_cast<system_clock::duration>(*this)); 
    }
    constexpr date::year_month_day ymd() const
    {
        return  { date::floor<date::days>(*this) };
    }

    friend std::ostream &operator<<(std::ostream &out, const Timestamp &t)
    {
        return out << std::to_string(t);
    }
};


template<typename T>
inline auto toStorage(const T &t) { return t; }
inline auto toStorage(const Timestamp &t) { return t.toStorage(); }

namespace std
{
    template <>
    struct hash<Timestamp>
    {
        using argument_type = Timestamp;
        using result_type = std::size_t;

        result_type operator()(const argument_type &value) const noexcept
        {
            return value.time_since_epoch().count();
        }
    };
}

template<typename T>
constexpr auto ValueTypeToId()
{
    if constexpr(std::is_same_v<T, int64_t>)
        return arrow::Type::INT64;
    else if constexpr(std::is_same_v<T, double>)
        return arrow::Type::DOUBLE;
    else if constexpr(std::is_same_v<T, std::string>)
        return arrow::Type::STRING;
    else if constexpr(std::is_same_v<T, Timestamp>)
        return arrow::Type::TIMESTAMP;
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
    using IntervalType = ValueType;
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

struct ListElemView
{
    arrow::Array *array{};
    int32_t offset{};
    int32_t length{};

    ListElemView(arrow::Array *array, int32_t offset, int32_t length)
        : array(array), offset(offset), length(length)
    {}
};

template<> struct TypeDescription<arrow::Type::TIMESTAMP>
{
    using ArrowType = arrow::TimestampType;
    using BuilderType = arrow::TimestampBuilder;
    using ValueType = Timestamp;
    using ObservedType = Timestamp;
    using CType = int64_t;
    using Array = arrow::TimestampArray;
    using StorageValueType = int64_t;
    using OffsetType = void;
    using IntervalType = TimestampDuration;
    static constexpr arrow::Type::type id = ArrowType::type_id;
};

template<> struct TypeDescription<arrow::Type::LIST>
{
    using ArrowType = arrow::ListType;
    using BuilderType = arrow::ListBuilder;
    //using ValueType = std::string;
    using ObservedType = ListElemView;
    //using CType = const char *;
    using Array = arrow::ListArray;
    // using StorageValueType = uint8_t;
    using OffsetType = int32_t;
    static constexpr arrow::Type::type id = ArrowType::type_id;
};

template<typename Array>
using ArrayTypeDescription = TypeDescription<std::decay_t<std::remove_pointer_t<Array>>::TypeClass::type_id>;

template<typename ArrowType>
using ArrowTypeDescription = TypeDescription<ArrowType::type_id>;

template<arrow::Type::type id>
using BuilderFor = typename TypeDescription<id>::BuilderType;

template<typename T>
constexpr arrow::Type::type getID(const std::shared_ptr<T> &)
{
    return T::type_id;
}

template<typename F>
auto visitDataType3(const std::shared_ptr<arrow::DataType> &type, F &&f)
{
    switch(type->id())
    {
        case arrow::Type::INT64: return f(std::static_pointer_cast<arrow::Int64Type>(type));
        case arrow::Type::DOUBLE: return f(std::static_pointer_cast<arrow::DoubleType>(type));
        case arrow::Type::STRING: return f(std::static_pointer_cast<arrow::StringType>(type));
        case arrow::Type::TIMESTAMP: return f(std::static_pointer_cast<arrow::TimestampType>(type));
        default: throw std::runtime_error("type not supported to downcast: " + type->ToString());
    }
}

template<typename F>
auto visitDataType(const std::shared_ptr<arrow::DataType> &type, F &&f)
{
    switch(type->id())
    {
    case arrow::Type::INT64: return f(std::static_pointer_cast<arrow::Int64Type>(type));
    case arrow::Type::DOUBLE: return f(std::static_pointer_cast<arrow::DoubleType>(type));
    case arrow::Type::STRING: return f(std::static_pointer_cast<arrow::StringType>(type));
    case arrow::Type::TIMESTAMP: return f(std::static_pointer_cast<arrow::TimestampType>(type));
    case arrow::Type::LIST: return f(std::static_pointer_cast<arrow::ListType>(type));
    default: throw std::runtime_error("type not supported to downcast: " + type->ToString());
    }
}

template<typename F>
auto visitType(const arrow::Type::type &id, F &&f)
{
    switch(id)
    {
    case arrow::Type::INT64 : return f(std::integral_constant<arrow::Type::type, arrow::Type::INT64 >{});
    case arrow::Type::DOUBLE: return f(std::integral_constant<arrow::Type::type, arrow::Type::DOUBLE>{});
    case arrow::Type::STRING: return f(std::integral_constant<arrow::Type::type, arrow::Type::STRING>{});
    case arrow::Type::TIMESTAMP: return f(std::integral_constant<arrow::Type::type, arrow::Type::TIMESTAMP>{});
    //case arrow::Type::LIST: return f(std::integral_constant<arrow::Type::type, arrow::Type::LIST>{});
    default: throw std::runtime_error("array type not supported to downcast: " + std::to_string((int)id));
    }
}

template<typename F>
auto visitType(const arrow::DataType &type, F &&f)
{
    switch(type.id())
    {
    case arrow::Type::INT64 : return f(std::integral_constant<arrow::Type::type, arrow::Type::INT64 >{});
    case arrow::Type::DOUBLE: return f(std::integral_constant<arrow::Type::type, arrow::Type::DOUBLE>{});
    case arrow::Type::STRING: return f(std::integral_constant<arrow::Type::type, arrow::Type::STRING>{});
    case arrow::Type::TIMESTAMP: return f(std::integral_constant<arrow::Type::type, arrow::Type::TIMESTAMP>{});
    //case arrow::Type::LIST: return f(std::integral_constant<arrow::Type::type, arrow::Type::LIST>{});
    default: throw std::runtime_error("array type not supported to downcast: " + type.ToString());
    }
}

template<typename F>
auto visitType4(const TypePtr &type, F &&f)
{
    switch(type->id())
    {
    case arrow::Type::INT64: return f(std::integral_constant<arrow::Type::type, arrow::Type::INT64 >{});
    case arrow::Type::DOUBLE: return f(std::integral_constant<arrow::Type::type, arrow::Type::DOUBLE>{});
    case arrow::Type::STRING: return f(std::integral_constant<arrow::Type::type, arrow::Type::STRING>{});
    case arrow::Type::TIMESTAMP: return f(std::integral_constant<arrow::Type::type, arrow::Type::TIMESTAMP>{});
    case arrow::Type::LIST: return f(std::integral_constant<arrow::Type::type, arrow::Type::LIST>{});
    default: throw std::runtime_error("array type not supported to downcast: " + type->ToString());
    }
}

template <typename Array>
auto arrayValueAtTyped(const Array &array, int32_t index)
{
    if constexpr(std::is_same_v<arrow::StringArray, Array>)
    {
        int32_t length = 0;
        auto ptr = array.GetValue(index, &length);
        return std::string_view(reinterpret_cast<const char *>(ptr), length);
    }
    else if constexpr(std::is_same_v<arrow::ListArray, Array>)
    {
        const auto length = array.value_length(index);
        const auto offset = array.value_offset(index);
        return ListElemView{ array.values().get(), offset, length };
    }
    else if constexpr(std::is_same_v<arrow::TimestampArray, Array>)
    {
        return Timestamp(TimestampDuration(array.Value(index)));
    }
    else
        return array.Value(index);
}

//////////////////////////////////////////////////////////////////////////


// for now we just assume that all timestamps are nanoseconds based
//DFH_EXPORT extern std::shared_ptr<arrow::TimestampType> timestampTypeSingleton;

template<arrow::Type::type id>
auto getTypeSingleton()
{
    using ArrowType = typename TypeDescription<id>::ArrowType;
    if constexpr(id == arrow::Type::TIMESTAMP)
    {
        // TODO: be smarter
        return std::make_shared<arrow::TimestampType>(arrow::TimeUnit::NANO);
    }
    else
        return std::static_pointer_cast<ArrowType>(arrow::TypeTraits<ArrowType>::type_singleton());
}

//////////////////////////////////////////////////////////////////////////

DFH_EXPORT std::shared_ptr<arrow::ArrayBuilder> makeBuilder(const TypePtr &type);

template<typename TypeT>
auto makeBuilder(const std::shared_ptr<TypeT> &type)
{
    using TT = arrow::TypeTraits<TypeT>;
    using Builder = typename TT::BuilderType;
    constexpr auto id = TypeT::type_id;
    if constexpr(id == arrow::Type::LIST)
    {
        if(type->num_children() != 1)
            throw std::runtime_error("list type must have a single child type");

        // list builder must additionally take a nested builder for a value buffer
        auto nestedBuilder = makeBuilder(type->child(0)->type());
        return std::make_shared<Builder>(nullptr, nestedBuilder);
    }
    else if constexpr(TT::is_parameter_free)
    {
        return std::make_shared<Builder>();
    }
    else
    {
        return std::make_shared<Builder>(type, arrow::default_memory_pool());
    }
}

inline auto append(arrow::StringBuilder &builder, std::string_view sv)
{
    return builder.Append(sv.data(), static_cast<int32_t>(sv.length()));
}

template<typename N, typename V>
auto append(arrow::NumericBuilder<N> &builder, const V &value, std::enable_if_t<std::is_arithmetic_v<V>> * = nullptr)
{
    return builder.Append(value);
}

inline auto append(arrow::TimestampBuilder &builder, const Timestamp &value)
{
    // TODO support other units than nanoseconds
    assert(std::dynamic_pointer_cast<arrow::TimestampType>(builder.type())->unit() == arrow::TimeUnit::NANO);
    static_assert(std::is_same_v<Timestamp::period, std::nano>);
    return builder.Append(value.time_since_epoch().count());
}

template<typename Builder, typename T>
inline auto append(Builder &builder, const std::optional<T> &value)
{
    if(value)
        append(builder, *value);
    else
        builder.AppendNull();
}


//////////////////////////////////////////////////////////////////////////

template <arrow::Type::type type>
auto arrayValueAt(const arrow::Array &array, int32_t index)
{
    using Array = typename TypeDescription<type>::Array;
    return arrayValueAtTyped(static_cast<const Array &>(array), index);
}
template <arrow::Type::type type>
auto tryArrayValueAt(const arrow::Array &array, int32_t index)
{
    using T = typename TypeDescription<type>::ObservedType;
    if(array.IsValid(index))
    {
        using Array = typename TypeDescription<type>::Array;
        return std::optional<T>(arrayValueAtTyped(static_cast<const Array &>(array), index));
    }
    return std::optional<T>{};
}

template <arrow::Type::type type, typename ElementF, typename NullF>
void iterateOver(const arrow::Array &array, ElementF &&handleElem, NullF &&handleNull)
{
    const auto N = static_cast<int32_t>(array.length());
    const auto nullCount = array.null_count();
    //const auto nullBitmapData = array.null_bitmap_data();

    // special fast paths when there are no nulls or array is all nulls
    if(nullCount == 0)
    {
        for(int32_t row = 0; row < N; row++)
            handleElem(arrayValueAt<type>(array, row));
    }
    else if(nullCount == N)
    {
        for(int32_t row = 0; row < N; row++)
            handleNull();
    }
    else
    {
        for(int32_t row = 0; row < N; row++)
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
    return visitType4(array.type(), [&] (auto id) { return iterateOver<id.value>(array, handleElem, handleNull); });
}

template <typename ElementF, typename NullF>
void iterateOverGeneric(const arrow::ChunkedArray &array, ElementF &&handleElem, NullF &&handleNull)
{
    return visitType(*array.type(), [&] (auto id) { return iterateOver<id.value>(array, handleElem, handleNull); });
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
    return visitType(*array.type(), [&] (auto id)
    {
        return f(staticDowncastArray<id.value>(&array));
    });
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
void toVector(std::vector<T> &out, const arrow::Array &array);

template<typename T>
void toVector(std::vector<std::vector<T>> &out, const arrow::Array &array)
{
    if(array.type_id() != arrow::Type::LIST)
        throw std::runtime_error(std::string("Type mismatch: expected `list` got " + array.type()->ToString()));

    iterateOver<arrow::Type::LIST>(array, [&](ListElemView elem)
        {
            std::vector<T> nestedOut;
            nestedOut.reserve(elem.length);
            toVector(nestedOut, *elem.array->Slice(elem.offset, elem.length));
            out.push_back(std::move(nestedOut));
        },
        [&] { out.push_back({}); }
    );
}

template<typename T>
void toVector(std::vector<T> &out, const arrow::Array &array)
{
    iterateOverGeneric
    (
        array, 
        [&] (auto &&elem)
        {
            using ElemT = decltype(elem);
            if constexpr(std::is_constructible_v<T, ElemT>)
                out.push_back(T(std::forward<decltype(elem)>(elem)));
            else
                throw std::runtime_error(std::string("Type mismatch: expected ") + typeid(T).name() + " got " + typeid(elem).name());
        },
        [&] { out.push_back({}); }
    );
}

template<typename ArrowSth>
int64_t count(const ArrowSth &sth, bool withNull)
{
    if(withNull)
        return sth.length();
    else
        return sth.length() - sth.null_count();
}

template<typename T>
std::vector<T> toVector(const arrow::Array &array)
{
    std::vector<T> ret;
    ret.reserve(count(array, is_optional_v<T>));
    toVector(ret, array);
    return ret;
}

template<typename T>
std::vector<T> toVector(const arrow::ChunkedArray &array)
{
    std::vector<T> ret;
    ret.reserve(count(array, is_optional_v<T>));
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
std::shared_ptr<arrow::Array> toArray(const std::vector<T> &elems)
{
    using ValueT = strip_optional_t<T>;
    using TD = TypeDescription<ValueTypeToId<ValueT>()>;
    using BuilderT = typename TD::BuilderType;
    auto builder = makeBuilder(getTypeSingleton<TD::id>());
    builder->Reserve(elems.size());
    for(auto &&elem : elems)
    {
        if constexpr(is_optional_v<T>)
        {
            if(elem)
                append(*builder, *elem);
            else
                builder->AppendNull();
        }
        else
            append(*builder, elem);
    }

    return finish(*builder);
}

template<typename T>
auto toColumn(const std::vector<T> &elems, std::string name = "col")
{
    auto array = toArray(elems);
    auto field = arrow::field(name, array->type(), is_optional_v<T>);
    return std::make_shared<arrow::Column>(field, array);
}

DFH_EXPORT std::shared_ptr<arrow::Column> toColumn(std::shared_ptr<arrow::ChunkedArray> chunks, std::string name = "col");
DFH_EXPORT std::shared_ptr<arrow::Column> toColumn(std::shared_ptr<arrow::Array> array, std::string name = "col");

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

DFH_EXPORT std::shared_ptr<arrow::Column> getColumn(const arrow::Table &table, std::string_view name);
DFH_EXPORT std::vector<std::shared_ptr<arrow::Column>> getColumns(const arrow::Table &table);
DFH_EXPORT std::unordered_map<std::string, std::shared_ptr<arrow::Column>> getColumnMap(const arrow::Table &table);

struct DFH_EXPORT BitmaskGenerator
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

using PossiblyChunkedArray = variant<std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::ChunkedArray>>;

DFH_EXPORT std::shared_ptr<arrow::Table> tableFromArrays(std::vector<PossiblyChunkedArray> arrays, std::vector<std::string> names = {}, std::vector<bool> nullables = {});
DFH_EXPORT std::shared_ptr<arrow::Table> tableFromColumns(const std::vector<std::shared_ptr<arrow::Column>> &columns, const std::shared_ptr<arrow::Schema> &schema);
DFH_EXPORT std::shared_ptr<arrow::Table> tableFromColumns(const std::vector<std::shared_ptr<arrow::Column>> &columns);

template<typename ...Ts>
std::shared_ptr<arrow::Table> tableFromVectors(const std::vector<Ts> & ...ts)
{
    return tableFromArrays({toArray(ts)...});
}

using DynamicField = variant<int64_t, double, std::string_view, std::string, ListElemView, Timestamp, TimestampDuration, std::nullopt_t>;

using DynamicJustVector = variant<std::vector<int64_t>, std::vector<double>, std::vector<std::string_view>, std::vector<ListElemView>, std::vector<Timestamp>>;
DFH_EXPORT DynamicJustVector toJustVector(const arrow::ChunkedArray &chunkedArray);
DFH_EXPORT DynamicJustVector toJustVector(const arrow::Column &column);

DFH_EXPORT DynamicField arrayAt(const arrow::Array &array, int64_t index);
DFH_EXPORT DynamicField arrayAt(const arrow::ChunkedArray &array, int64_t index);
DFH_EXPORT DynamicField arrayAt(const arrow::Column &column, int64_t index);

struct DFH_EXPORT ChunkAccessor
{
    std::vector<std::shared_ptr<arrow::Array>> chunks;
    std::vector<int64_t> chunkStartIndices;

    ChunkAccessor(const arrow::ChunkedArray &array);
    ChunkAccessor(const arrow::Column &column);
    std::pair<const arrow::Array *, int32_t> locate(int64_t index) const;

    template <arrow::Type::type type>
    auto valueAt(int64_t index) const
    {
        auto [chunk, chunkIndex] = locate(index);
        return arrayValueAt<type>(*chunk, chunkIndex);
    }

    bool isNull(int64_t index);
};

DFH_EXPORT std::pair<std::shared_ptr<arrow::Array>, int32_t> locateChunk(const arrow::ChunkedArray &chunkedArray, int64_t index);

template <arrow::Type::type type>
auto columnValueAt(const arrow::Column &column, int64_t index)
{
	auto [chunk, chunkIndex] = locateChunk(*column.data(), index);
	return arrayValueAt<type>(*chunk, chunkIndex);
}

template <arrow::Type::type type>
auto tryColumnValueAt(const arrow::Column &column, int64_t index)
{
    auto[chunk, chunkIndex] = locateChunk(*column.data(), index);
    return tryArrayValueAt<type>(*chunk, chunkIndex);
}

DFH_EXPORT std::vector<DynamicField> rowAt(const arrow::Table &table, int64_t index);

DFH_EXPORT void validateIndex(const arrow::Array &array, int64_t index);
DFH_EXPORT void validateIndex(const arrow::ChunkedArray &array, int64_t index);
DFH_EXPORT void validateIndex(const arrow::Column &column, int64_t index);

template<arrow::Type::type id1, arrow::Type::type id2, typename F>
void iterateOverJustPairs(const arrow::ChunkedArray &array1, const arrow::ChunkedArray &array2, F &&f)
{
    assert(array1.length() == array2.length());
    const auto N = array1.length();

    auto chunks1Itr = array1.chunks().begin();
    auto chunks2Itr = array2.chunks().begin();

    int64_t row = 0;

    int32_t chunk1Length = (int32_t) (*chunks1Itr)->length(); // arrow specification says that array size is 32-bit
    int32_t chunk2Length = (int32_t) (*chunks2Itr)->length(); // arrow specification says that array size is 32-bit

    int32_t index1 = -1, index2 = -1;
    for( ; row < N; row++)
    {
        if(++index1 >= chunk1Length)
        {
            ++chunks1Itr;
            chunk1Length = (int32_t)(*chunks1Itr)->length();
            index1 = 0;
        }
        if(++index2 >= chunk2Length)
        {
            ++chunks2Itr;
            chunk2Length = (int32_t)(*chunks2Itr)->length();
            index2 = 0;
        }

        if((*chunks1Itr)->IsValid(index1))
            if((*chunks2Itr)->IsValid(index2))
                f(arrayValueAt<id1>(**chunks1Itr, index1), arrayValueAt<id2>(**chunks2Itr, index2));
    }
}

template<arrow::Type::type id1, arrow::Type::type id2, typename F>
void iterateOverJustPairs(const arrow::Column &column1, const arrow::Column &column2, F &&f)
{
    return iterateOverJustPairs<id1, id2>(*column1.data(), *column2.data(), f);
}


DFH_EXPORT std::shared_ptr<arrow::Array> makeNullsArray(TypePtr type, int64_t length);


template<arrow::Type::type id, bool nullable>
struct FixedSizeArrayBuilder
{
    using T = typename TypeDescription<id>::StorageValueType;
    using Array = typename TypeDescription<id>::Array;

    std::shared_ptr<arrow::DataType> type;
    int64_t length;
    std::shared_ptr<arrow::Buffer> valueBuffer;
    T *nextValueToWrite{};


    FixedSizeArrayBuilder(std::shared_ptr<arrow::DataType> type, int32_t length)
        : type(std::move(type))
        , length(length)
    {
        std::tie(valueBuffer, nextValueToWrite) = allocateBuffer<T>(length);

        static_assert(nullable == false); // would need null mask
        static_assert(id == arrow::Type::INT64 || arrow::Type::DOUBLE); // would need another buffer
    }

    explicit FixedSizeArrayBuilder(int32_t length)
        : FixedSizeArrayBuilder(getTypeSingleton<id>(), length)
    {
        // TODO: should eventually require that type is parameter-free
    }

    void Append(T value)
    {
        *nextValueToWrite++ = value;
    }

    auto Finish()
    {
        return std::make_shared<Array>(type, length, valueBuffer, nullptr, 0);
    }
};


#define MAKE_INTEGRAL_CONSTANT(value) std::integral_constant<decltype(value), value>{} 
#define CASE_DISPATCH(value) case value: return f(MAKE_INTEGRAL_CONSTANT(value));

template<typename F>
auto dispatch(bool value, F &&f)
{
    if(value)
        return f(MAKE_INTEGRAL_CONSTANT(true));
    else
        return f(MAKE_INTEGRAL_CONSTANT(false));
}


template<typename ArrowDataTypePtr>
using ArrowTypeFromPtr = typename std::decay_t<ArrowDataTypePtr>::element_type;

template<typename ArrowDataTypePtr>
constexpr arrow::Type::type idFromDataPointer = std::decay_t<ArrowDataTypePtr>::element_type::type_id;

DFH_EXPORT std::shared_ptr<arrow::Column> consolidate(std::shared_ptr<arrow::Column> column);
