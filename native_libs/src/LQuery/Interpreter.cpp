#include "Interpreter.h"

#include <string_view>
#include <arrow/buffer.h>
#include <arrow/table.h>
#include <regex>

#include "Core/ArrowUtilities.h"
#include "AST.h"
#include "Core/Common.h"

using namespace std::literals;

template<typename T> struct StorageToArrowType {};
template<> struct StorageToArrowType<bool> { using type = arrow::BooleanType; };
template<> struct StorageToArrowType<int64_t> { using type = arrow::Int64Type; };
template<> struct StorageToArrowType<double> { using type = arrow::DoubleType; };
template<> struct StorageToArrowType<std::string> { using type = arrow::StringType; };
template<> struct StorageToArrowType<ListElemView> { using type = arrow::ListType; };

template<typename T> using StorageToArrowType_t = typename StorageToArrowType<T>::type;

template<typename ArrowType>
struct Scalar
{
    static constexpr arrow::Type::type id = ArrowType::type_id;
    using T = typename TypeDescription<id>::ValueType;

    T value;
    Scalar(T value) : value(value) {}
};

template<class T> 
Scalar(T)  -> Scalar<StorageToArrowType_t<T>>;  // #4

template<typename>
struct is_scalar : std::false_type {};
template<typename T>
struct is_scalar<Scalar<T>> : std::true_type {};
template<typename T>
constexpr bool is_scalar_v = is_optional<T>::value;

template<typename T>
struct FixedSizeValueWriter
{
    std::shared_ptr<arrow::Buffer> buffer;

    auto mutable_data() { return reinterpret_cast<T *>(buffer->mutable_data()); }
    auto data() const { return reinterpret_cast<const T*>(buffer->data()); }

    explicit FixedSizeValueWriter(std::shared_ptr<arrow::Buffer> buffer)
        : buffer(std::move(buffer))
    {}
    explicit FixedSizeValueWriter(size_t length)
        : FixedSizeValueWriter(allocateBuffer<T>(length).first)
    {}
    void store(size_t index, T value)
    {
        mutable_data()[index] = value;
    }
};
template<>
struct FixedSizeValueWriter<bool> : FixedSizeValueWriter<uint8_t>
{
    FixedSizeValueWriter(size_t length)
        : FixedSizeValueWriter<uint8_t>(arrow::BitUtil::BytesForBits(length))
    {}
    explicit FixedSizeValueWriter(std::shared_ptr<arrow::Buffer> buffer)
        : FixedSizeValueWriter<uint8_t>(std::move(buffer))
    {}
    void store(size_t index, bool value)
    {
        if(value)
            arrow::BitUtil::SetBit(mutable_data(), index);
        else
            arrow::BitUtil::ClearBit(mutable_data(), index);
    }
};

//namespace
//{

#define COMPLAIN_ABOUT_OPERAND_TYPES \
        throw std::runtime_error(__FUNCTION__ + ": not supported operand types: "s + typeid(lhs).name() + " and "s + typeid(rhs).name());

#define BINARY_REL_OPERATOR(op)                                                                          \
    template<typename Lhs, typename Rhs>                                                                 \
    static bool exec(const Lhs &lhs, const Rhs &rhs)                                                     \
    { /* below we protect against mixed types like int/string (eg. for ==)  */                           \
        if constexpr(std::is_same_v<Lhs, Rhs> || std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>) \
            return lhs op rhs;                                                                           \
        else                                                                                             \
        {                                                                                                \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
            return {}; /* just for type inference  */                                                    \
        }                                                                                                \
    }

#define BINARY_ARIT_OPERATOR(op)                                                                         \
    template<typename Lhs, typename Rhs>                                                                 \
    static auto exec(const Lhs &lhs, const Rhs &rhs)                                                     \
    {                                                                                                    \
        if constexpr(std::is_same_v<Lhs, Rhs> || std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>) \
            return lhs op rhs;                                                                           \
        else                                                                                             \
        {                                                                                                \
    COMPLAIN_ABOUT_OPERAND_TYPES;                                                                        \
        return lhs; /* just for type inference  */                                                       \
        }                                                                                                \
    }
#define FAIL_ON_STRING(ret)                                                                              \
    static ret exec(const std::string &lhs, const std::string &rhs)                                      \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    static ret exec(const std::string_view &lhs, const std::string_view &rhs)                            \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    template<typename Rhs>                                                                               \
    static ret exec(const std::string_view &lhs, const Rhs &rhs)                                         \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    template<typename Lhs>                                                                               \
    static ret exec(const Lhs &lhs, const std::string_view &rhs)                                         \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    } 

#define FAIL_ON_LIST(ret)                                                                                \
    static ret exec(const ListElemView &lhs, const ListElemView &rhs)                                    \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }  
    struct GreaterThan { BINARY_REL_OPERATOR(>); FAIL_ON_STRING(bool); FAIL_ON_LIST(bool); };
    struct LessThan    { BINARY_REL_OPERATOR(<); FAIL_ON_STRING(bool); FAIL_ON_LIST(bool); };
    struct EqualTo
    {
        BINARY_REL_OPERATOR(==);
        static bool exec(const ListElemView &lhs, const ListElemView &rhs)
        {
            if(!lhs.array->type()->Equals(rhs.array->type()))
                return false;
            if(lhs.length != rhs.length)
                return false;
            
            return lhs.toArray()->Equals(rhs.toArray());
        }
    };
    struct StartsWith
    {
        static bool exec(const std::string_view &lhs, const std::string_view &rhs)
        {
            return lhs.length() >= rhs.length()
                && std::memcmp(lhs.data(), rhs.data(), rhs.length()) == 0;
        }

        template<typename Lhs, typename Rhs>
        static bool exec(const Lhs &lhs, const Rhs &rhs)
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
        }
    };
    struct Matches
    {
        static bool exec(const std::string_view &lhs, const std::string_view &rhs)
        {
            std::regex regex{std::string(rhs)};
            return std::regex_match(lhs.begin(), lhs.end(), regex);
        }

        template<typename Lhs, typename Rhs>
        static bool exec(const Lhs &lhs, const Rhs &rhs)
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
        }
    };

    struct Plus
    {
        template<typename Lhs, typename Rhs>
        static auto exec(const Lhs &lhs, const Rhs &rhs)
        {
            if constexpr(std::is_same_v<Lhs, Rhs> || std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>)
                return lhs + rhs;
            else
            {
                COMPLAIN_ABOUT_OPERAND_TYPES;
                return lhs; /* just for type inference  */
            }
        }
    };

    //struct Plus        { BINARY_ARIT_OPERATOR(+); FAIL_ON_STRING(int64_t); FAIL_ON_LIST(int64_t); };
    struct Minus       { BINARY_ARIT_OPERATOR(-); FAIL_ON_STRING(int64_t); FAIL_ON_LIST(int64_t); };
    struct Times       { BINARY_ARIT_OPERATOR(*); FAIL_ON_STRING(int64_t); FAIL_ON_LIST(int64_t); };
    struct Divide      { BINARY_ARIT_OPERATOR(/); FAIL_ON_STRING(int64_t); FAIL_ON_LIST(int64_t); };
    struct Modulo      
    {
        static constexpr int64_t exec(const int64_t &lhs, const int64_t &rhs)
        {
            return lhs % rhs;
        }
        static double exec(const double &lhs, const double &rhs)
        {
            return std::fmod(lhs, rhs);
        }
        template<typename Lhs, typename Rhs>
        static int64_t exec(const Lhs &lhs, const Rhs &rhs)
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
        }
    };
    struct Negate
    {
        template<typename Lhs>
        static constexpr Lhs exec(const Lhs &lhs)
        {
            if constexpr(std::is_arithmetic_v<Lhs>)
                return -lhs;
            else
                throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
        }

        static int64_t exec(const std::string_view &lhs)
        {
            throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
        }
    };
    struct Mean
    {
        template<typename Lhs>
        static double exec(const Lhs &lhs)
        {
            throw std::runtime_error("not implemented");
        }
    };
    struct Condition
    {
        template<typename A, typename B>
        using Ret = std::conditional_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>,
            std::common_type_t<A, B>,
            A>;

        template<typename Lhs, typename Rhs>
        static auto exec(const bool &mask, const Lhs &lhs, const Rhs &rhs)
        {
            if constexpr(std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>)
                return mask ? lhs : rhs;
            else if constexpr(std::is_same_v<Lhs, Rhs> && std::is_same_v<Lhs, std::string_view>)
                return std::string(mask ? lhs : rhs);
            else
            {
                COMPLAIN_ABOUT_OPERAND_TYPES;
                return int64_t{}; // to deduct type
            }
        }

        // template<typename Mask, typename Lhs, typename Rhs>
        // static Ret<Lhs, Rhs> exec(const Mask &mask, const Lhs &lhs, const Rhs &rhs)
        // {
        //     throw std::runtime_error("condition operator got unexpected condition type: "s + typeid(Mask).name());
        // }
    };
    struct And
    {
        static bool exec(const bool &lhs, const bool &rhs)
        {
            return lhs && rhs;
        }

        template<typename Lhs, typename Rhs>
        static bool exec(const Lhs &lhs, const Rhs &rhs)
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
        }
    };
    struct Or
    {
        static bool exec(const bool &lhs, const bool &rhs)
        {
            return lhs || rhs;
        }

        template<typename Lhs, typename Rhs>
        static bool exec(const Lhs &lhs, const Rhs &rhs)
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
        }
    };
    struct Not
    {
        static bool exec(const bool &lhs)
        {
            return !lhs;
        }

        template<typename Lhs>
        static bool exec(const Lhs &lhs)
        {
            throw std::runtime_error("Not: wrong operand type "s + typeid(Lhs).name());
        }
    };


    template<typename T>
    auto getValue(const Scalar<T> &src, int64_t)
    {
        return src.value;
    }
    template<typename Array>
    auto getValue(const std::shared_ptr<Array> &src, int64_t index)
    {
        return arrayValueAtTyped(*src, static_cast<int32_t>(index)); // TODO to be checked when chunks are supported
    }

    template<typename Operation, typename ... Operands>
    auto exec(int64_t count, const Operands & ...operands)
    {
        // Operation between scalars shall yield a scalar
        if constexpr((is_scalar_v<Operands> && ...))
        {
            return Operation::exec(operands...);
        }
        else
        {
            using OperationResult = decltype(Operation::exec(getValue(operands, 0)...));
            using ResultArrowType = StorageToArrowType_t<OperationResult>;
            using Builder = typename arrow::TypeTraits<ResultArrowType>::BuilderType;
            using Array = typename arrow::TypeTraits<ResultArrowType>::ArrayType;

//             ArrayOperand<ResultArrowType> ret{ (size_t)count };
//             for(int64_t i = 0; i < count; i++)
//             {
//                 auto result = Operation::exec(getValue(operands, i)...);
//                 ret.store(i, result);
//             }
//             return ret;
// 
//             std::shared_ptr<arrow::Array> ret;
// 
//             throw std::runtime_error("not implemented");
//             return ret;
            Builder b;
            auto ret = finish(b);
            return staticDowncastArray<ResultArrowType::type_id>(ret);
            //return std::shared_ptr<arrow::Int64Array>{};
        }
    }

struct Interpreter
{
    Interpreter(const arrow::Table &table, const ColumnMapping &mapping)
        : table(table)
    {
        for(int i = 0; i < mapping.size(); i++)
            columns.push_back(table.column(mapping.at(i)));
    }

    const arrow::Table &table;
    std::vector<std::shared_ptr<arrow::Column>> columns;

    using Field = std::variant
        < Scalar<arrow::Int64Type>
        , Scalar<arrow::DoubleType>
        , Scalar<arrow::StringType>
        , std::shared_ptr<arrow::BooleanArray>
        , std::shared_ptr<arrow::Int64Array>
        , std::shared_ptr<arrow::DoubleArray>
        , std::shared_ptr<arrow::StringArray>
        , std::shared_ptr<arrow::ListArray>
        >;


    Field fieldFromColumn(const arrow::Column &column)
    {
        const auto data = column.data();
        if(data->num_chunks() != 1)
            throw std::runtime_error("not implemented: processing of chunked arrays");

        const auto chunk = data->chunk(0);
        return visitArray(chunk, [&](auto &&array)
        {
            return Field(array);
        });
    }
    std::vector<Field> evaluateOperands(const std::vector<ast::Value> &operands)
    {
        return transformToVector(operands, 
            [this] (auto &&operand) { return evaluateValue(operand); });
    }
    std::vector<Field> evaluatePredicates(const std::vector<ast::Predicate> &operands)
    {
        return transformToVector(operands, 
            [this] (auto &&operand) { return Field(evaluate(operand)); });
    }

    template<typename T>
    static auto getOperand(const T &t, unsigned index)
    {
        if(index < t.size())
            return t[index];

        throw std::runtime_error("failed to get operand by index " + std::to_string(index) + ": has only " + std::to_string(t.size()));
    }

    Field evaluateValue(const ast::Value &value)
    {
        return std::visit(overloaded{
            [&] (const ast::ColumnReference &col)    -> Field { return fieldFromColumn(*columns[col.columnRefId]); },
            [&] (const ast::ValueOperation &op)      -> Field 
            {
#define VALUE_UNARY_OP(opname)                                               \
            case ast::ValueOperator::opname:                                 \
                return std::visit(                                        \
                    [&] (auto &&lhs) -> Field                                \
                        { return exec<opname>(table.num_rows(), lhs);},      \
                    getOperand(operands, 0));
#define VALUE_BINARY_OP(opname)                                              \
            case ast::ValueOperator::opname:                                 \
                return std::visit(                                        \
                    [&] (auto &&lhs, auto &&rhs) -> Field                    \
                        { return exec<opname>(table.num_rows(), lhs, rhs);}, \
                    getOperand(operands, 0), getOperand(operands, 1));

                const auto operands = evaluateOperands(op.operands);
                switch(op.what)
                {
                    VALUE_BINARY_OP(Plus);
//                     VALUE_BINARY_OP(Minus);
//                     VALUE_BINARY_OP(Times);
//                     VALUE_BINARY_OP(Divide);
//                     VALUE_BINARY_OP(Modulo);
//                     VALUE_UNARY_OP(Negate);
//                     VALUE_UNARY_OP(Mean);
                default:
                    throw std::runtime_error("not implemented: value operator " + std::to_string((int)op.what));
                }
            },
            [&] (const ast::Literal<int64_t> &l)     -> Field { return Scalar(l.literal); },
            [&] (const ast::Literal<double> &l)      -> Field { return Scalar(l.literal); },
            [&] (const ast::Literal<std::string> &l) -> Field { return Scalar(l.literal); },
            [&] (const ast::Condition &condition)    -> Field 
            {
                auto mask = this->evaluate(*condition.predicate);
                auto onTrue = this->evaluateValue(*condition.onTrue);
                auto onFalse = this->evaluateValue(*condition.onFalse);
                return std::visit([&](auto &&t, auto &&f) -> Field
                {
                    return exec<Condition>(table.num_rows(), mask, t, f);
                }, onTrue, onFalse);
            },
            //[&] (const ast::Literal<std::string> &l) -> Field { return l.literal; },
            [&] (auto &&t) -> Field { throw std::runtime_error("not implemented: value node of type "s + typeid(decltype(t)).name()); }
            }, (const ast::ValueBase &) value);
    }

    std::shared_ptr<arrow::BooleanArray> evaluate(const ast::Predicate &p)
    {
        return std::visit(overloaded{
            [&] (const ast::PredicateFromValueOperation &elem) -> std::shared_ptr<arrow::BooleanArray>
        {
            const auto operands = evaluateOperands(elem.operands);
            switch(elem.what)
            {
            case ast::PredicateFromValueOperator::Greater:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<GreaterThan>(table.num_rows(), lhs, rhs);},
                    getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateFromValueOperator::Lesser:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<LessThan>(table.num_rows(), lhs, rhs);},
                    getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateFromValueOperator::Equal:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<EqualTo>(table.num_rows(), lhs, rhs);},
                    getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateFromValueOperator::StartsWith:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<StartsWith>(table.num_rows(), lhs, rhs);},
                    getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateFromValueOperator::Matches:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<Matches>(table.num_rows(), lhs, rhs);},
                    getOperand(operands, 0), getOperand(operands, 1));
            default:
                throw std::runtime_error("not implemented: predicate operator " + std::to_string((int)elem.what));
            }
        },
            [&] (const ast::PredicateOperation &op) -> std::shared_ptr<arrow::BooleanArray>
        {
            const auto operands = evaluatePredicates(op.operands);
            switch(op.what)
            {
            case ast::PredicateOperator::And:
                return std::visit(
                    [&](auto &&lhs, auto &&rhs) 
                    { 
                        return exec<And>(table.num_rows(), lhs, rhs); 
                    },  getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateOperator::Or:
                return std::visit(
                    [&](auto &&lhs, auto &&rhs)
                    {
                        return exec<Or>(table.num_rows(), lhs, rhs);
                    }, getOperand(operands, 0), getOperand(operands, 1));
            case ast::PredicateOperator::Not:
                return std::visit(
                    [&](auto &&lhs) 
                    { 
                        return exec<Not>(table.num_rows(), lhs); 
                    },  getOperand(operands, 0));
            default:
                throw std::runtime_error("not implemented: predicate operator " + std::to_string((int)op.what));
            }
        }
            }, (const ast::PredicateBase &) p);
    }
};

//}

std::shared_ptr<arrow::Buffer> execute(const arrow::Table &table, const ast::Predicate &predicate, ColumnMapping mapping)
{
    Interpreter interpreter{table, mapping};
    auto result = interpreter.evaluate(predicate);
    auto ret = FixedSizeValueWriter<bool>(result->values()); // TODO: proper ownership passing, buffer should be unique or copied
    for(auto && [refid, columnIndex] : mapping)
    {
        const auto column = table.column(columnIndex);
        if(column->null_count() == 0)
            continue;

        int i = 0;
        iterateOverGeneric(*column, 
            [&] (auto &&) { i++;              }, 
            [&]           { ret.store(i++, false); });
    }

    return ret.buffer;
}

// template<typename ArrowType>
// auto arrayWith(const arrow::Table &table, const Array &values, std::shared_ptr<arrow::Buffer> nullBuffer)
// {
//     using T = typename TypeDescription<ArrowType::type_id>::ValueType;
//     const auto N = table.num_rows();
//     constexpr auto id = ValueTypeToId<T>();
//     if constexpr(std::is_arithmetic_v<T>)
//     {
//         return std::make_shared<typename TypeDescription<id>::Array>(N, arrayProto.buffer, nullBuffer, -1);
//     }
//     else
//     {
//         arrow::StringBuilder builder;
//         checkStatus(builder.Reserve(N));
// 
//         if(nullBuffer)
//         {
//             for(int i = 0; i < N; i++)
//             {
//                 if(arrow::BitUtil::GetBit(nullBuffer->data(), i))
//                 {
//                     const auto sv = arrayProto.load(i);
//                     checkStatus(builder.Append(sv.data(), (int)sv.size()));
//                 }
//                 else
//                     checkStatus(builder.AppendNull());
//             }
//         }
//         else
//         {
//             for(int i = 0; i < N; i++)
//             {
//                 const auto sv = arrayProto.load(i);
//                 checkStatus(builder.Append(sv.data(), (int)sv.size()));
//             }
//         }
// 
//         return finish(builder);
//     }
// }
// 
// template<typename T>
// auto arrayWith(const arrow::Table &table, const T &constant, std::shared_ptr<arrow::Buffer> nullBuffer)
// {
//     const auto N = table.num_rows();
//     constexpr auto id = ValueTypeToId<T>();
//     if constexpr(std::is_arithmetic_v<T>)
//     {
//         auto [buffer, raw] = allocateBuffer<T>(N);
//         std::fill_n(raw, N, constant);
//         return std::make_shared<typename TypeDescription<id>::Array>(N, buffer, nullBuffer, -1);
//     }
//     else
//     {
//         arrow::StringBuilder builder;
//         checkStatus(builder.Reserve(N));
// 
//         const auto length = (int32_t)constant.size();
//         if(!nullBuffer)
//         {
//             for(int i = 0; i < N; i++)
//                 checkStatus(builder.Append(constant.data(), length));
//         }
//         else
//         {
//             for(int i = 0; i < N; i++)
//             {
//                 if(arrow::BitUtil::GetBit(nullBuffer->data(), i))
//                     checkStatus(builder.Append(constant.data(), length));
//                 else
//                     checkStatus(builder.AppendNull());
//             }
//         }
// 
//         return finish(builder);
//     }
// }

constexpr bool fitsIntoChunk(int64_t value)
{
    return value < std::numeric_limits<int32_t>::max();
}


template<typename T>
auto arrayWithConstant(int64_t length, T value, std::shared_ptr<arrow::Buffer> nulls = nullptr)
{
    assert(fitsIntoChunk(length));
    if constexpr(std::is_arithmetic_v<T>)
    {
        using ArrowType = StorageToArrowType_t<T>;
        using Array = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto [buffer, ptr] = allocateBuffer<T>(length);
        for(int i = 0; i < length; i++)
            ptr[i] = value;

        return std::make_shared<Array>(length, buffer, nulls, -1);
    }
}

template<typename Array>
std::shared_ptr<Array> arrayWithNulls(int64_t length, const std::shared_ptr<Array> &array, std::shared_ptr<arrow::Buffer> nulls = nullptr)
{
    // TODO if array had offset, than out nulls buffer should have offset as well
    if(array->offset() == 0  &&  nulls)
        throw std::runtime_error("not implemented: resulting array with offset and nulls");

    return std::make_shared<Array>(array->length(), array->data(), nulls, -1, array->offset());
}

std::shared_ptr<arrow::Array> arrayFromField(int64_t length, const Interpreter::Field &field, std::shared_ptr<arrow::Buffer> nulls = nullptr)
{
    return std::visit([&](auto &&fieldValue) -> std::shared_ptr<arrow::Array>
    {
        using T = std::decay_t<decltype(fieldValue)>;
        if constexpr(is_scalar<T>::value)
        {
            return arrayWithConstant(length, fieldValue.value, nulls);
        }
        else
        {
            static_assert(!is_scalar<T>::value);
            return arrayWithNulls<typename T::element_type>(length, fieldValue, nulls);
        }
    }, field);
}

std::shared_ptr<arrow::Array> execute(const arrow::Table &table, const ast::Value &value, ColumnMapping mapping)
{
    Interpreter interpreter{table, mapping};
    auto field = interpreter.evaluateValue(value);


    bool usedNullableColumns = false;
    BitmaskGenerator bitmask{table.num_rows(), true};

    for(auto && [refid, columnIndex] : mapping)
    {
        const auto column = table.column(columnIndex);
        if(column->null_count() == 0)
            continue;
 
        usedNullableColumns = true;

        int64_t i = 0;
        iterateOverGeneric(*column, 
            [&] (auto &&) { i++;}, 
            [&]           { bitmask.clear(i++); });
    }

    const auto nullBufferToBeUsed = usedNullableColumns ? bitmask.buffer : nullptr;

    return arrayFromField(table.num_rows(), field, nullBufferToBeUsed);
}
