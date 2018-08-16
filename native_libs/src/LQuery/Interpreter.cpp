#include "Interpreter.h"

#include <arrow/buffer.h>
#include <arrow/table.h>

#include "Core/ArrowUtilities.h"
#include "AST.h"

using namespace std::literals;

//namespace
//{

    // type-aware wrapper for buffer that is either owned or belongs to an external Column
    template<typename T>
    struct ArrayOperand 
    {
        std::shared_ptr<arrow::Buffer> buffer;

        auto mutable_data() { return reinterpret_cast<T *>(buffer->mutable_data()); }
        auto data() const { return reinterpret_cast<const T*>(buffer->data()); }
        
        explicit ArrayOperand(std::shared_ptr<arrow::Buffer> buffer)
            : buffer(std::move(buffer))
        {}
        explicit ArrayOperand(size_t length)
        {
            buffer = allocateBuffer<T>(length);
        }

        T &operator[](size_t index) { return mutable_data()[index]; }
        const T &operator[](size_t index) const { return data()[index]; }
    };

//     template<>
//     struct ArrayOperand<std::string>
//     {
//     };

    template<typename T>
    auto getValue(const ArrayOperand<T> &src, int64_t index)
    {
        return src[index];
    }
    template<typename T>
    auto getValue(const T &src, int64_t index)
    {
        return src;
    }

#define BINARY_OPERATOR(op)                                                                                              \
    template<typename Lhs>                                                                                               \
    static auto exec(const Lhs &lhs, const Lhs &rhs)                                                                     \
    {                                                                                                                    \
        return lhs op rhs;                                                                                               \
    }                                                                                                                    \
    template<typename Lhs, typename Rhs>                                                                                 \
    static auto exec(const Lhs &lhs, const Rhs &rhs)                                                                     \
    {                                                                                                                    \
        throw std::runtime_error("not supported operand types: "s + typeid(lhs).name() + " and "s + typeid(rhs).name()); \
        return exec(lhs, lhs);                                                                                           \
    }

#define FAIL_ON_STRING(ret)                                                                                              \
    static ret exec(const std::string &lhs, const std::string &rhs)                                                      \
    {                                                                                                                    \
        throw std::runtime_error("not supported operand types: "s + typeid(lhs).name() + " and "s + typeid(rhs).name()); \
    }

    struct GreaterThan { BINARY_OPERATOR(>); FAIL_ON_STRING(bool); };
    struct LessThan    { BINARY_OPERATOR(<); FAIL_ON_STRING(bool); };
    struct EqualTo     { BINARY_OPERATOR(==);};
    struct Plus        { BINARY_OPERATOR(+); };
    struct Minus       { BINARY_OPERATOR(-); FAIL_ON_STRING(std::string); };
    struct Times       { BINARY_OPERATOR(*); FAIL_ON_STRING(std::string); };
    struct Divide      { BINARY_OPERATOR(/); FAIL_ON_STRING(std::string); };
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
    };


    template<typename Operation, typename Lhs>
    auto exec(const Lhs &lhs, int64_t count)
    {
        using OperationResult = decltype(Operation::exec(getValue(lhs, 0)));
        using OperandValue = std::conditional_t<std::is_same_v<bool, OperationResult>, unsigned char, OperationResult>;

        // TODO: optimization opportunity: boolean constant support (remove the last part of if below and fix the build)
        if constexpr(std::is_arithmetic_v<Lhs> && !std::is_same_v<unsigned char, OperandValue>)
        {
            return Operation::exec(lhs);
        }
        else
        {
            ArrayOperand<OperandValue> ret{ (size_t)count };
            static_assert(sizeof(OperandValue) >= sizeof(OperationResult));
            for(int64_t i = 0; i < count; i++)
            {
                ret[i] = Operation::exec(getValue(lhs, i));
            }
            return ret;
        }
    }

    template<typename Operation, typename Lhs, typename Rhs>
    auto exec(const Lhs &lhs, const Rhs &rhs, int64_t count)
    {
        using OperationResult = decltype(Operation::exec(getValue(lhs, 0), getValue(rhs, 0)));
        using OperandValue = std::conditional_t<std::is_same_v<bool, OperationResult>, unsigned char, OperationResult>;

        // TODO: optimization opportunity: boolean constant support (remove the last part of if below and fix the build)
        if constexpr(std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs> && !std::is_same_v<unsigned char, OperandValue>)
        {
            return Operation::exec(lhs, rhs);
        }
        else
        {
            ArrayOperand<OperandValue> ret{ (size_t)count };
            static_assert(sizeof(OperandValue) >= sizeof(OperationResult));
            for(int64_t i = 0; i < count; i++)
            {
                ret[i] = Operation::exec(getValue(lhs, i), getValue(rhs, i));
            }
            return ret;
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

    using Field = std::variant<int64_t, double, ArrayOperand<int64_t>, ArrayOperand<double>, ArrayOperand<std::string>>;

    Field fieldFromColumn(const arrow::Column &column)
    {
        const auto data = column.data();
        if(data->num_chunks() != 1)
            throw std::runtime_error("not implemented: processing of chunked arrays");

        const auto chunkPtr = data->chunk(0).get();
        return visitArray(chunkPtr, [] (auto *array) -> Field
        {
            using ArrowType = typename std::remove_pointer_t<decltype(array)>::TypeClass;
            using T = typename TypeDescription<ArrowType::type_id>::ValueType;
            return ArrayOperand<T>(array->data()->buffers.at(1)); // NOTE: this works only for primitive arrays
        });
    }
    

    std::array<Field, ast::MaxOperatorArity> evaluateOperands(const std::array<std::unique_ptr<ast::Value>, ast::MaxOperatorArity> &operands)
    {
        static_assert(ast::MaxOperatorArity == 2); // if changed, adjust entries below :(
        return 
        {{
            evaluateValue(*operands[0]),
            evaluateValue(*operands[1])
        }};
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
                        { return exec<opname>(lhs, table.num_rows());},      \
                    operands[0]);
#define VALUE_BINARY_OP(opname)                                              \
            case ast::ValueOperator::opname:                                 \
                return std::visit(                                        \
                    [&] (auto &&lhs, auto &&rhs) -> Field                    \
                        { return exec<opname>(lhs, rhs, table.num_rows());}, \
                    operands[0], operands[1]);

                const auto operands = evaluateOperands(op.operands);
                switch(op.what)
                {
                    VALUE_BINARY_OP(Plus);
                    VALUE_BINARY_OP(Minus);
                    VALUE_BINARY_OP(Times);
                    VALUE_BINARY_OP(Divide);
                    VALUE_UNARY_OP(Negate);
                default:
                    throw std::runtime_error("not implemented: value operator " + std::to_string((int)op.what));
                }
            },
            [&] (const ast::Literal<int64_t> &l)     -> Field { return l.literal; },
            [&] (const ast::Literal<double> &l)      -> Field { return l.literal; },
            //[&] (const ast::Literal<std::string> &l) -> Field { return l.literal; },
            [&] (auto &&t) -> Field { throw std::runtime_error("not implemented: value node of type "s + typeid(decltype(t)).name()); }
            }, (const ast::ValueBase &) value);

    }

    ArrayOperand<unsigned char> evaluate(const ast::Predicate &p)
    {
        return std::visit(overloaded{
            [&] (const ast::PredicateFromValueOperation &elem) -> ArrayOperand<unsigned char>
        {
            const auto operands = evaluateOperands(elem.operands);
            switch(elem.what)
            {
            case ast::PredicateFromValueOperator::Greater:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<GreaterThan>(lhs, rhs, table.num_rows());},
                    operands[0], operands[1]);
            case ast::PredicateFromValueOperator::Lesser:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<LessThan>(lhs, rhs, table.num_rows());},
                    operands[0], operands[1]);
            case ast::PredicateFromValueOperator::Equal:
                return std::visit(
                    [&] (auto &&lhs, auto &&rhs) { return exec<EqualTo>(lhs, rhs, table.num_rows());},
                    operands[0], operands[1]);
            default:
                throw std::runtime_error("not implemented: predicate operator " + std::to_string((int)elem.what));
            }
        },
            [&] (const ast::PredicateOperation &) -> ArrayOperand<unsigned char> { throw std::runtime_error("not implemented: PredicateOperation"); }
            }, (const ast::PredicateBase &) p);
    }
};

//}

std::shared_ptr<arrow::Buffer> execute(const arrow::Table &table, const ast::Predicate &predicate, ColumnMapping mapping)
{
    Interpreter interpreter{table, mapping};
    return interpreter.evaluate(predicate).buffer;
}