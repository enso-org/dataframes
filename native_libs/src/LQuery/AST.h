#pragma once

#include "Core/ArrowUtilities.h"
#include "Core/Common.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "variant.h"

namespace arrow
{
    class Table;
}

using ColumnReferenceId = int;
using ColumnIndexInTable = int;
using ColumnMapping = std::unordered_map<ColumnReferenceId, ColumnIndexInTable>;

template<typename T>
struct HeapHolder
{
    std::unique_ptr<T> ptr;

    HeapHolder() : ptr(std::make_unique<T>()) {};
    HeapHolder(const T &t) : ptr(std::make_unique<T>(t)) {};
    HeapHolder(T &&t) : ptr(std::make_unique<T>(std::move(t))) {};
    
    HeapHolder(const HeapHolder &rhs) : ptr(ptr ? std::make_unique<T>(*rhs.ptr) : nullptr) {};
    HeapHolder(HeapHolder &&rhs) : ptr(std::move(rhs.ptr)) {};

    T * operator->() const { return ptr.get(); }
    T & operator*() const { return *ptr; }
    explicit operator bool() const { return ptr; }
};

namespace ast
{
    struct Value;
    struct Predicate;

    enum class ValueOperator
    {
        Plus, Minus, Times, Divide, Modulo,
        Negate, Abs,

        // timestamp operations
        Day, Month, Year
    };

    ValueOperator valueOperatorFromName(const std::string &name);

    template<typename OperatorTag, typename OperandType>
    struct OperationNode
    {
        OperationNode(OperatorTag what, std::vector<OperandType> operands)
            : what(what), operands(std::move(operands))
        {}

        OperatorTag what;
        std::vector<OperandType> operands; // unused are null
    };

    template<typename Storage>
    struct Literal
    {
        Literal(Storage value) : literal(std::move(value))
        {}
        Storage literal;
    };
    struct ColumnReference
    {
        ColumnReference(int columnRefId) : columnRefId(columnRefId) {}
        int columnRefId;
    };

    struct Condition
    {
        Condition(const Predicate &p, const Value &onTrue, const Value &onFalse);

        HeapHolder<Predicate> predicate;
        HeapHolder<Value> onTrue, onFalse;
    };

    using ValueOperation = OperationNode<ValueOperator, Value>;
    using ValueBase = variant<Literal<int64_t>, Literal<double>, Literal<std::string>, Literal<Timestamp>, ColumnReference, ValueOperation, Condition>;

    struct Value : ValueBase
    {
        using ValueBase::variant;
    };
    
    enum class PredicateOperator
    {
        And, Or, Not
    };
    PredicateOperator predicateBooleanOperatorFromName(const std::string &name);
    using PredicateOperation = OperationNode<PredicateOperator, Predicate>;

    enum class PredicateFromValueOperator
    {
        Greater, Lesser,  // works for int/real
        Equal, // works for all types
        StartsWith, Matches // works for strings
    };


    PredicateFromValueOperator predicateOperatorFromName(const std::string &name);

    using PredicateFromValueOperation = OperationNode<PredicateFromValueOperator, Value>;
    
    using PredicateBase = variant<PredicateOperation, PredicateFromValueOperation>;
    struct Predicate : PredicateBase
    {
        using PredicateBase::variant;
    };

    DFH_EXPORT std::pair<ColumnMapping, Predicate> parsePredicate(const arrow::Table &table, const char *lqueryJsonText);
    DFH_EXPORT std::pair<ColumnMapping, Value> parseValue(const arrow::Table &table, const char *lqueryJsonText);
}
