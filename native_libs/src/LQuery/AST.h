#pragma once

#include "Core/Common.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>

#include "variant.h"

namespace arrow
{
    class Table;
}

using ColumnReferenceId = int;
using ColumnIndexInTable = int;
using ColumnMapping = std::unordered_map<ColumnReferenceId, ColumnIndexInTable>;

namespace ast
{
    constexpr auto MaxOperatorArity = 2;

    struct Value;
    struct Predicate;

    enum class ValueOperator
    {
        Plus, Minus, Times, Divide, 
        Negate
    };

    ValueOperator valueOperatorFromName(const std::string &name);

    template<typename OperatorTag, typename OperandType>
    struct OperationNode
    {
        OperationNode(OperatorTag what, std::array<std::unique_ptr<OperandType>, MaxOperatorArity> operands)
            : what(what), operands(std::move(operands))
        {}

        OperatorTag what;
        std::array<std::unique_ptr<OperandType>, MaxOperatorArity> operands; // unused are null
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

    using ValueOperation = OperationNode<ValueOperator, Value>;
    using ValueBase = std::variant<Literal<int64_t>, Literal<double>, Literal<std::string>, ColumnReference, ValueOperation>;

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
    
    using PredicateBase = std::variant<PredicateOperation, PredicateFromValueOperation>;
    struct Predicate : PredicateBase
    {
        using PredicateBase::variant;
    };

    EXPORT std::pair<ColumnMapping, Predicate> parsePredicate(const arrow::Table &table, const char *lqueryJsonText);
}
