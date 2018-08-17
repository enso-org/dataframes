#pragma once

#include <vector>
#include "AST.h"

namespace arrow
{
    class Array;
    class Buffer;
    class Table;
}

namespace ast
{
    struct Predicate;
}

using ArrayMask = std::vector<unsigned char>;

std::shared_ptr<arrow::Buffer> execute(const arrow::Table &table, const ast::Predicate &predicate, ColumnMapping mapping);
std::shared_ptr<arrow::Array> execute(const arrow::Table &table, const ast::Value &value, ColumnMapping mapping);