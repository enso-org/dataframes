#pragma once

#include "Core/Common.h"

namespace arrow
{
    class Array;
    class Column;
    class Table;
}

#include <memory>

EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText);
EXPORT std::shared_ptr<arrow::Array> each(std::shared_ptr<arrow::Table> table, const char *dslJsonText);