#pragma once

#include "Core/Common.h"

namespace arrow
{
    class Table;
}

#include <memory>

EXPORT std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table);