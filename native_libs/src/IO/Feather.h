#pragma once

#include <memory>
#include <string>

#include "Core/Common.h"

namespace arrow
{
    class Table;
}

EXPORT void saveTableToFeatherFile(const std::string &filepath, const arrow::Table &table);
EXPORT std::shared_ptr<arrow::Table> loadTableFromFeatherFile(const std::string &filepath);
