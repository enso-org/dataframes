#pragma once
#include "Core/Common.h"
#include "IO.h"

namespace arrow
{
    class Table;
}

EXPORT std::shared_ptr<arrow::Table> readXlsxFile(const char *filepath, HeaderPolicy header, std::vector<ColumnType> columnTypes);
EXPORT void writeXlsx(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy);
