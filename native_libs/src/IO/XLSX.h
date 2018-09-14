#pragma once
#include "Core/Common.h"
#include "IO.h"

namespace arrow
{
    class Table;
}

DFH_EXPORT std::shared_ptr<arrow::Table> readXlsxFile(const char *filepath, HeaderPolicy header = TakeFirstRowAsHeaders{}, std::vector<ColumnType> columnTypes = {});
DFH_EXPORT void writeXlsx(const char *filepath, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy = GeneratorHeaderPolicy::GenerateHeaderLine);
DFH_EXPORT void writeXlsx(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy = GeneratorHeaderPolicy::GenerateHeaderLine);
