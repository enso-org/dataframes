#include "Processing.h"

#include <vector>

#include <arrow/table.h>
#include "Core/ArrowUtilities.h"
// 
// namespace
// {
// 
// 
//     std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const std::vector<bool> &mask)
//     {
//     }
// }
// 
// std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table)
// {
//     std::vector<bool> mask;
//     mask.resize(table->num_rows());
// 
//     auto N = 50;
//     int row = 0;
//     iterateOver<arrow::Type::INT64>(*table->column(3)->data()
//         , [&](int64_t i) { mask[row++] = i > N; }
//         , [&]() { row++; });
// 
// }
// 
