#include "Processing.h"

#include <cassert>
#include <iostream>
#include <vector>

#include <arrow/table.h>
#include <arrow/type_traits.h>

#include "Core/ArrowUtilities.h"

namespace
{
    template<typename ArrowValueType, typename BooleanSequence>
    std::shared_ptr<arrow::Column> filteredColumn(const arrow::Column &column, const BooleanSequence &mask, int filteredCount)
    {
        std::vector<std::shared_ptr<arrow::Array>> newChunks;

        assert(column.length() == std::size(mask));
        typename arrow::TypeTraits<ArrowValueType>::BuilderType builder;
        builder.Reserve(filteredCount);
        // throwingCast<typename arrow::TypeTraits<ArrowValueType>::ArrayType*>

        int row = 0;
        for(auto chunk : column.data()->chunks())
        {
            const int chunkLength = chunk->length();
            auto chunkT = throwingCast<typename arrow::TypeTraits<ArrowValueType>::ArrayType*>(chunk.get());

            for(int i = 0; i < chunkLength; ++i)
            {
                if(mask[row++])
                {
                    if constexpr(std::is_same_v<arrow::StringType, ArrowValueType>)
                        builder.Append(chunkT->GetString(i));
                    else
                        builder.Append(chunkT->Value(i));
                }
            }

            newChunks.push_back(finish(builder));
        }

        return std::make_shared<arrow::Column>(column.field(), newChunks);
    }

    template<typename BooleanSequence>
    std::shared_ptr<arrow::Column> filteredColumn(const arrow::Column &column, const BooleanSequence &mask, int filteredCount)
    {
        switch(column.type()->id())
        {
        case arrow::Type::INT64: return filteredColumn<arrow::Int64Type>(column, mask, filteredCount);
        case arrow::Type::DOUBLE: return filteredColumn<arrow::DoubleType>(column, mask, filteredCount);
        case arrow::Type::STRING: return filteredColumn<arrow::StringType>(column, mask, filteredCount);
        default                 : throw  std::runtime_error(__FUNCTION__ + std::string(": not supported array type ") + column.type()->ToString());
        }
    }

    template<typename BooleanSequence>
    std::shared_ptr<arrow::Table> filter(const arrow::Table &table, const BooleanSequence &mask, int filteredCount)
    {
        std::vector<std::shared_ptr<arrow::Column>> newColumns;


        for(int columnIndex = 0; columnIndex < table.num_columns(); columnIndex++)
        {
            newColumns.push_back(filteredColumn(*table.column(columnIndex), mask, filteredCount));
        }

        return arrow::Table::Make(table.schema(), newColumns);
    }
}

std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table)
{
    std::vector<char> mask;
    mask.resize(table->num_rows());

    constexpr auto N = 50;
    int row = 0;

//     std::cout << table->schema()->ToString() << std::endl;
//     for(int columnIndex = 0; columnIndex < table->num_columns(); columnIndex++)
//     {
//         std::cout << table->column(columnIndex)->field()->ToString() << std::endl;
//     }
//     for(int columnIndex = 0; columnIndex < table->num_columns(); columnIndex++)
//     {
//         std::cout << table->column(columnIndex)->name() << "  -> " << table->column(columnIndex)->data()->type()->ToString() << std::endl;
//     }
// 
    int filteredCount = 0;
    auto c = table->column(3);
    iterateOver<arrow::Type::INT64>(*c->data()
        , [&](int64_t i) { mask[row++] = (i > N); filteredCount++; }
        , [&]() { row++; });

    return filter(*table, mask, filteredCount);
}

