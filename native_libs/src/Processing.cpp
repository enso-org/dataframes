#include "Processing.h"

#include <cassert>
#include <iostream>
#include <vector>

#include <arrow/table.h>
#include <arrow/type_traits.h>

#define RAPIDJSON_NOMEMBERITERATORCLASS 
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "Core/ArrowUtilities.h"
#include "LQuery/AST.h"
#include "LQuery/Interpreter.h"

using namespace std::literals;


std::shared_ptr<arrow::Table> filter(std::shared_ptr<arrow::Table> table, const char *dslJsonText)
{
    auto [mapping, predicate] = ast::parsePredicate(*table, dslJsonText);
    auto mask = execute(*table, predicate, mapping);

    const auto oldRowCount = table->num_rows();
    int64_t newRowCount = 0;
    for(int i = 0; i < table->num_rows(); i++)
        newRowCount += mask->mutable_data()[i];
    
    std::vector<std::shared_ptr<arrow::Array>> newColumns;

    for(int columnIndex = 0; columnIndex < table->num_columns(); columnIndex++)
    {
        const auto column = table->column(columnIndex);

        // TODO handle zero chunks?
        // TODO handle more chunks
        const auto chunk = column->data()->chunk(0);


        visitArray(chunk.get(), [&](auto *array) 
        {
            // TODO string
            using TD = ArrayTypeDescription<std::remove_pointer_t<decltype(array)>>;
            using T = typename TD::ValueType;
            if constexpr(std::is_scalar_v<T>)
            {
                auto valueBuffer = std::make_shared<arrow::PoolBuffer>();
                valueBuffer->TypedResize<T>(newRowCount);


                const unsigned char *maskBuffer = mask->data();
                const T *sourceBuffer = array->raw_values();
                T *outputBuffer = reinterpret_cast<T*>(valueBuffer->mutable_data());

                int filteredItr = 0;
                for(int64_t sourceItr = 0; sourceItr < oldRowCount; sourceItr++)
                {
                    if(maskBuffer[sourceItr])
                        outputBuffer[filteredItr++] = sourceBuffer[sourceItr];
                }


                newColumns.push_back(std::make_shared<typename TD::Array>(newRowCount, valueBuffer));
            }
            else
                throw std::runtime_error("not implemented: filtering strings");// TODO string
        });


    }

    return arrow::Table::Make(table->schema(), newColumns);
}

