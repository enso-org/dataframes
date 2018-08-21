#include "Analysis.h"

#include <unordered_map>

template<arrow::Type::type id>
std::shared_ptr<arrow::Table> countValueTyped(const arrow::Column &column)
{
    using T = typename TypeDescription<id>::ObservedType;
    using Builder = typename TypeDescription<id>::BuilderType;
    std::unordered_map<T, int64_t> valueCounts;

    iterateOver<id>(column, 
        [&] (auto &&elem) { valueCounts[elem]++; },
        [] () {});

    Builder valueBuilder;
    arrow::Int64Builder countBuilder;

    valueBuilder.Reserve(valueCounts.size());
    countBuilder.Reserve(valueCounts.size());

    for(auto && [value, count] : valueCounts)
    {
        append(valueBuilder, value);
        append(countBuilder, count);
    }

    if(column.null_count())
    {
        valueBuilder.AppendNull();
        countBuilder.Append(column.null_count());
    }

    auto valueArray = finish(valueBuilder);
    auto countArray = finish(countBuilder);

    auto valueColumn = std::make_shared<arrow::Column>(arrow::field("value", column.type(), column.null_count()), valueArray);
    auto countColumn = std::make_shared<arrow::Column>(arrow::field("count", countArray->type(), false), countArray);

    return tableFromArrays({valueArray, countArray}, {"value", "count"});
}

std::shared_ptr<arrow::Table> countValues(const arrow::Column &column)
{
    return visitType(*column.type(), [&] (auto id) 
    {
        return countValueTyped<id.value>(column);
    });
}
