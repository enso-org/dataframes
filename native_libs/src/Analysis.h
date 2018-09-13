#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"

DFH_EXPORT std::shared_ptr<arrow::Table> countValues(const arrow::Column &column);

DFH_EXPORT std::shared_ptr<arrow::Column> calculateMin(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMax(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMean(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMedian(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateVariance(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateQuantile(const arrow::Column &column, double q);
DFH_EXPORT double calculateCorrelation(const arrow::Column &xCol, const arrow::Column &yCol);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateCorrelation(const arrow::Table &table, const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Table> calculateCorrelationMatrix(const arrow::Table &table);

DFH_EXPORT double autoCorrelation(const std::shared_ptr<arrow::Column> &column, int lag = 1);

template<arrow::Type::type id>
struct GroupedKeyInfo
{
    using KeyT = typename TypeDescription<id>::ObservedType;
    bool hasNulls;
    std::unordered_map<KeyT, int64_t> uniqueValues; // key value => group id
    std::vector<int64_t> groupIds; // [row index] => group id

    explicit GroupedKeyInfo(const arrow::Column &keyColumn)
        : hasNulls(keyColumn.null_count() != 0)
        , groupIds(keyColumn.length())
    {
        auto *rowGroupId = groupIds.data();
        iterateOver<id>(keyColumn, 
            [&] (auto &&value)
            {
                if(auto itr = uniqueValues.find(value); itr != uniqueValues.end())
                {
                    *rowGroupId++ = itr->second;
                }
                else
                {
                    *rowGroupId++ = uniqueValues.emplace(value, uniqueValues.size() + 1).first->second;
                }
            },
            [&] ()
            {
                *rowGroupId++ = 0;
            });
    }

    int64_t groupCount() const
    {
        // Null is not included in unique values
        return uniqueValues.size() + hasNulls;
    }
};


enum class AggregateFunction : int8_t
{
    Minimum, Maximum, Mean, Length, Median, First, Last
};

DFH_EXPORT std::shared_ptr<arrow::Table> abominableGroupAggregate(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Column> keyColumn, std::vector<std::pair<std::shared_ptr<arrow::Column>, std::vector<AggregateFunction>>> toAggregate);