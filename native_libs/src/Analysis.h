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
DFH_EXPORT std::shared_ptr<arrow::Column> calculateRSI(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateQuantile(const arrow::Column &column, double q);
DFH_EXPORT double calculateCorrelation(const arrow::Column &xCol, const arrow::Column &yCol);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateCorrelation(const arrow::Table &table, const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Table> calculateCorrelationMatrix(const arrow::Table &table);

DFH_EXPORT double autoCorrelation(const std::shared_ptr<arrow::Column> &column, int64_t lag = 1);

template<typename ArrowType>
struct GroupedKeyInfo
{
    using KeyT = typename ArrowTypeDescription<ArrowType>::ObservedType;
    bool hasNulls;
    std::unordered_map<KeyT, int64_t> uniqueValues; // key value => group id
    std::vector<int64_t> groupIds; // [row index] => group id

    explicit GroupedKeyInfo(const arrow::Column &keyColumn)
        : hasNulls(keyColumn.null_count() != 0)
        , groupIds(keyColumn.length())
    {
        auto *rowGroupId = groupIds.data();
        iterateOver<ArrowType::type_id>(keyColumn, 
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
    Minimum, Maximum, Mean, Length, Median, First, Last, Sum, RSI, StdDev
};

template<typename Function>
auto dispatchAggregateByEnum(AggregateFunction aggregateEnum, Function &&f)
{
    switch(aggregateEnum)
    {
    CASE_DISPATCH(AggregateFunction::Minimum)
    CASE_DISPATCH(AggregateFunction::Maximum)
    CASE_DISPATCH(AggregateFunction::Mean)
    CASE_DISPATCH(AggregateFunction::Length)
    CASE_DISPATCH(AggregateFunction::Median)
    CASE_DISPATCH(AggregateFunction::First)
    CASE_DISPATCH(AggregateFunction::Last)
    CASE_DISPATCH(AggregateFunction::Sum)
    CASE_DISPATCH(AggregateFunction::RSI)
    CASE_DISPATCH(AggregateFunction::StdDev)
    default: throw std::runtime_error("not supported aggregate function " + std::to_string((int)aggregateEnum));
    }
}

DFH_EXPORT std::string to_string(AggregateFunction a);

DFH_EXPORT std::shared_ptr<arrow::Table> abominableGroupAggregate(std::shared_ptr<arrow::Column> keyColumn, std::vector<std::pair<std::shared_ptr<arrow::Column>, std::vector<AggregateFunction>>> toAggregate);

DFH_EXPORT std::vector<int64_t> collectRollingIntervalSizes(std::shared_ptr<arrow::Column> keyColumn, DynamicField interval);
DFH_EXPORT std::shared_ptr<arrow::Table> rollingInterval(std::shared_ptr<arrow::Column> keyColumn, DynamicField interval, std::vector<std::pair<std::shared_ptr<arrow::Column>, std::vector<AggregateFunction>>> toAggregate);