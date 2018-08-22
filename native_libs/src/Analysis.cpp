#include "Analysis.h"
#include <unordered_map>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>


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

template<typename T>
struct Minimum
{
    T accumulator = std::numeric_limits<T>::max();
    std::string name = "min";
    void operator() (T elem) {  accumulator = std::min<T>(accumulator, elem); }
    auto get() { return accumulator; }
};

template<typename T>
struct Maximum
{
    T accumulator = std::numeric_limits<T>::min();
    std::string name = "max";
    void operator() (T elem) {  accumulator = std::max<T>(accumulator, elem); }
    auto get() { return accumulator; }
};

// TODO naive implementation, look for something numerically better
template<typename T>
struct Mean
{
    int64_t count = 0;
    double accumulator = 0;
    std::string name = "mean";
    void operator() (T elem) {  accumulator += elem; count++; }
    auto get() { return accumulator / count; }
};

template<typename T>
struct Median
{
    boost::accumulators::accumulator_set<T, boost::accumulators::features<boost::accumulators::tag::median>> accumulator;

    std::string name = "median";
    void operator() (T elem) {  accumulator(elem); }
    auto get() { return boost::accumulators::median(accumulator); }
};

template<typename T>
struct Variance
{
    boost::accumulators::accumulator_set<T, boost::accumulators::features<boost::accumulators::tag::variance>> accumulator;

    std::string name = "variance";
    void operator() (T elem) {  accumulator(elem); }
    auto get() { return boost::accumulators::variance(accumulator); }
};

template<typename T>
struct StdDev : Variance<T>
{
    auto get() { return std::sqrt(Variance<T>::get()); }
};

template<typename T>
struct Sum : Variance<T>
{
    T accumulator{};
    std::string name = "sum";
    void operator() (T elem) {  accumulator += elem; }
    auto get() { return accumulator; }
};


template<arrow::Type::type id, typename Processor>
auto calculateStatScalar(const arrow::Column &column, Processor &p)
{
    iterateOver<id>(column, 
        [&] (auto elem) { p(elem); },
        [] {});

    return p.get();
}

template<template <typename> typename Processor>
std::shared_ptr<arrow::Column> calculateStat(const arrow::Column &column)
{
    return visitType(*column.type(), [&](auto id) -> std::shared_ptr<arrow::Column>
    {
        if constexpr(id.value != arrow::Type::STRING)
        {
            using T = typename TypeDescription<id.value>::ValueType;
            Processor<T> p;
            using ResultT = decltype(p.get());

            if(column.length() - column.null_count() <= 0)
                return toColumn(std::vector<std::optional<ResultT>>{std::nullopt}, p.name);

            const auto result = calculateStatScalar<id.value>(column, p);
            return toColumn(std::vector<ResultT>{result}, { p.name });
        }
        else
            throw std::runtime_error("Operation not supported for type " + column.type()->ToString());
    });
}


std::shared_ptr<arrow::Column> calculateMin(const arrow::Column &column)
{
    return calculateStat<Minimum>(column);
}

std::shared_ptr<arrow::Column> calculateMax(const arrow::Column &column)
{
    return calculateStat<Maximum>(column);
}

std::shared_ptr<arrow::Column> calculateMean(const arrow::Column &column)
{
    return calculateStat<Mean>(column);
}

std::shared_ptr<arrow::Column> calculateMedian(const arrow::Column &column)
{
    return calculateStat<Median>(column);
}

std::shared_ptr<arrow::Column> calculateVariance(const arrow::Column &column)
{
    return calculateStat<Variance>(column);
}

std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column)
{
    return calculateStat<StdDev>(column);
}

std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column)
{
    return calculateStat<Sum>(column);
}
