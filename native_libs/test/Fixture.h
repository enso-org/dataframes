#pragma once

#include <memory>
#include <random>
#include <vector>

#include "Core/ArrowUtilities.h"

namespace arrow
{
    class ChunkedArray;
    class Column;
    class Table;
}

// TODO: use non-aligned begins with nulls
struct ChunkedFixture
{
    int N;

    std::vector<int64_t> intsV;
    std::vector<double> doublesV;
    std::mt19937 generator{ std::random_device{}() };

    std::shared_ptr<arrow::Column> ints, doubles;
    std::shared_ptr<arrow::Table> table;

    template <typename T>
    std::shared_ptr<arrow::ChunkedArray> toRandomChunks(std::vector<T> &v);

    explicit ChunkedFixture(int N = 10'000'000);
};


struct DataGenerator
{
    std::mt19937 generator{ std::random_device{}() };

    template<typename Distribution>
    std::shared_ptr<arrow::Column> generateColumn(arrow::Type::type id, int64_t N, std::string name, double nullShare, Distribution distribution)
    {
        std::bernoulli_distribution nullDistribution{ nullShare };
        auto arr = visitType(id, [&](auto id)
        {
            auto builder = makeBuilder(getTypeSingleton<id.value>());
            builder->Reserve(N);
            for(int64_t i = 0; i < N; i++)
            {
                if(nullDistribution(generator))
                {
                    builder->AppendNull();
                }
                else
                {
                    const auto value = distribution(generator);
                    if constexpr(id.value != arrow::Type::STRING)
                        builder->Append(value);
                    else
                        builder->Append(std::to_string(value));
                }
            }

            return finish(*builder);
        });

        return std::make_shared<arrow::Column>(arrow::field(name, arr->type(), arr->null_count()), arr);
    }

    std::shared_ptr<arrow::Column> generateColumn(arrow::Type::type id, int64_t N, std::string name, double nullShare = 0.0);
    std::shared_ptr<arrow::Table> generateNumericTable(int N);
};


#define BOOST_CHECK_EQUAL_RANGES(a, b) BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(a), std::end(a), std::begin(b), std::end(b))
