#include "Fixture.h"

#include "Core/ArrowUtilities.h"
#include <numeric>

template <typename T>
std::shared_ptr<arrow::ChunkedArray>
ChunkedFixture::toRandomChunks(std::vector<T> &v)
{
    auto continuousArray = toArray(v);
    std::vector<std::shared_ptr<arrow::Array>> chunks;

    const auto maxChunk = 200;
    std::uniform_int_distribution<> distribution{ 1, maxChunk };

    size_t currentPos = 0;
    while(currentPos < v.size())
    {
        auto chunkSize = distribution(generator);
        chunks.push_back(continuousArray->Slice(currentPos, chunkSize));
        currentPos += chunkSize;
    }

    std::shuffle(chunks.begin(), chunks.end(), generator);
    auto ret = std::make_shared<arrow::ChunkedArray>(chunks);
    v = toVector<T>(*ret);
    return ret;
}


ChunkedFixture::ChunkedFixture(int N /*= 10'000'000*/) : N(N)
{
    intsV.resize(N);
    doublesV.resize(N);
    std::iota(intsV.begin(), intsV.end(), 0);
    std::iota(doublesV.begin(), doublesV.end(), 0.0);

    ints = toColumn(toRandomChunks(intsV), "a");
    doubles = toColumn(toRandomChunks(doublesV), "b");
    table = tableFromColumns({ ints, doubles });
}

std::shared_ptr<arrow::Column> DataGenerator::generateColumn(arrow::Type::type id, int64_t N, std::string name, double nullShare /*= 0.0*/)
{
    if(id == arrow::Type::INT64)
        return generateColumn(id, N, name, nullShare, std::uniform_int_distribution<int64_t>{});
    if(id == arrow::Type::DOUBLE)
        return generateColumn(id, N, name, nullShare, std::uniform_real_distribution<double>{});
    if(id == arrow::Type::STRING)
        return generateColumn(id, N, name, nullShare, std::uniform_int_distribution<int64_t>{});
    throw std::runtime_error("wrong type");
}

std::shared_ptr<arrow::Table> DataGenerator::generateNumericTable(int N)
{
    auto intColumn1 = generateColumn(arrow::Type::INT64, N, "intsNonNull");
    auto intColumn2 = generateColumn(arrow::Type::INT64, N, "intsSomeNulls", 0.2);
    auto intColumn3 = generateColumn(arrow::Type::INT64, N, "intsManyNulls", 0.7);
    auto intColumn4 = generateColumn(arrow::Type::INT64, N, "intsAllNulls", 1.0);
    auto doubleColumn1 = generateColumn(arrow::Type::DOUBLE, N, "doublesNonNull");
    auto doubleColumn2 = generateColumn(arrow::Type::DOUBLE, N, "doublesSomeNulls", 0.2);
    auto doubleColumn3 = generateColumn(arrow::Type::DOUBLE, N, "doublesManyNulls", 0.7);
    auto doubleColumn4 = generateColumn(arrow::Type::DOUBLE, N, "doublesAllNulls", 1.0);

    return tableFromColumns({ intColumn1, intColumn2, intColumn3, intColumn4, doubleColumn1, doubleColumn2, doubleColumn3, doubleColumn4 });
}
