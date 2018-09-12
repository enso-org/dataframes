#include "Sort.h"

#include <numeric>
#include "Core/ArrowUtilities.h"

template<typename F>
auto dispatch(SortOrder order, F &&f)
{
    switch(order)
    {
        CASE_DISPATCH(SortOrder::Ascending);
        CASE_DISPATCH(SortOrder::Descending);
    default: throw std::runtime_error(__FUNCTION__ + ": invalid value"s);
    }
}
template<typename F>
auto dispatch(NullPosition nullPosition, F &&f)
{
    switch(nullPosition)
    {
        CASE_DISPATCH(NullPosition::Before);
        CASE_DISPATCH(NullPosition::After);
    default: throw std::runtime_error(__FUNCTION__ + ": invalid value"s);
    }
}

namespace
{

template<typename ArrowType, bool nullable>
struct ColumnPermuter
{
    static constexpr arrow::Type::type id = ArrowType::type_id;

    std::shared_ptr<arrow::Column> column;
    std::shared_ptr<ArrowType> type;
    Permutation indices;

    ColumnPermuter(std::shared_ptr<arrow::Column> column, std::shared_ptr<ArrowType> type, Permutation indices, std::bool_constant<nullable> n={})
        : column(std::move(column))
        , type(std::move(type))
        , indices(std::move(indices))
    {}

    std::shared_ptr<arrow::Array> operator()() const
    {
        if(column->length() > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("not implemented: too big array");

        using T = typename TypeDescription<ArrowType::type_id>::StorageValueType;

        const auto length = (int32_t)column->length();

        const ChunkAccessor chunks{ *column->data() };
        if constexpr(!nullable && (id == arrow::Type::INT64 || id == arrow::Type::DOUBLE))
        {
            FixedSizeArrayBuilder<id, nullable> b{ type, length };
            {
                T * __restrict target = b.nextValueToWrite;
                for(auto index : indices)
                {
                    const auto[chunk, indexInChunk] = chunks.locate(index);
                    const auto value = arrayValueAt<id>(*chunk, indexInChunk);
                    // unfortunately gives performance edge over b.Append(value)
                    // TODO: can we have something nice and fast?
                    *target++ = value;
                }
            }
            return b.Finish();
        }
        else
        {
            auto b = makeBuilder(type);
            b->Reserve(indices.size());
            for(auto index : indices)
            {
                const auto[chunk, indexInChunk] = chunks.locate(index);
                if constexpr(nullable)
                {
                    if(chunk->IsValid(indexInChunk))
                    {
                        const auto value = arrayValueAt<id>(*chunk, indexInChunk);
                        append(*b, value);
                    }
                    else
                        b->AppendNull();
                }
                else
                    append(*b, arrayValueAt<id>(*chunk, indexInChunk));
            }

            return finish(*b);
        }
    }
};



std::shared_ptr<arrow::Array> permuteInnerToArray(std::shared_ptr<arrow::Column> column, const Permutation &indices)
{
    return visitDataType3(column->type(), [&](auto &&datatype)
    {
        return dispatch(column->null_count() != 0, [&](auto nullable)
        {
            using ArrowType = typename std::decay_t<decltype(datatype)>::element_type;
            return ColumnPermuter<ArrowType, nullable.value>(column, datatype, indices)();
        });
    });
}

std::shared_ptr<arrow::Column> permuteInner(std::shared_ptr<arrow::Column> column, const Permutation &indices)
{
    return std::make_shared<arrow::Column>(column->field(), permuteInnerToArray(column, indices));
}

std::shared_ptr<arrow::Table> permuteInner(std::shared_ptr<arrow::Table> table, const Permutation &indices)
{
    auto oldColumns = getColumns(*table);
    auto newColumns = transformToVector(oldColumns, [&](auto &&col) { return permuteInner(col, indices); });
    return arrow::Table::Make(table->schema(), newColumns);
}

bool isPermuteId(const Permutation &indices)
{
    for(auto i = 0_z; i < indices.size(); i++)
        if(indices[i] != i)
            return false;
    return true;
}

template<arrow::Type::type id, bool nullable, SortOrder order, NullPosition nulls>
void sortPermutationInner(Permutation &indices, const arrow::Column &sortBy)
{
    // Note: Measures shown that it is usually much faster to copy data into vector
    // rather than keep it in array and each time lookup index for the given chunk.
    //
    // For now this is simple and fast enough.
    using ElementType = typename TypeDescription<id>::ObservedType;
    using ActualObservedType = std::conditional_t<nullable, std::optional<ElementType>, ElementType>;

    const auto compareRawValues = [](ElementType lhs, ElementType rhs)
    {
        if constexpr(order == SortOrder::Ascending)
            return lhs < rhs;
        else
            return lhs > rhs;
    };

    const auto compareValues = [=](auto &&lhs, auto &&rhs)
    {

        if constexpr(nullable)
        {
            const auto lhsValid = lhs.has_value();
            const auto rhsValid = rhs.has_value();
            if(lhsValid && rhsValid)
                return compareRawValues(*lhs, *rhs);
            if(!lhsValid && !rhsValid) // null < null
                return false;
            if(lhsValid) // lhs < null
                return nulls == NullPosition::After;
            if(rhsValid) // rhs < null
                return nulls == NullPosition::Before;

            throw std::runtime_error("sort internal error");
        }
        else
        {
            return compareRawValues(lhs, rhs);
        }
    };

    // TODO: special faster path (no-copy) can be provided for single chunk columns
    // however gains will rather be limited, as sorting and permuting data dominates
    const auto valuesAsVector = toVector<ActualObservedType>(sortBy);
    std::stable_sort(indices.begin(), indices.end(), [&](int64_t lhsIndex, int64_t rhsIndex)
    {
        const auto &lhs = valuesAsVector[lhsIndex];
        const auto &rhs = valuesAsVector[rhsIndex];
        return compareValues(lhs, rhs);
    });
}

void sortPermutation(Permutation &indices, const arrow::Column &sortBy, SortOrder order, NullPosition nullPosition)
{
    // Hoist runtime constant values to the compile-time values.
    visitType(*sortBy.type(), [&] (auto id)
    {
        dispatch(nullPosition, [&] (auto nullPosition)
        {
            dispatch(order, [&] (auto orderC)
            {
                dispatch(sortBy.null_count() != 0, [&] (auto hasNullsC)
                {
                    sortPermutationInner<id.value, hasNullsC.value, orderC.value, nullPosition.value>(indices, sortBy);
                });
            });
        });
    });
}

std::vector<int64_t> sortPermutation(std::vector<SortBy> sortBy)
{
    if(sortBy.empty())
        throw std::runtime_error("no column to sort by");

    Permutation indices = iotaVector<int64_t>(sortBy.front().column->length());

    std::reverse(sortBy.begin(), sortBy.end());
    for(auto &sortBy : sortBy)
    {
        sortPermutation(indices, *sortBy.column, sortBy.order, sortBy.nulls);
    }

    return indices;
}

}


std::shared_ptr<arrow::Array> permuteToArray(const std::shared_ptr<arrow::Column> &column, const Permutation &indices)
{
    if(isPermuteId(indices) && column->data()->num_chunks() == 1)
        return column->data()->chunk(0);

    return permuteInnerToArray(column, indices);
}

std::shared_ptr<arrow::Column> permute(const std::shared_ptr<arrow::Column> &column, const Permutation &indices)
{
    if(isPermuteId(indices))
        return column;

    return permuteInner(column, indices);
}

std::shared_ptr<arrow::Table> permute(const std::shared_ptr<arrow::Table> &table, const Permutation &indices)
{
    if(isPermuteId(indices))
        return table;

    return permuteInner(table, indices);
}

std::shared_ptr<arrow::Table> sortTable(const std::shared_ptr<arrow::Table> &table, const std::vector<SortBy> &sortBy)
{
    auto permutation = sortPermutation(sortBy);
    return permute(table, permutation);
}
