#include "ArrowUtilities.h"

using namespace std::literals;

//DFH_EXPORT std::shared_ptr<arrow::TimestampType> timestampTypeSingleton = std::make_shared<arrow::TimestampType>(arrow::TimeUnit::NANO);

std::shared_ptr<arrow::Column> toColumn(std::shared_ptr<arrow::ChunkedArray> chunks, std::string name /*= "col"*/)
{
	auto field = arrow::field(name, chunks->type(), chunks->null_count() != 0);
	return std::make_shared<arrow::Column>(field, std::move(chunks));
}

std::shared_ptr<arrow::Column> toColumn(std::shared_ptr<arrow::Array> array, std::string name /*= "col"*/)
{
    auto field = arrow::field(name, array->type(), array->null_count() != 0);
    return std::make_shared<arrow::Column>(field, std::move(array));
}

std::vector<std::shared_ptr<arrow::Column>> getColumns(const arrow::Table &table)
{
    std::vector<std::shared_ptr<arrow::Column>> columns;
    for(int i = 0; i < table.num_columns(); i++)
    {
        columns.push_back(table.column(i));
    }
    return columns;
}

std::shared_ptr<arrow::Column> getColumn(const arrow::Table &table, std::string_view name)
{
    auto columns = getColumns(table);
    for(auto &col : columns)
        if(col->name() == name)
            return col;

    auto names = transformToVector(columns, [](auto &col) { return col->name(); });
    THROW("Failed to find column by name `{}`. Available columns: `{}`", 
        name, names);
}

std::unordered_map<std::string, std::shared_ptr<arrow::Column>> getColumnMap(const arrow::Table &table)
{
    std::unordered_map<std::string, std::shared_ptr<arrow::Column>> ret;

    for(auto &&column : getColumns(table))
        ret[column->name()] = column;

    return ret;
}

std::shared_ptr<arrow::Field> setNullable(bool nullable, std::shared_ptr<arrow::Field> field)
{
    if(field->nullable() == nullable)
        return field;

    return arrow::field(field->name(), field->type(), nullable, field->metadata());
}

std::shared_ptr<arrow::Schema> setNullable(bool nullable, std::shared_ptr<arrow::Schema> schema)
{
    std::vector<std::shared_ptr<arrow::Field>> newFields;
    for(auto &&field : schema->fields())
        newFields.push_back(setNullable(nullable, field));

    return arrow::schema(newFields, schema->metadata());
}

std::shared_ptr<arrow::Table> tableFromArrays(std::vector<PossiblyChunkedArray> arrays, std::vector<std::string> names, std::vector<bool> nullables)
{
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunkedArrays;
    for(auto &someArray: arrays)
    {
        auto chunk = visit(overloaded {
            [&] (std::shared_ptr<arrow::Array> array) { return std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{array}); },
            [&] (std::shared_ptr<arrow::ChunkedArray> array) { return array; }
            }, someArray);
        chunkedArrays.push_back(chunk);
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    for(int i = 0; i < chunkedArrays.size(); i++)
    {
        const auto arr = chunkedArrays.at(i);
        if(names.size() <= i)
            names.push_back("col" + std::to_string(i));
        if(nullables.size() <= i)
            nullables.push_back(arr->null_count());

        fields.push_back(arrow::field(names.at(i), arr->type(), nullables.at(i)));
    }

    std::vector<std::shared_ptr<arrow::Column>> columns;
    for(int i = 0; i < chunkedArrays.size(); i++)
    {
        auto field = fields.at(i);
        auto chunks = chunkedArrays.at(i);
        auto column = std::make_shared<arrow::Column>(field, chunks);
        columns.push_back(column);
    }


    auto schema = arrow::schema(std::move(fields));
    return arrow::Table::Make(schema, std::move(columns));
}

std::shared_ptr<arrow::Table> tableFromColumns(const std::vector<std::shared_ptr<arrow::Column>> &columns)
{
    auto fields = transformToVector(columns, [](auto &&col) { return col->field(); });
    auto schema = arrow::schema(fields);
    return tableFromColumns(columns, schema);
}

std::shared_ptr<arrow::Table> tableFromColumns(const std::vector<std::shared_ptr<arrow::Column>> &columns, const std::shared_ptr<arrow::Schema> &schema)
{
    auto rowCount = maxElementValue(columns, int64_t(0), [](auto &&col) { return col->length(); });
    auto tableColumns = transformToVector(columns, [&](auto &&col)
    {
        // Columns with proper length can go into table as-is
        if(col->length() == rowCount)
            return col;

        auto nullCountToAdd = rowCount - col->length();
        auto padding = makeNullsArray(col->type(), nullCountToAdd);
        auto chunks = col->data()->chunks();
        chunks.push_back(padding);

        // Adjust column field information - if we pad it with nulls, it must be noted.
        auto field = setNullable(true, col->field());
        return std::make_shared<arrow::Column>(field, chunks);
    });

    return arrow::Table::Make(schema, tableColumns);
}

DynamicJustVector toJustVector(const arrow::ChunkedArray &chunkedArray)
{
    return visitType(*chunkedArray.type(), [&] (auto id)  -> DynamicJustVector
    {
        using T = typename TypeDescription<id.value>::ObservedType;

        std::vector<T> ret;
        ret.reserve(chunkedArray.length() - chunkedArray.null_count());
        iterateOver<id.value>(chunkedArray, 
            [&] (auto elem) { ret.push_back(elem); },
            [] {});
        return ret;
    });
}
DynamicJustVector toJustVector(const arrow::Column &column)
{
    return toJustVector(*column.data());
}

DynamicField arrayAt(const arrow::Array &array, int32_t index)
{
    return visitArray(array, [&](auto *array) -> DynamicField
    {
        if(array->IsValid(index))
            return arrayValueAtTyped(*array, index);
        else
            return std::nullopt;
    });
}

DynamicField arrayAt(const arrow::ChunkedArray &array, int64_t index)
{
    auto [chunk, indexInChunk] = locateChunk(array, index);
    return arrayAt(*chunk, indexInChunk);
}

DynamicField arrayAt(const arrow::Column &column, int64_t index)
{
    return arrayAt(*column.data(), index);
}

std::pair<std::shared_ptr<arrow::Array>, int32_t> locateChunk(const arrow::ChunkedArray &chunkedArray, int64_t index)
{
    validateIndex(chunkedArray, index);

    int64_t i = index;
    for(auto &chunk : chunkedArray.chunks())
    {
        const auto length = chunk->length(); // Note: having this assigned to variable greatly improves performance (MSVC)
        if(i < length)
            return {chunk, static_cast<int32_t>(i)};

        i -= length;
    }

    throw std::runtime_error(__FUNCTION__ + ": unexpected failure"s);
}

std::vector<DynamicField> rowAt(const arrow::Table &table, int64_t index)
{
    std::vector<DynamicField> ret;
    for(int i = 0; i < table.num_columns(); i++)
    {
        auto column = table.column(i);
        ret.push_back(arrayAt(*column, index));
    }
    return ret;
}

void validateIndex(const arrow::Array &array, int64_t index)
{
    validateIndex(array.length(), index);
}

void validateIndex(const arrow::ChunkedArray &array, int64_t index)
{
    validateIndex(array.length(), index);
}

void validateIndex(const arrow::Column &column, int64_t index)
{
    validateIndex(column.length(), index);
}

template<typename SharedPtrToType>
struct GetTypeS {};

template<typename T>
struct GetTypeS<std::shared_ptr<T>> { using type = T; };

template<typename SharedPtrToType>
using GetType = typename GetTypeS<std::decay_t<SharedPtrToType>>::type;

std::shared_ptr<arrow::Array> makeNullsArray(TypePtr type, int64_t length)
{
    return visitDataType(type, [&](auto &&typeDer)
    {
        using Type = GetType<decltype(typeDer)>;
        static_assert(std::is_base_of_v<arrow::DataType, GetType<decltype(typeDer)>>);
        auto builder = makeBuilder<Type>(typeDer);
        for(int64_t i = 0; i < length; i++)
            builder->AppendNull();
        return finish(*builder);
    });
}

std::shared_ptr<arrow::ArrayBuilder> makeBuilder(const TypePtr &type)
{
    return visitDataType(type, [&](auto &&typeDer) -> std::shared_ptr<arrow::ArrayBuilder>
    {
        return makeBuilder(typeDer);
    });
}

BitmaskGenerator::BitmaskGenerator(int64_t length, bool initialValue) : length(length)
{
    auto bytes = arrow::BitUtil::BytesForBits(length);
    std::tie(buffer, data) = allocateBuffer<uint8_t>(bytes);
    std::memset(data, initialValue ? 0xFF : 0, bytes);
    // TODO: above sets by bytes, the last byte should have only part of bits set
}

bool BitmaskGenerator::get(int64_t index)
{
    return arrow::BitUtil::GetBit(data, index);
}

void BitmaskGenerator::set(int64_t index)
{
    arrow::BitUtil::SetBit(data, index);
}

void BitmaskGenerator::clear(int64_t index)
{
    arrow::BitUtil::ClearBit(data, index);
}

ChunkAccessor::ChunkAccessor(const arrow::ChunkedArray &array)
    : chunks(array.chunks())
{
    int64_t index = 0;
    for(auto &chunk : chunks)
    {
        chunkStartIndices.push_back(index);
        index += chunk->length();
    }
}

ChunkAccessor::ChunkAccessor(const arrow::Column &column)
    : ChunkAccessor(*column.data())
{}

std::pair<const arrow::Array *, int32_t> ChunkAccessor::locate(int64_t index) const
{
    auto itr = std::upper_bound(chunkStartIndices.begin(), chunkStartIndices.end(), index);
    if(itr != chunkStartIndices.begin())
    {
        auto chunkStart = itr - 1;
        auto chunkIndex = std::distance(chunkStartIndices.begin(), chunkStart);
        auto indexWithinChunk = (int32_t)(index - *chunkStart);
        return { chunks[chunkIndex].get(), indexWithinChunk };
    }
    else
        throw std::runtime_error("wrong index");
}

bool ChunkAccessor::isNull(int64_t index)
{
    auto[chunk, chunkIndex] = locate(index);
    return chunk->IsNull(chunkIndex);
}

std::string std::to_string(const Timestamp &t)
{
    std::ostringstream out;
    out << date::format("%F", t);
    return out.str();
}

Timestamp::Timestamp(date::year_month_day ymd)
    : Base(date::sys_days(ymd))
{
}

std::shared_ptr<arrow::Column> consolidate(std::shared_ptr<arrow::Column> column)
{
    if(column->data()->num_chunks() <= 1)
        return column;

    return visitType(*column->type(), [&](auto id)
    {
        using TD = TypeDescription<id.value>;
        using ArrowT = typename TD::ArrowType;
        using Builder = typename TD::BuilderType;

        auto builder = makeBuilder(std::dynamic_pointer_cast<ArrowT>(column->type()));
        iterateOver<id.value>(*column, [&](auto &&elem)
        {
            append(*builder, elem);
        },
            [&]()
        {
            builder->AppendNull();
        });

        auto arr = finish(*builder);
        return std::make_shared<arrow::Column>(column->field(), arr);
    });
}
