#include "Utils.h"
#include <numeric>

std::optional<Timestamp> parseTimestamp(std::string_view text)
{
    std::istringstream input((std::string)text);

    Timestamp out;
    input >> date::parse("%F", out);
    if(input && input.rdbuf()->in_avail() == 0)
        return out;
    return std::nullopt;
}


template<typename Range, typename Reader>
void printList(std::ostream &out, Range &&r, Reader &&f)
{
    auto end = std::end(r);
    auto itr = std::begin(r);
    if(itr != end)
    {
        out << std::invoke(f, *itr++);
    }

    while(itr != end)
    {
        out << "\t" << std::invoke(f, *itr++);
    }
}


template<typename T>
std::string formatColumnElem(const std::optional<T> &elem);

template<typename T>
std::string formatColumnElem(const T &elem)
{
    return std::to_string(elem);
}
std::string formatColumnElem(const std::string_view &elem)
{
    return '"' + std::string(elem) + '"';
}
std::string formatColumnElem(const ListElemView &elem)
{
    std::ostringstream out;
    out << "[";

    visitType4(elem.array->type(), [&](auto id)
    {
        if(elem.length)
        {
            auto value = tryArrayValueAt<id.value>(*elem.array, elem.offset + 0);
            out << formatColumnElem(value);
        }

        for(int i = 1; i < elem.length; i++)
        {
            out << ", ";
            auto value = tryArrayValueAt<id.value>(*elem.array, elem.offset + i);
            out << formatColumnElem(value);
        }
    });

    out << "]";
    return out.str();
}

std::string formatColumnElem(const std::nullopt_t &)
{
    return "null"s;
}

template<typename T>
std::string formatColumnElem(const std::optional<T> &elem)
{
    if(elem)
        return formatColumnElem(*elem);
    return "null"s;
}

std::vector<std::vector<std::string>> formatElements(const arrow::Table &table, int rows)
{
    std::vector<std::vector<std::string>> cellsByRow;
    auto cols = getColumns(table);

    cellsByRow.push_back(transformToVector(cols, [](auto col) { return col->name(); }));

    int64_t partsSize = rows / 2;
    auto printedElement = [&](const arrow::Column &col, int64_t row)
    {
        return visitType4(col.type(), [&](auto id) -> std::string
        {
            auto[chunk, chunkIndex] = locateChunk(*col.data(), row);
            if(chunk->IsValid(chunkIndex))
            {
                const auto value = arrayValueAt<id.value>(*chunk, chunkIndex);
                return formatColumnElem(value);
            }
            else
                return formatColumnElem(std::nullopt);
        });
    };

    
    auto printRow = [&] (int64_t row) -> std::vector<std::string>
    {
        return transformToVector(cols, [&](const auto &col) -> std::string
        {
            return printedElement(*col, row);
        });
    };

    for(int64_t row = 0; row < partsSize && row < table.num_rows(); row++)
        cellsByRow.push_back(printRow(row));
    for(int64_t row = std::max<int64_t>(partsSize, std::max<int64_t>(0, table.num_rows() - partsSize)); row < table.num_rows(); row++)
        cellsByRow.push_back(printRow(row));

    return cellsByRow;
}

void uglyPrint(const arrow::Table &table, std::ostream &out, int rows /*= 20*/)
{
    std::vector<std::vector<std::string>> cells = formatElements(table, rows);

    std::vector<size_t> cellWidths(table.num_columns());

    for(int64_t row = 0; row < (int64_t)cells.size(); row++)
    {
        for(int64_t col = 0; col < table.num_columns(); ++col)
        {
            cellWidths[col] = std::max(cellWidths[col], cells[row][col].size());
        }
    }

    auto rowWidth = std::accumulate(cellWidths.begin(), cellWidths.end(), 0_z);

    auto printRow = [&](int64_t row)
    {
        auto rowCells = cells.at(row);
        out << "| ";
        for(int64_t col = 0; col < (int64_t)rowCells.size(); ++col)
        {
            out.width(cellWidths.at(col));
            out << rowCells.at(col);
            out << " | ";
        }
        out << std::endl;
    };


    int64_t partsSize = rows / 2;

    for(int64_t row = 0; row < partsSize && row < (int64_t)cells.size(); row++)
        printRow(row);

    auto skippedRows = table.num_rows() - partsSize * 2;
    if(skippedRows > 0)
        out << "... " << skippedRows << " more rows ...\n";

    for(int64_t row = partsSize + 1; row < (int64_t)cells.size(); row++)
        printRow(row);

    out << "[" << table.num_rows() << " rows x " << table.num_columns() << " cols]" << std::endl;
}

void uglyPrint(const std::shared_ptr<arrow::Column> &column, std::ostream &out /*= std::cout*/, int rows /*= 20*/)
{
    auto table = tableFromColumns({column});
    uglyPrint(*table, out, rows);
}
