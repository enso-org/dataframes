#include "Utils.h"

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

void uglyPrint(const arrow::Table &table, std::ostream &out, int rows /*= 20*/)
{

    auto cols = getColumns(table);
    out << "\t| ";
    printList(out, cols, [](auto col){ return col->name(); });
    out << '\n';
    for(int i = 0; i < 80; i++)
        out << "-";
    out << '\n';

    int64_t partsSize = rows / 2;

    auto printedElement = [&](const arrow::Column &col, int row)
    {
        return visitType4(col.type(), [&](auto id) -> std::string
        {
            auto [chunk, chunkIndex] = locateChunk(*col.data(), row);
            if(chunk->IsValid(chunkIndex))
            {
                const auto value = arrayValueAt<id.value>(*chunk, row);
                return formatColumnElem(value);
            }
            else
                return formatColumnElem(std::nullopt);
        });
    };

    auto printRow = [&](int64_t row)
    {
        out << row << "\t| ";
        printList(out, cols, [&](const auto &col) -> std::string
        {
            return printedElement(*col, row);
        });
        out << '\n';
    };

    for(int64_t row = 0; row < partsSize && row < table.num_rows(); row++)
        printRow(row);
    if(table.num_rows() > partsSize * 2)
        out << "... " << (table.num_rows() - partsSize * 2) << " more rows ...\n";
    for(int64_t row = std::max<int64_t>(partsSize, std::max<int64_t>(0, table.num_rows() - partsSize)); row < table.num_rows(); row++)
        printRow(row);

    out << "[" << table.num_rows() << " rows x " << table.num_columns() << " cols]" << std::endl;
}
