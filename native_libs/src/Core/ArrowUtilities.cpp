#include "ArrowUtilities.h"


std::shared_ptr<arrow::Table> tableFromArrays(std::vector<std::shared_ptr<arrow::Array>> arrays, std::vector<std::string> names, std::vector<bool> nullables)
{
    std::vector<std::shared_ptr<arrow::Field>> fields;

    for(int i = 0; i < arrays.size(); i++)
    {
        const auto arr = arrays.at(i);
        if(names.size() <= i)
            names.push_back("col" + std::to_string(i));
        if(nullables.size() <= i)
            nullables.push_back(arr->null_count());

        fields.push_back(arrow::field(names.at(i), arr->type(), nullables.at(i)));
    }

    auto schema = arrow::schema(std::move(fields));
    return arrow::Table::Make(schema, std::move(arrays));
}
