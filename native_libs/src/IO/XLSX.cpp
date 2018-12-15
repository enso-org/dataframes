#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include "XLSX.h"

#include "Core/ArrowUtilities.h"
#include "Core/Common.h"
#include "Core/Error.h"

#include <arrow/builder.h>
#include <arrow/table.h>

#include <fstream>

#ifdef DISABLE_XLSX

#pragma message("Note: DataframeHelper is being compiled without XLSX support.")  

namespace
{
    std::shared_ptr<arrow::Table> readXlsxInput(std::istream &input, HeaderPolicy header, std::vector<ColumnType> columnTypes)
    {
        THROW("The library was compiled without XLSX support!");
    }
    void writeXlsx(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy)
    {
        THROW("The library was compiled without XLSX support!");
    }
}

#else // DISABLE_XLSX

#include <xlnt/xlnt.hpp>

namespace
{
    using namespace std::literals;

    struct ColumnBuilderBase
    {
        virtual void addFromCell(const xlnt::cell &field) = 0;
        virtual void addMissing() = 0;
        virtual void reserve(int64_t count) = 0;
        virtual std::shared_ptr<arrow::Array> finish() = 0;
    };

    template<arrow::Type::type type>
    struct ColumnBuilder : ColumnBuilderBase
    {
        bool nullable{};
        std::shared_ptr<BuilderFor<type>> builder;

        ColumnBuilder(bool nullable)
            : nullable(nullable) 
            , builder(makeBuilder(getTypeSingleton<type>()))
        {}

        virtual void addFromCell(const xlnt::cell &field) override
        {
            if(field.has_value())
            {
                if constexpr(type == arrow::Type::STRING)
                    checkStatus(builder->Append(field.to_string()));
                else if constexpr(type == arrow::Type::INT64)
                    checkStatus(builder->Append(field.value<long long int>())); // NOTE: cannot be int64_t -- no such overload, fails on GCC
                else if constexpr(type == arrow::Type::DOUBLE)
                    checkStatus(builder->Append(field.value<double>()));
                else if constexpr(type == arrow::Type::TIMESTAMP)
                {
                    using namespace std::chrono;
                    using namespace date;
                    auto v = field.value<xlnt::datetime>();
                    auto day = year(v.year)/v.month/v.day;
                    auto timeOfDay = hours(v.hour) + minutes(v.minute) + seconds(v.second) + microseconds(v.microsecond);
                    Timestamp t{(sys_days)day + timeOfDay};
                    checkStatus(builder->Append(t.toStorage()));
                }
                else
                    throw std::runtime_error("wrong type");
            }
            else
                addMissing();
        }
        virtual void addMissing() override
        {
            if(nullable)
                checkStatus(builder->AppendNull());
            else
                checkStatus(builder->Append(defaultValue<type>()));
        }
        virtual void reserve(int64_t count) override
        {
            checkStatus(builder->Reserve(count));
        }

        virtual std::shared_ptr<arrow::Array> finish() override
        {
            return ::finish(*builder);
        }
    };

    std::shared_ptr<arrow::Table> readXlsxInput(std::istream &input, HeaderPolicy header, std::vector<ColumnType> columnTypes)
    {
        try
        {
            xlnt::workbook wb;
            wb.load(input);
            const auto sheet = wb.active_sheet();

            // We keep the object under unique_ptr, so there will be
            // no leak if exception is thrown before the end of function
            const auto rowCount = (int64_t)sheet.highest_row();
            const auto columnCount = (int32_t)sheet.highest_column().index;

            // If there is no type info for column, default to non-nullable Text (it always works)
            if((int)columnTypes.size() < columnCount)
            {
                const ColumnType nonNullableText{ std::make_shared<arrow::StringType>(), false, false };
                columnTypes.resize(columnCount, nonNullableText);
            }
            const auto names = decideColumnNames(columnCount, header, [&](int column)
            {
                return sheet.cell(column + 1, 0 + 1).to_string();
            });
            const bool useFirstRowAsHeaders = holds_alternative<TakeFirstRowAsHeaders>(header);

            // setup column builders
            std::vector<std::unique_ptr<ColumnBuilderBase>> columnBuilders;
            for(auto columnType : columnTypes)
            {
                auto ptr = visitType(*columnType.type, [&](auto id) -> std::unique_ptr<ColumnBuilderBase>
                {
                    return std::make_unique<ColumnBuilder<id.value>>(columnType.nullable);
                });
                ptr->reserve(rowCount);
                columnBuilders.push_back(std::move(ptr));
            }
            for(auto i = columnBuilders.size(); i < columnCount; i++)
            {
                columnBuilders.push_back(std::make_unique<ColumnBuilder<arrow::Type::STRING>>(false));
            }

            for(int column = 0; column < columnCount; column++)
            {
                for(int row = useFirstRowAsHeaders; row < rowCount; row++)
                {
                    xlnt::cell_reference cellPos(column + 1, row + 1);
                    if(sheet.has_cell(cellPos))
                        columnBuilders[column]->addFromCell(sheet.cell(cellPos));
                    else
                        columnBuilders[column]->addMissing();
                }
            }

            std::vector<std::shared_ptr<arrow::Array>> arrays;
            for(auto &builder : columnBuilders)
                arrays.push_back(builder->finish());


            return buildTable(names, arrays, columnTypes);

        }
        catch(std::exception &e)
        {
            THROW("failed to load XLSX: {}", e.what());
        }
    }

    void writeXlsx(std::ostream &out, const arrow::Table &table, GeneratorHeaderPolicy headerPolicy)
    {
        xlnt::workbook wb;
        auto sheet = wb.active_sheet();
        sheet.title("Table");

        if(headerPolicy == GeneratorHeaderPolicy::GenerateHeaderLine)
            for(int column = 0; column < table.num_columns(); column++)
                sheet.cell(column + 1, 1).value(table.column(column)->name());

        for(int column = 0; column < table.num_columns(); column++)
        {
            int32_t row = headerPolicy == GeneratorHeaderPolicy::GenerateHeaderLine;
            const auto writeValue = [&](auto &&field)
            {
                using FieldType = std::decay_t<decltype(field)>;
                auto cell = sheet.cell(column + 1, row + 1);
                // NOTE: workaround for GCC: otherwise call to xlnt::cell::value would be ambiguous
                // as int64_t is long int and there is no such overload (just ints and long long ints)
                if constexpr(std::is_same_v<int64_t, FieldType>)
                    cell.value((long long)field);
                else if constexpr(std::is_same_v<std::string_view, FieldType>)
                    cell.value(std::string(field));
                else if constexpr(std::is_same_v<Timestamp, FieldType>)
                {
                    using namespace date;
                    auto daypoint = floor<days>(field);
                    auto ymd = year_month_day(daypoint);   // calendar date
                    time_of_day tod = make_time(field - daypoint); // Yields time_of_day type

                    // Obtain individual components as integers
                    auto y = (int)ymd.year();
                    auto m = (int)(unsigned)ymd.month();
                    auto d = (int)(unsigned)ymd.day();
                    auto h = (int)tod.hours().count();
                    auto min = (int)tod.minutes().count();
                    auto s = (int)tod.seconds().count();
                    auto us = (int)std::chrono::duration_cast<std::chrono::microseconds>(tod.subseconds()).count();

                    xlnt::datetime timestamp{ y, m, d, h, min, s, us };
                    cell.value(timestamp);
                }
                else
                    cell.value(field);
                ++row;
            };
            const auto writeNull = [&]
            {
                ++row;
            };

            iterateOverGeneric(*table.column(column), writeValue, writeNull);
        }

        wb.save(out);
    }
}

#endif // DISABLE_XLSX

std::string FormatXLSX::fileSignature() const
{
    // Note: Office Open XML file format is actually just a zipped XML.
    // Because of that we test for ZIP file signature.
    // But that might just be some random zip or other office document (like MS Word).
    return { 0x50, 0x4B, 0x03, 0x04 };
}

std::shared_ptr<arrow::Table> FormatXLSX::read(std::string_view filePath, const XlsxReadOptions &options) const
{
    try
    {
        auto input = openFileToRead(filePath);
        return readXlsxInput(input, options.header, options.columnTypes);
    }
    catch(std::exception &e)
    {
        THROW("Failed to load file {} as XLSX: {}", filePath, e);
    }
}

void FormatXLSX::write(std::string_view filePath, const arrow::Table &table, const XlsxWriteOptions &options) const
{
    auto out = openFileToWrite(filePath);
    return writeXlsx(out, table, options.headerPolicy);
}

std::vector<std::string> FormatXLSX::fileExtensions() const
{
    return { "xlsx" };
}
