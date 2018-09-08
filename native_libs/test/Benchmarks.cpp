#define NOMINMAX
#include <boost/test/unit_test.hpp>

#include <rapidjson/filereadstream.h>

#include <cstdio>
#include <fstream>
#include <random>

#include "IO/csv.h"
#include "IO/Feather.h"
#include "Analysis.h"
#include "Core/Benchmark.h"
#include "Processing.h"
#include "Fixture.h"

using namespace std::literals;
using boost::unit_test_framework::disabled;

DFH_EXPORT std::shared_ptr<arrow::Column> calculateMin(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMax(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMean(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMedian(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateVariance(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateQuantile(const arrow::Column &column, double q);
DFH_EXPORT double calculateCorrelation(const arrow::Column &xCol, const arrow::Column &yCol);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateCorrelation(const arrow::Table &table, const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Table> calculateCorrelationMatrix(const arrow::Table &table);

struct BenchmarkingFixture
{
	DataGenerator g;
	MeasureAtLeast policy{10, 15s};
	std::vector<MeasureSeries> measures;

	std::shared_ptr<arrow::Table> numericTable = g.generateNumericTable(10'000'000);


	template<typename F, typename ...Args>
	auto benchmark(std::string text, F &&f, Args && ...args)
	{
		auto result = ::measure(text, policy, std::forward<F>(f), std::forward<Args>(args)...);
		measures.push_back(result.second);
		return result;
	}

	~BenchmarkingFixture()
	{
		std::cout << "Run " << measures.size() << " benchmarks:" << std::endl;
		for(auto &&measure : measures)
		{
			std::cout << "  " << measure.name << ":\t" << measure.bestTime().count() / 1000.0 << " ms" << std::endl;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE(Bench, BenchmarkingFixture, *disabled());

BOOST_AUTO_TEST_CASE(GeneralBenchmark)
{
	benchmark("write feather", [&] { return saveTableToFeatherFile("numtable-temp.feather", *numericTable); });
	benchmark("read feather", [&] { return loadTableFromFeatherFile("numtable-temp.feather"); });
	benchmark("count values", [&] { return countValues(*numericTable->column(1)); });
	benchmark("calculate min", [&] { return calculateMin(*numericTable->column(1)); });
	benchmark("calculate max", [&] { return calculateMax(*numericTable->column(1)); });
	auto [mean, blah3] = benchmark("calculate mean", [&] { return calculateMean(*numericTable->column(1)); });
	auto [medianCol, blah4] = benchmark("calculate median", [&] { return calculateMedian(*numericTable->column(1)); });
	benchmark("calculate variance", [&] { return calculateVariance(*numericTable->column(1)); });
	benchmark("calculate std", [&] { return calculateStandardDeviation(*numericTable->column(1)); });
	benchmark("calculate sum", [&] { return calculateSum(*numericTable->column(1)); });
	benchmark("calculate quantile 1/3", [&] { return calculateQuantile(*numericTable->column(1), 1.0/3.0); });
	benchmark("calculate correlationMatrix", [&] { return calculateCorrelationMatrix(*numericTable); });
 	benchmark("num table to csv", [&] { return generateCsv("numtable-temp.csv", *numericTable);});
 	benchmark("num table from csv", [&] { return parseCsvFile("numtable-temp.csv");});

	auto intColumn1 = g.generateColumn(arrow::Type::INT64, 10'000'000, "intsNonNull");
	BOOST_CHECK_EQUAL(intColumn1->null_count(), 0);

	auto v = benchmark("int column to vector", [&] { return toVector<int64_t>(*intColumn1); }).first;
	auto [intVector, blah2] = benchmark("int column from vector", [&] { return toColumn(v); });
}

BOOST_AUTO_TEST_CASE(FilterBigFile0)
{
	const auto jsonQuery = R"({"predicate": "gt", "arguments": [ {"column": "NUM_INSTALMENT_NUMBER"}, 50 ] } )";
	auto table = loadTableFromFeatherFile("C:/installments_payments.feather");
	for(int i  = 0; i < 2000; i++)
	{
		measure("filter installments_payments", [&]
		{
			auto table2 = filter(table, jsonQuery);
			// 			std::ofstream out{"tescik.csv"};
			// 			generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
		});
	}
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(MapBigFile)
{
	const auto jsonQuery = R"({"operation": "plus", "arguments": [ {"column": "NUM_INSTALMENT_NUMBER"}, 50 ] } )";
	auto table = loadTableFromFeatherFile("C:/installments_payments.feather");
	for(int i  = 0; i < 2000; i++)
	{
		measure("map installments_payments", [&]
		{
			auto column = each(table, jsonQuery);
			// 			std::ofstream out{"tescik.csv"};
			// 			generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
		});
	}
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(FilterBigFile1)
{
	const auto jsonQuery = R"({"predicate": "eq", "arguments": [ {"column": "NAME_TYPE_SUITE"}, "Unaccompanied" ] } )";

	auto table = loadTableFromFeatherFile("C:/temp/application_train.feather");


	for(int i  = 0; i < 2000; i++)
	{
		measure("filter application_train", [&]
		{
			auto table2 = filter(table, jsonQuery);
			// 			std::ofstream out{"tescik.csv"};
			// 			generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
		});
	}
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(DropNABigFile)
{
	auto csv = parseCsvFile("F:/dev/csv/application_train.csv");
	auto table = csvToArrowTable(std::move(csv), TakeFirstRowAsHeaders{}, {});

	auto table2 = dropNA(table);

	auto row = rowAt(*table, 307'407);

	{
		std::ofstream out{ "trained_filtered_nasze.csv" };
		if(!out)
			throw std::runtime_error("Cannot write to file ");
		generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
	}

	measure("drop NA from application_train", 5000, [&]
	{
		auto table2 = dropNA(table);
	});
}

BOOST_AUTO_TEST_CASE(FillNABigFile)
{
	auto csv = parseCsvFile("F:/dev/csv/application_train.csv");
	auto table = csvToArrowTable(std::move(csv), TakeFirstRowAsHeaders{}, {});

	std::unordered_map<std::string, DynamicField> valuesToFillWith;
	for(auto column : getColumns(*table))
	{
		valuesToFillWith[column->name()] = adjustTypeForFilling("80"s, *column->type());
	}

	measure("fill NA from application_train", 5000, [&]
	{
		auto table2 = dropNA(table);
	});
}

BOOST_AUTO_TEST_CASE(FilterBigFile)
{
	const auto jsonQuery = R"({"predicate": "gt", "arguments": [ {"column": "NUM_INSTALMENT_NUMBER"}, {"operation": "plus", "arguments": [50, 1]} ] } )";
	auto table = loadTableFromFeatherFile("C:/installments_payments.feather");
	measure("filter installments_payments", 2000, [&]
	{
		auto table2 = filter(table, jsonQuery);
		//  			std::ofstream out{"tescik100.csv"};
		//  			generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
	});
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(StatisticsBigFile)
{
	const auto jsonQuery = R"({"predicate": "gt", "arguments": [ {"column": "NUM_INSTALMENT_NUMBER"}, {"operation": "plus", "arguments": [50, 1]} ] } )";
	auto table = loadTableFromFeatherFile("C:/installments_payments.feather");
	measure("median installments_payments", 2000, [&]
	{
		calculateCorrelationMatrix(*table);
		//		toJustVector()
		// 		auto m = calculateQuantile(*table->column(7), 0.7);
		// 		std::cout << "median: " << toVector<double>(*m).at(0) << std::endl;

	});
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(ParseBigFile)
{
	measure("parse big file", 20, [&]
	{
		auto csv = parseCsvFile("C:/installments_payments.csv");
		auto table = csvToArrowTable(std::move(csv), TakeFirstRowAsHeaders{}, {});

		//saveTableToFeatherFile("C:/installments_payments.feather", *table);
	});

	auto integerType = std::make_shared<arrow::Int64Type>();
	auto doubleType = std::make_shared<arrow::DoubleType>();
	auto textType = std::make_shared<arrow::StringType>();

	std::vector<ColumnType> expectedTypes
	{
		ColumnType{ integerType, false, true },
		ColumnType{ integerType, false, true },
		ColumnType{ doubleType , false, true },
		ColumnType{ integerType, false, true },
		ColumnType{ doubleType , false, true },
		ColumnType{ doubleType , false, true },
		ColumnType{ doubleType , false, true },
		ColumnType{ doubleType , false, true },
	};
	auto csv = parseCsvFile("C:/installments_payments.csv");
	auto table = csvToArrowTable(std::move(csv), TakeFirstRowAsHeaders{}, {});

	//std::vector<ColumnType> typesEncountered;
	for(int i = 0; i < table->num_columns(); i++)
	{
		const auto column = table->column(i);
		//typesEncountered.emplace_back(column->type(), column->field()->nullable());

		BOOST_CHECK_EQUAL(expectedTypes.at(i).type->id(), column->type()->id());
		BOOST_CHECK_EQUAL(expectedTypes.at(i).nullable, column->field()->nullable());
	}
}


BOOST_AUTO_TEST_CASE(WriteBigFile)
{
	const auto path = R"(E:/hmda_lar-florida.csv)";
	auto csv = parseCsvFile(path);
	auto table = csvToArrowTable(csv, TakeFirstRowAsHeaders{}, {});
	// 	for(int i  = 0; i < 20; i++)
	// 	{
	// 		measure("load big file contents1", getFileContents, path);
	// 		measure("load big file contents2", get_file_contents, path);
	// 	}

	for(int i = 0; i < 10; i++)
	{
		measure("write big file", [&]
		{
			std::ofstream out{"ffffff.csv"};
			generateCsv(out, *table, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QueteAllFields);
		});
	}
}

BOOST_FIXTURE_TEST_CASE(InterpolateBigColumn, DataGenerator)
{
    auto doubles30pct = generateColumn(arrow::Type::DOUBLE, 10'000'000, "double", 0.3);
    auto doubles90pct = generateColumn(arrow::Type::DOUBLE, 10'000'000, "double", 0.9);
    MeasureAtLeast p{100, 5s};
    measure("interpolating doubles with 30% nulls", p, [&]
    {
        return interpolateNA(doubles30pct);
    });
//     measure("interpolating doubles with 90% nulls", p, [&]
//     {
//         return interpolateNA(doubles90pct);
//     });
}


template<typename F>
auto visitType4(const arrow::Type::type &id, F &&f)
{
    switch(id)
    {
    case arrow::Type::INT64: return f(std::integral_constant<arrow::Type::type, arrow::Type::INT64 >{});
    case arrow::Type::DOUBLE: return f(std::integral_constant<arrow::Type::type, arrow::Type::DOUBLE>{});
    case arrow::Type::STRING: return f(std::integral_constant<arrow::Type::type, arrow::Type::STRING>{});
    case arrow::Type::LIST: return f(std::integral_constant<arrow::Type::type, arrow::Type::LIST>{});
    default: throw std::runtime_error("array type not supported to downcast: " + std::to_string((int)id));
    }
}

template<typename Range, typename Reader>
void printList(Range &&r, Reader &&f)
{
    auto end = std::end(r);
    auto itr = std::begin(r);
    if(itr != end)
    {
        std::cout << std::invoke(f, *itr++);
    }

    while(itr != end)
    {
        std::cout << "\t" << std::invoke(f, *itr++);
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
    return std::string(elem);
}
std::string formatColumnElem(const ListElemView &elem)
{
    std::ostringstream out;
    out << "[";

    visitType4(elem.array->type_id(), [&](auto id)
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

template<typename T>
std::string formatColumnElem(const std::optional<T> &elem)
{
    if(elem)
        return formatColumnElem(*elem);
    return "null"s;
}

void uglyPrint(const arrow::Table &table)
{
    auto cols = getColumns(table);
    std::cout << "[i]\t";
    printList(cols, [](auto col){ return col->name(); });
    std::cout << '\n';

    int64_t partsSize = 5;

    auto printedElement = [&](const arrow::Column &col, int row)
    {
        return visitType4(col.type()->id(), [&](auto id) -> std::string
        {
            const auto value = columnValueAt<id.value>(col, row);
            return formatColumnElem(value);
        });
    };

    auto printRow = [&](int64_t row)
    {
        std::cout << row << "\t";
        printList(cols, [&](const auto &col) -> std::string
        {
            return printedElement(*col, row);
        });
        std::cout << '\n';
    };

    for(int64_t row = 0; row < partsSize && row < table.num_rows(); row++)
        printRow(row);
    if(table.num_rows() > partsSize*2)
        std::cout << "... " << (table.num_rows() - partsSize * 2) << " more rows ...\n";
    for(int64_t row = std::max<int64_t>(partsSize, std::max<int64_t>(0, table.num_rows() - partsSize)); row < table.num_rows(); row++)
        printRow(row);

    std::cout << "[" << table.num_rows() << " rows x " << table.num_columns() << " cols]" << std::endl;
}

BOOST_FIXTURE_TEST_CASE(GroupExperiments, DataGenerator)
{
    std::vector<ColumnType> types;
    types.emplace_back(arrow::TypeTraits<arrow::Int64Type>::type_singleton(), true, false);
    types.emplace_back(arrow::TypeTraits<arrow::Int64Type>::type_singleton(), true, false);
    for(int i = 0; i < 108; i++)
        types.emplace_back(arrow::TypeTraits<arrow::DoubleType>::type_singleton(), true, false);

    //auto table = loadTableFromCsvFile("F:/dev/temp.csv", types);
    //auto table = loadTableFromCsvFile("F:/dev/train.csv", types);
    //uglyPrint(*table);
    //std::cout << "table rows " << table->num_rows() << std::endl;
    //auto table = loadTableFromCsvFile("F:/dev/train.csv", types);
    //generateCsv("F:/dev/trainSel.csv", *tableFromColumns({table->column(0), table->column(1)}));
    //saveTableToFeatherFile("F:/dev/train-nasze3.feather", *table);
    auto table = loadTableFromFeatherFile("F:/dev/train-nasze3.feather");
    auto grouped = abominableGroupAggregate(table, table->column(0), {table->column(1)});

    const auto idCol = table->column(0);
    const auto timestampCol = table->column(1);
    const auto yCol = table->column(110);
    auto tableSelected = tableFromColumns({ idCol, timestampCol, yCol });


    int row = 0;
    iterateOver<arrow::Type::INT64>(*timestampCol, [&](auto) {row++;}, [&] 
    {
        std::cout << "ALART " << row++ << std::endl;
    });

//     auto row1 = rowAt(*table, 1710755);
//     auto row2 = rowAt(*table, 1710756);
    //auto row3 = rowAt(*table, 1710757);
    auto hlp = groupBy(tableSelected, table->column(0));
    uglyPrint(*hlp);

    MeasureAtLeast p{ 100, 15s };
    measure("groupBy", p, [&]
    {
        return groupBy(tableSelected, table->column(0));
    });

    measure("interpolating doubles with 30% nulls", p, [&]
    {
        return abominableGroupAggregate(table, idCol, { timestampCol, yCol });
    });

    generateCsv("F:/dev/aggr.csv", *grouped);
}

BOOST_FIXTURE_TEST_CASE(GroupBy, DataGenerator)
{
    auto id = std::vector<int64_t>{ 1, 1, 2, 3, 1, 2, 3, 4, 5, 4 };
    auto iota = iotaVector(10);
    auto iotaNulls = std::vector<std::optional<int64_t>>{ 0, std::nullopt, 2, std::nullopt, 4, 6, 7, std::nullopt, std::nullopt, 9 };
    auto idCol = toColumn(id, "id");
    auto iotaCol = toColumn(iota, "iota");
    auto iotaNullsCol = toColumn(iotaNulls, "iotaNulls");
    auto table = tableFromColumns({ idCol, iotaCol, iotaNullsCol });

    auto groupedTable = groupBy(table, idCol);
    uglyPrint(*groupedTable);
}


BOOST_AUTO_TEST_SUITE_END();