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

using namespace std::literals;
using boost::unit_test_framework::disabled;

struct DataGenerator
{
	std::mt19937 generator{std::random_device{}()};

	template<typename Distribution>
	std::shared_ptr<arrow::Column> generateColumn(arrow::Type::type id, int64_t N, std::string name, double nullShare, Distribution distribution)
	{
		std::bernoulli_distribution nullDistribution{nullShare};
		auto arr = visitType(id, [&] (auto id)
		{
			typename TypeDescription<id.value>::BuilderType b;
			b.Reserve(N);
			for(int64_t i = 0; i < N; i++)
			{
				if(nullDistribution(generator))
				{
					b.AppendNull();
				}
				else
				{
					const auto value = distribution(generator);
					if constexpr(id.value != arrow::Type::STRING)
						b.Append(value);
					else
						b.Append(std::to_string(value));
				}
			}

			return finish(b);
		});

		return std::make_shared<arrow::Column>(arrow::field(name, arr->type(), arr->null_count()), arr);
	}

	std::shared_ptr<arrow::Column> generateColumn(arrow::Type::type id, int64_t N, std::string name, double nullShare = 0.0)
	{
		if(id == arrow::Type::INT64)
			return generateColumn(id, N, name, nullShare, std::uniform_int_distribution<int64_t>{});
		if(id == arrow::Type::DOUBLE)
			return generateColumn(id, N, name, nullShare, std::uniform_real_distribution<double>{});
		if(id == arrow::Type::STRING)
			return generateColumn(id, N, name, nullShare, std::uniform_int_distribution<int64_t>{});
		throw std::runtime_error("wrong type");
	}

	std::shared_ptr<arrow::Table> generateNumericTable(int N)
	{
		auto intColumn1 = generateColumn(arrow::Type::INT64, N, "intsNonNull");
		auto intColumn2 = generateColumn(arrow::Type::INT64, N, "intsSomeNulls", 0.2);
		auto intColumn3 = generateColumn(arrow::Type::INT64, N, "intsManyNulls", 0.7);
		auto intColumn4 = generateColumn(arrow::Type::INT64, N, "intsAllNulls", 1.0);
		auto doubleColumn1 = generateColumn(arrow::Type::DOUBLE, N, "doublesNonNull");
		auto doubleColumn2 = generateColumn(arrow::Type::DOUBLE, N, "doublesSomeNulls", 0.2);
		auto doubleColumn3 = generateColumn(arrow::Type::DOUBLE, N, "doublesManyNulls", 0.7);
		auto doubleColumn4 = generateColumn(arrow::Type::DOUBLE, N, "doublesAllNulls", 1.0);

		return tableFromColumns({intColumn1, intColumn2, intColumn3, intColumn4, doubleColumn1, doubleColumn2, doubleColumn3, doubleColumn4});
	}
};


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

	auto [v, blah] = benchmark("int column to vector", [&] { return toVector<int64_t>(*intColumn1); });
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

BOOST_AUTO_TEST_SUITE_END();