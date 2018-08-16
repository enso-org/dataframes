#define BOOST_TEST_MODULE CsvTests
#define NOMINMAX
#include <boost/test/included/unit_test.hpp>
#include <chrono>

#include "IO/csv.h"
#include "IO/IO.h"
#include "IO/Feather.h"
#include "Core/ArrowUtilities.h"
#include "Processing.h"

#pragma comment(lib, "DataframeHelper.lib")
#ifdef _DEBUG
#pragma comment(lib, "arrowd.lib")
#else
#pragma comment(lib, "arrow.lib")
#endif

void testFieldParser(std::string input, std::string expectedContent, int expectedPosition)
{
	CsvParser parser{input};
	auto nsv = parser.parseField();

	BOOST_TEST_CONTEXT("Parsing `" << input << "`")
	{
		BOOST_CHECK_EQUAL(nsv.str(), expectedContent);
		BOOST_CHECK_EQUAL(std::distance(parser.bufferStart, parser.bufferIterator), expectedPosition);
	}
}

BOOST_AUTO_TEST_CASE(ParseField)
{
	testFieldParser("foo", "foo", 3);
	testFieldParser("foo,bar", "foo", 3);
	testFieldParser(",bar", "", 0);
	testFieldParser(R"("foo")", "foo", 5);
	testFieldParser(R"("fo""o")", "fo\"o", 7);
	testFieldParser(R"("fo""o,"",bar")", "fo\"o,\",bar", 14);
	std::string buffer = "foo,bar";
}

void testRecordParser(std::string input, std::vector<std::string> expectedContents)
{
	CsvParser parser{input};
	auto fields = parser.parseRecord();

	BOOST_TEST_CONTEXT("Parsing `" << input << "`")
	{
		BOOST_REQUIRE_EQUAL(fields.size(), expectedContents.size());
		for(int i = 0; i < expectedContents.size(); i++)
			BOOST_CHECK_EQUAL(fields.at(i).str(), expectedContents.at(i));
	}
}

BOOST_AUTO_TEST_CASE(ParseRecord)
{
	testRecordParser("foo,bar,b az", {"foo", "bar", "b az"});
	testRecordParser("foo,bar,b az\n\n\n", {"foo", "bar", "b az"});
	testRecordParser("foo", {"foo"});
	testRecordParser("foo\nbar", {"foo"});
	testRecordParser("\nfoo", {""});
	testRecordParser("\n\n\n", {""});
	testRecordParser(R"("f""o,o")", {R"(f"o,o)"});
}

void testCsvParser(std::string input, std::vector<std::vector<std::string>> expectedContents)
{
	CsvParser parser{input};
	auto rows = parser.parseCsvTable();

	BOOST_TEST_CONTEXT("Parsing `" << input << "`")
	{
		BOOST_REQUIRE_EQUAL(rows.size(), expectedContents.size());
		for(int i = 0; i < expectedContents.size(); i++)
		{
			BOOST_TEST_CONTEXT("row " << i)
			{
				auto &readRow = rows.at(i);
				auto &expectedRow = expectedContents.at(i);
				BOOST_REQUIRE_EQUAL(readRow.size(), expectedRow.size());
				for(int j = 0; j < readRow.size(); j++)
					BOOST_CHECK_EQUAL(readRow.at(j).str(), expectedRow.at(j));
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(ParseCsv)
{
	testCsvParser("foo\nbar\nbaz", {{"foo"}, {"bar"}, {"baz"}});
}

BOOST_AUTO_TEST_CASE(ParseFile)
{
	auto path = R"(F:\dev\Dataframes\data\simple_empty.csv)";
	auto csv = parseCsvFile(path);
	auto table = csvToArrowTable(csv, TakeFirstRowAsHeaders{}, {});
}

BOOST_AUTO_TEST_CASE(HelperConversionFunctions)
{
	std::vector<int64_t> numbers;
	std::vector<double> numbersD;
	std::vector<std::string> numbersS;

	for(int i = 0; i < 50; i++) 
	{
		numbers.push_back(i);
		numbersD.push_back(i);
	}

	for(int i = 0; i < 40; i++) 
		numbersS.push_back(std::to_string(i));

	auto numbersArray = toArray(numbers);
	auto numbersDArray = toArray(numbersD);
	auto numbersSArray = toArray(numbersS);

	auto table = tableFromArrays({numbersArray, numbersDArray, numbersSArray});
	auto [retI, retD, retS] = toVectors<int64_t, double, std::string>(*table);
	BOOST_CHECK(retI == numbers);
	BOOST_CHECK(retD == numbersD);
	BOOST_CHECK(retS == numbersS);
}
 
std::string get_file_contents(const char *filename)
{
	std::ifstream in(filename, std::ios::in);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return(contents);
	}
	throw(errno);
}

struct FilteringFixture
{
	std::vector<int64_t> a = {-1, 2, 3, -4, 5};
	std::vector<double> b = {5, 10, 0, -10, -5};
	std::vector<std::string> c = {"foo", "bar", "baz", "", "1"};

	std::shared_ptr<arrow::Table> table = tableFromArrays({toArray(a), toArray(b), toArray(c)}, {"a", "b", "c"});

	void testQuery(const char *jsonQuery, std::vector<int> expectedIndices)
	{
		auto expected = [&](auto &&v)
		{
			std::decay_t<decltype(v)> ret;
			for(auto index : expectedIndices)
				ret.push_back(v.at(index));
			return ret;
		};

		auto expectedA = expected(a);
		auto expectedB = expected(b);
		auto expectedC = expected(c);

		const auto filteredTable = filter(table, jsonQuery);
		auto[a2, b2, c2] = toVectors<int64_t, double, std::string>(*filteredTable);
		BOOST_CHECK(a2 == expectedA);
		BOOST_CHECK(b2 == expectedB);
		BOOST_CHECK(c2 == expectedC);
	}
};

BOOST_FIXTURE_TEST_CASE(FilterSimpleCase, FilteringFixture)
{
	
	{
		// a > 0
		const auto jsonQuery = R"(
			{
				"predicate": "gt", 
				"arguments": 
					[ 
						{"column": "a"}, 
						0 
					] 
			})";
		testQuery(jsonQuery, {1, 2, 4});
	}

	{
		// a > b
		// tests not only using two columns but also mixed-type comparison
		const auto jsonQuery = R"(
			{
				"predicate": "gt", 
				"arguments": 
					[ 
						{"column": "a"},
						{"column": "b"}
					] 
			})";

		testQuery(jsonQuery, {2, 3, 4});
	}
	{
		// c == "baz"
		// tests not only using two columns but also mixed-type comparison
		const auto jsonQuery = R"(
			{
				"predicate": "eq", 
				"arguments": 
					[ 
						{"column": "c"},
						"baz"
					] 
			})";

		testQuery(jsonQuery, {2});
	}

	{
		// c == 8
		// error: cannot compare string column against number literal
		const auto jsonQuery = R"(
			{
				"predicate": "eq", 
				"arguments": 
					[ 
						{"column": "c"},
						8
					] 
			})";

		BOOST_CHECK_THROW(filter(table, jsonQuery), std::exception);
	}
	{
		// c.startsWith "f"
		const auto jsonQuery = R"(
			{
				"predicate": "startsWith", 
				"arguments": 
					[ 
						{"column": "c"},
						"f"
					] 
			})";

		testQuery(jsonQuery, {0});
	}
	{
		// c.startsWith "ba"
		const auto jsonQuery = R"(
			{
				"predicate": "startsWith", 
				"arguments": 
					[ 
						{"column": "c"},
						"ba"
					] 
			})";

		testQuery(jsonQuery, {1, 2});
	}
	{
		// c.startsWith "ba"
		const auto jsonQuery = R"(
			{
				"predicate": "startsWith", 
				"arguments": 
					[ 
						{"column": "c"},
						"baa"
					] 
			})";

		testQuery(jsonQuery, {});
	}
	{
		// c.matches "ba."
		const auto jsonQuery = R"(
			{
				"predicate": "matches", 
				"arguments": 
					[ 
						{"column": "c"},
						"ba."
					] 
			})";

		testQuery(jsonQuery, {1, 2});
	}
	{
		// a > 0 || b < 0
		const auto jsonQuery = R"(
			{
				"boolean": "or",
				"arguments":
				[
					{
						"predicate": "gt", 
						"arguments": 
						[ 
							{"column": "a"},
							0
						] 
					},
					{
						"predicate": "lt", 
						"arguments": 
						[ 
							{"column": "b"},
							0
						] 
					}
				]
			})";

		testQuery(jsonQuery, {1, 2, 3, 4});
	}
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

BOOST_AUTO_TEST_CASE(FilterBigFile)
{
	const auto jsonQuery = R"({"predicate": "gt", "arguments": [ {"column": "NUM_INSTALMENT_NUMBER"}, {"operation": "plus", "arguments": [50, 1]} ] } )";
	auto table = loadTableFromFeatherFile("C:/installments_payments.feather");
	for(int i  = 0; i < 2000; i++)
	{
		measure("filter installments_payments", [&]
		{
			auto table2 = filter(table, jsonQuery);
//  			std::ofstream out{"tescik100.csv"};
//  			generateCsv(out, *table2, GeneratorHeaderPolicy::GenerateHeaderLine, GeneratorQuotingPolicy::QuoteWhenNeeded);
		});
	}
	std::cout<<"";
}

BOOST_AUTO_TEST_CASE(ParseBigFile)
{

	auto integerType = std::make_shared<arrow::Int64Type>();
	auto doubleType = std::make_shared<arrow::DoubleType>();
	auto textType = std::make_shared<arrow::StringType>();

	std::vector<ColumnType> types
	{
		ColumnType{ integerType, false },
		ColumnType{ integerType, false },
		ColumnType{ doubleType , false },
		ColumnType{ integerType, false },
		ColumnType{ doubleType , false },
		ColumnType{ doubleType , false },
		ColumnType{ doubleType , false },
		ColumnType{ doubleType , false },
	};
// 
// 

 	//const auto path = R"(E:/hmda_lar-florida.csv)";
	
 	for(int i  = 0; i < 20; i++)
 	{
		measure("parse big file", [&]
		{
			auto csv = parseCsvFile("C:/installments_payments.csv");
			auto table = csvToArrowTable(std::move(csv), TakeFirstRowAsHeaders{}, types);

			//saveTableToFeatherFile("C:/installments_payments.feather", *table);
		});
 	}

// 	for(int i = 0; i < 10; i++)
// 	{
// 		measure("parse big file", [&]
// 		{
// 			auto csv = parseCsvFile(path);
// 			auto table = csvToArrowTable(csv, TakeFirstRowAsHeaders{}, {});
// 		});
// 	}
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
