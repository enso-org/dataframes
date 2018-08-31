#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE DataframeHelperTests
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <fstream>

#include "IO/csv.h"
#include "IO/IO.h"
#include "IO/Feather.h"
#include "Core/ArrowUtilities.h"
#include "optional.h"
#include "Processing.h"
#include "Analysis.h"

using namespace std::literals;

void testFieldParser(std::string input, std::string expectedContent, int expectedPosition)
{
	CsvParser parser{input};
	auto nsv = parser.parseField();

	BOOST_TEST_CONTEXT("Parsing `" << input << "`")
	{
		BOOST_CHECK_EQUAL(nsv, expectedContent);
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
			BOOST_CHECK_EQUAL(fields.at(i), expectedContents.at(i));
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
					BOOST_CHECK_EQUAL(readRow.at(j), expectedRow.at(j));
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
	auto path = "data/simple_empty.csv";
	auto csv = parseCsvFile(path);
	auto table = csvToArrowTable(csv, TakeFirstRowAsHeaders{}, {});
}

BOOST_AUTO_TEST_CASE(HelperConversionFunctions)
{
	std::vector<int64_t> numbers;
	std::vector<double> numbersD;
	std::vector<std::string> numbersS;
	std::vector<std::optional<double>> numbersOD;
	std::vector<std::optional<std::string>> numbersOS;

	for(int i = 0; i < 50; i++) 
	{
		numbers.push_back(i);
		numbersD.push_back(i);
		if(i % 5)
			numbersOD.push_back(i);
		else
			numbersOD.push_back(std::nullopt);
	}

	for(int i = 0; i < 40; i++) 
	{
		numbersS.push_back(std::to_string(i));
		if(i % 5)
			numbersOS.push_back(std::to_string(i));
		else
			numbersOS.push_back(std::nullopt);
	}

	auto numbersArray = toArray(numbers);
	auto numbersDArray = toArray(numbersD);
	auto numbersSArray = toArray(numbersS);
	auto numbersODArray = toArray(numbersOD);
	auto numbersOSArray = toArray(numbersOS);

	auto table = tableFromArrays({numbersArray, numbersDArray, numbersSArray, numbersODArray, numbersOSArray});
	auto [retI, retD, retS, retOD, retOS] = toVectors<int64_t, double, std::string, std::optional<double>, std::optional<std::string>>(*table);
	BOOST_CHECK(retI == numbers);
	BOOST_CHECK(retD == numbersD);
	BOOST_CHECK(retS == numbersS);
	BOOST_CHECK(retOD == numbersOD);
	BOOST_CHECK(retOS == numbersOS);
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
	std::vector<std::optional<double>> d = {1.0, 2.0, std::nullopt, 4.0, std::nullopt};

	std::shared_ptr<arrow::Table> table = tableFromArrays({toArray(a), toArray(b), toArray(c), toArray(d)}, {"a", "b", "c", "d"});

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
		auto expectedD = expected(d);

		const auto filteredTable = filter(table, jsonQuery);
		auto[a2, b2, c2, d2] = toVectors<int64_t, double, std::string, std::optional<double>>(*filteredTable);
		BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), expectedA.begin(), expectedA.end());
		BOOST_CHECK_EQUAL_COLLECTIONS(b2.begin(), b2.end(), expectedB.begin(), expectedB.end());
		BOOST_CHECK_EQUAL_COLLECTIONS(c2.begin(), c2.end(), expectedC.begin(), expectedC.end());
		BOOST_CHECK_EQUAL_COLLECTIONS(d2.begin(), d2.end(), expectedD.begin(), expectedD.end());
	}

	template<typename T>
	void testMap(const char *jsonQuery, std::vector<T> expectedValues)
	{
		const auto column = each(table, jsonQuery);
		const auto result = toVector<T>(*column);
		BOOST_CHECK(result == expectedValues);
	}
};

BOOST_FIXTURE_TEST_CASE(MappingSimpleCase, FilteringFixture)
{
	{
		const auto jsonQuery = R"(5)";
		testMap<int64_t>(jsonQuery, { 5, 5, 5, 5, 5 });
	}
	{
		const auto jsonQuery = R"(5.0)";
		testMap<double>(jsonQuery, { 5.0, 5.0, 5.0, 5.0, 5.0 });
	}
	{
		const auto jsonQuery = R"("foo")";
		testMap<std::string>(jsonQuery, { "foo", "foo", "foo", "foo", "foo" });
	}
	{
		const auto jsonQuery = R"(
			{
				"operation": "times", 
				"arguments": 
				[ 
					{"column": "a"},
					{"column": "b"}
				] 
			})";
		testMap<double>(jsonQuery, { -5, 20, 0, 40, -25 });
	}
	{
		const auto jsonQuery = R"(
			{
				"operation": "plus", 
				"arguments": 
				[ 
					{
						"operation": "times", 
						"arguments": 
						[ 
							{"column": "a"},
							2
						] 
					},
					4
				] 
			})";
		testMap<double>(jsonQuery, { 2, 8, 10, -4, 14 });
	}
	{
		const auto jsonQuery = R"(
			{
				"operation": "negate", 
				"arguments": [ {"column": "a"} ]
			})";
		testMap<double>(jsonQuery, { 1, -2, -3, 4, -5 });
	}
 	{
 		// (a > 0) ? a : -a
 		const auto jsonQuery = R"(
 			{
 				"condition": {
 					"predicate": "gt",
 					"arguments": [{"column": "a"}, 0] },
 				"onTrue":  {
 					"column": "a"},
 				"onFalse": {
 					"operation": "negate",
 					"arguments": [{"column": "a"}]}
 			})";
 
		testMap<double>(jsonQuery, {1, 2, 3, 4, 5});
 	}
}

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
	{
		// !(a > 0)
		const auto jsonQuery = R"(
			{
				"boolean": "not",
				"arguments":
				[
					{
						"predicate": "gt", 
						"arguments": 
						[ 
							{"column": "a"},
							0
						] 
					}
				]
			})";

		testQuery(jsonQuery, {0, 3});
	}
	{
		// (a > 0) ||
		// error: missing second argument for `or`
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
					}
				]
			})";

		BOOST_CHECK_THROW(filter(table, jsonQuery), std::exception);
	}
}

BOOST_AUTO_TEST_CASE(FilterWithNulls)
{
	std::vector<std::optional<int64_t>> ints;
	std::vector<std::optional<std::string>> strings;
	for(int i = 0; i < 256; i++)
	{
		if(i % 3)
			ints.push_back(i);
		else
			ints.push_back(std::nullopt);

		if(i % 7)
			strings.push_back(std::to_string(i));
		else
			strings.push_back(std::nullopt);
	}


	std::vector<int64_t> expectedI;
	std::vector<std::optional<std::string>> expectedS;
	std::vector<int64_t> expectedTail10;
	for(int i = 0; i < 256; i++)
	{
		if(i % 3  &&  (i % 2 == 0))
		{
			expectedI.push_back(i);
			if(i >= 246)
				expectedTail10.push_back(i);

			if(i % 7)
				expectedS.push_back(std::to_string(i));
			else
				expectedS.push_back(std::nullopt);
		}
	}

	auto arrayI = toArray(ints);
	auto arrayS = toArray(strings);
	auto arrayTail10 = arrayI->Slice(246);

	// query: a%2 == 0
	const auto jsonQuery = R"(
			{
				"predicate": "eq", 
				"arguments": 
					[ 
						{
							"operation": "mod", 
							"arguments": 
							[ 
								{"column": "a"},
								2
							] 
						},
						0
					] 
			})";

	auto table = tableFromArrays({arrayI, arrayS}, {"a", "b"});
	auto [filteredI, filteredS] = toVectors<std::optional<int64_t>, std::optional<std::string>>(*filter(table, jsonQuery));


	BOOST_CHECK_EQUAL_COLLECTIONS(expectedI.begin(), expectedI.end(), filteredI.begin(), filteredI.end());
	BOOST_CHECK_EQUAL_COLLECTIONS(expectedS.begin(), expectedS.end(), filteredS.begin(), filteredS.end());

	auto tableTail10 = tableFromArrays({arrayTail10}, {"a"});
	auto [filteredVTail10] = toVectors<std::optional<int64_t>>(*filter(tableTail10, jsonQuery));
	BOOST_CHECK_EQUAL_COLLECTIONS(expectedTail10.begin(), expectedTail10.end(), filteredVTail10.begin(), filteredVTail10.end());

	arrow::ArrayVector chunksVectorI;
	int currentPos = 0;
	int currentChunkSize = 1;
	while(currentPos < ints.size())
	{
		chunksVectorI.push_back(arrayI->Slice(currentPos, currentChunkSize));
		currentPos += currentChunkSize;
		currentChunkSize++;
	}
	auto chunksI = std::make_shared<arrow::ChunkedArray>(chunksVectorI);
	auto table2 = tableFromArrays({arrayI, arrayS, chunksI}, {"a", "b", "a2"});
	auto [aa, bb, aa2] = toVectors<std::optional<int64_t>, std::optional<std::string>, std::optional<int64_t>>(*table2);

	BOOST_CHECK_EQUAL_COLLECTIONS(aa.begin(), aa.end(), aa2.begin(), aa2.end());
	auto [filtered2I, filtered2S, filtered2AI] = toVectors<std::optional<int64_t>, std::optional<std::string>, std::optional<int64_t>>(*filter(table2, jsonQuery));

	BOOST_CHECK_EQUAL_COLLECTIONS(expectedI.begin(), expectedI.end(), filtered2AI.begin(), filtered2AI.end());
}

BOOST_AUTO_TEST_CASE(TypeDeducing)
{
	BOOST_CHECK_EQUAL(deduceType("5.0"), arrow::Type::DOUBLE);
	BOOST_CHECK_EQUAL(deduceType("5"), arrow::Type::INT64);
	BOOST_CHECK_EQUAL(deduceType("five"), arrow::Type::STRING);
	BOOST_CHECK_EQUAL(deduceType(""), arrow::Type::NA);

	auto csv = parseCsvFile("data/variedColumn.csv"); 
	auto table = csvToArrowTable(csv, GenerateColumnNames{}, {});
	BOOST_REQUIRE_BITWISE_EQUAL(table->num_columns(), 5);
	BOOST_CHECK_EQUAL(table->column(0)->type()->id(), arrow::Type::INT64);
	BOOST_CHECK_EQUAL(table->column(1)->type()->id(), arrow::Type::INT64);
	BOOST_CHECK_EQUAL(table->column(2)->type()->id(), arrow::Type::DOUBLE);
	BOOST_CHECK_EQUAL(table->column(3)->type()->id(), arrow::Type::STRING);
	BOOST_CHECK_EQUAL(table->column(4)->type()->id(), arrow::Type::STRING);
}

BOOST_AUTO_TEST_CASE(FillingNAInts)
{
	int64_t fillWith = -70;
	std::vector<std::optional<int64_t>> values = {std::nullopt, 1, 2, std::nullopt, 4, 5, std::nullopt, std::nullopt};
	std::vector<std::optional<int64_t>> expectedFilled = {fillWith, 1, 2, fillWith, 4, 5, fillWith, fillWith};
	auto column = toColumn(values);
	auto columnFilled = fillNA(column, fillWith);
	auto valuesFilled = toVector<int64_t>(*columnFilled);
	BOOST_CHECK_EQUAL_COLLECTIONS(valuesFilled.begin(), valuesFilled.end(), expectedFilled.begin(), expectedFilled.end());
}

BOOST_AUTO_TEST_CASE(FillingNAStrings)
{
	std::string fillWith = "foo";
	std::vector<std::optional<std::string>> values = {std::nullopt, "1", "2", std::nullopt, "4", "5", std::nullopt, std::nullopt};
	std::vector<std::optional<std::string>> expectedFilled = {fillWith, "1", "2", fillWith, "4", "5", fillWith, fillWith};
	auto column = toColumn(values);
	auto columnFilled = fillNA(column, fillWith);
	auto valuesFilled = toVector<std::string>(*columnFilled);
	BOOST_CHECK_EQUAL_COLLECTIONS(valuesFilled.begin(), valuesFilled.end(), expectedFilled.begin(), expectedFilled.end());
}

BOOST_AUTO_TEST_CASE(Statistics)
{
	std::vector<std::optional<int64_t>> ints{1, 1, std::nullopt, 3, std::nullopt, 11};
	std::vector<std::optional<double>> doubles{std::nullopt, 1, std::nullopt, 3, 8.0, 11};
	std::vector<std::optional<int64_t>> intsNulls{std::nullopt};
	auto intsColumn = toColumn(ints, "ints");
	auto doublesColumn = toColumn(doubles, "doubles");
	auto intsMin = calculateMin(*intsColumn);
	BOOST_CHECK_EQUAL(toVector<std::optional<int64_t>>(*intsMin), std::vector<std::optional<int64_t>>{1});
	auto nullIntsMin = toVector<std::optional<int64_t>>(*calculateMin(*toColumn(intsNulls)));
	BOOST_CHECK_EQUAL(nullIntsMin, std::vector<std::optional<int64_t>>{std::nullopt});

	auto intsMax = calculateMax(*intsColumn);
	BOOST_CHECK_EQUAL(toVector<std::optional<int64_t>>(*intsMax), std::vector<std::optional<int64_t>>{11});
	auto nullIntsMax = toVector<std::optional<int64_t>>(*calculateMin(*toColumn(intsNulls)));
	BOOST_CHECK_EQUAL(nullIntsMax, std::vector<std::optional<int64_t>>{std::nullopt});

	auto intsMean = calculateMean(*intsColumn);
	BOOST_CHECK_EQUAL(toVector<std::optional<int64_t>>(*intsMean), std::vector<std::optional<int64_t>>{4});
	auto nullIntsMean = toVector<std::optional<int64_t>>(*calculateMin(*toColumn(intsNulls)));
	BOOST_CHECK_EQUAL(nullIntsMean, std::vector<std::optional<int64_t>>{std::nullopt});

	auto intsMedian = calculateMedian(*intsColumn);
	BOOST_CHECK_EQUAL(toVector<std::optional<int64_t>>(*intsMedian), std::vector<std::optional<int64_t>>{2});
	auto nullIntsMedian = toVector<std::optional<int64_t>>(*calculateMin(*toColumn(intsNulls)));
	BOOST_CHECK_EQUAL(nullIntsMedian,  std::vector<std::optional<int64_t>>{std::nullopt});

	calculateCorrelation(*intsColumn, *doublesColumn);
// 
// 	std::vector<double> a = {14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 
// 	25.1, 23.4, 18.1, 22.6, 17.2};
// 	std::vector<double> b = {215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408};;
// 	calculateCorrelation(*toColumn(a), *toColumn(b));
}
