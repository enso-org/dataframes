#ifndef  _MSC_VER
#define BOOST_TEST_DYN_LINK  // otherwise GCC gets undefined main error
#endif // ! _MSC_VER

#define BOOST_TEST_MODULE DataframeHelperTests
#include <boost/test/unit_test.hpp>
#include <boost/algorithm/string.hpp>

#include <chrono>
#include <fstream>
#include <numeric>
#include <random>

#include <date/date.h>

#include "IO/csv.h"
#include "IO/IO.h"
#include "IO/Feather.h"
#include "Core/ArrowUtilities.h"
#include "Core/Benchmark.h"
#include "optional.h"
#include "Processing.h"
#include "Sort.h"
#include "Analysis.h"

#include "Fixture.h"
#include "Core/Utils.h"
#include "IO/XLSX.h"

using namespace std::literals;
using namespace date::literals;

BOOST_AUTO_TEST_CASE(FooBar, *boost::unit_test_framework::disabled())
{
    auto pythonPath = std::getenv("{PYTHONPATH");
    std::cout << pythonPath << std::endl;
}

// TODO: fails now, because lquery interpreter was implemented without support for chunked arrays
BOOST_FIXTURE_TEST_CASE(MappingChunked, ChunkedFixture, *boost::unit_test_framework::disabled())
{
    // a + b
	const auto jsonQuery = R"(
		{
			"operation": "plus",
			"arguments":
			[
				{"column": "a"},
				{"column": "b"}
			]
		})";
	each(table, jsonQuery);
}

BOOST_AUTO_TEST_CASE(LoadCsvWithTimestamp, *boost::unit_test_framework::disabled())
{
    auto table = FormatCSV{}.read("F:/usa.us.txt");
    uglyPrint(*table);
}


BOOST_FIXTURE_TEST_CASE(SortChunked, ChunkedFixture, *boost::unit_test_framework::disabled())
{
    MeasureAtLeast p{50000};
    measure("sorting", p, [&] { return sortTable(table, {table->column(0)}); });
    auto sortedTable0 = sortTable(table, {table->column(0)});

    auto [sortedTable0Ints, sortedTable0Doubles] = toVectors<int64_t, double>(*sortedTable0);

    BOOST_CHECK(std::is_sorted(sortedTable0Ints.begin(), sortedTable0Ints.end()));
    std::cout << sortedTable0Ints.front() << std::endl;
	std::cout << sortedTable0Doubles.front() << std::endl;
}

BOOST_AUTO_TEST_CASE(SortSimple)
{
    std::vector<std::optional<int64_t>> ints{ std::nullopt, 1, 2, std::nullopt, 1, 2, std::nullopt, 2, 1 };
    std::vector<std::optional<double>> doubles{ 20.0, 8.0, std::nullopt, std::nullopt, 16.0, 9.0, 10.0, 3.0, std::nullopt };
    std::vector<std::optional<std::string>> strings{ std::nullopt, "one"s, std::nullopt, "4"s, "4"s, "five"s, std::nullopt, "7"s, "7"s };
    auto iota = iotaVector(ints.size());

    auto table = tableFromVectors(ints, doubles, strings, iota);

    auto testSort = [&](std::vector<SortBy> sortBy, Permutation expectedOrder)
    {
        auto sortedTable = sortTable(table, sortBy);
        auto[ints2, doubles2, strings2, iota2] = toVectors<std::optional<int64_t>, std::optional<double>, std::optional<std::string>, int64_t>(*sortedTable);
        //BOOST_CHECK(std::is_sorted(ints2.begin(), ints2.end()));
        BOOST_CHECK_EQUAL_RANGES(iota2, expectedOrder);
        // std::cout << iota2 << std::endl;
    };

    testSort(
        { { table->column(0), SortOrder::Ascending, NullPosition::Before }
        , { table->column(1), SortOrder::Ascending, NullPosition::Before }},
        { 3, 6, 0, 8, 1, 4, 2, 7, 5});
    testSort(
        { { table->column(0), SortOrder::Ascending, NullPosition::Before }
        , { table->column(1), SortOrder::Ascending, NullPosition::After  } },
        { 6, 0, 3, 1, 4, 8, 7, 5, 2 });
    testSort(
        { { table->column(0), SortOrder::Ascending,  NullPosition::Before}
        , { table->column(1), SortOrder::Descending, NullPosition::After } },
        { 0, 6, 3, 4, 1, 8, 5, 7, 2 });
    testSort(
        { { table->column(0), SortOrder::Ascending,  NullPosition::Before }
        , { table->column(1), SortOrder::Descending, NullPosition::Before } },
        { 3, 0, 6, 8, 4, 1, 2, 5, 7 });
    testSort(
        { { table->column(0), SortOrder::Ascending, NullPosition::After }
        , { table->column(1), SortOrder::Ascending, NullPosition::Before } },
        { 8, 1, 4, 2, 7, 5, 3, 6, 0 });
    testSort(
        { { table->column(0), SortOrder::Ascending, NullPosition::After }
        , { table->column(1), SortOrder::Ascending, NullPosition::After  } },
        { 1, 4, 8, 7, 5, 2, 6, 0, 3 });
    testSort(
        { { table->column(0), SortOrder::Ascending,  NullPosition::After}
        , { table->column(1), SortOrder::Descending, NullPosition::After } },
        { 4, 1, 8, 5, 7, 2, 0, 6, 3 });
    testSort(
        { { table->column(0), SortOrder::Ascending,  NullPosition::After }
        , { table->column(1), SortOrder::Descending, NullPosition::Before } },
        { 8, 4, 1, 2, 5, 7, 3, 0, 6 });

    testSort(
        { { table->column(2), SortOrder::Ascending,  NullPosition::After }
        , { table->column(1), SortOrder::Ascending, NullPosition::Before } },
        { 3, 4, 8, 7, 5, 1, 2, 6, 0 });
}

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
    auto performTest = [&]
    {
        CsvParser parser{ input };
        auto rows = parser.parseCsvTable();
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
    };
	BOOST_TEST_CONTEXT("Parsing `" << input << "` with default line endings")
	{
        performTest();
	}

    if(boost::algorithm::contains(input, "\n") && !boost::algorithm::contains(input, "\r\n"))
    {
        boost::algorithm::replace_all(input, "\n", "\r\n");
        BOOST_TEST_CONTEXT("Parsing `" << input << "` with CRLF")
        {
            performTest();
        }
    }
    else if(boost::algorithm::contains(input, "\r\n"))
    {
        boost::algorithm::replace_all(input, "\r\n", "\n");
        BOOST_TEST_CONTEXT("Parsing `" << input << "` with LF")
        {
            performTest();
        }
    }
}

BOOST_AUTO_TEST_CASE(ParseTimespamp)
{
    using namespace date::literals;
    BOOST_CHECK(date::sys_days(2005_y / mar / 16) == Parser::as<Timestamp>("2005-03-16"));
    BOOST_CHECK(std::nullopt == Parser::as<Timestamp>("2005-03-16 ghdiohgbrodizhbgfro"));
    BOOST_CHECK(std::nullopt == Parser::as<Timestamp>("2005"));
}

BOOST_AUTO_TEST_CASE(ParseCsv)
{
    testCsvParser("foo\nbar\nbaz", { {"foo"}, {"bar"}, {"baz"} });
    testCsvParser("\"\"\n10", { {""}, {"10"} });
    testCsvParser("\"\"\n10\n", { {""}, {"10"} });
    testCsvParser("a,v\n10,20\n", { {"a", "v"}, {"10", "20"} });
}

BOOST_AUTO_TEST_CASE(ParseFile)
{
	auto path = "data/simple_empty.csv";
	auto table = FormatCSV{}.read(path);
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

struct FilteringFixture
{
	std::vector<int64_t> a = {-1, 2, 3, -4, 5};
	std::vector<double> b = {5, 10, 0, -10, -5};
	std::vector<std::string> c = {"foo", "bar", "baz", "", "1"};
    std::vector<std::optional<double>> d = { 1.0, 2.0, std::nullopt, 4.0, std::nullopt };
    std::vector<std::optional<Timestamp>> e = { {2018_y/sep/01}, {2018_y/sep/02}, std::nullopt, {2020_y/nov/04}, std::nullopt };

	std::shared_ptr<arrow::Table> table = tableFromArrays({toArray(a), toArray(b), toArray(c), toArray(d), toArray(e)}, {"a", "b", "c", "d", "e"});

	void testQuery(const char *jsonQuery, std::vector<int> expectedIndices)
	{
        BOOST_TEST_CONTEXT("tesing query: " << jsonQuery)
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
            auto expectedE = expected(e);

            const auto filteredTable = filter(table, jsonQuery);
            auto[a2, b2, c2, d2, e2] = toVectors<int64_t, double, std::string, std::optional<double>, std::optional<Timestamp>>(*filteredTable);
            BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), expectedA.begin(), expectedA.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(b2.begin(), b2.end(), expectedB.begin(), expectedB.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(c2.begin(), c2.end(), expectedC.begin(), expectedC.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(d2.begin(), d2.end(), expectedD.begin(), expectedD.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(e2.begin(), e2.end(), expectedE.begin(), expectedE.end());
        }
	}

	template<typename T>
	void testMap(const char *jsonQuery, std::vector<T> expectedValues)
	{
		const auto column = each(table, jsonQuery);
		const auto result = toVector<T>(*column);
        BOOST_CHECK_EQUAL_RANGES(result, expectedValues);
	}
};

BOOST_FIXTURE_TEST_CASE(MapToIntLiteral, FilteringFixture)
{
	const auto jsonQuery = R"(5)";
	testMap<int64_t>(jsonQuery, { 5, 5, 5, 5, 5 });
}

BOOST_FIXTURE_TEST_CASE(MapToDoubleLiteral, FilteringFixture)
{
	const auto jsonQuery = R"(5.0)";
	testMap<double>(jsonQuery, { 5.0, 5.0, 5.0, 5.0, 5.0 });
}

BOOST_FIXTURE_TEST_CASE(MapToStringLiteral, FilteringFixture)
{
	const auto jsonQuery = R"("foo")";
	testMap<std::string>(jsonQuery, { "foo", "foo", "foo", "foo", "foo" });
}

BOOST_FIXTURE_TEST_CASE(MapToProduct, FilteringFixture)
{
    // a * b
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

BOOST_FIXTURE_TEST_CASE(MapToSumWithProduct, FilteringFixture)
{
    // a*2 + 4
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

BOOST_FIXTURE_TEST_CASE(MapToNegated, FilteringFixture)
{
	const auto jsonQuery = R"(
		{
			"operation": "negate",
			"arguments": [ {"column": "a"} ]
		})";
	testMap<double>(jsonQuery, { 1, -2, -3, 4, -5 });
}

BOOST_FIXTURE_TEST_CASE(MapToAbs, FilteringFixture)
{
	const auto jsonQuery = R"(
		{
			"operation": "abs", 
			"arguments": [ {"column": "a"} ]
		})";
	testMap<double>(jsonQuery, { 1, 2, 3, 4, 5 });
}

BOOST_FIXTURE_TEST_CASE(MapToAbsByCondition, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(MapTimestampDay, FilteringFixture)
{
    // day(e)
    const auto jsonQuery = R"(
 		{
			"operation": "day",
			"arguments":
			[
				{"column": "e"}
			]
 		})";
    testMap<std::optional<int64_t>>(jsonQuery, { 1, 2, std::nullopt, 4, std::nullopt });
}

BOOST_FIXTURE_TEST_CASE(MapTimestampMonth, FilteringFixture)
{
    // month(e)
    const auto jsonQuery = R"(
 		{
			"operation": "month",
			"arguments":
			[
				{"column": "e"}
			]
 		})";
    testMap<std::optional<int64_t>>(jsonQuery, { 9, 9, std::nullopt, 11, std::nullopt });
}

BOOST_FIXTURE_TEST_CASE(MapTimestampYear, FilteringFixture)
{
    // year(e)
    const auto jsonQuery = R"(
 		{
			"operation": "year",
			"arguments":
			[
				{"column": "e"}
			]
 		})";
    testMap<std::optional<int64_t>>(jsonQuery, { 2018, 2018, std::nullopt, 2020, std::nullopt });
}

BOOST_FIXTURE_TEST_CASE(FilterGreaterThanLiteral, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterGreaterThanOtherColumn, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterEqualString, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterEqualInt, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterStringStartsWith, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterStartsWith2, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterStartsWith3, FilteringFixture)
{
	// c.startsWith "baa"
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

BOOST_FIXTURE_TEST_CASE(FilterMatches, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterPredicateOr, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterPredicateNegate, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterInvalidLQuery, FilteringFixture)
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

BOOST_FIXTURE_TEST_CASE(FilterTimestampRelationalOps, FilteringFixture)
{
    BOOST_CHECK_EQUAL(Timestamp(2018_y/sep/2).toStorage(), 1535846400000000000);

    // (e > 2018-09-02)
    const auto jsonQueryGt = R"(
		{
			"predicate": "gt",
			"arguments":
				[
					{"column": "e"},
                    {"timestampNs" : 1535846400000000000 }
				]
		})";
    testQuery(jsonQueryGt, { 3 });

    // (e < 2018-09-02)
    const auto jsonQueryLt = R"(
		{
			"predicate": "lt",
			"arguments":
				[
					{"column": "e"},
                    {"timestampNs" : 1535846400000000000 }
				]
		})";
    testQuery(jsonQueryLt, { 0 });


    // (e == 2018-09-02)
    const auto jsonQueryEq = R"(
		{
			"predicate": "eq",
			"arguments":
				[
					{"column": "e"},
                    {"timestampNs" : 1535846400000000000 }
				]
		})";
    testQuery(jsonQueryEq, { 1 });
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

BOOST_AUTO_TEST_CASE(TimestampParsingFromCsv)
{
    CsvReadOptions opts;
    opts.header = GenerateColumnNames{};
    auto table = FormatCSV{}.read("data/variedColumn.csv", opts);
    auto col = table->column(1);
    BOOST_CHECK_EQUAL(col->type()->id(), arrow::Type::TIMESTAMP);
    BOOST_CHECK_EQUAL(col->null_count(), 1);
    auto t0 = columnValueAt<arrow::Type::TIMESTAMP>(*col, 0);
    BOOST_CHECK_EQUAL(t0, Timestamp(2005_y/feb/25));

    // check that table with timestamps roundtrips
    FormatCSV{}.write("_Temp.csv", *table);
    auto table2 = FormatCSV{}.read("_Temp.csv");
    BOOST_CHECK(table->Equals(*table2));

    FormatXLSX{}.write("_Temp.xlsx", *table);
    XlsxReadOptions xlsReadOpts;
    xlsReadOpts.columnTypes = transformToVector(getColumns(*table),
        [](auto col) { return ColumnType{ *col, false }; });
    auto tableXlsx = FormatXLSX{}.read("_Temp.xlsx", xlsReadOpts);
    BOOST_CHECK(table->Equals(*tableXlsx));
}

BOOST_AUTO_TEST_CASE(ReadTableDeducingFileType)
{
    std::vector<int64_t> ints{50,100};
    auto table = tableFromVectors(ints);

    const std::string filename = "ReadTableDeducingFileType";

    // write to CSV and read
    const auto filenameCsv = filename + ".csv";
    FormatCSV{}.write(filenameCsv, *table);
    auto table2 = readTableFromFile(filenameCsv);
    BOOST_CHECK(table->Equals(*table2));

    // write to feather and read
    const auto filenameFeather = filename + ".feather";
    FormatFeather{}.write(filenameFeather, *table);
    auto table3 = readTableFromFile(filenameFeather);
    // We can't just use Equals method, as feather format does not preserve information
    // whether a field is allowed to contain null values.
    BOOST_REQUIRE_EQUAL(table->num_columns(), 1);
    auto [ints3] = toVectors<int64_t>(*table3);
    BOOST_CHECK_EQUAL_RANGES(ints, ints3);

    // write to XLSX and read
    const auto filenameXlsx = filename + ".xlsx";
    FormatXLSX{}.write(filenameXlsx, *table);
    auto table4 = readTableFromFile(filenameXlsx);
    // FIXME
    // XLSX is not able yet to properly deduce column types, so we get string values
    // test should be adjusted after https://github.com/luna/Dataframes/issues/34
    auto [ints4] = toVectors<std::string>(*table4);
    std::vector strings{ "50", "100" };
    BOOST_CHECK_EQUAL_RANGES(strings, ints4);
}

BOOST_AUTO_TEST_CASE(WriteTableDeducingFileType)
{
    std::vector<int64_t> ints{ 50,100 };
    auto table = tableFromVectors(ints);

    auto testRoundTrip = [&] (const std::string &extension, const TableFileHandler &reader)
    {
        std::string path = "WriteTableDeducingFileType." + extension;
        writeTableToFile(path, *table);

        auto readTable = reader.read(path);

        BOOST_REQUIRE_EQUAL(readTable->num_columns(), table->num_columns());
        auto[readInts] = toVectors<int>(*table);
        BOOST_CHECK_EQUAL_RANGES(ints, readInts);
    };

    testRoundTrip("csv", FormatCSV{});
    testRoundTrip("CSV", FormatCSV{});
    testRoundTrip("cSV", FormatCSV{}); // case shouldn't matter
    testRoundTrip("txt", FormatCSV{});
    testRoundTrip("xlsx", FormatXLSX{});
    testRoundTrip("feather", FormatFeather{});
    BOOST_CHECK_THROW(writeTableToFile("WriteTableDeducingFileType.7z", *table), std::exception); // not a valid extension
}

BOOST_AUTO_TEST_CASE(TimestampInterpolation)
{
    std::vector<std::optional<Timestamp>> times{ {2018_y/sep/1}, std::nullopt, std::nullopt, {2018_y/sep/10} };
    auto interpolatedTimes = toVector<std::optional<Timestamp>>(*interpolateNA(toColumn(times)));
    std::vector<std::optional<Timestamp>> expectedInterpolatedTimes
    {
        {2018_y/sep/1}, {2018_y/sep/4}, {2018_y/sep/7}, {2018_y/sep/10}
    };

    BOOST_CHECK_EQUAL_RANGES(interpolatedTimes, expectedInterpolatedTimes);
}

BOOST_AUTO_TEST_CASE(TimestampStats, *boost::unit_test_framework::disabled())
{
    std::vector<std::optional<Timestamp>> times{ {2018_y/sep/1}, std::nullopt, std::nullopt, {2018_y/sep/10} };
    auto col = toColumn(times);

    std::vector<std::optional<Timestamp>> expectedMin{{ 2018_y / sep / 1 }};
    auto minCol = calculateMin(*col);
    auto minColV = toVector<Timestamp>(*minCol);
    BOOST_CHECK_EQUAL(minCol->type()->id(), col->type()->id());
    BOOST_CHECK_EQUAL_RANGES(minColV, expectedMin);

    // TODO other stats
}

BOOST_AUTO_TEST_CASE(TypeDeducing)
{
    BOOST_CHECK_EQUAL(deduceType("5.0"), arrow::Type::DOUBLE);
    BOOST_CHECK_EQUAL(deduceType("-1.060828e-39"), arrow::Type::DOUBLE);
    BOOST_CHECK_EQUAL(deduceType("5"), arrow::Type::INT64);
    BOOST_CHECK_EQUAL(deduceType("2005-10-11"), arrow::Type::TIMESTAMP);
	BOOST_CHECK_EQUAL(deduceType("five"), arrow::Type::STRING);
	BOOST_CHECK_EQUAL(deduceType(""), arrow::Type::NA);

	auto table = FormatCSV{}.read("data/variedColumn.csv");
    BOOST_REQUIRE_EQUAL(table->num_columns(), 7);
    BOOST_CHECK_EQUAL(table->column(0)->type()->id(), arrow::Type::STRING);
    BOOST_CHECK_EQUAL(table->column(1)->type()->id(), arrow::Type::TIMESTAMP);
	BOOST_CHECK_EQUAL(table->column(2)->type()->id(), arrow::Type::INT64);
	BOOST_CHECK_EQUAL(table->column(3)->type()->id(), arrow::Type::INT64);
	BOOST_CHECK_EQUAL(table->column(4)->type()->id(), arrow::Type::DOUBLE);
	BOOST_CHECK_EQUAL(table->column(5)->type()->id(), arrow::Type::STRING);
	BOOST_CHECK_EQUAL(table->column(6)->type()->id(), arrow::Type::STRING);
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
	std::vector<std::optional<std::string>> values = {std::nullopt, "1"s, "2"s, std::nullopt, "4"s, "5"s, std::nullopt, std::nullopt};
	std::vector<std::optional<std::string>> expectedFilled = {fillWith, "1"s, "2"s, fillWith, "4"s, "5"s, fillWith, fillWith};
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

struct RsiTestingFixture
{
    auto rsi(const std::vector<std::optional<double>> &vector)
    {
        auto col = toColumn(vector);
        auto resultCol = calculateRSI(*col);
        return toVector<std::optional<double>>(*resultCol);
    }

    auto test(const std::vector<std::optional<double>> &input, const std::vector<std::optional<double>> &expectedOutput)
    {
        auto actualOutput = rsi(input);
        BOOST_CHECK_EQUAL_RANGES(expectedOutput, actualOutput);
    }
};

BOOST_FIXTURE_TEST_CASE(RSI, RsiTestingFixture)
{
    test({ 5.0, 10.0, 6.0    }, { 100.0        });
    test({ -5.0, -10.0, -6.0 }, { 0.0          });
    test({                   }, { std::nullopt });
    test({ std::nullopt      }, { std::nullopt });
}

template<typename T>
void testInterpolation(std::vector<std::optional<T>> input, std::vector<std::optional<T>> expectedOutput)
{
    auto column = toColumn(input);
    auto columnInterpolated = interpolateNA(column);
    auto inputInterpolated = toVector<std::optional<T>>(*columnInterpolated);
    BOOST_CHECK_EQUAL_RANGES(inputInterpolated, expectedOutput);
}

BOOST_AUTO_TEST_CASE(InterpolateNA)
{
    testInterpolation<double>(
        { std::nullopt, std::nullopt, 1, 2, std::nullopt, 3, std::nullopt, std::nullopt, std::nullopt, 4, std::nullopt },
        { 1,            1,            1, 2, 2.5,          3, 3.25,         3.5,          3.75,         4, 4 });
    testInterpolation<int64_t>(
        { std::nullopt, 10, std::nullopt, std::nullopt, 16, std::nullopt },
        { 10,           10, 12,           14,           16, 16 });
    // strings cannot be interpolated
    BOOST_CHECK_THROW(testInterpolation<std::string>({"foo"s, std::nullopt, "bar"s}, {}), std::exception);
}

BOOST_AUTO_TEST_CASE(MakeNullsArray)
{
    auto nullInts = makeNullsArray(arrow::TypeTraits<arrow::Int64Type>::type_singleton(), 5);
    BOOST_CHECK_EQUAL(nullInts->type_id(), arrow::Type::INT64);
    BOOST_CHECK_EQUAL(nullInts->length(), 5);
    BOOST_CHECK_EQUAL(nullInts->null_count(), 5);

    auto nullIntsV = toVector<std::optional<int64_t>>(*nullInts);
    std::vector<std::optional<int64_t>> nullIntsVExpected(5);
    BOOST_CHECK_EQUAL_RANGES(nullIntsV, nullIntsVExpected);
}

BOOST_AUTO_TEST_CASE(CsvWithUtf8Path)
{
    using namespace date;

    const auto utfPath = u8"temp-zażółć鵞鳥.csv";
    std::vector<int64_t> nums = { 1, 2, 3 };
    std::vector<Timestamp> dates = { 2018_y/sep/13, 2018_y/sep/14, 2018_y/sep/15 };
    auto table = tableFromVectors(nums, dates);
    FormatCSV{}.write(utfPath, *table);
    auto table2 = FormatCSV{}.read(utfPath);

    auto [readInts, readDates] = toVectors<int64_t, Timestamp>(*table2);
    BOOST_CHECK_EQUAL_RANGES(nums, readInts);
    BOOST_CHECK_EQUAL_RANGES(dates, readDates);
}

BOOST_AUTO_TEST_CASE(ColumnShift)
{
    std::vector<int64_t> ints{ 1, 2, 3 };
    auto intsCol = toColumn(ints);

    auto test = [&] (int lag, std::vector<std::optional<int64_t>> expected)
    {
        BOOST_TEST_CONTEXT("shifting by " << lag)
        {
            auto shiftedCol = shift(intsCol, lag);
            auto shiftedVector = toVector<std::optional<int64_t>>(*shiftedCol);
            BOOST_CHECK_EQUAL_RANGES(shiftedVector, expected);
        }
    };

    BOOST_CHECK_EQUAL(intsCol, shift(intsCol, 0));
    test(0, { 1, 2, 3 });
    test(1, { std::nullopt, 1, 2 });
    test(2, { std::nullopt, std::nullopt, 1 });
    test(3, { std::nullopt, std::nullopt, std::nullopt });
    test(4, { std::nullopt, std::nullopt, std::nullopt });

    test(-1, { 2, 3, std::nullopt });
    test(-2, { 3, std::nullopt, std::nullopt });
    test(-3, { std::nullopt, std::nullopt, std::nullopt });
    test(-4, { std::nullopt, std::nullopt, std::nullopt });
}

BOOST_AUTO_TEST_CASE(AutoCorrelation)
{
    std::vector<int64_t> ints{ 1, 2, 3, 2, 5, 6, 7 };
    auto intsCol = toColumn(ints);

    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, 0), 1.0, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, 1), 0.811749, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, -1), 0.811749, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, 2), 0.7313574508612273, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, -2), 0.7313574508612273, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, 3), 0.7559289460184545, 0.0001);
    BOOST_CHECK_CLOSE(autoCorrelation(intsCol, -3), 0.7559289460184545, 0.0001);
    BOOST_CHECK(std::isnan(autoCorrelation(intsCol, 50)));
    BOOST_CHECK(std::isnan(autoCorrelation(intsCol, -50)));
    // Note: values above are calculated by pandas.
}

BOOST_AUTO_TEST_CASE(TableFromColumnsWithVaryingLengths)
{
    std::vector<int64_t> ints = { 1, 2, 3 };
    std::vector<std::optional<double>> doubles = {1.0, 2.0, std::nullopt, 4.0};

    auto table = tableFromColumns({toColumn(ints), toColumn(doubles)});
    BOOST_CHECK_EQUAL(table->num_rows(), 4);
    for(auto col : getColumns(*table))
        BOOST_CHECK_EQUAL(col->length(), table->num_rows());

    auto [ints2, doubles2] = toVectors<std::optional<int64_t>, std::optional<double>>(*table);
    std::vector<std::optional<int64_t>> expectedInts2{ 1, 2, 3, std::nullopt };
    std::vector<std::optional<double>> expectedDoubles2{ 1.0, 2.0, std::nullopt, 4.0 };
    BOOST_CHECK_EQUAL_RANGES(ints2, expectedInts2);
    BOOST_CHECK_EQUAL_RANGES(doubles2, expectedDoubles2);
}

BOOST_AUTO_TEST_CASE(Rolling, *boost::unit_test_framework::disabled())
{
    const date::sys_days day = 2013_y / jan / 01;
    const std::vector<Timestamp> ts
    {
        day + 9h + 0s,
        day + 9h + 2s,
        day + 9h + 3s,
        day + 9h + 5s,
        day + 9h + 6s,
    };


    const auto tsCol = toColumn(ts);
    const auto numCol = toColumn(std::vector<std::optional<double>>{0.0, 1.0, 2.0, std::nullopt, 4.0});
    const auto table = tableFromColumns({ tsCol, numCol });

    const auto samplesPerWindow = collectRollingIntervalSizes(tsCol, 2s);
    const auto expectedSamplesPerWindow = std::vector<int>{ 1, 1, 2, 1, 2 };
    BOOST_CHECK_EQUAL_RANGES(samplesPerWindow, expectedSamplesPerWindow);

    const auto sumsPerWindowT = rollingInterval(tsCol, 2s, { {numCol, {AggregateFunction::Sum}} });
    const auto sumsPerWindowV = toVectors<Timestamp, double>(*sumsPerWindowT);
    const auto expectedSumsPerWindow = std::vector<double>{ 0, 1, 3, 0, 4 };
    BOOST_CHECK_EQUAL_RANGES(get<0>(sumsPerWindowV), ts); // timestamps column should not be modified
    BOOST_CHECK_EQUAL_RANGES(get<1>(sumsPerWindowV), expectedSumsPerWindow);
}

BOOST_AUTO_TEST_CASE(SliceBoundsChecking)
{
    auto column = toColumn<int64_t>({ 1,2,3,4,5 });
    BOOST_CHECK_THROW(slice(column, -1, 2), std::exception);
    BOOST_CHECK_THROW(slice(column, 4, 2), std::exception);
    BOOST_CHECK_NO_THROW(slice(column, 4, 1));
    BOOST_CHECK_NO_THROW(slice(column, 4, 0));
    BOOST_CHECK_NO_THROW(slice(column, 5, 0));
    BOOST_CHECK_THROW(slice(column, 5, 1), std::exception);
    BOOST_CHECK_NO_THROW(slice(column, 0, 5));
}
