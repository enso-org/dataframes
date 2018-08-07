#define BOOST_TEST_MODULE CsvTests
#include <boost/test/included/unit_test.hpp>

#include "csv.h"

#pragma comment(lib, "DataframeHelper.lib")

void testFieldParser(std::string input, std::string expectedContent, int expectedPosition)
{
	auto itr = input.data();
	auto nsv = parseField(itr, itr + input.length(), ',', '\n', '"');

	BOOST_TEST_CONTEXT("Parsing `" << input << "`")
	{
		BOOST_CHECK_EQUAL(nsv.str(), expectedContent);
		BOOST_CHECK_EQUAL(std::distance(input.data(), itr), expectedPosition);
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
	auto itr = input.data();
	auto fields = parseRecord(itr, itr + input.length(), ',', '\n', '"');

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
	auto itr = input.data();
	auto rows = parseCsvTable(itr, itr + input.length(), ',', '\n', '"');

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
	auto table = csvToArrowTable(csv);
}
