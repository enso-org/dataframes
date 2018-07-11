#include "Core/Matrix2d.h"
#include "Core/Common.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std::literals;

namespace
{
struct ParsedCsv
{
	std::vector<std::vector<std::string>> records; // [record_index][field_index] => field content
	std::vector<std::string> *currentRecord = nullptr;
	std::string *currentField = nullptr;

	ParsedCsv()
	{
		newRecord();
	}

	void newField()
	{
		currentRecord->emplace_back();
		currentField = &currentRecord->back();
	};

	void newRecord()
	{
		records.emplace_back();
		currentRecord = &records.back();
		newField();
	};

	void append(char c)
	{
		*currentField += c;
	}

	std::unique_ptr<Matrix2d> toMatrix() &&
	{
		const auto rowCount = records.size();
		const auto columnCount = [&]
		{
			const auto biggestRecord = std::max_element(records.begin(), records.end(), 
				[] (auto &record1, auto &record2) { return record1.size() < record2.size(); });
			if(biggestRecord != records.end())
				return biggestRecord->size();
			return std::size_t(0);
		}();


		auto ret = std::make_unique<Matrix2d>(rowCount, columnCount);
		for(auto row = 0ull; row < rowCount; row++)
		{
			const auto columnsInRow = records.at(row).size();
			for(auto column = 0ull; column < columnsInRow; column++)
			{
				ret->store(row, column, std::move(records.at(row).at(column)));
			}
		}
		return ret;
	}
};

auto parseCsvContents(const std::string &contents, char separator)
{
	// Each line is a record. Record consist of string fields separated by separator.
	// Additionally, a field can be enclosed within a quote, eg: "field"
	// In such case, it is possible to use newline and separator characters within the field.
	// To use quote character within a quoted field, the character should be repeated: "blah "" blah"
	// Using quote in an unquoted field is not allowed but we don't really care about enforcing this. 
	// (preference is to carry on delivering possibly sensible results)
	ParsedCsv results;

	bool withinQuotes = false;
	for(auto itr = contents.begin(); itr != contents.end(); itr++)
	{
		char c = *itr;

		if(c == '"')
		{
			if(withinQuotes)
			{
				auto nextItr = itr+1;
				bool nextIsAlsoQuote = nextItr!=contents.end() && *nextItr == '"';
				if(nextIsAlsoQuote)
				{
					results.append(c);
					itr++; // skip one character -- double quote within quote codes a single quote
				}
				else
				{
					withinQuotes = false;
				}
			}
			else
				withinQuotes = true;
		}
		else if(withinQuotes)
		{
			results.append(c);
		}
		else
		{
			if(c == separator)
				results.newField();
			else if(c == '\n')
				results.newRecord();
			else
				results.append(c);
		}
	}

	if(withinQuotes)
		throw std::runtime_error("reached the end of the file with an unmatched quote character");

	return std::move(results).toMatrix();
}

std::unique_ptr<Matrix2d> loadCSV(const char *filename, char separator)
{
	try
	{
		std::ifstream input{filename};
		if(!input)
			throw std::runtime_error("Failed to open the file");

		std::stringstream buffer;
		buffer << input.rdbuf();
		return parseCsvContents(buffer.str(), separator);
	}
	catch(std::exception &e)
	{
		// make sure that filename is recorded in the error message
		std::stringstream errMsg;
		errMsg << "Failed to load file `" << filename << "`: " << e.what();
		throw std::runtime_error(errMsg.str());
	}
}

bool needsEscaping(const std::string &record, char seperator)
{
	if(record.empty())
		return false;

	if(record.front() == ' ' || record.back() == ' ')
		return true;

	if(record.find(seperator) != std::string::npos)
		return true;

	return false;
}

}

extern "C"
{
	EXPORT MatrixDataPtr read_csv(const char *filename, size_t *rows, size_t *cols, int* error) noexcept
	{
		try
		{
			auto ret = loadCSV(filename, ',');
			*rows = ret->rowCount;
			*cols = ret->columnCount;
			*error = 0;
			return ret.release()->data();
		}
		catch(std::exception &e)
		{
			*error = 2;
		}
		catch(...)
		{
			*error = 1;
		}
		return nullptr;
	}

	EXPORT void write_csv(const char *filename, MatrixDataPtr mat, size_t rows, size_t cols, int* error) noexcept
	{
		if(!mat)
		{
			*error = 2;
			return;
		}

		char separator = ',';
		*error = 0;

		try
		{
			std::ofstream out{filename};
			if(!out)
				throw std::runtime_error("cannot write csv: file `"s + filename + "` cannot be opened!");

			for(size_t r = 0; r < rows; ++r)
			{
				if(r > 0)
					out << '\n';

				for(size_t c = 0; c < cols; ++c)
				{
					auto index = r * cols + c;
					auto record = mat[index];
				
					if(record)
					{
						if(needsEscaping(record, separator))
							out << std::quoted(record, '"', '"');
						else
							out << record;
					}

					if(c != cols - 1)
						out << separator;

				}
			}
		}
		catch(std::exception &e)
		{
			*error = 1;
		}
	}
}

//int main()
//{
//	try
//	{
//		auto matrix = loadCSV(R"(F:\dev\Dataframes\data\simple_empty.csv)", ',');
//		int i = 0;
//		write_csv(R"(F:\dev\Dataframes\data\simple_empty2.csv)", matrix->data(), matrix->rowCount, matrix->columnCount, &i);
//		auto matrix2 = loadCSV(R"(F:\dev\Dataframes\data\simple_empty2.csv)", ',');
//  		std::cout << "";
//	}
//	catch(std::exception &e)
//	{
//  		std::cout << e.what() << std::endl;
//	}
//	return EXIT_SUCCESS;
//}
