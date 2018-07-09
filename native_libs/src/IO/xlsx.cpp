#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <xlnt/xlnt.hpp>

#include "Core/Matrix2d.h"
#include "Core/Common.h"
#include "Core/Error.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#pragma comment(lib, "xlntd.lib")
#else
#pragma comment(lib, "xlnt.lib")
#endif
#endif

namespace
{
	using namespace std::literals;
	auto xlsxParseFile(const char *filename)
	{
		try
		{
			xlnt::workbook wb;
			wb.load(filename);
			const auto sheet = wb.active_sheet();

			// We keep the object under unique_ptr, so there will be
			// no leak if exception is thrown before the end of function
			auto ret = std::make_unique<Matrix2d>(sheet.highest_row(), sheet.highest_column().index);

			for (auto row : sheet.rows(false))
			{
				for (auto cell : row)
				{ 
					// subtract 1, as xlsx is indexed from 1, and we are indexed from 0
					ret->store(cell.row() - 1, cell.column().index - 1, cell.to_string());
				}
			}

			return ret;
		}
		catch(std::exception &e)
		{
			throw std::runtime_error("Failed to parse file `"s + filename + "` : " + e.what());
		}
	}

	void xlsxPrintToFile(const Matrix2d &matrix, const char *filename)
	{
		xlnt::workbook wb;
		auto sheet = wb.active_sheet();
		sheet.title("Written from Luna");

		for(auto row = 0ull; row < matrix.rowCount; row++)
		{
			for(auto column = 0ull; column < matrix.columnCount; column++)
			{
				// translate indices to from-1-indexed xlnt types
				const auto xlsRow = static_cast<xlnt::row_t>(row + 1);
				const auto xlsColumn = static_cast<xlnt::column_t::index_t>(column + 1);

				const auto cellContents = matrix.load(row, column);
				sheet.cell(xlsColumn, xlsRow).value(cellContents);
			}
		}

		wb.save(filename);
	}
}

extern "C"
{

	EXPORT MatrixDataPtr read_xlsx(const char *filename, const char **error) noexcept
	{
		return translateExceptionToError(error, [&] 
		{
			auto mat = xlsxParseFile(filename);
			return mat.release()->data();
		});
	}

	EXPORT void write_xlsx(MatrixDataPtr mat, const char *filename, const char **error) noexcept
	{
		translateExceptionToError(error, [&] 
		{
			xlsxPrintToFile(*Matrix2d::fromData(mat), filename); 
		});
	}


}

// int main()
// {
// 	auto matrix = xlsxParseFile(R"(C:\Users\mwurb\Documents\kalkulator.xlsx)");
// 	xlsxPrintToFile(*matrix, R"(C:\Users\mwurb\Documents\kalkulator2.xlsx)");
// 	return EXIT_SUCCESS;
// }
