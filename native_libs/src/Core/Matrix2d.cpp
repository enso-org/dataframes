#include "Matrix2d.h"
#include "Error.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

using namespace std::literals;

namespace
{
	// Note: Luna gets pointer to memory managed by Matrix2d object
	// and calls mat_delete with that pointer.
	// We need to keep this mapping, so we know what to delete.

	// TODO: at some point in future thread-safety should be considered
	// (either hide this map behind mutex or document safety requirements)
	std::unordered_map<MatrixDataPtr, Matrix2d*> pointersToTheirManagers;

	std::ostream &printIndex(std::ostream &out, size_t row, size_t column)
	{
		return out << "(" << row << ", " << column << ")";
	}
}

void Matrix2d::verifyIndex(size_t row, size_t column) const
{
	const auto index = makeIndex(row, column);
	if(index < 0 || index >= cellContents.size())
	{
		std::ostringstream errorMsg;
		errorMsg << "Invalid index access: ";
		printIndex(errorMsg, row, column);
		errorMsg << ", matrix size is: ";
		printIndex(errorMsg, rowCount, columnCount);
		throw std::out_of_range(errorMsg.str());
	}
}

std::string & Matrix2d::access(size_t row, size_t column)
{
	verifyIndex(row, column);
	const auto index = makeIndex(row, column);
	return cellContents.at(index);
}

void Matrix2d::fixPointer(size_t row, size_t column)
{
	verifyIndex(row, column);
	const auto index = makeIndex(row, column);
	const auto &value = cellContents.at(index);
	items.at(index) = value.empty()
		? nullptr
		: value.c_str();
}

Matrix2d::Matrix2d(size_t rowCount, size_t columnCount) : rowCount(rowCount), columnCount(columnCount)
{
	items.resize(cellCount());
	cellContents.resize(cellCount());

	pointersToTheirManagers[data()] = this;
}

Matrix2d::Matrix2d(const Matrix2d &rhs)
	: cellContents(rhs.cellContents), items(rhs.items), rowCount(rhs.rowCount), columnCount(rhs.columnCount)
{
	pointersToTheirManagers[data()] = this;
}

Matrix2d::Matrix2d(const Matrix2d &top, const Matrix2d &bottom)
	: Matrix2d(top.rowCount + bottom.rowCount, std::max(top.columnCount, bottom.columnCount))
{
	top.foreach_index([&](auto row, auto column)
	{	
		//std::printf("TOP: row: %zu column: %zu\n", row, column);
		store(row, column, top.load(row, column));
	});
	bottom.foreach_index([&](auto row, auto column)
	{
 		//std::printf("BOTTOM: row: %zu column: %zu\n", row, column);
 		//std::printf("BOTTOM2: row: %zu column: %zu\n", top.rowCount + row, column);
		store(top.rowCount + row, column, bottom.load(row, column));
	});
}

Matrix2d::~Matrix2d()
{
	pointersToTheirManagers.erase(data());
}

size_t Matrix2d::cellCount() const
{
	return rowCount * columnCount;
}

size_t Matrix2d::makeIndex(size_t row, size_t column) const noexcept
{
	return row * columnCount + column;
}

void Matrix2d::store(size_t row, size_t column, std::string contents)
{
	access(row, column) = std::move(contents);
	fixPointer(row, column);
}

const std::string& Matrix2d::load(size_t row, size_t column) const
{
	verifyIndex(row, column);
	const auto index = makeIndex(row, column);
	return cellContents.at(index);
}

MatrixDataPtr Matrix2d::data() const noexcept
{
	return items.data();
}

Matrix2d * Matrix2d::fromData(MatrixDataPtr data)
{
	try
	{
		return pointersToTheirManagers.at(data);
	}
	catch(std::exception &e)
	{
		std::ostringstream errorMessage;
		errorMessage << "failed to match data pointer " << data << " to a known Matrix2d object: " << e.what();
		throw std::runtime_error(errorMessage.str());
	}
}


std::unique_ptr<Matrix2d> Matrix2d::copyColumns(size_t columnCount, int *columnsToCopy) const
{
	auto ret = std::make_unique<Matrix2d>(rowCount, columnCount);
	for(auto row = 0ull; row < rowCount; row++)
	{
		for(auto column = 0ull; column < columnCount; column++)
		{
			const auto sourceColumnIndex = columnsToCopy[column];
			auto value = load(row, sourceColumnIndex);
			ret->store(row, column, std::move(value));
		}
	}
	return ret;
}

std::unique_ptr<Matrix2d> Matrix2d::copyRows(size_t rowCount, int *rowsToCopy) const
{
	auto ret = std::make_unique<Matrix2d>(rowCount, columnCount);
	for (auto row = 0ull; row < rowCount; row++)
	{
		const auto sourceRowIndex = rowsToCopy[row];
		for (auto column = 0ull; column < columnCount; column++)
		{
			auto value = load(sourceRowIndex, column);
			ret->store(row, column, std::move(value));
		}
	}
	return ret;
}

std::unique_ptr<Matrix2d> Matrix2d::dropRow(int rowToDrop) const 
{
	auto ret = std::make_unique<Matrix2d>(rowCount-1, columnCount);
	int *rowsToCopy = new int[rowCount-1];
	for (int i = 0ull; i < rowToDrop; i++)
	{
		rowsToCopy[i]=i;
	}
	for (int i = rowToDrop+1; i < rowCount; i++)
	{
		rowsToCopy[i-1] = i;
	}

	for(auto row = 0ull; row < rowCount-1; row++) //should call copyRows 
	{
		for (auto column = 0ull; column < columnCount; column++)
		{
			const auto sourceRowIndex = rowsToCopy[row];
			auto value = load(sourceRowIndex, column);
			ret->store(row, column, std::move(value));
		}
	}
	
	return ret;
}

std::unique_ptr<Matrix2d> Matrix2d::transpose() const
{
	auto ret = std::make_unique<Matrix2d>(columnCount, rowCount);
	for (auto row = 0ull; row < rowCount; row++)
	{
		for (auto column = 0ull; column < columnCount; column++)
		{
			auto value = load(row, column);
			ret->store(column, row, std::move(value));
		}
	}
	return ret;
}

extern "C" 
{
	// Note: as an exception, this function does not take ourError argument
	// because ManagedPtr expects single argument function to call
	void mat_delete(MatrixDataPtr mat) noexcept
	{
		try
		{
			delete Matrix2d::fromData(mat);
		}
		catch(...) 
		{
			// Note: generally we don't want this library to print anything
			// but this should really not happen and we have no other means
			// of thelling the world that something went wrong.
			std::cout << __FUNCTION__ << ": " << "error" << std::endl;
		}
	}
	MatrixDataPtr allocate(size_t rowCount, size_t columnCount, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return std::make_unique<Matrix2d>(rowCount, columnCount).release()->data();
		};
	}

	MatrixDataPtr copyColumns(MatrixDataPtr mat, size_t columnCount, int *columnsToCopy, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->copyColumns(columnCount, columnsToCopy).release()->data();
		};
	}

	MatrixDataPtr copyRows(MatrixDataPtr mat, size_t rowCount, int *rowsToCopy, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->copyRows(rowCount, rowsToCopy).release()->data();
		};
	}

	MatrixDataPtr dropRow(MatrixDataPtr mat, int rowToDrop, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->dropRow(rowToDrop).release()->data();
		};
	}
	
	MatrixDataPtr transpose(MatrixDataPtr mat, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->transpose().release()->data();
		};
	}

	void store(MatrixDataPtr mat, size_t row, size_t column, const char *string, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->store(row, column, string);
		};
	}

	MatrixDataPtr mat_clone(MatrixDataPtr mat, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return std::make_unique<Matrix2d>(*Matrix2d::fromData(mat)).release()->data();
		};
	}

	MatrixDataPtr join(MatrixDataPtr top, MatrixDataPtr bottom, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return std::make_unique<Matrix2d>(*Matrix2d::fromData(top), *Matrix2d::fromData(bottom)).release()->data();
		};
	}

	size_t columnCount(MatrixDataPtr mat, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->columnCount;
		};
	}

	size_t rowCount(MatrixDataPtr mat, const char **outError) noexcept
	{
		return TRANSLATE_EXCEPTION(outError)
		{
			return Matrix2d::fromData(mat)->rowCount;
		};
	}
}
