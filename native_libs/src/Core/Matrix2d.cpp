#include "Matrix2d.h"

#include <unordered_map>
#include <utility>

namespace
{
	// Note: Luna gets pointer to memory managed by Matrix2d object
	// and calls mat_delete with that pointer.
	// We need to keep this mapping, so we know what to delete.

	// TODO: at some point in future thread-safety should be considered
	// (either hide this map behind mutex or document safety requirements)
	std::unordered_map<MatrixDataPtr, Matrix2d*> pointersToTheirManagers;
}

std::string & Matrix2d::access(size_t row, size_t column)
{
	const auto index = makeIndex(row, column);
	return cellContents.at(index);
}

void Matrix2d::fixPointer(size_t row, size_t column)
{
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
	const auto index = makeIndex(row, column);
	return cellContents.at(index);
}

MatrixDataPtr Matrix2d::data() const noexcept
{
	return items.data();
}

Matrix2d * Matrix2d::fromData(MatrixDataPtr data)
{
	return pointersToTheirManagers.at(data);
}

std::unique_ptr<Matrix2d> Matrix2d::copyColumns(size_t columnCount, size_t *columnsToCopy) const
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

extern "C" 
{
	void mat_delete(MatrixDataPtr mat) noexcept
	{
		try
		{
			delete Matrix2d::fromData(mat);
		}
		catch(...) {}
	}

	MatrixDataPtr copyColums(MatrixDataPtr mat, size_t colummCount, size_t *columnsToCopy) noexcept
	{
		try
		{
			return Matrix2d::fromData(mat)->copyColumns(colummCount, columnsToCopy).release()->data();
		}
		catch(...) 
		{
			return nullptr;
		}
	}
}

EXPORT void store(MatrixDataPtr mat, size_t row, size_t column, const char *string) noexcept
{
		try
		{
			return Matrix2d::fromData(mat)->store(row, column, string);
		}
		catch(...) 
		{}
}

