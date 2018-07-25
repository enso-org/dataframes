#pragma once

#include <string>
#include <vector>
#include <memory>

#include "Common.h"

using MatrixDataPtr = const char * const *;

// Helper class that manages resources for 2D-array for Luna language.
// NOTE: its sizes cannot be changed once the object is created.
class Matrix2d
{
	std::vector<std::string> cellContents; // manages memory for cell contents
	std::vector<const char *> items; // exposes strings as Luna-consumable array of C-style strings

	void verifyIndex(size_t row, size_t column) const;
	std::string &access(size_t row, size_t column);
	void fixPointer(size_t row, size_t column);

public:
	const size_t rowCount = 0;
	const size_t columnCount = 0;

	Matrix2d(size_t rowCount, size_t columnCount);
	explicit Matrix2d(const Matrix2d &rhs);
	Matrix2d(const Matrix2d &top, const Matrix2d &bottom); // joins matrices -- places the first one above the second one
	~Matrix2d();

	Matrix2d &operator=(const Matrix2d&) = delete;

	void store(size_t row, size_t column, std::string contents);
	const std::string& load(size_t row, size_t column) const;

	size_t cellCount() const;
	size_t makeIndex(size_t row, size_t column) const noexcept;

	MatrixDataPtr data() const noexcept;
	static Matrix2d *fromData(MatrixDataPtr data);

	std::unique_ptr<Matrix2d> copyColumns(size_t columnCount, int *columnsToCopy) const;
	std::unique_ptr<Matrix2d> copyRows(size_t rowCount, int *rowsToCopy) const;
	std::unique_ptr<Matrix2d> dropRow(int rowToDrop) const;
	std::unique_ptr<Matrix2d> transpose() const;

	template<typename Function>
	void foreach_index(Function &&f) const
	{
		for(auto row = 0ull; row < rowCount; row++)
		{
			for(auto column = 0ull; column < columnCount; column++)
			{
				f(row, column);
			}
		}
	}
};

extern "C"
{
	EXPORT MatrixDataPtr mat_clone(MatrixDataPtr mat, const char **outError) noexcept;
	EXPORT void mat_delete(MatrixDataPtr mat) noexcept;
	EXPORT MatrixDataPtr allocate(size_t rowCount, size_t columnCount, const char **outError) noexcept;
	EXPORT size_t columnCount(MatrixDataPtr mat, const char **outError) noexcept;
	EXPORT size_t rowCount(MatrixDataPtr mat, const char **outError) noexcept;
	EXPORT MatrixDataPtr join(MatrixDataPtr top, MatrixDataPtr bottom, const char **outError) noexcept;
	EXPORT MatrixDataPtr copyColumns(MatrixDataPtr mat, size_t columnCount, int *columnsToCopy, const char **outError) noexcept;
	EXPORT MatrixDataPtr copyRows(MatrixDataPtr mat, size_t rowCount, int *rowsToCopy, const char **outError) noexcept;
	EXPORT MatrixDataPtr dropRow(MatrixDataPtr mat, int rowToDrop, const char **outError) noexcept;
	EXPORT MatrixDataPtr transpose(MatrixDataPtr mat, const char **outError) noexcept;
	EXPORT void store(MatrixDataPtr mat, size_t row, size_t column, const char *string, const char **outError) noexcept;
}
