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

	std::string &access(size_t row, size_t column);
	void fixPointer(size_t row, size_t column);

public:
	const size_t rowCount = 0;
	const size_t columnCount = 0;

	Matrix2d(size_t rowCount, size_t columnCount);
	explicit Matrix2d(const Matrix2d &rhs);
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
};

extern "C"
{
	EXPORT MatrixDataPtr mat_clone(MatrixDataPtr mat) noexcept;
	EXPORT void mat_delete(MatrixDataPtr mat) noexcept;
	EXPORT MatrixDataPtr copyColumns(MatrixDataPtr mat, size_t columnCount, int *columnsToCopy) noexcept;
	EXPORT MatrixDataPtr copyRows(MatrixDataPtr mat, size_t rowCount, int *rowsToCopy) noexcept;
	EXPORT MatrixDataPtr dropRow(MatrixDataPtr mat, int rowToDrop) noexcept;
	EXPORT MatrixDataPtr transpose(MatrixDataPtr mat) noexcept;
	EXPORT void store(MatrixDataPtr mat, size_t row, size_t column, const char *string) noexcept;
}
