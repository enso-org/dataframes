#pragma once

#include <string>
#include <vector>

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
	~Matrix2d();

	void store(size_t row, size_t column, std::string contents);
	const std::string& load(size_t row, size_t column) const;

	size_t cellCount() const;
	size_t makeIndex(size_t row, size_t column) const noexcept;

	MatrixDataPtr data() const noexcept;
	static Matrix2d *fromData(MatrixDataPtr data);

	std::unique_ptr<Matrix2d> copyColumns(size_t columnCount, size_t *columnsToCopy) const;
};

extern "C"
{
	EXPORT void mat_delete(MatrixDataPtr mat) noexcept; // NOTE: mat is not the Matrix2d object but its data() value
	EXPORT MatrixDataPtr copyColums(MatrixDataPtr mat, size_t colummCount, size_t *columnsToCopy) noexcept;
	EXPORT void store(MatrixDataPtr mat, size_t row, size_t column, const char *string) noexcept;
}
