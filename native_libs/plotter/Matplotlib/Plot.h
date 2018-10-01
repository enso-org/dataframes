#pragma once

#include <Core/Common.h>

namespace arrow
{
	class ChunkedArray;
	class Table;
}

extern "C"
{
	EXPORT void plot_date(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys);
	EXPORT void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, const char* label, const char *style);
	EXPORT void kdeplot2(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, const char* colormap);
	EXPORT void kdeplot(arrow::ChunkedArray *xs, const char* label);
	EXPORT void heatmap(arrow::Table* xs, const char* cmap, const char* annot);
	EXPORT void histogram(arrow::ChunkedArray *xs, size_t bins);
	EXPORT void show();
	EXPORT void init(size_t w, size_t h);
	EXPORT void subplot(long nrows, long ncols, long plot_number);
	EXPORT const char* getPNG(const char **outError) noexcept;
}