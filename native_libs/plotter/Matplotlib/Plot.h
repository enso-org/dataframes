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
	EXPORT void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* label, char* color, double alpha, const char *style);
	EXPORT void kdeplot2(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* colormap);
	EXPORT void kdeplot(arrow::ChunkedArray *xs, char* label);
	EXPORT void filled_between(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys1, arrow::ChunkedArray *ys2, char* label, char* color, double alpha);
	EXPORT void heatmap(arrow::Table* xs, char* cmap, char* annot);
	EXPORT void histogram(arrow::ChunkedArray *xs, size_t bins);
	EXPORT void show();
	EXPORT void init(size_t w, size_t h);
	EXPORT void subplot(long nrows, long ncols, long plot_number);
	EXPORT char* getPNG();
}