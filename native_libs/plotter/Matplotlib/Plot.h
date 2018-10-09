#pragma once

#include <Core/Common.h>

namespace arrow
{
    class ChunkedArray;
    class Column;
	class Table;
}

EXPORT std::string getPNG();

extern "C"
{
    EXPORT void plot(const arrow::Column *xs, const arrow::Column *ys, const char *label, const char *style, const char *color, double alpha, const char **outError) noexcept;
	EXPORT void plotDate(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept;
    EXPORT void scatter(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept;
    EXPORT void kdeplot(const arrow::Column *xs, const char *label, const char **outError) noexcept;
	EXPORT void kdeplot2(const arrow::Column *xs, const arrow::Column *ys, const char *colormap, const char **outError) noexcept;
    EXPORT void fillBetween(const arrow::Column *xs, const arrow::Column *ys1, const arrow::Column *ys2, const char *label, const char *color, double alpha, const char **outError) noexcept;
	EXPORT void heatmap(const arrow::Table* xs, const char* cmap, const char* annot, const char **outError) noexcept;
	EXPORT void histogram(const arrow::Column *xs, size_t bins, const char **outError) noexcept;
	EXPORT void show(const char **outError) noexcept;
	EXPORT void init(size_t w, size_t h, const char **outError) noexcept;
	EXPORT void subplot(long nrows, long ncols, long plot_number, const char **outError) noexcept;
	EXPORT const char* getPngBase64(const char **outError) noexcept;
}