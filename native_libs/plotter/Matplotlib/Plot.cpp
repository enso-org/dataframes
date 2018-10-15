#include "Plot.h"
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"
#include "ValueHolder.h"
#include "Core/Error.h"

#include "Python/IncludePython.h"
#include <matplotlibcpp.h>
#include "Python/PythonInterpreter.h"

namespace
{
    thread_local ValueHolder returnedString;
}
///////////////////////////////////////////////////////////////////////////////

namespace plt = matplotlibcpp;

struct PyListBuilder
{
protected:
    pybind11::list list;
    size_t ind = 0;
public:
    PyListBuilder(size_t length)
        : list(length)
    {
    }
    ~PyListBuilder()
    {}

    void append(pybind11::object item)
    {
        setAt(list, ind++, item);
    }

    void append(int64_t i)
    {
        append(pybind11::int_(i));
    }

    void append(double d)
    {
        append(pybind11::float_(d));
    }

    void append(std::string_view s)
    {
        append(pybind11::str(s.data(), s.length()));
    }

    void append(Timestamp t)
    {
        append(PythonInterpreter::instance().toPyDateTime(t));
    }

    void appendNull()
    {
        append(pybind11::float_(std::numeric_limits<double>::quiet_NaN()));
    }

    auto release()
    {
        assert(list);
        assert(ind == list.size());
        return std::exchange(list, pybind11::list{});
    }
};

pybind11::list toPyList(const arrow::ChunkedArray &arr)
{
    try
    {
        PyListBuilder builder{ (size_t)arr.length() };
        iterateOverGeneric(arr,
            [&](auto &&elem) { builder.append(elem); },
            [&]()            { builder.appendNull(); });
        return builder.release();
    }
    catch(std::exception &e)
    {
        throw std::runtime_error("failed to convert chunked array to python list: "s + e.what());
    }
}

pybind11::list toPyList(const arrow::Column &column)
{
    try
    {
        return toPyList(*column.data());
    }
    catch(std::exception &e)
    {
        throw std::runtime_error("column " + column.name() + ": " + e.what());
    }
}

pybind11::list toPyList(const arrow::Table &table)
{
    auto cols = getColumns(table);
    pybind11::list result(table.num_columns());
    for(int i = 0; i < table.num_columns(); i++)
    {
        auto columnAsPyList = toPyList(*cols[i]);
        pybind11::setAt(result, i, columnAsPyList);
    }

    return result;
}

std::string getPNG()
{
    plt::tight_layout();
    return plt::getPNG();
}

extern "C"
{
    void plot(const arrow::Column *xs, const arrow::Column *ys, const char* label, const char *style, const char *color, double alpha, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray = toPyList(*ys);
            plt::plot(xsarray, ysarray, label, style, color, alpha);
        };
    }

    void plotDate(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray = toPyList(*ys);
            plt::plot_date(xsarray, ysarray);
        };
    }

    void scatter(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray = toPyList(*ys);
            plt::scatter(xsarray, ysarray);
        };
    }

    void kdeplot(const arrow::Column *xs, const char *label, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            plt::kdeplot(xsarray, label);
        };
    }

    void kdeplot2(const arrow::Column *xs, const arrow::Column *ys, const char *colormap, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray = toPyList(*ys);
            plt::kdeplot2(xsarray, ysarray, colormap);
        };
    }

    void fillBetween(const arrow::Column *xs, const arrow::Column *ys1, const arrow::Column *ys2, const char *label, const char *color, double alpha, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray1 = toPyList(*ys1);
            auto ysarray2 = toPyList(*ys2);
            plt::fill_between(xsarray, ysarray1, ysarray2, label, color, alpha);
        };
    }

    void heatmap(const arrow::Table* xs, const char* cmap, const char* annot, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            plt::heatmap(xsarray, cmap, annot);
        };
    }

    void histogram(const arrow::Column *xs, size_t bins, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            plt::hist(xsarray, bins);
        };
    }

    void show(const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            plt::show();
        };
    }

    void init(size_t w, size_t h, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            plt::backend("Agg");
            plt::detail::_interpreter::get();
            plt::figure_size(w, h);
            plt::rotate_ticks(45);
        };
    }

    void subplot(long nrows, long ncols, long plot_number, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            plt::subplot(nrows, ncols, plot_number);
        };
    }

    const char* getPngBase64(const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto png = ::getPNG();
            auto encodedPng = base64_encode(png);
            return returnedString.store(std::move(encodedPng));
        };
    }
}
