#include "Plot.h"
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"
#include "ValueHolder.h"
#include "Core/Error.h"


namespace
{
    thread_local ValueHolder returnedString;
}

// Windows-specific issue workaround:
// Note [MU]:
// When _DEBUG is defined, Python defines Py_DEBUG and Py_DEBUG leads to
// Py_DECREF expanding to special checking code that uses symbols available
// only in debug binaries of Python. That causes linker error on Windows
// when using Release Python binaries with Debug Dataframe build.
//
// And we don't want to use Debug binaries of Python, because they are 
// incompatible with Release packages installed through pip (e.g. numpy)
// and having Debug and Release packages side-by-side looks non-trivial.
// Perhaps someone with better Python knowledge can improve this is future.
// 
// For now just let's try to trick Python into thinking that we are in Release
// mode and hope that no one else includes this header. 
// And that standard library/runtime won't explode.
#if defined(_DEBUG) && defined(_MSC_VER)
#define WAS_DEBUG
#undef _DEBUG
#endif

#include <Python.h>
#include <matplotlibcpp.h>
#include <datetime.h>

#ifdef WAS_DEBUG
#define _DEBUG 1
#endif
///////////////////////////////////////////////////////////////////////////////

using namespace std;
namespace plt = matplotlibcpp;

struct PyListBuilder
{
protected:
    PyObject* list = NULL;
    size_t ind = 0;
public:
    PyListBuilder(size_t length)
        : list(PyList_New(length))
    {
        // TODO: move to some kind of general purpose initialization procedure
        if(PyDateTimeAPI == nullptr)
        {
            PyDateTime_IMPORT;
        }
    }
    ~PyListBuilder()
    {
        if(list)
            Py_DECREF(list);
    }

    void append(PyObject *item)
    {
        PyList_SetItem(list, ind++, item);
    }

    void append(int64_t i)
    {
        append(PyLong_FromLongLong(i));
    }

    void append(double d)
    {
        append(PyFloat_FromDouble(d));
    }

    void append(std::string_view s)
    {
        append(PyString_FromString(std::string(s).c_str()));
    }

    void append(Timestamp t)
    {
        using namespace date;
        auto daypoint = floor<days>(t);
        auto ymd = year_month_day(daypoint);   // calendar date
        time_of_day tod = make_time(t - daypoint); // Yields time_of_day type

        // Obtain individual components as integers
        auto y = (int)ymd.year();
        auto m = (int)(unsigned)ymd.month();
        auto d = (int)(unsigned)ymd.day();
        auto h = (int)tod.hours().count();
        auto min = (int)tod.minutes().count();
        auto s = (int)tod.seconds().count();
        auto us = (int)std::chrono::duration_cast<std::chrono::microseconds>(tod.subseconds()).count();
        append(PyDateTime_FromDateAndTime(y, m, d, h, min, s, us));
    }

    void appendNull()
    {
        append(PyFloat_FromDouble(std::nan(" ")));
    }

    auto release()
    {
        assert(list);
        assert(ind == PyList_Size(list));
        return std::exchange(list, nullptr);
    }
};

PyObject* chunkedArrayToPyObj(const arrow::ChunkedArray &arr)
{
    try
    {
        PyListBuilder builder{(size_t)arr.length()};
        iterateOverGeneric(arr,
            [&] (auto &&elem) { builder.append(elem); },
            [&] ()            { builder.appendNull(); });
        return builder.release();
    }
    catch(std::exception &e)
    {
        throw std::runtime_error("failed to convert chunked array to python list: "s + e.what());
    }
}

PyObject* chunkedArrayToPyObj(const arrow::Column &arr)
{
    try
    {
        return chunkedArrayToPyObj(*arr.data());
    }
    catch(std::exception &e)
    {
        throw std::runtime_error("column " + arr.name() + ": " + e.what());
    }
}

PyObject* tableToPyObj(const arrow::Table &table)
{
    auto cols = getColumns(table);
    PyObject *result = PyList_New(table.num_columns());
    for (int i = 0; i < table.num_columns(); i++)
        PyList_SetItem(result, i, chunkedArrayToPyObj(*(cols[i]->data())));

    return result;
}

extern "C"
{
    void plot(const arrow::Column *xs, const arrow::Column *ys, const char* label, const char *style, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
            auto ysarray = chunkedArrayToPyObj(*xs);
            plt::plot(xsarray, ysarray, label, style);
        };
    }

    void plotDate(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
            auto ysarray = chunkedArrayToPyObj(*ys);
            plt::plot_date(xsarray, ysarray);
        };
    }

    void scatter(const arrow::Column *xs, const arrow::Column *ys, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
            auto ysarray = chunkedArrayToPyObj(*ys);
            plt::scatter(xsarray, ysarray);
        };
    }

    void kdeplot(const arrow::Column *xs, const char *label, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
            plt::kdeplot(xsarray, label);
        };
    }

    void kdeplot2(const arrow::Column *xs, const arrow::Column *ys, const char *colormap, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
            auto ysarray = chunkedArrayToPyObj(*ys);
            plt::kdeplot2(xsarray, ysarray, colormap);
        };
    }

    void heatmap(const arrow::Table* xs, const char* cmap, const char* annot, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = tableToPyObj(*xs);
            plt::heatmap(xsarray, cmap, annot);
        };
    }

    void histogram(const arrow::Column *xs, size_t bins, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = chunkedArrayToPyObj(*xs);
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
            plt::tight_layout();
            plt::legend();
            auto png = plt::getPNG();
            auto encodedPng = base64_encode(png);
            return returnedString.store(std::move(encodedPng));
        };
    }
}
