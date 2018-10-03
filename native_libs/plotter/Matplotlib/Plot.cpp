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
    PyObject* list = NULL;
    size_t ind = 0;
public:
    PyListBuilder(size_t length)
        : list(PyList_New(length))
    {
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
        append(PythonInterpreter::instance().toPyDateTime(t).release().ptr());
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

PyObject *toPyList(const arrow::ChunkedArray &arr)
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

PyObject *toPyList(const arrow::Column &column)
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

PyObject *toPyList(const arrow::Table &table)
{
    // TODO resource safety
    auto cols = getColumns(table);
    PyObject *result = PyList_New(table.num_columns());
    for(int i = 0; i < table.num_columns(); i++)
        PyList_SetItem(result, i, toPyList(*(cols[i]->data())));

    return result;
}

std::string getPNG()
{
    plt::tight_layout();
    plt::legend();
    return plt::getPNG();
}

extern "C"
{
    void plot(const arrow::Column *xs, const arrow::Column *ys, const char* label, const char *style, const char **outError) noexcept
    {
        return TRANSLATE_EXCEPTION(outError)
        {
            auto xsarray = toPyList(*xs);
            auto ysarray = toPyList(*ys);
            plt::plot(xsarray, ysarray, label, style);
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
