#include "Plot.h"
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"


// Windows-specific issue workaround:
// Note [MU]:
// When _DEBUG is defined, Python defines Py_DEBUG and Py_DEBUG leads to
// Py_DECREF expanding to special checking code that uses symbols available
// only in debug binaries of Python. That causes linker error on Windows
// when using Release Python binaries with Debug Dataframe build.
//
// And we don't want to use Debug binaries of Python, because they are 
// incompatible with Release packages installed through pip (e.g. numpy)
// and having Debug and Release packages side-by-side looks non-trival.
// Perhaps someone with better Python knowledge can improve this is future.
// 
// For now just let's try to trick Python into thinking that we are in Release
// mode and hope that noone else includes this header. 
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

    void append(const std::string_view &s)
    {
        append(PyString_FromString(std::string(s).c_str()));
    }

    // TODO: use date2num or sth???
    void append(const Timestamp &t)
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
        append(PyFloat_FromDouble(nan(" ")));
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
    PyListBuilder builder{(size_t)arr.length()};
    iterateOverGeneric(arr,
        [&] (auto &&elem) { builder.append(elem); },
        [&] ()            { builder.appendNull(); });
    return builder.release();
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
    void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* label, const char *style) {
        std::string st(style);
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "PLOT BEG" << std::endl;
            plt::plot(xsarray, ysarray, label, st);
            std::cout << "PLOT END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void plot_date(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys) {

        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "PLOT_DATE BEG" << std::endl;
            plt::plot_date(xsarray, ysarray);
            std::cout << "PLOT_DATE END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void scatter(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys) {

        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "SCATTER BEG" << std::endl;
            plt::scatter(xsarray, ysarray);
            std::cout << "SCATTER END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void kdeplot2(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* colormap) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "KDEPLOT2 BEG" << std::endl;
            plt::kdeplot2(xsarray, ysarray, colormap);
            std::cout << "KDEPLOT2 END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void kdeplot(arrow::ChunkedArray *xs, char* label) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "KDE BEG" << std::endl;
          plt::kdeplot(xsarray, label);
          std::cout << "KDE END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void heatmap(arrow::Table* xs, char* cmap, char* annot) {
        auto xsarray = tableToPyObj(*xs);
        try {
            std::cout << "HEATMAP BEG" << std::endl;
            plt::heatmap(xsarray, cmap, annot);
            std::cout << "HEATMAP END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void fill_between(arrow::ChunkedArray* xs, arrow::ChunkedArray* ys1, arrow::ChunkedArray* ys2) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray1 = chunkedArrayToPyObj(*ys1);
        std::cout << "YS1 " << ysarray1 << std::endl;
        auto ysarray2 = chunkedArrayToPyObj(*ys2);
        std::cout << "YS2 " << ysarray2 << std::endl;
        try {
            std::cout << "FILL BEG" << std::endl;
            plt::fill_between(xsarray, ysarray1, ysarray2);
            std::cout << "FILL END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void histogram(arrow::ChunkedArray *xs, size_t bins) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "HIST BEG" << std::endl;
          plt::hist(xsarray, bins);
          std::cout << "HIST END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void show() {
        try {
          std::cout << "SHOW BEG" << std::endl;
          plt::show();
          std::cout << "SHOW END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void init(size_t w, size_t h) {
        try {
          std::cout << "INIT BEG" << std::endl;
          plt::backend("Agg");
          plt::detail::_interpreter::get();
          std::cout << "figsize BEG " << w << " " << h << std::endl;
          plt::figure_size(w, h);
          plt::rotate_ticks(45);
          std::cout << "figsize END" << std::endl;
          std::cout << "INIT END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void subplot(long nrows, long ncols, long plot_number) {
        plt::subplot(nrows, ncols, plot_number);
    }

    char* getPNG() {
        char* b;
        char* out = NULL;
        size_t l;
        try {
          plt::tight_layout();
          plt::legend();
          std::cout << "PNG BEG" << std::endl;
          plt::getPNG(&b, &l);
          std::cout << "PNG END" << std::endl;
          std::cout << "B64 BEG" << std::endl;
          out = (char*) base64_encode((unsigned char *) b, l, NULL);
          std::cout << "B64 END" << std::endl;
          std::cout << "FREE BEG" << std::endl;
          free(b);
          std::cout << "FREE END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
        return out;
    }
}
