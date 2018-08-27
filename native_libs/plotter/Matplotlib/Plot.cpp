#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"

#include <Python.h>
#include <matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;

struct PyListBuilder {
  PyObject* list = NULL;
  size_t ind = 0;

  void init(size_t s) {
    list = PyList_New(s);
  }

  void append(PyObject *item) {
    PyList_SetItem(list, ind++, item);
  }

  void append(int64_t i) {
    append(PyLong_FromLong(i));
  }

  void append(double d) {
    append(PyFloat_FromDouble(d));
  }

  void append(const std::string_view &s) {
    append(PyString_FromString(std::string(s).c_str()));
  }

  void appendNull() {
    append(PyFloat_FromDouble(nan(" ")));
  }
};

PyObject* chunkedArrayToPyObj(const arrow::ChunkedArray &arr) {
    PyListBuilder builder;
    builder.init(arr.length());
    iterateOverGeneric(arr,
        [&] (auto &&elem) { builder.append(elem); },
        [&] () { builder.appendNull(); });
    return builder.list;
}

PyObject* tableToPyObj(const arrow::Table &table) {
    auto cols = getColumns(table);
    PyObject *result = PyList_New(table.num_columns());
    for (int i = 0; i < table.num_columns(); i++) {
        PyList_SetItem(result, i, chunkedArrayToPyObj(*(cols[i]->data())));
    }
    return result;
}

extern "C"
{
    EXPORT void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* style) {
        std::string st(style);
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "PLOT BEG" << std::endl;
            plt::plot(xsarray, ysarray, st);
            std::cout << "PLOT END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    EXPORT void kdeplot2(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* colormap) {
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

    EXPORT void kdeplot(arrow::ChunkedArray *xs, char* label) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "KDE BEG" << std::endl;
          plt::kdeplot(xsarray, label);
          std::cout << "KDE END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    EXPORT void heatmap(arrow::Table* xs, char* cmap, char* annot) {
        auto xsarray = tableToPyObj(*xs);
        try {
            std::cout << "HEATMAP BEG" << std::endl;
            plt::heatmap(xsarray, cmap, annot);
            std::cout << "HEATMAP END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    EXPORT void histogram(arrow::ChunkedArray *xs, size_t bins) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "HIST BEG" << std::endl;
          plt::hist(xsarray, bins);
          std::cout << "HIST END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    EXPORT void show() {
        try {
          std::cout << "SHOW BEG" << std::endl;
          plt::show();
          std::cout << "SHOW END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    EXPORT void init(size_t w, size_t h) {
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

    EXPORT void subplot(long nrows, long ncols, long plot_number) {
        plt::subplot(nrows, ncols, plot_number);
    }

    EXPORT char* getPNG() {
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
