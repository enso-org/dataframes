#include <matplotlibcpp.h>
#include <Python.h>
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"

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

  void append(const std::string &s) {
    append(PyString_FromString(s.c_str()));
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
        [&] () { });
    return builder.list;
}

extern "C"
{
    void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys, char* style) {
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

    void kdeplot2(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "KDEPLOT2 BEG" << std::endl;
            plt::kdeplot2(xsarray, ysarray);
            std::cout << "KDEPLOT2 END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void kdeplot(arrow::ChunkedArray *xs) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "KDE BEG" << std::endl;
          plt::kdeplot(xsarray);
          std::cout << "KDE END" << std::endl;
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

    char* getPNG() {
        char* b;
        char* out = NULL;
        size_t l;
        try {
          plt::tight_layout();
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
