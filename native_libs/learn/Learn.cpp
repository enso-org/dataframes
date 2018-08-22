#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "SKLearn.h"
#include <numpy/arrayobject.h>
#include <Python.h>
#include <LifetimeManager.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

namespace skl = sklearn;

struct NPArrayBuilder {
  double* data = NULL;
  size_t rows = 0, cols = 0;
  void init(size_t _rows, size_t _cols) {
    rows = _rows;
    cols = _cols;
    data = (double*) malloc(sizeof(double) * rows * cols);
  }
  void setAt(int64_t row, int64_t col, double d) {
    data[row*cols + col] = d;
  }
  void setAt(int64_t row, int64_t col, int64_t i) {
    if (row == 0) std::cout << (double) i << std::endl;
    data[row*cols + col] = (double) i;
  }
  void setAt(int64_t row, int64_t col, std::string_view &s) {
    throw std::runtime_error("Cannot use strings with numpy array.");
  }
  void setNaAt(int64_t row, int64_t col) {
    data[row*cols + col] = nan(" ");
  }
  PyObject* getNPMatrix() {
    npy_intp dims[2] = { (long) rows, (long) cols };
    PyObject* r = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, data);
    return r;
  }
  PyObject* getNPArr() {
    npy_intp dim = rows * cols;
    PyObject* r = PyArray_SimpleNewFromData(1, &dim, NPY_DOUBLE, data);
    return r;
  }
};

PyObject* tableToNpMatrix(const arrow::Table &table) {
  Py_Initialize();
  import_array();
  NPArrayBuilder builder;
  builder.init(table.num_rows(), table.num_columns());
  auto cols = getColumns(table);
  int colIndex = 0;
  for (auto& col : cols) {
    int rowIndex = 0;
    iterateOverGeneric(*col,
        [&] (auto &&elem) {
          builder.setAt(rowIndex, colIndex, elem);
          rowIndex++;
        },
        [&] () {
          builder.setNaAt(rowIndex, colIndex);
          rowIndex++;
        });
    colIndex++;
  }
  auto res = builder.getNPMatrix();
  PyObject_Print(res, stdout, 0);
  return res;
}

PyObject* tableToNpArr(const arrow::Table &table) {
  Py_Initialize();
  import_array();
  NPArrayBuilder builder;
  builder.init(table.num_rows(), table.num_columns());
  auto cols = getColumns(table);
  int colIndex = 0;
  for (auto& col : cols) {
    int rowIndex = 0;
    iterateOverGeneric(*col,
        [&] (auto &&elem) {
          builder.setAt(rowIndex, colIndex, elem);
          rowIndex++;
        },
        [&] () {
          builder.setNaAt(rowIndex, colIndex);
          rowIndex++;
        });
    colIndex++;
  }
  auto res = builder.getNPArr();
  PyObject_Print(res, stdout, 0);
  return res;
}

arrow::Column* npArrayToColumn(PyObject* arO) {
  PyArrayObject* ar = reinterpret_cast<PyArrayObject*>(PyArray_ContiguousFromObject(arO, NPY_DOUBLE, 0, 0));
  int size = PyArray_DIM(ar, 0);
  double* data = (double*) PyArray_DATA(ar);
  arrow::DoubleBuilder builder;
  builder.AppendValues(data, size);
  auto arr = finish(builder);
  auto field = arrow::field("Predictions", arr->type(), arr->null_count());
  auto p = std::make_shared<arrow::Column>(field, arr);
  return LifetimeManager::instance().addOwnership(std::move(p));
}

extern "C" {
  void toNpArr(arrow::Table *tb) {
    tableToNpMatrix(*tb);
  }

  void freePyObj(PyObject* o) {
    Py_DECREF(o);
  }

  PyObject* newLogisticRegression() {
    skl::interpreter::get();
    return skl::newLogisticRegression();
  }

  void fit(PyObject* model, arrow::Table *xs, arrow::Table *y) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = tableToNpArr(*y);
    skl::fit(model, xsO, yO);
  }

  arrow::Column* predict(PyObject* model, arrow::Table *xs) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = skl::predict(model, xsO);
    return npArrayToColumn(yO);
  }

  void trainAndScoreLR(arrow::Table *xs, arrow::Table *y) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = tableToNpArr(*y);
    PyObject* model = skl::newLogisticRegression();
    std::cout << "\nFIT\n";
    skl::fit(model, xsO, yO);
    std::cout << "SCORE\n";
    double r = skl::score(model, xsO, yO);
    std::cout << r << std::endl;
    std::cout << "PREDICT\n";
    PyObject* pred = skl::predict(model, xsO);
    PyObject_Print(pred, stdout, 0);
  }
}
