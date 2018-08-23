#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "SKLearn.h"
#include <numpy/arrayobject.h>
#include <variant.h>
#include <Python.h>
#include <LifetimeManager.h>
#include <Analysis.h>

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

PyObject* columnToNpArr(const arrow::Column &col) {
  Py_Initialize();
  import_array();
  NPArrayBuilder builder;
  builder.init(col.length(), 1);
  int rowIndex = 0;
  iterateOverGeneric(col,
      [&] (auto &&elem) {
        builder.setAt(rowIndex, 0, elem);
        rowIndex++;
      },
      [&] () {
        builder.setNaAt(rowIndex, 0);
        rowIndex++;
      });
  auto res = builder.getNPArr();
  PyObject_Print(res, stdout, 0);
  return res;
}

arrow::Column* npArrayToColumn(PyObject* arO) {
  PyArrayObject* ar = reinterpret_cast<PyArrayObject*>(PyArray_ContiguousFromObject(arO, NPY_DOUBLE, 0, 0));
  std::cout << "MEAN IS: " << PyFloat_AsDouble(PyArray_Mean(ar, 0, NPY_DOUBLE, NULL)) << std::endl;
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

  PyObject* newLogisticRegression(double C) {
    skl::interpreter::get();
    return skl::newLogisticRegression(C);
  }

  void fit(PyObject* model, arrow::Table *xs, arrow::Column *y) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = columnToNpArr(*y);
    skl::fit(model, xsO, yO);
  }

  double score(PyObject* model, arrow::Table *xs, arrow::Column *y) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = columnToNpArr(*y);
    return skl::score(model, xsO, yO);
  }

  arrow::Column* predict(PyObject* model, arrow::Table *xs) {
    skl::interpreter::get();
    PyObject* xsO = tableToNpMatrix(*xs);
    PyObject* yO  = skl::predict(model, xsO);
    PyObject_Print(yO, stdout, 0);
    return npArrayToColumn(yO);
  }

  arrow::Table* confusionMatrix(arrow::Column* ytrue, arrow::Column* ypred) {
    auto c1 = columnToNpArr(*ytrue);
    auto c2 = columnToNpArr(*ypred);
    auto t1 = skl::confusion_matrix(c1, c2);
    std::cout << "\nBEFORE\n";
    PyObject_Print(t1, stdout, 0);
    std::cout << "\nAFTER\n";
    return NULL;
  }

  arrow::Table* oneHotEncode(arrow::Column* col) {
    std::unordered_map<std::string_view, int> valIndexes;
    int lastIndex = 0;
    iterateOver<arrow::Type::STRING>(*col,
      [&] (auto &&elem) {
        if (valIndexes.find(elem) == valIndexes.end()) valIndexes[elem] = lastIndex++;
      },
      [] () {});
    std::vector<arrow::DoubleBuilder> builders(valIndexes.size());
    iterateOver<arrow::Type::STRING>(*col,
      [&] (auto &&elem) {
        int ind = valIndexes[elem];
        for (int i = 0; i < builders.size(); i++) {
          builders[i].Append(i == ind ? 1 : 0);
        }
      },
      [&] () {
        for (int i = 0; i < builders.size(); i++) {
          builders[i].Append(0);
        }
      });
    std::vector<PossiblyChunkedArray> arrs;
    for (auto& bldr : builders) {
      arrs.push_back(finish(bldr));
    }
    std::vector<std::string> names(valIndexes.size());
    for (auto& item : valIndexes) {
      names[item.second] = col->name() + "_" + std::string(item.first);
    }
    auto t = tableFromArrays(arrs, names);
    return LifetimeManager::instance().addOwnership(std::move(t));
  }
}
