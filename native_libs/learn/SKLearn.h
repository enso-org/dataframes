#pragma once

#include <Python.h>
#include <stdexcept>
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#endif

namespace sklearn {
static std::string s_backend;

struct interpreter {
    PyObject *s_python_function_logistic_regression;
    PyObject *s_python_function_test_train_split;
    PyObject *s_python_function_confusion_matrix;
    PyObject *s_python_empty_tuple;

    static interpreter& get() {
        static interpreter ctx;
        return ctx;
    }

private:

#ifndef WITHOUT_NUMPY
#  if PY_MAJOR_VERSION >= 3

    void *import_numpy() {
        import_array(); // initialize C-API
        return NULL;
    }

#  else

    void import_numpy() {
        import_array(); // initialize C-API
    }

#  endif
#endif

    interpreter() {

        // optional but recommended
#if PY_MAJOR_VERSION >= 3
        wchar_t name[] = L"plotting";
#else
        char name[] = "plotting";
#endif
        Py_SetProgramName(name);
        Py_Initialize();

        import_numpy(); // initialize numpy C-API

        PyObject* sklearn_linname = PyString_FromString("sklearn.linear_model");
        if (!sklearn_linname) {
            throw std::runtime_error("couldnt create string");
        }

        PyObject* sklearn_selname = PyString_FromString("sklearn.model_selection");
        if (!sklearn_selname) {
            throw std::runtime_error("couldnt create string");
        }

        PyObject* sklearn_metricsname = PyString_FromString("sklearn.metrics");
        if (!sklearn_metricsname) {
            throw std::runtime_error("couldnt create string");
        }

        PyObject* sklearnlinmod = PyImport_Import(sklearn_linname);
        Py_DECREF(sklearn_linname);
        if (!sklearnlinmod) { throw std::runtime_error("Error loading module sklearn.linear_model!"); }

        PyObject* sklearnselmod = PyImport_Import(sklearn_selname);
        Py_DECREF(sklearn_selname);
        if (!sklearnselmod) { throw std::runtime_error("Error loading module sklearn.model_selection!"); }

        PyObject* sklearnmetricsmod = PyImport_Import(sklearn_metricsname);
        Py_DECREF(sklearn_metricsname);
        if (!sklearnmetricsmod) { throw std::runtime_error("Error loading module sklearn.metrics!"); }

        s_python_function_logistic_regression = PyObject_GetAttrString(sklearnlinmod, "LogisticRegression");
        s_python_function_test_train_split = PyObject_GetAttrString(sklearnlinmod, "LogisticRegression");
        s_python_function_confusion_matrix = PyObject_GetAttrString(sklearnmetricsmod, "confusion_matrix");

        if(!s_python_function_logistic_regression) {
          throw std::runtime_error("Couldn't find required function!");
        }

        if(!s_python_function_test_train_split) {
          throw std::runtime_error("Couldn't find required function!");
        }

        s_python_empty_tuple = PyTuple_New(0);
    }

    ~interpreter() {
        Py_Finalize();
    }
};

inline PyObject* newLogisticRegression(double c) {
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "C", PyFloat_FromDouble(c));
  PyObject* model = PyObject_Call(interpreter::get().s_python_function_logistic_regression, interpreter::get().s_python_empty_tuple, kwargs);
  return model;
}

inline void fit(PyObject* model, PyObject* xs, PyObject* y) {
  PyObject *n = PyString_FromString("fit");
  PyObject *r = PyObject_CallMethodObjArgs(model, n, xs, y, NULL);
}

inline double score(PyObject* model, PyObject* xs, PyObject* y) {
  PyObject *n = PyString_FromString("score");
  PyObject *r = PyObject_CallMethodObjArgs(model, n, xs, y, NULL);
  return PyFloat_AsDouble(r);
}

inline PyObject* predict(PyObject* model, PyObject* xs) {
  PyObject *n = PyString_FromString("predict");
  PyObject *r = PyObject_CallMethodObjArgs(model, n, xs, NULL);
  return r;
}

inline PyObject* testTrainSplit(PyObject *xs, PyObject* y) {
  PyObject* res = PyObject_CallFunctionObjArgs(xs, y, NULL);
  return res;
}

inline PyObject* confusion_matrix(PyObject* ytrue, PyObject* ypred) {
  return PyObject_CallFunctionObjArgs(interpreter::get().s_python_function_confusion_matrix, ytrue, ypred, NULL);
}

} // end sklearn

