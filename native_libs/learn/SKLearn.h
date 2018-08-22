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

        PyObject* sklearnlinmod = PyImport_Import(sklearn_linname);
        Py_DECREF(sklearn_linname);
        if (!sklearnlinmod) { throw std::runtime_error("Error loading module sklearn.linear_model!"); }

        s_python_function_logistic_regression = PyObject_GetAttrString(sklearnlinmod, "LogisticRegression");

        if(!s_python_function_logistic_regression) {
          throw std::runtime_error("Couldn't find required function!");
        }

        s_python_empty_tuple = PyTuple_New(0);
    }

    ~interpreter() {
        Py_Finalize();
    }
};

inline PyObject* newLogisticRegression() {
  PyObject* model = PyObject_CallObject(interpreter::get().s_python_function_logistic_regression, interpreter::get().s_python_empty_tuple);
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

} // end sklearn

