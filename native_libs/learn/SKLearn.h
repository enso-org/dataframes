#pragma once

#include <stdexcept>
#include <iostream>

#if defined(_DEBUG) && defined(_MSC_VER)
#define WAS_DEBUG
#undef _DEBUG
#endif

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/arrayobject.h>
#include <Python.h>
#include <datetime.h>
using namespace pybind11::literals;

#ifdef WAS_DEBUG
#define _DEBUG 1
#endif

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#endif

namespace arrow
{
    class Column;
}

namespace sklearn
{

struct EXPORT interpreter
{
    pybind11::object s_python_function_logistic_regression;
    pybind11::object s_python_function_linear_regression;
    pybind11::object s_python_function_test_train_split;
    pybind11::object s_python_function_confusion_matrix;

    static interpreter& get();

private:

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

    interpreter()
    {
        // optional but recommended
#if PY_MAJOR_VERSION >= 3
        wchar_t name[] = L"plotting";
#else
        char name[] = "plotting";
#endif
        Py_SetProgramName(name);
        pybind11::initialize_interpreter();
        import_numpy(); // initialize numpy C-API

        pybind11::module sklearnlinmod = pybind11::module::import("sklearn.linear_model");
        //pybind11::module sklearnselmod = pybind11::module::import("sklearn.model_selection");
        pybind11::module sklearnmetricsmod = pybind11::module::import("sklearn.metrics");

        s_python_function_logistic_regression = sklearnlinmod.attr("LogisticRegression");
        s_python_function_linear_regression = sklearnlinmod.attr("LinearRegression");
        s_python_function_test_train_split = sklearnlinmod.attr("LogisticRegression");
        s_python_function_confusion_matrix = sklearnmetricsmod.attr("confusion_matrix");
    }

    ~interpreter()
    {
        pybind11::finalize_interpreter();
    }
};

inline pybind11::object newLogisticRegression(double c)
{
    return interpreter::get().s_python_function_logistic_regression("C"_a=c);
}

inline pybind11::object newLinearRegression()
{
    return interpreter::get().s_python_function_linear_regression();
}

inline void fit(pybind11::object model, pybind11::array xs, pybind11::array y)
{
    model.attr("fit")(xs, y);
}

inline double score(pybind11::object model, pybind11::array xs, pybind11::array y)
{
    auto result = model.attr("score")(xs, y);
    return result.cast<double>();
}

inline pybind11::object predict(pybind11::object model, pybind11::array xs)
{
    return model.attr("predict")(xs);
}

// inline PyObject* testTrainSplit(PyObject *xs, PyObject* y)
// {
// 
//   PyObject* res = PyObject_CallFunctionObjArgs(xs, y, NULL);
//   return res;
// }

inline pybind11::object confusion_matrix(pybind11::object ytrue, pybind11::object ypred)
{
  return interpreter::get().s_python_function_confusion_matrix.call(ytrue, ypred); 
}

} // end sklearn

EXPORT pybind11::array_t<double> columnToNpArr(const arrow::Column &col);
EXPORT std::shared_ptr<arrow::Column> npArrayToColumn(pybind11::array_t<double> arr, std::string name);
