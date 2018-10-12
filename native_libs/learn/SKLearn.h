#pragma once

#include <stdexcept>
#include <iostream>

#include "Python/IncludePython.h"
using namespace pybind11::literals;

namespace arrow
{
    class Column;
}

namespace sklearn
{

struct EXPORT interpreter
{
    pybind11::function s_python_function_logistic_regression;
    pybind11::function s_python_function_linear_regression;
    pybind11::function s_python_function_test_train_split;
    pybind11::function s_python_function_confusion_matrix;

    static interpreter& get();

private:

    interpreter()
    {
        //pybind11::module sklearnselmod = pybind11::module::import("sklearn.model_selection");
        pybind11::module sklearnlinmod = pybind11::module::import("sklearn.linear_model");
        s_python_function_logistic_regression = getMethod(sklearnlinmod, "LogisticRegression");
        s_python_function_linear_regression   = getMethod(sklearnlinmod, "LinearRegression");
        s_python_function_test_train_split    = getMethod(sklearnlinmod, "LogisticRegression");

        pybind11::module sklearnmetricsmod = pybind11::module::import("sklearn.metrics");
        s_python_function_confusion_matrix = getMethod(sklearnmetricsmod, "confusion_matrix");
    }

    ~interpreter()
    {
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
