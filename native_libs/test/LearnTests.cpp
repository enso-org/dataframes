#include <boost/test/unit_test.hpp>

#include "Core/ArrowUtilities.h"

#include "../plotter/Matplotlib/Plot.h"

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

#include "../learn/SKLearn.h"
#ifdef WAS_DEBUG
#define _DEBUG 1
#endif


BOOST_AUTO_TEST_CASE(LinearRegresion, *boost::unit_test_framework::disabled())
{
    sklearn::interpreter::get();
    auto logReg = sklearn::newLogisticRegression(5.25);
    BOOST_CHECK_EQUAL((std::string)logReg.get_type().str(), "<class 'sklearn.linear_model.logistic.LogisticRegression'>");
    BOOST_CHECK_EQUAL(logReg.attr("C").cast<double>(), 5.25);

    auto linReg = sklearn::newLinearRegression();
    BOOST_CHECK_EQUAL((std::string)linReg.get_type().str(), "<class 'sklearn.linear_model.base.LinearRegression'>");

    auto col = toColumn<std::optional<double>>({1.0, 2.0, std::nullopt, 3.0});
    auto nar = columnToNpArr(*col);
    auto col2 = npArrayToColumn(nar, col->name());
    BOOST_CHECK(col->Equals(*col2));

    //pybind11::print(logReg);
    //PyObject_Print(ttt, stdout, 0);
}
