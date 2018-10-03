#include <boost/test/unit_test.hpp>

#include "Core/ArrowUtilities.h"
#include "../plotter/Matplotlib/Plot.h"
#include "../learn/SKLearn.h"

BOOST_AUTO_TEST_CASE(DoubleColumnNumpyRoundtrip)
{
    auto col = toColumn<std::optional<double>>({ 1.0, 2.0, std::nullopt, 3.0 });
    auto nar = columnToNpArr(*col);
    auto col2 = npArrayToColumn(nar, col->name());
    BOOST_CHECK(col->Equals(*col2));
}

BOOST_AUTO_TEST_CASE(LinearRegresion, *boost::unit_test_framework::disabled())
{
    sklearn::interpreter::get();
    auto logReg = sklearn::newLogisticRegression(5.25);
    BOOST_CHECK_EQUAL((std::string)logReg.get_type().str(), "<class 'sklearn.linear_model.logistic.LogisticRegression'>");
    BOOST_CHECK_EQUAL(logReg.attr("C").cast<double>(), 5.25);

    const auto coef = 2.0;
    const auto intercept = 1.0;
    const auto linearMap = [&] (double x) { return coef*x + intercept; };

    std::vector<double> xsVector{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> ysVector = transformToVector(xsVector, linearMap);

    auto xs = tableToNpMatrix(*tableFromVectors(xsVector));
    auto ys = columnToNpArr(*toColumn(ysVector));

    auto linReg = sklearn::newLinearRegression();
    sklearn::fit(linReg, xs, ys);
    pybind11::print(linReg, linReg.attr("coef_"), linReg.attr("intercept_"));

    auto inferredCoef = linReg.attr("coef_").cast<double>();
    auto inferredIntercept = linReg.attr("intercept_").cast<double>();

    BOOST_CHECK_EQUAL(inferredCoef, coef);
    BOOST_CHECK_EQUAL(inferredIntercept, intercept);
    std::vector<double> newSample{20.0};
    auto predictedAt20 = linReg.attr("predict")(tableToNpMatrix(*tableFromVectors(newSample))).cast<double>();
    BOOST_CHECK_EQUAL(predictedAt20, linearMap(20));

    BOOST_CHECK_EQUAL((std::string)linReg.get_type().str(), "<class 'sklearn.linear_model.base.LinearRegression'>");
    //pybind11::print(logReg);
    //PyObject_Print(ttt, stdout, 0);
}
