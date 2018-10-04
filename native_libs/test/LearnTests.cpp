#include <boost/test/unit_test.hpp>

#include "Core/ArrowUtilities.h"
#include "../plotter/Matplotlib/Plot.h"
#include "../learn/Learn.h"
#include "../learn/SKLearn.h"

BOOST_AUTO_TEST_CASE(DoubleColumnNumpyRoundtrip)
{
    auto col = toColumn<std::optional<double>>({ 1.0, 2.0, std::nullopt, 3.0 });
    auto nar = columnToNpArr(*col);
    auto col2 = npArrayToColumn(nar, col->name());
    BOOST_CHECK(col->Equals(*col2));
}

struct RegressionFixture
{
    double coef = 2.0;
    double intercept = 1.0;
    double linearMap(double x) { return coef * x + intercept; };

    std::vector<double> xsVector{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> ysVector = transformToVector(xsVector, [this](double x) { return linearMap(x); });

    std::vector<double> newSample{ 20.0 };

    pybind11::array xs = tableToNpMatrix(*tableFromVectors(xsVector));
    pybind11::array ys = columnToNpArr(*toColumn(ysVector));
};

BOOST_FIXTURE_TEST_CASE(LinearRegression, RegressionFixture)
{
    auto linReg = sklearn::newLinearRegression();
    BOOST_CHECK_EQUAL((std::string)linReg.get_type().str(), "<class 'sklearn.linear_model.base.LinearRegression'>");
    sklearn::fit(linReg, xs, ys);
    //pybind11::print(linReg, linReg.attr("coef_"), linReg.attr("intercept_"));

    auto inferredCoef = linReg.attr("coef_").cast<double>();
    auto inferredIntercept = linReg.attr("intercept_").cast<double>();
    BOOST_CHECK_EQUAL(inferredCoef, coef);
    BOOST_CHECK_EQUAL(inferredIntercept, intercept);

    auto predictedAt20 = sklearn::predict(linReg, tableToNpMatrix(*tableFromVectors(newSample))).cast<double>();
    BOOST_CHECK_EQUAL(predictedAt20, linearMap(20));
}

BOOST_FIXTURE_TEST_CASE(LogisticRegression, RegressionFixture)
{
    auto logReg = sklearn::newLogisticRegression(5.25);
    BOOST_CHECK_EQUAL((std::string)logReg.get_type().str(), "<class 'sklearn.linear_model.logistic.LogisticRegression'>");
    BOOST_CHECK_EQUAL(logReg.attr("C").cast<double>(), 5.25);


}