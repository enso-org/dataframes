#include <boost/test/unit_test.hpp>

#include <fstream>

#include "../plotter/Matplotlib/Plot.h"
#include "Core/ArrowUtilities.h"

 BOOST_AUTO_TEST_CASE(Plots, *boost::unit_test_framework::disabled())
 {
 	std::vector<int64_t> ints{1,2,3,4,5};
 	auto intsColumn = toColumn(ints);

 	init(800, 600);
 	plot(intsColumn->data().get(), intsColumn->data().get(), "label", "o");
 	auto p = getPNG(nullptr);

 }

