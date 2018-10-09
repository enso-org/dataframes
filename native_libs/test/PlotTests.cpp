#include <boost/test/unit_test.hpp>

#include <fstream>

#include "../plotter/Matplotlib/Plot.h"
#include "Core/ArrowUtilities.h"
#include "IO/IO.h"

BOOST_AUTO_TEST_CASE(Plots)
{
    std::vector<int64_t> ints{ 1,2,3,4,5 };
    auto intsColumn = toColumn(ints);

    init(800, 600, nullptr);
    plot(intsColumn.get(), intsColumn.get(), "label", "", "", 1.0, nullptr);
    auto p = getPNG();
    const auto pngHeaderSignature = "\x89PNG\r\n\x1a\n"; // first 8 bytes of PNG header
    BOOST_CHECK_EQUAL(p.substr(0, 8), pngHeaderSignature); // bytes we got look like PNG file
    writeFile("temp.png", p);
}
