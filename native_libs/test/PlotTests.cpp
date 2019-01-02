#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "../plotter/Matplotlib/Plot.h"
#include "Core/ArrowUtilities.h"
#include "IO/IO.h"

namespace
{
    void checkThatLooksLikePNG(std::string_view contents)
    {
        const auto pngHeaderSignature = "\x89PNG\r\n\x1a\n"; // first 8 bytes of PNG header
        BOOST_CHECK_EQUAL(contents.substr(0, 8), pngHeaderSignature); // bytes we got look like PNG file
    }
}

BOOST_AUTO_TEST_CASE(Plots)
{
    std::vector<int64_t> ints{ 1,2,3,4,5 };
    auto intsColumn = toColumn(ints);

    init(800, 600, nullptr);
    plot(intsColumn.get(), intsColumn.get(), "label", "", "", 1.0, nullptr);
    auto p = getPNG();
    checkThatLooksLikePNG(p);
    //writeFile("temp.png", p);
}

BOOST_AUTO_TEST_CASE(PlotSaveFile)
{
    std::vector<int64_t> ints{ 1,2,3,4,5 };
    auto intsColumn = toColumn(ints);

    init(800, 600, nullptr);
    plot(intsColumn.get(), intsColumn.get(), "label", "", "", 1.0, nullptr);
    
    const auto fname = "temp.png";

    boost::filesystem::remove(fname);
    BOOST_CHECK_NO_THROW(saveFigure(fname));
    BOOST_CHECK(boost::filesystem::exists(fname));
    auto contents = getFileContents(fname);
    checkThatLooksLikePNG(contents);
}
