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

    auto writePlot = [&](int w, int h, const std::string &fname)
    {
        const char *error = nullptr;
        boost::filesystem::remove(fname);
        init(w, h, &error);
        if(error)
            THROW("{}", error);
        plot(intsColumn.get(), intsColumn.get(), "label", "", "", 1.0, nullptr);
        saveFigure(fname);
    };

    auto checkIfSaves = [&] (int w, int h, const std::string &fname)
    {
        BOOST_TEST_CONTEXT("saving chart with sizes " << w << "x" << h << " to " << fname)
        {
            writePlot(w, h, fname);
            BOOST_CHECK(boost::filesystem::exists(fname));
            BOOST_CHECK_GT(boost::filesystem::file_size(fname), 0);
        }
    };
    auto checkIfFailsToSave = [&](int w, int h, const std::string &fname)
    {
        BOOST_TEST_CONTEXT("failing to save chart with sizes " << w << "x" << h << " to " << fname)
        {
            BOOST_CHECK_THROW(writePlot(w, h, fname), std::exception);
            BOOST_CHECK(!boost::filesystem::exists(fname));
        }
    };

    // Note: matplotlib also supports pgf format but that would require xelatex command available
    for (auto extension : { "eps", "pdf", "png", "ps", "raw", "rgba", "svg", "svgz" })
    {
        auto fname = "temp."s + extension;
        checkIfSaves(800, 600, fname);
        if(extension == "png")
        {
            auto contents = getFileContents(fname);
            checkThatLooksLikePNG(contents);
        }
    }

    // wrong sizes
    checkIfFailsToSave(0, 600, "temp.png");
    checkIfFailsToSave(800, 0, "temp.png");
    checkIfFailsToSave(0, 0, "temp.png");

    // unknown format
    checkIfFailsToSave(0, 0, "temp.pnhg");
    checkIfFailsToSave(0, 0, "tempppp");
}
