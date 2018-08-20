#include <matplotlibcpp.h>
#include <Python.h>
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "B64.h"

using namespace std;
namespace plt = matplotlibcpp;


PyObject* chunkedArrayToPyObj(const arrow::ChunkedArray &arr) {
    PyObject* list = PyList_New(arr.length());
    size_t chunkStart = 0;
    size_t pyInd;
    for (auto &chunk : arr.chunks()) {
        for (size_t row = 0; row < chunk->length(); row++) {
            pyInd = row + chunkStart;
            if (!chunk->IsNull(row)) {
                PyObject *item = NULL;
                switch (arr.type()->id()) {
                  case arrow::Type::DOUBLE:
                    item = PyFloat_FromDouble(arrayAt<arrow::Type::DOUBLE>(*chunk, row));
                    break;
                  case arrow::Type::STRING:
                    item = PyString_FromString(arrayAt<arrow::Type::STRING>(*chunk, row).c_str());
                    break;
                }
                PyList_SetItem(list, pyInd, item);
            } else {
                PyList_SetItem(list, pyInd, PyFloat_FromDouble(nan(" ")));
            }
        }
        chunkStart += chunk->length();
    }
    return list;
}

void testMpl()
{
	int n = 5000; // number of data points
	vector<double> x(n),y(n);
	for(int i=0; i<n; ++i) {
		double t = 2*M_PI*i/n;
		x.at(i) = 16*sin(t)*sin(t)*sin(t);
		y.at(i) = 13*cos(t) - 5*cos(2*t) - 2*cos(3*t) - cos(4*t);
	}

	// plot() takes an arbitrary number of (x,y,format)-triples.
	// x must be iterable (that is, anything providing begin(x) and end(x)),
	// y must either be callable (providing operator() const) or iterable.
	plt::plot(x, y, "r-", x, [](double d) { return 12.5+abs(sin(d)); }, "k-");


	// show plots
	plt::show();
}

extern "C"
{
    void plot(arrow::ChunkedArray *xs, arrow::ChunkedArray *ys) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        std::cout << "XS " << xsarray << std::endl;
        auto ysarray = chunkedArrayToPyObj(*ys);
        std::cout << "YS " << ysarray << std::endl;
        try {
            std::cout << "PLOT BEG" << std::endl;
            plt::plot(xsarray, ysarray, "o");
            std::cout << "PLOT END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void histogram(arrow::ChunkedArray *xs) {
        auto xsarray = chunkedArrayToPyObj(*xs);
        try {
          std::cout << "HIST BEG" << std::endl;
          plt::hist(xsarray);
          std::cout << "HIST END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void show() {
        try {
          std::cout << "SHOW BEG" << std::endl;
          plt::show();
          std::cout << "SHOW END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    void init() {
        try {
          std::cout << "INIT BEG" << std::endl;
          plt::backend("Agg");
          plt::detail::_interpreter::get();
          plt::figure_size(400, 400);
          plt::clf();
          std::cout << "INIT END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
    }

    char* getPNG() {
        char* b;
        char* out = NULL;
        size_t l;
        try {
          std::cout << "PNG BEG" << std::endl;
          plt::getPNG(&b, &l);
          std::cout << "PNG END" << std::endl;
          std::cout << "B64 BEG" << std::endl;
          out = (char*) base64_encode((unsigned char *) b, l, NULL);
          std::cout << "B64 END" << std::endl;
          std::cout << "FREE BEG" << std::endl;
          free(b);
          std::cout << "FREE END" << std::endl;
        } catch (const runtime_error& e) {
          std::cout << e.what() << std::endl;
        }
        return out;
    }
}
