#pragma once

#ifdef __linux__
#include <dlfcn.h> 
#endif 

#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint> 
#include <functional>
#include <string_view>

#include "Python/IncludePython.h"

using namespace pybind11::literals;

#define PyString_FromString PyUnicode_FromString

inline PyObject *PyString_FromStringView(std::string_view sv)
{
    return PyUnicode_FromStringAndSize(sv.data(), sv.size());
}

namespace matplotlibcpp {
namespace detail {

static std::string s_backend;

struct _interpreter
{
    pybind11::function s_python_function_show;
    pybind11::function s_python_function_close;
    pybind11::function s_python_function_draw;
    pybind11::function s_python_function_pause;
    pybind11::function s_python_function_save;
    pybind11::function s_python_function_figure;
    pybind11::function s_python_function_plot;
    pybind11::function s_python_function_scatter;
    pybind11::function s_python_function_plot_date;
    pybind11::function s_python_function_kdeplot;
    pybind11::function s_python_function_heatmap;
    pybind11::function s_python_function_semilogx;
    pybind11::function s_python_function_semilogy;
    pybind11::function s_python_function_loglog;
    pybind11::function s_python_function_fill_between;
    pybind11::function s_python_function_hist;
    pybind11::function s_python_function_subplot;
    pybind11::function s_python_function_legend;
    pybind11::function s_python_function_xlim;
    pybind11::function s_python_function_xticks;
    pybind11::function s_python_function_ion;
    pybind11::function s_python_function_ylim;
    pybind11::function s_python_function_title;
    pybind11::function s_python_function_axis;
    pybind11::function s_python_function_xlabel;
    pybind11::function s_python_function_ylabel;
    pybind11::function s_python_function_grid;
    pybind11::function s_python_function_clf;
    pybind11::function s_python_function_errorbar;
    pybind11::function s_python_function_annotate;
    pybind11::function s_python_function_tight_layout;
    pybind11::function s_python_function_stem;
    pybind11::function s_python_function_xkcd;
    pybind11::function s_python_function_newbytesio;
    //pybind11::tuple s_python_empty_tuple;


    /* For now, _interpreter is implemented as a singleton since its currently not possible to have
       multiple independent embedded python interpreters without patching the python source code
       or starting a separate process for each.
        http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
       */

    static _interpreter& get()
    {
        static _interpreter ctx;
        return ctx;
    }

private:
    _interpreter() 
    {
        auto matplotlib = pybind11::module::import("matplotlib");
        auto pystylemod = pybind11::module::import("matplotlib.style");

        // matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
        // or matplotlib.backends is imported for the first time
        matplotlib.attr("use")(s_backend);

        auto pymod = pybind11::module::import("matplotlib.pyplot");
        auto pylabmod = pybind11::module::import("pylab");
        auto seabornmod = pybind11::module::import("seaborn");
        auto iomod = pybind11::module::import("io");

        s_python_function_show         = getMethod(pymod, "show");
        s_python_function_show         = getMethod(pymod, "show");
        s_python_function_close        = getMethod(pymod, "close");
        s_python_function_draw         = getMethod(pymod, "draw");
        s_python_function_pause        = getMethod(pymod, "pause");
        s_python_function_figure       = getMethod(pymod, "figure");
        s_python_function_plot         = getMethod(pymod, "plot");
        s_python_function_scatter      = getMethod(pymod, "scatter");
        s_python_function_plot_date    = getMethod(pymod, "plot_date");
        s_python_function_kdeplot      = getMethod(seabornmod, "kdeplot");
        s_python_function_heatmap      = getMethod(seabornmod, "heatmap");
        s_python_function_semilogx     = getMethod(pymod, "semilogx");
        s_python_function_semilogy     = getMethod(pymod, "semilogy");
        s_python_function_loglog       = getMethod(pymod, "loglog");
        s_python_function_fill_between = getMethod(pymod, "fill_between");
        s_python_function_hist         = getMethod(pymod,"hist");
        s_python_function_subplot      = getMethod(pymod, "subplot");
        s_python_function_legend       = getMethod(pymod, "legend");
        s_python_function_ylim         = getMethod(pymod, "ylim");
        s_python_function_title        = getMethod(pymod, "title");
        s_python_function_axis         = getMethod(pymod, "axis");
        s_python_function_xlabel       = getMethod(pymod, "xlabel");
        s_python_function_ylabel       = getMethod(pymod, "ylabel");
        s_python_function_grid         = getMethod(pymod, "grid");
        s_python_function_xlim         = getMethod(pymod, "xlim");
        s_python_function_xticks       = getMethod(pymod, "xticks");
        s_python_function_ion          = getMethod(pymod, "ion");
        s_python_function_save         = getMethod(pylabmod, "savefig");
        s_python_function_annotate     = getMethod(pymod,"annotate");
        s_python_function_clf          = getMethod(pymod, "clf");
        s_python_function_errorbar     = getMethod(pymod, "errorbar");
        s_python_function_tight_layout = getMethod(pymod, "tight_layout");
        s_python_function_stem         = getMethod(pymod, "stem");
        s_python_function_xkcd         = getMethod(pymod, "xkcd");
        s_python_function_newbytesio   = getMethod(iomod, "BytesIO");

        auto usestyle = getMethod(pystylemod, "use");
        usestyle("dark_background");
        //s_python_empty_tuple = PyTuple_New(0);
    }

    ~_interpreter()
    {
    }
};

} // end namespace detail


void legendIfLabelPresent(std::string_view label);

// must be called before the first regular call to matplotlib to have any effect
inline void backend(const std::string& name)
{
    detail::s_backend = name;
}
// 
// inline bool annotate(std::string annotation, double x, double y)
// {
//     PyObject * xy = PyTuple_New(2);
//     PyObject * str = PyString_FromString(annotation.c_str());
// 
//     PyTuple_SetItem(xy,0,PyFloat_FromDouble(x));
//     PyTuple_SetItem(xy,1,PyFloat_FromDouble(y));
// 
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "xy", xy);
// 
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, str);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_annotate, args, kwargs);
// 
//     Py_DECREF(args);
//     Py_DECREF(kwargs);
// 
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// #ifndef WITHOUT_NUMPY
// // Type selector for numpy array conversion
// template <typename T> struct select_npy_type { const static NPY_TYPES type = NPY_NOTYPE; }; //Default
// template <> struct select_npy_type<double> { const static NPY_TYPES type = NPY_DOUBLE; };
// template <> struct select_npy_type<float> { const static NPY_TYPES type = NPY_FLOAT; };
// template <> struct select_npy_type<bool> { const static NPY_TYPES type = NPY_BOOL; };
// template <> struct select_npy_type<int8_t> { const static NPY_TYPES type = NPY_INT8; };
// template <> struct select_npy_type<int16_t> { const static NPY_TYPES type = NPY_SHORT; };
// template <> struct select_npy_type<int32_t> { const static NPY_TYPES type = NPY_INT; };
// template <> struct select_npy_type<int64_t> { const static NPY_TYPES type = NPY_INT64; };
// template <> struct select_npy_type<uint8_t> { const static NPY_TYPES type = NPY_UINT8; };
// template <> struct select_npy_type<uint16_t> { const static NPY_TYPES type = NPY_USHORT; };
// template <> struct select_npy_type<uint32_t> { const static NPY_TYPES type = NPY_ULONG; };
// template <> struct select_npy_type<uint64_t> { const static NPY_TYPES type = NPY_UINT64; };
// 
// template<typename Numeric>
// PyObject* get_array(const std::vector<Numeric>& v)
// {
//     detail::_interpreter::get();    //interpreter needs to be initialized for the numpy commands to work
//     NPY_TYPES type = select_npy_type<Numeric>::type;
//     if (type == NPY_NOTYPE)
//     {
//         std::vector<double> vd(v.size());
//         npy_intp vsize = v.size();
//         std::copy(v.begin(),v.end(),vd.begin());
//         PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, NPY_DOUBLE, (void*)(vd.data()));
//         return varray;
//     }
// 
//     npy_intp vsize = v.size();
//     PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, type, (void*)(v.data()));
//     return varray;
// }
// 
// #else // fallback if we don't have numpy: copy every element of the given vector
// 
// template<typename Numeric>
// PyObject* get_array(const std::vector<Numeric>& v)
// {
//     PyObject* list = PyList_New(v.size());
//     for(size_t i = 0; i < v.size(); ++i) {
//         PyList_SetItem(list, i, PyFloat_FromDouble(v.at(i)));
//     }
//     return list;
// }
// 
// #endif // WITHOUT_NUMPY
// 
// template<typename Numeric>
// bool plot(const std::vector<Numeric> &x, const std::vector<Numeric> &y, const std::map<std::string, std::string>& keywords)
// {
//     assert(x.size() == y.size());
// 
//     // using numpy arrays
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     // construct positional args
//     PyObject* args = PyTuple_New(2);
//     PyTuple_SetItem(args, 0, xarray);
//     PyTuple_SetItem(args, 1, yarray);
// 
//     // construct keyword args
//     PyObject* kwargs = PyDict_New();
//     for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
//     {
//         PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
//     }
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, args, kwargs);
// 
//     Py_DECREF(args);
//     Py_DECREF(kwargs);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
void plot_date(pybind11::list xarray, pybind11::list yarray)
{
    detail::_interpreter::get().s_python_function_plot_date(xarray, yarray);
}

void fill_between(pybind11::list xarray, pybind11::list yarray1, pybind11::list yarray2, std::string_view label, std::string_view color, double alpha)
{
    pybind11::dict kwargs{ "alpha"_a = alpha };
    if(!label.empty())
        insert(kwargs, "label", label);
    if(!color.empty())
        insert(kwargs, "color", color);

    detail::_interpreter::get().s_python_function_fill_between(xarray, yarray1, yarray2, **kwargs);
    legendIfLabelPresent(label);
}

void scatter(pybind11::list xarray, pybind11::list yarray)
{
    detail::_interpreter::get().s_python_function_scatter(xarray, yarray);
}
 
// template<typename Numeric>
// bool stem(const std::vector<Numeric> &x, const std::vector<Numeric> &y, const std::map<std::string, std::string>& keywords)
// {
//     assert(x.size() == y.size());
// 
//     // using numpy arrays
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     // construct positional args
//     PyObject* args = PyTuple_New(2);
//     PyTuple_SetItem(args, 0, xarray);
//     PyTuple_SetItem(args, 1, yarray);
// 
//     // construct keyword args
//     PyObject* kwargs = PyDict_New();
//     for (std::map<std::string, std::string>::const_iterator it =
//             keywords.begin(); it != keywords.end(); ++it) {
//         PyDict_SetItemString(kwargs, it->first.c_str(),
//                 PyString_FromString(it->second.c_str()));
//     }
// 
//     PyObject* res = PyObject_Call(
//             detail::_interpreter::get().s_python_function_stem, args, kwargs);
// 
//     Py_DECREF(args);
//     Py_DECREF(kwargs);
//     if (res)
//         Py_DECREF(res);
// 
//     return res;
// }
// 
// template< typename Numeric >
// bool fill_between(const std::vector<Numeric>& x, const std::vector<Numeric>& y1, const std::vector<Numeric>& y2, const std::map<std::string, std::string>& keywords)
// {
//     assert(x.size() == y1.size());
//     assert(x.size() == y2.size());
// 
//     // using numpy arrays
//     PyObject* xarray = get_array(x);
//     PyObject* y1array = get_array(y1);
//     PyObject* y2array = get_array(y2);
// 
//     // construct positional args
//     PyObject* args = PyTuple_New(3);
//     PyTuple_SetItem(args, 0, xarray);
//     PyTuple_SetItem(args, 1, y1array);
//     PyTuple_SetItem(args, 2, y2array);
// 
//     // construct keyword args
//     PyObject* kwargs = PyDict_New();
//     for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
//     {
//         PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
//     }
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_fill_between, args, kwargs);
// 
//     Py_DECREF(args);
//     Py_DECREF(kwargs);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
void hist(pybind11::list yarray, size_t bins = 20,std::string color = "b", double alpha = 1.0)
{
    detail::_interpreter::get().s_python_function_hist(yarray, "bins"_a=bins, "color"_a=color, "alpha"_a=alpha);
}
// 
// template< typename Numeric>
// bool hist(const std::vector<Numeric>& y, long bins=10,std::string color="b", double alpha=1.0)
// {
// 
//     PyObject* yarray = get_array(y);
// 
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
//     PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
//     PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
// 
// 
//     PyObject* plot_args = PyTuple_New(1);
// 
//     PyTuple_SetItem(plot_args, 0, yarray);
// 
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);
// 
// 
//     Py_DECREF(plot_args);
//     Py_DECREF(kwargs);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template< typename Numeric>
// bool named_hist(std::string label,const std::vector<Numeric>& y, long bins=10, std::string color="b", double alpha=1.0)
// {
//     PyObject* yarray = get_array(y);
// 
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
//     PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
//     PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
//     PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
// 
// 
//     PyObject* plot_args = PyTuple_New(1);
//     PyTuple_SetItem(plot_args, 0, yarray);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);
// 
//     Py_DECREF(plot_args);
//     Py_DECREF(kwargs);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
void kdeplot2(pybind11::list xarray, pybind11::list yarray, const char* colorMap)
{
    detail::_interpreter::get().s_python_function_kdeplot(xarray, yarray, "cmap"_a=colorMap);
}
 
void heatmap(pybind11::list xarray, std::string_view colorMap, std::string_view annot)
{
    pybind11::dict kwargs;

    if(!colorMap.empty())
        insert(kwargs, "cmap", colorMap);

    if(!annot.empty())
    {
        insert(kwargs, "annot", pybind11::bool_(true));
        insert(kwargs, "fmt", annot);
    }
    detail::_interpreter::get().s_python_function_heatmap(xarray, **kwargs);
}
 
void kdeplot(pybind11::list xarray, const char* label)
{
    pybind11::dict kwargs;
    if(label)
        insert(kwargs, "label", pybind11::str(label));

    detail::_interpreter::get().s_python_function_kdeplot(xarray, **kwargs);
    legendIfLabelPresent(label);
}

void plot(pybind11::list xarray, pybind11::list yarray, std::string_view label, std::string_view format = "", std::string_view color = "", double alpha = 1.0)
{
    pybind11::dict kwargs;
    if(!label.empty())
        insert(kwargs, "label", label);
    if(!color.empty())
        insert(kwargs, "color", color);
    insert(kwargs, "alpha", pybind11::float_(alpha));

    auto lines = detail::_interpreter::get().s_python_function_plot(xarray, yarray, format, **kwargs);
    legendIfLabelPresent(label);
}
// 
// template<typename NumericX, typename NumericY>
// bool plot(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(s.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, plot_args);
// 
//     Py_DECREF(plot_args);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename NumericX, typename NumericY>
// bool stem(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(s.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_CallObject(
//             detail::_interpreter::get().s_python_function_stem, plot_args);
// 
//     Py_DECREF(plot_args);
//     if (res)
//         Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename NumericX, typename NumericY>
// bool semilogx(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(s.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_semilogx, plot_args);
// 
//     Py_DECREF(plot_args);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename NumericX, typename NumericY>
// bool semilogy(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(s.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_semilogy, plot_args);
// 
//     Py_DECREF(plot_args);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename NumericX, typename NumericY>
// bool loglog(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(s.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_loglog, plot_args);
// 
//     Py_DECREF(plot_args);
//     if(res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename NumericX, typename NumericY>
// bool errorbar(const std::vector<NumericX> &x, const std::vector<NumericY> &y, const std::vector<NumericX> &yerr, const std::string &s = "")
// {
//     assert(x.size() == y.size());
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
//     PyObject* yerrarray = get_array(yerr);
// 
//     PyObject *kwargs = PyDict_New();
// 
//     PyDict_SetItemString(kwargs, "yerr", yerrarray);
// 
//     PyObject *pystring = PyString_FromString(s.c_str());
// 
//     PyObject *plot_args = PyTuple_New(2);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
// 
//     PyObject *res = PyObject_Call(detail::_interpreter::get().s_python_function_errorbar, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
// 
//     if (res)
//         Py_DECREF(res);
//     else
//         throw std::runtime_error("Call to errorbar() failed.");
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool named_plot(const std::string& name, const std::vector<Numeric>& y, const std::string& format = "")
// {
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));
// 
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(format.c_str());
// 
//     PyObject* plot_args = PyTuple_New(2);
// 
//     PyTuple_SetItem(plot_args, 0, yarray);
//     PyTuple_SetItem(plot_args, 1, pystring);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
//     if (res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool named_plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
// {
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(format.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
//     if (res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool named_semilogx(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
// {
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(format.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_semilogx, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
//     if (res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool named_semilogy(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
// {
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(format.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_semilogy, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
//     if (res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool named_loglog(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
// {
//     PyObject* kwargs = PyDict_New();
//     PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));
// 
//     PyObject* xarray = get_array(x);
//     PyObject* yarray = get_array(y);
// 
//     PyObject* pystring = PyString_FromString(format.c_str());
// 
//     PyObject* plot_args = PyTuple_New(3);
//     PyTuple_SetItem(plot_args, 0, xarray);
//     PyTuple_SetItem(plot_args, 1, yarray);
//     PyTuple_SetItem(plot_args, 2, pystring);
// 
//     PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_loglog, plot_args, kwargs);
// 
//     Py_DECREF(kwargs);
//     Py_DECREF(plot_args);
//     if (res) Py_DECREF(res);
// 
//     return res;
// }
// 
// template<typename Numeric>
// bool plot(const std::vector<Numeric>& y, const std::string& format = "")
// {
//     std::vector<Numeric> x;
//     x.resize(y.size());
//     for(size_t i=0; i < x.size(); ++i) 
//         x.at(i) = (Numeric)i;
// 
//     return plot(x,y,format);
// }
// 
// template<typename Numeric>
// bool stem(const std::vector<Numeric>& y, const std::string& format = "")
// {
//     std::vector<Numeric> x(y.size());
//     for (size_t i = 0; i < x.size(); ++i) x.at(i) = i;
//     return stem(x, y, format);
// }
// 
// inline void figure()
// {
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure, detail::_interpreter::get().s_python_empty_tuple);
//     if(!res) throw std::runtime_error("Call to figure() failed.");
// 
//     Py_DECREF(res);
// }
// 
inline void figure_size(size_t w, size_t h)
{
    double dpi = 100;
    auto size = pybind11::make_tuple(w / dpi, h / dpi); 
    detail::_interpreter::get().s_python_function_figure("figsize"_a=size, "dpi"_a = dpi);
}
 
inline void rotate_ticks(long rot)
{
    detail::_interpreter::get().s_python_function_xticks("rotation"_a = rot, "ha"_a = "right");
}

inline void legend()
{
    detail::_interpreter::get().s_python_function_legend();
}

inline void legendIfLabelPresent(std::string_view label)
{
    if(!label.empty())
        legend();
}
// 
// template<typename Numeric>
// void ylim(Numeric left, Numeric right)
// {
//     PyObject* list = PyList_New(2);
//     PyList_SetItem(list, 0, PyFloat_FromDouble(left));
//     PyList_SetItem(list, 1, PyFloat_FromDouble(right));
// 
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, list);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylim, args);
//     if(!res) throw std::runtime_error("Call to ylim() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// template<typename Numeric>
// void xlim(Numeric left, Numeric right)
// {
//     PyObject* list = PyList_New(2);
//     PyList_SetItem(list, 0, PyFloat_FromDouble(left));
//     PyList_SetItem(list, 1, PyFloat_FromDouble(right));
// 
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, list);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlim, args);
//     if(!res) throw std::runtime_error("Call to xlim() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// 
// inline double* xlim()
// {
//     PyObject* args = PyTuple_New(0);
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlim, args);
//     PyObject* left = PyTuple_GetItem(res,0);
//     PyObject* right = PyTuple_GetItem(res,1);
// 
//     double* arr = new double[2];
//     arr[0] = PyFloat_AsDouble(left);
//     arr[1] = PyFloat_AsDouble(right);
// 
//     if(!res) throw std::runtime_error("Call to xlim() failed.");
// 
//     Py_DECREF(res);
//     return arr;
// }
// 
// 
// inline double* ylim()
// {
//     PyObject* args = PyTuple_New(0);
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylim, args);
//     PyObject* left = PyTuple_GetItem(res,0);
//     PyObject* right = PyTuple_GetItem(res,1);
// 
//     double* arr = new double[2];
//     arr[0] = PyFloat_AsDouble(left);
//     arr[1] = PyFloat_AsDouble(right);
// 
//     if(!res) throw std::runtime_error("Call to ylim() failed.");
// 
//     Py_DECREF(res);
//     return arr;
// }
// 
inline void subplot(long nrows, long ncols, long plot_number)
{
    detail::_interpreter::get().s_python_function_subplot(nrows, ncols, plot_number);
}
// 
// inline void title(const std::string &titlestr)
// {
//     PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, pytitlestr);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_title, args);
//     if(!res) throw std::runtime_error("Call to title() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// inline void axis(const std::string &axisstr)
// {
//     PyObject* str = PyString_FromString(axisstr.c_str());
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, str);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_axis, args);
//     if(!res) throw std::runtime_error("Call to title() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// inline void xlabel(const std::string &str)
// {
//     PyObject* pystr = PyString_FromString(str.c_str());
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, pystr);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlabel, args);
//     if(!res) throw std::runtime_error("Call to xlabel() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// inline void ylabel(const std::string &str)
// {
//     PyObject* pystr = PyString_FromString(str.c_str());
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, pystr);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylabel, args);
//     if(!res) throw std::runtime_error("Call to ylabel() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// inline void grid(bool flag)
// {
//     PyObject* pyflag = flag ? Py_True : Py_False;
//     Py_INCREF(pyflag);
// 
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, pyflag);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_grid, args);
//     if(!res) throw std::runtime_error("Call to grid() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
inline void show(pybind11::bool_ block = true)
{
    detail::_interpreter::get().s_python_function_show("block"_a = block);
}
// 
// inline void close()
// {
//     PyObject* res = PyObject_CallObject(
//             detail::_interpreter::get().s_python_function_close,
//             detail::_interpreter::get().s_python_empty_tuple);
// 
//     if (!res) throw std::runtime_error("Call to close() failed.");
// 
//     Py_DECREF(res);
// }
// 
// inline void xkcd() {
//     PyObject* res;
//     PyObject *kwargs = PyDict_New();
// 
//     res = PyObject_Call(detail::_interpreter::get().s_python_function_xkcd,
//             detail::_interpreter::get().s_python_empty_tuple, kwargs);
// 
//     Py_DECREF(kwargs);
// 
//     if (!res)
//         throw std::runtime_error("Call to show() failed.");
// 
//     Py_DECREF(res);
// }
// 
// inline void draw()
// {
//     PyObject* res = PyObject_CallObject(
//         detail::_interpreter::get().s_python_function_draw,
//         detail::_interpreter::get().s_python_empty_tuple);
// 
//     if (!res) throw std::runtime_error("Call to draw() failed.");
// 
//     Py_DECREF(res);
// }
// 
// template<typename Numeric>
// inline void pause(Numeric interval)
// {
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_pause, args);
//     if(!res) throw std::runtime_error("Call to pause() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
// inline void save(const std::string& filename)
// {
//     PyObject* pyfilename = PyString_FromString(filename.c_str());
// 
//     PyObject* args = PyTuple_New(1);
//     PyTuple_SetItem(args, 0, pyfilename);
// 
//     PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_save, args);
//     if (!res) throw std::runtime_error("Call to save() failed.");
// 
//     Py_DECREF(args);
//     Py_DECREF(res);
// }
// 
inline std::string getPNG()
{
    auto bytesIO = detail::_interpreter::get().s_python_function_newbytesio();
    auto res = detail::_interpreter::get().s_python_function_save(bytesIO, "transparent"_a=true);

    pybind11::bytes bytes = getMethod(bytesIO, "getvalue")();
    return bytes;
}
// 
// inline void clf() {
//     PyObject *res = PyObject_CallObject(
//         detail::_interpreter::get().s_python_function_clf,
//         detail::_interpreter::get().s_python_empty_tuple);
// 
//     if (!res) throw std::runtime_error("Call to clf() failed.");
// 
//     Py_DECREF(res);
// }
// 
//     inline void ion() {
//     PyObject *res = PyObject_CallObject(
//         detail::_interpreter::get().s_python_function_ion,
//         detail::_interpreter::get().s_python_empty_tuple);
// 
//     if (!res) throw std::runtime_error("Call to ion() failed.");
// 
//     Py_DECREF(res);
// }
// 
// Actually, is there any reason not to call this automatically for every plot?
inline void tight_layout()
{
    detail::_interpreter::get().s_python_function_tight_layout();
}

// #if __cplusplus > 199711L || _MSC_VER > 1800
// // C++11-exclusive content starts here (variadic plot() and initializer list support)
// 
// namespace detail {
// 
// template<typename T>
// using is_function = typename std::is_function<std::remove_pointer<std::remove_reference<T>>>::type;
// 
// template<bool obj, typename T>
// struct is_callable_impl;
// 
// template<typename T>
// struct is_callable_impl<false, T>
// {
//     typedef is_function<T> type;
// }; // a non-object is callable iff it is a function
// 
// template<typename T>
// struct is_callable_impl<true, T>
// {
//     struct Fallback { void operator()(); };
//     struct Derived : T, Fallback { };
// 
//     template<typename U, U> struct Check;
// 
//     template<typename U>
//     static std::true_type test( ... ); // use a variadic function to make sure (1) it accepts everything and (2) its always the worst match
// 
//     template<typename U>
//     static std::false_type test( Check<void(Fallback::*)(), &U::operator()>* );
// 
// public:
//     typedef decltype(test<Derived>(nullptr)) type;
//     typedef decltype(&Fallback::operator()) dtype;
//     static constexpr bool value = type::value;
// }; // an object is callable iff it defines operator()
// 
// template<typename T>
// struct is_callable
// {
//     // dispatch to is_callable_impl<true, T> or is_callable_impl<false, T> depending on whether T is of class type or not
//     typedef typename is_callable_impl<std::is_class<T>::value, T>::type type;
// };
// 
// template<typename IsYDataCallable>
// struct plot_impl { };
// 
// template<>
// struct plot_impl<std::false_type>
// {
//     template<typename IterableX, typename IterableY>
//     bool operator()(const IterableX& x, const IterableY& y, const std::string& format)
//     {
//         // 2-phase lookup for distance, begin, end
//         using std::distance;
//         using std::begin;
//         using std::end;
// 
//         auto xs = distance(begin(x), end(x));
//         auto ys = distance(begin(y), end(y));
//         assert(xs == ys && "x and y data must have the same number of elements!");
// 
//         PyObject* xlist = PyList_New(xs);
//         PyObject* ylist = PyList_New(ys);
//         PyObject* pystring = PyString_FromString(format.c_str());
// 
//         auto itx = begin(x), ity = begin(y);
//         for(size_t i = 0; i < xs; ++i) {
//             PyList_SetItem(xlist, i, PyFloat_FromDouble(*itx++));
//             PyList_SetItem(ylist, i, PyFloat_FromDouble(*ity++));
//         }
// 
//         PyObject* plot_args = PyTuple_New(3);
//         PyTuple_SetItem(plot_args, 0, xlist);
//         PyTuple_SetItem(plot_args, 1, ylist);
//         PyTuple_SetItem(plot_args, 2, pystring);
// 
//         PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, plot_args);
// 
//         Py_DECREF(plot_args);
//         if(res) Py_DECREF(res);
// 
//         return res;
//     }
// };
// 
// template<>
// struct plot_impl<std::true_type>
// {
//     template<typename Iterable, typename Callable>
//     bool operator()(const Iterable& ticks, const Callable& f, const std::string& format)
//     {
//         if(begin(ticks) == end(ticks)) return true;
// 
//         // We could use additional meta-programming to deduce the correct element type of y,
//         // but all values have to be convertible to double anyways
//         std::vector<double> y;
//         for(auto x : ticks) y.push_back(f(x));
//         return plot_impl<std::false_type>()(ticks,y,format);
//     }
// };
// 
// } // end namespace detail
// 
// // recursion stop for the above
// template<typename... Args>
// bool plot() { return true; }
// 
// template<typename A, typename B, typename... Args>
// bool plot(const A& a, const B& b, const std::string& format, Args... args)
// {
//     return detail::plot_impl<typename detail::is_callable<B>::type>()(a,b,format) && plot(args...);
// }
// 
// /*
//  * This group of plot() functions is needed to support initializer lists, i.e. calling
//  *    plot( {1,2,3,4} )
//  */
// inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
//     return plot<double,double>(x,y,format);
// }
// 
// inline bool plot(const std::vector<double>& y, const std::string& format = "") {
//     return plot<double>(y,format);
// }
// 
// inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::map<std::string, std::string>& keywords) {
//     return plot<double>(x,y,keywords);
// }
// 
// #endif

} // end namespace matplotlibcpp
