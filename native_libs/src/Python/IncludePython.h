#pragma once

// Windows-specific issue workaround:
// Note [MU]:
// When _DEBUG is defined, Python defines Py_DEBUG and Py_DEBUG leads to
// Py_DECREF expanding to special checking code that uses symbols available
// only in debug binaries of Python. That causes linker error on Windows
// when using Release Python binaries with Debug Dataframe build.
//
// And we don't want to use Debug binaries of Python, because they are 
// incompatible with Release packages installed through pip (e.g. numpy)
// and having Debug and Release packages side-by-side looks non-trivial.
// Perhaps someone with better Python knowledge can improve this is future.
// 
// For now just let's try to trick Python into thinking that we are in Release
// mode and hope that no one else includes this header. 
// And that standard library/runtime won't explode.
#if defined(_DEBUG) && defined(_MSC_VER)
#define WAS_DEBUG
#undef _DEBUG
#endif

// Declare that we don't want deprecated APIs -- otherwise we get a warning.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifndef WITHOUT_NUMPY
#include <numpy/arrayobject.h>
#endif // WITHOUT_NUMPY

#include <Python.h>
#include <datetime.h>

#ifdef WAS_DEBUG
#define _DEBUG 1
#endif

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Python makes some defines that break other headers (namely Arrow's)
#undef timezone

#include <Core/Common.h>

DFH_EXPORT pybind11::function getMethod(pybind11::object module, const std::string &attributeName);

namespace pybind11
{
    DFH_EXPORT void insert(pybind11::dict dict, const char *key, std::string_view value);
    DFH_EXPORT void insert(pybind11::dict dict, const char *key, pybind11::object value);
    DFH_EXPORT void setAt(pybind11::list list, size_t index, pybind11::object value);
}
