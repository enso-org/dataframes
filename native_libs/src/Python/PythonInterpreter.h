#pragma once
#include "Core/Common.h"
#include "Core/ArrowUtilities.h"
#include "IncludePython.h"

struct DFH_EXPORT PythonInterpreter
{
    PythonInterpreter();
    ~PythonInterpreter();

    static PythonInterpreter &instance();
    static std::string libraryName();
#ifndef _WIN32
    static void setEnvironment();
#endif

    pybind11::object toPyDateTime(Timestamp timestamp) const;
};

// TLDR: If .cpp file uses numpy it must contain this macro.
//
// NOTE: [MWU]
// NumPy uses not entirely fortunate approach that upon initialization call
// a `PyArray_API` variable is set. However the variable is declared as static
// in the header -- so every compilation unit including NumPy gets its own copy
// of `PyArray_API`. And every single copy needs to be initialized.
//
// NumPy provides a few macros that allow controlling the behavior to some
// degree. While it may be enough for projects consisting of multiple 
// compilation units in a single assembly, I don't think it is possible to get
// this working with our project layout that spans across several shared 
// libraries. (PyArray_API would have been needed to be marked with dllimport)
//
// Perhaps someone smarter (or with more time to spend on this) can devise a 
// solution that works on all 3 supported platforms and is not as ugly as this
// macro-based workaround.
#define COMPILATION_UNIT_USING_NUMPY                  \
namespace                                             \
{                                                     \
    struct NumpyGuard                                 \
    {                                                 \
        NumpyGuard()                                  \
        {                                             \
            PythonInterpreter::instance();            \
            if(_import_array() < 0)                   \
                throw pybind11::error_already_set();  \
        }                                             \
    } numpyGuard;                                     \
}