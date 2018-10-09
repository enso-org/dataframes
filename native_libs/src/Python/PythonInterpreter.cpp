#include <date/date.h>
#include "Core/Common.h"
#include "PythonInterpreter.h"

PythonInterpreter::PythonInterpreter()
{
    try
    {
        const auto programName = L"Dataframes";
        Py_SetProgramName(const_cast<wchar_t *>(programName));
        pybind11::initialize_interpreter();
        PyDateTime_IMPORT;
        if(PyDateTimeAPI == nullptr)
            throw pybind11::error_already_set();

        if(_import_array() < 0)
            throw pybind11::error_already_set();
    }
    catch(std::exception &e)
    {
        auto s = fmt::format("failed to initialize Python bindings: {}", e);

        // As an exception, we shall complain to output:
        // Such error can happen during shared library loading and can be pretty unpleasant to debug.
        std::cerr << s << std::endl;
        throw std::runtime_error(s);
    }
}

PythonInterpreter::~PythonInterpreter()
{
    pybind11::finalize_interpreter();
}

PythonInterpreter &PythonInterpreter::instance()
{
    static PythonInterpreter p;
    return p;
}

pybind11::object PythonInterpreter::toPyDateTime(Timestamp timestamp) const
{
    using namespace date;
    auto daypoint = floor<days>(timestamp);
    auto ymd = year_month_day(daypoint);   // calendar date
    time_of_day tod = make_time(timestamp - daypoint); // Yields time_of_day type

                                               // Obtain individual components as integers
    auto y = (int)ymd.year();
    auto m = (int)(unsigned)ymd.month();
    auto d = (int)(unsigned)ymd.day();
    auto h = (int)tod.hours().count();
    auto min = (int)tod.minutes().count();
    auto s = (int)tod.seconds().count();
    auto us = (int)std::chrono::duration_cast<std::chrono::microseconds>(tod.subseconds()).count();
    
    auto ret = PyDateTime_FromDateAndTime(y, m, d, h, min, s, us);
    if(!ret)
        throw pybind11::error_already_set();

    return pybind11::reinterpret_steal<pybind11::object>(ret);
}

namespace
{
    struct AutomaticallyCreateInterpreter
    {
        AutomaticallyCreateInterpreter()
        {
            PythonInterpreter::instance();
        }
    } guardian;
}
