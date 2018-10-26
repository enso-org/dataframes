#include <date/date.h>
#include <fstream>

#include "Core/Common.h"
#include "PythonInterpreter.h"
#include "IO/IO.h"
#include <cstdlib>

#ifdef __linux__
#include <boost/filesystem.hpp>
#endif

void printPyEnv()
{
    auto printvar = [] (const char *var) {  std::cout << var << "=" << std::getenv(var) << std::endl; };
    printvar("PYTHONHOME");
    printvar("PYTHONPATH");
}

PythonInterpreter::PythonInterpreter()
{
    try
    {
        std::cout << "Python interpreter setup" << std::endl;
        const auto programName = L"Dataframes";
        Py_SetProgramName(const_cast<wchar_t *>(programName));
        printPyEnv();

#ifdef __linux__
        // If needed, environment must be set before initializing the interpreter.
        // Otherwise, we'll get tons of error like:
        // Could not find platform independent libraries <prefix>
        // Could not find platform dependent libraries <exec_prefix>
        // Consider setting $PYTHONHOME to <prefix>[:<exec_prefix>]
        // Fatal Python error: initfsencoding: unable to load the file system codec
        // ModuleNotFoundError: No module named 'encodings'

        setEnvironment();
#endif

        std::cout << "will initialize\n";
        printPyEnv();
        pybind11::initialize_interpreter();
        PyDateTime_IMPORT;
        if(PyDateTimeAPI == nullptr)
            throw pybind11::error_already_set();

        printPyEnv();
        std::cout << "will import array\n";
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

#ifdef __linux__
std::string PythonInterpreter::libraryName()
{
    return fmt::format("libpython{}.{}m.so", PY_MAJOR_VERSION, PY_MINOR_VERSION);
}

boost::filesystem::path pythonInterprerLibraryPath(std::string_view libraryName)
{
    auto pid = getpid();
    auto mapsIn = openFileToRead(fmt::format("/proc/{}/maps", pid));

    std::string line;
    while (std::getline(mapsIn, line))
    {
        if(line.find(libraryName) == std::string::npos)
            continue;

        // we expect line like that below:
        // 7f568cd8a000-7f568d17c000 r-xp 00000000 08:01 11543023                   /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0
        // we assume that path is from the first slash to the end of line
        auto firstSlash = line.find('/');
        if(firstSlash == std::string::npos)
        {
            // should not happen, as the paths are absolute
            std::cerr << "Unexpected entry in /proc/pid/maps: " << line << std::endl;
            continue;
        }

        return line.substr(firstSlash);
    }

    THROW("Failed to found {}", libraryName);
}

void PythonInterpreter::setEnvironment()
{
    std::cout << "Will prepare Python environemt" << std::endl;
    // Python interpreter library typically lies in path like: /home/mwu/Dataframes/lib/libpython3.6m.so
    // In such case we want to set following paths:
    // PYTHONHOME=/home/mwu/Dataframes/lib/
    // PYTHONPATH=/home/mwu/Dataframes/python-libs:/home/mwu/Dataframes/python-libs/lib-dynload:/home/mwu/Dataframes/python-libs/site-packages
    auto pythonSo = pythonInterprerLibraryPath(PythonInterpreter::libraryName());
    std::cout << "Deduced Python library location is " << pythonSo << std::endl;

    auto pythonHome = pythonSo.parent_path();

    auto pythonLibs = pythonHome.parent_path() / "python-libs";
    auto pythonPath = fmt::format("{}:{}:{}", pythonLibs.c_str(), (pythonLibs / "lib-dynload").c_str(), (pythonLibs / "site-packages").c_str());

    std::cout << "home " << pythonHome << std::endl;
    std::cout << "path " << pythonPath << std::endl;
    setenv("PYTHONHOME", pythonHome.c_str(), 1);
    setenv("PYTHONPATH", pythonPath.c_str(), 1);
    printPyEnv();
}
#endif

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
