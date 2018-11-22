#include <date/date.h>
#include <fstream>

#include "Core/Common.h"
#include "PythonInterpreter.h"
#include "IO/IO.h"
#include <cstdlib>

#ifndef _WIN32
#include <boost/filesystem.hpp>
#endif

#ifdef __linux__
#include <dlfcn.h> // for dlopen
#endif

std::wstring widenString(const char *narrow)
{
    auto tmp = Py_DecodeLocale(narrow, nullptr);
    std::wstring ret = tmp;
    PyMem_RawFree(tmp);
    return ret;
}

PythonInterpreter::PythonInterpreter()
{
    try
    {
        // Workaround - various shared libraries being part of Python packages depend on libpython symbols
        // but to not declare an explicit dependency on libpython. Because of that an attempt to load them
        // (that will be made by e.g. when importing modules, like multiarray)
        // will end up in failure due to undefined symbol (like PyFloat_Type).
        // See also:
        // * https://github.com/luna/Dataframes/issues/48
        // * https://bugs.python.org/issue4434
        // * https://github.com/Kitware/kwiver/pull/388
#ifdef __linux__
        // NOTE: [MWU] It's not really clear to me why we have to pass the whole path to the library instead of just library name.
        // Library name fails to be resolved by dlopen even though this library has RPATH set to the library's location.
        // Perhaps it is that because dlopen uses our process executable (typically shell luna or luna-empire) RPATH?
        // Not really sure. It works with just a name when I try to run our (RPATH-adjusted) test executable
        // but doesn't work when doing exactly the same from Luna.
        // Well, the approach below seemingly does work for all cases, so let's just be happy with that.
        boost::filesystem::path pythonInterprerLibraryPath(std::string_view libraryName);
        auto pythonLibraryPath = pythonInterprerLibraryPath(libraryName());
        if(!dlopen(pythonLibraryPath.c_str(), RTLD_LAZY | RTLD_GLOBAL))
            THROW("Failed to load {}: {}", pythonLibraryPath, dlerror());
#endif

        const auto programName = L"Dataframes";
        Py_SetProgramName(const_cast<wchar_t *>(programName));

#ifndef _WIN32
        // If needed, environment must be set before initializing the interpreter.
        // Otherwise, we'll get tons of error like:
        // Could not find platform independent libraries <prefix>
        // Could not find platform dependent libraries <exec_prefix>
        // Consider setting $PYTHONHOME to <prefix>[:<exec_prefix>]
        // Fatal Python error: initfsencoding: unable to load the file system codec
        // ModuleNotFoundError: No module named 'encodings'
        setEnvironment();
#endif

        pybind11::initialize_interpreter();
        PyDateTime_IMPORT;
        if(PyDateTimeAPI == nullptr)
            throw pybind11::error_already_set();

        if(_import_array() < 0)
            throw pybind11::error_already_set();

        // Without workaround below Dataframes behave like a fork-bomb on mac.
        // multiprocessing package is our transitive dependency (through sklearn).
        // upon initialization multiprocessing tries to spawn python process that
        // will act as semaphore_tracker.
        // To spawn process sys.executable is called. Which is fine... if it is a
        // python interpreter. However, it fails with embedded interpreters.
        // We don't want our process to start another luna-empire (or whatever
        // executable uses dataframes libray), so we need explicitly tell it to use
        // python3 as an executable name.
#ifdef __APPLE__
        auto multiprocessing = pybind11::module::import("multiprocessing");
        multiprocessing.attr("set_executable")("python3");
#endif
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

std::string PythonInterpreter::libraryName()
{

#if defined(__APPLE__)
    return fmt::format("libpython{}.{}m.dylib", PY_MAJOR_VERSION, PY_MINOR_VERSION);
#elif defined(__linux__)
    return fmt::format("libpython{}.{}m.so", PY_MAJOR_VERSION, PY_MINOR_VERSION);
#elif defined(_WIN32)
    return fmt::format("python{}{}.dll", PY_MAJOR_VERSION, PY_MINOR_VERSION);
#else
    #error "unknown system"
#endif
}

#ifdef __linux__
boost::filesystem::path loadedLibraryPath(std::string_view libraryName)
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

    THROW("Failed to find {}", libraryName);
}
#endif

#ifdef __APPLE__


boost::filesystem::path loadedLibraryPath(std::string libraryName)
{
    auto pid = getpid();
    std::string command = fmt::format("vmmap {}", pid);
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe)
        throw std::runtime_error("popen() failed, command was: " + command);

    char line[4096];
    while (!feof(pipe.get()))
    {
        if (fgets(line, std::size(line), pipe.get()) != nullptr)
        {
            if(std::strstr(line, libraryName.c_str()) != nullptr)
            {
                if (auto pos = std::strstr(line, " /"))
                {
                    return std::string(pos + 1);
                }
                else
                {
                    // unexpected
                    std::cerr << "unexpected line in vmmap output: " << line << std::endl;
                    continue;
                }
            }
        }
    }

    THROW("Failed to find {}", libraryName);
}

#endif // __APPLE__


#ifndef _WIN32
void PythonInterpreter::setEnvironment()
{
    // Python interpreter library typically lies in path like: /home/mwu/Dataframes/lib/libpython3.6m.so
    // In such case we want to set following paths:
    // PYTHONHOME=/home/mwu/Dataframes/lib/
    // PYTHONPATH=/home/mwu/Dataframes/python-libs:/home/mwu/Dataframes/python-libs/lib-dynload:/home/mwu/Dataframes/python-libs/site-packages
    const auto pythonSo = loadedLibraryPath(PythonInterpreter::libraryName());

    // However, we want to set home and path only when we use our packaged python ditribution.
    // If this is developer build using the system-wide Python, the we should not touch anything
    // or else we will only break things.
    // Current packaging scheme assumes that there's directory named Dataframes (package name).
    // Typical Python installation from system repository doesn't have such folder in path.
    // (for developer builds such heuristic should suffice just fine)
    if(pythonSo.string().find("Dataframes") == std::string::npos)
        return;

    const auto pythonHome = pythonSo.parent_path();
    const auto pythonLibs =  pythonHome.parent_path() / fmt::format("python{}.{}", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    const auto pythonPath = fmt::format("{}:{}:{}", pythonLibs.c_str(), (pythonLibs / "lib-dynload").c_str(), (pythonLibs / "site-packages").c_str());
    std::cout << "setting PYTHONHOME " << pythonHome << std::endl;
    std::cout << "setting PYTHONPATH " << pythonPath << std::endl;
    Py_SetPythonHome(widenString(pythonHome.c_str()).data());
    Py_SetPath(widenString(pythonPath.c_str()).data());
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
