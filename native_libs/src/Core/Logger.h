#pragma once
#include <atomic>
#include "Common.h"

class DFH_EXPORT Logger
{
public:
    std::atomic_bool enabled;

	Logger();
	~Logger();

    static Logger &instance() 
    {
        static Logger logger;
        return logger;
    }
};



#ifdef VERBOSE
    #include <fmt/format.h>
    #include <iostream>
    #pragma comment(lib, "fmt.lib")
    #define LOG(...) do                                       \
    {                                                         \
        if(Logger::instance().enabled.load())                 \
        {                                                     \
            fmt::print("C++ {}: ", __FUNCTION__);             \
            fmt::print(__VA_ARGS__); std::cout << std::endl;  \
        }                                                     \
    } while(0)
#else
    #define LOG(...) do {} while(0)
#endif