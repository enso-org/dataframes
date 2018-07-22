#pragma once

#ifdef _MSC_VER
#define EXPORT _declspec(dllexport)
#else
#define EXPORT
#endif

constexpr size_t operator"" _z (unsigned long long n)
{
	return n;
}

#ifdef VERBOSE
#include "fmt/format.h"
	#pragma comment(lib, "fmt.lib")
	#define LOG(...) do { fmt::print("C++ {}: ", __FUNCTION__); fmt::print(__VA_ARGS__); std::cout << std::endl; } while(0)
#else
#define LOG(...) 
#endif