#pragma once

#include <cstddef>

#ifdef _MSC_VER
#define EXPORT _declspec(dllexport)
#else
#define EXPORT
#endif

constexpr size_t operator"" _z (unsigned long long n)
{
    return n;
}
