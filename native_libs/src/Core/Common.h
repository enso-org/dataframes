#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>

#ifdef _MSC_VER
#define EXPORT _declspec(dllexport)
#else
#define EXPORT
#endif

constexpr size_t operator"" _z (unsigned long long n)
{
    return n;
}

// C++17 at its "best"
// helpers for variant visitation with lambda set
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;


// Abominable workaround - standard library on Mac does not have invoke_result
// And we don't just use std::result_of_t, as conforming compilers hate it
// (it has been deperecated in C++17 and removed in C++20)
#if !defined(_MSC_VER) && !defined(__cpp_lib_is_invocable)
namespace std
{
    template<typename F, typename ...Args>
    using invoke_result_t = std::result_of_t<F(Args...)>;
}
#endif

template<typename F, typename ...Args>
static auto duration(F&& func, Args&&... args)
{
    const auto start = std::chrono::steady_clock::now();
    std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    return std::chrono::steady_clock::now() - start;
}

template<typename F, typename ...Args>
static auto measure(std::string text, F&& func, Args&&... args)
{
    const auto t = duration(std::forward<F>(func), std::forward<Args>(args)...);
    std::cout << text << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(t).count() << " ms" << std::endl;
    return t;
}