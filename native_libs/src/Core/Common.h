#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>

#include "optional.h"

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif 

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

// to allow conditional static_asserts 
template<class T> struct always_false : std::false_type {};
template<class T> constexpr bool always_false_v = always_false<T>::value;

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

template<typename F, typename ...Args>
static auto measure(std::string text, int N, F&& func, Args&&... args)
{
    std::chrono::microseconds bestTime = std::chrono::hours{150};
    for(int i = 0; i < N; i++)
    {
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(duration(std::forward<F>(func), std::forward<Args>(args)...));
        bestTime = std::min(bestTime, t);
        std::cout << text << " took " << t.count() / 1000.0 << " ms, best: " << bestTime.count() / 1000. << " ms" << std::endl;
    }
}

template<typename Range, typename F>
auto transformToVector(Range &&range, F &&f)
{
    using SourceT = decltype(*std::begin(range));
    using T = std::invoke_result_t<F, SourceT>;

    std::vector<T> ret;
    // ret.reserve(std::size(range));
    ret.reserve(std::distance(range.begin(), range.end())); // ugly because rapidjson array

    for(auto &&elem : range)
        ret.push_back(f(elem));

    return ret;
}

template<typename>
struct is_optional : std::false_type {};

template<typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template<typename T>
constexpr bool is_optional_v = is_optional<T>::value;

namespace std
{
    inline std::ostream &operator<<(std::ostream &out, std::nullopt_t)
    {
        return out << "[none]";
    }

    template<typename T>
    inline std::ostream &operator<<(std::ostream &out, const std::optional<T> &opt)
    {
        if(opt)
            return out << *opt;
        else
            return out << std::nullopt;
    }
}

void validateIndex(const size_t size, int64_t index);
