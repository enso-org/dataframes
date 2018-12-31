#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

#include "optional.h"

#include <fmt/format.h>
#include <fmt/ostream.h> // needed for fmt to use user-provided operator<<

using namespace std::literals;
using namespace std::chrono_literals;

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

// NO_INLINE macro is meant to prevent compiler from inlining function.
// primary intended use-case is for dbeugging / profiling purposes.
#ifdef _MSC_VER
#define NO_INLINE __declspec(noinline)
#else
#define NO_INLINE __attribute__((noinline))
#endif

// intellisense is checked because of MSVC bug: https://developercommunity.visualstudio.com/content/problem/335672/c-intellisense-stops-working-with-given-code.html
#if defined(_MSC_VER) && !defined(__INTELLISENSE__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT [[gnu::visibility ("default")]]
#endif

#ifdef BUILDING_DATAFRAME_HELPER
#define DFH_EXPORT EXPORT
#else
#define DFH_EXPORT
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
// (it has been deprecated in C++17 and removed in C++20)
//
// The issue was fixed in XCode 10 but they didn't update the feature macro. :/
#if !defined(_MSC_VER) && !defined(__cpp_lib_is_invocable) && __clang_major__ < 10
namespace std
{
    template<typename F, typename ...Args>
    using invoke_result_t = std::result_of_t<F(Args...)>;
}
#endif

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
    inline std::ostream &operator<<(std::ostream &out, const std::exception &e)
    {
        return out << e.what();
    }

    inline std::ostream &operator<<(std::ostream &out, std::nullopt_t)
    {
        return out << "[none]";
    }

    inline std::ostream &operator<<(std::ostream &out, std::chrono::duration<int64_t, std::nano> d)
    {
        return out << d.count() << "ns";
    }

    inline std::ostream &operator<<(std::ostream &out, const std::type_info &info)
    {
        return out << info.name();
    }

    template<typename T>
    inline std::ostream &operator<<(std::ostream &out, const std::optional<T> &opt)
    {
        if(opt)
            return out << *opt;
        else
            return out << std::nullopt;
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const std::vector<T> &arr)
    {
        out << "{ ";
        if(arr.size())
            out << arr[0];

        for(int i = 1; i < (int)arr.size(); i++)
            out << ", " << arr[i];
        out << " }";
        return out;
    }
}

void validateIndex(int64_t size, int64_t index);
void validateSlice(int64_t dataLength, int64_t sliceStart, int64_t sliceLength);

template<typename T>
T lerp(T v0, T v1, double t)
{
    return (1 - t) * v0 + t * v1;
}

// Disabled due to MSVC bug: https://developercommunity.visualstudio.com/content/problem/327775/problem-with-auto-template-non-type-parameter-and.html
// Very similar bug in GCC 7.
// template<auto Value>
// auto makeIntegralConstant()
// {
//     using T = decltype(Value);
//     return std::integral_constant<T, Value>{};
// }

template<typename T = int64_t>
std::vector<T> iotaVector(size_t N, T from = T{})
{
    std::vector<T> ret;
    ret.resize(N);
    for(auto &elem : ret)
        elem = from++;
    return ret;
}


#define THROW(message, ...)  do {                                         \
	auto msg_ =  fmt::format(message, ##__VA_ARGS__);                     \
	std::cerr << __FILE__ << " " << __LINE__ << " " << msg_ << std::endl; \
	std::runtime_error e_to_throw{fmt::format(message, ##__VA_ARGS__)};   \
	throw e_to_throw;                                                     \
} while(0)

namespace detail
{
    template <template <class...> class Trait, class Enabler, class... Args>
    struct is_detected : std::false_type{};

    template <template <class...> class Trait, class... Args>
    struct is_detected<Trait, std::void_t<Trait<Args...>>, Args...> : std::true_type{};
}

template <template <class...> class Trait, class... Args>
using is_detected = typename detail::is_detected<Trait, void, Args...>::type;

template <template <class...> class Trait, class... Args>
constexpr bool is_detected_v = is_detected<Trait, Args...>::value;

template<typename Range, typename Functor, typename Value>
Value maxElementValue(Range &&range, Value forEmptyRange, Functor &&f)
{
    if(std::empty(range))
        return forEmptyRange;

    auto minItr = std::max_element(std::begin(range), std::end(range), [&] (auto &&lhs, auto &&rhs)
        { return f(lhs) < f(rhs); });
    return f(*minItr);
}
