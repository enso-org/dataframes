#pragma once

#include <cassert>
#include <cstdlib>
#include "optional.h"
#include <string_view>
#include <type_traits>
#include <date/date.h>

#include "Common.h"
#include "ArrowUtilities.h"

DFH_EXPORT std::optional<Timestamp> parseTimestamp(std::string_view text);

struct OldStyleNumberParser
{
    static constexpr bool requiresNull = true;

    template<typename T>
    static std::optional<T> as(std::string_view text)
    {
        if(text.empty())
            return {};

        if constexpr(std::is_same_v<Timestamp, T>)
        {
            return parseTimestamp(text);
        }
        else
        {
            char* next = nullptr;
            auto v = [&]
            {
                if constexpr(std::is_same_v<double, T>)
                    return std::strtod(text.data(), &next);
                else if constexpr(std::is_same_v<int64_t, T>)
                    return std::strtoll(text.data(), &next, 10);
                else
                    assert(0);
            }();
            if (next==text.data()+text.size())
                return v;
            else
                return {};
        }
    }
};

#if __cpp_lib_to_chars >= 201611 || _MSC_VER >= 1915

#include <charconv>

struct NewStyleNumberParser
{
    static constexpr bool requiresNull = false;

    template<typename T>
    static std::optional<T> as(std::string_view text)
    {
        if(text.empty())
            return {};


        T out;
        if constexpr(std::is_same_v<double, T>)
        {
            auto result = std::from_chars(text.data(), text.data() + text.size(), out);
            if(result.ptr == text.data() + text.size())
                return out;
        }
        else if constexpr(std::is_same_v<int64_t, T>)
        {
            auto result = std::from_chars(text.data(), text.data() + text.size(), out, 10);
            if(result.ptr == text.data() + text.size())
                return out;
        }
        else if constexpr (std::is_same_v<Timestamp, T>)
        {
            return parseTimestamp(text);
        }
        else
            static_assert(always_false_v<T>, "not supported target type");

        return {};
    }
};
using Parser = NewStyleNumberParser;
#else
using Parser = OldStyleNumberParser;
#endif

DFH_EXPORT void uglyPrint(const arrow::Table &table, std::ostream &out = std::cout, int rows = 20);
DFH_EXPORT void uglyPrint(const std::shared_ptr<arrow::Column> &column, std::ostream &out = std::cout, int rows = 20);
