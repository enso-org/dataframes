#pragma once

#include <cassert>
#include <cstdlib>
#include <optional>
#include <optional>
#include <string_view>
#include <type_traits>

template<typename T>
std::optional<T> parseAs(std::string_view text)
{
    if(text.empty())
        return {};

    char *next = nullptr;
    auto v = [&] 
    {
        if constexpr(std::is_same_v<double, T>)
            return std::strtod(text.data(), &next);
        else if constexpr(std::is_same_v<int64_t, T>)
            return std::strtoll(text.data(), &next, 10);
        else
            assert(0);
    }();
    if(next == text.data() + text.size())
        return v;
    else
        return {};
}
