#pragma once


#include <cmath>
#include <regex>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include <date/date.h>
#include "Core/ArrowUtilities.h"

using namespace std::literals;

#define COMPLAIN_ABOUT_OPERAND_TYPES \
        throw std::runtime_error(__FUNCTION__ + ": not supported operand types: "s + typeid(lhs).name() + " and "s + typeid(rhs).name());

#define BINARY_REL_OPERATOR(op)                                                                          \
    template<typename Lhs, typename Rhs>                                                                 \
    static bool exec(const Lhs &lhs, const Rhs &rhs)                                                     \
    { /* below we protect against mixed types like int/string (eg. for ==)  */                           \
        if constexpr(std::is_same_v<Lhs, Rhs> || std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>) \
            return lhs op rhs;                                                                           \
        else                                                                                             \
        {                                                                                                \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
            return {}; /* just for type inference  */                                                    \
        }                                                                                                \
    }

#define BINARY_ARIT_OPERATOR(op)                                                                         \
    template<typename Lhs, typename Rhs>                                                                 \
    static auto exec(const Lhs &lhs, const Rhs &rhs)                                                     \
    {                                                                                                    \
        if constexpr(std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>)                             \
            return lhs op rhs;                                                                           \
        else                                                                                             \
        {                                                                                                \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
            return lhs; /* just for type inference  */                                                   \
        }                                                                                                \
    }
#define FAIL_ON_STRING(ret)                                                                              \
    static ret exec(const std::string &lhs, const std::string &rhs)                                      \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    static ret exec(const std::string_view &lhs, const std::string_view &rhs)                            \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    template<typename Rhs>                                                                               \
    static ret exec(const std::string_view &lhs, const Rhs &rhs)                                         \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    }                                                                                                    \
    template<typename Lhs>                                                                               \
    static ret exec(const Lhs &lhs, const std::string_view &rhs)                                         \
    {                                                                                                    \
            COMPLAIN_ABOUT_OPERAND_TYPES;                                                                \
    } 

struct GreaterThan { BINARY_REL_OPERATOR(> ); FAIL_ON_STRING(bool); };
struct LessThan    { BINARY_REL_OPERATOR(< ); FAIL_ON_STRING(bool); };
struct EqualTo     { BINARY_REL_OPERATOR(== ); };
struct StartsWith
{
    static bool exec(const std::string_view &lhs, const std::string_view &rhs)
    {
        return lhs.length() >= rhs.length()
            && std::memcmp(lhs.data(), rhs.data(), rhs.length()) == 0;
    }

    template<typename Lhs, typename Rhs>
    static bool exec(const Lhs &lhs, const Rhs &rhs)
    {
        COMPLAIN_ABOUT_OPERAND_TYPES;
    }
};
struct Matches
{
    static bool exec(const std::string_view &lhs, const std::string_view &rhs)
    {
        std::regex regex{ std::string(rhs) };
        return std::regex_match(lhs.begin(), lhs.end(), regex);
    }

    template<typename Lhs, typename Rhs>
    static bool exec(const Lhs &lhs, const Rhs &rhs)
    {
        COMPLAIN_ABOUT_OPERAND_TYPES;
    }
};


struct Plus
{
    BINARY_ARIT_OPERATOR(+);
    FAIL_ON_STRING(int64_t); 
};

struct Minus       { BINARY_ARIT_OPERATOR(-); FAIL_ON_STRING(int64_t); };
struct Times       { BINARY_ARIT_OPERATOR(*); FAIL_ON_STRING(int64_t); };
struct Divide      { BINARY_ARIT_OPERATOR(/ ); FAIL_ON_STRING(int64_t); };
struct Modulo
{
    static constexpr int64_t exec(const int64_t &lhs, const int64_t &rhs)
    {
        return lhs % rhs;
    }
    static double exec(const double &lhs, const double &rhs)
    {
        return std::fmod(lhs, rhs);
    }
    template<typename Lhs, typename Rhs>
    static int64_t exec(const Lhs &lhs, const Rhs &rhs)
    {
        COMPLAIN_ABOUT_OPERAND_TYPES;
    }
};
struct Negate
{
    template<typename Lhs>
    static constexpr Lhs exec(const Lhs &lhs)
    {
        if constexpr(std::is_arithmetic_v<Lhs>)
            return -lhs;
        else
            throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
    }

    static int64_t exec(const std::string_view &lhs)
    {
        throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
    }
};

struct Abs
{
    template<typename Lhs>
    static constexpr Lhs exec(const Lhs &lhs)
    {
        if constexpr(std::is_arithmetic_v<Lhs>)
            if (lhs < 0)
                return -lhs;
            else return lhs;
        else
            throw std::runtime_error("abs does not support operand of type: "s + typeid(lhs).name());
    }

    static int64_t exec(const std::string_view &lhs)
    {
        throw std::runtime_error("abs does not support operand of type: "s + typeid(lhs).name());
    }
};

struct Day
{
    template<typename Lhs>
    static int64_t exec(const Lhs &lhs)
    {
        throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
    }
    static constexpr int64_t exec(const Timestamp &lhs)
    {
        return (unsigned) lhs.ymd().day();
    }
};
struct Month
{
    template<typename Lhs>
    static int64_t exec(const Lhs &lhs)
    {
        throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
    }
    static constexpr int64_t exec(const Timestamp &lhs)
    {
        return (unsigned) lhs.ymd().month();
    }
};
struct Year
{
    template<typename Lhs>
    static int64_t exec(const Lhs &lhs)
    {
        throw std::runtime_error("negate does not support operand of type: "s + typeid(lhs).name());
    }
    static constexpr int64_t exec(const Timestamp &lhs)
    {
        return (int) lhs.ymd().year();
    }
};

struct Condition
{
    template<typename A, typename B>
    using Ret = std::conditional_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>,
        std::common_type_t<A, B>,
        A>;

    template<typename Lhs, typename Rhs>
    static auto exec(const bool &mask, const Lhs &lhs, const Rhs &rhs)
    {
        if constexpr(std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>)
            return mask ? lhs : rhs;
        else if constexpr(std::is_same_v<Lhs, Rhs> && std::is_same_v<Lhs, std::string_view>)
            return std::string(mask ? lhs : rhs);
        else
        {
            COMPLAIN_ABOUT_OPERAND_TYPES;
            return int64_t{}; // to deduct type
        }
    }
};

struct And
{
    static bool exec(const bool &lhs, const bool &rhs)
    {
        return lhs && rhs;
    }

    template<typename Lhs, typename Rhs>
    static bool exec(const Lhs &lhs, const Rhs &rhs)
    {
        COMPLAIN_ABOUT_OPERAND_TYPES;
    }
};
struct Or
{
    static bool exec(const bool &lhs, const bool &rhs)
    {
        return lhs || rhs;
    }

    template<typename Lhs, typename Rhs>
    static bool exec(const Lhs &lhs, const Rhs &rhs)
    {
        COMPLAIN_ABOUT_OPERAND_TYPES;
    }
};
struct Not
{
    static bool exec(const bool &lhs)
    {
        return !lhs;
    }

    template<typename Lhs>
    static bool exec(const Lhs &lhs)
    {
        throw std::runtime_error("Not: wrong operand type "s + typeid(Lhs).name());
    }
};
