#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Common.h"

using namespace std::chrono_literals;

// returns pair [f(args...), elapsed time]
template<typename F, typename ...Args>
static auto callDuration(F&& func, Args&&... args)
{
    const auto start = std::chrono::steady_clock::now();
    if constexpr(std::is_same_v<void, std::invoke_result_t<F, Args...>>)
    {
        std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        return std::make_pair(nullptr, std::chrono::steady_clock::now() - start);
    }
    else
    {
        auto ret = std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
        return std::make_pair(std::move(ret), std::chrono::steady_clock::now() - start);
    }
}

struct MeasureOnce
{
    bool measuredEnough(int count, std::chrono::milliseconds timeSpent)
    {
        return true;
    }
};

struct MeasureAtLeast
{
    int64_t requiredCount{};
    std::chrono::milliseconds requiredTime{};

    MeasureAtLeast(int64_t requiredCount, std::chrono::milliseconds requiredTime) 
        : requiredCount(requiredCount), requiredTime(requiredTime)
    {}

    MeasureAtLeast(int64_t requiredCount) 
        : requiredCount(requiredCount)
    {
    }

    template<typename Duration>
    bool measuredEnough(int count, Duration timeSpent)
    {
        return requiredCount <= count && timeSpent >= requiredTime;
    }
};

template<typename F, typename ...Args>
static auto measure(std::string text, F&& func, Args&&... args)
{
    const auto results = callDuration(std::forward<F>(func), std::forward<Args>(args)...);
    const auto t = std::chrono::duration_cast<std::chrono::microseconds>(results.second).count();
    std::cout << text << " took " << t / 1000.0 << " ms" << std::endl;
    return results;
}

struct MeasureSeries
{
    using Duration = std::chrono::microseconds;
    std::string name;
    std::vector<Duration> times;

    explicit MeasureSeries(std::string name)
        : name(std::move(name)) 
    {}

    Duration bestTime()
    {
        if(times.empty())
            throw std::runtime_error("no measures");

        return *std::min_element(times.begin(), times.end());
    }

    void add(Duration d)
    {
        times.push_back(d);
    }
};


template<typename Policy, typename F, typename ...Args>
static auto measure(std::string text, Policy &&p, F&& func, Args&&... args)
{
    using namespace std::chrono;
    using namespace std::literals;

    MeasureSeries measures{text};

    microseconds bestTime = 9999h;
    int N = 0;

    const auto startTime = system_clock::now();
    while(true)
    {
        auto [value, tn] = callDuration(func, args...);
        const auto t = duration_cast<microseconds>(tn);
        
        measures.add(t);

        bestTime = std::min<microseconds>(bestTime, t);
        std::cout << text << " took " << t.count() / 1000.0 << " ms, best: " << bestTime.count() / 1000. << " ms" << std::endl;
        if constexpr(std::is_arithmetic_v<Policy>)
        {
            if(N >= p)
                return std::make_pair(std::move(value), measures);
        }
        else if(p.measuredEnough(++N, system_clock::now() - startTime))
            return std::make_pair(std::move(value), measures);

    }
}
