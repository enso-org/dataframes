#pragma once
#include <string>
#include "Logger.h"

// Abominable workaround - standard library on Mac does not have invoke_result
#if !defined(_MSC_VER) && !defined(__cpp_lib_is_invocable)
namespace std
{
    template<typename F, typename ...Args>
    using invoke_result_t = std::result_of_t<F(Args...)>;
}
#endif


void setError(const char **outError, const char *errorToSet, const char *functionName) noexcept;
void clearError(const char **outError) noexcept;

template<typename Function>
auto translateExceptionToError(const char *functionName, const char **outError, Function &&f)
{
    using ResultType = std::invoke_result_t<Function>;
    constexpr auto returnVoid = std::is_same_v<void, ResultType>;

    try
    {
        clearError(outError);

        if constexpr(!returnVoid)
        {
#ifdef VERBOSE
            auto ret = f();
            if constexpr(std::is_pointer_v<decltype(ret)>)
                LOG("returning: {}", (void*)ret);
            else
                LOG("returning: {}", ret);
			//if(outError)
			//	std::cout << "Out error " << (void*)outError << " is set to " << (void*)*outError << std::endl;
            return ret;
#else
            return f();
#endif
        }
        else
            f();
    }
    catch(std::exception &e)
    {
        setError(outError, e.what(), functionName);
    }
    catch(...)
    {
        setError(outError, "unknown exception", functionName);
    }

    if constexpr(!returnVoid)
        return ResultType{};
    else
        return;
}

struct ExceptionHelper 
{
    const char *functionName;
    const char **outError;
    
    explicit ExceptionHelper(const char *functionName, const char **outError) 
        : functionName(functionName), outError(outError)
    {}

    template<typename Function>
    auto operator<<(Function &&f) const noexcept
    {
        return translateExceptionToError(functionName, outError, std::forward<Function>(f));
    }
};

// Helper macro for translating between C++ exceptions and C API output error arguments.
// Takes an OutError argument expected to be of type const char **.
// Then takes a lambda body to be executed. All exceptions thrown from it will be catched
// and translated to error messages that will be written to OutError.
// If no exception is thrown, OutError shall be written with nullptr.
// Macro yields a call that returns the value that nested body returns.
// Body requires a semicolon after end. 
#define TRANSLATE_EXCEPTION(OutError) ExceptionHelper(__FUNCTION__, OutError) << [&] () mutable
