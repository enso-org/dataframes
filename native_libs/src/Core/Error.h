#pragma once
#include <string>

extern thread_local std::string errorMessage;

template<typename Function>
auto translateExceptionToError(const char **outError, Function &&f)
{
	try
	{
		*outError = nullptr;
		return f();
	}
	catch(std::exception &e)
	{
		errorMessage = e.what();
		*outError = errorMessage.c_str();
	}
	catch(...)
	{
		errorMessage = "unknown exception";
		*outError = errorMessage.c_str();
	}

	using ResultType = decltype(f());
	if constexpr(!std::is_same_v<void, ResultType>)
	{
		return ResultType{};
	}
}