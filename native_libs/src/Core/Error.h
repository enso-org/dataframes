#pragma once
#include <string>

void setError(const char **outError, const char *errorToSet) noexcept;
void clearError(const char **outError) noexcept;

template<typename Function>
auto translateExceptionToError(const char **outError, Function &&f)
{
	try
	{
		clearError(outError);
		return f();
	}
	catch(std::exception &e)
	{
		setError(outError, e.what());
	}
	catch(...)
	{
		setError(outError, "unknown exception");
	}

	using ResultType = decltype(f());
	if constexpr(!std::is_same_v<void, ResultType>)
		return ResultType{};
	else
		return;
}