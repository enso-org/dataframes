#include "Error.h"

#include <utility>

namespace
{
	const auto unknownInternalErrorText = "Unknown internal error encountered";
	thread_local std::string errorMessage;
}

void setError(const char **outError, const char *errorToSet, const char *functionName) noexcept
{
	try
	{
		if(functionName)
			errorMessage = std::string(functionName) + ": " + errorToSet;
		else
			errorMessage = errorToSet;

		*outError = errorMessage.c_str();
	}
	catch(...)
	{
		// should happen practically never, perhaps if error string is too long to fit in memory....
		// but to be on the safe side
		*outError = unknownInternalErrorText;
	}
}

void clearError(const char **outError) noexcept
{
	*outError = nullptr;
}
