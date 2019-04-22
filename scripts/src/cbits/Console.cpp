#include <Windows.h>
#include <cstdint>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace
{
std::string GetLastErrorAsString()
{
	//Get the error message, if any.
	DWORD errorMessageID = ::GetLastError();
	if(errorMessageID == 0)
		return std::string(); //No error message has been recorded

	LPSTR messageBuffer = nullptr;
	size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)& messageBuffer, 0, nullptr);

	std::string message(messageBuffer, size);
	LocalFree(messageBuffer);
	return message;
}

std::string utf16ToUtf8(const wchar_t* text, int32_t textLength)
{
	std::string retStr;
	if(textLength > 0)
	{
		const int sizeRequired = WideCharToMultiByte(CP_UTF8, 0, text, textLength, nullptr, 0, nullptr, nullptr);
		if(sizeRequired > 0)
		{
			retStr.resize(sizeRequired);
			const int bytesConverted = WideCharToMultiByte(CP_UTF8, 0, text, textLength, retStr.data(), retStr.size(), nullptr, nullptr);
			if(bytesConverted == 0)
				throw std::runtime_error("Conversion failed: " + GetLastErrorAsString());
		}
	}
	return retStr;
}

std::string utf16ToUtf8(const std::wstring &text)
{
	return utf16ToUtf8(text.data(), text.length());
}

bool isConsole(HANDLE handle) noexcept
{
	// if we can get console mode, then it is console
	DWORD temp;
	return GetConsoleMode(handle, &temp);
}
}

extern "C"
{
    int64_t writeText(const char16_t* text, int32_t length, HANDLE handle) noexcept
    {
        auto utf8Buffer = utf16ToUtf8(std::wstring(reinterpret_cast<const wchar_t*>(text), length));
        DWORD writtenCount = 0;
        if(isConsole(handle))
        {
            WriteConsoleW(handle, text, length, &writtenCount, nullptr);
        }
        else
        {
            auto utf8Buffer = utf16ToUtf8(std::wstring(reinterpret_cast<const wchar_t*>(text), length));
            if(!WriteFile(handle, utf8Buffer.data(), utf8Buffer.size(), &writtenCount, nullptr))
            {
                //error..
                return -1;
            }
        }
        return writtenCount;
    }
}
