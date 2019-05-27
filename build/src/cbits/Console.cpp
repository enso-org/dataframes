#include <Windows.h>
#include <cstdint>
#include <algorithm>
#include <string>
#include <stdexcept>

extern "C"
{
    bool isConsole(HANDLE handle) noexcept
    {
        // if we can get console mode, then it is console
        DWORD temp;
        return GetConsoleMode(handle, &temp);
    }
    int64_t writeConsole(const char16_t* text, int32_t length, HANDLE handle) noexcept
    {
        DWORD writtenCount = 0;
        if(!WriteConsoleW(handle, text, length, &writtenCount, nullptr))
            return -1;
        return writtenCount;
    }
}
