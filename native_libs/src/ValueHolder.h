#pragma once

#include <string>

#include "Core/Common.h"

struct DFH_EXPORT ValueHolder
{
    std::string buffer;

    const char *store(std::string s);
    void clear();
};
