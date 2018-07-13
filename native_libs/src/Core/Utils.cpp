#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>

#include "Common.h"

extern "C"
{

EXPORT double luna_to_double(char *data, const char **error)
{
    double d;
    std::sscanf(data,"%lf", &d);
    return d;
}

}
