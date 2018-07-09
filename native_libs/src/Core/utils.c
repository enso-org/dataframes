#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "Common.h"

EXPORT double luna_to_double(char *data)
{
    double d;
    sscanf(data,"%lf", &d);
    return d;
}
