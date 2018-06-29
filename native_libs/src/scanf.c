#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

double luna_to_double(char *data)
{
    double d;
    sscanf(data,"%lf", &d);
    return d;
}
