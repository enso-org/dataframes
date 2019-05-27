/******************************************************************************
NOTE: non-portable, unix-specific program (should work on macOS and Linux)
Simple C++ program that uses dlopen to load all the libraries given to it as
arguments. Returns 0 if all libraries were loaded, 1 otherwise.
It was created as a helper for a dependency discovery - by observing what files
get loaded when this program loads library, dependency list can be collected.
******************************************************************************/

#include <dlfcn.h>
#include <cstdio>

int main(int argc, char **argv)
{
    bool wasError = false;
    for(int i = 1; i < argc; i++)
    {
        if(!dlopen(argv[i], RTLD_NOW))
        {
            std::printf("failed to load library %s: %s", argv[i], dlerror());
            wasError = true;
        }
    }

    return wasError;
}