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