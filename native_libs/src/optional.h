#if __has_include(<optional>) && !defined(__apple__)

#include <optional>

#else  // Mac OS

#include <nonstd/optional.hpp>
namespace std
{
    using namespace nonstd;
}

#endif