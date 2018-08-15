
#if __has_include(<variant>)

#include <variant>

#else  // Mac OS

#include <mpark/variant.hpp>
namespace std
{
    using namespace mpark;
}

#endif