#include <sstream>
#include <stdexcept>

#include "Common.h"


void validateIndex(const size_t size, int64_t index)
{
    if(index < 0 || index >= (int64_t)size)
    {
        std::ostringstream out;
        out << "wrong index " << index << " when array length is " << size;
        throw std::out_of_range{ out.str() };
    }
}
