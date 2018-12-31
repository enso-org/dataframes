#include <sstream>
#include <stdexcept>

#include "Common.h"

bool isValidIndex(int64_t size, int64_t index)
{
    return index >= 0 && index < size;
}

void validateIndex(int64_t size, int64_t index)
{
    if(!isValidIndex(size, index))
        THROW("wrong index={} when array's length={}", index, size);
}

void validateSlice(int64_t dataLength, int64_t sliceStart, int64_t sliceLength)
{
    if(sliceLength < 0)
        THROW("slice length must be greater than zero, requested length is {}", sliceLength);

    // slices of size 0 don't require valid indices, because they are empty anyway
    if(sliceLength > 0)
    {
        if(!isValidIndex(dataLength, sliceStart))
            THROW("requested slice starting at invalid index={}, data length is {}", sliceStart, dataLength);

        const auto lastIndex = sliceStart + sliceLength - 1;
        if (lastIndex >= dataLength)
            THROW("Slice would end at invalid index={}, length is {}", lastIndex, dataLength);
    }
}