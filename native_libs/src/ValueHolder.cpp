#include "ValueHolder.h"
#include <utility>

const char * ValueHolder::store(std::string s)
{
    buffer = std::move(s);
    return buffer.c_str();
}

void ValueHolder::clear()
{
    buffer.clear();
    buffer.shrink_to_fit();
}
