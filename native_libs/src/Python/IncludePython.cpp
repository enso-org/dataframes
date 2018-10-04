#include "IncludePython.h"


pybind11::function getMethod(pybind11::object module, const std::string &attributeName)
{
    auto elem = module.attr(attributeName.c_str());
    auto ret = elem.cast<pybind11::function>();
    if(!ret.check())
        THROW("`{}` exists but is not a function: {}", attributeName, (std::string)elem.get_type().str());

    return ret;
}


