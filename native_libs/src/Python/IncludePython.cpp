#include "IncludePython.h"


pybind11::function getMethod(pybind11::object module, const std::string &attributeName)
{
    auto elem = module.attr(attributeName.c_str());
    auto ret = elem.cast<pybind11::function>();
    if(!ret.check())
        THROW("`{}` exists but is not a function: {}", attributeName, (std::string)elem.get_type().str());

    return ret;
}

void pybind11::insert(dict dict, const char *key, object value)
{
    if(PyDict_SetItemString(dict.ptr(), key, value.ptr()))
        THROW("failed to insert to map: key=`{}`, value=`{}`", key, (std::string)value.str());
}

void pybind11::insert(dict dict, const char *key, std::string_view value)
{
    insert(dict, key, pybind11::str(std::string(value)));
}

void pybind11::setAt(list list, size_t index, object value)
{
    // NOTE: PyList_SetItem steals reference
    if(PyList_SetItem(list.ptr(), index, value.release().ptr()))
        THROW("failed to list item: index=`{}`, value=`{}`", index, (std::string)value.str());
}
