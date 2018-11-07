
#ifndef __APPLE__

#include <variant>
using std::variant;
using std::visit;
using std::get;
using std::get_if;
using std::holds_alternative;

#else  // Mac OS

#include <mpark/variant.hpp>
using mpark::variant;
using mpark::visit;
using mpark::get;
using mpark::get_if;
using mpark::holds_alternative;

#endif