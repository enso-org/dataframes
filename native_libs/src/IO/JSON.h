#pragma once 
#include <rapidjson/document.h>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"


DFH_EXPORT rapidjson::Document parseJSON(const char *json);
DFH_EXPORT std::string toJsonString(const rapidjson::Value &v);

DFH_EXPORT DynamicField parseAsField(const char *jsonText);
DFH_EXPORT DynamicField parseAsField(const rapidjson::Value &doc);
