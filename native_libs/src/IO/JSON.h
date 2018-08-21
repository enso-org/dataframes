#pragma once 
#include <rapidjson/document.h>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"


EXPORT rapidjson::Document parseJSON(const char *json);
EXPORT std::string toJsonString(const rapidjson::Value &v);

EXPORT DynamicField parseAsField(const char *jsonText);
EXPORT DynamicField parseAsField(const rapidjson::Value &doc);
