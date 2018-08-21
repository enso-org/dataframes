#include "JSON.h"

#include <rapidjson/error/en.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

using namespace std::literals;

rapidjson::Document parseJSON(const char *json)
{
    rapidjson::Document doc{};
    doc.Parse(json);

    if(doc.HasParseError())
        throw std::runtime_error("Failed to parse JSON: "s + GetParseError_En(doc.GetParseError()));

    return doc;
}

std::string toJsonString(const rapidjson::Value &v)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    v.Accept(writer);
    return buffer.GetString();
}

DynamicField parseAsField(const char *jsonText)
{
    auto doc = parseJSON(jsonText);
    return parseAsField(doc);
}

DynamicField parseAsField(const rapidjson::Value &doc)
{
    if(doc.IsInt64())
        return doc.GetInt64();
    if(doc.IsDouble())
        return double(doc.GetDouble());
    if(doc.IsString())
        return std::string(doc.GetString());

    throw std::runtime_error("Failed to use JSON as a single value: "s + toJsonString(doc));
}
