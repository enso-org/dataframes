#include "Utils.h"

std::optional<Timestamp> parseTimestamp(std::string_view text)
{
    std::istringstream input((std::string)text);

    Timestamp out;
    input >> date::parse("%F", out);
    if(input && input.rdbuf()->in_avail() == 0)
        return out;
    return std::nullopt;
}
