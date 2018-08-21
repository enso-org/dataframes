#pragma once

#include <memory>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"

EXPORT std::shared_ptr<arrow::Table> countValues(const arrow::Column &column);