#pragma once

#include <memory>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"

EXPORT std::shared_ptr<arrow::Table> countValues(const arrow::Column &column);

EXPORT std::shared_ptr<arrow::Column> calculateMin(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateMax(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateMean(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateMedian(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateVariance(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column);
EXPORT std::shared_ptr<arrow::Column> calculateQuantile(const arrow::Column &column, double q);
EXPORT double calculateCorrelation(const arrow::Column &xCol, const arrow::Column &yCol);
EXPORT std::shared_ptr<arrow::Table> calculateCorrelationMatrix(const arrow::Table &table);
