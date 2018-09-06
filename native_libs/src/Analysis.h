#pragma once

#include <memory>

#include "Core/Common.h"
#include "Core/ArrowUtilities.h"

DFH_EXPORT std::shared_ptr<arrow::Table> countValues(const arrow::Column &column);

DFH_EXPORT std::shared_ptr<arrow::Column> calculateMin(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMax(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMean(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateMedian(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateVariance(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateStandardDeviation(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateSum(const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateQuantile(const arrow::Column &column, double q);
DFH_EXPORT double calculateCorrelation(const arrow::Column &xCol, const arrow::Column &yCol);
DFH_EXPORT std::shared_ptr<arrow::Column> calculateCorrelation(const arrow::Table &table, const arrow::Column &column);
DFH_EXPORT std::shared_ptr<arrow::Table> calculateCorrelationMatrix(const arrow::Table &table);

DFH_EXPORT double autoCorrelation(const std::shared_ptr<arrow::Column> &column, int lag = 1);

DFH_EXPORT std::shared_ptr<arrow::Table> abominableGroupAggregate(std::shared_ptr<arrow::Table> table, std::shared_ptr<arrow::Column> keyColumn, std::vector<std::shared_ptr<arrow::Column>> toAggregate);