#pragma once

#include <memory>
#include <string>

#include "Core/Common.h"
#include "IO.h"

namespace arrow
{
    class Table;
}

struct DFH_EXPORT FormatFeather : TableFileHandler
{
    virtual std::string fileSignature() const override;
    virtual std::shared_ptr<arrow::Table> read(std::string_view filePath) const override;
    virtual void write(std::string_view filePath, const arrow::Table &table) const override;
    virtual std::vector<std::string> fileExtensions() const override;
};
