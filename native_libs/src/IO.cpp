#include "IO.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::string getFileContents(const char *filepath)
{
    try
    {
        std::ifstream input{filepath};
        if(!input)
            throw std::runtime_error("Failed to open the file");

        std::stringstream buffer;
        buffer << input.rdbuf();
        return std::move(buffer.str());
    }
    catch(std::exception &e)
    {
        // make sure that filename is recorded in the error message
        std::stringstream errMsg;
        errMsg << "Failed to load file `" << filepath << "`: " << e.what();
        throw std::runtime_error(errMsg.str());
    }
}