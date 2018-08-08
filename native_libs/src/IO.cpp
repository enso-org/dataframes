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

        std::string contents;
        input.seekg(0, std::ios::end);
        contents.resize(input.tellg());
        input.seekg(0, std::ios::beg);
        input.read(&contents[0], contents.size());
        input.close();
        return(contents);
    }
    catch(std::exception &e)
    {
        // make sure that filename is recorded in the error message
        std::stringstream errMsg;
        errMsg << "Failed to load file `" << filepath << "`: " << e.what();
        throw std::runtime_error(errMsg.str());
    }
}