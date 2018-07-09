cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

# decide on subdirectory name within native_libs
if(WIN32)
    set(PLATFORM_DIR_SUFFIX windows)
elseif(APPLE)
    set(PLATFORM_DIR_SUFFIX macos)
else()
    set(PLATFORM_DIR_SUFFIX linux)
endif()


# set output directory
set(OUT_DIR ${CMAKE_CURRENT_LIST_DIR}/../${PLATFORM_DIR_SUFFIX})
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUT_DIR})
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUT_DIR})
endif()

project(DataframeHelper)



find_path(XLNT_INCLUDE xlnt/xlnt.hpp)
if(NOT XLNT_INCLUDE)
    message(WARNING "Cannot find xlnt include dir with xlnt/xlnt.hpp")
endif()

if(WIN32)
    find_library(XLNT_LIBRARY_DEBUG xlntd)
    if(NOT XLNT_LIBRARY_DEBUG)
        message(WARNING "Cannot find xlntd library " ${XLNT_LIBRARY_DEBUG})
    endif()
endif()

find_library(XLNT_LIBRARY xlnt)
if(NOT XLNT_LIBRARY)
    message(WARNING "Cannot find xlnt library " ${XLNT_LIBRARY})
endif()

file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(DataframeHelper ${HEADER_FILES} ${SRC_FILES})

target_include_directories(DataframeHelper PRIVATE ${XLNT_INCLUDE} ${PROJECT_SOURCE_DIR})
if(WIN32)
target_link_libraries(DataframeHelper debug ${XLNT_LIBRARY_DEBUG})
target_link_libraries(DataframeHelper optimized ${XLNT_LIBRARY})
else()
target_link_libraries(DataframeHelper ${XLNT_LIBRARY})
endif()

