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
    message(WARNING "Cannot find xlnt include dir with xlnt/xlnt.hpp. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()

if(WIN32)
    find_library(XLNT_LIBRARY_DEBUG xlntd)
    if(NOT XLNT_LIBRARY_DEBUG)
        message(WARNING "Cannot find xlntd (debug build) library. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH.")
    endif()
endif()

find_library(XLNT_LIBRARY xlnt)
if(NOT XLNT_LIBRARY)
    message(WARNING "Cannot find xlnt library. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH.")
endif()

# Glob for the project files
file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

# Define the compilation target
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SRC_FILES})

# Add root src directory to the includes path
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

if(XLNT_LIBRARY AND XLNT_INCLUDE)
    target_include_directories(${PROJECT_NAME} PRIVATE ${XLNT_INCLUDE})
    if(WIN32)
        # Note [MU] Windows builds have different names for debug and release binaries (xlntd and xlnt respectively)
        # (as they are ABI-incompatible)
        target_link_libraries(${PROJECT_NAME} debug ${XLNT_LIBRARY_DEBUG})
        target_link_libraries(${PROJECT_NAME} optimized ${XLNT_LIBRARY})
    else()
        target_link_libraries(${PROJECT_NAME} ${XLNT_LIBRARY})
    endif()
else()
    message(WARNING ${PROJECT_NAME} " will be built without XLSX format support, as xlnt library was not found!")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DISABLE_XLSX)
endif()
