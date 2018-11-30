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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
if(NOT WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIR})
endif()

project(DataframeHelper)

find_package(Boost 1.62.0 REQUIRED COMPONENTS filesystem unit_test_framework)
if(NOT Boost_FOUND)
    message(WARNING "Cannot find Boost libraries")
endif()

find_path(RAPIDJSON_INCLUDE rapidjson/document.h)
if(NOT RAPIDJSON_INCLUDE)
    message(WARNING "Cannot find rapidjson include dir with rapidjson/document.h. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()

find_path(FMT_INCLUDE fmt/format.h)
if(NOT FMT_INCLUDE)
    message(WARNING "Cannot find {fmt} include dir with fmt/format.h. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()
if(NOT WIN32)
    find_library(FMT_LIBRARY fmt)
    if(NOT FMT_LIBRARY)
        message(WARNING "Cannot find fmt library. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH.")
    endif()
endif()

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

find_path(ARROW_INCLUDE arrow/array.h)
if(NOT ARROW_INCLUDE)
    message(WARNING "Cannot find Apache Arrow include dir with arrow/array.h. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()
if(WIN32)
    find_library(ARROW_LIBRARY_DEBUG arrowd)
    if(NOT ARROW_LIBRARY_DEBUG)
        message(WARNING "Cannot find arrowd (debug build) library. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH.")
    endif()
endif()
find_library(ARROW_LIBRARY arrow)
if(NOT ARROW_LIBRARY)
    message(WARNING "Cannot find arrow library. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_LIBRARY_PATH.")
endif()


find_path(DATE_INCLUDE date/date.h)
if(NOT DATE_INCLUDE)
    message(WARNING "Cannot find date library include dir with date/date.h. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()

find_path(PYBIND_INCLUDE pybind11/embed.h)
if(NOT PYBIND_INCLUDE)
    message(WARNING "Cannot find pybind11 library include dir with pybind11/embed.h. If it is present, consider setting CMAKE_PREFIX_PATH or CMAKE_INCLUDE_PATH.")
endif()

find_package(PythonLibs 3.6)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR};${CMAKE_MODULE_PATH}")
find_package(NumPy)

# Glob for the project files
file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

# Define the compilation target
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SRC_FILES})
target_compile_definitions(${PROJECT_NAME} PRIVATE BUILDING_DATAFRAME_HELPER)
target_compile_definitions(${PROJECT_NAME} PUBLIC _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS) # rapidjson is so deprecated)

# Boost libraries dependency
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Boost::filesystem)

# Includes path: project root, arrow, third-party any-lite
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${ARROW_INCLUDE} ${PROJECT_SOURCE_DIR}/../third-party/any-lite ${PROJECT_SOURCE_DIR}/../third-party/optional-lite ${PROJECT_SOURCE_DIR}/../third-party/variant ${RAPIDJSON_INCLUDE} ${DATE_INCLUDE} ${FMT_INCLUDE} ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR} ${PYBIND_INCLUDE})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FMT_HEADER_ONLY)
else()
    target_link_libraries(${PROJECT_NAME} ${FMT_LIBRARY})
endif()

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

if(WIN32)
    # Note [MU] Windows builds have different names for debug and release binaries (arrowd and arrow respectively)
    # (as they are ABI-incompatible)
    target_link_libraries(${PROJECT_NAME} debug ${ARROW_LIBRARY_DEBUG})
    target_link_libraries(${PROJECT_NAME} optimized ${ARROW_LIBRARY})
else()
    target_link_libraries(${PROJECT_NAME} ${ARROW_LIBRARY})
endif()

###################

file(GLOB_RECURSE PLOTTER_HEADER_FILES ${PROJECT_SOURCE_DIR}/../plotter/*.h)
file(GLOB_RECURSE PLOTTER_SRC_FILES    ${PROJECT_SOURCE_DIR}/../plotter/*.cpp)

add_library(DataframePlotter SHARED ${PLOTTER_HEADER_FILES} ${PLOTTER_SRC_FILES})

target_include_directories(DataframePlotter PRIVATE ${PROJECT_SOURCE_DIR}/../third-party/matplotlib-cpp)
target_link_libraries(DataframePlotter ${PROJECT_NAME})

###################

file(GLOB_RECURSE LEARN_HEADER_FILES ${PROJECT_SOURCE_DIR}/../learn/*.h)
file(GLOB_RECURSE LEARN_SRC_FILES    ${PROJECT_SOURCE_DIR}/../learn/*.cpp)

add_library(Learn SHARED ${LEARN_HEADER_FILES} ${LEARN_SRC_FILES})

# target_include_directories(Learn PUBLIC )
target_link_libraries(Learn ${PROJECT_NAME})

##################
# TESTS & BENCHMARKS
file(GLOB_RECURSE TEST_HEADER_FILES ${PROJECT_SOURCE_DIR}/../test/*.h)
file(GLOB_RECURSE TEST_SRC_FILES    ${PROJECT_SOURCE_DIR}/../test/*.cpp)

add_executable(DataframeHelperTests ${TEST_HEADER_FILES} ${TEST_SRC_FILES})
target_include_directories(DataframeHelperTests PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(DataframeHelperTests Learn ${PROJECT_NAME} DataframePlotter Boost::unit_test_framework)
###################