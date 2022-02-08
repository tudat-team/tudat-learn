# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(tudat-learn VERSION 0.1.0 LANGUAGES CXX)

# Set environment variables
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# # Library examples
# add_subdirectory(example_internal)