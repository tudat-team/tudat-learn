cmake_minimum_required(VERSION 3.10)

project(tudat_learn VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(TUDAT_LEARN_SRC_FILES
        # Core classes
        "${CMAKE_CURRENT_SOURCE_DIR}/src/estimator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dataset.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/processing.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sampling.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark

        )

add_library(tudat_learn SHARED )
add_subdirectory(app)