cmake_minimum_required(VERSION 3.10)

project(tudat_learn VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

include(CTest)
include(GNUInstallDirs)

set(TUDAT_LEARN_SRC_FILES
        ### Core Classes ###
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/estimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/processing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sampling.cpp

        ## Estimators ##
        ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/classifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/clustering.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/regressor.cpp

        # Regressors #
        ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/regressors/rbfn.cpp

)


add_library(tudat_learn SHARED
        ${TUDAT_LEARN_SRC_FILES}
)


target_include_directories(tudat_learn PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        # ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark might not be needed
)


add_subdirectory(app)
add_subdirectory(tests)
