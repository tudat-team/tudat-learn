cmake_minimum_required(VERSION 3.10)

project(tudat_learn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

include(CTest)
include(GNUInstallDirs)

# TUDAT_LEARN_EXTERNAL_LIBS contains the external libraries
set(TUDAT_EXTERNAL_LIBS "")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# set(TUDAT_EXTERNAL_LIBS "${TUDAT_EXTERNAL_LIBS}" Eigen3::Eigen) doesn't work


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
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/regressors/rbf.cpp CHANGED TO .tpp FOR TEMPLATE DEFINITION
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/regressors/rbfn.cpp CHANGED TO .tpp FOR TEMPLATE DEFINITION

)


add_library(tudat_learn SHARED
        ${TUDAT_LEARN_SRC_FILES}
)


target_include_directories(tudat_learn PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIRS}
)


add_subdirectory(app)
add_subdirectory(tests)
