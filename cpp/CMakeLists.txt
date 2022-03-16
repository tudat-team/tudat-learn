# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(tudat-learn VERSION 0.1.0 LANGUAGES CXX)

# Set environment variables
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

enable_testing()

include(CTest)
include(GNUInstallDirs)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Set SOURCES variable
set(SOURCES
        ### Core Classes ###
        src/estimator.cpp
        src/function.cpp
        src/processing.cpp
        src/random.cpp
        src/sampler.cpp

        ## Estimators ##
        src/estimators/classifier.cpp
        src/estimators/clustering.cpp
        src/estimators/regressor.cpp
)

# Set HEADERS_PUBLIC variable (public headers, included in the library)
set(HEADERS_PUBLIC
        # .hpp
        include/tudat-learn/dataset.hpp
        include/tudat-learn/estimator.hpp
        include/tudat-learn/function.hpp
        include/tudat-learn/io.hpp
        include/tudat-learn/operator.hpp
        include/tudat-learn/processing.hpp
        include/tudat-learn/random.hpp
        include/tudat-learn/sampler.hpp
        include/tudat-learn/types.hpp

        include/tudat-learn/estimators/classifier.hpp
        include/tudat-learn/estimators/clustering.hpp
        include/tudat-learn/estimators/regressor.hpp

        include/tudat-learn/estimators/regressors/grnn.hpp
        include/tudat-learn/estimators/regressors/rbf.hpp
        include/tudat-learn/estimators/regressors/rbfn.hpp

        include/tudat-learn/model_evaluation/split.hpp
        include/tudat-learn/model_evaluation/cross_validation.hpp
        include/tudat-learn/model_evaluation/splits/k_fold_split.hpp

        include/tudat-learn/processing/scaler.hpp
        include/tudat-learn/processing/scalers/minmax_scaler.hpp
        include/tudat-learn/processing/scalers/standard_scaler.hpp

        include/tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp

        # .tpp
        include/tudat-learn/dataset.tpp

        include/tudat-learn/estimators/regressors/grnn.tpp
        include/tudat-learn/estimators/regressors/rbf.tpp
        include/tudat-learn/estimators/regressors/rbfn.tpp

        include/tudat-learn/model_evaluation/cross_validation.tpp
        include/tudat-learn/model_evaluation/splits/k_fold_split.tpp

        include/tudat-learn/processing/scalers/minmax_scaler.tpp
        include/tudat-learn/processing/scalers/standard_scaler.tpp

        include/tudat-learn/samplers/random_samplers/latin_hypercube_sampler.tpp
)

# Set HEADERS_PRIVATE variable, if needed.
# (these private headers won't be included in the library)
set(HEADERS_PRIVATE
        
)

include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)

include(${PROJECT_SOURCE_DIR}/docs/DoxygenConfig.cmake)


add_subdirectory(app)
add_subdirectory(tests)