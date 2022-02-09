set(CMAKE_CXX_STANDARD 17)

enable_testing()

include(CTest)
include(GNUInstallDirs)

set(CMAKE_PREFIX_PATH "$CONDA_PREFIX")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# set(TUDAT_LEARN_SRC_FILES
#         ### Core Classes ###
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/estimator.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/processing.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/random.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/sampler.cpp

#         ## Estimators ##
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/classifier.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/clustering.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/estimators/regressor.cpp

# )


# add_library(tudat_learn SHARED ${TUDAT_LEARN_SRC_FILES})

# set_target_properties(tudat_learn PROPERTIES VERSION ${PROJECT_VERSION})

# set_target_properties(tudat_learn PROPERTIES SOVERSION 0)



# target_include_directories(tudat_learn PUBLIC 
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${EIGEN3_INCLUDE_DIRS}
# )


# add_subdirectory(app)
# add_subdirectory(tests)


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

# target_include_directories(tudat_learn PUBLIC 
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${EIGEN3_INCLUDE_DIRS}
# )

set(HEADERS_PUBLIC
        include/tudat-learn/dataset.hpp
        include/tudat-learn/estimator.hpp
        include/tudat-learn/function.hpp
        include/tudat-learn/io.hpp
        include/tudat-learn/learn.hpp
        include/tudat-learn/operator.hpp
        include/tudat-learn/processing.hpp
        include/tudat-learn/random.hpp
        include/tudat-learn/response.hpp
        include/tudat-learn/sampler.hpp
        include/tudat-learn/types.hpp

        include/tudat-learn/estimators/classifier.hpp
        include/tudat-learn/estimators/clustering.hpp
        include/tudat-learn/estimators/regressor.hpp

        include/tudat-learn/estimators/regressors/grnn.hpp
        include/tudat-learn/estimators/regressors/rbf.hpp
        include/tudat-learn/estimators/regressors/rbfn.hpp

        include/tudat-learn/processing/scaler.hpp
        include/tudat-learn/processing/scalers/minmax_scaler.hpp
        include/tudat-learn/processing/scalers/standard_scaler.hpp

        include/tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp
)

include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)


add_subdirectory(app)
add_subdirectory(tests)