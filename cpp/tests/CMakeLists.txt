add_executable(dataset_test dataset_test.cpp)
target_link_libraries(dataset_test tudat-learn Eigen3::Eigen)
add_test(NAME dataset_test COMMAND dataset_test)

add_subdirectory(estimators)
add_subdirectory(model_evaluation)
add_subdirectory(processing)
add_subdirectory(samplers)