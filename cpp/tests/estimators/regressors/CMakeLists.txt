add_executable(rbf_test rbf_test.cpp)
target_link_libraries(rbf_test tudat-learn Eigen3::Eigen)
add_test(NAME rbf_test COMMAND rbf_test)

add_executable(rbfn_test rbfn_test.cpp)
target_link_libraries(rbfn_test tudat-learn Eigen3::Eigen)
add_test(NAME rbfn_test COMMAND rbfn_test)

add_executable(grnn_test grnn_test.cpp)
target_link_libraries(grnn_test tudat-learn Eigen3::Eigen)
add_test(NAME grnn_test COMMAND grnn_test)