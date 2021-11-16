add_executable(rbf_test rbf_test.cpp)
target_link_libraries(rbf_test tudat_learn)
add_test(NAME rbf_test COMMAND rbf_test)