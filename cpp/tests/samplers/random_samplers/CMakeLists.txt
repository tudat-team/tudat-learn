add_executable(latin_hypercube_sampler_test latin_hypercube_sampler_test.cpp)
target_link_libraries(latin_hypercube_sampler_test tudat-learn Eigen3::Eigen)
add_test(NAME latin_hypercube_sampler_test COMMAND latin_hypercube_sampler_test)
