add_executable(testFirst test.cpp)
target_link_libraries(testFirst tudat_learn)
add_test(NAME testFirst COMMAND testFirst)