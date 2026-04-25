cmake_minimum_required(VERSION 3.25)

add_executable(test_pipeline_mock test_pipeline_mock.cpp)
target_link_libraries(test_pipeline_mock PRIVATE biona_core)
add_test(NAME test_pipeline_mock COMMAND test_pipeline_mock)
