cmake_minimum_required(VERSION 3.25)

add_executable(eval_harness eval_harness.cpp)
target_link_libraries(eval_harness PRIVATE biona_sdk biona_core biona_security biona_onnx_engine)
