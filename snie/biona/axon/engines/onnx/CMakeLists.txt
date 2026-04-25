cmake_minimum_required(VERSION 3.25)

find_package(ONNXRuntime REQUIRED)

add_library(biona_onnx_engine STATIC
    src/onnx_engine.cpp
)

target_include_directories(biona_onnx_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(biona_onnx_engine
    PUBLIC  biona_core
            biona_security
    PRIVATE OnnxRuntime::OnnxRuntime
            biona_compiler_flags
)
