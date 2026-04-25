cmake_minimum_required(VERSION 3.25)

add_library(biona_sdk STATIC
    src/biona_sdk.cpp
)

target_include_directories(biona_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(biona_sdk
    PUBLIC  biona_core
    PRIVATE biona_security
            biona_onnx_engine
            biona_compiler_flags
    PUBLIC  biona_adapters
)
