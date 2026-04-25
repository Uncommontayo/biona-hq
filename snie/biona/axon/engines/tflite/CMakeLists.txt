cmake_minimum_required(VERSION 3.25)

# TFLite engine — stub, implemented in a future prompt
add_library(biona_tflite_engine INTERFACE)

target_include_directories(biona_tflite_engine
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(biona_tflite_engine
    INTERFACE biona_core
)
