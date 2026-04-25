cmake_minimum_required(VERSION 3.25)

add_library(biona_adapters INTERFACE)

target_include_directories(biona_adapters
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(biona_adapters
    INTERFACE biona_core
)
