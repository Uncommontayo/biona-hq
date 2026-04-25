cmake_minimum_required(VERSION 3.25)

find_package(OpenSSL REQUIRED)

add_library(biona_security STATIC
    src/safe_log.cpp
    src/secret_manager_env.cpp
    src/secret_manager_factory.cpp
    src/model_loader.cpp
)

target_include_directories(biona_security
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(biona_security
    PUBLIC  biona_core
    PRIVATE OpenSSL::SSL
            OpenSSL::Crypto
            biona_compiler_flags
)
