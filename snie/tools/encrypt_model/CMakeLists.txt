cmake_minimum_required(VERSION 3.25)

find_package(OpenSSL REQUIRED)

add_executable(encrypt_model encrypt_model.cpp)

target_compile_features(encrypt_model PRIVATE cxx_std_17)

target_link_libraries(encrypt_model
    PRIVATE OpenSSL::SSL
            OpenSSL::Crypto
)
