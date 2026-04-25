# android-arm64.cmake — Android NDK ARM64 toolchain
#
# Requires the ANDROID_NDK environment variable to point to the NDK root.
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-arm64.cmake \
#         -DANDROID_NDK=$ANDROID_NDK ...

if(NOT DEFINED ENV{ANDROID_NDK} AND NOT DEFINED ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK environment variable is not set.")
endif()

if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK $ENV{ANDROID_NDK})
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 26)            # Android API level minimum
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
set(CMAKE_ANDROID_STL_TYPE c++_static)

set(CMAKE_C_COMPILER   ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang)
set(CMAKE_CXX_COMPILER ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
