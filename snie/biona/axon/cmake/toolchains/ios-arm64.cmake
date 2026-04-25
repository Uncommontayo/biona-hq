# ios-arm64.cmake — iOS ARM64 toolchain (requires Xcode generator)
#
# Usage:
#   cmake -G Xcode \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/ios-arm64.cmake ...

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_SYSTEM_PROCESSOR arm64)

set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum iOS deployment version")

# Use Xcode-managed sysroot
set(CMAKE_OSX_SYSROOT iphoneos)

set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
set(CMAKE_IOS_INSTALL_COMBINED NO)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
