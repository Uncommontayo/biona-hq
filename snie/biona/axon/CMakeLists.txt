cmake_minimum_required(VERSION 3.25)
project(BionaAxon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Platform detection ──────────────────────────────────────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(BIONA_PLATFORM_LINUX TRUE)
    add_compile_definitions(BIONA_PLATFORM_LINUX)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(BIONA_PLATFORM_ANDROID TRUE)
    add_compile_definitions(BIONA_PLATFORM_ANDROID)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(BIONA_PLATFORM_IOS TRUE)
    add_compile_definitions(BIONA_PLATFORM_IOS)
endif()

# ── Compiler flags ──────────────────────────────────────────────────────────
include(cmake/CompilerFlags.cmake)

# ── ONNX Runtime finder ─────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── Build options ───────────────────────────────────────────────────────────
option(BUILD_TESTING "Build tests" ON)

# ── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(core)
add_subdirectory(security)
add_subdirectory(engines)
add_subdirectory(adapters)
add_subdirectory(sdk)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
