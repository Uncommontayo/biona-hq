cmake_minimum_required(VERSION 3.25)

include(FetchContent)

# libfvad — WebRTC VAD C port (Apache 2.0)
FetchContent_Declare(
    libfvad
    GIT_REPOSITORY https://github.com/dpirch/libfvad.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(libfvad)

# core — zero external dependencies on engines/, security/, adapters/, sdk/
add_library(biona_core STATIC
    src/signal.cpp
    src/inference_engine_factory.cpp
    src/onnx_contract.cpp
    src/vad/webrtc_vad.cpp
    src/vad/energy_vad.cpp
    src/vad/vad_factory.cpp
    src/signal_thread_pool.cpp
)

target_include_directories(biona_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(biona_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vad
        ${libfvad_SOURCE_DIR}/include
        # signal_thread_pool.cpp uses safe_log — include security headers for .cpp only
        ${CMAKE_CURRENT_SOURCE_DIR}/../security/include
)

target_link_libraries(biona_core
    PUBLIC  biona_compiler_flags
    PRIVATE fvad
)

# Enforce no linkage to restricted modules
# (CI can verify by checking target_link_libraries of biona_core contains none of:
#  biona_engines, biona_security, biona_adapters, biona_sdk)
