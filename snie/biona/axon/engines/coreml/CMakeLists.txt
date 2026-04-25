cmake_minimum_required(VERSION 3.25)

# CoreML engine — stub, iOS only, implemented in a future prompt
if(BIONA_PLATFORM_IOS)
    add_library(biona_coreml_engine INTERFACE)

    target_include_directories(biona_coreml_engine
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(biona_coreml_engine
        INTERFACE biona_core
    )
else()
    # On non-iOS platforms, provide an empty stub so top-level CMake doesn't fail
    add_library(biona_coreml_engine INTERFACE)
endif()
