# CompilerFlags.cmake — Hardened compiler flags for Biona Axon
# Supports: GCC/Clang (Linux/macOS/Android/iOS) and MSVC (Windows)

add_library(biona_compiler_flags INTERFACE)

if(MSVC)
    target_compile_options(biona_compiler_flags INTERFACE
        /W4         # High warning level (≈ -Wall -Wextra)
        /WX         # Warnings as errors (≈ -Werror)
        /GS         # Buffer security check (≈ -fstack-protector-strong, on by default)
        /sdl        # Additional SDL checks
        /permissive-# Strict conformance (≈ -Wpedantic)
        /wd4324     # Suppress C4324: structure padded due to alignas (expected for SPSC queue)
        /wd4458     # Suppress C4458: declaration hides class member (common in ORT headers)
    )
    # /DYNAMICBASE = ASLR (≈ -fPIE), on by default in MSVC
else()
    # GCC / Clang
    target_compile_options(biona_compiler_flags INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -fstack-protector-strong
        -fPIE
    )

    # -D_FORTIFY_SOURCE=2 only in Release (requires -O2 or higher)
    target_compile_options(biona_compiler_flags INTERFACE
        $<$<CONFIG:Release>:-D_FORTIFY_SOURCE=2>
        $<$<CONFIG:RelWithDebInfo>:-D_FORTIFY_SOURCE=2>
    )
endif()
