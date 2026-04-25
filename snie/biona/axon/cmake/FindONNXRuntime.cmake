# FindONNXRuntime.cmake — Locates ONNX Runtime headers and libraries
#
# Imported targets:
#   OnnxRuntime::OnnxRuntime
#
# Cache variables:
#   ONNXRUNTIME_ROOT_DIR   — root of the ONNX Runtime installation
#   ONNXRUNTIME_INCLUDE_DIR
#   ONNXRUNTIME_LIBRARY

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT_DIR}/include
        $ENV{ONNXRUNTIME_ROOT_DIR}/include
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT_DIR}/lib
        $ENV{ONNXRUNTIME_ROOT_DIR}/lib
        /usr/local/lib
        /usr/lib
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
)

if(ONNXRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)

    # On Windows the prebuilt release has onnxruntime.dll + onnxruntime.lib (import lib)
    if(WIN32)
        find_file(ONNXRUNTIME_DLL
            NAMES onnxruntime.dll
            HINTS
                ${ONNXRUNTIME_ROOT_DIR}/lib
                ${ONNXRUNTIME_ROOT_DIR}/bin
                $ENV{ONNXRUNTIME_ROOT_DIR}/lib
                $ENV{ONNXRUNTIME_ROOT_DIR}/bin
        )
        if(ONNXRUNTIME_DLL)
            set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
                IMPORTED_LOCATION             "${ONNXRUNTIME_DLL}"
                IMPORTED_IMPLIB               "${ONNXRUNTIME_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
        else()
            set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
                IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
        endif()
    else()
        set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
            IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
