cmake_minimum_required(VERSION 3.18)
project(2048_AI_Core LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

set(_nanobind_found FALSE)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_result
)

if(nanobind_result EQUAL 0 AND EXISTS "${nanobind_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
    find_package(nanobind CONFIG QUIET)
    if(nanobind_FOUND)
        set(_nanobind_found TRUE)
    endif()
endif()

if(NOT _nanobind_found AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/nanobind/CMakeLists.txt")
    add_subdirectory(extern/nanobind)
    set(_nanobind_found TRUE)
endif()

if(NOT _nanobind_found)
    message(FATAL_ERROR
        "nanobind is unavailable. Install it with `pip install nanobind` or provide ai_and_sort/extern/nanobind."
    )
endif()

find_package(OpenMP REQUIRED)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

add_library(common_config INTERFACE)
target_compile_features(common_config INTERFACE cxx_std_17)
target_compile_definitions(common_config INTERFACE NB_NDOC NB_LEAK_WARNINGS=0)
target_include_directories(common_config INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    target_compile_options(common_config INTERFACE /O2 /arch:AVX2 /openmp)
else()
    target_compile_options(common_config INTERFACE
        -O3 -march=x86-64-v2 -ffast-math -fopenmp
        -ffunction-sections -fdata-sections
    )
    target_link_options(common_config INTERFACE
        -static-libstdc++ -Wl,--gc-sections -s -fopenmp
    )
endif()

set(AI_CORE_SOURCES
    src/AIPlayer.cpp
    src/egtb_query.cpp
    src/egtb_data_512.cpp
    src/egtb_data_256.cpp
    src/egtb_data_1256.cpp
    src/bindings_ai.cpp
)

nanobind_add_module(ai_core ${AI_CORE_SOURCES})
target_link_libraries(ai_core PRIVATE common_config OpenMP::OpenMP_CXX)

set(MOVER_CORE_SOURCES
    src/bindings_mover.cpp
)

nanobind_add_module(mover_core ${MOVER_CORE_SOURCES})
target_link_libraries(mover_core PRIVATE common_config)

set(ALL_PYD_MODULES ai_core mover_core)

foreach(target ${ALL_PYD_MODULES})
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(ipo_supported)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endforeach()
