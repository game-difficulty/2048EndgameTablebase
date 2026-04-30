cmake_minimum_required(VERSION 3.18)
project(2048_AI_Core LANGUAGES CXX)

option(NATIVE_ENABLE_ASAN "Enable AddressSanitizer for native_core debug builds." OFF)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

set(_nanobind_found FALSE)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
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
        "nanobind is unavailable. Install it with `pip install nanobind` or provide native_core/extern/nanobind."
    )
endif()

find_package(OpenMP REQUIRED)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

set(PORTABLE_X86_64_ARCH "x86-64-v3" CACHE STRING
    "Portable x86-64 ISA baseline for GCC/Clang builds."
)
set(PORTABLE_X86_64_TUNE "generic" CACHE STRING
    "CPU tuning target for GCC/Clang builds."
)

get_filename_component(PYTHON_ROOT_DIR "${Python_EXECUTABLE}" DIRECTORY)
set(LIBLZMA_HINT_DIRS
    "${PYTHON_ROOT_DIR}/Library/include"
    "/usr/include"
    "/usr/local/include"
)
find_path(LIBLZMA_INCLUDE_DIR lzma.h HINTS ${LIBLZMA_HINT_DIRS})
if(NOT LIBLZMA_INCLUDE_DIR)
    message(FATAL_ERROR "lzma.h not found. Install liblzma headers or provide them in the Python environment.")
endif()

add_library(common_config INTERFACE)
target_compile_features(common_config INTERFACE cxx_std_17)
target_compile_definitions(common_config INTERFACE NB_NDOC NB_LEAK_WARNINGS=0)
target_include_directories(common_config INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${LIBLZMA_INCLUDE_DIR}"
)

if(MSVC)
    target_compile_options(common_config INTERFACE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<NOT:$<CONFIG:Debug>>:/O2 /Ot /GL /fp:fast>
        /Zo
        /Oy-
        /arch:AVX2
        /openmp
    )
    target_link_options(common_config INTERFACE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )
else()
    target_compile_options(common_config INTERFACE
        $<$<CONFIG:Debug>:-O0 -g3 -fno-omit-frame-pointer -fno-inline -fno-optimize-sibling-calls>
        $<$<NOT:$<CONFIG:Debug>>:-Ofast -ffast-math -fno-math-errno -fno-trapping-math -fomit-frame-pointer -funroll-loops>
        -march=${PORTABLE_X86_64_ARCH}
        -mtune=${PORTABLE_X86_64_TUNE}
        -fopenmp
        -ffunction-sections -fdata-sections
    )
    target_link_options(common_config INTERFACE
        $<$<CONFIG:Debug>:-g3>
        $<$<CONFIG:RelWithDebInfo>:-Wl,--gc-sections>
        $<$<CONFIG:Release>:-Wl,--gc-sections -s>
        $<$<CONFIG:MinSizeRel>:-Wl,--gc-sections -s>
        -static-libstdc++
        -fopenmp
    )
endif()

if(NATIVE_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(common_config INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    target_link_options(common_config INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
endif()

set(REQUIRED_EGTB_DATA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/egtb_data_512.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/egtb_data_256.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/egtb_data_1256.cpp"
)

foreach(required_source ${REQUIRED_EGTB_DATA_SOURCES})
    if(NOT EXISTS "${required_source}")
        message(FATAL_ERROR
            "Missing ${required_source}. Extract native_core/egtb_data.7z into native_core/src before configuring CMake, or use native_core/make.sh on Linux."
        )
    endif()
endforeach()

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

set(FORMATION_CORE_SOURCES
    src/FileIOUtils.cpp
    src/CompressionBridge.cpp
    src/NativeLzma.cpp
    src/TrieCompression.cpp
    src/ReaderRuntime.cpp
    src/AdaptiveIndex.cpp
    src/HybridSearch.cpp
    src/BookGeneratorUtils.cpp
    src/BookGenerator.cpp
    src/BoardMaskerAD.cpp
    src/BoardMoverAD.cpp
    src/BookGeneratorAD.cpp
    src/BookSolverAD.cpp
    src/BookSolver.cpp
    src/bindings_formation.cpp
)

nanobind_add_module(formation_core ${FORMATION_CORE_SOURCES})
target_link_libraries(formation_core PRIVATE common_config OpenMP::OpenMP_CXX)

add_executable(adaptive_index_test
    src/AdaptiveIndex.cpp
    tests/test_adaptive_index.cpp
)
target_link_libraries(adaptive_index_test PRIVATE common_config OpenMP::OpenMP_CXX)

add_library(bookgen_native SHARED
    xss_wrapper.cpp
    src/HybridSearch.cpp
    src/bookgen_native_wrapper.cpp
    src/booksolver_utils_wrapper.cpp
    src/unique_wrapper.cpp
    src/merge_tree_wrapper.cpp
)
target_include_directories(bookgen_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(bookgen_native PRIVATE
    OpenMP::OpenMP_CXX
    "${CMAKE_CURRENT_SOURCE_DIR}/libx86simdsortcpp.a"
)
set_target_properties(bookgen_native PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    PREFIX ""
    OUTPUT_NAME "bookgen_native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<NOT:$<CONFIG:Debug>>:/O2 /Ot /GL /fp:fast>
        /Zo
        /Oy-
        /arch:AVX2
        /openmp
    )
    target_link_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )
else()
    target_compile_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:-O0 -g3 -fno-omit-frame-pointer -fno-inline -fno-optimize-sibling-calls>
        $<$<NOT:$<CONFIG:Debug>>:-Ofast -ffast-math -fno-math-errno -fno-trapping-math -fomit-frame-pointer -funroll-loops>
        -march=${PORTABLE_X86_64_ARCH}
        -mtune=${PORTABLE_X86_64_TUNE}
        -fopenmp
        -ffunction-sections -fdata-sections
    )
    target_link_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:-g3>
        $<$<CONFIG:RelWithDebInfo>:-Wl,--gc-sections>
        $<$<CONFIG:Release>:-Wl,--gc-sections -s>
        $<$<CONFIG:MinSizeRel>:-Wl,--gc-sections -s>
        -static-libstdc++
        -fopenmp
    )
endif()

if(NATIVE_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    target_link_options(bookgen_native PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
endif()

set(ALL_PYD_MODULES ai_core mover_core formation_core)

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
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endif()
endforeach()

if(ipo_supported)
    set_property(TARGET bookgen_native PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET bookgen_native PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()
