cmake_minimum_required(VERSION 3.18)
project(2048_AI_Core LANGUAGES CXX)

option(NATIVE_ENABLE_ASAN "Enable AddressSanitizer for native_core debug builds." OFF)
option(NATIVE_BUILD_TESTS "Build native_core test executables when test sources are available." OFF)

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
target_compile_definitions(common_config INTERFACE
    NB_NDOC
    NB_LEAK_WARNINGS=0
    BC_PORTABLE_X86_64_ARCH="${PORTABLE_X86_64_ARCH}"
)
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
    src/CanonicalBatch.cpp
    src/BookGeneratorUtils.cpp
    src/BookGenerator.cpp
    src/EXPrefix40Layer.cpp
    src/EXPrefix36Runtime.cpp
    src/EXFrozenLayer.cpp
    src/EXCompressedResult.cpp
    src/BookGeneratorEX.cpp
    src/EXADLayer.cpp
    src/EXADIO.cpp
    src/EXADCompressedResult.cpp
    src/EXADBuilder.cpp
    src/BCBacksolve.cpp
    src/BookGeneratorEXAD.cpp
    src/BookSolverEXAD.cpp
    src/BoardMaskerAD.cpp
    src/BoardMoverAD.cpp
    src/BookGeneratorAD.cpp
    src/BookSolverEX.cpp
    src/BookSolverAD.cpp
    src/BookSolver.cpp
    src/bindings_formation.cpp
)

nanobind_add_module(formation_core ${FORMATION_CORE_SOURCES})
target_link_libraries(formation_core PRIVATE common_config OpenMP::OpenMP_CXX)

set(CANONICAL_BATCH_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_canonical_batch.cpp")
set(BC_FAMILY_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_family.cpp")
set(BC_FAMILY_ROUTE_PLANNER_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_family_route_planner.cpp")
set(BC_FAMILY_PARTITION_ANALYSIS_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_family_partition_analysis.cpp")
set(BC_KEY_RANK_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_key_rank.cpp")
set(BC_CELL_BUILDER_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_cell_builder.cpp")
set(BC_POSITION_FILE_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_position_file.cpp")
set(BC_POSITION_CELL_LOADER_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_position_cell_loader.cpp")
set(BC_SUCCESS_IO_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_success_io.cpp")
set(BC_POSITION_SCANNER_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_position_scanner.cpp")
set(BC_BOARD_CODEC_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_board_codec.cpp")
set(BC_BOARD_OPS_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_board_ops.cpp")
set(BC_BACKSOLVE_RESIDENT_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_backsolve_resident.cpp")
set(BC_RESIDENT_SOLVE_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_resident_solve.cpp")
set(BC_SINGLE_CHUNK_SOLVE_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_single_chunk_solve.cpp")
set(BC_FAMILY_GENERATION_STATE_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_family_generation_state.cpp")
set(BC_RESIDENT_GENERATION_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_resident_generation.cpp")
set(BC_FILE_IO_TEST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/test_bc_file_io.cpp")
set(BC_FAMILY_GENERATION_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_family_generation.cpp")
set(BC_RESIDENT_GENERATION_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_resident_generation.cpp")
set(BC_SINGLE_CHUNK_GENERATION_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_single_chunk_generation.cpp")
set(BC_BACKSOLVE_RESIDENT_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_backsolve_resident.cpp")
set(BC_RESIDENT_SOLVE_FULL_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_resident_solve_full.cpp")
set(BC_RESIDENT_SOLVE_LAYER_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_resident_solve_layer.cpp")
set(BC_SINGLE_CHUNK_SOLVE_LAYER_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_single_chunk_solve_layer.cpp")
set(BC_SINGLE_CHUNK_SOLVE_FULL_BENCH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/bench_bc_single_chunk_solve_full.cpp")
set(BC_FAMILY_LAYER_DIFF_TOOL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/tool_bc_family_layer_diff.cpp")
set(BC_POSITION_REMAP_EXACT_TOOL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests_src/tool_bc_position_remap_exact.cpp")

if(NATIVE_BUILD_TESTS)
    foreach(_bc_required_test_source
        "${CANONICAL_BATCH_TEST_SOURCE}"
        "${BC_FAMILY_TEST_SOURCE}"
        "${BC_FAMILY_ROUTE_PLANNER_TEST_SOURCE}"
        "${BC_FAMILY_PARTITION_ANALYSIS_TEST_SOURCE}"
        "${BC_KEY_RANK_TEST_SOURCE}"
        "${BC_CELL_BUILDER_TEST_SOURCE}"
        "${BC_POSITION_FILE_TEST_SOURCE}"
        "${BC_POSITION_CELL_LOADER_TEST_SOURCE}"
        "${BC_SUCCESS_IO_TEST_SOURCE}"
        "${BC_POSITION_SCANNER_TEST_SOURCE}"
        "${BC_BOARD_CODEC_TEST_SOURCE}"
        "${BC_BOARD_OPS_TEST_SOURCE}"
        "${BC_BACKSOLVE_RESIDENT_TEST_SOURCE}"
        "${BC_RESIDENT_SOLVE_TEST_SOURCE}"
        "${BC_SINGLE_CHUNK_SOLVE_TEST_SOURCE}"
        "${BC_FAMILY_GENERATION_STATE_TEST_SOURCE}"
        "${BC_RESIDENT_GENERATION_TEST_SOURCE}"
        "${BC_FILE_IO_TEST_SOURCE}"
        "${BC_FAMILY_GENERATION_BENCH_SOURCE}"
        "${BC_RESIDENT_GENERATION_BENCH_SOURCE}"
        "${BC_SINGLE_CHUNK_GENERATION_BENCH_SOURCE}"
        "${BC_BACKSOLVE_RESIDENT_BENCH_SOURCE}"
        "${BC_RESIDENT_SOLVE_FULL_BENCH_SOURCE}"
        "${BC_RESIDENT_SOLVE_LAYER_BENCH_SOURCE}"
        "${BC_SINGLE_CHUNK_SOLVE_LAYER_BENCH_SOURCE}"
        "${BC_SINGLE_CHUNK_SOLVE_FULL_BENCH_SOURCE}"
        "${BC_FAMILY_LAYER_DIFF_TOOL_SOURCE}"
        "${BC_POSITION_REMAP_EXACT_TOOL_SOURCE}"
    )
        if(NOT EXISTS "${_bc_required_test_source}")
            message(FATAL_ERROR
                "NATIVE_BUILD_TESTS=ON but ${_bc_required_test_source} is missing. "
                "Test sources are optional and are not required for normal Linux or Windows builds."
            )
        endif()
    endforeach()

    add_executable(canonical_batch_test
        src/CanonicalBatch.cpp
        tests_src/test_canonical_batch.cpp
    )
    target_link_libraries(canonical_batch_test PRIVATE common_config)

    add_executable(bc_family_test
        tests_src/test_bc_family.cpp
    )
    target_link_libraries(bc_family_test PRIVATE common_config)

    add_executable(bc_family_route_planner_test
        tests_src/test_bc_family_route_planner.cpp
    )
    target_link_libraries(bc_family_route_planner_test PRIVATE common_config)

    add_executable(bc_family_partition_analysis_test
        tests_src/test_bc_family_partition_analysis.cpp
    )
    target_link_libraries(bc_family_partition_analysis_test PRIVATE common_config)

    add_executable(bc_key_rank_test
        tests_src/test_bc_key_rank.cpp
    )
    target_link_libraries(bc_key_rank_test PRIVATE common_config)

    add_executable(bc_cell_builder_test
        tests_src/test_bc_cell_builder.cpp
    )
    target_link_libraries(bc_cell_builder_test PRIVATE common_config)

    add_executable(bc_position_file_test
        src/BCPositionFile.cpp
        tests_src/test_bc_position_file.cpp
    )
    target_link_libraries(bc_position_file_test PRIVATE common_config)

    add_executable(bc_position_cell_loader_test
        src/BCPositionFile.cpp
        tests_src/test_bc_position_cell_loader.cpp
    )
    target_link_libraries(bc_position_cell_loader_test PRIVATE common_config)

    add_executable(bc_success_io_test
        src/BCPositionFile.cpp
        tests_src/test_bc_success_io.cpp
    )
    target_link_libraries(bc_success_io_test PRIVATE common_config)

    add_executable(bc_position_scanner_test
        src/BCPositionFile.cpp
        tests_src/test_bc_position_scanner.cpp
    )
    target_link_libraries(bc_position_scanner_test PRIVATE common_config)

    add_executable(bc_board_codec_test
        tests_src/test_bc_board_codec.cpp
    )
    target_link_libraries(bc_board_codec_test PRIVATE common_config)

    add_executable(bc_board_ops_test
        tests_src/test_bc_board_ops.cpp
    )
    target_link_libraries(bc_board_ops_test PRIVATE common_config)

    add_executable(bc_backsolve_resident_test
        src/BCPositionFile.cpp
        src/BCBacksolve.cpp
        src/CanonicalBatch.cpp
        tests_src/test_bc_backsolve_resident.cpp
    )
    target_link_libraries(bc_backsolve_resident_test PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_resident_solve_test
        src/BCPositionFile.cpp
        src/BCBacksolve.cpp
        src/CanonicalBatch.cpp
        tests_src/test_bc_resident_solve.cpp
    )
    target_link_libraries(bc_resident_solve_test PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_single_chunk_solve_test
        src/BCPositionFile.cpp
        src/CanonicalBatch.cpp
        tests_src/test_bc_single_chunk_solve.cpp
    )
    target_link_libraries(bc_single_chunk_solve_test PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_family_generation_state_test
        src/BCPositionFile.cpp
        src/BCFamilyGeneration.cpp
        src/BCResidentGeneration.cpp
        src/BCResidentGenerationStreaming.cpp
        src/CanonicalBatch.cpp
        tests_src/test_bc_family_generation_state.cpp
    )
    target_link_libraries(bc_family_generation_state_test PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_resident_generation_test
        src/BCPositionFile.cpp
        src/BCResidentGeneration.cpp
        src/CanonicalBatch.cpp
        tests_src/test_bc_resident_generation.cpp
    )
    target_link_libraries(bc_resident_generation_test PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_file_io_test
        src/BCPositionFile.cpp
        tests_src/test_bc_file_io.cpp
    )
    target_link_libraries(bc_file_io_test PRIVATE common_config)

    add_executable(bc_family_generation_bench
        src/BCPositionFile.cpp
        src/BCFamilyGeneration.cpp
        src/BCResidentGeneration.cpp
        src/BCResidentGenerationStreaming.cpp
        src/BCSingleChunkGeneration.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_family_generation.cpp
    )
    target_link_libraries(bc_family_generation_bench PRIVATE common_config OpenMP::OpenMP_CXX)
    if(WIN32)
        target_link_libraries(bc_family_generation_bench PRIVATE psapi)
    endif()

    add_executable(bc_resident_generation_bench
        src/BCPositionFile.cpp
        src/BCResidentGeneration.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_resident_generation.cpp
    )
    target_link_libraries(bc_resident_generation_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_generation_compute_bench
        src/BCPositionFile.cpp
        src/BCResidentGeneration.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_generation_compute.cpp
    )
    target_link_libraries(bc_generation_compute_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_single_chunk_generation_bench
        src/BCPositionFile.cpp
        src/BCResidentGeneration.cpp
        src/BCResidentGenerationStreaming.cpp
        src/BCSingleChunkGeneration.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_single_chunk_generation.cpp
    )
    target_link_libraries(bc_single_chunk_generation_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_backsolve_resident_bench
        src/BCPositionFile.cpp
        src/BCBacksolve.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_backsolve_resident.cpp
    )
    target_link_libraries(bc_backsolve_resident_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_resident_solve_full_bench
        src/BCPositionFile.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_resident_solve_full.cpp
    )
    target_link_libraries(bc_resident_solve_full_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_resident_solve_layer_bench
        src/BCPositionFile.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_resident_solve_layer.cpp
    )
    target_link_libraries(bc_resident_solve_layer_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_single_chunk_solve_layer_bench
        src/BCPositionFile.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_single_chunk_solve_layer.cpp
    )
    target_link_libraries(bc_single_chunk_solve_layer_bench PRIVATE common_config OpenMP::OpenMP_CXX)

    add_executable(bc_single_chunk_solve_full_bench
        src/BCPositionFile.cpp
        src/CanonicalBatch.cpp
        tests_src/bench_bc_single_chunk_solve_full.cpp
    )
    target_link_libraries(bc_single_chunk_solve_full_bench PRIVATE common_config OpenMP::OpenMP_CXX)
    if(WIN32)
        target_link_libraries(bc_single_chunk_solve_full_bench PRIVATE psapi)
    endif()

    add_executable(bc_family_layer_diff_tool
        src/BCPositionFile.cpp
        tests_src/tool_bc_family_layer_diff.cpp
    )
    target_link_libraries(bc_family_layer_diff_tool PRIVATE common_config)

    add_executable(bc_position_remap_exact_tool
        src/BCPositionFile.cpp
        tests_src/tool_bc_position_remap_exact.cpp
    )
    target_link_libraries(bc_position_remap_exact_tool PRIVATE common_config OpenMP::OpenMP_CXX)
endif()

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
