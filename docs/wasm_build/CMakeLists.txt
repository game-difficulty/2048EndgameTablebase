cmake_minimum_required(VERSION 3.15)
project(2048_AI_Core_WASM)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "This CMake file must be built using emcmake (Emscripten).")
endif()

set(CANONICAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../native_core")
set(DOCS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

if(NOT EXISTS "${CANONICAL_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "native_core/ is required as the canonical native source workspace.")
endif()

add_library(common_config INTERFACE)
target_compile_features(common_config INTERFACE cxx_std_17)
target_compile_options(common_config INTERFACE
    -O3 -ffast-math -flto
)
target_link_options(common_config INTERFACE
    -O3 -flto
    -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=1024MB
)
target_include_directories(common_config INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CANONICAL_ROOT}/include"
)

set(AI_CORE_SOURCES
    "${CANONICAL_ROOT}/src/AIPlayer - wasm.cpp"
    "${CANONICAL_ROOT}/src/egtb_query.cpp"
    "${CANONICAL_ROOT}/src/egtb_data_512.cpp"
    "${CANONICAL_ROOT}/src/egtb_data_256.cpp"
    "${CANONICAL_ROOT}/src/egtb_data_1256.cpp"
    "${CANONICAL_ROOT}/src/bindings_wasm.cpp"
)

add_executable(ai_core_wasm ${AI_CORE_SOURCES})
target_link_libraries(ai_core_wasm PRIVATE common_config)
target_link_options(ai_core_wasm PRIVATE
    -lembind
    -sEXPORT_ES6=1
    -sMODULARIZE=1
    -sEXPORT_NAME="createAICore"
    -sWASM_BIGINT=1
    -sERROR_ON_UNDEFINED_SYMBOLS=0
    -sALLOW_MEMORY_GROWTH=1
)
set_target_properties(ai_core_wasm PROPERTIES
    OUTPUT_NAME "ai_core"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${DOCS_ROOT}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${DOCS_ROOT}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${DOCS_ROOT}"
)
