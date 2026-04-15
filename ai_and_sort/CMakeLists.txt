cmake_minimum_required(VERSION 3.15)
project(2048_AI_Core)

# 1. 查找依赖项
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
if (UNIX)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(nanobind CONFIG REQUIRED)
else()
    add_subdirectory(extern/nanobind)
endif ()
find_package(OpenMP REQUIRED)

# 2. 先定义目标 (target)
nanobind_add_module(ai_core AIPlayer.cpp egtb_query.cpp egtb_data_512.cpp egtb_data_256.cpp egtb_data_1256.cpp)

# 3. 设置基础特性
target_compile_features(ai_core PRIVATE cxx_std_17)
target_link_libraries(ai_core PRIVATE OpenMP::OpenMP_CXX)

# 4. 符号可见性优化 (减小体积)
set_target_properties(ai_core PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# 5. 开启 LTO / IPO (性能提升 + 体积优化)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set_target_properties(ai_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# 6. 编译器和链接器参数优化
# GCC/Clang 优化：剔除死代码，Strip 符号表
target_compile_options(ai_core PRIVATE 
    -O3 -march=x86-64-v3 -ffast-math -fopenmp
    -ffunction-sections -fdata-sections
)
target_link_options(ai_core PRIVATE 
    -static-libstdc++ 
    -Wl,--gc-sections -s -fopenmp
)

target_compile_definitions(ai_core PRIVATE 
    NB_NDOC 
    NB_LEAK_WARNINGS=0)
