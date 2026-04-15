#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include "formation.h"
#include "BookGeneratorUtils.h"

namespace BookGenerator {

    // ========================================================================
    // 1. 数据结构
    // ========================================================================

    // 封装 gen_boards 返回的计数结果
    struct GenBoardsResult {
        size_t total_arr1;
        size_t total_arr2;
    };

    // ========================================================================
    // 2. 核心生成引擎 (模板化以支持 Mover 的静态分发)
    // ========================================================================

    /**
     * 超大数组的并行生成核心
     * 使用裸指针以获得极致的内存访问速度和对非对齐内存的支持
     */
    template <typename Mover>
    GenBoardsResult gen_boards(
        const uint64_t* arr0, size_t arr0_size,
        int target,
        const PatternConfig& config,
        uint64_t* hashmap1, size_t hashmap1_size,
        uint64_t* hashmap2, size_t hashmap2_size,
        uint64_t* arr1, uint64_t* arr2,
        int num_threads,
        bool do_check,
        bool isfree
    );

    /**
     * 简单生成引擎 (对应 Python 的 gen_boards_simple)
     * 用于处理初期层级（局面数较少）的情况
     */
    template <typename Mover>
    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> 
    gen_boards_simple(
        const uint64_t* arr0, size_t arr0_size,
        int target,
        const PatternConfig& config,
        bool do_check,
        bool isfree
    );

    // ========================================================================
    // 3. 宏观调度与断点控制
    // ========================================================================

    /**
     * 处理超大数组的切片调度 (对应 Python 的 gen_boards_big)
     * 负责将 arr0 拆分为 Segments 逐个处理，并在层内进行归并去重
     */
    struct GenBoardsResult {
        size_t total_arr1; 
        size_t total_arr2; 
        std::vector<size_t> counts1; // 新增
        std::vector<size_t> counts2; // 新增
    };

    std::tuple<
        std::vector<std::vector<uint64_t>>, 
        std::vector<std::vector<uint64_t>>, 
        std::vector<uint64_t>,              
        std::vector<uint64_t>               
    >
    gen_boards_big(
        const std::vector<uint64_t>& arr0,
        int target,
        const PatternConfig& config,
        std::vector<uint64_t>& hashmap1,
        std::vector<uint64_t>& hashmap2,
        int num_threads,
        std::vector<std::vector<double>>& length_factors_list, 
        double& length_factor_multiplier,                      
        bool do_check,
        bool isfree,
        bool is_variant
    );

    /**
     * 生成定式查表文件的主流程
     */
    std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> 
    generate_process(
        const std::vector<uint64_t>& arr_init,
        const PatternConfig& config,
        int target,
        int steps,
        int tile_sum,
        const std::string& pathname,
        bool isfree,
        bool is_variant
    );

    /**
     * 处理断点重连逻辑：检查磁盘文件是否存在，决定是否跳过当前 step
     */
    std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> 
    handle_restart(int step_i, const std::string& pathname, const std::vector<uint64_t>& arr_init, bool started);

    // 辅助工具：调整无锁哈希表的大小
    void update_hashmap_length(std::vector<uint64_t>& hashmap, size_t current_arr_size);

} // namespace BookGenerator