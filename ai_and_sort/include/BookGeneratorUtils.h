#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace BookGeneratorUtils {

    // ========================================================================
    // 外部排序库 (sort_wrapper.dll) 接口声明
    // 请在 CMakeLists.txt 中通过 target_link_libraries 链接对应的 .lib 文件
    // ========================================================================
    extern "C" void sort_uint64(uint64_t* data, size_t size, bool descending);

    /**
     * 包装外部排序调用
     */
    void sort_array(uint64_t* arr, size_t length, int num_threads = 1);

    // ========================================================================
    // 内存处理与去重工具
    // ========================================================================

    /**
     * 多线程原地去重已排序数组 (对应 parallel_unique)
     * @return 去重后的有效长度
     */
    size_t parallel_unique(uint64_t* arr, size_t length, int num_threads);

    /**
     * 将每一段有效数据原地紧缩合并到前一段末尾 (对应 merge_inplace)
     * @return 紧缩后的总有效长度
     */
    size_t merge_inplace(uint64_t* arr, const std::vector<size_t>& segment_ends, const std::vector<size_t>& segment_starts);

    /**
     * 将多个数组并行拼接为一个大数组 (对应 concatenate)
     */
    std::vector<uint64_t> concatenate(const std::vector<std::vector<uint64_t>>& arrays);

    // ========================================================================
    // 多路归并工具 (K-way Merge)
    // ========================================================================

    /**
     * 内部使用：代表一个数组的数据段
     */
    struct ArraySegment {
        const uint64_t* data;
        size_t length;
        size_t index;
    };

    /**
     * 多线程归并并去重多个完整的有序数组 (对应 merge_deduplicate_all)
     */
    std::vector<std::vector<uint64_t>> merge_deduplicate_all(
        const std::vector<std::vector<uint64_t>>& arrays,
        const std::vector<uint64_t>& pivots,
        int n_threads
    );

    /**
     * 合并并去重两个有序数组 (对应 merge_and_deduplicate)
     */
    std::vector<uint64_t> merge_and_deduplicate(const std::vector<uint64_t>& arr1, const std::vector<uint64_t>& arr2);

    // ========================================================================
    // 数学工具
    // ========================================================================

    inline uint64_t largest_power_of_2(uint64_t n) {
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    inline uint64_t hash_board(uint64_t board) {
        return ((board ^ (board >> 27)) * 0x1A85EC53ULL + board) >> 23;
    }

} // namespace BookGeneratorUtils