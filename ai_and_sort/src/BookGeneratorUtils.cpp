#include "BookGeneratorUtils.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace BookGeneratorUtils {

    // ------------------------------------------------------------------
    // 排序与检查
    // ------------------------------------------------------------------

    void sort_array(uint64_t* arr, size_t length, int num_threads) {
        if (length < 100000) {
            std::sort(arr, arr + length);
        } else {
            // 调用 DLL 导出的 C 接口
            sort_uint64(arr, length, false);
        }
    }

    // ------------------------------------------------------------------
    // 原地紧缩去重与合并
    // ------------------------------------------------------------------

    size_t parallel_unique(uint64_t* arr, size_t length, int num_threads) {
        if (length < 128) {
            auto it = std::unique(arr, arr + length);
            return std::distance(arr, it);
        }

        size_t step = length / num_threads;
        std::vector<size_t> c_list(num_threads, 0);

        // 并行区间内去重
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_threads; ++i) {
            size_t start = i * step;
            size_t end = (i == num_threads - 1) ? length : (i + 1) * step;
            
            if (start >= end) continue;

            size_t c = (i == 0) ? 1 : 0;
            size_t start_ = (i == 0) ? start + 1 : start;

            for (size_t j = start_; j < end; ++j) {
                // arr[j-1] 访问是安全的，因为后续的 memmove 在并行的这步之后执行
                if (arr[j] != arr[j - 1]) {
                    arr[start + c] = arr[j];
                    c++;
                }
            }
            c_list[i] = c;
        }

        // 串行紧缩收集结果
        size_t result_cumulative = c_list[0];
        for (int i = 1; i < num_threads; ++i) {
            size_t start = i * step;
            size_t count = c_list[i];
            if (count > 0) {
                // memmove 允许内存重叠，非常适合这种原地覆盖
                std::memmove(arr + result_cumulative, arr + start, count * sizeof(uint64_t));
                result_cumulative += count;
            }
        }

        return result_cumulative;
    }

    size_t merge_inplace(uint64_t* arr, const std::vector<size_t>& segment_ends, const std::vector<size_t>& segment_starts) {
        size_t counts = segment_ends[0];
        size_t num_segments = segment_starts.size();

        for (size_t i = 1; i < num_segments; ++i) {
            size_t start = segment_starts[i];
            size_t end = segment_ends[i];
            size_t size = end - start;
            size_t dest_start = counts;
            
            std::memmove(arr + dest_start, arr + start, size * sizeof(uint64_t));
            counts += size;
        }

        return counts;
    }

    std::vector<uint64_t> concatenate(const std::vector<std::vector<uint64_t>>& arrays) {
        size_t total_length = 0;
        for (const auto& arr : arrays) {
            total_length += arr.size();
        }

        std::vector<uint64_t> res(total_length);
        size_t offset = 0;
        for (const auto& arr : arrays) {
            std::memcpy(res.data() + offset, arr.data(), arr.size() * sizeof(uint64_t));
            offset += arr.size();
        }
        return res;
    }

    // ------------------------------------------------------------------
    // 多路归并逻辑
    // ------------------------------------------------------------------

    std::vector<uint64_t> _merge_deduplicate_all(std::vector<ArraySegment>& segments) {
        size_t max_possible_length = 0;
        for (const auto& seg : segments) {
            max_possible_length += seg.length;
        }

        std::vector<uint64_t> merged_array;
        merged_array.reserve(max_possible_length);

        uint64_t last_added = 0; // 仅作为参考，实际依赖 merged_array.empty()
        
        while (true) {
            uint64_t current_min = 0;
            int min_index = -1;

            // 寻找当前可用元素中的最小值
            for (size_t i = 0; i < segments.size(); ++i) {
                if (segments[i].index < segments[i].length) {
                    uint64_t val = segments[i].data[segments[i].index];
                    // 修复 Python 中的 Sentinel Bug，使用 min_index 标记是否为首次赋值
                    if (min_index == -1 || val < current_min) {
                        current_min = val;
                        min_index = (int)i;
                    }
                }
            }

            if (min_index == -1) break; // 所有片段处理完毕

            // 去重判断
            if (merged_array.empty() || current_min != last_added) {
                merged_array.push_back(current_min);
                last_added = current_min;
            }

            segments[min_index].index++;
        }

        // 释放多余预留的内存
        merged_array.shrink_to_fit(); 
        return merged_array;
    }

    std::vector<std::vector<uint64_t>> merge_deduplicate_all(
        const std::vector<std::vector<uint64_t>>& arrays,
        const std::vector<uint64_t>& pivots,
        int n_threads
    ) {
        size_t num_arrays = arrays.size();
        
        // 寻找每个枢轴(pivot)的切割位置
        std::vector<std::vector<size_t>> split_positions(num_arrays, std::vector<size_t>(n_threads + 1, 0));

        for (size_t a = 0; a < num_arrays; ++a) {
            for (int t = 0; t < n_threads - 1; ++t) {
                // 等价于 Python 的 binary_search
                auto it = std::lower_bound(arrays[a].begin(), arrays[a].end(), pivots[t]);
                split_positions[a][t + 1] = std::distance(arrays[a].begin(), it);
            }
            split_positions[a][n_threads] = arrays[a].size();
        }

        std::vector<std::vector<uint64_t>> res(n_threads);

        // 并行归并每个分区
        #pragma omp parallel for num_threads(n_threads)
        for (int t = 0; t < n_threads; ++t) {
            std::vector<ArraySegment> temp_segments(num_arrays);
            for (size_t a = 0; a < num_arrays; ++a) {
                size_t s = split_positions[a][t];
                size_t e = split_positions[a][t + 1];
                temp_segments[a] = {arrays[a].data() + s, e - s, 0};
            }
            res[t] = _merge_deduplicate_all(temp_segments);
        }

        return res;
    }

    std::vector<uint64_t> merge_and_deduplicate(const std::vector<uint64_t>& arr1, const std::vector<uint64_t>& arr2) {
        std::vector<uint64_t> result;
        result.reserve(arr1.size() + arr2.size());
        
        // C++ 标准库直接支持此操作
        std::set_union(
            arr1.begin(), arr1.end(),
            arr2.begin(), arr2.end(),
            std::back_inserter(result)
        );
        
        result.shrink_to_fit();
        return result;
    }

} // namespace BookGeneratorUtils