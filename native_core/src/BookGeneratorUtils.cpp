#include "BookGeneratorUtils.h"
#include "HybridSearch.h"
#include "UniqueUtils.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <omp.h>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace BookGeneratorUtils {

    namespace {

    using SortFn = void (*)(uint64_t *, size_t, bool);
    using MergeTwoPartitionedFn = size_t (*)(const uint64_t *, size_t, const uint64_t *, size_t, const uint64_t *, size_t, uint64_t *, size_t *, size_t *);
    using MergeTreePartitionedFn = size_t (*)(const uint64_t *const *, const size_t *, size_t, const uint64_t *, size_t, uint64_t *, size_t *, size_t *, uint64_t *);

    SortFn resolve_sort_uint64() {
        static SortFn fn = []() -> SortFn {
#ifdef _WIN32
            std::vector<fs::path> candidates;

            HMODULE module = nullptr;
            if (GetModuleHandleExA(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(&resolve_sort_uint64),
                    &module) &&
                module != nullptr) {
                char module_path[MAX_PATH];
                DWORD module_len = GetModuleFileNameA(module, module_path, MAX_PATH);
                if (module_len > 0) {
                    fs::path module_dir = fs::path(module_path).parent_path();
                    candidates.push_back(module_dir / "bookgen_native.dll");
                }
            }

            candidates.insert(candidates.end(), {
                "bookgen_native.dll",
                fs::path("native_core") / "bookgen_native.dll",
            });

            char module_path[MAX_PATH];
            DWORD path_len = GetModuleFileNameA(nullptr, module_path, MAX_PATH);
            if (path_len > 0) {
                fs::path exe_dir = fs::path(module_path).parent_path();
                candidates.push_back(exe_dir / "bookgen_native.dll");
                candidates.push_back(exe_dir / "native_core" / "bookgen_native.dll");
            }

            for (const auto &candidate : candidates) {
                if (!fs::exists(candidate)) {
                    continue;
                }
                HMODULE lib = LoadLibraryA(candidate.string().c_str());
                if (!lib) {
                    continue;
                }
                auto proc = reinterpret_cast<SortFn>(GetProcAddress(lib, "sort_uint64"));
                if (proc) {
                    return proc;
                }
            }
            return nullptr;
#else
            void *lib = dlopen("bookgen_native.so", RTLD_LAZY);
            if (!lib) {
                lib = dlopen("native_core/bookgen_native.so", RTLD_LAZY);
            }
            return lib ? reinterpret_cast<SortFn>(dlsym(lib, "sort_uint64")) : nullptr;
#endif
        }();
        return fn;
    }

    template <typename Fn> Fn resolve_bookgen_symbol(const char *symbol_name) {
        static_assert(std::is_pointer_v<Fn>, "Fn must be a function pointer");
#ifdef _WIN32
        std::vector<fs::path> candidates;

        HMODULE module = nullptr;
        if (GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(&resolve_sort_uint64),
                &module) &&
            module != nullptr) {
            char module_path[MAX_PATH];
            DWORD module_len = GetModuleFileNameA(module, module_path, MAX_PATH);
            if (module_len > 0) {
                fs::path module_dir = fs::path(module_path).parent_path();
                candidates.push_back(module_dir / "bookgen_native.dll");
            }
        }

        candidates.insert(candidates.end(), {
            "bookgen_native.dll",
            fs::path("native_core") / "bookgen_native.dll",
        });

        char module_path[MAX_PATH];
        DWORD path_len = GetModuleFileNameA(nullptr, module_path, MAX_PATH);
        if (path_len > 0) {
            fs::path exe_dir = fs::path(module_path).parent_path();
            candidates.push_back(exe_dir / "bookgen_native.dll");
            candidates.push_back(exe_dir / "native_core" / "bookgen_native.dll");
        }

        for (const auto &candidate : candidates) {
            if (!fs::exists(candidate)) {
                continue;
            }
            HMODULE lib = LoadLibraryA(candidate.string().c_str());
            if (!lib) {
                continue;
            }
            auto proc = reinterpret_cast<Fn>(GetProcAddress(lib, symbol_name));
            if (proc) {
                return proc;
            }
        }
        return nullptr;
#else
        void *lib = dlopen("bookgen_native.so", RTLD_LAZY);
        if (!lib) {
            lib = dlopen("native_core/bookgen_native.so", RTLD_LAZY);
        }
        return lib ? reinterpret_cast<Fn>(dlsym(lib, symbol_name)) : nullptr;
#endif
    }

    MergeTwoPartitionedFn resolve_merge_two_partitioned_u64_dedup() {
        static MergeTwoPartitionedFn fn = resolve_bookgen_symbol<MergeTwoPartitionedFn>("merge_two_u64_partitioned_dedup");
        return fn;
    }

    MergeTreePartitionedFn resolve_merge_tree_partitioned_u64_dedup() {
        static MergeTreePartitionedFn fn = resolve_bookgen_symbol<MergeTreePartitionedFn>("merge_tree_partitioned_u64_dedup");
        return fn;
    }

    } // namespace

    // ------------------------------------------------------------------
    // 排序与检查
    // ------------------------------------------------------------------

    void sort_array(uint64_t* arr, size_t length, int num_threads) {
        (void) num_threads;
        if (length < 10000) {
            std::sort(arr, arr + length);
        } else {
            // 调用 DLL 导出的 C 接口
            if (auto fn = resolve_sort_uint64()) {
                fn(arr, length, false);
            } else {
                std::sort(arr, arr + length);
            }
        }
    }

    // ------------------------------------------------------------------
    // 原地紧缩去重与合并
    // ------------------------------------------------------------------

    size_t parallel_unique(uint64_t* arr, size_t length, int num_threads) {
        (void) num_threads;
        return UniqueUtils::unique_sorted_u64_inplace(arr, length);
        if (length < 2) {
            return length;
        }

        const size_t worker_count = std::min<size_t>(length, static_cast<size_t>(std::max(num_threads, 1)));
        if (worker_count <= 1 || length < 131072) {
            return UniqueUtils::unique_sorted_u64_inplace(arr, length);
        }

        const size_t step = (length + worker_count - 1) / worker_count;
        std::vector<size_t> c_list(worker_count, 0);

        // 并行区间内去重
        #pragma omp parallel for num_threads(static_cast<int>(worker_count))
        for (int i = 0; i < static_cast<int>(worker_count); ++i) {
            const size_t start = static_cast<size_t>(i) * step;
            const size_t end = std::min(length, start + step);
            
            if (start >= end) {
                continue;
            }

            size_t c = UniqueUtils::unique_sorted_u64_inplace(arr + start, end - start);
            if (i > 0 && c > 0 && arr[start] == arr[start - 1]) {
                if (c > 1) {
                    std::memmove(arr + start, arr + start + 1, (c - 1) * sizeof(uint64_t));
                }
                --c;
            }

                // arr[j-1] 访问是安全的，因为后续的 memmove 在并行的这步之后执行
            c_list[static_cast<size_t>(i)] = c;
        }

        // 串行紧缩收集结果
        size_t result_cumulative = c_list[0];
        for (size_t i = 1; i < worker_count; ++i) {
            const size_t start = i * step;
            const size_t count = c_list[i];
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
        std::vector<uint64_t> sorted_pivots = pivots;
        std::sort(sorted_pivots.begin(), sorted_pivots.end());
        
        // 寻找每个枢轴(pivot)的切割位置
        std::vector<std::vector<size_t>> split_positions(num_arrays, std::vector<size_t>(n_threads + 1, 0));

        for (size_t a = 0; a < num_arrays; ++a) {
            for (int t = 0; t < n_threads - 1; ++t) {
                // 等价于 Python 的 binary_search
                split_positions[a][t + 1] = HybridSearch::lower_bound(
                    arrays[a].data(), arrays[a].size(), sorted_pivots[t]
                );
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
                if (e < s) {
                    e = s;
                }
                temp_segments[a] = {arrays[a].data() + s, e - s, 0};
            }
            res[t] = _merge_deduplicate_all(temp_segments);
        }

        return res;
    }

    std::vector<uint64_t> merge_deduplicate_all_concat(
        const std::vector<std::vector<uint64_t>>& arrays,
        const std::vector<uint64_t>& pivots,
        int n_threads
    ) {
        if (n_threads <= 0 || arrays.empty()) {
            return {};
        }

        size_t total_length = 0;
        for (const auto &array : arrays) {
            total_length += array.size();
        }

        if (arrays.size() == 2) {
            if (auto fn = resolve_merge_two_partitioned_u64_dedup()) {
                std::vector<uint64_t> result(total_length);
                std::vector<size_t> offsets(static_cast<size_t>(n_threads) + 1U, 0U);
                std::vector<size_t> sizes(static_cast<size_t>(n_threads), 0U);
                size_t merged = fn(
                    arrays[0].data(),
                    arrays[0].size(),
                    arrays[1].data(),
                    arrays[1].size(),
                    pivots.empty() ? nullptr : pivots.data(),
                    static_cast<size_t>(n_threads),
                    result.data(),
                    offsets.data(),
                    sizes.data()
                );
                result.resize(merged);
                return result;
            }
        } else if (auto fn = resolve_merge_tree_partitioned_u64_dedup()) {
            std::vector<const uint64_t *> ptrs(arrays.size(), nullptr);
            std::vector<size_t> lengths(arrays.size(), 0U);
            for (size_t i = 0; i < arrays.size(); ++i) {
                ptrs[i] = arrays[i].data();
                lengths[i] = arrays[i].size();
            }
            std::vector<uint64_t> result(total_length);
            std::vector<size_t> offsets(static_cast<size_t>(n_threads) + 1U, 0U);
            std::vector<size_t> sizes(static_cast<size_t>(n_threads), 0U);
            std::vector<uint64_t> scratch(total_length);
            size_t merged = fn(
                ptrs.data(),
                lengths.data(),
                arrays.size(),
                pivots.empty() ? nullptr : pivots.data(),
                static_cast<size_t>(n_threads),
                result.data(),
                offsets.data(),
                sizes.data(),
                scratch.data()
            );
            result.resize(merged);
            return result;
        }

        return concatenate(merge_deduplicate_all(arrays, pivots, n_threads));
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
