#include "BookGeneratorUtils.h"
#include "HybridSearch.h"
#include "NativeDiagnostics.h"
#include "UniqueUtils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <numeric>
#include <omp.h>
#include <string>
#include <thread>
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
    using KeyValueSortUint64Uint32Fn = void (*)(uint64_t *, uint32_t *, size_t, bool);
    using MergeTwoPartitionedFn = size_t (*)(const uint64_t *, size_t, const uint64_t *, size_t, const uint64_t *, size_t, uint64_t *, size_t *, size_t *);
    using MergeTreePartitionedFn = size_t (*)(const uint64_t *const *, const size_t *, size_t, const uint64_t *, size_t, uint64_t *, size_t *, size_t *, uint64_t *);

    std::vector<fs::path> bookgen_dll_candidates() {
        std::vector<fs::path> candidates;
        auto append_unique = [&candidates](const fs::path &candidate) {
            if (candidate.empty()) {
                return;
            }
            if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
                candidates.push_back(candidate);
            }
        };

#ifdef _WIN32
        HMODULE module = nullptr;
        if (GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(&bookgen_dll_candidates),
                &module) &&
            module != nullptr) {
            char module_path[MAX_PATH];
            DWORD module_len = GetModuleFileNameA(module, module_path, MAX_PATH);
            if (module_len > 0) {
                fs::path module_dir = fs::path(module_path).parent_path();
                append_unique(module_dir / "bookgen_native.dll");
                append_unique(module_dir / "native_core" / "bookgen_native.dll");
            }
        }

        char exe_path[MAX_PATH];
        DWORD exe_len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
        if (exe_len > 0) {
            fs::path exe_dir = fs::path(exe_path).parent_path();
            append_unique(exe_dir / "native_core" / "bookgen_native.dll");
            append_unique(exe_dir / "bookgen_native.dll");
        }

        append_unique(fs::path("native_core") / "bookgen_native.dll");
        append_unique("bookgen_native.dll");
#endif

        return candidates;
    }

    SortFn resolve_sort_uint64() {
        static SortFn fn = []() -> SortFn {
            NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 begin");
            try {
#ifdef _WIN32
                for (const auto &candidate : bookgen_dll_candidates()) {
                    const std::string candidate_string = candidate.string();
                    NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 candidate=" + candidate_string);
                    if (!fs::exists(candidate)) {
                        NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 missing path=" + candidate_string);
                        continue;
                    }
                    SetEnvironmentVariableA("XSS_DISABLE_AVX512", "1");
                    NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 forced XSS_DISABLE_AVX512=1");
                    NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 LoadLibrary begin path=" + candidate_string);
                    HMODULE lib = LoadLibraryA(candidate_string.c_str());
                    NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 LoadLibrary done path=" + candidate_string);
                    if (!lib) {
                        NativeDiagnostics::mark(
                            "BookGeneratorUtils.resolve_sort_uint64 LoadLibrary failed path=" + candidate_string +
                            " error=" + std::to_string(static_cast<unsigned long>(GetLastError()))
                        );
                        continue;
                    }
                    auto proc = reinterpret_cast<SortFn>(GetProcAddress(lib, "sort_uint64"));
                    if (proc) {
                        NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 found path=" + candidate_string);
                        return proc;
                    }
                    NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 symbol missing path=" + candidate_string);
                }
                return nullptr;
#else
                void *lib = dlopen("bookgen_native.so", RTLD_LAZY);
                if (!lib) {
                    lib = dlopen("native_core/bookgen_native.so", RTLD_LAZY);
                }
                return lib ? reinterpret_cast<SortFn>(dlsym(lib, "sort_uint64")) : nullptr;
#endif
            } catch (const std::exception &ex) {
                NativeDiagnostics::mark(
                    std::string("BookGeneratorUtils.resolve_sort_uint64 exception; fallback what=") + ex.what()
                );
                return nullptr;
            } catch (...) {
                NativeDiagnostics::mark("BookGeneratorUtils.resolve_sort_uint64 unknown exception; fallback");
                return nullptr;
            }
        }();
        return fn;
    }

    template <typename Fn> Fn resolve_bookgen_symbol(const char *symbol_name) {
        static_assert(std::is_pointer_v<Fn>, "Fn must be a function pointer");
        try {
#ifdef _WIN32
            for (const auto &candidate : bookgen_dll_candidates()) {
                const std::string candidate_string = candidate.string();
                NativeDiagnostics::mark(std::string("BookGeneratorUtils.resolve ") + symbol_name + " candidate=" + candidate_string);
                if (!fs::exists(candidate)) {
                    continue;
                }
                NativeDiagnostics::mark(std::string("BookGeneratorUtils.resolve ") + symbol_name + " LoadLibrary begin path=" + candidate_string);
                HMODULE lib = LoadLibraryA(candidate_string.c_str());
                NativeDiagnostics::mark(std::string("BookGeneratorUtils.resolve ") + symbol_name + " LoadLibrary done path=" + candidate_string);
                if (!lib) {
                    NativeDiagnostics::mark(
                        std::string("BookGeneratorUtils.resolve ") + symbol_name + " LoadLibrary failed path=" +
                        candidate_string + " error=" +
                        std::to_string(static_cast<unsigned long>(GetLastError()))
                    );
                    continue;
                }
                auto proc = reinterpret_cast<Fn>(GetProcAddress(lib, symbol_name));
                if (proc) {
                    NativeDiagnostics::mark(std::string("BookGeneratorUtils.resolve ") + symbol_name + " found path=" + candidate_string);
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
        } catch (const std::exception &ex) {
            NativeDiagnostics::mark(
                std::string("BookGeneratorUtils.resolve ") + symbol_name + " exception; fallback what=" + ex.what()
            );
            return nullptr;
        } catch (...) {
            NativeDiagnostics::mark(std::string("BookGeneratorUtils.resolve ") + symbol_name + " unknown exception; fallback");
            return nullptr;
        }
    }

    MergeTwoPartitionedFn resolve_merge_two_partitioned_u64_dedup() {
        static MergeTwoPartitionedFn fn = resolve_bookgen_symbol<MergeTwoPartitionedFn>("merge_two_u64_partitioned_dedup");
        return fn;
    }

    MergeTreePartitionedFn resolve_merge_tree_partitioned_u64_dedup() {
        static MergeTreePartitionedFn fn = resolve_bookgen_symbol<MergeTreePartitionedFn>("merge_tree_partitioned_u64_dedup");
        return fn;
    }

    KeyValueSortUint64Uint32Fn resolve_keyvalue_sort_uint64_uint32() {
        static KeyValueSortUint64Uint32Fn fn =
            resolve_bookgen_symbol<KeyValueSortUint64Uint32Fn>("keyvalue_sort_uint64_uint32");
        return fn;
    }

    } // namespace

    // ------------------------------------------------------------------
    // 排序与检查
    // ------------------------------------------------------------------

    void sort_array(uint64_t* arr, size_t length, int num_threads) {
        (void) num_threads;
        if (length < 10000) {
            NativeDiagnostics::Scope scope("BookGeneratorUtils.sort_array fallback len=" + std::to_string(length));
            std::sort(arr, arr + length);
        } else {
            // 调用 DLL 导出的 C 接口
            if (auto fn = resolve_sort_uint64()) {
                NativeDiagnostics::Scope scope("BookGeneratorUtils.sort_array external len=" + std::to_string(length));
                fn(arr, length, false);
            } else {
                NativeDiagnostics::Scope scope("BookGeneratorUtils.sort_array fallback-no-dll len=" + std::to_string(length));
                std::sort(arr, arr + length);
            }
        }
    }

    void sort_keyvalue_uint64_uint32(
        uint64_t *keys,
        uint32_t *values,
        size_t length,
        bool descending
    ) {
        if (keys == nullptr || values == nullptr || length < 2) {
            return;
        }
        if (auto fn = resolve_keyvalue_sort_uint64_uint32(); fn != nullptr && length >= 10000) {
            NativeDiagnostics::Scope scope(
                "BookGeneratorUtils.sort_keyvalue external len=" + std::to_string(length) +
                " descending=" + (descending ? "1" : "0")
            );
            fn(keys, values, length, descending);
            return;
        }
        NativeDiagnostics::Scope scope(
            "BookGeneratorUtils.sort_keyvalue fallback len=" + std::to_string(length) +
            " descending=" + (descending ? "1" : "0")
        );
        std::vector<size_t> order(length);
        std::iota(order.begin(), order.end(), size_t {0});
        if (descending) {
            std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
                return keys[lhs] > keys[rhs];
            });
        } else {
            std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
                return keys[lhs] < keys[rhs];
            });
        }
        std::vector<uint64_t> sorted_keys(length);
        std::vector<uint32_t> sorted_values(length);
        for (size_t i = 0; i < length; ++i) {
            sorted_keys[i] = keys[order[i]];
            sorted_values[i] = values[order[i]];
        }
        std::memcpy(keys, sorted_keys.data(), length * sizeof(uint64_t));
        std::memcpy(values, sorted_values.data(), length * sizeof(uint32_t));
    }

    // ------------------------------------------------------------------
    // 原地紧缩去重与合并
    // ------------------------------------------------------------------

    size_t parallel_unique(uint64_t* arr, size_t length, int num_threads) {
        (void) num_threads;
        return UniqueUtils::unique_sorted_u64_inplace(arr, length);
    }

    std::pair<size_t, size_t> sort_and_unique_two_arrays_concurrently(
        uint64_t *arr1,
        size_t len1,
        uint64_t *arr2,
        size_t len2,
        int total_threads,
        int concurrent_threads_per_sort,
        size_t min_length
    ) {
        NativeDiagnostics::mark(
            "BookGeneratorUtils.sort_unique entry len1=" + std::to_string(len1) +
            " len2=" + std::to_string(len2) +
            " total_threads=" + std::to_string(total_threads) +
            " concurrent_threads_per_sort=" + std::to_string(concurrent_threads_per_sort) +
            " min_length=" + std::to_string(min_length)
        );

        auto sequential = [&]() -> std::pair<size_t, size_t> {
            NativeDiagnostics::mark(
                "BookGeneratorUtils.sort_unique sequential begin len1=" + std::to_string(len1) +
                " len2=" + std::to_string(len2)
            );
            sort_array(arr1, len1, total_threads);
            sort_array(arr2, len2, total_threads);
            auto result = std::pair<size_t, size_t>{
                parallel_unique(arr1, len1, total_threads),
                parallel_unique(arr2, len2, total_threads)
            };
            NativeDiagnostics::mark(
                "BookGeneratorUtils.sort_unique sequential done unique1=" + std::to_string(result.first) +
                " unique2=" + std::to_string(result.second)
            );
            return result;
        };

        const char *enable_concurrent_sort = std::getenv("TABLEBASE_ENABLE_CONCURRENT_SORT");
        if (enable_concurrent_sort == nullptr || std::strcmp(enable_concurrent_sort, "1") != 0) {
            NativeDiagnostics::mark("BookGeneratorUtils.sort_unique concurrent disabled; using sequential");
            return sequential();
        }

        if (concurrent_threads_per_sort <= 0 ||
            total_threads < concurrent_threads_per_sort * 2 ||
            len1 == 0 ||
            len2 == 0 ||
            len1 < min_length ||
            len2 < min_length) {
            return sequential();
        }

        size_t unique1 = len1;
        size_t unique2 = len2;
        std::exception_ptr error1;
        std::exception_ptr error2;

        NativeDiagnostics::mark(
            "BookGeneratorUtils.sort_unique concurrent begin len1=" + std::to_string(len1) +
            " len2=" + std::to_string(len2)
        );

        auto worker = [concurrent_threads_per_sort](uint64_t *arr, size_t len, size_t &unique_len, std::exception_ptr &error) {
            try {
                omp_set_dynamic(0);
                omp_set_num_threads(concurrent_threads_per_sort);
                sort_array(arr, len, concurrent_threads_per_sort);
                unique_len = parallel_unique(arr, len, concurrent_threads_per_sort);
            } catch (...) {
                error = std::current_exception();
            }
        };

        std::thread t1;
        std::thread t2;
        try {
            t1 = std::thread(worker, arr1, len1, std::ref(unique1), std::ref(error1));
            t2 = std::thread(worker, arr2, len2, std::ref(unique2), std::ref(error2));
            t1.join();
            t2.join();
        } catch (const std::exception &ex) {
            NativeDiagnostics::mark(
                "BookGeneratorUtils.concurrent_sort thread setup failed; fallback sequential len1=" +
                std::to_string(len1) +
                " len2=" + std::to_string(len2) +
                " what=" + ex.what()
            );
            if (t1.joinable()) {
                t1.join();
            }
            if (t2.joinable()) {
                t2.join();
            }
            return sequential();
        } catch (...) {
            NativeDiagnostics::mark(
                "BookGeneratorUtils.concurrent_sort thread setup failed; fallback sequential len1=" +
                std::to_string(len1) +
                " len2=" + std::to_string(len2)
            );
            if (t1.joinable()) {
                t1.join();
            }
            if (t2.joinable()) {
                t2.join();
            }
            return sequential();
        }

        if (error1) {
            try {
                std::rethrow_exception(error1);
            } catch (const std::exception &ex) {
                NativeDiagnostics::mark(
                    "BookGeneratorUtils.concurrent_sort worker1 failed; fallback sequential len=" +
                    std::to_string(len1) +
                    " what=" + ex.what()
                );
                return sequential();
            } catch (...) {
                NativeDiagnostics::mark(
                    "BookGeneratorUtils.concurrent_sort worker1 failed; fallback sequential len=" +
                    std::to_string(len1)
                );
                return sequential();
            }
        }
        if (error2) {
            try {
                std::rethrow_exception(error2);
            } catch (const std::exception &ex) {
                NativeDiagnostics::mark(
                    "BookGeneratorUtils.concurrent_sort worker2 failed; fallback sequential len=" +
                    std::to_string(len2) +
                    " what=" + ex.what()
                );
                return sequential();
            } catch (...) {
                NativeDiagnostics::mark(
                    "BookGeneratorUtils.concurrent_sort worker2 failed; fallback sequential len=" +
                    std::to_string(len2)
                );
                return sequential();
            }
        }

        return {unique1, unique2};
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
                NativeDiagnostics::Scope scope(
                    "BookGeneratorUtils.merge_two external total=" + std::to_string(total_length) +
                    " threads=" + std::to_string(n_threads)
                );
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
            NativeDiagnostics::Scope scope(
                "BookGeneratorUtils.merge_tree external arrays=" + std::to_string(arrays.size()) +
                " total=" + std::to_string(total_length) +
                " threads=" + std::to_string(n_threads)
            );
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
