#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "HybridSearch.h"
#include "UniqueUtils.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#include "../x86simdsort/x86-simd-sort/src/xss-common-includes.h"
#include "../x86simdsort/x86-simd-sort/src/xss-common-qsort.h"
#include "../x86simdsort/x86-simd-sort/src/avx512-64bit-common.h"
#pragma GCC pop_options
#endif

#if defined(_WIN32)
#define MERGE_TREE_EXPORT extern "C" __declspec(dllexport)
#else
#define MERGE_TREE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Segment {
    const uint64_t *data = nullptr;
    size_t length = 0;
};

struct MergeTask {
    size_t partition = 0;
    size_t pair = 0;
    size_t offset = 0;
};

inline void emit_unique(uint64_t value,
                        uint64_t *output,
                        size_t &out,
                        bool &has_last,
                        uint64_t &last) {
    if (!has_last || value != last) {
        output[out++] = value;
        last = value;
        has_last = true;
    }
}

inline void emit_unique_block(const uint64_t *block,
                              size_t count,
                              uint64_t *output,
                              size_t &out,
                              bool &has_last,
                              uint64_t &last) {
    for (size_t lane = 0; lane < count; ++lane) {
        emit_unique(block[lane], output, out, has_last, last);
    }
}

inline size_t merge_two_scalar_dedup(const uint64_t *left,
                                     size_t left_len,
                                     const uint64_t *right,
                                     size_t right_len,
                                     uint64_t *output) {
    size_t i = 0;
    size_t j = 0;
    size_t out = 0;
    bool has_last = false;
    uint64_t last = 0;

    while (i < left_len && j < right_len) {
        uint64_t value = 0;
        if (left[i] < right[j]) {
            value = left[i++];
        } else if (left[i] > right[j]) {
            value = right[j++];
        } else {
            value = left[i];
            ++i;
            ++j;
        }
        emit_unique(value, output, out, has_last, last);
    }

    while (i < left_len) {
        emit_unique(left[i++], output, out, has_last, last);
    }
    while (j < right_len) {
        emit_unique(right[j++], output, out, has_last, last);
    }

    return out;
}

#if defined(__GNUC__) || defined(__clang__)

using U64Vec = zmm_vector<uint64_t>;
using U64Comparator = Comparator<U64Vec, false>;
using U64Reg = U64Vec::reg_t;

__attribute__((target("avx512f")))
inline U64Reg sort_block_prefix_8x8(U64Reg left_vec, U64Reg right_vec) {
    U64Reg regs[2];
    regs[0] = left_vec;
    regs[1] = U64Vec::reverse(right_vec);
    sort_vectors<U64Vec, U64Comparator, 2>(regs);
    return regs[0];
}

__attribute__((target("avx512f")))
inline U64Reg previous_lanes_8x8(U64Reg value) {
    const __m512i previous_index = U64Vec::seti(6, 5, 4, 3, 2, 1, 0, 0);
    return U64Vec::permutexvar(previous_index, value);
}

__attribute__((target("avx512f")))
inline U64Reg broadcast_last_lane_8x8(U64Reg value) {
    const __m512i last_index = U64Vec::seti(7, 7, 7, 7, 7, 7, 7, 7);
    return U64Vec::permutexvar(last_index, value);
}

__attribute__((target("avx512f")))
inline void count_prefix_consumed_8x8(U64Reg left_vec,
                                      U64Reg right_vec,
                                      U64Reg sorted_low,
                                      size_t &consume_left,
                                      size_t &consume_right) {
    const U64Reg pivot = broadcast_last_lane_8x8(sorted_low);

    const __mmask8 left_lt = _mm512_cmp_epu64_mask(left_vec, pivot, _MM_CMPINT_LT);
    const __mmask8 right_lt = _mm512_cmp_epu64_mask(right_vec, pivot, _MM_CMPINT_LT);
    const __mmask8 left_eq = _mm512_cmp_epu64_mask(left_vec, pivot, _MM_CMPINT_EQ);
    const __mmask8 right_eq = _mm512_cmp_epu64_mask(right_vec, pivot, _MM_CMPINT_EQ);

    const size_t less_left = UniqueUtils::popcount_mask(static_cast<unsigned>(left_lt));
    const size_t less_right = UniqueUtils::popcount_mask(static_cast<unsigned>(right_lt));
    const size_t need_equal = 8 - less_left - less_right;
    const size_t equal_left = UniqueUtils::popcount_mask(static_cast<unsigned>(left_eq));

    const size_t take_left_equal = std::min(equal_left, need_equal);
    consume_left = less_left + take_left_equal;
    consume_right = 8 - consume_left;
}

__attribute__((target("avx512f")))
inline void emit_unique_sorted_vec(U64Reg sorted_low,
                                   uint64_t *output,
                                   size_t &out,
                                   bool &has_last,
                                   uint64_t &last) {
    U64Reg previous = previous_lanes_8x8(sorted_low);
    if (has_last) {
        previous = _mm512_mask_mov_epi64(previous, 0x01, U64Vec::set1(last));
    }

    __mmask8 keep_mask = _mm512_cmp_epu64_mask(sorted_low, previous, _MM_CMPINT_NE);
    if (!has_last) {
        keep_mask = static_cast<__mmask8>(keep_mask | 0x01);
    }

    U64Vec::mask_compressstoreu(output + out, keep_mask, sorted_low);
    out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
    has_last = true;
    last = U64Vec::reducemax(sorted_low);
}

__attribute__((target("avx512f")))
size_t merge_two_avx512_dedup(const uint64_t *left,
                              size_t left_len,
                              const uint64_t *right,
                              size_t right_len,
                              uint64_t *output) {
    size_t i = 0;
    size_t j = 0;
    size_t out = 0;
    bool has_last = false;
    uint64_t last = 0;

    while (i + 8 <= left_len && j + 8 <= right_len) {
        const U64Reg left_vec = U64Vec::loadu(left + i);
        const U64Reg right_vec = U64Vec::loadu(right + j);
        const U64Reg sorted_low = sort_block_prefix_8x8(left_vec, right_vec);

        size_t consume_left = 0;
        size_t consume_right = 0;
        count_prefix_consumed_8x8(left_vec, right_vec, sorted_low, consume_left, consume_right);

        if (consume_left + consume_right != 8) {
            break;
        }

        emit_unique_sorted_vec(sorted_low, output, out, has_last, last);
        i += consume_left;
        j += consume_right;
    }

    while (i < left_len && j < right_len) {
        uint64_t value = 0;
        if (left[i] < right[j]) {
            value = left[i++];
        } else if (left[i] > right[j]) {
            value = right[j++];
        } else {
            value = left[i];
            ++i;
            ++j;
        }
        emit_unique(value, output, out, has_last, last);
    }

    while (i < left_len) {
        emit_unique(left[i++], output, out, has_last, last);
    }
    while (j < right_len) {
        emit_unique(right[j++], output, out, has_last, last);
    }

    return out;
}

#endif

inline size_t merge_two_serial_dedup(const uint64_t *left,
                                     size_t left_len,
                                     const uint64_t *right,
                                     size_t right_len,
                                     uint64_t *output,
                                     bool use_avx512) {
#if defined(__GNUC__) || defined(__clang__)
    if (use_avx512 && left_len >= 8 && right_len >= 8) {
        return merge_two_avx512_dedup(left, left_len, right, right_len, output);
    }
#else
    (void)use_avx512;
#endif
    return merge_two_scalar_dedup(left, left_len, right, right_len, output);
}

inline size_t merge_path_partition(const uint64_t *left,
                                   size_t left_len,
                                   const uint64_t *right,
                                   size_t right_len,
                                   size_t diag) {
    size_t low = (diag > right_len) ? (diag - right_len) : 0;
    size_t high = std::min(diag, left_len);

    while (low <= high) {
        const size_t left_mid = low + (high - low) / 2;
        const size_t right_mid = diag - left_mid;

        if (left_mid > 0 && right_mid < right_len && left[left_mid - 1] > right[right_mid]) {
            high = left_mid - 1;
            continue;
        }
        if (right_mid > 0 && left_mid < left_len && right[right_mid - 1] >= left[left_mid]) {
            low = left_mid + 1;
            continue;
        }
        return left_mid;
    }

    return low;
}

inline size_t merge_threads_for_length(size_t total_len) {
    constexpr size_t activation_threshold = 1u << 19;
    constexpr size_t min_chunk = 1u << 17;
    constexpr size_t max_chunk = 1u << 18;

#if defined(_OPENMP)
    if (omp_in_parallel()) {
        return 1;
    }
    const int max_threads = std::max(omp_get_max_threads(), 1);
#else
    const int max_threads = 1;
#endif

    if (total_len < activation_threshold) {
        return 1;
    }

    const size_t target_chunk = std::min(max_chunk, std::max(min_chunk, total_len / 2));
    const size_t by_length = (total_len + target_chunk - 1) / target_chunk;
    return std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(max_threads), by_length));
}

inline size_t compact_partitioned_merge_output(uint64_t *output,
                                               const std::vector<size_t> &diagonals,
                                               const std::vector<size_t> &partition_lengths) {
    size_t result_end = 0;

    for (size_t part = 0; part < partition_lengths.size(); ++part) {
        size_t part_length = partition_lengths[part];
        if (part_length == 0) {
            continue;
        }

        uint64_t *src = output + diagonals[part];
        if (result_end > 0 && output[result_end - 1] == src[0]) {
            ++src;
            --part_length;
        }

        if (part_length == 0) {
            continue;
        }

        if (src != output + result_end) {
            std::memmove(output + result_end, src, part_length * sizeof(uint64_t));
        }
        result_end += part_length;
    }

    return result_end;
}

inline size_t compact_partitioned_output_inplace(uint64_t *output,
                                                 size_t *partition_offsets,
                                                 size_t *partition_lengths,
                                                 size_t partition_count) {
    size_t write_offset = 0;
    for (size_t part = 0; part < partition_count; ++part) {
        const size_t read_offset = partition_offsets[part];
        const size_t part_length = partition_lengths[part];
        partition_offsets[part] = write_offset;
        if (part_length > 0 && read_offset != write_offset) {
            std::memmove(output + write_offset,
                         output + read_offset,
                         part_length * sizeof(uint64_t));
        }
        write_offset += part_length;
    }
    partition_offsets[partition_count] = write_offset;
    return write_offset;
}

inline size_t merge_two_dedup(const uint64_t *left,
                              size_t left_len,
                              const uint64_t *right,
                              size_t right_len,
                              uint64_t *output,
                              bool use_avx512);

inline bool can_spawn_parallel_region();

inline void fill_partition_cuts(const uint64_t *data,
                                size_t length,
                                const uint64_t *pivots,
                                size_t partition_count,
                                size_t *cuts) {
    std::fill(cuts, cuts + partition_count + 1, 0);
    if (data == nullptr || length == 0) {
        return;
    }

    const uint64_t *search_begin = data;
    for (size_t part = 1; part < partition_count; ++part) {
        const size_t offset = static_cast<size_t>(search_begin - data);
        const size_t remaining = length - offset;
        const size_t cut = HybridSearch::lower_bound(search_begin, remaining, pivots[part - 1]);
        const uint64_t *it = search_begin + cut;
        cuts[part] = static_cast<size_t>(it - data);
        search_begin = it;
    }
    cuts[partition_count] = length;
}

inline size_t merge_two_partitioned_direct_dedup(const uint64_t *left,
                                                 size_t left_len,
                                                 const uint64_t *right,
                                                 size_t right_len,
                                                 const uint64_t *pivots,
                                                 size_t partition_count,
                                                 uint64_t *output,
                                                 size_t *partition_offsets,
                                                 size_t *partition_lengths,
                                                 bool use_avx512) {
    if (partition_count == 0) {
        return 0;
    }
    if (partition_count == 1) {
        partition_offsets[0] = 0;
        partition_offsets[1] = left_len + right_len;
        partition_lengths[0] =
            merge_two_dedup(left, left_len, right, right_len, output, use_avx512);
        return partition_lengths[0];
    }

    constexpr size_t stack_partition_limit = 64;
    std::array<size_t, stack_partition_limit + 1> left_cuts_stack{};
    std::array<size_t, stack_partition_limit + 1> right_cuts_stack{};
    std::vector<size_t> left_cuts_heap;
    std::vector<size_t> right_cuts_heap;

    size_t *left_cuts = nullptr;
    size_t *right_cuts = nullptr;
    if (partition_count <= stack_partition_limit) {
        left_cuts = left_cuts_stack.data();
        right_cuts = right_cuts_stack.data();
    } else {
        left_cuts_heap.resize(partition_count + 1);
        right_cuts_heap.resize(partition_count + 1);
        left_cuts = left_cuts_heap.data();
        right_cuts = right_cuts_heap.data();
    }

    fill_partition_cuts(left, left_len, pivots, partition_count, left_cuts);
    fill_partition_cuts(right, right_len, pivots, partition_count, right_cuts);

    partition_offsets[0] = 0;
    for (size_t part = 0; part < partition_count; ++part) {
        const size_t reserved =
            (left_cuts[part + 1] - left_cuts[part]) + (right_cuts[part + 1] - right_cuts[part]);
        partition_offsets[part + 1] = partition_offsets[part] + reserved;
    }

#if defined(_OPENMP)
#pragma omp parallel for if(partition_count > 1 && can_spawn_parallel_region())
#endif
    for (size_t part = 0; part < partition_count; ++part) {
        const size_t left_start = left_cuts[part];
        const size_t left_stop = left_cuts[part + 1];
        const size_t right_start = right_cuts[part];
        const size_t right_stop = right_cuts[part + 1];
        partition_lengths[part] = merge_two_serial_dedup(left + left_start,
                                                         left_stop - left_start,
                                                         right + right_start,
                                                         right_stop - right_start,
                                                         output + partition_offsets[part],
                                                         use_avx512);
    }

    return compact_partitioned_output_inplace(
        output, partition_offsets, partition_lengths, partition_count
    );
}

inline size_t merge_two_partitioned_dedup(const uint64_t *left,
                                          size_t left_len,
                                          const uint64_t *right,
                                          size_t right_len,
                                          uint64_t *output,
                                          bool use_avx512) {
    const size_t total_len = left_len + right_len;
    const size_t worker_count = merge_threads_for_length(total_len);
    if (worker_count <= 1 || left_len == 0 || right_len == 0) {
        return merge_two_serial_dedup(left, left_len, right, right_len, output, use_avx512);
    }

    std::vector<size_t> diagonals(worker_count + 1, 0);
    std::vector<size_t> left_cuts(worker_count + 1, 0);
    std::vector<size_t> right_cuts(worker_count + 1, 0);
    std::vector<size_t> partition_lengths(worker_count, 0);

    for (size_t part = 0; part <= worker_count; ++part) {
        const size_t diag = (total_len * part) / worker_count;
        const size_t left_cut = merge_path_partition(left, left_len, right, right_len, diag);
        diagonals[part] = diag;
        left_cuts[part] = left_cut;
        right_cuts[part] = diag - left_cut;
    }

#if defined(_OPENMP)
#pragma omp parallel for if(worker_count > 1)
#endif
    for (size_t part = 0; part < worker_count; ++part) {
        const size_t left_start = left_cuts[part];
        const size_t left_stop = left_cuts[part + 1];
        const size_t right_start = right_cuts[part];
        const size_t right_stop = right_cuts[part + 1];
        partition_lengths[part] = merge_two_serial_dedup(left + left_start,
                                                          left_stop - left_start,
                                                          right + right_start,
                                                          right_stop - right_start,
                                                          output + diagonals[part],
                                                          use_avx512);
    }

    return compact_partitioned_merge_output(output, diagonals, partition_lengths);
}

inline size_t merge_two_dedup(const uint64_t *left,
                              size_t left_len,
                              const uint64_t *right,
                              size_t right_len,
                              uint64_t *output,
                              bool use_avx512) {
    return merge_two_partitioned_dedup(left, left_len, right, right_len, output, use_avx512);
}

inline bool can_spawn_parallel_region() {
#if defined(_OPENMP)
    return !omp_in_parallel();
#else
    return false;
#endif
}

inline std::vector<Segment> make_segments(const uint64_t *const *inputs,
                                          const size_t *lengths,
                                          size_t count) {
    std::vector<Segment> segments(count);
    for (size_t i = 0; i < count; ++i) {
        segments[i].data = inputs[i];
        segments[i].length = lengths[i];
    }
    return segments;
}

inline size_t merge_tree_segments_dedup_impl(const std::vector<Segment> &segments,
                                             uint64_t *output,
                                             uint64_t *scratch,
                                             bool use_avx512) {
    if (output == nullptr || scratch == nullptr) {
        return 0;
    }
    if (segments.empty()) {
        return 0;
    }
    if (segments.size() == 1) {
        if (segments[0].length == 0 || segments[0].data == nullptr) {
            return 0;
        }
        std::memmove(output, segments[0].data, segments[0].length * sizeof(uint64_t));
        return segments[0].length;
    }
    if (segments.size() == 2) {
        return merge_two_dedup(segments[0].data,
                               segments[0].length,
                               segments[1].data,
                               segments[1].length,
                               output,
                               use_avx512);
    }

    std::vector<Segment> current = segments;
    std::vector<Segment> next((current.size() + 1) / 2);
    uint64_t *dest = output;
    uint64_t *alternate = scratch;

    while (current.size() > 1) {
        const size_t pair_count = current.size() / 2;
        std::vector<size_t> pair_offsets(pair_count + 1, 0);

        for (size_t pair = 0; pair < pair_count; ++pair) {
            pair_offsets[pair + 1] =
                pair_offsets[pair] + current[pair * 2].length + current[pair * 2 + 1].length;
        }

#if defined(_OPENMP)
#pragma omp parallel for if(pair_count > 1 && can_spawn_parallel_region())
#endif
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const Segment &left = current[pair * 2];
            const Segment &right = current[pair * 2 + 1];
            const size_t pair_offset = pair_offsets[pair];
            const size_t merged_length = merge_two_dedup(left.data,
                                                         left.length,
                                                         right.data,
                                                         right.length,
                                                         dest + pair_offset,
                                                         use_avx512);
            next[pair].data = dest + pair_offset;
            next[pair].length = merged_length;
        }

        if ((current.size() & 1u) != 0u) {
            next[pair_count] = current.back();
            next.resize(pair_count + 1);
        } else {
            next.resize(pair_count);
        }

        current.swap(next);
        next.resize((current.size() + 1) / 2);
        std::swap(dest, alternate);
    }

    if (current[0].data != output && current[0].length > 0) {
        std::memmove(output, current[0].data, current[0].length * sizeof(uint64_t));
    }

    return current[0].length;
}

inline size_t merge_tree_segments_dedup_impl(const std::vector<Segment> &segments,
                                             uint64_t *output,
                                             uint64_t *scratch) {
    return merge_tree_segments_dedup_impl(
        segments, output, scratch, UniqueUtils::cpu_has_avx512()
    );
}

inline size_t merge_tree_partitioned_segments_dedup_impl(
    const std::vector<std::vector<Segment>> &segments_by_partition,
    const std::vector<size_t> &partition_offsets,
    uint64_t *output,
    uint64_t *scratch,
    size_t *partition_lengths,
    bool use_avx512) {
    if (output == nullptr || scratch == nullptr || partition_lengths == nullptr) {
        return 0;
    }
    if (segments_by_partition.empty()) {
        return 0;
    }

    std::vector<std::vector<Segment>> current = segments_by_partition;
    std::vector<std::vector<Segment>> next(current.size());
    uint64_t *dest = output;
    uint64_t *alternate = scratch;

    bool needs_merge = false;
    for (const auto &segments : current) {
        if (segments.size() > 1) {
            needs_merge = true;
            break;
        }
    }

    while (needs_merge) {
        std::vector<MergeTask> tasks;
        tasks.reserve(current.size() * 4);

        for (size_t part = 0; part < current.size(); ++part) {
            const size_t pair_count = current[part].size() / 2;
            next[part].resize((current[part].size() + 1) / 2);

            size_t offset = 0;
            for (size_t pair = 0; pair < pair_count; ++pair) {
                tasks.push_back(MergeTask{part, pair, offset});
                offset += current[part][pair * 2].length + current[part][pair * 2 + 1].length;
            }
            if ((current[part].size() & 1u) != 0u) {
                next[part][pair_count] = current[part].back();
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for if(tasks.size() > 1 && can_spawn_parallel_region())
#endif
        for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
            const MergeTask task = tasks[task_index];
            const Segment &left = current[task.partition][task.pair * 2];
            const Segment &right = current[task.partition][task.pair * 2 + 1];
            const size_t merged_length = merge_two_dedup(left.data,
                                                         left.length,
                                                         right.data,
                                                         right.length,
                                                         dest + partition_offsets[task.partition] + task.offset,
                                                         use_avx512);
            next[task.partition][task.pair].data =
                dest + partition_offsets[task.partition] + task.offset;
            next[task.partition][task.pair].length = merged_length;
        }

        current.swap(next);
        for (size_t part = 0; part < next.size(); ++part) {
            next[part].clear();
        }
        std::swap(dest, alternate);

        needs_merge = false;
        for (const auto &segments : current) {
            if (segments.size() > 1) {
                needs_merge = true;
                break;
            }
        }
    }

    size_t total_length = 0;
    for (size_t part = 0; part < current.size(); ++part) {
        size_t length = 0;
        if (!current[part].empty() && current[part][0].data != nullptr) {
            length = current[part][0].length;
            const uint64_t *src = current[part][0].data;
            uint64_t *dst = output + partition_offsets[part];
            if (src != dst && length > 0) {
                std::memmove(dst, src, length * sizeof(uint64_t));
            }
        }
        partition_lengths[part] = length;
        total_length += length;
    }

    return total_length;
}

inline size_t merge_tree_u64_dedup_impl(const uint64_t *const *inputs,
                                        const size_t *lengths,
                                        size_t count,
                                        uint64_t *output,
                                        uint64_t *scratch) {
    if (inputs == nullptr || lengths == nullptr) {
        return 0;
    }
    const bool use_avx512 = UniqueUtils::cpu_has_avx512();
    if (count == 2) {
        return merge_two_dedup(inputs[0], lengths[0], inputs[1], lengths[1], output, use_avx512);
    }
    return merge_tree_segments_dedup_impl(
        make_segments(inputs, lengths, count), output, scratch, use_avx512
    );
}

inline size_t merge_tree_partitioned_u64_dedup_impl(const uint64_t *const *inputs,
                                                    const size_t *lengths,
                                                    size_t count,
                                                    const uint64_t *pivots,
                                                    size_t partition_count,
                                                    uint64_t *output,
                                                    size_t *partition_offsets,
                                                    size_t *partition_lengths,
                                                    uint64_t *scratch) {
    if (inputs == nullptr || lengths == nullptr || output == nullptr || scratch == nullptr ||
        partition_offsets == nullptr || partition_lengths == nullptr || partition_count == 0) {
        return 0;
    }
    const bool use_avx512 = UniqueUtils::cpu_has_avx512();
    if (count == 2) {
        return merge_two_partitioned_direct_dedup(inputs[0],
                                                  lengths[0],
                                                  inputs[1],
                                                  lengths[1],
                                                  pivots,
                                                  partition_count,
                                                  output,
                                                  partition_offsets,
                                                  partition_lengths,
                                                  use_avx512);
    }

    const size_t positions_per_array = partition_count + 1;
    std::vector<size_t> split_positions(count * positions_per_array, 0);

    for (size_t a = 0; a < count; ++a) {
        const uint64_t *data = inputs[a];
        const size_t length = lengths[a];
        if (data == nullptr || length == 0) {
            split_positions[a * positions_per_array + partition_count] = 0;
            continue;
        }

        const uint64_t *search_begin = data;
        for (size_t t = 0; t + 1 < partition_count; ++t) {
            const size_t offset = static_cast<size_t>(search_begin - data);
            const size_t remaining = length - offset;
            const size_t cut = HybridSearch::lower_bound(search_begin, remaining, pivots[t]);
            const uint64_t *it = search_begin + cut;
            split_positions[a * positions_per_array + t + 1] = static_cast<size_t>(it - data);
            search_begin = it;
        }
        split_positions[a * positions_per_array + partition_count] = length;
    }

    partition_offsets[0] = 0;
    for (size_t t = 0; t < partition_count; ++t) {
        size_t reserved = 0;
        for (size_t a = 0; a < count; ++a) {
            const size_t base = a * positions_per_array;
            reserved += split_positions[base + t + 1] - split_positions[base + t];
        }
        partition_offsets[t + 1] = partition_offsets[t] + reserved;
    }
    std::vector<std::vector<Segment>> segments_by_partition(
        partition_count, std::vector<Segment>(count)
    );

    for (size_t t = 0; t < partition_count; ++t) {
        for (size_t a = 0; a < count; ++a) {
            const size_t base = a * positions_per_array;
            const size_t start = split_positions[base + t];
            const size_t stop = split_positions[base + t + 1];
            segments_by_partition[t][a].data = inputs[a] == nullptr ? nullptr : inputs[a] + start;
            segments_by_partition[t][a].length = stop - start;
        }
    }

    std::vector<size_t> partition_base_offsets(partition_count, 0);
    for (size_t t = 0; t < partition_count; ++t) {
        partition_base_offsets[t] = partition_offsets[t];
    }

    merge_tree_partitioned_segments_dedup_impl(
        segments_by_partition, partition_base_offsets, output, scratch, partition_lengths, use_avx512
    );
    return compact_partitioned_output_inplace(
        output, partition_offsets, partition_lengths, partition_count
    );
}

} // namespace

MERGE_TREE_EXPORT size_t merge_tree_u64_dedup(const uint64_t *const *inputs,
                                              const size_t *lengths,
                                              size_t count,
                                              uint64_t *output,
                                              uint64_t *scratch) {
    return merge_tree_u64_dedup_impl(inputs, lengths, count, output, scratch);
}

MERGE_TREE_EXPORT size_t merge_two_u64_dedup(const uint64_t *left,
                                             size_t left_len,
                                             const uint64_t *right,
                                             size_t right_len,
                                             uint64_t *output) {
    return merge_two_dedup(left, left_len, right, right_len, output, UniqueUtils::cpu_has_avx512());
}

MERGE_TREE_EXPORT size_t merge_two_u64_partitioned_dedup(const uint64_t *left,
                                                         size_t left_len,
                                                         const uint64_t *right,
                                                         size_t right_len,
                                                         const uint64_t *pivots,
                                                         size_t partition_count,
                                                         uint64_t *output,
                                                         size_t *partition_offsets,
                                                         size_t *partition_lengths) {
    return merge_two_partitioned_direct_dedup(left,
                                              left_len,
                                              right,
                                              right_len,
                                              pivots,
                                              partition_count,
                                              output,
                                              partition_offsets,
                                              partition_lengths,
                                              UniqueUtils::cpu_has_avx512());
}

MERGE_TREE_EXPORT int merge_tree_u64_dedup_has_avx512() {
    return UniqueUtils::cpu_has_avx512() ? 1 : 0;
}

MERGE_TREE_EXPORT size_t merge_tree_partitioned_u64_dedup(
    const uint64_t *const *inputs,
    const size_t *lengths,
    size_t count,
    const uint64_t *pivots,
    size_t partition_count,
    uint64_t *output,
    size_t *partition_offsets,
    size_t *partition_lengths,
    uint64_t *scratch) {
    return merge_tree_partitioned_u64_dedup_impl(
        inputs,
        lengths,
        count,
        pivots,
        partition_count,
        output,
        partition_offsets,
        partition_lengths,
        scratch
    );
}
