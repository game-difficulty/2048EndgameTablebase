#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace BookGeneratorUtils {

void sort_array(uint64_t *arr, size_t length, int num_threads = 1);

size_t parallel_unique(uint64_t *arr, size_t length, int num_threads);

std::pair<size_t, size_t> sort_and_unique_two_arrays_concurrently(
    uint64_t *arr1,
    size_t len1,
    uint64_t *arr2,
    size_t len2,
    int total_threads,
    int concurrent_threads_per_sort = 8,
    size_t min_length = 100000
);

size_t merge_inplace(uint64_t *arr, const std::vector<size_t> &segment_ends, const std::vector<size_t> &segment_starts);

std::vector<uint64_t> concatenate(const std::vector<std::vector<uint64_t>> &arrays);

struct ArraySegment {
    const uint64_t *data = nullptr;
    size_t length = 0;
    size_t index = 0;
};

std::vector<std::vector<uint64_t>> merge_deduplicate_all(
    const std::vector<std::vector<uint64_t>> &arrays,
    const std::vector<uint64_t> &pivots,
    int n_threads
);

std::vector<uint64_t> merge_deduplicate_all_concat(
    const std::vector<std::vector<uint64_t>> &arrays,
    const std::vector<uint64_t> &pivots,
    int n_threads
);

std::vector<uint64_t> merge_and_deduplicate(const std::vector<uint64_t> &arr1, const std::vector<uint64_t> &arr2);

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
    board = (board ^ (board >> 27)) * 0x1A85EC53ULL + (board >> 23) + board;
    return (board ^ (board >> 27)) * 0x1A85EC53ULL + (board >> 23) + board;
}

} // namespace BookGeneratorUtils
