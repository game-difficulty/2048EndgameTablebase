#include "AdaptiveIndex.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <utility>

namespace AdaptiveIndex {
namespace {

constexpr uint64_t kExperimentalHybridDelayedL3Threshold = 10'000'000ULL;

struct TempL3Bucket {
    uint16_t h2_value = 0;
    uint32_t h3_count = 0;
    uint64_t h3_values_offset = 0;
    uint64_t h3_starts_offset = 0;
    uint64_t starts_offset = 0;
};

struct TempL2Node {
    uint32_t h1_value = 0;
    uint64_t parent_begin = 0;
    uint32_t h2_count = 0;
    uint64_t h2_values_offset = 0;
    uint64_t h2_starts_offset = 0;
    uint32_t l3_bucket_count = 0;
    uint64_t l3_buckets_offset = 0;
    uint64_t starts_offset = 0;
    uint64_t l3_nodes_offset = 0;
};

struct ExperimentalTempL3Bucket {
    uint16_t h2_value = 0;
    std::vector<uint16_t> h3_values;
    std::vector<uint32_t> rel_starts;
    uint64_t starts_offset = 0;
};

struct ExperimentalTempL2Node {
    uint32_t h1_value = 0;
    uint64_t parent_begin = 0;
    std::vector<uint16_t> h2_values;
    std::vector<uint32_t> rel_starts;
    std::vector<ExperimentalTempL3Bucket> l3_buckets;
    uint64_t starts_offset = 0;
    uint64_t l3_nodes_offset = 0;
    bool active = false;
};

struct H1Transition {
    uint32_t h1_value = 0;
    uint64_t position = 0;
};

uint32_t effective_num_threads(const Config &config) {
    if (config.num_threads > 0) {
        return static_cast<uint32_t>(config.num_threads);
    }
    return static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
}

double wall_time_seconds() {
    return omp_get_wtime();
}

uint32_t h1_of(uint64_t key) {
    return static_cast<uint32_t>(key >> 44U);
}

uint16_t h2_of(uint64_t key) {
    return static_cast<uint16_t>((key >> 32U) & 0xFFFU);
}

uint16_t h3_of(uint64_t key) {
    return static_cast<uint16_t>((key >> 20U) & 0xFFFU);
}

uint32_t popcount64(uint64_t value) {
    return static_cast<uint32_t>(__builtin_popcountll(value));
}

void clear_sub_rank(BitRank4096 &map) {
    map.words.fill(0ULL);
    map.ranks.fill(0U);
}

void finalize_sub_rank(BitRank4096 &map) {
    uint16_t total = 0;
    for (size_t i = 0; i < map.words.size(); ++i) {
        map.ranks[i] = total;
        total = static_cast<uint16_t>(total + popcount64(map.words[i]));
    }
    map.ranks[map.words.size()] = total;
}

void build_sub_rank(const uint16_t *values, uint32_t count, BitRank4096 &map) {
    clear_sub_rank(map);
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t value = values[i];
        map.words[value >> 6U] |= (1ULL << (value & 63U));
    }
    finalize_sub_rank(map);
}

bool cpu_supports_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
#else
    return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void build_sub_rank_avx2(const uint16_t *values, uint32_t count, BitRank4096 &map) {
    clear_sub_rank(map);
    alignas(32) uint16_t word_indices[16];
    alignas(32) uint16_t bit_indices[16];
    const __m256i bit_mask = _mm256_set1_epi16(63);

    uint32_t i = 0;
    for (; i + 16U <= count; i += 16U) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values + i));
        const __m256i words = _mm256_srli_epi16(v, 6);
        const __m256i bits = _mm256_and_si256(v, bit_mask);
        _mm256_store_si256(reinterpret_cast<__m256i *>(word_indices), words);
        _mm256_store_si256(reinterpret_cast<__m256i *>(bit_indices), bits);
        for (uint32_t lane = 0; lane < 16U; ++lane) {
            map.words[word_indices[lane]] |= (1ULL << bit_indices[lane]);
        }
    }
    for (; i < count; ++i) {
        const uint16_t value = values[i];
        map.words[value >> 6U] |= (1ULL << (value & 63U));
    }
    finalize_sub_rank(map);
}

void build_sub_rank_fast(const uint16_t *values, uint32_t count, BitRank4096 &map) {
    if (count >= 16U && cpu_supports_avx2()) {
        build_sub_rank_avx2(values, count, map);
    } else {
        build_sub_rank(values, count, map);
    }
}

bool root_bit_test(const Index &index, uint32_t h1) {
    return ((index.l2_root_bitmap[h1 >> 6U] >> (h1 & 63U)) & 1ULL) != 0ULL;
}

uint32_t root_rank_before(const Index &index, uint32_t h1) {
    uint32_t word = h1 >> 6U;
    uint32_t bit = h1 & 63U;
    uint64_t mask = bit == 0U ? 0ULL : ((1ULL << bit) - 1ULL);
    return index.l2_root_rank[word] + popcount64(index.l2_root_bitmap[word] & mask);
}

bool sub_bit_test(const BitRank4096 &map, uint16_t value) {
    return ((map.words[value >> 6U] >> (value & 63U)) & 1ULL) != 0ULL;
}

uint32_t sub_rank_before(const BitRank4096 &map, uint16_t value) {
    uint32_t word = value >> 6U;
    uint32_t bit = value & 63U;
    uint64_t mask = bit == 0U ? 0ULL : ((1ULL << bit) - 1ULL);
    return static_cast<uint32_t>(map.ranks[word]) + popcount64(map.words[word] & mask);
}

uint64_t load_u64_unaligned(const unsigned char *ptr) {
    uint64_t value = 0;
    std::memcpy(&value, ptr, sizeof(uint64_t));
    return value;
}

Index build_contiguous_impl(
    const uint64_t *keys,
    uint64_t size,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    Index index;
    index.l1_offsets.resize(kL1Size, 0U);
    if (size == 0U) {
        return index;
    }
    const double t0 = profile ? wall_time_seconds() : 0.0;

    const uint64_t reserve_l2 = std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    );
    const uint64_t reserve_l3 = std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l3_split_threshold, 1U))
    );

    std::vector<TempL2Node> temp_nodes;
    temp_nodes.reserve(static_cast<size_t>(reserve_l2));
    std::vector<uint16_t> temp_h2_values;
    temp_h2_values.reserve(static_cast<size_t>(reserve_l2));
    std::vector<uint32_t> temp_h2_starts;
    temp_h2_starts.reserve(static_cast<size_t>(reserve_l2 + 1U));
    std::vector<TempL3Bucket> temp_l3_buckets;
    temp_l3_buckets.reserve(static_cast<size_t>(reserve_l3));
    std::vector<uint16_t> temp_h3_values;
    temp_h3_values.reserve(static_cast<size_t>(reserve_l3));
    std::vector<uint32_t> temp_h3_starts;
    temp_h3_starts.reserve(static_cast<size_t>(reserve_l3 + 1U));

    uint64_t cursor = 0U;
    uint32_t fill_from = 0U;
    while (cursor < size) {
        const uint64_t key0 = keys[cursor];
        const uint32_t h1 = h1_of(key0);
        const uint64_t h1_begin = cursor;
        const uint64_t h2_values_offset = temp_h2_values.size();
        const uint64_t h2_starts_offset = temp_h2_starts.size();
        const uint64_t l3_buckets_offset = temp_l3_buckets.size();
        const uint64_t h3_values_offset_base = temp_h3_values.size();
        const uint64_t h3_starts_offset_base = temp_h3_starts.size();
        uint32_t h2_count = 0U;
        uint32_t l3_bucket_count = 0U;

        while (cursor < size) {
            const uint64_t h1_key = keys[cursor];
            if (h1_of(h1_key) != h1) {
                break;
            }
            const uint16_t h2 = h2_of(h1_key);
            const uint64_t h2_begin = cursor;
            const uint64_t h3_values_offset = temp_h3_values.size();
            const uint64_t h3_starts_offset = temp_h3_starts.size();
            uint32_t h3_count = 0U;

            temp_h2_values.push_back(h2);
            temp_h2_starts.push_back(static_cast<uint32_t>(h2_begin - h1_begin));
            ++h2_count;

            while (cursor < size) {
                const uint64_t h2_key = keys[cursor];
                if (h1_of(h2_key) != h1 || h2_of(h2_key) != h2) {
                    break;
                }
                const uint16_t h3 = h3_of(h2_key);
                temp_h3_values.push_back(h3);
                temp_h3_starts.push_back(static_cast<uint32_t>(cursor - h2_begin));
                ++h3_count;
                ++cursor;
                while (cursor < size) {
                    const uint64_t next_key = keys[cursor];
                    if (h1_of(next_key) != h1 || h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++cursor;
                }
            }

            const uint64_t h2_span = cursor - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                temp_h3_starts.push_back(static_cast<uint32_t>(h2_span));
                TempL3Bucket l3_bucket;
                l3_bucket.h2_value = h2;
                l3_bucket.h3_count = h3_count;
                l3_bucket.h3_values_offset = h3_values_offset;
                l3_bucket.h3_starts_offset = h3_starts_offset;
                temp_l3_buckets.push_back(l3_bucket);
                ++l3_bucket_count;
            } else {
                temp_h3_values.resize(static_cast<size_t>(h3_values_offset));
                temp_h3_starts.resize(static_cast<size_t>(h3_starts_offset));
            }
        }

        std::fill(
            index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(h1 + 1U),
            h1_begin
        );
        fill_from = h1 + 1U;

        const uint64_t h1_span = cursor - h1_begin;
        if (
            h1_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
            h1_span > static_cast<uint64_t>(config.l2_split_threshold) &&
            h2_count >= 2U
        ) {
            temp_h2_starts.push_back(static_cast<uint32_t>(h1_span));
            TempL2Node node;
            node.h1_value = h1;
            node.parent_begin = h1_begin;
            node.h2_count = h2_count;
            node.h2_values_offset = h2_values_offset;
            node.h2_starts_offset = h2_starts_offset;
            node.l3_bucket_count = l3_bucket_count;
            node.l3_buckets_offset = l3_buckets_offset;
            temp_nodes.push_back(node);
        } else {
            temp_h2_values.resize(static_cast<size_t>(h2_values_offset));
            temp_h2_starts.resize(static_cast<size_t>(h2_starts_offset));
            temp_l3_buckets.resize(static_cast<size_t>(l3_buckets_offset));
            temp_h3_values.resize(static_cast<size_t>(h3_values_offset_base));
            temp_h3_starts.resize(static_cast<size_t>(h3_starts_offset_base));
        }
    }

    std::fill(index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), index.l1_offsets.end(), size);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    uint64_t total_l2_starts = 0U;
    uint64_t total_l3_starts = 0U;
    uint64_t total_l3_nodes = 0U;
    for (TempL2Node &node : temp_nodes) {
        node.starts_offset = total_l2_starts;
        total_l2_starts += static_cast<uint64_t>(node.h2_count) + 1U;
        node.l3_nodes_offset = total_l3_nodes;
        total_l3_nodes += static_cast<uint64_t>(node.l3_bucket_count);
        for (uint32_t bucket_index = 0; bucket_index < node.l3_bucket_count; ++bucket_index) {
            TempL3Bucket &bucket = temp_l3_buckets[static_cast<size_t>(node.l3_buckets_offset + bucket_index)];
            bucket.starts_offset = total_l3_starts;
            total_l3_starts += static_cast<uint64_t>(bucket.h3_count) + 1U;
        }
    }

    index.l2_relative_starts.resize(static_cast<size_t>(total_l2_starts));
    index.l3_relative_starts.resize(static_cast<size_t>(total_l3_starts));
    index.l2_nodes.resize(temp_nodes.size());
    index.l3_nodes.resize(static_cast<size_t>(total_l3_nodes));

    index.l2_root_bitmap.fill(0ULL);
    index.l2_root_rank.fill(0U);
    for (const TempL2Node &node : temp_nodes) {
        index.l2_root_bitmap[node.h1_value >> 6U] |= (1ULL << (node.h1_value & 63U));
    }
    uint32_t root_total = 0U;
    for (size_t i = 0; i < index.l2_root_bitmap.size(); ++i) {
        index.l2_root_rank[i] = root_total;
        root_total += popcount64(index.l2_root_bitmap[i]);
    }
    index.l2_root_rank[index.l2_root_bitmap.size()] = root_total;
    const double t2 = profile ? wall_time_seconds() : 0.0;

    const int num_threads = static_cast<int>(effective_num_threads(config));
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t node_index = 0; node_index < static_cast<int64_t>(temp_nodes.size()); ++node_index) {
        const TempL2Node &temp = temp_nodes[static_cast<size_t>(node_index)];
        L2Node &dst = index.l2_nodes[static_cast<size_t>(node_index)];
        dst.parent_begin = temp.parent_begin;
        dst.starts_offset = temp.starts_offset;
        dst.l3_nodes_offset = temp.l3_nodes_offset;
        build_sub_rank(
            temp_h2_values.data() + static_cast<std::ptrdiff_t>(temp.h2_values_offset),
            temp.h2_count,
            dst.h2_map
        );

        clear_sub_rank(dst.l3_map);
        for (uint32_t bucket_index = 0; bucket_index < temp.l3_bucket_count; ++bucket_index) {
            const TempL3Bucket &bucket = temp_l3_buckets[static_cast<size_t>(temp.l3_buckets_offset + bucket_index)];
            dst.l3_map.words[bucket.h2_value >> 6U] |= (1ULL << (bucket.h2_value & 63U));
        }
        finalize_sub_rank(dst.l3_map);

        std::copy(
            temp_h2_starts.begin() + static_cast<std::ptrdiff_t>(temp.h2_starts_offset),
            temp_h2_starts.begin() + static_cast<std::ptrdiff_t>(temp.h2_starts_offset + temp.h2_count + 1U),
            index.l2_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp.starts_offset)
        );

        for (uint32_t bucket_index = 0; bucket_index < temp.l3_bucket_count; ++bucket_index) {
            const TempL3Bucket &temp_bucket = temp_l3_buckets[static_cast<size_t>(temp.l3_buckets_offset + bucket_index)];
            L3Node &dst_bucket = index.l3_nodes[static_cast<size_t>(temp.l3_nodes_offset + bucket_index)];
            dst_bucket.starts_offset = temp_bucket.starts_offset;
            build_sub_rank(
                temp_h3_values.data() + static_cast<std::ptrdiff_t>(temp_bucket.h3_values_offset),
                temp_bucket.h3_count,
                dst_bucket.h3_map
            );
            std::copy(
                temp_h3_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.h3_starts_offset),
                temp_h3_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.h3_starts_offset + temp_bucket.h3_count + 1U),
                index.l3_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.starts_offset)
            );
        }
    }

    if (profile != nullptr) {
        const double t3 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
        profile->phase3_seconds = t3 - t2;
        profile->phase4_seconds = 0.0;
    }

    return index;
}

Index build_strided_impl(
    const unsigned char *base,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    Index index;
    index.l1_offsets.resize(kL1Size, 0U);
    if (size == 0U) {
        return index;
    }
    const double t0 = profile ? wall_time_seconds() : 0.0;

    const uint64_t reserve_l2 = std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    );
    const uint64_t reserve_l3 = std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l3_split_threshold, 1U))
    );

    std::vector<TempL2Node> temp_nodes;
    temp_nodes.reserve(static_cast<size_t>(reserve_l2));
    std::vector<uint16_t> temp_h2_values;
    temp_h2_values.reserve(static_cast<size_t>(reserve_l2));
    std::vector<uint32_t> temp_h2_starts;
    temp_h2_starts.reserve(static_cast<size_t>(reserve_l2 + 1U));
    std::vector<TempL3Bucket> temp_l3_buckets;
    temp_l3_buckets.reserve(static_cast<size_t>(reserve_l3));
    std::vector<uint16_t> temp_h3_values;
    temp_h3_values.reserve(static_cast<size_t>(reserve_l3));
    std::vector<uint32_t> temp_h3_starts;
    temp_h3_starts.reserve(static_cast<size_t>(reserve_l3 + 1U));

    uint64_t cursor = 0U;
    uint32_t fill_from = 0U;
    while (cursor < size) {
        const uint64_t key0 = load_u64_unaligned(base + cursor * stride + key_offset);
        const uint32_t h1 = h1_of(key0);
        const uint64_t h1_begin = cursor;
        const uint64_t h2_values_offset = temp_h2_values.size();
        const uint64_t h2_starts_offset = temp_h2_starts.size();
        const uint64_t l3_buckets_offset = temp_l3_buckets.size();
        const uint64_t h3_values_offset_base = temp_h3_values.size();
        const uint64_t h3_starts_offset_base = temp_h3_starts.size();
        uint32_t h2_count = 0U;
        uint32_t l3_bucket_count = 0U;

        while (cursor < size) {
            const uint64_t h1_key = load_u64_unaligned(base + cursor * stride + key_offset);
            if (h1_of(h1_key) != h1) {
                break;
            }
            const uint16_t h2 = h2_of(h1_key);
            const uint64_t h2_begin = cursor;
            const uint64_t h3_values_offset = temp_h3_values.size();
            const uint64_t h3_starts_offset = temp_h3_starts.size();
            uint32_t h3_count = 0U;

            temp_h2_values.push_back(h2);
            temp_h2_starts.push_back(static_cast<uint32_t>(h2_begin - h1_begin));
            ++h2_count;

            while (cursor < size) {
                const uint64_t h2_key = load_u64_unaligned(base + cursor * stride + key_offset);
                if (h1_of(h2_key) != h1 || h2_of(h2_key) != h2) {
                    break;
                }
                const uint16_t h3 = h3_of(h2_key);
                temp_h3_values.push_back(h3);
                temp_h3_starts.push_back(static_cast<uint32_t>(cursor - h2_begin));
                ++h3_count;
                ++cursor;
                while (cursor < size) {
                    const uint64_t next_key = load_u64_unaligned(base + cursor * stride + key_offset);
                    if (h1_of(next_key) != h1 || h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++cursor;
                }
            }

            const uint64_t h2_span = cursor - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                temp_h3_starts.push_back(static_cast<uint32_t>(h2_span));
                TempL3Bucket l3_bucket;
                l3_bucket.h2_value = h2;
                l3_bucket.h3_count = h3_count;
                l3_bucket.h3_values_offset = h3_values_offset;
                l3_bucket.h3_starts_offset = h3_starts_offset;
                temp_l3_buckets.push_back(l3_bucket);
                ++l3_bucket_count;
            } else {
                temp_h3_values.resize(static_cast<size_t>(h3_values_offset));
                temp_h3_starts.resize(static_cast<size_t>(h3_starts_offset));
            }
        }

        std::fill(
            index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(h1 + 1U),
            h1_begin
        );
        fill_from = h1 + 1U;

        const uint64_t h1_span = cursor - h1_begin;
        if (
            h1_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
            h1_span > static_cast<uint64_t>(config.l2_split_threshold) &&
            h2_count >= 2U
        ) {
            temp_h2_starts.push_back(static_cast<uint32_t>(h1_span));
            TempL2Node node;
            node.h1_value = h1;
            node.parent_begin = h1_begin;
            node.h2_count = h2_count;
            node.h2_values_offset = h2_values_offset;
            node.h2_starts_offset = h2_starts_offset;
            node.l3_bucket_count = l3_bucket_count;
            node.l3_buckets_offset = l3_buckets_offset;
            temp_nodes.push_back(node);
        } else {
            temp_h2_values.resize(static_cast<size_t>(h2_values_offset));
            temp_h2_starts.resize(static_cast<size_t>(h2_starts_offset));
            temp_l3_buckets.resize(static_cast<size_t>(l3_buckets_offset));
            temp_h3_values.resize(static_cast<size_t>(h3_values_offset_base));
            temp_h3_starts.resize(static_cast<size_t>(h3_starts_offset_base));
        }
    }

    std::fill(index.l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), index.l1_offsets.end(), size);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    uint64_t total_l2_starts = 0U;
    uint64_t total_l3_starts = 0U;
    uint64_t total_l3_nodes = 0U;
    for (TempL2Node &node : temp_nodes) {
        node.starts_offset = total_l2_starts;
        total_l2_starts += static_cast<uint64_t>(node.h2_count) + 1U;
        node.l3_nodes_offset = total_l3_nodes;
        total_l3_nodes += static_cast<uint64_t>(node.l3_bucket_count);
        for (uint32_t bucket_index = 0; bucket_index < node.l3_bucket_count; ++bucket_index) {
            TempL3Bucket &bucket = temp_l3_buckets[static_cast<size_t>(node.l3_buckets_offset + bucket_index)];
            bucket.starts_offset = total_l3_starts;
            total_l3_starts += static_cast<uint64_t>(bucket.h3_count) + 1U;
        }
    }

    index.l2_relative_starts.resize(static_cast<size_t>(total_l2_starts));
    index.l3_relative_starts.resize(static_cast<size_t>(total_l3_starts));
    index.l2_nodes.resize(temp_nodes.size());
    index.l3_nodes.resize(static_cast<size_t>(total_l3_nodes));

    index.l2_root_bitmap.fill(0ULL);
    index.l2_root_rank.fill(0U);
    for (const TempL2Node &node : temp_nodes) {
        index.l2_root_bitmap[node.h1_value >> 6U] |= (1ULL << (node.h1_value & 63U));
    }
    uint32_t root_total = 0U;
    for (size_t i = 0; i < index.l2_root_bitmap.size(); ++i) {
        index.l2_root_rank[i] = root_total;
        root_total += popcount64(index.l2_root_bitmap[i]);
    }
    index.l2_root_rank[index.l2_root_bitmap.size()] = root_total;
    const double t2 = profile ? wall_time_seconds() : 0.0;

    const int num_threads = static_cast<int>(effective_num_threads(config));
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t node_index = 0; node_index < static_cast<int64_t>(temp_nodes.size()); ++node_index) {
        const TempL2Node &temp = temp_nodes[static_cast<size_t>(node_index)];
        L2Node &dst = index.l2_nodes[static_cast<size_t>(node_index)];
        dst.parent_begin = temp.parent_begin;
        dst.starts_offset = temp.starts_offset;
        dst.l3_nodes_offset = temp.l3_nodes_offset;
        build_sub_rank(
            temp_h2_values.data() + static_cast<std::ptrdiff_t>(temp.h2_values_offset),
            temp.h2_count,
            dst.h2_map
        );

        clear_sub_rank(dst.l3_map);
        for (uint32_t bucket_index = 0; bucket_index < temp.l3_bucket_count; ++bucket_index) {
            const TempL3Bucket &bucket = temp_l3_buckets[static_cast<size_t>(temp.l3_buckets_offset + bucket_index)];
            dst.l3_map.words[bucket.h2_value >> 6U] |= (1ULL << (bucket.h2_value & 63U));
        }
        finalize_sub_rank(dst.l3_map);

        std::copy(
            temp_h2_starts.begin() + static_cast<std::ptrdiff_t>(temp.h2_starts_offset),
            temp_h2_starts.begin() + static_cast<std::ptrdiff_t>(temp.h2_starts_offset + temp.h2_count + 1U),
            index.l2_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp.starts_offset)
        );

        for (uint32_t bucket_index = 0; bucket_index < temp.l3_bucket_count; ++bucket_index) {
            const TempL3Bucket &temp_bucket = temp_l3_buckets[static_cast<size_t>(temp.l3_buckets_offset + bucket_index)];
            L3Node &dst_bucket = index.l3_nodes[static_cast<size_t>(temp.l3_nodes_offset + bucket_index)];
            dst_bucket.starts_offset = temp_bucket.starts_offset;
            build_sub_rank(
                temp_h3_values.data() + static_cast<std::ptrdiff_t>(temp_bucket.h3_values_offset),
                temp_bucket.h3_count,
                dst_bucket.h3_map
            );
            std::copy(
                temp_h3_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.h3_starts_offset),
                temp_h3_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.h3_starts_offset + temp_bucket.h3_count + 1U),
                index.l3_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.starts_offset)
            );
        }
    }

    if (profile != nullptr) {
        const double t3 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
        profile->phase3_seconds = t3 - t2;
        profile->phase4_seconds = 0.0;
    }

    return index;
}

Index materialize_experimental_nodes(
    std::vector<uint64_t> l1_offsets,
    std::vector<ExperimentalTempL2Node> &&temp_nodes,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    Index index;
    index.l1_offsets = std::move(l1_offsets);
    const double t0 = profile ? wall_time_seconds() : 0.0;

    uint64_t total_l2_starts = 0U;
    uint64_t total_l3_starts = 0U;
    uint64_t total_l3_nodes = 0U;
    for (ExperimentalTempL2Node &node : temp_nodes) {
        node.starts_offset = total_l2_starts;
        total_l2_starts += static_cast<uint64_t>(node.h2_values.size()) + 1U;
        node.l3_nodes_offset = total_l3_nodes;
        total_l3_nodes += static_cast<uint64_t>(node.l3_buckets.size());
        for (ExperimentalTempL3Bucket &bucket : node.l3_buckets) {
            bucket.starts_offset = total_l3_starts;
            total_l3_starts += static_cast<uint64_t>(bucket.h3_values.size()) + 1U;
        }
    }

    index.l2_relative_starts.resize(static_cast<size_t>(total_l2_starts));
    index.l3_relative_starts.resize(static_cast<size_t>(total_l3_starts));
    index.l2_nodes.resize(temp_nodes.size());
    index.l3_nodes.resize(static_cast<size_t>(total_l3_nodes));

    index.l2_root_bitmap.fill(0ULL);
    index.l2_root_rank.fill(0U);
    for (const ExperimentalTempL2Node &node : temp_nodes) {
        index.l2_root_bitmap[node.h1_value >> 6U] |= (1ULL << (node.h1_value & 63U));
    }
    uint32_t root_total = 0U;
    for (size_t i = 0; i < index.l2_root_bitmap.size(); ++i) {
        index.l2_root_rank[i] = root_total;
        root_total += popcount64(index.l2_root_bitmap[i]);
    }
    index.l2_root_rank[index.l2_root_bitmap.size()] = root_total;
    const double t1 = profile ? wall_time_seconds() : 0.0;

    const int num_threads = static_cast<int>(effective_num_threads(config));
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t node_index = 0; node_index < static_cast<int64_t>(temp_nodes.size()); ++node_index) {
        const ExperimentalTempL2Node &temp = temp_nodes[static_cast<size_t>(node_index)];
        L2Node &dst = index.l2_nodes[static_cast<size_t>(node_index)];
        dst.parent_begin = temp.parent_begin;
        dst.starts_offset = temp.starts_offset;
        dst.l3_nodes_offset = temp.l3_nodes_offset;
        build_sub_rank_fast(
            temp.h2_values.data(),
            static_cast<uint32_t>(temp.h2_values.size()),
            dst.h2_map
        );

        if (!temp.l3_buckets.empty()) {
            clear_sub_rank(dst.l3_map);
            for (const ExperimentalTempL3Bucket &bucket : temp.l3_buckets) {
                dst.l3_map.words[bucket.h2_value >> 6U] |= (1ULL << (bucket.h2_value & 63U));
            }
            finalize_sub_rank(dst.l3_map);
        }

        std::copy(
            temp.rel_starts.begin(),
            temp.rel_starts.end(),
            index.l2_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp.starts_offset)
        );

        for (size_t bucket_index = 0; bucket_index < temp.l3_buckets.size(); ++bucket_index) {
            const ExperimentalTempL3Bucket &temp_bucket = temp.l3_buckets[bucket_index];
            L3Node &dst_bucket = index.l3_nodes[static_cast<size_t>(temp.l3_nodes_offset + bucket_index)];
            dst_bucket.starts_offset = temp_bucket.starts_offset;
            build_sub_rank_fast(
                temp_bucket.h3_values.data(),
                static_cast<uint32_t>(temp_bucket.h3_values.size()),
                dst_bucket.h3_map
            );
            std::copy(
                temp_bucket.rel_starts.begin(),
                temp_bucket.rel_starts.end(),
                index.l3_relative_starts.begin() + static_cast<std::ptrdiff_t>(temp_bucket.starts_offset)
            );
        }
    }

    if (profile != nullptr) {
        const double t2 = wall_time_seconds();
        if (profile->phase3_seconds == 0.0) {
            profile->phase3_seconds = t1 - t0;
            profile->phase4_seconds = t2 - t1;
        } else {
            profile->phase4_seconds = t2 - t0;
        }
    }

    return index;
}

void collect_h1_transitions_scalar(
    const uint64_t *keys,
    uint64_t size,
    std::vector<H1Transition> &transitions,
    uint32_t first_h1
) {
    transitions.clear();
    transitions.reserve(static_cast<size_t>(std::min<uint64_t>(size / 64U + 1U, 1U << 20U)));

    uint32_t prev_h1 = first_h1;
    for (uint64_t i = 1U; i < size; ++i) {
        const uint32_t h1 = h1_of(keys[i]);
        if (h1 == prev_h1) {
            continue;
        }
        transitions.push_back(H1Transition{h1, i});
        prev_h1 = h1;
    }
}

void collect_h1_transitions_parallel(
    const uint64_t *keys,
    uint64_t size,
    std::vector<H1Transition> &transitions,
    uint32_t first_h1,
    const Config &config
) {
    const size_t worker_count =
        (size >= (1U << 20U))
            ? static_cast<size_t>(std::max(1U, effective_num_threads(config)))
            : 1U;

    if (worker_count <= 1U) {
        collect_h1_transitions_scalar(keys, size, transitions, first_h1);
        return;
    }

    const uint64_t chunk = (size + worker_count - 1U) / worker_count;
    std::vector<std::vector<H1Transition>> local_transitions(worker_count);

#pragma omp parallel for num_threads(static_cast<int>(worker_count))
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const uint64_t begin = static_cast<uint64_t>(t) * chunk;
        const uint64_t end = std::min<uint64_t>(size, begin + chunk);
        if (begin >= end) {
            continue;
        }

        auto &local = local_transitions[static_cast<size_t>(t)];
        local.reserve(static_cast<size_t>(std::min<uint64_t>((end - begin) / 64U + 1U, 1U << 18U)));

        uint32_t prev_h1 = first_h1;
        uint64_t i = 1U;
        if (begin > 0U) {
            prev_h1 = h1_of(keys[begin - 1U]);
            i = begin;
        }

        for (; i < end; ++i) {
            const uint32_t h1 = h1_of(keys[i]);
            if (h1 == prev_h1) {
                continue;
            }
            local.push_back(H1Transition{h1, i});
            prev_h1 = h1;
        }
    }

    size_t total_transitions = 0U;
    for (const auto &local : local_transitions) {
        total_transitions += local.size();
    }
    transitions.clear();
    transitions.reserve(total_transitions);

    uint32_t prev_h1 = first_h1;
    for (const auto &local : local_transitions) {
        for (const H1Transition &transition : local) {
            if (transition.h1_value == prev_h1) {
                continue;
            }
            transitions.push_back(transition);
            prev_h1 = transition.h1_value;
        }
    }
}

void collect_h1_transitions_scalar_strided(
    const unsigned char *base,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    std::vector<H1Transition> &transitions,
    uint32_t first_h1
) {
    transitions.clear();
    transitions.reserve(static_cast<size_t>(std::min<uint64_t>(size / 64U + 1U, 1U << 20U)));

    uint32_t prev_h1 = first_h1;
    for (uint64_t i = 1U; i < size; ++i) {
        const uint32_t h1 = h1_of(load_u64_unaligned(base + i * stride + key_offset));
        if (h1 == prev_h1) {
            continue;
        }
        transitions.push_back(H1Transition{h1, i});
        prev_h1 = h1;
    }
}

void collect_h1_transitions_parallel_strided(
    const unsigned char *base,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    std::vector<H1Transition> &transitions,
    uint32_t first_h1,
    const Config &config
) {
    const size_t worker_count =
        (size >= (1U << 20U))
            ? static_cast<size_t>(std::max(1U, effective_num_threads(config)))
            : 1U;

    if (worker_count <= 1U) {
        collect_h1_transitions_scalar_strided(base, size, stride, key_offset, transitions, first_h1);
        return;
    }

    const uint64_t chunk = (size + worker_count - 1U) / worker_count;
    std::vector<std::vector<H1Transition>> local_transitions(worker_count);

#pragma omp parallel for num_threads(static_cast<int>(worker_count))
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const uint64_t begin = static_cast<uint64_t>(t) * chunk;
        const uint64_t end = std::min<uint64_t>(size, begin + chunk);
        if (begin >= end) {
            continue;
        }

        auto &local = local_transitions[static_cast<size_t>(t)];
        local.reserve(static_cast<size_t>(std::min<uint64_t>((end - begin) / 64U + 1U, 1U << 18U)));

        uint32_t prev_h1 = first_h1;
        uint64_t i = 1U;
        if (begin > 0U) {
            prev_h1 = h1_of(load_u64_unaligned(base + (begin - 1U) * stride + key_offset));
            i = begin;
        }

        for (; i < end; ++i) {
            const uint32_t h1 = h1_of(load_u64_unaligned(base + i * stride + key_offset));
            if (h1 == prev_h1) {
                continue;
            }
            local.push_back(H1Transition{h1, i});
            prev_h1 = h1;
        }
    }

    size_t total_transitions = 0U;
    for (const auto &local : local_transitions) {
        total_transitions += local.size();
    }
    transitions.clear();
    transitions.reserve(total_transitions);

    uint32_t prev_h1 = first_h1;
    for (const auto &local : local_transitions) {
        for (const H1Transition &transition : local) {
            if (transition.h1_value == prev_h1) {
                continue;
            }
            transitions.push_back(transition);
            prev_h1 = transition.h1_value;
        }
    }
}

Index build_experimental_refined_impl(
    const uint64_t *keys,
    std::vector<uint64_t> l1_offsets,
    const std::vector<uint32_t> &active_h1_values,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    if (l1_offsets.empty()) {
        Index index;
        index.l1_offsets.resize(kL1Size, 0U);
        return index;
    }

    std::vector<ExperimentalTempL2Node> node_slots(active_h1_values.size());
    const int num_threads = static_cast<int>(effective_num_threads(config));
    const double t0 = profile ? wall_time_seconds() : 0.0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (int64_t idx = 0; idx < static_cast<int64_t>(active_h1_values.size()); ++idx) {
        const uint32_t h1 = active_h1_values[static_cast<size_t>(idx)];
        const uint64_t begin = l1_offsets[h1];
        const uint64_t end = l1_offsets[h1 + 1U];
        const uint64_t span = end - begin;
        if (
            span <= static_cast<uint64_t>(config.l2_split_threshold) ||
            span > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        ) {
            continue;
        }

        ExperimentalTempL2Node node;
        node.h1_value = h1;
        node.parent_begin = begin;
        node.h2_values.reserve(static_cast<size_t>(std::max<uint64_t>(2U, span / 64U)));
        node.rel_starts.reserve(static_cast<size_t>(std::max<uint64_t>(3U, span / 64U + 1U)));

        uint64_t local_cursor = begin;
        while (local_cursor < end) {
            const uint16_t h2 = h2_of(keys[local_cursor]);
            const uint64_t h2_begin = local_cursor;
            node.h2_values.push_back(h2);
            node.rel_starts.push_back(static_cast<uint32_t>(h2_begin - begin));

            ExperimentalTempL3Bucket l3_bucket;
            l3_bucket.h2_value = h2;
            uint32_t h3_count = 0U;
            while (local_cursor < end) {
                const uint64_t h2_key = keys[local_cursor];
                if (h2_of(h2_key) != h2) {
                    break;
                }
                ++h3_count;
                const uint16_t h3 = h3_of(h2_key);
                l3_bucket.h3_values.push_back(h3);
                l3_bucket.rel_starts.push_back(static_cast<uint32_t>(local_cursor - h2_begin));
                ++local_cursor;
                while (local_cursor < end) {
                    const uint64_t next_key = keys[local_cursor];
                    if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++local_cursor;
                }
            }

            const uint64_t h2_span = local_cursor - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                l3_bucket.rel_starts.push_back(static_cast<uint32_t>(h2_span));
                node.l3_buckets.push_back(std::move(l3_bucket));
            }
        }

        if (node.h2_values.size() >= 2U) {
            node.rel_starts.push_back(static_cast<uint32_t>(span));
            node.active = true;
            node_slots[static_cast<size_t>(idx)] = std::move(node);
        }
    }

    std::vector<ExperimentalTempL2Node> temp_nodes;
    temp_nodes.reserve(node_slots.size());
    for (auto &slot : node_slots) {
        if (slot.active) {
            temp_nodes.push_back(std::move(slot));
        }
    }
    if (profile != nullptr) {
        const double t1 = wall_time_seconds();
        if (profile->phase2_seconds == 0.0) {
            profile->phase2_seconds = t1 - t0;
        } else {
            profile->phase3_seconds = t1 - t0;
        }
    }

    return materialize_experimental_nodes(std::move(l1_offsets), std::move(temp_nodes), config, profile);
}

Index build_experimental_refined_delayed_l3_impl(
    const uint64_t *keys,
    std::vector<uint64_t> l1_offsets,
    const std::vector<uint32_t> &active_h1_values,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    if (l1_offsets.empty()) {
        Index index;
        index.l1_offsets.resize(kL1Size, 0U);
        return index;
    }

    std::vector<ExperimentalTempL2Node> node_slots(active_h1_values.size());
    const int num_threads = static_cast<int>(effective_num_threads(config));
    const double t0 = profile ? wall_time_seconds() : 0.0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (int64_t idx = 0; idx < static_cast<int64_t>(active_h1_values.size()); ++idx) {
        const uint32_t h1 = active_h1_values[static_cast<size_t>(idx)];
        const uint64_t begin = l1_offsets[h1];
        const uint64_t end = l1_offsets[h1 + 1U];
        const uint64_t span = end - begin;
        if (
            span <= static_cast<uint64_t>(config.l2_split_threshold) ||
            span > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        ) {
            continue;
        }

        ExperimentalTempL2Node node;
        node.h1_value = h1;
        node.parent_begin = begin;
        node.h2_values.reserve(static_cast<size_t>(std::max<uint64_t>(2U, span / 64U)));
        node.rel_starts.reserve(static_cast<size_t>(std::max<uint64_t>(3U, span / 64U + 1U)));

        uint64_t local_cursor = begin;
        while (local_cursor < end) {
            const uint16_t h2 = h2_of(keys[local_cursor]);
            const uint64_t h2_begin = local_cursor;
            node.h2_values.push_back(h2);
            node.rel_starts.push_back(static_cast<uint32_t>(h2_begin - begin));

            uint32_t h3_count = 0U;
            uint64_t scan_cursor = h2_begin;
            while (scan_cursor < end) {
                if (h2_of(keys[scan_cursor]) != h2) {
                    break;
                }
                ++h3_count;
                const uint16_t h3 = h3_of(keys[scan_cursor]);
                ++scan_cursor;
                while (scan_cursor < end) {
                    const uint64_t next_key = keys[scan_cursor];
                    if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++scan_cursor;
                }
            }

            const uint64_t h2_end = scan_cursor;
            const uint64_t h2_span = h2_end - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                ExperimentalTempL3Bucket bucket;
                bucket.h2_value = h2;
                bucket.h3_values.reserve(h3_count);
                bucket.rel_starts.reserve(static_cast<size_t>(h3_count) + 1U);

                uint64_t materialize_cursor = h2_begin;
                while (materialize_cursor < h2_end) {
                    const uint16_t h3 = h3_of(keys[materialize_cursor]);
                    bucket.h3_values.push_back(h3);
                    bucket.rel_starts.push_back(static_cast<uint32_t>(materialize_cursor - h2_begin));
                    ++materialize_cursor;
                    while (materialize_cursor < h2_end) {
                        const uint64_t next_key = keys[materialize_cursor];
                        if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                            break;
                        }
                        ++materialize_cursor;
                    }
                }
                bucket.rel_starts.push_back(static_cast<uint32_t>(h2_span));
                node.l3_buckets.push_back(std::move(bucket));
            }

            local_cursor = h2_end;
        }

        if (node.h2_values.size() >= 2U) {
            node.rel_starts.push_back(static_cast<uint32_t>(span));
            node.active = true;
            node_slots[static_cast<size_t>(idx)] = std::move(node);
        }
    }

    std::vector<ExperimentalTempL2Node> temp_nodes;
    temp_nodes.reserve(node_slots.size());
    for (auto &slot : node_slots) {
        if (slot.active) {
            temp_nodes.push_back(std::move(slot));
        }
    }
    if (profile != nullptr) {
        const double t1 = wall_time_seconds();
        if (profile->phase2_seconds == 0.0) {
            profile->phase2_seconds = t1 - t0;
        } else {
            profile->phase3_seconds = t1 - t0;
        }
    }

    return materialize_experimental_nodes(std::move(l1_offsets), std::move(temp_nodes), config, profile);
}

Index build_experimental_refined_strided_impl(
    const unsigned char *base,
    uint64_t stride,
    uint64_t key_offset,
    std::vector<uint64_t> l1_offsets,
    const std::vector<uint32_t> &active_h1_values,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    if (l1_offsets.empty()) {
        Index index;
        index.l1_offsets.resize(kL1Size, 0U);
        return index;
    }

    std::vector<ExperimentalTempL2Node> node_slots(active_h1_values.size());
    const int num_threads = static_cast<int>(effective_num_threads(config));
    const double t0 = profile ? wall_time_seconds() : 0.0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (int64_t idx = 0; idx < static_cast<int64_t>(active_h1_values.size()); ++idx) {
        const uint32_t h1 = active_h1_values[static_cast<size_t>(idx)];
        const uint64_t begin = l1_offsets[h1];
        const uint64_t end = l1_offsets[h1 + 1U];
        const uint64_t span = end - begin;
        if (
            span <= static_cast<uint64_t>(config.l2_split_threshold) ||
            span > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        ) {
            continue;
        }

        ExperimentalTempL2Node node;
        node.h1_value = h1;
        node.parent_begin = begin;
        node.h2_values.reserve(static_cast<size_t>(std::max<uint64_t>(2U, span / 64U)));
        node.rel_starts.reserve(static_cast<size_t>(std::max<uint64_t>(3U, span / 64U + 1U)));

        uint64_t local_cursor = begin;
        while (local_cursor < end) {
            const uint16_t h2 = h2_of(load_u64_unaligned(base + local_cursor * stride + key_offset));
            const uint64_t h2_begin = local_cursor;
            node.h2_values.push_back(h2);
            node.rel_starts.push_back(static_cast<uint32_t>(h2_begin - begin));

            ExperimentalTempL3Bucket l3_bucket;
            l3_bucket.h2_value = h2;
            uint32_t h3_count = 0U;
            while (local_cursor < end) {
                const uint64_t h2_key = load_u64_unaligned(base + local_cursor * stride + key_offset);
                if (h2_of(h2_key) != h2) {
                    break;
                }
                ++h3_count;
                const uint16_t h3 = h3_of(h2_key);
                l3_bucket.h3_values.push_back(h3);
                l3_bucket.rel_starts.push_back(static_cast<uint32_t>(local_cursor - h2_begin));
                ++local_cursor;
                while (local_cursor < end) {
                    const uint64_t next_key = load_u64_unaligned(base + local_cursor * stride + key_offset);
                    if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++local_cursor;
                }
            }

            const uint64_t h2_span = local_cursor - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                l3_bucket.rel_starts.push_back(static_cast<uint32_t>(h2_span));
                node.l3_buckets.push_back(std::move(l3_bucket));
            }
        }

        if (node.h2_values.size() >= 2U) {
            node.rel_starts.push_back(static_cast<uint32_t>(span));
            node.active = true;
            node_slots[static_cast<size_t>(idx)] = std::move(node);
        }
    }

    std::vector<ExperimentalTempL2Node> temp_nodes;
    temp_nodes.reserve(node_slots.size());
    for (auto &slot : node_slots) {
        if (slot.active) {
            temp_nodes.push_back(std::move(slot));
        }
    }
    if (profile != nullptr) {
        const double t1 = wall_time_seconds();
        if (profile->phase2_seconds == 0.0) {
            profile->phase2_seconds = t1 - t0;
        } else {
            profile->phase3_seconds = t1 - t0;
        }
    }

    return materialize_experimental_nodes(std::move(l1_offsets), std::move(temp_nodes), config, profile);
}

Index build_experimental_refined_delayed_l3_strided_impl(
    const unsigned char *base,
    uint64_t stride,
    uint64_t key_offset,
    std::vector<uint64_t> l1_offsets,
    const std::vector<uint32_t> &active_h1_values,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    if (l1_offsets.empty()) {
        Index index;
        index.l1_offsets.resize(kL1Size, 0U);
        return index;
    }

    std::vector<ExperimentalTempL2Node> node_slots(active_h1_values.size());
    const int num_threads = static_cast<int>(effective_num_threads(config));
    const double t0 = profile ? wall_time_seconds() : 0.0;
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (int64_t idx = 0; idx < static_cast<int64_t>(active_h1_values.size()); ++idx) {
        const uint32_t h1 = active_h1_values[static_cast<size_t>(idx)];
        const uint64_t begin = l1_offsets[h1];
        const uint64_t end = l1_offsets[h1 + 1U];
        const uint64_t span = end - begin;
        if (
            span <= static_cast<uint64_t>(config.l2_split_threshold) ||
            span > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        ) {
            continue;
        }

        ExperimentalTempL2Node node;
        node.h1_value = h1;
        node.parent_begin = begin;
        node.h2_values.reserve(static_cast<size_t>(std::max<uint64_t>(2U, span / 64U)));
        node.rel_starts.reserve(static_cast<size_t>(std::max<uint64_t>(3U, span / 64U + 1U)));

        uint64_t local_cursor = begin;
        while (local_cursor < end) {
            const uint16_t h2 = h2_of(load_u64_unaligned(base + local_cursor * stride + key_offset));
            const uint64_t h2_begin = local_cursor;
            node.h2_values.push_back(h2);
            node.rel_starts.push_back(static_cast<uint32_t>(h2_begin - begin));

            uint32_t h3_count = 0U;
            uint64_t scan_cursor = h2_begin;
            while (scan_cursor < end) {
                if (h2_of(load_u64_unaligned(base + scan_cursor * stride + key_offset)) != h2) {
                    break;
                }
                ++h3_count;
                const uint16_t h3 = h3_of(load_u64_unaligned(base + scan_cursor * stride + key_offset));
                ++scan_cursor;
                while (scan_cursor < end) {
                    const uint64_t next_key = load_u64_unaligned(base + scan_cursor * stride + key_offset);
                    if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                        break;
                    }
                    ++scan_cursor;
                }
            }

            const uint64_t h2_end = scan_cursor;
            const uint64_t h2_span = h2_end - h2_begin;
            if (
                h2_span <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
                h2_span > static_cast<uint64_t>(config.l3_split_threshold) &&
                h3_count >= 2U
            ) {
                ExperimentalTempL3Bucket bucket;
                bucket.h2_value = h2;
                bucket.h3_values.reserve(h3_count);
                bucket.rel_starts.reserve(static_cast<size_t>(h3_count) + 1U);

                uint64_t materialize_cursor = h2_begin;
                while (materialize_cursor < h2_end) {
                    const uint16_t h3 = h3_of(load_u64_unaligned(base + materialize_cursor * stride + key_offset));
                    bucket.h3_values.push_back(h3);
                    bucket.rel_starts.push_back(static_cast<uint32_t>(materialize_cursor - h2_begin));
                    ++materialize_cursor;
                    while (materialize_cursor < h2_end) {
                        const uint64_t next_key = load_u64_unaligned(base + materialize_cursor * stride + key_offset);
                        if (h2_of(next_key) != h2 || h3_of(next_key) != h3) {
                            break;
                        }
                        ++materialize_cursor;
                    }
                }
                bucket.rel_starts.push_back(static_cast<uint32_t>(h2_span));
                node.l3_buckets.push_back(std::move(bucket));
            }

            local_cursor = h2_end;
        }

        if (node.h2_values.size() >= 2U) {
            node.rel_starts.push_back(static_cast<uint32_t>(span));
            node.active = true;
            node_slots[static_cast<size_t>(idx)] = std::move(node);
        }
    }

    std::vector<ExperimentalTempL2Node> temp_nodes;
    temp_nodes.reserve(node_slots.size());
    for (auto &slot : node_slots) {
        if (slot.active) {
            temp_nodes.push_back(std::move(slot));
        }
    }
    if (profile != nullptr) {
        const double t1 = wall_time_seconds();
        if (profile->phase2_seconds == 0.0) {
            profile->phase2_seconds = t1 - t0;
        } else {
            profile->phase3_seconds = t1 - t0;
        }
    }

    return materialize_experimental_nodes(std::move(l1_offsets), std::move(temp_nodes), config, profile);
}

Index build_experimental_impl(
    const uint64_t *keys,
    uint64_t size,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    std::vector<uint64_t> l1_offsets(kL1Size, 0U);
    if (size == 0U) {
        Index index;
        index.l1_offsets = std::move(l1_offsets);
        return index;
    }

    std::vector<uint32_t> active_h1_values;
    active_h1_values.reserve(std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    ));

    uint64_t cursor = 0U;
    uint32_t fill_from = 0U;
    const double t0 = profile ? wall_time_seconds() : 0.0;
    while (cursor < size) {
        const uint32_t h1 = h1_of(keys[cursor]);
        const uint64_t h1_begin = cursor;
        ++cursor;
        while (cursor < size && h1_of(keys[cursor]) == h1) {
            ++cursor;
        }
        std::fill(
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(h1 + 1U),
            h1_begin
        );
        fill_from = h1 + 1U;
        if (cursor - h1_begin > static_cast<uint64_t>(config.l2_split_threshold)) {
            active_h1_values.push_back(h1);
        }
    }
    std::fill(l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), l1_offsets.end(), size);
    if (profile != nullptr) {
        const double t1 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
    }

    return build_experimental_refined_impl(keys, std::move(l1_offsets), active_h1_values, config, profile);
}

Index build_experimental_parallel_l1_impl(
    const uint64_t *keys,
    uint64_t size,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    std::vector<uint64_t> l1_offsets(kL1Size, 0U);
    if (size == 0U) {
        Index index;
        index.l1_offsets = std::move(l1_offsets);
        return index;
    }

    const uint32_t first_h1 = h1_of(keys[0]);

    std::vector<H1Transition> transitions;
    const double t0 = profile ? wall_time_seconds() : 0.0;
    collect_h1_transitions_parallel(keys, size, transitions, first_h1, config);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    std::vector<uint32_t> active_h1_values;
    active_h1_values.reserve(std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    ));

    std::fill(
        l1_offsets.begin(),
        l1_offsets.begin() + static_cast<std::ptrdiff_t>(first_h1 + 1U),
        0U
    );

    uint32_t fill_from = first_h1 + 1U;
    uint32_t prev_h1 = first_h1;
    uint64_t prev_pos = 0U;
    for (const H1Transition &transition : transitions) {
        std::fill(
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(transition.h1_value + 1U),
            transition.position
        );
        fill_from = transition.h1_value + 1U;

        if (transition.position - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
            active_h1_values.push_back(prev_h1);
        }
        prev_h1 = transition.h1_value;
        prev_pos = transition.position;
    }

    if (size - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
        active_h1_values.push_back(prev_h1);
    }
    std::fill(l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), l1_offsets.end(), size);

    if (profile != nullptr) {
        const double t2 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
    }

    return build_experimental_refined_impl(keys, std::move(l1_offsets), active_h1_values, config, profile);
}

Index build_experimental_parallel_l1_delayed_l3_impl(
    const uint64_t *keys,
    uint64_t size,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    std::vector<uint64_t> l1_offsets(kL1Size, 0U);
    if (size == 0U) {
        Index index;
        index.l1_offsets = std::move(l1_offsets);
        return index;
    }

    const uint32_t first_h1 = h1_of(keys[0]);

    std::vector<H1Transition> transitions;
    const double t0 = profile ? wall_time_seconds() : 0.0;
    collect_h1_transitions_parallel(keys, size, transitions, first_h1, config);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    std::vector<uint32_t> active_h1_values;
    active_h1_values.reserve(std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    ));

    std::fill(
        l1_offsets.begin(),
        l1_offsets.begin() + static_cast<std::ptrdiff_t>(first_h1 + 1U),
        0U
    );

    uint32_t fill_from = first_h1 + 1U;
    uint32_t prev_h1 = first_h1;
    uint64_t prev_pos = 0U;
    for (const H1Transition &transition : transitions) {
        std::fill(
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(transition.h1_value + 1U),
            transition.position
        );
        fill_from = transition.h1_value + 1U;

        if (transition.position - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
            active_h1_values.push_back(prev_h1);
        }
        prev_h1 = transition.h1_value;
        prev_pos = transition.position;
    }

    if (size - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
        active_h1_values.push_back(prev_h1);
    }
    std::fill(l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), l1_offsets.end(), size);

    if (profile != nullptr) {
        const double t2 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
    }

    return build_experimental_refined_delayed_l3_impl(
        keys,
        std::move(l1_offsets),
        active_h1_values,
        config,
        profile
    );
}

Index build_experimental_parallel_l1_strided_impl(
    const unsigned char *base,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    std::vector<uint64_t> l1_offsets(kL1Size, 0U);
    if (size == 0U) {
        Index index;
        index.l1_offsets = std::move(l1_offsets);
        return index;
    }

    const uint32_t first_h1 = h1_of(load_u64_unaligned(base + key_offset));

    std::vector<H1Transition> transitions;
    const double t0 = profile ? wall_time_seconds() : 0.0;
    collect_h1_transitions_parallel_strided(base, size, stride, key_offset, transitions, first_h1, config);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    std::vector<uint32_t> active_h1_values;
    active_h1_values.reserve(std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    ));

    std::fill(
        l1_offsets.begin(),
        l1_offsets.begin() + static_cast<std::ptrdiff_t>(first_h1 + 1U),
        0U
    );

    uint32_t fill_from = first_h1 + 1U;
    uint32_t prev_h1 = first_h1;
    uint64_t prev_pos = 0U;
    for (const H1Transition &transition : transitions) {
        std::fill(
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(transition.h1_value + 1U),
            transition.position
        );
        fill_from = transition.h1_value + 1U;

        if (transition.position - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
            active_h1_values.push_back(prev_h1);
        }
        prev_h1 = transition.h1_value;
        prev_pos = transition.position;
    }

    if (size - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
        active_h1_values.push_back(prev_h1);
    }
    std::fill(l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), l1_offsets.end(), size);

    if (profile != nullptr) {
        const double t2 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
    }

    return build_experimental_refined_strided_impl(
        base,
        stride,
        key_offset,
        std::move(l1_offsets),
        active_h1_values,
        config,
        profile
    );
}

Index build_experimental_parallel_l1_delayed_l3_strided_impl(
    const unsigned char *base,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    const Config &config,
    BuildProfile *profile = nullptr
) {
    std::vector<uint64_t> l1_offsets(kL1Size, 0U);
    if (size == 0U) {
        Index index;
        index.l1_offsets = std::move(l1_offsets);
        return index;
    }

    const uint32_t first_h1 = h1_of(load_u64_unaligned(base + key_offset));

    std::vector<H1Transition> transitions;
    const double t0 = profile ? wall_time_seconds() : 0.0;
    collect_h1_transitions_parallel_strided(base, size, stride, key_offset, transitions, first_h1, config);
    const double t1 = profile ? wall_time_seconds() : 0.0;

    std::vector<uint32_t> active_h1_values;
    active_h1_values.reserve(std::max<uint64_t>(
        1024U,
        size / static_cast<uint64_t>(std::max(config.l2_split_threshold, 1U))
    ));

    std::fill(
        l1_offsets.begin(),
        l1_offsets.begin() + static_cast<std::ptrdiff_t>(first_h1 + 1U),
        0U
    );

    uint32_t fill_from = first_h1 + 1U;
    uint32_t prev_h1 = first_h1;
    uint64_t prev_pos = 0U;
    for (const H1Transition &transition : transitions) {
        std::fill(
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from),
            l1_offsets.begin() + static_cast<std::ptrdiff_t>(transition.h1_value + 1U),
            transition.position
        );
        fill_from = transition.h1_value + 1U;

        if (transition.position - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
            active_h1_values.push_back(prev_h1);
        }
        prev_h1 = transition.h1_value;
        prev_pos = transition.position;
    }

    if (size - prev_pos > static_cast<uint64_t>(config.l2_split_threshold)) {
        active_h1_values.push_back(prev_h1);
    }
    std::fill(l1_offsets.begin() + static_cast<std::ptrdiff_t>(fill_from), l1_offsets.end(), size);

    if (profile != nullptr) {
        const double t2 = wall_time_seconds();
        profile->phase1_seconds = t1 - t0;
        profile->phase2_seconds = t2 - t1;
    }

    return build_experimental_refined_delayed_l3_strided_impl(
        base,
        stride,
        key_offset,
        std::move(l1_offsets),
        active_h1_values,
        config,
        profile
    );
}

} // namespace

bool Index::empty() const {
    return l1_offsets.empty() || l1_offsets.back() == 0U;
}

Range Index::locate(uint64_t key) const {
    if (l1_offsets.empty()) {
        return {};
    }

    const uint32_t h1 = h1_of(key);
    Range range{l1_offsets[h1], l1_offsets[h1 + 1U]};
    if (range.empty() || !root_bit_test(*this, h1)) {
        return range;
    }

    const uint32_t l2_index = root_rank_before(*this, h1);
    const L2Node &l2_node = l2_nodes[l2_index];
    const uint16_t h2 = h2_of(key);
    const uint32_t h2_rank = sub_rank_before(l2_node.h2_map, h2);
    const uint64_t l2_begin = l2_node.parent_begin + l2_relative_starts[l2_node.starts_offset + h2_rank];

    if (!sub_bit_test(l2_node.h2_map, h2)) {
        return {l2_begin, l2_begin};
    }

    const uint64_t l2_end = l2_node.parent_begin + l2_relative_starts[l2_node.starts_offset + h2_rank + 1U];
    range = {l2_begin, l2_end};
    if (range.empty() || !sub_bit_test(l2_node.l3_map, h2)) {
        return range;
    }

    const uint32_t l3_index = sub_rank_before(l2_node.l3_map, h2);
    const L3Node &l3_node = l3_nodes[l2_node.l3_nodes_offset + l3_index];
    const uint16_t h3 = h3_of(key);
    const uint32_t h3_rank = sub_rank_before(l3_node.h3_map, h3);
    const uint64_t l3_begin = range.begin + l3_relative_starts[l3_node.starts_offset + h3_rank];

    if (!sub_bit_test(l3_node.h3_map, h3)) {
        return {l3_begin, l3_begin};
    }

    const uint64_t l3_end = range.begin + l3_relative_starts[l3_node.starts_offset + h3_rank + 1U];
    return {l3_begin, l3_end};
}

Index build(const std::vector<uint64_t> &keys, const Config &config) {
    return build(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

Index build(const uint64_t *keys, uint64_t size, const Config &config) {
    if (size == 0U || keys == nullptr) {
        return {};
    }
    return build_contiguous_impl(keys, size, config);
}

Index build_strided(const void *records, uint64_t size, uint64_t stride, uint64_t key_offset, const Config &config) {
    if (size == 0U || records == nullptr) {
        return {};
    }
    const auto *base = static_cast<const unsigned char *>(records);
    if (stride == sizeof(uint64_t) && key_offset == 0U) {
        return build_contiguous_impl(reinterpret_cast<const uint64_t *>(records), size, config);
    }
    return build_strided_impl(base, size, stride, key_offset, config);
}

Index build_experimental_hybrid_strided(
    const void *records,
    uint64_t size,
    uint64_t stride,
    uint64_t key_offset,
    const Config &config
) {
    if (size == 0U || records == nullptr) {
        return {};
    }
    const auto *base = static_cast<const unsigned char *>(records);
    if (size > kExperimentalHybridDelayedL3Threshold) {
        return build_experimental_parallel_l1_strided_impl(base, size, stride, key_offset, config);
    }
    return build_experimental_parallel_l1_delayed_l3_strided_impl(base, size, stride, key_offset, config);
}

Index build_experimental(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

Index build_experimental(const uint64_t *keys, uint64_t size, const Config &config) {
    if (size == 0U || keys == nullptr) {
        return {};
    }
    return build_experimental_impl(keys, size, config);
}

Index build_experimental_parallel_l1(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_parallel_l1(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

Index build_experimental_parallel_l1(const uint64_t *keys, uint64_t size, const Config &config) {
    if (size == 0U || keys == nullptr) {
        return {};
    }
    return build_experimental_parallel_l1_impl(keys, size, config);
}

Index build_experimental_parallel_l1_delayed_l3(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_parallel_l1_delayed_l3(
        keys.data(),
        static_cast<uint64_t>(keys.size()),
        config
    );
}

Index build_experimental_parallel_l1_delayed_l3(const uint64_t *keys, uint64_t size, const Config &config) {
    if (size == 0U || keys == nullptr) {
        return {};
    }
    return build_experimental_parallel_l1_delayed_l3_impl(keys, size, config);
}

Index build_experimental_hybrid(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_hybrid(
        keys.data(),
        static_cast<uint64_t>(keys.size()),
        config
    );
}

Index build_experimental_hybrid(const uint64_t *keys, uint64_t size, const Config &config) {
    if (size == 0U || keys == nullptr) {
        return {};
    }
    if (size > kExperimentalHybridDelayedL3Threshold) {
        return build_experimental_parallel_l1_impl(keys, size, config);
    }
    return build_experimental_parallel_l1_delayed_l3_impl(keys, size, config);
}

ProfiledBuild build_profiled(const std::vector<uint64_t> &keys, const Config &config) {
    return build_profiled(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

ProfiledBuild build_profiled(const uint64_t *keys, uint64_t size, const Config &config) {
    ProfiledBuild result;
    if (size == 0U || keys == nullptr) {
        return result;
    }
    result.index = build_contiguous_impl(keys, size, config, &result.profile);
    return result;
}

ProfiledBuild build_experimental_profiled(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_profiled(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

ProfiledBuild build_experimental_profiled(const uint64_t *keys, uint64_t size, const Config &config) {
    ProfiledBuild result;
    if (size == 0U || keys == nullptr) {
        return result;
    }
    result.index = build_experimental_impl(keys, size, config, &result.profile);
    return result;
}

ProfiledBuild build_experimental_parallel_l1_profiled(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_parallel_l1_profiled(keys.data(), static_cast<uint64_t>(keys.size()), config);
}

ProfiledBuild build_experimental_parallel_l1_profiled(const uint64_t *keys, uint64_t size, const Config &config) {
    ProfiledBuild result;
    if (size == 0U || keys == nullptr) {
        return result;
    }
    result.index = build_experimental_parallel_l1_impl(keys, size, config, &result.profile);
    return result;
}

ProfiledBuild build_experimental_parallel_l1_delayed_l3_profiled(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_parallel_l1_delayed_l3_profiled(
        keys.data(),
        static_cast<uint64_t>(keys.size()),
        config
    );
}

ProfiledBuild build_experimental_parallel_l1_delayed_l3_profiled(
    const uint64_t *keys,
    uint64_t size,
    const Config &config
) {
    ProfiledBuild result;
    if (size == 0U || keys == nullptr) {
        return result;
    }
    result.index = build_experimental_parallel_l1_delayed_l3_impl(keys, size, config, &result.profile);
    return result;
}

ProfiledBuild build_experimental_hybrid_profiled(const std::vector<uint64_t> &keys, const Config &config) {
    return build_experimental_hybrid_profiled(
        keys.data(),
        static_cast<uint64_t>(keys.size()),
        config
    );
}

ProfiledBuild build_experimental_hybrid_profiled(
    const uint64_t *keys,
    uint64_t size,
    const Config &config
) {
    ProfiledBuild result;
    if (size == 0U || keys == nullptr) {
        return result;
    }
    if (size > kExperimentalHybridDelayedL3Threshold) {
        result.index = build_experimental_parallel_l1_impl(keys, size, config, &result.profile);
    } else {
        result.index = build_experimental_parallel_l1_delayed_l3_impl(keys, size, config, &result.profile);
    }
    return result;
}

Range locate(const Index &index, uint64_t key) {
    return index.locate(key);
}

} // namespace AdaptiveIndex
