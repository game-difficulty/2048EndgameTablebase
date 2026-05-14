#pragma once

#include "BoardMaskerAD.h"
#include "EXFrozenLayer.h"
#include "FormationRuntime.h"

#include <array>
#include <cstdint>
#include <vector>

namespace EXAD {

constexpr uint32_t kDefaultThresholdBits = 240U;
constexpr uint64_t kInvalidBucketKey = ~0ULL;
constexpr uint32_t kPrefixBits = 36U;
constexpr uint32_t kSuffixBits = 28U;
constexpr uint64_t kSuffixMask = (1ULL << kSuffixBits) - 1ULL;
constexpr uint32_t kLow24StateCount = 1U << 24U;
constexpr uint32_t kMaxSemanticSuffixSum = 7U * (1U << 15U);
constexpr uint32_t kSemanticGroupCount = (kMaxSemanticSuffixSum >> 1U) + 1U;
constexpr uint32_t kInvalidSuffix28 = 0xFFFFFFFFU;

struct Luts {
    ZMaskFrozen::TileLimitConfig config;
    uint64_t config_signature = 0;
    std::vector<std::vector<uint16_t>> rank_tables;
    std::vector<uint32_t> packed_rank_pair_table;
    std::vector<uint32_t> size_table;
    std::vector<uint32_t> offset_table;
    std::vector<uint32_t> unrank_array;
    std::vector<uint16_t> high_base;
    std::array<uint8_t, 16> table_for_high{};
    std::vector<uint32_t> row16_sum;
    uint8_t packed_table0 = 0xFFU;
    uint8_t packed_table1 = 0xFFU;
    uint64_t valid_suffix_count = 0;
};

struct BucketEntry {
    uint64_t key = 0;
    uint32_t bitmap_offset = 0;
    uint32_t dense_offset = 0;
};

struct BoardSet {
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    std::vector<BucketEntry> buckets;
    std::vector<uint8_t> small_bitmap_bytes;
    std::vector<uint64_t> large_bitmap_words;

    [[nodiscard]] bool empty() const {
        return buckets.empty();
    }
};

struct Layer {
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint64_t lut_signature = 0;
    uint64_t live_board_count = 0;
    std::array<BoardSet, bucket_slot_count()> sets{};

    [[nodiscard]] bool empty() const {
        return live_board_count == 0;
    }
};

uint32_t tile_value(uint32_t tile);
uint32_t sum_index(uint32_t sum);
uint64_t config_signature(const ZMaskFrozen::TileLimitConfig &config);
ZMaskFrozen::TileLimitConfig make_exad_lut_tile_limit_config(
    int target_exponent,
    const std::vector<uint64_t> &seed_boards,
    const PatternSpec &spec,
    bool is_free,
    bool is_variant
);
Luts build_luts(const ZMaskFrozen::TileLimitConfig &config, int num_threads = 8);
void initialize_runtime_tables(Luts &luts);

uint32_t semantic_suffix_sum(uint32_t suffix28, const Luts &luts);
uint64_t pack_bucket_key(uint64_t prefix36, uint32_t semantic_suffix_sum);
uint64_t bucket_key_prefix36(uint64_t key);
uint32_t bucket_key_semantic_sum(uint64_t key);
uint32_t lut_group_index(uint32_t semantic_suffix_sum);
bool suffix28_rank_group(const Luts &luts, uint32_t suffix28, uint32_t &group, uint32_t &rank);
bool suffix28_rank_group_sum(
    const Luts &luts,
    uint32_t suffix28,
    uint32_t &group,
    uint32_t &rank,
    uint32_t &semantic_sum
);

inline uint32_t low24_sum_from_row_lut_fast(const Luts &luts, uint32_t low24) {
    return luts.row16_sum[low24 & 0xFFFFU] +
        luts.row16_sum[((low24 >> 16U) & 0xFFU) << 8U];
}

inline uint32_t low24_group_from_row_lut_fast(const Luts &luts, uint32_t low24) {
    return low24_sum_from_row_lut_fast(luts, low24) >> 1U;
}

inline uint32_t tile_half_value_fast(uint32_t tile) {
    static constexpr std::array<uint32_t, 16> kHalfValues = {
        0U, 1U, 2U, 4U, 8U, 16U, 32U, 64U,
        128U, 256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U
    };
    return kHalfValues[tile & 0xFU];
}

inline uint16_t low24_rank_for_table_fast(const Luts &luts, uint8_t table_id, uint32_t low24) {
    if (!luts.packed_rank_pair_table.empty() &&
        (table_id == luts.packed_table0 || table_id == luts.packed_table1)) {
        const uint32_t packed = luts.packed_rank_pair_table[low24];
        return table_id == luts.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
    }
    if (table_id >= luts.rank_tables.size()) {
        return ZMaskFrozen::kInvalidRank;
    }
    return luts.rank_tables[table_id][low24];
}

inline bool suffix28_rank_group_sum_fast(
    const Luts &luts,
    uint32_t suffix28,
    uint32_t &group,
    uint32_t &rank,
    uint32_t &semantic_sum
) {
    constexpr uint8_t kInvalidTableId = 0xFFU;
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = luts.table_for_high[high];
    if (table_id == kInvalidTableId) {
        return false;
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    const uint16_t low_rank = low24_rank_for_table_fast(luts, table_id, low24);
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return false;
    }
    group = low24_group_from_row_lut_fast(luts, low24) + tile_half_value_fast(high);
    if (group >= luts.size_table.size()) {
        return false;
    }
    semantic_sum = group << 1U;
    rank = luts.high_base[static_cast<size_t>(high) * kSemanticGroupCount + group] + low_rank;
    return rank < luts.size_table[group];
}

int8_t ad_bucket_key_for_board(
    uint64_t board,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param
);

uint64_t live_count_by_bitmap(
    const BoardSet &set,
    uint32_t bucket_index,
    const Luts &luts
);

template <typename Fn>
void for_each_live_board(const Layer &layer, const Luts &luts, Fn &&fn) {
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        const BoardSet &set = layer.sets[slot];
        const int ad_key = bucket_key_min() + static_cast<int>(slot);
        for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(set.buckets.size()); ++bucket_idx) {
            const BucketEntry &bucket = set.buckets[bucket_idx];
            const uint64_t key = bucket.key;
            const uint64_t prefix36 = bucket_key_prefix36(key);
            const uint32_t semantic_sum = bucket_key_semantic_sum(key);
            const uint32_t group = lut_group_index(semantic_sum);
            const uint32_t valid_count = luts.size_table[group];
            const uint32_t unrank_base = luts.offset_table[group];
            const uint32_t bitmap_offset = bucket.bitmap_offset;
            if (valid_count <= set.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = set.small_bitmap_bytes[bitmap_offset + byte_idx];
                    while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit_idx = static_cast<uint32_t>(__builtin_ctz(value));
#else
                        uint32_t bit_idx = 0;
                        while (((value >> bit_idx) & 1U) == 0U) {
                            ++bit_idx;
                        }
#endif
                        const uint32_t rank = byte_idx * 8U + bit_idx;
                        if (rank >= valid_count) {
                            break;
                        }
                        const uint64_t board = (prefix36 << 28U) | luts.unrank_array[unrank_base + rank];
                        fn(static_cast<int8_t>(ad_key), board);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = set.large_bitmap_words[bitmap_offset + word_idx];
                    while (value != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit_idx = static_cast<uint32_t>(__builtin_ctzll(value));
#else
                        uint32_t bit_idx = 0;
                        while (((value >> bit_idx) & 1ULL) == 0ULL) {
                            ++bit_idx;
                        }
#endif
                        const uint32_t rank = word_idx * 64U + bit_idx;
                        if (rank >= valid_count) {
                            break;
                        }
                        const uint64_t board = (prefix36 << 28U) | luts.unrank_array[unrank_base + rank];
                        fn(static_cast<int8_t>(ad_key), board);
                        value &= (value - 1ULL);
                    }
                }
            }
        }
    }
}

std::vector<uint64_t> extract_boards_sorted(const Layer &layer, const Luts &luts, int num_threads);

} // namespace EXAD
