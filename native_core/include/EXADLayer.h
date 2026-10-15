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
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
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
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
    std::array<BoardSet, bucket_slot_count()> sets{};

    [[nodiscard]] bool empty() const {
        return live_board_count == 0;
    }
};

inline bool physical_metadata_matches(const Layer &layer, const Luts &luts) {
    return layer.physical_transform == luts.physical_transform &&
        layer.inverse_physical_transform == luts.inverse_physical_transform &&
        layer.logical_pattern_signature == luts.logical_pattern_signature &&
        layer.physical_pattern_signature == luts.physical_pattern_signature;
}

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

uint64_t pack_bucket_key(uint64_t prefix36, uint32_t semantic_suffix_sum);
uint64_t bucket_key_prefix36(uint64_t key);
uint32_t bucket_key_semantic_sum(uint64_t key);
uint32_t lut_group_index(uint32_t semantic_suffix_sum);
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




} // namespace EXAD
