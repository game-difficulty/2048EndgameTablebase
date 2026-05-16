#pragma once

#include "FormationRuntime.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ZMaskFrozen {
struct TileLimitConfig;
}

namespace Prefix40Baseline {

constexpr int kPrefixBits = 40;
constexpr int kSuffixBits = 24;
constexpr uint32_t kSuffixStateCount = 1u << kSuffixBits;
constexpr uint32_t kMaxSum = 6u * (1u << 15);
constexpr uint16_t kInvalidRank = 0xFFFFu;
constexpr uint32_t kDefaultThresholdBits = 240u;

struct Luts {
    std::vector<uint16_t> rank_table;
    std::vector<uint16_t> size_table;
    std::vector<uint32_t> offset_table;
    std::vector<uint32_t> unrank_array;
    std::vector<uint32_t> row16_sum;
};

struct Layer {
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    std::vector<uint64_t> bucket_keys;
    std::vector<uint32_t> bitmap_offsets;
    std::vector<uint32_t> dense_offsets;
    std::vector<uint8_t> small_bitmap_bytes;
    std::vector<uint64_t> large_bitmap_words;

    [[nodiscard]] bool empty() const {
        return bucket_keys.empty();
    }
};

uint32_t tile_value(uint32_t tile);
uint32_t sum_index(uint32_t sum);
uint64_t bytes_for_bits(uint32_t bits);
uint64_t words_for_bits(uint32_t bits);
uint32_t board_layer_sum(uint64_t board, const Luts &luts);
uint32_t prefix40_sum(uint64_t prefix40, const Luts &luts);
uint32_t prefix40_remaining_sum(uint64_t prefix40, uint32_t layer_sum, const Luts &luts);

uint64_t pack_bucket_key(uint64_t prefix40, uint32_t remaining_sum);
uint64_t bucket_key_prefix40(uint64_t key);
uint32_t bucket_key_remaining_sum(uint64_t key);

Luts build_luts(const ZMaskFrozen::TileLimitConfig &config, int num_threads = 8);

void set_small_bit(std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank);
void set_large_bit(std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank);

Layer build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const Luts &luts,
    int num_threads
);

} // namespace Prefix40Baseline
