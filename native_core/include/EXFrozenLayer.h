#pragma once

#include "AdaptiveIndex.h"
#include "FormationRuntime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ZMaskFrozen {

constexpr int kPrefixBits = 40;
constexpr int kSuffixBits = 24;
constexpr int kZeroMaskBits = 6;
constexpr uint32_t kZeroMaskCount = 1u << kZeroMaskBits;
constexpr uint32_t kSuffixStateCount = 1u << kSuffixBits;
constexpr uint32_t kMaxSum = 6u * (1u << 15);
constexpr uint16_t kInvalidRank = std::numeric_limits<uint16_t>::max();
constexpr uint32_t kEstimatedLargeSuccessStride = 30U;
constexpr std::array<uint32_t, 4> kEstimatedLargeSuccessStrides = {8U, 16U, 30U, 32U};

struct BucketDirectEntry {
    uint64_t key = 0;
    uint32_t bitmap_offset = 0;
    uint32_t success_offset = 0;
};

struct BucketDirectIndex {
    static constexpr uint32_t kEmptyBucketIndex = std::numeric_limits<uint32_t>::max();
    uint64_t table_size = 0;
    std::vector<uint32_t> bucket_indices;

    [[nodiscard]] bool empty() const {
        return bucket_indices.empty();
    }
};

struct TileLimitConfig {
    std::array<int8_t, 16> max_counts{};
    std::vector<uint32_t> valid_suffix_masks;
    uint32_t required_suffix24 = 0;
};

struct ThresholdStats {
    uint32_t threshold_bits = 0;
    uint64_t small_bytes = 0;
    uint64_t large_bytes = 0;
    uint64_t abs_diff_bytes = 0;
    uint64_t aligned_total_bits = 0;
};

struct ZMaskLuts {
    std::vector<uint16_t> rank_table;
    std::vector<uint16_t> size_table;
    std::vector<uint32_t> offset_table;
    std::vector<uint32_t> unrank_array;
    std::vector<uint32_t> row16_sum;
    std::vector<uint32_t> suffix_group_table;
};

struct PreparedDirectQuery {
    uint64_t target_direct_key46 = 0;
    uint32_t valid_count = 0;
    uint16_t rank = kInvalidRank;
    bool valid = false;
};

template <typename T>
struct Layer {
    SuccessRateKind kind = SuccessRateKindMap<T>::value;
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 0;
    uint64_t large_success_base = 0;
    uint32_t large_success_stride = 1;
    uint64_t large_success_dense_count = 0;
    uint64_t large_success_padded_count = 0;
    uint64_t large_success_padding_count = 0;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    std::vector<uint64_t> bucket_keys;
    std::vector<uint32_t> bitmap_offsets;
    std::vector<uint32_t> success_offsets;
    std::vector<BucketDirectEntry> bucket_entries;
    std::vector<uint8_t> small_bitmap_bytes;
    std::vector<uint64_t> large_bitmap_words;
    std::vector<T> success_values;
    AdaptiveIndex::Index index;
    BucketDirectIndex direct_index;

    [[nodiscard]] bool empty() const {
        return bucket_keys.empty();
    }
};

template <typename T>
struct LookupResult {
    T success{};
    uint64_t global_dense_index = 0;
    bool found = false;
};

bool tile_limit_configs_equal(const TileLimitConfig &lhs, const TileLimitConfig &rhs);
TileLimitConfig make_target_tile_limit_config(int target_exponent);
TileLimitConfig make_lut_tile_limit_config(
    int target_exponent,
    const std::vector<uint64_t> &seed_boards,
    const PatternSpec &spec,
    bool is_free,
    bool is_variant
);
uint32_t tile_value(uint32_t tile);
bool decode_suffix24_if_valid(uint32_t state, uint32_t &sum, const TileLimitConfig &config);
uint8_t suffix24_zero_mask(uint32_t state);
uint64_t pack_bucket_key(uint64_t prefix40, uint8_t zero_mask, uint32_t remaining_sum);
uint64_t pack_direct_lookup_key46(uint64_t prefix40, uint8_t zero_mask);
uint64_t bucket_key_prefix40(uint64_t key);
uint8_t bucket_key_zero_mask(uint64_t key);
uint32_t bucket_key_remaining_sum(uint64_t key);
uint64_t bucket_key_direct_lookup_key46(uint64_t key);
uint32_t sum_index(uint32_t sum);
uint32_t lut_group_index(uint32_t sum, uint8_t zero_mask);
uint64_t bytes_for_bits(uint32_t bits);
uint64_t words_for_bits(uint32_t bits);
uint64_t round_up_multiple_u64(uint64_t value, uint32_t multiple);
uint32_t choose_large_success_stride(uint64_t dense_count);
uint64_t large_success_partition_base(uint64_t large_success_base, uint32_t success_offset, uint32_t stride);
uint32_t board_layer_sum(uint64_t board, const ZMaskLuts &luts);
uint32_t prefix40_sum(uint64_t prefix40, const ZMaskLuts &luts);
uint32_t prefix40_remaining_sum(uint64_t prefix40, uint32_t layer_sum, const ZMaskLuts &luts);
ZMaskLuts build_zmask_luts(const TileLimitConfig &config, int num_threads = 8);
void initialize_runtime_tables(ZMaskLuts &luts);
ThresholdStats choose_best_threshold(const std::vector<ThresholdStats> &scan);
bool test_small_bit(const std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank);
bool test_large_bit(const std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank);
void set_small_bit(std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank);
void set_large_bit(std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank);

template <typename T>
Layer<T> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<T> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
);

template <typename T>
void build_index(Layer<T> &layer, int num_threads);

template <typename T>
LookupResult<T> lookup_success_and_index(
    const Layer<T> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    T zero_value
);

template <typename T>
T lookup_success(
    const Layer<T> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    T zero_value
);

PreparedDirectQuery prepare_direct_suffix_query(const ZMaskLuts &luts, uint64_t board);

template <typename T>
bool lookup_direct_success_index_prepared(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
);

template <typename T>
void lookup_direct_success_indices_prepared_batch(
    const Layer<T> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
);

template <typename T>
T lookup_direct_prepared(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    T zero_value
);

template <typename T>
uint64_t live_count_by_bitmap(
    const Layer<T> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
);

extern template Layer<uint32_t> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<uint32_t> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
);
extern template Layer<uint64_t> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<uint64_t> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
);
extern template Layer<float> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<float> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
);
extern template Layer<double> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<double> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
);

extern template void build_index(Layer<uint32_t> &layer, int num_threads);
extern template void build_index(Layer<uint64_t> &layer, int num_threads);
extern template void build_index(Layer<float> &layer, int num_threads);
extern template void build_index(Layer<double> &layer, int num_threads);

extern template LookupResult<uint32_t> lookup_success_and_index(
    const Layer<uint32_t> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    uint32_t zero_value
);
extern template LookupResult<uint64_t> lookup_success_and_index(
    const Layer<uint64_t> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    uint64_t zero_value
);
extern template LookupResult<float> lookup_success_and_index(
    const Layer<float> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    float zero_value
);
extern template LookupResult<double> lookup_success_and_index(
    const Layer<double> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    double zero_value
);

extern template uint32_t lookup_success(
    const Layer<uint32_t> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    uint32_t zero_value
);
extern template uint64_t lookup_success(
    const Layer<uint64_t> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    uint64_t zero_value
);
extern template float lookup_success(
    const Layer<float> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    float zero_value
);
extern template double lookup_success(
    const Layer<double> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    double zero_value
);

extern template bool lookup_direct_success_index_prepared(
    const Layer<uint32_t> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
);
extern template bool lookup_direct_success_index_prepared(
    const Layer<uint64_t> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
);
extern template bool lookup_direct_success_index_prepared(
    const Layer<float> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
);
extern template bool lookup_direct_success_index_prepared(
    const Layer<double> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
);

extern template void lookup_direct_success_indices_prepared_batch(
    const Layer<uint32_t> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
);
extern template void lookup_direct_success_indices_prepared_batch(
    const Layer<uint64_t> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
);
extern template void lookup_direct_success_indices_prepared_batch(
    const Layer<float> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
);
extern template void lookup_direct_success_indices_prepared_batch(
    const Layer<double> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
);

extern template uint32_t lookup_direct_prepared(
    const Layer<uint32_t> &layer,
    const PreparedDirectQuery &query,
    uint32_t zero_value
);
extern template uint64_t lookup_direct_prepared(
    const Layer<uint64_t> &layer,
    const PreparedDirectQuery &query,
    uint64_t zero_value
);
extern template float lookup_direct_prepared(
    const Layer<float> &layer,
    const PreparedDirectQuery &query,
    float zero_value
);
extern template double lookup_direct_prepared(
    const Layer<double> &layer,
    const PreparedDirectQuery &query,
    double zero_value
);

extern template uint64_t live_count_by_bitmap(
    const Layer<uint32_t> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
);
extern template uint64_t live_count_by_bitmap(
    const Layer<uint64_t> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
);
extern template uint64_t live_count_by_bitmap(
    const Layer<float> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
);
extern template uint64_t live_count_by_bitmap(
    const Layer<double> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
);

} // namespace ZMaskFrozen
