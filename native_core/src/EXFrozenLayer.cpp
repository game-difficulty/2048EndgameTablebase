#include "EXFrozenLayer.h"

#include "HybridSearch.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ZMaskFrozen {

namespace {

struct BucketPlan {
    uint64_t key = 0;
    uint32_t valid_count = 0;
    uint32_t live_count = 0;
    bool is_small = false;
};

template <typename T>
struct RankValue {
    uint16_t rank = 0;
    T success{};
};

inline int effective_threads(int requested) {
#if defined(_OPENMP)
    if (requested > 0) {
        return requested;
    }
    return std::max(1, omp_get_max_threads());
#else
    return requested > 0 ? requested : 1;
#endif
}

inline uint32_t countr_zero_u32(uint32_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return static_cast<uint32_t>(index);
#else
    return static_cast<uint32_t>(__builtin_ctz(value));
#endif
}

inline uint32_t countr_zero_u64(uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<uint32_t>(index);
#elif defined(_MSC_VER)
    const uint32_t lo = static_cast<uint32_t>(value);
    if (lo != 0U) {
        return countr_zero_u32(lo);
    }
    return 32U + countr_zero_u32(static_cast<uint32_t>(value >> 32U));
#else
    return static_cast<uint32_t>(__builtin_ctzll(value));
#endif
}

inline uint32_t popcount_u32(uint32_t value) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt(value));
#else
    return static_cast<uint32_t>(__builtin_popcount(value));
#endif
}

inline uint32_t popcount_u64(uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<uint32_t>(__popcnt64(value));
#elif defined(_MSC_VER)
    return popcount_u32(static_cast<uint32_t>(value)) + popcount_u32(static_cast<uint32_t>(value >> 32U));
#else
    return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
}

inline uint64_t mix_u64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

inline bool is_prime_u64(uint64_t value) {
    if (value < 2ULL) {
        return false;
    }
    if ((value & 1ULL) == 0ULL) {
        return value == 2ULL;
    }
    if (value % 3ULL == 0ULL) {
        return value == 3ULL;
    }
    for (uint64_t divisor = 5ULL; divisor <= value / divisor; divisor += 6ULL) {
        if (value % divisor == 0ULL || value % (divisor + 2ULL) == 0ULL) {
            return false;
        }
    }
    return true;
}

inline uint64_t next_prime_u64(uint64_t value) {
    if (value <= 2ULL) {
        return 2ULL;
    }
    if ((value & 1ULL) == 0ULL) {
        ++value;
    }
    while (!is_prime_u64(value)) {
        value += 2ULL;
    }
    return value;
}

inline uint64_t choose_bucket_index_direct_table_size(uint64_t bucket_count) {
    constexpr uint64_t kMinTableSize = 1024ULL;
    constexpr uint64_t kLoadNumerator = 55ULL;
    constexpr uint64_t kLoadDenominator = 100ULL;
    const uint64_t required =
        (bucket_count * kLoadDenominator + (kLoadNumerator - 1ULL)) / kLoadNumerator;
    return next_prime_u64(std::max<uint64_t>(kMinTableSize, required));
}

inline uint64_t direct_hash_slot(const BucketDirectIndex &index, uint64_t key) {
    const uint64_t mixed = mix_u64(key);
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(mixed) * index.table_size) >> 64U);
#else
    return mixed % index.table_size;
#endif
}

inline uint64_t direct_next_slot(const BucketDirectIndex &index, uint64_t slot) {
    ++slot;
    return slot == index.table_size ? 0ULL : slot;
}

inline uint64_t direct_lookup_key46_from_bucket_key(uint64_t key) {
    return key >> 18U;
}

template <typename T>
inline void ensure_bucket_entries(Layer<T> &layer) {
    if (layer.bucket_entries.size() == layer.bucket_keys.size()) {
        return;
    }
    layer.bucket_entries.resize(layer.bucket_keys.size());
    for (size_t i = 0; i < layer.bucket_keys.size(); ++i) {
        layer.bucket_entries[i] = BucketDirectEntry{
            layer.bucket_keys[i],
            layer.bitmap_offsets[i],
            layer.success_offsets[i]
        };
    }
}

inline bool suffix_tile_limit_allows(const TileLimitConfig &config, uint32_t tile, uint8_t new_count) {
    const int8_t limit = config.max_counts[tile];
    return limit < 0 || new_count <= static_cast<uint8_t>(limit);
}

inline bool suffix_structural_constraints_allow(const TileLimitConfig &config, uint32_t state) {
    if ((state & config.required_suffix24) != config.required_suffix24) {
        return false;
    }
    if (config.valid_suffix_masks.empty()) {
        return true;
    }
    for (const uint32_t mask : config.valid_suffix_masks) {
        if ((state & mask) == mask) {
            return true;
        }
    }
    return false;
}

inline void validate_sorted_boards(const std::vector<uint64_t> &boards, const ZMaskLuts &luts) {
    if (boards.empty()) {
        return;
    }
    uint64_t previous = boards.front();
    const uint32_t layer_sum = board_layer_sum(previous, luts);
    const uint32_t first_suffix = static_cast<uint32_t>(previous & 0xFFFFFFULL);
    if (luts.rank_table[first_suffix] == kInvalidRank) {
        throw std::runtime_error("board suffix is disallowed by zmask LUT");
    }
    for (size_t i = 1; i < boards.size(); ++i) {
        const uint64_t current = boards[i];
        if (current <= previous) {
            throw std::runtime_error("boards must be strictly increasing");
        }
        if (board_layer_sum(current, luts) != layer_sum) {
            throw std::runtime_error("boards span multiple layer sums");
        }
        const uint32_t suffix24 = static_cast<uint32_t>(current & 0xFFFFFFULL);
        if (luts.rank_table[suffix24] == kInvalidRank) {
            throw std::runtime_error("board suffix is disallowed by zmask LUT");
        }
        previous = current;
    }
}

inline uint32_t dense_ordinal_small(const std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t byte_idx = rank >> 3U;
    uint32_t total = 0U;
    for (uint32_t i = 0; i < byte_idx; ++i) {
        total += popcount_u32(bitmap[offset + i]);
    }
    const uint32_t bit_idx = rank & 7U;
    if (bit_idx != 0U) {
        const uint32_t mask = (1U << bit_idx) - 1U;
        total += popcount_u32(static_cast<uint32_t>(bitmap[offset + byte_idx] & static_cast<uint8_t>(mask)));
    }
    return total;
}

inline uint32_t dense_ordinal_large(const std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t word_idx = rank >> 6U;
    uint32_t total = 0U;
    for (uint32_t i = 0; i < word_idx; ++i) {
        total += popcount_u64(bitmap[offset + i]);
    }
    const uint32_t bit_idx = rank & 63U;
    if (bit_idx != 0U) {
        const uint64_t mask = (bit_idx == 64U) ? ~0ULL : ((1ULL << bit_idx) - 1ULL);
        total += popcount_u64(bitmap[offset + word_idx] & mask);
    }
    return total;
}

template <typename T>
inline bool direct_lookup_bucket_offsets(
    const Layer<T> &layer,
    uint64_t target_direct_key46,
    uint32_t &bitmap_offset,
    uint32_t &success_offset
) {
    const BucketDirectIndex &index = layer.direct_index;
    uint64_t slot = direct_hash_slot(index, target_direct_key46);
    while (true) {
        const uint32_t bucket_index = index.bucket_indices[static_cast<size_t>(slot)];
        if (bucket_index == BucketDirectIndex::kEmptyBucketIndex) {
            return false;
        }
        const BucketDirectEntry &bucket = layer.bucket_entries[static_cast<size_t>(bucket_index)];
        if (direct_lookup_key46_from_bucket_key(bucket.key) == target_direct_key46) {
            bitmap_offset = bucket.bitmap_offset;
            success_offset = bucket.success_offset;
            return true;
        }
        slot = direct_next_slot(index, slot);
    }
}

template <typename T>
inline bool success_index_from_offsets(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    uint32_t bitmap_offset,
    uint32_t success_offset,
    uint64_t &success_index
) {
    const bool is_small = query.valid_count <= layer.threshold_bits;
    if (is_small) {
        if (!test_small_bit(layer.small_bitmap_bytes, bitmap_offset, query.rank)) {
            return false;
        }
        const uint32_t dense_ordinal = dense_ordinal_small(layer.small_bitmap_bytes, bitmap_offset, query.rank);
        success_index = static_cast<uint64_t>(success_offset) + dense_ordinal;
        return true;
    }
    if (!test_large_bit(layer.large_bitmap_words, bitmap_offset, query.rank)) {
        return false;
    }
    const uint32_t dense_ordinal = dense_ordinal_large(layer.large_bitmap_words, bitmap_offset, query.rank);
    success_index =
        large_success_partition_base(layer.large_success_base, success_offset, layer.large_success_stride)
        + dense_ordinal;
    return true;
}

template <typename T>
inline bool lookup_direct_success_index_prepared_impl(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
) {
    if (!query.valid || layer.direct_index.empty()) {
        return false;
    }
    uint32_t bitmap_offset = 0U;
    uint32_t success_offset = 0U;
    if (!direct_lookup_bucket_offsets(layer, query.target_direct_key46, bitmap_offset, success_offset)) {
        return false;
    }
    return success_index_from_offsets(layer, query, bitmap_offset, success_offset, success_index);
}

} // namespace

bool tile_limit_configs_equal(const TileLimitConfig &lhs, const TileLimitConfig &rhs) {
    return lhs.max_counts == rhs.max_counts
        && lhs.required_suffix24 == rhs.required_suffix24
        && lhs.valid_suffix_masks == rhs.valid_suffix_masks;
}

TileLimitConfig make_target_tile_limit_config(int target_exponent) {
    if (target_exponent <= 0 || target_exponent > 14) {
        throw std::runtime_error("zmask target exponent must be in [1, 14]");
    }
    TileLimitConfig config{};
    config.max_counts.fill(-1);
    config.max_counts[15] = -1;
    for (int exponent = target_exponent + 1; exponent <= 14; ++exponent) {
        config.max_counts[static_cast<size_t>(exponent)] = 0;
    }
    return config;
}

namespace {

void normalize_valid_suffix_masks(std::vector<uint32_t> &masks) {
    for (uint32_t &mask : masks) {
        mask &= 0xFFFFFFU;
    }
    if (std::find(masks.begin(), masks.end(), 0U) != masks.end()) {
        masks.clear();
        return;
    }
    std::sort(masks.begin(), masks.end());
    masks.erase(std::unique(masks.begin(), masks.end()), masks.end());
}

} // namespace

TileLimitConfig make_lut_tile_limit_config(
    int target_exponent,
    const std::vector<uint64_t> &seed_boards,
    const PatternSpec &spec,
    bool is_free,
    bool is_variant
) {
    TileLimitConfig config = make_target_tile_limit_config(target_exponent);

    const bool generated_free_seed = is_free && !is_variant && spec.pattern_masks.empty();
    if (!generated_free_seed && !seed_boards.empty()) {
        std::array<uint8_t, 16> seed_max_counts{};
        for (uint64_t board : seed_boards) {
            std::array<uint8_t, 16> counts{};
            for (uint32_t cell = 0; cell < 16U; ++cell) {
                const uint32_t tile = static_cast<uint32_t>((board >> (cell * 4U)) & 0xFULL);
                if (tile > static_cast<uint32_t>(target_exponent) && tile < 15U) {
                    ++counts[tile];
                }
            }
            for (uint32_t tile = static_cast<uint32_t>(target_exponent + 1); tile < 15U; ++tile) {
                seed_max_counts[tile] = std::max(seed_max_counts[tile], counts[tile]);
            }
        }
        for (uint32_t tile = static_cast<uint32_t>(target_exponent + 1); tile < 15U; ++tile) {
            if (seed_max_counts[tile] != 0U) {
                config.max_counts[tile] = -1;
            }
        }
    }

    config.valid_suffix_masks.reserve(spec.pattern_masks.size());
    for (uint64_t mask : spec.pattern_masks) {
        config.valid_suffix_masks.push_back(static_cast<uint32_t>(mask & 0xFFFFFFULL));
    }
    normalize_valid_suffix_masks(config.valid_suffix_masks);

    if (is_variant) {
        uint32_t required_suffix24 = 0U;
        for (uint64_t board : seed_boards) {
            for (uint32_t cell = 0; cell < 6U; ++cell) {
                const uint32_t shift = cell * 4U;
                if (((board >> shift) & 0xFULL) == 0xFU) {
                    required_suffix24 |= (0xFU << shift);
                }
            }
        }
        config.required_suffix24 = required_suffix24 & 0xFFFFFFU;
    }

    return config;
}

uint32_t tile_value(uint32_t tile) {
    return tile == 0U ? 0U : (1U << tile);
}

bool decode_suffix24_if_valid(uint32_t state, uint32_t &sum, const TileLimitConfig &config) {
    sum = 0U;
    if (!suffix_structural_constraints_allow(config, state)) {
        return false;
    }
    std::array<uint8_t, 16> tile_counts{};
    uint32_t x = state;
    for (int i = 0; i < 6; ++i) {
        const uint32_t tile = x & 0xFU;
        const uint8_t new_count = static_cast<uint8_t>(tile_counts[tile] + 1U);
        if (!suffix_tile_limit_allows(config, tile, new_count)) {
            return false;
        }
        tile_counts[tile] = new_count;
        sum += tile_value(tile);
        x >>= 4U;
    }
    return true;
}

uint8_t suffix24_zero_mask(uint32_t state) {
    uint8_t mask = 0U;
    for (uint32_t i = 0; i < 6U; ++i) {
        if ((state & 0xFU) == 0U) {
            mask |= static_cast<uint8_t>(1U << i);
        }
        state >>= 4U;
    }
    return mask;
}

uint64_t pack_bucket_key(uint64_t prefix40, uint8_t zero_mask, uint32_t remaining_sum) {
    return (prefix40 << 24U) | (static_cast<uint64_t>(zero_mask) << 18U) | static_cast<uint64_t>(remaining_sum);
}

uint64_t pack_direct_lookup_key46(uint64_t prefix40, uint8_t zero_mask) {
    return (prefix40 << kZeroMaskBits) | static_cast<uint64_t>(zero_mask);
}

uint64_t bucket_key_prefix40(uint64_t key) {
    return key >> 24U;
}

uint8_t bucket_key_zero_mask(uint64_t key) {
    return static_cast<uint8_t>((key >> 18U) & 0x3FULL);
}

uint32_t bucket_key_remaining_sum(uint64_t key) {
    return static_cast<uint32_t>(key & ((1ULL << 18U) - 1ULL));
}

uint64_t bucket_key_direct_lookup_key46(uint64_t key) {
    return direct_lookup_key46_from_bucket_key(key);
}

uint32_t sum_index(uint32_t sum) {
    if ((sum & 1U) != 0U) {
        throw std::runtime_error("encountered odd sum");
    }
    return sum >> 1U;
}

uint32_t lut_group_index(uint32_t sum, uint8_t zero_mask) {
    return sum_index(sum) * kZeroMaskCount + zero_mask;
}

uint64_t bytes_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 7U) >> 3U);
}

uint64_t words_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 63U) >> 6U);
}

uint64_t round_up_multiple_u64(uint64_t value, uint32_t multiple) {
    if (multiple == 0U) {
        throw std::runtime_error("round_up_multiple_u64 called with zero multiple");
    }
    const uint64_t m = static_cast<uint64_t>(multiple);
    return ((value + m - 1U) / m) * m;
}

uint32_t choose_large_success_stride(uint64_t dense_count) {
    if (dense_count <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return 1U;
    }
    for (const uint32_t stride : kEstimatedLargeSuccessStrides) {
        if (dense_count <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * stride) {
            return stride;
        }
    }
    throw std::runtime_error("large_success_dense_count exceeds supported stride coverage");
}

uint64_t large_success_partition_base(uint64_t large_success_base, uint32_t success_offset, uint32_t stride) {
    return large_success_base + static_cast<uint64_t>(success_offset) * static_cast<uint64_t>(stride);
}

uint32_t board_layer_sum(uint64_t board, const ZMaskLuts &luts) {
    return luts.row16_sum[(board >> 48U) & 0xFFFFULL]
         + luts.row16_sum[(board >> 32U) & 0xFFFFULL]
         + luts.row16_sum[(board >> 16U) & 0xFFFFULL]
         + luts.row16_sum[board & 0xFFFFULL];
}

uint32_t prefix40_sum(uint64_t prefix40, const ZMaskLuts &luts) {
    const uint32_t row0 = static_cast<uint32_t>((prefix40 >> 24U) & 0xFFFFULL);
    const uint32_t row1 = static_cast<uint32_t>((prefix40 >> 8U) & 0xFFFFULL);
    const uint32_t tail8 = static_cast<uint32_t>(prefix40 & 0xFFULL);
    return luts.row16_sum[row0] + luts.row16_sum[row1] + luts.row16_sum[tail8];
}

uint32_t prefix40_remaining_sum(uint64_t prefix40, uint32_t layer_sum, const ZMaskLuts &luts) {
    const uint32_t prefix_sum = prefix40_sum(prefix40, luts);
    if (prefix_sum > layer_sum) {
        throw std::runtime_error("prefix sum exceeds layer sum");
    }
    return layer_sum - prefix_sum;
}

ZMaskLuts build_zmask_luts(const TileLimitConfig &config, int num_threads) {
    const int thread_count = std::max(1, num_threads);
    ZMaskLuts luts;
    luts.rank_table.assign(kSuffixStateCount, kInvalidRank);
    luts.size_table.assign(((kMaxSum >> 1U) + 1U) * kZeroMaskCount, static_cast<uint16_t>(0));
    luts.offset_table.assign(((kMaxSum >> 1U) + 1U) * kZeroMaskCount, 0U);
    luts.row16_sum.resize(1U << 16U);

    {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(thread_count));
        for (int tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back([&, tid]() {
                const uint32_t begin = ((1U << 16U) * static_cast<uint32_t>(tid)) / static_cast<uint32_t>(thread_count);
                const uint32_t end = ((1U << 16U) * static_cast<uint32_t>(tid + 1)) / static_cast<uint32_t>(thread_count);
                for (uint32_t row = begin; row < end; ++row) {
                    uint32_t sum = 0U;
                    uint32_t x = row;
                    for (int i = 0; i < 4; ++i) {
                        sum += tile_value(x & 0xFU);
                        x >>= 4U;
                    }
                    luts.row16_sum[row] = sum;
                }
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
    }

    std::vector<uint32_t> local_counts(static_cast<size_t>(thread_count) * luts.size_table.size(), 0U);
    {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(thread_count));
        for (int tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back([&, tid]() {
                const uint32_t begin = (kSuffixStateCount * static_cast<uint32_t>(tid)) / static_cast<uint32_t>(thread_count);
                const uint32_t end = (kSuffixStateCount * static_cast<uint32_t>(tid + 1)) / static_cast<uint32_t>(thread_count);
                uint32_t *counts = local_counts.data() + static_cast<size_t>(tid) * luts.size_table.size();
                for (uint32_t state = begin; state < end; ++state) {
                    uint32_t sum = 0U;
                    if (!decode_suffix24_if_valid(state, sum, config)) {
                        continue;
                    }
                    ++counts[lut_group_index(sum, suffix24_zero_mask(state))];
                }
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
    }

    uint64_t total_valid_states = 0U;
    for (size_t group = 0; group < luts.size_table.size(); ++group) {
        uint32_t total = 0U;
        for (int tid = 0; tid < thread_count; ++tid) {
            total += local_counts[static_cast<size_t>(tid) * luts.size_table.size() + group];
        }
        if (total > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("zmask LUT valid_count exceeds uint16_t");
        }
        luts.size_table[group] = static_cast<uint16_t>(total);
        luts.offset_table[group] = static_cast<uint32_t>(total_valid_states);
        total_valid_states += total;
    }

    luts.unrank_array.resize(total_valid_states);
    std::vector<uint16_t> current_rank(luts.size_table.size(), 0U);
    for (uint32_t state = 0; state < kSuffixStateCount; ++state) {
        uint32_t sum = 0U;
        if (!decode_suffix24_if_valid(state, sum, config)) {
            continue;
        }
        const uint8_t zero_mask = suffix24_zero_mask(state);
        const uint32_t group = lut_group_index(sum, zero_mask);
        const uint16_t rank = current_rank[group]++;
        luts.rank_table[state] = rank;
        luts.unrank_array[luts.offset_table[group] + rank] = state;
    }

    initialize_runtime_tables(luts);
    return luts;
}

void initialize_runtime_tables(ZMaskLuts &luts) {
    constexpr uint32_t kInvalidGroup = std::numeric_limits<uint32_t>::max();
    luts.suffix_group_table.assign(kSuffixStateCount, kInvalidGroup);
    for (uint32_t suffix24 = 0; suffix24 < kSuffixStateCount; ++suffix24) {
        const uint16_t rank = luts.rank_table[suffix24];
        if (rank == kInvalidRank) {
            continue;
        }
        const uint32_t remaining_sum =
            luts.row16_sum[suffix24 & 0xFFFFU]
            + luts.row16_sum[(suffix24 >> 16U) & 0xFFU];
        const uint8_t zero_mask = suffix24_zero_mask(suffix24);
        luts.suffix_group_table[suffix24] = lut_group_index(remaining_sum, zero_mask);
    }
}

ThresholdStats choose_best_threshold(const std::vector<ThresholdStats> &scan) {
    if (scan.empty()) {
        throw std::runtime_error("threshold scan is empty");
    }
    auto better = [](const ThresholdStats &a, const ThresholdStats &b) {
        if (a.abs_diff_bytes != b.abs_diff_bytes) {
            return a.abs_diff_bytes < b.abs_diff_bytes;
        }
        if (a.aligned_total_bits != b.aligned_total_bits) {
            return a.aligned_total_bits < b.aligned_total_bits;
        }
        return a.threshold_bits < b.threshold_bits;
    };
    ThresholdStats best = scan.front();
    for (size_t i = 1; i < scan.size(); ++i) {
        if (better(scan[i], best)) {
            best = scan[i];
        }
    }
    return best;
}

bool test_small_bit(const std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t byte_idx = rank >> 3U;
    const uint32_t bit_idx = rank & 7U;
    return (bitmap[offset + byte_idx] & static_cast<uint8_t>(1U << bit_idx)) != 0U;
}

bool test_large_bit(const std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t word_idx = rank >> 6U;
    const uint32_t bit_idx = rank & 63U;
    return (bitmap[offset + word_idx] & (1ULL << bit_idx)) != 0ULL;
}

void set_small_bit(std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t byte_idx = rank >> 3U;
    const uint32_t bit_idx = rank & 7U;
    bitmap[offset + byte_idx] = static_cast<uint8_t>(bitmap[offset + byte_idx] | static_cast<uint8_t>(1U << bit_idx));
}

void set_large_bit(std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t word_idx = rank >> 6U;
    const uint32_t bit_idx = rank & 63U;
    bitmap[offset + word_idx] |= (1ULL << bit_idx);
}

template <typename T>
Layer<T> build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const std::vector<T> *initial_success,
    const ZMaskLuts &luts,
    int num_threads
) {
    Layer<T> layer;
    if (boards.empty()) {
        return layer;
    }
    if (initial_success && initial_success->size() != boards.size()) {
        throw std::runtime_error("initial_success size mismatch");
    }
    validate_sorted_boards(boards, luts);
    const int thread_count = effective_threads(num_threads);
    layer.layer_sum = board_layer_sum(boards.front(), luts);
    layer.live_board_count = boards.size();

    std::vector<size_t> range_begin;
    std::vector<uint32_t> range_count;
    std::vector<uint64_t> prefixes;
    std::vector<uint32_t> remaining_sums;
    range_begin.reserve(boards.size() / 16U + 1U);
    range_count.reserve(boards.size() / 16U + 1U);
    prefixes.reserve(boards.size() / 16U + 1U);
    remaining_sums.reserve(boards.size() / 16U + 1U);

    size_t begin = 0U;
    while (begin < boards.size()) {
        const uint64_t prefix40 = boards[begin] >> 24U;
        size_t end = begin + 1U;
        while (end < boards.size() && (boards[end] >> 24U) == prefix40) {
            ++end;
        }
        range_begin.push_back(begin);
        range_count.push_back(static_cast<uint32_t>(end - begin));
        prefixes.push_back(prefix40);
        remaining_sums.push_back(prefix40_remaining_sum(prefix40, layer.layer_sum, luts));
        begin = end;
    }

    const size_t prefix_count = prefixes.size();
    std::vector<uint32_t> child_counts(prefix_count, 0U);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t idx = 0; idx < static_cast<int64_t>(prefix_count); ++idx) {
        std::array<uint32_t, 64> counts{};
        const size_t start = range_begin[static_cast<size_t>(idx)];
        const size_t end = start + range_count[static_cast<size_t>(idx)];
        for (size_t pos = start; pos < end; ++pos) {
            const uint32_t suffix24 = static_cast<uint32_t>(boards[pos] & 0xFFFFFFULL);
            ++counts[suffix24_zero_mask(suffix24)];
        }
        uint32_t child_count = 0U;
        for (uint32_t count : counts) {
            child_count += count != 0U ? 1U : 0U;
        }
        child_counts[static_cast<size_t>(idx)] = child_count;
    }

    std::vector<uint32_t> prefix_bucket_begin(prefix_count + 1U, 0U);
    uint32_t bucket_cursor = 0U;
    for (size_t i = 0; i < prefix_count; ++i) {
        prefix_bucket_begin[i] = bucket_cursor;
        bucket_cursor += child_counts[i];
    }
    prefix_bucket_begin[prefix_count] = bucket_cursor;
    std::vector<BucketPlan> bucket_plans(bucket_cursor);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t idx = 0; idx < static_cast<int64_t>(prefix_count); ++idx) {
        std::array<uint32_t, 64> counts{};
        const size_t start = range_begin[static_cast<size_t>(idx)];
        const size_t end = start + range_count[static_cast<size_t>(idx)];
        for (size_t pos = start; pos < end; ++pos) {
            const uint32_t suffix24 = static_cast<uint32_t>(boards[pos] & 0xFFFFFFULL);
            ++counts[suffix24_zero_mask(suffix24)];
        }
        uint32_t plan_pos = prefix_bucket_begin[static_cast<size_t>(idx)];
        for (uint32_t zero_mask = 0; zero_mask < 64U; ++zero_mask) {
            if (counts[zero_mask] == 0U) {
                continue;
            }
            const uint32_t valid_count = luts.size_table[lut_group_index(remaining_sums[static_cast<size_t>(idx)], static_cast<uint8_t>(zero_mask))];
            if (valid_count == 0U) {
                throw std::runtime_error("zmask bucket has zero valid_count");
            }
            bucket_plans[plan_pos++] = BucketPlan{
                pack_bucket_key(prefixes[static_cast<size_t>(idx)], static_cast<uint8_t>(zero_mask), remaining_sums[static_cast<size_t>(idx)]),
                valid_count,
                counts[zero_mask],
                false
            };
        }
    }

    std::vector<uint32_t> unique_valid_counts;
    unique_valid_counts.reserve(bucket_plans.size());
    for (const BucketPlan &bucket : bucket_plans) {
        unique_valid_counts.push_back(bucket.valid_count);
        layer.exact_bitmap_bits += bucket.valid_count;
    }
    std::sort(unique_valid_counts.begin(), unique_valid_counts.end());
    unique_valid_counts.erase(std::unique(unique_valid_counts.begin(), unique_valid_counts.end()), unique_valid_counts.end());

    std::vector<ThresholdStats> scan;
    scan.reserve(unique_valid_counts.size());
    for (const uint32_t threshold : unique_valid_counts) {
        uint64_t small_bytes = 0U;
        uint64_t large_bytes = 0U;
        for (const BucketPlan &bucket : bucket_plans) {
            if (bucket.valid_count <= threshold) {
                small_bytes += bytes_for_bits(bucket.valid_count);
            } else {
                large_bytes += words_for_bits(bucket.valid_count) * sizeof(uint64_t);
            }
        }
        const uint64_t abs_diff = small_bytes >= large_bytes ? (small_bytes - large_bytes) : (large_bytes - small_bytes);
        scan.push_back(ThresholdStats{threshold, small_bytes, large_bytes, abs_diff, (small_bytes + large_bytes) * 8ULL});
    }
    const ThresholdStats best = choose_best_threshold(scan);
    layer.threshold_bits = best.threshold_bits;
    layer.aligned_bitmap_bits = best.aligned_total_bits;

    layer.bucket_keys.resize(bucket_plans.size());
    layer.bitmap_offsets.resize(bucket_plans.size());
    layer.success_offsets.resize(bucket_plans.size());

    uint64_t small_bitmap_cursor = 0U;
    uint64_t large_bitmap_cursor = 0U;
    uint64_t small_success_cursor = 0U;
    uint64_t large_success_dense_count = 0U;
    for (size_t i = 0; i < bucket_plans.size(); ++i) {
        BucketPlan &bucket = bucket_plans[i];
        layer.bucket_keys[i] = bucket.key;
        if (bucket.valid_count <= layer.threshold_bits) {
            bucket.is_small = true;
            layer.bitmap_offsets[i] = static_cast<uint32_t>(small_bitmap_cursor);
            small_bitmap_cursor += bytes_for_bits(bucket.valid_count);
        } else {
            bucket.is_small = false;
            layer.bitmap_offsets[i] = static_cast<uint32_t>(large_bitmap_cursor);
            large_bitmap_cursor += words_for_bits(bucket.valid_count);
            large_success_dense_count += bucket.live_count;
        }
    }
    layer.large_success_stride = choose_large_success_stride(large_success_dense_count);
    if (small_success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("small success partition exceeds uint32_t offset range");
    }
    for (size_t i = 0; i < bucket_plans.size(); ++i) {
        const BucketPlan &bucket = bucket_plans[i];
        if (bucket.is_small) {
            layer.success_offsets[i] = static_cast<uint32_t>(small_success_cursor);
            small_success_cursor += bucket.live_count;
        }
    }
    uint64_t large_success_units_cursor = 0U;
    for (size_t i = 0; i < bucket_plans.size(); ++i) {
        const BucketPlan &bucket = bucket_plans[i];
        if (!bucket.is_small) {
            layer.success_offsets[i] = static_cast<uint32_t>(large_success_units_cursor);
            large_success_units_cursor +=
                round_up_multiple_u64(bucket.live_count, layer.large_success_stride) / layer.large_success_stride;
        }
    }
    if (small_success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("small success partition exceeds uint32_t offset range");
    }
    if (large_success_units_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("large success partition exceeds uint32_t offset range after stride packing");
    }
    layer.large_success_dense_count = large_success_dense_count;
    layer.large_success_padded_count = large_success_units_cursor * static_cast<uint64_t>(layer.large_success_stride);
    layer.large_success_padding_count = layer.large_success_padded_count - layer.large_success_dense_count;
    layer.small_bitmap_bytes.assign(small_bitmap_cursor, 0U);
    layer.large_bitmap_words.assign(large_bitmap_cursor, 0ULL);
    layer.large_success_base = small_success_cursor;
    layer.success_values.assign(layer.large_success_base + layer.large_success_padded_count, T{});

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t idx = 0; idx < static_cast<int64_t>(prefix_count); ++idx) {
        std::array<std::vector<RankValue<T>>, 64> groups;
        const size_t start = range_begin[static_cast<size_t>(idx)];
        const size_t end = start + range_count[static_cast<size_t>(idx)];
        for (size_t pos = start; pos < end; ++pos) {
            const uint32_t suffix24 = static_cast<uint32_t>(boards[pos] & 0xFFFFFFULL);
            const uint8_t zero_mask = suffix24_zero_mask(suffix24);
            const uint16_t rank = luts.rank_table[suffix24];
            if (rank == kInvalidRank) {
                throw std::runtime_error("invalid zmask rank during layer build");
            }
            groups[zero_mask].push_back(RankValue<T>{rank, initial_success ? (*initial_success)[pos] : T{}});
        }

        uint32_t plan_pos = prefix_bucket_begin[static_cast<size_t>(idx)];
        for (uint32_t zero_mask = 0; zero_mask < 64U; ++zero_mask) {
            auto &group = groups[zero_mask];
            if (group.empty()) {
                continue;
            }
            std::sort(group.begin(), group.end(), [](const RankValue<T> &a, const RankValue<T> &b) {
                return a.rank < b.rank;
            });
            const BucketPlan &bucket = bucket_plans[plan_pos];
            const uint32_t bitmap_offset = layer.bitmap_offsets[plan_pos];
            const uint64_t success_base = bucket.is_small
                ? static_cast<uint64_t>(layer.success_offsets[plan_pos])
                : large_success_partition_base(layer.large_success_base,
                                               layer.success_offsets[plan_pos],
                                               layer.large_success_stride);
            if (group.size() != bucket.live_count) {
                throw std::runtime_error("bucket live_count mismatch during layer materialization");
            }
            for (size_t k = 0; k < group.size(); ++k) {
                if (bucket.is_small) {
                    set_small_bit(layer.small_bitmap_bytes, bitmap_offset, group[k].rank);
                } else {
                    set_large_bit(layer.large_bitmap_words, bitmap_offset, group[k].rank);
                }
                layer.success_values[success_base + k] = group[k].success;
            }
            ++plan_pos;
        }
    }

    return layer;
}

template <typename T>
void build_index(Layer<T> &layer, int num_threads) {
    (void)num_threads;
    if (layer.bucket_keys.empty()) {
        layer.index = {};
        layer.direct_index = {};
        layer.bucket_entries = {};
        return;
    }
    layer.index = {};

    if (layer.bucket_keys.size() >= BucketDirectIndex::kEmptyBucketIndex) {
        throw std::runtime_error("zmask direct bucket-index table requires uint32 bucket indices");
    }
    const uint32_t bucket_count = static_cast<uint32_t>(layer.bucket_keys.size());
    ensure_bucket_entries(layer);
    const uint64_t table_size = choose_bucket_index_direct_table_size(bucket_count);
    layer.direct_index = {};
    layer.direct_index.table_size = table_size;
    layer.direct_index.bucket_indices.assign(
        static_cast<size_t>(table_size),
        BucketDirectIndex::kEmptyBucketIndex
    );
    for (uint32_t bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
        const uint64_t key = layer.bucket_keys[static_cast<size_t>(bucket_idx)];
        const uint64_t direct_key46 = direct_lookup_key46_from_bucket_key(key);
        uint64_t slot = direct_hash_slot(layer.direct_index, direct_key46);
        while (layer.direct_index.bucket_indices[static_cast<size_t>(slot)] != BucketDirectIndex::kEmptyBucketIndex) {
            slot = direct_next_slot(layer.direct_index, slot);
        }
        layer.direct_index.bucket_indices[static_cast<size_t>(slot)] = bucket_idx;
    }
}

template <typename T>
LookupResult<T> lookup_success_and_index(
    const Layer<T> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    T zero_value
) {
    LookupResult<T> result{zero_value, 0U, false};
    if (layer.bucket_keys.empty()) {
        return result;
    }
    if (!layer.direct_index.empty()) {
        const PreparedDirectQuery query = prepare_direct_suffix_query(luts, board);
        if (!lookup_direct_success_index_prepared(layer, query, result.global_dense_index)) {
            return result;
        }
        result.success = layer.success_values[result.global_dense_index];
    } else {
        const uint64_t prefix40 = board >> 24U;
        const uint32_t suffix24 = static_cast<uint32_t>(board & 0xFFFFFFULL);
        const uint8_t zero_mask = suffix24_zero_mask(suffix24);
        const uint32_t remaining_sum =
            luts.row16_sum[suffix24 & 0xFFFFU] +
            luts.row16_sum[(suffix24 >> 16U) & 0xFFU];
        const uint32_t group = lut_group_index(remaining_sum, zero_mask);
        const uint32_t valid_count = luts.size_table[group];
        const uint16_t rank = luts.rank_table[suffix24];
        if (valid_count == 0U || rank == kInvalidRank || rank >= valid_count) {
            return result;
        }

        const uint64_t target_key = pack_bucket_key(prefix40, zero_mask, remaining_sum);
        uint64_t low = 0U;
        uint64_t high = static_cast<uint64_t>(layer.bucket_keys.size() - 1U);
        if (!layer.index.empty()) {
            AdaptiveIndex::Range range = layer.index.locate(target_key);
            if (range.empty()) {
                return result;
            }
            low = range.begin;
            high = range.end - 1U;
        }

        if (low > high) {
            return result;
        }
        const size_t begin = static_cast<size_t>(low);
        const size_t length = static_cast<size_t>(high - low + 1U);
        const size_t pos = HybridSearch::exact_search(layer.bucket_keys.data() + begin, length, target_key);
        if (pos == HybridSearch::kNotFound) {
            return result;
        }
        const uint32_t bucket_index = static_cast<uint32_t>(begin + pos);
        const uint32_t bitmap_offset = layer.bitmap_offsets[bucket_index];
        const uint32_t success_offset = layer.success_offsets[bucket_index];
        const bool is_small = valid_count <= layer.threshold_bits;
        if (is_small) {
            if (!test_small_bit(layer.small_bitmap_bytes, bitmap_offset, rank)) {
                return result;
            }
            const uint32_t dense_ordinal = dense_ordinal_small(layer.small_bitmap_bytes, bitmap_offset, rank);
            result.global_dense_index = static_cast<uint64_t>(success_offset) + dense_ordinal;
            result.success = layer.success_values[result.global_dense_index];
        } else {
            if (!test_large_bit(layer.large_bitmap_words, bitmap_offset, rank)) {
                return result;
            }
            const uint32_t dense_ordinal = dense_ordinal_large(layer.large_bitmap_words, bitmap_offset, rank);
            result.global_dense_index = large_success_partition_base(layer.large_success_base, success_offset, layer.large_success_stride) + dense_ordinal;
            result.success = layer.success_values[result.global_dense_index];
        }
    }
    result.found = true;
    return result;
}

template <typename T>
T lookup_success(
    const Layer<T> &layer,
    const ZMaskLuts &luts,
    uint64_t board,
    T zero_value
) {
    return lookup_success_and_index(layer, luts, board, zero_value).success;
}

PreparedDirectQuery prepare_direct_suffix_query(const ZMaskLuts &luts, uint64_t board) {
    constexpr uint32_t kInvalidGroup = std::numeric_limits<uint32_t>::max();
    PreparedDirectQuery query;
    const uint64_t prefix40 = board >> 24U;
    const uint32_t suffix24 = static_cast<uint32_t>(board & 0xFFFFFFULL);
    const uint16_t rank = luts.rank_table[suffix24];
    if (rank == kInvalidRank) {
        return query;
    }
    const uint32_t group = luts.suffix_group_table.empty()
        ? kInvalidGroup
        : luts.suffix_group_table[suffix24];
    if (group == kInvalidGroup) {
        return query;
    }
    const uint32_t valid_count = luts.size_table[group];
    if (valid_count == 0U || rank >= valid_count) {
        return query;
    }
    const uint8_t zero_mask = static_cast<uint8_t>(group & (kZeroMaskCount - 1U));
    query.target_direct_key46 = pack_direct_lookup_key46(prefix40, zero_mask);
    query.valid_count = valid_count;
    query.rank = rank;
    query.valid = true;
    return query;
}

template <typename T>
bool lookup_direct_success_index_prepared(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    uint64_t &success_index
) {
    return lookup_direct_success_index_prepared_impl(layer, query, success_index);
}

template <typename T>
void lookup_direct_success_indices_prepared_batch(
    const Layer<T> &layer,
    const PreparedDirectQuery *queries,
    uint64_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
) {
    if (count == 0U) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        success_indices[i] = 0U;
        found_flags[i] = 0U;
    }
    const BucketDirectIndex &index = layer.direct_index;
    if (index.empty()) {
        return;
    }

    constexpr uint32_t kMaxPipelinedProbeRounds = 4U;
    uint64_t slots[128];
    uint32_t bucket_indices[128];
    uint8_t active[128];
    const uint32_t local_count = std::min<uint32_t>(count, 128U);

    for (uint32_t i = 0; i < local_count; ++i) {
        if (!queries[i].valid) {
            active[i] = 0U;
            continue;
        }
        slots[i] = direct_hash_slot(index, queries[i].target_direct_key46);
        active[i] = 1U;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&index.bucket_indices[static_cast<size_t>(slots[i])], 0, 1);
#endif
    }

    for (uint32_t round = 0; round < kMaxPipelinedProbeRounds; ++round) {
        bool any_active = false;
        for (uint32_t i = 0; i < local_count; ++i) {
            if (!active[i]) {
                continue;
            }
            any_active = true;
            const uint32_t bucket_index = index.bucket_indices[static_cast<size_t>(slots[i])];
            bucket_indices[i] = bucket_index;
            if (bucket_index == BucketDirectIndex::kEmptyBucketIndex) {
                active[i] = 0U;
                continue;
            }
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&layer.bucket_entries[static_cast<size_t>(bucket_index)], 0, 1);
#endif
        }
        if (!any_active) {
            break;
        }

        for (uint32_t i = 0; i < local_count; ++i) {
            if (!active[i] || bucket_indices[i] == BucketDirectIndex::kEmptyBucketIndex) {
                continue;
            }
            const BucketDirectEntry &bucket = layer.bucket_entries[static_cast<size_t>(bucket_indices[i])];
            if (direct_lookup_key46_from_bucket_key(bucket.key) == queries[i].target_direct_key46) {
#if defined(__GNUC__) || defined(__clang__)
                if (queries[i].valid_count <= layer.threshold_bits) {
                    __builtin_prefetch(
                        &layer.small_bitmap_bytes[bucket.bitmap_offset + (queries[i].rank >> 3U)],
                        0,
                        1
                    );
                } else {
                    __builtin_prefetch(
                        &layer.large_bitmap_words[bucket.bitmap_offset + (queries[i].rank >> 6U)],
                        0,
                        1
                    );
                }
#endif
                found_flags[i] = success_index_from_offsets(
                    layer,
                    queries[i],
                    bucket.bitmap_offset,
                    bucket.success_offset,
                    success_indices[i]
                ) ? 1U : 0U;
                active[i] = 0U;
                continue;
            }
            slots[i] = direct_next_slot(index, slots[i]);
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&index.bucket_indices[static_cast<size_t>(slots[i])], 0, 1);
#endif
        }
    }

    for (uint32_t i = 0; i < local_count; ++i) {
        if (!active[i]) {
            continue;
        }
        found_flags[i] = lookup_direct_success_index_prepared_impl(layer, queries[i], success_indices[i]) ? 1U : 0U;
    }
    for (uint32_t i = local_count; i < count; ++i) {
        found_flags[i] = lookup_direct_success_index_prepared_impl(layer, queries[i], success_indices[i]) ? 1U : 0U;
    }
}

template <typename T>
T lookup_direct_prepared(
    const Layer<T> &layer,
    const PreparedDirectQuery &query,
    T zero_value
) {
    uint64_t success_index = 0U;
    if (!lookup_direct_success_index_prepared(layer, query, success_index)) {
        return zero_value;
    }
    return layer.success_values[success_index];
}

template <typename T>
uint64_t live_count_by_bitmap(
    const Layer<T> &layer,
    uint32_t bucket_index,
    const ZMaskLuts &luts
) {
    const uint64_t key = layer.bucket_keys[bucket_index];
    const uint8_t zero_mask = bucket_key_zero_mask(key);
    const uint32_t remaining_sum = bucket_key_remaining_sum(key);
    const uint32_t valid_count = luts.size_table[lut_group_index(remaining_sum, zero_mask)];
    const bool is_small = valid_count <= layer.threshold_bits;
    uint64_t total = 0U;
    if (is_small) {
        const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
        const uint32_t bitmap_offset = layer.bitmap_offsets[bucket_index];
        for (uint32_t i = 0; i < bytes; ++i) {
            total += popcount_u32(layer.small_bitmap_bytes[bitmap_offset + i]);
        }
    } else {
        const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
        const uint32_t bitmap_offset = layer.bitmap_offsets[bucket_index];
        for (uint32_t i = 0; i < words; ++i) {
            total += popcount_u64(layer.large_bitmap_words[bitmap_offset + i]);
        }
    }
    return total;
}

template Layer<uint32_t> build_layer_from_sorted_boards(const std::vector<uint64_t> &, const std::vector<uint32_t> *, const ZMaskLuts &, int);
template Layer<uint64_t> build_layer_from_sorted_boards(const std::vector<uint64_t> &, const std::vector<uint64_t> *, const ZMaskLuts &, int);
template Layer<float> build_layer_from_sorted_boards(const std::vector<uint64_t> &, const std::vector<float> *, const ZMaskLuts &, int);
template Layer<double> build_layer_from_sorted_boards(const std::vector<uint64_t> &, const std::vector<double> *, const ZMaskLuts &, int);

template void build_index(Layer<uint32_t> &, int);
template void build_index(Layer<uint64_t> &, int);
template void build_index(Layer<float> &, int);
template void build_index(Layer<double> &, int);

template LookupResult<uint32_t> lookup_success_and_index(const Layer<uint32_t> &, const ZMaskLuts &, uint64_t, uint32_t);
template LookupResult<uint64_t> lookup_success_and_index(const Layer<uint64_t> &, const ZMaskLuts &, uint64_t, uint64_t);
template LookupResult<float> lookup_success_and_index(const Layer<float> &, const ZMaskLuts &, uint64_t, float);
template LookupResult<double> lookup_success_and_index(const Layer<double> &, const ZMaskLuts &, uint64_t, double);

template uint32_t lookup_success(const Layer<uint32_t> &, const ZMaskLuts &, uint64_t, uint32_t);
template uint64_t lookup_success(const Layer<uint64_t> &, const ZMaskLuts &, uint64_t, uint64_t);
template float lookup_success(const Layer<float> &, const ZMaskLuts &, uint64_t, float);
template double lookup_success(const Layer<double> &, const ZMaskLuts &, uint64_t, double);

template bool lookup_direct_success_index_prepared(const Layer<uint32_t> &, const PreparedDirectQuery &, uint64_t &);
template bool lookup_direct_success_index_prepared(const Layer<uint64_t> &, const PreparedDirectQuery &, uint64_t &);
template bool lookup_direct_success_index_prepared(const Layer<float> &, const PreparedDirectQuery &, uint64_t &);
template bool lookup_direct_success_index_prepared(const Layer<double> &, const PreparedDirectQuery &, uint64_t &);

template void lookup_direct_success_indices_prepared_batch(const Layer<uint32_t> &, const PreparedDirectQuery *, uint64_t *, uint8_t *, uint32_t);
template void lookup_direct_success_indices_prepared_batch(const Layer<uint64_t> &, const PreparedDirectQuery *, uint64_t *, uint8_t *, uint32_t);
template void lookup_direct_success_indices_prepared_batch(const Layer<float> &, const PreparedDirectQuery *, uint64_t *, uint8_t *, uint32_t);
template void lookup_direct_success_indices_prepared_batch(const Layer<double> &, const PreparedDirectQuery *, uint64_t *, uint8_t *, uint32_t);

template uint32_t lookup_direct_prepared(const Layer<uint32_t> &, const PreparedDirectQuery &, uint32_t);
template uint64_t lookup_direct_prepared(const Layer<uint64_t> &, const PreparedDirectQuery &, uint64_t);
template float lookup_direct_prepared(const Layer<float> &, const PreparedDirectQuery &, float);
template double lookup_direct_prepared(const Layer<double> &, const PreparedDirectQuery &, double);

template uint64_t live_count_by_bitmap(const Layer<uint32_t> &, uint32_t, const ZMaskLuts &);
template uint64_t live_count_by_bitmap(const Layer<uint64_t> &, uint32_t, const ZMaskLuts &);
template uint64_t live_count_by_bitmap(const Layer<float> &, uint32_t, const ZMaskLuts &);
template uint64_t live_count_by_bitmap(const Layer<double> &, uint32_t, const ZMaskLuts &);

} // namespace ZMaskFrozen
