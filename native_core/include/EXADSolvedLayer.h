#pragma once

#include "EXADIO.h"
#include "FileIOUtils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace EXAD {

constexpr const char *kSolvedFileExtension = ".exadbook";
constexpr uint32_t kSolvedFileVersion = 2U;
constexpr uint32_t kInvalidDirectBucket = 0xFFFFFFFFU;
constexpr uint32_t kLookupBatchSize = 256U;

template <typename T>
struct NoInitAllocator : std::allocator<T> {
    using value_type = T;

    NoInitAllocator() noexcept = default;

    template <typename U>
    NoInitAllocator(const NoInitAllocator<U> &) noexcept {}

    template <typename U>
    struct rebind {
        using other = NoInitAllocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U *ptr, Args &&...args) {
        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<U>) {
            ::new (static_cast<void *>(ptr)) U;
        } else {
            ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...);
        }
    }
};

template <typename T, typename U>
inline bool operator==(const NoInitAllocator<T> &, const NoInitAllocator<U> &) noexcept {
    return true;
}

template <typename T, typename U>
inline bool operator!=(const NoInitAllocator<T> &, const NoInitAllocator<U> &) noexcept {
    return false;
}

template <typename T>
using SuccessVector = std::vector<T, NoInitAllocator<T>>;

template <typename T>
inline void fill_success_values(SuccessVector<T> &values, T value, int num_threads) {
    const int requested_threads = std::max(1, num_threads);
    const int64_t count = static_cast<int64_t>(values.size());
#pragma omp parallel for schedule(static) num_threads(requested_threads) if(requested_threads > 1 && count > 1048576)
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = value;
    }
}

enum class DTypeMode : uint32_t {
    UInt32 = 1,
    UInt64 = 2,
    Float32 = 3,
    Float64 = 4,
    OneMinusFloat32 = 5,
    OneMinusFloat64 = 6,
};

inline DTypeMode dtype_mode_from_name(const std::string &name) {
    if (name == "uint64") {
        return DTypeMode::UInt64;
    }
    if (name == "float32") {
        return DTypeMode::Float32;
    }
    if (name == "float64") {
        return DTypeMode::Float64;
    }
    if (name == "1-float32") {
        return DTypeMode::OneMinusFloat32;
    }
    if (name == "1-float64") {
        return DTypeMode::OneMinusFloat64;
    }
    return DTypeMode::UInt32;
}

template <typename T>
inline uint32_t dtype_value_size() {
    return static_cast<uint32_t>(sizeof(T));
}

inline uint32_t dtype_value_size(DTypeMode mode) {
    switch (mode) {
        case DTypeMode::UInt64:
        case DTypeMode::Float64:
        case DTypeMode::OneMinusFloat64:
            return 8U;
        case DTypeMode::UInt32:
        case DTypeMode::Float32:
        case DTypeMode::OneMinusFloat32:
        default:
            return 4U;
    }
}

template <typename T>
inline bool dtype_matches_type(DTypeMode mode) {
    if constexpr (std::is_same_v<T, uint64_t>) {
        return mode == DTypeMode::UInt64;
    } else if constexpr (std::is_same_v<T, float>) {
        return mode == DTypeMode::Float32 || mode == DTypeMode::OneMinusFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        return mode == DTypeMode::Float64 || mode == DTypeMode::OneMinusFloat64;
    } else {
        return mode == DTypeMode::UInt32;
    }
}

inline size_t solved_derive_size_for_bucket(int count_32k, uint8_t num_free_32k) {
    constexpr std::array<uint32_t, 19> factorials = {
        1U, 1U, 2U, 6U, 24U, 120U, 720U, 5040U, 40320U, 362880U,
        3628800U, 39916800U, 1U, 1U, 1U, 1U, 1U, 1U, 1U
    };
    if (count_32k == -1) {
        return 0U;
    }
    if (count_32k < 0) {
        return static_cast<size_t>(factorials[static_cast<size_t>(-count_32k - 2)] / factorials[num_free_32k]);
    }
    if (count_32k > 15) {
        return static_cast<size_t>(factorials[static_cast<size_t>(count_32k - 16)] / factorials[num_free_32k]);
    }
    return static_cast<size_t>(factorials[static_cast<size_t>(count_32k)] / factorials[num_free_32k]);
}

struct DirectIndex {
    std::vector<uint32_t> slots;
    uint32_t mask = 0;

    [[nodiscard]] bool empty() const {
        return slots.empty();
    }
};

struct DirectEntry {
    uint64_t key = kInvalidBucketKey;
    uint32_t bitmap_offset = 0;
    uint32_t dense_offset = 0;
};
static_assert(sizeof(DirectEntry) == 16U, "EXAD DirectEntry must stay compact");

struct DirectEntryIndex {
    std::vector<DirectEntry> entries;
    uint32_t mask = 0;

    [[nodiscard]] bool empty() const {
        return entries.empty();
    }
};

template <typename T>
struct SolvedLayer {
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint64_t lut_signature = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
    DTypeMode dtype_mode = DTypeMode::UInt32;
    std::array<BoardSet, bucket_slot_count()> sets{};
    std::array<uint64_t, bucket_slot_count()> slot_row_base{};
    std::array<uint64_t, bucket_slot_count()> slot_value_base{};
    std::array<uint32_t, bucket_slot_count()> row_width{};
    SuccessVector<T> success_values;
    std::array<DirectIndex, bucket_slot_count()> direct_indices{};
    std::array<DirectEntryIndex, bucket_slot_count()> direct_entry_indices{};
    std::array<std::vector<uint32_t>, bucket_slot_count()> large_rank_bases{};

    [[nodiscard]] bool empty() const {
        return live_board_count == 0;
    }
};

struct LookupResult {
    bool found = false;
    uint64_t global_row = 0;
    uint64_t local_row = 0;
};

inline std::string solved_file_path(const std::string &pathname, int step) {
    return pathname + std::to_string(step) + kSolvedFileExtension;
}

inline uint64_t mix_key64(uint64_t value) {
    value ^= value >> 33U;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33U;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33U;
    return value;
}

inline uint32_t next_power_of_two_u32(uint64_t value) {
    uint64_t cap = 1;
    while (cap < value) {
        cap <<= 1U;
    }
    if (cap > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("EXAD direct index capacity exceeds uint32 range");
    }
    return static_cast<uint32_t>(cap);
}

inline uint32_t popcount8(uint8_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(value)));
#else
    uint32_t count = 0;
    while (value != 0U) {
        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
        ++count;
    }
    return count;
#endif
}

inline uint32_t popcount64_local(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t count = 0;
    while (value != 0ULL) {
        value &= value - 1ULL;
        ++count;
    }
    return count;
#endif
}

inline bool dense_ordinal_for_rank(
    const BoardSet &set,
    const BucketEntry &bucket,
    uint32_t valid_count,
    uint32_t rank,
    uint32_t &ordinal
) {
    if (rank >= valid_count) {
        return false;
    }
    if (valid_count <= set.threshold_bits) {
        const uint32_t byte_idx = rank >> 3U;
        const uint32_t bit = rank & 7U;
        const uint32_t offset = bucket.bitmap_offset;
        const uint8_t current = set.small_bitmap_bytes[offset + byte_idx];
        if (((current >> bit) & 1U) == 0U) {
            return false;
        }
        uint32_t count = 0;
        for (uint32_t i = 0; i < byte_idx; ++i) {
            count += popcount8(set.small_bitmap_bytes[offset + i]);
        }
        const uint8_t lower_mask = bit == 0U ? 0U : static_cast<uint8_t>((1U << bit) - 1U);
        count += popcount8(static_cast<uint8_t>(current & lower_mask));
        ordinal = count;
        return true;
    }

    const uint32_t word_idx = rank >> 6U;
    const uint32_t bit = rank & 63U;
    const uint32_t offset = bucket.bitmap_offset;
    const uint64_t current = set.large_bitmap_words[offset + word_idx];
    if (((current >> bit) & 1ULL) == 0ULL) {
        return false;
    }
    uint32_t count = 0;
    for (uint32_t i = 0; i < word_idx; ++i) {
        count += popcount64_local(set.large_bitmap_words[offset + i]);
    }
    const uint64_t lower_mask = bit == 0U ? 0ULL : ((1ULL << bit) - 1ULL);
    count += popcount64_local(current & lower_mask);
    ordinal = count;
    return true;
}

inline bool dense_ordinal_for_rank_fast(
    const BoardSet &set,
    const DirectEntry &entry,
    uint32_t valid_count,
    uint32_t rank,
    const std::vector<uint32_t> &large_rank_bases,
    uint32_t &ordinal
) {
    if (rank >= valid_count) {
        return false;
    }
    if (valid_count <= set.threshold_bits) {
        const uint32_t byte_idx = rank >> 3U;
        const uint32_t bit = rank & 7U;
        const uint32_t offset = entry.bitmap_offset;
        const uint8_t current = set.small_bitmap_bytes[offset + byte_idx];
        if (((current >> bit) & 1U) == 0U) {
            return false;
        }
        uint32_t count = 0;
        for (uint32_t i = 0; i < byte_idx; ++i) {
            count += popcount8(set.small_bitmap_bytes[offset + i]);
        }
        const uint8_t lower_mask = bit == 0U ? 0U : static_cast<uint8_t>((1U << bit) - 1U);
        count += popcount8(static_cast<uint8_t>(current & lower_mask));
        ordinal = count;
        return true;
    }

    const uint32_t word_idx = rank >> 6U;
    const uint32_t bit = rank & 63U;
    const uint32_t offset = entry.bitmap_offset;
    const uint64_t current = set.large_bitmap_words[offset + word_idx];
    if (((current >> bit) & 1ULL) == 0ULL) {
        return false;
    }
    if (static_cast<size_t>(entry.bitmap_offset) + word_idx >= large_rank_bases.size()) {
        return false;
    }
    uint32_t count = large_rank_bases[static_cast<size_t>(entry.bitmap_offset) + word_idx];
    const uint64_t lower_mask = bit == 0U ? 0ULL : ((1ULL << bit) - 1ULL);
    count += popcount64_local(current & lower_mask);
    ordinal = count;
    return true;
}

inline void build_direct_index_for_set(const BoardSet &set, DirectIndex &index) {
    index = DirectIndex{};
    if (set.buckets.empty()) {
        return;
    }
    const uint32_t capacity = next_power_of_two_u32(static_cast<uint64_t>(set.buckets.size()) * 2ULL);
    index.slots.assign(capacity, kInvalidDirectBucket);
    index.mask = capacity - 1U;
    for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(set.buckets.size()); ++bucket_idx) {
        const uint64_t key = set.buckets[bucket_idx].key;
        uint32_t slot = static_cast<uint32_t>(mix_key64(key)) & index.mask;
        while (index.slots[slot] != kInvalidDirectBucket) {
            slot = (slot + 1U) & index.mask;
        }
        index.slots[slot] = bucket_idx;
    }
}

inline uint32_t choose_direct_entry_capacity(uint64_t bucket_count) {
    constexpr uint64_t kMinCapacity = 1024ULL;
    constexpr uint64_t kLoadPercent = 40ULL;
    const uint64_t required = (bucket_count * 100ULL + (kLoadPercent - 1ULL)) / kLoadPercent;
    return next_power_of_two_u32(std::max<uint64_t>(kMinCapacity, required));
}

inline uint64_t masked_large_bitmap_word(const BoardSet &set, const BucketEntry &bucket, uint32_t valid_count, uint32_t word_idx) {
    uint64_t value = set.large_bitmap_words[bucket.bitmap_offset + word_idx];
    const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
    if (word_idx + 1U == words && (valid_count & 63U) != 0U) {
        value &= ((1ULL << (valid_count & 63U)) - 1ULL);
    }
    return value;
}

inline void build_direct_entry_index_for_set(
    const BoardSet &set,
    const Luts &luts,
    DirectEntryIndex &index,
    std::vector<uint32_t> &large_rank_bases
) {
    index = DirectEntryIndex{};
    large_rank_bases.clear();
    if (set.buckets.empty()) {
        return;
    }

    const uint64_t rank_words = static_cast<uint64_t>(set.large_bitmap_words.size());
    if (rank_words > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("EXAD runtime large rank bases exceed uint32 range");
    }
    large_rank_bases.assign(static_cast<size_t>(rank_words), 0U);
    for (size_t bucket_idx = 0; bucket_idx < set.buckets.size(); ++bucket_idx) {
        const BucketEntry &bucket = set.buckets[bucket_idx];
        const uint32_t group = lut_group_index(bucket_key_semantic_sum(bucket.key));
        const uint32_t valid_count = luts.size_table[group];
        if (valid_count <= set.threshold_bits) {
            continue;
        }
        const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
        const uint32_t base = bucket.bitmap_offset;
        if (static_cast<uint64_t>(base) + words > rank_words) {
            throw std::runtime_error("EXAD large rank base offset exceeds large bitmap size");
        }
        uint32_t count = 0U;
        for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
            large_rank_bases[static_cast<size_t>(base) + word_idx] = count;
            count += popcount64_local(masked_large_bitmap_word(set, bucket, valid_count, word_idx));
        }
    }

    const uint32_t capacity = choose_direct_entry_capacity(set.buckets.size());
    DirectEntry empty_entry{};
    index.entries.assign(capacity, empty_entry);
    index.mask = capacity - 1U;
    for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(set.buckets.size()); ++bucket_idx) {
        const BucketEntry &bucket = set.buckets[bucket_idx];
        uint32_t slot = static_cast<uint32_t>(mix_key64(bucket.key)) & index.mask;
        while (index.entries[slot].key != kInvalidBucketKey) {
            slot = (slot + 1U) & index.mask;
        }
        index.entries[slot] = DirectEntry{
            bucket.key,
            bucket.bitmap_offset,
            bucket.dense_offset
        };
    }
}

template <typename T>
inline void build_direct_indexes(SolvedLayer<T> &layer, const Luts &luts) {
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        layer.direct_indices[slot] = DirectIndex{};
        build_direct_entry_index_for_set(
            layer.sets[slot],
            luts,
            layer.direct_entry_indices[slot],
            layer.large_rank_bases[slot]
        );
    }
}

template <typename T>
inline SolvedLayer<T> make_solved_layer_from_generation(
    const Layer &source,
    const AdvancedMaskParam &param,
    DTypeMode mode,
    T fill_value
) {
    SolvedLayer<T> out;
    out.original_board_sum = source.original_board_sum;
    out.threshold_bits = source.threshold_bits;
    out.lut_signature = source.lut_signature;
    out.physical_transform = source.physical_transform;
    out.inverse_physical_transform = source.inverse_physical_transform;
    out.logical_pattern_signature = source.logical_pattern_signature;
    out.physical_pattern_signature = source.physical_pattern_signature;
    out.live_board_count = source.live_board_count;
    out.dtype_mode = mode;
    out.sets = source.sets;
    (void)fill_value;

    uint64_t row_cursor = 0;
    uint64_t value_cursor = 0;
    for (size_t slot = 0; slot < out.sets.size(); ++slot) {
        const int ad_key = bucket_key_min() + static_cast<int>(slot);
        const uint32_t width = static_cast<uint32_t>(solved_derive_size_for_bucket(ad_key, param.num_free_32k));
        out.row_width[slot] = width;
        out.slot_row_base[slot] = row_cursor;
        out.slot_value_base[slot] = value_cursor;
        row_cursor += out.sets[slot].live_board_count;
        value_cursor += out.sets[slot].live_board_count * static_cast<uint64_t>(width);
    }
    out.success_values.resize(static_cast<size_t>(value_cursor));
    return out;
}

template <typename T>
inline SolvedLayer<T> make_solved_layer_from_generation(
    Layer &&source,
    const AdvancedMaskParam &param,
    DTypeMode mode,
    T fill_value
) {
    SolvedLayer<T> out;
    out.original_board_sum = source.original_board_sum;
    out.threshold_bits = source.threshold_bits;
    out.lut_signature = source.lut_signature;
    out.physical_transform = source.physical_transform;
    out.inverse_physical_transform = source.inverse_physical_transform;
    out.logical_pattern_signature = source.logical_pattern_signature;
    out.physical_pattern_signature = source.physical_pattern_signature;
    out.live_board_count = source.live_board_count;
    out.dtype_mode = mode;
    out.sets = std::move(source.sets);
    (void)fill_value;

    uint64_t row_cursor = 0;
    uint64_t value_cursor = 0;
    for (size_t slot = 0; slot < out.sets.size(); ++slot) {
        const int ad_key = bucket_key_min() + static_cast<int>(slot);
        const uint32_t width = static_cast<uint32_t>(solved_derive_size_for_bucket(ad_key, param.num_free_32k));
        out.row_width[slot] = width;
        out.slot_row_base[slot] = row_cursor;
        out.slot_value_base[slot] = value_cursor;
        row_cursor += out.sets[slot].live_board_count;
        value_cursor += out.sets[slot].live_board_count * static_cast<uint64_t>(width);
    }
    out.success_values.resize(static_cast<size_t>(value_cursor));
    return out;
}

template <typename T>
inline T *row_ptr(SolvedLayer<T> &layer, size_t slot, uint64_t local_row) {
    const uint32_t width = layer.row_width[slot];
    return layer.success_values.data() + static_cast<size_t>(layer.slot_value_base[slot] + local_row * width);
}

template <typename T>
inline const T *row_ptr(const SolvedLayer<T> &layer, size_t slot, uint64_t local_row) {
    const uint32_t width = layer.row_width[slot];
    return layer.success_values.data() + static_cast<size_t>(layer.slot_value_base[slot] + local_row * width);
}

template <typename T>
inline LookupResult lookup_row(
    const SolvedLayer<T> &layer,
    const Luts &luts,
    int ad_key,
    uint64_t board
) {
    if (ad_key < bucket_key_min() || ad_key > bucket_key_max()) {
        return {};
    }
    const size_t slot_index = bucket_to_index(ad_key);
    const BoardSet &set = layer.sets[slot_index];
    const DirectEntryIndex &index = layer.direct_entry_indices[slot_index];
    if (set.buckets.empty() || index.empty()) {
        return {};
    }
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    uint32_t group = 0;
    uint32_t rank = 0;
    uint32_t semantic_sum = 0;
    if (!suffix28_rank_group_sum_fast(luts, suffix28, group, rank, semantic_sum)) {
        return {};
    }
    const uint64_t key = pack_bucket_key(board >> kSuffixBits, semantic_sum);
    uint32_t hash_slot = static_cast<uint32_t>(mix_key64(key)) & index.mask;
    while (true) {
        const DirectEntry &entry = index.entries[hash_slot];
        if (entry.key == kInvalidBucketKey) {
            return {};
        }
        if (entry.key == key) {
            const uint32_t valid_count = luts.size_table[group];
            uint32_t ordinal = 0;
            if (!dense_ordinal_for_rank_fast(
                    set,
                    entry,
                    valid_count,
                    rank,
                    layer.large_rank_bases[slot_index],
                    ordinal)) {
                return {};
            }
            const uint64_t local_row = static_cast<uint64_t>(entry.dense_offset) + ordinal;
            return {true, layer.slot_row_base[slot_index] + local_row, local_row};
        }
        hash_slot = (hash_slot + 1U) & index.mask;
    }
}

template <typename T>
inline const T *lookup_row_ptr(
    const SolvedLayer<T> &layer,
    const Luts &luts,
    int ad_key,
    uint64_t board
) {
    const LookupResult result = lookup_row(layer, luts, ad_key, board);
    if (!result.found) {
        return nullptr;
    }
    return row_ptr(layer, bucket_to_index(ad_key), result.local_row);
}

struct PreparedQuery {
    uint64_t key = 0;
    uint32_t rank = 0;
    uint32_t group = 0;
    uint32_t valid_count = 0;
    uint32_t column = 0;
    uint16_t ref = 0;
    uint8_t slot = 0;
    uint8_t valid = 0;
};

inline bool prepare_query(
    const Luts &luts,
    int ad_key,
    uint64_t board,
    uint16_t ref,
    uint32_t column,
    PreparedQuery &query
) {
    query = PreparedQuery{};
    if (ad_key < bucket_key_min() || ad_key > bucket_key_max()) {
        return false;
    }
    uint32_t group = 0;
    uint32_t rank = 0;
    uint32_t semantic_sum = 0;
    if (!suffix28_rank_group_sum_fast(luts, static_cast<uint32_t>(board & kSuffixMask), group, rank, semantic_sum)) {
        return false;
    }
    query.key = pack_bucket_key(board >> kSuffixBits, semantic_sum);
    query.rank = rank;
    query.group = group;
    query.valid_count = luts.size_table[group];
    query.column = column;
    query.ref = ref;
    query.slot = static_cast<uint8_t>(bucket_to_index(ad_key));
    query.valid = 1U;
    return true;
}

template <typename T>
inline uint64_t lookup_reduce_prepared_queries(
    const SolvedLayer<T> &layer,
    const PreparedQuery *queries,
    uint32_t count,
    T *best
) {
    uint64_t found = 0U;
    for (uint32_t base = 0; base < count; base += kLookupBatchSize) {
        const uint32_t block_count = std::min<uint32_t>(kLookupBatchSize, count - base);
        std::array<const DirectEntry *, kLookupBatchSize> entries{};
        std::array<const T *, kLookupBatchSize> values{};
        std::array<uint32_t, kLookupBatchSize> hash_slots{};

        for (uint32_t i = 0; i < block_count; ++i) {
            const PreparedQuery &query = queries[base + i];
            if (query.valid == 0U || query.slot >= bucket_slot_count()) {
                continue;
            }
            const DirectEntryIndex &index = layer.direct_entry_indices[query.slot];
            if (index.empty()) {
                continue;
            }
            hash_slots[i] = static_cast<uint32_t>(mix_key64(query.key)) & index.mask;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&index.entries[hash_slots[i]], 0, 1);
#endif
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const PreparedQuery &query = queries[base + i];
            if (query.valid == 0U || query.slot >= bucket_slot_count()) {
                continue;
            }
            const DirectEntryIndex &index = layer.direct_entry_indices[query.slot];
            if (index.empty()) {
                continue;
            }
            uint32_t hash_slot = hash_slots[i];
            for (;;) {
                const DirectEntry &entry = index.entries[hash_slot];
                if (entry.key == kInvalidBucketKey) {
                    break;
                }
                if (entry.key == query.key) {
                    entries[i] = &entry;
                    break;
                }
                hash_slot = (hash_slot + 1U) & index.mask;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&index.entries[hash_slot], 0, 1);
#endif
            }
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const DirectEntry *entry = entries[i];
            if (entry == nullptr) {
                continue;
            }
            const PreparedQuery &query = queries[base + i];
            const BoardSet &set = layer.sets[query.slot];
            if (query.valid_count <= set.threshold_bits) {
                const uint32_t byte_idx = query.rank >> 3U;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&set.small_bitmap_bytes[entry->bitmap_offset + byte_idx], 0, 1);
#endif
            } else {
                const uint32_t word_idx = query.rank >> 6U;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&set.large_bitmap_words[entry->bitmap_offset + word_idx], 0, 1);
                __builtin_prefetch(&layer.large_rank_bases[query.slot][entry->bitmap_offset + word_idx], 0, 1);
#endif
            }
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const DirectEntry *entry = entries[i];
            if (entry == nullptr) {
                continue;
            }
            const PreparedQuery &query = queries[base + i];
            const uint32_t width = layer.row_width[query.slot];
            const BoardSet &set = layer.sets[query.slot];
            if (query.column >= width || query.group >= kSemanticGroupCount) {
                continue;
            }
            uint32_t ordinal = 0;
            if (!dense_ordinal_for_rank_fast(
                    set,
                    *entry,
                    query.valid_count,
                    query.rank,
                    layer.large_rank_bases[query.slot],
                    ordinal)) {
                continue;
            }
            const uint64_t local_row = static_cast<uint64_t>(entry->dense_offset) + ordinal;
            values[i] = row_ptr(layer, query.slot, local_row) + query.column;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(values[i], 0, 1);
#endif
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const T *value_ptr = values[i];
            if (value_ptr == nullptr) {
                continue;
            }
            const PreparedQuery &query = queries[base + i];
            T &slot = best[query.ref];
            if (*value_ptr > slot) {
                slot = *value_ptr;
            }
            ++found;
        }
    }
    return found;
}

template <typename T>
inline uint64_t value_count(const SolvedLayer<T> &layer) {
    return static_cast<uint64_t>(layer.success_values.size());
}

template <typename T>
inline std::pair<uint64_t, T> solved_value_count_and_max(const SolvedLayer<T> &layer, T zero_val) {
    T max_value = zero_val;
    for (const T value : layer.success_values) {
        if (value > max_value) {
            max_value = value;
        }
    }
    return {static_cast<uint64_t>(layer.success_values.size()), max_value};
}

template <typename T>
inline uint64_t metadata_bytes(const SolvedLayer<T> &layer) {
    uint64_t bytes = sizeof(uint64_t) * bucket_slot_count() * 2ULL + sizeof(uint32_t) * bucket_slot_count();
    for (const BoardSet &set : layer.sets) {
        bytes += static_cast<uint64_t>(set.buckets.size()) * sizeof(BucketEntry);
        bytes += static_cast<uint64_t>(set.small_bitmap_bytes.size());
        bytes += static_cast<uint64_t>(set.large_bitmap_words.size()) * sizeof(uint64_t);
    }
    return bytes;
}

template <typename T>
inline double bitmap_density(const SolvedLayer<T> &layer) {
    uint64_t live = 0;
    uint64_t aligned = 0;
    for (const BoardSet &set : layer.sets) {
        live += set.live_board_count;
        aligned += set.aligned_bitmap_bits;
    }
    return aligned == 0 ? 0.0 : static_cast<double>(live) / static_cast<double>(aligned);
}

template <typename T>
inline bool row_has_value_above(const T *row, uint32_t width, T threshold) {
    for (uint32_t col = 0; col < width; ++col) {
        if (row[col] > threshold) {
            return true;
        }
    }
    return false;
}

inline uint32_t bucket_dense_count(const BoardSet &set, uint32_t bucket_idx) {
    const uint32_t begin = set.buckets[bucket_idx].dense_offset;
    const uint32_t end = bucket_idx + 1U < static_cast<uint32_t>(set.buckets.size())
        ? set.buckets[static_cast<size_t>(bucket_idx) + 1U].dense_offset
        : static_cast<uint32_t>(set.live_board_count);
    return end - begin;
}

template <typename T>
inline SolvedLayer<T> compact_solved_layer(
    const SolvedLayer<T> &input,
    const Luts &luts,
    T threshold,
    int num_threads
) {
    SolvedLayer<T> out;
    out.original_board_sum = input.original_board_sum;
    out.threshold_bits = input.threshold_bits;
    out.lut_signature = input.lut_signature;
    out.physical_transform = input.physical_transform;
    out.inverse_physical_transform = input.inverse_physical_transform;
    out.logical_pattern_signature = input.logical_pattern_signature;
    out.physical_pattern_signature = input.physical_pattern_signature;
    out.dtype_mode = input.dtype_mode;
    out.row_width = input.row_width;

    struct CompactPlan {
        uint32_t slot = 0;
        uint32_t bucket_idx = 0;
        uint32_t valid_count = 0;
        uint32_t width = 0;
        bool small = false;
        uint32_t bitmap_units = 0;
        uint32_t kept = 0;
        uint32_t out_bucket_idx = 0;
        uint32_t out_bitmap_offset = 0;
        uint32_t out_dense_offset = 0;
    };

    std::vector<CompactPlan> plans;
    size_t total_bucket_count = 0;
    for (const BoardSet &set : input.sets) {
        total_bucket_count += set.buckets.size();
    }
    plans.reserve(total_bucket_count);
    for (size_t slot = 0; slot < input.sets.size(); ++slot) {
        const BoardSet &in_set = input.sets[slot];
        const uint32_t width = input.row_width[slot];
        for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(in_set.buckets.size()); ++bucket_idx) {
            const BucketEntry &bucket = in_set.buckets[bucket_idx];
            const uint32_t group = lut_group_index(bucket_key_semantic_sum(bucket.key));
            const uint32_t valid_count = luts.size_table[group];
            const bool small = valid_count <= in_set.threshold_bits;
            CompactPlan plan{};
            plan.slot = static_cast<uint32_t>(slot);
            plan.bucket_idx = bucket_idx;
            plan.valid_count = valid_count;
            plan.width = width;
            plan.small = small;
            plan.bitmap_units = small
                ? static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count))
                : static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
            plans.push_back(plan);
        }
    }

    auto count_kept = [&](CompactPlan &plan) {
        const BoardSet &in_set = input.sets[plan.slot];
        const BucketEntry &bucket = in_set.buckets[plan.bucket_idx];
        uint32_t kept = 0;
        const uint32_t dense_count = bucket_dense_count(in_set, plan.bucket_idx);
        for (uint32_t ordinal = 0; ordinal < dense_count; ++ordinal) {
            const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
            const T *src = row_ptr(input, plan.slot, local_row);
            if (row_has_value_above(src, plan.width, threshold)) {
                ++kept;
            }
        }
        plan.kept = kept;
    };

    const int requested_threads = std::max(1, num_threads);
#pragma omp parallel for schedule(dynamic, 64) num_threads(requested_threads) if(requested_threads > 1 && plans.size() > 64)
    for (int64_t plan_idx = 0; plan_idx < static_cast<int64_t>(plans.size()); ++plan_idx) {
        count_kept(plans[static_cast<size_t>(plan_idx)]);
    }

    std::array<uint32_t, bucket_slot_count()> out_bucket_counts{};
    std::array<uint64_t, bucket_slot_count()> out_small_bytes{};
    std::array<uint64_t, bucket_slot_count()> out_large_words{};
    std::array<uint64_t, bucket_slot_count()> out_live_rows{};
    std::array<uint64_t, bucket_slot_count()> out_exact_bits{};
    std::array<uint64_t, bucket_slot_count()> out_aligned_bits{};

    for (CompactPlan &plan : plans) {
        if (plan.kept == 0U) {
            continue;
        }
        const uint32_t slot = plan.slot;
        plan.out_bucket_idx = out_bucket_counts[slot]++;
        plan.out_dense_offset = static_cast<uint32_t>(out_live_rows[slot]);
        out_live_rows[slot] += plan.kept;
        out_exact_bits[slot] += plan.valid_count;
        if (plan.small) {
            plan.out_bitmap_offset = static_cast<uint32_t>(out_small_bytes[slot]);
            out_small_bytes[slot] += plan.bitmap_units;
            out_aligned_bits[slot] += static_cast<uint64_t>(plan.bitmap_units) * 8ULL;
        } else {
            plan.out_bitmap_offset = static_cast<uint32_t>(out_large_words[slot]);
            out_large_words[slot] += plan.bitmap_units;
            out_aligned_bits[slot] += static_cast<uint64_t>(plan.bitmap_units) * 64ULL;
        }
    }

    uint64_t total_rows = 0;
    uint64_t total_values = 0;
    for (size_t slot = 0; slot < out.sets.size(); ++slot) {
        const BoardSet &in_set = input.sets[slot];
        BoardSet &out_set = out.sets[slot];
        out_set.threshold_bits = in_set.threshold_bits;
        out_set.live_board_count = out_live_rows[slot];
        out_set.exact_bitmap_bits = out_exact_bits[slot];
        out_set.aligned_bitmap_bits = out_aligned_bits[slot];
        out_set.buckets.resize(out_bucket_counts[slot]);
        out_set.small_bitmap_bytes.assign(static_cast<size_t>(out_small_bytes[slot]), 0U);
        out_set.large_bitmap_words.assign(static_cast<size_t>(out_large_words[slot]), 0ULL);
        out.slot_row_base[slot] = total_rows;
        out.slot_value_base[slot] = total_values;
        total_rows += out_live_rows[slot];
        total_values += out_live_rows[slot] * static_cast<uint64_t>(out.row_width[slot]);
    }
    out.live_board_count = total_rows;
    out.success_values.resize(static_cast<size_t>(total_values));
    fill_success_values(out.success_values, T{}, num_threads);

    auto rewrite_bucket = [&](const CompactPlan &plan) {
        if (plan.kept == 0U) {
            return;
        }
        const BoardSet &in_set = input.sets[plan.slot];
        const BucketEntry &bucket = in_set.buckets[plan.bucket_idx];
        BoardSet &out_set = out.sets[plan.slot];
        BucketEntry &out_bucket = out_set.buckets[plan.out_bucket_idx];
        out_bucket.key = bucket.key;
        out_bucket.bitmap_offset = plan.out_bitmap_offset;
        out_bucket.dense_offset = plan.out_dense_offset;

        uint32_t ordinal = 0;
        uint32_t kept = 0;
        auto keep_rank = [&](uint32_t rank) {
            const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
            const T *src = row_ptr(input, plan.slot, local_row);
            if (!row_has_value_above(src, plan.width, threshold)) {
                ++ordinal;
                return;
            }
            if (plan.small) {
                const uint32_t byte_idx = rank >> 3U;
                const uint32_t bit = rank & 7U;
                const uint32_t target = plan.out_bitmap_offset + byte_idx;
                out_set.small_bitmap_bytes[target] = static_cast<uint8_t>(out_set.small_bitmap_bytes[target] | (1U << bit));
            } else {
                const uint32_t word_idx = rank >> 6U;
                const uint32_t bit = rank & 63U;
                out_set.large_bitmap_words[plan.out_bitmap_offset + word_idx] |= (1ULL << bit);
            }
            T *dst = out.success_values.data()
                + static_cast<size_t>(out.slot_value_base[plan.slot]
                    + (static_cast<uint64_t>(plan.out_dense_offset) + kept) * plan.width);
            std::copy(src, src + plan.width, dst);
            ++kept;
            ++ordinal;
        };

        if (plan.small) {
            for (uint32_t byte_idx = 0; byte_idx < plan.bitmap_units; ++byte_idx) {
                uint8_t value = in_set.small_bitmap_bytes[bucket.bitmap_offset + byte_idx];
                while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                    const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(value));
#else
                    uint32_t bit = 0;
                    while (((value >> bit) & 1U) == 0U) {
                        ++bit;
                    }
#endif
                    const uint32_t rank = byte_idx * 8U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    keep_rank(rank);
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                }
            }
        } else {
            for (uint32_t word_idx = 0; word_idx < plan.bitmap_units; ++word_idx) {
                uint64_t value = in_set.large_bitmap_words[bucket.bitmap_offset + word_idx];
                while (value != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
                    const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(value));
#else
                    uint32_t bit = 0;
                    while (((value >> bit) & 1ULL) == 0ULL) {
                        ++bit;
                    }
#endif
                    const uint32_t rank = word_idx * 64U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    keep_rank(rank);
                    value &= value - 1ULL;
                }
            }
        }
    };

#pragma omp parallel for schedule(dynamic, 64) num_threads(requested_threads) if(requested_threads > 1 && plans.size() > 64)
    for (int64_t plan_idx = 0; plan_idx < static_cast<int64_t>(plans.size()); ++plan_idx) {
        rewrite_bucket(plans[static_cast<size_t>(plan_idx)]);
    }

    return out;
}

namespace detail {

struct SolvedFileHeader {
    char magic[8];
    uint32_t version = kSolvedFileVersion;
    uint32_t dtype_mode = 0;
    uint32_t value_size = 0;
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t slot_count = static_cast<uint32_t>(bucket_slot_count());
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved = 0;
    uint64_t lut_signature = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
    uint64_t success_value_count = 0;
};

struct SolvedSlotHeader {
    uint64_t bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    uint64_t row_base = 0;
    uint64_t value_base = 0;
    uint32_t row_width = 0;
    uint32_t reserved = 0;
};

inline constexpr char kSolvedMagic[8] = {'E', 'X', 'A', 'D', '7', 'S', 'L', 'V'};

inline uint64_t file_size_or_throw(const std::string &path, const char *kind) {
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(std::string("failed to determine EXAD ") + kind + " file size: " + path);
    }
    return static_cast<uint64_t>(size);
}

inline void read_direct_exact(
    FileIOUtils::DirectSequentialReader &in,
    void *dst,
    size_t bytes,
    const std::string &
) {
    if (bytes != 0U) {
        in.read(dst, bytes);
    }
}

} // namespace detail

template <typename T>
inline uint64_t solved_serialized_size(const SolvedLayer<T> &layer) {
    uint64_t size = sizeof(detail::SolvedFileHeader)
        + sizeof(detail::SolvedSlotHeader) * bucket_slot_count();
    for (const BoardSet &set : layer.sets) {
        size += static_cast<uint64_t>(set.buckets.size()) * sizeof(BucketEntry);
        size += static_cast<uint64_t>(set.small_bitmap_bytes.size());
        size += static_cast<uint64_t>(set.large_bitmap_words.size()) * sizeof(uint64_t);
    }
    size += static_cast<uint64_t>(layer.success_values.size()) * sizeof(T);
    return size;
}

template <typename T>
inline void write_solved_layer_file(
    const std::string &path,
    const SolvedLayer<T> &layer,
    FileIOUtils::DirectIoConfig config = {}
) {
    if (std::filesystem::is_directory(path)) {
        throw std::runtime_error("refusing to overwrite old EXAD directory-format solved layer: " + path);
    }
    detail::SolvedFileHeader header{};
    std::memcpy(header.magic, detail::kSolvedMagic, sizeof(header.magic));
    header.dtype_mode = static_cast<uint32_t>(layer.dtype_mode);
    header.value_size = sizeof(T);
    header.original_board_sum = layer.original_board_sum;
    header.threshold_bits = layer.threshold_bits;
    header.physical_transform = layer.physical_transform;
    header.inverse_physical_transform = layer.inverse_physical_transform;
    header.lut_signature = layer.lut_signature;
    header.logical_pattern_signature = layer.logical_pattern_signature;
    header.physical_pattern_signature = layer.physical_pattern_signature;
    header.live_board_count = layer.live_board_count;
    header.success_value_count = layer.success_values.size();

    std::array<detail::SolvedSlotHeader, bucket_slot_count()> slots{};
    for (size_t i = 0; i < layer.sets.size(); ++i) {
        const BoardSet &set = layer.sets[i];
        slots[i].bucket_count = set.buckets.size();
        slots[i].small_bitmap_bytes = set.small_bitmap_bytes.size();
        slots[i].large_bitmap_words = set.large_bitmap_words.size();
        slots[i].live_board_count = set.live_board_count;
        slots[i].exact_bitmap_bits = set.exact_bitmap_bits;
        slots[i].aligned_bitmap_bits = set.aligned_bitmap_bits;
        slots[i].row_base = layer.slot_row_base[i];
        slots[i].value_base = layer.slot_value_base[i];
        slots[i].row_width = layer.row_width[i];
    }

    FileIOUtils::DirectAppendWriter out(path, solved_serialized_size(layer), config);
    out.append(&header, sizeof(header));
    out.append(slots.data(), slots.size() * sizeof(slots[0]));
    for (const BoardSet &set : layer.sets) {
        if (!set.buckets.empty()) {
            out.append(set.buckets.data(), set.buckets.size() * sizeof(BucketEntry));
        }
        if (!set.small_bitmap_bytes.empty()) {
            out.append(set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size());
        }
        if (!set.large_bitmap_words.empty()) {
            out.append(set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t));
        }
    }
    if (!layer.success_values.empty()) {
        out.append(layer.success_values.data(), layer.success_values.size() * sizeof(T));
    }
    out.close();
}

template <typename T>
inline SolvedLayer<T> read_solved_layer_file(
    const std::string &path,
    DTypeMode expected_mode,
    FileIOUtils::DirectIoConfig config = {}
) {
    if (std::filesystem::is_directory(path)) {
        throw std::runtime_error("old EXAD directory-format solved layer is not compatible: " + path);
    }
    FileIOUtils::DirectSequentialReader in(
        path,
        detail::file_size_or_throw(path, "solved layer"),
        config
    );
    detail::SolvedFileHeader header{};
    detail::read_direct_exact(in, &header, sizeof(header), path);
    if (std::memcmp(header.magic, detail::kSolvedMagic, sizeof(header.magic)) != 0 ||
        header.version != kSolvedFileVersion ||
        header.slot_count != static_cast<uint32_t>(bucket_slot_count())) {
        throw std::runtime_error("invalid EXAD solved layer file magic/version: " + path);
    }
    const DTypeMode file_mode = static_cast<DTypeMode>(header.dtype_mode);
    if (file_mode != expected_mode || !dtype_matches_type<T>(file_mode) || header.value_size != sizeof(T)) {
        throw std::runtime_error("EXAD solved layer dtype mismatch: " + path);
    }

    std::array<detail::SolvedSlotHeader, bucket_slot_count()> slots{};
    detail::read_direct_exact(in, slots.data(), slots.size() * sizeof(slots[0]), path);

    SolvedLayer<T> layer;
    layer.dtype_mode = file_mode;
    layer.original_board_sum = header.original_board_sum;
    layer.threshold_bits = header.threshold_bits;
    layer.lut_signature = header.lut_signature;
    layer.physical_transform = header.physical_transform;
    layer.inverse_physical_transform = header.inverse_physical_transform;
    layer.logical_pattern_signature = header.logical_pattern_signature;
    layer.physical_pattern_signature = header.physical_pattern_signature;
    layer.live_board_count = header.live_board_count;
    for (size_t i = 0; i < layer.sets.size(); ++i) {
        BoardSet &set = layer.sets[i];
        set.threshold_bits = header.threshold_bits;
        set.live_board_count = slots[i].live_board_count;
        set.exact_bitmap_bits = slots[i].exact_bitmap_bits;
        set.aligned_bitmap_bits = slots[i].aligned_bitmap_bits;
        set.buckets.resize(static_cast<size_t>(slots[i].bucket_count));
        set.small_bitmap_bytes.resize(static_cast<size_t>(slots[i].small_bitmap_bytes));
        set.large_bitmap_words.resize(static_cast<size_t>(slots[i].large_bitmap_words));
        layer.slot_row_base[i] = slots[i].row_base;
        layer.slot_value_base[i] = slots[i].value_base;
        layer.row_width[i] = slots[i].row_width;
        if (!set.buckets.empty()) {
            detail::read_direct_exact(in, set.buckets.data(), set.buckets.size() * sizeof(BucketEntry), path);
        }
        if (!set.small_bitmap_bytes.empty()) {
            detail::read_direct_exact(in, set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size(), path);
        }
        if (!set.large_bitmap_words.empty()) {
            detail::read_direct_exact(in, set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t), path);
        }
    }
    layer.success_values.resize(static_cast<size_t>(header.success_value_count));
    if (!layer.success_values.empty()) {
        detail::read_direct_exact(in, layer.success_values.data(), layer.success_values.size() * sizeof(T), path);
    }
    in.close();
    return layer;
}

inline bool solved_file_exists(const std::string &path) {
    if (std::filesystem::is_directory(path)) {
        throw std::runtime_error("old EXAD directory-format solved layer is not compatible: " + path);
    }
    if (!std::filesystem::exists(path)) {
        return false;
    }
    if (std::filesystem::file_size(path) < sizeof(detail::SolvedFileHeader)) {
        throw std::runtime_error("incomplete EXAD solved layer file: " + path);
    }
    return true;
}

} // namespace EXAD
