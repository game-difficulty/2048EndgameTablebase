#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"
#include "EXFrozenLayer.h"
#include "EXPrefix40Layer.h"
#include "NativeSortPolicy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t kPrefixBits = 36U;
constexpr uint32_t kSuffixBits = 28U;
constexpr uint64_t kSuffixMask = (1ULL << kSuffixBits) - 1ULL;
constexpr uint32_t kMaxSuffix28Sum = 7U * (1U << 15U);
constexpr uint32_t kInvalidSuffix = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kBatchSize = 256U;

constexpr uint32_t kDirectIndexLoadPercent = 25U;
constexpr int kRecalcDynamicChunk = 16;

uint32_t large_rank_bases_for_words(uint32_t words) {
    return words;
}

double wall_time_seconds() {
#if defined(_OPENMP)
    return omp_get_wtime();
#else
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    return std::chrono::duration<double>(clock::now() - epoch).count();
#endif
}

double mbps_for(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1e6 : 0.0;
}

uint32_t popcount_u32(uint32_t value) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt(value));
#else
    return static_cast<uint32_t>(__builtin_popcount(value));
#endif
}

uint32_t popcount_u64(uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<uint32_t>(__popcnt64(value));
#elif defined(_MSC_VER)
    return popcount_u32(static_cast<uint32_t>(value)) + popcount_u32(static_cast<uint32_t>(value >> 32U));
#else
    return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
}

uint32_t countr_zero_u32(uint32_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return static_cast<uint32_t>(index);
#else
    return static_cast<uint32_t>(__builtin_ctz(value));
#endif
}

uint32_t zero_cell_mask16(uint64_t board) {
    constexpr uint64_t kNibbleLsbMask = 0x1111111111111111ULL;
    const uint64_t nonzero_lsb =
        (board | (board >> 1U) | (board >> 2U) | (board >> 3U)) & kNibbleLsbMask;
    const uint64_t zero_lsb = (~nonzero_lsb) & kNibbleLsbMask;
#if defined(__BMI2__)
    return static_cast<uint32_t>(_pext_u64(zero_lsb, kNibbleLsbMask));
#else
    uint32_t mask = 0U;
    for (uint32_t cell = 0; cell < 16U; ++cell) {
        if (((zero_lsb >> (4U * cell)) & 1ULL) != 0ULL) {
            mask |= (1U << cell);
        }
    }
    return mask;
#endif
}

uint32_t countr_zero_u64(uint64_t value) {
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

uint64_t mix_u64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

bool is_prime_u64(uint64_t value) {
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

uint64_t next_prime_u64(uint64_t value) {
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

uint64_t choose_direct_table_size(uint64_t bucket_count) {
    constexpr uint64_t kMinTableSize = 1024ULL;
    constexpr uint64_t kLoadDenominator = 100ULL;
    constexpr uint64_t kLoadNumerator = kDirectIndexLoadPercent;
    const uint64_t required =
        (bucket_count * kLoadDenominator + (kLoadNumerator - 1ULL)) / kLoadNumerator;
    return next_prime_u64(std::max<uint64_t>(kMinTableSize, required));
}

uint64_t choose_suffix_meta_table_size(uint64_t item_count) {
    constexpr uint64_t kMinTableSize = 1024ULL;
    constexpr uint64_t kLoadNumerator = 70ULL;
    constexpr uint64_t kLoadDenominator = 100ULL;
    const uint64_t required =
        (item_count * kLoadDenominator + (kLoadNumerator - 1ULL)) / kLoadNumerator;
    return next_prime_u64(std::max<uint64_t>(kMinTableSize, required));
}

uint32_t sum_index(uint32_t sum) {
    if ((sum & 1U) != 0U) {
        throw std::runtime_error("prefix36 suffix28 LUT encountered odd tile sum");
    }
    return sum >> 1U;
}

uint64_t bytes_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 7U) >> 3U);
}

uint64_t words_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 63U) >> 6U);
}

uint32_t tile_value(uint32_t tile) {
    return tile == 0U ? 0U : (1U << tile);
}

uint32_t tile_half_value(uint32_t tile) {
    static constexpr std::array<uint32_t, 16> kHalfValues = {
        0U, 1U, 2U, 4U, 8U, 16U, 32U, 64U,
        128U, 256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U
    };
    return kHalfValues[tile & 0xFU];
}

uint64_t pack_key(uint64_t prefix36, uint32_t remaining_sum) {
    return (prefix36 << 18U) | static_cast<uint64_t>(remaining_sum);
}

uint64_t key_prefix36(uint64_t key) {
    return key >> 18U;
}

uint32_t key_remaining_sum(uint64_t key) {
    return static_cast<uint32_t>(key & ((1ULL << 18U) - 1ULL));
}

void set_small_bit(std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    bitmap[offset + (rank >> 3U)] =
        static_cast<uint8_t>(bitmap[offset + (rank >> 3U)] | static_cast<uint8_t>(1U << (rank & 7U)));
}

void set_large_bit(std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    bitmap[offset + (rank >> 6U)] |= (1ULL << (rank & 63U));
}

bool test_small_bit(const std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    return (bitmap[offset + (rank >> 3U)] & static_cast<uint8_t>(1U << (rank & 7U))) != 0U;
}

bool test_large_bit(const std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    return (bitmap[offset + (rank >> 6U)] & (1ULL << (rank & 63U))) != 0ULL;
}

struct SuffixState {
    uint32_t suffix = 0;
    uint32_t sum = 0;
};

struct SuffixMeta {
    uint32_t rank = 0;
    uint32_t sum = 0;
};

struct Suffix28Lut {
    std::vector<uint32_t> size_table;
    std::vector<uint32_t> slot_suffixes;
    std::vector<SuffixMeta> slot_meta;
    uint64_t table_size = 0;
    uint64_t valid_suffix_count = 0;
};

struct DenseLow24RankLut {
    static constexpr uint8_t kInvalidTable = std::numeric_limits<uint8_t>::max();
    std::vector<std::vector<uint16_t>> rank_tables;
    std::vector<uint32_t> packed_rank_pair_table;
    std::vector<uint64_t> packed_meta_table;
    std::vector<std::vector<uint32_t>> counts_by_table;
    std::vector<uint32_t> size_table;
    std::vector<uint32_t> offset_table;
    std::vector<uint32_t> unrank_array;
    std::vector<uint16_t> high_base;
    std::array<uint8_t, 16> table_for_high{};
    uint8_t packed_table0 = kInvalidTable;
    uint8_t packed_table1 = kInvalidTable;
    uint32_t rank_table_variant_count = 0;
    uint64_t valid_suffix_count = 0;
    uint64_t rank_table_bytes = 0;
    uint64_t packed_meta_bytes = 0;
    uint64_t unrank_bytes = 0;
};

uint16_t dense_low24_rank_for_table(const DenseLow24RankLut &lut, uint8_t table_id, uint32_t low24);
uint16_t dense_low24_rank_for_high(const DenseLow24RankLut &lut, uint32_t high, uint32_t low24);

uint32_t low24_sum_fast(uint32_t low24) {
    static std::array<uint32_t, 1U << 16U> row16_sum{};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t row = 0; row < row16_sum.size(); ++row) {
            uint32_t sum = 0U;
            for (uint32_t cell = 0; cell < 4U; ++cell) {
                sum += tile_value((row >> (cell * 4U)) & 0xFU);
            }
            row16_sum[row] = sum;
        }
        initialized = true;
    }
    return row16_sum[low24 & 0xFFFFU] + row16_sum[(low24 >> 16U) & 0xFFU];
}

uint64_t suffix_slot(const Suffix28Lut &lut, uint32_t suffix) {
    const uint64_t mixed = mix_u64(static_cast<uint64_t>(suffix));
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(mixed) * lut.table_size) >> 64U);
#else
    return mixed % lut.table_size;
#endif
}

const SuffixMeta *lookup_suffix_meta(const Suffix28Lut &lut, uint32_t suffix) {
    uint64_t slot = suffix_slot(lut, suffix);
    for (;;) {
        const uint32_t found = lut.slot_suffixes[static_cast<size_t>(slot)];
        if (found == suffix) {
            return &lut.slot_meta[static_cast<size_t>(slot)];
        }
        if (found == kInvalidSuffix) {
            return nullptr;
        }
        ++slot;
        if (slot == lut.table_size) {
            slot = 0ULL;
        }
    }
}

bool tile_limit_allows(const ZMaskFrozen::TileLimitConfig &config, uint32_t tile, uint8_t new_count) {
    const int8_t limit = config.max_counts[tile];
    return limit < 0 || new_count <= static_cast<uint8_t>(limit);
}

bool suffix28_structural_constraints_allow(const ZMaskFrozen::TileLimitConfig &config, uint32_t suffix) {
    if ((suffix & config.required_suffix24) != config.required_suffix24) {
        return false;
    }
    if (config.valid_suffix_masks.empty()) {
        return true;
    }
    for (const uint32_t mask : config.valid_suffix_masks) {
        if ((suffix & mask) == mask) {
            return true;
        }
    }
    return false;
}

void enumerate_suffix28_states_recursive(
    const ZMaskFrozen::TileLimitConfig &config,
    uint32_t pos,
    uint32_t suffix,
    uint32_t sum,
    std::array<uint8_t, 16> &counts,
    std::vector<SuffixState> &states
) {
    if (pos == 7U) {
        if (suffix28_structural_constraints_allow(config, suffix)) {
            states.push_back(SuffixState{suffix, sum});
        }
        return;
    }
    for (uint32_t tile = 0; tile < 16U; ++tile) {
        const int8_t limit = config.max_counts[tile];
        if (limit == 0) {
            continue;
        }
        const uint8_t next_count = static_cast<uint8_t>(counts[tile] + 1U);
        if (!tile_limit_allows(config, tile, next_count)) {
            continue;
        }
        counts[tile] = next_count;
        enumerate_suffix28_states_recursive(
            config,
            pos + 1U,
            suffix | (tile << (4U * pos)),
            sum + tile_value(tile),
            counts,
            states
        );
        counts[tile] = static_cast<uint8_t>(counts[tile] - 1U);
    }
}

Suffix28Lut build_suffix28_lut(const ZMaskFrozen::TileLimitConfig &config) {
    std::vector<SuffixState> states;
    states.reserve(8U * 1024U * 1024U);
    std::array<uint8_t, 16> counts{};
    enumerate_suffix28_states_recursive(config, 0U, 0U, 0U, counts, states);
    std::sort(states.begin(), states.end(), [](const SuffixState &lhs, const SuffixState &rhs) {
        return lhs.suffix < rhs.suffix;
    });

    Suffix28Lut lut;
    lut.valid_suffix_count = states.size();
    lut.size_table.assign((kMaxSuffix28Sum >> 1U) + 1U, 0U);
    for (const SuffixState &state : states) {
        ++lut.size_table[sum_index(state.sum)];
    }
    lut.table_size = choose_suffix_meta_table_size(states.size());
    lut.slot_suffixes.assign(static_cast<size_t>(lut.table_size), kInvalidSuffix);
    lut.slot_meta.assign(static_cast<size_t>(lut.table_size), SuffixMeta{});
    std::vector<uint32_t> rank_cursor(lut.size_table.size(), 0U);
    for (const SuffixState &state : states) {
        const uint32_t group = sum_index(state.sum);
        const uint32_t rank = rank_cursor[group]++;
        uint64_t slot = suffix_slot(lut, state.suffix);
        while (lut.slot_suffixes[static_cast<size_t>(slot)] != kInvalidSuffix) {
            ++slot;
            if (slot == lut.table_size) {
                slot = 0ULL;
            }
        }
        lut.slot_suffixes[static_cast<size_t>(slot)] = state.suffix;
        lut.slot_meta[static_cast<size_t>(slot)] = SuffixMeta{rank, state.sum};
    }
    return lut;
}

ZMaskFrozen::TileLimitConfig adjusted_low24_config_for_high(
    const ZMaskFrozen::TileLimitConfig &config,
    uint32_t high_tile,
    bool &valid_high
) {
    ZMaskFrozen::TileLimitConfig adjusted = config;
    valid_high = true;
    const int8_t limit = adjusted.max_counts[high_tile];
    if (limit == 0) {
        valid_high = false;
        return adjusted;
    }
    if (limit > 0) {
        adjusted.max_counts[high_tile] = static_cast<int8_t>(limit - 1);
    }
    return adjusted;
}

uint64_t config_signature_for_rank_table(const ZMaskFrozen::TileLimitConfig &config) {
    uint64_t hash = 1469598103934665603ULL;
    for (int8_t value : config.max_counts) {
        hash ^= static_cast<uint8_t>(value);
        hash *= 1099511628211ULL;
    }
    hash ^= config.required_suffix24;
    hash *= 1099511628211ULL;
    hash ^= static_cast<uint64_t>(config.valid_suffix_masks.size());
    hash *= 1099511628211ULL;
    for (uint32_t mask : config.valid_suffix_masks) {
        hash ^= mask;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::pair<std::vector<uint16_t>, std::vector<uint32_t>> build_low24_rank_table(
    const ZMaskFrozen::TileLimitConfig &config
) {
    std::vector<uint16_t> rank_table(ZMaskFrozen::kSuffixStateCount, ZMaskFrozen::kInvalidRank);
    std::vector<uint32_t> counts((6U * (1U << 15U) >> 1U) + 1U, 0U);
    std::vector<uint32_t> sums(ZMaskFrozen::kSuffixStateCount, 0U);
    std::vector<uint8_t> valid(ZMaskFrozen::kSuffixStateCount, 0U);
    for (uint32_t state = 0; state < ZMaskFrozen::kSuffixStateCount; ++state) {
        uint32_t sum = 0U;
        if (!ZMaskFrozen::decode_suffix24_if_valid(state, sum, config)) {
            continue;
        }
        valid[state] = 1U;
        sums[state] = sum;
        ++counts[sum_index(sum)];
    }
    std::vector<uint32_t> cursor(counts.size(), 0U);
    for (uint32_t state = 0; state < ZMaskFrozen::kSuffixStateCount; ++state) {
        if (valid[state] == 0U) {
            continue;
        }
        const uint32_t idx = sum_index(sums[state]);
        const uint32_t rank = cursor[idx]++;
        if (rank > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("dense low24 rank exceeds uint16 range");
        }
        rank_table[state] = static_cast<uint16_t>(rank);
    }
    return {std::move(rank_table), std::move(counts)};
}

DenseLow24RankLut build_dense_low24_rank_lut(const ZMaskFrozen::TileLimitConfig &config) {
    DenseLow24RankLut lut;
    lut.table_for_high.fill(DenseLow24RankLut::kInvalidTable);
    lut.size_table.assign((kMaxSuffix28Sum >> 1U) + 1U, 0U);
    lut.high_base.assign(16ULL * lut.size_table.size(), 0U);

    std::vector<uint64_t> signatures;
    for (uint32_t high = 0; high < 16U; ++high) {
        bool valid_high = false;
        ZMaskFrozen::TileLimitConfig adjusted = adjusted_low24_config_for_high(config, high, valid_high);
        if (!valid_high) {
            continue;
        }
        const uint64_t signature = config_signature_for_rank_table(adjusted);
        auto it = std::find(signatures.begin(), signatures.end(), signature);
        uint8_t table_id = 0U;
        if (it == signatures.end()) {
            auto built = build_low24_rank_table(adjusted);
            if (lut.rank_tables.size() >= DenseLow24RankLut::kInvalidTable) {
                throw std::runtime_error("too many dense low24 rank table variants");
            }
            table_id = static_cast<uint8_t>(lut.rank_tables.size());
            signatures.push_back(signature);
            lut.rank_tables.push_back(std::move(built.first));
            lut.counts_by_table.push_back(std::move(built.second));
        } else {
            table_id = static_cast<uint8_t>(std::distance(signatures.begin(), it));
        }
        lut.table_for_high[high] = table_id;
    }

    for (uint32_t high = 0; high < 16U; ++high) {
        const uint8_t table_id = lut.table_for_high[high];
        if (table_id == DenseLow24RankLut::kInvalidTable) {
            continue;
        }
        const uint32_t high_value = tile_value(high);
        const std::vector<uint32_t> &counts = lut.counts_by_table[table_id];
        for (uint32_t low_idx = 0; low_idx < counts.size(); ++low_idx) {
            const uint32_t count = counts[low_idx];
            if (count == 0U) {
                continue;
            }
            const uint32_t low_sum = low_idx << 1U;
            const uint32_t total_sum = low_sum + high_value;
            if (total_sum > kMaxSuffix28Sum) {
                continue;
            }
            lut.size_table[sum_index(total_sum)] += count;
        }
    }

    for (uint32_t high = 0; high < 16U; ++high) {
        for (uint32_t total_idx = 0; total_idx < lut.size_table.size(); ++total_idx) {
            const uint32_t total_sum = total_idx << 1U;
        uint32_t base = 0U;
            for (uint32_t prev_high = 0; prev_high < high; ++prev_high) {
                const uint8_t table_id = lut.table_for_high[prev_high];
                if (table_id == DenseLow24RankLut::kInvalidTable) {
                    continue;
                }
                const uint32_t high_value = tile_value(prev_high);
                if (high_value > total_sum) {
                    continue;
                }
                const uint32_t low_sum = total_sum - high_value;
                if ((low_sum & 1U) != 0U) {
                    continue;
                }
                const uint32_t low_idx = low_sum >> 1U;
                const std::vector<uint32_t> &counts = lut.counts_by_table[table_id];
                if (low_idx < counts.size()) {
                    base += counts[low_idx];
                }
            }
            if (base > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("dense suffix28 high_base exceeds uint16 range");
            }
            lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + total_idx] =
                static_cast<uint16_t>(base);
        }
    }

    lut.offset_table.resize(lut.size_table.size(), 0U);
    uint64_t unrank_total = 0U;
    for (uint32_t group = 0; group < static_cast<uint32_t>(lut.size_table.size()); ++group) {
        lut.offset_table[group] = static_cast<uint32_t>(unrank_total);
        unrank_total += lut.size_table[group];
    }
    if (unrank_total > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("dense suffix28 unrank table exceeds uint32 offsets");
    }
    lut.unrank_array.assign(static_cast<size_t>(unrank_total), kInvalidSuffix);
    for (uint32_t high = 0; high < 16U; ++high) {
        const uint8_t table_id = lut.table_for_high[high];
        if (table_id == DenseLow24RankLut::kInvalidTable) {
            continue;
        }
        const uint32_t high_value = tile_value(high);
        const std::vector<uint16_t> &rank_table = lut.rank_tables[table_id];
        for (uint32_t low24 = 0; low24 < ZMaskFrozen::kSuffixStateCount; ++low24) {
            const uint16_t low_rank = rank_table[low24];
            if (low_rank == ZMaskFrozen::kInvalidRank) {
                continue;
            }
            const uint32_t total_sum = low24_sum_fast(low24) + high_value;
            if (total_sum > kMaxSuffix28Sum) {
                continue;
            }
            const uint32_t group = sum_index(total_sum);
            const uint32_t rank =
                lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
            if (rank >= lut.size_table[group]) {
                std::ostringstream oss;
                oss << "dense suffix28 unrank rank exceeds group size"
                    << " high=" << high
                    << " low24=" << low24
                    << " total_sum=" << total_sum
                    << " group=" << group
                    << " low_rank=" << low_rank
                    << " base=" << lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group]
                    << " size=" << lut.size_table[group];
                throw std::runtime_error(oss.str());
            }
            lut.unrank_array[static_cast<size_t>(lut.offset_table[group]) + rank] =
                low24 | (high << 24U);
        }
    }
    lut.unrank_bytes = static_cast<uint64_t>(lut.unrank_array.size()) * sizeof(uint32_t);

    lut.rank_table_variant_count = static_cast<uint32_t>(lut.rank_tables.size());
    if (lut.rank_tables.size() == 2U) {
        lut.packed_table0 = 0U;
        lut.packed_table1 = 1U;
        lut.packed_rank_pair_table.resize(ZMaskFrozen::kSuffixStateCount);
        for (uint32_t state = 0; state < ZMaskFrozen::kSuffixStateCount; ++state) {
            lut.packed_rank_pair_table[state] =
                static_cast<uint32_t>(lut.rank_tables[0][state]) |
                (static_cast<uint32_t>(lut.rank_tables[1][state]) << 16U);
        }
        lut.rank_table_bytes = static_cast<uint64_t>(lut.packed_rank_pair_table.size()) * sizeof(uint32_t);
        std::vector<std::vector<uint16_t>>().swap(lut.rank_tables);
    } else {
        for (const auto &table : lut.rank_tables) {
            lut.rank_table_bytes += static_cast<uint64_t>(table.size()) * sizeof(uint16_t);
        }
    }
    for (uint32_t count : lut.size_table) {
        lut.valid_suffix_count += count;
    }
    return lut;
}

uint16_t dense_low24_rank_from_source_for_table(
    const DenseLow24RankLut &lut,
    uint8_t table_id,
    uint32_t low24
) {
    if (!lut.packed_meta_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint64_t packed = lut.packed_meta_table[low24];
        return table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>((packed >> 16U) & 0xFFFFU);
    }
    if (!lut.packed_rank_pair_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t packed = lut.packed_rank_pair_table[low24];
        return table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
    }
    if (table_id >= lut.rank_tables.size()) {
        return ZMaskFrozen::kInvalidRank;
    }
    return lut.rank_tables[table_id][low24];
}

uint32_t dense_low24_rank_variant_count(const DenseLow24RankLut &lut) {
    uint32_t variant_count = lut.rank_table_variant_count;
    variant_count = std::max<uint32_t>(variant_count, static_cast<uint32_t>(lut.rank_tables.size()));
    if (!lut.packed_rank_pair_table.empty() || !lut.packed_meta_table.empty()) {
        if (lut.packed_table0 != DenseLow24RankLut::kInvalidTable) {
            variant_count = std::max<uint32_t>(variant_count, static_cast<uint32_t>(lut.packed_table0) + 1U);
        }
        if (lut.packed_table1 != DenseLow24RankLut::kInvalidTable) {
            variant_count = std::max<uint32_t>(variant_count, static_cast<uint32_t>(lut.packed_table1) + 1U);
        }
    }
    return variant_count;
}

uint16_t dense_low24_rank_for_table(const DenseLow24RankLut &lut, uint8_t table_id, uint32_t low24) {
    return dense_low24_rank_from_source_for_table(lut, table_id, low24);
}

struct BucketEntry {
    static constexpr uint32_t kRankOffsetMask = 0x0FFFFFFFU;
    static constexpr uint32_t kSmallRankOffset = kRankOffsetMask;
    uint32_t prefix_low32 = 0;
    uint32_t bitmap_offset = 0;
    uint32_t success_offset = 0;
    uint32_t rank_offset_prefix_high = kSmallRankOffset;
};
static_assert(sizeof(BucketEntry) == 16, "BucketEntry must stay compact");

uint32_t pack_rank_offset_prefix_high(uint64_t prefix36, uint32_t rank_offset) {
    if ((prefix36 >> 36U) != 0ULL) {
        throw std::runtime_error("prefix36 entry prefix overflow");
    }
    const uint32_t prefix_high = static_cast<uint32_t>(prefix36 >> 32U);
    if ((rank_offset & ~BucketEntry::kRankOffsetMask) != 0U) {
        throw std::runtime_error("prefix36 entry rank offset exceeds packed 28-bit storage");
    }
    return (prefix_high << 28U) | rank_offset;
}

bool bucket_prefix_matches(const BucketEntry &bucket, uint64_t prefix36) {
    return bucket.prefix_low32 == static_cast<uint32_t>(prefix36) &&
           (bucket.rank_offset_prefix_high >> 28U) == static_cast<uint32_t>(prefix36 >> 32U);
}

bool bucket_is_small(const BucketEntry &bucket) {
    return (bucket.rank_offset_prefix_high & BucketEntry::kRankOffsetMask) == BucketEntry::kSmallRankOffset;
}

uint32_t bucket_rank_offset(const BucketEntry &bucket) {
    return bucket.rank_offset_prefix_high & BucketEntry::kRankOffsetMask;
}

struct DirectIndex {
    static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
    uint64_t table_size = 0;
    std::vector<uint32_t> bucket_indices;
};

struct DirectEntryIndex {
    static constexpr uint32_t kEmptyBitmapOffset = std::numeric_limits<uint32_t>::max();
    uint64_t table_size = 0;
    std::vector<BucketEntry> entries;
};

struct Prefix36Layer {
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = ZMaskFrozen::kEstimatedLargeSuccessStride * 8U;
    uint64_t live_board_count = 0;
    std::vector<uint64_t> bucket_keys;
    std::vector<uint32_t> bitmap_offsets;
    std::vector<uint32_t> success_offsets;
    std::vector<uint32_t> large_rank_offsets;
    std::vector<BucketEntry> bucket_entries;
    std::vector<uint8_t> small_bitmap_bytes;
    std::vector<uint64_t> large_bitmap_words;
    std::vector<uint16_t> large_rank_bases;
    std::vector<uint32_t> success_values;
    DirectIndex direct_index;
    DirectEntryIndex direct_entry_index;
};

uint64_t direct_slot(const DirectIndex &index, uint64_t key) {
    const uint64_t mixed = key * 11400714819323198485ULL;
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(mixed) * index.table_size) >> 64U);
#else
    return mixed % index.table_size;
#endif
}

uint64_t direct_next_slot(const DirectIndex &index, uint64_t slot) {
    ++slot;
    return slot == index.table_size ? 0ULL : slot;
}

uint64_t direct_entry_slot(const DirectEntryIndex &index, uint64_t key) {
    const uint64_t mixed = key * 11400714819323198485ULL;
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(mixed) * index.table_size) >> 64U);
#else
    return mixed % index.table_size;
#endif
}

uint64_t direct_entry_next_slot(const DirectEntryIndex &index, uint64_t slot) {
    ++slot;
    return slot == index.table_size ? 0ULL : slot;
}

bool direct_entry_is_empty(const BucketEntry &entry) {
    return entry.bitmap_offset == DirectEntryIndex::kEmptyBitmapOffset;
}

uint32_t fill_large_rank_bases(Prefix36Layer &layer, uint32_t offset, uint32_t rank_offset, uint32_t words) {
    uint32_t running = 0U;
    for (uint32_t word = 0; word < words; ++word) {
        if (running > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("prefix36 uint16 large rank base overflow");
        }
        layer.large_rank_bases[rank_offset + word] = static_cast<uint16_t>(running);
        running += popcount_u64(layer.large_bitmap_words[offset + word]);
    }
    return running;
}

void build_direct_index(Prefix36Layer &layer, const std::vector<uint32_t> &size_table) {
    layer.bucket_entries.resize(layer.bucket_keys.size());
    for (size_t i = 0; i < layer.bucket_keys.size(); ++i) {
        const uint64_t prefix36 = key_prefix36(layer.bucket_keys[i]);
        const uint32_t valid_count = size_table[sum_index(key_remaining_sum(layer.bucket_keys[i]))];
        const uint32_t rank_offset =
            valid_count <= layer.threshold_bits ? BucketEntry::kSmallRankOffset : layer.large_rank_offsets[i];
        layer.bucket_entries[i] = BucketEntry{
            static_cast<uint32_t>(prefix36),
            layer.bitmap_offsets[i],
            layer.success_offsets[i],
            pack_rank_offset_prefix_high(prefix36, rank_offset)
        };
    }
    layer.direct_index.table_size = choose_direct_table_size(layer.bucket_keys.size());
    layer.direct_index.bucket_indices.assign(
        static_cast<size_t>(layer.direct_index.table_size),
        DirectIndex::kEmpty
    );
    for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(layer.bucket_keys.size()); ++bucket_idx) {
        const uint64_t direct_key = key_prefix36(layer.bucket_keys[bucket_idx]);
        uint64_t slot = direct_slot(layer.direct_index, direct_key);
        while (layer.direct_index.bucket_indices[static_cast<size_t>(slot)] != DirectIndex::kEmpty) {
            slot = direct_next_slot(layer.direct_index, slot);
        }
        layer.direct_index.bucket_indices[static_cast<size_t>(slot)] = bucket_idx;
    }
    layer.direct_entry_index.table_size = layer.direct_index.table_size;
    BucketEntry empty_entry{};
    empty_entry.bitmap_offset = DirectEntryIndex::kEmptyBitmapOffset;
    layer.direct_entry_index.entries.assign(
        static_cast<size_t>(layer.direct_entry_index.table_size),
        empty_entry
    );
    for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(layer.bucket_keys.size()); ++bucket_idx) {
        const uint64_t direct_key = key_prefix36(layer.bucket_keys[bucket_idx]);
        uint64_t slot = direct_entry_slot(layer.direct_entry_index, direct_key);
        while (!direct_entry_is_empty(layer.direct_entry_index.entries[static_cast<size_t>(slot)])) {
            slot = direct_entry_next_slot(layer.direct_entry_index, slot);
        }
        layer.direct_entry_index.entries[static_cast<size_t>(slot)] = layer.bucket_entries[bucket_idx];
    }
}

uint32_t dense_ordinal_small(const std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t word_idx = rank >> 6U;
    const uint32_t bit_idx = rank & 63U;
    const size_t word_offset = static_cast<size_t>(offset) + static_cast<size_t>(word_idx) * sizeof(uint64_t);
    if (word_offset + sizeof(uint64_t) <= bitmap.size()) {
        uint32_t total = 0U;
        for (uint32_t word = 0; word < word_idx; ++word) {
            uint64_t bits = 0ULL;
            std::memcpy(&bits, bitmap.data() + static_cast<size_t>(offset) + static_cast<size_t>(word) * sizeof(bits),
                        sizeof(bits));
            total += popcount_u64(bits);
        }
        uint64_t bits = 0ULL;
        std::memcpy(&bits, bitmap.data() + word_offset, sizeof(bits));
        if (bit_idx != 0U) {
            total += popcount_u64(bits & ((1ULL << bit_idx) - 1ULL));
        }
        return total;
    }
    const uint32_t byte_idx = rank >> 3U;
    uint32_t total = 0U;
    for (uint32_t i = 0; i < byte_idx; ++i) {
        total += popcount_u32(bitmap[offset + i]);
    }
    const uint32_t bit_idx8 = rank & 7U;
    if (bit_idx8 != 0U) {
        total += popcount_u32(
            static_cast<uint32_t>(bitmap[offset + byte_idx] & static_cast<uint8_t>((1U << bit_idx8) - 1U))
        );
    }
    return total;
}

struct PreparedQuery {
    uint64_t key = 0;
    uint32_t rank = 0;
    uint16_t ref = 0;
    uint8_t valid = 0;
};
static_assert(sizeof(PreparedQuery) == 16, "PreparedQuery must stay cache-compact");

struct LateQuery {
    uint64_t prefix36 = 0;
    uint32_t suffix28 = 0;
    uint16_t ref = 0;
    uint8_t valid = 0;
};
static_assert(sizeof(LateQuery) == 16, "LateQuery must stay cache-compact");

PreparedQuery make_prepared_query(uint64_t prefix36, uint32_t rank) {
    return PreparedQuery{
        prefix36,
        rank,
        0U,
        1U
    };
}

bool query_is_valid(const PreparedQuery &query) {
    return query.valid != 0U;
}

uint64_t query_prefix36(const PreparedQuery &query) {
    return query.key;
}

LateQuery make_late_query(uint64_t board, uint16_t ref) {
    return LateQuery{
        board >> kSuffixBits,
        static_cast<uint32_t>(board & kSuffixMask),
        ref,
        1U
    };
}

PreparedQuery prepare_query(
    const Suffix28Lut &lut,
    uint64_t board,
    uint32_t threshold_bits = ZMaskFrozen::kEstimatedLargeSuccessStride * 8U
) {
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    const SuffixMeta *meta = lookup_suffix_meta(lut, suffix28);
    if (meta == nullptr) {
        return PreparedQuery{};
    }
    const uint32_t valid_count = lut.size_table[sum_index(meta->sum)];
    if (meta->rank >= valid_count) {
        return PreparedQuery{};
    }
    const uint64_t prefix36 = board >> kSuffixBits;
    return make_prepared_query(prefix36, meta->rank);
}

uint32_t low24_sum_from_row_lut(const ZMaskFrozen::ZMaskLuts &z_luts, uint32_t low24) {
    return z_luts.row16_sum[static_cast<uint16_t>(low24)]
         + z_luts.row16_sum[static_cast<uint16_t>((low24 >> 16U) << 8U)];
}

uint32_t low24_group_from_row_lut(const ZMaskFrozen::ZMaskLuts &z_luts, uint32_t low24) {
    return low24_sum_from_row_lut(z_luts, low24) >> 1U;
}

PreparedQuery prepare_query_dense(
    const DenseLow24RankLut &lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    uint64_t board,
    uint32_t threshold_bits = ZMaskFrozen::kEstimatedLargeSuccessStride * 8U
) {
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = lut.table_for_high[high];
    if (table_id == DenseLow24RankLut::kInvalidTable) {
        return PreparedQuery{};
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    uint16_t low_rank = ZMaskFrozen::kInvalidRank;
    uint32_t group = 0U;
    if (!lut.packed_meta_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint64_t packed = lut.packed_meta_table[low24];
        low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>((packed >> 16U) & 0xFFFFU);
        group = static_cast<uint32_t>(packed >> 32U) + (tile_value(high) >> 1U);
    } else {
        low_rank = dense_low24_rank_for_high(lut, high, low24);
        const uint32_t sum = low24_sum_from_row_lut(z_luts, low24) + tile_value(high);
        if (sum > kMaxSuffix28Sum) {
            return PreparedQuery{};
        }
        group = sum_index(sum);
    }
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return PreparedQuery{};
    }
    if (group >= lut.size_table.size()) {
        return PreparedQuery{};
    }
    const uint32_t valid_count = lut.size_table[group];
    const uint32_t rank =
        lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
    if (rank >= valid_count) {
        return PreparedQuery{};
    }
    const uint64_t prefix36 = board >> kSuffixBits;
    return make_prepared_query(prefix36, rank);
}

PreparedQuery prepare_query_dense_hot(
    const DenseLow24RankLut &lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    uint64_t board,
    uint32_t threshold_bits = ZMaskFrozen::kEstimatedLargeSuccessStride * 8U
) {
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = lut.table_for_high[high];
    if (!lut.packed_meta_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint64_t packed = lut.packed_meta_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>((packed >> 16U) & 0xFFFFU);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return PreparedQuery{};
        }
        const uint32_t group = static_cast<uint32_t>(packed >> 32U) + tile_half_value(high);
        const uint32_t rank =
            lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return make_prepared_query(board >> kSuffixBits, rank);
    }
    if (!lut.packed_rank_pair_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint32_t packed = lut.packed_rank_pair_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return PreparedQuery{};
        }
        const uint32_t group = low24_group_from_row_lut(z_luts, low24) + tile_half_value(high);
        const uint32_t rank =
            lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return make_prepared_query(board >> kSuffixBits, rank);
    }
    return prepare_query_dense(lut, z_luts, board, threshold_bits);
}

bool suffix28_rank_dense_hot(
    const DenseLow24RankLut &lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    uint32_t suffix28,
    uint32_t &rank
) {
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = lut.table_for_high[high];
    if (!lut.packed_meta_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint64_t packed = lut.packed_meta_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>((packed >> 16U) & 0xFFFFU);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return false;
        }
        const uint32_t group = static_cast<uint32_t>(packed >> 32U) + tile_half_value(high);
        rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return true;
    }
    if (!lut.packed_rank_pair_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint32_t packed = lut.packed_rank_pair_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return false;
        }
        const uint32_t group = low24_group_from_row_lut(z_luts, low24) + tile_half_value(high);
        rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return true;
    }
    if (table_id == DenseLow24RankLut::kInvalidTable) {
        return false;
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    const uint16_t low_rank = dense_low24_rank_for_high(lut, high, low24);
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return false;
    }
    const uint32_t sum = low24_sum_from_row_lut(z_luts, low24) + tile_value(high);
    if (sum > kMaxSuffix28Sum) {
        return false;
    }
    const uint32_t group = sum_index(sum);
    if (group >= lut.size_table.size()) {
        return false;
    }
    rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
    return rank < lut.size_table[group];
}

bool suffix28_rank_group_dense_hot(
    const DenseLow24RankLut &lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    uint32_t suffix28,
    uint32_t &group,
    uint32_t &rank
) {
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = lut.table_for_high[high];
    if (!lut.packed_meta_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint64_t packed = lut.packed_meta_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>((packed >> 16U) & 0xFFFFU);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return false;
        }
        group = static_cast<uint32_t>(packed >> 32U) + tile_half_value(high);
        rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return rank < lut.size_table[group];
    }
    if (!lut.packed_rank_pair_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint32_t packed = lut.packed_rank_pair_table[low24];
        const uint16_t low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return false;
        }
        group = low24_group_from_row_lut(z_luts, low24) + tile_half_value(high);
        rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
        return rank < lut.size_table[group];
    }
    if (table_id == DenseLow24RankLut::kInvalidTable) {
        return false;
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    const uint16_t low_rank = dense_low24_rank_for_high(lut, high, low24);
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return false;
    }
    const uint32_t sum = low24_sum_from_row_lut(z_luts, low24) + tile_value(high);
    if (sum > kMaxSuffix28Sum) {
        return false;
    }
    group = sum_index(sum);
    if (group >= lut.size_table.size()) {
        return false;
    }
    rank = lut.high_base[static_cast<size_t>(high) * lut.size_table.size() + group] + low_rank;
    return rank < lut.size_table[group];
}

bool success_index_from_bucket(
    const Prefix36Layer &layer,
    const PreparedQuery &query,
    const BucketEntry &bucket,
    uint32_t bucket_index,
    uint32_t &success_index
) {
    if (!bucket_is_small(bucket)) {
        const uint32_t word_idx = query.rank >> 6U;
        const uint32_t bit_idx = query.rank & 63U;
        const uint64_t word = layer.large_bitmap_words[bucket.bitmap_offset + word_idx];
        if ((word & (1ULL << bit_idx)) == 0ULL) {
            return false;
        }
        const uint32_t rank_offset = bucket_rank_offset(bucket);
        uint32_t ordinal = layer.large_rank_bases[rank_offset + word_idx];
        if (bit_idx != 0U) {
            ordinal += popcount_u64(word & ((1ULL << bit_idx) - 1ULL));
        }
        success_index = bucket.success_offset + ordinal;
        return true;
    }
    const uint32_t byte_idx = query.rank >> 3U;
    const uint32_t bit_idx = query.rank & 7U;
    const uint8_t current_byte = layer.small_bitmap_bytes[bucket.bitmap_offset + byte_idx];
    if ((current_byte & static_cast<uint8_t>(1U << bit_idx)) == 0U) {
        return false;
    }
    const uint32_t ordinal =
        dense_ordinal_small(layer.small_bitmap_bytes, bucket.bitmap_offset, query.rank);
    success_index = bucket.success_offset + ordinal;
    return true;
}

void lookup_prepared_batch(
    const Prefix36Layer &layer,
    const PreparedQuery *queries,
    uint32_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
) {
    for (uint32_t i = 0; i < count; ++i) {
        found_flags[i] = 0U;
    }
    if (count == 0U || layer.live_board_count == 0U ||
        layer.direct_index.table_size == 0U || layer.direct_index.bucket_indices.empty()) {
        return;
    }
    uint64_t slots[kBatchSize];
    uint32_t bucket_indices[kBatchSize];
    uint8_t hit[kBatchSize];
    uint16_t retry_storage_a[kBatchSize];
    uint16_t retry_storage_b[kBatchSize];
    uint16_t *retry_indices = retry_storage_a;
    uint16_t *next_retry_indices = retry_storage_b;
    for (uint32_t i = 0; i < count; ++i) {
        bucket_indices[i] = DirectIndex::kEmpty;
        hit[i] = 0U;
        if (query_is_valid(queries[i])) {
            slots[i] = direct_slot(layer.direct_index, query_prefix36(queries[i]));
            __builtin_prefetch(&layer.direct_index.bucket_indices[static_cast<size_t>(slots[i])], 0, 1);
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (query_is_valid(queries[i])) {
            const uint32_t bucket_index = layer.direct_index.bucket_indices[static_cast<size_t>(slots[i])];
            bucket_indices[i] = bucket_index;
            if (bucket_index != DirectIndex::kEmpty) {
                __builtin_prefetch(&layer.bucket_entries[static_cast<size_t>(bucket_index)], 0, 1);
            }
        }
    }
    uint32_t retry_count = 0U;
    for (uint32_t i = 0; i < count; ++i) {
        if (!query_is_valid(queries[i]) || bucket_indices[i] == DirectIndex::kEmpty) {
            continue;
        }
        const BucketEntry &bucket = layer.bucket_entries[static_cast<size_t>(bucket_indices[i])];
        if (bucket_prefix_matches(bucket, query_prefix36(queries[i]))) {
            hit[i] = 1U;
        } else {
            slots[i] = direct_next_slot(layer.direct_index, slots[i]);
            retry_indices[retry_count++] = static_cast<uint16_t>(i);
        }
    }
    while (retry_count != 0U) {
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            __builtin_prefetch(&layer.direct_index.bucket_indices[static_cast<size_t>(slots[i])], 0, 1);
        }
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            const uint32_t bucket_index = layer.direct_index.bucket_indices[static_cast<size_t>(slots[i])];
            bucket_indices[i] = bucket_index;
            if (bucket_index != DirectIndex::kEmpty) {
                __builtin_prefetch(&layer.bucket_entries[static_cast<size_t>(bucket_index)], 0, 1);
            }
        }
        uint32_t next_retry_count = 0U;
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            if (bucket_indices[i] == DirectIndex::kEmpty) {
                continue;
            }
            const BucketEntry &bucket = layer.bucket_entries[static_cast<size_t>(bucket_indices[i])];
            if (bucket_prefix_matches(bucket, query_prefix36(queries[i]))) {
                hit[i] = 1U;
            } else {
                slots[i] = direct_next_slot(layer.direct_index, slots[i]);
                next_retry_indices[next_retry_count++] = static_cast<uint16_t>(i);
            }
        }
        std::swap(retry_indices, next_retry_indices);
        retry_count = next_retry_count;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (hit[i] == 0U) {
            continue;
        }
        const BucketEntry &bucket = layer.bucket_entries[static_cast<size_t>(bucket_indices[i])];
        if (bucket_is_small(bucket)) {
            __builtin_prefetch(
                &layer.small_bitmap_bytes[bucket.bitmap_offset + (queries[i].rank >> 3U)],
                0,
                1
            );
        } else {
            const uint32_t word_idx = queries[i].rank >> 6U;
            __builtin_prefetch(
                &layer.large_bitmap_words[bucket.bitmap_offset + word_idx],
                0,
                1
            );
            __builtin_prefetch(
                &layer.large_rank_bases[bucket_rank_offset(bucket) + word_idx],
                0,
                1
            );
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (hit[i] == 0U) {
            continue;
        }
        found_flags[i] = success_index_from_bucket(
            layer,
            queries[i],
            layer.bucket_entries[static_cast<size_t>(bucket_indices[i])],
            bucket_indices[i],
            success_indices[i]
        ) ? 1U : 0U;
    }
}

void lookup_prepared_batch_direct_entry(
    const Prefix36Layer &layer,
    const PreparedQuery *queries,
    uint32_t *success_indices,
    uint8_t *found_flags,
    uint32_t count
) {
    for (uint32_t i = 0; i < count; ++i) {
        found_flags[i] = 0U;
    }
    if (count == 0U || layer.live_board_count == 0U ||
        layer.direct_entry_index.table_size == 0U || layer.direct_entry_index.entries.empty()) {
        return;
    }
    uint64_t slots[kBatchSize];
    BucketEntry buckets[kBatchSize];
    uint8_t hit[kBatchSize];
    uint16_t retry_storage_a[kBatchSize];
    uint16_t retry_storage_b[kBatchSize];
    uint16_t *retry_indices = retry_storage_a;
    uint16_t *next_retry_indices = retry_storage_b;
    for (uint32_t i = 0; i < count; ++i) {
        buckets[i].bitmap_offset = DirectEntryIndex::kEmptyBitmapOffset;
        hit[i] = 0U;
        if (query_is_valid(queries[i])) {
            slots[i] = direct_entry_slot(layer.direct_entry_index, query_prefix36(queries[i]));
            __builtin_prefetch(&layer.direct_entry_index.entries[static_cast<size_t>(slots[i])], 0, 1);
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (!query_is_valid(queries[i])) {
            continue;
        }
        buckets[i] = layer.direct_entry_index.entries[static_cast<size_t>(slots[i])];
    }
    uint32_t retry_count = 0U;
    for (uint32_t i = 0; i < count; ++i) {
        if (!query_is_valid(queries[i]) || direct_entry_is_empty(buckets[i])) {
            continue;
        }
        if (bucket_prefix_matches(buckets[i], query_prefix36(queries[i]))) {
            hit[i] = 1U;
        } else {
            slots[i] = direct_entry_next_slot(layer.direct_entry_index, slots[i]);
            retry_indices[retry_count++] = static_cast<uint16_t>(i);
        }
    }
    while (retry_count != 0U) {
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            __builtin_prefetch(&layer.direct_entry_index.entries[static_cast<size_t>(slots[i])], 0, 1);
        }
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            buckets[i] = layer.direct_entry_index.entries[static_cast<size_t>(slots[i])];
        }
        uint32_t next_retry_count = 0U;
        for (uint32_t r = 0; r < retry_count; ++r) {
            const uint32_t i = retry_indices[r];
            if (direct_entry_is_empty(buckets[i])) {
                continue;
            }
            if (bucket_prefix_matches(buckets[i], query_prefix36(queries[i]))) {
                hit[i] = 1U;
            } else {
                slots[i] = direct_entry_next_slot(layer.direct_entry_index, slots[i]);
                next_retry_indices[next_retry_count++] = static_cast<uint16_t>(i);
            }
        }
        std::swap(retry_indices, next_retry_indices);
        retry_count = next_retry_count;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (hit[i] == 0U) {
            continue;
        }
        const BucketEntry &bucket = buckets[i];
        if (bucket_is_small(bucket)) {
            __builtin_prefetch(
                &layer.small_bitmap_bytes[bucket.bitmap_offset + (queries[i].rank >> 3U)],
                0,
                1
            );
        } else {
            const uint32_t word_idx = queries[i].rank >> 6U;
            __builtin_prefetch(&layer.large_bitmap_words[bucket.bitmap_offset + word_idx], 0, 1);
            __builtin_prefetch(&layer.large_rank_bases[bucket_rank_offset(bucket) + word_idx], 0, 1);
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (hit[i] == 0U) {
            continue;
        }
        found_flags[i] = success_index_from_bucket(
            layer,
            queries[i],
            buckets[i],
            0U,
            success_indices[i]
        ) ? 1U : 0U;
    }
}

struct Prefix36Range {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint64_t prefix36 = 0;
};

uint16_t dense_low24_rank_for_high(const DenseLow24RankLut &lut, uint32_t high, uint32_t low24) {
    const uint8_t table_id = lut.table_for_high[high];
    if (table_id == DenseLow24RankLut::kInvalidTable) {
        return ZMaskFrozen::kInvalidRank;
    }
    return dense_low24_rank_for_table(lut, table_id, low24);
}

std::vector<Prefix36Range> build_prefix36_ranges(const Prefix40Baseline::Layer &source) {
    std::vector<Prefix36Range> ranges;
    ranges.reserve(source.bucket_keys.size() / 8U + 1U);
    uint32_t begin = 0U;
    while (begin < source.bucket_keys.size()) {
        const uint64_t prefix36 =
            Prefix40Baseline::bucket_key_prefix40(source.bucket_keys[begin]) >> 4U;
        uint32_t end = begin + 1U;
        while (end < source.bucket_keys.size() &&
               (Prefix40Baseline::bucket_key_prefix40(source.bucket_keys[end]) >> 4U) == prefix36) {
            ++end;
        }
        ranges.push_back(Prefix36Range{begin, end, prefix36});
        begin = end;
    }
    return ranges;
}

struct Prefix36SingleBucketPlan {
    uint64_t key = 0;
    uint32_t valid_count = 0;
    uint32_t out_bucket = 0;
    uint64_t small_base = 0;
    uint64_t large_base = 0;
    uint64_t large_rank_base = 0;
    uint64_t small_bytes = 0;
    uint64_t large_words = 0;
    uint64_t large_rank_words = 0;
};

Prefix36SingleBucketPlan plan_prefix36_single_bucket_range(
    const Prefix40Baseline::Layer &source,
    const DenseLow24RankLut &dense_lut,
    const Prefix36Range &range,
    uint32_t threshold_bits
) {
    const uint64_t first_key = source.bucket_keys[range.begin];
    const uint64_t prefix40 = Prefix40Baseline::bucket_key_prefix40(first_key);
    const uint32_t high = static_cast<uint32_t>(prefix40 & 0xFULL);
    const uint32_t remaining_sum = Prefix40Baseline::bucket_key_remaining_sum(first_key);
    const uint32_t total_sum = remaining_sum + tile_value(high);
    const uint32_t valid_count = dense_lut.size_table[sum_index(total_sum)];
    Prefix36SingleBucketPlan plan;
    plan.key = pack_key(range.prefix36, total_sum);
    plan.valid_count = valid_count;
    if (valid_count <= threshold_bits) {
        plan.small_bytes = bytes_for_bits(valid_count);
    } else {
        plan.large_words = words_for_bits(valid_count);
        plan.large_rank_words = large_rank_bases_for_words(static_cast<uint32_t>(plan.large_words));
    }
    return plan;
}

void or_shifted_word_to_large(std::vector<uint64_t> &target, uint32_t target_offset, uint32_t bit_base, uint64_t word) {
    if (word == 0ULL) {
        return;
    }
    const uint32_t word_base = bit_base >> 6U;
    const uint32_t shift = bit_base & 63U;
    target[target_offset + word_base] |= word << shift;
    if (shift != 0U) {
        target[target_offset + word_base + 1U] |= word >> (64U - shift);
    }
}

void or_shifted_small_bytes_to_large(
    std::vector<uint64_t> &target,
    uint32_t target_offset,
    uint32_t rank_base,
    const std::vector<uint8_t> &source,
    uint32_t source_offset,
    uint32_t valid_count
) {
    const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
    for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
        uint64_t value = source[source_offset + byte_idx];
        or_shifted_word_to_large(target, target_offset, rank_base + byte_idx * 8U, value);
    }
}

void or_shifted_large_words_to_large(
    std::vector<uint64_t> &target,
    uint32_t target_offset,
    uint32_t rank_base,
    const std::vector<uint64_t> &source,
    uint32_t source_offset,
    uint32_t valid_count
) {
    const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
    for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
        uint64_t value = source[source_offset + word_idx];
        or_shifted_word_to_large(target, target_offset, rank_base + word_idx * 64U, value);
    }
}

void write_prefix36_single_bucket_range_rank_identity(
    Prefix36Layer &out,
    const Prefix40Baseline::Layer &source,
    const Prefix40Baseline::Luts &prefix_luts,
    const DenseLow24RankLut &dense_lut,
    const Prefix36Range &range,
    const Prefix36SingleBucketPlan &plan
) {
    const uint32_t out_bucket = plan.out_bucket;
    out.bucket_keys[out_bucket] = plan.key;
    out.success_offsets[out_bucket] = 0U;

    if (plan.valid_count <= out.threshold_bits) {
        const uint32_t target_offset = static_cast<uint32_t>(plan.small_base);
        out.bitmap_offsets[out_bucket] = target_offset;
        out.large_rank_offsets[out_bucket] = 0U;
        for (uint32_t bucket_idx = range.begin; bucket_idx < range.end; ++bucket_idx) {
            const uint64_t key = source.bucket_keys[bucket_idx];
            const uint64_t prefix40 = Prefix40Baseline::bucket_key_prefix40(key);
            const uint32_t high = static_cast<uint32_t>(prefix40 & 0xFULL);
            const uint32_t remaining_sum = Prefix40Baseline::bucket_key_remaining_sum(key);
            const uint32_t group = Prefix40Baseline::sum_index(remaining_sum);
            const uint32_t valid_count = prefix_luts.size_table[group];
            const uint32_t total_group = sum_index(remaining_sum + tile_value(high));
            const uint32_t rank_base =
                dense_lut.high_base[static_cast<size_t>(high) * dense_lut.size_table.size() + total_group];
            const uint32_t source_offset = source.bitmap_offsets[bucket_idx];
            if (valid_count <= source.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = source.small_bitmap_bytes[source_offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        set_small_bit(out.small_bitmap_bytes, target_offset, rank_base + byte_idx * 8U + bit);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = source.large_bitmap_words[source_offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        set_small_bit(out.small_bitmap_bytes, target_offset, rank_base + word_idx * 64U + bit);
                        value &= value - 1ULL;
                    }
                }
            }
        }
        return;
    }

    const uint32_t target_offset = static_cast<uint32_t>(plan.large_base);
    const uint32_t target_rank_offset = static_cast<uint32_t>(plan.large_rank_base);
    out.bitmap_offsets[out_bucket] = target_offset;
    out.large_rank_offsets[out_bucket] = target_rank_offset;
    const uint32_t target_words = static_cast<uint32_t>(words_for_bits(plan.valid_count));
    for (uint32_t bucket_idx = range.begin; bucket_idx < range.end; ++bucket_idx) {
        const uint64_t key = source.bucket_keys[bucket_idx];
        const uint64_t prefix40 = Prefix40Baseline::bucket_key_prefix40(key);
        const uint32_t high = static_cast<uint32_t>(prefix40 & 0xFULL);
        const uint32_t remaining_sum = Prefix40Baseline::bucket_key_remaining_sum(key);
        const uint32_t group = Prefix40Baseline::sum_index(remaining_sum);
        const uint32_t valid_count = prefix_luts.size_table[group];
        const uint32_t total_group = sum_index(remaining_sum + tile_value(high));
        const uint32_t rank_base =
            dense_lut.high_base[static_cast<size_t>(high) * dense_lut.size_table.size() + total_group];
        const uint32_t source_offset = source.bitmap_offsets[bucket_idx];
        if (valid_count <= source.threshold_bits) {
            or_shifted_small_bytes_to_large(
                out.large_bitmap_words,
                target_offset,
                rank_base,
                source.small_bitmap_bytes,
                source_offset,
                valid_count
            );
        } else {
            or_shifted_large_words_to_large(
                out.large_bitmap_words,
                target_offset,
                rank_base,
                source.large_bitmap_words,
                source_offset,
                valid_count
            );
        }
    }
    fill_large_rank_bases(out, target_offset, target_rank_offset, target_words);
}

Prefix36Layer build_prefix36_metadata_from_prefix40_single_bucket_parallel(
    const Prefix40Baseline::Layer &source,
    const Prefix40Baseline::Luts &prefix_luts,
    const DenseLow24RankLut &dense_lut,
    int num_threads,
    uint32_t threshold_bits
) {
    Prefix36Layer out;
    out.layer_sum = source.layer_sum;
    out.threshold_bits = threshold_bits;
    out.live_board_count = source.live_board_count;

    const std::vector<Prefix36Range> ranges = build_prefix36_ranges(source);
    std::vector<Prefix36SingleBucketPlan> plans(ranges.size());

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(ranges.size()); ++i) {
        plans[static_cast<size_t>(i)] = plan_prefix36_single_bucket_range(
            source,
            dense_lut,
            ranges[static_cast<size_t>(i)],
            threshold_bits
        );
    }

    uint64_t small_bytes = 0U;
    uint64_t large_words = 0U;
    uint64_t large_rank_words = 0U;
    for (uint32_t i = 0; i < static_cast<uint32_t>(plans.size()); ++i) {
        Prefix36SingleBucketPlan &plan = plans[i];
        plan.out_bucket = i;
        plan.small_base = small_bytes;
        plan.large_base = large_words;
        plan.large_rank_base = large_rank_words;
        small_bytes += plan.small_bytes;
        large_words += plan.large_words;
        large_rank_words += plan.large_rank_words;
    }
    if (small_bytes > std::numeric_limits<uint32_t>::max() ||
        large_words > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("single-bucket prefix36 builder bitmap offsets exceed uint32");
    }

    out.bucket_keys.resize(ranges.size());
    out.bitmap_offsets.resize(ranges.size());
    out.success_offsets.resize(ranges.size());
    out.large_rank_offsets.resize(ranges.size());
    out.small_bitmap_bytes.assign(static_cast<size_t>(small_bytes), 0U);
    out.large_bitmap_words.assign(static_cast<size_t>(large_words), 0ULL);
    out.large_rank_bases.assign(static_cast<size_t>(large_rank_words), 0U);

#pragma omp parallel for schedule(dynamic, 256) num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(ranges.size()); ++i) {
        write_prefix36_single_bucket_range_rank_identity(
            out,
            source,
            prefix_luts,
            dense_lut,
            ranges[static_cast<size_t>(i)],
            plans[static_cast<size_t>(i)]
        );
    }
    return out;
}

struct Prefix36DynamicState {
    static constexpr uint64_t kEmptyKey = std::numeric_limits<uint64_t>::max();
    static constexpr uint32_t kPendingOffset = std::numeric_limits<uint32_t>::max();
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 64U;
    uint32_t hash_capacity = 0;
    uint64_t reserved_small_bytes = 0;
    uint64_t reserved_large_words = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> key_array;
    std::unique_ptr<std::atomic<uint32_t>[]> offset_array;
    std::unique_ptr<std::atomic<uint8_t>[]> small_arena;
    std::unique_ptr<std::atomic<uint64_t>[]> large_arena;
    std::atomic<uint32_t> small_cursor_bytes{0};
    std::atomic<uint32_t> large_cursor_words{0};
    std::atomic<bool> overflowed{false};

    Prefix36DynamicState() = default;
    Prefix36DynamicState(const Prefix36DynamicState &) = delete;
    Prefix36DynamicState &operator=(const Prefix36DynamicState &) = delete;
    Prefix36DynamicState(Prefix36DynamicState &&other) noexcept
        : layer_sum(other.layer_sum),
          threshold_bits(other.threshold_bits),
          hash_capacity(other.hash_capacity),
          reserved_small_bytes(other.reserved_small_bytes),
          reserved_large_words(other.reserved_large_words),
          key_array(std::move(other.key_array)),
          offset_array(std::move(other.offset_array)),
          small_arena(std::move(other.small_arena)),
          large_arena(std::move(other.large_arena)),
          small_cursor_bytes(other.small_cursor_bytes.load(std::memory_order_relaxed)),
          large_cursor_words(other.large_cursor_words.load(std::memory_order_relaxed)),
          overflowed(other.overflowed.load(std::memory_order_relaxed)) {}
    Prefix36DynamicState &operator=(Prefix36DynamicState &&other) noexcept {
        if (this != &other) {
            layer_sum = other.layer_sum;
            threshold_bits = other.threshold_bits;
            hash_capacity = other.hash_capacity;
            reserved_small_bytes = other.reserved_small_bytes;
            reserved_large_words = other.reserved_large_words;
            key_array = std::move(other.key_array);
            offset_array = std::move(other.offset_array);
            small_arena = std::move(other.small_arena);
            large_arena = std::move(other.large_arena);
            small_cursor_bytes.store(other.small_cursor_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            large_cursor_words.store(other.large_cursor_words.load(std::memory_order_relaxed), std::memory_order_relaxed);
            overflowed.store(other.overflowed.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

struct Prefix36DynamicThreadChunks {
    uint32_t small_next = 0;
    uint32_t small_end = 0;
    uint32_t large_next = 0;
    uint32_t large_end = 0;
};

struct Prefix36DynamicPending {
    uint64_t key = 0;
    uint32_t valid_count = 0;
    uint32_t rank = 0;
    uint32_t home_slot = 0;
};

struct Prefix36DynamicResolved {
    uint32_t bitmap_offset = 0;
    uint32_t valid_count = 0;
    uint32_t rank = 0;
};

struct Prefix36DynamicFinalizeTiming {
    double count_seconds = 0.0;
    double collect_seconds = 0.0;
    double sort_seconds = 0.0;
    double plan_seconds = 0.0;
    double allocate_seconds = 0.0;
    double copy_seconds = 0.0;
};

constexpr uint32_t kPrefix36DynamicInsertBufferSize = 128U;
constexpr uint32_t kPrefix36DynamicPrefetchDistance = 16U;
constexpr uint32_t kPrefix36DynamicSmallChunkBytes = 4U * 1024U;
constexpr uint32_t kPrefix36DynamicLargeChunkWords = 512U;
constexpr double kPrefix36DynamicHashLoadUpper = 0.40;

uint32_t choose_prefix36_dynamic_capacity(uint64_t bucket_estimate) {
    const uint64_t required = static_cast<uint64_t>(
        static_cast<double>(std::max<uint64_t>(bucket_estimate, 1ULL)) / kPrefix36DynamicHashLoadUpper) + 1ULL;
    if (required > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("prefix36 dynamic builder hash capacity exceeds uint32");
    }
    return static_cast<uint32_t>(next_prime_u64(std::max<uint64_t>(1024ULL, required)));
}

uint32_t prefix36_dynamic_home_slot(uint64_t prefix36, uint32_t capacity) {
    const uint64_t mixed = prefix36 * 11400714819323198485ULL;
#if defined(__SIZEOF_INT128__)
    return static_cast<uint32_t>((static_cast<unsigned __int128>(mixed) * capacity) >> 64U);
#else
    return static_cast<uint32_t>(mixed % capacity);
#endif
}

using KeyValueSortUint64Uint32Fn = void (*)(uint64_t *, uint32_t *, size_t, bool);

KeyValueSortUint64Uint32Fn resolve_keyvalue_sort_uint64_uint32() {
    static KeyValueSortUint64Uint32Fn fn = []() -> KeyValueSortUint64Uint32Fn {
        if (NativeSortPolicy::native_sort_disabled()) {
            return nullptr;
        }
        try {
#if defined(_WIN32)
            std::vector<std::filesystem::path> candidates;
            candidates.emplace_back("native_core/bookgen_native.dll");
            candidates.emplace_back("bookgen_native.dll");
            candidates.emplace_back(std::filesystem::current_path() / "native_core" / "build-formation" / "bookgen_native.dll");
            candidates.emplace_back(std::filesystem::current_path() / "native_core" / "bookgen_native.dll");
            for (const auto &candidate : candidates) {
                if (!std::filesystem::exists(candidate)) {
                    continue;
                }
                NativeSortPolicy::prepare_bookgen_native_load(candidate);
                HMODULE lib = LoadLibraryA(candidate.string().c_str());
                if (!lib) {
                    continue;
                }
                auto proc = reinterpret_cast<KeyValueSortUint64Uint32Fn>(GetProcAddress(lib, "keyvalue_sort_uint64_uint32"));
                if (proc) {
                    return proc;
                }
            }
            return nullptr;
#else
            void *lib = dlopen("bookgen_native.so", RTLD_LAZY);
            if (!lib) {
                lib = dlopen("native_core/bookgen_native.so", RTLD_LAZY);
            }
            return lib ? reinterpret_cast<KeyValueSortUint64Uint32Fn>(dlsym(lib, "keyvalue_sort_uint64_uint32")) : nullptr;
#endif
        } catch (...) {
            return nullptr;
        }
    }();
    return fn;
}

Prefix36DynamicState make_prefix36_dynamic_state(
    uint32_t layer_sum,
    uint32_t threshold_bits,
    uint64_t bucket_estimate,
    uint64_t small_bytes,
    uint64_t large_words
) {
    Prefix36DynamicState state;
    state.layer_sum = layer_sum;
    state.threshold_bits = threshold_bits;
    state.hash_capacity = choose_prefix36_dynamic_capacity(bucket_estimate);
    state.reserved_small_bytes = std::max<uint64_t>(small_bytes, kPrefix36DynamicSmallChunkBytes);
    state.reserved_large_words = std::max<uint64_t>(large_words, kPrefix36DynamicLargeChunkWords);
    state.key_array = std::make_unique<std::atomic<uint64_t>[]>(state.hash_capacity);
    state.offset_array = std::make_unique<std::atomic<uint32_t>[]>(state.hash_capacity);
    for (uint32_t i = 0; i < state.hash_capacity; ++i) {
        state.key_array[i].store(Prefix36DynamicState::kEmptyKey, std::memory_order_relaxed);
        state.offset_array[i].store(Prefix36DynamicState::kPendingOffset, std::memory_order_relaxed);
    }
    state.small_arena = std::make_unique<std::atomic<uint8_t>[]>(
        static_cast<size_t>(state.reserved_small_bytes));
    state.large_arena = std::make_unique<std::atomic<uint64_t>[]>(
        static_cast<size_t>(state.reserved_large_words));
    return state;
}

uint32_t prefix36_dynamic_acquire_small(
    Prefix36DynamicState &state,
    Prefix36DynamicThreadChunks &chunks,
    uint32_t bytes
) {
    if (chunks.small_next + bytes <= chunks.small_end) {
        const uint32_t out = chunks.small_next;
        chunks.small_next += bytes;
        return out;
    }
    const uint32_t chunk_bytes = std::max<uint32_t>(kPrefix36DynamicSmallChunkBytes, bytes);
    const uint32_t begin = state.small_cursor_bytes.fetch_add(chunk_bytes, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(begin) + chunk_bytes > state.reserved_small_bytes) {
        state.overflowed.store(true, std::memory_order_release);
        return Prefix36DynamicState::kPendingOffset;
    }
    chunks.small_next = begin + bytes;
    chunks.small_end = begin + chunk_bytes;
    return begin;
}

uint32_t prefix36_dynamic_acquire_large(
    Prefix36DynamicState &state,
    Prefix36DynamicThreadChunks &chunks,
    uint32_t words
) {
    if (chunks.large_next + words <= chunks.large_end) {
        const uint32_t out = chunks.large_next;
        chunks.large_next += words;
        return out;
    }
    const uint32_t chunk_words = std::max<uint32_t>(kPrefix36DynamicLargeChunkWords, words);
    const uint32_t begin = state.large_cursor_words.fetch_add(chunk_words, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(begin) + chunk_words > state.reserved_large_words) {
        state.overflowed.store(true, std::memory_order_release);
        return Prefix36DynamicState::kPendingOffset;
    }
    chunks.large_next = begin + words;
    chunks.large_end = begin + chunk_words;
    return begin;
}

void prefix36_dynamic_clear_slice(Prefix36DynamicState &state, uint32_t offset, uint32_t valid_count) {
    if (valid_count <= state.threshold_bits) {
        const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
        for (uint32_t i = 0; i < bytes; ++i) {
            state.small_arena[offset + i].store(0U, std::memory_order_relaxed);
        }
    } else {
        const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
        for (uint32_t i = 0; i < words; ++i) {
            state.large_arena[offset + i].store(0ULL, std::memory_order_relaxed);
        }
    }
}

uint32_t prefix36_dynamic_find_or_insert(
    Prefix36DynamicState &state,
    const Prefix36DynamicPending &entry,
    Prefix36DynamicThreadChunks &chunks
) {
    const uint64_t prefix36 = key_prefix36(entry.key);
    uint32_t slot = entry.home_slot;
    for (;;) {
        const uint64_t current = state.key_array[slot].load(std::memory_order_acquire);
        if (current == entry.key) {
            uint32_t offset = state.offset_array[slot].load(std::memory_order_acquire);
            while (offset == Prefix36DynamicState::kPendingOffset) {
                if (state.overflowed.load(std::memory_order_acquire)) {
                    return Prefix36DynamicState::kPendingOffset;
                }
                offset = state.offset_array[slot].load(std::memory_order_acquire);
            }
            return offset;
        }
        if (current == Prefix36DynamicState::kEmptyKey) {
            uint64_t expected = Prefix36DynamicState::kEmptyKey;
            if (state.key_array[slot].compare_exchange_strong(
                    expected,
                    entry.key,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                uint32_t offset = 0U;
                if (entry.valid_count <= state.threshold_bits) {
                    offset = prefix36_dynamic_acquire_small(
                        state,
                        chunks,
                        static_cast<uint32_t>(bytes_for_bits(entry.valid_count))
                    );
                } else {
                    offset = prefix36_dynamic_acquire_large(
                        state,
                        chunks,
                        static_cast<uint32_t>(words_for_bits(entry.valid_count))
                    );
                }
                if (offset != Prefix36DynamicState::kPendingOffset) {
                    prefix36_dynamic_clear_slice(state, offset, entry.valid_count);
                    state.offset_array[slot].store(offset, std::memory_order_release);
                }
                return offset;
            }
            continue;
        }
        if (key_prefix36(current) == prefix36 && current != entry.key) {
            throw std::runtime_error("prefix36 dynamic builder saw inconsistent suffix sum for prefix36");
        }
        ++slot;
        if (slot == state.hash_capacity) {
            slot = 0U;
        }
        __builtin_prefetch(&state.key_array[slot], 0, 1);
    }
}

void prefix36_dynamic_flush(
    const Prefix36DynamicPending *pending,
    uint32_t count,
    Prefix36DynamicState &state,
    Prefix36DynamicThreadChunks &chunks
) {
    if (count == 0U || state.overflowed.load(std::memory_order_acquire)) {
        return;
    }
    std::array<Prefix36DynamicResolved, kPrefix36DynamicInsertBufferSize> resolved{};
    const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefix36DynamicPrefetchDistance);
    for (uint32_t i = 0; i < prefetch_count; ++i) {
        __builtin_prefetch(&state.key_array[pending[i].home_slot], 0, 1);
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (i + kPrefix36DynamicPrefetchDistance < count) {
            __builtin_prefetch(&state.key_array[pending[i + kPrefix36DynamicPrefetchDistance].home_slot], 0, 1);
        }
        resolved[i].bitmap_offset = prefix36_dynamic_find_or_insert(state, pending[i], chunks);
        if (resolved[i].bitmap_offset == Prefix36DynamicState::kPendingOffset) {
            return;
        }
        resolved[i].valid_count = pending[i].valid_count;
        resolved[i].rank = pending[i].rank;
        if (resolved[i].valid_count <= state.threshold_bits) {
            __builtin_prefetch(
                &state.small_arena[resolved[i].bitmap_offset + (resolved[i].rank >> 3U)],
                1,
                1
            );
        } else {
            __builtin_prefetch(
                &state.large_arena[resolved[i].bitmap_offset + (resolved[i].rank >> 6U)],
                1,
                1
            );
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        const Prefix36DynamicResolved &entry = resolved[i];
        if (entry.valid_count <= state.threshold_bits) {
            std::atomic<uint8_t> &target = state.small_arena[entry.bitmap_offset + (entry.rank >> 3U)];
            const uint8_t mask = static_cast<uint8_t>(1U << (entry.rank & 7U));
            if ((target.load(std::memory_order_relaxed) & mask) != 0U) {
                continue;
            }
            target.fetch_or(mask, std::memory_order_relaxed);
        } else {
            std::atomic<uint64_t> &target = state.large_arena[entry.bitmap_offset + (entry.rank >> 6U)];
            const uint64_t mask = 1ULL << (entry.rank & 63U);
            if ((target.load(std::memory_order_relaxed) & mask) != 0ULL) {
                continue;
            }
            target.fetch_or(mask, std::memory_order_relaxed);
        }
    }
}

void prefix36_dynamic_push_board(
    Prefix36DynamicState &state,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    uint64_t canonical,
    Prefix36DynamicPending *pending,
    uint32_t &pending_count,
    Prefix36DynamicThreadChunks &chunks
) {
    uint32_t group = 0U;
    uint32_t rank = 0U;
    if (!suffix28_rank_group_dense_hot(
            dense_lut,
            z_luts,
            static_cast<uint32_t>(canonical & kSuffixMask),
            group,
            rank)) {
        return;
    }
    const uint32_t valid_count = dense_lut.size_table[group];
    const uint64_t prefix36 = canonical >> kSuffixBits;
    pending[pending_count++] = Prefix36DynamicPending{
        pack_key(prefix36, group << 1U),
        valid_count,
        rank,
        prefix36_dynamic_home_slot(prefix36, state.hash_capacity)
    };
    if (pending_count == kPrefix36DynamicInsertBufferSize) {
        prefix36_dynamic_flush(pending, pending_count, state, chunks);
        pending_count = 0U;
    }
}

void prefix36_dynamic_generate_into(
    const Prefix36Layer &current,
    Prefix36DynamicState &arr1_state,
    Prefix36DynamicState &arr2_state,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads,
    int symm_mode
) {
#pragma omp parallel num_threads(num_threads)
    {
        std::array<Prefix36DynamicPending, kPrefix36DynamicInsertBufferSize> pending1{};
        std::array<Prefix36DynamicPending, kPrefix36DynamicInsertBufferSize> pending2{};
        uint32_t pending_count1 = 0U;
        uint32_t pending_count2 = 0U;
        Prefix36DynamicThreadChunks chunks1{};
        Prefix36DynamicThreadChunks chunks2{};

        auto flush1 = [&]() {
            prefix36_dynamic_flush(pending1.data(), pending_count1, arr1_state, chunks1);
            pending_count1 = 0U;
        };
        auto flush2 = [&]() {
            prefix36_dynamic_flush(pending2.data(), pending_count2, arr2_state, chunks2);
            pending_count2 = 0U;
        };
        std::array<uint64_t, kPrefix36DynamicInsertBufferSize> canonical1{};
        std::array<uint64_t, kPrefix36DynamicInsertBufferSize> canonical2{};
        uint32_t canonical_count1 = 0U;
        uint32_t canonical_count2 = 0U;
        auto flush_canonical1 = [&]() {
            if (canonical_count1 == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(canonical1.data(), canonical_count1, symm_mode);
            for (uint32_t i = 0; i < canonical_count1; ++i) {
                prefix36_dynamic_push_board(
                    arr1_state,
                    dense_lut,
                    z_luts,
                    canonical1[i],
                    pending1.data(),
                    pending_count1,
                    chunks1
                );
            }
            canonical_count1 = 0U;
        };
        auto flush_canonical2 = [&]() {
            if (canonical_count2 == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(canonical2.data(), canonical_count2, symm_mode);
            for (uint32_t i = 0; i < canonical_count2; ++i) {
                prefix36_dynamic_push_board(
                    arr2_state,
                    dense_lut,
                    z_luts,
                    canonical2[i],
                    pending2.data(),
                    pending_count2,
                    chunks2
                );
            }
            canonical_count2 = 0U;
        };
        auto push_canonical1 = [&](uint64_t moved) {
            canonical1[canonical_count1++] = moved;
            if (canonical_count1 == canonical1.size()) {
                flush_canonical1();
            }
        };
        auto push_canonical2 = [&](uint64_t moved) {
            canonical2[canonical_count2++] = moved;
            if (canonical_count2 == canonical2.size()) {
                flush_canonical2();
            }
        };

#pragma omp for schedule(dynamic, 16)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            if (arr1_state.overflowed.load(std::memory_order_acquire) ||
                arr2_state.overflowed.load(std::memory_order_acquire)) {
                continue;
            }
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t total_sum = key_remaining_sum(key);
            const uint32_t group = sum_index(total_sum);
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            auto handle_board = [&](uint64_t board) {
                uint32_t empty_mask = zero_cell_mask16(board);
                while (empty_mask != 0U) {
                    const uint32_t cell = countr_zero_u32(empty_mask);
                    empty_mask &= empty_mask - 1U;
                    const uint64_t spawn2 = board | (1ULL << (4U * cell));
                    const auto moves2 = BoardMover::move_all_dir(spawn2);
                    const uint64_t boards2[4] = {
                        std::get<0>(moves2), std::get<1>(moves2),
                        std::get<2>(moves2), std::get<3>(moves2)
                    };
                    for (uint64_t moved : boards2) {
                        if (moved != spawn2) {
                            push_canonical1(moved);
                        }
                    }
                    const uint64_t spawn4 = board | (2ULL << (4U * cell));
                    const auto moves4 = BoardMover::move_all_dir(spawn4);
                    const uint64_t boards4[4] = {
                        std::get<0>(moves4), std::get<1>(moves4),
                        std::get<2>(moves4), std::get<3>(moves4)
                    };
                    for (uint64_t moved : boards4) {
                        if (moved != spawn4) {
                            push_canonical2(moved);
                        }
                    }
                }
            };
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        handle_board(
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank])
                        );
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        handle_board(
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank])
                        );
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush_canonical1();
        flush_canonical2();
        flush1();
        flush2();
    }
}

struct Prefix36DynamicKeyOffset {
    uint64_t key = 0;
    uint32_t offset = 0;
};

void fallback_sort_key_offsets(std::vector<uint64_t> &keys, std::vector<uint32_t> &offsets) {
    std::vector<Prefix36DynamicKeyOffset> items(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        items[i] = Prefix36DynamicKeyOffset{keys[i], offsets[i]};
    }
    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
        return a.key < b.key;
    });
    for (size_t i = 0; i < items.size(); ++i) {
        keys[i] = items[i].key;
        offsets[i] = items[i].offset;
    }
}

Prefix36Layer finalize_prefix36_dynamic_state(
    const Prefix36DynamicState &state,
    const DenseLow24RankLut &dense_lut,
    int num_threads,
    Prefix36DynamicFinalizeTiming *timing = nullptr
) {
    const double count_t0 = wall_time_seconds();
    std::vector<size_t> thread_counts(static_cast<size_t>(num_threads), 0U);
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const uint64_t begin =
            (static_cast<uint64_t>(state.hash_capacity) * static_cast<uint64_t>(tid)) /
            static_cast<uint64_t>(num_threads);
        const uint64_t end =
            (static_cast<uint64_t>(state.hash_capacity) * static_cast<uint64_t>(tid + 1)) /
            static_cast<uint64_t>(num_threads);
        size_t local = 0U;
        for (uint64_t slot_i = begin; slot_i < end; ++slot_i) {
            const uint64_t key = state.key_array[static_cast<uint32_t>(slot_i)].load(std::memory_order_relaxed);
            if (key != Prefix36DynamicState::kEmptyKey) {
                ++local;
            }
        }
        thread_counts[static_cast<size_t>(tid)] = local;
    }
    const double count_t1 = wall_time_seconds();
    std::vector<size_t> thread_offsets(static_cast<size_t>(num_threads + 1), 0U);
    for (int i = 0; i < num_threads; ++i) {
        thread_offsets[static_cast<size_t>(i + 1)] =
            thread_offsets[static_cast<size_t>(i)] + thread_counts[static_cast<size_t>(i)];
    }
    const double collect_t0 = wall_time_seconds();
    const size_t item_count = thread_offsets.back();
    std::vector<uint64_t> keys(item_count);
    std::vector<uint32_t> arena_offsets(item_count);
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        size_t out = thread_offsets[static_cast<size_t>(tid)];
        const uint64_t begin =
            (static_cast<uint64_t>(state.hash_capacity) * static_cast<uint64_t>(tid)) /
            static_cast<uint64_t>(num_threads);
        const uint64_t end =
            (static_cast<uint64_t>(state.hash_capacity) * static_cast<uint64_t>(tid + 1)) /
            static_cast<uint64_t>(num_threads);
        for (uint64_t slot_i = begin; slot_i < end; ++slot_i) {
            const uint32_t slot = static_cast<uint32_t>(slot_i);
            const uint64_t key = state.key_array[slot].load(std::memory_order_relaxed);
            if (key == Prefix36DynamicState::kEmptyKey) {
                continue;
            }
            const uint32_t offset = state.offset_array[slot].load(std::memory_order_relaxed);
            if (offset == Prefix36DynamicState::kPendingOffset) {
                throw std::runtime_error("prefix36 dynamic finalize saw pending offset");
            }
            keys[out] = key;
            arena_offsets[out] = offset;
            ++out;
        }
    }
    const double collect_t1 = wall_time_seconds();
    const double sort_t0 = wall_time_seconds();
    if (auto keyvalue_sort_fn = resolve_keyvalue_sort_uint64_uint32(); keyvalue_sort_fn != nullptr && keys.size() >= 10000U) {
        keyvalue_sort_fn(keys.data(), arena_offsets.data(), keys.size(), false);
    } else {
        fallback_sort_key_offsets(keys, arena_offsets);
    }
    const double sort_t1 = wall_time_seconds();

    Prefix36Layer out;
    out.layer_sum = state.layer_sum;
    out.threshold_bits = state.threshold_bits;
    out.bucket_keys.resize(item_count);
    out.bitmap_offsets.resize(item_count);
    out.success_offsets.resize(item_count);
    out.large_rank_offsets.resize(item_count);

    uint64_t small_bytes = 0U;
    uint64_t large_words = 0U;
    uint64_t large_rank_words = 0U;
    const double plan_t0 = wall_time_seconds();
    std::vector<uint32_t> live_counts(item_count, 0U);
    if (item_count >= 10000U && num_threads > 1) {
        std::vector<uint64_t> block_small(static_cast<size_t>(num_threads + 1), 0U);
        std::vector<uint64_t> block_large(static_cast<size_t>(num_threads + 1), 0U);
        std::vector<uint64_t> block_rank(static_cast<size_t>(num_threads + 1), 0U);
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const size_t begin =
                (item_count * static_cast<size_t>(tid)) / static_cast<size_t>(num_threads);
            const size_t end =
                (item_count * static_cast<size_t>(tid + 1)) / static_cast<size_t>(num_threads);
            uint64_t local_small = 0U;
            uint64_t local_large = 0U;
            uint64_t local_rank = 0U;
            for (size_t i = begin; i < end; ++i) {
                const uint32_t group = sum_index(key_remaining_sum(keys[i]));
                const uint32_t valid_count = dense_lut.size_table[group];
                if (valid_count <= state.threshold_bits) {
                    local_small += bytes_for_bits(valid_count);
                } else {
                    const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                    local_large += words;
                    local_rank += large_rank_bases_for_words(words);
                }
            }
            block_small[static_cast<size_t>(tid + 1)] = local_small;
            block_large[static_cast<size_t>(tid + 1)] = local_large;
            block_rank[static_cast<size_t>(tid + 1)] = local_rank;
        }
        for (int i = 0; i < num_threads; ++i) {
            block_small[static_cast<size_t>(i + 1)] += block_small[static_cast<size_t>(i)];
            block_large[static_cast<size_t>(i + 1)] += block_large[static_cast<size_t>(i)];
            block_rank[static_cast<size_t>(i + 1)] += block_rank[static_cast<size_t>(i)];
        }
        small_bytes = block_small[static_cast<size_t>(num_threads)];
        large_words = block_large[static_cast<size_t>(num_threads)];
        large_rank_words = block_rank[static_cast<size_t>(num_threads)];
        if (small_bytes > std::numeric_limits<uint32_t>::max() ||
            large_words > std::numeric_limits<uint32_t>::max() ||
            large_rank_words > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("prefix36 dynamic finalize bitmap offsets exceed uint32");
        }
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const size_t begin =
                (item_count * static_cast<size_t>(tid)) / static_cast<size_t>(num_threads);
            const size_t end =
                (item_count * static_cast<size_t>(tid + 1)) / static_cast<size_t>(num_threads);
            uint64_t small_cursor = block_small[static_cast<size_t>(tid)];
            uint64_t large_cursor = block_large[static_cast<size_t>(tid)];
            uint64_t rank_cursor = block_rank[static_cast<size_t>(tid)];
            for (size_t i = begin; i < end; ++i) {
                const uint64_t key = keys[i];
                const uint32_t group = sum_index(key_remaining_sum(key));
                const uint32_t valid_count = dense_lut.size_table[group];
                out.bucket_keys[i] = key;
                if (valid_count <= state.threshold_bits) {
                    const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                    out.bitmap_offsets[i] = static_cast<uint32_t>(small_cursor);
                    out.large_rank_offsets[i] = 0U;
                    small_cursor += bytes;
                } else {
                    const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                    out.bitmap_offsets[i] = static_cast<uint32_t>(large_cursor);
                    out.large_rank_offsets[i] = static_cast<uint32_t>(rank_cursor);
                    large_cursor += words;
                    rank_cursor += large_rank_bases_for_words(words);
                }
            }
        }
    } else {
        for (size_t i = 0; i < item_count; ++i) {
            const uint32_t group = sum_index(key_remaining_sum(keys[i]));
            const uint32_t valid_count = dense_lut.size_table[group];
            if (valid_count <= state.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                out.bitmap_offsets[i] = static_cast<uint32_t>(small_bytes);
                out.large_rank_offsets[i] = 0U;
                small_bytes += bytes;
            } else {
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                out.bitmap_offsets[i] = static_cast<uint32_t>(large_words);
                out.large_rank_offsets[i] = static_cast<uint32_t>(large_rank_words);
                large_words += words;
                large_rank_words += large_rank_bases_for_words(words);
            }
            out.bucket_keys[i] = keys[i];
        }
    }
    const double plan_t1 = wall_time_seconds();
    const double allocate_t0 = wall_time_seconds();
    // std::vector still value-initializes here; true uninitialized allocation would require a container/allocator change.
    out.small_bitmap_bytes.resize(static_cast<size_t>(small_bytes));
    out.large_bitmap_words.resize(static_cast<size_t>(large_words));
    out.large_rank_bases.resize(static_cast<size_t>(large_rank_words));
    const double allocate_t1 = wall_time_seconds();

    const double copy_t0 = wall_time_seconds();
#pragma omp parallel for schedule(dynamic, 256) num_threads(num_threads)
    for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(item_count); ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        const uint32_t group = sum_index(key_remaining_sum(keys[i]));
        const uint32_t valid_count = dense_lut.size_table[group];
        if (valid_count <= state.threshold_bits) {
            const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
            const uint32_t dst = out.bitmap_offsets[i];
            uint32_t live = 0U;
            for (uint32_t j = 0; j < bytes; ++j) {
                uint8_t value =
                    state.small_arena[arena_offsets[i] + j].load(std::memory_order_relaxed);
                if (j + 1U == bytes && (valid_count & 7U) != 0U) {
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>((1U << (valid_count & 7U)) - 1U));
                }
                out.small_bitmap_bytes[dst + j] = value;
                live += popcount_u32(value);
            }
            live_counts[i] = live;
        } else {
            const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
            const uint32_t dst = out.bitmap_offsets[i];
            for (uint32_t j = 0; j < words; ++j) {
                uint64_t value = state.large_arena[arena_offsets[i] + j].load(std::memory_order_relaxed);
                if (j + 1U == words && (valid_count & 63U) != 0U) {
                    value &= ((1ULL << (valid_count & 63U)) - 1ULL);
                }
                out.large_bitmap_words[dst + j] = value;
            }
            live_counts[i] = fill_large_rank_bases(out, dst, out.large_rank_offsets[i], words);
        }
    }
    if (item_count >= 10000U && num_threads > 1) {
        std::vector<uint64_t> block_success(static_cast<size_t>(num_threads + 1), 0U);
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const size_t begin =
                (item_count * static_cast<size_t>(tid)) / static_cast<size_t>(num_threads);
            const size_t end =
                (item_count * static_cast<size_t>(tid + 1)) / static_cast<size_t>(num_threads);
            uint64_t local = 0U;
            for (size_t i = begin; i < end; ++i) {
                local += live_counts[i];
            }
            block_success[static_cast<size_t>(tid + 1)] = local;
        }
        for (int i = 0; i < num_threads; ++i) {
            block_success[static_cast<size_t>(i + 1)] += block_success[static_cast<size_t>(i)];
        }
        if (block_success[static_cast<size_t>(num_threads)] > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("prefix36 dynamic finalize success offsets exceed uint32");
        }
#pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            const size_t begin =
                (item_count * static_cast<size_t>(tid)) / static_cast<size_t>(num_threads);
            const size_t end =
                (item_count * static_cast<size_t>(tid + 1)) / static_cast<size_t>(num_threads);
            uint64_t success_cursor = block_success[static_cast<size_t>(tid)];
            for (size_t i = begin; i < end; ++i) {
                out.success_offsets[i] = static_cast<uint32_t>(success_cursor);
                success_cursor += live_counts[i];
            }
        }
        out.live_board_count = block_success[static_cast<size_t>(num_threads)];
    } else {
        uint64_t success_cursor = 0U;
        for (size_t i = 0; i < live_counts.size(); ++i) {
            out.success_offsets[i] = static_cast<uint32_t>(success_cursor);
            success_cursor += live_counts[i];
        }
        out.live_board_count = success_cursor;
    }
    const double copy_t1 = wall_time_seconds();
    if (timing != nullptr) {
        timing->count_seconds = count_t1 - count_t0;
        timing->collect_seconds = collect_t1 - collect_t0;
        timing->sort_seconds = sort_t1 - sort_t0;
        timing->plan_seconds = plan_t1 - plan_t0;
        timing->allocate_seconds = allocate_t1 - allocate_t0;
        timing->copy_seconds = copy_t1 - copy_t0;
    }
    return out;
}

struct RecalcWorkspace {
    std::vector<PreparedQuery> queries1;
    std::vector<PreparedQuery> queries2;
    std::vector<LateQuery> late_queries1;
    std::vector<LateQuery> late_queries2;
    std::vector<uint32_t> success_indices1;
    std::vector<uint32_t> success_indices2;
    std::vector<uint8_t> found_flags1;
    std::vector<uint8_t> found_flags2;
    std::array<uint16_t, kBatchSize> empty_masks{};
    std::array<uint32_t, kBatchSize * 16U> best2{};
    std::array<uint32_t, kBatchSize * 16U> best4{};

    RecalcWorkspace() {
        constexpr size_t kMaxQueriesPerBatch = static_cast<size_t>(kBatchSize) * 16U * 4U;
        queries1.reserve(kMaxQueriesPerBatch);
        queries2.reserve(kMaxQueriesPerBatch);
        late_queries1.reserve(kMaxQueriesPerBatch);
        late_queries2.reserve(kMaxQueriesPerBatch);
    }
};

struct RecalcStats {
    double seconds = 0.0;
    double mbps = 0.0;
    double generate_prepare_seconds = 0.0;
    double lookup1_seconds = 0.0;
    double lookup2_seconds = 0.0;
    double reduce_seconds = 0.0;
    double finalize_seconds = 0.0;
    uint64_t boards = 0;
    uint64_t queries1 = 0;
    uint64_t queries2 = 0;
    uint64_t found1 = 0;
    uint64_t found2 = 0;
    uint64_t checksum = 0;
};

void lookup_query_vector_dense(
    const Prefix36Layer &future,
    std::vector<PreparedQuery> &queries,
    std::vector<uint32_t> &success_indices,
    std::vector<uint8_t> &found_flags
) {
    success_indices.resize(queries.size());
    found_flags.resize(queries.size());
    for (uint32_t base = 0; base < static_cast<uint32_t>(queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(queries.size()) - base);
        lookup_prepared_batch(
            future,
            queries.data() + base,
            success_indices.data() + base,
            found_flags.data() + base,
            count
        );
    }
}

uint64_t lookup_reduce_query_vector_dense(
    const Prefix36Layer &future,
    std::vector<PreparedQuery> &queries,
    std::array<uint32_t, kBatchSize * 16U> &best
) {
    uint64_t found = 0U;
    uint32_t success_indices[kBatchSize];
    uint8_t found_flags[kBatchSize];
    for (uint32_t base = 0; base < static_cast<uint32_t>(queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(queries.size()) - base);
        lookup_prepared_batch(
            future,
            queries.data() + base,
            success_indices,
            found_flags,
            count
        );
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] != 0U) {
                __builtin_prefetch(&future.success_values[static_cast<size_t>(success_indices[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] == 0U) {
                continue;
            }
            ++found;
            uint32_t &slot = best[queries[base + i].ref];
            slot = std::max(slot, future.success_values[static_cast<size_t>(success_indices[i])]);
        }
    }
    return found;
}

uint64_t lookup_reduce_query_vector_dense_direct_entry(
    const Prefix36Layer &future,
    std::vector<PreparedQuery> &queries,
    std::array<uint32_t, kBatchSize * 16U> &best
) {
    uint64_t found = 0U;
    uint32_t success_indices[kBatchSize];
    uint8_t found_flags[kBatchSize];
    for (uint32_t base = 0; base < static_cast<uint32_t>(queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(queries.size()) - base);
        lookup_prepared_batch_direct_entry(
            future,
            queries.data() + base,
            success_indices,
            found_flags,
            count
        );
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] != 0U) {
                __builtin_prefetch(&future.success_values[static_cast<size_t>(success_indices[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] == 0U) {
                continue;
            }
            ++found;
            uint32_t &slot = best[queries[base + i].ref];
            slot = std::max(slot, future.success_values[static_cast<size_t>(success_indices[i])]);
        }
    }
    return found;
}

struct SuccessRef {
    uint32_t success_index = 0;
    uint16_t ref = 0;
};

uint64_t lookup_reduce_query_vector_dense_direct_entry_sorted_success(
    const Prefix36Layer &future,
    std::vector<PreparedQuery> &queries,
    std::array<uint32_t, kBatchSize * 16U> &best
) {
    uint64_t found = 0U;
    uint32_t success_indices[kBatchSize];
    uint8_t found_flags[kBatchSize];
    SuccessRef refs[kBatchSize];
    for (uint32_t base = 0; base < static_cast<uint32_t>(queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(queries.size()) - base);
        lookup_prepared_batch_direct_entry(
            future,
            queries.data() + base,
            success_indices,
            found_flags,
            count
        );
        uint32_t ref_count = 0U;
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] == 0U) {
                continue;
            }
            refs[ref_count++] = SuccessRef{success_indices[i], queries[base + i].ref};
        }
        std::sort(refs, refs + ref_count, [](const SuccessRef &lhs, const SuccessRef &rhs) {
            return lhs.success_index < rhs.success_index;
        });
        for (uint32_t i = 0; i < ref_count; ++i) {
            __builtin_prefetch(&future.success_values[static_cast<size_t>(refs[i].success_index)], 0, 1);
        }
        for (uint32_t i = 0; i < ref_count; ++i) {
            ++found;
            uint32_t &slot = best[refs[i].ref];
            slot = std::max(slot, future.success_values[static_cast<size_t>(refs[i].success_index)]);
        }
    }
    return found;
}

void lookup_reduce_query_vectors_dense_interleaved(
    const Prefix36Layer &future1,
    std::vector<PreparedQuery> &queries1,
    std::array<uint32_t, kBatchSize * 16U> &best1,
    const Prefix36Layer &future2,
    std::vector<PreparedQuery> &queries2,
    std::array<uint32_t, kBatchSize * 16U> &best2,
    uint64_t &found1,
    uint64_t &found2
) {
    found1 = 0U;
    found2 = 0U;
    uint32_t success_indices1[kBatchSize];
    uint32_t success_indices2[kBatchSize];
    uint8_t found_flags1[kBatchSize];
    uint8_t found_flags2[kBatchSize];
    const uint32_t size1 = static_cast<uint32_t>(queries1.size());
    const uint32_t size2 = static_cast<uint32_t>(queries2.size());
    uint32_t base1 = 0U;
    uint32_t base2 = 0U;
    while (base1 < size1 || base2 < size2) {
        uint32_t count1 = 0U;
        uint32_t count2 = 0U;
        if (base1 < size1) {
            count1 = std::min<uint32_t>(kBatchSize, size1 - base1);
            lookup_prepared_batch(
                future1,
                queries1.data() + base1,
                success_indices1,
                found_flags1,
                count1
            );
        }
        if (base2 < size2) {
            count2 = std::min<uint32_t>(kBatchSize, size2 - base2);
            lookup_prepared_batch(
                future2,
                queries2.data() + base2,
                success_indices2,
                found_flags2,
                count2
            );
        }
        for (uint32_t i = 0; i < count1; ++i) {
            if (found_flags1[i] != 0U) {
                __builtin_prefetch(&future1.success_values[static_cast<size_t>(success_indices1[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count2; ++i) {
            if (found_flags2[i] != 0U) {
                __builtin_prefetch(&future2.success_values[static_cast<size_t>(success_indices2[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count1; ++i) {
            if (found_flags1[i] == 0U) {
                continue;
            }
            ++found1;
            uint32_t &slot = best1[queries1[base1 + i].ref];
            slot = std::max(slot, future1.success_values[static_cast<size_t>(success_indices1[i])]);
        }
        for (uint32_t i = 0; i < count2; ++i) {
            if (found_flags2[i] == 0U) {
                continue;
            }
            ++found2;
            uint32_t &slot = best2[queries2[base2 + i].ref];
            slot = std::max(slot, future2.success_values[static_cast<size_t>(success_indices2[i])]);
        }
        base1 += count1;
        base2 += count2;
    }
}

uint64_t lookup_reduce_late_query_vector_dense(
    const Prefix36Layer &future,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    std::vector<LateQuery> &queries,
    std::array<uint32_t, kBatchSize * 16U> &best
) {
    uint64_t found = 0U;
    uint64_t slots[kBatchSize];
    BucketEntry buckets[kBatchSize];
    uint8_t hit[kBatchSize];
    uint32_t ranks[kBatchSize];
    uint32_t success_indices[kBatchSize];
    uint8_t found_flags[kBatchSize];
    uint16_t retry_storage_a[kBatchSize];
    uint16_t retry_storage_b[kBatchSize];

    for (uint32_t base = 0; base < static_cast<uint32_t>(queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(queries.size()) - base);
        for (uint32_t i = 0; i < count; ++i) {
            buckets[i].bitmap_offset = DirectEntryIndex::kEmptyBitmapOffset;
            hit[i] = 0U;
            found_flags[i] = 0U;
            if (queries[base + i].valid != 0U) {
                slots[i] = direct_entry_slot(future.direct_entry_index, queries[base + i].prefix36);
                __builtin_prefetch(&future.direct_entry_index.entries[static_cast<size_t>(slots[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (queries[base + i].valid == 0U) {
                continue;
            }
            buckets[i] = future.direct_entry_index.entries[static_cast<size_t>(slots[i])];
        }
        uint16_t *retry_indices = retry_storage_a;
        uint16_t *next_retry_indices = retry_storage_b;
        uint32_t retry_count = 0U;
        for (uint32_t i = 0; i < count; ++i) {
            if (queries[base + i].valid == 0U || direct_entry_is_empty(buckets[i])) {
                continue;
            }
            if (bucket_prefix_matches(buckets[i], queries[base + i].prefix36)) {
                hit[i] = 1U;
            } else {
                slots[i] = direct_entry_next_slot(future.direct_entry_index, slots[i]);
                retry_indices[retry_count++] = static_cast<uint16_t>(i);
            }
        }
        while (retry_count != 0U) {
            for (uint32_t r = 0; r < retry_count; ++r) {
                const uint32_t i = retry_indices[r];
                __builtin_prefetch(&future.direct_entry_index.entries[static_cast<size_t>(slots[i])], 0, 1);
            }
            for (uint32_t r = 0; r < retry_count; ++r) {
                const uint32_t i = retry_indices[r];
                buckets[i] = future.direct_entry_index.entries[static_cast<size_t>(slots[i])];
            }
            uint32_t next_retry_count = 0U;
            for (uint32_t r = 0; r < retry_count; ++r) {
                const uint32_t i = retry_indices[r];
                if (direct_entry_is_empty(buckets[i])) {
                    continue;
                }
                if (bucket_prefix_matches(buckets[i], queries[base + i].prefix36)) {
                    hit[i] = 1U;
                } else {
                    slots[i] = direct_entry_next_slot(future.direct_entry_index, slots[i]);
                    next_retry_indices[next_retry_count++] = static_cast<uint16_t>(i);
                }
            }
            std::swap(retry_indices, next_retry_indices);
            retry_count = next_retry_count;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (hit[i] == 0U) {
                continue;
            }
            if (!suffix28_rank_dense_hot(dense_lut, z_luts, queries[base + i].suffix28, ranks[i])) {
                hit[i] = 0U;
                continue;
            }
            const BucketEntry &bucket = buckets[i];
            if (bucket_is_small(bucket)) {
                __builtin_prefetch(&future.small_bitmap_bytes[bucket.bitmap_offset + (ranks[i] >> 3U)], 0, 1);
            } else {
                const uint32_t word_idx = ranks[i] >> 6U;
                __builtin_prefetch(&future.large_bitmap_words[bucket.bitmap_offset + word_idx], 0, 1);
                __builtin_prefetch(&future.large_rank_bases[bucket_rank_offset(bucket) + word_idx], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (hit[i] == 0U) {
                continue;
            }
            PreparedQuery query = make_prepared_query(queries[base + i].prefix36, ranks[i]);
            found_flags[i] = success_index_from_bucket(
                future,
                query,
                buckets[i],
                0U,
                success_indices[i]
            ) ? 1U : 0U;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] != 0U) {
                __builtin_prefetch(&future.success_values[static_cast<size_t>(success_indices[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] == 0U) {
                continue;
            }
            ++found;
            uint32_t &slot = best[queries[base + i].ref];
            slot = std::max(slot, future.success_values[static_cast<size_t>(success_indices[i])]);
        }
    }
    return found;
}

void recalculate_batch_prefix36(
    const uint64_t *boards,
    const uint64_t *output_positions,
    uint32_t board_count,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int symm_mode,
    RecalcWorkspace &workspace,
    RecalcStats &stats,
    bool interleave_lookups = false,
    bool dedup_canonical_moves = false,
    bool use_direct_entries = false,
    bool sort_success_reads = false
) {
    workspace.queries1.clear();
    workspace.queries2.clear();

    const double gen_t0 = wall_time_seconds();
    constexpr size_t kMaxCandidatesPerSide = static_cast<size_t>(kBatchSize) * 16U * 4U;
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates1{};
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates2{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs1{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs2{};
    size_t canonical_count1 = 0U;
    size_t canonical_count2 = 0U;
    auto flush_canonical1 = [&]() {
        if (canonical_count1 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates1.data(), canonical_count1, symm_mode);
        for (size_t i = 0; i < canonical_count1; ++i) {
            PreparedQuery query = prepare_query_dense_hot(
                dense_lut,
                z_luts,
                canonical_candidates1[i],
                future1.threshold_bits
            );
            query.ref = candidate_refs1[i];
            workspace.queries1.push_back(query);
        }
        canonical_count1 = 0U;
    };
    auto flush_canonical2 = [&]() {
        if (canonical_count2 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates2.data(), canonical_count2, symm_mode);
        for (size_t i = 0; i < canonical_count2; ++i) {
            PreparedQuery query = prepare_query_dense_hot(
                dense_lut,
                z_luts,
                canonical_candidates2[i],
                future2.threshold_bits
            );
            query.ref = candidate_refs2[i];
            workspace.queries2.push_back(query);
        }
        canonical_count2 = 0U;
    };
    auto push_canonical1 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates1[canonical_count1] = moved;
        candidate_refs1[canonical_count1] = ref;
        ++canonical_count1;
        if (canonical_count1 == canonical_candidates1.size()) {
            flush_canonical1();
        }
    };
    auto push_canonical2 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates2[canonical_count2] = moved;
        candidate_refs2[canonical_count2] = ref;
        ++canonical_count2;
        if (canonical_count2 == canonical_candidates2.size()) {
            flush_canonical2();
        }
    };
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= (empty_mask - 1U);

            const uint64_t spawn2 = board | (1ULL << (4U * cell));
            const auto moves2 = BoardMover::move_all_dir(spawn2);
            const uint64_t boards2[4] = {
                std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)
            };
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            workspace.best2[ref] = 0U;
            workspace.best4[ref] = 0U;
            uint64_t seen_canon2[4];
            uint32_t seen_canon2_count = 0U;
            if (dedup_canonical_moves) {
                uint64_t canonical2[4];
                uint32_t canonical2_count = 0U;
                for (uint64_t moved : boards2) {
                    if (moved != spawn2) {
                        canonical2[canonical2_count++] = moved;
                    }
                }
                CanonicalBatch::canonicalize_inplace(canonical2, canonical2_count, symm_mode);
                for (uint32_t candidate_idx = 0; candidate_idx < canonical2_count; ++candidate_idx) {
                    const uint64_t canonical = canonical2[candidate_idx];
                    bool duplicate = false;
                    for (uint32_t seen_idx = 0; seen_idx < seen_canon2_count; ++seen_idx) {
                        duplicate = duplicate || (seen_canon2[seen_idx] == canonical);
                    }
                    if (duplicate) {
                        continue;
                    }
                    seen_canon2[seen_canon2_count++] = canonical;
                    PreparedQuery query = prepare_query_dense_hot(
                        dense_lut,
                        z_luts,
                        canonical,
                        future1.threshold_bits
                    );
                    query.ref = ref;
                    workspace.queries1.push_back(query);
                }
            } else {
                for (uint64_t moved : boards2) {
                    if (moved != spawn2) {
                        push_canonical1(moved, ref);
                    }
                }
            }

            const uint64_t spawn4 = board | (2ULL << (4U * cell));
            const auto moves4 = BoardMover::move_all_dir(spawn4);
            const uint64_t boards4[4] = {
                std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)
            };
            uint64_t seen_canon4[4];
            uint32_t seen_canon4_count = 0U;
            if (dedup_canonical_moves) {
                uint64_t canonical4[4];
                uint32_t canonical4_count = 0U;
                for (uint64_t moved : boards4) {
                    if (moved != spawn4) {
                        canonical4[canonical4_count++] = moved;
                    }
                }
                CanonicalBatch::canonicalize_inplace(canonical4, canonical4_count, symm_mode);
                for (uint32_t candidate_idx = 0; candidate_idx < canonical4_count; ++candidate_idx) {
                    const uint64_t canonical = canonical4[candidate_idx];
                    bool duplicate = false;
                    for (uint32_t seen_idx = 0; seen_idx < seen_canon4_count; ++seen_idx) {
                        duplicate = duplicate || (seen_canon4[seen_idx] == canonical);
                    }
                    if (duplicate) {
                        continue;
                    }
                    seen_canon4[seen_canon4_count++] = canonical;
                    PreparedQuery query = prepare_query_dense_hot(
                        dense_lut,
                        z_luts,
                        canonical,
                        future2.threshold_bits
                    );
                    query.ref = ref;
                    workspace.queries2.push_back(query);
                }
            } else {
                for (uint64_t moved : boards4) {
                    if (moved != spawn4) {
                        push_canonical2(moved, ref);
                    }
                }
            }
        }
    }
    flush_canonical1();
    flush_canonical2();
    const double gen_t1 = wall_time_seconds();

    double lookup1_t0 = wall_time_seconds();
    uint64_t found1 = 0U;
    uint64_t found2 = 0U;
    double lookup1_t1 = lookup1_t0;
    double lookup2_t0 = lookup1_t0;
    double lookup2_t1 = lookup1_t0;
    if (use_direct_entries) {
        found1 = sort_success_reads
            ? lookup_reduce_query_vector_dense_direct_entry_sorted_success(future1, workspace.queries1, workspace.best2)
            : lookup_reduce_query_vector_dense_direct_entry(future1, workspace.queries1, workspace.best2);
        lookup1_t1 = wall_time_seconds();
        lookup2_t0 = wall_time_seconds();
        found2 = sort_success_reads
            ? lookup_reduce_query_vector_dense_direct_entry_sorted_success(future2, workspace.queries2, workspace.best4)
            : lookup_reduce_query_vector_dense_direct_entry(future2, workspace.queries2, workspace.best4);
        lookup2_t1 = wall_time_seconds();
    } else if (interleave_lookups) {
        lookup_reduce_query_vectors_dense_interleaved(
            future1,
            workspace.queries1,
            workspace.best2,
            future2,
            workspace.queries2,
            workspace.best4,
            found1,
            found2
        );
        lookup1_t1 = wall_time_seconds();
        lookup2_t0 = lookup1_t1;
        lookup2_t1 = lookup1_t1;
    } else {
        found1 = lookup_reduce_query_vector_dense(future1, workspace.queries1, workspace.best2);
        lookup1_t1 = wall_time_seconds();
        lookup2_t0 = wall_time_seconds();
        found2 = lookup_reduce_query_vector_dense(future2, workspace.queries2, workspace.best4);
        lookup2_t1 = wall_time_seconds();
    }

    const double reduce_t0 = wall_time_seconds();
    const double reduce_t1 = reduce_t0;

    const double finalize_t0 = wall_time_seconds();
    uint64_t checksum = 0U;
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        double success_probability = 0.0;
        uint32_t empty_count = 0U;
        uint32_t empty_mask = workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= (empty_mask - 1U);
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + static_cast<size_t>(cell);
            success_probability += static_cast<double>(workspace.best2[best_index]) * 0.9;
            success_probability += static_cast<double>(workspace.best4[best_index]) * 0.1;
            ++empty_count;
        }
        const uint32_t value =
            empty_count > 0U ? static_cast<uint32_t>(success_probability / static_cast<double>(empty_count)) : 0U;
        checksum += static_cast<uint64_t>(value) * (output_positions[board_slot] + 1ULL);
    }
    const double finalize_t1 = wall_time_seconds();

    stats.generate_prepare_seconds += gen_t1 - gen_t0;
    stats.lookup1_seconds += lookup1_t1 - lookup1_t0;
    stats.lookup2_seconds += lookup2_t1 - lookup2_t0;
    stats.reduce_seconds += reduce_t1 - reduce_t0;
    stats.finalize_seconds += finalize_t1 - finalize_t0;
    stats.boards += board_count;
    stats.queries1 += workspace.queries1.size();
    stats.queries2 += workspace.queries2.size();
    stats.found1 += found1;
    stats.found2 += found2;
    stats.checksum += checksum;
}

void recalculate_batch_prefix36_late(
    const uint64_t *boards,
    const uint64_t *output_positions,
    uint32_t board_count,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int symm_mode,
    RecalcWorkspace &workspace,
    RecalcStats &stats
) {
    workspace.late_queries1.clear();
    workspace.late_queries2.clear();

    const double gen_t0 = wall_time_seconds();
    constexpr size_t kMaxCandidatesPerSide = static_cast<size_t>(kBatchSize) * 16U * 4U;
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates1{};
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates2{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs1{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs2{};
    size_t canonical_count1 = 0U;
    size_t canonical_count2 = 0U;
    auto flush_canonical1 = [&]() {
        if (canonical_count1 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates1.data(), canonical_count1, symm_mode);
        for (size_t i = 0; i < canonical_count1; ++i) {
            workspace.late_queries1.push_back(make_late_query(canonical_candidates1[i], candidate_refs1[i]));
        }
        canonical_count1 = 0U;
    };
    auto flush_canonical2 = [&]() {
        if (canonical_count2 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates2.data(), canonical_count2, symm_mode);
        for (size_t i = 0; i < canonical_count2; ++i) {
            workspace.late_queries2.push_back(make_late_query(canonical_candidates2[i], candidate_refs2[i]));
        }
        canonical_count2 = 0U;
    };
    auto push_canonical1 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates1[canonical_count1] = moved;
        candidate_refs1[canonical_count1] = ref;
        ++canonical_count1;
        if (canonical_count1 == canonical_candidates1.size()) {
            flush_canonical1();
        }
    };
    auto push_canonical2 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates2[canonical_count2] = moved;
        candidate_refs2[canonical_count2] = ref;
        ++canonical_count2;
        if (canonical_count2 == canonical_candidates2.size()) {
            flush_canonical2();
        }
    };
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= (empty_mask - 1U);

            const uint64_t spawn2 = board | (1ULL << (4U * cell));
            const auto moves2 = BoardMover::move_all_dir(spawn2);
            const uint64_t boards2[4] = {
                std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)
            };
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            workspace.best2[ref] = 0U;
            workspace.best4[ref] = 0U;
            for (uint64_t moved : boards2) {
                if (moved == spawn2) {
                    continue;
                }
                push_canonical1(moved, ref);
            }

            const uint64_t spawn4 = board | (2ULL << (4U * cell));
            const auto moves4 = BoardMover::move_all_dir(spawn4);
            const uint64_t boards4[4] = {
                std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)
            };
            for (uint64_t moved : boards4) {
                if (moved == spawn4) {
                    continue;
                }
                push_canonical2(moved, ref);
            }
        }
    }
    flush_canonical1();
    flush_canonical2();
    const double gen_t1 = wall_time_seconds();

    const double lookup1_t0 = wall_time_seconds();
    const uint64_t found1 = lookup_reduce_late_query_vector_dense(
        future1,
        dense_lut,
        z_luts,
        workspace.late_queries1,
        workspace.best2
    );
    const double lookup1_t1 = wall_time_seconds();
    const double lookup2_t0 = wall_time_seconds();
    const uint64_t found2 = lookup_reduce_late_query_vector_dense(
        future2,
        dense_lut,
        z_luts,
        workspace.late_queries2,
        workspace.best4
    );
    const double lookup2_t1 = wall_time_seconds();

    const double reduce_t0 = wall_time_seconds();
    const double reduce_t1 = reduce_t0;

    const double finalize_t0 = wall_time_seconds();
    uint64_t checksum = 0U;
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        double success_probability = 0.0;
        uint32_t empty_count = 0U;
        uint32_t empty_mask = workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= (empty_mask - 1U);
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + static_cast<size_t>(cell);
            success_probability += static_cast<double>(workspace.best2[best_index]) * 0.9;
            success_probability += static_cast<double>(workspace.best4[best_index]) * 0.1;
            ++empty_count;
        }
        const uint32_t value =
            empty_count > 0U ? static_cast<uint32_t>(success_probability / static_cast<double>(empty_count)) : 0U;
        checksum += static_cast<uint64_t>(value) * (output_positions[board_slot] + 1ULL);
    }
    const double finalize_t1 = wall_time_seconds();

    stats.generate_prepare_seconds += gen_t1 - gen_t0;
    stats.lookup1_seconds += lookup1_t1 - lookup1_t0;
    stats.lookup2_seconds += lookup2_t1 - lookup2_t0;
    stats.reduce_seconds += reduce_t1 - reduce_t0;
    stats.finalize_seconds += finalize_t1 - finalize_t0;
    stats.boards += board_count;
    stats.queries1 += workspace.late_queries1.size();
    stats.queries2 += workspace.late_queries2.size();
    stats.found1 += found1;
    stats.found2 += found2;
    stats.checksum += checksum;
}

RecalcStats run_recalculate_prefix36(
    const Prefix40Baseline::Layer &current,
    const Prefix40Baseline::Luts &prefix_luts,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads,
    int symm_mode,
    bool dedup_canonical_moves = false,
    bool use_direct_entries = false,
    bool sort_success_reads = false
) {
    const double t0 = wall_time_seconds();
    std::vector<RecalcStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        RecalcStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        std::array<uint64_t, kBatchSize> output_buffer{};
        uint32_t board_buffer_count = 0U;
        RecalcWorkspace workspace;

        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            recalculate_batch_prefix36(
                board_buffer.data(),
                output_buffer.data(),
                board_buffer_count,
                future1,
                future2,
                dense_lut,
                z_luts,
                symm_mode,
                workspace,
                stats,
                false,
                dedup_canonical_moves,
                use_direct_entries,
                sort_success_reads
            );
            board_buffer_count = 0U;
        };

#pragma omp for schedule(dynamic, kRecalcDynamicChunk)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix40 = Prefix40Baseline::bucket_key_prefix40(key);
            const uint32_t remaining_sum = Prefix40Baseline::bucket_key_remaining_sum(key);
            const uint32_t group = Prefix40Baseline::sum_index(remaining_sum);
            const uint32_t valid_count = prefix_luts.size_table[group];
            const uint32_t unrank_offset = prefix_luts.offset_table[group];
            const uint64_t success_base = current.dense_offsets[bucket_idx];
            uint32_t ordinal = 0U;
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix40 << 24U) | static_cast<uint64_t>(prefix_luts.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix40 << 24U) | static_cast<uint64_t>(prefix_luts.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    const double t1 = wall_time_seconds();

    RecalcStats total;
    total.seconds = t1 - t0;
    for (const RecalcStats &stats : per_thread) {
        total.boards += stats.boards;
        total.generate_prepare_seconds += stats.generate_prepare_seconds;
        total.lookup1_seconds += stats.lookup1_seconds;
        total.lookup2_seconds += stats.lookup2_seconds;
        total.reduce_seconds += stats.reduce_seconds;
        total.finalize_seconds += stats.finalize_seconds;
        total.queries1 += stats.queries1;
        total.queries2 += stats.queries2;
        total.found1 += stats.found1;
        total.found2 += stats.found2;
        total.checksum += stats.checksum;
    }
    total.mbps = mbps_for(total.boards, total.seconds);
    return total;
}

RecalcStats run_recalculate_prefix36_current(
    const Prefix36Layer &current,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads,
    int symm_mode,
    bool dedup_canonical_moves = false,
    bool use_direct_entries = false,
    bool sort_success_reads = false
) {
    const double t0 = wall_time_seconds();
    std::vector<RecalcStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        RecalcStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        std::array<uint64_t, kBatchSize> output_buffer{};
        uint32_t board_buffer_count = 0U;
        RecalcWorkspace workspace;

        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            recalculate_batch_prefix36(
                board_buffer.data(),
                output_buffer.data(),
                board_buffer_count,
                future1,
                future2,
                dense_lut,
                z_luts,
                symm_mode,
                workspace,
                stats,
                false,
                dedup_canonical_moves,
                use_direct_entries,
                sort_success_reads
            );
            board_buffer_count = 0U;
        };

#pragma omp for schedule(dynamic, kRecalcDynamicChunk)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t total_sum = key_remaining_sum(key);
            const uint32_t group = sum_index(total_sum);
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            const uint64_t success_base = current.success_offsets[bucket_idx];
            uint32_t ordinal = 0U;
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    const double t1 = wall_time_seconds();

    RecalcStats total;
    total.seconds = t1 - t0;
    for (const RecalcStats &stats : per_thread) {
        total.boards += stats.boards;
        total.generate_prepare_seconds += stats.generate_prepare_seconds;
        total.lookup1_seconds += stats.lookup1_seconds;
        total.lookup2_seconds += stats.lookup2_seconds;
        total.reduce_seconds += stats.reduce_seconds;
        total.finalize_seconds += stats.finalize_seconds;
        total.queries1 += stats.queries1;
        total.queries2 += stats.queries2;
        total.found1 += stats.found1;
        total.found2 += stats.found2;
        total.checksum += stats.checksum;
    }
    total.mbps = mbps_for(total.boards, total.seconds);
    return total;
}

RecalcStats run_recalculate_prefix36_current_interleaved(
    const Prefix36Layer &current,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads,
    int symm_mode
) {
    const double t0 = wall_time_seconds();
    std::vector<RecalcStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        RecalcStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        std::array<uint64_t, kBatchSize> output_buffer{};
        uint32_t board_buffer_count = 0U;
        RecalcWorkspace workspace;

        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            recalculate_batch_prefix36(
                board_buffer.data(),
                output_buffer.data(),
                board_buffer_count,
                future1,
                future2,
                dense_lut,
                z_luts,
                symm_mode,
                workspace,
                stats,
                true
            );
            board_buffer_count = 0U;
        };

#pragma omp for schedule(dynamic, kRecalcDynamicChunk)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t total_sum = key_remaining_sum(key);
            const uint32_t group = sum_index(total_sum);
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            const uint64_t success_base = current.success_offsets[bucket_idx];
            uint32_t ordinal = 0U;
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    const double t1 = wall_time_seconds();

    RecalcStats total;
    total.seconds = t1 - t0;
    for (const RecalcStats &stats : per_thread) {
        total.boards += stats.boards;
        total.generate_prepare_seconds += stats.generate_prepare_seconds;
        total.lookup1_seconds += stats.lookup1_seconds;
        total.lookup2_seconds += stats.lookup2_seconds;
        total.reduce_seconds += stats.reduce_seconds;
        total.finalize_seconds += stats.finalize_seconds;
        total.queries1 += stats.queries1;
        total.queries2 += stats.queries2;
        total.found1 += stats.found1;
        total.found2 += stats.found2;
        total.checksum += stats.checksum;
    }
    total.mbps = mbps_for(total.boards, total.seconds);
    return total;
}

RecalcStats run_recalculate_prefix36_current_late(
    const Prefix36Layer &current,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads,
    int symm_mode
) {
    const double t0 = wall_time_seconds();
    std::vector<RecalcStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        RecalcStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        std::array<uint64_t, kBatchSize> output_buffer{};
        uint32_t board_buffer_count = 0U;
        RecalcWorkspace workspace;

        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            recalculate_batch_prefix36_late(
                board_buffer.data(),
                output_buffer.data(),
                board_buffer_count,
                future1,
                future2,
                dense_lut,
                z_luts,
                symm_mode,
                workspace,
                stats
            );
            board_buffer_count = 0U;
        };

#pragma omp for schedule(dynamic, kRecalcDynamicChunk)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t total_sum = key_remaining_sum(key);
            const uint32_t group = sum_index(total_sum);
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            const uint64_t success_base = current.success_offsets[bucket_idx];
            uint32_t ordinal = 0U;
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        board_buffer[board_buffer_count] =
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                        output_buffer[board_buffer_count] = success_base + ordinal;
                        ++board_buffer_count;
                        ++ordinal;
                        if (board_buffer_count == kBatchSize) {
                            flush();
                        }
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    const double t1 = wall_time_seconds();

    RecalcStats total;
    total.seconds = t1 - t0;
    for (const RecalcStats &stats : per_thread) {
        total.boards += stats.boards;
        total.generate_prepare_seconds += stats.generate_prepare_seconds;
        total.lookup1_seconds += stats.lookup1_seconds;
        total.lookup2_seconds += stats.lookup2_seconds;
        total.reduce_seconds += stats.reduce_seconds;
        total.finalize_seconds += stats.finalize_seconds;
        total.queries1 += stats.queries1;
        total.queries2 += stats.queries2;
        total.found1 += stats.found1;
        total.found2 += stats.found2;
        total.checksum += stats.checksum;
    }
    total.mbps = mbps_for(total.boards, total.seconds);
    return total;
}

} // namespace
