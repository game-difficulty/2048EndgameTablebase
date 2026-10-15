#include "EXADLayer.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace EXAD {

namespace {

constexpr uint8_t kInvalidTable = 0xFFU;

inline uint32_t popcount64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t count = 0;
    while (value != 0ULL) {
        value &= (value - 1ULL);
        ++count;
    }
    return count;
#endif
}

bool suffix28_structural_constraints_allow(const ZMaskFrozen::TileLimitConfig &config, uint32_t suffix) {
    if ((suffix & config.required_suffix24) != config.required_suffix24) {
        return false;
    }
    if (config.valid_suffix_masks.empty()) {
        return true;
    }
    for (uint32_t mask : config.valid_suffix_masks) {
        if ((suffix & mask) == mask) {
            return true;
        }
    }
    return false;
}

ZMaskFrozen::TileLimitConfig adjusted_low24_config_for_high(
    const ZMaskFrozen::TileLimitConfig &config,
    uint32_t high_tile,
    bool &valid_high
) {
    ZMaskFrozen::TileLimitConfig adjusted = config;
    valid_high = true;
    const int8_t limit = adjusted.max_counts[high_tile & 0xFU];
    if (limit == 0) {
        valid_high = false;
        return adjusted;
    }
    if (limit > 0) {
        adjusted.max_counts[high_tile & 0xFU] = static_cast<int8_t>(limit - 1);
    }
    return adjusted;
}

uint32_t low24_sum_from_row_lut(const Luts &luts, uint32_t low24) {
    return luts.row16_sum[low24 & 0xFFFFU] +
        luts.row16_sum[((low24 >> 16U) & 0xFFU) << 8U];
}

uint32_t low24_group_from_row_lut(const Luts &luts, uint32_t low24) {
    return low24_sum_from_row_lut(luts, low24) >> 1U;
}

uint32_t tile_half_value(uint32_t tile) {
    static constexpr std::array<uint32_t, 16> kHalfValues = {
        0U, 1U, 2U, 4U, 8U, 16U, 32U, 64U,
        128U, 256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U
    };
    return kHalfValues[tile & 0xFU];
}

std::pair<std::vector<uint16_t>, std::vector<uint32_t>> build_low24_rank_table(
    const ZMaskFrozen::TileLimitConfig &config
) {
    std::vector<uint16_t> rank_table(kLow24StateCount, ZMaskFrozen::kInvalidRank);
    std::vector<uint32_t> counts(((6U * (1U << 15U)) >> 1U) + 1U, 0U);
    std::vector<uint32_t> sums(kLow24StateCount, 0U);
    std::vector<uint8_t> valid(kLow24StateCount, 0U);

    for (uint32_t state = 0; state < kLow24StateCount; ++state) {
        uint32_t sum = 0U;
        if (!ZMaskFrozen::decode_suffix24_if_valid(state, sum, config)) {
            continue;
        }
        valid[state] = 1U;
        sums[state] = sum;
        ++counts[sum_index(sum)];
    }

    std::vector<uint32_t> cursor(counts.size(), 0U);
    for (uint32_t state = 0; state < kLow24StateCount; ++state) {
        if (valid[state] == 0U) {
            continue;
        }
        const uint32_t group = sum_index(sums[state]);
        const uint32_t rank = cursor[group]++;
        if (rank > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("EXAD low24 rank exceeds uint16 range");
        }
        rank_table[state] = static_cast<uint16_t>(rank);
    }
    return {std::move(rank_table), std::move(counts)};
}

uint16_t low24_rank_for_table(const Luts &luts, uint8_t table_id, uint32_t low24) {
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

} // namespace

uint32_t tile_value(uint32_t tile) {
    return tile == 0U ? 0U : (1U << (tile & 0xFU));
}

uint32_t sum_index(uint32_t sum) {
    if ((sum & 1U) != 0U) {
        throw std::runtime_error("EXAD encountered odd semantic suffix sum");
    }
    return sum >> 1U;
}

uint64_t config_signature(const ZMaskFrozen::TileLimitConfig &config) {
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

ZMaskFrozen::TileLimitConfig make_exad_lut_tile_limit_config(
    int target_exponent,
    const std::vector<uint64_t> &seed_boards,
    const PatternSpec &spec,
    bool is_free,
    bool is_variant
) {
    ZMaskFrozen::TileLimitConfig config = ZMaskFrozen::make_lut_tile_limit_config(
        target_exponent,
        seed_boards,
        spec,
        is_free,
        is_variant
    );
    if (target_exponent >= 0 && target_exponent < static_cast<int>(config.max_counts.size())) {
        config.max_counts[static_cast<size_t>(target_exponent)] = 0;
    }
    return config;
}

void initialize_runtime_tables(Luts &luts) {
    luts.row16_sum.assign(1U << 16U, 0U);
    for (uint32_t row = 0; row < (1U << 16U); ++row) {
        uint32_t sum = 0U;
        for (uint32_t cell = 0; cell < 4U; ++cell) {
            sum += tile_value((row >> (cell * 4U)) & 0xFU);
        }
        luts.row16_sum[row] = sum;
    }
}

Luts build_luts(const ZMaskFrozen::TileLimitConfig &config, int /*num_threads*/) {
    Luts luts;
    luts.config = config;
    luts.config_signature = config_signature(config);
    luts.table_for_high.fill(kInvalidTable);
    luts.size_table.assign(kSemanticGroupCount, 0U);
    luts.high_base.assign(16ULL * kSemanticGroupCount, 0U);
    initialize_runtime_tables(luts);

    std::vector<uint64_t> signatures;
    std::vector<std::vector<uint32_t>> counts_by_table;
    for (uint32_t high = 0; high < 16U; ++high) {
        bool valid_high = false;
        ZMaskFrozen::TileLimitConfig adjusted = adjusted_low24_config_for_high(config, high, valid_high);
        if (!valid_high) {
            continue;
        }
        const uint64_t signature = config_signature(adjusted);
        auto found = std::find(signatures.begin(), signatures.end(), signature);
        uint8_t table_id = 0U;
        if (found == signatures.end()) {
            auto built = build_low24_rank_table(adjusted);
            if (luts.rank_tables.size() >= kInvalidTable) {
                throw std::runtime_error("EXAD too many low24 rank table variants");
            }
            table_id = static_cast<uint8_t>(luts.rank_tables.size());
            signatures.push_back(signature);
            luts.rank_tables.push_back(std::move(built.first));
            counts_by_table.push_back(std::move(built.second));
        } else {
            table_id = static_cast<uint8_t>(std::distance(signatures.begin(), found));
        }
        luts.table_for_high[high] = table_id;
    }

    for (uint32_t high = 0; high < 16U; ++high) {
        const uint8_t table_id = luts.table_for_high[high];
        if (table_id == kInvalidTable) {
            continue;
        }
        const uint32_t high_value = tile_value(high);
        const std::vector<uint32_t> &counts = counts_by_table[table_id];
        for (uint32_t low_group = 0; low_group < counts.size(); ++low_group) {
            const uint32_t count = counts[low_group];
            if (count == 0U) {
                continue;
            }
            const uint32_t total_sum = (low_group << 1U) + high_value;
            if (total_sum > kMaxSemanticSuffixSum) {
                continue;
            }
            luts.size_table[sum_index(total_sum)] += count;
        }
    }

    for (uint32_t high = 0; high < 16U; ++high) {
        for (uint32_t group = 0; group < kSemanticGroupCount; ++group) {
            const uint32_t total_sum = group << 1U;
            uint32_t base = 0U;
            for (uint32_t prev_high = 0; prev_high < high; ++prev_high) {
                const uint8_t table_id = luts.table_for_high[prev_high];
                if (table_id == kInvalidTable) {
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
                const uint32_t low_group = low_sum >> 1U;
                if (table_id < counts_by_table.size() && low_group < counts_by_table[table_id].size()) {
                    base += counts_by_table[table_id][low_group];
                }
            }
            if (base > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("EXAD suffix28 high_base exceeds uint16 range");
            }
            luts.high_base[static_cast<size_t>(high) * kSemanticGroupCount + group] =
                static_cast<uint16_t>(base);
        }
    }

    luts.offset_table.resize(kSemanticGroupCount, 0U);
    uint64_t unrank_total = 0U;
    for (uint32_t group = 0; group < kSemanticGroupCount; ++group) {
        luts.offset_table[group] = static_cast<uint32_t>(unrank_total);
        unrank_total += luts.size_table[group];
    }
    if (unrank_total > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("EXAD suffix28 unrank table exceeds uint32 range");
    }
    luts.unrank_array.assign(static_cast<size_t>(unrank_total), kInvalidSuffix28);

    std::vector<uint32_t> cursor(kSemanticGroupCount, 0U);
    for (uint32_t high = 0; high < 16U; ++high) {
        const uint8_t table_id = luts.table_for_high[high];
        if (table_id == kInvalidTable) {
            continue;
        }
        for (uint32_t low24 = 0; low24 < kLow24StateCount; ++low24) {
            const uint16_t low_rank = low24_rank_for_table(luts, table_id, low24);
            if (low_rank == ZMaskFrozen::kInvalidRank) {
                continue;
            }
            const uint32_t suffix = low24 | (high << 24U);
            if (!suffix28_structural_constraints_allow(config, suffix)) {
                continue;
            }
            const uint32_t sum = low24_sum_from_row_lut(luts, low24) + tile_value(high);
            if (sum > kMaxSemanticSuffixSum) {
                continue;
            }
            const uint32_t group = sum_index(sum);
            const uint32_t rank =
                luts.high_base[static_cast<size_t>(high) * kSemanticGroupCount + group] + low_rank;
            if (rank >= luts.size_table[group]) {
                throw std::runtime_error("EXAD suffix28 rank exceeds group size");
            }
            luts.unrank_array[static_cast<size_t>(luts.offset_table[group]) + rank] = suffix;
            ++cursor[group];
        }
    }

    luts.valid_suffix_count = unrank_total;
    if (luts.rank_tables.size() == 2U) {
        luts.packed_table0 = 0U;
        luts.packed_table1 = 1U;
        luts.packed_rank_pair_table.resize(kLow24StateCount);
        for (uint32_t state = 0; state < kLow24StateCount; ++state) {
            luts.packed_rank_pair_table[state] =
                static_cast<uint32_t>(luts.rank_tables[0][state]) |
                (static_cast<uint32_t>(luts.rank_tables[1][state]) << 16U);
        }
        std::vector<std::vector<uint16_t>>().swap(luts.rank_tables);
    }
    return luts;
}

uint64_t pack_bucket_key(uint64_t prefix36, uint32_t semantic_suffix_sum) {
    if ((prefix36 >> 36U) != 0ULL) {
        throw std::runtime_error("EXAD prefix36 overflow");
    }
    if ((semantic_suffix_sum >> 18U) != 0U) {
        throw std::runtime_error("EXAD semantic_suffix_sum overflow");
    }
    return (prefix36 << 18U) | static_cast<uint64_t>(semantic_suffix_sum);
}

uint64_t bucket_key_prefix36(uint64_t key) {
    return key >> 18U;
}

uint32_t bucket_key_semantic_sum(uint64_t key) {
    return static_cast<uint32_t>(key & 0x3FFFFULL);
}

uint32_t lut_group_index(uint32_t semantic_suffix_sum) {
    return sum_index(semantic_suffix_sum);
}

bool suffix28_rank_group_sum(
    const Luts &luts,
    uint32_t suffix28,
    uint32_t &group,
    uint32_t &rank,
    uint32_t &semantic_sum
) {
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = luts.table_for_high[high];
    if (table_id == kInvalidTable) {
        return false;
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    const uint16_t low_rank = low24_rank_for_table(luts, table_id, low24);
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return false;
    }
    group = low24_group_from_row_lut(luts, low24) + tile_half_value(high);
    if (group >= luts.size_table.size()) {
        return false;
    }
    semantic_sum = group << 1U;
    rank = luts.high_base[static_cast<size_t>(high) * kSemanticGroupCount + group] + low_rank;
    return rank < luts.size_table[group];
}

} // namespace EXAD
