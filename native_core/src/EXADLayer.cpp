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

bool tile_limit_allows(const ZMaskFrozen::TileLimitConfig &config, uint32_t tile, uint8_t new_count) {
    const int8_t limit = config.max_counts[tile & 0xFU];
    return limit < 0 || new_count <= static_cast<uint8_t>(limit);
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

uint32_t semantic_suffix_sum(uint32_t suffix28, const Luts &luts) {
    return low24_sum_from_row_lut(luts, suffix28 & 0xFFFFFFU) + tile_value(suffix28 >> 24U);
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

bool suffix28_rank_group(const Luts &luts, uint32_t suffix28, uint32_t &group, uint32_t &rank) {
    uint32_t semantic_sum = 0U;
    return suffix28_rank_group_sum(luts, suffix28, group, rank, semantic_sum);
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

int8_t ad_bucket_key_for_board(
    uint64_t board,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param
) {
    const uint64_t masked_board = board & ~param.pos_fixed_32k_mask;
    const auto &table = FormationAD::info_table1();
    const uint64_t block0 = masked_board & 0xFFFFULL;
    const uint64_t block1 = (masked_board >> 16U) & 0xFFFFULL;
    const uint64_t block2 = (masked_board >> 32U) & 0xFFFFULL;
    const uint64_t block3 = (masked_board >> 48U) & 0xFFFFULL;
    const FormationAD::InfoEntry &r0 = table[block0];
    const FormationAD::InfoEntry &r1 = table[block1];
    const FormationAD::InfoEntry &r2 = table[block2];
    const FormationAD::InfoEntry &r3 = table[block3];
    const int8_t count_32k = static_cast<int8_t>(r0.count_32k + r1.count_32k + r2.count_32k + r3.count_32k);
    if (count_32k == static_cast<int8_t>(param.num_free_32k)) {
        return count_32k;
    }
    const int remaining_count = static_cast<int>(count_32k) - static_cast<int>(param.num_free_32k);
    if (remaining_count < 0 || remaining_count > 9) {
        return -1;
    }
    const uint32_t total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    const int64_t large_tiles_sum64 =
        static_cast<int64_t>(original_board_sum) -
        static_cast<int64_t>(total_sum) -
        (static_cast<int64_t>(param.num_free_32k + param.num_fixed_32k) << 15);
    if (large_tiles_sum64 < 0 || (large_tiles_sum64 & 63LL) != 0LL) {
        return -1;
    }
    const uint64_t large_tiles_sum = static_cast<uint64_t>(large_tiles_sum64);
    if ((large_tiles_sum >> 6U) > 255U) {
        return -1;
    }
    auto tiles = FormationAD::tiles_combination_view(
        tiles_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(remaining_count)
    );
    if (tiles.empty()) {
        return -1;
    }
    if (remaining_count == 1) {
        return count_32k;
    }
    if (tiles.size > 1 && tiles[0] == tiles[1]) {
        if (tiles.size > 2 && tiles[0] == tiles[2]) {
            return static_cast<int8_t>(count_32k - 3 + 16);
        }
        return static_cast<int8_t>(-count_32k);
    }
    return count_32k;
}

uint64_t live_count_by_bitmap(
    const BoardSet &set,
    uint32_t bucket_index,
    const Luts &luts
) {
    if (bucket_index >= set.buckets.size()) {
        return 0U;
    }
    const uint64_t key = set.buckets[bucket_index].key;
    const uint32_t valid_count = luts.size_table[lut_group_index(bucket_key_semantic_sum(key))];
    const uint32_t offset = set.buckets[bucket_index].bitmap_offset;
    uint64_t live = 0U;
    if (valid_count <= set.threshold_bits) {
        const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
        for (uint32_t i = 0; i < bytes; ++i) {
            live += popcount64(set.small_bitmap_bytes[offset + i]);
        }
    } else {
        const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
        for (uint32_t i = 0; i < words; ++i) {
            live += popcount64(set.large_bitmap_words[offset + i]);
        }
    }
    return live;
}

std::vector<uint64_t> extract_boards_sorted(const Layer &layer, const Luts &luts, int num_threads) {
    std::vector<uint64_t> boards;
    boards.reserve(static_cast<size_t>(layer.live_board_count));
    for_each_live_board(layer, luts, [&boards](int8_t, uint64_t board) {
        boards.push_back(board);
    });
    if (num_threads > 1 && boards.size() > 100000U) {
#pragma omp parallel
        {
#pragma omp single nowait
            std::sort(boards.begin(), boards.end());
        }
    } else {
        std::sort(boards.begin(), boards.end());
    }
    boards.erase(std::unique(boards.begin(), boards.end()), boards.end());
    return boards;
}

} // namespace EXAD
