#include "BoardMaskerAD.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <utility>

namespace FormationAD {

namespace {

constexpr std::array<uint32_t, 8> kLargeTiles = {8192U, 4096U, 2048U, 1024U, 512U, 256U, 128U, 64U};

uint8_t log2_uint(uint32_t value) {
    uint8_t result = 0;
    while (value > 1U) {
        value >>= 1U;
        ++result;
    }
    return result;
}

std::array<InfoEntry, 65536> generate_board_info_table(bool masked_only) {
    std::array<InfoEntry, 65536> lookup{};
    for (uint32_t block = 0; block < 65536U; ++block) {
        InfoEntry entry;
        for (int i = 4; i >= 0; --i) {
            uint32_t shift = static_cast<uint32_t>(i * 4);
            uint64_t tile_value = (static_cast<uint64_t>(block) >> shift) & 0xFULL;
            bool is_large = masked_only ? (tile_value == 0xFULL) : (tile_value > 5ULL);
            if (is_large) {
                entry.pos_bitmap |= static_cast<uint16_t>(0xFU << shift);
                ++entry.count_32k;
                if (tile_value == 0xFULL) {
                    if (entry.merged_count == 0) {
                        ++entry.pos_rank;
                    }
                } else {
                    ++entry.merged_count;
                    entry.merged_xor ^= static_cast<uint8_t>(tile_value);
                }
            } else if (tile_value > 0ULL) {
                entry.total_sum = static_cast<uint16_t>(entry.total_sum + (1U << tile_value));
            }
        }
        lookup[block] = entry;
    }
    return lookup;
}

const std::array<InfoEntry, 65536> &info_table_impl1() {
    static const std::array<InfoEntry, 65536> table = generate_board_info_table(false);
    return table;
}

const std::array<InfoEntry, 65536> &info_table_impl2() {
    static const std::array<InfoEntry, 65536> table = generate_board_info_table(true);
    return table;
}

PermutationSlot make_slot_from_rows(const std::vector<std::vector<uint8_t>> &rows) {
    PermutationSlot slot;
    slot.rows = rows.size();
    slot.cols = rows.empty() ? 0 : rows.front().size();
    slot.data.reserve(slot.rows * slot.cols);
    for (const auto &row : rows) {
        slot.data.insert(slot.data.end(), row.begin(), row.end());
    }
    return slot;
}

PermutationSlot make_slot_from_sorted_rows(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> &rows) {
    PermutationSlot slot;
    slot.rows = rows.size();
    slot.cols = rows.empty() ? 0 : rows.front().second.size();
    slot.data.reserve(slot.rows * slot.cols);
    for (const auto &row : rows) {
        slot.data.insert(slot.data.end(), row.second.begin(), row.second.end());
    }
    return slot;
}

} // namespace

const std::array<InfoEntry, 65536> &info_table1() {
    return info_table_impl1();
}

const std::array<InfoEntry, 65536> &info_table2() {
    return info_table_impl2();
}

uint64_t fixed_32k_mask(const std::vector<uint8_t> &fixed_shifts) {
    uint64_t mask = 0;
    for (uint8_t shift : fixed_shifts) {
        mask += (0xFULL << shift);
    }
    return mask;
}

AdvancedMaskParam build_mask_param(const AdvancedPatternSpec &spec) {
    AdvancedMaskParam param;
    param.small_tile_sum_limit = spec.small_tile_sum_limit;
    param.target = spec.target;
    param.pos_fixed_32k_mask = fixed_32k_mask(spec.fixed_32k_shifts);
    param.num_free_32k = spec.num_free_32k;
    param.num_fixed_32k = static_cast<uint8_t>(spec.fixed_32k_shifts.size());
    return param;
}

uint64_t mask_board(uint64_t board, int threshold) {
    uint64_t masked_board = board;
    for (int k = 0; k < 16; ++k) {
        uint64_t encoded_num = (board >> static_cast<uint64_t>(4 * k)) & 0xFULL;
        if (encoded_num >= static_cast<uint64_t>(threshold)) {
            masked_board |= (0xFULL << static_cast<uint64_t>(4 * k));
        }
    }
    return masked_board;
}

TileCountResult tile_sum_and_32k_count(uint64_t masked_board, const AdvancedMaskParam &param) {
    masked_board &= ~param.pos_fixed_32k_mask;
    const auto &table = info_table2();
    uint64_t block3 = (masked_board >> 48U) & 0xFFFFULL;
    uint64_t block2 = (masked_board >> 32U) & 0xFFFFULL;
    uint64_t block1 = (masked_board >> 16U) & 0xFFFFULL;
    uint64_t block0 = masked_board & 0xFFFFULL;

    const InfoEntry &r0 = table[block0];
    const InfoEntry &r1 = table[block1];
    const InfoEntry &r2 = table[block2];
    const InfoEntry &r3 = table[block3];

    TileCountResult result;
    result.total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    result.count_32k = static_cast<int8_t>(r0.count_32k + r1.count_32k + r2.count_32k + r3.count_32k);
    result.pos_bitmap = static_cast<uint64_t>(r0.pos_bitmap)
        | (static_cast<uint64_t>(r1.pos_bitmap) << 16U)
        | (static_cast<uint64_t>(r2.pos_bitmap) << 32U)
        | (static_cast<uint64_t>(r3.pos_bitmap) << 48U);
    return result;
}

TileCountPositionsResult tile_sum_and_32k_count_positions(uint64_t masked_board, const AdvancedMaskParam &param) {
    masked_board &= ~param.pos_fixed_32k_mask;
    const auto &table = info_table2();
    const uint64_t block3 = (masked_board >> 48U) & 0xFFFFULL;
    const uint64_t block2 = (masked_board >> 32U) & 0xFFFFULL;
    const uint64_t block1 = (masked_board >> 16U) & 0xFFFFULL;
    const uint64_t block0 = masked_board & 0xFFFFULL;

    const InfoEntry &r0 = table[block0];
    const InfoEntry &r1 = table[block1];
    const InfoEntry &r2 = table[block2];
    const InfoEntry &r3 = table[block3];

    TileCountPositionsResult result;
    result.total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    const uint64_t pos_bitmap = static_cast<uint64_t>(r0.pos_bitmap)
        | (static_cast<uint64_t>(r1.pos_bitmap) << 16U)
        | (static_cast<uint64_t>(r2.pos_bitmap) << 32U)
        | (static_cast<uint64_t>(r3.pos_bitmap) << 48U);
    result.count_32k = static_cast<int8_t>(extract_f_positions_compact(pos_bitmap, result.pos_32k.data()));
    return result;
}

TileCountResult tile_sum_and_32k_count2(uint64_t masked_board, const AdvancedMaskParam &param) {
    masked_board &= ~param.pos_fixed_32k_mask;
    const auto &table = info_table1();
    uint64_t block3 = (masked_board >> 48U) & 0xFFFFULL;
    uint64_t block2 = (masked_board >> 32U) & 0xFFFFULL;
    uint64_t block1 = (masked_board >> 16U) & 0xFFFFULL;
    uint64_t block0 = masked_board & 0xFFFFULL;

    const InfoEntry &r0 = table[block0];
    const InfoEntry &r1 = table[block1];
    const InfoEntry &r2 = table[block2];
    const InfoEntry &r3 = table[block3];

    TileCountResult result;
    result.total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    result.count_32k = static_cast<int8_t>(r0.count_32k + r1.count_32k + r2.count_32k + r3.count_32k);
    result.pos_bitmap = static_cast<uint64_t>(r0.pos_bitmap)
        | (static_cast<uint64_t>(r1.pos_bitmap) << 16U)
        | (static_cast<uint64_t>(r2.pos_bitmap) << 32U)
        | (static_cast<uint64_t>(r3.pos_bitmap) << 48U);
    return result;
}

TileCount3Result tile_sum_and_32k_count3(uint64_t masked_board, const AdvancedMaskParam &param) {
    TileCount3Result result;
    result.masked_board = masked_board;
    int tile64_count = 0;
    uint64_t tile64_pos = 0;
    for (int i = 60; i >= 0; i -= 4) {
        uint64_t tile_value = (masked_board >> static_cast<uint64_t>(i)) & 0xFULL;
        if (tile_value > 5ULL) {
            if (((param.pos_fixed_32k_mask >> static_cast<uint64_t>(i)) & 0xFULL) == 0ULL) {
                result.pos_32k[static_cast<size_t>(result.count_32k)] = static_cast<uint64_t>(i);
                ++result.count_32k;
                if (tile_value == 6ULL) {
                    ++tile64_count;
                    tile64_pos = static_cast<uint64_t>(i);
                }
            }
        } else if (tile_value > 0ULL) {
            result.total_sum = static_cast<uint32_t>(result.total_sum + (1U << tile_value));
        }
    }
    if (tile64_count == 1) {
        result.masked_board |= (0xFULL << tile64_pos);
    }
    result.tile64_count = tile64_count;
    return result;
}

TileCount4Result tile_sum_and_32k_count4(uint64_t board, const AdvancedMaskParam &param) {
    board &= ~param.pos_fixed_32k_mask;
    const auto &table = info_table1();
    uint64_t block3 = (board >> 48U) & 0xFFFFULL;
    uint64_t block2 = (board >> 32U) & 0xFFFFULL;
    uint64_t block1 = (board >> 16U) & 0xFFFFULL;
    uint64_t block0 = board & 0xFFFFULL;

    const InfoEntry &r0 = table[block0];
    const InfoEntry &r1 = table[block1];
    const InfoEntry &r2 = table[block2];
    const InfoEntry &r3 = table[block3];

    TileCount4Result result;
    result.total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    result.count_32k = static_cast<int8_t>(r0.count_32k + r1.count_32k + r2.count_32k + r3.count_32k);
    result.pos_bitmap = static_cast<uint64_t>(r0.pos_bitmap)
        | (static_cast<uint64_t>(r1.pos_bitmap) << 16U)
        | (static_cast<uint64_t>(r2.pos_bitmap) << 32U)
        | (static_cast<uint64_t>(r3.pos_bitmap) << 48U);
    result.merged_tile_found = static_cast<uint8_t>(r0.merged_count + r1.merged_count + r2.merged_count + r3.merged_count);
    result.merged_tile = static_cast<uint8_t>(r0.merged_xor ^ r1.merged_xor ^ r2.merged_xor ^ r3.merged_xor);

    uint8_t s3 = r3.pos_rank;
    uint8_t s2 = static_cast<uint8_t>(s3 + r2.pos_rank);
    uint8_t s1 = static_cast<uint8_t>(s2 + r1.pos_rank);
    uint8_t s0 = static_cast<uint8_t>(s1 + r0.pos_rank);

    bool f3 = (r3.merged_count != 0);
    bool f2 = (r3.merged_count == 0) && (r2.merged_count != 0);
    bool f1 = (r3.merged_count == 0) && (r2.merged_count == 0) && (r1.merged_count != 0);
    bool f0 = (r3.merged_count == 0) && (r2.merged_count == 0) && (r1.merged_count == 0);
    result.pos_rank = static_cast<uint8_t>(s0 * static_cast<uint8_t>(f0)
        + s1 * static_cast<uint8_t>(f1)
        + s2 * static_cast<uint8_t>(f2)
        + s3 * static_cast<uint8_t>(f3));
    result.is_success = (result.merged_tile == param.target) && (result.merged_tile_found == 1);
    return result;
}

std::vector<uint8_t> masked_tiles_combinations(uint64_t remaining_sum, int remaining_count) {
    std::vector<uint8_t> result;
    result.reserve(10);
    for (uint32_t tile : kLargeTiles) {
        if (tile > remaining_sum) {
            continue;
        }
        if (tile == remaining_sum) {
            if (remaining_count == 1) {
                result.push_back(log2_uint(tile));
                return result;
            }
        } else {
            if ((static_cast<uint64_t>(tile) << 1U) == remaining_sum && remaining_count == 2) {
                result.push_back(log2_uint(tile));
                result.push_back(log2_uint(tile));
                return result;
            }
            if (tile != 64U && remaining_sum == 192U && remaining_count == 3) {
                result.push_back(6);
                result.push_back(6);
                result.push_back(6);
                return result;
            }
            remaining_sum -= tile;
            --remaining_count;
            result.push_back(log2_uint(tile));
        }
    }
    return {};
}

PermutationSlot resort_permutations(int m, int n, const PermutationSlot &permutation, bool type_flag) {
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> keyed_rows;
    keyed_rows.reserve(permutation.rows);

    std::vector<uint8_t> pos(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        pos[static_cast<size_t>(i)] = static_cast<uint8_t>((m - 1 - i) * 4);
    }

    std::vector<uint64_t> tiles(static_cast<size_t>(n));
    if (type_flag) {
        for (int i = 0; i < n; ++i) {
            tiles[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
        }
    } else {
        if (n > 0) {
            tiles[0] = 0;
            for (int i = 1; i < n; ++i) {
                tiles[static_cast<size_t>(i)] = static_cast<uint64_t>(i - 1);
            }
        }
    }

    auto view = permutation.view();
    for (size_t row_index = 0; row_index < view.rows; ++row_index) {
        const uint8_t *row = view.row(row_index);
        uint64_t clear_mask = 0;
        uint64_t set_mask = 0;
        for (int j = 0; j < n; ++j) {
            uint8_t shift = pos[row[static_cast<size_t>(j)]];
            clear_mask += (0xFULL << shift);
            set_mask += (tiles[static_cast<size_t>(j)] << shift);
        }
        uint64_t masked = (~clear_mask) | set_mask;
        keyed_rows.push_back({masked, std::vector<uint8_t>(row, row + n)});
    }

    std::sort(keyed_rows.begin(), keyed_rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
    });
    return make_slot_from_sorted_rows(keyed_rows);
}

PermutationSlot generate_permutations(int m, int n) {
    std::vector<std::vector<uint8_t>> rows;
    std::vector<uint8_t> current;
    std::vector<bool> used(static_cast<size_t>(m), false);
    current.reserve(static_cast<size_t>(n));

    std::function<void()> dfs = [&]() {
        if (static_cast<int>(current.size()) == n) {
            rows.push_back(current);
            return;
        }
        for (int i = 0; i < m; ++i) {
            if (used[static_cast<size_t>(i)]) {
                continue;
            }
            used[static_cast<size_t>(i)] = true;
            current.push_back(static_cast<uint8_t>(i));
            dfs();
            current.pop_back();
            used[static_cast<size_t>(i)] = false;
        }
    };
    dfs();
    return resort_permutations(m, n, make_slot_from_rows(rows), true);
}

void build_permutation_subsets(PermutationSlot &slot) {
    slot.first_subset_offsets.fill(0);
    slot.first_subset_indices.clear();
    slot.pair_subset_offsets.fill(0);
    slot.pair_subset_indices.clear();

    auto view = slot.view();
    if (view.empty()) {
        return;
    }

    std::array<uint32_t, 16> first_counts{};
    std::array<uint32_t, 256> pair_counts{};

    for (size_t row_index = 0; row_index < view.rows; ++row_index) {
        const uint8_t *row = view.row(row_index);
        ++first_counts[row[0]];
        if (view.cols > 1) {
            const size_t pair_code = (static_cast<size_t>(row[0]) << 4U) + row[1];
            ++pair_counts[pair_code];
        }
    }

    for (size_t i = 0; i < first_counts.size(); ++i) {
        slot.first_subset_offsets[i + 1] = slot.first_subset_offsets[i] + first_counts[i];
    }
    slot.first_subset_indices.resize(view.rows);
    auto first_write_offsets = slot.first_subset_offsets;
    for (size_t row_index = 0; row_index < view.rows; ++row_index) {
        const uint8_t *row = view.row(row_index);
        slot.first_subset_indices[first_write_offsets[row[0]]++] = static_cast<uint32_t>(row_index);
    }

    if (view.cols <= 1) {
        return;
    }

    for (size_t i = 0; i < pair_counts.size(); ++i) {
        slot.pair_subset_offsets[i + 1] = slot.pair_subset_offsets[i] + pair_counts[i];
    }
    slot.pair_subset_indices.resize(slot.pair_subset_offsets.back());
    auto pair_write_offsets = slot.pair_subset_offsets;
    for (size_t row_index = 0; row_index < view.rows; ++row_index) {
        const uint8_t *row = view.row(row_index);
        const size_t pair_code = (static_cast<size_t>(row[0]) << 4U) + row[1];
        slot.pair_subset_indices[pair_write_offsets[pair_code]++] = static_cast<uint32_t>(row_index);
    }
}

bool validate(uint64_t board, uint32_t original_board_sum, const TilesCombinationTable &tiles_combinations_table, const AdvancedMaskParam &param) {
    TileCountResult stats = tile_sum_and_32k_count2(board, param);
    if (stats.total_sum >= param.small_tile_sum_limit + 64U) {
        return false;
    }
    if (stats.count_32k == static_cast<int8_t>(param.num_free_32k)) {
        return true;
    }
    uint32_t large_tiles_sum = original_board_sum - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles_combinations = tiles_combination_view(
        tiles_combinations_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles_combinations.empty()) {
        return false;
    }
    if (tiles_combinations[tiles_combinations.size - 1] == param.target) {
        return false;
    }
    if (stats.count_32k < 2) {
        return true;
    }
    if (tiles_combinations.size > 2 && tiles_combinations[0] == tiles_combinations[2]) {
        if (stats.total_sum >= param.small_tile_sum_limit - 64U) {
            return false;
        }
    } else if (tiles_combinations.size > 1 && tiles_combinations[0] == tiles_combinations[1]) {
        if (stats.total_sum >= param.small_tile_sum_limit) {
            return false;
        }
    }
    return true;
}

void unmask_board_into(
    uint64_t board,
    uint32_t original_board_sum,
    const TilesCombinationTable &tiles_combinations_table,
    const PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    std::vector<uint64_t> &boards
) {
    TileCountPositionsResult stats = tile_sum_and_32k_count_positions(board, param);
    if (stats.count_32k == 0 || stats.count_32k == static_cast<int8_t>(param.num_free_32k)) {
        boards.resize(1);
        boards[0] = board;
        return;
    }
    uint32_t large_tiles_sum = original_board_sum - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles_combinations = tiles_combination_view(
        tiles_combinations_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles_combinations.empty()) {
        boards.clear();
        return;
    }

    auto permutation_all = permutation_view(
        permutation_table,
        static_cast<uint8_t>(stats.count_32k),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (permutation_all.empty()) {
        boards.clear();
        return;
    }
    boards.clear();
    boards.reserve(permutation_all.rows);

    for (size_t row_index = 0; row_index < permutation_all.rows; ++row_index) {
        const uint8_t *permutation = permutation_all.row(row_index);
        if (tiles_combinations.size > 1 && tiles_combinations[0] == tiles_combinations[1]) {
            if (!(permutation[1] < permutation[0])) {
                continue;
            }
        }
        uint64_t clear_mask = 0;
        uint64_t set_mask = 0;
        for (size_t j = 0; j < tiles_combinations.size; ++j) {
            const uint64_t shift = static_cast<uint64_t>(stats.pos_32k[permutation[j]]);
            clear_mask += (0xFULL << shift);
            set_mask += (static_cast<uint64_t>(tiles_combinations[j]) << shift);
        }
        uint64_t masked = (board & ~clear_mask) | set_mask;
        boards.push_back(masked);
    }
}

std::vector<uint64_t> unmask_board(
    uint64_t board,
    uint32_t original_board_sum,
    const TilesCombinationTable &tiles_combinations_table,
    const PermutationTable &permutation_table,
    const AdvancedMaskParam &param
) {
    std::vector<uint64_t> boards;
    unmask_board_into(board, original_board_sum, tiles_combinations_table, permutation_table, param, boards);
    return boards;
}

MaskerContext init_masker(const AdvancedPatternSpec &spec) {
    MaskerContext context;
    context.param = build_mask_param(spec);
    for (int n = 1; n < static_cast<int>(spec.target) - 4; ++n) {
        int m = n + static_cast<int>(spec.num_free_32k);
        auto slot = generate_permutations(m, n);
        build_permutation_subsets(slot);
        context.permutation_table[permutation_index(static_cast<uint8_t>(m), static_cast<uint8_t>(n))] = std::move(slot);
    }
    for (int a = 1; a < 256; ++a) {
        for (int b = 1; b < 10; ++b) {
            std::vector<uint8_t> tile = masked_tiles_combinations(static_cast<uint64_t>(a << 6), b);
            if (!tile.empty()) {
                std::reverse(tile.begin(), tile.end());
                context.tiles_combination_table[tiles_combination_index(static_cast<uint8_t>(a), static_cast<uint8_t>(b))] = std::move(tile);
            }
        }
    }
    return context;
}

} // namespace FormationAD
