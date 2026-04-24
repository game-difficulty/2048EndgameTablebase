#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "FormationRuntime.h"

namespace FormationAD {

struct InfoEntry {
    uint16_t total_sum = 0;
    uint8_t count_32k = 0;
    uint16_t pos_bitmap = 0;
    uint8_t merged_count = 0;
    uint8_t merged_xor = 0;
    uint8_t pos_rank = 0;
};

struct PermutationSlot {
    std::vector<uint8_t> data;
    size_t rows = 0;
    size_t cols = 0;
    std::array<uint32_t, 17> first_subset_offsets{};
    std::vector<uint32_t> first_subset_indices;
    std::array<uint32_t, 257> pair_subset_offsets{};
    std::vector<uint32_t> pair_subset_indices;

    [[nodiscard]] bool empty() const {
        return data.empty() || rows == 0 || cols == 0;
    }

    [[nodiscard]] MatrixView<uint8_t> view() {
        return {data.data(), rows, cols};
    }

    [[nodiscard]] MatrixView<const uint8_t> view() const {
        return {data.data(), rows, cols};
    }
};

using PermutationTable = std::array<PermutationSlot, 256>;
using TilesCombinationTable = std::array<std::vector<uint8_t>, 2560>;

struct MaskerContext {
    AdvancedMaskParam param;
    PermutationTable permutation_table{};
    TilesCombinationTable tiles_combination_table{};
};

struct TileCountResult {
    uint32_t total_sum = 0;
    int8_t count_32k = 0;
    uint64_t pos_bitmap = 0;
};

struct TileCount3Result {
    uint32_t total_sum = 0;
    int8_t count_32k = 0;
    std::array<uint64_t, 16> pos_32k{};
    uint64_t masked_board = 0;
    int tile64_count = 0;
};

struct TileCount4Result {
    uint32_t total_sum = 0;
    int8_t count_32k = 0;
    uint64_t pos_bitmap = 0;
    uint8_t pos_rank = 0;
    uint8_t merged_tile_found = 0;
    uint8_t merged_tile = 0;
    bool is_success = false;
};

uint64_t fixed_32k_mask(const std::vector<uint8_t> &fixed_shifts);
AdvancedMaskParam build_mask_param(const AdvancedPatternSpec &spec);
MaskerContext init_masker(const AdvancedPatternSpec &spec);

uint64_t mask_board(uint64_t board, int threshold = 6);

TileCountResult tile_sum_and_32k_count(uint64_t masked_board, const AdvancedMaskParam &param);
TileCountResult tile_sum_and_32k_count2(uint64_t masked_board, const AdvancedMaskParam &param);
TileCount3Result tile_sum_and_32k_count3(uint64_t masked_board, const AdvancedMaskParam &param);
TileCount4Result tile_sum_and_32k_count4(uint64_t board, const AdvancedMaskParam &param);

std::vector<uint8_t> masked_tiles_combinations(uint64_t remaining_sum, int remaining_count);
bool validate(uint64_t board, uint32_t original_board_sum, const TilesCombinationTable &tiles_combinations_table, const AdvancedMaskParam &param);
std::vector<uint64_t> unmask_board(
    uint64_t board,
    uint32_t original_board_sum,
    const TilesCombinationTable &tiles_combinations_table,
    const PermutationTable &permutation_table,
    const AdvancedMaskParam &param
);
void unmask_board_into(
    uint64_t board,
    uint32_t original_board_sum,
    const TilesCombinationTable &tiles_combinations_table,
    const PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    std::vector<uint64_t> &boards
);

std::vector<uint64_t> extract_f_positions(uint64_t pos_bitmap);
PermutationSlot generate_permutations(int m, int n);
PermutationSlot resort_permutations(int m, int n, const PermutationSlot &permutation, bool type_flag);
void build_permutation_subsets(PermutationSlot &slot);

const std::array<InfoEntry, 65536> &info_table1();
const std::array<InfoEntry, 65536> &info_table2();

inline size_t permutation_index(uint8_t a, uint8_t b) {
    return static_cast<size_t>((static_cast<uint16_t>(a) << 4U) + b);
}

inline size_t tiles_combination_index(uint8_t a, uint8_t b) {
    return static_cast<size_t>(a + (static_cast<uint16_t>(b) << 8U));
}

inline MatrixView<const uint8_t> permutation_view(const PermutationTable &table, uint8_t a, uint8_t b) {
    return table[permutation_index(a, b)].view();
}

inline ArrayView<const uint32_t> permutation_first_subset(
    const PermutationTable &table,
    uint8_t a,
    uint8_t b,
    uint8_t first
) {
    const auto &slot = table[permutation_index(a, b)];
    const size_t begin = slot.first_subset_offsets[first];
    const size_t end = slot.first_subset_offsets[static_cast<size_t>(first) + 1];
    return {slot.first_subset_indices.data() + begin, end - begin};
}

inline ArrayView<const uint32_t> permutation_pair_subset(
    const PermutationTable &table,
    uint8_t a,
    uint8_t b,
    uint8_t first,
    uint8_t second
) {
    const auto &slot = table[permutation_index(a, b)];
    const size_t pair_code = (static_cast<size_t>(first) << 4U) + second;
    const size_t begin = slot.pair_subset_offsets[pair_code];
    const size_t end = slot.pair_subset_offsets[pair_code + 1];
    return {slot.pair_subset_indices.data() + begin, end - begin};
}

inline ArrayView<const uint8_t> tiles_combination_view(const TilesCombinationTable &table, uint8_t a, uint8_t b) {
    const auto &slot = table[tiles_combination_index(a, b)];
    return {slot.data(), slot.size()};
}

} // namespace FormationAD
