#include "BoardMoverAD.h"

#include <algorithm>
#include <vector>

namespace FormationAD {

namespace {

struct ADRowMove {
    uint16_t movel = 0;
    uint16_t mover = 0;
};

std::array<ADRowMove, 65536> g_row_table{};
std::array<uint64_t, 1024> g_mask_new_tile_bits{};

inline void set_mask_new_tile(uint32_t row) {
    g_mask_new_tile_bits[row >> 6U] |= (1ULL << (row & 63U));
}

inline bool has_mask_new_tile(uint64_t row) {
    return ((g_mask_new_tile_bits[row >> 6U] >> (row & 63U)) & 1ULL) != 0ULL;
}

struct ADRowTableInitializer {
    ADRowTableInitializer() {
        for (uint32_t i = 0; i < 65536; ++i) {
            bool mask_new_tile = false;
            uint16_t left = merge_line(static_cast<uint16_t>(i), false, mask_new_tile);

            bool ignored = false;
            uint16_t right = merge_line(static_cast<uint16_t>(i), true, ignored);
            g_row_table[i] = ADRowMove{
                static_cast<uint16_t>(left ^ static_cast<uint16_t>(i)),
                static_cast<uint16_t>(right ^ static_cast<uint16_t>(i))
            };
            if (mask_new_tile) {
                set_mask_new_tile(i);
            }
        }
    }
};

ADRowTableInitializer g_initializer;

} // namespace

uint16_t merge_line(uint16_t row, bool reverse_line, bool &mask_new_tile) {
    uint32_t line[4];
    decode_row(row, line);
    if (reverse_line) {
        std::reverse(std::begin(line), std::end(line));
    }

    std::vector<uint32_t> non_zero;
    non_zero.reserve(4);
    for (uint32_t value : line) {
        if (value != 0) {
            non_zero.push_back(value);
        }
    }

    uint32_t merged[4] = {0, 0, 0, 0};
    size_t merged_count = 0;
    bool skip = false;
    for (size_t i = 0; i < non_zero.size(); ++i) {
        if (skip) {
            skip = false;
            continue;
        }
        if (i + 1 < non_zero.size() && non_zero[i] == non_zero[i + 1] && non_zero[i] != 32768U) {
            uint32_t merged_value = non_zero[i] * 2U;
            if (merged_value >= 64U) {
                mask_new_tile = true;
                merged_value = 32768U;
            }
            merged[merged_count++] = merged_value;
            skip = true;
        } else {
            merged[merged_count++] = non_zero[i];
        }
    }

    if (reverse_line) {
        std::reverse(std::begin(merged), std::end(merged));
    }
    return encode_row(merged);
}

uint64_t m_move_left(uint64_t board) {
    board ^= static_cast<uint64_t>(g_row_table[board & 0xFFFFULL].movel);
    board ^= static_cast<uint64_t>(g_row_table[(board >> 16U) & 0xFFFFULL].movel) << 16U;
    board ^= static_cast<uint64_t>(g_row_table[(board >> 32U) & 0xFFFFULL].movel) << 32U;
    board ^= static_cast<uint64_t>(g_row_table[(board >> 48U) & 0xFFFFULL].movel) << 48U;
    return board;
}

std::pair<uint64_t, bool> m_move_left2(uint64_t board) {
    const uint64_t r0 = board & 0xFFFFULL;
    const uint64_t r1 = (board >> 16U) & 0xFFFFULL;
    const uint64_t r2 = (board >> 32U) & 0xFFFFULL;
    const uint64_t r3 = (board >> 48U) & 0xFFFFULL;
    const ADRowMove &e0 = g_row_table[r0];
    const ADRowMove &e1 = g_row_table[r1];
    const ADRowMove &e2 = g_row_table[r2];
    const ADRowMove &e3 = g_row_table[r3];
    const bool mnt = has_mask_new_tile(r0) || has_mask_new_tile(r1) || has_mask_new_tile(r2) || has_mask_new_tile(r3);
    board ^= static_cast<uint64_t>(e0.movel);
    board ^= static_cast<uint64_t>(e1.movel) << 16U;
    board ^= static_cast<uint64_t>(e2.movel) << 32U;
    board ^= static_cast<uint64_t>(e3.movel) << 48U;
    return {board, mnt};
}

uint64_t m_move_right(uint64_t board) {
    board ^= static_cast<uint64_t>(g_row_table[board & 0xFFFFULL].mover);
    board ^= static_cast<uint64_t>(g_row_table[(board >> 16U) & 0xFFFFULL].mover) << 16U;
    board ^= static_cast<uint64_t>(g_row_table[(board >> 32U) & 0xFFFFULL].mover) << 32U;
    board ^= static_cast<uint64_t>(g_row_table[(board >> 48U) & 0xFFFFULL].mover) << 48U;
    return board;
}

uint64_t m_move_up(uint64_t board, uint64_t board_rev) {
    (void) board;
    return reverse_board(m_move_left(board_rev));
}

std::pair<uint64_t, bool> m_move_up2(uint64_t board, uint64_t board_rev) {
    (void) board;
    auto [moved_rev, mnt] = m_move_left2(board_rev);
    return {reverse_board(moved_rev), mnt};
}

uint64_t m_move_down(uint64_t board, uint64_t board_rev) {
    (void) board;
    return reverse_board(m_move_right(board_rev));
}

uint64_t m_move_board(uint64_t board, int direction) {
    if (direction == 1) {
        return m_move_left(board);
    }
    if (direction == 2) {
        return m_move_right(board);
    }
    if (direction == 3) {
        uint64_t board_rev = reverse(board);
        return m_move_up(board, board_rev);
    }
    if (direction == 4) {
        uint64_t board_rev = reverse(board);
        return m_move_down(board, board_rev);
    }
    return board;
}

uint64_t m_move_board2(uint64_t board, uint64_t board_rev, int direction) {
    if (direction == 1) {
        return m_move_left(board);
    }
    if (direction == 2) {
        return m_move_right(board);
    }
    if (direction == 3) {
        return m_move_up(board, board_rev);
    }
    return m_move_down(board, board_rev);
}

std::array<ADMoveResult, 4> m_move_all_dir(uint64_t board) {
    uint64_t board_rev = reverse(board);
    uint64_t md = m_move_down(board, board_rev);
    uint64_t mr = m_move_right(board);
    auto [ml, mnt_h] = m_move_left2(board);
    auto [mu, mnt_v] = m_move_up2(board, board_rev);
    return {ADMoveResult{ml, mnt_h}, ADMoveResult{mr, mnt_h}, ADMoveResult{mu, mnt_v}, ADMoveResult{md, mnt_v}};
}

std::array<ADMoveResult, 4> m_move_all_dir2(uint64_t board, uint64_t board_rev) {
    uint64_t md = m_move_down(board, board_rev);
    uint64_t mr = m_move_right(board);
    auto [ml, mnt_h] = m_move_left2(board);
    auto [mu, mnt_v] = m_move_up2(board, board_rev);
    return {ADMoveResult{ml, mnt_h}, ADMoveResult{mr, mnt_h}, ADMoveResult{mu, mnt_v}, ADMoveResult{md, mnt_v}};
}

} // namespace FormationAD
