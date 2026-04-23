#include "BoardMoverAD.h"

#include <algorithm>
#include <vector>

namespace FormationAD {

namespace {

std::array<uint16_t, 65536> g_movel{};
std::array<uint16_t, 65536> g_mover{};
std::array<uint64_t, 65536> g_moveu{};
std::array<uint64_t, 65536> g_moved{};
std::array<bool, 65536> g_mask_new_tiles{};

struct ADRowTableInitializer {
    ADRowTableInitializer() {
        for (uint32_t i = 0; i < 65536; ++i) {
            bool mask_new_tile = false;
            uint16_t left = merge_line(static_cast<uint16_t>(i), false, mask_new_tile);
            g_movel[i] = static_cast<uint16_t>(left ^ static_cast<uint16_t>(i));
            g_mask_new_tiles[i] = mask_new_tile;

            bool ignored = false;
            uint16_t right = merge_line(static_cast<uint16_t>(i), true, ignored);
            g_mover[i] = static_cast<uint16_t>(right ^ static_cast<uint16_t>(i));
        }
        for (uint32_t i = 0; i < 65536; ++i) {
            g_moveu[i] = reverse_board(static_cast<uint64_t>(g_movel[i]));
            g_moved[i] = reverse_board(static_cast<uint64_t>(g_mover[i]));
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
    board ^= static_cast<uint64_t>(g_movel[board & 0xFFFFULL]);
    board ^= static_cast<uint64_t>(g_movel[(board >> 16U) & 0xFFFFULL]) << 16U;
    board ^= static_cast<uint64_t>(g_movel[(board >> 32U) & 0xFFFFULL]) << 32U;
    board ^= static_cast<uint64_t>(g_movel[(board >> 48U) & 0xFFFFULL]) << 48U;
    return board;
}

std::pair<uint64_t, bool> m_move_left2(uint64_t board) {
    bool mnt = false;
    mnt = mnt || g_mask_new_tiles[board & 0xFFFFULL];
    board ^= static_cast<uint64_t>(g_movel[board & 0xFFFFULL]);
    mnt = mnt || g_mask_new_tiles[(board >> 16U) & 0xFFFFULL];
    board ^= static_cast<uint64_t>(g_movel[(board >> 16U) & 0xFFFFULL]) << 16U;
    mnt = mnt || g_mask_new_tiles[(board >> 32U) & 0xFFFFULL];
    board ^= static_cast<uint64_t>(g_movel[(board >> 32U) & 0xFFFFULL]) << 32U;
    mnt = mnt || g_mask_new_tiles[(board >> 48U) & 0xFFFFULL];
    board ^= static_cast<uint64_t>(g_movel[(board >> 48U) & 0xFFFFULL]) << 48U;
    return {board, mnt};
}

uint64_t m_move_right(uint64_t board) {
    board ^= static_cast<uint64_t>(g_mover[board & 0xFFFFULL]);
    board ^= static_cast<uint64_t>(g_mover[(board >> 16U) & 0xFFFFULL]) << 16U;
    board ^= static_cast<uint64_t>(g_mover[(board >> 32U) & 0xFFFFULL]) << 32U;
    board ^= static_cast<uint64_t>(g_mover[(board >> 48U) & 0xFFFFULL]) << 48U;
    return board;
}

uint64_t m_move_up(uint64_t board, uint64_t board_rev) {
    board ^= g_moveu[board_rev & 0xFFFFULL];
    board ^= g_moveu[(board_rev >> 16U) & 0xFFFFULL] << 4U;
    board ^= g_moveu[(board_rev >> 32U) & 0xFFFFULL] << 8U;
    board ^= g_moveu[(board_rev >> 48U) & 0xFFFFULL] << 12U;
    return board;
}

std::pair<uint64_t, bool> m_move_up2(uint64_t board, uint64_t board_rev) {
    bool mnt = false;
    mnt = mnt || g_mask_new_tiles[board_rev & 0xFFFFULL];
    board ^= g_moveu[board_rev & 0xFFFFULL];
    mnt = mnt || g_mask_new_tiles[(board_rev >> 16U) & 0xFFFFULL];
    board ^= g_moveu[(board_rev >> 16U) & 0xFFFFULL] << 4U;
    mnt = mnt || g_mask_new_tiles[(board_rev >> 32U) & 0xFFFFULL];
    board ^= g_moveu[(board_rev >> 32U) & 0xFFFFULL] << 8U;
    mnt = mnt || g_mask_new_tiles[(board_rev >> 48U) & 0xFFFFULL];
    board ^= g_moveu[(board_rev >> 48U) & 0xFFFFULL] << 12U;
    return {board, mnt};
}

uint64_t m_move_down(uint64_t board, uint64_t board_rev) {
    board ^= g_moved[board_rev & 0xFFFFULL];
    board ^= g_moved[(board_rev >> 16U) & 0xFFFFULL] << 4U;
    board ^= g_moved[(board_rev >> 32U) & 0xFFFFULL] << 8U;
    board ^= g_moved[(board_rev >> 48U) & 0xFFFFULL] << 12U;
    return board;
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
