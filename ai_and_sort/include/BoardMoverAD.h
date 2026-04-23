#pragma once

#include <array>
#include <cstdint>
#include <tuple>

#include "CommonMover.h"

namespace FormationAD {

struct ADMoveResult {
    uint64_t board = 0;
    bool mask_new_tile = false;
};

uint16_t merge_line(uint16_t row, bool reverse_line, bool &mask_new_tile);
uint64_t m_move_left(uint64_t board);
uint64_t m_move_right(uint64_t board);
uint64_t m_move_up(uint64_t board, uint64_t board_rev);
uint64_t m_move_down(uint64_t board, uint64_t board_rev);
std::pair<uint64_t, bool> m_move_left2(uint64_t board);
std::pair<uint64_t, bool> m_move_up2(uint64_t board, uint64_t board_rev);
uint64_t m_move_board(uint64_t board, int direction);
uint64_t m_move_board2(uint64_t board, uint64_t board_rev, int direction);
std::array<ADMoveResult, 4> m_move_all_dir(uint64_t board);
std::array<ADMoveResult, 4> m_move_all_dir2(uint64_t board, uint64_t board_rev);

inline uint64_t reverse(uint64_t board) {
    return reverse_board(board);
}

} // namespace FormationAD
