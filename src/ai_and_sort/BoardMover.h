#pragma once
#include <cstdint>
#include <tuple>
#include <array>
#include <algorithm>

// ------------------------------------------------------------------
// 合并后的全局预计算表
// ------------------------------------------------------------------
struct alignas(8) RowEntry {
    uint16_t movel;
    uint16_t mover;
    uint16_t score;
    uint16_t _pad;
};

inline RowEntry row_table[65536];

// ------------------------------------------------------------------
// 基础辅助函数
// ------------------------------------------------------------------
inline uint64_t reverse_board(uint64_t board) {
    board = (board & 0xFF00FF0000FF00FFULL) | ((board & 0x00FF00FF00000000ULL) >> 24) | ((board & 0x00000000FF00FF00ULL) << 24);
    board = (board & 0xF0F00F0FF0F00F0FULL) | ((board & 0x0F0F00000F0F0000ULL) >> 12) | ((board & 0x0000F0F00000F0F0ULL) << 12);
    return board;
}

uint64_t canonical_diagonal(uint64_t board) {
    uint64_t board2 = reverse_board(board);
    return board < board2 ? board : board2;
}

inline int get_log2(int val) {
    if (val == 0) return 0;
    int res = 0;
    while (val >>= 1) ++res;
    return res;
}

void init_tables() {
    for (uint32_t orig = 0; orig < 65536; ++orig) {
        int k[4] = {
            (int)((orig >> 12) & 0xF),
            (int)((orig >> 8)  & 0xF),
            (int)((orig >> 4)  & 0xF),
            (int)((orig >> 0)  & 0xF)
        };
        int vals[4] = {
            k[0] ? (1 << k[0]) : 0,
            k[1] ? (1 << k[1]) : 0,
            k[2] ? (1 << k[2]) : 0,
            k[3] ? (1 << k[3]) : 0
        };

        auto merge_line = [&](bool reverse_line) -> std::pair<uint16_t, uint16_t> {
            int non_zero[4];
            int nz_count = 0;
            if (reverse_line) {
                for (int i = 3; i >= 0; --i) if (vals[i] != 0) non_zero[nz_count++] = vals[i];
            } else {
                for (int i = 0; i < 4; ++i) if (vals[i] != 0) non_zero[nz_count++] = vals[i];
            }

            int merged[4] = {0, 0, 0, 0};
            uint32_t score = 0;
            int m_count = 0;
            for (int i = 0; i < nz_count; ++i) {
                if (i + 1 < nz_count && non_zero[i] == non_zero[i + 1] && non_zero[i] != 32768) {
                    merged[m_count++] = non_zero[i] * 2;
                    score += non_zero[i] * 2;
                    ++i;
                } else {
                    merged[m_count++] = non_zero[i];
                }
            }

            uint16_t res = 0;
            for (int i = 0; i < 4; ++i) {
                int log_v = get_log2(reverse_line ? merged[3 - i] : merged[i]);
                res |= (log_v << (4 * (3 - i)));
            }
            return {res, (uint16_t)std::min(score, 65535U)};
        };

        auto [l_move, l_score] = merge_line(false);
        auto [r_move, r_score] = merge_line(true);

        row_table[orig].movel = l_move ^ (uint16_t)orig;
        row_table[orig].mover = r_move ^ (uint16_t)orig;
        row_table[orig].score = l_score;
        row_table[orig]._pad  = 0;
    }
}

struct TableInitializer {
    TableInitializer() { init_tables(); }
};
inline TableInitializer _initializer;

// ------------------------------------------------------------------
// 移动函数
// ------------------------------------------------------------------
inline std::tuple<uint64_t, uint32_t> s_move_left(uint64_t board) {
    // 获取四行的索引
    uint16_t rows[4] = {
        (uint16_t)board,
        (uint16_t)(board >> 16),
        (uint16_t)(board >> 32),
        (uint16_t)(board >> 48)
    };

    const RowEntry& e0 = row_table[rows[0]];
    const RowEntry& e1 = row_table[rows[1]];
    const RowEntry& e2 = row_table[rows[2]];
    const RowEntry& e3 = row_table[rows[3]];

    uint32_t total_score = e0.score + e1.score + e2.score + e3.score;

    board ^= (uint64_t)e0.movel;
    board ^= (uint64_t)e1.movel << 16;
    board ^= (uint64_t)e2.movel << 32;
    board ^= (uint64_t)e3.movel << 48;

    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_right(uint64_t board) {
    uint16_t rows[4] = {
        (uint16_t)board,
        (uint16_t)(board >> 16),
        (uint16_t)(board >> 32),
        (uint16_t)(board >> 48)
    };

    const RowEntry& e0 = row_table[rows[0]];
    const RowEntry& e1 = row_table[rows[1]];
    const RowEntry& e2 = row_table[rows[2]];
    const RowEntry& e3 = row_table[rows[3]];

    uint32_t total_score = e0.score + e1.score + e2.score + e3.score;

    board ^= (uint64_t)e0.mover;
    board ^= (uint64_t)e1.mover << 16;
    board ^= (uint64_t)e2.mover << 32;
    board ^= (uint64_t)e3.mover << 48;

    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_up(uint64_t board, uint64_t board2) {
    auto [bd, total_score] = s_move_left(board2);
    board = reverse_board(bd);
    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_down(uint64_t board, uint64_t board2) {
    auto [bd, total_score] = s_move_right(board2);
    board = reverse_board(bd);
    return {board, total_score};
}

std::tuple<uint64_t, uint32_t> s_move_board(uint64_t board, int direction) {
    if (direction == 1) return s_move_left(board);
    else if (direction == 2) return s_move_right(board);
    else if (direction == 3) return s_move_up(board, reverse_board(board));
    else if (direction == 4) return s_move_down(board, reverse_board(board));
    return {board, 0};
}

struct MoveResult {
    uint64_t board;
    uint32_t score;
    bool is_valid; // 是否是有效移动
};

inline std::array<MoveResult, 4> s_move_board_all(uint64_t board) {
    uint64_t rev_board = reverse_board(board); 
    
    auto [t_l, s_l] = s_move_left(board);
    auto [t_r, s_r] = s_move_right(board);
    auto [t_u, s_u] = s_move_up(board, rev_board);
    auto [t_d, s_d] = s_move_down(board, rev_board);

    return {{
        {t_l, s_l, t_l != board},
        {t_r, s_r, t_r != board},
        {t_u, s_u, t_u != board},
        {t_d, s_d, t_d != board}
    }};
}
