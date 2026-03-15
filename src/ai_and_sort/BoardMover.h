#pragma once
#include <cstdint>
#include <tuple>
#include <random>
#include <array>

// ------------------------------------------------------------------
// 全局预计算表
// ------------------------------------------------------------------
struct alignas(4) LR_Move {
    uint16_t movel;
    uint16_t mover;
};

struct alignas(16) UD_Move {
    uint64_t moveu;
    uint64_t moved;
};

inline LR_Move _lr_moves[65536];
inline UD_Move _ud_moves[65536];
inline uint32_t score_table[65536];

// ------------------------------------------------------------------
// 基础辅助函数
// ------------------------------------------------------------------
inline uint64_t reverse_board(uint64_t board) {
    board = (board & 0xFF00FF0000FF00FFULL) | ((board & 0x00FF00FF00000000ULL) >> 24) | ((board & 0x00000000FF00FF00ULL) << 24);
    board = (board & 0xF0F00F0FF0F00F0FULL) | ((board & 0x0F0F00000F0F0000ULL) >> 12) | ((board & 0x0000F0F00000F0F0ULL) << 12);
    return board;
}

// 模拟 log2 操作，用于预计算
inline int get_log2(int val) {
    if (val == 0) return 0;
    int res = 0;
    while (val >>= 1) ++res;
    return res;
}

// 初始化所有查找表 (在模块加载时自动执行一次)
void init_tables() {
    for (uint32_t orig = 0; orig < 65536; ++orig) {
        // 解析 16 位整数为 4 个独立数字 (基于 2 的幂)
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

        // 处理向左合并
        auto merge_line = [&](bool reverse_line) -> std::pair<uint16_t, uint32_t> {
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
                    ++i; // skip
                } else {
                    merged[m_count++] = non_zero[i];
                }
            }

            uint16_t res = 0;
            if (reverse_line) {
                for (int i = 0; i < 4; ++i) {
                    int log_v = get_log2(merged[3 - i]);
                    res |= (log_v << (4 * (3 - i)));
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    int log_v = get_log2(merged[i]);
                    res |= (log_v << (4 * (3 - i)));
                }
            }
            return {res, score};
        };

        auto left_res = merge_line(false);
        auto right_res = merge_line(true);

        score_table[orig] = left_res.second;

        _lr_moves[orig].movel = left_res.first ^ orig;
        _lr_moves[orig].mover = right_res.first ^ orig;

        _ud_moves[orig].moveu = reverse_board((uint64_t)_lr_moves[orig].movel);
        _ud_moves[orig].moved = reverse_board((uint64_t)_lr_moves[orig].mover);
    }
}

struct TableInitializer {
    TableInitializer() { init_tables(); }
};
inline TableInitializer _initializer;

// ------------------------------------------------------------------
// 带分数的移动函数
// ------------------------------------------------------------------
inline std::tuple<uint64_t, uint32_t> s_move_left(uint64_t board) {
    uint16_t l0 = (uint16_t)board;
    uint16_t l1 = (uint16_t)(board >> 16);
    uint16_t l2 = (uint16_t)(board >> 32);
    uint16_t l3 = (uint16_t)(board >> 48);

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= (uint64_t)_lr_moves[l0].movel;
    board ^= (uint64_t)_lr_moves[l1].movel << 16;
    board ^= (uint64_t)_lr_moves[l2].movel << 32;
    board ^= (uint64_t)_lr_moves[l3].movel << 48;

    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_right(uint64_t board) {
    uint16_t l0 = (uint16_t)board;
    uint16_t l1 = (uint16_t)(board >> 16);
    uint16_t l2 = (uint16_t)(board >> 32);
    uint16_t l3 = (uint16_t)(board >> 48);

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= (uint64_t)_lr_moves[l0].mover;
    board ^= (uint64_t)_lr_moves[l1].mover << 16;
    board ^= (uint64_t)_lr_moves[l2].mover << 32;
    board ^= (uint64_t)_lr_moves[l3].mover << 48;

    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_up(uint64_t board, uint64_t board2) {
    uint16_t l0 = (uint16_t)board2;
    uint16_t l1 = (uint16_t)(board2 >> 16);
    uint16_t l2 = (uint16_t)(board2 >> 32);
    uint16_t l3 = (uint16_t)(board2 >> 48);

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= _ud_moves[l0].moveu;
    board ^= _ud_moves[l1].moveu << 4;
    board ^= _ud_moves[l2].moveu << 8;
    board ^= _ud_moves[l3].moveu << 12;

    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_down(uint64_t board, uint64_t board2) {
    uint16_t l0 = (uint16_t)board2;
    uint16_t l1 = (uint16_t)(board2 >> 16);
    uint16_t l2 = (uint16_t)(board2 >> 32);
    uint16_t l3 = (uint16_t)(board2 >> 48);

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= _ud_moves[l0].moved;
    board ^= _ud_moves[l1].moved << 4;
    board ^= _ud_moves[l2].moved << 8;
    board ^= _ud_moves[l3].moved << 12;

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
