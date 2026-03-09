#include <cstdint>
#include <tuple>
#include <vector>
#include <random>
#include <iostream>

// ------------------------------------------------------------------
// 全局预计算表
// ------------------------------------------------------------------
uint16_t movel[65536];
uint16_t mover[65536];
uint64_t moveu[65536];
uint64_t moved[65536];
uint32_t score_table[65536];

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
        movel[orig] = left_res.first ^ orig;
        score_table[orig] = left_res.second;

        auto right_res = merge_line(true);
        mover[orig] = right_res.first ^ orig;

        moveu[orig] = reverse_board((uint64_t)movel[orig]);
        moved[orig] = reverse_board((uint64_t)mover[orig]);
    }
}

struct TableInitializer {
    TableInitializer() { init_tables(); }
};
static TableInitializer _initializer;

// ------------------------------------------------------------------
// 带分数的移动函数
// ------------------------------------------------------------------
inline std::tuple<uint64_t, uint32_t> s_move_left(uint64_t board) {
    uint16_t l0 = board & 0xFFFF;
    uint16_t l1 = (board >> 16) & 0xFFFF;
    uint16_t l2 = (board >> 32) & 0xFFFF;
    uint16_t l3 = (board >> 48) & 0xFFFF;

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= (uint64_t)movel[l0];
    board ^= (uint64_t)movel[l1] << 16;
    board ^= (uint64_t)movel[l2] << 32;
    board ^= (uint64_t)movel[l3] << 48;
    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_right(uint64_t board) {
    uint16_t l0 = board & 0xFFFF;
    uint16_t l1 = (board >> 16) & 0xFFFF;
    uint16_t l2 = (board >> 32) & 0xFFFF;
    uint16_t l3 = (board >> 48) & 0xFFFF;

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= (uint64_t)mover[l0];
    board ^= (uint64_t)mover[l1] << 16;
    board ^= (uint64_t)mover[l2] << 32;
    board ^= (uint64_t)mover[l3] << 48;
    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_up(uint64_t board, uint64_t board2) {
    uint16_t l0 = board2 & 0xFFFF;
    uint16_t l1 = (board2 >> 16) & 0xFFFF;
    uint16_t l2 = (board2 >> 32) & 0xFFFF;
    uint16_t l3 = (board2 >> 48) & 0xFFFF;

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= moveu[l0];
    board ^= moveu[l1] << 4;
    board ^= moveu[l2] << 8;
    board ^= moveu[l3] << 12;
    return {board, total_score};
}

inline std::tuple<uint64_t, uint32_t> s_move_down(uint64_t board, uint64_t board2) {
    uint16_t l0 = board2 & 0xFFFF;
    uint16_t l1 = (board2 >> 16) & 0xFFFF;
    uint16_t l2 = (board2 >> 32) & 0xFFFF;
    uint16_t l3 = (board2 >> 48) & 0xFFFF;

    uint32_t total_score = score_table[l0] + score_table[l1] + score_table[l2] + score_table[l3];

    board ^= moved[l0];
    board ^= moved[l1] << 4;
    board ^= moved[l2] << 8;
    board ^= moved[l3] << 12;
    return {board, total_score};
}

std::tuple<uint64_t, uint32_t> s_move_board(uint64_t board, int direction) {
    if (direction == 1) return s_move_left(board);
    else if (direction == 2) return s_move_right(board);
    else if (direction == 3) return s_move_up(board, reverse_board(board));
    else if (direction == 4) return s_move_down(board, reverse_board(board));
    
    std::cout << "bad direction input:" << direction << std::endl;
    return {board, 0};
}
