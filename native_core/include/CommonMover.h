#pragma once
#include <tuple>
#include <array>
#include <random>
#include <cstdint>



// 场景 1：仅移动，不计分 (4字节)
struct alignas(4) RowMove {
    uint16_t movel; // 左移 XOR 掩码
    uint16_t mover; // 右移 XOR 掩码
};

// 场景 2：移动并计分 (8字节)
struct alignas(8) RowEntry {
    uint16_t movel;
    uint16_t mover;
    uint16_t score;
    uint16_t _pad;
};

struct MoveResult {
    uint64_t board;
    uint32_t score;
    bool is_valid; // 是否是有效移动
};

// ------------------------------------------------------------------
// 1. 基础转换与位运算接口
// ------------------------------------------------------------------

inline uint64_t reverse_board(uint64_t board) {
    board = (board & 0xFF00FF0000FF00FFULL) | ((board & 0x00FF00FF00000000ULL) >> 24) | ((board & 0x00000000FF00FF00ULL) << 24);
    board = (board & 0xF0F00F0FF0F00F0FULL) | ((board & 0x0F0F00000F0F0000ULL) >> 12) | ((board & 0x0000F0F00000F0F0ULL) << 12);
    return board;
}

inline uint16_t encode_row(const uint32_t row[4]) {
    uint16_t encoded = 0;
    for (int i = 0; i < 4; ++i) {
        uint16_t log_v = 0;
        if (row[i] >= 2) {
            uint32_t tmp = row[i];
            while (tmp >>= 1) log_v++;
        }
        encoded |= (log_v & 0xF) << (4 * (3 - i));
    }
    return encoded;
}

inline void decode_row(uint16_t encoded, uint32_t row[4]) {
    for (int i = 0; i < 4; ++i) {
        uint16_t val = (encoded >> (4 * (3 - i))) & 0xF;
        row[i] = (val == 0) ? 0 : (1u << val);
    }
}

inline std::pair<uint64_t, int> gen_new_num(uint64_t t, float p = 0.1) {
    static thread_local std::mt19937 gen(std::random_device{}());
    int empty[16], cnt = 0;
    for (int i = 0; i < 16; ++i) if (((t >> (4 * i)) & 0xF) == 0) empty[cnt++] = i;
    if (cnt == 0) return {t, 0};
    int pos = empty[std::uniform_int_distribution<>(0, cnt - 1)(gen)];
    uint64_t val = (std::uniform_real_distribution<float>(0, 1)(gen) < p) ? 2 : 1;
    return { t | (val << (4 * pos)), cnt };
}

inline std::tuple<uint64_t, int, int, int> s_gen_new_num(uint64_t t, float p = 0.1) {
    static thread_local std::mt19937 gen(std::random_device{}());
    int empty[16], cnt = 0;
    for (int i = 0; i < 16; ++i) if (((t >> (4 * i)) & 0xF) == 0) empty[cnt++] = i;
    if (cnt == 0) return {t, 0, 0, 0};
    int pos = empty[std::uniform_int_distribution<>(0, cnt - 1)(gen)];
    int val = (std::uniform_real_distribution<float>(0, 1)(gen) < p) ? 2 : 1;
    t |= ((uint64_t)val << (4 * pos));
    return { t, cnt, 15 - pos, val };
}

// 定义模板引擎
template <typename MergePolicy>
struct MoverEngine {
    // 静态查找表：编译器会为不同的策略生成完全独立的数组
    inline static RowMove  row_move_table[65536];
    inline static RowEntry row_score_table[65536];

    static inline void init_tables() {
        for (uint32_t i = 0; i < 65536; ++i) {
            auto [l_res, l_score] = MergePolicy::_internal_merge((uint16_t)i, false);
            auto [r_res, r_score] = MergePolicy::_internal_merge((uint16_t)i, true);
            row_move_table[i] = { (uint16_t)(l_res ^ i), (uint16_t)(r_res ^ i) };
            row_score_table[i] = { (uint16_t)(l_res ^ i), (uint16_t)(r_res ^ i), l_score, 0 };
        }
    }

    // 触发初始化
    struct Initializer { Initializer() { init_tables(); } };
    inline static Initializer _init_trigger;

    static inline uint64_t move_left(uint64_t board) {
        board ^= (uint64_t)row_move_table[board & 0xFFFF].movel;
        board ^= (uint64_t)row_move_table[(board >> 16) & 0xFFFF].movel << 16;
        board ^= (uint64_t)row_move_table[(board >> 32) & 0xFFFF].movel << 32;
        board ^= (uint64_t)row_move_table[(board >> 48) & 0xFFFF].movel << 48;
        return board;
    }

    static inline uint64_t move_right(uint64_t board) {
        board ^= (uint64_t)row_move_table[board & 0xFFFF].mover;
        board ^= (uint64_t)row_move_table[(board >> 16) & 0xFFFF].mover << 16;
        board ^= (uint64_t)row_move_table[(board >> 32) & 0xFFFF].mover << 32;
        board ^= (uint64_t)row_move_table[(board >> 48) & 0xFFFF].mover << 48;
        return board;
    }

    static inline uint64_t move_up(uint64_t b) {
        uint64_t trans = reverse_board(b);
        return reverse_board(move_left(trans));
    }

    static inline uint64_t move_down(uint64_t b) {
        uint64_t trans = reverse_board(b);
        return reverse_board(move_right(trans));
    }

    static inline uint64_t move_board(uint64_t board, int direction) {
        switch (direction) {
            case 1: return move_left(board);
            case 2: return move_right(board);
            case 3: return move_up(board);
            case 4: return move_down(board);
            default: return board;
        }
    }

    static inline std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> move_all_dir(uint64_t board) {
        uint64_t trans = reverse_board(board);
        return { move_left(board), move_right(board), reverse_board(move_left(trans)), reverse_board(move_right(trans)) };
    }

    static inline std::pair<uint64_t, uint32_t> s_move_left(uint64_t b) {
        uint64_t s = 0;
        const RowEntry& e0 = row_score_table[b & 0xFFFF];
        const RowEntry& e1 = row_score_table[(b >> 16) & 0xFFFF];
        const RowEntry& e2 = row_score_table[(b >> 32) & 0xFFFF];
        const RowEntry& e3 = row_score_table[(b >> 48) & 0xFFFF];
        s = (uint64_t)e0.score + e1.score + e2.score + e3.score;
        b ^= (uint64_t)e0.movel | ((uint64_t)e1.movel << 16) | ((uint64_t)e2.movel << 32) | ((uint64_t)e3.movel << 48);
        return {b, s};
    }

    static inline std::pair<uint64_t, uint32_t> s_move_right(uint64_t b) {
        uint64_t s = 0;
        const RowEntry& e0 = row_score_table[b & 0xFFFF];
        const RowEntry& e1 = row_score_table[(b >> 16) & 0xFFFF];
        const RowEntry& e2 = row_score_table[(b >> 32) & 0xFFFF];
        const RowEntry& e3 = row_score_table[(b >> 48) & 0xFFFF];
        s = (uint64_t)e0.score + e1.score + e2.score + e3.score;
        b ^= (uint64_t)e0.mover | ((uint64_t)e1.mover << 16) | ((uint64_t)e2.mover << 32) | ((uint64_t)e3.mover << 48);
        return {b, s};
    }

    static inline std::pair<uint64_t, uint32_t> s_move_up(uint64_t b) {
        auto [nb, s] = s_move_left(reverse_board(b));
        return {reverse_board(nb), s};
    }

    static inline std::pair<uint64_t, uint32_t> s_move_down(uint64_t b) {
        auto [nb, s] = s_move_right(reverse_board(b));
        return {reverse_board(nb), s};
    }

    static inline std::pair<uint64_t, uint32_t> s_move_board(uint64_t board, int direction) {
        if (direction == 1) return s_move_left(board);
        if (direction == 2) return s_move_right(board);
        if (direction == 3) return s_move_up(board);
        if (direction == 4) return s_move_down(board);
        return {board, 0};
    }

    static inline std::array<MoveResult, 4> s_move_board_all(uint64_t board) {
        auto [t_l, s_l] = s_move_left(board);
        auto [t_r, s_r] = s_move_right(board);
        auto [t_u, s_u] = s_move_up(board);
        auto [t_d, s_d] = s_move_down(board);

        return {{
            {t_l, s_l, t_l != board},
            {t_r, s_r, t_r != board},
            {t_u, s_u, t_u != board},
            {t_d, s_d, t_d != board}
        }};
    }
};

inline uint64_t canonical_diagonal(uint64_t board) {
    uint64_t board2 = reverse_board(board);
    return board < board2 ? board : board2;
}