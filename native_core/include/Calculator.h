#pragma once
#include <cstdint>
#include <algorithm>
#include <tuple>

namespace Calculator {
    // ------------------------------------------------------------------
    // 1. 基础位运算与检查
    // ------------------------------------------------------------------

    /**
     * 使用 SWAR 技术检查棋盘中是否存在目标值
     * target_stacked 需预先广播，例如目标是 0x9，则为 0x9999999999999999ULL
     */
    static inline bool is_success(uint64_t board_encoded, uint64_t target_stacked, uint64_t mask = 0xffffffffffffffffULL) {
        board_encoded &= mask;
        uint64_t diff = board_encoded ^ target_stacked;
        uint64_t mask_7 = 0x7777777777777777ULL;
        // SWAR 零检测逻辑：如果某 4-bit 为 0，则结果对应位为 1
        uint64_t res = ~(((diff & mask_7) + mask_7) | diff | mask_7);
        return res != 0;
    }

    // ------------------------------------------------------------------
    // 2. 4x4 棋盘对称变换 (Reflection & Rotation)
    // ------------------------------------------------------------------

    static inline uint64_t ReverseLR(uint64_t board) {
        board = ((board & 0xff00ff00ff00ff00ULL) >> 8) | ((board & 0x00ff00ff00ff00ffULL) << 8);
        board = ((board & 0xf0f0f0f0f0f0f0f0ULL) >> 4) | ((board & 0x0f0f0f0f0f0f0f0fULL) << 4);
        return board;
    }

    static inline uint64_t ReverseUD(uint64_t board) {
        board = ((board & 0xffffffff00000000ULL) >> 32) | ((board & 0x00000000ffffffffULL) << 32);
        board = ((board & 0xffff0000ffff0000ULL) >> 16) | ((board & 0x0000ffff0000ffffULL) << 16);
        return board;
    }

    static inline uint64_t ReverseUL(uint64_t board) {
        board = (board & 0xff00ff0000ff00ffULL) | ((board & 0x00ff00ff00000000ULL) >> 24) | ((board & 0x00000000ff00ff00ULL) << 24);
        board = (board & 0xf0f00f0ff0f00f0fULL) | ((board & 0x0f0f00000f0f0000ULL) >> 12) | ((board & 0x0000f0f00000f0f0ULL) << 12);
        return board;
    }

    static inline uint64_t ReverseUR(uint64_t board) {
        board = (board & 0x0f0ff0f00f0ff0f0ULL) | ((board & 0xf0f00000f0f00000ULL) >> 20) | ((board & 0x00000f0f00000f0fULL) << 20);
        board = (board & 0x00ff00ffff00ff00ULL) | ((board & 0xff00ff0000000000ULL) >> 40) | ((board & 0x0000000000ff00ffULL) << 40);
        return board;
    }

    static inline uint64_t Rotate180(uint64_t board) {
        board = ((board & 0xffffffff00000000ULL) >> 32) | ((board & 0x00000000ffffffffULL) << 32);
        board = ((board & 0xffff0000ffff0000ULL) >> 16) | ((board & 0x0000ffff0000ffffULL) << 16);
        board = ((board & 0xff00ff00ff00ff00ULL) >> 8) | ((board & 0x00ff00ff00ff00ffULL) << 8);
        board = ((board & 0xf0f0f0f0f0f0f0f0ULL) >> 4) | ((board & 0x0f0f0f0f0f0f0f0fULL) << 4);
        return board;
    }

    static inline uint64_t RotateL(uint64_t board) {
        board = ((board & 0xff00ff0000000000ULL) >> 32) | ((board & 0x00ff00ff00000000ULL) << 8) |
                ((board & 0x00000000ff00ff00ULL) >> 8)  | ((board & 0x0000000000ff00ffULL) << 32);
        board = ((board & 0xf0f00000f0f00000ULL) >> 16) | ((board & 0x0f0f00000f0f0000ULL) << 4) |
                ((board & 0x0000f0f00000f0f0ULL) >> 4)  | ((board & 0x00000f0f00000f0fULL) << 16);
        return board;
    }

    static inline uint64_t RotateR(uint64_t board) {
        board = ((board & 0xff00ff0000000000ULL) >> 8)  | ((board & 0x00ff00ff00000000ULL) >> 32) |
                ((board & 0x00000000ff00ff00ULL) << 32) | ((board & 0x0000000000ff00ffULL) << 8);
        board = ((board & 0xf0f00000f0f00000ULL) >> 4)  | ((board & 0x0f0f00000f0f0000ULL) >> 16) |
                ((board & 0x0000f0f00000f0f0ULL) << 16) | ((board & 0x00000f0f00000f0fULL) << 4);
        return board;
    }

    // ------------------------------------------------------------------
    // 3. 正则化 (Canonicalization)
    // ------------------------------------------------------------------

    static inline uint64_t canonical_full(uint64_t board) {
        return std::min({ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board),
                         Rotate180(board), RotateL(board), RotateR(board), board});
    }

    static inline uint64_t canonical_diagonal(uint64_t bd) {
        uint64_t board = (bd & 0xff00ff0000ff00ffULL) | ((bd & 0x00ff00ff00000000ULL) >> 24) | ((bd & 0x00000000ff00ff00ULL) << 24);
        board = (board & 0xf0f00f0ff0f00f0fULL) | ((board & 0x0f0f00000f0f0000ULL) >> 12) | ((board & 0x0000f0f00000f0f0ULL) << 12);
        return std::min(bd, board);
    }

    static inline uint64_t canonical_horizontal(uint64_t board) {
        return std::min(ReverseLR(board), board);
    }

    static inline uint64_t canonical_identity(uint64_t encoded_board) {
        return encoded_board;
    }

    // ------------------------------------------------------------------
    // 4. 对称性配对 (Pair functions)
    // ------------------------------------------------------------------

    static inline std::pair<uint64_t, int> canonical_diagonal_pair(uint64_t bd1) {
        uint64_t board = ReverseUL(bd1);
        return (bd1 <= board) ? std::make_pair(bd1, 0) : std::make_pair(board, 3);
    }

    static inline std::pair<uint64_t, int> canonical_horizontal_pair(uint64_t bd1) {
        uint64_t board = ReverseLR(bd1);
        return (bd1 <= board) ? std::make_pair(bd1, 0) : std::make_pair(board, 1);
    }

    static inline std::pair<uint64_t, int> canonical_full_pair(uint64_t bd1) {
        uint64_t min_v = bd1;
        int best_symm = 0;

        uint64_t t;
        if ((t = ReverseLR(bd1)) < min_v) { min_v = t; best_symm = 1; }
        if ((t = ReverseUD(bd1)) < min_v) { min_v = t; best_symm = 2; }
        if ((t = ReverseUL(bd1)) < min_v) { min_v = t; best_symm = 3; }
        if ((t = ReverseUR(bd1)) < min_v) { min_v = t; best_symm = 4; }
        if ((t = Rotate180(bd1)) < min_v) { min_v = t; best_symm = 5; }
        if ((t = RotateL(bd1))   < min_v) { min_v = t; best_symm = 6; }
        if ((t = RotateR(bd1))   < min_v) { min_v = t; best_symm = 7; }

        return {min_v, best_symm};
    }
    static inline std::pair<uint64_t, int> canonical_identity_pair(uint64_t bd1) { return {bd1, 0}; }

    // ------------------------------------------------------------------
    // 5. 子棋盘特定变换 (3x3, 3x4, 2x4)
    // ------------------------------------------------------------------

    static inline uint64_t exchange_row12(uint64_t board) {
        return (board & 0xffff00000000ffffULL) | ((board & 0x00000000ffff0000ULL) << 16) | ((board & 0x0000ffff00000000ULL) >> 16);
    }

    static inline uint64_t exchange_row02(uint64_t board) {
        return (board & 0x0000ffff0000ffffULL) | ((board & 0x00000000ffff0000ULL) << 32) | ((board & 0xffff000000000000ULL) >> 32);
    }

    static inline uint64_t exchange_col02(uint64_t board) {
        return (board & 0x0f0f0f0f0f0f0f0fULL) | ((board & 0xf000f000f000f000ULL) >> 8) | ((board & 0x00f000f000f000f0ULL) << 8);
    }

    static inline uint64_t R90_33(uint64_t board) {
        return ((board & 0xf000000000000000ULL) >> 32) | ((board & 0x0f00000000000000ULL) >> 12) |
               ((board & 0x00f0000000000000ULL) << 8)  | ((board & 0x0000f00000000000ULL) >> 20) |
               ((board & 0x000000f000000000ULL) << 20) | ((board & 0x00000000f0000000ULL) >> 8)  |
               ((board & 0x000000000f000000ULL) << 12) | ((board & 0x0000000000f00000ULL) << 32) |
               (board & 0x000f0f0f000fffffULL);
    }

    static inline uint64_t L90_33(uint64_t board) {
        return ((board & 0xf000000000000000ULL) >> 8)  | ((board & 0x0f00000000000000ULL) >> 20) |
               ((board & 0x00f0000000000000ULL) >> 32) | ((board & 0x0000f00000000000ULL) << 12) |
               ((board & 0x000000f000000000ULL) >> 12) | ((board & 0x00000000f0000000ULL) << 32) |
               ((board & 0x000000000f000000ULL) << 20) | ((board & 0x0000000000f00000ULL) << 8)  |
               (board & 0x000f0f0f000fffffULL);
    }

    static inline uint64_t R180_33(uint64_t board) {
        return ((board & 0xf000000000000000ULL) >> 40) | ((board & 0x0f00000000000000ULL) >> 32) |
               ((board & 0x00f0000000000000ULL) >> 24) | ((board & 0x0000f00000000000ULL) >> 8)  |
               ((board & 0x000000f000000000ULL) << 8)  | ((board & 0x00000000f0000000ULL) << 24) |
               ((board & 0x000000000f000000ULL) << 32) | ((board & 0x0000000000f00000ULL) << 40) |
               (board & 0x000f0f0f000fffffULL);
    }

    static inline uint64_t UL_33(uint64_t board) {
        return ((board & 0x0f0000f000000000ULL) >> 12) | ((board & 0x0000f0000f000000ULL) << 12) |
               ((board & 0x00f0000000000000ULL) >> 24) | ((board & 0x00000000f0000000ULL) << 24) |
               (board & 0xf00f0f0f00ffffffULL);
    }

    static inline uint64_t UR_33(uint64_t board) {
        return ((board & 0x0f00f00000000000ULL) >> 20) | ((board & 0x000000f00f000000ULL) << 20) |
               ((board & 0xf000000000000000ULL) >> 40) | ((board & 0x0000000000f00000ULL) << 40) |
               (board & 0x00ff0f0ff00fffffULL);
    }

    static inline uint64_t Rotate18034(uint64_t board) {
        uint64_t res = Rotate180(board);
        return ((res & 0xffff000000000000ULL) >> 48) | ((res & 0x0000ffffffffffffULL) << 16);
    }

    static inline uint64_t ReverseUD34(uint64_t board) {
        return (board & 0x0000ffff0000ffffULL) | ((board & 0xffff000000000000ULL) >> 32) |
               ((board & 0x00000000ffff0000ULL) << 32);
    }

    static inline uint64_t canonical_min33(uint64_t board) {
        return std::min({exchange_col02(board), exchange_row02(board), R90_33(board), L90_33(board),
                         R180_33(board), UR_33(board), UL_33(board), board});
    }

    static inline uint64_t canonical_min24(uint64_t board) {
        return std::min({ReverseUD(board), ReverseLR(board), Rotate180(board), board});
    }

    static inline uint64_t canonical_min34(uint64_t board) {
        return std::min({ReverseLR(board), ReverseUD34(board), Rotate18034(board), board});
    }
}