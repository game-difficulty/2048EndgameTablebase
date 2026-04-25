#pragma once

#include <array>
#include <cstdint>
#include <vector>

using BoardMatrix = std::array<std::array<uint32_t, 4>, 4>;

inline uint32_t encode_tile_value(uint32_t value) {
    if (value == 0U) {
        return 0U;
    }
    uint32_t log_v = 0U;
    while (value >>= 1U) {
        ++log_v;
    }
    return log_v;
}

inline uint64_t encode_board_matrix(const BoardMatrix &board) {
    uint64_t encoded = 0ULL;
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            const uint32_t shift = static_cast<uint32_t>(4U * ((3U - row) * 4U + (3U - col)));
            encoded |= static_cast<uint64_t>(encode_tile_value(board[row][col])) << shift;
        }
    }
    return encoded;
}

inline uint64_t encode_board_vector(const std::vector<std::vector<int>> &board) {
    uint64_t encoded = 0ULL;
    const size_t rows = board.size() < 4U ? board.size() : 4U;
    for (size_t row = 0; row < rows; ++row) {
        const size_t cols = board[row].size() < 4U ? board[row].size() : 4U;
        for (size_t col = 0; col < cols; ++col) {
            const uint32_t shift = static_cast<uint32_t>(4U * ((3U - row) * 4U + (3U - col)));
            encoded |= static_cast<uint64_t>(encode_tile_value(static_cast<uint32_t>(board[row][col]))) << shift;
        }
    }
    return encoded;
}

inline BoardMatrix decode_board_matrix(uint64_t encoded_board) {
    BoardMatrix board{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            const uint32_t shift = static_cast<uint32_t>(4U * ((3U - row) * 4U + (3U - col)));
            const uint32_t encoded_num = static_cast<uint32_t>((encoded_board >> shift) & 0xFULL);
            board[row][col] = (encoded_num == 0U) ? 0U : (1U << encoded_num);
        }
    }
    return board;
}

inline std::vector<std::vector<uint32_t>> decode_board_vector(uint64_t encoded_board) {
    const BoardMatrix board = decode_board_matrix(encoded_board);
    return {
        {board[0][0], board[0][1], board[0][2], board[0][3]},
        {board[1][0], board[1][1], board[1][2], board[1][3]},
        {board[2][0], board[2][1], board[2][2], board[2][3]},
        {board[3][0], board[3][1], board[3][2], board[3][3]},
    };
}

inline uint32_t board_value_sum(const BoardMatrix &board) {
    uint32_t total = 0U;
    for (const auto &row : board) {
        for (uint32_t value : row) {
            total += value;
        }
    }
    return total;
}
