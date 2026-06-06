#pragma once

#include "BCKeyRank.h"

#include <cstdint>
#include <stdexcept>

namespace BC {

inline constexpr uint32_t kBCBoardCellCount = 16U;
inline constexpr uint32_t kBCBoardTileBits = 4U;
inline constexpr uint64_t kBCBoardTileMask = 0xFULL;

// Internal board cell indices are low-nibble first. In design-doc coordinates:
// D1=0, C1=1, B1=2, A1=3, ..., D4=12, C4=13, B4=14, A4=15.

[[nodiscard]] inline uint8_t board_tile_unchecked(uint64_t board, uint32_t cell_index) {
    return static_cast<uint8_t>(
        (board >> (cell_index * kBCBoardTileBits)) & kBCBoardTileMask
    );
}

[[nodiscard]] inline uint64_t set_board_tile_unchecked(
    uint64_t board,
    uint32_t cell_index,
    uint8_t tile
) {
    const uint32_t shift = cell_index * kBCBoardTileBits;
    const uint64_t mask = kBCBoardTileMask << shift;
    return (board & ~mask) | (static_cast<uint64_t>(tile & 0xFU) << shift);
}

[[nodiscard]] inline uint8_t board_tile_checked(uint64_t board, uint32_t cell_index) {
    if (cell_index >= kBCBoardCellCount) {
        throw std::out_of_range("BC board tile cell index out of range");
    }
    return board_tile_unchecked(board, cell_index);
}

[[nodiscard]] inline uint64_t set_board_tile_checked(
    uint64_t board,
    uint32_t cell_index,
    uint8_t tile
) {
    if (cell_index >= kBCBoardCellCount) {
        throw std::out_of_range("BC set board tile cell index out of range");
    }
    if ((tile & 0xF0U) != 0U) {
        throw std::out_of_range("BC set board tile value does not fit in a nibble");
    }
    return set_board_tile_unchecked(board, cell_index, tile);
}

[[nodiscard]] inline uint8_t board_tile(uint64_t board, uint32_t cell_index) {
    return board_tile_checked(board, cell_index);
}

[[nodiscard]] inline uint64_t set_board_tile(
    uint64_t board,
    uint32_t cell_index,
    uint8_t tile
) {
    return set_board_tile_checked(board, cell_index, tile);
}

[[nodiscard]] inline BCQuadrantWords unpack_board_to_quadrants(uint64_t b) {
    const uint16_t nw = static_cast<uint16_t>(
        ((b >> 60U) & 0x000FULL) |
        ((b >> 52U) & 0x00F0ULL) |
        ((b >> 36U) & 0x0F00ULL) |
        ((b >> 28U) & 0xF000ULL)
    );
    const uint16_t ne = static_cast<uint16_t>(
        ((b >> 52U) & 0x000FULL) |
        ((b >> 44U) & 0x00F0ULL) |
        ((b >> 28U) & 0x0F00ULL) |
        ((b >> 20U) & 0xF000ULL)
    );
    const uint16_t sw = static_cast<uint16_t>(
        ((b >> 28U) & 0x000FULL) |
        ((b >> 20U) & 0x00F0ULL) |
        ((b >> 4U) & 0x0F00ULL) |
        ((b << 4U) & 0xF000ULL)
    );
    const uint16_t se = static_cast<uint16_t>(
        ((b >> 20U) & 0x000FULL) |
        ((b >> 12U) & 0x00F0ULL) |
        ((b << 4U) & 0x0F00ULL) |
        ((b << 12U) & 0xF000ULL)
    );
    return BCQuadrantWords{nw, ne, sw, se};
}

[[nodiscard]] inline uint64_t pack_quadrants_to_board(const BCQuadrantWords &q) {
    uint64_t b = 0U;

    b |= (static_cast<uint64_t>(q.nw & 0x000FU) << 60U);
    b |= (static_cast<uint64_t>(q.nw & 0x00F0U) << 52U);
    b |= (static_cast<uint64_t>(q.nw & 0x0F00U) << 36U);
    b |= (static_cast<uint64_t>(q.nw & 0xF000U) << 28U);

    b |= (static_cast<uint64_t>(q.ne & 0x000FU) << 52U);
    b |= (static_cast<uint64_t>(q.ne & 0x00F0U) << 44U);
    b |= (static_cast<uint64_t>(q.ne & 0x0F00U) << 28U);
    b |= (static_cast<uint64_t>(q.ne & 0xF000U) << 20U);

    b |= (static_cast<uint64_t>(q.sw & 0x000FU) << 28U);
    b |= (static_cast<uint64_t>(q.sw & 0x00F0U) << 20U);
    b |= (static_cast<uint64_t>(q.sw & 0x0F00U) << 4U);
    b |= (static_cast<uint64_t>(q.sw & 0xF000U) >> 4U);

    b |= (static_cast<uint64_t>(q.se & 0x000FU) << 20U);
    b |= (static_cast<uint64_t>(q.se & 0x00F0U) << 12U);
    b |= (static_cast<uint64_t>(q.se & 0x0F00U) >> 4U);
    b |= (static_cast<uint64_t>(q.se & 0xF000U) >> 12U);

    return b;
}

[[nodiscard]] inline BCEncodedKeyRank encode_canonical_board_to_key_rank(
    const BCLut &lut,
    uint64_t canonical_board
) {
    const BCQuadrantWords q = unpack_board_to_quadrants(canonical_board);
    return encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
}

[[nodiscard]] inline BCEncodedKeyRank encode_board_to_key_rank(
    const BCLut &lut,
    uint64_t canonical_board
) {
    return encode_canonical_board_to_key_rank(lut, canonical_board);
}

} // namespace BC
