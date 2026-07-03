#pragma once

#include "BCKeyRank.h"

#include <array>
#include <cstdint>
#include <stdexcept>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

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

[[nodiscard]] inline uint32_t bc_zero_cell_mask16(uint64_t board) {
    constexpr uint64_t kNibbleLsbMask = 0x1111111111111111ULL;
    const uint64_t nonzero_lsb =
        (board | (board >> 1U) | (board >> 2U) | (board >> 3U)) & kNibbleLsbMask;
    const uint64_t zero_lsb = (~nonzero_lsb) & kNibbleLsbMask;
#if defined(__BMI2__)
    return static_cast<uint32_t>(_pext_u64(zero_lsb, kNibbleLsbMask));
#else
    uint32_t mask = 0U;
    for (uint32_t cell = 0U; cell < kBCBoardCellCount; ++cell) {
        if (((zero_lsb >> (4U * cell)) & 1ULL) != 0ULL) {
            mask |= 1U << cell;
        }
    }
    return mask;
#endif
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

[[nodiscard]] inline uint64_t bc_swapped_nibble_pair_bits(uint16_t word, uint32_t byte_index) {
    const uint32_t byte = (static_cast<uint32_t>(word) >> (byte_index * 8U)) & 0xFFU;
    return static_cast<uint64_t>(((byte & 0x0FU) << 4U) | (byte >> 4U));
}

[[nodiscard]] inline uint64_t pack_nw_quadrant_to_board_bits(uint16_t word) {
    return (bc_swapped_nibble_pair_bits(word, 0U) << 56U) |
           (bc_swapped_nibble_pair_bits(word, 1U) << 40U);
}

[[nodiscard]] inline uint64_t pack_ne_quadrant_to_board_bits(uint16_t word) {
    return (bc_swapped_nibble_pair_bits(word, 0U) << 48U) |
           (bc_swapped_nibble_pair_bits(word, 1U) << 32U);
}

[[nodiscard]] inline uint64_t pack_sw_quadrant_to_board_bits(uint16_t word) {
    return (bc_swapped_nibble_pair_bits(word, 0U) << 24U) |
           (bc_swapped_nibble_pair_bits(word, 1U) << 8U);
}

[[nodiscard]] inline uint64_t pack_se_quadrant_to_board_bits(uint16_t word) {
    return (bc_swapped_nibble_pair_bits(word, 0U) << 16U) |
           bc_swapped_nibble_pair_bits(word, 1U);
}

[[nodiscard]] inline uint64_t pack_quadrants_to_board(const BCQuadrantWords &q) {
    return pack_nw_quadrant_to_board_bits(q.nw) |
           pack_ne_quadrant_to_board_bits(q.ne) |
           pack_sw_quadrant_to_board_bits(q.sw) |
           pack_se_quadrant_to_board_bits(q.se);
}

[[nodiscard]] inline uint16_t bc_quadrant_empty_masks_to_board_mask16(
    uint8_t nw_empty_mask,
    uint8_t ne_empty_mask,
    uint8_t sw_empty_mask,
    uint8_t se_empty_mask
) noexcept {
    return static_cast<uint16_t>(
        (((nw_empty_mask >> 0U) & 1U) << 15U) |
        (((nw_empty_mask >> 1U) & 1U) << 14U) |
        (((nw_empty_mask >> 2U) & 1U) << 11U) |
        (((nw_empty_mask >> 3U) & 1U) << 10U) |
        (((ne_empty_mask >> 0U) & 1U) << 13U) |
        (((ne_empty_mask >> 1U) & 1U) << 12U) |
        (((ne_empty_mask >> 2U) & 1U) << 9U) |
        (((ne_empty_mask >> 3U) & 1U) << 8U) |
        (((sw_empty_mask >> 0U) & 1U) << 7U) |
        (((sw_empty_mask >> 1U) & 1U) << 6U) |
        (((sw_empty_mask >> 2U) & 1U) << 3U) |
        (((sw_empty_mask >> 3U) & 1U) << 2U) |
        (((se_empty_mask >> 0U) & 1U) << 5U) |
        (((se_empty_mask >> 1U) & 1U) << 4U) |
        (((se_empty_mask >> 2U) & 1U) << 1U) |
        (((se_empty_mask >> 3U) & 1U) << 0U)
    );
}

[[nodiscard]] inline uint16_t bc_bucket_empty_mask16(const BCBucketRankDecoder &decoder) noexcept {
    return bc_quadrant_empty_masks_to_board_mask16(
        decoder.nw_empty_mask,
        decoder.ne_empty_mask,
        decoder.sw_empty_mask,
        decoder.se_empty_mask
    );
}

struct BCBucketBoardDecoder {
    BCBucketRankDecoder rank_decoder;
    uint64_t nw_bits = 0U;
    std::array<uint64_t, kBCMaxWordsPerSumMask> ne_bits{};
    std::array<uint64_t, kBCMaxWordsPerSumMask> sw_bits{};
    std::array<uint64_t, kBCMaxWordsPerSumMask> se_bits{};

    BCBucketBoardDecoder() = default;

    BCBucketBoardDecoder(const BCLut &lut, uint64_t bucket_key) {
        reset(lut, bucket_key);
    }

    void reset(const BCLut &lut, uint64_t bucket_key) {
        rank_decoder.reset(lut, bucket_key);
        nw_bits = pack_nw_quadrant_to_board_bits(rank_decoder.nw);
        for (uint32_t i = 0U; i < rank_decoder.count_ne; ++i) {
            ne_bits[i] = pack_ne_quadrant_to_board_bits(rank_decoder.ne_group.words[i]);
        }
        for (uint32_t i = 0U; i < rank_decoder.count_sw; ++i) {
            sw_bits[i] = pack_sw_quadrant_to_board_bits(rank_decoder.sw_group.words[i]);
        }
        for (uint32_t i = 0U; i < rank_decoder.count_se; ++i) {
            se_bits[i] = pack_se_quadrant_to_board_bits(rank_decoder.se_group.words[i]);
        }
    }

    [[nodiscard]] BucketBitmapLen bitmap_len() const {
        return rank_decoder.bitmap_len;
    }

    struct RankParts {
        uint32_t tmp = 0U;
        uint32_t rank_ne = 0U;
        uint32_t rank_sw = 0U;
        uint32_t rank_se = 0U;
    };

    [[nodiscard]] RankParts split_rank_unchecked(uint32_t rank_u32) const {
        const uint32_t tmp = rank_decoder.div_count_se.div(rank_u32);
        const uint32_t rank_se =
            rank_u32 - tmp * static_cast<uint32_t>(rank_decoder.count_se);
        const uint32_t rank_ne = rank_decoder.div_count_sw.div(tmp);
        const uint32_t rank_sw =
            tmp - rank_ne * static_cast<uint32_t>(rank_decoder.count_sw);
        if (rank_ne >= rank_decoder.count_ne ||
            rank_sw >= rank_decoder.count_sw ||
            rank_se >= rank_decoder.count_se) {
            throw std::logic_error("BC bucket board decoder computed quadrant rank outside count");
        }
        return RankParts{tmp, rank_ne, rank_sw, rank_se};
    }

    [[nodiscard]] uint64_t ne_sw_base_bits(uint32_t rank_ne, uint32_t rank_sw) const {
        return nw_bits | ne_bits[rank_ne] | sw_bits[rank_sw];
    }

    [[nodiscard]] uint64_t board_from_base_and_se(uint64_t base_bits, uint32_t rank_se) const {
        return base_bits | se_bits[rank_se];
    }

    [[nodiscard]] uint64_t board(BucketRank rank) const {
        if (rank >= rank_decoder.bitmap_len) {
            throw std::out_of_range("BC bucket board decoder rank is outside bucket bitmap");
        }
        const RankParts parts = split_rank_unchecked(static_cast<uint32_t>(rank));
        return board_from_base_and_se(
            ne_sw_base_bits(parts.rank_ne, parts.rank_sw),
            parts.rank_se
        );
    }

    [[nodiscard]] uint64_t board_with_tmp_cache(
        BucketRank rank,
        uint32_t &last_tmp,
        uint64_t &last_base_bits
    ) const {
        if (rank >= rank_decoder.bitmap_len) {
            throw std::out_of_range("BC bucket board decoder rank is outside bucket bitmap");
        }
        const RankParts parts = split_rank_unchecked(static_cast<uint32_t>(rank));
        if (parts.tmp != last_tmp) {
            last_tmp = parts.tmp;
            last_base_bits = ne_sw_base_bits(parts.rank_ne, parts.rank_sw);
        }
        return board_from_base_and_se(last_base_bits, parts.rank_se);
    }

    [[nodiscard]] uint64_t base_bits_for_tmp(uint32_t tmp) const {
        const uint32_t rank_ne = rank_decoder.div_count_sw.div(tmp);
        const uint32_t rank_sw =
            tmp - rank_ne * static_cast<uint32_t>(rank_decoder.count_sw);
        if (rank_ne >= rank_decoder.count_ne || rank_sw >= rank_decoder.count_sw) {
            throw std::logic_error("BC bucket board decoder computed tmp outside quadrant count");
        }
        return ne_sw_base_bits(rank_ne, rank_sw);
    }
};

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
