#pragma once

#include "BCBoardCodec.h"
#include "BCFamilyTable.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace BC {

struct BCEmptyCells {
    uint8_t count = 0U;
    uint8_t cells[16] = {};
};

struct BCBoardEncodedPosition {
    CellId cid = 0U;
    FamilyId row_family = 0U;
    FamilyId col_family = 0U;
    uint64_t key = 0U;
    BucketRank rank = 0U;
    BucketBitmapLen bitmap_len = 0U;
    uint16_t count_ne = 0U;
    uint16_t count_sw = 0U;
    uint16_t count_se = 0U;
    bool valid = false;
};

[[nodiscard]] inline BCBoardEncodedPosition encode_canonical_quadrants_position(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q
);

[[nodiscard]] inline BCBoardEncodedPosition bc_encode_canonical_quadrants_position_hot(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q
);

[[nodiscard]] inline BCEmptyCells enumerate_empty_cells(uint64_t board) {
    BCEmptyCells out;
    for (uint32_t cell = 0U; cell < kBCBoardCellCount; ++cell) {
        if (board_tile_unchecked(board, cell) == 0U) {
            out.cells[out.count++] = static_cast<uint8_t>(cell);
        }
    }
    return out;
}

[[nodiscard]] inline uint64_t spawn_tile(
    uint64_t board,
    uint8_t cell_index,
    uint8_t tile_rank
) {
    if (cell_index >= kBCBoardCellCount) {
        throw std::out_of_range("BC spawn cell index out of range");
    }
    if (tile_rank == 0U || tile_rank > 15U) {
        throw std::out_of_range("BC spawn tile rank must be in 1..15");
    }
    if (board_tile_unchecked(board, cell_index) != 0U) {
        throw std::invalid_argument("BC spawn target cell is not empty");
    }
    return set_board_tile_unchecked(board, cell_index, tile_rank);
}

[[nodiscard]] inline bool bc_quadrant_sum_value(
    const BCLut &lut,
    uint16_t word,
    uint32_t &sum_out
) {
    const BCWordDesc &desc = lut.word_desc(word);
    if (!desc.valid) {
        return false;
    }
    sum_out = desc.sum;
    return true;
}

[[nodiscard]] inline bool bc_min_side_coord(
    uint32_t first_sum,
    uint32_t second_sum,
    uint16_t family_unit,
    FamilyCoord &coord_out
) {
    if (family_unit == 0U) {
        throw std::logic_error("BC family unit must not be zero");
    }
    const uint32_t min_sum = std::min(first_sum, second_sum);
    if ((min_sum % family_unit) != 0U) {
        return false;
    }
    const uint32_t coord = min_sum / family_unit;
    if (coord > std::numeric_limits<FamilyCoord>::max()) {
        return false;
    }
    coord_out = static_cast<FamilyCoord>(coord);
    return true;
}

[[nodiscard]] inline bool bc_min_side_coord_u64(
    uint64_t first_sum,
    uint64_t second_sum,
    uint16_t family_unit,
    FamilyCoord &coord_out
) {
    if (family_unit == 0U) {
        throw std::logic_error("BC family unit must not be zero");
    }
    const uint64_t min_sum = std::min(first_sum, second_sum);
    if ((min_sum % family_unit) != 0U) {
        return false;
    }
    const uint64_t coord = min_sum / family_unit;
    if (coord > std::numeric_limits<FamilyCoord>::max()) {
        return false;
    }
    coord_out = static_cast<FamilyCoord>(coord);
    return true;
}

[[nodiscard]] inline BCEncodedKeyRank bc_encode_key_rank_from_descs(
    const BCLut &lut,
    uint16_t nw,
    const BCWordDesc &nw_desc,
    const BCWordDesc &ne_desc,
    const BCWordDesc &sw_desc,
    const BCWordDesc &se_desc
) {
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return {};
    }

    const uint16_t count_ne = ne_desc.group_count;
    const uint16_t count_sw = sw_desc.group_count;
    const uint16_t count_se = se_desc.group_count;
    const uint32_t bitmap_len =
        static_cast<uint32_t>(count_ne) *
        static_cast<uint32_t>(count_sw) *
        static_cast<uint32_t>(count_se);
    if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen ||
        bitmap_len > std::numeric_limits<BucketBitmapLen>::max()) {
        throw std::logic_error("BC encoded bitmap length is outside uint16 bounds");
    }

    const uint32_t rank =
        (static_cast<uint32_t>(ne_desc.rank) * count_sw + sw_desc.rank) * count_se + se_desc.rank;
    if (rank >= bitmap_len || rank > std::numeric_limits<BucketRank>::max()) {
        throw std::logic_error("BC encoded rank is outside bitmap length");
    }

    const uint64_t key =
        (static_cast<uint64_t>(nw) << 48U) |
        (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
        (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
        static_cast<uint64_t>(se_desc.packed_sum_mask);

    return BCEncodedKeyRank{
        key,
        static_cast<BucketRank>(rank),
        static_cast<BucketBitmapLen>(bitmap_len),
        count_ne,
        count_sw,
        count_se,
        true
    };
}

[[nodiscard]] inline BCBoardEncodedPosition encode_canonical_board_position(
    const BCLut &lut,
    const BCFamilyTable &axis,
    uint64_t canonical_board
) {
    return encode_canonical_quadrants_position(
        lut,
        axis,
        unpack_board_to_quadrants(canonical_board)
    );
}

[[nodiscard]] inline BCBoardEncodedPosition encode_canonical_quadrants_position(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q
) {
    return bc_encode_canonical_quadrants_position_hot(lut, axis, q);
}

[[nodiscard]] inline BCBoardEncodedPosition bc_encode_canonical_quadrants_position_hot(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q
) {
    BCBoardEncodedPosition out;

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = nw_desc.sum;
    const uint64_t ne_sum = ne_desc.sum;
    const uint64_t sw_sum = sw_desc.sum;
    const uint64_t se_sum = se_desc.sum;

    const uint64_t total_sum =
        nw_sum +
        ne_sum +
        sw_sum +
        se_sum;
    if (total_sum != axis.layer_sum()) {
        return out;
    }

    const uint64_t top_sum = nw_sum + ne_sum;
    const uint64_t bottom_sum = sw_sum + se_sum;
    const uint64_t left_sum = nw_sum + sw_sum;
    const uint64_t right_sum = ne_sum + se_sum;

    const uint16_t family_unit = axis.family_unit();
    auto min_side_coord_fast = [family_unit](uint64_t first_sum, uint64_t second_sum, FamilyCoord &coord_out) {
        const uint64_t min_sum = std::min(first_sum, second_sum);
        uint64_t coord = 0U;
        if (family_unit == 2U) {
            if ((min_sum & 1ULL) != 0ULL) {
                return false;
            }
            coord = min_sum >> 1U;
        } else {
            if (family_unit == 0U || (min_sum % family_unit) != 0U) {
                return false;
            }
            coord = min_sum / family_unit;
        }
        if (coord > std::numeric_limits<FamilyCoord>::max()) {
            return false;
        }
        coord_out = static_cast<FamilyCoord>(coord);
        return true;
    };

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!min_side_coord_fast(top_sum, bottom_sum, row_coord) ||
        !min_side_coord_fast(left_sum, right_sum, col_coord)) {
        return out;
    }

    const uint32_t family_count = axis.family_count();
    const FamilyId row_id = axis.try_coord_to_id(row_coord);
    const FamilyId col_id = axis.try_coord_to_id(col_coord);
    if (row_id == BCFamilyTable::kInvalidFamilyId ||
        col_id == BCFamilyTable::kInvalidFamilyId) {
        return out;
    }

    const BCEncodedKeyRank encoded =
        bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return out;
    }

    out.row_family = row_id;
    out.col_family = col_id;

    const uint64_t cid =
        static_cast<uint64_t>(out.row_family) * family_count +
        static_cast<uint32_t>(out.col_family);
    if (cid > std::numeric_limits<CellId>::max()) {
        throw std::overflow_error("BC encoded cell id exceeds CellId");
    }
    out.cid = static_cast<CellId>(cid);
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

[[nodiscard]] inline BCBoardEncodedPosition encode_spawned_canonical_board(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    uint64_t canonical_board
) {
    return encode_canonical_board_position(lut, target_axis, canonical_board);
}

} // namespace BC
