#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCQuadrantWords;
using BC::BucketRank;
using BC::CellId;
using BC::FinalizedCellPayload;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throws(Fn &&fn, const char *message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

void check_empty_cells(
    const BC::BCEmptyCells &cells,
    const std::vector<uint8_t> &expected
) {
    check(cells.count == expected.size(), "empty cell count mismatch");
    for (size_t i = 0; i < expected.size(); ++i) {
        check(cells.cells[i] == expected[i], "empty cell index mismatch");
    }
}

void test_empty_cell_enumeration() {
    check_empty_cells(
        BC::enumerate_empty_cells(0U),
        {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U}
    );

    uint64_t full = 0U;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        full = BC::set_board_tile_unchecked(full, cell, 1U);
    }
    check_empty_cells(BC::enumerate_empty_cells(full), {});

    uint64_t board = 0U;
    board = BC::set_board_tile_unchecked(board, 0U, 1U);   // D1
    board = BC::set_board_tile_unchecked(board, 3U, 2U);   // A1
    board = BC::set_board_tile_unchecked(board, 15U, 3U);  // A4
    check_empty_cells(
        BC::enumerate_empty_cells(board),
        {1U, 2U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U}
    );
}

void test_spawn_tile() {
    uint64_t board = 0U;
    board = BC::spawn_tile(board, 0U, 1U);
    check(BC::board_tile(board, 0U) == 1U, "spawn tile 2 rank mismatch");
    board = BC::spawn_tile(board, 15U, 2U);
    check(BC::board_tile(board, 15U) == 2U, "spawn tile 4 rank mismatch");

    expect_throws(
        [&] {
            (void)BC::spawn_tile(board, 0U, 1U);
        },
        "spawn into non-empty cell should throw"
    );
    expect_throws(
        [] {
            (void)BC::spawn_tile(0U, 16U, 1U);
        },
        "spawn cell index out of range should throw"
    );
    expect_throws(
        [] {
            (void)BC::spawn_tile(0U, 0U, 0U);
        },
        "spawn tile rank 0 should throw"
    );
    expect_throws(
        [] {
            (void)BC::spawn_tile(0U, 0U, 16U);
        },
        "spawn tile rank >15 should throw"
    );
}

void test_encode_canonical_board_position() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(14U, 2U, 1U, 3U);
    const BCCellMatrix matrix(axis);
    const BCQuadrantWords q{
        0x1111U, // sum 8
        0x0000U, // sum 0
        0x0011U, // sum 4
        0x0001U  // sum 2
    };
    const uint64_t board = BC::pack_quadrants_to_board(q);
    const auto encoded = BC::encode_canonical_board_position(lut, axis, board);
    check(encoded.valid, "canonical board should encode");
    check(encoded.row_family == axis.coord_to_id(3U), "row family id mismatch");
    check(encoded.col_family == axis.coord_to_id(1U), "col family id mismatch");
    check(encoded.cid == matrix.cid(encoded.row_family, encoded.col_family), "encoded cid mismatch");

    const BCEncodedKeyRank key_rank = BC::encode_canonical_board_to_key_rank(lut, board);
    check(key_rank.valid, "board codec key/rank should be valid");
    check(encoded.key == key_rank.key, "encoded key mismatch");
    check(encoded.rank == key_rank.rank, "encoded rank mismatch");

    const auto spawned_alias = BC::encode_spawned_canonical_board(lut, axis, board);
    check(spawned_alias.valid, "spawned canonical board alias should encode");
    check(spawned_alias.cid == encoded.cid, "spawned canonical board alias cid mismatch");
    check(spawned_alias.key == encoded.key, "spawned canonical board alias key mismatch");
    check(spawned_alias.rank == encoded.rank, "spawned canonical board alias rank mismatch");

    const BCLut small_lut({0U, 1U, 2U});
    const uint64_t invalid_tile_board = BC::pack_quadrants_to_board(
        BCQuadrantWords{0xF000U, 0U, 0U, 0U}
    );
    check(
        !BC::encode_canonical_board_position(small_lut, BCFamilyTable::from_range(0U, 1U, 0U, 0U), invalid_tile_board).valid,
        "illegal tile board should return invalid"
    );

    const BCFamilyTable indivisible_axis = BCFamilyTable::from_range(6U, 3U, 0U, 1U);
    const uint64_t indivisible_board = BC::pack_quadrants_to_board(
        BCQuadrantWords{0x0001U, 0x0001U, 0U, 0U}
    );
    check(
        !BC::encode_canonical_board_position(lut, indivisible_axis, indivisible_board).valid,
        "side sum not divisible by family_unit should return invalid"
    );

    const BCFamilyTable missing_axis = BCFamilyTable::from_range(14U, 2U, 0U, 0U);
    check(
        !BC::encode_canonical_board_position(lut, missing_axis, board).valid,
        "coord outside axis should return invalid"
    );

    const BCFamilyTable mismatched_sum_axis = BCFamilyTable::from_range(14U, 2U, 0U, 3U);
    const uint64_t mismatched_sum_board = BC::pack_quadrants_to_board(
        BCQuadrantWords{
            0x0011U, // sum 4
            0x0011U, // sum 4
            0x0001U, // sum 2
            0x0001U  // sum 2
        }
    );
    const auto mismatched_sum =
        BC::encode_canonical_board_position(lut, mismatched_sum_axis, mismatched_sum_board);
    check(!mismatched_sum.valid, "layer sum mismatch should return invalid");
}

struct CellFixture {
    CellId cid = 0U;
    FinalizedCellPayload payload;
    std::set<std::pair<uint64_t, BucketRank>> oracle;
};

CellFixture build_cell(
    const BCLut &lut,
    CellId cid,
    const std::vector<uint64_t> &boards
) {
    BCCellBuilder builder(lut);
    CellFixture fixture;
    fixture.cid = cid;
    for (uint64_t board : boards) {
        const BCEncodedKeyRank encoded = BC::encode_canonical_board_to_key_rank(lut, board);
        check(encoded.valid, "scanner fixture board should encode");
        builder.insert(encoded.key, encoded.rank);
        fixture.oracle.insert({encoded.key, encoded.rank});
    }
    fixture.payload = builder.finalize();
    return fixture;
}

std::vector<uint8_t> write_position_layer(
    const BCFamilyTable &axis,
    const std::vector<CellFixture> &fixtures
) {
    const BCCellMatrix matrix(axis);
    std::map<CellId, const CellFixture *> by_cid;
    for (const CellFixture &fixture : fixtures) {
        by_cid[fixture.cid] = &fixture;
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        const auto it = by_cid.find(cid);
        if (it == by_cid.end()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, it->second->payload);
    }
    return writer.finish_layer();
}

void test_scanner_board_roundtrip() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(8U, 1U, 0U, 0U);
    const BCCellMatrix matrix(axis);
    const CellId cid = matrix.cid(0U, 0U);

    const std::vector<uint64_t> boards = {
        BC::pack_quadrants_to_board(BCQuadrantWords{0x1111U, 0U, 0U, 0U}),
        BC::pack_quadrants_to_board(BCQuadrantWords{0U, 0x1111U, 0U, 0U}),
        BC::pack_quadrants_to_board(BCQuadrantWords{0U, 0U, 0x1111U, 0U}),
        BC::pack_quadrants_to_board(BCQuadrantWords{0U, 0U, 0U, 0x1111U}),
    };
    const CellFixture fixture = build_cell(lut, cid, boards);
    const std::vector<uint8_t> bytes = write_position_layer(axis, {fixture});
    const BCPositionLayerReader reader(bytes, lut);

    uint32_t seen = 0U;
    BCPositionCellScanner(reader, cid).for_each_board(
        [&](const BC::BCScannedBoardEntry &entry) {
            const auto encoded = BC::encode_canonical_board_position(lut, reader.axis(), entry.board);
            check(encoded.valid, "scanner board should encode through board ops");
            check(encoded.cid == cid, "scanner board encoded cid mismatch");
            check(encoded.key == entry.key, "scanner board encoded key mismatch");
            check(encoded.rank == entry.rank, "scanner board encoded rank mismatch");
            check(
                fixture.oracle.find({entry.key, entry.rank}) != fixture.oracle.end(),
                "scanner board produced unknown key/rank"
            );
            ++seen;
        }
    );
    check(seen == fixture.oracle.size(), "scanner board roundtrip count mismatch");
}

} // namespace

int main() {
    try {
        std::cerr << "test_empty_cell_enumeration\n";
        test_empty_cell_enumeration();
        std::cerr << "test_spawn_tile\n";
        test_spawn_tile();
        std::cerr << "test_encode_canonical_board_position\n";
        test_encode_canonical_board_position();
        std::cerr << "test_scanner_board_roundtrip\n";
        test_scanner_board_roundtrip();
    } catch (const std::exception &ex) {
        std::cerr << "bc_board_ops_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_board_ops_test passed\n";
    return 0;
}
