#include "BCCellMatrix.h"
#include "BCPositionCellLayout.h"
#include "BCLut.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using BC::BCCellMatrix;
using BC::BCCellSetBuilder;
using BC::BCFamilyTable;
using BC::CellId;
using BC::FamilyCoord;
using BC::FamilyId;
using BC::FamilyIdList2;
using BC::LayerSum;

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

std::vector<CellId> sorted_unique(std::vector<CellId> cells) {
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

void expect_cells_equal(std::vector<CellId> actual, std::vector<CellId> expected, const char *message) {
    actual = sorted_unique(std::move(actual));
    expected = sorted_unique(std::move(expected));
    check(actual == expected, message);
}

FamilyId coord_id(const BCFamilyTable &axis, FamilyCoord coord) {
    return axis.coord_to_id(coord);
}

void expect_target_coords(
    const BCFamilyTable &source,
    const BCFamilyTable &target,
    FamilyCoord source_coord,
    BC::SpawnDeltaCoord delta,
    const std::vector<FamilyCoord> &expected_coords
) {
    const FamilyId source_id = source.coord_to_id(source_coord);
    const FamilyIdList2 ids =
        BC::map_source_family_to_target_families(source, target, source_id, delta);
    check(ids.size() == expected_coords.size(), "unexpected mapped target family count");
    for (size_t i = 0; i < expected_coords.size(); ++i) {
        check(target.id_to_coord(ids[i]) == expected_coords[i], "unexpected mapped target coord");
    }
}

std::vector<CellId> brute_force_crosses(
    const BCCellMatrix &matrix,
    const std::vector<FamilyId> &families
) {
    std::vector<CellId> out;
    for (uint32_t r = 0; r < matrix.family_count(); ++r) {
        for (uint32_t c = 0; c < matrix.family_count(); ++c) {
            const bool in_row =
                std::find(families.begin(), families.end(), static_cast<FamilyId>(r)) != families.end();
            const bool in_col =
                std::find(families.begin(), families.end(), static_cast<FamilyId>(c)) != families.end();
            if (in_row || in_col) {
                out.push_back(matrix.cid(static_cast<FamilyId>(r), static_cast<FamilyId>(c)));
            }
        }
    }
    return out;
}

std::vector<CellId> brute_force_boundary(
    const BCCellMatrix &matrix,
    FamilyCoord prev_boundary,
    FamilyCoord current_boundary
) {
    std::vector<CellId> out;
    const BCFamilyTable &axis = matrix.axis();
    for (uint32_t r = 0; r < matrix.family_count(); ++r) {
        for (uint32_t c = 0; c < matrix.family_count(); ++c) {
            const FamilyCoord row_coord = axis.id_to_coord(static_cast<FamilyId>(r));
            const FamilyCoord col_coord = axis.id_to_coord(static_cast<FamilyId>(c));
            const FamilyCoord max_coord = std::max(row_coord, col_coord);
            if (prev_boundary < max_coord && max_coord <= current_boundary) {
                out.push_back(matrix.cid(static_cast<FamilyId>(r), static_cast<FamilyId>(c)));
            }
        }
    }
    return out;
}

void test_family_axis_roundtrip() {
    const BCFamilyTable axis = BCFamilyTable::from_range(40U, 2U, 3U, 10U);
    check(axis.layer_sum() == 40U, "wrong layer_sum");
    check(axis.family_unit() == 2U, "wrong family_unit");
    check(axis.total_coord() == 20U, "wrong total_coord");
    check(axis.axis_base_coord() == 3U, "wrong axis_base_coord");
    check(axis.family_count() == 8U, "wrong family_count");
    for (FamilyId id = 0; id < axis.family_count(); ++id) {
        const FamilyCoord coord = axis.id_to_coord(id);
        check(axis.contains_coord(coord), "axis should contain id_to_coord result");
        check(axis.coord_to_id(coord) == id, "coord/id roundtrip failed");
    }

    const BCFamilyTable sparse_axis(40U, 2U, std::vector<FamilyCoord>{0U, 2U, 5U, 9U});
    check(sparse_axis.family_count() == 4U, "sparse axis family count mismatch");
    check(sparse_axis.id_to_coord(0U) == 0U, "sparse axis id 0 mismatch");
    check(sparse_axis.id_to_coord(1U) == 2U, "sparse axis id 1 mismatch");
    check(sparse_axis.id_to_coord(2U) == 5U, "sparse axis id 2 mismatch");
    check(sparse_axis.id_to_coord(3U) == 9U, "sparse axis id 3 mismatch");
    check(!sparse_axis.contains_coord(1U), "sparse axis should not contain gap coord");
    expect_throws(
        [&] { (void)sparse_axis.coord_to_id(1U); },
        "sparse coord_to_id gap should throw"
    );
    expect_throws(
        [] { BCFamilyTable(40U, 2U, std::vector<FamilyCoord>{4U, 3U}); },
        "descending axis should throw"
    );
    expect_throws(
        [] { BCFamilyTable(41U, 2U, std::vector<FamilyCoord>{0U, 1U}); },
        "non-divisible layer sum should throw"
    );
}

void test_theory_axis_builder() {
    std::vector<uint8_t> legal_tiles;
    for (uint8_t tile = 0U; tile <= 10U; ++tile) {
        legal_tiles.push_back(tile);
    }
    legal_tiles.push_back(15U);
    const std::array<uint32_t, 16U> tile_values = BC::default_2048_tile_sum_values();
    check(tile_values[15] == 32768U, "tile 15 must use true sum value");

    const std::vector<LayerSum> possible =
        BC::build_possible_8tile_sums(legal_tiles, tile_values);
    const std::unordered_set<LayerSum> possible_set(possible.begin(), possible.end());

    const LayerSum layer_sum = 32768ULL + 64ULL;
    std::vector<FamilyCoord> expected;
    for (LayerSum h : possible) {
        const LayerSum other = layer_sum >= h ? layer_sum - h : std::numeric_limits<LayerSum>::max();
        if (other == std::numeric_limits<LayerSum>::max() ||
            possible_set.find(other) == possible_set.end()) {
            continue;
        }
        const LayerSum small = std::min(h, other);
        if ((small % 2U) != 0U) {
            continue;
        }
        expected.push_back(static_cast<FamilyCoord>(small / 2U));
    }
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    check(!expected.empty(), "brute force theory axis should not be empty");

    const BCFamilyTable axis = BC::build_family_axis_for_layer(layer_sum, 2U, possible);
    check(axis.coords() == expected, "theory axis coords differ from brute force");
    check(axis.family_count() != layer_sum / 4U + 1U, "theory axis should not equal dense range size");
    check(axis.contains_coord(0U), "theory axis should contain zero small-side coord for rank-15-heavy layer");
    check(axis.total_coord() == layer_sum / 2U, "theory axis total coord mismatch");
}

void test_t_delta_mapping() {
    const BCFamilyTable source11 = BCFamilyTable::from_range(22U, 2U, 0U, 5U);
    const BCFamilyTable target13 = BCFamilyTable::from_range(26U, 2U, 0U, 6U);
    expect_target_coords(source11, target13, 5U, 2U, {5U, 6U});

    const BCFamilyTable source20 = BCFamilyTable::from_range(40U, 2U, 0U, 10U);
    const BCFamilyTable target24 = BCFamilyTable::from_range(48U, 2U, 0U, 12U);
    expect_target_coords(source20, target24, 8U, 4U, {8U, 12U});
    expect_target_coords(source20, target24, 9U, 4U, {9U, 11U});
    expect_target_coords(source20, target24, 10U, 4U, {10U});

    const BCFamilyTable source_base3 = BCFamilyTable::from_range(40U, 2U, 3U, 10U);
    const BCFamilyTable target_base3 = BCFamilyTable::from_range(48U, 2U, 3U, 12U);
    expect_target_coords(source_base3, target_base3, 8U, 4U, {8U, 12U});
    expect_target_coords(source_base3, target_base3, 9U, 4U, {9U, 11U});
    expect_target_coords(source_base3, target_base3, 10U, 4U, {10U});

    const BCFamilyTable wrong_total_target = BCFamilyTable::from_range(44U, 2U, 0U, 11U);
    expect_throws(
        [&] {
            (void)BC::map_source_family_to_target_families(
                source20,
                wrong_total_target,
                source20.coord_to_id(8U),
                4U
            );
        },
        "target total coord mismatch should throw"
    );

    const BCFamilyTable wrong_unit_target = BCFamilyTable::from_range(96U, 4U, 0U, 12U);
    expect_throws(
        [&] {
            (void)BC::map_source_family_to_target_families(
                source20,
                wrong_unit_target,
                source20.coord_to_id(8U),
                4U
            );
        },
        "family unit mismatch should throw"
    );

    const BCFamilyTable too_short_target = BCFamilyTable::from_range(26U, 2U, 0U, 5U);
    expect_throws(
        [&] {
            (void)BC::map_source_family_to_target_families(
                source11,
                too_short_target,
                source11.coord_to_id(5U),
                2U
            );
        },
        "missing normalized target coord should throw"
    );

    const BCFamilyTable sparse_missing_target(
        48U,
        2U,
        std::vector<FamilyCoord>{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 12U}
    );
    expect_throws(
        [&] {
            (void)BC::map_source_family_to_target_families(
                source20,
                sparse_missing_target,
                source20.coord_to_id(9U),
                4U
            );
        },
        "sparse target missing normalized coord should throw"
    );
}

void test_cell_matrix_roundtrip() {
    const BCFamilyTable axis = BCFamilyTable::from_range(12U, 2U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    check(matrix.family_count() == 4U, "wrong matrix family count");
    check(matrix.cell_count() == 16U, "wrong matrix cell count");
    for (uint32_t r = 0; r < matrix.family_count(); ++r) {
        for (uint32_t c = 0; c < matrix.family_count(); ++c) {
            const CellId id = matrix.cid(static_cast<FamilyId>(r), static_cast<FamilyId>(c));
            check(matrix.row(id) == r, "matrix row roundtrip failed");
            check(matrix.col(id) == c, "matrix col roundtrip failed");
        }
    }
}

void test_collect_family_crosses() {
    const BCFamilyTable axis = BCFamilyTable::from_range(12U, 2U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    BCCellSetBuilder builder(matrix);

    builder.begin_epoch();
    builder.add_family_cross(2U);
    check(builder.cells().size() == 2U * matrix.family_count() - 1U, "single cross size should be 2F-1");
    expect_cells_equal(builder.cells(), brute_force_crosses(matrix, {2U}), "single cross differs from brute force");

    builder.begin_epoch();
    builder.add_family_crosses(std::vector<FamilyId>{1U, 2U});
    expect_cells_equal(
        builder.cells(),
        brute_force_crosses(matrix, {1U, 2U}),
        "two-family crosses differ from brute force"
    );
    check(
        builder.cells().size() == sorted_unique(builder.cells()).size(),
        "family crosses should be deduplicated within epoch"
    );

    const BCFamilyTable large_axis = BCFamilyTable::from_range(12648U, 2U, 0U, 3162U);
    const BCCellMatrix large_matrix(large_axis);
    expect_throws(
        [&] {
            BCCellSetBuilder too_large(large_matrix);
            (void)too_large;
        },
        "large dense cell set builder should trip practical guard"
    );
}

void test_boundary_cells() {
    const BCFamilyTable axis = BCFamilyTable::from_range(24U, 2U, 0U, 6U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellId> actual = BC::collect_boundary_cells(matrix, 3U, 5U);
    const std::vector<CellId> expected = brute_force_boundary(matrix, 3U, 5U);
    expect_cells_equal(actual, expected, "boundary cells differ from brute force");
    check(actual.size() == sorted_unique(actual).size(), "boundary cells should not contain duplicates");

    const CellId diag4 = matrix.cid(coord_id(axis, 4U), coord_id(axis, 4U));
    const CellId diag5 = matrix.cid(coord_id(axis, 5U), coord_id(axis, 5U));
    const std::vector<CellId> unique_actual = sorted_unique(actual);
    check(std::binary_search(unique_actual.begin(), unique_actual.end(), diag4), "boundary missed diagonal 4");
    check(std::binary_search(unique_actual.begin(), unique_actual.end(), diag5), "boundary missed diagonal 5");

    const BCFamilyTable nonzero_axis = BCFamilyTable::from_range(32U, 2U, 3U, 8U);
    const BCCellMatrix nonzero_matrix(nonzero_axis);
    const std::vector<CellId> nonzero_actual =
        BC::collect_boundary_cells(nonzero_matrix, 4U, 6U);
    const std::vector<CellId> nonzero_expected =
        brute_force_boundary(nonzero_matrix, 4U, 6U);
    expect_cells_equal(
        nonzero_actual,
        nonzero_expected,
        "nonzero-axis boundary cells differ from brute force"
    );
    check(
        nonzero_actual.size() == sorted_unique(nonzero_actual).size(),
        "nonzero-axis boundary cells should not contain duplicates"
    );
}

void test_modulo_position_cell_layout() {
    const uint32_t modulus = 13U;
    const BC::BCPositionCellLayout layout =
        BC::build_modulo_position_cell_layout_for_layer(
            100U,
            2U,
            std::vector<LayerSum>{0U},
            modulus
        );

    check(layout.uses_modulo_cells(), "modulo layout should report modulo cells");
    check(layout.side_coord_count() == modulus, "modulo physical side count mismatch");
    check(layout.side_index_to_coord(0U) == 0U, "modulo physical coord 0 mismatch");
    check(layout.side_index_to_coord(12U) == 12U, "modulo physical coord 12 mismatch");
    check(layout.try_raw_side_coord_to_physical_index(0U) == 0U, "raw coord 0 should map to id 0");
    check(layout.try_raw_side_coord_to_physical_index(12U) == 12U, "raw coord 12 should map to id 12");
    check(layout.try_raw_side_coord_to_physical_index(13U) == 0U, "raw coord 13 should map to id 0");
    check(layout.try_raw_side_coord_to_physical_index(25U) == 12U, "raw coord 25 should map to id 12");
    check(
        layout.try_raw_side_coord_to_physical_index(26U) ==
            BC::BCPositionCellLayout::kInvalidSideIndex,
        "raw coord beyond min-side range should be invalid"
    );
    check(layout.cell_id(12U, 1U) == 157U, "modulo cell id mismatch");
}

} // namespace

int main() {
    try {
        std::cerr << "test_family_axis_roundtrip\n";
        test_family_axis_roundtrip();
        std::cerr << "test_theory_axis_builder\n";
        test_theory_axis_builder();
        std::cerr << "test_t_delta_mapping\n";
        test_t_delta_mapping();
        std::cerr << "test_cell_matrix_roundtrip\n";
        test_cell_matrix_roundtrip();
        std::cerr << "test_collect_family_crosses\n";
        test_collect_family_crosses();
        std::cerr << "test_boundary_cells\n";
        test_boundary_cells();
        std::cerr << "test_modulo_position_cell_layout\n";
        test_modulo_position_cell_layout();
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_family_test passed\n";
    return 0;
}
