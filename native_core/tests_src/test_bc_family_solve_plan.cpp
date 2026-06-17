#include "BCFamilySolve.h"

#include "BCCellBuilder.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyPartialCellLayout;
using BC::BCFamilySolveOptions;
using BC::BCFamilySolvePlanner;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCSolveEdgeOptions;
using BC::BCSolveEdgeWorkspace;
using BC::BCSolveSpawnPhase;
using BC::BCSolveTargetFamilyFilter;
using BC::BCLoadedCell;
using BC::BCLoadedSuccessCell;
using BC::BCSuccessDTypeMode;
using BC::CellId;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

std::vector<BC::LayerSum> dense_possible_sums(uint32_t max_sum) {
    std::vector<BC::LayerSum> out;
    out.reserve(max_sum + 1U);
    for (uint32_t i = 0U; i <= max_sum; ++i) {
        out.push_back(i);
    }
    return out;
}

std::vector<BC::LayerSum> default_possible_sums() {
    return BC::build_possible_8tile_sums(
        test_alphabet(),
        BC::default_2048_tile_sum_values()
    );
}

uint64_t make_board(const std::vector<std::pair<uint32_t, uint8_t>> &tiles) {
    uint64_t board = 0U;
    for (const auto &[cell, tile] : tiles) {
        board = BC::set_board_tile(board, cell, tile);
    }
    return board;
}

uint64_t board_sum(const BCLut &lut, uint64_t board) {
    uint64_t sum = 0U;
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        sum += lut.tile_sum_value(BC::board_tile(board, cell));
    }
    return sum;
}

BCFamilyPartialCellLayout make_single_board_layout(
    const BCLut &lut,
    uint64_t board,
    uint32_t row_width,
    BCLoadedCell &loaded
) {
    const std::vector<BC::LayerSum> possible = default_possible_sums();
    const BCFamilyTable axis =
        BC::build_family_axis_for_layer(board_sum(lut, board), 1U, possible);
    const auto encoded = BC::encode_canonical_board_position(lut, axis, board);
    check(encoded.valid, "test board should encode into BC cell");

    BCCellBuilder builder(lut);
    builder.insert(encoded.key, encoded.rank);
    BC::FinalizedCellPayload payload = builder.finalize();
    loaded.cid = encoded.cid;
    loaded.success_rows = payload.success_rows;
    loaded.buckets = std::move(payload.buckets);
    loaded.rank_payload = std::move(payload.rank_payload);
    return BC::bc_family_make_partial_cell_layout(lut, loaded.view(), row_width);
}

void test_exact_planner_visit_order() {
    const BCFamilyTable current = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCFamilyTable future = BCFamilyTable::from_range(8U, 1U, 0U, 4U);
    const auto current_map = BC::build_family_partition_layer_map_from_axis(current);
    const auto future_map = BC::build_family_partition_layer_map_from_axis(future);
    const BCFamilySolvePlanner planner(current, current_map, future, future_map);

    const auto pass = planner.make_pass(1U, BCSolveSpawnPhase::Spawn4, 2U);
    check(pass.future_families.size() == 2U, "exact solve fanout should be two");
    check(pass.future_families[0] == 1U && pass.future_families[1] == 3U,
        "exact solve fanout ids mismatch");
    check(pass.current_cells.size() == 7U, "source cross size mismatch");

    const BCCellMatrix matrix(current);
    const auto find_work = [&](CellId cid) -> const BC::BCFamilySolveCellWork & {
        const auto it = std::find_if(
            pass.current_cells.begin(),
            pass.current_cells.end(),
            [cid](const BC::BCFamilySolveCellWork &work) {
                return work.cid == cid;
            }
        );
        check(it != pass.current_cells.end(), "expected current work cell missing");
        return *it;
    };

    check(find_work(matrix.cid(1U, 1U)).directions == BC::BCDirectionMask::Both,
        "diagonal work should use both directions");
    check(find_work(matrix.cid(1U, 1U)).visit == BC::BCFamilySolveCellVisitKind::DiagonalBoth,
        "diagonal work visit kind mismatch");
    check(find_work(matrix.cid(1U, 0U)).visit == BC::BCFamilySolveCellVisitKind::SecondDirection,
        "row-left cell should be second visit");
    check(find_work(matrix.cid(0U, 1U)).visit == BC::BCFamilySolveCellVisitKind::SecondDirection,
        "col-upper cell should be second visit");
    check(find_work(matrix.cid(1U, 2U)).visit == BC::BCFamilySolveCellVisitKind::FirstDirection,
        "row-right cell should be first visit");
    check(find_work(matrix.cid(2U, 1U)).visit == BC::BCFamilySolveCellVisitKind::FirstDirection,
        "col-lower cell should be first visit");
}

void test_modulo_planner_three_fanout() {
    const std::vector<BC::LayerSum> possible = dense_possible_sums(18U);
    const BC::BCFamilyPartitionPolicy policy = BC::BCFamilyPartitionPolicy::modulo(5U);
    const BCFamilyTable current =
        BC::build_family_partition_axis_for_layer(14U, 1U, possible, policy);
    const BCFamilyTable future =
        BC::build_family_partition_axis_for_layer(18U, 1U, possible, policy);
    const auto current_map = BC::build_family_partition_layer_map(current, possible, policy);
    const auto future_map = BC::build_family_partition_layer_map(future, possible, policy);
    const BCFamilySolvePlanner planner(current, current_map, future, future_map);

    const auto pass = planner.make_pass(1U, BCSolveSpawnPhase::Spawn4, 4U);
    check(pass.future_families.size() == 3U, "modulo solve fanout should reach three");
    check(pass.future_families[0] == 0U, "modulo fanout first id mismatch");
    check(pass.future_families[1] == 1U, "modulo fanout second id mismatch");
    check(pass.future_families[2] == 3U, "modulo fanout third id mismatch");

    const BCSolveTargetFamilyFilter filter = BC::bc_family_make_solve_filter(pass.future_families);
    check(filter.enabled, "fanout filter should be enabled");
    check(filter.families.size() == 3U, "fanout filter should retain three ids");
}

void test_cached_pass_keep_window_four_families() {
    BCFamilySolveOptions<uint32_t> options;
    check(options.future_reuse_max_families == 4U,
        "family solve default future reuse window should be four families");
    check(options.interleave_block_fids == 1U,
        "family solve default interleaved block size should be one fid");

    const std::vector<BC::LayerSum> possible = dense_possible_sums(18U);
    const BC::BCFamilyPartitionPolicy policy = BC::BCFamilyPartitionPolicy::modulo(5U);
    const BCFamilyTable current =
        BC::build_family_partition_axis_for_layer(14U, 1U, possible, policy);
    const BCFamilyTable future =
        BC::build_family_partition_axis_for_layer(18U, 1U, possible, policy);
    const auto current_map = BC::build_family_partition_layer_map(current, possible, policy);
    const auto future_map = BC::build_family_partition_layer_map(future, possible, policy);
    const BCFamilySolvePlanner planner(current, current_map, future, future_map);

    const auto passes = BC::bc_family_build_passes(
        planner,
        BCSolveSpawnPhase::Spawn4,
        4U,
        current.family_count()
    );
    const auto cached = BC::bc_family_build_cached_passes(
        passes,
        options.future_reuse_max_families
    );
    check(cached.size() == passes.size(), "cached solve pass count mismatch");
    for (size_t i = 0U; i < cached.size(); ++i) {
        check(cached[i].pass.future_families.size() <= 3U,
            "cached solve pass fanout should not exceed three families");
        check(cached[i].keep_families.size() <= 4U,
            "cached solve keep window should not exceed four families");
        for (CellId cid : cached[i].keep_cids) {
            check(cid < BCCellMatrix(future).cell_count(),
                "cached solve keep cell id should be in future matrix");
        }
    }
}

void test_partial_layout_and_buffer() {
    const BCLut lut(test_alphabet());
    const uint64_t board = make_board({{0U, 1U}, {5U, 1U}});
    constexpr uint32_t kRowWidth = 2U;

    BCLoadedCell loaded;
    const BCFamilyPartialCellLayout layout =
        make_single_board_layout(lut, board, kRowWidth, loaded);
    check(layout.success_rows == 1U, "partial fixture should have one success row");
    check(layout.buckets.size() == 1U, "partial fixture should have one bucket");
    const uint32_t empty_count = BC::popcount64(layout.buckets[0].empty_mask);
    check(empty_count == 14U, "partial fixture empty count mismatch");
    check(layout.value_count == static_cast<uint64_t>(empty_count) * kRowWidth,
        "partial layout compact value count mismatch");

    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> best{};
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        for (uint32_t lane = 0U; lane < kRowWidth; ++lane) {
            best[static_cast<size_t>(cell) * kRowWidth + lane] =
                100U + cell * 10U + lane;
        }
    }

    BC::BCFamilyPartialMaxCellBuffer<uint32_t> buffer;
    buffer.reset(layout, 0U);
    buffer.write_success_row(layout, 0U, best.data());

    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> roundtrip{};
    buffer.read_success_row(layout, 0U, roundtrip.data(), 0U);
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        const bool is_empty = (layout.buckets[0].empty_mask & (1U << cell)) != 0U;
        for (uint32_t lane = 0U; lane < kRowWidth; ++lane) {
            const uint32_t got = roundtrip[static_cast<size_t>(cell) * kRowWidth + lane];
            const uint32_t expected =
                is_empty ? best[static_cast<size_t>(cell) * kRowWidth + lane] : 0U;
            check(got == expected, "partial buffer roundtrip mismatch");
        }
    }

    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> higher = best;
    higher[static_cast<size_t>(1U) * kRowWidth + 0U] += 1000U;
    buffer.merge_success_row(layout, 0U, higher.data());
    const uint64_t offset = layout.value_offset_for(0U, 1U, 0U);
    check(buffer.values().at(static_cast<size_t>(offset)) == higher[static_cast<size_t>(1U) * kRowWidth],
        "partial merge should retain max value");
}

void test_success_scratch_weights() {
    constexpr uint32_t kRowWidth = 2U;
    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> best4{};
    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> best2{};
    const uint16_t empty_mask = static_cast<uint16_t>((1U << 1U) | (1U << 3U));
    best4[1U * kRowWidth + 0U] = 100U;
    best4[3U * kRowWidth + 0U] = 300U;
    best4[1U * kRowWidth + 1U] = 200U;
    best4[3U * kRowWidth + 1U] = 400U;
    best2[1U * kRowWidth + 0U] = 1000U;
    best2[3U * kRowWidth + 0U] = 3000U;
    best2[1U * kRowWidth + 1U] = 2000U;
    best2[3U * kRowWidth + 1U] = 4000U;

    BC::BCFamilyCellSuccessScratch<uint32_t> scratch;
    scratch.reset(7U, 1U, kRowWidth, 0U);
    scratch.write_spawn4_contribution(0U, best4.data(), empty_mask, 2U, 0.1, 0U);
    check(scratch.values()[0U] == 20U, "spawn4 lane0 contribution mismatch");
    check(scratch.values()[1U] == 30U, "spawn4 lane1 contribution mismatch");
    scratch.finalize_spawn2_row(0U, best2.data(), empty_mask, 2U, 0.1, 0U);
    check(scratch.values()[0U] == 1820U, "spawn2 lane0 final value mismatch");
    check(scratch.values()[1U] == 2730U, "spawn2 lane1 final value mismatch");
    scratch.write_terminal_row(0U, 42U);
    check(scratch.values()[0U] == 42U && scratch.values()[1U] == 42U,
        "terminal scratch row mismatch");
}

void test_loaded_future_lookup_and_directional_kernel() {
    const BCLut lut(test_alphabet());
    const std::vector<BC::LayerSum> possible = default_possible_sums();
    const uint64_t source_board = make_board({{0U, 1U}});
    const uint64_t future_sum = board_sum(lut, source_board) + lut.tile_sum_value(1U);
    const BCFamilyTable future_axis =
        BC::build_family_axis_for_layer(future_sum, 1U, possible);
    const BCCellMatrix future_matrix(future_axis);

    BCSolveEdgeOptions options;
    options.canonical_batch_size = 4U;
    options.spawn2_tile_rank = 1U;
    options.spawn4_tile_rank = 2U;

    BCSolveEdgeWorkspace<uint32_t> collect_workspace;
    const auto summary = BC::bc_solve_collect_board_phase_queries<uint32_t>(
        lut,
        future_axis,
        source_board,
        BC::BCDirectionMask::Both,
        BCSolveTargetFamilyFilter{},
        BCSolveSpawnPhase::Spawn2,
        collect_workspace,
        options
    );
    check(!summary.terminal_success, "directional fixture should not be terminal");
    check(summary.empty_count > 0U, "directional fixture should have empty slots");
    check(!collect_workspace.queries2.empty(), "directional fixture should emit queries");

    std::vector<std::unique_ptr<BCCellBuilder>> builders(future_matrix.cell_count());
    for (const BC::BCSolvePreparedQuery &query : collect_workspace.queries2) {
        check(query.cid < builders.size(), "query cid out of range");
        if (!builders[query.cid]) {
            builders[query.cid] = std::make_unique<BCCellBuilder>(lut);
        }
        builders[query.cid]->insert(query.key, query.rank);
    }

    constexpr uint32_t kRowWidth = 2U;
    std::vector<BCLoadedCell> position_cells;
    std::vector<BCLoadedSuccessCell> success_cells;
    for (CellId cid = 0U; cid < future_matrix.cell_count(); ++cid) {
        if (!builders[cid]) {
            continue;
        }
        BC::FinalizedCellPayload payload = builders[cid]->finalize();
        BCLoadedCell position;
        position.cid = cid;
        position.success_rows = payload.success_rows;
        position.buckets = std::move(payload.buckets);
        position.rank_payload = std::move(payload.rank_payload);

        BCLoadedSuccessCell success;
        success.cid = cid;
        success.dtype = static_cast<uint32_t>(BCSuccessDTypeMode::UInt32);
        success.row_width = kRowWidth;
        success.success_rows = position.success_rows;
        success.values.reserve(static_cast<size_t>(success.success_rows) * kRowWidth);
        for (uint32_t row = 0U; row < success.success_rows; ++row) {
            for (uint32_t lane = 0U; lane < kRowWidth; ++lane) {
                const uint32_t value =
                    1000U + static_cast<uint32_t>(cid) * 100U + row * 10U + lane;
                success.values.push_back(value);
                BC::bc_append_success_value_le<uint32_t>(success.raw_bytes, value);
            }
        }

        position_cells.push_back(std::move(position));
        success_cells.push_back(std::move(success));
    }

    auto lookup = BC::bc_family_open_loaded_future_lookup<uint32_t>(
        lut,
        future_matrix.cell_count(),
        std::move(position_cells),
        success_cells,
        kRowWidth,
        BCSuccessDTypeMode::UInt32
    );

    std::array<uint32_t, BC::kBCBoardCellCount * kRowWidth> best{};
    BCSolveEdgeWorkspace<uint32_t> kernel_workspace;
    const auto kernel_summary = BC::bc_family_fill_directional_phase_best<uint32_t>(
        lut,
        future_axis,
        source_board,
        BC::BCDirectionMask::Both,
        BCSolveTargetFamilyFilter{},
        BCSolveSpawnPhase::Spawn2,
        lookup,
        kRowWidth,
        0U,
        best.data(),
        kernel_workspace,
        options,
        nullptr,
        nullptr,
        nullptr,
        true
    );
    check(kernel_summary.empty_mask == summary.empty_mask, "kernel empty mask mismatch");
    check(kernel_summary.empty_count == summary.empty_count, "kernel empty count mismatch");

    for (uint32_t lane = 0U; lane < kRowWidth; ++lane) {
        std::array<uint32_t, BC::kBCBoardCellCount> expected{};
        (void)lookup.reduce_max_queries(
            collect_workspace.queries2,
            expected.data(),
            expected.size(),
            lane,
            nullptr,
            true
        );
        uint32_t mask = summary.empty_mask;
        while (mask != 0U) {
            const uint32_t cell = BC::bc_solve_pop_lowest_set_bit_index(mask);
            check(best[static_cast<size_t>(cell) * kRowWidth + lane] == expected[cell],
                "directional kernel best value mismatch");
        }
    }
}

} // namespace

int main() {
    try {
        test_exact_planner_visit_order();
        test_modulo_planner_three_fanout();
        test_cached_pass_keep_window_four_families();
        test_partial_layout_and_buffer();
        test_success_scratch_weights();
        test_loaded_future_lookup_and_directional_kernel();
    } catch (const std::exception &ex) {
        std::cerr << "test_bc_family_solve_plan failed: " << ex.what() << '\n';
        return 1;
    }
    std::cout << "test_bc_family_solve_plan passed\n";
    return 0;
}
