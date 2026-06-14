#include "BCBacksolve.h"
#include "BCCellBuilder.h"
#include "BCFutureSuccessLookup.h"
#include "BCPositionScanner.h"
#include "BCResidentSolve.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "SymmetryUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCFutureSuccessLookupView;
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCResidentSolveOptions;
using BC::BCSuccessDTypeMode;
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BucketRank;
using BC::CellId;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void check_value_close(T actual, T expected, const char *message) {
    if constexpr (std::is_floating_point_v<T>) {
        const double diff = std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
        if (diff > 1.0e-5) {
            throw std::runtime_error(message);
        }
    } else {
        if (actual != expected) {
            throw std::runtime_error(message);
        }
    }
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

uint64_t make_board(const std::vector<std::pair<uint8_t, uint8_t>> &tiles) {
    uint64_t board = 0U;
    for (const auto &[cell, tile] : tiles) {
        board = BC::set_board_tile(board, cell, tile);
    }
    return board;
}

uint64_t full_rank1_board() {
    uint64_t board = 0U;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        board = BC::set_board_tile(board, cell, 1U);
    }
    return board;
}

BCFamilyTable full_axis_for_sum(uint32_t layer_sum) {
    return BCFamilyTable::from_range(layer_sum, 2U, 0U, static_cast<BC::FamilyCoord>(layer_sum / 4U));
}

BC::FamilyCoord side_coord_for_sums(uint64_t first, uint64_t second) {
    const uint64_t coord = std::min<uint64_t>(first, second) / 2U;
    if (coord > std::numeric_limits<BC::FamilyCoord>::max()) {
        throw std::overflow_error("test side coord exceeds FamilyCoord");
    }
    return static_cast<BC::FamilyCoord>(coord);
}

std::set<BC::FamilyCoord> coords_for_board(const BCLut &lut, uint64_t board) {
    const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
    const uint64_t nw = lut.word_desc(q.nw).sum;
    const uint64_t ne = lut.word_desc(q.ne).sum;
    const uint64_t sw = lut.word_desc(q.sw).sum;
    const uint64_t se = lut.word_desc(q.se).sum;
    return {
        side_coord_for_sums(nw + ne, sw + se),
        side_coord_for_sums(nw + sw, ne + se),
    };
}

BCFamilyTable compact_axis_for_boards(
    const BCLut &lut,
    uint32_t layer_sum,
    const std::set<uint64_t> &boards
) {
    std::set<BC::FamilyCoord> coords;
    for (uint64_t board : boards) {
        const std::set<BC::FamilyCoord> board_coords = coords_for_board(lut, board);
        coords.insert(board_coords.begin(), board_coords.end());
    }
    if (coords.empty()) {
        return full_axis_for_sum(layer_sum);
    }
    return BCFamilyTable(layer_sum, 2U, std::vector<BC::FamilyCoord>(coords.begin(), coords.end()));
}

struct PositionLayer {
    std::vector<uint8_t> bytes;
    BCPositionLayerReader reader;
    std::vector<uint64_t> stored_boards;
};

PositionLayer write_position_layer(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &boards,
    int symm_mode
) {
    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    PositionLayer layer;
    std::set<uint64_t> unique_boards;
    for (uint64_t board : boards) {
        const uint64_t canonical = canonical_by_mode(board, symm_mode);
        if (!unique_boards.insert(canonical).second) {
            continue;
        }
        const auto encoded = BC::encode_canonical_board_position(lut, axis, canonical);
        check(encoded.valid, "test board should encode into position axis");
        if (!builders[encoded.cid]) {
            builders[encoded.cid] = std::make_unique<BCCellBuilder>(lut);
        }
        builders[encoded.cid]->insert(encoded.key, encoded.rank);
        layer.stored_boards.push_back(canonical);
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (!builders[cid]) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, builders[cid]->finalize());
    }
    layer.bytes = writer.finish_layer();
    layer.reader.open(layer.bytes, lut);
    return layer;
}

template <typename T>
T typed_value(uint64_t index, uint32_t lane, uint32_t base) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return static_cast<uint32_t>(base + index * 17U + lane * 3U);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return static_cast<uint64_t>(base) * 1000000ULL + index * 17000ULL + lane * 3000ULL;
    } else if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(
            static_cast<double>(base) / 10000.0 + static_cast<double>(index) * 0.001 +
            static_cast<double>(lane) * 0.0001
        );
    } else {
        return static_cast<double>(base) / 10000.0 + static_cast<double>(index) * 0.001 +
            static_cast<double>(lane) * 0.0001;
    }
}

template <typename T>
std::map<uint64_t, std::vector<T>> assign_typed_values(
    const std::set<uint64_t> &boards,
    uint32_t row_width,
    uint32_t base
) {
    std::map<uint64_t, std::vector<T>> values;
    uint64_t i = 0U;
    for (uint64_t board : boards) {
        std::vector<T> lanes;
        lanes.reserve(row_width);
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            lanes.push_back(typed_value<T>(i, lane, base));
        }
        values[board] = std::move(lanes);
        ++i;
    }
    return values;
}

template <typename T>
void apply_dtype_storage_semantics(
    std::map<uint64_t, std::vector<T>> &values,
    BCSuccessDTypeMode dtype
) {
    if constexpr (std::is_floating_point_v<T>) {
        if (BC::bc_success_dtype_is_one_minus(dtype)) {
            for (auto &[board, lanes] : values) {
                (void)board;
                for (T &value : lanes) {
                    value = value - static_cast<T>(1);
                }
            }
        }
    }
}

template <typename T>
std::vector<uint8_t> write_typed_success_layer(
    const BCPositionLayerReader &position,
    const std::map<uint64_t, std::vector<T>> &values_by_board,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    std::vector<std::vector<T>> values(position.cell_count());
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        values[static_cast<size_t>(cid)].assign(
            static_cast<size_t>(desc.success_rows) * row_width,
            T{}
        );
        if (desc.success_rows == 0U || desc.empty()) {
            continue;
        }
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                const auto it = values_by_board.find(entry.board);
                if (it == values_by_board.end()) {
                    return;
                }
                for (uint32_t lane = 0U; lane < row_width; ++lane) {
                    values[static_cast<size_t>(cid)][
                        static_cast<size_t>(entry.local_success_row) * row_width + lane
                    ] = it->second[lane];
                }
            }
        );
    }

    BCSuccessLayerWriter writer;
    writer.begin_layer(position, row_width, dtype);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell_typed<T>(cid, values[static_cast<size_t>(cid)]);
    }
    return writer.finish_layer();
}

template <typename T>
std::vector<T> flat_values_from_success(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &success
) {
    std::vector<T> values;
    values.reserve(static_cast<size_t>(BC::bc_success_total_values_for(position, success.row_width())));
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        std::vector<T> cell = success.template read_cell_typed<T>(cid);
        values.insert(values.end(), cell.begin(), cell.end());
    }
    return values;
}

template <typename T>
BC::BCResidentSolvedLayer<T> solved_layer_from_success_bytes(
    const BCLut &lut,
    const PositionLayer &position,
    const std::vector<uint8_t> &success_bytes,
    uint32_t row_width
) {
    const BCSuccessLayerReader success(success_bytes, position.reader, row_width);
    BC::BCResidentSolvedLayer<T> out;
    out.open(
        position.bytes,
        flat_values_from_success<T>(position.reader, success),
        lut,
        row_width,
        success.dtype_mode()
    );
    return out;
}

struct FutureSets {
    std::set<uint64_t> boards2;
    std::set<uint64_t> boards4;
};

bool terminal_success(uint64_t board, const BC::BCSolveEdgeOptions &options) {
    if (options.success_target_rank <= 0 ||
        options.success_shifts == nullptr ||
        options.success_shifts->empty()) {
        return false;
    }
    for (uint8_t shift : *options.success_shifts) {
        if (((board >> shift) & 0xFULL) == static_cast<uint64_t>(options.success_target_rank)) {
            return true;
        }
    }
    return false;
}

FutureSets collect_future_sets(
    const std::vector<uint64_t> &current_boards,
    const BC::BCSolveEdgeOptions &options
) {
    FutureSets out;
    for (uint64_t board : current_boards) {
        if (terminal_success(board, options)) {
            continue;
        }
        uint32_t empty_mask = BC::bc_zero_cell_mask16(board);
        while (empty_mask != 0U) {
#if defined(__GNUC__) || defined(__clang__)
            const uint32_t cell = static_cast<uint32_t>(__builtin_ctz(empty_mask));
#else
            uint32_t cell = 0U;
            while (((empty_mask >> cell) & 1U) == 0U) {
                ++cell;
            }
#endif
            empty_mask &= empty_mask - 1U;
            const uint64_t spawn2 = BC::set_board_tile_unchecked(board, cell, 1U);
            const auto moves2 = BoardMover::move_all_dir(spawn2);
            const uint64_t moved2[4] = {
                std::get<0>(moves2), std::get<1>(moves2),
                std::get<2>(moves2), std::get<3>(moves2)
            };
            for (uint64_t moved : moved2) {
                if (moved != spawn2) {
                    out.boards2.insert(canonical_by_mode(moved, options.canonical_symm_mode));
                }
            }
            const uint64_t spawn4 = BC::set_board_tile_unchecked(board, cell, 2U);
            const auto moves4 = BoardMover::move_all_dir(spawn4);
            const uint64_t moved4[4] = {
                std::get<0>(moves4), std::get<1>(moves4),
                std::get<2>(moves4), std::get<3>(moves4)
            };
            for (uint64_t moved : moved4) {
                if (moved != spawn4) {
                    out.boards4.insert(canonical_by_mode(moved, options.canonical_symm_mode));
                }
            }
        }
    }
    return out;
}

template <typename T>
T oracle_value(
    uint64_t board,
    uint32_t lane,
    const std::map<uint64_t, std::vector<T>> &future2,
    const std::map<uint64_t, std::vector<T>> &future4,
    const BCResidentSolveOptions<T> &options
) {
    if (terminal_success(board, options.edge_options)) {
        return options.terminal_value;
    }
    long double success_probability = 0.0L;
    uint32_t empty_count = 0U;
    uint32_t empty_mask = BC::bc_zero_cell_mask16(board);
    while (empty_mask != 0U) {
#if defined(__GNUC__) || defined(__clang__)
        const uint32_t cell = static_cast<uint32_t>(__builtin_ctz(empty_mask));
#else
        uint32_t cell = 0U;
        while (((empty_mask >> cell) & 1U) == 0U) {
            ++cell;
        }
#endif
        empty_mask &= empty_mask - 1U;
        ++empty_count;

        T best2 = options.zero_value;
        const uint64_t spawn2 = BC::set_board_tile_unchecked(board, cell, 1U);
        const auto moves2 = BoardMover::move_all_dir(spawn2);
        const uint64_t moved2[4] = {
            std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)
        };
        for (uint64_t moved : moved2) {
            if (moved == spawn2) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(moved, options.edge_options.canonical_symm_mode);
            const auto it = future2.find(canonical);
            if (it != future2.end() && it->second[lane] > best2) {
                best2 = it->second[lane];
            }
        }

        T best4 = options.zero_value;
        const uint64_t spawn4 = BC::set_board_tile_unchecked(board, cell, 2U);
        const auto moves4 = BoardMover::move_all_dir(spawn4);
        const uint64_t moved4[4] = {
            std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)
        };
        for (uint64_t moved : moved4) {
            if (moved == spawn4) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(moved, options.edge_options.canonical_symm_mode);
            const auto it = future4.find(canonical);
            if (it != future4.end() && it->second[lane] > best4) {
                best4 = it->second[lane];
            }
        }

        success_probability += static_cast<long double>(best2) *
            (1.0L - static_cast<long double>(options.edge_options.spawn_rate4));
        success_probability += static_cast<long double>(best4) *
            static_cast<long double>(options.edge_options.spawn_rate4);
    }
    return empty_count == 0U
        ? options.zero_value
        : static_cast<T>(success_probability / static_cast<long double>(empty_count));
}

template <typename T>
struct ResidentFixture {
    BCLut lut;
    std::vector<uint8_t> success_shifts;
    PositionLayer current;
    PositionLayer future2;
    PositionLayer future4;
    std::map<uint64_t, std::vector<T>> future2_values;
    std::map<uint64_t, std::vector<T>> future4_values;
    std::vector<uint8_t> future2_success_bytes;
    std::vector<uint8_t> future4_success_bytes;

    ResidentFixture()
        : lut(test_alphabet()) {}
};

template <typename T>
void make_resident_fixture(
    ResidentFixture<T> &fixture,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    int symm_mode
) {
    fixture.success_shifts.clear();
    fixture.future2_values.clear();
    fixture.future4_values.clear();
    fixture.future2_success_bytes.clear();
    fixture.future4_success_bytes.clear();
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        fixture.success_shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }

    BC::BCSolveEdgeOptions edge_options;
    edge_options.canonical_batch_size = 4U;
    edge_options.canonical_symm_mode = symm_mode;
    edge_options.spawn_rate4 = 0.25;
    edge_options.success_target_rank = 3;
    edge_options.success_shifts = &fixture.success_shifts;

    const std::vector<uint64_t> current_input{
        make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
        make_board({{0U, 2U}, {15U, 2U}}),
        make_board({{7U, 3U}}),
    };
    fixture.current = write_position_layer(
        fixture.lut,
        full_axis_for_sum(8U),
        current_input,
        symm_mode
    );

    FutureSets futures = collect_future_sets(fixture.current.stored_boards, edge_options);
    check(!futures.boards2.empty(), "test requires spawn2 future boards");
    futures.boards2.erase(*futures.boards2.begin());

    fixture.future2 = write_position_layer(
        fixture.lut,
        compact_axis_for_boards(fixture.lut, 10U, futures.boards2),
        std::vector<uint64_t>(futures.boards2.begin(), futures.boards2.end()),
        static_cast<int>(SymmMode::Identity)
    );
    fixture.future4 = write_position_layer(
        fixture.lut,
        compact_axis_for_boards(fixture.lut, 12U, futures.boards4),
        std::vector<uint64_t>(futures.boards4.begin(), futures.boards4.end()),
        static_cast<int>(SymmMode::Identity)
    );
    fixture.future2_values = assign_typed_values<T>(futures.boards2, row_width, 1000U);
    fixture.future4_values = assign_typed_values<T>(futures.boards4, row_width, 2000U);
    apply_dtype_storage_semantics<T>(fixture.future2_values, dtype);
    apply_dtype_storage_semantics<T>(fixture.future4_values, dtype);
    fixture.future2_success_bytes = write_typed_success_layer<T>(
        fixture.future2.reader,
        fixture.future2_values,
        row_width,
        dtype
    );
    fixture.future4_success_bytes = write_typed_success_layer<T>(
        fixture.future4.reader,
        fixture.future4_values,
        row_width,
        dtype
    );
}

template <typename T>
BCResidentSolveOptions<T> make_resident_options(
    const std::vector<uint8_t> &success_shifts,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    int symm_mode
) {
    BCResidentSolveOptions<T> options;
    options.num_threads = 2;
    options.row_width = row_width;
    options.set_dtype(dtype);
    options.edge_options.canonical_batch_size = 4U;
    options.edge_options.canonical_symm_mode = symm_mode;
    options.edge_options.spawn_rate4 = 0.25;
    options.edge_options.success_target_rank = 3;
    options.edge_options.success_shifts = &success_shifts;
    return options;
}

template <typename T>
void verify_typed_resident_result(
    const ResidentFixture<T> &fixture,
    const BC::BCResidentSolvedLayer<T> &actual,
    const BCResidentSolveOptions<T> &options
) {
    for (uint64_t board : fixture.current.stored_boards) {
        std::vector<T> expected_lanes;
        expected_lanes.reserve(options.row_width);
        bool keep_expected = false;
        for (uint32_t lane = 0U; lane < options.row_width; ++lane) {
            const T expected = oracle_value<T>(
                board,
                lane,
                fixture.future2_values,
                fixture.future4_values,
                options
            );
            expected_lanes.push_back(expected);
            if (expected != options.zero_value) {
                keep_expected = true;
            }
        }
        for (uint32_t lane = 0U; lane < options.row_width; ++lane) {
            T actual_value{};
            const bool found = actual.lookup.lookup(board, actual_value, lane);
            if (!keep_expected) {
                check(!found, "resident compacted output should prune all-zero row");
                continue;
            }
            check(found, "resident compacted output is missing non-zero row");
            check_value_close<T>(actual_value, expected_lanes[lane], "resident solve output mismatch");
        }
    }
}

template <typename T>
void run_direct_lookup_case(BCSuccessDTypeMode dtype) {
    constexpr uint32_t row_width = 2U;
    ResidentFixture<T> fixture;
    make_resident_fixture<T>(fixture, row_width, dtype, static_cast<int>(SymmMode::Identity));
    const BCSuccessLayerReader future2_success(
        fixture.future2_success_bytes,
        fixture.future2.reader,
        row_width
    );
    BCFutureSuccessLookupView<T> lookup(
        fixture.lut,
        fixture.future2.reader,
        future2_success
    );
    check(lookup.row_width() == row_width, "future lookup row_width mismatch");

    for (const auto &[board, lanes] : fixture.future2_values) {
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            T value{};
            check(lookup.lookup(board, value, lane), "future lookup should find stored board");
            check_value_close<T>(value, lanes[lane], "future lookup stored value mismatch");
        }
        const auto encoded = BC::encode_canonical_board_position(
            fixture.lut,
            fixture.future2.reader.axis(),
            board
        );
        check(encoded.valid, "stored board should encode for query lookup");
        BC::BCSolvePreparedQuery query;
        query.cid = encoded.cid;
        query.key = encoded.key;
        query.rank = encoded.rank;
        query.bitmap_len = encoded.bitmap_len;
        query.spawn_tile_rank = 1U;
        query.valid = true;
        const BC::BCSolveLookupResult<T> query_result = lookup.lookup(query, 1U);
        check(query_result.found, "future query lookup should find stored board");
        check_value_close<T>(query_result.value, lanes[1], "future query value mismatch");
        query.rank = encoded.bitmap_len;
        check(!lookup.lookup(query, 0U).found, "future query bad rank should miss");
        break;
    }

    T miss_value{};
    check(
        !lookup.lookup(full_rank1_board(), miss_value, 0U),
        "future lookup should miss non-stored board"
    );
}

template <typename T>
void run_typed_resident_case(BCSuccessDTypeMode dtype) {
    constexpr uint32_t row_width = 2U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    ResidentFixture<T> fixture;
    make_resident_fixture<T>(fixture, row_width, dtype, symm_mode);
    const BCSuccessLayerReader future2_success(
        fixture.future2_success_bytes,
        fixture.future2.reader,
        row_width
    );
    const BCSuccessLayerReader future4_success(
        fixture.future4_success_bytes,
        fixture.future4.reader,
        row_width
    );
    BCResidentSolveOptions<T> options =
        make_resident_options<T>(fixture.success_shifts, row_width, dtype, symm_mode);
    check(
        options.zero_value == BC::bc_success_zero_value_for_dtype<T>(dtype),
        "resident option zero value should follow dtype"
    );
    check(
        options.terminal_value == BC::bc_success_terminal_value_for_dtype<T>(dtype),
        "resident option terminal value should follow dtype"
    );
    BC::BCResidentSolvedLayer<T> future2_layer =
        solved_layer_from_success_bytes<T>(
            fixture.lut,
            fixture.future2,
            fixture.future2_success_bytes,
            row_width
        );
    BC::BCResidentSolvedLayer<T> future4_layer =
        solved_layer_from_success_bytes<T>(
            fixture.lut,
            fixture.future4,
            fixture.future4_success_bytes,
            row_width
        );
    const auto result = BC::bc_resident_solve_compacted_layer<T>(
        fixture.current.reader,
        future2_layer,
        future4_layer,
        options
    );
    verify_typed_resident_result<T>(fixture, result.layer, options);
    check(result.solve_stats.current_rows == fixture.current.stored_boards.size(), "resident stats current_rows mismatch");
    check(result.solve_stats.terminal_success_rows == 1U, "resident stats terminal_success_rows mismatch");
    check(
        result.solve_stats.queries2 != 0U && result.solve_stats.queries4 != 0U,
        "resident stats should count queries"
    );
    check(
        result.solve_stats.found2 != 0U && result.solve_stats.found4 != 0U,
        "resident stats should count found queries"
    );
}

void run_uint32_matches_backsolve() {
    constexpr uint32_t row_width = 1U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    ResidentFixture<uint32_t> fixture;
    make_resident_fixture<uint32_t>(fixture, row_width, BCSuccessDTypeMode::UInt32, symm_mode);
    const BCSuccessLayerReader future2_success(
        fixture.future2_success_bytes,
        fixture.future2.reader,
        row_width
    );
    const BCSuccessLayerReader future4_success(
        fixture.future4_success_bytes,
        fixture.future4.reader,
        row_width
    );

    BC::BCBacksolveOptions backsolve_options;
    backsolve_options.num_threads = 2;
    backsolve_options.canonical_batch_size = 4U;
    backsolve_options.canonical_symm_mode = symm_mode;
    backsolve_options.spawn_rate4 = 0.25;
    backsolve_options.success_target_rank = 3;
    backsolve_options.success_shifts = &fixture.success_shifts;

    const BC::BCBacksolveResult backsolve = BC::backsolve_resident_layer(
        fixture.lut,
        fixture.current.reader,
        fixture.future2.reader,
        future2_success,
        fixture.future4.reader,
        future4_success,
        backsolve_options
    );

    BCResidentSolveOptions<uint32_t> resident_options =
        make_resident_options<uint32_t>(
            fixture.success_shifts,
            row_width,
            BCSuccessDTypeMode::UInt32,
            symm_mode
        );
    BC::BCResidentSolvedLayer<uint32_t> compact_future2 =
        solved_layer_from_success_bytes<uint32_t>(
            fixture.lut,
            fixture.future2,
            fixture.future2_success_bytes,
            row_width
        );
    BC::BCResidentSolvedLayer<uint32_t> compact_future4 =
        solved_layer_from_success_bytes<uint32_t>(
            fixture.lut,
            fixture.future4,
            fixture.future4_success_bytes,
            row_width
        );
    const auto resident = BC::bc_resident_solve_compacted_layer<uint32_t>(
        fixture.current.reader,
        compact_future2,
        compact_future4,
        resident_options
    );
    const BCSuccessLayerReader backsolve_success(backsolve.success_bytes, fixture.current.reader, row_width);
    for (CellId cid = 0U; cid < fixture.current.reader.cell_count(); ++cid) {
        if (fixture.current.reader.descriptor(cid).success_rows == 0U) {
            continue;
        }
        BCPositionCellScanner(fixture.current.reader, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                const uint32_t expected = backsolve_success.read_value(cid, entry.local_success_row);
                uint32_t actual = 0U;
                const bool found = resident.layer.lookup.lookup(entry.board, actual);
                if (expected == 0U) {
                    check(!found, "ResidentSolve UInt32 compacted zero row should be pruned");
                } else {
                    check(found, "ResidentSolve UInt32 compacted row should be present");
                    check(actual == expected, "ResidentSolve UInt32 must match BCBacksolve value");
                }
            }
        );
    }
    check(
        resident.solve_stats.current_rows == backsolve.stats.current_rows,
        "ResidentSolve UInt32 stats current_rows mismatch"
    );
}

std::vector<uint32_t> flat_uint32_values_for_position(
    const BCPositionLayerReader &position,
    const std::map<uint64_t, std::vector<uint32_t>> &values_by_board
) {
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    std::vector<uint32_t> values(static_cast<size_t>(offsets.back()), 0U);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        if (position.descriptor(cid).success_rows == 0U || position.descriptor(cid).empty()) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                const auto it = values_by_board.find(entry.board);
                if (it != values_by_board.end() && !it->second.empty()) {
                    values[static_cast<size_t>(cell_base + entry.local_success_row)] = it->second[0];
                }
            }
        );
    }
    return values;
}

template <typename T>
BC::BCResidentSolvedLayer<T> make_compacted_typed_layer(
    const BCPositionLayerReader &position,
    const std::map<uint64_t, std::vector<T>> &values_by_board,
    BCSuccessDTypeMode dtype,
    int num_threads = 2
) {
    const uint32_t row_width = values_by_board.empty()
        ? 1U
        : static_cast<uint32_t>(values_by_board.begin()->second.size());
    check(row_width != 0U, "compacted typed layer row_width must be non-zero");
    const T zero = BC::bc_success_zero_value_for_dtype<T>(dtype);
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    std::vector<T> values(static_cast<size_t>(offsets.back() * row_width), zero);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        if (position.descriptor(cid).success_rows == 0U || position.descriptor(cid).empty()) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                const auto it = values_by_board.find(entry.board);
                if (it != values_by_board.end() && !it->second.empty()) {
                    check(it->second.size() == row_width, "typed compact value row_width mismatch");
                    const uint64_t row_base =
                        (cell_base + entry.local_success_row) * static_cast<uint64_t>(row_width);
                    for (uint32_t lane = 0U; lane < row_width; ++lane) {
                        values[static_cast<size_t>(row_base + lane)] = it->second[lane];
                    }
                }
            }
        );
    }
    BC::BCResidentRawSolveResult<T> raw;
    raw.values = std::move(values);
    raw.cell_value_offsets = offsets;
    return BC::bc_resident_compact_zero_in_place<T>(
        position,
        raw,
        position.lut(),
        row_width,
        dtype,
        zero,
        num_threads
    );
}

BC::BCResidentUInt32SolvedLayer make_compacted_uint32_layer(
    const BCPositionLayerReader &position,
    const std::map<uint64_t, std::vector<uint32_t>> &values_by_board,
    int num_threads = 2
) {
    return make_compacted_typed_layer<uint32_t>(
        position,
        values_by_board,
        BCSuccessDTypeMode::UInt32,
        num_threads
    );
}

void test_compact_uint32_layer() {
    ResidentFixture<uint32_t> fixture;
    make_resident_fixture<uint32_t>(
        fixture,
        1U,
        BCSuccessDTypeMode::UInt32,
        static_cast<int>(SymmMode::Identity)
    );
    std::map<uint64_t, std::vector<uint32_t>> values_by_board;
    uint32_t index = 0U;
    for (uint64_t board : fixture.current.stored_boards) {
        values_by_board[board] = {index == 0U ? 0U : (100U + index)};
        ++index;
    }

    BC::BCResidentUInt32SolvedLayer compacted =
        make_compacted_uint32_layer(fixture.current.reader, values_by_board);
    check(
        compacted.compact_stats.input_rows == fixture.current.stored_boards.size(),
        "compact input row count mismatch"
    );
    check(compacted.compact_stats.zero_pruned_rows == 1U, "compact should prune one zero row");
    check(
        compacted.compact_stats.live_rows + compacted.compact_stats.zero_pruned_rows ==
            compacted.compact_stats.input_rows,
        "compact live+zero rows mismatch"
    );

    for (uint64_t board : fixture.current.stored_boards) {
        uint32_t value = 0U;
        const bool found = compacted.lookup.lookup(board, value);
        const uint32_t expected = values_by_board[board][0];
        if (expected == 0U) {
            check(!found, "compact lookup should miss pruned zero row");
        } else {
            check(found, "compact lookup should find non-zero row");
            check(value == expected, "compact lookup value mismatch");
        }
    }

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "bc_resident_compact_success_roundtrip.bcsuc";
    {
        BC::BCBufferedFileWriter writer(path);
        BC::write_success_values_to_file(
            writer,
            compacted.position,
            1U,
            BCSuccessDTypeMode::UInt32,
            compacted.success_values
        );
    }
    const BC::BCSuccessFileReader readback =
        BC::BCSuccessFileReader::open_buffered(path, compacted.position, 1U);
    for (uint64_t board : fixture.current.stored_boards) {
        uint32_t expected = values_by_board[board][0];
        if (expected == 0U) {
            continue;
        }
        const auto encoded =
            BC::encode_canonical_board_position(fixture.lut, compacted.position.axis(), board);
        check(encoded.valid, "compacted board should encode");
        uint32_t direct_value = 0U;
        check(compacted.lookup.lookup(board, direct_value), "compacted lookup should find readback row");
        uint32_t row_value = 0U;
        BCPositionCellScanner(compacted.position, encoded.cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                if (entry.board == board) {
                    row_value = readback.reader().read_value(encoded.cid, entry.local_success_row);
                }
            }
        );
        check(row_value == direct_value && row_value == expected, "success streaming roundtrip mismatch");
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);

    BC::BCResidentUInt32SolvedLayer archive_zero_threshold =
        make_compacted_uint32_layer(fixture.current.reader, values_by_board);
    const std::vector<uint8_t> zero_threshold_position_bytes =
        archive_zero_threshold.position.bytes();
    const std::vector<uint32_t> zero_threshold_success_values =
        archive_zero_threshold.success_values;
    const BC::BCResidentArchivePruneResult zero_threshold_prune =
        BC::bc_resident_archive_prune_if_threshold_enabled_in_place<uint32_t>(
            archive_zero_threshold,
            0U,
            2
        );
    check(!zero_threshold_prune.pruned, "archive threshold 0 should skip prune");
    check(
        zero_threshold_prune.stats.input_rows == 0U &&
            zero_threshold_prune.stats.live_rows == 0U &&
            zero_threshold_prune.stats.zero_pruned_rows == 0U &&
            zero_threshold_prune.stats.compact_seconds == 0.0,
        "archive threshold 0 should produce empty stats"
    );
    check(
        archive_zero_threshold.position.bytes() == zero_threshold_position_bytes,
        "archive threshold 0 should not rewrite position"
    );
    check(
        archive_zero_threshold.success_values == zero_threshold_success_values,
        "archive threshold 0 should not rewrite success values"
    );

    BC::BCResidentUInt32SolvedLayer archive_pruned =
        make_compacted_uint32_layer(fixture.current.reader, values_by_board);
    const BC::BCResidentArchivePruneResult archive_prune =
        BC::bc_resident_archive_prune_if_threshold_enabled_in_place<uint32_t>(
            archive_pruned,
            101U,
            2
        );
    check(archive_prune.pruned, "archive positive threshold should prune");
    const BC::BCResidentCompactStats &archive_stats = archive_prune.stats;
    check(
        archive_stats.input_rows == compacted.compact_stats.live_rows,
        "archive prune should scan zero-compacted rows"
    );
    for (uint64_t board : fixture.current.stored_boards) {
        uint32_t value = 0U;
        const bool found = archive_pruned.lookup.lookup(board, value);
        const uint32_t expected = values_by_board[board][0];
        if (expected <= 101U) {
            check(!found, "archive prune should miss below-threshold row");
        } else {
            check(found, "archive prune should keep above-threshold row");
            check(value == expected, "archive prune kept row value mismatch");
        }
    }
}

void test_optimized_uint32_matches_resident() {
    constexpr uint32_t row_width = 1U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    ResidentFixture<uint32_t> fixture;
    make_resident_fixture<uint32_t>(fixture, row_width, BCSuccessDTypeMode::UInt32, symm_mode);
    BCResidentSolveOptions<uint32_t> options =
        make_resident_options<uint32_t>(
            fixture.success_shifts,
            row_width,
            BCSuccessDTypeMode::UInt32,
            symm_mode
        );

    BC::BCResidentUInt32SolvedLayer compact_future2 =
        make_compacted_uint32_layer(fixture.future2.reader, fixture.future2_values);
    BC::BCResidentUInt32SolvedLayer compact_future4 =
        make_compacted_uint32_layer(fixture.future4.reader, fixture.future4_values);
    BC::BCResidentLayerResult<uint32_t> optimized =
        BC::bc_resident_solve_compacted_layer<uint32_t>(
            fixture.current.reader,
            compact_future2,
            compact_future4,
            options
    );
    check(
        optimized.layer.compact_stats.input_rows == fixture.current.stored_boards.size(),
        "optimized compact input row count mismatch"
    );
    verify_typed_resident_result<uint32_t>(fixture, optimized.layer, options);
}

void test_one_minus_dtype_value_semantics() {
    check(
        BC::bc_success_zero_value_for_dtype<float>(BCSuccessDTypeMode::OneMinusFloat32) == -1.0f,
        "one-minus float32 zero should be -1"
    );
    check(
        BC::bc_success_terminal_value_for_dtype<float>(BCSuccessDTypeMode::OneMinusFloat32) == 0.0f,
        "one-minus float32 terminal should be 0"
    );
    check(
        BC::bc_success_zero_value_for_dtype<double>(BCSuccessDTypeMode::OneMinusFloat64) == -1.0,
        "one-minus float64 zero should be -1"
    );
    check(
        BC::bc_success_terminal_value_for_dtype<double>(BCSuccessDTypeMode::OneMinusFloat64) == 0.0,
        "one-minus float64 terminal should be 0"
    );

    BCResidentSolveOptions<float> options;
    options.set_dtype(BCSuccessDTypeMode::OneMinusFloat32);
    check(options.zero_value == -1.0f, "one-minus resident option zero should be -1");
    check(options.terminal_value == 0.0f, "one-minus resident option terminal should be 0");

    ResidentFixture<float> fixture;
    make_resident_fixture<float>(
        fixture,
        1U,
        BCSuccessDTypeMode::OneMinusFloat32,
        static_cast<int>(SymmMode::Identity)
    );
    std::set<uint64_t> current_boards(
        fixture.current.stored_boards.begin(),
        fixture.current.stored_boards.end()
    );
    std::map<uint64_t, std::vector<float>> values_by_board =
        assign_typed_values<float>(current_boards, 1U, 1000U);
    apply_dtype_storage_semantics<float>(values_by_board, BCSuccessDTypeMode::OneMinusFloat32);

    BC::BCResidentSolvedLayer<float> archive_zero_threshold =
        make_compacted_typed_layer<float>(
            fixture.current.reader,
            values_by_board,
            BCSuccessDTypeMode::OneMinusFloat32
        );
    const std::vector<uint8_t> zero_threshold_position_bytes =
        archive_zero_threshold.position.bytes();
    const std::vector<float> zero_threshold_success_values =
        archive_zero_threshold.success_values;
    const BC::BCResidentArchivePruneResult zero_threshold_prune =
        BC::bc_resident_archive_prune_if_threshold_enabled_in_place<float>(
            archive_zero_threshold,
            -1.0f,
            2
        );
    check(!zero_threshold_prune.pruned, "one-minus archive zero threshold should skip prune");
    check(
        archive_zero_threshold.position.bytes() == zero_threshold_position_bytes,
        "one-minus zero threshold should not rewrite position"
    );
    check(
        archive_zero_threshold.success_values == zero_threshold_success_values,
        "one-minus zero threshold should not rewrite success values"
    );

    BC::BCResidentSolvedLayer<float> archive_pruned =
        make_compacted_typed_layer<float>(
            fixture.current.reader,
            values_by_board,
            BCSuccessDTypeMode::OneMinusFloat32
        );
    const BC::BCResidentArchivePruneResult archive_prune =
        BC::bc_resident_archive_prune_if_threshold_enabled_in_place<float>(
            archive_pruned,
            -0.8995f,
            2
        );
    check(archive_prune.pruned, "one-minus negative archive threshold should prune");
    check(
        archive_prune.stats.input_rows == fixture.current.stored_boards.size(),
        "one-minus archive prune input rows mismatch"
    );
    check(
        archive_prune.stats.zero_pruned_rows != 0U,
        "one-minus archive prune should remove below-threshold rows"
    );
}

void test_no_empty_board_is_zero() {
    const BCLut lut(test_alphabet());
    PositionLayer current = write_position_layer(
        lut,
        full_axis_for_sum(32U),
        {full_rank1_board()},
        static_cast<int>(SymmMode::Identity)
    );
    PositionLayer future2 = write_position_layer(
        lut,
        full_axis_for_sum(34U),
        {},
        static_cast<int>(SymmMode::Identity)
    );
    PositionLayer future4 = write_position_layer(
        lut,
        full_axis_for_sum(36U),
        {},
        static_cast<int>(SymmMode::Identity)
    );
    const std::vector<uint8_t> future2_success_bytes =
        write_typed_success_layer<uint32_t>(future2.reader, {}, 1U, BCSuccessDTypeMode::UInt32);
    const std::vector<uint8_t> future4_success_bytes =
        write_typed_success_layer<uint32_t>(future4.reader, {}, 1U, BCSuccessDTypeMode::UInt32);
    BCResidentSolveOptions<uint32_t> options;
    options.num_threads = 1;
    options.set_dtype(BCSuccessDTypeMode::UInt32);
    options.edge_options.canonical_symm_mode = static_cast<int>(SymmMode::Identity);
    BC::BCResidentSolvedLayer<uint32_t> future2_layer =
        solved_layer_from_success_bytes<uint32_t>(lut, future2, future2_success_bytes, 1U);
    BC::BCResidentSolvedLayer<uint32_t> future4_layer =
        solved_layer_from_success_bytes<uint32_t>(lut, future4, future4_success_bytes, 1U);
    const auto result = BC::bc_resident_solve_compacted_layer<uint32_t>(
        current.reader,
        future2_layer,
        future4_layer,
        options
    );
    uint32_t seen_rows = 0U;
    for (CellId cid = 0U; cid < current.reader.cell_count(); ++cid) {
        if (current.reader.descriptor(cid).success_rows == 0U) {
            continue;
        }
        BCPositionCellScanner(current.reader, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                ++seen_rows;
                check(entry.board == full_rank1_board(), "unexpected board in full-board layer");
                uint32_t value = 1U;
                check(
                    !result.layer.lookup.lookup(entry.board, value),
                    "full board with zero result should be pruned"
                );
            }
        );
    }
    check(seen_rows == 1U, "full-board layer should contain exactly one row");
}

} // namespace

int main() {
    try {
        std::cerr << "direct lookup uint32\n";
        run_direct_lookup_case<uint32_t>(BCSuccessDTypeMode::UInt32);
        std::cerr << "direct lookup uint64\n";
        run_direct_lookup_case<uint64_t>(BCSuccessDTypeMode::UInt64);
        std::cerr << "direct lookup float one-minus\n";
        run_direct_lookup_case<float>(BCSuccessDTypeMode::OneMinusFloat32);
        std::cerr << "direct lookup double one-minus\n";
        run_direct_lookup_case<double>(BCSuccessDTypeMode::OneMinusFloat64);

        std::cerr << "resident uint32 matches backsolve\n";
        run_uint32_matches_backsolve();
        std::cerr << "resident compact uint32\n";
        test_compact_uint32_layer();
        std::cerr << "resident optimized uint32\n";
        test_optimized_uint32_matches_resident();
        std::cerr << "resident uint64\n";
        run_typed_resident_case<uint64_t>(BCSuccessDTypeMode::UInt64);
        std::cerr << "resident float32\n";
        run_typed_resident_case<float>(BCSuccessDTypeMode::Float32);
        std::cerr << "resident float64\n";
        run_typed_resident_case<double>(BCSuccessDTypeMode::Float64);
        std::cerr << "resident one-minus float32\n";
        run_typed_resident_case<float>(BCSuccessDTypeMode::OneMinusFloat32);
        std::cerr << "resident one-minus float64\n";
        run_typed_resident_case<double>(BCSuccessDTypeMode::OneMinusFloat64);
        std::cerr << "resident one-minus dtype semantics\n";
        test_one_minus_dtype_value_semantics();
        std::cerr << "resident no empty board\n";
        test_no_empty_board_is_zero();
    } catch (const std::exception &ex) {
        std::cerr << "bc_resident_solve_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_resident_solve_test passed\n";
    return 0;
}
