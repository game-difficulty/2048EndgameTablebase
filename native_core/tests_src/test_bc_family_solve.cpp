#include "BCFamilySolve.h"

#include "BCCellBuilder.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "SymmetryUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilySolveOptions;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCPositionStreamingReader;
using BC::BCSingleChunkSolveOptions;
using BC::BCSuccessDTypeMode;
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BCSuccessStreamingReader;
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
        const uint64_t actual_u = static_cast<uint64_t>(actual);
        const uint64_t expected_u = static_cast<uint64_t>(expected);
        const uint64_t diff = actual_u > expected_u
            ? actual_u - expected_u
            : expected_u - actual_u;
        if (diff > 2U) {
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

BCFamilyTable full_axis_for_sum(uint32_t layer_sum) {
    return BCFamilyTable::from_range(layer_sum, 2U, 0U, static_cast<BC::FamilyCoord>(layer_sum / 4U));
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

uint64_t board_sum(const BCLut &lut, uint64_t board) {
    uint64_t sum = 0U;
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        sum += lut.tile_sum_value(BC::board_tile(board, cell));
    }
    return sum;
}

PositionLayer write_modulo_position_layer(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<BC::LayerSum> &possible_sums,
    uint32_t modulus,
    const std::vector<uint64_t> &boards,
    int symm_mode
) {
    const BCCellMatrix matrix(axis);
    const BCFamilyTable exact_axis =
        BC::build_family_axis_for_layer(axis.layer_sum(), axis.family_unit(), possible_sums);
    const BCCellMatrix exact_matrix(exact_axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    PositionLayer layer;
    std::set<uint64_t> unique_boards;
    for (uint64_t board : boards) {
        check(board_sum(lut, board) == axis.layer_sum(), "modulo test board layer sum mismatch");
        const uint64_t canonical = canonical_by_mode(board, symm_mode);
        if (!unique_boards.insert(canonical).second) {
            continue;
        }
        const auto exact = BC::encode_canonical_board_position(lut, exact_axis, canonical);
        check(exact.valid, "test board should encode into exact position axis");
        const BC::FamilyId exact_row = exact_matrix.row(exact.cid);
        const BC::FamilyId exact_col = exact_matrix.col(exact.cid);
        const BC::FamilyId row = static_cast<BC::FamilyId>(
            exact_axis.id_to_coord(exact_row) % modulus
        );
        const BC::FamilyId col = static_cast<BC::FamilyId>(
            exact_axis.id_to_coord(exact_col) % modulus
        );
        const CellId cid = matrix.cid(row, col);
        if (!builders[cid]) {
            builders[cid] = std::make_unique<BCCellBuilder>(lut);
        }
        builders[cid]->insert(exact.key, exact.rank);
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

struct FutureSets {
    std::set<uint64_t> boards2;
    std::set<uint64_t> boards4;
};

FutureSets collect_future_sets(
    const std::vector<uint64_t> &current_boards,
    const BC::BCSolveEdgeOptions &options
) {
    FutureSets out;
    for (uint64_t board : current_boards) {
        if (BC::bc_solve_is_success_board(board, options)) {
            continue;
        }
        uint32_t empty_mask = BC::bc_zero_cell_mask16(board);
        while (empty_mask != 0U) {
            const uint32_t cell = BC::bc_solve_pop_lowest_set_bit_index(empty_mask);
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

void write_bytes_file(const std::filesystem::path &path, const std::vector<uint8_t> &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open test output file: " + path.string());
    }
    if (!bytes.empty()) {
        out.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    if (!out) {
        throw std::runtime_error("failed to write test output file: " + path.string());
    }
}

std::filesystem::path temp_root(const char *name) {
    static uint32_t counter = 0U;
    return std::filesystem::temp_directory_path() /
        ("bc_family_solve_" + std::to_string(++counter) + "_" + name);
}

template <typename T>
void compare_success_on_current_boards(
    const PositionLayer &current,
    const BCSuccessStreamingReader &expected_success,
    const BCSuccessStreamingReader &actual_success,
    const BCLut &lut,
    uint32_t row_width
) {
    (void)lut;
    for (CellId cid = 0U; cid < current.reader.cell_count(); ++cid) {
        const auto &desc = current.reader.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        for (uint32_t row = 0U; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0U; lane < row_width; ++lane) {
                const T expected = expected_success.template read_value_typed<T>(cid, row, lane);
                const T actual = actual_success.template read_value_typed<T>(cid, row, lane);
                check_value_close<T>(actual, expected, "family solve output value mismatch");
            }
        }
    }
}

template <typename T>
void compare_success_layers(
    const BCPositionStreamingReader &expected_position,
    const BCSuccessStreamingReader &expected_success,
    const BCPositionStreamingReader &actual_position,
    const BCSuccessStreamingReader &actual_success,
    uint32_t row_width
) {
    check(expected_position.cell_count() == actual_position.cell_count(),
        "output position cell count mismatch");
    for (CellId cid = 0U; cid < expected_position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &expected_desc = expected_position.descriptor(cid);
        const BC::BCPositionCellDescriptor &actual_desc = actual_position.descriptor(cid);
        check(expected_desc.success_rows == actual_desc.success_rows,
            "output success row count mismatch");
        check(expected_desc.empty() == actual_desc.empty(), "output empty descriptor mismatch");
        for (uint32_t row = 0U; row < expected_desc.success_rows; ++row) {
            for (uint32_t lane = 0U; lane < row_width; ++lane) {
                const T expected = expected_success.template read_value_typed<T>(cid, row, lane);
                const T actual = actual_success.template read_value_typed<T>(cid, row, lane);
                check_value_close<T>(actual, expected, "output success value mismatch");
            }
        }
    }
}

template <typename T>
std::map<uint64_t, std::vector<T>> collect_success_by_board(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &success,
    uint32_t row_width
) {
    check(success.row_width() == row_width, "logical success row_width mismatch");
    std::map<uint64_t, std::vector<T>> rows;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                std::vector<T> lanes;
                lanes.reserve(row_width);
                for (uint32_t lane = 0U; lane < row_width; ++lane) {
                    lanes.push_back(
                        success.template read_value_typed<T>(
                            cid,
                            entry.local_success_row,
                            lane
                        )
                    );
                }
                const auto inserted = rows.emplace(entry.board, std::move(lanes));
                check(inserted.second, "duplicate board in logical success output");
            }
        );
    }
    return rows;
}

template <typename T>
void compare_output_layers_logically(
    const BCPositionLayerReader &expected_position,
    const BCSuccessLayerReader &expected_success,
    const BCPositionLayerReader &actual_position,
    const BCSuccessLayerReader &actual_success,
    uint32_t row_width
) {
    check(expected_position.cell_count() == actual_position.cell_count(),
        "logical output position cell count mismatch");
    check(expected_success.dtype_mode() == actual_success.dtype_mode(),
        "logical output dtype mismatch");
    check(expected_success.row_width() == actual_success.row_width(),
        "logical output row_width mismatch");
    for (CellId cid = 0U; cid < expected_position.cell_count(); ++cid) {
        const auto &expected_desc = expected_position.descriptor(cid);
        const auto &actual_desc = actual_position.descriptor(cid);
        check(expected_desc.empty() == actual_desc.empty(),
            "logical output empty descriptor mismatch");
        check(expected_desc.success_rows == actual_desc.success_rows,
            "logical output success row count mismatch");
        check(expected_desc.bucket_count == actual_desc.bucket_count,
            "logical output bucket count mismatch");
    }

    const auto expected = collect_success_by_board<T>(
        expected_position,
        expected_success,
        row_width
    );
    const auto actual = collect_success_by_board<T>(
        actual_position,
        actual_success,
        row_width
    );
    check(expected.size() == actual.size(), "logical output board count mismatch");
    for (const auto &[board, expected_lanes] : expected) {
        const auto actual_it = actual.find(board);
        check(actual_it != actual.end(), "logical output missing board");
        check(expected_lanes.size() == actual_it->second.size(),
            "logical output lane count mismatch");
        for (size_t lane = 0U; lane < expected_lanes.size(); ++lane) {
            check_value_close<T>(
                actual_it->second[lane],
                expected_lanes[lane],
                "logical output success value mismatch"
            );
        }
    }
}

template <typename T>
T direct_oracle_board_value(
    uint64_t board,
    uint32_t lane,
    const std::map<uint64_t, std::vector<T>> &future2_values,
    const std::map<uint64_t, std::vector<T>> &future4_values,
    const BC::BCResidentSolveOptions<T> &options
) {
    if (BC::bc_solve_is_success_board(board, options.edge_options)) {
        return options.terminal_value;
    }

    uint32_t empty_count = 0U;
    long double sum2 = 0.0L;
    long double sum4 = 0.0L;
    uint32_t empty_mask = BC::bc_zero_cell_mask16(board);
    while (empty_mask != 0U) {
        const uint32_t cell = BC::bc_solve_pop_lowest_set_bit_index(empty_mask);
        ++empty_count;

        T best2 = options.zero_value;
        const uint64_t spawn2 = BC::set_board_tile_unchecked(
            board,
            cell,
            options.edge_options.spawn2_tile_rank
        );
        const auto moves2 = BoardMover::move_all_dir(spawn2);
        const uint64_t moved2[4] = {
            std::get<0>(moves2), std::get<1>(moves2),
            std::get<2>(moves2), std::get<3>(moves2)
        };
        for (uint64_t moved : moved2) {
            if (moved == spawn2) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(
                moved,
                options.edge_options.canonical_symm_mode
            );
            const auto it = future2_values.find(canonical);
            if (it != future2_values.end() && it->second[lane] > best2) {
                best2 = it->second[lane];
            }
        }

        T best4 = options.zero_value;
        const uint64_t spawn4 = BC::set_board_tile_unchecked(
            board,
            cell,
            options.edge_options.spawn4_tile_rank
        );
        const auto moves4 = BoardMover::move_all_dir(spawn4);
        const uint64_t moved4[4] = {
            std::get<0>(moves4), std::get<1>(moves4),
            std::get<2>(moves4), std::get<3>(moves4)
        };
        for (uint64_t moved : moved4) {
            if (moved == spawn4) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(
                moved,
                options.edge_options.canonical_symm_mode
            );
            const auto it = future4_values.find(canonical);
            if (it != future4_values.end() && it->second[lane] > best4) {
                best4 = it->second[lane];
            }
        }

        sum2 += static_cast<long double>(best2);
        sum4 += static_cast<long double>(best4);
    }

    if (empty_count == 0U) {
        return options.zero_value;
    }
    const T spawn4_contribution = BC::bc_single_chunk_scale_average_contribution<T>(
        sum4,
        empty_count,
        options.edge_options.spawn_rate4,
        true,
        options.zero_value
    );
    const T spawn2_contribution = BC::bc_single_chunk_scale_average_contribution<T>(
        sum2,
        empty_count,
        options.edge_options.spawn_rate4,
        false,
        options.zero_value
    );
    T value = spawn4_contribution;
    if (spawn2_contribution != options.zero_value) {
        value = BC::bc_single_chunk_add_success_contribution<T>(
            value,
            spawn2_contribution
        );
    }
    return value;
}

template <typename T>
void verify_sampled_board_success_rates(
    const PositionLayer &current,
    const BCPositionLayerReader &actual_position,
    const BCSuccessLayerReader &actual_success,
    const std::map<uint64_t, std::vector<T>> &future2_values,
    const std::map<uint64_t, std::vector<T>> &future4_values,
    const BC::BCResidentSolveOptions<T> &options,
    uint32_t row_width
) {
    const auto actual_by_board = collect_success_by_board<T>(
        actual_position,
        actual_success,
        row_width
    );
    const size_t expected_samples = std::min<size_t>(current.stored_boards.size(), 3U);
    size_t checked = 0U;
    for (uint64_t board : current.stored_boards) {
        if (checked >= expected_samples) {
            break;
        }
        std::vector<T> expected_lanes;
        expected_lanes.reserve(row_width);
        bool expected_live = false;
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            const T expected = direct_oracle_board_value<T>(
                board,
                lane,
                future2_values,
                future4_values,
                options
            );
            expected_lanes.push_back(expected);
            expected_live = expected_live || expected != options.zero_value;
        }

        const auto actual_it = actual_by_board.find(board);
        if (actual_it == actual_by_board.end()) {
            check(!expected_live, "sampled non-zero board missing from output");
            ++checked;
            continue;
        }
        check(actual_it->second.size() == row_width, "sampled output lane count mismatch");
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            check_value_close<T>(
                actual_it->second[lane],
                expected_lanes[lane],
                "sampled board success rate mismatch"
            );
        }
        ++checked;
    }
    check(checked == expected_samples, "sampled board verifier did not run enough checks");
}

template <typename T>
void run_exact_family_matches_single_case(BCSuccessDTypeMode dtype, const char *name) {
    constexpr uint32_t row_width = 2U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    const BCLut lut(test_alphabet());

    std::vector<uint8_t> success_shifts;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        success_shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }

    BC::BCSolveEdgeOptions edge_options;
    edge_options.canonical_batch_size = 4U;
    edge_options.canonical_symm_mode = symm_mode;
    edge_options.spawn_rate4 = 0.25;
    edge_options.success_target_rank = 3;
    edge_options.success_shifts = &success_shifts;

    const std::vector<uint64_t> current_input{
        make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
        make_board({{0U, 2U}, {15U, 2U}}),
        make_board({{7U, 3U}}),
    };
    PositionLayer current = write_position_layer(
        lut,
        full_axis_for_sum(8U),
        current_input,
        symm_mode
    );
    FutureSets futures = collect_future_sets(current.stored_boards, edge_options);
    check(!futures.boards2.empty(), "test requires spawn2 future boards");
    futures.boards2.erase(*futures.boards2.begin());

    PositionLayer future2 = write_position_layer(
        lut,
        full_axis_for_sum(10U),
        std::vector<uint64_t>(futures.boards2.begin(), futures.boards2.end()),
        static_cast<int>(SymmMode::Identity)
    );
    PositionLayer future4 = write_position_layer(
        lut,
        full_axis_for_sum(12U),
        std::vector<uint64_t>(futures.boards4.begin(), futures.boards4.end()),
        static_cast<int>(SymmMode::Identity)
    );

    std::map<uint64_t, std::vector<T>> future2_values =
        assign_typed_values<T>(futures.boards2, row_width, 1000U);
    std::map<uint64_t, std::vector<T>> future4_values =
        assign_typed_values<T>(futures.boards4, row_width, 2000U);
    apply_dtype_storage_semantics<T>(future2_values, dtype);
    apply_dtype_storage_semantics<T>(future4_values, dtype);
    const std::vector<uint8_t> future2_success_bytes =
        write_typed_success_layer<T>(future2.reader, future2_values, row_width, dtype);
    const std::vector<uint8_t> future4_success_bytes =
        write_typed_success_layer<T>(future4.reader, future4_values, row_width, dtype);

    const std::filesystem::path root = temp_root(name);
    std::filesystem::create_directories(root);
    const std::filesystem::path current_path = root / "current.bcpos";
    const std::filesystem::path future2_path = root / "future2.bcpos";
    const std::filesystem::path future4_path = root / "future4.bcpos";
    const std::filesystem::path success2_path = root / "future2.bcsuc";
    const std::filesystem::path success4_path = root / "future4.bcsuc";
    const std::filesystem::path single_position_path = root / "single_out.bcpos";
    const std::filesystem::path single_success_path = root / "single_out.bcsuc";
    const std::filesystem::path family_position_path = root / "family_out.bcpos";
    const std::filesystem::path family_success_path = root / "family_out.bcsuc";
    const std::filesystem::path family_block_position_path = root / "family_block_out.bcpos";
    const std::filesystem::path family_block_success_path = root / "family_block_out.bcsuc";

    BC::write_position_layer_to_file(current_path, current.bytes);
    BC::write_position_layer_to_file(future2_path, future2.bytes);
    BC::write_position_layer_to_file(future4_path, future4.bytes);
    write_bytes_file(success2_path, future2_success_bytes);
    write_bytes_file(success4_path, future4_success_bytes);

    BCPositionStreamingReader current_stream =
        BCPositionStreamingReader::open_buffered(current_path, lut);
    BCPositionStreamingReader future2_stream =
        BCPositionStreamingReader::open_buffered(future2_path, lut);
    BCPositionStreamingReader future4_stream =
        BCPositionStreamingReader::open_buffered(future4_path, lut);
    BCSuccessStreamingReader future2_success =
        BCSuccessStreamingReader::open_buffered(success2_path, future2_stream, row_width);
    BCSuccessStreamingReader future4_success =
        BCSuccessStreamingReader::open_buffered(success4_path, future4_stream, row_width);

    BCSingleChunkSolveOptions<T> single_options;
    single_options.solve.row_width = row_width;
    single_options.solve.set_dtype(dtype);
    single_options.solve.edge_options = edge_options;
    single_options.current_chunk_rows = 1U;
    {
        BC::BCBufferedFileWriter position_writer(single_position_path);
        BC::BCBufferedFileWriter success_writer(single_success_path);
        const auto result = BC::bc_single_chunk_solve_strict_1x_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            root / "single_tmp",
            single_options
        );
        check(result.position_bytes != 0U, "single strict position output should be non-empty");
        check(result.success_bytes != 0U, "single strict success output should be non-empty");
    }

    BCFamilySolveOptions<T> family_options;
    family_options.solve = single_options.solve;
    {
        BC::BCBufferedFileWriter position_writer(family_position_path);
        BC::BCBufferedFileWriter success_writer(family_success_path);
        const auto result = BC::bc_family_solve_exact_layer_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            root / "family_tmp",
            family_options
        );
        check(result.position_bytes != 0U, "family position output should be non-empty");
        check(result.success_bytes != 0U, "family success output should be non-empty");
        check(result.stats.spawn4_passes == current_stream.axis().family_count(),
            "family spawn4 pass count mismatch");
        check(result.stats.spawn2_passes == current_stream.axis().family_count(),
            "family spawn2 pass count mismatch");
        check(result.stats.spawn4_future_reuse_groups != 0U,
            "family spawn4 reuse group count should be non-zero");
        check(result.stats.spawn2_future_reuse_groups != 0U,
            "family spawn2 reuse group count should be non-zero");
        check(result.stats.spawn4_future_reuse_groups <= result.stats.spawn4_passes,
            "family spawn4 reuse groups exceed pass count");
        check(result.stats.spawn2_future_reuse_groups <= result.stats.spawn2_passes,
            "family spawn2 reuse groups exceed pass count");
        check(result.stats.final_stage_bytes_written == 0U,
            "family solve should not stage finalized success values");
        check(result.stats.final_stage_bytes_read == 0U,
            "family solve should not reread finalized success values");
    }
    {
        BCFamilySolveOptions<T> block_options = family_options;
        block_options.interleave_spawn_phases = true;
        block_options.interleave_block_fids = 1U;
        BC::BCBufferedFileWriter position_writer(family_block_position_path);
        BC::BCBufferedFileWriter success_writer(family_block_success_path);
        const auto result = BC::bc_family_solve_exact_layer_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            root / "family_block_tmp",
            block_options
        );
        check(result.position_bytes != 0U,
            "family block-interleaved position output should be non-empty");
        check(result.success_bytes != 0U,
            "family block-interleaved success output should be non-empty");
        check(result.stats.scratch4_cells_written == 0U,
            "family block-interleaved solve should not write scratch4 temp");
        check(result.stats.scratch4_cells_read == 0U,
            "family block-interleaved solve should not read scratch4 temp");
        check(result.stats.final_stage_bytes_written == 0U,
            "family block-interleaved solve should not stage finalized success values");
        check(result.stats.final_stage_bytes_read == 0U,
            "family block-interleaved solve should not reread finalized success values");
    }
    {
        BCFamilySolveOptions<T> block_options = family_options;
        block_options.interleave_spawn_phases = true;
        block_options.interleave_block_fids = 3U;
        BC::BCBufferedFileWriter position_writer(root / "family_block_reject.bcpos");
        BC::BCBufferedFileWriter success_writer(root / "family_block_reject.bcsuc");
        bool rejected = false;
        try {
            (void)BC::bc_family_solve_exact_layer_to_files<T>(
                current_stream,
                future2_stream,
                future2_success,
                future4_stream,
                future4_success,
                position_writer,
                success_writer,
                root / "family_block_reject_tmp",
                block_options
            );
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "family interleaved block_fids > 1 should be rejected");
    }

    BCPositionStreamingReader single_position =
        BCPositionStreamingReader::open_buffered(single_position_path, lut);
    BCSuccessStreamingReader single_success =
        BCSuccessStreamingReader::open_buffered(single_success_path, single_position, row_width);
    BCPositionStreamingReader family_position =
        BCPositionStreamingReader::open_buffered(family_position_path, lut);
    BCSuccessStreamingReader family_success =
        BCSuccessStreamingReader::open_buffered(family_success_path, family_position, row_width);
    BCPositionStreamingReader family_block_position =
        BCPositionStreamingReader::open_buffered(family_block_position_path, lut);
    BCSuccessStreamingReader family_block_success =
        BCSuccessStreamingReader::open_buffered(
            family_block_success_path,
            family_block_position,
            row_width
        );
    check(family_success.dtype_mode() == dtype, "family output dtype mismatch");
    check(family_success.row_width() == row_width, "family output row_width mismatch");
    compare_success_on_current_boards<T>(
        current,
        single_success,
        family_success,
        lut,
        row_width
    );
    compare_success_layers<T>(
        single_position,
        single_success,
        family_position,
        family_success,
        row_width
    );
    compare_success_layers<T>(
        single_position,
        single_success,
        family_block_position,
        family_block_success,
        row_width
    );
    const BCPositionLayerReader single_position_mem(
        BC::read_position_layer_from_file(single_position_path),
        lut
    );
    const BCSuccessLayerReader single_success_mem(
        BC::read_success_layer_from_file(single_success_path),
        single_position_mem,
        row_width
    );
    const BCPositionLayerReader family_position_mem(
        BC::read_position_layer_from_file(family_position_path),
        lut
    );
    const BCSuccessLayerReader family_success_mem(
        BC::read_success_layer_from_file(family_success_path),
        family_position_mem,
        row_width
    );
    compare_output_layers_logically<T>(
        single_position_mem,
        single_success_mem,
        family_position_mem,
        family_success_mem,
        row_width
    );
    verify_sampled_board_success_rates<T>(
        current,
        family_position_mem,
        family_success_mem,
        future2_values,
        future4_values,
        single_options.solve,
        row_width
    );

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

template <typename T>
void run_modulo_family_matches_single_case(BCSuccessDTypeMode dtype, const char *name) {
    constexpr uint32_t row_width = 2U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    constexpr uint32_t modulus = 5U;
    const BCLut lut(test_alphabet());
    const std::vector<BC::LayerSum> possible_sums =
        BC::build_possible_8tile_sums(test_alphabet(), BC::default_2048_tile_sum_values());
    const BC::BCFamilyPartitionPolicy policy = BC::BCFamilyPartitionPolicy::modulo(modulus);

    std::vector<uint8_t> success_shifts;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        success_shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }

    BC::BCSolveEdgeOptions edge_options;
    edge_options.canonical_batch_size = 4U;
    edge_options.canonical_symm_mode = symm_mode;
    edge_options.spawn_rate4 = 0.25;
    edge_options.success_target_rank = 3;
    edge_options.success_shifts = &success_shifts;
    edge_options.future_cell_modulus = modulus;

    const std::vector<uint64_t> current_input{
        make_board({{0U, 2U}, {1U, 2U}, {4U, 1U}, {5U, 2U}}),
        make_board({{0U, 1U}, {1U, 2U}, {2U, 3U}}),
        make_board({{7U, 3U}, {8U, 2U}, {9U, 1U}}),
    };
    const BCFamilyTable current_axis =
        BC::build_family_partition_axis_for_layer(14U, 1U, possible_sums, policy);
    const BCFamilyTable future2_axis =
        BC::build_family_partition_axis_for_layer(16U, 1U, possible_sums, policy);
    const BCFamilyTable future4_axis =
        BC::build_family_partition_axis_for_layer(18U, 1U, possible_sums, policy);
    PositionLayer current = write_modulo_position_layer(
        lut,
        current_axis,
        possible_sums,
        modulus,
        current_input,
        symm_mode
    );
    FutureSets futures = collect_future_sets(current.stored_boards, edge_options);
    check(!futures.boards2.empty(), "modulo test requires spawn2 future boards");
    futures.boards2.erase(*futures.boards2.begin());

    PositionLayer future2 = write_modulo_position_layer(
        lut,
        future2_axis,
        possible_sums,
        modulus,
        std::vector<uint64_t>(futures.boards2.begin(), futures.boards2.end()),
        static_cast<int>(SymmMode::Identity)
    );
    PositionLayer future4 = write_modulo_position_layer(
        lut,
        future4_axis,
        possible_sums,
        modulus,
        std::vector<uint64_t>(futures.boards4.begin(), futures.boards4.end()),
        static_cast<int>(SymmMode::Identity)
    );

    std::map<uint64_t, std::vector<T>> future2_values =
        assign_typed_values<T>(futures.boards2, row_width, 3000U);
    std::map<uint64_t, std::vector<T>> future4_values =
        assign_typed_values<T>(futures.boards4, row_width, 4000U);
    apply_dtype_storage_semantics<T>(future2_values, dtype);
    apply_dtype_storage_semantics<T>(future4_values, dtype);
    const std::vector<uint8_t> future2_success_bytes =
        write_typed_success_layer<T>(future2.reader, future2_values, row_width, dtype);
    const std::vector<uint8_t> future4_success_bytes =
        write_typed_success_layer<T>(future4.reader, future4_values, row_width, dtype);

    const std::filesystem::path root = temp_root(name);
    std::filesystem::create_directories(root);
    const std::filesystem::path current_path = root / "current.bcpos";
    const std::filesystem::path future2_path = root / "future2.bcpos";
    const std::filesystem::path future4_path = root / "future4.bcpos";
    const std::filesystem::path success2_path = root / "future2.bcsuc";
    const std::filesystem::path success4_path = root / "future4.bcsuc";
    const std::filesystem::path single_position_path = root / "single_out.bcpos";
    const std::filesystem::path single_success_path = root / "single_out.bcsuc";
    const std::filesystem::path family_position_path = root / "family_out.bcpos";
    const std::filesystem::path family_success_path = root / "family_out.bcsuc";

    BC::write_position_layer_to_file(current_path, current.bytes);
    BC::write_position_layer_to_file(future2_path, future2.bytes);
    BC::write_position_layer_to_file(future4_path, future4.bytes);
    write_bytes_file(success2_path, future2_success_bytes);
    write_bytes_file(success4_path, future4_success_bytes);

    BCPositionStreamingReader current_stream =
        BCPositionStreamingReader::open_buffered(current_path, lut);
    BCPositionStreamingReader future2_stream =
        BCPositionStreamingReader::open_buffered(future2_path, lut);
    BCPositionStreamingReader future4_stream =
        BCPositionStreamingReader::open_buffered(future4_path, lut);
    BCSuccessStreamingReader future2_success =
        BCSuccessStreamingReader::open_buffered(success2_path, future2_stream, row_width);
    BCSuccessStreamingReader future4_success =
        BCSuccessStreamingReader::open_buffered(success4_path, future4_stream, row_width);

    BCSingleChunkSolveOptions<T> single_options;
    single_options.solve.row_width = row_width;
    single_options.solve.set_dtype(dtype);
    single_options.solve.edge_options = edge_options;
    single_options.current_chunk_rows = 1U;
    {
        BC::BCBufferedFileWriter position_writer(single_position_path);
        BC::BCBufferedFileWriter success_writer(single_success_path);
        const auto result = BC::bc_single_chunk_solve_strict_1x_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            root / "single_tmp",
            single_options
        );
        check(result.position_bytes != 0U, "modulo single position output should be non-empty");
        check(result.success_bytes != 0U, "modulo single success output should be non-empty");
    }

    const auto current_partition = BC::build_family_partition_layer_map(current_axis, possible_sums, policy);
    const auto future2_partition = BC::build_family_partition_layer_map(future2_axis, possible_sums, policy);
    const auto future4_partition = BC::build_family_partition_layer_map(future4_axis, possible_sums, policy);
    BCFamilySolveOptions<T> family_options;
    family_options.solve = single_options.solve;
    {
        BC::BCBufferedFileWriter position_writer(family_position_path);
        BC::BCBufferedFileWriter success_writer(family_success_path);
        const auto result = BC::bc_family_solve_layer_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            current_partition,
            future2_partition,
            future4_partition,
            position_writer,
            success_writer,
            root / "family_tmp",
            family_options
        );
        check(result.position_bytes != 0U, "modulo family position output should be non-empty");
        check(result.success_bytes != 0U, "modulo family success output should be non-empty");
        check(result.stats.spawn4_future_reuse_groups != 0U,
            "modulo family spawn4 reuse group count should be non-zero");
        check(result.stats.spawn2_future_reuse_groups != 0U,
            "modulo family spawn2 reuse group count should be non-zero");
        check(result.stats.spawn4_future_reuse_groups <= result.stats.spawn4_passes,
            "modulo family spawn4 reuse groups exceed pass count");
        check(result.stats.spawn2_future_reuse_groups <= result.stats.spawn2_passes,
            "modulo family spawn2 reuse groups exceed pass count");
    }

    BCPositionStreamingReader single_position =
        BCPositionStreamingReader::open_buffered(single_position_path, lut);
    BCSuccessStreamingReader single_success =
        BCSuccessStreamingReader::open_buffered(single_success_path, single_position, row_width);
    BCPositionStreamingReader family_position =
        BCPositionStreamingReader::open_buffered(family_position_path, lut);
    BCSuccessStreamingReader family_success =
        BCSuccessStreamingReader::open_buffered(family_success_path, family_position, row_width);
    compare_success_layers<T>(
        single_position,
        single_success,
        family_position,
        family_success,
        row_width
    );
    const BCPositionLayerReader single_position_mem(
        BC::read_position_layer_from_file(single_position_path),
        lut
    );
    const BCSuccessLayerReader single_success_mem(
        BC::read_success_layer_from_file(single_success_path),
        single_position_mem,
        row_width
    );
    const BCPositionLayerReader family_position_mem(
        BC::read_position_layer_from_file(family_position_path),
        lut
    );
    const BCSuccessLayerReader family_success_mem(
        BC::read_success_layer_from_file(family_success_path),
        family_position_mem,
        row_width
    );
    compare_output_layers_logically<T>(
        single_position_mem,
        single_success_mem,
        family_position_mem,
        family_success_mem,
        row_width
    );
    verify_sampled_board_success_rates<T>(
        current,
        family_position_mem,
        family_success_mem,
        future2_values,
        future4_values,
        single_options.solve,
        row_width
    );

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

} // namespace

int main() {
    try {
        run_exact_family_matches_single_case<uint32_t>(BCSuccessDTypeMode::UInt32, "uint32");
        run_exact_family_matches_single_case<uint64_t>(BCSuccessDTypeMode::UInt64, "uint64");
        run_exact_family_matches_single_case<double>(BCSuccessDTypeMode::Float64, "float64");
        run_modulo_family_matches_single_case<uint32_t>(BCSuccessDTypeMode::UInt32, "mod_uint32");
    } catch (const std::exception &ex) {
        std::cerr << "test_bc_family_solve failed: " << ex.what() << '\n';
        return 1;
    }
    std::cout << "test_bc_family_solve passed\n";
    return 0;
}
