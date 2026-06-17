#include "BCCellBuilder.h"
#include "BCResidentSolve.h"
#include "BCSingleChunkSolve.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "SymmetryUtils.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCPositionStreamingReader;
using BC::BCResidentSolveOptions;
using BC::BCSingleChunkSolveOptions;
using BC::BCSuccessDTypeMode;
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BCSuccessStreamingReader;
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
struct SingleChunkFixture {
    BCLut lut;
    std::vector<uint8_t> success_shifts;
    PositionLayer current;
    PositionLayer future2;
    PositionLayer future4;
    std::vector<uint8_t> future2_success_bytes;
    std::vector<uint8_t> future4_success_bytes;

    SingleChunkFixture()
        : lut(test_alphabet()) {}
};

template <typename T>
void make_single_chunk_fixture(
    SingleChunkFixture<T> &fixture,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    int symm_mode
) {
    fixture.success_shifts.clear();
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

    std::map<uint64_t, std::vector<T>> future2_values =
        assign_typed_values<T>(futures.boards2, row_width, 1000U);
    std::map<uint64_t, std::vector<T>> future4_values =
        assign_typed_values<T>(futures.boards4, row_width, 2000U);
    apply_dtype_storage_semantics<T>(future2_values, dtype);
    apply_dtype_storage_semantics<T>(future4_values, dtype);
    fixture.future2_success_bytes = write_typed_success_layer<T>(
        fixture.future2.reader,
        future2_values,
        row_width,
        dtype
    );
    fixture.future4_success_bytes = write_typed_success_layer<T>(
        fixture.future4.reader,
        future4_values,
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

template <typename T, typename ActualLayerT>
void compare_layers_on_current_boards(
    const SingleChunkFixture<T> &fixture,
    const BC::BCResidentSolvedLayer<T> &expected,
    const ActualLayerT &actual,
    uint32_t row_width,
    const char *label = "single chunk"
) {
    for (uint64_t board : fixture.current.stored_boards) {
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            T expected_value{};
            T actual_value{};
            const bool expected_found = expected.lookup.lookup(board, expected_value, lane);
            const bool actual_found = actual.lookup.lookup(board, actual_value, lane);
            check(expected_found == actual_found, "single chunk found state mismatch");
            if (expected_found) {
                if constexpr (std::is_floating_point_v<T>) {
                    const double diff = std::fabs(
                        static_cast<double>(actual_value) - static_cast<double>(expected_value)
                    );
                    if (diff > 1.0e-5) {
                        throw std::runtime_error(
                            std::string(label) + " value mismatch board=" + std::to_string(board) +
                            " lane=" + std::to_string(lane) +
                            " expected=" + std::to_string(static_cast<double>(expected_value)) +
                            " actual=" + std::to_string(static_cast<double>(actual_value))
                        );
                    }
                } else {
                    const uint64_t actual_u = static_cast<uint64_t>(actual_value);
                    const uint64_t expected_u = static_cast<uint64_t>(expected_value);
                    const uint64_t diff = actual_u > expected_u
                        ? actual_u - expected_u
                        : expected_u - actual_u;
                    if (diff > 2U) {
                        throw std::runtime_error(
                            std::string(label) + " value mismatch board=" + std::to_string(board) +
                            " lane=" + std::to_string(lane) +
                            " expected=" + std::to_string(expected_u) +
                            " actual=" + std::to_string(actual_u)
                        );
                    }
                }
            }
        }
    }
}

template <typename T>
void run_single_chunk_case(BCSuccessDTypeMode dtype, const char *name) {
    constexpr uint32_t row_width = 2U;
    constexpr int symm_mode = static_cast<int>(SymmMode::Full);
    SingleChunkFixture<T> fixture;
    make_single_chunk_fixture<T>(fixture, row_width, dtype, symm_mode);

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

    BCResidentSolveOptions<T> resident_options =
        make_resident_options<T>(fixture.success_shifts, row_width, dtype, symm_mode);
    const auto resident = BC::bc_resident_solve_compacted_layer<T>(
        fixture.current.reader,
        future2_layer,
        future4_layer,
        resident_options
    );

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        (std::string("bc_single_chunk_solve_test_") + name);
    std::filesystem::create_directories(root);
    const std::filesystem::path current_path = root / "current.bcpos";
    const std::filesystem::path future2_path = root / "future2.bcpos";
    const std::filesystem::path future4_path = root / "future4.bcpos";
    const std::filesystem::path success2_path = root / "future2.bcsuc";
    const std::filesystem::path success4_path = root / "future4.bcsuc";
    const std::filesystem::path out_position_path = root / "single_out.bcpos";
    const std::filesystem::path out_success_path = root / "single_out.bcsuc";
    const std::filesystem::path strict_position_path = root / "strict_out.bcpos";
    const std::filesystem::path strict_success_path = root / "strict_out.bcsuc";
    const std::filesystem::path strict_temp_dir = root / "strict_tmp";
    const std::filesystem::path strict_frontier_position_path = root / "strict_frontier_out.bcpos";
    const std::filesystem::path strict_frontier_success_path = root / "strict_frontier_out.bcsuc";
    const std::filesystem::path strict_frontier_temp_dir = root / "strict_frontier_tmp";
    const std::filesystem::path strict_direct_position_path = root / "strict_direct_out.bcpos";
    const std::filesystem::path strict_direct_success_path = root / "strict_direct_out.bcsuc";
    const std::filesystem::path strict_direct_temp_dir = root / "strict_direct_tmp";

    BC::write_position_layer_to_file(current_path, fixture.current.bytes);
    BC::write_position_layer_to_file(future2_path, fixture.future2.bytes);
    BC::write_position_layer_to_file(future4_path, fixture.future4.bytes);
    write_bytes_file(success2_path, fixture.future2_success_bytes);
    write_bytes_file(success4_path, fixture.future4_success_bytes);

    BCPositionStreamingReader current_stream =
        BCPositionStreamingReader::open_buffered(current_path, fixture.lut);
    BCPositionStreamingReader future2_stream =
        BCPositionStreamingReader::open_buffered(future2_path, fixture.lut);
    BCPositionStreamingReader future4_stream =
        BCPositionStreamingReader::open_buffered(future4_path, fixture.lut);
    BCSuccessStreamingReader future2_success =
        BCSuccessStreamingReader::open_buffered(success2_path, future2_stream, row_width);
    BCSuccessStreamingReader future4_success =
        BCSuccessStreamingReader::open_buffered(success4_path, future4_stream, row_width);

    BCSingleChunkSolveOptions<T> options;
    options.solve = resident_options;
    options.current_chunk_cells = 1U;
    options.current_chunk_max_rows = 2U;
    const auto single = BC::bc_single_chunk_solve_compacted_layer<T>(
        current_stream,
        future2_stream,
        future2_success,
        future4_stream,
        future4_success,
        options
    );
    compare_layers_on_current_boards<T>(fixture, resident.layer, single.layer, row_width, "single memory");

    const auto frontier_single = BC::bc_single_chunk_solve_compacted_layer_from_frontier<T>(
        current_stream,
        future2_layer,
        future4_layer,
        options
    );
    compare_layers_on_current_boards<T>(fixture, resident.layer, frontier_single.layer, row_width, "frontier memory");

    check(single.stats.current_chunks > 1U, "single chunk test should split current layer");
    check(single.stats.future_resident_layers_max == 1U, "single chunk should hold one future lookup at a time");
    check(
        frontier_single.stats.future_resident_layers_max == 2U,
        "single chunk frontier should account for two resident future layers"
    );
    check(
        frontier_single.stats.future2_position_read_seconds == 0.0 &&
            frontier_single.stats.future4_position_read_seconds == 0.0 &&
            frontier_single.stats.future2_success_read_seconds == 0.0 &&
            frontier_single.stats.future4_success_read_seconds == 0.0 &&
            frontier_single.stats.future2_index_seconds == 0.0 &&
            frontier_single.stats.future4_index_seconds == 0.0 &&
            frontier_single.stats.future_release_seconds == 0.0,
        "single chunk frontier should not read/index/release future layers"
    );
    check(
        single.stats.compact_live_rows == resident.layer.compact_stats.live_rows,
        "single chunk compact live rows mismatch"
    );
    check(
        single.stats.compact_zero_pruned_rows == resident.layer.compact_stats.zero_pruned_rows,
        "single chunk compact zero rows mismatch"
    );

    {
        BC::BCBufferedFileWriter position_writer(out_position_path);
        BC::BCBufferedFileWriter success_writer(out_success_path);
        const auto file_result = BC::bc_single_chunk_solve_compacted_layer_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            options
        );
        check(file_result.position_bytes != 0U, "single chunk file position output should be non-empty");
        check(file_result.success_bytes != 0U, "single chunk file success output should be non-empty");
    }

    BCPositionStreamingReader output_position =
        BCPositionStreamingReader::open_buffered(out_position_path, fixture.lut);
    BCSuccessStreamingReader output_success =
        BCSuccessStreamingReader::open_buffered(out_success_path, output_position, row_width);
    check(output_success.dtype_mode() == dtype, "single chunk file output dtype mismatch");
    check(output_success.row_width() == row_width, "single chunk file output row_width mismatch");
    const auto file_layer = BC::bc_single_chunk_load_frontier_layer<T>(
        output_position,
        output_success,
        fixture.lut,
        row_width,
        dtype
    );
    compare_layers_on_current_boards<T>(fixture, resident.layer, file_layer, row_width, "file output");

    {
        BC::BCBufferedFileWriter position_writer(strict_position_path);
        BC::BCBufferedFileWriter success_writer(strict_success_path);
        const auto strict_result = BC::bc_single_chunk_solve_strict_1x_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            strict_temp_dir,
            options
        );
        check(strict_result.position_bytes != 0U, "strict single position output should be non-empty");
        check(strict_result.success_bytes != 0U, "strict single success output should be non-empty");
        check(
            strict_result.stats.future_resident_layers_max == 1U,
            "strict single should hold at most one future layer"
        );
        check(
            strict_result.stats.current_board_cache_bytes == 0U,
            "strict single should not build a current board cache"
        );
    }

    BCPositionStreamingReader strict_position =
        BCPositionStreamingReader::open_buffered(strict_position_path, fixture.lut);
    BCSuccessStreamingReader strict_success =
        BCSuccessStreamingReader::open_buffered(strict_success_path, strict_position, row_width);
    check(strict_success.dtype_mode() == dtype, "strict single file output dtype mismatch");
    check(strict_success.row_width() == row_width, "strict single file output row_width mismatch");
    const auto strict_layer = BC::bc_single_chunk_load_frontier_layer<T>(
        strict_position,
        strict_success,
        fixture.lut,
        row_width,
        dtype
    );
    compare_layers_on_current_boards<T>(fixture, resident.layer, strict_layer, row_width, "strict output");

    {
        BC::BCSingleChunkFrontierLayer<T> loaded_future4 =
            BC::bc_single_chunk_load_frontier_layer<T>(
                future4_stream,
                future4_success,
                fixture.lut,
                row_width,
                dtype
            );
        BC::BCBufferedFileWriter position_writer(strict_frontier_position_path);
        BC::BCBufferedFileWriter success_writer(strict_frontier_success_path);
        const auto strict_frontier_result =
            BC::bc_single_chunk_solve_strict_1x_to_files_from_future4_frontier<T>(
                current_stream,
                future2_stream,
                future2_success,
                std::move(loaded_future4),
                position_writer,
                success_writer,
                strict_frontier_temp_dir,
                options
            );
        check(strict_frontier_result.position_bytes != 0U, "strict frontier position output should be non-empty");
        check(strict_frontier_result.success_bytes != 0U, "strict frontier success output should be non-empty");
        check(
            strict_frontier_result.stats.future_resident_layers_max == 1U,
            "strict frontier should hold at most one future layer"
        );
        check(
            strict_frontier_result.stats.future4_position_read_seconds == 0.0 &&
                strict_frontier_result.stats.future4_success_read_seconds == 0.0 &&
                strict_frontier_result.stats.future4_index_seconds == 0.0,
            "strict frontier should reuse preloaded future4 without read/index"
        );
        check(
            strict_frontier_result.stats.future2_position_read_seconds >= 0.0 &&
                strict_frontier_result.stats.future2_success_read_seconds >= 0.0 &&
                strict_frontier_result.stats.future2_index_seconds >= 0.0,
            "strict frontier should account future2 load/index fields"
        );
        check(
            strict_frontier_result.next_future4_layer.position.axis().layer_sum() ==
                future2_stream.axis().layer_sum(),
            "strict frontier should return loaded future2 as next future4"
        );
    }

    BCPositionStreamingReader strict_frontier_position =
        BCPositionStreamingReader::open_buffered(strict_frontier_position_path, fixture.lut);
    BCSuccessStreamingReader strict_frontier_success =
        BCSuccessStreamingReader::open_buffered(
            strict_frontier_success_path,
            strict_frontier_position,
            row_width
        );
    check(strict_frontier_success.dtype_mode() == dtype, "strict frontier file output dtype mismatch");
    check(strict_frontier_success.row_width() == row_width, "strict frontier file output row_width mismatch");
    const auto strict_frontier_layer = BC::bc_single_chunk_load_frontier_layer<T>(
        strict_frontier_position,
        strict_frontier_success,
        fixture.lut,
        row_width,
        dtype
    );
    compare_layers_on_current_boards<T>(fixture, resident.layer, strict_frontier_layer, row_width, "strict frontier output");

    {
        BC::BCDirectFileIOOptions direct_options;
        direct_options.queue_depth = 4U;
        direct_options.overlapped = true;
        BC::BCDirectFileWriter position_writer(strict_direct_position_path, direct_options);
        BC::BCDirectFileWriter success_writer(strict_direct_success_path, direct_options);
        const auto strict_direct_result = BC::bc_single_chunk_solve_strict_1x_to_files<T>(
            current_stream,
            future2_stream,
            future2_success,
            future4_stream,
            future4_success,
            position_writer,
            success_writer,
            strict_direct_temp_dir,
            options
        );
        check(strict_direct_result.position_bytes != 0U, "strict direct position output should be non-empty");
        check(strict_direct_result.success_bytes != 0U, "strict direct success output should be non-empty");
    }

    BCPositionStreamingReader strict_direct_position =
        BCPositionStreamingReader::open_direct_auto(strict_direct_position_path, fixture.lut, 4U, true);
    BCSuccessStreamingReader strict_direct_success =
        BCSuccessStreamingReader::open_direct_auto(
            strict_direct_success_path,
            strict_direct_position,
            row_width,
            4U,
            true
        );
    check(strict_direct_success.dtype_mode() == dtype, "strict direct file output dtype mismatch");
    check(strict_direct_success.row_width() == row_width, "strict direct file output row_width mismatch");
    const auto strict_direct_layer = BC::bc_single_chunk_load_frontier_layer<T>(
        strict_direct_position,
        strict_direct_success,
        fixture.lut,
        row_width,
        dtype
    );
    compare_layers_on_current_boards<T>(
        fixture,
        resident.layer,
        strict_direct_layer,
        row_width,
        "strict direct output"
    );

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

} // namespace

int main() {
    try {
        run_single_chunk_case<uint32_t>(BCSuccessDTypeMode::UInt32, "uint32");
        run_single_chunk_case<uint64_t>(BCSuccessDTypeMode::UInt64, "uint64");
        run_single_chunk_case<float>(BCSuccessDTypeMode::Float32, "float32");
        run_single_chunk_case<double>(BCSuccessDTypeMode::Float64, "float64");
        run_single_chunk_case<float>(BCSuccessDTypeMode::OneMinusFloat32, "one_minus_float32");
        run_single_chunk_case<double>(BCSuccessDTypeMode::OneMinusFloat64, "one_minus_float64");
    } catch (const std::exception &ex) {
        std::cerr << "bc_single_chunk_solve_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_single_chunk_solve_test passed\n";
    return 0;
}
