#include "BCBacksolve.h"

#include "BCCellBuilder.h"
#include "BCPositionScanner.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "SymmetryUtils.h"

#include <algorithm>
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
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BucketRank;
using BC::CellId;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
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

std::vector<uint8_t> write_success_layer(
    const BCPositionLayerReader &position,
    const std::map<uint64_t, uint32_t> &values_by_board
) {
    std::vector<std::vector<uint32_t>> values(position.cell_count());
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        values[static_cast<size_t>(cid)].assign(desc.success_rows, 0U);
        if (desc.success_rows == 0U || desc.empty()) {
            continue;
        }
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                const auto it = values_by_board.find(entry.board);
                values[static_cast<size_t>(cid)][entry.local_success_row] =
                    it == values_by_board.end() ? 0U : it->second;
            }
        );
    }

    BCSuccessLayerWriter writer;
    writer.begin_layer(position, 1U);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, values[static_cast<size_t>(cid)]);
    }
    return writer.finish_layer();
}

struct FutureSets {
    std::set<uint64_t> boards2;
    std::set<uint64_t> boards4;
};

bool terminal_success(uint64_t board, const BC::BCBacksolveOptions &options) {
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
    const BC::BCBacksolveOptions &options
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

std::map<uint64_t, uint32_t> assign_values(const std::set<uint64_t> &boards, uint32_t base) {
    std::map<uint64_t, uint32_t> values;
    uint32_t i = 0U;
    for (uint64_t board : boards) {
        values[board] = base + i * 17U;
        ++i;
    }
    return values;
}

uint32_t oracle_value(
    uint64_t board,
    const std::map<uint64_t, uint32_t> &future2,
    const std::map<uint64_t, uint32_t> &future4,
    const BC::BCBacksolveOptions &options
) {
    if (terminal_success(board, options)) {
        return max_scale_value<uint32_t>();
    }
    double success_probability = 0.0;
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

        uint32_t best2 = 0U;
        const uint64_t spawn2 = BC::set_board_tile_unchecked(board, cell, 1U);
        const auto moves2 = BoardMover::move_all_dir(spawn2);
        const uint64_t moved2[4] = {
            std::get<0>(moves2), std::get<1>(moves2),
            std::get<2>(moves2), std::get<3>(moves2)
        };
        for (uint64_t moved : moved2) {
            if (moved == spawn2) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(moved, options.canonical_symm_mode);
            const auto it = future2.find(canonical);
            if (it != future2.end()) {
                best2 = std::max(best2, it->second);
            }
        }

        uint32_t best4 = 0U;
        const uint64_t spawn4 = BC::set_board_tile_unchecked(board, cell, 2U);
        const auto moves4 = BoardMover::move_all_dir(spawn4);
        const uint64_t moved4[4] = {
            std::get<0>(moves4), std::get<1>(moves4),
            std::get<2>(moves4), std::get<3>(moves4)
        };
        for (uint64_t moved : moved4) {
            if (moved == spawn4) {
                continue;
            }
            const uint64_t canonical = canonical_by_mode(moved, options.canonical_symm_mode);
            const auto it = future4.find(canonical);
            if (it != future4.end()) {
                best4 = std::max(best4, it->second);
            }
        }

        success_probability += static_cast<double>(best2) * (1.0 - options.spawn_rate4);
        success_probability += static_cast<double>(best4) * options.spawn_rate4;
    }
    return empty_count == 0U
        ? 0U
        : static_cast<uint32_t>(success_probability / static_cast<double>(empty_count));
}

void verify_result(
    const BCPositionLayerReader &current,
    const BCSuccessLayerReader &success,
    const std::vector<uint64_t> &current_boards,
    const std::map<uint64_t, uint32_t> &future2_values,
    const std::map<uint64_t, uint32_t> &future4_values,
    const BC::BCBacksolveOptions &options
) {
    std::map<uint64_t, uint32_t> actual_by_board;
    for (CellId cid = 0U; cid < current.cell_count(); ++cid) {
        if (current.descriptor(cid).success_rows == 0U) {
            continue;
        }
        BCPositionCellScanner(current, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                actual_by_board[entry.board] = success.read_value(cid, entry.local_success_row);
            }
        );
    }
    for (uint64_t board : current_boards) {
        const uint32_t expected = oracle_value(board, future2_values, future4_values, options);
        const auto it = actual_by_board.find(board);
        check(it != actual_by_board.end(), "backsolve output is missing current board");
        if (it->second != expected) {
            throw std::runtime_error("backsolve output value mismatch");
        }
    }
}

void run_resident_case(int symm_mode) {
    const BCLut lut(test_alphabet());
    std::vector<uint8_t> all_success_shifts;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        all_success_shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }

    BC::BCBacksolveOptions options;
    options.num_threads = 2;
    options.canonical_batch_size = 4U;
    options.canonical_symm_mode = symm_mode;
    options.spawn_rate4 = 0.25;
    options.success_target_rank = 3;
    options.success_shifts = &all_success_shifts;

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

    FutureSets futures = collect_future_sets(current.stored_boards, options);
    check(!futures.boards2.empty(), "test requires spawn2 future boards");
    futures.boards2.erase(*futures.boards2.begin()); // Deliberate future miss.

    const BCFamilyTable future2_axis = compact_axis_for_boards(lut, 10U, futures.boards2);
    const BCFamilyTable future4_axis = compact_axis_for_boards(lut, 12U, futures.boards4);
    PositionLayer future2 = write_position_layer(
        lut,
        future2_axis,
        std::vector<uint64_t>(futures.boards2.begin(), futures.boards2.end()),
        static_cast<int>(SymmMode::Identity)
    );
    PositionLayer future4 = write_position_layer(
        lut,
        future4_axis,
        std::vector<uint64_t>(futures.boards4.begin(), futures.boards4.end()),
        static_cast<int>(SymmMode::Identity)
    );
    const std::map<uint64_t, uint32_t> future2_values = assign_values(futures.boards2, 1000U);
    const std::map<uint64_t, uint32_t> future4_values = assign_values(futures.boards4, 2000U);
    const std::vector<uint8_t> future2_success_bytes =
        write_success_layer(future2.reader, future2_values);
    const std::vector<uint8_t> future4_success_bytes =
        write_success_layer(future4.reader, future4_values);
    const BCSuccessLayerReader future2_success(future2_success_bytes, future2.reader, 1U);
    const BCSuccessLayerReader future4_success(future4_success_bytes, future4.reader, 1U);

    const BC::BCBacksolveResult result = BC::backsolve_resident_layer(
        lut,
        current.reader,
        future2.reader,
        future2_success,
        future4.reader,
        future4_success,
        options
    );
    const BCSuccessLayerReader current_success(result.success_bytes, current.reader, 1U);
    verify_result(
        current.reader,
        current_success,
        current.stored_boards,
        future2_values,
        future4_values,
        options
    );
    check(result.stats.current_rows == current.stored_boards.size(), "stats current_rows mismatch");
    check(result.stats.terminal_success_rows == 1U, "stats terminal_success_rows mismatch");
    check(result.stats.queries2 != 0U && result.stats.queries4 != 0U, "stats should count future queries");

    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() /
        ("bc_backsolve_resident_test_" + std::to_string(symm_mode) + ".bcsuc");
    std::error_code ec;
    std::filesystem::remove(output_path, ec);
    const BC::BCBacksolveStats file_stats = BC::backsolve_resident_layer_to_file(
        output_path,
        lut,
        current.reader,
        future2.reader,
        future2_success,
        future4.reader,
        future4_success,
        options
    );
    const std::vector<uint8_t> file_bytes = BC::read_success_layer_from_file(output_path);
    std::filesystem::remove(output_path, ec);
    check(file_bytes == result.success_bytes, "file backsolve output bytes mismatch");
    check(file_stats.current_rows == result.stats.current_rows, "file stats current_rows mismatch");
}

void test_no_empty_board_is_zero() {
    const BCLut lut(test_alphabet());
    BC::BCBacksolveOptions options;
    options.canonical_symm_mode = static_cast<int>(SymmMode::Identity);
    options.num_threads = 1;

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
    const std::vector<uint8_t> future2_success_bytes = write_success_layer(future2.reader, {});
    const std::vector<uint8_t> future4_success_bytes = write_success_layer(future4.reader, {});
    const BCSuccessLayerReader future2_success(future2_success_bytes, future2.reader, 1U);
    const BCSuccessLayerReader future4_success(future4_success_bytes, future4.reader, 1U);
    const BC::BCBacksolveResult result = BC::backsolve_resident_layer(
        lut,
        current.reader,
        future2.reader,
        future2_success,
        future4.reader,
        future4_success,
        options
    );
    const BCSuccessLayerReader current_success(result.success_bytes, current.reader, 1U);
    uint32_t seen_rows = 0U;
    for (CellId cid = 0U; cid < current.reader.cell_count(); ++cid) {
        if (current.reader.descriptor(cid).success_rows == 0U) {
            continue;
        }
        BCPositionCellScanner(current.reader, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                ++seen_rows;
                check(entry.board == full_rank1_board(), "unexpected board in full-board layer");
                check(
                    current_success.read_value(cid, entry.local_success_row) == 0U,
                    "full board with no empty cells should solve to zero"
                );
            }
        );
    }
    check(seen_rows == 1U, "full-board layer should contain exactly one row");
}

} // namespace

int main() {
    try {
        std::cerr << "resident backsolve full canonical\n";
        run_resident_case(static_cast<int>(SymmMode::Full));
        std::cerr << "resident backsolve identity canonical\n";
        run_resident_case(static_cast<int>(SymmMode::Identity));
        std::cerr << "resident backsolve no empty board\n";
        test_no_empty_board_is_zero();
    } catch (const std::exception &ex) {
        std::cerr << "bc_backsolve_resident_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_backsolve_resident_test passed\n";
    return 0;
}
