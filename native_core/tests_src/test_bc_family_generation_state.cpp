#include "BCFamilyMutableStore.h"

#include "BCBoardOps.h"
#include "BCDirectFileIO.h"
#include "BCFamilyGeneration.h"
#include "BCFamilyPositionWriter.h"
#include "BCFileIO.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"
#include "BCResidentGeneration.h"
#include "BoardMover.h"
#include "Calculator.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyGenerationPass;
using BC::BCFamilyGenerationScheduler;
using BC::BCFamilyGenerationOptions;
using BC::BCFamilyStreamingGenerationSource;
using BC::BCFamilyMutableStore;
using BC::BCFamilyTable;
using BC::BCFileGenerationBlobIO;
using BC::BCGenerationBlobIO;
using BC::BCLut;
using BC::BCMutableCellState;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCResidentGenerationOptions;
using BC::BCResidentGenerationSource;
using BC::BCSourceCellWork;
using BC::BucketRank;
using BC::CellId;
using Candidate = std::tuple<CellId, uint64_t, BucketRank>;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

BC::FamilyIdList2 family_ids(std::initializer_list<BC::FamilyId> ids) {
    BC::FamilyIdList2 out;
    for (BC::FamilyId id : ids) {
        out.push_back(id);
    }
    return out;
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

uint64_t make_board(const std::vector<std::pair<uint8_t, uint8_t>> &tiles) {
    uint64_t board = 0U;
    for (const auto &[cell, tile] : tiles) {
        board = BC::set_board_tile(board, cell, tile);
    }
    return board;
}

[[nodiscard]] uint64_t canonicalize(uint64_t board) {
    return Calculator::canonical_full(board);
}

[[nodiscard]] bool move_one(uint64_t board, uint32_t direction, uint64_t &moved) {
    switch (direction) {
    case 0U:
        moved = BoardMover::move_left(board);
        break;
    case 1U:
        moved = BoardMover::move_right(board);
        break;
    case 2U:
        moved = BoardMover::move_up(board);
        break;
    case 3U:
        moved = BoardMover::move_down(board);
        break;
    default:
        throw std::invalid_argument("test move direction out of range");
    }
    return moved != board;
}

std::vector<uint8_t> make_position_bytes(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &boards
) {
    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    for (uint64_t board : boards) {
        const auto encoded =
            BC::encode_canonical_board_position(lut, axis, canonicalize(board));
        check(encoded.valid, "test source board should encode");
        if (!builders[encoded.cid]) {
            builders[encoded.cid] = std::make_unique<BCCellBuilder>(lut);
        }
        builders[encoded.cid]->insert(encoded.key, encoded.rank);
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (builders[cid]) {
            writer.write_cell(cid, builders[cid]->finalize());
        } else {
            writer.mark_empty_cell(cid);
        }
    }
    return writer.finish_layer();
}

std::vector<uint8_t> make_position_bytes_for_cell(
    const BCLut &lut,
    const BCFamilyTable &axis,
    CellId cid,
    uint64_t canonical_board
) {
    const BCCellMatrix matrix(axis);
    check(cid < matrix.cell_count(), "manual source cell id out of range");
    const auto encoded = BC::encode_canonical_board_position(lut, axis, canonical_board);
    check(encoded.valid, "manual source board should encode");
    BCCellBuilder builder(lut);
    builder.insert(encoded.key, encoded.rank);

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cell = 0U; cell < matrix.cell_count(); ++cell) {
        if (cell == cid) {
            writer.write_cell(cell, builder.finalize());
        } else {
            writer.mark_empty_cell(cell);
        }
    }
    return writer.finish_layer();
}

std::set<Candidate> collect_candidates(const BCPositionLayerReader &reader) {
    std::set<Candidate> out;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        BCPositionCellScanner(reader, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                out.insert(Candidate{cid, entry.key, entry.rank});
            }
        );
    }
    return out;
}

std::filesystem::path temp_path(const char *name) {
    static uint32_t counter = 0U;
    return std::filesystem::temp_directory_path() /
        ("bc_family_generation_state_" + std::to_string(++counter) + "_" + name);
}

void cleanup_file(const std::filesystem::path &path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

bool direct_io_tests_enabled() {
    const char *value = std::getenv("BC_RUN_DIRECT_IO_TESTS");
    return value != nullptr && std::string(value) == "1";
}

uint32_t direct_blob_queue_depth() {
    const char *value = std::getenv("BC_DIRECT_BLOB_QD");
    if (value == nullptr || *value == '\0') {
        return 8U;
    }
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    return parsed == 0UL ? 1U : static_cast<uint32_t>(parsed);
}

BC::FinalizedCellPayload make_payload(
    const BCLut &lut,
    const std::vector<BCEncodedKeyRank> &encoded
) {
    BCCellBuilder builder(lut);
    for (const BCEncodedKeyRank &item : encoded) {
        check(item.valid, "test encoded payload item should be valid");
        builder.insert(item.key, item.rank);
    }
    return builder.finalize();
}

[[nodiscard]] std::vector<uint8_t> run_toy_family_generation(
    const BCLut &lut,
    const BCPositionLayerReader &source,
    const BCFamilyTable &target_axis,
    uint8_t spawn_tile_rank,
    BC::SpawnDeltaCoord delta_coord
) {
    BCFamilyGenerationScheduler scheduler(source.axis(), target_axis);
    BCGenerationBlobIO blob;
    BCFamilyMutableStore store(lut, target_axis, blob);
    const BCCellMatrix target_matrix(target_axis);

    for (BC::FamilyId source_id = 0U; source_id < source.axis().family_count(); ++source_id) {
        const BCFamilyGenerationPass pass =
            scheduler.make_pass(source_id, delta_coord, spawn_tile_rank);
        const std::vector<CellId> need = scheduler.target_need_cells(pass);
        std::set<CellId> need_set(need.begin(), need.end());
        store.prepare_target_window_for_families(pass.target_families);
        check(pass.target_families.size() <= 2U, "target fanout must be <=2 families");

        for (const BCSourceCellWork &work : scheduler.source_cells(pass)) {
            BCPositionCellScanner(source, work.cid).for_each_board(
                [&](const BC::BCScannedBoardEntry &entry) {
                    const BC::BCEmptyCells empties = BC::enumerate_empty_cells(entry.board);
                    for (uint8_t empty_i = 0U; empty_i < empties.count; ++empty_i) {
                        const uint64_t spawned =
                            BC::spawn_tile(entry.board, empties.cells[empty_i], spawn_tile_rank);
                        const uint32_t begin_dir = BC::bc_has_horizontal(work.directions) ? 0U : 2U;
                        const uint32_t end_dir = BC::bc_has_vertical(work.directions) ? 4U : 2U;
                        for (uint32_t direction = begin_dir; direction < end_dir; ++direction) {
                            if (direction >= 2U && !BC::bc_has_vertical(work.directions)) {
                                continue;
                            }
                            if (direction < 2U && !BC::bc_has_horizontal(work.directions)) {
                                continue;
                            }
                            uint64_t moved = 0U;
                            if (!move_one(spawned, direction, moved)) {
                                continue;
                            }
                            const uint64_t canonical = canonicalize(moved);
                            const auto encoded =
                                BC::encode_canonical_board_position(lut, target_axis, canonical);
                            check(encoded.valid, "toy family candidate should encode");
                            check(
                                need_set.find(encoded.cid) != need_set.end(),
                                "toy family candidate escaped target fanout window"
                            );
                            (void)store.get_or_create(encoded.cid).insert(encoded.key, encoded.rank);
                        }
                    }
                }
            );
        }

        std::vector<CellId> keep;
        if (source_id + 1U < source.axis().family_count()) {
            keep = scheduler.target_need_cells_for_source(
                static_cast<BC::FamilyId>(source_id + 1U),
                delta_coord
            );
        }
        store.release_except(keep);
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(target_axis);
    for (CellId cid = 0U; cid < target_matrix.cell_count(); ++cid) {
        const BC::FinalizedCellPayload payload = store.finalize_cell(cid);
        if (payload.success_rows == 0U && payload.buckets.empty() && payload.rank_payload.empty()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, payload);
    }
    return writer.finish_layer();
}

void test_scheduler() {
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    BCFamilyGenerationScheduler scheduler(source_axis, target_axis);
    const BCFamilyGenerationPass pass = scheduler.make_pass(1U, 1U, 1U);
    check(pass.source_coord == 1U, "scheduler source coord mismatch");
    check(pass.target_families.size() == 2U, "scheduler target fanout should have two families");

    const std::vector<BCSourceCellWork> source_cells = scheduler.source_cells(pass);
    check(source_cells.size() == 5U, "source family view should contain 2F-1 cells");
    const BCCellMatrix source_matrix(source_axis);
    bool saw_diag = false;
    bool saw_row = false;
    bool saw_col = false;
    for (const BCSourceCellWork &work : source_cells) {
        const BC::FamilyId row = source_matrix.row(work.cid);
        const BC::FamilyId col = source_matrix.col(work.cid);
        if (row == 1U && col == 1U) {
            saw_diag = work.directions == BC::BCDirectionMask::Both;
        } else if (row == 1U) {
            saw_row = saw_row || work.directions == BC::BCDirectionMask::Horizontal;
        } else if (col == 1U) {
            saw_col = saw_col || work.directions == BC::BCDirectionMask::Vertical;
        }
    }
    check(saw_diag && saw_row && saw_col, "source direction masks are incorrect");

    const std::vector<CellId> need = scheduler.target_need_cells(pass);
    std::set<CellId> need_set(need.begin(), need.end());
    BC::BCCellSetBuilder brute(scheduler.target_matrix());
    brute.begin_epoch();
    brute.add_family_crosses(pass.target_families);
    check(
        need_set == std::set<CellId>(brute.cells().begin(), brute.cells().end()),
        "target NeedCells must equal union of fanout crosses"
    );

}

void test_mutable_store_lifecycle() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const BCCellMatrix matrix(axis);
    BCGenerationBlobIO blob;
    BCFamilyMutableStore store(lut, axis, blob);
    const CellId cid = matrix.cid(0U, 1U);
    const BCEncodedKeyRank first = BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0000U, 0x0000U);
    const BCEncodedKeyRank second = BC::encode_key_and_rank(lut, 0x0001U, 0x0000U, 0x0000U, 0x0000U);
    check(first.valid && second.valid, "test encoded keys should be valid");

    BC::BCCellMutableBuilder direct_builder(lut, cid);
    (void)direct_builder.insert_encoded(first);
    check(direct_builder.finalize_count_for_testing() == 0U, "mutable builder should not finalize during insert");
    (void)direct_builder.dump(99U);
    check(direct_builder.finalize_count_for_testing() == 0U, "dump must not call finalize");

    store.prepare_target_window_for_families(family_ids({0U}));
    (void)store.insert_encoded(cid, first);
    check(store.resident_cell_count() == 1U, "resident list should track created cell");
    check(store.resident_cells_for_testing().size() == 1U, "resident cell vector size mismatch");
    store.release_except(std::vector<CellId>{cid});
    check(store.state(cid) == BCMutableCellState::Resident, "immediate reuse should keep resident");
    check(store.resident_cell_count() == 1U, "immediate reuse should remain resident");
    check(blob.append_count() == 0U, "immediate reuse should not dump");

    store.release_except({});
    check(store.state(cid) == BCMutableCellState::Dumped, "dirty release should dump cell");
    check(store.resident_cell_count() == 0U, "dumped release should clear resident list");
    check(blob.append_count() == 1U, "dirty release should append one dump");

    store.prepare_target_window_for_families(family_ids({0U}));
    check(store.state(cid) == BCMutableCellState::Resident, "prepare should reload dumped cell");
    check(store.resident_cell_count() == 1U, "reload should add resident cell");
    check(store.stats().reloaded_builders == 1U, "reload count mismatch");
    check(blob.append_count() == 1U, "reload should not append a new dump");
    store.release_except({});
    check(blob.append_count() == 1U, "clean release should not append another dump");
    check(store.resident_cell_count() == 0U, "clean release should clear resident list");

    store.prepare_target_window_for_families(family_ids({0U}));
    (void)store.insert_encoded(cid, second);
    store.release_except({});
    check(blob.append_count() == 2U, "dirty reload release should append new dump");

    const BC::FinalizedCellPayload payload = store.finalize_cell(cid);
    check(store.state(cid) == BCMutableCellState::Finalized, "finalize state mismatch");
    check(payload.lookup(lut, first.key, first.rank).found, "finalized payload missing first entry");
    check(payload.lookup(lut, second.key, second.rank).found, "finalized payload missing second entry");
    expect_throws(
        [&] { store.prepare_target_window_for_cells_for_testing(std::vector<CellId>{cid}); },
        "finalized cell should reject prepare"
    );
    expect_throws(
        [&] { store.prepare_target_window_for_families(std::vector<BC::FamilyId>{0U, 1U, 2U, 3U}); },
        "more than three target families should be rejected"
    );
}

void test_file_blob_roundtrip() {
    const BCLut lut(test_alphabet());
    const CellId cid0 = 3U;
    const CellId cid1 = 7U;
    BC::BCCellMutableBuilder builder0(lut, cid0);
    BC::BCCellMutableBuilder builder1(lut, cid1);
    const BCEncodedKeyRank a = BC::encode_key_and_rank(lut, 0x0001U, 0x0000U, 0x0000U, 0x0000U);
    const BCEncodedKeyRank b = BC::encode_key_and_rank(lut, 0x0011U, 0x0000U, 0x0000U, 0x0000U);
    check(a.valid && b.valid, "blob test encoded keys should be valid");
    (void)builder0.insert_encoded(a);
    (void)builder1.insert_encoded(b);
    const auto dump0 = builder0.dump(1U);
    const auto dump1 = builder1.dump(2U);

    const std::filesystem::path blob_path = temp_path("blob.bin");
    cleanup_file(blob_path);
    {
        BC::BCBufferedFileWriter writer(blob_path);
        BCFileGenerationBlobIO blob(writer);
        const BC::BCDumpRef ref0 = blob.append_cell_dump(dump0);
        const BC::BCDumpRef ref1 = blob.append_cell_dump(dump1);
        blob.flush_pending_appends();
        writer.flush();
        BC::BCBufferedFileReader reader(blob_path);
        blob.set_reader(reader);
        BC::BCGenerationBlobIOStats stats;
        const auto buffers = blob.read_many({{cid1, ref1}, {cid0, ref0}}, &stats);
        check(buffers.size() == 2U, "file blob read_many size mismatch");
        auto restored1 = BC::BCCellMutableBuilder::restore(lut, cid1, buffers[0].view());
        auto restored0 = BC::BCCellMutableBuilder::restore(lut, cid0, buffers[1].view());
        check(restored0->contains(a.key, a.rank), "file blob restored first dump missing rank");
        check(restored1->contains(b.key, b.rank), "file blob restored second dump missing rank");
        check(stats.coalesced_extents <= stats.requested_extents, "file blob coalescing stats invalid");
    }
    cleanup_file(blob_path);
}

void test_direct_file_blob_restore_roundtrip() {
#if defined(_WIN32) || defined(__linux__)
    if (!direct_io_tests_enabled()) {
        return;
    }
    const BCLut lut(test_alphabet());
    const std::vector<BCEncodedKeyRank> encoded{
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0001U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0011U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0101U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x1111U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0001U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0001U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0000U, 0x0001U),
    };
    for (const BCEncodedKeyRank &item : encoded) {
        check(item.valid, "direct blob test encoded item should be valid");
    }
    const std::vector<BCEncodedKeyRank> mutation_encoded{
        BC::encode_key_and_rank(lut, 0x0111U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x1011U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x1101U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x1110U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0012U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0021U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0201U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x2001U, 0x0000U, 0x0000U, 0x0000U),
    };
    for (const BCEncodedKeyRank &item : mutation_encoded) {
        check(item.valid, "direct blob mutation item should be valid");
    }

    struct ExpectedCell {
        CellId cid = 0U;
        std::vector<BCEncodedKeyRank> items;
    };
    std::vector<ExpectedCell> expected_cells;
    expected_cells.reserve(96U);
    for (uint32_t i = 0U; i < 96U; ++i) {
        ExpectedCell cell;
        cell.cid = static_cast<CellId>(1000U + i * 7U);
        const uint32_t count = 1U + (i % encoded.size());
        for (uint32_t j = 0U; j < count; ++j) {
            cell.items.push_back(encoded[(i + j * 3U) % encoded.size()]);
        }
        expected_cells.push_back(std::move(cell));
    }

    const std::filesystem::path blob_path = temp_path("direct_blob.bin");
    cleanup_file(blob_path);
    {
        BC::BCDirectFileIOOptions write_options;
        write_options.queue_depth = direct_blob_queue_depth();
        write_options.overlapped = write_options.queue_depth > 1U;
        write_options.preserve_unwritten_bytes = false;
        BC::BCDirectFileWriter writer(blob_path, write_options);
        BCFileGenerationBlobIO blob(writer);

        std::vector<std::pair<CellId, BC::BCDumpRef>> refs;
        refs.reserve(expected_cells.size());
        uint32_t generation = 1U;
        for (const ExpectedCell &cell : expected_cells) {
            BC::BCCellMutableBuilder builder(lut, cell.cid);
            builder.reserve(32U, 128U);
            for (const BCEncodedKeyRank &item : cell.items) {
                (void)builder.insert_encoded(item);
            }
            refs.emplace_back(cell.cid, blob.append_cell_dump(builder.dump(generation++)));
        }
        blob.flush_pending_appends();
        writer.flush();

        BC::BCDirectFileIOOptions read_options;
        read_options.queue_depth = direct_blob_queue_depth();
        read_options.overlapped = read_options.queue_depth > 1U;
        BC::BCDirectFileReader reader(blob_path, read_options);
        blob.set_reader(reader);

        std::vector<std::pair<CellId, BC::BCDumpRef>> shuffled_refs;
        shuffled_refs.reserve(refs.size());
        for (size_t i = 0U; i < refs.size(); ++i) {
            shuffled_refs.push_back(refs[(i * 37U) % refs.size()]);
        }
        BC::BCGenerationBlobIOStats stats;
        const auto restored = blob.restore_many_builders(lut, shuffled_refs, &stats);
        check(restored.size() == shuffled_refs.size(), "direct blob restored builder count mismatch");
        std::vector<std::pair<CellId, BC::BCDumpRef>> mutated_refs;
        mutated_refs.reserve(restored.size());
        for (size_t i = 0U; i < restored.size(); ++i) {
            const CellId cid = shuffled_refs[i].first;
            auto expected_it = std::find_if(
                expected_cells.begin(),
                expected_cells.end(),
                [&](const ExpectedCell &cell) { return cell.cid == cid; }
            );
            check(expected_it != expected_cells.end(), "direct blob restored unexpected cid");
            for (const BCEncodedKeyRank &item : expected_it->items) {
                check(
                    restored[i].builder->contains(item.key, item.rank),
                    "direct blob restored builder missing rank"
                );
            }
            const BCEncodedKeyRank extra = mutation_encoded[i % mutation_encoded.size()];
            (void)restored[i].builder->insert_encoded(extra);
            expected_it->items.push_back(extra);
            mutated_refs.emplace_back(
                cid,
                blob.append_cell_dump_streamed(*restored[i].builder, generation++)
            );
        }
        check(stats.backend_read_ops != 0U, "direct blob restore should report backend reads");
        check(stats.backend_read_bytes >= stats.bytes_read, "direct blob backend read bytes should include alignment");

        blob.flush_pending_appends();
        writer.flush();
        std::vector<std::pair<CellId, BC::BCDumpRef>> reshuffled_mutated_refs;
        reshuffled_mutated_refs.reserve(mutated_refs.size());
        for (size_t i = 0U; i < mutated_refs.size(); ++i) {
            reshuffled_mutated_refs.push_back(mutated_refs[(i * 53U) % mutated_refs.size()]);
        }
        BC::BCGenerationBlobIOStats mutated_stats;
        const auto mutated = blob.restore_many_builders(lut, reshuffled_mutated_refs, &mutated_stats);
        check(mutated.size() == reshuffled_mutated_refs.size(), "direct blob mutated restore count mismatch");
        for (size_t i = 0U; i < mutated.size(); ++i) {
            const CellId cid = reshuffled_mutated_refs[i].first;
            const auto expected_it = std::find_if(
                expected_cells.begin(),
                expected_cells.end(),
                [&](const ExpectedCell &cell) { return cell.cid == cid; }
            );
            check(expected_it != expected_cells.end(), "direct blob mutated restore unexpected cid");
            for (const BCEncodedKeyRank &item : expected_it->items) {
                check(
                    mutated[i].builder->contains(item.key, item.rank),
                    "direct blob mutated builder missing rank"
                );
            }
        }
        check(mutated_stats.backend_read_ops != 0U, "direct blob mutated restore should report backend reads");
    }
    cleanup_file(blob_path);
#endif
}

void test_concurrent_mutable_builder() {
    const BCLut lut(test_alphabet());
    BC::BCCellMutableBuilder builder(lut, 5U);
    builder.reserve(64U, 256U);
    const std::vector<BCEncodedKeyRank> encoded{
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0001U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0011U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0101U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x1111U, 0x0000U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0001U, 0x0000U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0001U, 0x0000U),
        BC::encode_key_and_rank(lut, 0x0000U, 0x0000U, 0x0000U, 0x0001U),
    };
    for (const BCEncodedKeyRank &item : encoded) {
        check(item.valid, "concurrent builder encoded item should be valid");
    }

    constexpr uint32_t kThreads = 8U;
    constexpr uint32_t kIters = 2000U;
    std::vector<std::thread> threads;
    for (uint32_t t = 0U; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (uint32_t i = 0U; i < kIters; ++i) {
                const BCEncodedKeyRank &item = encoded[(i + t) % encoded.size()];
                (void)builder.insert_encoded(item);
            }
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
    const BC::FinalizedCellPayload payload = builder.finalize();
    check(payload.success_rows == encoded.size(), "concurrent duplicate insert should dedup ranks");
    for (const BCEncodedKeyRank &item : encoded) {
        check(payload.lookup(lut, item.key, item.rank).found, "concurrent builder finalized payload missing rank");
    }

    const uint64_t finalize_count = builder.finalize_count_for_testing();
    const BC::BCCellBuilderDump dump = builder.dump(1U);
    check(builder.finalize_count_for_testing() == finalize_count, "dump should not increase finalize_count");
    auto restored = BC::BCCellMutableBuilder::restore(lut, 5U, BC::BCCellBuilderDumpView{
        dump.cid,
        dump.dump_generation,
        dump.bytes.data(),
        static_cast<uint32_t>(dump.bytes.size())
    });
    std::vector<std::thread> more_threads;
    for (uint32_t t = 0U; t < kThreads; ++t) {
        more_threads.emplace_back([&, t] {
            for (uint32_t i = 0U; i < 256U; ++i) {
                const BCEncodedKeyRank &item = encoded[(encoded.size() - 1U - ((i + t) % encoded.size()))];
                (void)restored->insert_encoded(item);
            }
        });
    }
    for (std::thread &thread : more_threads) {
        thread.join();
    }
    const BC::FinalizedCellPayload restored_payload = restored->finalize();
    check(restored_payload.success_rows == encoded.size(), "restored concurrent builder should preserve dedup");
}

void test_concurrent_mutable_builder_grow_stress() {
    const BCLut lut(test_alphabet());
    BC::BCCellMutableBuilder builder(lut, 17U);

    std::vector<BCEncodedKeyRank> encoded;
    encoded.reserve(4096U);
    std::set<std::pair<uint64_t, uint16_t>> oracle;
    for (uint32_t word = 0U; word <= 0xFFFFU && encoded.size() < 4096U; ++word) {
        const uint16_t nw = static_cast<uint16_t>(word);
        if (!lut.word_desc(nw).valid) {
            continue;
        }
        BCEncodedKeyRank item = BC::encode_key_and_rank(lut, nw, 0x0000U, 0x0000U, 0x0000U);
        if (!item.valid) {
            continue;
        }
        if (oracle.insert({item.key, item.rank}).second) {
            encoded.push_back(item);
        }
    }
    check(encoded.size() == 4096U, "grow stress should build enough unique encoded positions");

    constexpr uint32_t kThreads = 16U;
    std::vector<std::thread> threads;
    for (uint32_t t = 0U; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (uint32_t i = t; i < encoded.size(); i += kThreads) {
                (void)builder.insert_encoded(encoded[i]);
            }
            for (uint32_t i = t; i < encoded.size(); i += kThreads * 4U) {
                (void)builder.insert_encoded(encoded[i]);
            }
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }

    check(builder.hash_grow_count() != 0U, "grow stress should trigger hash table growth");
    const BC::FinalizedCellPayload payload = builder.finalize();
    check(payload.success_rows == oracle.size(), "concurrent grow stress should preserve unique ranks");
    for (const BCEncodedKeyRank &item : encoded) {
        check(payload.lookup(lut, item.key, item.rank).found, "concurrent grow stress missing inserted rank");
    }
}

void test_mutable_builder_source_is_concurrent_backend() {
    std::filesystem::path source_path = std::filesystem::path(__FILE__).parent_path() /
        ".." / "include" / "BCCellMutableBuilder.h";
    source_path = source_path.lexically_normal();
    std::ifstream in(source_path);
    check(in.good(), "test should open actual BCCellMutableBuilder.h source");
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    auto has = [&](const char *needle) {
        return text.find(needle) != std::string::npos;
    };
    check(has("struct SlotArrays"), "mutable builder should use SoA slot arrays");
    check(has("std::unique_ptr<std::atomic<uint8_t>[]> states"), "mutable builder should split slot state into a hot array");
    check(has("BCCellAlignedBytePtr metadata_storage"), "mutable builder should keep slot metadata in one flat storage block");
    check(has("uint64_t *keys"), "mutable builder should expose keys as a flat SoA pointer");
    check(has("uint32_t *bitmap_offsets"), "mutable builder should store bitmap offsets separately");
    check(has("BCCellBitmapArenaPtr bitmap_arena_"), "mutable builder should use an aligned flat bitmap arena");
    check(has("bc_allocate_cell_bitmap_arena"), "mutable builder should allocate bitmap arena through the aligned allocator");
    check(has("__atomic_fetch_or") || has("_InterlockedOr64"), "mutable builder insert should use atomic fetch_or on raw bitmap words");
    check(has("grow_hash_table"), "mutable builder should have explicit flat-table resize/retry");
    check(has("capacity_for_bucket_count"), "mutable builder should size the flat table from predicted buckets");
    check(has("kLoadNumerator = 3U"), "mutable builder production load threshold should be 0.75");
    check(has("kDumpVersion = 1U"), "mutable builder dump should have one current transient format version");
    check(!has("kDumpCompactVersion"), "mutable builder should not keep old dump format versions");
    check(!has("kDumpRawSnapshotVersion"), "mutable builder should not keep old raw snapshot dump versions");
    check(!has("kDumpCompactNoMetaVersion"), "mutable builder should not keep old no-meta dump versions");
    check(!has("kDumpRawArenaVersion"), "mutable builder should not keep retired raw-arena version names");
    check(!has("struct BucketSlot"), "mutable builder should not use AoS bucket slots");
    check(!has("BitmapBlock"), "mutable builder should not store per-slot bitmap block pointers");
    check(!has("struct HashSegment"), "mutable builder should not use the retired segment table");
    check(!has("std::vector<BucketSlot> table_"), "mutable builder must not use movable vector table");
    check(!has("bitmap_arena_.resize"), "mutable builder must not use resizable bitmap_arena vector");
    check(!has("++live_rows_"), "mutable builder must not hot-update live_rows_");
    check(!has("bitmap[word] |="), "mutable builder must not use non-atomic bitmap OR");
    check(!has("void rehash("), "mutable builder should use explicit grow_hash_table retry, not old rehash");
}

void test_family_position_writer() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const BCCellMatrix matrix(axis);
    std::vector<BC::FinalizedCellPayload> payloads(matrix.cell_count());
    const CellId cid0 = matrix.cid(1U, 0U);
    const CellId cid1 = matrix.cid(0U, 2U);
    payloads[cid0] = make_payload(
        lut,
        {
            BC::encode_key_and_rank(lut, 0x0001U, 0x0000U, 0x0000U, 0x0000U),
            BC::encode_key_and_rank(lut, 0x0011U, 0x0000U, 0x0000U, 0x0000U),
        }
    );
    payloads[cid1] = make_payload(
        lut,
        {
            BC::encode_key_and_rank(lut, 0x0101U, 0x0000U, 0x0000U, 0x0000U),
        }
    );

    BCPositionLayerWriter memory_writer;
    memory_writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (payloads[cid].buckets.empty()) {
            memory_writer.mark_empty_cell(cid);
        } else {
            memory_writer.write_cell(cid, payloads[cid]);
        }
    }
    const BCPositionLayerReader memory_reader(memory_writer.finish_layer(), lut);

    const std::filesystem::path final_path = temp_path("position.bcpos");
    const std::filesystem::path spool_path = temp_path("position_rank.spool");
    cleanup_file(final_path);
    cleanup_file(spool_path);
    {
        BC::BCBufferedFileWriter final_file(final_path);
        BC::BCBufferedFileWriter spool_file(spool_path);
        BC::BCFamilyPositionWriter writer;
        expect_throws(
            [&] {
                BC::BCFamilyPositionWriter rejected;
                rejected.begin_layer(
                    final_file,
                    spool_file,
                    axis,
                    BC::BCFamilyPositionWriterOptions{1024U, false}
                );
            },
            "family position writer should reject non-preserving direct-style backend"
        );
        writer.begin_layer(final_file, spool_file, axis);
        writer.write_empty_cell(0U);
        writer.write_finalized_cell(cid1, payloads[cid1]);
        writer.write_finalized_cell(cid0, payloads[cid0]);
        for (CellId cid = 1U; cid < matrix.cell_count(); ++cid) {
            if (cid == cid0 || cid == cid1) {
                continue;
            }
            writer.write_empty_cell(cid);
        }
        writer.flush_pending_streams_for_reader();
        BC::BCBufferedFileReader spool_reader(spool_path);
        const uint64_t logical_size = writer.finish_layer(spool_reader);
        final_file.flush();
        check(logical_size > 0U, "family position writer logical size should be non-zero");
    }
    const std::vector<uint8_t> bytes = BC::read_position_layer_from_file(final_path);
    const BCPositionLayerReader family_reader(bytes, lut);
    check(
        collect_candidates(family_reader) == collect_candidates(memory_reader),
        "family position writer output should match memory position writer"
    );
    cleanup_file(final_path);
    cleanup_file(spool_path);

    const std::filesystem::path rank_first_path = temp_path("position_rank_first.bcpos");
    const std::filesystem::path bucket_spool_path = temp_path("position_bucket.spool");
    cleanup_file(rank_first_path);
    cleanup_file(bucket_spool_path);
    {
        BC::BCBufferedFileWriter final_file(rank_first_path);
        BC::BCBufferedFileWriter bucket_spool_file(bucket_spool_path);
        BC::BCFamilyPositionWriter writer;
        BC::BCFamilyPositionWriterOptions options;
        options.staging_bytes = 1024U;
        options.backend_preserves_unaligned_positioned_writes = false;
        options.rank_first_direct_layout = true;
        writer.begin_layer(final_file, bucket_spool_file, axis, options);
        writer.write_empty_cell(0U);
        writer.write_finalized_cell(cid1, payloads[cid1]);
        writer.write_finalized_cell(cid0, payloads[cid0]);
        for (CellId cid = 1U; cid < matrix.cell_count(); ++cid) {
            if (cid == cid0 || cid == cid1) {
                continue;
            }
            writer.write_empty_cell(cid);
        }
        writer.flush_pending_streams_for_reader();
        BC::BCBufferedFileReader bucket_spool_reader(bucket_spool_path);
        const uint64_t logical_size = writer.finish_layer(bucket_spool_reader);
        final_file.flush();
        check(logical_size > 0U, "rank-first family position writer logical size should be non-zero");
    }
    const std::vector<uint8_t> rank_first_bytes = BC::read_position_layer_from_file(rank_first_path);
    const BCPositionLayerReader rank_first_reader(rank_first_bytes, lut);
    check(
        collect_candidates(rank_first_reader) == collect_candidates(memory_reader),
        "rank-first family position writer output should match memory position writer"
    );
    cleanup_file(rank_first_path);
    cleanup_file(bucket_spool_path);
}

std::unique_ptr<BC::BCPositionStreamingReader> make_streaming_reader(
    const BCLut &lut,
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    cleanup_file(path);
    BC::write_position_layer_to_file(path, bytes);
    return std::make_unique<BC::BCPositionStreamingReader>(
        std::make_unique<BC::BCBufferedFileReader>(path),
        lut
    );
}

std::set<Candidate> run_family_generator_to_candidates(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource *source4,
    const BCFamilyStreamingGenerationSource &source2,
    BC::BCFamilyGenerationStats *stats_out = nullptr,
    const BCFamilyGenerationOptions *options_override = nullptr
) {
    const std::filesystem::path final_path = temp_path("family_gen.bcpos");
    const std::filesystem::path spool_path = temp_path("family_gen_rank.spool");
    const std::filesystem::path blob_path = temp_path("family_gen_blob.bin");
    cleanup_file(final_path);
    cleanup_file(spool_path);
    cleanup_file(blob_path);
    {
        BC::BCBufferedFileWriter final_file(final_path);
        BC::BCBufferedFileWriter spool_file(spool_path);
        BC::BCBufferedFileWriter blob_file(blob_path);
        BC::BCBufferedFileReader blob_reader(blob_path);
        BC::BCFileGenerationBlobIO blob(blob_file);
        blob.set_reader(blob_reader);
        BCFamilyMutableStore store(lut, target_axis, blob);
        BC::BCFamilyPositionWriter writer;
        writer.begin_layer(final_file, spool_file, target_axis);
        BCFamilyGenerationOptions options;
        options.num_threads = 2;
        options.canonical_batch_size = 3U;
        options.family_partition_policy = BC::BCFamilyPartitionPolicy::exact();
        if (options_override != nullptr) {
            options = *options_override;
        }
        const BC::BCFamilyGenerationStats stats = BC::generate_family_position_layer_v1(
            lut,
            target_axis,
            source4,
            source2,
            store,
            writer,
            options
        );
        check(stats.target_cells_finalized == target_axis.family_count() * target_axis.family_count(),
            "Family generator should finalize every target cell exactly once");
        if (stats_out != nullptr) {
            *stats_out = stats;
        }
        writer.flush_pending_streams_for_reader();
        BC::BCBufferedFileReader spool_reader(spool_path);
        (void)writer.finish_layer(spool_reader);
        final_file.flush();
    }
    const std::vector<uint8_t> family_bytes = BC::read_position_layer_from_file(final_path);
    const BCPositionLayerReader family_reader(family_bytes, lut);
    std::set<Candidate> out = collect_candidates(family_reader);
    cleanup_file(final_path);
    cleanup_file(spool_path);
    cleanup_file(blob_path);
    return out;
}

void test_parallel_family_generator_matches_resident() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source2_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const std::vector<uint64_t> boards2{
        make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
        make_board({{0U, 2U}, {15U, 2U}}),
    };
    const std::vector<uint8_t> source2_bytes = make_position_bytes(lut, source2_axis, boards2);
    const BCPositionLayerReader source2_memory(source2_bytes, lut);
    const std::filesystem::path source2_path = temp_path("family_source2.bcpos");
    auto source2_stream = make_streaming_reader(lut, source2_path, source2_bytes);
    const BCFamilyStreamingGenerationSource source2{
        source2_stream.get(),
        nullptr,
        1U,
        1U
    };

    BCResidentGenerationOptions resident_options;
    resident_options.num_threads = 1;
    resident_options.canonical_batch_size = 4U;
    const auto resident = BC::generate_resident_position_layer(
        lut,
        target_axis,
        std::vector<BCResidentGenerationSource>{BCResidentGenerationSource{&source2_memory, 1U, 1U}},
        resident_options
    );
    const BCPositionLayerReader resident_reader(resident.position_bytes, lut);
    check(
        run_family_generator_to_candidates(lut, target_axis, nullptr, source2) ==
            collect_candidates(resident_reader),
        "parallel FamilyGenerator +2-only output should match Resident generation"
    );
    cleanup_file(source2_path);
}

void test_parallel_family_generator_combined_matches_resident() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source4_axis = BCFamilyTable::from_range(6U, 2U, 0U, 1U);
    const BCFamilyTable source2_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const std::vector<uint64_t> boards4{
        make_board({{0U, 2U}, {15U, 1U}}),
    };
    const std::vector<uint64_t> boards2{
        make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
        make_board({{0U, 2U}, {15U, 2U}}),
    };
    const std::vector<uint8_t> source4_bytes = make_position_bytes(lut, source4_axis, boards4);
    const std::vector<uint8_t> source2_bytes = make_position_bytes(lut, source2_axis, boards2);
    const BCPositionLayerReader source4_memory(source4_bytes, lut);
    const BCPositionLayerReader source2_memory(source2_bytes, lut);
    const std::filesystem::path source4_path = temp_path("family_source4.bcpos");
    const std::filesystem::path source2_path = temp_path("family_source2_combined.bcpos");
    auto source4_stream = make_streaming_reader(lut, source4_path, source4_bytes);
    auto source2_stream = make_streaming_reader(lut, source2_path, source2_bytes);
    const BCFamilyStreamingGenerationSource source4{source4_stream.get(), nullptr, 2U, 2U};
    const BCFamilyStreamingGenerationSource source2{source2_stream.get(), nullptr, 1U, 1U};

    BCResidentGenerationOptions resident_options;
    resident_options.num_threads = 1;
    resident_options.canonical_batch_size = 4U;
    const auto resident = BC::generate_resident_position_layer(
        lut,
        target_axis,
        std::vector<BCResidentGenerationSource>{
            BCResidentGenerationSource{&source4_memory, 2U, 2U},
            BCResidentGenerationSource{&source2_memory, 1U, 1U}
        },
        resident_options
    );
    const BCPositionLayerReader resident_reader(resident.position_bytes, lut);
    check(
        run_family_generator_to_candidates(lut, target_axis, &source4, source2) ==
            collect_candidates(resident_reader),
        "parallel FamilyGenerator +4/+2 output should match Resident generation"
    );
    cleanup_file(source4_path);
    cleanup_file(source2_path);
}

void test_family_generator_direction_masks_generate_candidates() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const BCCellMatrix source_matrix(source_axis);
    const uint64_t board = canonicalize(make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}));

    auto run_for_cid = [&](CellId cid) {
        const std::vector<uint8_t> bytes = make_position_bytes_for_cell(lut, source_axis, cid, board);
        const std::filesystem::path path = temp_path("family_selective_source.bcpos");
        auto stream = make_streaming_reader(lut, path, bytes);
        const BCFamilyStreamingGenerationSource source{stream.get(), nullptr, 1U, 1U};
        BCFamilyGenerationOptions options;
        options.num_threads = 2;
        options.canonical_batch_size = 3U;
        options.family_partition_policy = BC::BCFamilyPartitionPolicy::exact();
        const std::set<Candidate> generated =
            run_family_generator_to_candidates(lut, target_axis, nullptr, source, nullptr, &options);
        cleanup_file(path);
        return generated;
    };

    const std::set<Candidate> offdiag = run_for_cid(source_matrix.cid(0U, 1U));
    check(!offdiag.empty(), "off-diagonal source cells should generate candidates");

    const std::set<Candidate> diagonal = run_for_cid(source_matrix.cid(0U, 0U));
    check(!diagonal.empty(), "diagonal source cells should generate candidates");
}

void test_family_generator_skips_success_sources() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const std::vector<uint64_t> boards{
        make_board({{0U, 2U}, {15U, 2U}}),
    };
    const std::vector<uint8_t> source_bytes = make_position_bytes(lut, source_axis, boards);
    const std::filesystem::path source_path = temp_path("family_success_source.bcpos");
    auto source_stream = make_streaming_reader(lut, source_path, source_bytes);
    const BCFamilyStreamingGenerationSource source{source_stream.get(), nullptr, 1U, 1U};

    std::vector<uint8_t> shifts;
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }

    BCFamilyGenerationOptions skip_options;
    skip_options.num_threads = 2;
    skip_options.canonical_batch_size = 3U;
    skip_options.family_partition_policy = BC::BCFamilyPartitionPolicy::exact();
    skip_options.success_target_rank = 2;
    skip_options.success_shifts = &shifts;
    skip_options.success_check_min_source_layer_sum = source_axis.layer_sum();

    const std::set<Candidate> skipped =
        run_family_generator_to_candidates(lut, target_axis, nullptr, source, nullptr, &skip_options);
    check(skipped.empty(), "FamilyGenerator should not expand already-successful source boards");

    BCFamilyGenerationOptions no_skip_options = skip_options;
    no_skip_options.success_target_rank = 0;
    const std::set<Candidate> generated =
        run_family_generator_to_candidates(lut, target_axis, nullptr, source, nullptr, &no_skip_options);
    check(!generated.empty(), "FamilyGenerator success-skip test source should generate candidates when disabled");

    cleanup_file(source_path);
}

void test_toy_family_matches_resident() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const std::vector<uint64_t> boards{
        make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
        make_board({{0U, 2U}, {15U, 2U}}),
    };
    const std::vector<uint8_t> source_bytes = make_position_bytes(lut, source_axis, boards);
    const BCPositionLayerReader source_reader(source_bytes, lut);

    BCResidentGenerationOptions options;
    options.num_threads = 1;
    options.canonical_batch_size = 4U;
    options.pending_insert_buffer_size = 4U;
    const auto resident = BC::generate_resident_position_layer(
        lut,
        target_axis,
        std::vector<BCResidentGenerationSource>{BCResidentGenerationSource{&source_reader, 1U, 1U}},
        options
    );
    const BCPositionLayerReader resident_reader(resident.position_bytes, lut);

    const std::vector<uint8_t> family_bytes =
        run_toy_family_generation(lut, source_reader, target_axis, 1U, 1U);
    const BCPositionLayerReader family_reader(family_bytes, lut);
    check(
        collect_candidates(family_reader) == collect_candidates(resident_reader),
        "toy FamilyChain candidate set should match Resident generation"
    );
}

} // namespace

int main() {
    try {
        const bool progress = std::getenv("BC_TEST_PROGRESS") != nullptr;
        const auto run = [&](const char *name, auto &&fn) {
            if (progress) {
                std::cerr << "RUN " << name << '\n';
            }
            fn();
        };
        if (std::getenv("BC_ONLY_DIRECT_BLOB_TEST") != nullptr) {
            test_direct_file_blob_restore_roundtrip();
            std::cout << "bc_family_generation_state_test passed\n";
            return 0;
        }
        run("scheduler", test_scheduler);
        run("mutable_store_lifecycle", test_mutable_store_lifecycle);
        run("file_blob_roundtrip", test_file_blob_roundtrip);
        run("direct_file_blob_restore_roundtrip", test_direct_file_blob_restore_roundtrip);
        run("concurrent_mutable_builder", test_concurrent_mutable_builder);
        run("concurrent_mutable_builder_grow_stress", test_concurrent_mutable_builder_grow_stress);
        run("mutable_builder_source_is_concurrent_backend", test_mutable_builder_source_is_concurrent_backend);
        run("family_position_writer", test_family_position_writer);
        run("parallel_family_generator_matches_resident", test_parallel_family_generator_matches_resident);
        run("parallel_family_generator_combined_matches_resident", test_parallel_family_generator_combined_matches_resident);
        run("family_generator_direction_masks_generate_candidates", test_family_generator_direction_masks_generate_candidates);
        run("family_generator_skips_success_sources", test_family_generator_skips_success_sources);
        run("toy_family_matches_resident", test_toy_family_matches_resident);
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_generation_state_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_family_generation_state_test passed\n";
    return 0;
}
