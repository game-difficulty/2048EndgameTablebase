#include "BCResidentGeneration.h"

#include "BCBoardOps.h"
#include "BCPositionScanner.h"
#include "BoardMover.h"
#include "Calculator.h"

#include <cstdint>
#include <exception>
#include <iostream>
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
using BC::BCQuadrantWords;
using BC::BCResidentGenerationResult;
using BC::BCResidentGenerationSource;
using BC::BucketRank;
using BC::CellId;
using Candidate = std::tuple<CellId, uint64_t, BucketRank>;

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

uint64_t make_board(const std::vector<std::pair<uint8_t, uint8_t>> &tiles) {
    uint64_t board = 0U;
    for (const auto &[cell, tile] : tiles) {
        board = BC::set_board_tile(board, cell, tile);
    }
    return board;
}

[[nodiscard]] uint64_t oracle_canonicalize(uint64_t board) {
    return Calculator::canonical_full(board);
}

[[nodiscard]] bool oracle_move(
    uint64_t board,
    uint32_t direction,
    uint64_t &moved_board
) {
    switch (direction) {
    case 0U:
        moved_board = BoardMover::move_left(board);
        break;
    case 1U:
        moved_board = BoardMover::move_right(board);
        break;
    case 2U:
        moved_board = BoardMover::move_up(board);
        break;
    case 3U:
        moved_board = BoardMover::move_down(board);
        break;
    default:
        throw std::invalid_argument("oracle move direction out of range");
    }
    return moved_board != board;
}

struct TestPositionLayer {
    std::vector<uint8_t> bytes;
    BCPositionLayerReader reader;
    std::vector<uint64_t> canonical_boards;
};

TestPositionLayer write_source_layer(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &boards
) {
    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    TestPositionLayer layer;
    for (uint64_t board : boards) {
        const uint64_t canonical = oracle_canonicalize(board);
        const auto encoded = BC::encode_canonical_board_position(lut, axis, canonical);
        check(encoded.valid, "source board should encode into source axis");
        if (!builders[encoded.cid]) {
            builders[encoded.cid] = std::make_unique<BCCellBuilder>(lut);
        }
        builders[encoded.cid]->insert(encoded.key, encoded.rank);
        layer.canonical_boards.push_back(canonical);
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

struct OracleSource {
    std::vector<uint64_t> source_boards;
    uint8_t spawn_tile_rank = 0U;
};

struct OracleResult {
    std::set<Candidate> candidates;
    uint64_t source_boards_scanned = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_candidates = 0U;
    uint64_t valid_candidates = 0U;
};

OracleResult compute_oracle(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<OracleSource> &sources
) {
    OracleResult out;
    for (const OracleSource &source : sources) {
        for (uint64_t source_board : source.source_boards) {
            ++out.source_boards_scanned;
            const BC::BCEmptyCells empties = BC::enumerate_empty_cells(source_board);
            for (uint32_t empty_i = 0U; empty_i < empties.count; ++empty_i) {
                const uint64_t spawned =
                    BC::spawn_tile(source_board, empties.cells[empty_i], source.spawn_tile_rank);
                ++out.spawned_boards;

                for (uint32_t direction = 0U; direction < 4U; ++direction) {
                    ++out.move_candidates;
                    uint64_t moved = 0U;
                    if (!oracle_move(spawned, direction, moved)) {
                        continue;
                    }
                    const uint64_t canonical = oracle_canonicalize(moved);
                    const auto encoded =
                        BC::encode_canonical_board_position(lut, target_axis, canonical);
                    check(encoded.valid, "oracle candidate should encode into target axis");
                    out.candidates.insert(Candidate{encoded.cid, encoded.key, encoded.rank});
                    ++out.valid_candidates;
                }
            }
        }
    }
    return out;
}

std::set<Candidate> collect_generated_candidates(const BCPositionLayerReader &reader) {
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

uint64_t descriptor_success_rows_sum(const BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

void check_dynamic_stats(const BCResidentGenerationResult &generated) {
    check(generated.generation_retries == 0U, "small resident generation should not retry capacity");
    check(generated.dynamic_hash_capacity != 0U, "resident generation should report dynamic hash capacity");
    check(
        generated.dynamic_bucket_slots_used <= generated.dynamic_hash_capacity,
        "resident generation dynamic bucket slots exceed hash capacity"
    );
    check(generated.dynamic_bucket_slots_used != 0U, "resident generation should report used dynamic buckets");
    check(generated.dynamic_bitmap_words_used != 0U, "resident generation should report used bitmap words");
    check(
        generated.dynamic_bitmap_words_used <= generated.dynamic_bitmap_words_allocated,
        "resident generation used bitmap words exceed allocated words"
    );
    check(
        generated.dynamic_bitmap_words_allocated <= generated.dynamic_bitmap_words_reserved,
        "resident generation allocated bitmap words exceed reserved words"
    );
}

void check_stats_match(
    const BCResidentGenerationResult &generated,
    const OracleResult &oracle
) {
    check(
        generated.source_boards_scanned == oracle.source_boards_scanned,
        "resident generation source board scan count mismatch"
    );
    check(generated.spawned_boards == oracle.spawned_boards, "resident generation spawn count mismatch");
    check(
        generated.move_candidates == oracle.move_candidates,
        "resident generation move candidate count mismatch"
    );
    check(
        generated.valid_candidates == oracle.valid_candidates,
        "resident generation valid candidate count mismatch"
    );
    check_dynamic_stats(generated);
}

void test_single_source_generation_matches_oracle() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const TestPositionLayer source = write_source_layer(
        lut,
        source_axis,
        {
            make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
            make_board({{0U, 2U}, {15U, 2U}}),
        }
    );

    const BCResidentGenerationSource source2{&source.reader, 1U, 1U};
    const BCResidentGenerationResult generated =
        BC::generate_resident_position_layer(lut, target_axis, std::vector<BCResidentGenerationSource>{source2});
    const OracleResult oracle = compute_oracle(
        lut,
        target_axis,
        {OracleSource{source.canonical_boards, 1U}}
    );
    const BCPositionLayerReader generated_reader(generated.position_bytes, lut);
    check(collect_generated_candidates(generated_reader) == oracle.candidates,
        "single-source generated candidates mismatch oracle");
    check(descriptor_success_rows_sum(generated_reader) == oracle.candidates.size(),
        "single-source generated success_rows should match unique oracle");
    check_stats_match(generated, oracle);
}

void test_combined_generation_matches_oracle() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source4_axis = BCFamilyTable::from_range(6U, 2U, 0U, 1U);
    const BCFamilyTable source2_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);

    const TestPositionLayer source4 = write_source_layer(
        lut,
        source4_axis,
        {
            make_board({{0U, 1U}, {1U, 1U}, {2U, 1U}}),
            make_board({{0U, 2U}, {1U, 1U}}),
        }
    );
    const TestPositionLayer source2 = write_source_layer(
        lut,
        source2_axis,
        {
            make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
            make_board({{0U, 2U}, {15U, 2U}}),
        }
    );

    const BCResidentGenerationSource src4{&source4.reader, 2U, 2U};
    const BCResidentGenerationSource src2{&source2.reader, 1U, 1U};
    const BCResidentGenerationResult generated =
        BC::generate_resident_position_layer(lut, target_axis, src4, src2);
    const OracleResult oracle = compute_oracle(
        lut,
        target_axis,
        {
            OracleSource{source4.canonical_boards, 2U},
            OracleSource{source2.canonical_boards, 1U},
        }
    );
    const BCPositionLayerReader generated_reader(generated.position_bytes, lut);
    check(collect_generated_candidates(generated_reader) == oracle.candidates,
        "+4/+2 generated candidates mismatch oracle");
    check(descriptor_success_rows_sum(generated_reader) == oracle.candidates.size(),
        "+4/+2 generated success_rows should match unique oracle");
    check_stats_match(generated, oracle);

    std::cerr
        << "sample_stats source_boards=" << generated.source_boards_scanned
        << " spawned=" << generated.spawned_boards
        << " move_candidates=" << generated.move_candidates
        << " valid_candidates=" << generated.valid_candidates
        << " duplicate_candidates_possible=" << generated.duplicate_candidates_possible
        << " unique_targets=" << oracle.candidates.size()
        << "\n";
}

void test_parallel_and_small_batch_match_scalar_output() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const TestPositionLayer source = write_source_layer(
        lut,
        source_axis,
        {
            make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
            make_board({{0U, 2U}, {15U, 2U}}),
            make_board({{0U, 1U}, {2U, 1U}, {4U, 2U}}),
        }
    );

    const BCResidentGenerationSource source2{&source.reader, 1U, 1U};
    BC::BCResidentGenerationOptions one_thread;
    one_thread.num_threads = 1;
    one_thread.canonical_batch_size = 3U;
    one_thread.pending_insert_buffer_size = 3U;
    one_thread.dynamic_finalize_mode = BC::BCDynamicFinalizeMode::PerCellSort;
    BC::BCResidentGenerationOptions two_threads = one_thread;
    two_threads.num_threads = 2;
    BC::BCResidentGenerationOptions global_finalize = two_threads;
    global_finalize.dynamic_finalize_mode = BC::BCDynamicFinalizeMode::GlobalSort;

    const BCResidentGenerationResult generated_one =
        BC::generate_resident_position_layer(
            lut,
            target_axis,
            std::vector<BCResidentGenerationSource>{source2},
            one_thread
        );
    const BCResidentGenerationResult generated_two =
        BC::generate_resident_position_layer(
            lut,
            target_axis,
            std::vector<BCResidentGenerationSource>{source2},
            two_threads
        );
    const BCResidentGenerationResult generated_global =
        BC::generate_resident_position_layer(
            lut,
            target_axis,
            std::vector<BCResidentGenerationSource>{source2},
            global_finalize
        );

    const BCPositionLayerReader reader_one(generated_one.position_bytes, lut);
    const BCPositionLayerReader reader_two(generated_two.position_bytes, lut);
    const BCPositionLayerReader reader_global(generated_global.position_bytes, lut);
    check(
        collect_generated_candidates(reader_one) == collect_generated_candidates(reader_two),
        "num_threads=1 and num_threads=2 generation outputs should match"
    );
    check(
        descriptor_success_rows_sum(reader_one) == descriptor_success_rows_sum(reader_two),
        "parallel generation output row counts should match"
    );
    check(
        collect_generated_candidates(reader_two) == collect_generated_candidates(reader_global),
        "global-sort and per-cell-sort finalize outputs should match"
    );
    check(
        descriptor_success_rows_sum(reader_two) == descriptor_success_rows_sum(reader_global),
        "per-cell finalize output row count should match global finalize"
    );
}

void test_pair_generation_stats() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable primary_axis = BCFamilyTable::from_range(10U, 2U, 0U, 2U);
    const BCFamilyTable secondary_axis = BCFamilyTable::from_range(12U, 2U, 0U, 3U);
    const TestPositionLayer source = write_source_layer(
        lut,
        source_axis,
        {
            make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}}),
            make_board({{0U, 2U}, {15U, 1U}, {1U, 1U}}),
        }
    );

    BC::BCResidentGenerationOptions options;
    options.num_threads = 2;
    const BC::BCResidentGenerationPairResult pair =
        BC::generate_resident_position_layer_pair(
            lut,
            primary_axis,
            source.reader,
            nullptr,
            &secondary_axis,
            options
        );

    const OracleResult primary_oracle = compute_oracle(
        lut,
        primary_axis,
        {OracleSource{source.canonical_boards, 1U}}
    );
    const OracleResult secondary_oracle = compute_oracle(
        lut,
        secondary_axis,
        {OracleSource{source.canonical_boards, 2U}}
    );
    const BCPositionLayerReader primary_reader(pair.primary.position_bytes, lut);
    const BCPositionLayerReader secondary_reader(pair.secondary.position_bytes, lut);
    check(
        collect_generated_candidates(primary_reader) == primary_oracle.candidates,
        "pair primary generated candidates mismatch oracle"
    );
    check(
        collect_generated_candidates(secondary_reader) == secondary_oracle.candidates,
        "pair secondary generated candidates mismatch oracle"
    );
    check(pair.has_secondary, "pair generation should report secondary");
    check(
        pair.current_boards_scanned == descriptor_success_rows_sum(source.reader),
        "pair current_boards_scanned should match current source rows"
    );
    check(
        pair.primary.source_boards_scanned == pair.current_boards_scanned,
        "pair primary source stats should own current scan count"
    );
    check(
        pair.secondary.source_boards_scanned == 0U,
        "pair secondary source scan count should remain zero; use pair.current_boards_scanned"
    );
    check(
        pair.shared_generation_seconds == pair.primary.generation_seconds &&
            pair.shared_generation_seconds == pair.secondary.generation_seconds,
        "pair shared generation seconds should match both layer result aliases"
    );
    check(
        pair.total_pair_compute_seconds >= pair.shared_generation_seconds,
        "pair total compute seconds should include shared generation time"
    );
    check_stats_match(pair.primary, primary_oracle);
    check(pair.secondary.spawned_boards == secondary_oracle.spawned_boards,
        "pair secondary spawn count mismatch");
    check(pair.secondary.move_candidates == secondary_oracle.move_candidates,
        "pair secondary move candidate count mismatch");
    check(pair.secondary.valid_candidates == secondary_oracle.valid_candidates,
        "pair secondary valid candidate count mismatch");
    check_dynamic_stats(pair.secondary);
}

void test_duplicate_candidate_dedup_and_invalid_moves() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(0U, 2U, 0U, 0U);
    const BCFamilyTable target_axis = BCFamilyTable::from_range(2U, 2U, 0U, 0U);
    const TestPositionLayer source = write_source_layer(lut, source_axis, {0U});

    const BCResidentGenerationSource source2{&source.reader, 1U, 1U};
    const BCResidentGenerationResult generated =
        BC::generate_resident_position_layer(lut, target_axis, std::vector<BCResidentGenerationSource>{source2});
    const OracleResult oracle = compute_oracle(
        lut,
        target_axis,
        {OracleSource{source.canonical_boards, 1U}}
    );
    const BCPositionLayerReader generated_reader(generated.position_bytes, lut);
    const std::set<Candidate> generated_candidates = collect_generated_candidates(generated_reader);
    check(generated_candidates == oracle.candidates, "duplicate generated candidates mismatch oracle");
    check(generated_candidates.size() < generated.valid_candidates, "generation should dedup repeated paths");
    check(
        descriptor_success_rows_sum(generated_reader) == generated_candidates.size(),
        "dedup success_rows should equal unique generated targets"
    );
    check(generated.move_candidates > generated.valid_candidates, "invalid moves should be skipped");
    check(
        generated.duplicate_candidates_possible + generated_candidates.size() == generated.valid_candidates,
        "duplicate stats should explain valid candidate paths"
    );
}

void test_axis_validation() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable source_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCFamilyTable bad_target_axis = BCFamilyTable::from_range(12U, 2U, 0U, 3U);
    const TestPositionLayer source = write_source_layer(
        lut,
        source_axis,
        {make_board({{0U, 1U}, {1U, 1U}, {4U, 1U}, {5U, 1U}})}
    );
    const BCResidentGenerationSource source2{&source.reader, 1U, 1U};
    expect_throws(
        [&] {
            (void)BC::generate_resident_position_layer(
                lut,
                bad_target_axis,
                std::vector<BCResidentGenerationSource>{source2}
            );
        },
        "resident generation should reject target total_coord mismatch"
    );
}

} // namespace

int main() {
    try {
        std::cerr << "test_single_source_generation_matches_oracle\n";
        test_single_source_generation_matches_oracle();
        std::cerr << "test_combined_generation_matches_oracle\n";
        test_combined_generation_matches_oracle();
        std::cerr << "test_parallel_and_small_batch_match_scalar_output\n";
        test_parallel_and_small_batch_match_scalar_output();
        std::cerr << "test_pair_generation_stats\n";
        test_pair_generation_stats();
        std::cerr << "test_duplicate_candidate_dedup_and_invalid_moves\n";
        test_duplicate_candidate_dedup_and_invalid_moves();
        std::cerr << "test_axis_validation\n";
        test_axis_validation();
    } catch (const std::exception &ex) {
        std::cerr << "bc_resident_generation_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_resident_generation_test passed\n";
    return 0;
}
