#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCBoardOps.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"
#include "BCResidentGeneration.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCResidentGenerationOptions;
using BC::BCResidentGenerationResult;
using BC::BCResidentGenerationSource;
using BC::CellId;

struct Args {
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    uint32_t target_extra_override = 0U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 128U;
    uint32_t warmup_extra = 16U;
    bool verify_layer_rows = true;
    bool detail_timing = true;
};

struct ResidentLayer {
    uint32_t layer_sum = 0U;
    std::vector<uint8_t> bytes;
    std::unique_ptr<BCPositionLayerReader> reader;
    uint64_t rows = 0U;
};

struct AggregateStats {
    uint32_t layers = 0U;
    int effective_threads = 0;
    uint64_t source_boards = 0U;
    uint64_t output_rows = 0U;
    uint64_t throughput_live = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_candidates = 0U;
    uint64_t moved_candidates = 0U;
    uint64_t encoded_candidates = 0U;
    uint64_t duplicate_candidates = 0U;
    uint64_t position_bytes = 0U;
    uint64_t bucket_count = 0U;
    uint64_t bitmap_live_bits = 0U;
    uint64_t bitmap_logical_bits = 0U;
    uint64_t bitmap_physical_bits = 0U;
    uint64_t rank_payload_bytes = 0U;
    double generation_seconds = 0.0;
    double scan_seconds = 0.0;
    double thread_spawn_move_seconds = 0.0;
    double thread_canonical_seconds = 0.0;
    double thread_encode_insert_seconds = 0.0;
    double merge_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
};

struct BitmapStats {
    uint64_t bucket_count = 0U;
    uint64_t live_bits = 0U;
    uint64_t logical_bits = 0U;
    uint64_t physical_bits = 0U;
    uint64_t rank_payload_bytes = 0U;

    [[nodiscard]] double logical_density() const {
        return logical_bits != 0U ? static_cast<double>(live_bits) / static_cast<double>(logical_bits) : 0.0;
    }

    [[nodiscard]] double physical_density() const {
        return physical_bits != 0U ? static_cast<double>(live_bits) / static_cast<double>(physical_bits) : 0.0;
    }

    [[nodiscard]] double rank_payload_bytes_per_live() const {
        return live_bits != 0U
            ? static_cast<double>(rank_payload_bytes) / static_cast<double>(live_bits)
            : 0.0;
    }
};

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] double mbps(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1.0e6 : 0.0;
}

bool board_has_target_rank(uint64_t board, uint32_t target_rank, const std::vector<uint8_t> &success_shifts) {
    const uint64_t target = static_cast<uint64_t>(target_rank);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

uint32_t parse_rank_to_extra(const std::string &value) {
    const uint32_t rank = static_cast<uint32_t>(std::stoul(value));
    if (rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large for uint32 target_extra");
    }
    return 1U << rank;
}

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char *label) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + label);
            }
            return argv[++i];
        };
        if (key == "--pattern") {
            args.pattern = require_value("--pattern");
        } else if (key == "--target-extra") {
            args.target_extra_override = static_cast<uint32_t>(std::stoul(require_value("--target-extra")));
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--extra-steps") {
            args.extra_steps = static_cast<uint32_t>(std::stoul(require_value("--extra-steps")));
        } else if (key == "--num-threads") {
            args.num_threads = std::stoi(require_value("--num-threads"));
        } else if (key == "--batch-size") {
            args.batch_size = static_cast<uint32_t>(std::stoul(require_value("--batch-size")));
        } else if (key == "--pending-buffer") {
            args.pending_buffer = static_cast<uint32_t>(std::stoul(require_value("--pending-buffer")));
        } else if (key == "--warmup-extra") {
            args.warmup_extra = static_cast<uint32_t>(std::stoul(require_value("--warmup-extra")));
        } else if (key == "--no-verify-layer-rows") {
            args.verify_layer_rows = false;
        } else if (key == "--no-detail-timing") {
            args.detail_timing = false;
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.pattern != "free9") {
        throw std::invalid_argument("bc_generation_compute_bench currently supports --pattern free9 only");
    }
    if (args.target_rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large");
    }
    if (args.target_extra_override != 0U && (args.target_extra_override & 1U) != 0U) {
        throw std::invalid_argument("--target-extra must be an even semantic sum when provided");
    }
    if (args.batch_size == 0U) {
        throw std::invalid_argument("--batch-size must be non-zero");
    }
    if (args.pending_buffer == 0U) {
        throw std::invalid_argument("--pending-buffer must be non-zero");
    }
    return args;
}

uint32_t ex_forward_steps(const Args &args) {
    if (args.target_extra_override != 0U) {
        return args.target_extra_override / 2U;
    }
    const uint32_t target_tile = parse_rank_to_extra(std::to_string(args.target_rank));
    return target_tile / 2U + args.extra_steps - 1U;
}

std::vector<uint8_t> all_board_success_shifts() {
    std::vector<uint8_t> shifts;
    shifts.reserve(16U);
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }
    return shifts;
}

uint32_t ex_docheck_step_for_target_rank(uint32_t target_rank) {
    const uint32_t target_tile = parse_rank_to_extra(std::to_string(target_rank));
    if (target_tile < 8U) {
        return 0U;
    }
    return target_tile / 2U - 4U;
}

std::filesystem::path find_patterns_config() {
    std::filesystem::path current = std::filesystem::current_path();
    for (uint32_t i = 0; i < 8U; ++i) {
        const std::filesystem::path candidate =
            current / "docs_and_configs" / "patterns_config.json";
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        if (!current.has_parent_path() || current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    }
    throw std::runtime_error("failed to locate docs_and_configs/patterns_config.json");
}

uint64_t parse_hex_u64(const std::string &text) {
    size_t parsed = 0U;
    const uint64_t value = std::stoull(text, &parsed, 0);
    if (parsed != text.size()) {
        throw std::runtime_error("failed to parse full hex seed board");
    }
    return value;
}

uint64_t load_free9_seed_board() {
    const std::filesystem::path config_path = find_patterns_config();
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("failed to open patterns_config.json");
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str();
    const size_t free9 = text.find("\"free9\"");
    if (free9 == std::string::npos) {
        throw std::runtime_error("patterns_config.json does not contain free9");
    }
    const size_t seed_boards = text.find("\"seed boards\"", free9);
    if (seed_boards == std::string::npos) {
        throw std::runtime_error("free9 does not contain seed boards");
    }
    const size_t hex_begin = text.find("0x", seed_boards);
    if (hex_begin == std::string::npos) {
        throw std::runtime_error("free9 seed board has no hex value");
    }
    size_t hex_end = hex_begin + 2U;
    while (hex_end < text.size() &&
           std::isxdigit(static_cast<unsigned char>(text[hex_end])) != 0) {
        ++hex_end;
    }
    return parse_hex_u64(text.substr(hex_begin, hex_end - hex_begin));
}

std::array<uint32_t, 16U> free9_semantic_tile_sums() {
    std::array<uint32_t, 16U> values = BC::default_2048_tile_sum_values();
    values[15] = 0U;
    return values;
}

BCLut make_free9_lut() {
    return BCLut({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U});
}

uint32_t board_semantic_sum(uint64_t board, const std::array<uint32_t, 16U> &tile_sums) {
    uint64_t sum = 0U;
    for (uint32_t cell = 0U; cell < 16U; ++cell) {
        const uint8_t tile = static_cast<uint8_t>((board >> (4U * cell)) & 0xFU);
        sum += tile_sums[tile];
    }
    if (sum > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("semantic layer sum exceeds uint32");
    }
    return static_cast<uint32_t>(sum);
}

BCFamilyTable make_axis(uint32_t layer_sum) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free9 BC compute benchmark requires even layer sums");
    }
    const uint32_t total_coord = layer_sum / 2U;
    const uint32_t max_coord = total_coord / 2U;
    if (max_coord > std::numeric_limits<BC::FamilyCoord>::max()) {
        throw std::overflow_error("free9 BC compute benchmark axis exceeds FamilyCoord");
    }
    return BCFamilyTable::from_range(
        layer_sum,
        2U,
        0U,
        static_cast<BC::FamilyCoord>(max_coord)
    );
}

uint64_t descriptor_success_rows_sum(const BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

BitmapStats collect_bitmap_stats(const BCPositionLayerReader &reader) {
    BitmapStats stats;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = reader.descriptor(cid);
        if (desc.empty()) {
            continue;
        }
        const BC::BCBucketEntryView buckets = reader.bucket_entries_for_cell(cid);
        if (buckets.size != desc.bucket_count) {
            throw std::runtime_error("BC bitmap stats bucket view size mismatch");
        }

        uint64_t bucket_live_sum = 0U;
        for (uint32_t i = 0U; i < buckets.size; ++i) {
            const BC::BCBucketEntry &bucket = buckets.data[i];
            const uint32_t next_success_offset =
                i + 1U < buckets.size
                    ? buckets.data[i + 1U].success_row_offset
                    : desc.success_rows;
            if (next_success_offset < bucket.success_row_offset ||
                next_success_offset > desc.success_rows) {
                throw std::runtime_error("BC bitmap stats bucket success offsets are invalid");
            }
            bucket_live_sum +=
                static_cast<uint64_t>(next_success_offset - bucket.success_row_offset);

            const uint32_t bitmap_len = BC::bitmap_len_from_key(reader.lut(), bucket.key);
            stats.logical_bits += bitmap_len;
            stats.physical_bits += static_cast<uint64_t>(BC::words_for_bits(bitmap_len)) * 64ULL;
        }
        if (bucket_live_sum != desc.success_rows) {
            throw std::runtime_error("BC bitmap stats bucket live sum mismatch");
        }

        stats.bucket_count += desc.bucket_count;
        stats.live_bits += desc.success_rows;
        stats.rank_payload_bytes += desc.rank_payload_bytes;
    }
    if (stats.rank_payload_bytes != reader.header().rank_payload_bytes) {
        throw std::runtime_error("BC bitmap stats rank payload byte sum mismatch");
    }
    return stats;
}

BitmapStats collect_bitmap_stats(const BCLut &lut, const std::vector<uint8_t> &position_bytes) {
    const BCPositionLayerReader reader(position_bytes, lut);
    return collect_bitmap_stats(reader);
}

template <class Fn>
void choose_positions(
    const std::vector<uint8_t> &values,
    uint32_t need,
    uint32_t start,
    std::vector<uint8_t> &chosen,
    Fn &&fn
) {
    if (chosen.size() == need) {
        fn(chosen);
        return;
    }
    const uint32_t remaining = need - static_cast<uint32_t>(chosen.size());
    for (uint32_t i = start; i + remaining <= values.size(); ++i) {
        chosen.push_back(values[i]);
        choose_positions(values, need, i + 1U, chosen, fn);
        chosen.pop_back();
    }
}

void sort_unique_boards(std::vector<uint64_t> &boards) {
    std::sort(boards.begin(), boards.end());
    boards.erase(std::unique(boards.begin(), boards.end()), boards.end());
}

std::vector<uint64_t> collect_canonical_successors(const std::vector<uint64_t> &boards) {
    std::vector<uint64_t> out;
    out.reserve(boards.size());
    for (uint64_t board : boards) {
        const auto moved = BoardMover::move_all_dir(board);
        const uint64_t candidates[4] = {
            std::get<0>(moved),
            std::get<1>(moved),
            std::get<2>(moved),
            std::get<3>(moved)
        };
        for (uint64_t candidate : candidates) {
            if (candidate == Calculator::canonical_full(candidate)) {
                out.push_back(candidate);
            }
        }
    }
    sort_unique_boards(out);
    return out;
}

bool is_reachable_free_init(uint64_t board) {
    constexpr uint32_t kCorners[4] = {0U, 3U, 12U, 15U};
    uint32_t large_corners = 0U;
    for (uint32_t cell : kCorners) {
        const uint8_t tile = static_cast<uint8_t>((board >> (4U * cell)) & 0xFU);
        if (tile > 2U) {
            ++large_corners;
        }
    }
    return large_corners < 4U;
}

std::vector<uint64_t> generate_free9_initial_boards() {
    constexpr uint32_t kFree9LargeTiles = 7U;
    constexpr uint32_t kFree9InitialTwos = 8U;
    std::vector<uint8_t> cells(16U);
    for (uint8_t i = 0U; i < cells.size(); ++i) {
        cells[i] = i;
    }

    std::vector<uint64_t> generated;
    generated.reserve(120000U);
    std::vector<uint8_t> large_positions;
    choose_positions(cells, kFree9LargeTiles, 0U, large_positions, [&](const std::vector<uint8_t> &positions_32k) {
        uint64_t base = 0U;
        bool used[16] = {};
        for (uint8_t pos : positions_32k) {
            base |= 15ULL << (4U * pos);
            used[pos] = true;
        }
        std::vector<uint8_t> remaining;
        remaining.reserve(16U - positions_32k.size());
        for (uint8_t cell : cells) {
            if (!used[cell]) {
                remaining.push_back(cell);
            }
        }
        std::vector<uint8_t> two_positions;
        choose_positions(remaining, kFree9InitialTwos, 0U, two_positions, [&](const std::vector<uint8_t> &positions_2) {
            uint64_t board = base;
            for (uint8_t pos : positions_2) {
                board |= 1ULL << (4U * pos);
            }
            generated.push_back(board);
        });
    });
    sort_unique_boards(generated);

    std::vector<uint64_t> canonical_a = collect_canonical_successors(generated);
    std::vector<uint64_t> canonical_b = collect_canonical_successors(canonical_a);
    if (!canonical_b.empty()) {
        canonical_a.insert(canonical_a.end(), canonical_b.begin(), canonical_b.end());
        sort_unique_boards(canonical_a);
    }

    std::vector<uint64_t> reachable;
    reachable.reserve(canonical_a.size());
    for (uint64_t board : canonical_a) {
        if (is_reachable_free_init(board)) {
            reachable.push_back(board);
        }
    }
    sort_unique_boards(reachable);
    return reachable;
}

ResidentLayer write_initial_layer(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &initial_boards,
    const std::array<uint32_t, 16U> &family_tile_sums
) {
    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    for (uint64_t board : initial_boards) {
        const uint64_t canonical = Calculator::canonical_full(board);
        const auto encoded = BC::encode_canonical_quadrants_position(
            lut,
            axis,
            BC::unpack_board_to_quadrants(canonical),
            &family_tile_sums
        );
        check(encoded.valid, "free9 initial board should encode into seed axis");
        if (!builders[encoded.cid]) {
            builders[encoded.cid] = std::make_unique<BCCellBuilder>(lut);
        }
        BC::BCEncodedKeyRank key_rank;
        key_rank.key = encoded.key;
        key_rank.rank = encoded.rank;
        key_rank.bitmap_len = encoded.bitmap_len;
        key_rank.count_ne = encoded.count_ne;
        key_rank.count_sw = encoded.count_sw;
        key_rank.count_se = encoded.count_se;
        key_rank.valid = true;
        (void)builders[encoded.cid]->insert_encoded_and_report(key_rank);
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

    ResidentLayer layer;
    layer.layer_sum = axis.layer_sum();
    layer.bytes = writer.finish_layer();
    layer.reader = std::make_unique<BCPositionLayerReader>(layer.bytes, lut);
    layer.rows = descriptor_success_rows_sum(*layer.reader);
    return layer;
}

ResidentLayer make_generated_layer(
    uint32_t layer_sum,
    const BCLut &lut,
    BCResidentGenerationResult &&result
) {
    ResidentLayer layer;
    layer.layer_sum = layer_sum;
    layer.bytes = std::move(result.position_bytes);
    layer.reader = std::make_unique<BCPositionLayerReader>(layer.bytes, lut);
    layer.rows = descriptor_success_rows_sum(*layer.reader);
    return layer;
}

ResidentLayer compact_position_layer_to_success(
    uint32_t layer_sum,
    const BCLut &lut,
    const std::vector<uint8_t> &bytes,
    uint32_t target_rank,
    const std::vector<uint8_t> &success_shifts
) {
    BCPositionLayerReader reader(bytes, lut);
    const BCCellMatrix matrix(reader.axis());
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        BC::BCPositionCellScanner scanner(reader, cid);
        scanner.for_each_board([&](const BC::BCScannedBoardEntry &entry) {
            if (!board_has_target_rank(entry.board, target_rank, success_shifts)) {
                return;
            }
            if (!builders[cid]) {
                builders[cid] = std::make_unique<BCCellBuilder>(lut);
            }
            builders[cid]->insert(entry.key, entry.rank);
        });
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(reader.axis());
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (!builders[cid]) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, builders[cid]->finalize());
    }

    ResidentLayer layer;
    layer.layer_sum = layer_sum;
    layer.bytes = writer.finish_layer();
    layer.reader = std::make_unique<BCPositionLayerReader>(layer.bytes, lut);
    layer.rows = descriptor_success_rows_sum(*layer.reader);
    return layer;
}

void accumulate(
    AggregateStats &aggregate,
    const BCResidentGenerationResult &result,
    const BitmapStats &bitmap_stats
) {
    ++aggregate.layers;
    aggregate.effective_threads = std::max(aggregate.effective_threads, result.effective_threads);
    aggregate.source_boards += result.source_boards_scanned;
    aggregate.output_rows += result.output_success_rows;
    aggregate.throughput_live += result.ex_generate_throughput_live();
    aggregate.spawned_boards += result.spawned_boards;
    aggregate.move_candidates += result.move_candidates;
    aggregate.moved_candidates += result.moved_candidates;
    aggregate.encoded_candidates += result.encoded_candidates;
    aggregate.duplicate_candidates += result.duplicate_candidates;
    aggregate.position_bytes += static_cast<uint64_t>(result.position_bytes.size());
    aggregate.bucket_count += bitmap_stats.bucket_count;
    aggregate.bitmap_live_bits += bitmap_stats.live_bits;
    aggregate.bitmap_logical_bits += bitmap_stats.logical_bits;
    aggregate.bitmap_physical_bits += bitmap_stats.physical_bits;
    aggregate.rank_payload_bytes += bitmap_stats.rank_payload_bytes;
    aggregate.generation_seconds += result.generation_seconds;
    aggregate.scan_seconds += result.scan_seconds;
    aggregate.thread_spawn_move_seconds += result.thread_spawn_move_seconds;
    aggregate.thread_canonical_seconds += result.thread_canonical_seconds;
    aggregate.thread_encode_insert_seconds += result.thread_encode_insert_seconds;
    aggregate.merge_seconds += result.merge_seconds;
    aggregate.finalize_seconds += result.finalize_seconds;
    aggregate.write_seconds += result.write_seconds;
    aggregate.compute_seconds += result.compute_seconds;
    aggregate.total_seconds += result.total_seconds;
}

void add_terminal_secondary_to_aggregate(
    AggregateStats &aggregate,
    uint64_t secondary_rows,
    uint64_t secondary_position_bytes,
    const BitmapStats &secondary_bitmap_stats
) {
    aggregate.output_rows += secondary_rows;
    aggregate.throughput_live += secondary_rows;
    aggregate.position_bytes += secondary_position_bytes;
    aggregate.bucket_count += secondary_bitmap_stats.bucket_count;
    aggregate.bitmap_live_bits += secondary_bitmap_stats.live_bits;
    aggregate.bitmap_logical_bits += secondary_bitmap_stats.logical_bits;
    aggregate.bitmap_physical_bits += secondary_bitmap_stats.physical_bits;
    aggregate.rank_payload_bytes += secondary_bitmap_stats.rank_payload_bytes;
}

double thread_hot_seconds(const AggregateStats &stats) {
    return stats.thread_spawn_move_seconds +
           stats.thread_canonical_seconds +
           stats.thread_encode_insert_seconds;
}

void print_aggregate(
    const char *label,
    const AggregateStats &stats,
    uint32_t reported_layers,
    bool fallback_to_aggregate = false
) {
    std::cout
        << (fallback_to_aggregate ? std::string(label) + "_fallback=aggregate " : "")
        << label << "_layers=" << reported_layers
        << ' ' << label << "_effective_threads=" << stats.effective_threads
        << ' ' << label << "_source_boards=" << stats.source_boards
        << ' ' << label << "_output_rows=" << stats.output_rows
        << ' ' << label << "_throughput_live=" << stats.throughput_live
        << ' ' << label << "_moved_candidates=" << stats.moved_candidates
        << ' ' << label << "_encoded_candidates=" << stats.encoded_candidates
        << ' ' << label << "_duplicates=" << stats.duplicate_candidates
        << ' ' << label << "_position_bytes=" << stats.position_bytes
        << ' ' << label << "_bucket_count=" << stats.bucket_count
        << ' ' << label << "_bitmap_live_bits=" << stats.bitmap_live_bits
        << ' ' << label << "_bitmap_logical_bits=" << stats.bitmap_logical_bits
        << ' ' << label << "_bitmap_physical_bits=" << stats.bitmap_physical_bits
        << ' ' << label << "_bitmap_density="
        << (stats.bitmap_logical_bits != 0U
                ? static_cast<double>(stats.bitmap_live_bits) / static_cast<double>(stats.bitmap_logical_bits)
                : 0.0)
        << ' ' << label << "_bitmap_physical_density="
        << (stats.bitmap_physical_bits != 0U
                ? static_cast<double>(stats.bitmap_live_bits) / static_cast<double>(stats.bitmap_physical_bits)
                : 0.0)
        << ' ' << label << "_rank_payload_bytes=" << stats.rank_payload_bytes
        << ' ' << label << "_rank_payload_bytes_per_live="
        << (stats.bitmap_live_bits != 0U
                ? static_cast<double>(stats.rank_payload_bytes) / static_cast<double>(stats.bitmap_live_bits)
                : 0.0)
        << ' ' << label << "_generation_wall_seconds=" << stats.generation_seconds
        << ' ' << label << "_thread_scan_spawn_move_seconds=" << stats.thread_spawn_move_seconds
        << ' ' << label << "_thread_canonical_seconds=" << stats.thread_canonical_seconds
        << ' ' << label << "_thread_encode_insert_seconds=" << stats.thread_encode_insert_seconds
        << ' ' << label << "_thread_hot_seconds=" << thread_hot_seconds(stats)
        << ' ' << label << "_avg_hot_threads="
        << (stats.generation_seconds > 0.0 ? thread_hot_seconds(stats) / stats.generation_seconds : 0.0)
        << ' ' << label << "_compute_seconds=" << stats.compute_seconds
        << ' ' << label << "_total_seconds=" << stats.total_seconds
        << ' ' << label << "_compute_throughput_mbps="
        << mbps(stats.throughput_live, stats.compute_seconds)
        << ' ' << label << "_total_throughput_mbps="
        << mbps(stats.throughput_live, stats.total_seconds)
        << ' ' << label << "_moved_candidate_mbps="
        << mbps(stats.moved_candidates, stats.compute_seconds)
        << ' ' << label << "_source_board_mbps="
        << mbps(stats.source_boards, stats.compute_seconds)
        << ' ' << label << "_output_board_mbps="
        << mbps(stats.output_rows, stats.compute_seconds)
        << "\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::array<uint32_t, 16U> tile_sums = free9_semantic_tile_sums();
        const BCLut lut = make_free9_lut();
        const uint64_t seed_board = load_free9_seed_board();
        const uint32_t seed_sum = board_semantic_sum(seed_board, tile_sums);
        const uint32_t forward_steps = ex_forward_steps(args);
        const uint32_t final_sum = seed_sum + forward_steps * 2U;
        const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
        const uint32_t success_check_min_source_layer_sum = seed_sum + 2U * (docheck_step + 1U);
        const std::vector<uint8_t> success_shifts = all_board_success_shifts();
        std::vector<uint64_t> initial_boards = generate_free9_initial_boards();
        check(initial_boards.size() == 21283U, "free9 C++ initial boards must match EX generate_free_inits(7,8)");

        BCResidentGenerationOptions options;
        options.num_threads = args.num_threads;
        options.canonical_batch_size = args.batch_size;
        options.pending_insert_buffer_size = args.pending_buffer;
        options.family_tile_sum_values = &tile_sums;
        options.collect_timing = args.detail_timing;
        options.success_target_rank = static_cast<int>(args.target_rank);
        options.success_shifts = &success_shifts;
        options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;

        ResidentLayer current =
            write_initial_layer(lut, make_axis(seed_sum), initial_boards, tile_sums);
        check(current.rows == initial_boards.size(), "free9 initial layer row count mismatch");
        std::unique_ptr<ResidentLayer> carry_layer;

        AggregateStats aggregate;
        AggregateStats warm;

        std::cout << std::setprecision(9);
        std::cout
            << "benchmark=bc_generation_compute"
            << " no_cheat=actual_resident_generation"
            << " pattern=" << args.pattern
            << " seed=0x" << std::hex << seed_board << std::dec
            << " seed_sum=" << seed_sum
            << " target_rank=" << args.target_rank
            << " extra_steps=" << args.extra_steps
            << " target_extra_override=" << args.target_extra_override
            << " forward_steps=" << forward_steps
            << " target_sum=" << final_sum
            << " initial_live=" << initial_boards.size()
            << " num_threads=" << args.num_threads
            << " batch_size=" << args.batch_size
            << " pending_buffer=" << args.pending_buffer
            << " success_target_rank=" << args.target_rank
            << " success_check_min_source_layer_sum=" << success_check_min_source_layer_sum
            << " verify_layer_rows=" << (args.verify_layer_rows ? 1 : 0)
            << " detail_timing=" << (args.detail_timing ? 1 : 0)
            << " move=BoardMover::move_all_dir"
            << " canonical_batch=CanonicalBatch::canonicalize_inplace"
            << " canonical_backend=" << CanonicalBatch::backend_name()
            << "\n";

        std::cout
            << "layer_sum,effective_threads,input_live,primary_live,throughput_live,source_rows,"
            << "secondary_live,"
            << "primary_bucket_count,primary_bitmap_bits,primary_bitmap_density,"
            << "primary_bitmap_physical_density,primary_rank_payload_bytes,"
            << "secondary_bucket_count,secondary_bitmap_bits,secondary_bitmap_density,"
            << "secondary_bitmap_physical_density,secondary_rank_payload_bytes,"
            << "spawned_boards,move_candidates,moved_candidates,encoded_candidates,"
            << "duplicate_candidates,position_bytes,generation_wall_seconds,scan_phase_wall_seconds,"
            << "thread_scan_spawn_move_seconds,thread_canonical_seconds,thread_encode_insert_seconds,"
            << "thread_hot_seconds,avg_hot_threads,merge_seconds,finalize_seconds,"
            << "write_seconds,compute_seconds,total_seconds,compute_throughput_mbps,"
            << "total_throughput_mbps,moved_candidate_mbps,source_board_mbps,output_board_mbps\n";

        const bool ex_terminal_mode = args.target_extra_override == 0U && final_sum >= seed_sum + 4U;
        const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
        for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
            const BCFamilyTable target_axis = make_axis(layer_sum);
            const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
            const bool need_secondary = layer_sum + 2U <= final_sum;
            BCFamilyTable secondary_axis = need_secondary ? make_axis(layer_sum + 2U) : target_axis;
            BC::BCResidentGenerationPairResult pair =
                BC::generate_resident_position_layer_pair(
                    lut,
                    target_axis,
                    *current.reader,
                    carry_layer ? carry_layer->reader.get() : nullptr,
                    need_secondary ? &secondary_axis : nullptr,
                    options
                );
            BCResidentGenerationResult &result = pair.primary;
            const uint64_t source_rows = current.rows;
            if (args.verify_layer_rows && source_rows != result.source_boards_scanned) {
                throw std::runtime_error("source descriptor rows do not match generated scanned rows");
            }

            uint64_t secondary_live =
                pair.has_secondary ? pair.secondary.output_success_rows : 0U;
            uint64_t secondary_position_bytes =
                pair.has_secondary ? static_cast<uint64_t>(pair.secondary.position_bytes.size()) : 0U;
            if (terminal) {
                const double compact_begin = now_seconds();
                ResidentLayer compacted_primary = compact_position_layer_to_success(
                    layer_sum,
                    lut,
                    result.position_bytes,
                    args.target_rank,
                    success_shifts
                );
                ResidentLayer compacted_secondary = compact_position_layer_to_success(
                    layer_sum + 2U,
                    lut,
                    pair.secondary.position_bytes,
                    args.target_rank,
                    success_shifts
                );
                const double compact_seconds = now_seconds() - compact_begin;
                result.position_bytes = std::move(compacted_primary.bytes);
                result.output_success_rows = compacted_primary.rows;
                result.finalize_seconds += compact_seconds;
                result.compute_seconds += compact_seconds;
                result.total_seconds += compact_seconds;
                pair.secondary.position_bytes = std::move(compacted_secondary.bytes);
                pair.secondary.output_success_rows = compacted_secondary.rows;
                secondary_live = pair.secondary.output_success_rows;
                secondary_position_bytes = static_cast<uint64_t>(pair.secondary.position_bytes.size());
            }

            const BitmapStats primary_bitmap = collect_bitmap_stats(lut, result.position_bytes);
            const BitmapStats secondary_bitmap =
                pair.has_secondary ? collect_bitmap_stats(lut, pair.secondary.position_bytes) : BitmapStats{};
            const uint64_t expected_output_rows = result.output_success_rows;
            const uint64_t throughput_live = result.ex_generate_throughput_live();
            std::cout
                << layer_sum << ','
                << result.effective_threads << ','
                << result.source_boards_scanned << ','
                << result.output_success_rows << ','
                << throughput_live << ','
                << source_rows << ','
                << secondary_live << ','
                << primary_bitmap.bucket_count << ','
                << primary_bitmap.logical_bits << ','
                << primary_bitmap.logical_density() << ','
                << primary_bitmap.physical_density() << ','
                << primary_bitmap.rank_payload_bytes << ','
                << secondary_bitmap.bucket_count << ','
                << secondary_bitmap.logical_bits << ','
                << secondary_bitmap.logical_density() << ','
                << secondary_bitmap.physical_density() << ','
                << secondary_bitmap.rank_payload_bytes << ','
                << result.spawned_boards << ','
                << result.move_candidates << ','
                << result.moved_candidates << ','
                << result.encoded_candidates << ','
                << result.duplicate_candidates << ','
                << result.position_bytes.size() << ','
                << result.generation_seconds << ','
                << result.scan_seconds << ','
                << result.thread_spawn_move_seconds << ','
                << result.thread_canonical_seconds << ','
                << result.thread_encode_insert_seconds << ','
                << result.thread_hot_seconds() << ','
                << result.avg_hot_threads() << ','
                << result.merge_seconds << ','
                << result.finalize_seconds << ','
                << result.write_seconds << ','
                << result.compute_seconds << ','
                << result.total_seconds << ','
                << result.compute_throughput_mbps() << ','
                << result.throughput_mbps() << ','
                << result.moved_candidate_mbps() << ','
                << result.source_board_mbps() << ','
                << result.output_board_mbps()
                << "\n";

            accumulate(aggregate, result, primary_bitmap);
            if (terminal && secondary_live != 0U) {
                add_terminal_secondary_to_aggregate(
                    aggregate,
                    secondary_live,
                    secondary_position_bytes,
                    secondary_bitmap
                );
            }
            if (layer_sum >= seed_sum + args.warmup_extra) {
                accumulate(warm, result, primary_bitmap);
                if (terminal && secondary_live != 0U) {
                    add_terminal_secondary_to_aggregate(
                        warm,
                        secondary_live,
                        secondary_position_bytes,
                        secondary_bitmap
                    );
                }
            }

            current = make_generated_layer(layer_sum, lut, std::move(result));
            if (args.verify_layer_rows &&
                current.rows != expected_output_rows) {
                throw std::runtime_error("generated position descriptor rows do not match output_success_rows");
            }
            if (args.verify_layer_rows &&
                current.rows != descriptor_success_rows_sum(*current.reader)) {
                throw std::runtime_error("generated position descriptor rows are not stable");
            }
            if (terminal) {
                carry_layer.reset();
            } else if (need_secondary) {
                carry_layer = std::make_unique<ResidentLayer>(
                    make_generated_layer(layer_sum + 2U, lut, std::move(pair.secondary))
                );
            } else {
                carry_layer.reset();
            }
        }

        print_aggregate("aggregate", aggregate, aggregate.layers);
        if (warm.layers == 0U) {
            print_aggregate("warm", aggregate, 0U, true);
        } else {
            print_aggregate("warm", warm, warm.layers);
        }

        const char *perf_assert = std::getenv("BC_PERF_ASSERT");
        if (perf_assert != nullptr && std::string(perf_assert) == "1" &&
            mbps(warm.layers == 0U ? aggregate.moved_candidates : warm.moved_candidates,
                 warm.layers == 0U ? aggregate.compute_seconds : warm.compute_seconds) < 100.0) {
            std::cerr << "BC_PERF_ASSERT failed: warm moved_candidate_mbps < 100\n";
            return 2;
        }
    } catch (const std::exception &ex) {
        std::cerr << "bc_generation_compute_bench failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
