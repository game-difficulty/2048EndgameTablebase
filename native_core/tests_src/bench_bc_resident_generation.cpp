#include "BCResidentGeneration.h"

#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCPositionFile.h"
#include "BCSortUtils.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef BC_PORTABLE_X86_64_ARCH
#define BC_PORTABLE_X86_64_ARCH "unknown"
#endif

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionCellLayout;
using BC::BCPositionLayerReader;
using BC::BCResidentGenerationOptions;
using BC::BCResidentGenerationResult;
using BC::BCResidentGenerationSource;
using BC::CellId;

struct Args {
    std::string pattern = "free9";
    std::string seed_mode = "expanded";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    uint32_t target_extra_override = 0U;
    uint32_t target_sum = 0U;
    uint32_t warmup_extra = 16U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 512U;
    bool detail_timing = false;
    uint32_t cell_modulus = 29U;
    std::filesystem::path output_dir;
    std::string prefix = "free9_256_";
    std::filesystem::path stats_csv;
};

struct LayerState {
    uint32_t layer_sum = 0U;
    uint64_t rows = 0U;
    BCPositionLayerReader reader;
};

struct AggregateStats {
    uint32_t layers = 0U;
    uint64_t source_boards = 0U;
    uint64_t output_rows = 0U;
    uint64_t throughput_live = 0U;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
};

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] double mbps(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1.0e6 : 0.0;
}

[[nodiscard]] uint32_t parse_rank_to_extra(uint32_t rank) {
    if (rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large for uint32 target_extra");
    }
    return 1U << rank;
}

[[nodiscard]] uint32_t ex_forward_steps(const Args &args) {
    if (args.target_extra_override != 0U) {
        return args.target_extra_override / 2U;
    }
    const uint32_t target_tile = parse_rank_to_extra(args.target_rank);
    return target_tile / 2U + args.extra_steps - 1U;
}

[[nodiscard]] uint32_t ex_docheck_step_for_target_rank(uint32_t target_rank) {
    const uint32_t target_tile = parse_rank_to_extra(target_rank);
    if (target_tile < 8U) {
        return 0U;
    }
    return target_tile / 2U - 4U;
}

[[nodiscard]] std::vector<uint8_t> all_board_success_shifts() {
    std::vector<uint8_t> shifts;
    shifts.reserve(16U);
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }
    return shifts;
}

[[nodiscard]] uint64_t descriptor_rows(const BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

[[nodiscard]] std::filesystem::path find_patterns_config() {
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

[[nodiscard]] uint64_t parse_hex_u64(const std::string &text) {
    size_t parsed = 0U;
    const uint64_t value = std::stoull(text, &parsed, 0);
    if (parsed != text.size()) {
        throw std::runtime_error("failed to parse full hex seed board");
    }
    return value;
}

[[nodiscard]] uint64_t load_pattern_seed_board(const std::string &pattern) {
    std::ifstream file(find_patterns_config());
    if (!file) {
        throw std::runtime_error("failed to open patterns_config.json");
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str();
    const size_t pattern_pos = text.find("\"" + pattern + "\"");
    if (pattern_pos == std::string::npos) {
        throw std::runtime_error("patterns_config.json does not contain " + pattern);
    }
    const size_t seed_boards = text.find("\"seed boards\"", pattern_pos);
    if (seed_boards == std::string::npos) {
        throw std::runtime_error(pattern + " does not contain seed boards");
    }
    const size_t hex_begin = text.find("0x", seed_boards);
    if (hex_begin == std::string::npos) {
        throw std::runtime_error(pattern + " seed board has no hex value");
    }
    size_t hex_end = hex_begin + 2U;
    while (hex_end < text.size() &&
           std::isxdigit(static_cast<unsigned char>(text[hex_end])) != 0) {
        ++hex_end;
    }
    return parse_hex_u64(text.substr(hex_begin, hex_end - hex_begin));
}

[[nodiscard]] std::array<uint32_t, 16U> free_tile_sums() {
    return BC::default_2048_tile_sum_values();
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("free benchmark target rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(target_rank + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] std::vector<BC::LayerSum> free_possible_8tile_sums(
    const std::vector<uint8_t> &legal_tiles,
    const std::array<uint32_t, 16U> &tile_sums
) {
    return BC::build_possible_8tile_sums(legal_tiles, tile_sums);
}

[[nodiscard]] BCLut make_free_lut(uint32_t target_rank) {
    return BCLut(make_free_legal_tiles(target_rank));
}

[[nodiscard]] uint32_t board_semantic_sum(
    uint64_t board,
    const std::array<uint32_t, 16U> &tile_sums
) {
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

[[nodiscard]] uint64_t word_semantic_sum(
    uint16_t word,
    const std::array<uint32_t, 16U> &tile_sums
) {
    uint64_t sum = 0U;
    for (uint32_t i = 0U; i < 4U; ++i) {
        sum += tile_sums[BC::word_tile(word, i)];
    }
    return sum;
}

[[nodiscard]] BC::BCBoardEncodedPosition encode_canonical_quadrants_position_custom_sums(
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const BC::BCQuadrantWords &q,
    const std::array<uint32_t, 16U> &tile_sums
) {
    BC::BCBoardEncodedPosition out;
    const BCFamilyTable &axis = layout.serialization_axis();
    const BC::BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BC::BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BC::BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BC::BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = word_semantic_sum(q.nw, tile_sums);
    const uint64_t ne_sum = word_semantic_sum(q.ne, tile_sums);
    const uint64_t sw_sum = word_semantic_sum(q.sw, tile_sums);
    const uint64_t se_sum = word_semantic_sum(q.se, tile_sums);
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!BC::bc_min_side_coord_u64(nw_sum + ne_sum, sw_sum + se_sum, axis.family_unit(), row_coord) ||
        !BC::bc_min_side_coord_u64(nw_sum + sw_sum, ne_sum + se_sum, axis.family_unit(), col_coord)) {
        return out;
    }
    const BC::FamilyId row_id = layout.try_raw_side_coord_to_physical_index(row_coord);
    const BC::FamilyId col_id = layout.try_raw_side_coord_to_physical_index(col_coord);
    if (row_id == BCPositionCellLayout::kInvalidSideIndex ||
        col_id == BCPositionCellLayout::kInvalidSideIndex) {
        return out;
    }

    const uint64_t cid64 =
        static_cast<uint64_t>(row_id) * static_cast<uint64_t>(axis.family_count()) + col_id;
    if (cid64 > std::numeric_limits<BC::CellId>::max()) {
        throw std::overflow_error("BC custom-sum encoded cell id exceeds CellId");
    }

    const BC::BCEncodedKeyRank encoded =
        BC::bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return out;
    }
    out.cid = static_cast<BC::CellId>(cid64);
    out.row_family = row_id;
    out.col_family = col_id;
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

[[nodiscard]] BCFamilyTable make_axis(
    uint32_t layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t cell_modulus
) {
    return BC::build_modulo_position_cell_layout_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        cell_modulus
    ).serialization_axis();
}

[[nodiscard]] BCPositionCellLayout make_layout(
    uint32_t layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t cell_modulus
) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free resident benchmark requires even layer sums");
    }
    return BC::build_modulo_position_cell_layout_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        cell_modulus
    );
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

[[nodiscard]] std::vector<uint64_t> collect_canonical_successors(const std::vector<uint64_t> &boards) {
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

[[nodiscard]] bool is_reachable_free_init(uint64_t board) {
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

[[nodiscard]] uint32_t free_pattern_index(const std::string &pattern) {
    constexpr char kPrefix[] = "free";
    if (pattern.rfind(kPrefix, 0U) != 0U || pattern.size() <= 4U) {
        throw std::invalid_argument("free benchmark pattern must be named freeN");
    }
    size_t parsed = 0U;
    const uint32_t value = static_cast<uint32_t>(std::stoul(pattern.substr(4U), &parsed));
    if (parsed != pattern.size() - 4U || value == 0U || value > 16U) {
        throw std::invalid_argument("invalid freeN benchmark pattern");
    }
    return value;
}

[[nodiscard]] std::vector<uint64_t> generate_free_initial_boards(uint32_t free_cells) {
    if (free_cells < 2U || free_cells > 16U) {
        throw std::invalid_argument("free initial generator requires 2..16 free cells");
    }
    const uint32_t large_tile_count = 16U - free_cells;
    const uint32_t initial_twos = free_cells - 1U;
    std::vector<uint8_t> cells(16U);
    for (uint8_t i = 0U; i < cells.size(); ++i) {
        cells[i] = i;
    }

    std::vector<uint64_t> generated;
    generated.reserve(120000U);
    std::vector<uint8_t> large_positions;
    choose_positions(cells, large_tile_count, 0U, large_positions, [&](const std::vector<uint8_t> &positions_32k) {
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
        choose_positions(remaining, initial_twos, 0U, two_positions, [&](const std::vector<uint8_t> &positions_2) {
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

[[nodiscard]] std::vector<uint8_t> build_initial_layer_bytes(
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const std::vector<uint64_t> &initial_boards,
    const std::array<uint32_t, 16U> &family_tile_sums
) {
    const BCFamilyTable &axis = layout.serialization_axis();
    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    for (uint64_t board : initial_boards) {
        const uint64_t canonical = Calculator::canonical_full(board);
        const auto encoded = encode_canonical_quadrants_position_custom_sums(
            lut,
            layout,
            BC::unpack_board_to_quadrants(canonical),
            family_tile_sums
        );
        check(encoded.valid, "free initial board should encode into seed axis");
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

    BC::BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (!builders[cid]) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, builders[cid]->finalize());
    }
    return writer.finish_layer();
}

[[nodiscard]] std::unique_ptr<LayerState> make_initial_layer(
    const BCLut &lut,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::string &pattern,
    const std::string &seed_mode,
    uint32_t free_cells,
    uint32_t cell_modulus
) {
    const uint64_t seed_board = load_pattern_seed_board(pattern);
    const uint32_t seed_sum = board_semantic_sum(seed_board, tile_sums);
    std::vector<uint64_t> initial_boards;
    if (seed_mode == "single") {
        initial_boards.push_back(seed_board);
    } else if (seed_mode == "expanded") {
        initial_boards = generate_free_initial_boards(free_cells);
        if (pattern == "free9") {
            check(initial_boards.size() == 21283U, "free9 initial board count must match EX");
        }
    } else {
        throw std::invalid_argument("--seed-mode must be single or expanded");
    }

    auto layer = std::make_unique<LayerState>();
    layer->layer_sum = seed_sum;
    const std::vector<uint8_t> bytes =
        build_initial_layer_bytes(
            lut,
            make_layout(seed_sum, possible_8tile_sums, cell_modulus),
            initial_boards,
            tile_sums
        );
    layer->reader.open(bytes, lut);
    layer->rows = descriptor_rows(layer->reader);
    check(layer->rows == initial_boards.size(), "free initial layer row count mismatch");
    return layer;
}

[[nodiscard]] std::filesystem::path output_position_path(const Args &args, uint32_t ordinal) {
    return args.output_dir / (args.prefix + std::to_string(ordinal) + ".bcpos");
}

void write_position_if_requested(
    const Args &args,
    uint32_t ordinal,
    const std::vector<uint8_t> &bytes
) {
    if (args.output_dir.empty()) {
        return;
    }
    std::filesystem::create_directories(args.output_dir);
    BC::write_position_layer_to_file(output_position_path(args, ordinal), bytes);
}

void accumulate(AggregateStats &agg, const BCResidentGenerationResult &result) {
    ++agg.layers;
    agg.source_boards += result.source_boards_scanned;
    agg.output_rows += result.output_success_rows;
    agg.throughput_live += result.ex_generate_throughput_live();
    agg.compute_seconds += result.compute_seconds;
    agg.total_seconds += result.total_seconds;
}

void print_layer_row(
    std::ostream &out,
    uint32_t layer_sum,
    const BCResidentGenerationResult &result
) {
    out << layer_sum << ','
        << result.source_boards_scanned << ','
        << result.output_success_rows << ','
        << result.output_success_rows << ','
        << result.source_boards_scanned << ','
        << result.output_success_rows << ','
        << result.source_boards_scanned << ','
        << result.total_seconds << ','
        << result.throughput_mbps() << ','
        << result.compute_seconds << ','
        << result.compute_throughput_mbps() << ','
        << result.source_board_mbps() << ','
        << result.output_board_mbps() << '\n';
}

void print_aggregate_row(
    std::ostream &out,
    const char *label,
    const AggregateStats &agg
) {
    out << label
        << "_source=" << agg.source_boards
        << " " << label << "_output=" << agg.output_rows
        << " " << label << "_throughput_live=" << agg.throughput_live
        << " " << label << "_total_seconds=" << agg.total_seconds
        << " " << label << "_compute_seconds=" << agg.compute_seconds
        << " " << label << "_throughput_mbps=" << mbps(agg.throughput_live, agg.total_seconds)
        << " " << label << "_compute_throughput_mbps=" << mbps(agg.throughput_live, agg.compute_seconds)
        << " warm_layers=" << agg.layers
        << '\n';
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
        } else if (key == "--seed-mode") {
            args.seed_mode = require_value("--seed-mode");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--extra-steps") {
            args.extra_steps = static_cast<uint32_t>(std::stoul(require_value("--extra-steps")));
        } else if (key == "--target-extra") {
            args.target_extra_override = static_cast<uint32_t>(std::stoul(require_value("--target-extra")));
        } else if (key == "--target-sum") {
            args.target_sum = static_cast<uint32_t>(std::stoul(require_value("--target-sum")));
        } else if (key == "--warmup-extra") {
            args.warmup_extra = static_cast<uint32_t>(std::stoul(require_value("--warmup-extra")));
        } else if (key == "--num-threads") {
            args.num_threads = std::stoi(require_value("--num-threads"));
        } else if (key == "--batch-size") {
            args.batch_size = static_cast<uint32_t>(std::stoul(require_value("--batch-size")));
        } else if (key == "--pending-buffer") {
            args.pending_buffer = static_cast<uint32_t>(std::stoul(require_value("--pending-buffer")));
        } else if (key == "--detail-timing") {
            args.detail_timing = true;
        } else if (key == "--no-detail-timing") {
            args.detail_timing = false;
        } else if (key == "--cell-modulus" || key == "--family-modulus") {
            args.cell_modulus = static_cast<uint32_t>(std::stoul(require_value(key.c_str())));
        } else if (key == "--output-dir") {
            args.output_dir = require_value("--output-dir");
        } else if (key == "--prefix") {
            args.prefix = require_value("--prefix");
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value("--stats-csv");
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.pattern.rfind("free", 0U) != 0U) {
        throw std::invalid_argument("bc_resident_generation_bench currently supports --pattern freeN only");
    }
    (void)free_pattern_index(args.pattern);
    if (args.seed_mode != "single" && args.seed_mode != "expanded") {
        throw std::invalid_argument("--seed-mode must be single or expanded");
    }
    if (args.target_rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large");
    }
    if (args.target_extra_override != 0U && (args.target_extra_override & 1U) != 0U) {
        throw std::invalid_argument("--target-extra must be even");
    }
    if (args.batch_size == 0U || args.pending_buffer == 0U) {
        throw std::invalid_argument("--batch-size and --pending-buffer must be non-zero");
    }
    if (args.cell_modulus == 0U || args.cell_modulus > std::numeric_limits<BC::FamilyId>::max()) {
        throw std::invalid_argument("--cell-modulus must be in 1..65535");
    }
    return args;
}

int run_free_chain(const Args &args, std::ostream &out) {
    const uint32_t free_cells = free_pattern_index(args.pattern);
    const std::vector<uint8_t> legal_tiles = make_free_legal_tiles(args.target_rank);
    const std::array<uint32_t, 16U> tile_sums = free_tile_sums();
    const std::vector<BC::LayerSum> possible_8tile_sums =
        free_possible_8tile_sums(legal_tiles, tile_sums);
    const BCLut lut = make_free_lut(args.target_rank);
    std::unique_ptr<LayerState> previous2 =
        make_initial_layer(
            lut,
            tile_sums,
            possible_8tile_sums,
            args.pattern,
            args.seed_mode,
            free_cells,
            args.cell_modulus
        );
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_to_primary;
    const uint32_t seed_sum = previous2->layer_sum;
    const uint32_t final_sum =
        args.target_sum != 0U ? args.target_sum : seed_sum + ex_forward_steps(args) * 2U;
    const bool ex_terminal_mode =
        args.target_sum == 0U &&
        args.target_extra_override == 0U &&
        final_sum >= seed_sum + 4U;
    const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
    if (final_primary_sum < seed_sum || ((final_primary_sum - seed_sum) & 1U) != 0U) {
        throw std::invalid_argument("target layer must be an even delta from seed");
    }
    const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
    const uint32_t success_check_min_source_layer_sum = seed_sum + 2U * (docheck_step + 1U);
    const std::vector<uint8_t> success_shifts = all_board_success_shifts();

    BCResidentGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    options.pending_insert_buffer_size = args.pending_buffer;
    options.dynamic_reserve_factor = 2.0;
    options.collect_timing = args.detail_timing;
    options.tile_sum_values = &tile_sums;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    options.finalize_options.keyvalue_sort = nullptr;
    options.finalize_options.simd_sort_min_bucket_count = 10000U;

    out << "canonical_backend=" << CanonicalBatch::backend_name()
        << " compile_arch=" << BC_PORTABLE_X86_64_ARCH
        << " pattern=" << args.pattern
        << " seed_mode=" << args.seed_mode
        << " seed=0x" << std::hex << load_pattern_seed_board(args.pattern) << std::dec
        << " seed_sum=" << seed_sum
        << " target_rank=" << args.target_rank
        << " extra_steps=" << args.extra_steps
        << " target_extra_override=" << args.target_extra_override
        << " target_sum=" << final_sum
        << " final_primary_sum=" << final_primary_sum
        << " num_threads=" << options.num_threads
        << " batch_size=" << options.canonical_batch_size
        << " pending_buffer=" << options.pending_insert_buffer_size
        << " dynamic_reserve_factor=" << options.dynamic_reserve_factor
        << " generation_mode=pair_mutable_carry"
        << " tile_sum_semantics=raw"
        << " cell_modulus=" << args.cell_modulus
        << "\n";

    AggregateStats aggregate;
    AggregateStats warm;
    write_position_if_requested(args, 0U, previous2->reader.bytes());

    for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
        const uint32_t ordinal = (layer_sum - seed_sum) / 2U;
        const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
        const BCPositionCellLayout target_layout =
            make_layout(layer_sum, possible_8tile_sums, args.cell_modulus);
        std::optional<BCPositionCellLayout> secondary_layout;
        if (!terminal && layer_sum + 2U <= final_primary_sum) {
            secondary_layout.emplace(make_layout(layer_sum + 2U, possible_8tile_sums, args.cell_modulus));
        }
        options.keep_only_success_generated_boards = terminal;
        options.keep_only_success_secondary_generated_boards =
            ex_terminal_mode && !terminal && layer_sum + 2U == final_primary_sum;
        BC::BCResidentGenerationPairResult pair =
            BC::generate_resident_position_layer_pair_with_mutable_carry(
                lut,
                target_layout,
                previous2->reader,
                std::move(carry_to_primary),
                secondary_layout ? &(*secondary_layout) : nullptr,
                options
            );
        BCResidentGenerationResult result = std::move(pair.primary);
        carry_to_primary = std::move(pair.secondary_carry);
        auto current = std::make_unique<LayerState>();
        current->layer_sum = layer_sum;
        current->reader.open(result.position_bytes, lut);
        current->rows = descriptor_rows(current->reader);
        if (current->rows != result.output_success_rows) {
            throw std::runtime_error("generated resident layer row count mismatch");
        }
        write_position_if_requested(args, ordinal, result.position_bytes);

        print_layer_row(out, layer_sum, result);
        accumulate(aggregate, result);
        if (layer_sum >= seed_sum + args.warmup_extra) {
            accumulate(warm, result);
        }

        previous2 = std::move(current);
    }

    out << "aggregate_source=" << aggregate.source_boards
        << " aggregate_output=" << aggregate.output_rows
        << " aggregate_throughput_live=" << aggregate.throughput_live
        << " aggregate_total_seconds=" << aggregate.total_seconds
        << " aggregate_compute_seconds=" << aggregate.compute_seconds
        << " aggregate_throughput_mbps=" << mbps(aggregate.throughput_live, aggregate.total_seconds)
        << " aggregate_compute_throughput_mbps=" << mbps(aggregate.throughput_live, aggregate.compute_seconds)
        << " warm_layers=" << warm.layers
        << " warm_throughput_live=" << warm.throughput_live
        << " warm_throughput_mbps=" << mbps(warm.throughput_live, warm.total_seconds)
        << " warm_compute_throughput_mbps=" << mbps(warm.throughput_live, warm.compute_seconds)
        << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        std::ofstream stats_file;
        std::ostream *out = &std::cout;
        if (!args.stats_csv.empty()) {
            stats_file.open(args.stats_csv, std::ios::binary | std::ios::trunc);
            if (!stats_file) {
                throw std::runtime_error("failed to open stats CSV: " + args.stats_csv.string());
            }
            out = &stats_file;
        }
        *out << std::setprecision(9);
        return run_free_chain(args, *out);
    } catch (const std::exception &ex) {
        std::cerr << "bc_resident_generation_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
