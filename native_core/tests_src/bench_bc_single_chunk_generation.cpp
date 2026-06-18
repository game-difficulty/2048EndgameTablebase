#include "BCSingleChunkGeneration.h"

#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCDirectFileIO.h"
#include "BCFileIO.h"
#include "BCPositionCellLoader.h"
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
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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
using BC::BCPositionStreamingReader;
using BC::BCResidentGenerationOptions;
using BC::BCResidentGenerationResult;
using BC::BCSingleChunkGenerationSource;
using BC::CellId;

struct Args {
    uint32_t target_sum = 0U;
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    uint32_t target_extra_override = 0U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 1024U;
    uint32_t cell_chunk_size = 0U;
    bool cell_chunk_size_explicit = false;
    uint32_t cell_chunk_count = 0U;
    uint32_t warmup_extra = 16U;
    uint64_t expected_output_rows = 0U;
    bool verify_layer_rows = true;
    bool detail_timing = false;
    std::string target_output_io = "direct";
    uint32_t target_direct_queue_depth = 8U;
    bool target_direct_overlapped = false;
    std::string position_io = "direct-auto";
    uint32_t position_direct_queue_depth = 8U;
    uint32_t cell_modulus = 29U;
    std::vector<std::filesystem::path> sources;
    std::filesystem::path output;
    std::filesystem::path output_dir =
        std::filesystem::path("tmp") / "bc_singlechunk_free9";
    std::filesystem::path stats_csv;
    std::filesystem::path ex_stats_csv;
};

struct LayerFile {
    uint32_t layer_sum = 0U;
    std::filesystem::path path;
    uint64_t logical_size = 0U;
    uint64_t physical_size = 0U;
    uint64_t rows = 0U;
    uint64_t bucket_count = 0U;
    uint64_t rank_payload_bytes = 0U;
};

struct AggregateStats {
    uint32_t layers = 0U;
    int effective_threads = 0;
    uint64_t input_live = 0U;
    uint64_t scanned_source_rows = 0U;
    uint64_t output_rows = 0U;
    uint64_t throughput_live = 0U;
    uint64_t generation_retries = 0U;
    uint64_t dynamic_hash_capacity_sum = 0U;
    uint64_t dynamic_bucket_slots_used = 0U;
    uint64_t source_read_bytes = 0U;
    uint64_t source_backend_read_ops = 0U;
    uint64_t target_logical_bytes = 0U;
    uint64_t target_backend_write_bytes = 0U;
    uint64_t target_backend_write_ops = 0U;
    double source_load_seconds = 0.0;
    double generation_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
};

struct ExLayerExpected {
    uint64_t input_live = 0U;
    uint64_t primary_live = 0U;
    uint64_t secondary_live = 0U;
    bool terminal = false;
};

struct ExExpectedStats {
    bool enabled = false;
    std::map<uint32_t, ExLayerExpected> by_layer_sum;
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

[[nodiscard]] uint32_t resolve_cell_chunk_size(const Args &args, uint32_t cell_count) {
    if (args.cell_chunk_count != 0U) {
        const uint64_t count = args.cell_chunk_count;
        return static_cast<uint32_t>(
            std::max<uint64_t>(
                1U,
                (static_cast<uint64_t>(cell_count) + count - 1U) / count
            )
        );
    }
    if (args.cell_chunk_size_explicit) {
        return args.cell_chunk_size == 0U ? cell_count : args.cell_chunk_size;
    }
    return std::min<uint32_t>(
        512U,
        std::max<uint32_t>(1U, (cell_count + 1U) / 2U)
    );
}

constexpr double kBCDefaultReserveFactor = 2.0;
constexpr double kBCEarlyLayerReserveFactor = 2.0;
constexpr uint32_t kBCEarlyLayerReserveFactorSteps = 10U;
constexpr double kBCLearnedReserveMinFactor = 1.08;
constexpr double kBCLearnedReserveQuantileGuard = 1.10;
constexpr double kBCLearnedReserveLastGuard = 1.12;
constexpr double kBCLearnedReserveRetryGuard = 1.25;
constexpr size_t kBCLearnedReserveHistoryWindow = 5U;

[[nodiscard]] double reserve_need_recent_quantile(const std::vector<double> &history) {
    const size_t begin =
        history.size() > kBCLearnedReserveHistoryWindow
            ? history.size() - kBCLearnedReserveHistoryWindow
            : 0U;
    std::vector<double> values(history.begin() + static_cast<std::ptrdiff_t>(begin), history.end());
    std::sort(values.begin(), values.end());
    const size_t index = ((values.size() - 1U) * 9U) / 10U;
    return values[index];
}

[[nodiscard]] double regular_reserve_factor(
    const std::vector<double> &history,
    double retry_guard_factor
) {
    if (history.empty()) {
        return kBCDefaultReserveFactor;
    }
    double factor = kBCLearnedReserveMinFactor;
    factor = std::max(factor, history.back() * kBCLearnedReserveLastGuard);
    factor = std::max(factor, reserve_need_recent_quantile(history) * kBCLearnedReserveQuantileGuard);
    factor = std::max(factor, retry_guard_factor);
    return std::min(kBCDefaultReserveFactor, std::max(kBCLearnedReserveMinFactor, factor));
}

[[nodiscard]] double reserve_factor_for_step(
    uint32_t current_step,
    const std::vector<double> &history,
    double retry_guard_factor
) {
    if (current_step < kBCEarlyLayerReserveFactorSteps) {
        return kBCEarlyLayerReserveFactor;
    }
    return regular_reserve_factor(history, retry_guard_factor);
}

[[nodiscard]] double reserve_need_component(
    uint64_t source_size,
    uint64_t used_size,
    uint64_t padding
) {
    if (used_size <= padding) {
        return 0.0;
    }
    if (source_size == 0U) {
        return kBCDefaultReserveFactor;
    }
    return static_cast<double>(used_size - padding) / static_cast<double>(source_size);
}

[[nodiscard]] double observed_reserve_need(
    uint64_t source_bucket_count,
    uint64_t source_rank_payload_bytes,
    const BCResidentGenerationResult &result
) {
    double need = 0.0;
    need = std::max(
        need,
        reserve_need_component(source_bucket_count, result.dynamic_bucket_slots_used, 4096ULL)
    );
    const uint64_t source_rank_words = source_rank_payload_bytes / sizeof(uint64_t) + 1U;
    need = std::max(
        need,
        reserve_need_component(
            source_rank_words,
            result.dynamic_bitmap_words_allocated,
            512ULL * 64ULL
        )
    );
    return need;
}

[[nodiscard]] std::vector<std::string> split_csv_simple(const std::string &line) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream in(line);
    while (std::getline(in, cell, ',')) {
        if (!cell.empty() && cell.back() == '\r') {
            cell.pop_back();
        }
        cells.push_back(cell);
    }
    if (!line.empty() && line.back() == ',') {
        cells.emplace_back();
    }
    return cells;
}

[[nodiscard]] uint64_t parse_u64_field(
    const std::vector<std::string> &cells,
    size_t index,
    const char *name
) {
    if (index >= cells.size()) {
        throw std::runtime_error(std::string("EX stats row missing field: ") + name);
    }
    return static_cast<uint64_t>(std::stoull(cells[index]));
}

[[nodiscard]] ExExpectedStats load_ex_expected_stats(
    const std::filesystem::path &path,
    uint32_t seed_sum
) {
    ExExpectedStats expected;
    if (path.empty()) {
        return expected;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open EX stats CSV: " + path.string());
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("EX stats CSV is empty: " + path.string());
    }
    const std::vector<std::string> header = split_csv_simple(header_line);
    std::map<std::string, size_t> column;
    for (size_t i = 0; i < header.size(); ++i) {
        column.emplace(header[i], i);
    }
    auto required = [&](const char *name) -> size_t {
        const auto it = column.find(name);
        if (it == column.end()) {
            throw std::runtime_error(std::string("EX stats CSV missing column: ") + name);
        }
        return it->second;
    };

    const size_t stage_col = required("stage");
    const size_t step_col = required("step");
    const size_t input_col = required("input_live");
    const size_t primary_col = required("primary_live");
    const size_t secondary_col = required("secondary_live");

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_simple(line);
        if (stage_col >= cells.size()) {
            throw std::runtime_error("EX stats row missing stage");
        }
        const std::string &stage = cells[stage_col];
        if (stage == "_total") {
            continue;
        }
        if (stage != "init" && stage != "forward" && stage != "forward_terminal") {
            continue;
        }
        const uint64_t step = parse_u64_field(cells, step_col, "step");
        if (step > std::numeric_limits<uint32_t>::max() / 2U) {
            throw std::runtime_error("EX stats step is too large");
        }
        const uint32_t layer_sum =
            stage == "init"
                ? seed_sum
                : seed_sum + static_cast<uint32_t>(step) * 2U;
        ExLayerExpected row;
        row.input_live = parse_u64_field(cells, input_col, "input_live");
        row.primary_live = parse_u64_field(cells, primary_col, "primary_live");
        row.secondary_live = parse_u64_field(cells, secondary_col, "secondary_live");
        row.terminal = stage == "forward_terminal";
        expected.by_layer_sum[layer_sum] = row;
    }
    expected.enabled = true;
    return expected;
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

[[nodiscard]] uint64_t load_free9_seed_board() {
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

[[nodiscard]] std::array<uint32_t, 16U> free9_tile_sums() {
    return BC::default_2048_tile_sum_values();
}

[[nodiscard]] std::vector<BC::LayerSum> free9_possible_8tile_sums(
    const std::array<uint32_t, 16U> &tile_sums
) {
    const std::vector<uint8_t> legal_tiles{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
    return BC::build_possible_8tile_sums(legal_tiles, tile_sums);
}

[[nodiscard]] BCLut make_free9_lut() {
    return BCLut({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U});
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
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free9 SingleChunk benchmark requires even layer sums");
    }
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
        throw std::invalid_argument("free9 SingleChunk benchmark requires even layer sums");
    }
    return BC::build_modulo_position_cell_layout_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        cell_modulus
    );
}

[[nodiscard]] uint64_t descriptor_rows(const BCPositionStreamingReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

[[nodiscard]] uint64_t descriptor_bucket_count(const BCPositionStreamingReader &reader) {
    return reader.header().bucket_meta_bytes / BC::kBCPositionBucketEntryBytes;
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

[[nodiscard]] std::vector<uint64_t> generate_free9_initial_boards() {
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

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_position_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.position_io == "buffered") {
        return std::make_unique<BC::BCBufferedFileWriter>(path);
    }
    if (args.position_io == "direct" || args.position_io == "direct-auto") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.position_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.position_direct_queue_depth > 1U;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    throw std::invalid_argument("--position-io must be buffered, direct, or direct-auto");
}

void write_raw_bytes_to_file(
    const Args &args,
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    std::unique_ptr<BC::BCWritableFile> writer = open_position_writer(args, path);
    writer->prepare_full_overwrite(static_cast<uint64_t>(bytes.size()));
    if (!bytes.empty()) {
        writer->write_at(0U, bytes.data(), static_cast<uint64_t>(bytes.size()));
    }
    writer->flush();
}

[[nodiscard]] std::unique_ptr<BCPositionStreamingReader> open_position_reader(
    const Args &args,
    const LayerFile &layer,
    const BCLut &lut
) {
    std::unique_ptr<BCPositionStreamingReader> reader;
    bool use_direct = args.position_io == "direct";
    if (args.position_io == "direct-auto") {
        const uint64_t aligned =
            BC::bc_direct_align_up(layer.logical_size, BC::BCDirectFileIOOptions{}.alignment);
        use_direct = layer.physical_size == aligned;
    }
    if (!use_direct) {
        reader = std::make_unique<BCPositionStreamingReader>(
            BCPositionStreamingReader::open_buffered(layer.path, lut)
        );
    } else if (args.position_io == "direct-auto") {
        reader = std::make_unique<BCPositionStreamingReader>(
            BCPositionStreamingReader::open_direct_auto(
                layer.path,
                lut,
                args.position_direct_queue_depth,
                args.position_direct_queue_depth > 1U
            )
        );
    } else {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.position_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.position_direct_queue_depth > 1U;
        options.logical_size = layer.logical_size;
        reader = std::make_unique<BCPositionStreamingReader>(
            std::make_unique<BC::BCDirectFileReader>(layer.path, options),
            lut
        );
    }
    reader->set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] std::unique_ptr<BCPositionStreamingReader> open_position_reader_path(
    const Args &args,
    const std::filesystem::path &path,
    const BCLut &lut
) {
    if (args.position_io == "buffered") {
        auto reader = std::make_unique<BCPositionStreamingReader>(
            BCPositionStreamingReader::open_buffered(path, lut)
        );
        reader->set_validate_loaded_cells(false);
        return reader;
    }
    if (args.position_io == "direct-auto") {
        auto reader = std::make_unique<BCPositionStreamingReader>(
            BCPositionStreamingReader::open_direct_auto(
                path,
                lut,
                args.position_direct_queue_depth,
                args.position_direct_queue_depth > 1U
            )
        );
        reader->set_validate_loaded_cells(false);
        return reader;
    }

    BC::BCBufferedFileReader probe(path);
    std::vector<uint8_t> header_bytes(BC::kBCPositionHeaderBytes);
    probe.read_at(0U, header_bytes.data(), header_bytes.size());
    const BC::BCPositionHeader header = BC::bc_read_header(header_bytes);
    const uint64_t logical_size = BC::bc_position_logical_size_from_header(header);
    BC::BCDirectFileIOOptions options;
    options.queue_depth = args.position_direct_queue_depth;
    options.overlapped = args.position_direct_queue_depth > 1U;
    options.logical_size = logical_size;
    const uint64_t required_physical = BC::bc_direct_align_up(logical_size, options.alignment);
    if (probe.size() < required_physical) {
        throw std::runtime_error("source position file is not padded for direct IO: " + path.string());
    }
    auto reader = std::make_unique<BCPositionStreamingReader>(
        std::make_unique<BC::BCDirectFileReader>(path, options),
        lut
    );
    reader->set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] LayerFile inspect_layer_file(
    const Args &args,
    uint32_t layer_sum,
    const std::filesystem::path &path,
    uint64_t logical_size,
    const BCLut &lut
) {
    LayerFile layer;
    layer.layer_sum = layer_sum;
    layer.path = path;
    layer.logical_size = logical_size;
    std::error_code ec;
    layer.physical_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("failed to stat layer file: " + ec.message());
    }
    std::unique_ptr<BCPositionStreamingReader> reader = open_position_reader(args, layer, lut);
    layer.rows = descriptor_rows(*reader);
    layer.bucket_count = descriptor_bucket_count(*reader);
    layer.rank_payload_bytes = reader->header().rank_payload_bytes;
    return layer;
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

[[nodiscard]] LayerFile write_initial_layer_file(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const std::vector<uint64_t> &initial_boards,
    const std::array<uint32_t, 16U> &family_tile_sums
) {
    const BCFamilyTable &axis = layout.serialization_axis();
    const std::vector<uint8_t> bytes =
        build_initial_layer_bytes(lut, layout, initial_boards, family_tile_sums);
    const std::filesystem::path path = args.output_dir / ("bc_layer_" + std::to_string(axis.layer_sum()) + ".bcpos");
    write_raw_bytes_to_file(args, path, bytes);
    return inspect_layer_file(args, axis.layer_sum(), path, bytes.size(), lut);
}

[[nodiscard]] std::filesystem::path layer_path(const Args &args, uint32_t layer_sum) {
    return args.output_dir / ("bc_layer_" + std::to_string(layer_sum) + ".bcpos");
}

void accumulate(AggregateStats &dst, const BCResidentGenerationResult &result, uint64_t input_live) {
    ++dst.layers;
    dst.effective_threads = std::max(dst.effective_threads, result.effective_threads);
    dst.input_live += input_live;
    dst.scanned_source_rows += result.source_boards_scanned;
    dst.output_rows += result.output_success_rows;
    dst.throughput_live += result.ex_generate_throughput_live();
    dst.generation_retries += result.generation_retries;
    dst.dynamic_hash_capacity_sum += result.dynamic_hash_capacity;
    dst.dynamic_bucket_slots_used += result.dynamic_bucket_slots_used;
    dst.source_read_bytes += result.source_position_load_read_bytes;
    dst.source_backend_read_ops += result.source_position_load_backend_read_ops;
    dst.target_logical_bytes += result.target_position_file_logical_bytes;
    dst.target_backend_write_bytes += result.target_position_write_backend_bytes;
    dst.target_backend_write_ops += result.target_position_write_backend_ops;
    dst.source_load_seconds += result.source_position_load_seconds;
    dst.generation_seconds += result.generation_seconds;
    dst.finalize_seconds += result.finalize_seconds;
    dst.write_seconds += result.write_seconds;
    dst.compute_seconds += result.compute_seconds;
    dst.total_seconds += result.total_seconds;
}

[[nodiscard]] BCResidentGenerationResult combine_strict_step_stats(
    BC::BCSingleChunkGenerationStepResult &&step
) {
    BCResidentGenerationResult result = std::move(step.primary);
    if (!step.has_carry) {
        return result;
    }

    const BCResidentGenerationResult &carry = step.carry;
    result.generation_retries += carry.generation_retries;
    result.dynamic_hash_capacity =
        std::max(result.dynamic_hash_capacity, carry.dynamic_hash_capacity);
    result.dynamic_bucket_slots_used =
        std::max(result.dynamic_bucket_slots_used, carry.dynamic_bucket_slots_used);
    result.dynamic_bitmap_words_used =
        std::max(result.dynamic_bitmap_words_used, carry.dynamic_bitmap_words_used);
    result.dynamic_bitmap_words_allocated =
        std::max(result.dynamic_bitmap_words_allocated, carry.dynamic_bitmap_words_allocated);
    result.dynamic_bitmap_words_reserved =
        std::max(result.dynamic_bitmap_words_reserved, carry.dynamic_bitmap_words_reserved);
    result.source_position_load_requested_extents += carry.source_position_load_requested_extents;
    result.source_position_load_coalesced_extents += carry.source_position_load_coalesced_extents;
    result.source_position_load_requested_bytes += carry.source_position_load_requested_bytes;
    result.source_position_load_read_bytes += carry.source_position_load_read_bytes;
    result.source_position_load_backend_read_ops += carry.source_position_load_backend_read_ops;
    result.source_position_load_backend_read_bytes += carry.source_position_load_backend_read_bytes;
    result.generation_seconds += carry.generation_seconds;
    result.scan_seconds += carry.scan_seconds;
    result.source_position_load_seconds += carry.source_position_load_seconds;
    result.thread_spawn_move_seconds += carry.thread_spawn_move_seconds;
    result.thread_canonical_seconds += carry.thread_canonical_seconds;
    result.thread_encode_insert_seconds += carry.thread_encode_insert_seconds;
    result.prepare_seconds += carry.prepare_seconds;
    result.work_seconds += carry.work_seconds;
    result.finalize_seconds += carry.finalize_seconds;
    result.cleanup_seconds += carry.cleanup_seconds;
    result.write_seconds += carry.write_seconds;
    result.compute_seconds += carry.compute_seconds;
    result.total_seconds += carry.total_seconds;
    return result;
}

void print_header(std::ostream &out) {
    out
        << "row_type,layer_sum,layers,effective_threads,input_live,scanned_source_rows,"
        << "output_rows,throughput_live,ex_input_live,ex_primary_live,ex_match,"
        << "dynamic_reserve_factor,generation_retries,dynamic_hash_capacity,"
        << "dynamic_bucket_slots_used,dynamic_hash_load,source_load_seconds,"
        << "source_read_bytes,source_backend_read_ops,target_logical_bytes,"
        << "target_backend_write_bytes,target_backend_write_ops,generation_seconds,"
        << "finalize_seconds,write_seconds,compute_seconds,total_seconds,"
        << "compute_throughput_mbps,total_throughput_mbps,"
        << "output_path\n";
}

void print_layer_row(
    std::ostream &out,
    uint32_t layer_sum,
    uint64_t input_live,
    uint64_t ex_input_live,
    uint64_t ex_primary_live,
    uint32_t ex_match,
    double dynamic_reserve_factor,
    const BCResidentGenerationResult &result,
    const std::filesystem::path &output_path
) {
    const double dynamic_hash_load =
        result.dynamic_hash_capacity != 0U
            ? static_cast<double>(result.dynamic_bucket_slots_used) /
                  static_cast<double>(result.dynamic_hash_capacity)
            : 0.0;
    out
        << "layer,"
        << layer_sum << ','
        << 1U << ','
        << result.effective_threads << ','
        << input_live << ','
        << result.source_boards_scanned << ','
        << result.output_success_rows << ','
        << result.ex_generate_throughput_live() << ','
        << ex_input_live << ','
        << ex_primary_live << ','
        << ex_match << ','
        << dynamic_reserve_factor << ','
        << result.generation_retries << ','
        << result.dynamic_hash_capacity << ','
        << result.dynamic_bucket_slots_used << ','
        << dynamic_hash_load << ','
        << result.source_position_load_seconds << ','
        << result.source_position_load_read_bytes << ','
        << result.source_position_load_backend_read_ops << ','
        << result.target_position_file_logical_bytes << ','
        << result.target_position_write_backend_bytes << ','
        << result.target_position_write_backend_ops << ','
        << result.generation_seconds << ','
        << result.finalize_seconds << ','
        << result.write_seconds << ','
        << result.compute_seconds << ','
        << result.total_seconds << ','
        << result.compute_throughput_mbps() << ','
        << result.throughput_mbps() << ','
        << output_path.string()
        << "\n";
}

void print_summary_row(
    std::ostream &out,
    const char *row_type,
    const AggregateStats &stats
) {
    const double dynamic_hash_load =
        stats.dynamic_hash_capacity_sum != 0U
            ? static_cast<double>(stats.dynamic_bucket_slots_used) /
                  static_cast<double>(stats.dynamic_hash_capacity_sum)
            : 0.0;
    out
        << row_type << ','
        << ',' // layer_sum
        << stats.layers << ','
        << stats.effective_threads << ','
        << stats.input_live << ','
        << stats.scanned_source_rows << ','
        << stats.output_rows << ','
        << stats.throughput_live << ','
        << ',' // ex_input_live
        << ',' // ex_primary_live
        << ',' // ex_match
        << ',' // dynamic_reserve_factor
        << stats.generation_retries << ','
        << stats.dynamic_hash_capacity_sum << ','
        << stats.dynamic_bucket_slots_used << ','
        << dynamic_hash_load << ','
        << stats.source_load_seconds << ','
        << stats.source_read_bytes << ','
        << stats.source_backend_read_ops << ','
        << stats.target_logical_bytes << ','
        << stats.target_backend_write_bytes << ','
        << stats.target_backend_write_ops << ','
        << stats.generation_seconds << ','
        << stats.finalize_seconds << ','
        << stats.write_seconds << ','
        << stats.compute_seconds << ','
        << stats.total_seconds << ','
        << mbps(stats.throughput_live, stats.compute_seconds) << ','
        << mbps(stats.throughput_live, stats.total_seconds) << ','
        << "\n";
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
        if (key == "--target-sum") {
            args.target_sum = static_cast<uint32_t>(std::stoul(require_value("--target-sum")));
        } else if (key == "--pattern") {
            args.pattern = require_value("--pattern");
        } else if (key == "--singlechunk-mode") {
            const std::string mode = require_value("--singlechunk-mode");
            if (mode != "strict") {
                throw std::invalid_argument("--singlechunk-mode only supports strict");
            }
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--target-extra") {
            args.target_extra_override = static_cast<uint32_t>(std::stoul(require_value("--target-extra")));
        } else if (key == "--extra-steps") {
            args.extra_steps = static_cast<uint32_t>(std::stoul(require_value("--extra-steps")));
        } else if (key == "--source") {
            args.sources.emplace_back(require_value("--source"));
        } else if (key == "--output") {
            args.output = require_value("--output");
        } else if (key == "--output-dir") {
            args.output_dir = require_value("--output-dir");
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value("--stats-csv");
        } else if (key == "--ex-stats-csv") {
            args.ex_stats_csv = require_value("--ex-stats-csv");
        } else if (key == "--num-threads") {
            args.num_threads = std::stoi(require_value("--num-threads"));
        } else if (key == "--batch-size") {
            args.batch_size = static_cast<uint32_t>(std::stoul(require_value("--batch-size")));
        } else if (key == "--pending-buffer") {
            args.pending_buffer = static_cast<uint32_t>(std::stoul(require_value("--pending-buffer")));
        } else if (key == "--cell-chunk-size") {
            args.cell_chunk_size = static_cast<uint32_t>(std::stoul(require_value("--cell-chunk-size")));
            args.cell_chunk_size_explicit = true;
        } else if (key == "--cell-chunk-count") {
            args.cell_chunk_count = static_cast<uint32_t>(std::stoul(require_value("--cell-chunk-count")));
        } else if (key == "--warmup-extra") {
            args.warmup_extra = static_cast<uint32_t>(std::stoul(require_value("--warmup-extra")));
        } else if (key == "--expected-output-rows") {
            args.expected_output_rows = static_cast<uint64_t>(std::stoull(require_value("--expected-output-rows")));
        } else if (key == "--target-output-io") {
            args.target_output_io = require_value("--target-output-io");
            args.position_io = args.target_output_io;
        } else if (key == "--target-direct-queue-depth") {
            args.target_direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value("--target-direct-queue-depth")));
            args.position_direct_queue_depth = args.target_direct_queue_depth;
        } else if (key == "--target-direct-overlapped") {
            args.target_direct_overlapped = true;
        } else if (key == "--position-io") {
            args.position_io = require_value("--position-io");
        } else if (key == "--position-direct-queue-depth") {
            args.position_direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value("--position-direct-queue-depth")));
        } else if (key == "--cell-modulus" || key == "--family-modulus") {
            args.cell_modulus = static_cast<uint32_t>(std::stoul(require_value(key.c_str())));
        } else if (key == "--free9-sum-mode") {
            const std::string mode = require_value("--free9-sum-mode");
            if (mode != "raw") {
                throw std::invalid_argument("--free9-sum-mode only supports raw sums");
            }
        } else if (key == "--no-verify-layer-rows") {
            args.verify_layer_rows = false;
        } else if (key == "--detail-timing") {
            args.detail_timing = true;
        } else if (key == "--no-detail-timing") {
            args.detail_timing = false;
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.pattern != "free9") {
        throw std::invalid_argument("bc_single_chunk_generation_bench currently supports --pattern free9 only");
    }
    if (args.cell_modulus == 0U || args.cell_modulus > 256U) {
        throw std::invalid_argument("--cell-modulus must be in 1..256");
    }
    if (args.target_rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large");
    }
    if (args.target_extra_override != 0U && (args.target_extra_override & 1U) != 0U) {
        throw std::invalid_argument("--target-extra must be an even semantic sum when provided");
    }
    if (args.batch_size == 0U || args.pending_buffer == 0U) {
        throw std::invalid_argument("--batch-size and --pending-buffer must be non-zero");
    }
    if (args.target_output_io != "buffered" && args.target_output_io != "direct") {
        throw std::invalid_argument("--target-output-io must be buffered or direct");
    }
    if (args.target_direct_queue_depth == 0U) {
        throw std::invalid_argument("--target-direct-queue-depth must be non-zero");
    }
    if (args.position_io != "buffered" &&
        args.position_io != "direct" &&
        args.position_io != "direct-auto") {
        throw std::invalid_argument("--position-io must be buffered, direct, or direct-auto");
    }
    if (args.position_direct_queue_depth == 0U) {
        throw std::invalid_argument("--position-direct-queue-depth must be non-zero");
    }
    if (args.target_sum != 0U && args.sources.empty()) {
        throw std::invalid_argument("single-layer mode requires at least one --source position file");
    }
    return args;
}

int run_single_layer(const Args &args, std::ostream &out) {
    const BCLut lut = make_free9_lut();
    const std::array<uint32_t, 16U> tile_sums = free9_tile_sums();
    const std::vector<BC::LayerSum> possible_8tile_sums =
        free9_possible_8tile_sums(tile_sums);
    const BCPositionCellLayout target_layout =
        make_layout(args.target_sum, possible_8tile_sums, args.cell_modulus);
    std::vector<std::unique_ptr<BCPositionStreamingReader>> readers;
    std::vector<BCSingleChunkGenerationSource> sources;
    readers.reserve(args.sources.size());
    sources.reserve(args.sources.size());

    uint16_t side_unit = 0U;
    uint64_t source_rows = 0U;
    uint64_t source_bucket_count = 0U;
    uint64_t source_rank_payload_bytes = 0U;
    for (const std::filesystem::path &source_path : args.sources) {
        auto reader = open_position_reader_path(args, source_path, lut);
        if (side_unit == 0U) {
            side_unit = reader->axis().family_unit();
        } else if (side_unit != reader->axis().family_unit()) {
            throw std::invalid_argument("all source files must use the same side unit");
        }
        const uint32_t source_total = reader->axis().total_coord();
        const uint32_t target_total = target_layout.total_coord();
        if (target_total <= source_total || target_total - source_total > 2U) {
            throw std::invalid_argument("source/target total_coord must differ by 1 (+2) or 2 (+4)");
        }
        const uint32_t delta = target_total - source_total;
        sources.push_back(BCSingleChunkGenerationSource{
            reader.get(),
            static_cast<uint8_t>(delta),
            static_cast<BC::SpawnDeltaCoord>(delta),
            resolve_cell_chunk_size(args, reader->cell_count())
        });
        source_rows += descriptor_rows(*reader);
        source_bucket_count += descriptor_bucket_count(*reader);
        source_rank_payload_bytes += reader->header().rank_payload_bytes;
        readers.push_back(std::move(reader));
    }

    BCResidentGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    options.pending_insert_buffer_size = args.pending_buffer;
    options.collect_timing = args.detail_timing;
    options.finalize_options.keyvalue_sort = nullptr;
    options.finalize_options.simd_sort_min_bucket_count = 10000U;

    BCResidentGenerationResult result;
    if (args.output.empty()) {
        result = BC::generate_single_chunk_position_layer(lut, target_layout, sources, options);
    } else {
        std::unique_ptr<BC::BCWritableFile> writer = open_position_writer(args, args.output);
        result = BC::generate_single_chunk_position_layer_to_file(
            lut,
            target_layout,
            sources,
            *writer,
            options
        );
    }
    if (args.expected_output_rows != 0U && result.output_success_rows != args.expected_output_rows) {
        throw std::runtime_error("SingleChunk output row count does not match expected rows");
    }
    (void)source_bucket_count;
    (void)source_rank_payload_bytes;
    print_header(out);
    print_layer_row(
        out,
        args.target_sum,
        source_rows,
        0U,
        args.expected_output_rows,
        args.expected_output_rows == 0U || result.output_success_rows == args.expected_output_rows ? 1U : 0U,
        options.dynamic_reserve_factor,
        result,
        args.output
    );
    AggregateStats total;
    accumulate(total, result, source_rows);
    print_summary_row(out, "total", total);
    return 0;
}

int run_free9_chain(const Args &args, std::ostream &out) {
    std::filesystem::create_directories(args.output_dir);
    const std::array<uint32_t, 16U> tile_sums = free9_tile_sums();
    const std::vector<BC::LayerSum> possible_8tile_sums =
        free9_possible_8tile_sums(tile_sums);
    const BCLut lut = make_free9_lut();
    const uint64_t seed_board = load_free9_seed_board();
    const uint32_t seed_sum = board_semantic_sum(seed_board, tile_sums);
    const uint32_t forward_steps = ex_forward_steps(args);
    const uint32_t final_sum = seed_sum + forward_steps * 2U;
    const bool ex_terminal_mode = args.target_extra_override == 0U && final_sum >= seed_sum + 4U;
    const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
    const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
    const uint32_t success_check_min_source_layer_sum = seed_sum + 2U * (docheck_step + 1U);
    const std::vector<uint8_t> success_shifts = all_board_success_shifts();
    const ExExpectedStats ex_expected = load_ex_expected_stats(args.ex_stats_csv, seed_sum);

    std::vector<uint64_t> initial_boards = generate_free9_initial_boards();
    check(initial_boards.size() == 21283U, "free9 initial board count must match EX");

    BCResidentGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    options.pending_insert_buffer_size = args.pending_buffer;
    options.tile_sum_values = &tile_sums;
    options.collect_timing = args.detail_timing;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    options.finalize_options.keyvalue_sort = nullptr;
    options.finalize_options.simd_sort_min_bucket_count = 10000U;
    std::map<uint32_t, LayerFile> layers;
    LayerFile seed_layer =
        write_initial_layer_file(
            args,
            lut,
            make_layout(seed_sum, possible_8tile_sums, args.cell_modulus),
            initial_boards,
            tile_sums
        );
    if (seed_layer.rows != initial_boards.size()) {
        throw std::runtime_error("free9 initial layer row count mismatch");
    }
    if (ex_expected.enabled) {
        const auto init_it = ex_expected.by_layer_sum.find(seed_sum);
        if (init_it == ex_expected.by_layer_sum.end()) {
            throw std::runtime_error("EX stats CSV is missing init row");
        }
        if (seed_layer.rows != init_it->second.input_live ||
            seed_layer.rows != init_it->second.primary_live) {
            throw std::runtime_error("BC seed layer rows do not match EX stats");
        }
    }
    layers.emplace(seed_sum, std::move(seed_layer));

    std::vector<double> reserve_need_history;
    double retry_guard_factor = 0.0;
    AggregateStats aggregate;
    AggregateStats warm;
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_to_primary;

    print_header(out);

    for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
        const uint32_t current_step = (layer_sum - seed_sum) / 2U - 1U;
        const double dynamic_reserve_factor =
            reserve_factor_for_step(current_step, reserve_need_history, retry_guard_factor);
        retry_guard_factor = 0.0;
        options.dynamic_reserve_factor = dynamic_reserve_factor;

        const BCPositionCellLayout target_layout =
            make_layout(layer_sum, possible_8tile_sums, args.cell_modulus);
        const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
        const std::filesystem::path final_path = layer_path(args, layer_sum);
        options.keep_only_success_generated_boards = terminal;
        options.keep_only_success_secondary_generated_boards = false;

        const auto current_it = layers.find(layer_sum - 2U);
        if (current_it == layers.end()) {
            throw std::runtime_error("SingleChunk chain lost required current layer");
        }
        const uint64_t input_live = current_it->second.rows;
        uint64_t scanned_source_rows_expected = 0U;
        uint64_t source_bucket_count = 0U;
        uint64_t source_rank_payload_bytes = 0U;
        BCResidentGenerationResult result;

        {
            const LayerFile &current_layer = current_it->second;
            std::unique_ptr<BCPositionStreamingReader> current_reader =
                open_position_reader(args, current_layer, lut);
            const uint32_t current_cell_chunk_size =
                resolve_cell_chunk_size(args, current_reader->cell_count());
            scanned_source_rows_expected = current_layer.rows;
            source_bucket_count = current_layer.bucket_count;
            source_rank_payload_bytes = current_layer.rank_payload_bytes;
            std::optional<BCPositionCellLayout> secondary_layout;
            if (!terminal) {
                secondary_layout.emplace(
                    make_layout(layer_sum + 2U, possible_8tile_sums, args.cell_modulus)
                );
                options.keep_only_success_secondary_generated_boards =
                    ex_terminal_mode && layer_sum + 2U == final_primary_sum;
            }
            std::unique_ptr<BC::BCWritableFile> writer = open_position_writer(args, final_path);
            BC::BCSingleChunkGenerationStepResult step =
                BC::generate_single_chunk_position_layer_strict_to_file(
                    lut,
                    target_layout,
                    *current_reader,
                    current_cell_chunk_size,
                    std::move(carry_to_primary),
                    secondary_layout ? &*secondary_layout : nullptr,
                    *writer,
                    options
            );
            carry_to_primary = std::move(step.next_carry);
            result = combine_strict_step_stats(std::move(step));
            current_reader.reset();
            writer.reset();
        }

        if (args.verify_layer_rows &&
            result.source_boards_scanned != 0U &&
            result.source_boards_scanned != scanned_source_rows_expected) {
            throw std::runtime_error("SingleChunk scanned source rows mismatch descriptors");
        }

        LayerFile generated_layer = inspect_layer_file(
            args,
            layer_sum,
            final_path,
            result.target_position_file_logical_bytes,
            lut
        );
        if (args.verify_layer_rows && generated_layer.rows != result.output_success_rows) {
            throw std::runtime_error("SingleChunk output rows mismatch generated file descriptors");
        }

        uint64_t ex_input_live = 0U;
        uint64_t ex_primary_live = 0U;
        uint32_t ex_match = 0U;
        if (ex_expected.enabled) {
            const auto ex_it = ex_expected.by_layer_sum.find(layer_sum);
            if (ex_it == ex_expected.by_layer_sum.end()) {
                throw std::runtime_error("EX stats CSV is missing generated layer " + std::to_string(layer_sum));
            }
            ex_input_live = ex_it->second.input_live;
            ex_primary_live = ex_it->second.primary_live;
            if (input_live != ex_input_live) {
                throw std::runtime_error("BC SingleChunk input_live does not match EX stats for layer " + std::to_string(layer_sum));
            }
            if (result.output_success_rows != ex_primary_live) {
                throw std::runtime_error("BC SingleChunk primary_live does not match EX stats for layer " + std::to_string(layer_sum));
            }
            ex_match = 1U;
        }

        print_layer_row(
            out,
            layer_sum,
            input_live,
            ex_input_live,
            ex_primary_live,
            ex_match,
            dynamic_reserve_factor,
            result,
            final_path
        );

        accumulate(aggregate, result, input_live);
        if (layer_sum >= seed_sum + args.warmup_extra) {
            accumulate(warm, result, input_live);
        }

        const double need = observed_reserve_need(
            source_bucket_count,
            source_rank_payload_bytes,
            result
        );
        if (need > 0.0) {
            reserve_need_history.push_back(need);
        }
        if (result.generation_retries != 0U) {
            retry_guard_factor =
                std::min(kBCDefaultReserveFactor, dynamic_reserve_factor * kBCLearnedReserveRetryGuard);
        }

        layers[layer_sum] = std::move(generated_layer);
    }

    print_summary_row(out, "total", aggregate);
    if (warm.layers != 0U) {
        print_summary_row(out, "warm", warm);
    }

    const char *perf_assert = std::getenv("BC_PERF_ASSERT");
    if (perf_assert != nullptr && std::string(perf_assert) == "1") {
        const AggregateStats &basis = warm.layers != 0U ? warm : aggregate;
        if (mbps(basis.throughput_live, basis.total_seconds) < 110.0) {
            std::cerr << "BC_PERF_ASSERT failed: SingleChunk total throughput < 110 Mboards/s\n";
            return 2;
        }
    }
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
        if (args.target_sum != 0U) {
            return run_single_layer(args, *out);
        }
        return run_free9_chain(args, *out);
    } catch (const std::exception &ex) {
        std::cerr << "bc_single_chunk_generation_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
