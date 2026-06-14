#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCBoardOps.h"
#include "BCDirectFileIO.h"
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
    uint32_t pending_buffer = 512U;
    uint32_t warmup_extra = 16U;
    uint32_t cell_modulus = 29U;
    bool verify_layer_rows = true;
    bool detail_timing = true;
    bool file_backed = false;
    std::string target_output_io = "memory";
    uint32_t target_direct_queue_depth = 8U;
    bool target_direct_overlapped = false;
    std::filesystem::path file_dir = std::filesystem::path("tmp") / "bc_generation_compute_files";
    std::filesystem::path ex_stats_csv;
};

struct ResidentLayer {
    uint32_t layer_sum = 0U;
    std::vector<uint8_t> bytes;
    std::unique_ptr<BCPositionLayerReader> reader;
    uint64_t rows = 0U;
    uint64_t bucket_count = 0U;
    uint64_t rank_payload_bytes = 0U;
};

struct AggregateStats {
    uint32_t layers = 0U;
    int effective_threads = 0;
    uint64_t source_boards = 0U;
    uint64_t output_rows = 0U;
    uint64_t throughput_live = 0U;
    uint64_t position_bytes = 0U;
    uint64_t generation_retries = 0U;
    uint32_t max_dynamic_hash_capacity = 0U;
    uint64_t dynamic_hash_capacity_sum = 0U;
    uint64_t dynamic_bucket_slots_used = 0U;
    uint64_t dynamic_bitmap_words_used = 0U;
    uint64_t dynamic_bitmap_words_allocated = 0U;
    uint64_t dynamic_bitmap_words_reserved = 0U;
    uint64_t bucket_count = 0U;
    uint64_t bitmap_live_bits = 0U;
    uint64_t bitmap_logical_bits = 0U;
    uint64_t bitmap_physical_bits = 0U;
    uint64_t rank_payload_bytes = 0U;
    uint64_t target_file_logical_bytes = 0U;
    uint64_t target_write_backend_ops = 0U;
    uint64_t target_write_backend_bytes = 0U;
    double generation_seconds = 0.0;
    double scan_seconds = 0.0;
    double thread_spawn_move_seconds = 0.0;
    double thread_canonical_seconds = 0.0;
    double thread_encode_insert_seconds = 0.0;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double finalize_seconds = 0.0;
    double cleanup_seconds = 0.0;
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

constexpr double kBCDefaultReserveFactor = 2.0;
constexpr double kBCEarlyLayerReserveFactor = 2.0;
constexpr uint32_t kBCEarlyLayerReserveFactorSteps = 10U;
constexpr double kBCLearnedReserveMinFactor = 1.08;
constexpr double kBCLearnedReserveQuantileGuard = 1.10;
constexpr double kBCLearnedReserveLastGuard = 1.12;
constexpr double kBCLearnedReserveRetryGuard = 1.25;
constexpr size_t kBCLearnedReserveHistoryWindow = 32U;

double bc_reserve_need_recent_quantile(const std::vector<double> &history) {
    const size_t begin = history.size() > kBCLearnedReserveHistoryWindow
        ? history.size() - kBCLearnedReserveHistoryWindow
        : 0U;
    std::vector<double> values(history.begin() + static_cast<std::ptrdiff_t>(begin), history.end());
    std::sort(values.begin(), values.end());
    const size_t index = ((values.size() - 1U) * 9U) / 10U;
    return values[index];
}

double bc_regular_reserve_factor(const std::vector<double> &history, double retry_guard_factor) {
    if (history.empty()) {
        return kBCDefaultReserveFactor;
    }
    double factor = kBCLearnedReserveMinFactor;
    factor = std::max(factor, history.back() * kBCLearnedReserveLastGuard);
    factor = std::max(factor, bc_reserve_need_recent_quantile(history) * kBCLearnedReserveQuantileGuard);
    factor = std::max(factor, retry_guard_factor);
    return std::min(kBCDefaultReserveFactor, std::max(kBCLearnedReserveMinFactor, factor));
}

double bc_reserve_factor_for_step(
    uint32_t current_step,
    const std::vector<double> &history,
    double retry_guard_factor
) {
    if (current_step < kBCEarlyLayerReserveFactorSteps) {
        return kBCEarlyLayerReserveFactor;
    }
    return bc_regular_reserve_factor(history, retry_guard_factor);
}

double bc_transition_component_reserve_need(uint64_t current_size, uint64_t next_size, uint64_t padding) {
    if (next_size <= padding) {
        return 0.0;
    }
    if (current_size == 0U) {
        return kBCDefaultReserveFactor;
    }
    return static_cast<double>(next_size - padding) / static_cast<double>(current_size);
}

double bc_transition_reserve_need(
    const ResidentLayer &current,
    const BitmapStats &next
) {
    double need = 0.0;
    need = std::max(
        need,
        bc_transition_component_reserve_need(current.bucket_count, next.bucket_count, 4096ULL)
    );
    need = std::max(
        need,
        bc_transition_component_reserve_need(
            current.rank_payload_bytes,
            next.rank_payload_bytes,
            512ULL * 64ULL * sizeof(uint64_t)
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

[[nodiscard]] uint64_t parse_u64_field(const std::vector<std::string> &cells, size_t index, const char *name) {
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
            throw std::runtime_error("EX stats row missing stage field");
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
        } else if (key == "--cell-modulus" || key == "--family-modulus") {
            args.cell_modulus = static_cast<uint32_t>(std::stoul(require_value(key.c_str())));
        } else if (key == "--no-verify-layer-rows") {
            args.verify_layer_rows = false;
        } else if (key == "--no-detail-timing") {
            args.detail_timing = false;
        } else if (key == "--file-backed") {
            args.file_backed = true;
        } else if (key == "--target-output-io") {
            args.target_output_io = require_value("--target-output-io");
        } else if (key == "--target-direct-queue-depth") {
            args.target_direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value("--target-direct-queue-depth")));
        } else if (key == "--target-direct-overlapped") {
            args.target_direct_overlapped = true;
        } else if (key == "--file-dir") {
            args.file_dir = require_value("--file-dir");
        } else if (key == "--ex-stats-csv") {
            args.ex_stats_csv = require_value("--ex-stats-csv");
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
    if (args.cell_modulus == 0U || args.cell_modulus > std::numeric_limits<BC::FamilyId>::max()) {
        throw std::invalid_argument("--cell-modulus must be in 1..65535");
    }
    if (args.target_output_io != "memory" &&
        args.target_output_io != "buffered" &&
        args.target_output_io != "direct") {
        throw std::invalid_argument("--target-output-io must be memory, buffered, or direct");
    }
    if (args.target_direct_queue_depth == 0U) {
        throw std::invalid_argument("--target-direct-queue-depth must be non-zero");
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
    return BC::default_2048_tile_sum_values();
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

BCPositionCellLayout make_layout(uint32_t layer_sum, uint32_t cell_modulus) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free9 BC compute benchmark requires even layer sums");
    }
    return BC::build_modulo_position_cell_layout_for_layer(
        layer_sum,
        2U,
        std::vector<BC::LayerSum>{0U},
        cell_modulus
    );
}

uint64_t descriptor_success_rows_sum(const BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

std::vector<uint8_t> roundtrip_position_file_if_requested(
    const Args &args,
    uint32_t layer_sum,
    const char *label,
    const std::vector<uint8_t> &bytes,
    double &seconds_out,
    uint64_t &bytes_out
) {
    if (!args.file_backed) {
        return bytes;
    }
    const double begin = now_seconds();
    std::filesystem::create_directories(args.file_dir);
    std::ostringstream name;
    name << "bc_" << label << "_" << layer_sum << ".bcpos";
    const std::filesystem::path path = args.file_dir / name.str();
    const std::string io_mode =
        args.target_output_io == "memory" ? std::string("buffered") : args.target_output_io;
    if (io_mode == "buffered") {
        BC::write_position_layer_to_file(path, bytes);
    } else if (io_mode == "direct") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.target_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.target_direct_queue_depth > 1U;
        BC::BCDirectFileWriter writer(path, options);
        writer.prepare_full_overwrite(static_cast<uint64_t>(bytes.size()));
        if (!bytes.empty()) {
            writer.write_at(0U, bytes.data(), static_cast<uint64_t>(bytes.size()));
        }
        writer.flush();
    } else {
        throw std::invalid_argument("--target-output-io must be memory, buffered, or direct");
    }
    std::error_code ec;
    const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("failed to stat file-backed BC position layer: " + ec.message());
    }
    const uint64_t expected_file_size = io_mode == "direct"
        ? BC::bc_direct_align_up(static_cast<uint64_t>(bytes.size()), BC::BCDirectFileIOOptions{}.alignment)
        : static_cast<uint64_t>(bytes.size());
    if (file_size != expected_file_size) {
        throw std::runtime_error("file-backed BC position layer size mismatch");
    }
    std::vector<uint8_t> read_bytes;
    if (io_mode == "buffered") {
        read_bytes = BC::read_position_layer_from_file(path);
    } else {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.target_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.target_direct_queue_depth > 1U;
        options.logical_size = static_cast<uint64_t>(bytes.size());
        BC::BCDirectFileReader reader(path, options);
        read_bytes.resize(bytes.size());
        if (!read_bytes.empty()) {
            reader.read_at(0U, read_bytes.data(), static_cast<uint64_t>(read_bytes.size()));
        }
    }
    if (read_bytes.size() != bytes.size()) {
        throw std::runtime_error("file-backed BC position read size mismatch");
    }
    seconds_out += now_seconds() - begin;
    bytes_out += static_cast<uint64_t>(bytes.size()) * 2ULL;
    return read_bytes;
}

void roundtrip_result_position_if_requested(
    const Args &args,
    uint32_t layer_sum,
    const char *label,
    BCResidentGenerationResult &result
) {
    double file_seconds = 0.0;
    uint64_t file_bytes = 0U;
    result.position_bytes = roundtrip_position_file_if_requested(
        args,
        layer_sum,
        label,
        result.position_bytes,
        file_seconds,
        file_bytes
    );
    (void)file_bytes;
    result.write_seconds += file_seconds;
    result.total_seconds += file_seconds;
}

[[nodiscard]] bool target_output_file_enabled(const Args &args) {
    return args.target_output_io != "memory";
}

[[nodiscard]] std::filesystem::path position_file_path(
    const Args &args,
    uint32_t layer_sum
) {
    std::ostringstream name;
    name << "bc_layer_" << layer_sum << ".bcpos";
    return args.file_dir / name.str();
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_target_position_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.target_output_io == "buffered") {
        return std::make_unique<BC::BCBufferedFileWriter>(path);
    }
    if (args.target_output_io == "direct") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.target_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.target_direct_queue_depth > 1U;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    throw std::invalid_argument("target position writer requested for memory output mode");
}

[[nodiscard]] std::vector<uint8_t> read_target_position_file(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size
) {
    if (logical_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("target position file exceeds addressable vector size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(logical_size));
    if (logical_size == 0U) {
        return bytes;
    }
    if (args.target_output_io == "buffered") {
        BC::BCBufferedFileReader reader(path);
        if (reader.size() != logical_size) {
            throw std::runtime_error("buffered target position file size mismatch");
        }
        reader.read_at(0U, bytes.data(), logical_size);
        return bytes;
    }
    if (args.target_output_io == "direct") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.target_direct_queue_depth;
        options.overlapped = args.target_direct_overlapped || args.target_direct_queue_depth > 1U;
        options.logical_size = logical_size;
        BC::BCDirectFileReader reader(path, options);
        reader.read_at(0U, bytes.data(), logical_size);
        return bytes;
    }
    throw std::invalid_argument("target position reader requested for memory output mode");
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

uint32_t quadrant_semantic_sum(
    uint16_t word,
    const std::array<uint32_t, 16U> &tile_sums
) {
    return tile_sums[word & 0xFU] +
           tile_sums[(word >> 4U) & 0xFU] +
           tile_sums[(word >> 8U) & 0xFU] +
           tile_sums[(word >> 12U) & 0xFU];
}

BC::BCBoardEncodedPosition encode_canonical_quadrants_position_for_family_sums(
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const BC::BCQuadrantWords &q,
    const std::array<uint32_t, 16U> &family_tile_sums
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

    const uint64_t nw_sum = quadrant_semantic_sum(q.nw, family_tile_sums);
    const uint64_t ne_sum = quadrant_semantic_sum(q.ne, family_tile_sums);
    const uint64_t sw_sum = quadrant_semantic_sum(q.sw, family_tile_sums);
    const uint64_t se_sum = quadrant_semantic_sum(q.se, family_tile_sums);
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    const uint16_t family_unit = axis.family_unit();
    auto min_side_coord = [family_unit](uint64_t first_sum, uint64_t second_sum, BC::FamilyCoord &coord_out) {
        const uint64_t min_sum = std::min(first_sum, second_sum);
        uint64_t coord = 0U;
        if (family_unit == 2U) {
            if ((min_sum & 1ULL) != 0ULL) {
                return false;
            }
            coord = min_sum >> 1U;
        } else {
            if (family_unit == 0U || (min_sum % family_unit) != 0U) {
                return false;
            }
            coord = min_sum / family_unit;
        }
        if (coord > std::numeric_limits<BC::FamilyCoord>::max()) {
            return false;
        }
        coord_out = static_cast<BC::FamilyCoord>(coord);
        return true;
    };

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!min_side_coord(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
        !min_side_coord(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
        return out;
    }
    const uint32_t family_count = axis.family_count();
    const BC::FamilyId row_id = layout.try_raw_side_coord_to_physical_index(row_coord);
    const BC::FamilyId col_id = layout.try_raw_side_coord_to_physical_index(col_coord);
    if (row_id == BCPositionCellLayout::kInvalidSideIndex ||
        col_id == BCPositionCellLayout::kInvalidSideIndex) {
        return out;
    }
    const uint64_t cid64 =
        static_cast<uint64_t>(row_id) * static_cast<uint64_t>(family_count) + col_id;
    if (cid64 > std::numeric_limits<CellId>::max()) {
        throw std::overflow_error("BC encoded cell id exceeds CellId");
    }

    const BC::BCEncodedKeyRank encoded =
        BC::bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return out;
    }

    out.cid = static_cast<CellId>(cid64);
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

ResidentLayer write_initial_layer(
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
        const auto encoded = encode_canonical_quadrants_position_for_family_sums(
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
    const BitmapStats bitmap = collect_bitmap_stats(*layer.reader);
    layer.bucket_count = bitmap.bucket_count;
    layer.rank_payload_bytes = bitmap.rank_payload_bytes;
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
    aggregate.position_bytes += static_cast<uint64_t>(result.position_bytes.size());
    aggregate.generation_retries += result.generation_retries;
    aggregate.max_dynamic_hash_capacity =
        std::max<uint32_t>(aggregate.max_dynamic_hash_capacity, result.dynamic_hash_capacity);
    aggregate.dynamic_hash_capacity_sum += result.dynamic_hash_capacity;
    aggregate.dynamic_bucket_slots_used += result.dynamic_bucket_slots_used;
    aggregate.dynamic_bitmap_words_used += result.dynamic_bitmap_words_used;
    aggregate.dynamic_bitmap_words_allocated += result.dynamic_bitmap_words_allocated;
    aggregate.dynamic_bitmap_words_reserved += result.dynamic_bitmap_words_reserved;
    aggregate.bucket_count += bitmap_stats.bucket_count;
    aggregate.bitmap_live_bits += bitmap_stats.live_bits;
    aggregate.bitmap_logical_bits += bitmap_stats.logical_bits;
    aggregate.bitmap_physical_bits += bitmap_stats.physical_bits;
    aggregate.rank_payload_bytes += bitmap_stats.rank_payload_bytes;
    aggregate.target_file_logical_bytes += result.target_position_file_logical_bytes;
    aggregate.target_write_backend_ops += result.target_position_write_backend_ops;
    aggregate.target_write_backend_bytes += result.target_position_write_backend_bytes;
    aggregate.generation_seconds += result.generation_seconds;
    aggregate.scan_seconds += result.scan_seconds;
    aggregate.thread_spawn_move_seconds += result.thread_spawn_move_seconds;
    aggregate.thread_canonical_seconds += result.thread_canonical_seconds;
    aggregate.thread_encode_insert_seconds += result.thread_encode_insert_seconds;
    aggregate.prepare_seconds += result.prepare_seconds;
    aggregate.work_seconds += result.work_seconds;
    aggregate.finalize_seconds += result.finalize_seconds;
    aggregate.cleanup_seconds += result.cleanup_seconds;
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

void print_csv_header() {
    std::cout
        << "row_type,layer_sum,layers,effective_threads,input_live,primary_live,secondary_live,"
        << "throughput_live,ex_input_live,ex_primary_live,ex_secondary_live,ex_match,"
        << "ex_checked_layers,dynamic_reserve_factor,generation_retries,dynamic_hash_capacity,"
        << "dynamic_hash_capacity_sum,dynamic_bucket_slots_used,dynamic_hash_load,"
        << "dynamic_bitmap_words_reserved,"
        << "bucket_count,bitmap_live_bits,bitmap_logical_bits,bitmap_density,"
        << "bitmap_physical_density,rank_payload_bytes,position_bytes,target_file_logical_bytes,"
        << "target_write_backend_ops,target_write_backend_bytes,generation_seconds,"
        << "prepare_seconds,work_seconds,finalize_seconds,write_seconds,compute_seconds,"
        << "total_seconds,compute_throughput_mbps,total_throughput_mbps,"
        << "source_board_mbps,output_board_mbps,avg_hot_threads,"
        << "thread_scan_spawn_move_seconds,thread_canonical_seconds,thread_encode_insert_seconds\n";
}

void print_summary_row(
    const char *row_type,
    const AggregateStats &stats,
    uint32_t reported_layers,
    uint32_t ex_checked_layers
) {
    const double bitmap_density = stats.bitmap_logical_bits != 0U
        ? static_cast<double>(stats.bitmap_live_bits) / static_cast<double>(stats.bitmap_logical_bits)
        : 0.0;
    const double bitmap_physical_density = stats.bitmap_physical_bits != 0U
        ? static_cast<double>(stats.bitmap_live_bits) / static_cast<double>(stats.bitmap_physical_bits)
        : 0.0;
    std::cout
        << row_type << ','
        << ',' // layer_sum
        << reported_layers << ','
        << stats.effective_threads << ','
        << stats.source_boards << ','
        << stats.output_rows << ','
        << ',' // secondary_live
        << stats.throughput_live << ','
        << ',' // ex_input_live
        << ',' // ex_primary_live
        << ',' // ex_secondary_live
        << ',' // ex_match
        << ex_checked_layers << ','
        << ',' // dynamic_reserve_factor
        << stats.generation_retries << ','
        << stats.max_dynamic_hash_capacity << ','
        << stats.dynamic_hash_capacity_sum << ','
        << stats.dynamic_bucket_slots_used << ','
        << (stats.dynamic_hash_capacity_sum != 0U
                ? static_cast<double>(stats.dynamic_bucket_slots_used) /
                  static_cast<double>(stats.dynamic_hash_capacity_sum)
                : 0.0) << ','
        << stats.dynamic_bitmap_words_reserved << ','
        << stats.bucket_count << ','
        << stats.bitmap_live_bits << ','
        << stats.bitmap_logical_bits << ','
        << bitmap_density << ','
        << bitmap_physical_density << ','
        << stats.rank_payload_bytes << ','
        << stats.position_bytes << ','
        << stats.target_file_logical_bytes << ','
        << stats.target_write_backend_ops << ','
        << stats.target_write_backend_bytes << ','
        << stats.generation_seconds << ','
        << stats.prepare_seconds << ','
        << stats.work_seconds << ','
        << stats.finalize_seconds << ','
        << stats.write_seconds << ','
        << stats.compute_seconds << ','
        << stats.total_seconds << ','
        << mbps(stats.throughput_live, stats.compute_seconds) << ','
        << mbps(stats.throughput_live, stats.total_seconds) << ','
        << mbps(stats.source_boards, stats.compute_seconds) << ','
        << mbps(stats.output_rows, stats.compute_seconds) << ','
        << (stats.generation_seconds > 0.0 ? thread_hot_seconds(stats) / stats.generation_seconds : 0.0) << ','
        << stats.thread_spawn_move_seconds << ','
        << stats.thread_canonical_seconds << ','
        << stats.thread_encode_insert_seconds
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
        const ExExpectedStats ex_expected = load_ex_expected_stats(args.ex_stats_csv, seed_sum);
        uint32_t ex_checked_layers = 0U;
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
            write_initial_layer(lut, make_layout(seed_sum, args.cell_modulus), initial_boards, tile_sums);
        if (args.file_backed) {
            double initial_file_seconds = 0.0;
            uint64_t initial_file_bytes = 0U;
            current.bytes = roundtrip_position_file_if_requested(
                args,
                current.layer_sum,
                "initial",
                current.bytes,
                initial_file_seconds,
                initial_file_bytes
            );
            current.reader = std::make_unique<BCPositionLayerReader>(current.bytes, lut);
            current.rows = descriptor_success_rows_sum(*current.reader);
        }
        if (target_output_file_enabled(args)) {
            std::filesystem::create_directories(args.file_dir);
        }
        check(current.rows == initial_boards.size(), "free9 initial layer row count mismatch");
        if (ex_expected.enabled) {
            const auto init_it = ex_expected.by_layer_sum.find(seed_sum);
            if (init_it == ex_expected.by_layer_sum.end()) {
                throw std::runtime_error("EX stats CSV is missing init row for seed layer");
            }
            if (current.rows != init_it->second.primary_live ||
                current.rows != init_it->second.input_live) {
                throw std::runtime_error("BC initial layer row count does not match EX stats CSV");
            }
        }
        std::unique_ptr<ResidentLayer> carry_layer;
        std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_dynamic;
        std::vector<double> reserve_need_history;
        double retry_guard_factor = 0.0;

        AggregateStats aggregate;
        AggregateStats warm;

        std::cout << std::setprecision(9);
        print_csv_header();

        const bool ex_terminal_mode = args.target_extra_override == 0U && final_sum >= seed_sum + 4U;
        const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
        for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
            const uint32_t current_step = (layer_sum - seed_sum) / 2U - 1U;
            const double dynamic_reserve_factor =
                bc_reserve_factor_for_step(current_step, reserve_need_history, retry_guard_factor);
            retry_guard_factor = 0.0;
            options.dynamic_reserve_factor = dynamic_reserve_factor;
            const BCPositionCellLayout target_layout = make_layout(layer_sum, args.cell_modulus);
            const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
            const bool need_secondary = layer_sum + 2U <= final_sum;
            BCPositionCellLayout secondary_layout =
                need_secondary ? make_layout(layer_sum + 2U, args.cell_modulus) : target_layout;
            BC::BCResidentGenerationPairResult pair;
            std::filesystem::path primary_target_path;
            if (target_output_file_enabled(args)) {
                primary_target_path = position_file_path(args, layer_sum);
                std::unique_ptr<BC::BCWritableFile> primary_writer =
                    open_target_position_writer(args, primary_target_path);
                pair = BC::generate_resident_position_layer_pair_with_mutable_carry_to_file(
                    lut,
                    target_layout,
                    *current.reader,
                    std::move(carry_dynamic),
                    need_secondary ? &secondary_layout : nullptr,
                    *primary_writer,
                    nullptr,
                    options
                );
                primary_writer.reset();
                pair.primary.position_bytes = read_target_position_file(
                    args,
                    primary_target_path,
                    pair.primary.target_position_file_logical_bytes
                );
            } else {
                pair = BC::generate_resident_position_layer_pair_with_mutable_carry(
                    lut,
                    target_layout,
                    *current.reader,
                    std::move(carry_dynamic),
                    need_secondary ? &secondary_layout : nullptr,
                    options
                );
                carry_layer.reset();
            }
            BCResidentGenerationResult &result = pair.primary;
            const uint64_t source_rows = current.rows;
            current.reader.reset();
            current.bytes.clear();
            current.bytes.shrink_to_fit();
            if (args.verify_layer_rows && source_rows != result.source_boards_scanned) {
                throw std::runtime_error("source descriptor rows do not match generated scanned rows");
            }

            const uint64_t mutable_secondary_live =
                pair.has_secondary ? pair.secondary.output_success_rows : 0U;
            (void)mutable_secondary_live;
            uint64_t secondary_live = 0U;
            uint64_t secondary_position_bytes =
                pair.has_secondary ? static_cast<uint64_t>(pair.secondary.position_bytes.size()) : 0U;
            const uint64_t target_file_logical_bytes =
                result.target_position_file_logical_bytes +
                (pair.has_secondary ? pair.secondary.target_position_file_logical_bytes : 0U);
            const uint64_t target_write_backend_ops =
                result.target_position_write_backend_ops +
                (pair.has_secondary ? pair.secondary.target_position_write_backend_ops : 0U);
            const uint64_t target_write_backend_bytes =
                result.target_position_write_backend_bytes +
                (pair.has_secondary ? pair.secondary.target_position_write_backend_bytes : 0U);
            double terminal_compact_seconds = 0.0;
            if (terminal) {
                const double compact_begin = now_seconds();
                ResidentLayer compacted_primary = compact_position_layer_to_success(
                    layer_sum,
                    lut,
                    result.position_bytes,
                    args.target_rank,
                    success_shifts
                );
                const double compact_seconds = now_seconds() - compact_begin;
                terminal_compact_seconds = compact_seconds;
                result.position_bytes = std::move(compacted_primary.bytes);
                result.output_success_rows = compacted_primary.rows;
                result.finalize_seconds += compact_seconds;
                result.compute_seconds += compact_seconds;
                result.total_seconds += compact_seconds;
                pair.secondary.position_bytes.clear();
                pair.secondary.output_success_rows = 0U;
                secondary_live = 0U;
                secondary_position_bytes = 0U;
            }

            if (!target_output_file_enabled(args)) {
                roundtrip_result_position_if_requested(args, layer_sum, "primary", result);
                if (pair.has_secondary) {
                    roundtrip_result_position_if_requested(args, layer_sum + 2U, "secondary", pair.secondary);
                    secondary_position_bytes = static_cast<uint64_t>(pair.secondary.position_bytes.size());
                }
            }

            uint64_t ex_input_live = 0U;
            uint64_t ex_primary_live = 0U;
            uint64_t ex_secondary_live = 0U;
            uint32_t ex_match = 0U;
            if (ex_expected.enabled) {
                const auto ex_it = ex_expected.by_layer_sum.find(layer_sum);
                if (ex_it == ex_expected.by_layer_sum.end()) {
                    throw std::runtime_error("EX stats CSV is missing generated layer " + std::to_string(layer_sum));
                }
                ex_input_live = ex_it->second.input_live;
                ex_primary_live = ex_it->second.primary_live;
                ex_secondary_live = ex_it->second.secondary_live;
                if (source_rows != ex_input_live) {
                    throw std::runtime_error("BC input_live does not match EX stats for layer " + std::to_string(layer_sum));
                }
                if (result.output_success_rows != ex_primary_live) {
                    throw std::runtime_error("BC primary_live does not match EX stats for layer " + std::to_string(layer_sum));
                }
                if (secondary_live != 0U && secondary_live != ex_secondary_live) {
                    throw std::runtime_error("BC secondary_live does not match EX stats for layer " + std::to_string(layer_sum));
                }
                ex_match = 1U;
                ++ex_checked_layers;
            }

            const BitmapStats primary_bitmap = collect_bitmap_stats(lut, result.position_bytes);
            const BitmapStats secondary_bitmap =
                pair.has_secondary && !pair.secondary.position_bytes.empty()
                    ? collect_bitmap_stats(lut, pair.secondary.position_bytes)
                    : BitmapStats{};
            double observed_reserve_need = bc_transition_reserve_need(current, primary_bitmap);
            if (pair.has_secondary) {
                observed_reserve_need =
                    std::max(observed_reserve_need, bc_transition_reserve_need(current, secondary_bitmap));
            }
            if (observed_reserve_need > 0.0) {
                reserve_need_history.push_back(observed_reserve_need);
            }
            if (result.generation_retries != 0U) {
                retry_guard_factor =
                    std::min(kBCDefaultReserveFactor, dynamic_reserve_factor * kBCLearnedReserveRetryGuard);
            }
            const uint64_t expected_output_rows = result.output_success_rows;
            const uint64_t throughput_live = result.ex_generate_throughput_live();
            const uint64_t row_bucket_count =
                primary_bitmap.bucket_count + secondary_bitmap.bucket_count;
            const uint64_t row_bitmap_live_bits =
                primary_bitmap.live_bits + secondary_bitmap.live_bits;
            const uint64_t row_bitmap_logical_bits =
                primary_bitmap.logical_bits + secondary_bitmap.logical_bits;
            const uint64_t row_bitmap_physical_bits =
                primary_bitmap.physical_bits + secondary_bitmap.physical_bits;
            const uint64_t row_rank_payload_bytes =
                primary_bitmap.rank_payload_bytes + secondary_bitmap.rank_payload_bytes;
            const double row_bitmap_density = row_bitmap_logical_bits != 0U
                ? static_cast<double>(row_bitmap_live_bits) / static_cast<double>(row_bitmap_logical_bits)
                : 0.0;
            const double row_bitmap_physical_density = row_bitmap_physical_bits != 0U
                ? static_cast<double>(row_bitmap_live_bits) / static_cast<double>(row_bitmap_physical_bits)
                : 0.0;
            const double dynamic_hash_load = result.dynamic_hash_capacity != 0U
                ? static_cast<double>(result.dynamic_bucket_slots_used) /
                  static_cast<double>(result.dynamic_hash_capacity)
                : 0.0;
            std::cout
                << "layer,"
                << layer_sum << ','
                << 1U << ','
                << result.effective_threads << ','
                << source_rows << ','
                << result.output_success_rows << ','
                << secondary_live << ','
                << throughput_live << ','
                << ex_input_live << ','
                << ex_primary_live << ','
                << ex_secondary_live << ','
                << ex_match << ','
                << (ex_match != 0U ? 1U : 0U) << ','
                << dynamic_reserve_factor << ','
                << result.generation_retries << ','
                << result.dynamic_hash_capacity << ','
                << result.dynamic_hash_capacity << ','
                << result.dynamic_bucket_slots_used << ','
                << dynamic_hash_load << ','
                << result.dynamic_bitmap_words_reserved << ','
                << row_bucket_count << ','
                << row_bitmap_live_bits << ','
                << row_bitmap_logical_bits << ','
                << row_bitmap_density << ','
                << row_bitmap_physical_density << ','
                << row_rank_payload_bytes << ','
                << (result.position_bytes.size() + secondary_position_bytes) << ','
                << target_file_logical_bytes << ','
                << target_write_backend_ops << ','
                << target_write_backend_bytes << ','
                << result.generation_seconds << ','
                << result.prepare_seconds << ','
                << result.work_seconds << ','
                << result.finalize_seconds << ','
                << result.write_seconds << ','
                << result.compute_seconds << ','
                << result.total_seconds << ','
                << result.compute_throughput_mbps() << ','
                << result.throughput_mbps() << ','
                << result.source_board_mbps() << ','
                << result.output_board_mbps() << ','
                << result.avg_hot_threads() << ','
                << result.thread_spawn_move_seconds << ','
                << result.thread_canonical_seconds << ','
                << result.thread_encode_insert_seconds
                << "\n";

            accumulate(aggregate, result, primary_bitmap);
            if (terminal && secondary_live != 0U) {
                add_terminal_secondary_to_aggregate(
                    aggregate,
                    secondary_live,
                    secondary_position_bytes,
                    secondary_bitmap
                );
                aggregate.target_file_logical_bytes += pair.secondary.target_position_file_logical_bytes;
                aggregate.target_write_backend_ops += pair.secondary.target_position_write_backend_ops;
                aggregate.target_write_backend_bytes += pair.secondary.target_position_write_backend_bytes;
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
                    warm.target_file_logical_bytes += pair.secondary.target_position_file_logical_bytes;
                    warm.target_write_backend_ops += pair.secondary.target_position_write_backend_ops;
                    warm.target_write_backend_bytes += pair.secondary.target_position_write_backend_bytes;
                }
            }

            current = make_generated_layer(layer_sum, lut, std::move(result));
            current.bucket_count = primary_bitmap.bucket_count;
            current.rank_payload_bytes = primary_bitmap.rank_payload_bytes;
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
                carry_dynamic.reset();
            } else if (need_secondary) {
                if (pair.secondary_carry) {
                    carry_dynamic = std::move(pair.secondary_carry);
                    carry_layer.reset();
                } else {
                    ResidentLayer secondary_layer =
                        make_generated_layer(layer_sum + 2U, lut, std::move(pair.secondary));
                    secondary_layer.bucket_count = secondary_bitmap.bucket_count;
                    secondary_layer.rank_payload_bytes = secondary_bitmap.rank_payload_bytes;
                    carry_layer = std::make_unique<ResidentLayer>(std::move(secondary_layer));
                    carry_dynamic.reset();
                }
            } else {
                carry_layer.reset();
                carry_dynamic.reset();
            }
        }

        print_summary_row("total", aggregate, aggregate.layers, ex_checked_layers);
        if (warm.layers != 0U) {
            print_summary_row("warm", warm, warm.layers, ex_checked_layers);
        }

        const char *perf_assert = std::getenv("BC_PERF_ASSERT");
        if (perf_assert != nullptr && std::string(perf_assert) == "1" &&
            mbps(warm.layers == 0U ? aggregate.throughput_live : warm.throughput_live,
                 warm.layers == 0U ? aggregate.compute_seconds : warm.compute_seconds) < 100.0) {
            std::cerr << "BC_PERF_ASSERT failed: warm compute_throughput_mbps < 100\n";
            return 2;
        }
    } catch (const std::exception &ex) {
        std::cerr << "bc_generation_compute_bench failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
