#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCDirectFileIO.h"
#include "BCFamilyGeneration.h"
#include "BCFileIO.h"
#include "BCGenerationBlobIO.h"
#include "BCPositionFile.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <array>
#include <atomic>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionStreamingReader;
using BC::CellId;
using BC::FamilyCoord;
using BC::FamilyId;

std::atomic<uint32_t> g_current_bench_layer_sum{0U};

#if defined(_WIN32)
LONG WINAPI bc_bench_unhandled_exception_filter(EXCEPTION_POINTERS *exception_info) {
    const DWORD code = exception_info != nullptr && exception_info->ExceptionRecord != nullptr
        ? exception_info->ExceptionRecord->ExceptionCode
        : 0U;
    const void *address = exception_info != nullptr && exception_info->ExceptionRecord != nullptr
        ? exception_info->ExceptionRecord->ExceptionAddress
        : nullptr;
    const HMODULE module = GetModuleHandleW(nullptr);
    uintptr_t rip = 0U;
#if defined(_M_X64) || defined(__x86_64__)
    if (exception_info != nullptr && exception_info->ContextRecord != nullptr) {
        rip = static_cast<uintptr_t>(exception_info->ContextRecord->Rip);
    }
#endif
    const uintptr_t base = reinterpret_cast<uintptr_t>(module);
    std::cerr
        << "BC_BENCH_UNHANDLED_EXCEPTION"
        << " code=0x" << std::hex << code
        << " address=0x" << reinterpret_cast<uintptr_t>(address)
        << " rip=0x" << rip
        << " module_base=0x" << base
        << " rip_offset=0x" << (rip >= base ? rip - base : 0U)
        << std::dec
        << " layer_sum=" << g_current_bench_layer_sum.load(std::memory_order_relaxed)
        << " canonical_backend=" << CanonicalBatch::backend_name()
        << " finalize_cid=" << BC::g_bc_cell_finalize_debug_cid.load(std::memory_order_relaxed)
        << " finalize_bucket_count=" << BC::g_bc_cell_finalize_debug_bucket_count.load(std::memory_order_relaxed)
        << " finalize_sorted_size=" << BC::g_bc_cell_finalize_debug_sorted_size.load(std::memory_order_relaxed)
        << " finalize_stage=" << BC::g_bc_cell_finalize_debug_stage.load(std::memory_order_relaxed)
        << '\n';
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] double mbps(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1.0e6 : 0.0;
}

[[nodiscard]] uint64_t process_current_working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)) == 0) {
        return 0U;
    }
    return static_cast<uint64_t>(info.WorkingSetSize);
#else
    std::ifstream statm("/proc/self/statm");
    uint64_t size_pages = 0U;
    uint64_t resident_pages = 0U;
    if (!(statm >> size_pages >> resident_pages)) {
        return 0U;
    }
    long page_size = 4096;
#if defined(_SC_PAGESIZE)
    const long sys_page_size = sysconf(_SC_PAGESIZE);
    if (sys_page_size > 0) {
        page_size = sys_page_size;
    }
#endif
    return resident_pages * static_cast<uint64_t>(page_size);
#endif
}

[[nodiscard]] uint64_t process_peak_working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)) == 0) {
        return 0U;
    }
    return static_cast<uint64_t>(info.PeakWorkingSetSize);
#else
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0U;
    }
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

struct Args {
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    uint32_t target_extra_override = 0U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 0U;
    uint32_t family_work_schedule_chunk = 1U;
    uint32_t family_source_words_per_item = 64U;
    uint32_t family_reserve_buckets = 0U;
    uint32_t family_reserve_bitmap_words = 0U;
    uint32_t warmup_extra = 16U;
    bool verify_layer_rows = true;
    bool output_inspect = true;
    std::string family_blob = "buffered";
    bool family_blob_checksum = false;
    bool family_hot_counters = false;
    bool family_memory_checkpoints = false;
    uint32_t family_modulus = 29U;
    uint32_t direct_queue_depth = 8U;
    std::filesystem::path output_dir =
        std::filesystem::path("tmp") / "bc_family_free";
    std::filesystem::path stats_csv;
    std::filesystem::path ex_stats_csv;
    uint32_t single_layer_sum = 0U;
    std::filesystem::path single_source2_file;
    std::filesystem::path single_source4_file;
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
    uint64_t output_rows = 0U;
    uint64_t source_family_passes = 0U;
    uint64_t source_boards_scanned = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_results_produced = 0U;
    uint64_t canonicalized_candidates = 0U;
    uint64_t encode_attempts = 0U;
    uint64_t encode_valid_candidates = 0U;
    uint64_t encoded_candidates = 0U;
    uint64_t duplicate_candidates = 0U;
    uint64_t source_read_bytes = 0U;
    uint64_t blob_read_bytes = 0U;
    uint64_t blob_write_bytes = 0U;
    uint64_t blob_read_ops = 0U;
    uint64_t blob_write_ops = 0U;
    uint64_t target_logical_bytes = 0U;
    uint64_t target_write_bytes = 0U;
    uint64_t target_write_ops = 0U;
    double source_load_seconds = 0.0;
    double parallel_seconds = 0.0;
    double dump_seconds = 0.0;
    double reload_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double generation_seconds = 0.0;
    double total_seconds = 0.0;
    uint64_t active_family_window_peak = 0U;
    uint64_t target_active_cell_peak = 0U;
    uint64_t source_loaded_cell_peak = 0U;
    uint64_t active_builder_bytes_peak = 0U;
    uint64_t thread_workspace_bytes_peak = 0U;
    uint64_t process_peak_working_set_bytes = 0U;
    uint64_t buffer_flushes = 0U;
    uint64_t builder_bind_calls = 0U;
    uint64_t hash_lookups = 0U;
    uint64_t hash_probe_steps = 0U;
    uint64_t target_cells_created = 0U;
    uint64_t target_cells_reloaded = 0U;
    uint64_t target_cells_dumped = 0U;
    uint64_t target_cells_finalized = 0U;
    uint64_t builder_hash_grows = 0U;
    uint64_t builder_bitmap_grows = 0U;
};

struct ExLayerExpected {
    uint64_t input_live = 0U;
    uint64_t primary_live = 0U;
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

[[nodiscard]] int bench_effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
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
        expected.by_layer_sum[layer_sum] = ExLayerExpected{
            parse_u64_field(cells, input_col, "input_live"),
            parse_u64_field(cells, primary_col, "primary_live")
        };
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

[[nodiscard]] uint64_t load_pattern_seed_board(const std::string &pattern) {
    const std::filesystem::path config_path = find_patterns_config();
    std::ifstream file(config_path);
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

[[nodiscard]] std::array<uint32_t, 16U> free_semantic_tile_sums() {
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

[[nodiscard]] BCLut make_free_lut(uint32_t target_rank) {
    return BCLut(make_free_legal_tiles(target_rank));
}

[[nodiscard]] BC::LayerSum board_semantic_sum(
    uint64_t board,
    const std::array<uint32_t, 16U> &tile_sums
) {
    BC::LayerSum sum = 0U;
    for (uint32_t cell = 0U; cell < 16U; ++cell) {
        const uint8_t tile = static_cast<uint8_t>((board >> (4U * cell)) & 0xFU);
        sum += tile_sums[tile];
    }
    return sum;
}

[[nodiscard]] BCFamilyTable make_axis(
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free family benchmark requires even layer sums");
    }
    if (family_modulus == 0U) {
        throw std::invalid_argument("BC benchmark family modulus must be non-zero");
    }
    return BC::build_family_partition_axis_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        BC::BCFamilyPartitionPolicy::modulo(family_modulus)
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

[[nodiscard]] std::vector<uint64_t> collect_canonical_successors(
    const std::vector<uint64_t> &boards
) {
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
    std::vector<uint8_t> large_positions;
    choose_positions(cells, large_tile_count, 0U, large_positions, [&](const std::vector<uint8_t> &positions_32k) {
        uint64_t base = 0U;
        bool used[16] = {};
        for (uint8_t pos : positions_32k) {
            base |= 15ULL << (4U * pos);
            used[pos] = true;
        }
        std::vector<uint8_t> remaining;
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

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_family_blob_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.family_blob == "direct") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.preserve_unwritten_bytes = false;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

[[nodiscard]] std::unique_ptr<BC::BCReadableFile> open_family_blob_reader(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.family_blob == "direct") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        return std::make_unique<BC::BCDirectFileReader>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileReader>(path);
}

void write_raw_bytes_to_file(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    BC::BCBufferedFileWriter writer(path);
    writer.resize(static_cast<uint64_t>(bytes.size()));
    if (!bytes.empty()) {
        writer.write_at(0U, bytes.data(), static_cast<uint64_t>(bytes.size()));
    }
    writer.flush();
}

[[nodiscard]] std::unique_ptr<BCPositionStreamingReader> open_position_reader(
    const LayerFile &layer,
    const BCLut &lut
) {
    auto reader = std::make_unique<BCPositionStreamingReader>(
        BCPositionStreamingReader::open_buffered(layer.path, lut)
    );
    reader->set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] LayerFile inspect_layer_file(
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
    std::unique_ptr<BCPositionStreamingReader> reader = open_position_reader(layer, lut);
    layer.rows = descriptor_rows(*reader);
    layer.bucket_count = descriptor_bucket_count(*reader);
    layer.rank_payload_bytes = reader->header().rank_payload_bytes;
    return layer;
}

[[nodiscard]] LayerFile inspect_existing_layer_file(
    const std::filesystem::path &path,
    const BCLut &lut
) {
    std::error_code ec;
    const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("failed to stat existing layer file: " + ec.message());
    }
    BCPositionStreamingReader reader = BCPositionStreamingReader::open_buffered(path, lut);
    reader.set_validate_loaded_cells(false);
    LayerFile layer;
    layer.layer_sum = BC::checked_u32_size(
        static_cast<size_t>(reader.axis().layer_sum()),
        "existing layer sum exceeds uint32"
    );
    layer.path = path;
    layer.logical_size = file_size;
    layer.physical_size = file_size;
    layer.rows = descriptor_rows(reader);
    layer.bucket_count = descriptor_bucket_count(reader);
    layer.rank_payload_bytes = reader.header().rank_payload_bytes;
    return layer;
}

[[nodiscard]] LayerFile layer_file_without_inspect(
    uint32_t layer_sum,
    const std::filesystem::path &path,
    uint64_t logical_size,
    uint64_t rows,
    uint64_t bucket_count = 0U,
    uint64_t rank_payload_bytes = 0U
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
    layer.rows = rows;
    layer.bucket_count = bucket_count;
    layer.rank_payload_bytes = rank_payload_bytes;
    return layer;
}

[[nodiscard]] std::vector<uint8_t> build_initial_layer_bytes(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &initial_boards,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    const BCCellMatrix matrix(axis);
    const BC::BCFamilyPartitionLayerMap partition =
        BC::build_family_partition_layer_map(
            axis,
            possible_8tile_sums,
            BC::BCFamilyPartitionPolicy::modulo(family_modulus)
        );
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    for (uint64_t board : initial_boards) {
        const uint64_t canonical = Calculator::canonical_full(board);
        const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(canonical);
        BC::BCEncodedKeyRank key_rank = BC::encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
        check(key_rank.valid, "free initial board should encode key/rank");
        const uint64_t nw_sum = lut.word_desc(q.nw).sum;
        const uint64_t ne_sum = lut.word_desc(q.ne).sum;
        const uint64_t sw_sum = lut.word_desc(q.sw).sum;
        const uint64_t se_sum = lut.word_desc(q.se).sum;
        const uint64_t top_sum = nw_sum + ne_sum;
        const uint64_t left_sum = nw_sum + sw_sum;
        const uint64_t bottom_sum = axis.layer_sum() - top_sum;
        const uint64_t right_sum = axis.layer_sum() - left_sum;
        const FamilyId row_id = partition.coord_to_family_id(
            static_cast<FamilyCoord>(std::min(top_sum, bottom_sum) / 2U)
        );
        const FamilyId col_id = partition.coord_to_family_id(
            static_cast<FamilyCoord>(std::min(left_sum, right_sum) / 2U)
        );
        const CellId cid = matrix.cid(row_id, col_id);
        if (!builders[cid]) {
            builders[cid] = std::make_unique<BCCellBuilder>(lut);
        }
        (void)builders[cid]->insert_encoded_and_report(key_rank);
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
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &initial_boards,
    const std::vector<BC::LayerSum> &possible_8tile_sums
) {
    const std::vector<uint8_t> bytes =
        build_initial_layer_bytes(
            lut,
            axis,
            initial_boards,
            possible_8tile_sums,
            args.family_modulus
        );
    const std::filesystem::path path =
        args.output_dir / ("bc_layer_" + std::to_string(axis.layer_sum()) + ".bcpos");
    write_raw_bytes_to_file(path, bytes);
    if (!args.output_inspect) {
        return layer_file_without_inspect(axis.layer_sum(), path, bytes.size(), initial_boards.size());
    }
    return inspect_layer_file(axis.layer_sum(), path, bytes.size(), lut);
}

[[nodiscard]] std::filesystem::path layer_path(const Args &args, uint32_t layer_sum) {
    return args.output_dir / ("bc_layer_" + std::to_string(layer_sum) + ".bcpos");
}

void cleanup_temp_file(const std::filesystem::path &path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

[[nodiscard]] BC::BCFamilyGenerationOptions family_options_from_args(
    const Args &args,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal
) {
    BC::BCFamilyGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    if (args.pending_buffer != 0U) {
        options.pending_insert_buffer_size = args.pending_buffer;
    }
    options.source_work_schedule_chunk = args.family_work_schedule_chunk;
    options.source_bitmap_words_per_work_item = args.family_source_words_per_item;
    if (args.family_reserve_buckets != 0U) {
        options.new_cell_reserve_buckets = args.family_reserve_buckets;
    }
    if (args.family_reserve_bitmap_words != 0U) {
        options.new_cell_reserve_bitmap_words = args.family_reserve_bitmap_words;
    }
    options.collect_hot_counters = args.family_hot_counters;
    options.family_partition_policy = BC::BCFamilyPartitionPolicy::modulo(args.family_modulus);
    options.family_possible_8tile_sums = &possible_8tile_sums;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    options.success_check_all_cells = true;
    options.keep_only_success_generated_boards = terminal;
    return options;
}

struct FamilyLayerResult {
    BC::BCFamilyGenerationStats stats;
    BC::BCFamilyPositionWriterStats writer_stats;
    BC::BCGenerationBlobIOStats blob_stats;
    uint64_t logical_size = 0U;
    uint64_t output_rows = 0U;
    int effective_threads = 1;
    uint32_t retries = 0U;
    double total_seconds = 0.0;
};

[[nodiscard]] FamilyLayerResult generate_family_layer_to_file(
    const Args &args,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCPositionStreamingReader *source4_reader,
    const BCPositionStreamingReader &source2_reader,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal,
    const std::filesystem::path &output_path
) {
    const std::filesystem::path rank_spool_path = output_path.string() + ".rank_spool.tmp";
    const std::filesystem::path blob_path = output_path.string() + ".family_blob.tmp";
    constexpr uint64_t kFamilyDirectBlobStagingBytes = 256ULL * 1024ULL;
    constexpr uint64_t kFamilyBufferedBlobStagingBytes = 1ULL * 1024ULL * 1024ULL;
    constexpr uint64_t kFamilyPositionWriterStagingBytes = 1ULL * 1024ULL * 1024ULL;

    uint32_t reserve_buckets = args.family_reserve_buckets == 0U ? 1024U : args.family_reserve_buckets;
    uint32_t reserve_bitmap_words =
        args.family_reserve_bitmap_words == 0U ? 4096U : args.family_reserve_bitmap_words;
    constexpr uint32_t kMaxFamilyBuilderRetries = 6U;
    const double retry_begin = now_seconds();
    for (uint32_t attempt = 0U; attempt <= kMaxFamilyBuilderRetries; ++attempt) {
        cleanup_temp_file(rank_spool_path);
        cleanup_temp_file(blob_path);
        cleanup_temp_file(output_path);
        try {
            FamilyLayerResult result;
            {
                BC::BCBufferedFileWriter final_writer(output_path);
                BC::BCBufferedFileWriter rank_spool_writer(rank_spool_path);
                std::unique_ptr<BC::BCWritableFile> blob_writer = open_family_blob_writer(args, blob_path);
                std::unique_ptr<BC::BCReadableFile> blob_reader = open_family_blob_reader(args, blob_path);
                const uint64_t blob_staging_bytes =
                    args.family_blob == "direct"
                        ? kFamilyDirectBlobStagingBytes
                        : kFamilyBufferedBlobStagingBytes;
                BC::BCFileGenerationBlobIO blob(
                    *blob_writer,
                    blob_reader.get(),
                    blob_staging_bytes,
                    args.family_blob_checksum
                );
                std::optional<BC::BCFamilyStreamingGenerationSource> source4;
                if (source4_reader != nullptr) {
                    source4 = BC::BCFamilyStreamingGenerationSource{source4_reader, 2U, 2U};
                }
                const BC::BCFamilyStreamingGenerationSource source2{&source2_reader, 1U, 1U};
                BC::BCFamilyGenerationOptions options = family_options_from_args(
                    args,
                    possible_8tile_sums,
                    success_shifts,
                    success_check_min_source_layer_sum,
                    terminal
                );
                options.new_cell_reserve_buckets = reserve_buckets;
                options.new_cell_reserve_bitmap_words = reserve_bitmap_words;

                BC::BCFamilyMutableStore target_store(lut, target_axis, blob);
                BC::BCFamilyPositionWriter position_writer;
                BC::BCFamilyPositionWriterOptions writer_options;
                writer_options.staging_bytes = kFamilyPositionWriterStagingBytes;
                position_writer.begin_layer(final_writer, rank_spool_writer, target_axis, writer_options);

                const double total_begin = now_seconds();
                BC::BCFamilyGenerationStats stats = BC::generate_family_position_layer_v1(
                    lut,
                    target_axis,
                    source4 ? &(*source4) : nullptr,
                    source2,
                    target_store,
                    position_writer,
                    options
                );
                position_writer.flush_pending_streams_for_reader();
                BC::BCBufferedFileReader rank_spool_reader(rank_spool_path);
                const double finish_begin = now_seconds();
                const uint64_t logical_size = position_writer.finish_layer(rank_spool_reader);
                stats.write_seconds += now_seconds() - finish_begin;

                result.stats = stats;
                result.writer_stats = position_writer.stats();
                result.blob_stats = blob.stats();
                result.logical_size = logical_size;
                result.output_rows = result.writer_stats.success_rows;
                result.effective_threads = bench_effective_threads(args.num_threads);
                result.retries = attempt;
                result.total_seconds = now_seconds() - total_begin;
                result.stats.generation_seconds = result.total_seconds;
            }
            cleanup_temp_file(rank_spool_path);
            cleanup_temp_file(blob_path);
            return result;
        } catch (const BC::BCCellMutableBuilderOverflow &) {
            cleanup_temp_file(rank_spool_path);
            cleanup_temp_file(blob_path);
            cleanup_temp_file(output_path);
            if (attempt == kMaxFamilyBuilderRetries) {
                throw;
            }
            reserve_buckets = reserve_buckets > (std::numeric_limits<uint32_t>::max() / 2U)
                ? std::numeric_limits<uint32_t>::max()
                : reserve_buckets * 2U;
            reserve_bitmap_words = reserve_bitmap_words > (std::numeric_limits<uint32_t>::max() / 2U)
                ? std::numeric_limits<uint32_t>::max()
                : reserve_bitmap_words * 2U;
        }
    }
    throw std::logic_error("FamilyChain generation retry loop exited unexpectedly");
}

void print_header(std::ostream &out) {
    out
        << "row_type,layer_sum,layers,effective_threads,input_live,output_rows,"
        << "ex_input_live,ex_primary_live,ex_match,"
        << "generation_seconds,total_seconds,generation_throughput_mbps,total_throughput_mbps,"
        << "source_load_seconds,parallel_seconds,dump_seconds,reload_seconds,finalize_seconds,write_seconds,"
        << "source_read_bytes,blob_read_bytes,blob_write_bytes,blob_read_ops,blob_write_ops,"
        << "target_logical_bytes,target_write_bytes,target_write_ops,"
        << "active_family_window_peak,target_active_cell_peak,source_loaded_cell_peak,"
        << "active_builder_bytes_peak,thread_workspace_bytes_peak,process_peak_working_set_bytes,"
        << "source_boards_scanned,spawned_boards,move_results_produced,canonicalized_candidates,"
        << "encode_attempts,encode_valid_candidates,encoded_candidates,duplicate_candidates,"
        << "buffer_flushes,builder_bind_calls,"
        << "hash_lookups,hash_probe_steps,target_cells_created,target_cells_reloaded,"
        << "target_cells_dumped,target_cells_finalized,builder_hash_grows,builder_bitmap_grows,"
        << "output_path\n";
}

void print_layer_row(
    std::ostream &out,
    uint32_t layer_sum,
    uint64_t input_live,
    uint64_t ex_input_live,
    uint64_t ex_primary_live,
    uint32_t ex_match,
    uint64_t source_family_passes,
    const FamilyLayerResult &result,
    const std::filesystem::path &output_path
) {
    const BC::BCFamilyGenerationStats &s = result.stats;
    const BC::BCFamilyPositionWriterStats &w = result.writer_stats;
    const BC::BCGenerationBlobIOStats &b = result.blob_stats;
    out
        << "layer," << layer_sum << ",1," << result.effective_threads << ','
        << input_live << ',' << result.output_rows << ','
        << ex_input_live << ',' << ex_primary_live << ',' << ex_match << ','
        << std::setprecision(9)
        << s.generation_seconds << ',' << result.total_seconds << ','
        << mbps(result.output_rows, s.generation_seconds) << ','
        << mbps(result.output_rows, result.total_seconds) << ','
        << s.source_load_seconds << ',' << s.parallel_seconds << ','
        << s.dump_seconds << ',' << s.reload_seconds << ','
        << s.finalize_seconds << ',' << s.write_seconds << ','
        << s.source_bytes_read << ',' << b.bytes_read << ',' << b.bytes_written << ','
        << b.backend_read_ops << ',' << b.backend_write_ops << ','
        << result.logical_size << ','
        << (w.bucket_stage_write_bytes + w.rank_stage_write_bytes + w.metadata_write_bytes + w.rank_copy_write_bytes) << ','
        << (w.bucket_stage_flushes + w.rank_stage_flushes + w.metadata_write_ops + w.rank_copy_chunks) << ','
        << s.active_family_window_peak << ',' << s.target_active_cell_peak << ','
        << s.source_loaded_cell_peak << ',' << s.active_builder_bytes_peak << ','
        << s.thread_workspace_bytes_peak << ',' << process_peak_working_set_bytes() << ','
        << s.source_boards_scanned << ',' << s.spawned_boards << ','
        << s.move_results_produced << ',' << s.canonicalized_candidates << ','
        << s.encode_attempts << ',' << s.encode_valid_candidates << ','
        << s.encoded_candidates << ',' << s.duplicate_candidates << ','
        << s.family_buffer_flushes << ',' << s.family_builder_bind_calls << ','
        << s.family_insert_hash_lookups << ',' << s.family_insert_hash_probe_steps << ','
        << s.target_cells_created << ',' << s.target_cells_reloaded << ','
        << s.target_cells_dumped << ',' << s.target_cells_finalized << ','
        << s.family_builder_hash_grows << ',' << s.family_builder_bitmap_grows << ','
        << output_path.string()
        << '\n';
    (void)source_family_passes;
}

void accumulate(
    AggregateStats &agg,
    const FamilyLayerResult &result,
    uint64_t input_live,
    uint64_t source_family_passes
) {
    const BC::BCFamilyGenerationStats &s = result.stats;
    const BC::BCFamilyPositionWriterStats &w = result.writer_stats;
    const BC::BCGenerationBlobIOStats &b = result.blob_stats;
    ++agg.layers;
    agg.effective_threads = result.effective_threads;
    agg.input_live += input_live;
    agg.output_rows += result.output_rows;
    agg.source_family_passes += source_family_passes;
    agg.source_boards_scanned += s.source_boards_scanned;
    agg.spawned_boards += s.spawned_boards;
    agg.move_results_produced += s.move_results_produced;
    agg.canonicalized_candidates += s.canonicalized_candidates;
    agg.encode_attempts += s.encode_attempts;
    agg.encode_valid_candidates += s.encode_valid_candidates;
    agg.encoded_candidates += s.encoded_candidates;
    agg.duplicate_candidates += s.duplicate_candidates;
    agg.source_read_bytes += s.source_bytes_read;
    agg.blob_read_bytes += b.bytes_read;
    agg.blob_write_bytes += b.bytes_written;
    agg.blob_read_ops += b.backend_read_ops;
    agg.blob_write_ops += b.backend_write_ops;
    agg.target_logical_bytes += result.logical_size;
    agg.target_write_bytes +=
        w.bucket_stage_write_bytes + w.rank_stage_write_bytes +
        w.metadata_write_bytes + w.rank_copy_write_bytes;
    agg.target_write_ops +=
        w.bucket_stage_flushes + w.rank_stage_flushes + w.metadata_write_ops + w.rank_copy_chunks;
    agg.source_load_seconds += s.source_load_seconds;
    agg.parallel_seconds += s.parallel_seconds;
    agg.dump_seconds += s.dump_seconds;
    agg.reload_seconds += s.reload_seconds;
    agg.finalize_seconds += s.finalize_seconds;
    agg.write_seconds += s.write_seconds;
    agg.generation_seconds += s.generation_seconds;
    agg.total_seconds += result.total_seconds;
    agg.active_family_window_peak = std::max(agg.active_family_window_peak, s.active_family_window_peak);
    agg.target_active_cell_peak = std::max(agg.target_active_cell_peak, s.target_active_cell_peak);
    agg.source_loaded_cell_peak = std::max(agg.source_loaded_cell_peak, s.source_loaded_cell_peak);
    agg.active_builder_bytes_peak = std::max(agg.active_builder_bytes_peak, s.active_builder_bytes_peak);
    agg.thread_workspace_bytes_peak = std::max(agg.thread_workspace_bytes_peak, s.thread_workspace_bytes_peak);
    agg.process_peak_working_set_bytes =
        std::max(agg.process_peak_working_set_bytes, process_peak_working_set_bytes());
    agg.buffer_flushes += s.family_buffer_flushes;
    agg.builder_bind_calls += s.family_builder_bind_calls;
    agg.hash_lookups += s.family_insert_hash_lookups;
    agg.hash_probe_steps += s.family_insert_hash_probe_steps;
    agg.target_cells_created += s.target_cells_created;
    agg.target_cells_reloaded += s.target_cells_reloaded;
    agg.target_cells_dumped += s.target_cells_dumped;
    agg.target_cells_finalized += s.target_cells_finalized;
    agg.builder_hash_grows += s.family_builder_hash_grows;
    agg.builder_bitmap_grows += s.family_builder_bitmap_grows;
}

void print_summary_row(std::ostream &out, const char *label, const AggregateStats &agg) {
    out
        << label << ",0," << agg.layers << ',' << agg.effective_threads << ','
        << agg.input_live << ',' << agg.output_rows << ",0,0,0,"
        << std::setprecision(9)
        << agg.generation_seconds << ',' << agg.total_seconds << ','
        << mbps(agg.output_rows, agg.generation_seconds) << ','
        << mbps(agg.output_rows, agg.total_seconds) << ','
        << agg.source_load_seconds << ',' << agg.parallel_seconds << ','
        << agg.dump_seconds << ',' << agg.reload_seconds << ','
        << agg.finalize_seconds << ',' << agg.write_seconds << ','
        << agg.source_read_bytes << ',' << agg.blob_read_bytes << ',' << agg.blob_write_bytes << ','
        << agg.blob_read_ops << ',' << agg.blob_write_ops << ','
        << agg.target_logical_bytes << ',' << agg.target_write_bytes << ',' << agg.target_write_ops << ','
        << agg.active_family_window_peak << ',' << agg.target_active_cell_peak << ','
        << agg.source_loaded_cell_peak << ',' << agg.active_builder_bytes_peak << ','
        << agg.thread_workspace_bytes_peak << ',' << agg.process_peak_working_set_bytes << ','
        << agg.source_boards_scanned << ',' << agg.spawned_boards << ','
        << agg.move_results_produced << ',' << agg.canonicalized_candidates << ','
        << agg.encode_attempts << ',' << agg.encode_valid_candidates << ','
        << agg.encoded_candidates << ',' << agg.duplicate_candidates << ','
        << agg.buffer_flushes << ',' << agg.builder_bind_calls << ','
        << agg.hash_lookups << ',' << agg.hash_probe_steps << ','
        << agg.target_cells_created << ',' << agg.target_cells_reloaded << ','
        << agg.target_cells_dumped << ',' << agg.target_cells_finalized << ','
        << agg.builder_hash_grows << ',' << agg.builder_bitmap_grows << ",\n";
}

int run_single_layer(const Args &args, std::ostream &out) {
    std::filesystem::create_directories(args.output_dir);
    const std::array<uint32_t, 16U> tile_sums = free_semantic_tile_sums();
    const std::vector<uint8_t> legal_tiles = make_free_legal_tiles(args.target_rank);
    const std::vector<BC::LayerSum> possible_8tile_sums =
        BC::build_possible_8tile_sums(legal_tiles, tile_sums);
    const BCLut lut = make_free_lut(args.target_rank);

    const uint64_t seed_board = load_pattern_seed_board(args.pattern);
    const BC::LayerSum seed_sum64 = board_semantic_sum(seed_board, tile_sums);
    if (seed_sum64 > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("free benchmark seed sum exceeds uint32");
    }
    const uint32_t seed_sum = static_cast<uint32_t>(seed_sum64);
    const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
    const uint32_t success_check_min_source_layer_sum = seed_sum + 2U * (docheck_step + 1U);
    const std::vector<uint8_t> success_shifts = all_board_success_shifts();
    const ExExpectedStats ex_expected = load_ex_expected_stats(args.ex_stats_csv, seed_sum);

    const LayerFile source2_layer = inspect_existing_layer_file(args.single_source2_file, lut);
    const LayerFile source4_layer = inspect_existing_layer_file(args.single_source4_file, lut);
    if (source2_layer.layer_sum + 2U != args.single_layer_sum) {
        throw std::invalid_argument("--source2-file layer_sum must equal --single-layer-sum - 2");
    }
    if (source4_layer.layer_sum + 4U != args.single_layer_sum) {
        throw std::invalid_argument("--source4-file layer_sum must equal --single-layer-sum - 4");
    }

    std::unique_ptr<BCPositionStreamingReader> source2_reader =
        open_position_reader(source2_layer, lut);
    std::unique_ptr<BCPositionStreamingReader> source4_reader =
        open_position_reader(source4_layer, lut);
    const uint64_t source_family_passes =
        static_cast<uint64_t>(source2_reader->axis().family_count()) +
        static_cast<uint64_t>(source4_reader->axis().family_count());
    const BCFamilyTable target_axis =
        make_axis(args.single_layer_sum, possible_8tile_sums, args.family_modulus);
    const std::filesystem::path final_path = layer_path(args, args.single_layer_sum);

    print_header(out);
    FamilyLayerResult result = generate_family_layer_to_file(
        args,
        lut,
        target_axis,
        source4_reader.get(),
        *source2_reader,
        possible_8tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        false,
        final_path
    );
    source4_reader.reset();
    source2_reader.reset();

    LayerFile generated_layer =
        args.output_inspect
            ? inspect_layer_file(args.single_layer_sum, final_path, result.logical_size, lut)
            : layer_file_without_inspect(
                  args.single_layer_sum,
                  final_path,
                  result.logical_size,
                  result.output_rows,
                  result.writer_stats.bucket_stage_write_bytes / BC::kBCPositionBucketEntryBytes,
                  result.writer_stats.rank_stage_write_bytes);
    result.output_rows = generated_layer.rows;

    uint64_t ex_input_live = 0U;
    uint64_t ex_primary_live = 0U;
    uint32_t ex_match = 0U;
    if (ex_expected.enabled) {
        const auto ex_it = ex_expected.by_layer_sum.find(args.single_layer_sum);
        if (ex_it == ex_expected.by_layer_sum.end()) {
            throw std::runtime_error("EX stats CSV is missing single target layer " + std::to_string(args.single_layer_sum));
        }
        ex_input_live = ex_it->second.input_live;
        ex_primary_live = ex_it->second.primary_live;
        if (args.verify_layer_rows && source2_layer.rows != ex_input_live) {
            throw std::runtime_error("BC Family single-layer input_live does not match EX stats for layer " + std::to_string(args.single_layer_sum));
        }
        if (args.verify_layer_rows && result.output_rows != ex_primary_live) {
            throw std::runtime_error("BC Family single-layer primary_live does not match EX stats for layer " + std::to_string(args.single_layer_sum));
        }
        ex_match = source2_layer.rows == ex_input_live && result.output_rows == ex_primary_live ? 1U : 0U;
    }

    print_layer_row(
        out,
        args.single_layer_sum,
        source2_layer.rows,
        ex_input_live,
        ex_primary_live,
        ex_match,
        source_family_passes,
        result,
        final_path
    );

    AggregateStats aggregate;
    accumulate(aggregate, result, source2_layer.rows, source_family_passes);
    print_summary_row(out, "total", aggregate);
    return 0;
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
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--target-extra") {
            args.target_extra_override = static_cast<uint32_t>(std::stoul(require_value("--target-extra")));
        } else if (key == "--single-layer-sum") {
            args.single_layer_sum = static_cast<uint32_t>(std::stoul(require_value("--single-layer-sum")));
        } else if (key == "--source2-file") {
            args.single_source2_file = require_value("--source2-file");
        } else if (key == "--source4-file") {
            args.single_source4_file = require_value("--source4-file");
        } else if (key == "--extra-steps") {
            args.extra_steps = static_cast<uint32_t>(std::stoul(require_value("--extra-steps")));
        } else if (key == "--output-dir") {
            args.output_dir = require_value("--output-dir");
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value("--stats-csv");
        } else if (key == "--ex-stats-csv" || key == "--verify-csv") {
            args.ex_stats_csv = require_value(key.c_str());
        } else if (key == "--num-threads") {
            args.num_threads = std::stoi(require_value("--num-threads"));
        } else if (key == "--batch-size") {
            args.batch_size = static_cast<uint32_t>(std::stoul(require_value("--batch-size")));
        } else if (key == "--pending-buffer") {
            args.pending_buffer = static_cast<uint32_t>(std::stoul(require_value("--pending-buffer")));
        } else if (key == "--family-work-schedule-chunk") {
            args.family_work_schedule_chunk =
                static_cast<uint32_t>(std::stoul(require_value("--family-work-schedule-chunk")));
        } else if (key == "--family-source-words-per-item") {
            args.family_source_words_per_item =
                static_cast<uint32_t>(std::stoul(require_value("--family-source-words-per-item")));
        } else if (key == "--family-reserve-buckets") {
            args.family_reserve_buckets =
                static_cast<uint32_t>(std::stoul(require_value("--family-reserve-buckets")));
        } else if (key == "--family-reserve-bitmap-words") {
            args.family_reserve_bitmap_words =
                static_cast<uint32_t>(std::stoul(require_value("--family-reserve-bitmap-words")));
        } else if (key == "--warmup-extra") {
            args.warmup_extra = static_cast<uint32_t>(std::stoul(require_value("--warmup-extra")));
        } else if (key == "--family-blob") {
            args.family_blob = require_value("--family-blob");
        } else if (key == "--family-blob-checksum") {
            args.family_blob_checksum = true;
        } else if (key == "--family-hot-counters") {
            args.family_hot_counters = true;
        } else if (key == "--family-memory-checkpoints") {
            args.family_memory_checkpoints = true;
        } else if (key == "--family-modulus") {
            args.family_modulus = static_cast<uint32_t>(std::stoul(require_value("--family-modulus")));
        } else if (key == "--target-direct-queue-depth") {
            args.direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value("--target-direct-queue-depth")));
        } else if (key == "--no-verify-layer-rows") {
            args.verify_layer_rows = false;
        } else if (key == "--no-output-inspect") {
            args.output_inspect = false;
        } else if (key == "--family-output") {
            const std::string value = require_value(key.c_str());
            if (value != "buffered") {
                throw std::invalid_argument("--family-output only supports buffered in this slim benchmark");
            }
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.pattern.rfind("free", 0U) != 0U) {
        throw std::invalid_argument("bc_family_generation_bench currently supports --pattern freeN only");
    }
    (void)free_pattern_index(args.pattern);
    if (args.target_rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large");
    }
    if (args.target_extra_override != 0U && (args.target_extra_override & 1U) != 0U) {
        throw std::invalid_argument("--target-extra must be an even semantic sum when provided");
    }
    if (args.batch_size == 0U || args.family_work_schedule_chunk == 0U ||
        args.family_source_words_per_item == 0U || args.family_modulus == 0U ||
        args.direct_queue_depth == 0U) {
        throw std::invalid_argument("family benchmark numeric options must be non-zero");
    }
    if (args.family_blob != "buffered" && args.family_blob != "direct") {
        throw std::invalid_argument("--family-blob must be buffered or direct");
    }
    const bool single_mode =
        args.single_layer_sum != 0U ||
        !args.single_source2_file.empty() ||
        !args.single_source4_file.empty();
    if (single_mode) {
        if (args.single_layer_sum == 0U) {
            throw std::invalid_argument("--single-layer-sum is required in single-layer mode");
        }
        if (args.single_source2_file.empty()) {
            throw std::invalid_argument("--source2-file is required in single-layer mode");
        }
        if (args.single_source4_file.empty()) {
            throw std::invalid_argument("--source4-file is required in single-layer mode");
        }
        if ((args.single_layer_sum & 1U) != 0U) {
            throw std::invalid_argument("--single-layer-sum must be even");
        }
    }
    return args;
}

int run_free_chain(const Args &args, std::ostream &out) {
    std::filesystem::create_directories(args.output_dir);
    const std::array<uint32_t, 16U> tile_sums = free_semantic_tile_sums();
    const std::vector<uint8_t> legal_tiles = make_free_legal_tiles(args.target_rank);
    const std::vector<BC::LayerSum> possible_8tile_sums =
        BC::build_possible_8tile_sums(legal_tiles, tile_sums);
    const BCLut lut = make_free_lut(args.target_rank);
    const uint64_t seed_board = load_pattern_seed_board(args.pattern);
    const BC::LayerSum seed_sum64 = board_semantic_sum(seed_board, tile_sums);
    if (seed_sum64 > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("free benchmark seed sum exceeds uint32");
    }
    const uint32_t seed_sum = static_cast<uint32_t>(seed_sum64);
    const uint32_t forward_steps = ex_forward_steps(args);
    const uint32_t final_sum = seed_sum + forward_steps * 2U;
    const bool ex_terminal_mode = args.target_extra_override == 0U && final_sum >= seed_sum + 4U;
    const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
    const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
    const uint32_t success_check_min_source_layer_sum = seed_sum + 2U * (docheck_step + 1U);
    const std::vector<uint8_t> success_shifts = all_board_success_shifts();
    const ExExpectedStats ex_expected = load_ex_expected_stats(args.ex_stats_csv, seed_sum);

    std::vector<uint64_t> initial_boards =
        generate_free_initial_boards(free_pattern_index(args.pattern));
    if (args.pattern == "free9") {
        check(initial_boards.size() == 21283U, "free9 initial board count must match EX");
    }

    std::map<uint32_t, LayerFile> layers;
    LayerFile seed_layer =
        write_initial_layer_file(
            args,
            lut,
            make_axis(seed_sum, possible_8tile_sums, args.family_modulus),
            initial_boards,
            possible_8tile_sums
        );
    if (seed_layer.rows != initial_boards.size()) {
        throw std::runtime_error("free initial layer row count mismatch");
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

    const uint64_t process_baseline_working_set = process_current_working_set_bytes();
    (void)process_baseline_working_set;
    AggregateStats aggregate;
    AggregateStats warm;
    print_header(out);

    for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
        g_current_bench_layer_sum.store(layer_sum, std::memory_order_relaxed);
        const BCFamilyTable target_axis =
            make_axis(layer_sum, possible_8tile_sums, args.family_modulus);
        const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
        const auto source2_it = layers.find(layer_sum - 2U);
        if (source2_it == layers.end()) {
            throw std::runtime_error("FamilyChain lost required +2 source layer");
        }
        std::unique_ptr<BCPositionStreamingReader> source2_reader =
            open_position_reader(source2_it->second, lut);
        std::unique_ptr<BCPositionStreamingReader> source4_reader;
        if (layer_sum >= seed_sum + 4U) {
            const auto source4_it = layers.find(layer_sum - 4U);
            if (source4_it == layers.end()) {
                throw std::runtime_error("FamilyChain lost required +4 source layer");
            }
            source4_reader = open_position_reader(source4_it->second, lut);
        }

        const uint64_t input_live = source2_it->second.rows;
        const uint64_t source_family_passes =
            source2_reader->axis().family_count() +
            (source4_reader != nullptr ? source4_reader->axis().family_count() : 0U);
        const std::filesystem::path final_path = layer_path(args, layer_sum);
        FamilyLayerResult result = generate_family_layer_to_file(
            args,
            lut,
            target_axis,
            source4_reader.get(),
            *source2_reader,
            possible_8tile_sums,
            success_shifts,
            success_check_min_source_layer_sum,
            terminal,
            final_path
        );
        source4_reader.reset();
        source2_reader.reset();

        LayerFile generated_layer =
            args.output_inspect
                ? inspect_layer_file(layer_sum, final_path, result.logical_size, lut)
                : layer_file_without_inspect(
                      layer_sum,
                      final_path,
                      result.logical_size,
                      result.output_rows,
                      result.writer_stats.bucket_stage_write_bytes / BC::kBCPositionBucketEntryBytes,
                      result.writer_stats.rank_stage_write_bytes);
        result.output_rows = generated_layer.rows;

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
            if (args.verify_layer_rows && input_live != ex_input_live) {
                throw std::runtime_error("BC Family input_live does not match EX stats for layer " + std::to_string(layer_sum));
            }
            if (args.verify_layer_rows && result.output_rows != ex_primary_live) {
                throw std::runtime_error("BC Family primary_live does not match EX stats for layer " + std::to_string(layer_sum));
            }
            ex_match = input_live == ex_input_live && result.output_rows == ex_primary_live ? 1U : 0U;
        }

        print_layer_row(
            out,
            layer_sum,
            input_live,
            ex_input_live,
            ex_primary_live,
            ex_match,
            source_family_passes,
            result,
            final_path
        );
        accumulate(aggregate, result, input_live, source_family_passes);
        if (layer_sum >= seed_sum + args.warmup_extra) {
            accumulate(warm, result, input_live, source_family_passes);
        }
        layers[layer_sum] = std::move(generated_layer);
    }

    print_summary_row(out, "total", aggregate);
    if (warm.layers != 0U) {
        print_summary_row(out, "warm", warm);
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
#if defined(_WIN32)
    SetUnhandledExceptionFilter(bc_bench_unhandled_exception_filter);
#endif
    try {
        const Args args = parse_args(argc, argv);
        if (!args.stats_csv.empty()) {
            std::ofstream out(args.stats_csv);
            if (!out) {
                throw std::runtime_error("failed to open --stats-csv");
            }
            return args.single_layer_sum != 0U
                ? run_single_layer(args, out)
                : run_free_chain(args, out);
        }
        return args.single_layer_sum != 0U
            ? run_single_layer(args, std::cout)
            : run_free_chain(args, std::cout);
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_generation_bench failed: " << ex.what() << '\n';
        return 1;
    }
}
