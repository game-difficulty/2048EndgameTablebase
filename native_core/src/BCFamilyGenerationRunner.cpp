#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCDirectFileIO.h"
#include "BCFamilyGeneration.h"
#include "BCFamilyGenerationRunner.h"
#include "BCFamilyRoutePlanner.h"
#include "BCFileIO.h"
#include "BCGenerationBlobIO.h"
#include "BCPositionFamilyRemapReader.h"
#include "BCPositionFile.h"
#include "BCResidentGeneration.h"
#include "BCSingleChunkGeneration.h"
#include "BCSortUtils.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"
#include "CompressionBridge.h"
#include "FormationRuntime.h"
#include "NativeLzma.h"
#include "PathUtils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
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
using BC::BCPositionCellLayout;
using BC::BCPositionStreamingReader;
using BC::CellId;
using BC::FinalizedCellPayload;

constexpr uint32_t kSingleRouteMaxCellChunkSize = 512U;
using BC::FamilyCoord;
using BC::FamilyId;

std::atomic<uint32_t> g_current_generation_layer_sum{0U};

#if defined(_WIN32)
LONG WINAPI bc_generation_unhandled_exception_filter(EXCEPTION_POINTERS *exception_info) {
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
        << "BC_GENERATION_UNHANDLED_EXCEPTION"
        << " code=0x" << std::hex << code
        << " address=0x" << reinterpret_cast<uintptr_t>(address)
        << " rip=0x" << rip
        << " module_base=0x" << base
        << " rip_offset=0x" << (rip >= base ? rip - base : 0U)
        << std::dec
        << " layer_sum=" << g_current_generation_layer_sum.load(std::memory_order_relaxed)
        << " canonical_backend=" << CanonicalBatch::backend_name()
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

[[nodiscard]] double gbps(uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1.0e9 : 0.0;
}

constexpr double kBCDefaultReserveFactor = 2.0;
constexpr double kBCMaxReserveFactor = 4.0;
constexpr double kBCEarlyLayerReserveFactor = 4.0;
constexpr uint32_t kBCEarlyLayerReserveFactorSteps = 10U;
constexpr double kBCLearnedReserveMinFactor = 1.08;
constexpr double kBCLearnedReserveQuantileGuard = 1.10;
constexpr double kBCLearnedReserveLastGuard = 1.12;
constexpr double kBCLearnedReserveRetryGuard = 1.25;
constexpr size_t kBCResidentReserveHistoryWindow = 32U;
constexpr size_t kBCSingleReserveHistoryWindow = 5U;
constexpr uint32_t kResidentRoutePendingBuffer = 2048U;
constexpr uint32_t kSingleRoutePendingBuffer = 1024U;

double bc_reserve_need_recent_quantile(const std::vector<double> &history, size_t history_window) {
    const size_t begin = history.size() > history_window
        ? history.size() - history_window
        : 0U;
    std::vector<double> values(history.begin() + static_cast<std::ptrdiff_t>(begin), history.end());
    std::sort(values.begin(), values.end());
    const size_t index = ((values.size() - 1U) * 9U) / 10U;
    return values[index];
}

double bc_regular_reserve_factor(
    const std::vector<double> &history,
    double retry_guard_factor,
    size_t history_window
) {
    if (history.empty()) {
        return kBCDefaultReserveFactor;
    }
    double factor = kBCLearnedReserveMinFactor;
    factor = std::max(factor, history.back() * kBCLearnedReserveLastGuard);
    factor = std::max(
        factor,
        bc_reserve_need_recent_quantile(history, history_window) * kBCLearnedReserveQuantileGuard
    );
    factor = std::max(factor, retry_guard_factor);
    return std::min(kBCMaxReserveFactor, std::max(kBCLearnedReserveMinFactor, factor));
}

double bc_reserve_factor_for_step(
    uint32_t current_step,
    const std::vector<double> &history,
    double retry_guard_factor,
    size_t history_window
) {
    if (current_step < kBCEarlyLayerReserveFactorSteps) {
        return kBCEarlyLayerReserveFactor;
    }
    return bc_regular_reserve_factor(history, retry_guard_factor, history_window);
}

[[nodiscard]] uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("alignment must be a non-zero power of two");
    }
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
        throw std::overflow_error("align_up_u64 overflow");
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

[[nodiscard]] uint64_t position_header_logical_size(const BC::BCPositionHeader &header) {
    const uint64_t bucket_end = BC::bc_checked_add_u64(
        header.bucket_meta_offset,
        header.bucket_meta_bytes,
        "position header bucket end overflow"
    );
    const uint64_t rank_end = BC::bc_checked_add_u64(
        header.rank_payload_offset,
        header.rank_payload_bytes,
        "position header rank end overflow"
    );
    return std::max(bucket_end, rank_end);
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

struct SystemMemorySnapshot {
    uint64_t available_bytes = 0U;
    uint64_t total_bytes = 0U;
};

[[nodiscard]] SystemMemorySnapshot system_memory_snapshot() {
    SystemMemorySnapshot snapshot;
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        snapshot.available_bytes = static_cast<uint64_t>(status.ullAvailPhys);
        snapshot.total_bytes = static_cast<uint64_t>(status.ullTotalPhys);
    }
#else
    long page_size = 4096;
#if defined(_SC_PAGESIZE)
    const long sys_page_size = sysconf(_SC_PAGESIZE);
    if (sys_page_size > 0) {
        page_size = sys_page_size;
    }
#endif
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
    if (pages > 0) {
        snapshot.total_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    if (avail_pages > 0) {
        snapshot.available_bytes = static_cast<uint64_t>(avail_pages) * static_cast<uint64_t>(page_size);
    }
#endif
    return snapshot;
}

struct ProcessMemorySnapshot {
    uint64_t working_set_bytes = 0U;
    uint64_t peak_working_set_bytes = 0U;
    uint64_t private_bytes = 0U;
    uint64_t pagefile_bytes = 0U;
    uint64_t peak_pagefile_bytes = 0U;
};

[[nodiscard]] ProcessMemorySnapshot process_memory_snapshot() {
    ProcessMemorySnapshot snapshot;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX info{};
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&info),
            sizeof(info)) == 0) {
        return snapshot;
    }
    snapshot.working_set_bytes = static_cast<uint64_t>(info.WorkingSetSize);
    snapshot.peak_working_set_bytes = static_cast<uint64_t>(info.PeakWorkingSetSize);
    snapshot.private_bytes = static_cast<uint64_t>(info.PrivateUsage);
    snapshot.pagefile_bytes = static_cast<uint64_t>(info.PagefileUsage);
    snapshot.peak_pagefile_bytes = static_cast<uint64_t>(info.PeakPagefileUsage);
#else
    snapshot.working_set_bytes = process_current_working_set_bytes();
    snapshot.peak_working_set_bytes = process_peak_working_set_bytes();
#endif
    return snapshot;
}

struct Args {
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    uint32_t target_extra_override = 0U;
    std::vector<uint64_t> seed_boards;
    std::vector<uint64_t> pattern_masks;
    std::vector<uint8_t> success_shifts;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    uint32_t success_check_min_source_layer_sum_override = 0U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 0U;
    uint32_t family_work_schedule_chunk = 1U;
    uint32_t family_source_words_per_item = 64U;
    uint32_t family_reserve_buckets = 0U;
    uint32_t family_reserve_bitmap_words = 0U;
    uint32_t warmup_extra = 16U;
    bool verify_layer_rows = true;
    bool output_inspect = false;
    std::string family_blob = "direct";
    std::string family_position_io = "direct-rank-first";
    std::string family_source_io = "direct-auto";
    bool family_blob_checksum = false;
    bool family_memory_checkpoints = false;
    bool compress_temp_files = false;
    uint32_t family_modulus = 29U;
    BC::BCFamilyGenerationRoute family_route = BC::BCFamilyGenerationRoute::Auto;
    std::filesystem::path family_route_script;
    uint32_t direct_queue_depth = 8U;
    std::filesystem::path output_dir =
        std::filesystem::path("tmp") / "bc_family_free";
    std::vector<std::filesystem::path> output_dirs;
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
    const LayerFile &current,
    const LayerFile &next
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

struct FamilyMemoryCheckpointContext {
    uint64_t baseline_working_set_bytes = 0U;
    std::vector<BC::BCFamilyMemoryCheckpoint> records;
};

void family_memory_checkpoint_callback(
    BC::BCFamilyMemoryCheckpoint &checkpoint,
    void *context
) {
    auto *ctx = static_cast<FamilyMemoryCheckpointContext *>(context);
    const ProcessMemorySnapshot snapshot = process_memory_snapshot();
    checkpoint.process_working_set_bytes = snapshot.working_set_bytes;
    checkpoint.process_peak_working_set_bytes = snapshot.peak_working_set_bytes;
    checkpoint.process_baseline_working_set_bytes =
        ctx == nullptr ? 0U : ctx->baseline_working_set_bytes;
    checkpoint.process_private_bytes = snapshot.private_bytes;
    checkpoint.process_pagefile_bytes = snapshot.pagefile_bytes;
    checkpoint.process_peak_pagefile_bytes = snapshot.peak_pagefile_bytes;
    if (checkpoint.process_working_set_bytes > checkpoint.accounted_bytes) {
        checkpoint.residual_bytes =
            checkpoint.process_working_set_bytes - checkpoint.accounted_bytes;
    }
    const uint64_t baseline = checkpoint.process_baseline_working_set_bytes;
    if (checkpoint.accounted_bytes <= std::numeric_limits<uint64_t>::max() - baseline) {
        const uint64_t baseline_plus_accounted = baseline + checkpoint.accounted_bytes;
        if (checkpoint.process_working_set_bytes > baseline_plus_accounted) {
            checkpoint.baseline_adjusted_residual_bytes =
                checkpoint.process_working_set_bytes - baseline_plus_accounted;
        }
    }
    if (ctx != nullptr) {
        ctx->records.push_back(checkpoint);
    }
}

struct AggregateStats {
    uint32_t layers = 0U;
    int effective_threads = 0;
    uint64_t input_live = 0U;
    uint64_t output_rows = 0U;
    uint64_t source_read_bytes = 0U;
    uint64_t blob_read_bytes = 0U;
    uint64_t blob_write_bytes = 0U;
    uint64_t blob_read_ops = 0U;
    uint64_t blob_write_ops = 0U;
    uint64_t blob_backend_read_bytes = 0U;
    uint64_t blob_backend_write_bytes = 0U;
    uint64_t target_logical_bytes = 0U;
    uint64_t target_write_bytes = 0U;
    uint64_t target_write_ops = 0U;
    uint64_t target_backend_read_bytes = 0U;
    uint64_t target_backend_write_bytes = 0U;
    double source_load_seconds = 0.0;
    double parallel_seconds = 0.0;
    double dump_seconds = 0.0;
    double reload_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double blob_backend_read_seconds = 0.0;
    double blob_backend_write_seconds = 0.0;
    double target_backend_read_seconds = 0.0;
    double target_backend_write_seconds = 0.0;
    double generation_seconds = 0.0;
    double total_seconds = 0.0;
    double resident_prepare_seconds = 0.0;
    double resident_work_seconds = 0.0;
    double resident_compute_seconds = 0.0;
    double resident_cleanup_seconds = 0.0;
    double resident_finalize_seconds = 0.0;
    double resident_write_seconds = 0.0;
    double resident_total_seconds = 0.0;
    double resident_mirror_seconds = 0.0;
    double resident_reader_open_seconds = 0.0;
    uint64_t active_family_window_peak = 0U;
    uint64_t target_active_cell_peak = 0U;
    uint64_t source_loaded_cell_peak = 0U;
    uint64_t active_builder_bytes_peak = 0U;
    uint64_t thread_workspace_bytes_peak = 0U;
    uint64_t process_peak_working_set_bytes = 0U;
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

[[nodiscard]] int generation_effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

[[nodiscard]] uint32_t single_route_cell_chunk_size(uint32_t cell_count) {
    return std::min<uint32_t>(
        kSingleRouteMaxCellChunkSize,
        std::max<uint32_t>(1U, (cell_count + 1U) / 2U)
    );
}

void configure_global_threads(int requested) {
#if defined(_OPENMP)
    if (requested > 0) {
        omp_set_num_threads(requested);
    }
#else
    (void)requested;
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

struct FamilyRouteScriptEntry {
    bool has_route = false;
    BC::BCFamilyGenerationRoute route = BC::BCFamilyGenerationRoute::Auto;
    uint32_t target_modulus = 0U;
    uint64_t available_memory_bytes = 0U;
};

struct FamilyRouteScript {
    std::map<uint32_t, FamilyRouteScriptEntry> by_layer_sum;
};

struct FamilyRoutePlannerState {
    BC::BCFamilyGenerationRoute previous_route = BC::BCFamilyGenerationRoute::Resident;
    uint32_t previous_modulus = 0U;
    uint32_t resident_upgrade_streak = 0U;
    uint32_t single_upgrade_streak = 0U;
};

[[nodiscard]] FamilyRouteScript load_family_route_script(const std::filesystem::path &path) {
    FamilyRouteScript script;
    if (path.empty()) {
        return script;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open --family-route-script: " + path.string());
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        return script;
    }
    const std::vector<std::string> header = split_csv_simple(header_line);
    std::map<std::string, size_t> column;
    for (size_t i = 0U; i < header.size(); ++i) {
        column.emplace(header[i], i);
    }
    auto optional_col = [&](const char *name) -> size_t {
        const auto it = column.find(name);
        return it == column.end() ? std::numeric_limits<size_t>::max() : it->second;
    };
    const size_t layer_col = optional_col("layer_sum");
    if (layer_col == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("--family-route-script missing layer_sum column");
    }
    const size_t route_col = optional_col("route");
    const size_t modulus_col = optional_col("target_modulus");
    const size_t avail_col = optional_col("available_memory_bytes");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> cells = split_csv_simple(line);
        if (layer_col >= cells.size() || cells[layer_col].empty()) {
            throw std::runtime_error("--family-route-script row missing layer_sum");
        }
        FamilyRouteScriptEntry entry;
        if (route_col < cells.size() && !cells[route_col].empty()) {
            entry.route = BC::bc_parse_family_route(cells[route_col]);
            entry.has_route = true;
        }
        if (modulus_col < cells.size() && !cells[modulus_col].empty()) {
            entry.target_modulus = static_cast<uint32_t>(std::stoul(cells[modulus_col]));
        }
        if (avail_col < cells.size() && !cells[avail_col].empty()) {
            entry.available_memory_bytes = static_cast<uint64_t>(std::stoull(cells[avail_col]));
        }
        script.by_layer_sum[static_cast<uint32_t>(std::stoul(cells[layer_col]))] = entry;
    }
    return script;
}

[[nodiscard]] BC::BCFamilyRouteDecision decide_family_route_for_layer(
    const Args &args,
    const FamilyRouteScript &script,
    FamilyRoutePlannerState &state,
    uint32_t layer_sum,
    const LayerFile &source2_layer,
    const LayerFile *source4_layer
) {
    const SystemMemorySnapshot memory = system_memory_snapshot();
    uint64_t available = memory.available_bytes;
    const auto script_it = script.by_layer_sum.find(layer_sum);
    const FamilyRouteScriptEntry *script_entry =
        script_it == script.by_layer_sum.end() ? nullptr : &script_it->second;
    if (script_entry != nullptr && script_entry->available_memory_bytes != 0U) {
        available = script_entry->available_memory_bytes;
    }
    BC::BCFamilyRouteInputs inputs;
    inputs.source2_size = source2_layer.physical_size != 0U
        ? source2_layer.physical_size
        : source2_layer.logical_size;
    inputs.source4_size = source4_layer == nullptr
        ? inputs.source2_size
        : (source4_layer->physical_size != 0U ? source4_layer->physical_size : source4_layer->logical_size);
    inputs.has_source4 = source4_layer != nullptr;
    inputs.available_memory_bytes = available;
    inputs.total_memory_bytes = memory.total_bytes;
    inputs.fixed_modulus = args.family_modulus;
    inputs.previous_modulus = state.previous_modulus == 0U ? args.family_modulus : state.previous_modulus;
    inputs.previous_route = state.previous_route;
    inputs.resident_upgrade_streak = state.resident_upgrade_streak;
    inputs.single_upgrade_streak = state.single_upgrade_streak;
    const BC::BCFamilyGenerationRoute requested =
        script_entry != nullptr && script_entry->has_route
            ? script_entry->route
            : args.family_route;
    BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, requested);
    if (script_entry != nullptr && script_entry->target_modulus != 0U) {
        if (script_entry->target_modulus != args.family_modulus) {
            throw std::invalid_argument(
                "--family-route-script target_modulus cannot change the fixed --family-modulus"
            );
        }
    }
    state.previous_route = decision.route;
    state.previous_modulus = decision.target_modulus;
    state.resident_upgrade_streak = decision.resident_upgrade_streak;
    state.single_upgrade_streak = decision.single_upgrade_streak;
    return decision;
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

[[nodiscard]] std::string layer_file_prefix(const Args &args) {
    return args.pattern + "_" + std::to_string(parse_rank_to_extra(args.target_rank));
}

[[nodiscard]] uint32_t layer_ordinal_for_sum(uint32_t seed_sum, uint32_t layer_sum) {
    if (layer_sum < seed_sum || ((layer_sum - seed_sum) & 1U) != 0U) {
        throw std::invalid_argument("layer sum does not belong to the generated sequence");
    }
    return (layer_sum - seed_sum) / 2U;
}

[[nodiscard]] uint64_t bc_position_bucket_stream_offset(const BCFamilyTable &axis) {
    const BCCellMatrix matrix(axis);
    const uint64_t descriptor_bytes =
        static_cast<uint64_t>(matrix.cell_count()) * BC::kBCPositionCellDescriptorBytes;
    return BC::bc_checked_add_u64(
        BC::bc_checked_add_u64(
            BC::kBCPositionHeaderBytes,
            BC::bc_axis_coord_table_bytes(axis.family_count()),
            "BC position archive axis table offset overflow"),
        descriptor_bytes,
        "BC position archive bucket stream offset overflow");
}

[[nodiscard]] std::filesystem::path layer_path(
    const Args &args,
    uint32_t seed_sum,
    uint32_t layer_sum
) {
    return args.output_dir /
        (layer_file_prefix(args) + "_" +
         std::to_string(layer_ordinal_for_sum(seed_sum, layer_sum)) + ".bcpos");
}

[[nodiscard]] std::filesystem::path layer_archive_path(const std::filesystem::path &position_path) {
    return std::filesystem::path(position_path.string() + ".7z");
}

[[nodiscard]] std::filesystem::path layer_cell_compressed_path(const std::filesystem::path &position_path) {
    return BC::bc_cell_compressed_position_path(position_path);
}

[[nodiscard]] bool is_position_archive_path(const std::filesystem::path &path) {
    const std::string name = path.filename().string();
    const std::string suffix = ".bcpos.7z";
    return name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[nodiscard]] bool is_cell_compressed_position_path(const std::filesystem::path &path) {
    return BC::bc_is_cell_compressed_position_path(path);
}

[[nodiscard]] std::string archive_entry_name_for_path(const std::filesystem::path &path) {
    std::string name = NativePath::to_utf8_string(path.filename());
    if (name.size() > 3U && name.compare(name.size() - 3U, 3U, ".7z") == 0) {
        name.resize(name.size() - 3U);
    }
    if (name.empty()) {
        name = "data.bcpos";
    }
    return name;
}

[[nodiscard]] std::filesystem::path layer_artifact_path(
    const Args &args,
    const std::filesystem::path &position_path
) {
    return args.compress_temp_files ? layer_archive_path(position_path) : position_path;
}

[[nodiscard]] std::filesystem::path layer_artifact_path(
    const Args &args,
    const std::filesystem::path &position_path,
    BC::BCFamilyGenerationRoute route
) {
    if (args.compress_temp_files && route == BC::BCFamilyGenerationRoute::Family) {
        return layer_cell_compressed_path(position_path);
    }
    return layer_artifact_path(args, position_path);
}

[[nodiscard]] bool same_path(
    const std::filesystem::path &lhs,
    const std::filesystem::path &rhs
) {
    return std::filesystem::absolute(lhs).lexically_normal() ==
        std::filesystem::absolute(rhs).lexically_normal();
}

[[nodiscard]] std::vector<std::filesystem::path> generated_candidate_dirs(const Args &args) {
    std::vector<std::filesystem::path> dirs;
    auto add = [&dirs](const std::filesystem::path &dir) {
        if (dir.empty()) {
            return;
        }
        for (const std::filesystem::path &existing : dirs) {
            if (same_path(existing, dir)) {
                return;
            }
        }
        dirs.push_back(dir);
    };
    add(args.output_dir);
    for (const std::filesystem::path &dir : args.output_dirs) {
        add(dir);
    }
    return dirs;
}

[[nodiscard]] std::map<uint32_t, std::filesystem::path> discover_existing_layer_paths(
    const Args &args
) {
    std::map<uint32_t, std::filesystem::path> layers;
    const std::string prefix = layer_file_prefix(args) + "_";
    const std::string suffix = ".bcpos";
    const std::string cell_suffix = ".bcposc";
    const std::string archive_suffix = ".bcpos.7z";
    for (const std::filesystem::path &dir : generated_candidate_dirs(args)) {
        if (!std::filesystem::is_directory(dir)) {
            continue;
        }
        for (const std::filesystem::directory_entry &entry :
             std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.size() <= prefix.size() + suffix.size() ||
                name.compare(0U, prefix.size(), prefix) != 0) {
                continue;
            }
            std::string ordinal_text;
            const bool is_raw =
                name.size() > prefix.size() + suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
            const bool is_cell_compressed =
                name.size() > prefix.size() + cell_suffix.size() &&
                name.compare(name.size() - cell_suffix.size(), cell_suffix.size(), cell_suffix) == 0;
            const bool is_archive =
                name.size() > prefix.size() + archive_suffix.size() &&
                name.compare(name.size() - archive_suffix.size(), archive_suffix.size(), archive_suffix) == 0;
            if (is_raw) {
                ordinal_text = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
            } else if (is_cell_compressed) {
                ordinal_text =
                    name.substr(prefix.size(), name.size() - prefix.size() - cell_suffix.size());
            } else if (is_archive) {
                ordinal_text =
                    name.substr(prefix.size(), name.size() - prefix.size() - archive_suffix.size());
            } else {
                continue;
            }
            if (ordinal_text.empty() ||
                !std::all_of(ordinal_text.begin(), ordinal_text.end(), [](char ch) {
                    return ch >= '0' && ch <= '9';
                })) {
                continue;
            }
            const unsigned long parsed = std::stoul(ordinal_text);
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                continue;
            }
            const uint32_t ordinal = static_cast<uint32_t>(parsed);
            const auto existing = layers.find(ordinal);
            const bool replace_archive =
                is_cell_compressed &&
                existing != layers.end() &&
                is_position_archive_path(existing->second);
            if (is_raw || replace_archive || existing == layers.end()) {
                layers[ordinal] = entry.path();
            }
        }
    }
    return layers;
}

[[nodiscard]] uint32_t contiguous_existing_layer_count(
    const std::map<uint32_t, std::filesystem::path> &layers
) {
    uint32_t expected = 0U;
    while (layers.find(expected) != layers.end()) {
        if (expected == std::numeric_limits<uint32_t>::max()) {
            break;
        }
        ++expected;
    }
    return expected;
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

[[nodiscard]] std::vector<uint8_t> make_bc_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("BC generation target rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(target_rank + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] BCLut make_bc_lut(uint32_t target_rank) {
    return BCLut(make_bc_legal_tiles(target_rank));
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
        throw std::invalid_argument("free family generation requires even layer sums");
    }
    if (family_modulus == 0U) {
        throw std::invalid_argument("BC generation family modulus must be non-zero");
    }
    return BC::build_family_partition_axis_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        BC::BCFamilyPartitionPolicy::modulo(family_modulus)
    );
}

[[nodiscard]] BCFamilyTable make_resident_axis(
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free resident generation requires even layer sums");
    }
    return make_axis(layer_sum, possible_8tile_sums, family_modulus);
}

[[nodiscard]] BCPositionCellLayout make_resident_layout(
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    return BC::build_modulo_position_cell_layout_for_layer(
        layer_sum,
        2U,
        possible_8tile_sums,
        family_modulus
    );
}

[[nodiscard]] bool axes_equivalent(
    const BCFamilyTable &lhs,
    const BCFamilyTable &rhs
) {
    return lhs.layer_sum() == rhs.layer_sum() &&
        lhs.family_unit() == rhs.family_unit() &&
        lhs.coords() == rhs.coords();
}

[[nodiscard]] BCFamilyTable make_target_axis_for_route(
    BC::BCFamilyGenerationRoute route,
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    switch (route) {
    case BC::BCFamilyGenerationRoute::Resident:
        return make_resident_axis(layer_sum, possible_8tile_sums, family_modulus);
    case BC::BCFamilyGenerationRoute::Single:
        return make_resident_axis(layer_sum, possible_8tile_sums, family_modulus);
    case BC::BCFamilyGenerationRoute::Family:
    case BC::BCFamilyGenerationRoute::Auto:
        return make_axis(layer_sum, possible_8tile_sums, family_modulus);
    }
    throw std::invalid_argument("unknown BC family generation route");
}

[[nodiscard]] bool layer_matches_layout(
    const BCPositionStreamingReader &reader,
    const BCPositionCellLayout &layout
) {
    return layout.equivalent_to_axis(reader.axis());
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
        throw std::invalid_argument("free generation pattern must be named freeN");
    }
    size_t parsed = 0U;
    const uint32_t value = static_cast<uint32_t>(std::stoul(pattern.substr(4U), &parsed));
    if (parsed != pattern.size() - 4U || value == 0U || value > 16U) {
        throw std::invalid_argument("invalid freeN generation pattern");
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

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_family_position_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.family_position_io == "direct-rank-first") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.preserve_unwritten_bytes = false;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_family_position_spool_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (args.family_position_io == "direct-rank-first") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.preserve_unwritten_bytes = false;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

[[nodiscard]] std::unique_ptr<BC::BCReadableFile> open_family_position_spool_reader(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size
) {
    if (args.family_position_io == "direct-rank-first") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        return std::make_unique<BC::BCDirectFileReader>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileReader>(path);
}

class BCSequentialArchiveWritableFile final : public BC::BCWritableFile {
public:
    explicit BCSequentialArchiveWritableFile(const std::filesystem::path &archive_path)
        : archive_path_(archive_path),
          writer_(
              NativePath::to_utf8_string(archive_path),
              archive_entry_name_for_path(archive_path),
              1) {}

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{offset, data, bytes}
        };
        write_many(requests);
    }

    void write_many(
        const std::vector<BC::BCFileWriteRequest> &requests,
        BC::BCFileIOStats *stats = nullptr
    ) override {
        if (stats != nullptr) {
            *stats = {};
        }
        if (closed_) {
            throw std::runtime_error("BC archive writer is already closed: " + archive_path_.string());
        }
        for (const BC::BCFileWriteRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC archive writer data pointer is null");
            }
            if (request.offset != cursor_) {
                throw std::runtime_error("BC archive writer requires sequential writes: " + archive_path_.string());
            }
            if (request.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC archive writer request exceeds size_t");
            }
            writer_.append(request.data, static_cast<size_t>(request.bytes));
            cursor_ = BC::bc_checked_add_u64(cursor_, request.bytes, "BC archive writer cursor overflow");
            BC::bc_fileio_accumulate_request(stats, request.bytes);
        }
    }

    void resize(uint64_t bytes) override {
        prepare_full_overwrite(bytes);
    }

    void prepare_full_overwrite(uint64_t bytes) override {
        if (cursor_ != 0U) {
            throw std::runtime_error("BC archive writer cannot resize after writing: " + archive_path_.string());
        }
        expected_size_ = bytes;
    }

    void flush() override {
        if (closed_) {
            return;
        }
        if (expected_size_.has_value() && cursor_ != *expected_size_) {
            throw std::runtime_error("BC archive writer byte count mismatch: " + archive_path_.string());
        }
        writer_.close();
        closed_ = true;
    }

private:
    std::filesystem::path archive_path_;
    SevenZipArchiveWriter writer_;
    std::optional<uint64_t> expected_size_;
    uint64_t cursor_ = 0U;
    bool closed_ = false;
};

class BCOffsetWritableFile final : public BC::BCWritableFile {
public:
    BCOffsetWritableFile(std::unique_ptr<BC::BCWritableFile> inner, uint64_t base_offset)
        : inner_(std::move(inner)), base_offset_(base_offset) {
        if (!inner_) {
            throw std::invalid_argument("BC offset writer requires an inner writer");
        }
    }

    [[nodiscard]] BC::BCFileIOMode mode() const override {
        return inner_->mode();
    }

    [[nodiscard]] uint32_t preferred_write_alignment() const override {
        return inner_->preferred_write_alignment();
    }

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{offset, data, bytes}
        };
        write_many(requests);
    }

    void write_many(
        const std::vector<BC::BCFileWriteRequest> &requests,
        BC::BCFileIOStats *stats = nullptr
    ) override {
        shifted_requests_.clear();
        shifted_requests_.reserve(requests.size());
        for (const BC::BCFileWriteRequest &request : requests) {
            if (request.offset < base_offset_) {
                throw std::runtime_error("BC offset writer saw write before base offset");
            }
            shifted_requests_.push_back(BC::BCFileWriteRequest{
                request.offset - base_offset_,
                request.data,
                request.bytes
            });
        }
        inner_->write_many(shifted_requests_, stats);
    }

    void resize(uint64_t bytes) override {
        if (bytes <= base_offset_) {
            inner_->resize(0U);
            return;
        }
        inner_->resize(bytes - base_offset_);
    }

    void prepare_full_overwrite(uint64_t bytes) override {
        resize(bytes);
    }

    void flush() override {
        inner_->flush();
    }

private:
    std::unique_ptr<BC::BCWritableFile> inner_;
    uint64_t base_offset_ = 0U;
    std::vector<BC::BCFileWriteRequest> shifted_requests_;
};

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_position_layer_artifact_writer(
    const Args &args,
    const std::filesystem::path &position_path
) {
    if (args.compress_temp_files) {
        return std::make_unique<BCSequentialArchiveWritableFile>(layer_archive_path(position_path));
    }
    return open_family_position_writer(args, position_path);
}

void write_position_bytes_to_file(
    const Args &args,
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes,
    BC::BCFileIOStats *stats = nullptr
) {
    std::unique_ptr<BC::BCWritableFile> writer = open_family_position_writer(args, path);
    writer->prepare_full_overwrite(static_cast<uint64_t>(bytes.size()));
    if (stats != nullptr) {
        *stats = {};
    }
    if (!bytes.empty()) {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{0U, bytes.data(), static_cast<uint64_t>(bytes.size())}
        };
        writer->write_many(requests, stats);
    }
    writer->flush();
}

[[nodiscard]] std::filesystem::path write_position_bytes_to_layer_artifact(
    const Args &args,
    const std::filesystem::path &position_path,
    const std::vector<uint8_t> &bytes,
    BC::BCFileIOStats *stats = nullptr
) {
    if (!args.compress_temp_files) {
        write_position_bytes_to_file(args, position_path, bytes, stats);
        std::error_code ec;
        std::filesystem::remove(layer_archive_path(position_path), ec);
        return position_path;
    }
    const std::filesystem::path archive_path = layer_archive_path(position_path);
    BCSequentialArchiveWritableFile writer(archive_path);
    writer.prepare_full_overwrite(static_cast<uint64_t>(bytes.size()));
    if (stats != nullptr) {
        *stats = {};
    }
    if (!bytes.empty()) {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{0U, bytes.data(), static_cast<uint64_t>(bytes.size())}
        };
        writer.write_many(requests, stats);
    }
    writer.flush();
    std::error_code ec;
    std::filesystem::remove(position_path, ec);
    return archive_path;
}

void write_position_bytes_to_file_buffered_padded(
    const Args &args,
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    BC::BCBufferedFileWriter writer(path);
    writer.prepare_full_overwrite(static_cast<uint64_t>(bytes.size()));
    if (!bytes.empty()) {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{0U, bytes.data(), static_cast<uint64_t>(bytes.size())}
        };
        writer.write_many(requests);
    }
    writer.flush();
    const uint64_t physical_size = args.family_position_io == "direct-rank-first"
        ? align_up_u64(static_cast<uint64_t>(bytes.size()), 4096ULL)
        : static_cast<uint64_t>(bytes.size());
    if (physical_size != bytes.size()) {
        std::error_code ec;
        std::filesystem::resize_file(path, physical_size, ec);
        if (ec) {
            throw std::runtime_error("BC buffered position writer padding resize failed: " + ec.message());
        }
    }
}

class BCAsyncPositionWriteQueue {
public:
    void enqueue(
        const Args &args,
        uint32_t layer_sum,
        const std::filesystem::path &path,
        std::shared_ptr<const BC::BCPositionLayerReader> reader
    ) {
        if (!reader) {
            throw std::invalid_argument("BC async position write requires a reader");
        }
        wait_until_below_limit();
        std::error_code ec;
        std::filesystem::remove(path, ec);
        Args args_copy = args;
        PendingWrite pending;
        pending.layer_sum = layer_sum;
        pending.path = path;
        pending.reader = std::move(reader);
        pending.future = std::async(
            std::launch::async,
            [args_copy = std::move(args_copy), path, reader = pending.reader] {
                (void)write_position_bytes_to_layer_artifact(args_copy, path, reader->bytes());
            }
        );
        pending_.push_back(std::move(pending));
    }

    void wait_for_layer(uint32_t layer_sum) {
        for (size_t i = 0U; i < pending_.size();) {
            if (pending_[i].layer_sum != layer_sum) {
                ++i;
                continue;
            }
            wait_one(i);
        }
    }

    void wait_all() {
        while (!pending_.empty()) {
            wait_one(0U);
        }
    }

private:
    struct PendingWrite {
        uint32_t layer_sum = 0U;
        std::filesystem::path path;
        std::shared_ptr<const BC::BCPositionLayerReader> reader;
        std::future<void> future;
    };

    static constexpr size_t kMaxPendingWrites = 2U;

    void wait_until_below_limit() {
        while (pending_.size() >= kMaxPendingWrites) {
            wait_one(0U);
        }
    }

    void wait_one(size_t index) {
        if (index >= pending_.size()) {
            throw std::out_of_range("BC async position write index out of range");
        }
        pending_[index].future.get();
        pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
    }

    std::vector<PendingWrite> pending_;
};

class BCMemoryMirrorWritableFile final : public BC::BCWritableFile {
public:
    explicit BCMemoryMirrorWritableFile(std::unique_ptr<BC::BCWritableFile> inner)
        : inner_(std::move(inner)) {
        if (!inner_) {
            throw std::invalid_argument("BC memory mirror writer requires an inner writer");
        }
    }

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        const std::vector<BC::BCFileWriteRequest> requests{
            BC::BCFileWriteRequest{offset, data, bytes}
        };
        write_many(requests);
    }

    void write_many_overlapped_mirror(
        const std::vector<BC::BCFileWriteRequest> &requests,
        BC::BCFileIOStats *stats
    ) {
        auto mirror_future = std::async(std::launch::async, [this, &requests] {
            mirror_requests(requests);
        });
        try {
            inner_->write_many(requests, stats);
        } catch (...) {
            try {
                mirror_future.get();
            } catch (...) {
            }
            throw;
        }
        mirror_future.get();
    }

    void write_many(
        const std::vector<BC::BCFileWriteRequest> &requests,
        BC::BCFileIOStats *stats = nullptr
    ) override {
        if (requests.empty()) {
            return;
        }
        uint64_t total_bytes = 0U;
        for (const BC::BCFileWriteRequest &request : requests) {
            if (total_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                throw std::overflow_error("BC memory mirror write request byte count overflow");
            }
            total_bytes += request.bytes;
        }
        constexpr uint64_t kAsyncMirrorThreshold = 8ULL * 1024ULL * 1024ULL;
        if (total_bytes >= kAsyncMirrorThreshold) {
            write_many_overlapped_mirror(requests, stats);
        } else {
            inner_->write_many(requests, stats);
            mirror_requests(requests);
        }
    }

    [[nodiscard]] BC::BCFileIOMode mode() const override {
        return inner_->mode();
    }

    [[nodiscard]] uint32_t preferred_write_alignment() const override {
        return inner_->preferred_write_alignment();
    }

    void resize(uint64_t bytes) override {
        inner_->resize(bytes);
        logical_size_ = bytes;
        mirror_.clear();
        reserve_mirror(bytes);
    }

    void prepare_full_overwrite(uint64_t bytes) override {
        inner_->prepare_full_overwrite(bytes);
        logical_size_ = bytes;
        mirror_.clear();
        reserve_mirror(bytes);
    }

    void flush() override {
        inner_->flush();
    }

    [[nodiscard]] std::vector<uint8_t> take_mirror() && {
        if (mirror_.size() > logical_size_) {
            throw std::runtime_error("BC memory mirror writer exceeded logical size");
        }
        if (mirror_.size() < logical_size_) {
            mirror_.resize(checked_size_t(logical_size_, "BC memory mirror logical size exceeds size_t"), 0U);
        }
        return std::move(mirror_);
    }

    [[nodiscard]] double mirror_seconds() const noexcept {
        return mirror_seconds_;
    }

private:
    static size_t checked_size_t(uint64_t value, const char *message) {
        if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error(message);
        }
        return static_cast<size_t>(value);
    }

    void reserve_mirror(uint64_t bytes) {
        if (bytes == 0U) {
            return;
        }
        mirror_.reserve(checked_size_t(bytes, "BC memory mirror reserve exceeds size_t"));
    }

    void mirror_request(const BC::BCFileWriteRequest &request) {
        if (request.bytes == 0U) {
            return;
        }
        if (request.data == nullptr) {
            throw std::invalid_argument("BC memory mirror write request data pointer is null");
        }
        if (request.offset > std::numeric_limits<uint64_t>::max() - request.bytes) {
            throw std::overflow_error("BC memory mirror write request end overflow");
        }
        const uint64_t end = request.offset + request.bytes;
        const size_t offset_size = checked_size_t(request.offset, "BC memory mirror offset exceeds size_t");
        const size_t end_size = checked_size_t(end, "BC memory mirror end exceeds size_t");
        if (end > logical_size_) {
            logical_size_ = end;
        }
        if (mirror_.size() < offset_size) {
            mirror_.resize(offset_size, 0U);
        }
        if (mirror_.size() < end_size) {
            mirror_.resize(end_size);
        }
        std::memcpy(
            mirror_.data() + offset_size,
            request.data,
            checked_size_t(request.bytes, "BC memory mirror request size exceeds size_t")
        );
    }

    void mirror_requests(const std::vector<BC::BCFileWriteRequest> &requests) {
        const double begin = now_seconds();
        for (const BC::BCFileWriteRequest &request : requests) {
            mirror_request(request);
        }
        mirror_seconds_ += now_seconds() - begin;
    }

    std::unique_ptr<BC::BCWritableFile> inner_;
    std::vector<uint8_t> mirror_;
    uint64_t logical_size_ = 0U;
    double mirror_seconds_ = 0.0;
};

[[nodiscard]] std::vector<uint8_t> read_position_file_to_memory(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size
) {
    if (logical_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC position file exceeds addressable vector size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(logical_size));
    if (logical_size == 0U) {
        return bytes;
    }
    if (args.family_position_io == "direct-rank-first") {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        BC::BCDirectFileReader reader(path, options);
        reader.read_at(0U, bytes.data(), logical_size);
        return bytes;
    }
    BC::BCBufferedFileReader reader(path);
    if (reader.size() != logical_size) {
        throw std::runtime_error("buffered BC position file size mismatch");
    }
    reader.read_at(0U, bytes.data(), logical_size);
    return bytes;
}

void fill_writer_stats_from_resident_result(
    BC::BCFamilyPositionWriterStats &writer_stats,
    const BC::BCResidentGenerationResult &resident
) {
    writer_stats = {};
    writer_stats.success_rows = resident.output_success_rows;
    writer_stats.metadata_write_ops = resident.target_position_write_requests;
    writer_stats.metadata_write_bytes = resident.target_position_write_requested_bytes;
    writer_stats.backend_write_ops = resident.target_position_write_backend_ops;
    writer_stats.backend_write_bytes = resident.target_position_write_backend_bytes;
    writer_stats.backend_write_seconds = resident.write_seconds;
    writer_stats.logical_size = resident.target_position_file_logical_bytes;
}

[[nodiscard]] std::unique_ptr<BCPositionStreamingReader> open_position_reader(
    const Args &args,
    const LayerFile &layer,
    const BCLut &lut
) {
    std::unique_ptr<BC::BCReadableFile> file;
    if (is_position_archive_path(layer.path)) {
        std::vector<uint8_t> bytes = read_temp_byte_archive(NativePath::to_utf8_string(layer.path));
        if (bytes.empty()) {
            throw std::runtime_error("failed to read BC generated position archive: " + layer.path.string());
        }
        file = std::make_unique<BC::BCMemoryReadableFile>(std::move(bytes));
        auto reader = std::make_unique<BCPositionStreamingReader>(std::move(file), lut);
        reader->set_validate_loaded_cells(false);
        return reader;
    }
    if (is_cell_compressed_position_path(layer.path)) {
        file = std::make_unique<BC::BCCellCompressedPositionReadableFile>(layer.path);
        auto reader = std::make_unique<BCPositionStreamingReader>(std::move(file), lut);
        reader->set_validate_loaded_cells(false);
        return reader;
    }
    if (args.family_source_io == "direct" || args.family_source_io == "direct-auto") {
        const uint64_t required_physical = align_up_u64(layer.logical_size, 4096ULL);
        if (layer.physical_size >= required_physical) {
            BC::BCDirectFileIOOptions options;
            options.queue_depth = args.direct_queue_depth;
            options.overlapped = args.direct_queue_depth > 1U;
            options.logical_size = layer.logical_size;
            file = std::make_unique<BC::BCDirectFileReader>(layer.path, options);
        } else if (args.family_source_io == "direct") {
            throw std::runtime_error(
                "source position file is not padded for direct IO: " + layer.path.string()
            );
        }
    }
    if (!file) {
        file = std::make_unique<BC::BCBufferedFileReader>(layer.path);
    }
    auto reader = std::make_unique<BCPositionStreamingReader>(std::move(file), lut);
    reader->set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] BC::BCPositionFileReader open_position_file_reader(
    const Args &args,
    const LayerFile &layer,
    const BCLut &lut
) {
    if (is_position_archive_path(layer.path)) {
        std::vector<uint8_t> bytes = read_temp_byte_archive(NativePath::to_utf8_string(layer.path));
        if (bytes.empty()) {
            throw std::runtime_error("failed to read BC generated position archive: " + layer.path.string());
        }
        return BC::BCPositionFileReader(
            std::make_unique<BC::BCMemoryReadableFile>(std::move(bytes)),
            lut);
    }
    if (is_cell_compressed_position_path(layer.path)) {
        return BC::BCPositionFileReader(
            std::make_unique<BC::BCCellCompressedPositionReadableFile>(layer.path),
            lut);
    }
    return BC::BCPositionFileReader::open_direct_auto(
        layer.path,
        lut,
        args.direct_queue_depth,
        args.direct_queue_depth > 1U);
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
    if (is_position_archive_path(path) || is_cell_compressed_position_path(path)) {
        layer.physical_size = logical_size;
    }
    auto reader = open_position_reader(args, layer, lut);
    reader->set_validate_loaded_cells(false);
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
    std::unique_ptr<BCPositionStreamingReader> archive_reader;
    std::unique_ptr<BCPositionStreamingReader> cell_compressed_reader;
    BCPositionStreamingReader raw_reader;
    BCPositionStreamingReader *reader_ptr = nullptr;
    if (is_position_archive_path(path)) {
        std::vector<uint8_t> bytes = read_temp_byte_archive(NativePath::to_utf8_string(path));
        if (bytes.empty()) {
            throw std::runtime_error("failed to read existing BC generated position archive: " + path.string());
        }
        archive_reader = std::make_unique<BCPositionStreamingReader>(
            std::make_unique<BC::BCMemoryReadableFile>(std::move(bytes)),
            lut);
        reader_ptr = archive_reader.get();
    } else if (is_cell_compressed_position_path(path)) {
        cell_compressed_reader = std::make_unique<BCPositionStreamingReader>(
            std::make_unique<BC::BCCellCompressedPositionReadableFile>(path),
            lut);
        reader_ptr = cell_compressed_reader.get();
    } else {
        raw_reader = BCPositionStreamingReader::open_direct_auto(path, lut, 8U, true);
        reader_ptr = &raw_reader;
    }
    BCPositionStreamingReader &reader = *reader_ptr;
    reader.set_validate_loaded_cells(false);
    LayerFile layer;
    layer.layer_sum = BC::checked_u32_size(
        static_cast<size_t>(reader.axis().layer_sum()),
        "existing layer sum exceeds uint32"
    );
    layer.path = path;
    layer.logical_size = position_header_logical_size(reader.header());
    layer.physical_size =
        (is_position_archive_path(path) || is_cell_compressed_position_path(path))
            ? layer.logical_size
            : file_size;
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
    if (is_cell_compressed_position_path(path) || is_position_archive_path(path)) {
        layer.physical_size = logical_size;
    }
    layer.rows = rows;
    layer.bucket_count = bucket_count;
    layer.rank_payload_bytes = rank_payload_bytes;
    return layer;
}

[[nodiscard]] uint64_t expected_position_physical_size(
    const Args &args,
    uint64_t logical_size
) {
    return args.family_position_io == "direct-rank-first"
        ? align_up_u64(logical_size, 4096ULL)
        : logical_size;
}

[[nodiscard]] LayerFile layer_file_from_known_metadata(
    const Args &args,
    uint32_t layer_sum,
    const std::filesystem::path &path,
    uint64_t logical_size,
    uint64_t rows,
    uint64_t bucket_count,
    uint64_t rank_payload_bytes
) {
    LayerFile layer;
    layer.layer_sum = layer_sum;
    layer.path = path;
    layer.logical_size = logical_size;
    layer.physical_size = expected_position_physical_size(args, logical_size);
    layer.rows = rows;
    layer.bucket_count = bucket_count;
    layer.rank_payload_bytes = rank_payload_bytes;
    return layer;
}

struct MaterializedLayerView {
    LayerFile layer;
    bool temporary = false;
};

[[nodiscard]] FinalizedCellPayload payload_from_loaded_cell(BC::BCLoadedCell &&cell) {
    FinalizedCellPayload payload;
    payload.buckets = std::move(cell.buckets);
    payload.rank_payload = std::move(cell.rank_payload);
    payload.success_rows = cell.success_rows;
    return payload;
}

[[nodiscard]] MaterializedLayerView ensure_layer_cell_layout(
    const Args &args,
    const BCLut &lut,
    const LayerFile &source_layer,
    const BCPositionCellLayout &layout,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::filesystem::path &temp_path
) {
    std::unique_ptr<BCPositionStreamingReader> reader =
        open_position_reader(args, source_layer, lut);
    if (layer_matches_layout(*reader, layout)) {
        return MaterializedLayerView{source_layer, false};
    }

    BC::BCPositionFamilyRemapReader remap(
        *reader,
        layout.serialization_axis(),
        possible_8tile_sums
    );
    std::vector<FinalizedCellPayload> payloads(layout.cell_count());
    const uint32_t cell_chunk_size = single_route_cell_chunk_size(layout.cell_count());
    std::vector<CellId> cids;
    cids.reserve(cell_chunk_size);
    std::vector<BC::BCLoadedCell> loaded;
    for (CellId begin = 0U; begin < layout.cell_count();) {
        const CellId end = static_cast<CellId>(
            std::min<uint64_t>(
                layout.cell_count(),
                static_cast<uint64_t>(begin) + cell_chunk_size
            )
        );
        cids.clear();
        for (CellId cid = begin; cid < end; ++cid) {
            cids.push_back(cid);
        }
        remap.load_cells_into(cids, loaded, nullptr);
        for (BC::BCLoadedCell &cell : loaded) {
            payloads[static_cast<size_t>(cell.cid)] =
                payload_from_loaded_cell(std::move(cell));
        }
        begin = end;
    }

    {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
    }
    std::unique_ptr<BC::BCWritableFile> writer =
        open_family_position_writer(args, temp_path);
    BC::BCFileIOStats stats;
    const uint64_t logical_size =
        BC::write_position_payloads_to_file(
            *writer,
            layout.serialization_axis(),
            payloads,
            &stats
        );
    writer.reset();
    (void)stats;

    LayerFile materialized =
        args.output_inspect
            ? inspect_layer_file(
                  args,
                  static_cast<uint32_t>(layout.layer_sum()),
                  temp_path,
                  logical_size,
                  lut)
            : layer_file_without_inspect(
                  static_cast<uint32_t>(layout.layer_sum()),
                  temp_path,
                  logical_size,
                  source_layer.rows);
    return MaterializedLayerView{std::move(materialized), true};
}

[[nodiscard]] std::vector<uint8_t> build_initial_layer_bytes(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const std::vector<uint64_t> &initial_boards,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus,
    int canonical_symm_mode
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
        uint64_t canonical = board;
        CanonicalBatch::canonicalize_inplace(&canonical, 1U, canonical_symm_mode);
        const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(canonical);
        BC::BCEncodedKeyRank key_rank = BC::encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
        check(key_rank.valid, "BC initial board should encode key/rank");
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
    uint32_t seed_sum,
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
            args.family_modulus,
            args.canonical_symm_mode
    );
    BC::BCPositionLayerReader initial_reader(bytes, lut);
    uint64_t success_rows = 0U;
    for (CellId cid = 0U; cid < initial_reader.cell_count(); ++cid) {
        success_rows += initial_reader.descriptor(cid).success_rows;
    }
    const uint64_t bucket_count =
        initial_reader.header().bucket_meta_bytes / BC::kBCPositionBucketEntryBytes;
    const uint64_t rank_payload_bytes = initial_reader.header().rank_payload_bytes;
    const std::filesystem::path path = layer_path(args, seed_sum, axis.layer_sum());
    const std::filesystem::path artifact_path =
        write_position_bytes_to_layer_artifact(args, path, bytes);
    if (!args.output_inspect) {
        return layer_file_without_inspect(
            axis.layer_sum(),
            artifact_path,
            bytes.size(),
            success_rows,
            bucket_count,
            rank_payload_bytes);
    }
    return inspect_layer_file(args, axis.layer_sum(), artifact_path, bytes.size(), lut);
}

void cleanup_temp_file(const std::filesystem::path &path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

[[nodiscard]] uint64_t layer_storage_size_bytes(const LayerFile &layer) {
    return layer.physical_size != 0U ? layer.physical_size : layer.logical_size;
}

[[nodiscard]] BC::BCFamilyGenerationOptions family_options_from_args(
    const Args &args,
    uint32_t target_modulus,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal
) {
    BC::BCFamilyGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    options.canonical_symm_mode = args.canonical_symm_mode;
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
    options.family_partition_policy = BC::BCFamilyPartitionPolicy::modulo(target_modulus);
    options.family_possible_8tile_sums = &possible_8tile_sums;
    options.pattern_masks = &args.pattern_masks;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    const bool is_free_pattern = args.pattern.rfind("free", 0U) == 0U;
    options.success_check_all_cells =
        is_free_pattern || args.success_shifts.empty();
    options.keep_only_success_generated_boards = terminal;
    options.finalize_options.keyvalue_sort = nullptr;
    options.finalize_options.simd_sort_min_bucket_count = 10000U;
    return options;
}

[[nodiscard]] BC::BCResidentGenerationOptions resident_options_from_args(
    const Args &args,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    uint32_t route_default_pending_buffer
) {
    BC::BCResidentGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    options.canonical_symm_mode = args.canonical_symm_mode;
    options.dynamic_reserve_factor = kBCDefaultReserveFactor;
    options.collect_timing = false;
    options.pending_insert_buffer_size =
        args.pending_buffer != 0U ? args.pending_buffer : route_default_pending_buffer;
    options.tile_sum_values = &tile_sums;
    options.pattern_masks = &args.pattern_masks;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    options.finalize_options.keyvalue_sort = nullptr;
    options.finalize_options.simd_sort_min_bucket_count = 10000U;
    options.collect_dynamic_state_stats = false;
    options.collect_mutable_output_stats = false;
    return options;
}

struct FamilyLayerResult {
    BC::BCFamilyGenerationStats stats;
    BC::BCFamilyPositionWriterStats writer_stats;
    BC::BCGenerationBlobIOStats blob_stats;
    std::vector<BC::BCFamilyMemoryCheckpoint> memory_checkpoints;
    uint64_t logical_size = 0U;
    uint64_t output_rows = 0U;
    uint64_t output_bucket_count = 0U;
    uint64_t output_rank_payload_bytes = 0U;
    int effective_threads = 1;
    uint32_t retries = 0U;
    double total_seconds = 0.0;
    BC::BCFamilyGenerationRoute route = BC::BCFamilyGenerationRoute::Family;
    uint32_t target_modulus = 0U;
    uint64_t available_memory_bytes = 0U;
    uint64_t route_estimated_peak_bytes = 0U;
    uint64_t route_budget_bytes = 0U;
    uint64_t dynamic_bucket_slots_used = 0U;
    uint64_t dynamic_bitmap_words_allocated = 0U;
    double resident_prepare_seconds = 0.0;
    double resident_work_seconds = 0.0;
    double resident_compute_seconds = 0.0;
    double resident_cleanup_seconds = 0.0;
    double resident_finalize_seconds = 0.0;
    double resident_write_seconds = 0.0;
    double resident_total_seconds = 0.0;
    double resident_mirror_seconds = 0.0;
    double resident_reader_open_seconds = 0.0;
    std::shared_ptr<BC::BCPositionLayerReader> resident_memory_layer;
};

double bc_observed_reserve_need(
    const LayerFile &current,
    const FamilyLayerResult &result
) {
    double need = 0.0;
    need = std::max(
        need,
        bc_transition_component_reserve_need(
            current.bucket_count,
            result.dynamic_bucket_slots_used,
            4096ULL
        )
    );
    const uint64_t current_rank_words = current.rank_payload_bytes / sizeof(uint64_t) + 1U;
    need = std::max(
        need,
        bc_transition_component_reserve_need(
            current_rank_words,
            result.dynamic_bitmap_words_allocated,
            512ULL * 64ULL
        )
    );
    return need;
}

BC::BCFamilyGenerationRunResult *g_run_result = nullptr;
const BC::BCFamilyGenerationLayerCallback *g_layer_callback = nullptr;

void emit_generation_layer_metric(
    uint32_t ordinal,
    uint32_t layer_sum,
    uint64_t input_rows,
    const FamilyLayerResult &result,
    const std::filesystem::path &output_path
) {
    BC::BCFamilyGenerationRunLayerMetric metric;
    metric.kind = "generation";
    metric.ordinal = ordinal;
    metric.layer_sum = layer_sum;
    metric.input_rows = input_rows;
    metric.output_rows = result.output_rows;
    metric.logical_size = result.logical_size;
    metric.target_modulus = result.target_modulus;
    metric.route = result.route;
    metric.total_seconds = result.total_seconds;
    metric.output_path = output_path;
    if (g_run_result != nullptr) {
        g_run_result->layers.push_back(metric);
    }
    if (g_layer_callback != nullptr && *g_layer_callback) {
        (*g_layer_callback)(metric);
    }
}

void emit_seed_layer_metric(
    uint32_t ordinal,
    uint32_t layer_sum,
    const LayerFile &layer
) {
    BC::BCFamilyGenerationRunLayerMetric metric;
    metric.kind = "seed";
    metric.ordinal = ordinal;
    metric.layer_sum = layer_sum;
    metric.input_rows = layer.rows;
    metric.output_rows = layer.rows;
    metric.logical_size = layer.logical_size;
    metric.target_modulus = 0U;
    metric.route = BC::BCFamilyGenerationRoute::Family;
    metric.total_seconds = 0.0;
    metric.output_path = layer.path;
    if (g_run_result != nullptr) {
        g_run_result->layers.push_back(metric);
    }
    if (g_layer_callback != nullptr && *g_layer_callback) {
        (*g_layer_callback)(metric);
    }
}

void apply_route_decision_to_result(
    FamilyLayerResult &result,
    const BC::BCFamilyRouteDecision &decision
) {
    result.route = decision.route;
    if (result.target_modulus == 0U) {
        result.target_modulus = decision.target_modulus;
    }
    result.available_memory_bytes = decision.available_memory_bytes;
    result.route_estimated_peak_bytes = decision.route_estimated_peak_bytes;
    result.route_budget_bytes = decision.route_budget_bytes;
}

[[nodiscard]] FamilyLayerResult generate_resident_layer_to_file(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const LayerFile *source4_layer,
    const LayerFile &source2_layer,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal,
    const std::filesystem::path &output_path
) {
    FamilyLayerResult result;
    const double total_begin = now_seconds();
    const double source_load_begin = now_seconds();
    BC::BCPositionFileReader source2_file =
        open_position_file_reader(args, source2_layer, lut);
    std::optional<BC::BCPositionFileReader> source4_file;
    std::vector<BC::BCResidentGenerationSource> sources;
    sources.reserve(source4_layer == nullptr ? 1U : 2U);
    uint64_t source_bytes = source2_layer.physical_size != 0U
        ? source2_layer.physical_size
        : source2_layer.logical_size;
    if (source4_layer != nullptr) {
        source4_file.emplace(open_position_file_reader(args, *source4_layer, lut));
        source_bytes += source4_layer->physical_size != 0U
            ? source4_layer->physical_size
            : source4_layer->logical_size;
        sources.push_back(BC::BCResidentGenerationSource{&source4_file->layer(), 2U, 2U});
    }
    sources.push_back(BC::BCResidentGenerationSource{&source2_file.layer(), 1U, 1U});
    const double source_load_seconds = now_seconds() - source_load_begin;

    BC::BCResidentGenerationOptions options = resident_options_from_args(
        args,
        tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        kResidentRoutePendingBuffer
    );
    options.keep_only_success_generated_boards = terminal;
    cleanup_temp_file(output_path);
    cleanup_temp_file(layer_archive_path(output_path));
    cleanup_temp_file(layer_cell_compressed_path(output_path));
    std::unique_ptr<BC::BCWritableFile> writer =
        open_position_layer_artifact_writer(args, output_path);
    BCMemoryMirrorWritableFile mirror_writer(std::move(writer));
    BC::BCResidentGenerationResult resident =
        BC::generate_resident_position_layer_to_file(
            lut,
            target_layout,
            sources,
            mirror_writer,
            options
        );
    mirror_writer.flush();

    result.logical_size = resident.target_position_file_logical_bytes;
    result.output_rows = resident.output_success_rows;
    result.effective_threads = resident.effective_threads;
    result.retries = resident.generation_retries;
    result.total_seconds = now_seconds() - total_begin;
    result.target_modulus = 0U;
    result.dynamic_bucket_slots_used = resident.dynamic_bucket_slots_used;
    result.dynamic_bitmap_words_allocated = resident.dynamic_bitmap_words_allocated;
    result.writer_stats.success_rows = resident.output_success_rows;
    fill_writer_stats_from_resident_result(result.writer_stats, resident);
    std::vector<uint8_t> position_bytes = std::move(mirror_writer).take_mirror();
    if (!position_bytes.empty()) {
        const BC::BCPositionHeader header = BC::bc_read_header(position_bytes);
        result.output_bucket_count = header.bucket_meta_bytes / BC::kBCPositionBucketEntryBytes;
        result.output_rank_payload_bytes = header.rank_payload_bytes;
    }
    result.resident_prepare_seconds = resident.prepare_seconds;
    result.resident_work_seconds = resident.work_seconds;
    result.resident_compute_seconds = resident.compute_seconds;
    result.resident_cleanup_seconds = resident.cleanup_seconds;
    result.resident_finalize_seconds = resident.finalize_seconds;
    result.resident_write_seconds = resident.write_seconds;
    result.resident_total_seconds = resident.total_seconds;
    const double reader_open_begin = now_seconds();
    result.resident_memory_layer =
        std::make_shared<BC::BCPositionLayerReader>(std::move(position_bytes), lut);
    result.resident_reader_open_seconds = now_seconds() - reader_open_begin;

    result.stats.source_bytes_read = source_bytes;
    result.stats.source_backend_read_ops = source4_layer == nullptr ? 1U : 2U;
    result.stats.source_backend_read_bytes = source_bytes;
    result.stats.source_load_seconds = source_load_seconds;
    result.stats.parallel_seconds = resident.generation_seconds;
    result.stats.finalize_seconds = resident.finalize_seconds;
    result.stats.write_seconds = resident.write_seconds;
    result.stats.generation_seconds = result.total_seconds;
    result.stats.target_cells_created = target_layout.cell_count();
    result.stats.target_cells_finalized = result.stats.target_cells_created;
    result.stats.family_builder_hash_grows = resident.generation_retries;
    result.stats.active_builder_bytes_peak =
        resident.dynamic_bitmap_words_allocated * static_cast<uint64_t>(sizeof(uint64_t));
    return result;
}

[[nodiscard]] FamilyLayerResult generate_resident_mutable_carry_layer_to_file(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout &current_layout,
    const BCPositionCellLayout *secondary_layout,
    const LayerFile &current_layer,
    const BC::BCPositionLayerReader *current_memory_layer,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_to_primary,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    double dynamic_reserve_factor,
    bool primary_terminal,
    bool secondary_terminal,
    const std::filesystem::path &output_path,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> &next_carry,
    uint32_t &next_carry_layer_sum
) {
    FamilyLayerResult result;
    const double total_begin = now_seconds();
    double source_load_seconds = 0.0;
    std::optional<BC::BCPositionFileReader> current_file;
    const BC::BCPositionLayerReader *current_reader = current_memory_layer;
    if (current_reader == nullptr) {
        const double source_load_begin = now_seconds();
        current_file.emplace(open_position_file_reader(args, current_layer, lut));
        source_load_seconds += now_seconds() - source_load_begin;
        current_reader = &current_file->layer();
    }
    if (!current_layout.equivalent_to_axis(current_reader->axis())) {
        throw std::runtime_error("resident route current layer was not materialized to requested cell layout");
    }

    BC::BCResidentGenerationOptions options = resident_options_from_args(
        args,
        tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        kResidentRoutePendingBuffer
    );
    options.dynamic_reserve_factor = dynamic_reserve_factor;
    options.keep_only_success_generated_boards = primary_terminal;
    options.keep_only_success_secondary_generated_boards = secondary_terminal;
    options.collect_mutable_output_stats = false;
    cleanup_temp_file(output_path);
    cleanup_temp_file(layer_archive_path(output_path));
    cleanup_temp_file(layer_cell_compressed_path(output_path));
    auto file_writer = open_position_layer_artifact_writer(args, output_path);
    BCMemoryMirrorWritableFile mirror_writer(std::move(file_writer));
    BC::BCResidentGenerationPairResult pair =
        BC::generate_resident_position_layer_pair_with_mutable_carry_to_file(
        lut,
        primary_layout,
        *current_reader,
        std::move(carry_to_primary),
        secondary_layout,
        mirror_writer,
        nullptr,
        options
    );
    mirror_writer.flush();
    current_reader = nullptr;
    current_file.reset();

    if (pair.secondary_carry) {
        next_carry_layer_sum = static_cast<uint32_t>(pair.secondary_carry->layout().layer_sum());
        next_carry = std::move(pair.secondary_carry);
    } else {
        next_carry_layer_sum = 0U;
        next_carry.reset();
    }

    result.resident_mirror_seconds = mirror_writer.mirror_seconds();
    pair.primary.position_bytes = std::move(mirror_writer).take_mirror();

    result.logical_size = pair.primary.target_position_file_logical_bytes;
    result.output_rows = pair.primary.output_success_rows;
    result.effective_threads = pair.primary.effective_threads;
    result.retries = std::max(pair.primary.generation_retries, pair.secondary.generation_retries);
    result.total_seconds = now_seconds() - total_begin;
    result.target_modulus = 0U;
    result.dynamic_bucket_slots_used =
        std::max(pair.primary.dynamic_bucket_slots_used, pair.secondary.dynamic_bucket_slots_used);
    result.dynamic_bitmap_words_allocated =
        std::max(pair.primary.dynamic_bitmap_words_allocated, pair.secondary.dynamic_bitmap_words_allocated);
    fill_writer_stats_from_resident_result(result.writer_stats, pair.primary);
    if (!pair.primary.position_bytes.empty()) {
        const BC::BCPositionHeader header = BC::bc_read_header(pair.primary.position_bytes);
        result.output_bucket_count = header.bucket_meta_bytes / BC::kBCPositionBucketEntryBytes;
        result.output_rank_payload_bytes = header.rank_payload_bytes;
    }
    result.resident_prepare_seconds = pair.primary.prepare_seconds;
    result.resident_work_seconds = pair.primary.work_seconds;
    result.resident_compute_seconds = pair.primary.compute_seconds;
    result.resident_cleanup_seconds = pair.primary.cleanup_seconds;
    result.resident_finalize_seconds = pair.primary.finalize_seconds;
    result.resident_write_seconds = pair.primary.write_seconds;
    result.resident_total_seconds = pair.primary.total_seconds;
    const double reader_open_begin = now_seconds();
    result.resident_memory_layer =
        std::make_shared<BC::BCPositionLayerReader>(std::move(pair.primary.position_bytes), lut);
    result.resident_reader_open_seconds = now_seconds() - reader_open_begin;

    const uint64_t current_bytes = current_memory_layer == nullptr
        ? (current_layer.physical_size != 0U ? current_layer.physical_size : current_layer.logical_size)
        : 0U;
    result.stats.source_bytes_read = current_bytes;
    result.stats.source_backend_read_ops = current_memory_layer == nullptr ? 1U : 0U;
    result.stats.source_backend_read_bytes = current_bytes;
    result.stats.source_load_seconds = source_load_seconds + pair.primary.source_position_load_seconds;
    result.stats.parallel_seconds = pair.shared_generation_seconds;
    result.stats.finalize_seconds = pair.primary.finalize_seconds;
    result.stats.write_seconds = pair.primary.write_seconds;
    result.stats.generation_seconds = result.total_seconds;
    result.stats.target_cells_created =
        primary_layout.cell_count() +
        (secondary_layout == nullptr
             ? 0U
             : secondary_layout->cell_count());
    result.stats.target_cells_finalized = primary_layout.cell_count();
    result.stats.family_builder_hash_grows = result.retries;
    result.stats.active_builder_bytes_peak =
        (pair.primary.dynamic_bitmap_words_allocated + pair.secondary.dynamic_bitmap_words_allocated) *
        static_cast<uint64_t>(sizeof(uint64_t));
    return result;
}

[[nodiscard]] FamilyLayerResult generate_resident_mutable_carry_layer_in_memory(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout &current_layout,
    const BCPositionCellLayout *secondary_layout,
    const LayerFile &current_layer,
    const BC::BCPositionLayerReader *current_memory_layer,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_to_primary,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    double dynamic_reserve_factor,
    bool primary_terminal,
    bool secondary_terminal,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> &next_carry,
    uint32_t &next_carry_layer_sum
) {
    (void)possible_8tile_sums;
    FamilyLayerResult result;
    const double total_begin = now_seconds();
    double source_load_seconds = 0.0;
    std::optional<BC::BCPositionFileReader> current_file;
    const BC::BCPositionLayerReader *current_reader = current_memory_layer;
    if (current_reader == nullptr) {
        const double source_load_begin = now_seconds();
        current_file.emplace(open_position_file_reader(args, current_layer, lut));
        source_load_seconds += now_seconds() - source_load_begin;
        current_reader = &current_file->layer();
    }
    if (!current_layout.equivalent_to_axis(current_reader->axis())) {
        throw std::runtime_error("resident route current layer was not materialized to requested cell layout");
    }

    BC::BCResidentGenerationOptions options = resident_options_from_args(
        args,
        tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        kResidentRoutePendingBuffer
    );
    options.dynamic_reserve_factor = dynamic_reserve_factor;
    options.keep_only_success_generated_boards = primary_terminal;
    options.keep_only_success_secondary_generated_boards = secondary_terminal;
    options.collect_mutable_output_stats = false;

    BC::BCResidentGenerationPairResult pair =
        BC::generate_resident_position_layer_pair_with_mutable_carry(
            lut,
            primary_layout,
            *current_reader,
            std::move(carry_to_primary),
            secondary_layout,
            options
        );
    current_reader = nullptr;
    current_file.reset();

    if (pair.secondary_carry) {
        next_carry_layer_sum = static_cast<uint32_t>(pair.secondary_carry->layout().layer_sum());
        next_carry = std::move(pair.secondary_carry);
    } else {
        next_carry_layer_sum = 0U;
        next_carry.reset();
    }

    result.logical_size = pair.primary.target_position_file_logical_bytes;
    result.output_rows = pair.primary.output_success_rows;
    result.effective_threads = pair.primary.effective_threads;
    result.retries = std::max(pair.primary.generation_retries, pair.secondary.generation_retries);
    result.target_modulus = 0U;
    result.dynamic_bucket_slots_used =
        std::max(pair.primary.dynamic_bucket_slots_used, pair.secondary.dynamic_bucket_slots_used);
    result.dynamic_bitmap_words_allocated =
        std::max(pair.primary.dynamic_bitmap_words_allocated, pair.secondary.dynamic_bitmap_words_allocated);
    fill_writer_stats_from_resident_result(result.writer_stats, pair.primary);

    std::vector<uint8_t> position_bytes = std::move(pair.primary.position_bytes);
    if (!position_bytes.empty()) {
        const BC::BCPositionHeader header = BC::bc_read_header(position_bytes);
        result.output_bucket_count = header.bucket_meta_bytes / BC::kBCPositionBucketEntryBytes;
        result.output_rank_payload_bytes = header.rank_payload_bytes;
    }

    result.resident_prepare_seconds = pair.primary.prepare_seconds;
    result.resident_work_seconds = pair.primary.work_seconds;
    result.resident_compute_seconds = pair.primary.compute_seconds;
    result.resident_cleanup_seconds = pair.primary.cleanup_seconds;
    result.resident_finalize_seconds = pair.primary.finalize_seconds;
    result.resident_write_seconds = pair.primary.write_seconds;
    result.resident_total_seconds = pair.primary.total_seconds;
    const double reader_open_begin = now_seconds();
    result.resident_memory_layer =
        std::make_shared<BC::BCPositionLayerReader>(std::move(position_bytes), lut);
    result.resident_reader_open_seconds = now_seconds() - reader_open_begin;
    result.total_seconds = now_seconds() - total_begin;

    const uint64_t current_bytes = current_memory_layer == nullptr
        ? (current_layer.physical_size != 0U ? current_layer.physical_size : current_layer.logical_size)
        : 0U;
    result.stats.source_bytes_read = current_bytes;
    result.stats.source_backend_read_ops = current_memory_layer == nullptr ? 1U : 0U;
    result.stats.source_backend_read_bytes = current_bytes;
    result.stats.source_load_seconds = source_load_seconds + pair.primary.source_position_load_seconds;
    result.stats.parallel_seconds = pair.shared_generation_seconds;
    result.stats.finalize_seconds = pair.primary.finalize_seconds;
    result.stats.write_seconds = pair.primary.write_seconds;
    result.stats.generation_seconds = result.total_seconds;
    result.stats.target_cells_created =
        primary_layout.cell_count() +
        (secondary_layout == nullptr
             ? 0U
             : secondary_layout->cell_count());
    result.stats.target_cells_finalized = primary_layout.cell_count();
    result.stats.family_builder_hash_grows = result.retries;
    result.stats.active_builder_bytes_peak =
        (pair.primary.dynamic_bitmap_words_allocated + pair.secondary.dynamic_bitmap_words_allocated) *
        static_cast<uint64_t>(sizeof(uint64_t));
    return result;
}

[[nodiscard]] std::unique_ptr<BC::BCResidentGenerationMutableLayer> build_streaming_carry_to_primary(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout &source_layout,
    const LayerFile &source4_layer,
    const LayerFile &current_layer,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    double dynamic_reserve_factor,
    uint32_t route_default_pending_buffer,
    bool terminal
) {
    std::unique_ptr<BCPositionStreamingReader> source4_reader =
        open_position_reader(args, source4_layer, lut);
    if (!source_layout.equivalent_to_axis(source4_reader->axis())) {
        throw std::runtime_error("resident carry source layer was not materialized to requested cell layout");
    }
    const uint32_t source_cell_chunk_size =
        single_route_cell_chunk_size(source4_reader->cell_count());
    BC::BCResidentGenerationOptions options = resident_options_from_args(
        args,
        tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        route_default_pending_buffer
    );
    options.dynamic_reserve_factor = dynamic_reserve_factor;
    options.collect_mutable_output_stats = false;
    options.collect_dynamic_state_stats = false;
    options.keep_only_success_generated_boards = terminal;
    const uint64_t source4_bytes = layer_storage_size_bytes(source4_layer);
    const uint64_t current_bytes = layer_storage_size_bytes(current_layer);
    if (source4_bytes != 0U) {
        const double combined_ratio =
            static_cast<double>(source4_bytes + current_bytes) /
            static_cast<double>(source4_bytes);
        options.dynamic_reserve_factor =
            std::max(options.dynamic_reserve_factor, options.dynamic_reserve_factor * combined_ratio);
    }
    BC::BCResidentMutableGenerationResult carry =
        BC::generate_resident_mutable_layer_from_streaming_source(
            lut,
            primary_layout,
            BC::BCResidentStreamingGenerationSource{
                source4_reader.get(),
                2U,
                2U,
                source_cell_chunk_size
            },
            nullptr,
            options
        );
    return std::move(carry.mutable_layer);
}

[[nodiscard]] FamilyLayerResult generate_single_chunk_strict_layer_to_file(
    const Args &args,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout &current_layout,
    const BCPositionCellLayout *secondary_layout,
    const LayerFile &current_layer,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> carry_to_primary,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    double dynamic_reserve_factor,
    bool primary_terminal,
    bool secondary_terminal,
    const std::filesystem::path &output_path,
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> &next_carry,
    uint32_t &next_carry_layer_sum
) {
    FamilyLayerResult result;
    const double total_begin = now_seconds();
    const double source_load_begin = now_seconds();
    std::unique_ptr<BCPositionStreamingReader> current_reader =
        open_position_reader(args, current_layer, lut);
    const double source_load_seconds = now_seconds() - source_load_begin;
    if (!current_layout.equivalent_to_axis(current_reader->axis())) {
        throw std::runtime_error("single route current layer was not materialized to requested cell layout");
    }
    const uint32_t current_cell_chunk_size =
        single_route_cell_chunk_size(current_reader->cell_count());

    BC::BCResidentGenerationOptions options = resident_options_from_args(
        args,
        tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        kSingleRoutePendingBuffer
    );
    options.dynamic_reserve_factor = dynamic_reserve_factor;
    options.keep_only_success_generated_boards = primary_terminal;
    options.keep_only_success_secondary_generated_boards = secondary_terminal;
    options.collect_mutable_output_stats = false;
    cleanup_temp_file(output_path);
    cleanup_temp_file(layer_archive_path(output_path));
    cleanup_temp_file(layer_cell_compressed_path(output_path));
    std::unique_ptr<BC::BCWritableFile> writer =
        open_position_layer_artifact_writer(args, output_path);
    BC::BCSingleChunkGenerationStepResult step =
        BC::generate_single_chunk_position_layer_strict_to_file(
            lut,
            primary_layout,
            BC::BCSingleChunkGenerationSource{
                current_reader.get(),
                1U,
                1U,
                current_cell_chunk_size
            },
            std::move(carry_to_primary),
            secondary_layout,
            *writer,
            options
        );
    current_reader.reset();
    writer->flush();
    writer.reset();

    if (step.next_carry) {
        next_carry_layer_sum = static_cast<uint32_t>(step.next_carry->layout().layer_sum());
        next_carry = std::move(step.next_carry);
    } else {
        next_carry_layer_sum = 0U;
        next_carry.reset();
    }

    result.logical_size = step.primary.target_position_file_logical_bytes;
    result.output_rows = step.primary.output_success_rows;
    result.effective_threads = step.primary.effective_threads;
    result.retries = std::max(step.primary.generation_retries, step.carry.generation_retries);
    result.total_seconds = now_seconds() - total_begin;
    result.target_modulus = 0U;
    result.dynamic_bucket_slots_used =
        std::max(step.primary.dynamic_bucket_slots_used, step.carry.dynamic_bucket_slots_used);
    result.dynamic_bitmap_words_allocated =
        std::max(step.primary.dynamic_bitmap_words_allocated, step.carry.dynamic_bitmap_words_allocated);
    fill_writer_stats_from_resident_result(result.writer_stats, step.primary);

    const uint64_t current_bytes = current_layer.physical_size != 0U
        ? current_layer.physical_size
        : current_layer.logical_size;
    result.stats.source_bytes_read = current_bytes * (secondary_layout == nullptr ? 1U : 2U);
    result.stats.source_backend_read_ops = secondary_layout == nullptr ? 1U : 2U;
    result.stats.source_backend_read_bytes = result.stats.source_bytes_read;
    result.stats.source_load_seconds = source_load_seconds;
    result.stats.parallel_seconds = step.total_step_compute_seconds;
    result.stats.finalize_seconds = step.primary.finalize_seconds;
    result.stats.write_seconds = step.primary.write_seconds;
    result.stats.generation_seconds = result.total_seconds;
    result.stats.target_cells_created =
        primary_layout.cell_count() +
        (secondary_layout == nullptr
             ? 0U
             : secondary_layout->cell_count());
    result.stats.target_cells_finalized = primary_layout.cell_count();
    result.stats.family_builder_hash_grows = result.retries;
    result.stats.active_builder_bytes_peak =
        (step.primary.dynamic_bitmap_words_allocated + step.carry.dynamic_bitmap_words_allocated) *
        static_cast<uint64_t>(sizeof(uint64_t));
    result.resident_prepare_seconds = step.primary.prepare_seconds;
    result.resident_work_seconds = step.primary.work_seconds;
    result.resident_compute_seconds = step.primary.compute_seconds;
    result.resident_cleanup_seconds = step.primary.cleanup_seconds;
    result.resident_finalize_seconds = step.primary.finalize_seconds;
    result.resident_write_seconds = step.primary.write_seconds;
    result.resident_total_seconds = step.primary.total_seconds;
    return result;
}

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
    const std::filesystem::path bucket_spool_path = output_path.string() + ".bucket_spool.tmp";
    const std::filesystem::path blob_path = output_path.string() + ".family_blob.tmp";
    constexpr uint64_t kFamilyDirectBlobStagingBytes = 32ULL * 1024ULL * 1024ULL;
    constexpr uint64_t kFamilyBufferedBlobStagingBytes = 1ULL * 1024ULL * 1024ULL;
    constexpr uint64_t kFamilyPositionWriterStagingBytes = 1ULL * 1024ULL * 1024ULL;
    constexpr uint64_t kFamilyDirectPositionWriterStagingBytes = 32ULL * 1024ULL * 1024ULL;

    uint32_t reserve_buckets = args.family_reserve_buckets == 0U ? 1024U : args.family_reserve_buckets;
    uint32_t reserve_bitmap_words =
        args.family_reserve_bitmap_words == 0U ? 4096U : args.family_reserve_bitmap_words;
    constexpr uint32_t kMaxFamilyBuilderRetries = 6U;
    const double retry_begin = now_seconds();
    for (uint32_t attempt = 0U; attempt <= kMaxFamilyBuilderRetries; ++attempt) {
        cleanup_temp_file(rank_spool_path);
        cleanup_temp_file(bucket_spool_path);
        cleanup_temp_file(blob_path);
        cleanup_temp_file(output_path);
        cleanup_temp_file(layer_archive_path(output_path));
        cleanup_temp_file(layer_cell_compressed_path(output_path));
        cleanup_temp_file(layer_cell_compressed_path(output_path));
        try {
            FamilyLayerResult result;
            FamilyMemoryCheckpointContext memory_checkpoint_context;
            {
                const bool cell_compressed_output = args.compress_temp_files;
                std::unique_ptr<BC::BCWritableFile> final_writer;
                std::unique_ptr<BC::BCWritableFile> rank_spool_writer;
                std::unique_ptr<BC::BCCellCompressedPositionWriter> compressed_writer;
                if (cell_compressed_output) {
                    compressed_writer = std::make_unique<BC::BCCellCompressedPositionWriter>(
                        layer_cell_compressed_path(output_path));
                } else {
                    rank_spool_writer = open_family_position_spool_writer(args, rank_spool_path);
                    final_writer = open_family_position_writer(args, output_path);
                }
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
                const uint32_t target_modulus = target_axis.family_count();
                const BCFamilyTable source2_axis = make_axis(
                    target_axis.layer_sum() - 2U,
                    possible_8tile_sums,
                    target_modulus
                );
                std::optional<BC::BCPositionFamilyRemapReader> source2_remap(
                    std::in_place,
                    source2_reader,
                    source2_axis,
                    possible_8tile_sums
                );
                const BC::BCPositionFamilyRemapReader *source2_remap_ptr =
                    source2_remap->direct() ? nullptr : &(*source2_remap);
                std::optional<BC::BCFamilyStreamingGenerationSource> source4;
                std::optional<BC::BCPositionFamilyRemapReader> source4_remap;
                if (source4_reader != nullptr) {
                    const BCFamilyTable source4_axis = make_axis(
                        target_axis.layer_sum() - 4U,
                        possible_8tile_sums,
                        target_modulus
                    );
                    source4_remap.emplace(
                        *source4_reader,
                        source4_axis,
                        possible_8tile_sums
                    );
                    const BC::BCPositionFamilyRemapReader *source4_remap_ptr =
                        source4_remap->direct() ? nullptr : &(*source4_remap);
                    source4 = BC::BCFamilyStreamingGenerationSource{source4_reader, source4_remap_ptr, 2U, 2U};
                }
                const BC::BCFamilyStreamingGenerationSource source2{&source2_reader, source2_remap_ptr, 1U, 1U};
                BC::BCFamilyGenerationOptions options = family_options_from_args(
                    args,
                    target_axis.family_count(),
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
                writer_options.rank_first_direct_layout =
                    !cell_compressed_output && args.family_position_io == "direct-rank-first";
                writer_options.backend_preserves_unaligned_positioned_writes =
                    cell_compressed_output || !writer_options.rank_first_direct_layout;
                writer_options.staging_bytes = writer_options.rank_first_direct_layout
                    ? kFamilyDirectPositionWriterStagingBytes
                    : kFamilyPositionWriterStagingBytes;
                if (cell_compressed_output) {
                    position_writer.begin_cell_compressed_layer(
                        *compressed_writer,
                        target_axis,
                        writer_options);
                } else {
                    position_writer.begin_layer(*final_writer, *rank_spool_writer, target_axis, writer_options);
                }
                if (args.family_memory_checkpoints) {
                    memory_checkpoint_context.baseline_working_set_bytes =
                        process_current_working_set_bytes();
                    options.memory_checkpoint_callback = family_memory_checkpoint_callback;
                    options.memory_checkpoint_context = &memory_checkpoint_context;
                    options.memory_checkpoint_external_staging_bytes = blob_staging_bytes;
                }

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
                const double prefinish_flush_begin = now_seconds();
                position_writer.flush_pending_streams_for_reader();
                stats.write_seconds += now_seconds() - prefinish_flush_begin;
                const uint64_t spool_logical_size = writer_options.rank_first_direct_layout
                    ? position_writer.bucket_meta_bytes()
                    : position_writer.rank_payload_bytes();
                const double finish_begin = now_seconds();
                uint64_t logical_size = 0U;
                if (cell_compressed_output) {
                    (void)spool_logical_size;
                    logical_size = position_writer.finish_cell_compressed_layer();
                    compressed_writer.reset();
                } else {
                    std::unique_ptr<BC::BCReadableFile> rank_spool_reader =
                        open_family_position_spool_reader(args, rank_spool_path, spool_logical_size);
                    logical_size = position_writer.finish_layer(*rank_spool_reader);
                }
                stats.write_seconds += now_seconds() - finish_begin;

                result.stats = stats;
                result.writer_stats = position_writer.stats();
                result.blob_stats = blob.stats();
                result.logical_size = logical_size;
                result.output_rows = result.writer_stats.success_rows;
                result.output_bucket_count =
                    result.writer_stats.bucket_stage_write_bytes / BC::kBCPositionBucketEntryBytes;
                result.output_rank_payload_bytes = result.writer_stats.rank_stage_write_bytes;
                result.effective_threads = generation_effective_threads(args.num_threads);
                result.retries = attempt;
                result.total_seconds = now_seconds() - total_begin;
                result.stats.generation_seconds = result.total_seconds;
                result.target_modulus = target_axis.family_count();
                result.memory_checkpoints = std::move(memory_checkpoint_context.records);
            }
            cleanup_temp_file(rank_spool_path);
            cleanup_temp_file(bucket_spool_path);
            cleanup_temp_file(blob_path);
            return result;
        } catch (const BC::BCCellMutableBuilderOverflow &) {
            cleanup_temp_file(rank_spool_path);
            cleanup_temp_file(bucket_spool_path);
            cleanup_temp_file(blob_path);
            cleanup_temp_file(output_path);
            cleanup_temp_file(layer_archive_path(output_path));
            cleanup_temp_file(layer_cell_compressed_path(output_path));
            cleanup_temp_file(layer_cell_compressed_path(output_path));
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

[[nodiscard]] FamilyLayerResult generate_layer_to_file_for_route(
    const Args &args,
    const BCLut &lut,
    BC::BCFamilyGenerationRoute route,
    const BCFamilyTable &target_axis,
    const LayerFile *source4_layer,
    const LayerFile &source2_layer,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal,
    const std::filesystem::path &output_path
) {
    if (route == BC::BCFamilyGenerationRoute::Resident) {
        const BCPositionCellLayout target_layout =
            BCPositionCellLayout::from_modulo_axis(target_axis, target_axis.family_count());
        const BCPositionCellLayout source2_layout =
            make_resident_layout(target_axis.layer_sum() - 2U, possible_8tile_sums, target_axis.family_count());
        MaterializedLayerView source2_view = ensure_layer_cell_layout(
            args,
            lut,
            source2_layer,
            source2_layout,
            possible_8tile_sums,
            output_path.string() + ".resident_source2.tmp"
        );
        std::optional<MaterializedLayerView> source4_view;
        if (source4_layer != nullptr) {
            const BCPositionCellLayout source4_layout =
                make_resident_layout(target_axis.layer_sum() - 4U, possible_8tile_sums, target_axis.family_count());
            source4_view.emplace(
                ensure_layer_cell_layout(
                    args,
                    lut,
                    *source4_layer,
                    source4_layout,
                    possible_8tile_sums,
                    output_path.string() + ".resident_source4.tmp"
                )
            );
        }
        FamilyLayerResult result = generate_resident_layer_to_file(
            args,
            lut,
            target_layout,
            source4_view ? &source4_view->layer : nullptr,
            source2_view.layer,
            tile_sums,
            success_shifts,
            success_check_min_source_layer_sum,
            terminal,
            output_path
        );
        if (source2_view.temporary) {
            cleanup_temp_file(source2_view.layer.path);
        }
        if (source4_view && source4_view->temporary) {
            cleanup_temp_file(source4_view->layer.path);
        }
        return result;
    }
    if (route == BC::BCFamilyGenerationRoute::Single) {
        const BCPositionCellLayout target_layout =
            BCPositionCellLayout::from_modulo_axis(target_axis, target_axis.family_count());
        const BCPositionCellLayout source2_layout =
            make_resident_layout(target_axis.layer_sum() - 2U, possible_8tile_sums, target_axis.family_count());
        MaterializedLayerView source2_view = ensure_layer_cell_layout(
            args,
            lut,
            source2_layer,
            source2_layout,
            possible_8tile_sums,
            output_path.string() + ".single_source2.tmp"
        );
        std::optional<MaterializedLayerView> source4_view;
        if (source4_layer != nullptr) {
            const BCPositionCellLayout source4_layout =
                make_resident_layout(target_axis.layer_sum() - 4U, possible_8tile_sums, target_axis.family_count());
            source4_view.emplace(
                ensure_layer_cell_layout(
                    args,
                    lut,
                    *source4_layer,
                    source4_layout,
                    possible_8tile_sums,
                    output_path.string() + ".single_source4.tmp"
                )
            );
        }

        std::unique_ptr<BCPositionStreamingReader> source2_reader =
            open_position_reader(args, source2_view.layer, lut);
        const uint32_t source2_cell_chunk_size =
            single_route_cell_chunk_size(source2_reader->cell_count());
        std::unique_ptr<BCPositionStreamingReader> source4_reader;
        uint32_t source4_cell_chunk_size = source2_cell_chunk_size;
        std::vector<BC::BCSingleChunkGenerationSource> sources;
        sources.reserve(source4_view ? 2U : 1U);
        if (source4_view) {
            source4_reader = open_position_reader(args, source4_view->layer, lut);
            source4_cell_chunk_size =
                single_route_cell_chunk_size(source4_reader->cell_count());
            sources.push_back(BC::BCSingleChunkGenerationSource{
                source4_reader.get(),
                2U,
                2U,
                source4_cell_chunk_size
            });
        }
        sources.push_back(BC::BCSingleChunkGenerationSource{
            source2_reader.get(),
            1U,
            1U,
            source2_cell_chunk_size
        });

        BC::BCResidentGenerationOptions options = resident_options_from_args(
            args,
            tile_sums,
            success_shifts,
            success_check_min_source_layer_sum,
            kSingleRoutePendingBuffer
        );
        options.keep_only_success_generated_boards = terminal;
        cleanup_temp_file(output_path);
        cleanup_temp_file(layer_archive_path(output_path));
        cleanup_temp_file(layer_cell_compressed_path(output_path));
        std::unique_ptr<BC::BCWritableFile> writer =
            open_position_layer_artifact_writer(args, output_path);
        const double total_begin = now_seconds();
        BC::BCResidentGenerationResult single =
            BC::generate_single_chunk_position_layer_to_file(
                lut,
                target_layout,
                sources,
                *writer,
                options
            );
        writer->flush();
        writer.reset();

        FamilyLayerResult result;
        result.logical_size = single.target_position_file_logical_bytes;
        result.output_rows = single.output_success_rows;
        result.effective_threads = single.effective_threads;
        result.retries = single.generation_retries;
        result.total_seconds = now_seconds() - total_begin;
        result.target_modulus = 0U;
        result.dynamic_bucket_slots_used = single.dynamic_bucket_slots_used;
        result.dynamic_bitmap_words_allocated = single.dynamic_bitmap_words_allocated;
        fill_writer_stats_from_resident_result(result.writer_stats, single);
        result.stats.parallel_seconds = single.generation_seconds;
        result.stats.finalize_seconds = single.finalize_seconds;
        result.stats.write_seconds = single.write_seconds;
        result.stats.generation_seconds = result.total_seconds;
        result.stats.target_cells_created = target_layout.cell_count();
        result.stats.target_cells_finalized = target_layout.cell_count();
        result.stats.family_builder_hash_grows = single.generation_retries;
        result.stats.active_builder_bytes_peak =
            single.dynamic_bitmap_words_allocated * static_cast<uint64_t>(sizeof(uint64_t));
        result.resident_prepare_seconds = single.prepare_seconds;
        result.resident_work_seconds = single.work_seconds;
        result.resident_compute_seconds = single.compute_seconds;
        result.resident_cleanup_seconds = single.cleanup_seconds;
        result.resident_finalize_seconds = single.finalize_seconds;
        result.resident_write_seconds = single.write_seconds;
        result.resident_total_seconds = single.total_seconds;
        result.stats.source_bytes_read =
            (source2_view.layer.physical_size != 0U ? source2_view.layer.physical_size : source2_view.layer.logical_size) +
            (source4_view
                 ? (source4_view->layer.physical_size != 0U ? source4_view->layer.physical_size : source4_view->layer.logical_size)
                 : 0U);
        result.stats.source_backend_read_ops = source4_view ? 2U : 1U;
        result.stats.source_backend_read_bytes = result.stats.source_bytes_read;
        result.stats.source_load_seconds = single.source_position_load_seconds;
        if (source2_view.temporary) {
            cleanup_temp_file(source2_view.layer.path);
        }
        if (source4_view && source4_view->temporary) {
            cleanup_temp_file(source4_view->layer.path);
        }
        return result;
    }

    std::unique_ptr<BCPositionStreamingReader> source2_reader =
        open_position_reader(args, source2_layer, lut);
    std::unique_ptr<BCPositionStreamingReader> source4_reader;
    if (source4_layer != nullptr) {
        source4_reader = open_position_reader(args, *source4_layer, lut);
    }
    return generate_family_layer_to_file(
        args,
        lut,
        target_axis,
        source4_reader.get(),
        *source2_reader,
        possible_8tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        terminal,
        output_path
    );
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
        << "target_cells_created,target_cells_reloaded,"
        << "target_cells_dumped,target_cells_finalized,builder_hash_grows,builder_bitmap_grows,"
        << "source_load_gbps,blob_dump_gbps,blob_reload_gbps,blob_rw_gbps,"
        << "blob_backend_read_seconds,blob_backend_read_gbps,"
        << "blob_backend_write_seconds,blob_backend_write_gbps,target_writer_gbps,"
        << "target_backend_read_seconds,target_backend_read_gbps,"
        << "target_backend_write_seconds,target_backend_write_gbps,"
        << "route,target_modulus,available_memory_bytes,route_estimated_peak_bytes,route_budget_bytes,"
        << "resident_prepare_seconds,resident_work_seconds,resident_compute_seconds,resident_cleanup_seconds,"
        << "resident_finalize_detail_seconds,resident_write_detail_seconds,resident_total_detail_seconds,"
        << "resident_mirror_copy_seconds,resident_reader_open_seconds,"
        << "output_path\n";
}

void print_layer_row(
    std::ostream &out,
    uint32_t layer_sum,
    uint64_t input_live,
    uint64_t ex_input_live,
    uint64_t ex_primary_live,
    uint32_t ex_match,
    const FamilyLayerResult &result,
    const std::filesystem::path &output_path
) {
    const BC::BCFamilyGenerationStats &s = result.stats;
    const BC::BCFamilyPositionWriterStats &w = result.writer_stats;
    const uint64_t target_write_bytes =
        w.bucket_stage_write_bytes + w.rank_stage_write_bytes +
        w.metadata_write_bytes + w.rank_copy_write_bytes;
    const double blob_rw_seconds = s.dump_seconds + s.reload_seconds;
    const uint64_t blob_rw_bytes = s.blob_read_bytes + s.blob_bytes_written;
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
        << s.source_bytes_read << ',' << s.blob_read_bytes << ',' << s.blob_bytes_written << ','
        << s.blob_backend_read_ops << ',' << s.blob_backend_write_ops << ','
        << result.logical_size << ','
        << target_write_bytes << ','
        << (w.bucket_stage_flushes + w.rank_stage_flushes + w.metadata_write_ops + w.rank_copy_chunks) << ','
        << s.active_family_window_peak << ',' << s.target_active_cell_peak << ','
        << s.source_loaded_cell_peak << ',' << s.active_builder_bytes_peak << ','
        << s.thread_workspace_bytes_peak << ',' << process_peak_working_set_bytes() << ','
        << s.target_cells_created << ',' << s.target_cells_reloaded << ','
        << s.target_cells_dumped << ',' << s.target_cells_finalized << ','
        << s.family_builder_hash_grows << ',' << s.family_builder_bitmap_grows << ','
        << gbps(s.source_bytes_read, s.source_load_seconds) << ','
        << gbps(s.blob_bytes_written, s.dump_seconds) << ','
        << gbps(s.blob_read_bytes, s.reload_seconds) << ','
        << gbps(blob_rw_bytes, blob_rw_seconds) << ','
        << s.blob_backend_read_seconds << ','
        << gbps(s.blob_backend_read_bytes, s.blob_backend_read_seconds) << ','
        << s.blob_backend_write_seconds << ','
        << gbps(s.blob_backend_write_bytes, s.blob_backend_write_seconds) << ','
        << gbps(target_write_bytes, s.write_seconds) << ','
        << w.backend_read_seconds << ','
        << gbps(w.backend_read_bytes, w.backend_read_seconds) << ','
        << w.backend_write_seconds << ','
        << gbps(w.backend_write_bytes, w.backend_write_seconds) << ','
        << BC::bc_family_route_name(result.route) << ','
        << result.target_modulus << ','
        << result.available_memory_bytes << ','
        << result.route_estimated_peak_bytes << ','
        << result.route_budget_bytes << ','
        << result.resident_prepare_seconds << ','
        << result.resident_work_seconds << ','
        << result.resident_compute_seconds << ','
        << result.resident_cleanup_seconds << ','
        << result.resident_finalize_seconds << ','
        << result.resident_write_seconds << ','
        << result.resident_total_seconds << ','
        << result.resident_mirror_seconds << ','
        << result.resident_reader_open_seconds << ','
        << output_path.string()
        << '\n';
}

void accumulate(
    AggregateStats &agg,
    const FamilyLayerResult &result,
    uint64_t input_live
) {
    const BC::BCFamilyGenerationStats &s = result.stats;
    const BC::BCFamilyPositionWriterStats &w = result.writer_stats;
    ++agg.layers;
    agg.effective_threads = result.effective_threads;
    agg.input_live += input_live;
    agg.output_rows += result.output_rows;
    agg.source_read_bytes += s.source_bytes_read;
    agg.blob_read_bytes += s.blob_read_bytes;
    agg.blob_write_bytes += s.blob_bytes_written;
    agg.blob_read_ops += s.blob_backend_read_ops;
    agg.blob_write_ops += s.blob_backend_write_ops;
    agg.blob_backend_read_bytes += s.blob_backend_read_bytes;
    agg.blob_backend_write_bytes += s.blob_backend_write_bytes;
    agg.target_logical_bytes += result.logical_size;
    agg.target_write_bytes +=
        w.bucket_stage_write_bytes + w.rank_stage_write_bytes +
        w.metadata_write_bytes + w.rank_copy_write_bytes;
    agg.target_write_ops +=
        w.bucket_stage_flushes + w.rank_stage_flushes + w.metadata_write_ops + w.rank_copy_chunks;
    agg.target_backend_read_bytes += w.backend_read_bytes;
    agg.target_backend_write_bytes += w.backend_write_bytes;
    agg.source_load_seconds += s.source_load_seconds;
    agg.parallel_seconds += s.parallel_seconds;
    agg.dump_seconds += s.dump_seconds;
    agg.reload_seconds += s.reload_seconds;
    agg.finalize_seconds += s.finalize_seconds;
    agg.write_seconds += s.write_seconds;
    agg.blob_backend_read_seconds += s.blob_backend_read_seconds;
    agg.blob_backend_write_seconds += s.blob_backend_write_seconds;
    agg.target_backend_read_seconds += w.backend_read_seconds;
    agg.target_backend_write_seconds += w.backend_write_seconds;
    agg.generation_seconds += s.generation_seconds;
    agg.total_seconds += result.total_seconds;
    agg.resident_prepare_seconds += result.resident_prepare_seconds;
    agg.resident_work_seconds += result.resident_work_seconds;
    agg.resident_compute_seconds += result.resident_compute_seconds;
    agg.resident_cleanup_seconds += result.resident_cleanup_seconds;
    agg.resident_finalize_seconds += result.resident_finalize_seconds;
    agg.resident_write_seconds += result.resident_write_seconds;
    agg.resident_total_seconds += result.resident_total_seconds;
    agg.resident_mirror_seconds += result.resident_mirror_seconds;
    agg.resident_reader_open_seconds += result.resident_reader_open_seconds;
    agg.active_family_window_peak = std::max(agg.active_family_window_peak, s.active_family_window_peak);
    agg.target_active_cell_peak = std::max(agg.target_active_cell_peak, s.target_active_cell_peak);
    agg.source_loaded_cell_peak = std::max(agg.source_loaded_cell_peak, s.source_loaded_cell_peak);
    agg.active_builder_bytes_peak = std::max(agg.active_builder_bytes_peak, s.active_builder_bytes_peak);
    agg.thread_workspace_bytes_peak = std::max(agg.thread_workspace_bytes_peak, s.thread_workspace_bytes_peak);
    agg.process_peak_working_set_bytes =
        std::max(agg.process_peak_working_set_bytes, process_peak_working_set_bytes());
    agg.target_cells_created += s.target_cells_created;
    agg.target_cells_reloaded += s.target_cells_reloaded;
    agg.target_cells_dumped += s.target_cells_dumped;
    agg.target_cells_finalized += s.target_cells_finalized;
    agg.builder_hash_grows += s.family_builder_hash_grows;
    agg.builder_bitmap_grows += s.family_builder_bitmap_grows;
}

void print_summary_row(std::ostream &out, const char *label, const AggregateStats &agg) {
    const double blob_rw_seconds = agg.dump_seconds + agg.reload_seconds;
    const uint64_t blob_rw_bytes = agg.blob_read_bytes + agg.blob_write_bytes;
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
        << agg.target_cells_created << ',' << agg.target_cells_reloaded << ','
        << agg.target_cells_dumped << ',' << agg.target_cells_finalized << ','
        << agg.builder_hash_grows << ',' << agg.builder_bitmap_grows << ','
        << gbps(agg.source_read_bytes, agg.source_load_seconds) << ','
        << gbps(agg.blob_write_bytes, agg.dump_seconds) << ','
        << gbps(agg.blob_read_bytes, agg.reload_seconds) << ','
        << gbps(blob_rw_bytes, blob_rw_seconds) << ','
        << agg.blob_backend_read_seconds << ','
        << gbps(agg.blob_backend_read_bytes, agg.blob_backend_read_seconds) << ','
        << agg.blob_backend_write_seconds << ','
        << gbps(agg.blob_backend_write_bytes, agg.blob_backend_write_seconds) << ','
        << gbps(agg.target_write_bytes, agg.write_seconds) << ','
        << agg.target_backend_read_seconds << ','
        << gbps(agg.target_backend_read_bytes, agg.target_backend_read_seconds) << ','
        << agg.target_backend_write_seconds << ','
        << gbps(agg.target_backend_write_bytes, agg.target_backend_write_seconds)
        << ",,0,0,0,0,"
        << agg.resident_prepare_seconds << ','
        << agg.resident_work_seconds << ','
        << agg.resident_compute_seconds << ','
        << agg.resident_cleanup_seconds << ','
        << agg.resident_finalize_seconds << ','
        << agg.resident_write_seconds << ','
        << agg.resident_total_seconds << ','
        << agg.resident_mirror_seconds << ','
        << agg.resident_reader_open_seconds << ",\n";
}

[[nodiscard]] std::filesystem::path memory_checkpoint_csv_path(const Args &args) {
    if (!args.stats_csv.empty()) {
        return std::filesystem::path(args.stats_csv.string() + ".memory.csv");
    }
    return args.output_dir / "bc_family_memory_checkpoints.csv";
}

void write_memory_checkpoint_header(std::ostream &out) {
    out
        << "layer_sum,index,label,working_set,peak_working_set,baseline_working_set,"
        << "private_bytes,pagefile_bytes,peak_pagefile_bytes,accounted,residual,"
        << "baseline_adjusted_residual,active_builder,thread_workspace,"
        << "source_loaded_payload,source_loaded_allocated,source_reader_metadata,"
        << "active_index,range_work,pass_cache,store_static_metadata,store_allocated,"
        << "position_writer,finalized_payload,external_staging,released_builder_total,"
        << "last_release_batch_builder\n";
}

void write_memory_checkpoint_row(
    std::ostream &out,
    uint32_t layer_sum,
    size_t index,
    const BC::BCFamilyMemoryCheckpoint &c
) {
    out
        << layer_sum << ',' << index << ',' << c.label << ','
        << c.process_working_set_bytes << ',' << c.process_peak_working_set_bytes << ','
        << c.process_baseline_working_set_bytes << ',' << c.process_private_bytes << ','
        << c.process_pagefile_bytes << ',' << c.process_peak_pagefile_bytes << ','
        << c.accounted_bytes << ',' << c.residual_bytes << ','
        << c.baseline_adjusted_residual_bytes << ',' << c.active_builder_bytes << ','
        << c.thread_workspace_bytes << ',' << c.source_loaded_payload_bytes << ','
        << c.source_loaded_allocated_bytes << ',' << c.source_reader_metadata_bytes << ','
        << c.active_index_bytes << ',' << c.range_work_bytes << ','
        << c.pass_cache_bytes << ',' << c.store_static_metadata_bytes << ','
        << c.store_allocated_bytes << ',' << c.position_writer_bytes << ','
        << c.finalized_payload_bytes << ',' << c.external_staging_bytes << ','
        << c.released_builder_bytes_total << ',' << c.last_release_batch_builder_bytes
        << '\n';
}

void write_memory_checkpoint_rows(
    const Args &args,
    uint32_t layer_sum,
    const std::vector<BC::BCFamilyMemoryCheckpoint> &checkpoints
) {
    if (!args.family_memory_checkpoints || checkpoints.empty()) {
        return;
    }
    const std::filesystem::path path = memory_checkpoint_csv_path(args);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open memory checkpoint CSV: " + path.string());
    }
    write_memory_checkpoint_header(out);
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        write_memory_checkpoint_row(out, layer_sum, i, checkpoints[i]);
    }
}

struct MemoryCheckpointLayerRows {
    uint32_t layer_sum = 0U;
    std::vector<BC::BCFamilyMemoryCheckpoint> checkpoints;
};

void write_memory_checkpoint_rows(
    const Args &args,
    const std::vector<MemoryCheckpointLayerRows> &layers
) {
    if (!args.family_memory_checkpoints || layers.empty()) {
        return;
    }
    bool has_checkpoints = false;
    for (const MemoryCheckpointLayerRows &layer : layers) {
        if (!layer.checkpoints.empty()) {
            has_checkpoints = true;
            break;
        }
    }
    if (!has_checkpoints) {
        return;
    }

    const std::filesystem::path path = memory_checkpoint_csv_path(args);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open memory checkpoint CSV: " + path.string());
    }
    write_memory_checkpoint_header(out);
    for (const MemoryCheckpointLayerRows &layer : layers) {
        for (size_t i = 0; i < layer.checkpoints.size(); ++i) {
            write_memory_checkpoint_row(out, layer.layer_sum, i, layer.checkpoints[i]);
        }
    }
}

[[nodiscard]] bool bc_pattern_is_free(const std::string &pattern) {
    return pattern.rfind("free", 0U) == 0U;
}

int run_bc_chain(const Args &args, std::ostream &out) {
    std::filesystem::create_directories(args.output_dir);
    const std::array<uint32_t, 16U> tile_sums = free_semantic_tile_sums();
    const std::vector<uint8_t> legal_tiles = make_bc_legal_tiles(args.target_rank);
    const std::vector<BC::LayerSum> possible_8tile_sums =
        BC::build_possible_8tile_sums(legal_tiles, tile_sums);
    const BCLut lut = make_bc_lut(args.target_rank);
    const uint64_t seed_board =
        !args.seed_boards.empty() ? args.seed_boards.front() : load_pattern_seed_board(args.pattern);
    const BC::LayerSum seed_sum64 = board_semantic_sum(seed_board, tile_sums);
    if (seed_sum64 > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC generation seed sum exceeds uint32");
    }
    const uint32_t seed_sum = static_cast<uint32_t>(seed_sum64);
    const uint32_t forward_steps = ex_forward_steps(args);
    const uint32_t final_sum = seed_sum + forward_steps * 2U;
    const bool ex_terminal_mode = args.target_extra_override == 0U && final_sum >= seed_sum + 4U;
    const uint32_t final_primary_sum = ex_terminal_mode ? final_sum - 2U : final_sum;
    const uint32_t docheck_step = ex_docheck_step_for_target_rank(args.target_rank);
    const uint32_t default_success_check_min_source_layer_sum =
        seed_sum + 2U * (docheck_step + 1U);
    const uint32_t success_check_min_source_layer_sum =
        args.success_check_min_source_layer_sum_override != 0U
            ? args.success_check_min_source_layer_sum_override
            : default_success_check_min_source_layer_sum;
    const bool is_free_pattern = bc_pattern_is_free(args.pattern);
    const std::vector<uint8_t> success_shifts =
        (is_free_pattern || args.success_shifts.empty())
            ? all_board_success_shifts()
            : args.success_shifts;
    const ExExpectedStats ex_expected = load_ex_expected_stats(args.ex_stats_csv, seed_sum);

    std::vector<uint64_t> initial_boards;
    if (is_free_pattern) {
        initial_boards = generate_free_initial_boards(free_pattern_index(args.pattern));
    } else {
        if (args.seed_boards.empty()) {
            throw std::invalid_argument("BC non-free generation requires seed_boards");
        }
        initial_boards = args.seed_boards;
        initial_boards.erase(
            std::remove_if(
                initial_boards.begin(),
                initial_boards.end(),
                [&](uint64_t board) {
                    return board_semantic_sum(board, tile_sums) != seed_sum64;
                }),
            initial_boards.end());
        sort_unique_boards(initial_boards);
        if (initial_boards.empty()) {
            throw std::invalid_argument("BC non-free generation has no seed boards at seed sum");
        }
    }
    if (args.pattern == "free9") {
        check(initial_boards.size() == 21283U, "free9 initial board count must match EX");
    }

    const uint32_t final_primary_ordinal =
        layer_ordinal_for_sum(seed_sum, final_primary_sum);
    const std::map<uint32_t, std::filesystem::path> existing_layer_paths =
        discover_existing_layer_paths(args);
    uint32_t existing_prefix_count =
        contiguous_existing_layer_count(existing_layer_paths);
    if (existing_prefix_count > final_primary_ordinal + 1U) {
        existing_prefix_count = final_primary_ordinal + 1U;
    }

    std::map<uint32_t, LayerFile> layers;
    uint32_t first_generation_ordinal = 1U;
    std::optional<LayerFile> seed_layer_for_validation;
    auto load_existing_layer = [&](uint32_t ordinal) {
        const auto existing_it = existing_layer_paths.find(ordinal);
        if (existing_it == existing_layer_paths.end()) {
            throw std::runtime_error(
                "BC generation resume lost existing layer ordinal " +
                std::to_string(ordinal));
        }
        LayerFile layer = inspect_existing_layer_file(existing_it->second, lut);
        const uint32_t expected_layer_sum = seed_sum + ordinal * 2U;
        if (layer.layer_sum != expected_layer_sum) {
            throw std::runtime_error(
                "BC generation resume existing layer sum mismatch at ordinal " +
                std::to_string(ordinal));
        }
        return layer;
    };

    if (existing_prefix_count == 0U) {
        LayerFile seed_layer =
            write_initial_layer_file(
                args,
                seed_sum,
                lut,
                make_axis(seed_sum, possible_8tile_sums, args.family_modulus),
                initial_boards,
                possible_8tile_sums
            );
        seed_layer_for_validation = seed_layer;
        emit_seed_layer_metric(layer_ordinal_for_sum(seed_sum, seed_sum), seed_sum, seed_layer);
        layers.emplace(seed_sum, std::move(seed_layer));
        first_generation_ordinal = 1U;
    } else if (existing_prefix_count > final_primary_ordinal) {
        print_header(out);
        return 0;
    } else {
        const uint32_t highest_existing_ordinal = existing_prefix_count - 1U;
        const uint32_t first_source_ordinal =
            highest_existing_ordinal == 0U ? 0U : highest_existing_ordinal - 1U;
        for (uint32_t ordinal = first_source_ordinal;
             ordinal <= highest_existing_ordinal;
             ++ordinal) {
            LayerFile layer = load_existing_layer(ordinal);
            if (ordinal == 0U) {
                seed_layer_for_validation = layer;
            }
            layers.emplace(layer.layer_sum, std::move(layer));
        }
        first_generation_ordinal = highest_existing_ordinal + 1U;
    }

    if (seed_layer_for_validation) {
        if (is_free_pattern && seed_layer_for_validation->rows != initial_boards.size()) {
            throw std::runtime_error("free initial layer row count mismatch");
        }
        if (ex_expected.enabled) {
            const auto init_it = ex_expected.by_layer_sum.find(seed_sum);
            if (init_it == ex_expected.by_layer_sum.end()) {
                throw std::runtime_error("EX stats CSV is missing init row");
            }
            if (seed_layer_for_validation->rows != init_it->second.input_live ||
                seed_layer_for_validation->rows != init_it->second.primary_live) {
                throw std::runtime_error("BC seed layer rows do not match EX stats");
            }
        }
    }

    const uint64_t process_baseline_working_set = process_current_working_set_bytes();
    (void)process_baseline_working_set;
    AggregateStats aggregate;
    AggregateStats warm;
    std::vector<MemoryCheckpointLayerRows> memory_checkpoint_layers;
    const FamilyRouteScript route_script = load_family_route_script(args.family_route_script);
    FamilyRoutePlannerState route_state;
    route_state.previous_modulus = args.family_modulus;
    route_state.previous_route = BC::BCFamilyGenerationRoute::Resident;
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> resident_carry;
    uint32_t resident_carry_layer_sum = 0U;
    std::unique_ptr<BC::BCResidentGenerationMutableLayer> single_carry;
    uint32_t single_carry_layer_sum = 0U;
    std::vector<double> resident_reserve_need_history;
    std::vector<double> single_reserve_need_history;
    double resident_retry_guard_factor = 0.0;
    double single_retry_guard_factor = 0.0;
    std::map<uint32_t, std::shared_ptr<BC::BCPositionLayerReader>> resident_memory_layers;
    BCAsyncPositionWriteQueue async_position_writes;
    print_header(out);

    for (uint32_t layer_sum = seed_sum + first_generation_ordinal * 2U;
         layer_sum <= final_primary_sum;
         layer_sum += 2U) {
        g_current_generation_layer_sum.store(layer_sum, std::memory_order_relaxed);
        const uint32_t current_step = layer_ordinal_for_sum(seed_sum, layer_sum) - 1U;
        const bool terminal = ex_terminal_mode && layer_sum == final_primary_sum;
        const auto source2_it = layers.find(layer_sum - 2U);
        if (source2_it == layers.end()) {
            throw std::runtime_error("FamilyChain lost required +2 source layer");
        }
        const LayerFile *source4_layer = nullptr;
        std::map<uint32_t, LayerFile>::const_iterator source4_it = layers.end();
        if (layer_sum >= seed_sum + 4U) {
            source4_it = layers.find(layer_sum - 4U);
            if (source4_it == layers.end()) {
                throw std::runtime_error("FamilyChain lost required +4 source layer");
            }
            source4_layer = &source4_it->second;
        }
        BC::BCFamilyRouteDecision route_decision = decide_family_route_for_layer(
            args,
            route_script,
            route_state,
            layer_sum,
            source2_it->second,
            source4_layer
        );
        const BCFamilyTable target_axis =
            make_target_axis_for_route(
                route_decision.route,
                layer_sum,
                possible_8tile_sums,
                route_decision.target_modulus
            );

        double dynamic_reserve_factor = kBCDefaultReserveFactor;
        if (route_decision.route == BC::BCFamilyGenerationRoute::Resident) {
            dynamic_reserve_factor = bc_reserve_factor_for_step(
                current_step,
                resident_reserve_need_history,
                resident_retry_guard_factor,
                kBCResidentReserveHistoryWindow
            );
            resident_retry_guard_factor = 0.0;
        } else if (route_decision.route == BC::BCFamilyGenerationRoute::Single) {
            dynamic_reserve_factor = bc_reserve_factor_for_step(
                current_step,
                single_reserve_need_history,
                single_retry_guard_factor,
                kBCSingleReserveHistoryWindow
            );
            single_retry_guard_factor = 0.0;
        }

        const uint64_t input_live = source2_it->second.rows;
        const std::filesystem::path final_path = layer_path(args, seed_sum, layer_sum);
        FamilyLayerResult result;
        if (route_decision.route == BC::BCFamilyGenerationRoute::Resident) {
            single_carry.reset();
            single_carry_layer_sum = 0U;
            const bool secondary_terminal =
                ex_terminal_mode && layer_sum + 2U == final_primary_sum;
            const BCPositionCellLayout primary_layout =
                make_resident_layout(layer_sum, possible_8tile_sums, route_decision.target_modulus);
            const BCPositionCellLayout current_layout =
                make_resident_layout(layer_sum - 2U, possible_8tile_sums, route_decision.target_modulus);
            std::shared_ptr<BC::BCPositionLayerReader> current_memory_layer;
            const auto current_memory_it = resident_memory_layers.find(layer_sum - 2U);
            if (current_memory_it != resident_memory_layers.end() &&
                current_layout.equivalent_to_axis(current_memory_it->second->axis())) {
                current_memory_layer = current_memory_it->second;
            }
            MaterializedLayerView current_view;
            const LayerFile *current_layer_for_generation = &source2_it->second;
            if (!current_memory_layer) {
                async_position_writes.wait_for_layer(layer_sum - 2U);
                current_view = ensure_layer_cell_layout(
                    args,
                    lut,
                    source2_it->second,
                    current_layout,
                    possible_8tile_sums,
                    final_path.string() + ".resident_current.tmp"
                );
                current_layer_for_generation = &current_view.layer;
            }
            if ((!resident_carry || resident_carry_layer_sum != layer_sum) &&
                source4_layer != nullptr) {
                const BCPositionCellLayout source4_layout =
                    make_resident_layout(layer_sum - 4U, possible_8tile_sums, route_decision.target_modulus);
                async_position_writes.wait_for_layer(layer_sum - 4U);
                MaterializedLayerView source4_view = ensure_layer_cell_layout(
                    args,
                    lut,
                    *source4_layer,
                    source4_layout,
                    possible_8tile_sums,
                    final_path.string() + ".resident_source4.tmp"
                );
                resident_carry = build_streaming_carry_to_primary(
                    args,
                    lut,
                    primary_layout,
                    source4_layout,
                    source4_view.layer,
                    *current_layer_for_generation,
                    tile_sums,
                    possible_8tile_sums,
                    success_shifts,
                    success_check_min_source_layer_sum,
                    dynamic_reserve_factor,
                    kResidentRoutePendingBuffer,
                    terminal
                );
                if (source4_view.temporary) {
                    cleanup_temp_file(source4_view.layer.path);
                }
                resident_carry_layer_sum = layer_sum;
            }
            std::optional<BCPositionCellLayout> secondary_layout;
            if (layer_sum + 2U <= final_primary_sum) {
                secondary_layout.emplace(
                    make_resident_layout(layer_sum + 2U, possible_8tile_sums, route_decision.target_modulus)
                );
            }
            result = generate_resident_mutable_carry_layer_in_memory(
                args,
                lut,
                primary_layout,
                current_layout,
                secondary_layout ? &(*secondary_layout) : nullptr,
                *current_layer_for_generation,
                current_memory_layer.get(),
                std::move(resident_carry),
                tile_sums,
                possible_8tile_sums,
                success_shifts,
                success_check_min_source_layer_sum,
                dynamic_reserve_factor,
                terminal,
                secondary_terminal,
                resident_carry,
                resident_carry_layer_sum
            );
            async_position_writes.enqueue(
                args,
                layer_sum,
                final_path,
                result.resident_memory_layer
            );
            if (!current_memory_layer && current_view.temporary) {
                cleanup_temp_file(current_view.layer.path);
            }
        } else if (route_decision.route == BC::BCFamilyGenerationRoute::Single) {
            resident_carry.reset();
            resident_carry_layer_sum = 0U;
            const bool secondary_terminal =
                ex_terminal_mode && layer_sum + 2U == final_primary_sum;
            const BCPositionCellLayout primary_layout =
                make_resident_layout(layer_sum, possible_8tile_sums, route_decision.target_modulus);
            const BCPositionCellLayout current_layout =
                make_resident_layout(layer_sum - 2U, possible_8tile_sums, route_decision.target_modulus);
            async_position_writes.wait_for_layer(layer_sum - 2U);
            MaterializedLayerView current_view = ensure_layer_cell_layout(
                args,
                lut,
                source2_it->second,
                current_layout,
                possible_8tile_sums,
                final_path.string() + ".single_current.tmp"
            );
            if ((!single_carry || single_carry_layer_sum != layer_sum) &&
                source4_layer != nullptr) {
                const BCPositionCellLayout source4_layout =
                    make_resident_layout(layer_sum - 4U, possible_8tile_sums, route_decision.target_modulus);
                async_position_writes.wait_for_layer(layer_sum - 4U);
                MaterializedLayerView source4_view = ensure_layer_cell_layout(
                    args,
                    lut,
                    *source4_layer,
                    source4_layout,
                    possible_8tile_sums,
                    final_path.string() + ".single_source4.tmp"
                );
                single_carry = build_streaming_carry_to_primary(
                    args,
                    lut,
                    primary_layout,
                    source4_layout,
                    source4_view.layer,
                    current_view.layer,
                    tile_sums,
                    possible_8tile_sums,
                    success_shifts,
                    success_check_min_source_layer_sum,
                    dynamic_reserve_factor,
                    kSingleRoutePendingBuffer,
                    terminal
                );
                if (source4_view.temporary) {
                    cleanup_temp_file(source4_view.layer.path);
                }
                single_carry_layer_sum = layer_sum;
            }
            std::optional<BCPositionCellLayout> secondary_layout;
            if (layer_sum + 2U <= final_primary_sum) {
                secondary_layout.emplace(
                    make_resident_layout(layer_sum + 2U, possible_8tile_sums, route_decision.target_modulus)
                );
            }
            result = generate_single_chunk_strict_layer_to_file(
                args,
                lut,
                primary_layout,
                current_layout,
                secondary_layout ? &(*secondary_layout) : nullptr,
                current_view.layer,
                std::move(single_carry),
                tile_sums,
                possible_8tile_sums,
                success_shifts,
                success_check_min_source_layer_sum,
                dynamic_reserve_factor,
                terminal,
                secondary_terminal,
                final_path,
                single_carry,
                single_carry_layer_sum
            );
            if (current_view.temporary) {
                cleanup_temp_file(current_view.layer.path);
            }
        } else {
            resident_carry.reset();
            resident_carry_layer_sum = 0U;
            single_carry.reset();
            single_carry_layer_sum = 0U;
            async_position_writes.wait_for_layer(layer_sum - 2U);
            if (source4_layer != nullptr) {
                async_position_writes.wait_for_layer(layer_sum - 4U);
            }
            result = generate_layer_to_file_for_route(
                args,
                lut,
                route_decision.route,
                target_axis,
                source4_layer,
                source2_it->second,
                tile_sums,
                possible_8tile_sums,
                success_shifts,
                success_check_min_source_layer_sum,
                terminal,
                final_path
            );
        }
        apply_route_decision_to_result(result, route_decision);

        if (args.output_inspect) {
            async_position_writes.wait_for_layer(layer_sum);
        }
        const std::filesystem::path final_artifact_path =
            layer_artifact_path(args, final_path, route_decision.route);
        LayerFile generated_layer =
            args.output_inspect
                ? inspect_layer_file(args, layer_sum, final_artifact_path, result.logical_size, lut)
                : layer_file_from_known_metadata(
                      args,
                      layer_sum,
                      final_artifact_path,
                      result.logical_size,
                      result.output_rows,
                      result.output_bucket_count,
                      result.output_rank_payload_bytes);
        result.output_rows = generated_layer.rows;
        if (route_decision.route == BC::BCFamilyGenerationRoute::Resident) {
            const double observed_reserve_need =
                bc_observed_reserve_need(source2_it->second, result);
            if (observed_reserve_need > 0.0) {
                resident_reserve_need_history.push_back(observed_reserve_need);
            }
            if (result.retries != 0U) {
                resident_retry_guard_factor =
                    std::min(kBCMaxReserveFactor, dynamic_reserve_factor * kBCLearnedReserveRetryGuard);
            }
        } else if (route_decision.route == BC::BCFamilyGenerationRoute::Single) {
            const double observed_reserve_need =
                bc_observed_reserve_need(source2_it->second, result);
            if (observed_reserve_need > 0.0) {
                single_reserve_need_history.push_back(observed_reserve_need);
            }
            if (result.retries != 0U) {
                single_retry_guard_factor =
                    std::min(kBCMaxReserveFactor, dynamic_reserve_factor * kBCLearnedReserveRetryGuard);
            }
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
            result,
            final_artifact_path
        );
        out.flush();
        emit_generation_layer_metric(
            layer_ordinal_for_sum(seed_sum, layer_sum),
            layer_sum,
            input_live,
            result,
            final_artifact_path
        );
        accumulate(aggregate, result, input_live);
        if (layer_sum >= seed_sum + args.warmup_extra) {
            accumulate(warm, result, input_live);
        }
        if (args.family_memory_checkpoints && !result.memory_checkpoints.empty()) {
            MemoryCheckpointLayerRows checkpoint_rows;
            checkpoint_rows.layer_sum = layer_sum;
            checkpoint_rows.checkpoints = std::move(result.memory_checkpoints);
            memory_checkpoint_layers.push_back(std::move(checkpoint_rows));
        }
        resident_memory_layers.clear();
        if (result.resident_memory_layer) {
            resident_memory_layers[layer_sum] = result.resident_memory_layer;
        }
        layers[layer_sum] = std::move(generated_layer);
    }

    async_position_writes.wait_all();
    print_summary_row(out, "total", aggregate);
    if (warm.layers != 0U) {
        print_summary_row(out, "warm", warm);
    }
    write_memory_checkpoint_rows(args, memory_checkpoint_layers);
    return 0;
}

} // namespace

namespace BC {

BCFamilyGenerationRunResult bc_family_generation_full_run(
    const BCFamilyGenerationRunOptions &options,
    const BCFamilyGenerationLayerCallback &callback
) {
    if (options.target_rank >= 31U) {
        throw std::invalid_argument("BC family generation target_rank is too large");
    }
    if (options.batch_size == 0U ||
        options.family_work_schedule_chunk == 0U ||
        options.family_source_words_per_item == 0U ||
        options.family_modulus == 0U ||
        options.direct_queue_depth == 0U) {
        throw std::invalid_argument("BC family generation numeric options must be non-zero");
    }
    if (options.family_modulus > BC::kBCMaxFamilyModulusForPackedCellId) {
        throw std::invalid_argument("BC family generation family_modulus is outside 1..256");
    }

    Args args;
    args.pattern = options.pattern;
    args.target_rank = options.target_rank;
    args.extra_steps = options.extra_steps;
    args.seed_boards = options.seed_boards;
    args.pattern_masks = options.pattern_masks;
    args.success_shifts = options.success_shifts;
    args.canonical_symm_mode = options.canonical_symm_mode;
    args.success_check_min_source_layer_sum_override =
        options.success_check_min_source_layer_sum;
    args.num_threads = options.num_threads;
    args.batch_size = options.batch_size;
    args.pending_buffer = options.pending_buffer;
    args.family_work_schedule_chunk = options.family_work_schedule_chunk;
    args.family_source_words_per_item = options.family_source_words_per_item;
    args.family_reserve_buckets = options.family_reserve_buckets;
    args.family_reserve_bitmap_words = options.family_reserve_bitmap_words;
    args.warmup_extra = options.warmup_extra;
    args.verify_layer_rows = options.verify_layer_rows;
    args.output_inspect = options.output_inspect;
    args.family_blob = options.family_blob;
    args.family_position_io = options.family_position_io;
    args.family_source_io = options.family_source_io;
    args.family_blob_checksum = options.family_blob_checksum;
    args.family_memory_checkpoints = options.family_memory_checkpoints;
    args.compress_temp_files = options.compress_temp_files;
    args.family_modulus = options.family_modulus;
    args.family_route = options.family_route;
    args.direct_queue_depth = options.direct_queue_depth;
    args.output_dir = options.output_dir;
    args.output_dirs = options.output_dirs;
    args.stats_csv = options.stats_csv;

    BCFamilyGenerationRunResult result;
    BCFamilyGenerationRunResult *previous_result = g_run_result;
    const BCFamilyGenerationLayerCallback *previous_callback = g_layer_callback;
    g_run_result = &result;
    g_layer_callback = &callback;
    struct CallbackGuard {
        BCFamilyGenerationRunResult *previous_result = nullptr;
        const BCFamilyGenerationLayerCallback *previous_callback = nullptr;
        ~CallbackGuard() {
            g_run_result = previous_result;
            g_layer_callback = previous_callback;
        }
    } guard{previous_result, previous_callback};

    configure_global_threads(args.num_threads);
    if (!args.stats_csv.empty()) {
        if (!args.stats_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.stats_csv.parent_path());
        }
        std::ofstream out(args.stats_csv);
        if (!out) {
            throw std::runtime_error("failed to open BC family generation stats CSV");
        }
        run_bc_chain(args, out);
    } else {
        std::ostringstream sink;
        run_bc_chain(args, sink);
    }
    result.completed = true;
    return result;
}

} // namespace BC
