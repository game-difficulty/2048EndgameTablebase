#include "BCBoardOps.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCDirectFileIO.h"
#include "BCFamilyGeneration.h"
#include "BCFamilyRoutePlanner.h"
#include "BCFileIO.h"
#include "BCGenerationBlobIO.h"
#include "BCPositionFamilyRemapReader.h"
#include "BCPositionFile.h"
#include "BCResidentGeneration.h"
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

[[nodiscard]] double gbps(uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1.0e9 : 0.0;
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
    std::string family_blob = "direct";
    std::string family_position_io = "direct-rank-first";
    std::string family_source_io = "direct-auto";
    bool family_blob_checksum = false;
    bool family_memory_checkpoints = false;
    uint32_t family_modulus = 29U;
    BC::BCFamilyGenerationRoute family_route = BC::BCFamilyGenerationRoute::Auto;
    std::filesystem::path family_route_script;
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

[[nodiscard]] int bench_effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
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
    BC::BCFamilyGenerationRoute previous_route = BC::BCFamilyGenerationRoute::Family;
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
    if (requested == BC::BCFamilyGenerationRoute::Family &&
        (script_entry == nullptr || script_entry->target_modulus == 0U)) {
        decision.target_modulus = args.family_modulus;
        decision.family_estimated_peak_bytes =
            BC::bc_family_estimate_for_modulus(
                std::max<uint64_t>(inputs.source2_size, inputs.source4_size),
                decision.target_modulus
            );
        decision.route_estimated_peak_bytes = decision.family_estimated_peak_bytes;
    }
    if (script_entry != nullptr && script_entry->target_modulus != 0U &&
        decision.route == BC::BCFamilyGenerationRoute::Family) {
        if (!BC::bc_family_route_is_supported_prime(script_entry->target_modulus)) {
            throw std::invalid_argument("--family-route-script target_modulus must be a supported prime in 13..293");
        }
        decision.target_modulus = script_entry->target_modulus;
        decision.family_estimated_peak_bytes =
            BC::bc_family_estimate_for_modulus(
                std::max<uint64_t>(inputs.source2_size, inputs.source4_size),
                decision.target_modulus
            );
        if (decision.route == BC::BCFamilyGenerationRoute::Family) {
            decision.route_estimated_peak_bytes = decision.family_estimated_peak_bytes;
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

[[nodiscard]] std::filesystem::path layer_path(
    const Args &args,
    uint32_t seed_sum,
    uint32_t layer_sum
) {
    return args.output_dir /
        (layer_file_prefix(args) + "_" +
         std::to_string(layer_ordinal_for_sum(seed_sum, layer_sum)) + ".bcpos");
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

[[nodiscard]] BCFamilyTable make_resident_axis(
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums
) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free resident benchmark requires even layer sums");
    }
    return BC::build_family_axis_for_layer(layer_sum, 2U, possible_8tile_sums);
}

[[nodiscard]] BCFamilyTable make_target_axis_for_route(
    BC::BCFamilyGenerationRoute route,
    BC::LayerSum layer_sum,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t family_modulus
) {
    switch (route) {
    case BC::BCFamilyGenerationRoute::Resident:
        return make_resident_axis(layer_sum, possible_8tile_sums);
    case BC::BCFamilyGenerationRoute::Single:
        return make_axis(layer_sum, possible_8tile_sums, 1U);
    case BC::BCFamilyGenerationRoute::Family:
    case BC::BCFamilyGenerationRoute::Auto:
        return make_axis(layer_sum, possible_8tile_sums, family_modulus);
    }
    throw std::invalid_argument("unknown BC family generation route");
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
    const Args &args,
    const LayerFile &layer,
    const BCLut &lut
) {
    std::unique_ptr<BC::BCReadableFile> file;
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
    auto reader = std::make_unique<BCPositionStreamingReader>(
        BCPositionStreamingReader::open_buffered(layer.path, lut)
    );
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
    BCPositionStreamingReader reader = BCPositionStreamingReader::open_buffered(path, lut);
    reader.set_validate_loaded_cells(false);
    LayerFile layer;
    layer.layer_sum = BC::checked_u32_size(
        static_cast<size_t>(reader.axis().layer_sum()),
        "existing layer sum exceeds uint32"
    );
    layer.path = path;
    layer.logical_size = position_header_logical_size(reader.header());
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
            args.family_modulus
        );
    const std::filesystem::path path = layer_path(args, seed_sum, axis.layer_sum());
    write_raw_bytes_to_file(path, bytes);
    if (!args.output_inspect) {
        return layer_file_without_inspect(axis.layer_sum(), path, bytes.size(), initial_boards.size());
    }
    return inspect_layer_file(axis.layer_sum(), path, bytes.size(), lut);
}

void cleanup_temp_file(const std::filesystem::path &path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
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
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    options.success_check_all_cells = true;
    options.keep_only_success_generated_boards = terminal;
    return options;
}

[[nodiscard]] BC::BCResidentGenerationOptions resident_options_from_args(
    const Args &args,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum
) {
    BC::BCResidentGenerationOptions options;
    options.num_threads = args.num_threads;
    options.canonical_batch_size = args.batch_size;
    if (args.pending_buffer != 0U) {
        options.pending_insert_buffer_size = args.pending_buffer;
    }
    options.family_tile_sum_values = &tile_sums;
    options.success_target_rank = static_cast<int>(args.target_rank);
    options.success_shifts = &success_shifts;
    options.success_check_min_source_layer_sum = success_check_min_source_layer_sum;
    return options;
}

struct FamilyLayerResult {
    BC::BCFamilyGenerationStats stats;
    BC::BCFamilyPositionWriterStats writer_stats;
    BC::BCGenerationBlobIOStats blob_stats;
    std::vector<BC::BCFamilyMemoryCheckpoint> memory_checkpoints;
    uint64_t logical_size = 0U;
    uint64_t output_rows = 0U;
    int effective_threads = 1;
    uint32_t retries = 0U;
    double total_seconds = 0.0;
    BC::BCFamilyGenerationRoute route = BC::BCFamilyGenerationRoute::Family;
    uint32_t target_modulus = 0U;
    uint64_t available_memory_bytes = 0U;
    uint64_t route_estimated_peak_bytes = 0U;
    uint64_t route_budget_bytes = 0U;
};

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
    const BCFamilyTable &target_axis,
    const LayerFile *source4_layer,
    const LayerFile &source2_layer,
    const std::array<uint32_t, 16U> &tile_sums,
    const std::vector<uint8_t> &success_shifts,
    BC::LayerSum success_check_min_source_layer_sum,
    bool terminal,
    const std::filesystem::path &output_path
) {
    if (terminal) {
        throw std::runtime_error("resident route does not support terminal keep-only-success generation");
    }

    FamilyLayerResult result;
    const double total_begin = now_seconds();
    const double source_load_begin = now_seconds();
    BC::BCPositionFileReader source2_file =
        BC::BCPositionFileReader::open_buffered(source2_layer.path, lut);
    std::optional<BC::BCPositionFileReader> source4_file;
    std::vector<BC::BCResidentGenerationSource> sources;
    sources.reserve(source4_layer == nullptr ? 1U : 2U);
    uint64_t source_bytes = source2_layer.physical_size != 0U
        ? source2_layer.physical_size
        : source2_layer.logical_size;
    if (source4_layer != nullptr) {
        source4_file.emplace(BC::BCPositionFileReader::open_buffered(source4_layer->path, lut));
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
        success_check_min_source_layer_sum
    );
    BC::BCResidentGenerationResult resident =
        BC::generate_resident_position_layer(lut, target_axis, sources, options);

    cleanup_temp_file(output_path);
    const double disk_write_begin = now_seconds();
    write_raw_bytes_to_file(output_path, resident.position_bytes);
    const double disk_write_seconds = now_seconds() - disk_write_begin;

    result.logical_size = static_cast<uint64_t>(resident.position_bytes.size());
    result.output_rows = resident.output_success_rows;
    result.effective_threads = resident.effective_threads;
    result.retries = resident.generation_retries;
    result.total_seconds = now_seconds() - total_begin;
    result.target_modulus = target_axis.family_count();
    result.writer_stats.success_rows = resident.output_success_rows;
    result.writer_stats.metadata_write_ops = result.logical_size == 0U ? 0U : 1U;
    result.writer_stats.metadata_write_bytes = result.logical_size;
    result.writer_stats.backend_write_ops = result.writer_stats.metadata_write_ops;
    result.writer_stats.backend_write_bytes = result.logical_size;
    result.writer_stats.backend_write_seconds = disk_write_seconds;
    result.writer_stats.logical_size = result.logical_size;

    result.stats.source_bytes_read = source_bytes;
    result.stats.source_backend_read_ops = source4_layer == nullptr ? 1U : 2U;
    result.stats.source_backend_read_bytes = source_bytes;
    result.stats.source_load_seconds = source_load_seconds;
    result.stats.parallel_seconds = resident.generation_seconds;
    result.stats.finalize_seconds = resident.finalize_seconds;
    result.stats.write_seconds = resident.write_seconds + disk_write_seconds;
    result.stats.generation_seconds = result.total_seconds;
    result.stats.target_cells_created = target_axis.family_count() * target_axis.family_count();
    result.stats.target_cells_finalized = result.stats.target_cells_created;
    result.stats.family_builder_hash_grows = resident.generation_retries;
    result.stats.active_builder_bytes_peak =
        resident.dynamic_bitmap_words_allocated * static_cast<uint64_t>(sizeof(uint64_t));
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
        cleanup_temp_file(blob_path);
        cleanup_temp_file(output_path);
        try {
            FamilyLayerResult result;
            FamilyMemoryCheckpointContext memory_checkpoint_context;
            {
                std::unique_ptr<BC::BCWritableFile> final_writer =
                    open_family_position_writer(args, output_path);
                std::unique_ptr<BC::BCWritableFile> rank_spool_writer =
                    open_family_position_spool_writer(args, rank_spool_path);
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
                    args.family_position_io == "direct-rank-first";
                writer_options.backend_preserves_unaligned_positioned_writes =
                    !writer_options.rank_first_direct_layout;
                writer_options.staging_bytes = writer_options.rank_first_direct_layout
                    ? kFamilyDirectPositionWriterStagingBytes
                    : kFamilyPositionWriterStagingBytes;
                position_writer.begin_layer(*final_writer, *rank_spool_writer, target_axis, writer_options);
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
                std::unique_ptr<BC::BCReadableFile> rank_spool_reader =
                    open_family_position_spool_reader(args, rank_spool_path, spool_logical_size);
                const double finish_begin = now_seconds();
                const uint64_t logical_size = position_writer.finish_layer(*rank_spool_reader);
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
                result.target_modulus = target_axis.family_count();
                result.memory_checkpoints = std::move(memory_checkpoint_context.records);
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
        return generate_resident_layer_to_file(
            args,
            lut,
            target_axis,
            source4_layer,
            source2_layer,
            tile_sums,
            success_shifts,
            success_check_min_source_layer_sum,
            terminal,
            output_path
        );
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
        << ",,0,0,0,0,\n";
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

    const FamilyRouteScript route_script = load_family_route_script(args.family_route_script);
    FamilyRoutePlannerState route_state;
    route_state.previous_modulus = args.family_modulus;
    route_state.previous_route = BC::BCFamilyGenerationRoute::Family;
    BC::BCFamilyRouteDecision route_decision = decide_family_route_for_layer(
        args,
        route_script,
        route_state,
        args.single_layer_sum,
        source2_layer,
        &source4_layer
    );

    const BCFamilyTable target_axis =
        make_target_axis_for_route(
            route_decision.route,
            args.single_layer_sum,
            possible_8tile_sums,
            route_decision.target_modulus
        );
    const std::filesystem::path final_path = layer_path(args, seed_sum, args.single_layer_sum);

    print_header(out);
    FamilyLayerResult result = generate_layer_to_file_for_route(
        args,
        lut,
        route_decision.route,
        target_axis,
        &source4_layer,
        source2_layer,
        tile_sums,
        possible_8tile_sums,
        success_shifts,
        success_check_min_source_layer_sum,
        false,
        final_path
    );
    apply_route_decision_to_result(result, route_decision);

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
        result,
        final_path
    );
    write_memory_checkpoint_rows(args, args.single_layer_sum, result.memory_checkpoints);

    AggregateStats aggregate;
    accumulate(aggregate, result, source2_layer.rows);
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
        } else if (key == "--family-position-io") {
            args.family_position_io = require_value("--family-position-io");
        } else if (key == "--family-source-io") {
            args.family_source_io = require_value("--family-source-io");
        } else if (key == "--family-blob-checksum") {
            args.family_blob_checksum = true;
        } else if (key == "--family-memory-checkpoints") {
            args.family_memory_checkpoints = true;
        } else if (key == "--family-modulus") {
            args.family_modulus = static_cast<uint32_t>(std::stoul(require_value("--family-modulus")));
        } else if (key == "--family-route") {
            args.family_route = BC::bc_parse_family_route(require_value("--family-route"));
        } else if (key == "--family-route-script") {
            args.family_route_script = require_value("--family-route-script");
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
    if (args.family_position_io != "buffered" &&
        args.family_position_io != "direct-rank-first") {
        throw std::invalid_argument("--family-position-io must be buffered or direct-rank-first");
    }
    if (args.family_source_io != "buffered" &&
        args.family_source_io != "direct" &&
        args.family_source_io != "direct-auto") {
        throw std::invalid_argument("--family-source-io must be buffered, direct, or direct-auto");
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
            seed_sum,
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
    std::vector<MemoryCheckpointLayerRows> memory_checkpoint_layers;
    const FamilyRouteScript route_script = load_family_route_script(args.family_route_script);
    FamilyRoutePlannerState route_state;
    route_state.previous_modulus = args.family_modulus;
    route_state.previous_route = BC::BCFamilyGenerationRoute::Family;
    print_header(out);

    for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_primary_sum; layer_sum += 2U) {
        g_current_bench_layer_sum.store(layer_sum, std::memory_order_relaxed);
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
        if (terminal && route_decision.route == BC::BCFamilyGenerationRoute::Resident) {
            route_decision.route = BC::BCFamilyGenerationRoute::Family;
            route_decision.route_estimated_peak_bytes = route_decision.family_estimated_peak_bytes;
            if (!BC::bc_family_route_is_supported_prime(route_decision.target_modulus)) {
                route_decision.target_modulus = args.family_modulus;
            }
        }
        const BCFamilyTable target_axis =
            make_target_axis_for_route(
                route_decision.route,
                layer_sum,
                possible_8tile_sums,
                route_decision.target_modulus
            );

        const uint64_t input_live = source2_it->second.rows;
        const std::filesystem::path final_path = layer_path(args, seed_sum, layer_sum);
        FamilyLayerResult result = generate_layer_to_file_for_route(
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
        apply_route_decision_to_result(result, route_decision);

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
            result,
            final_path
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
        layers[layer_sum] = std::move(generated_layer);
    }

    print_summary_row(out, "total", aggregate);
    if (warm.layers != 0U) {
        print_summary_row(out, "warm", warm);
    }
    write_memory_checkpoint_rows(args, memory_checkpoint_layers);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
#if defined(_WIN32)
    SetUnhandledExceptionFilter(bc_bench_unhandled_exception_filter);
#endif
    try {
        const Args args = parse_args(argc, argv);
        configure_global_threads(args.num_threads);
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
