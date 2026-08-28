#pragma once

#include "BCDirectFileIO.h"
#include "BCFamilyRoutePlanner.h"
#include "BCCellCompressedPositionFile.h"
#include "BCCompressedResult.h"
#include "BCFamilySolve.h"
#include "BCLoadedCellScanner.h"
#include "BCPositionScanner.h"
#include "BCResidentSolve.h"
#include "BCSingleChunkSolve.h"
#include "CompressionBridge.h"
#include "FormationRuntime.h"
#include "PathUtils.h"
#include "SymmetryUtils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

namespace BC {

struct BCFamilySolveRunOptions {
    std::filesystem::path generated_position_dir;
    std::filesystem::path solved_output_dir;
    std::filesystem::path archive_output_dir;
    std::vector<std::filesystem::path> generated_position_dirs;
    std::vector<std::filesystem::path> solved_output_dirs;
    std::vector<std::filesystem::path> archive_output_dirs;
    std::string prefix;
    uint32_t target_rank = 8U;
    int success_target_rank = -1;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    std::vector<uint64_t> pattern_masks;
    std::vector<uint8_t> success_shifts;
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t family_modulus = 17U;
    BCSolveRoute solve_route = BCSolveRoute::Auto;
    uint64_t available_memory_override_bytes = 0U;
    uint32_t future_reuse_max_families = 4U;
    uint64_t future_index_recycle_max_bytes = 0U;
    uint32_t source_words_per_item = 64U;
    uint32_t work_schedule_chunk = 1U;
    uint32_t cell_parallel_min_work_items = 4U;
    uint64_t final_pending_value_memory_cap_bytes = 0U;
    BCSuccessDTypeMode success_dtype = BCSuccessDTypeMode::UInt32;
    uint32_t direct_queue_depth = 16U;
    uint32_t direct_io_chunk_mib = 8U;
    std::optional<uint32_t> start_ordinal;
    std::optional<uint32_t> min_ordinal;
    bool direct_io = false;
    bool keep_direct_padding = false;
    bool compress = false;
    bool compress_temp_files = false;
    bool optimal_branch_only = false;
    bool resume_from_checkpoint = true;
    bool force_restart = false;
    double deletion_threshold = 0.0;
    double relative_deletion_threshold = 0.0;
    std::string deletion_threshold_signal_path;
};

struct BCFamilySolveRunLayerFile {
    uint32_t ordinal = 0U;
    std::filesystem::path path;
};

struct BCFamilySolveArchiveExactLayer {
    uint32_t ordinal = 0U;
    std::filesystem::path position_path;
    std::filesystem::path success_path;
};

struct BCFamilySolveRunLayerMetric {
    std::string kind = "solve";
    std::string solve_route = "";
    uint32_t ordinal = 0U;
    uint64_t layer_sum = 0U;
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t archive_live_rows = 0U;
    uint64_t threshold_pruned_rows = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t output_position_write_bytes = 0U;
    uint64_t output_success_write_bytes = 0U;
    uint64_t temp_compressed_bytes = 0U;
    uint64_t final_compress_bucket_blocks = 0U;
    uint64_t final_compress_value_blocks = 0U;
    uint64_t final_compress_success_values = 0U;
    uint64_t final_compress_bucket_raw_bytes = 0U;
    uint64_t final_compress_bucket_compressed_bytes = 0U;
    uint64_t final_compress_value_raw_bytes = 0U;
    uint64_t final_compress_value_compressed_bytes = 0U;
    uint64_t final_compress_output_bytes = 0U;
    uint64_t route_available_memory_bytes = 0U;
    uint64_t route_resident_required_bytes = 0U;
    uint64_t route_single_required_bytes = 0U;
    uint64_t route_required_bytes = 0U;
    double open_seconds = 0.0;
    double open_current_position_seconds = 0.0;
    double open_future2_position_seconds = 0.0;
    double open_future2_success_seconds = 0.0;
    double open_future4_position_seconds = 0.0;
    double open_future4_success_seconds = 0.0;
    double descriptor_rows_seconds = 0.0;
    double partition_seconds = 0.0;
    double writer_open_seconds = 0.0;
    double solve_call_seconds = 0.0;
    double archive_scan_seconds = 0.0;
    double archive_prune_write_seconds = 0.0;
    double final_compress_seconds = 0.0;
    double final_compress_read_seconds = 0.0;
    double final_compress_write_seconds = 0.0;
    double final_compress_worker_seconds = 0.0;
    double temp_compress_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double writer_close_seconds = 0.0;
    double post_resize_seconds = 0.0;
    double total_seconds = 0.0;
    BCFamilySolveStats family_stats;
    bool has_family_stats = false;
};

struct BCFamilySolveRunResult {
    std::vector<BCFamilySolveRunLayerMetric> layers;
    uint32_t min_ordinal = 0U;
    uint32_t max_ordinal = 0U;
    bool completed = false;
};

using BCFamilySolveLayerCallback =
    std::function<void(const BCFamilySolveRunLayerMetric &)>;

namespace detail {

[[nodiscard]] inline double bc_family_solve_runner_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline uint64_t bc_family_runner_available_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0U;
#else
    struct sysinfo info {};
    if (sysinfo(&info) == 0) {
        return static_cast<uint64_t>(info.freeram) * static_cast<uint64_t>(info.mem_unit);
    }
    return 0U;
#endif
}

[[nodiscard]] inline std::string bc_family_success_dtype_name(BCSuccessDTypeMode dtype) {
    switch (dtype) {
        case BCSuccessDTypeMode::UInt32: return "uint32";
        case BCSuccessDTypeMode::UInt64: return "uint64";
        case BCSuccessDTypeMode::Float32: return "float32";
        case BCSuccessDTypeMode::Float64: return "float64";
        case BCSuccessDTypeMode::OneMinusFloat32: return "1-float32";
        case BCSuccessDTypeMode::OneMinusFloat64: return "1-float64";
    }
    return "unknown";
}

[[nodiscard]] inline uint64_t bc_family_runner_success_value_rows(
    const BCSuccessStreamingReader &success
) {
    const uint32_t row_width = success.row_width();
    if (row_width == 0U) {
        throw std::runtime_error("BC family runner success row_width is zero");
    }
    return success.header().payload_bytes /
        static_cast<uint64_t>(success.value_size()) /
        static_cast<uint64_t>(row_width);
}

[[nodiscard]] inline std::vector<uint8_t> bc_family_runner_legal_tiles(
    uint32_t target_rank
) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("BC family runner target_rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] inline std::vector<uint8_t> bc_family_runner_success_shifts() {
    std::vector<uint8_t> shifts;
    shifts.reserve(16U);
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }
    return shifts;
}

[[nodiscard]] inline BCQuadrantWordSumTable bc_family_runner_word_sums(const BCLut &lut) {
    BCQuadrantWordSumTable sums(kBCQuadrantWordCount, 0U);
    for (uint32_t word = 0U; word < kBCQuadrantWordCount; ++word) {
        const BCWordDesc &desc = lut.word_desc(static_cast<uint16_t>(word));
        if (desc.valid) {
            sums[word] = desc.sum;
        }
    }
    return sums;
}

[[nodiscard]] inline bool bc_family_runner_board_has_target_rank(
    uint64_t board,
    int target_rank,
    const std::vector<uint8_t> &success_shifts
) {
    const uint64_t target = static_cast<uint64_t>(target_rank);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline std::filesystem::path bc_family_runner_position_path(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcpos");
}

[[nodiscard]] inline bool bc_family_runner_is_position_archive_path(
    const std::filesystem::path &path
) {
    const std::string name = path.filename().string();
    const std::string suffix = ".bcpos.7z";
    return name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[nodiscard]] inline bool bc_family_runner_is_cell_compressed_position_path(
    const std::filesystem::path &path
) {
    return bc_is_cell_compressed_position_path(path);
}

[[nodiscard]] inline std::filesystem::path bc_family_runner_success_path(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcsuc");
}

[[nodiscard]] inline std::filesystem::path bc_family_runner_checkpoint_path(
    const BCFamilySolveRunOptions &options
) {
    return options.solved_output_dir / (options.prefix + "family_checkpoint.csv");
}

[[nodiscard]] inline std::filesystem::path bc_family_runner_archive_dir(
    const BCFamilySolveRunOptions &options
) {
    return options.archive_output_dir.empty()
        ? options.solved_output_dir
        : options.archive_output_dir;
}

[[nodiscard]] inline bool bc_family_runner_same_path(
    const std::filesystem::path &lhs,
    const std::filesystem::path &rhs
) {
    if (lhs.empty() || rhs.empty()) {
        return false;
    }
    return std::filesystem::absolute(lhs).lexically_normal() ==
        std::filesystem::absolute(rhs).lexically_normal();
}

[[nodiscard]] inline std::vector<std::filesystem::path> bc_family_runner_unique_dirs(
    const std::filesystem::path &primary,
    const std::vector<std::filesystem::path> &extra
) {
    std::vector<std::filesystem::path> dirs;
    auto add = [&dirs](const std::filesystem::path &dir) {
        if (dir.empty()) {
            return;
        }
        for (const std::filesystem::path &existing : dirs) {
            if (bc_family_runner_same_path(existing, dir)) {
                return;
            }
        }
        dirs.push_back(dir);
    };
    add(primary);
    for (const std::filesystem::path &dir : extra) {
        add(dir);
    }
    return dirs;
}

[[nodiscard]] inline std::vector<std::filesystem::path> bc_family_runner_generated_dirs(
    const BCFamilySolveRunOptions &options
) {
    return bc_family_runner_unique_dirs(options.generated_position_dir, options.generated_position_dirs);
}

[[nodiscard]] inline std::vector<std::filesystem::path> bc_family_runner_solved_dirs(
    const BCFamilySolveRunOptions &options
) {
    return bc_family_runner_unique_dirs(options.solved_output_dir, options.solved_output_dirs);
}

[[nodiscard]] inline std::vector<std::filesystem::path> bc_family_runner_archive_dirs(
    const BCFamilySolveRunOptions &options
) {
    return bc_family_runner_unique_dirs(
        bc_family_runner_archive_dir(options),
        options.archive_output_dirs);
}

[[nodiscard]] inline std::map<uint32_t, BCFamilySolveRunLayerFile>
bc_family_runner_discover_layers_impl(
    const BCFamilySolveRunOptions &options,
    bool allow_missing_or_empty
) {
    const std::vector<std::filesystem::path> generated_dirs = bc_family_runner_generated_dirs(options);
    const bool has_any_dir = std::any_of(
        generated_dirs.begin(),
        generated_dirs.end(),
        [](const std::filesystem::path &dir) { return std::filesystem::is_directory(dir); });
    if (!has_any_dir) {
        if (allow_missing_or_empty) {
            return {};
        }
        throw std::runtime_error(
            "BC family runner generated position dir does not exist: " +
            options.generated_position_dir.string()
        );
    }
    const std::regex pattern("^" + options.prefix + "([0-9]+)\\.bcpos(c|\\.7z)?$");
    std::map<uint32_t, BCFamilySolveRunLayerFile> layers;
    for (const std::filesystem::path &dir : generated_dirs) {
        if (!std::filesystem::is_directory(dir)) {
            continue;
        }
        for (const std::filesystem::directory_entry &entry :
             std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            std::smatch match;
            if (!std::regex_match(name, match, pattern)) {
                continue;
            }
            const uint32_t ordinal = static_cast<uint32_t>(std::stoul(match[1].str()));
            const auto existing = layers.find(ordinal);
            const bool is_raw = !bc_family_runner_is_position_archive_path(entry.path()) &&
                !bc_family_runner_is_cell_compressed_position_path(entry.path());
            const bool replace_archive =
                bc_family_runner_is_cell_compressed_position_path(entry.path()) &&
                existing != layers.end() &&
                bc_family_runner_is_position_archive_path(existing->second.path);
            if (is_raw || replace_archive || existing == layers.end()) {
                layers[ordinal] = BCFamilySolveRunLayerFile{ordinal, entry.path()};
            }
        }
    }
    if (layers.empty()) {
        if (allow_missing_or_empty) {
            return {};
        }
        throw std::runtime_error(
            "BC family runner found no generated layers with prefix: " + options.prefix
        );
    }
    return layers;
}

[[nodiscard]] inline std::map<uint32_t, BCFamilySolveRunLayerFile>
bc_family_runner_discover_layers(const BCFamilySolveRunOptions &options) {
    return bc_family_runner_discover_layers_impl(options, false);
}

[[nodiscard]] inline std::map<uint32_t, BCFamilySolveRunLayerFile>
bc_family_runner_discover_layers_or_empty(const BCFamilySolveRunOptions &options) {
    return bc_family_runner_discover_layers_impl(options, true);
}

[[nodiscard]] inline std::map<uint32_t, BCFamilySolveArchiveExactLayer>
bc_family_runner_discover_archive_exact_layers(const BCFamilySolveRunOptions &options) {
    std::map<uint32_t, BCFamilySolveArchiveExactLayer> layers;
    const std::regex pattern("^" + options.prefix + "([0-9]+)\\.bcpos$");
    for (const std::filesystem::path &archive_dir : bc_family_runner_archive_dirs(options)) {
        if (archive_dir.empty() || !std::filesystem::is_directory(archive_dir)) {
            continue;
        }
        for (const std::filesystem::directory_entry &entry :
             std::filesystem::directory_iterator(archive_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            std::smatch match;
            if (!std::regex_match(name, match, pattern)) {
                continue;
            }
            const uint32_t ordinal = static_cast<uint32_t>(std::stoul(match[1].str()));
            const std::filesystem::path success_path =
                bc_family_runner_success_path(archive_dir, options.prefix, ordinal);
            std::error_code ec;
            if (!std::filesystem::exists(success_path, ec) || ec) {
                continue;
            }
            layers.emplace(
                ordinal,
                BCFamilySolveArchiveExactLayer{ordinal, entry.path(), success_path});
        }
    }
    return layers;
}

[[nodiscard]] inline std::map<uint32_t, std::filesystem::path>
bc_family_runner_discover_archive_compressed_layers(const BCFamilySolveRunOptions &options) {
    std::map<uint32_t, std::filesystem::path> layers;
    const std::regex pattern(
        "^" + options.prefix + "([0-9]+)\\.bccmp$");
    for (const std::filesystem::path &archive_dir : bc_family_runner_archive_dirs(options)) {
        if (archive_dir.empty() || !std::filesystem::is_directory(archive_dir)) {
            continue;
        }
        for (const std::filesystem::directory_entry &entry :
             std::filesystem::directory_iterator(archive_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            std::smatch match;
            if (!std::regex_match(name, match, pattern)) {
                continue;
            }
            const uint32_t ordinal = static_cast<uint32_t>(std::stoul(match[1].str()));
            layers.emplace(ordinal, entry.path());
        }
    }
    return layers;
}

[[nodiscard]] inline bool bc_family_runner_archive_layers_are_contiguous_from_zero(
    const BCFamilySolveRunOptions &options,
    uint32_t *max_ordinal_out = nullptr
) {
    std::map<uint32_t, uint8_t> ordinals;
    for (const auto &[ordinal, layer] :
         bc_family_runner_discover_archive_exact_layers(options)) {
        (void)layer;
        ordinals.emplace(ordinal, 1U);
    }
    for (const auto &[ordinal, path] :
         bc_family_runner_discover_archive_compressed_layers(options)) {
        (void)path;
        ordinals.emplace(ordinal, 1U);
    }
    if (ordinals.empty() || ordinals.begin()->first != 0U) {
        return false;
    }
    uint32_t expected = 0U;
    for (const auto &[ordinal, marker] : ordinals) {
        (void)marker;
        if (ordinal != expected) {
            return false;
        }
        ++expected;
    }
    if (max_ordinal_out != nullptr) {
        *max_ordinal_out = expected - 1U;
    }
    return true;
}

[[nodiscard]] inline uint64_t bc_family_runner_descriptor_rows(
    const BCPositionStreamingReader &reader
) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows = bc_checked_add_u64(
            rows,
            reader.descriptor(cid).success_rows,
            "BC family runner descriptor row count overflow"
        );
    }
    return rows;
}

[[nodiscard]] inline BCPositionStreamingReader bc_family_runner_open_position_stream(
    const BCFamilySolveRunOptions &options,
    const std::filesystem::path &path,
    const BCLut &lut
) {
    if (bc_family_runner_is_position_archive_path(path)) {
        std::vector<uint8_t> bytes = read_temp_byte_archive(NativePath::to_utf8_string(path));
        if (bytes.empty()) {
            throw std::runtime_error("failed to read BC generated position archive: " + path.string());
        }
        BCPositionStreamingReader reader(
            std::make_unique<BCMemoryReadableFile>(std::move(bytes)),
            lut);
        reader.set_validate_loaded_cells(false);
        return reader;
    }
    if (bc_family_runner_is_cell_compressed_position_path(path)) {
        BCPositionStreamingReader reader(
            std::make_unique<BCCellCompressedPositionReadableFile>(path),
            lut);
        reader.set_validate_loaded_cells(false);
        return reader;
    }
    BCPositionStreamingReader reader = options.direct_io
        ? BCPositionStreamingReader::open_direct_auto(
              path,
              lut,
              options.direct_queue_depth,
              options.direct_queue_depth > 1U,
              static_cast<uint64_t>(options.direct_io_chunk_mib) * 1024ULL * 1024ULL)
        : BCPositionStreamingReader::open_buffered(path, lut);
    reader.set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] inline BCPositionFileReader bc_family_runner_open_position_file(
    const BCFamilySolveRunOptions &options,
    const std::filesystem::path &path,
    const BCLut &lut
) {
    if (bc_family_runner_is_position_archive_path(path)) {
        std::vector<uint8_t> bytes = read_temp_byte_archive(NativePath::to_utf8_string(path));
        if (bytes.empty()) {
            throw std::runtime_error("failed to read BC generated position archive: " + path.string());
        }
        return BCPositionFileReader(
            std::make_unique<BCMemoryReadableFile>(std::move(bytes)),
            lut);
    }
    if (bc_family_runner_is_cell_compressed_position_path(path)) {
        return BCPositionFileReader(
            std::make_unique<BCCellCompressedPositionReadableFile>(path),
            lut);
    }
    return options.direct_io
        ? BCPositionFileReader::open_direct_auto(
              path,
              lut,
              options.direct_queue_depth,
              options.direct_queue_depth > 1U,
              static_cast<uint64_t>(options.direct_io_chunk_mib) * 1024ULL * 1024ULL)
        : BCPositionFileReader::open_buffered(path, lut);
}

template <class PositionReader>
[[nodiscard]] inline BCSuccessStreamingReader bc_family_runner_open_success_stream(
    const BCFamilySolveRunOptions &options,
    const std::filesystem::path &path,
    const PositionReader &position
) {
    return options.direct_io
        ? BCSuccessStreamingReader::open_direct_auto(
              path,
              position,
              1U,
              options.direct_queue_depth,
              options.direct_queue_depth > 1U,
              static_cast<uint64_t>(options.direct_io_chunk_mib) * 1024ULL * 1024ULL)
        : BCSuccessStreamingReader::open_buffered(path, position, 1U);
}

inline void bc_family_runner_remove_exact_layer_files(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal
) {
    std::error_code ec;
    for (const std::filesystem::path &dir : bc_family_runner_solved_dirs(options)) {
        std::filesystem::remove(
            bc_family_runner_position_path(dir, options.prefix, ordinal),
            ec);
        ec.clear();
        std::filesystem::remove(
            bc_family_runner_success_path(dir, options.prefix, ordinal),
            ec);
        ec.clear();
    }
}

inline void bc_family_runner_remove_file_quiet(const std::filesystem::path &path) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

inline void bc_family_runner_remove_generated_layer_file(
    const BCFamilySolveRunLayerFile &layer
) {
    bc_family_runner_remove_file_quiet(layer.path);
}

[[nodiscard]] inline std::optional<std::pair<std::filesystem::path, std::filesystem::path>>
bc_family_runner_find_exact_layer_paths(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal
) {
    for (const std::filesystem::path &dir : bc_family_runner_solved_dirs(options)) {
        const std::filesystem::path position_path =
            bc_family_runner_position_path(dir, options.prefix, ordinal);
        const std::filesystem::path success_path =
            bc_family_runner_success_path(dir, options.prefix, ordinal);
        std::error_code ec;
        const bool has_position = std::filesystem::exists(position_path, ec) && !ec;
        ec.clear();
        const bool has_success = std::filesystem::exists(success_path, ec) && !ec;
        if (has_position && has_success) {
            return std::make_pair(position_path, success_path);
        }
    }
    return std::nullopt;
}

template <typename StorageT>
[[nodiscard]] inline bool bc_family_runner_exact_layer_valid(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const BCLut &lut,
    bool remove_invalid = false
) {
    (void)lut;
    (void)remove_invalid;
    // Resume/checkpoint decisions intentionally only inspect filenames. File
    // contents are opened later only when the layer is actually used.
    return bc_family_runner_find_exact_layer_paths(options, ordinal).has_value();
}

[[nodiscard]] inline std::unique_ptr<BCWritableFile> bc_family_runner_make_writer(
    const BCFamilySolveRunOptions &options,
    const std::filesystem::path &path,
    uint64_t logical_size = 0U
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (options.direct_io) {
        BCDirectFileIOOptions io_options;
        io_options.queue_depth = options.direct_queue_depth;
        io_options.max_transfer_bytes =
            static_cast<uint64_t>(options.direct_io_chunk_mib) * 1024ULL * 1024ULL;
        io_options.overlapped = options.direct_queue_depth > 1U;
        io_options.logical_size = logical_size;
        return std::make_unique<BCDirectFileWriter>(path, io_options);
    }
    (void)logical_size;
    return std::make_unique<BCBufferedFileWriter>(path);
}

inline void bc_family_runner_publish_file(
    const std::filesystem::path &tmp,
    const std::filesystem::path &target
) {
    std::error_code ec;
    std::filesystem::rename(tmp, target, ec);
    if (!ec) {
        return;
    }
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        throw std::runtime_error("BC family runner failed to publish file: " + ec.message());
    }
}

inline void bc_family_runner_move_or_copy_file(
    const std::filesystem::path &source,
    const std::filesystem::path &target
) {
    if (source.empty() || target.empty()) {
        throw std::invalid_argument("BC family runner cannot publish empty path");
    }
    if (std::filesystem::absolute(source).lexically_normal() ==
        std::filesystem::absolute(target).lexically_normal()) {
        return;
    }
    if (!target.parent_path().empty()) {
        std::filesystem::create_directories(target.parent_path());
    }
    std::error_code ec;
    std::filesystem::rename(source, target, ec);
    if (!ec) {
        return;
    }
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(source, target, ec);
    if (!ec) {
        return;
    }
    ec.clear();
    std::filesystem::copy_file(
        source,
        target,
        std::filesystem::copy_options::overwrite_existing,
        ec);
    if (ec) {
        throw std::runtime_error("BC family runner failed to publish exact file: " + ec.message());
    }
    std::filesystem::remove(source, ec);
}

inline void bc_family_runner_add_final_compress_stats(
    BCFamilySolveRunLayerMetric &metric,
    const BCCompressedResult::CompressStats &stats
) {
    metric.final_compress_seconds += stats.total_seconds;
    metric.final_compress_read_seconds += stats.read_seconds;
    metric.final_compress_write_seconds += stats.write_seconds;
    metric.final_compress_worker_seconds += stats.compress_worker_seconds;
    metric.final_compress_bucket_blocks += stats.bucket_blocks;
    metric.final_compress_value_blocks += stats.value_blocks;
    metric.final_compress_success_values += stats.success_values;
    metric.final_compress_bucket_raw_bytes += stats.bucket_raw_bytes;
    metric.final_compress_bucket_compressed_bytes += stats.bucket_compressed_bytes;
    metric.final_compress_value_raw_bytes += stats.value_raw_bytes;
    metric.final_compress_value_compressed_bytes += stats.value_compressed_bytes;
    metric.final_compress_output_bytes += stats.output_bytes;
}

[[nodiscard]] inline BCCompressedResult::CompressOptions bc_family_runner_make_compress_options(
    const BCFamilySolveRunOptions &options
) {
    BCCompressedResult::CompressOptions compress_options;
    compress_options.worker_count =
        options.num_threads > 0 ? static_cast<uint32_t>(options.num_threads) : 0U;
    return compress_options;
}

[[nodiscard]] inline std::filesystem::path bc_family_runner_compressed_output_path(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const std::filesystem::path &position_path
) {
    const std::filesystem::path output_dir =
        options.archive_output_dir.empty() ? position_path.parent_path() : options.archive_output_dir;
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    return output_dir / (
        options.prefix + std::to_string(ordinal) +
        BCCompressedResult::kCompressedLayerFileExtension);
}

[[nodiscard]] inline std::vector<std::filesystem::path> bc_family_runner_compressed_output_candidates(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const std::filesystem::path &position_path
) {
    std::vector<std::filesystem::path> dirs = bc_family_runner_archive_dirs(options);
    if (dirs.empty()) {
        dirs.push_back(position_path.parent_path());
    }
    std::vector<std::filesystem::path> paths;
    paths.reserve(dirs.size());
    for (const std::filesystem::path &dir : dirs) {
        paths.push_back(dir / (
            options.prefix + std::to_string(ordinal) +
            BCCompressedResult::kCompressedLayerFileExtension));
    }
    return paths;
}

[[nodiscard]] inline uint64_t bc_family_runner_file_size_or_zero(const std::filesystem::path &path) {
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(path, ec);
    return ec ? 0ULL : static_cast<uint64_t>(size);
}

[[nodiscard]] inline uint64_t bc_family_runner_estimate_compressed_output_bytes(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    return StoragePaths::estimate_compressed_bytes(
        bc_family_runner_file_size_or_zero(position_path),
        bc_family_runner_file_size_or_zero(success_path));
}

[[nodiscard]] inline StoragePaths::ReservedWritePath bc_family_runner_reserve_compressed_output(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    std::vector<std::string> candidates;
    for (const std::filesystem::path &path :
         bc_family_runner_compressed_output_candidates(options, ordinal, position_path)) {
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        candidates.push_back(NativePath::to_utf8_string(path));
    }
    return StoragePaths::reserve_write_path_from_candidates(
        candidates,
        bc_family_runner_estimate_compressed_output_bytes(position_path, success_path));
}

[[nodiscard]] inline std::optional<uint32_t> bc_family_runner_parse_archive_tmp_ordinal(
    const std::string &name,
    const std::string &prefix,
    const std::string &suffix
) {
    if (name.size() <= prefix.size() + suffix.size()) {
        return std::nullopt;
    }
    if (name.compare(0U, prefix.size(), prefix) != 0) {
        return std::nullopt;
    }
    if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return std::nullopt;
    }
    const size_t begin = prefix.size();
    const size_t count = name.size() - prefix.size() - suffix.size();
    for (size_t i = 0U; i < count; ++i) {
        const char ch = name[begin + i];
        if (ch < '0' || ch > '9') {
            return std::nullopt;
        }
    }
    const unsigned long value = std::stoul(name.substr(begin, count));
    if (value > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] inline bool bc_family_runner_file_magic_matches(
    const std::filesystem::path &path,
    uint32_t expected_magic
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    uint8_t bytes[sizeof(uint32_t)] = {};
    in.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(bytes))) {
        return false;
    }
    return bc_load_u32_le(bytes) == expected_magic;
}

struct BCFamilyArchiveTmpPair {
    std::filesystem::path position_tmp;
    std::filesystem::path success_tmp;
};

inline void bc_family_runner_cleanup_archive_tmp_layers(
    const BCFamilySolveRunOptions &options
) {
    const std::string position_suffix = ".bcpos.archive_tmp";
    const std::string success_suffix = ".bcsuc.archive_tmp";
    for (const std::filesystem::path &archive_dir : bc_family_runner_archive_dirs(options)) {
        if (archive_dir.empty() || !std::filesystem::is_directory(archive_dir)) {
            continue;
        }
        std::map<uint32_t, BCFamilyArchiveTmpPair> tmp_layers;
        for (const std::filesystem::directory_entry &entry :
             std::filesystem::directory_iterator(archive_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (std::optional<uint32_t> ordinal =
                    bc_family_runner_parse_archive_tmp_ordinal(
                        name,
                        options.prefix,
                        position_suffix)) {
                tmp_layers[*ordinal].position_tmp = entry.path();
                continue;
            }
            if (std::optional<uint32_t> ordinal =
                    bc_family_runner_parse_archive_tmp_ordinal(
                        name,
                        options.prefix,
                        success_suffix)) {
                tmp_layers[*ordinal].success_tmp = entry.path();
            }
        }

        for (const auto &[ordinal, pair] : tmp_layers) {
            const std::filesystem::path position_target =
                bc_family_runner_position_path(archive_dir, options.prefix, ordinal);
            const std::filesystem::path success_target =
                bc_family_runner_success_path(archive_dir, options.prefix, ordinal);
            const std::filesystem::path compressed_target =
                archive_dir / (
                    options.prefix + std::to_string(ordinal) +
                    BCCompressedResult::kCompressedLayerFileExtension);

            std::error_code ec;
            const bool has_compressed =
                std::filesystem::exists(compressed_target, ec) && !ec;
            ec.clear();
            const bool has_position =
                std::filesystem::exists(position_target, ec) && !ec;
            ec.clear();
            const bool has_success =
                std::filesystem::exists(success_target, ec) && !ec;

            if (has_compressed || (has_position && has_success)) {
                bc_family_runner_remove_file_quiet(pair.position_tmp);
                bc_family_runner_remove_file_quiet(pair.success_tmp);
                continue;
            }

            const bool position_tmp_valid =
                !pair.position_tmp.empty() &&
                bc_family_runner_file_magic_matches(pair.position_tmp, kBCPositionMagic);
            const bool success_tmp_valid =
                !pair.success_tmp.empty() &&
                bc_family_runner_file_magic_matches(pair.success_tmp, kBCSuccessMagic);
            const bool can_recover_position = has_position || position_tmp_valid;
            const bool can_recover_success = has_success || success_tmp_valid;
            if (can_recover_position && can_recover_success) {
                if (!has_position) {
                    bc_family_runner_publish_file(pair.position_tmp, position_target);
                } else {
                    bc_family_runner_remove_file_quiet(pair.position_tmp);
                }
                if (!has_success) {
                    bc_family_runner_publish_file(pair.success_tmp, success_target);
                } else {
                    bc_family_runner_remove_file_quiet(pair.success_tmp);
                }
                continue;
            }

            bc_family_runner_remove_file_quiet(pair.position_tmp);
            bc_family_runner_remove_file_quiet(pair.success_tmp);
            if (has_position != has_success) {
                bc_family_runner_remove_file_quiet(position_target);
                bc_family_runner_remove_file_quiet(success_target);
            }
        }
    }
}

[[nodiscard]] inline bool bc_family_runner_compressed_file_is_fresh(
    const std::filesystem::path &output_path,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    std::error_code ec;
    if (!std::filesystem::exists(output_path, ec) || ec) {
        return false;
    }
    const auto compressed_time = std::filesystem::last_write_time(output_path, ec);
    if (ec) {
        return false;
    }
    const auto position_time = std::filesystem::last_write_time(position_path, ec);
    if (ec) {
        return false;
    }
    const auto success_time = std::filesystem::last_write_time(success_path, ec);
    if (ec) {
        return false;
    }
    return compressed_time >= position_time && compressed_time >= success_time;
}

[[nodiscard]] inline std::optional<std::filesystem::path> bc_family_runner_find_fresh_compressed_output(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    for (const std::filesystem::path &candidate :
         bc_family_runner_compressed_output_candidates(options, ordinal, position_path)) {
        if (bc_family_runner_compressed_file_is_fresh(candidate, position_path, success_path)) {
            return candidate;
        }
    }
    return std::nullopt;
}

inline void bc_family_runner_remove_exact_sources_after_compress(
    const std::filesystem::path &output_path,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    if (!bc_family_runner_compressed_file_is_fresh(output_path, position_path, success_path)) {
        throw std::runtime_error(
            "BC compressed layer is not fresh after compression: " +
            output_path.string());
    }
    auto remove_one = [](const std::filesystem::path &path) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (ec) {
            throw std::runtime_error(
                "BC family runner failed to remove exact file after compression: " +
                path.string() + ": " + ec.message());
        }
    };
    remove_one(position_path);
    remove_one(success_path);
}

[[nodiscard]] inline BCCompressedResult::CompressStats bc_family_runner_final_compression_hook(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    bool remove_sources_after_compress = false
) {
    if (!options.compress) {
        return {};
    }
    if (const std::optional<std::filesystem::path> fresh_output =
            bc_family_runner_find_fresh_compressed_output(options, ordinal, position_path, success_path)) {
        if (remove_sources_after_compress) {
            bc_family_runner_remove_exact_sources_after_compress(
                *fresh_output,
                position_path,
                success_path);
        }
        return {};
    }

    auto output_lease = bc_family_runner_reserve_compressed_output(
        options,
        ordinal,
        position_path,
        success_path);
    const std::filesystem::path output_path = NativePath::from_utf8(output_lease.path());
    const BCLut lut(bc_family_runner_legal_tiles(options.target_rank));
    BCCompressedResult::CompressStats stats =
        BCCompressedResult::compress_exact_layer_to_result(
        position_path,
        success_path,
        lut,
        output_path,
        bc_family_runner_make_compress_options(options));
    output_lease.release();
    if (remove_sources_after_compress) {
        bc_family_runner_remove_exact_sources_after_compress(
            output_path,
            position_path,
            success_path);
    }
    return stats;
}

template <typename StorageT>
struct BCFamilySolveFrontierLayer;

template <typename StorageT>
[[nodiscard]] inline BCCompressedResult::CompressStats
bc_family_runner_final_compression_hook_flat(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const BCPositionLayerReader &position,
    const StorageT *success_values,
    uint64_t success_value_count,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    bool remove_sources_after_compress = false
) {
    if (!options.compress) {
        return {};
    }
    if (const std::optional<std::filesystem::path> fresh_output =
            bc_family_runner_find_fresh_compressed_output(options, ordinal, position_path, success_path)) {
        if (remove_sources_after_compress) {
            bc_family_runner_remove_exact_sources_after_compress(
                *fresh_output,
                position_path,
                success_path);
        }
        return {};
    }

    auto output_lease = bc_family_runner_reserve_compressed_output(
        options,
        ordinal,
        position_path,
        success_path);
    const std::filesystem::path output_path = NativePath::from_utf8(output_lease.path());
    BCCompressedResult::CompressStats stats =
        BCCompressedResult::compress_flat_success_layer_to_result(
            position,
            success_values,
            success_value_count,
            row_width,
            dtype,
            output_path,
            bc_family_runner_make_compress_options(options));
    output_lease.release();
    if (remove_sources_after_compress) {
        bc_family_runner_remove_exact_sources_after_compress(
            output_path,
            position_path,
            success_path);
    }
    return stats;
}

template <typename StorageT>
[[nodiscard]] inline bool bc_family_runner_try_final_compress_retired_in_memory(
    const BCFamilySolveRunOptions &options,
    BCFamilySolveRunLayerMetric &metric,
    const BCFamilySolveFrontierLayer<StorageT> &retired,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path
) {
    if (!options.compress) {
        return false;
    }
    if (retired.resident_cache) {
        const BCResidentSolvedLayer<StorageT> &layer = *retired.resident_cache;
        bc_family_runner_add_final_compress_stats(
            metric,
            bc_family_runner_final_compression_hook_flat<StorageT>(
                options,
                metric.ordinal,
                layer.position,
                layer.success_values.empty() ? nullptr : layer.success_values.data(),
                static_cast<uint64_t>(layer.success_values.size()),
                layer.row_width,
                layer.dtype,
                position_path,
                success_path,
                true));
        return true;
    }
    if (retired.single_frontier_cache) {
        const BCSingleChunkFrontierLayer<StorageT> &layer = *retired.single_frontier_cache;
        bc_family_runner_add_final_compress_stats(
            metric,
            bc_family_runner_final_compression_hook_flat<StorageT>(
                options,
                metric.ordinal,
                layer.position,
                layer.success_values.empty() ? nullptr : layer.success_values.data(),
                static_cast<uint64_t>(layer.success_values.size()),
                layer.row_width,
                layer.dtype,
                position_path,
                success_path,
                true));
        return true;
    }
    return false;
}

[[nodiscard]] inline std::vector<uint8_t> bc_family_runner_make_empty_position_bytes(
    uint64_t layer_sum,
    uint32_t family_unit,
    uint32_t family_count
) {
    std::vector<FamilyCoord> coords;
    coords.reserve(family_count);
    for (uint32_t i = 0U; i < family_count; ++i) {
        coords.push_back(static_cast<FamilyCoord>(i));
    }
    const BCFamilyTable axis(layer_sum, static_cast<uint16_t>(family_unit), std::move(coords));
    const BCCellMatrix matrix(axis);
    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        writer.mark_empty_cell(cid);
    }
    return writer.finish_layer();
}

[[nodiscard]] inline BCFamilyPartitionLayerMap bc_family_runner_partition_for(
    const BCPositionStreamingReader &reader,
    const std::vector<LayerSum> &possible_8tile_sums,
    uint32_t modulus
) {
    if (reader.axis().family_count() != modulus) {
        throw std::runtime_error(
            "BC family runner position family_count does not match family_modulus"
        );
    }
    return build_family_partition_layer_map(
        reader.axis(),
        possible_8tile_sums,
        BCFamilyPartitionPolicy::modulo(modulus));
}

struct BCFamilySolveCheckpoint {
    int64_t next_ordinal = -1;
    int64_t exact_future2_ordinal = -1;
    int64_t exact_future4_ordinal = -1;
    uint32_t dtype = static_cast<uint32_t>(BCSuccessDTypeMode::UInt32);
    uint32_t family_modulus = 0U;
};

inline void bc_family_runner_write_checkpoint(
    const BCFamilySolveRunOptions &options,
    const BCFamilySolveCheckpoint &checkpoint
) {
    const std::filesystem::path path = bc_family_runner_checkpoint_path(options);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) {
            throw std::runtime_error(
                "BC family runner failed to open checkpoint: " + tmp.string()
            );
        }
        out << "next_ordinal,exact_future2_ordinal,exact_future4_ordinal,dtype,family_modulus\n"
            << checkpoint.next_ordinal << ','
            << checkpoint.exact_future2_ordinal << ','
            << checkpoint.exact_future4_ordinal << ','
            << checkpoint.dtype << ','
            << checkpoint.family_modulus << '\n';
    }
    bc_family_runner_publish_file(tmp, path);
}

[[nodiscard]] inline std::optional<BCFamilySolveCheckpoint>
bc_family_runner_read_checkpoint(const BCFamilySolveRunOptions &options) {
    const std::filesystem::path path = bc_family_runner_checkpoint_path(options);
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("BC family runner failed to read checkpoint: " + path.string());
    }
    std::string header;
    std::string row;
    std::getline(in, header);
    std::getline(in, row);
    std::stringstream ss(row);
    std::string item;
    std::vector<std::string> fields;
    while (std::getline(ss, item, ',')) {
        fields.push_back(item);
    }
    if (fields.size() < 5U) {
        throw std::runtime_error("BC family runner checkpoint has too few fields");
    }
    BCFamilySolveCheckpoint checkpoint;
    checkpoint.next_ordinal = std::stoll(fields[0]);
    checkpoint.exact_future2_ordinal = std::stoll(fields[1]);
    checkpoint.exact_future4_ordinal = std::stoll(fields[2]);
    checkpoint.dtype = static_cast<uint32_t>(std::stoul(fields[3]));
    checkpoint.family_modulus = static_cast<uint32_t>(std::stoul(fields[4]));
    return checkpoint;
}

template <typename StorageT>
[[nodiscard]] inline BCResidentSolvedLayer<StorageT> bc_family_runner_make_terminal_layer(
    const BCPositionLayerReader &position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank,
    BCSuccessDTypeMode dtype,
    int num_threads
) {
    if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
        throw std::invalid_argument("BC family runner terminal dtype mismatch");
    }
    const std::vector<uint64_t> offsets = bc_resident_cell_value_offsets(position);
    if (offsets.back() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC family runner terminal value count exceeds size_t");
    }
    const StorageT zero = bc_success_zero_value_for_dtype<StorageT>(dtype);
    const StorageT terminal = bc_success_terminal_value_for_dtype<StorageT>(dtype);
    std::vector<StorageT> values(static_cast<size_t>(offsets.back()), zero);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BCPositionCellScanner(position, cid).for_each_board(
            [&](const BCScannedBoardEntry &entry) {
                if (bc_family_runner_board_has_target_rank(
                        entry.board,
                        target_rank,
                        success_shifts)) {
                    values[static_cast<size_t>(cell_base + entry.local_success_row)] =
                        terminal;
                }
            });
    }
    BCResidentRawSolveResult<StorageT> raw;
    raw.values = std::move(values);
    raw.cell_value_offsets = offsets;
    return bc_resident_compact_zero_in_place<StorageT>(
        position,
        raw,
        position.lut(),
        1U,
        dtype,
        zero,
        num_threads);
}

template <typename StorageT>
[[nodiscard]] inline BCFamilySolveRunLayerMetric bc_family_runner_write_resident_layer(
    const BCFamilySolveRunOptions &options,
    uint32_t ordinal,
    const BCResidentSolvedLayer<StorageT> &layer,
    const std::string &kind
) {
    BCFamilySolveRunLayerMetric metric;
    metric.kind = kind;
    metric.ordinal = ordinal;
    metric.layer_sum = layer.position.header().layer_sum;
    metric.live_rows = layer.compact_stats.live_rows;
    metric.zero_pruned_rows = layer.compact_stats.zero_pruned_rows;
    metric.position_bytes = static_cast<uint64_t>(layer.position.bytes().size());
    const double position_t0 = bc_family_solve_runner_now_seconds();
    {
        std::unique_ptr<BCWritableFile> writer = bc_family_runner_make_writer(
            options,
            bc_family_runner_position_path(options.solved_output_dir, options.prefix, ordinal),
            metric.position_bytes);
        BCFileIOStats io_stats;
        bc_single_chunk_write_position_bytes<StorageT>(
            *writer,
            layer.position.bytes(),
            &io_stats);
        metric.output_position_write_bytes = io_stats.requested_bytes;
    }
    metric.position_write_seconds = bc_family_solve_runner_now_seconds() - position_t0;
    if (options.direct_io && !options.keep_direct_padding) {
        std::filesystem::resize_file(
            bc_family_runner_position_path(options.solved_output_dir, options.prefix, ordinal),
            metric.position_bytes);
    }

    const uint64_t success_payload_bytes =
        static_cast<uint64_t>(layer.success_values.size()) *
            bc_success_dtype_value_size(layer.dtype);
    BCSuccessHeader success_header;
    success_header.descriptor_count = layer.position.cell_count();
    success_header.cell_value_offsets_offset = kBCSuccessHeaderBytes;
    success_header.payload_offset = bc_checked_add_u64(
        success_header.cell_value_offsets_offset,
        bc_success_cell_value_offsets_bytes(success_header.descriptor_count),
        "BC family runner resident success offset table overflow");
    success_header.payload_bytes = success_payload_bytes;
    const uint64_t success_logical_size = bc_success_logical_size(success_header);
    const double success_t0 = bc_family_solve_runner_now_seconds();
    {
        std::unique_ptr<BCWritableFile> writer = bc_family_runner_make_writer(
            options,
            bc_family_runner_success_path(options.solved_output_dir, options.prefix, ordinal),
            success_logical_size);
        BCFileIOStats io_stats;
        metric.success_bytes = write_success_values_to_file<StorageT>(
            *writer,
            layer.position,
            1U,
            layer.dtype,
            layer.success_values,
            &io_stats);
        metric.output_success_write_bytes = io_stats.requested_bytes;
    }
    metric.success_write_seconds = bc_family_solve_runner_now_seconds() - success_t0;
    if (options.direct_io && !options.keep_direct_padding) {
        std::filesystem::resize_file(
            bc_family_runner_success_path(options.solved_output_dir, options.prefix, ordinal),
            metric.success_bytes);
    }
    metric.total_seconds =
        metric.position_write_seconds +
        metric.success_write_seconds;
    return metric;
}

template <typename StorageT>
struct BCFamilySolveFrontierLayer {
    int64_t ordinal = -1;
    std::filesystem::path position_path;
    std::filesystem::path success_path;
    BCPositionStreamingReader position;
    BCSuccessStreamingReader success;
    BCFamilyPartitionLayerMap partition;
    std::unique_ptr<BCResidentSolvedLayer<StorageT>> resident_cache;
    std::unique_ptr<BCSingleChunkFrontierLayer<StorageT>> single_frontier_cache;
    double open_position_seconds = 0.0;
    double open_success_seconds = 0.0;
    double partition_seconds = 0.0;

    [[nodiscard]] bool valid() const {
        return ordinal >= 0;
    }
};

template <typename StorageT>
[[nodiscard]] inline BCFamilySolveFrontierLayer<StorageT> bc_family_runner_open_frontier(
    const BCFamilySolveRunOptions &options,
    int64_t ordinal,
    const BCLut &lut,
    const std::vector<LayerSum> &possible_8tile_sums
) {
    if (ordinal < 0 || ordinal > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("BC family runner frontier ordinal is out of range");
    }
    const uint32_t ord = static_cast<uint32_t>(ordinal);
    BCFamilySolveFrontierLayer<StorageT> layer;
    layer.ordinal = ordinal;
    const std::optional<std::pair<std::filesystem::path, std::filesystem::path>> exact_paths =
        bc_family_runner_find_exact_layer_paths(options, ord);
    if (!exact_paths.has_value()) {
        throw std::runtime_error(
            "BC family runner missing exact frontier files for ordinal " +
            std::to_string(ord)
        );
    }
    layer.position_path = exact_paths->first;
    layer.success_path = exact_paths->second;
    double t0 = bc_family_solve_runner_now_seconds();
    layer.position = bc_family_runner_open_position_stream(options, layer.position_path, lut);
    layer.open_position_seconds = bc_family_solve_runner_now_seconds() - t0;
    t0 = bc_family_solve_runner_now_seconds();
    layer.success = bc_family_runner_open_success_stream(
        options,
        layer.success_path,
        layer.position);
    layer.open_success_seconds = bc_family_solve_runner_now_seconds() - t0;
    if (layer.success.dtype_mode() != options.success_dtype ||
        !bc_success_dtype_matches_type<StorageT>(layer.success.dtype_mode())) {
        throw std::runtime_error("BC family runner frontier success dtype mismatch");
    }
    (void)possible_8tile_sums;
    return layer;
}

template <typename StorageT>
inline void bc_family_runner_ensure_frontier_partition(
    BCFamilySolveFrontierLayer<StorageT> &frontier,
    const BCFamilySolveRunOptions &options,
    const std::vector<LayerSum> &possible_8tile_sums
) {
    if (!frontier.partition.empty()) {
        return;
    }
    const double t0 = bc_family_solve_runner_now_seconds();
    frontier.partition = bc_family_runner_partition_for(
        frontier.position,
        possible_8tile_sums,
        options.family_modulus);
    frontier.partition_seconds += bc_family_solve_runner_now_seconds() - t0;
}

template <typename StorageT>
[[nodiscard]] inline BCResidentSolveOptions<StorageT> bc_family_runner_make_resident_solve_options(
    const BCFamilySolveRunOptions &options,
    const std::vector<uint8_t> &success_shifts,
    const BCQuadrantWordSumTable &word_sums
) {
    BCResidentSolveOptions<StorageT> solve;
    solve.num_threads = options.num_threads;
    solve.row_width = 1U;
    solve.set_dtype(options.success_dtype);
    solve.edge_options.canonical_batch_size = options.canonical_batch_size;
    solve.edge_options.canonical_symm_mode = options.canonical_symm_mode;
    solve.edge_options.spawn_rate4 = options.spawn_rate4;
    solve.edge_options.success_target_rank =
        options.success_target_rank < 0
            ? static_cast<int>(options.target_rank)
            : options.success_target_rank;
    solve.edge_options.success_shifts = &success_shifts;
    solve.edge_options.success_check_all_cells = options.success_shifts.empty();
    solve.edge_options.pattern_masks = &options.pattern_masks;
    solve.edge_options.future_cell_modulus = options.family_modulus;
    solve.word_sums = &word_sums;
    return solve;
}

template <typename StorageT>
[[nodiscard]] inline BCSingleChunkSolveOptions<StorageT> bc_family_runner_make_single_solve_options(
    const BCFamilySolveRunOptions &options,
    const std::vector<uint8_t> &success_shifts,
    const BCQuadrantWordSumTable &word_sums
) {
    BCSingleChunkSolveOptions<StorageT> solve;
    solve.solve = bc_family_runner_make_resident_solve_options<StorageT>(
        options,
        success_shifts,
        word_sums);
    return solve;
}

template <typename StorageT>
[[nodiscard]] inline BCResidentSolvedLayer<StorageT> bc_family_runner_load_resident_frontier(
    const BCFamilySolveFrontierLayer<StorageT> &frontier,
    const BCLut &lut,
    const BCFamilySolveRunOptions &options,
    double *position_read_seconds = nullptr,
    double *success_read_seconds = nullptr
) {
    BCFileIOStats position_io;
    const double position_t0 = bc_family_solve_runner_now_seconds();
    std::vector<uint8_t> position_bytes = frontier.position.read_all_bytes(&position_io);
    if (position_read_seconds != nullptr) {
        *position_read_seconds += bc_family_solve_runner_now_seconds() - position_t0;
    }
    BCSuccessLoadStats success_stats;
    const double success_t0 = bc_family_solve_runner_now_seconds();
    std::vector<StorageT> success_values =
        frontier.success.template read_all_values_typed<StorageT>(&success_stats);
    if (success_read_seconds != nullptr) {
        *success_read_seconds += bc_family_solve_runner_now_seconds() - success_t0;
    }
    BCResidentSolvedLayer<StorageT> layer;
    layer.open(
        std::move(position_bytes),
        std::move(success_values),
        lut,
        1U,
        options.success_dtype);
    return layer;
}

template <typename StorageT>
[[nodiscard]] inline const BCResidentSolvedLayer<StorageT> &
bc_family_runner_resident_frontier_layer(
    BCFamilySolveFrontierLayer<StorageT> &frontier,
    const BCLut &lut,
    const BCFamilySolveRunOptions &options,
    double *position_read_seconds = nullptr,
    double *success_read_seconds = nullptr
) {
    if (!frontier.resident_cache) {
        frontier.resident_cache =
            std::make_unique<BCResidentSolvedLayer<StorageT>>(
                bc_family_runner_load_resident_frontier<StorageT>(
                    frontier,
                    lut,
                    options,
                    position_read_seconds,
                    success_read_seconds));
    }
    return *frontier.resident_cache;
}

template <typename StorageT>
[[nodiscard]] inline BCSingleChunkFrontierLayer<StorageT>
bc_family_runner_take_single_frontier_layer(
    BCFamilySolveFrontierLayer<StorageT> &frontier,
    const BCLut &lut,
    const BCFamilySolveRunOptions &options
) {
    if (!frontier.single_frontier_cache) {
        frontier.single_frontier_cache =
            std::make_unique<BCSingleChunkFrontierLayer<StorageT>>(
                bc_single_chunk_load_frontier_layer<StorageT>(
                    frontier.position,
                    frontier.success,
                    lut,
                    1U,
                    options.success_dtype));
    }
    BCSingleChunkFrontierLayer<StorageT> out =
        std::move(*frontier.single_frontier_cache);
    frontier.single_frontier_cache.reset();
    return out;
}

[[nodiscard]] inline BCPositionLayerReader bc_family_runner_load_current_resident_position(
    const BCPositionStreamingReader &current,
    const BCLut &lut,
    double *read_seconds = nullptr
) {
    BCFileIOStats read_stats;
    const double read_t0 = bc_family_solve_runner_now_seconds();
    std::vector<uint8_t> bytes = current.read_all_bytes(&read_stats);
    if (read_seconds != nullptr) {
        *read_seconds += bc_family_solve_runner_now_seconds() - read_t0;
    }
    BCPositionLayerReader layer;
    layer.open(std::move(bytes), lut);
    return layer;
}

template <typename StorageT>
[[nodiscard]] inline BCSolveRouteDecision bc_family_runner_decide_solve_route(
    const BCFamilySolveRunOptions &options,
    uint64_t current_rows,
    const BCFamilySolveFrontierLayer<StorageT> &future2,
    const BCFamilySolveFrontierLayer<StorageT> &future4
) {
    BCSolveRouteInputs route_inputs;
    route_inputs.current_rows = current_rows;
    route_inputs.future2_live_rows = bc_family_runner_success_value_rows(future2.success);
    route_inputs.future4_live_rows = bc_family_runner_success_value_rows(future4.success);
    route_inputs.available_memory_bytes = options.available_memory_override_bytes != 0U
        ? options.available_memory_override_bytes
        : bc_family_runner_available_memory_bytes();
    route_inputs.fixed_modulus = options.family_modulus;
    return bc_plan_solve_route(route_inputs, options.solve_route);
}

inline void bc_family_runner_apply_route_metric(
    BCFamilySolveRunLayerMetric &metric,
    const BCSolveRouteDecision &route
) {
    metric.solve_route = bc_solve_route_name(route.route);
    metric.route_available_memory_bytes = route.available_memory_bytes;
    metric.route_resident_required_bytes = route.resident_required_bytes;
    metric.route_single_required_bytes = route.single_required_bytes;
    metric.route_required_bytes = route.route_required_bytes;
}

template <typename StorageT>
[[nodiscard]] inline StorageT bc_family_runner_stream_layer_max(
    const BCFamilySolveFrontierLayer<StorageT> &layer,
    StorageT zero_value
) {
    StorageT max_value = zero_value;
    bool have_value = false;
    for (CellId cid = 0U; cid < layer.position.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = layer.position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        std::vector<BCLoadedSuccessCell> loaded =
            layer.success.load_cells(std::vector<CellId>{cid}, nullptr, true);
        if (loaded.size() != 1U) {
            throw std::logic_error("BC family runner max load returned wrong cell count");
        }
        const BCLoadedSuccessCell &success_cell = loaded.front();
        const size_t value_count =
            static_cast<size_t>(desc.success_rows) * layer.success.row_width();
        const StorageT *values = success_cell.typed_values_data<StorageT>();
        for (size_t index = 0U; index < value_count; ++index) {
            const StorageT value = values != nullptr
                ? values[index]
                : success_cell.read_value_typed<StorageT>(
                      static_cast<uint32_t>(index / layer.success.row_width()),
                      static_cast<uint32_t>(index % layer.success.row_width()));
            if (!have_value || value > max_value) {
                have_value = true;
                max_value = value;
            }
        }
    }
    return max_value;
}

template <typename StorageT>
inline void bc_family_runner_append_row_values(
    std::vector<StorageT> &out,
    const BCLoadedSuccessCell &success_cell,
    const StorageT *values,
    uint32_t local_row,
    uint32_t row_width
) {
    const uint64_t base = static_cast<uint64_t>(local_row) * row_width;
    if (values != nullptr) {
        out.insert(
            out.end(),
            values + static_cast<size_t>(base),
            values + static_cast<size_t>(base + row_width));
        return;
    }
    for (uint32_t lane = 0U; lane < row_width; ++lane) {
        out.push_back(success_cell.read_value_typed<StorageT>(local_row, lane));
    }
}

template <typename StorageT>
inline void bc_family_runner_compact_loaded_cell_threshold(
    const BCLut &lut,
    const BCLoadedCell &cell,
    const BCLoadedSuccessCell &success_cell,
    StorageT threshold,
    uint32_t row_width,
    FinalizedCellPayload &payload,
    std::vector<StorageT> &compact_values,
    BCFamilySolveRunLayerMetric &metric
) {
    if (cell.success_rows == 0U || cell.buckets.empty()) {
        return;
    }
    if (success_cell.success_rows != cell.success_rows) {
        throw std::runtime_error("BC family runner archive position/success row mismatch");
    }
    if (success_cell.row_width != row_width) {
        throw std::runtime_error("BC family runner archive success row_width mismatch");
    }
    metric.current_rows = bc_checked_add_u64(
        metric.current_rows,
        cell.success_rows,
        "BC family runner archive current row count overflow");

    const BCLoadedCellView view = cell.view();
    payload.buckets.reserve(view.buckets.size);
    payload.rank_payload.reserve(view.rank_payload.size);
    compact_values.clear();
    compact_values.reserve(static_cast<size_t>(cell.success_rows) * row_width);

    const StorageT *values = success_cell.typed_values_data<StorageT>();
    const size_t typed_value_count = success_cell.typed_value_count<StorageT>();
    if (values != nullptr &&
        typed_value_count < static_cast<size_t>(cell.success_rows) * row_width) {
        throw std::runtime_error("BC family runner archive success payload is truncated");
    }

    uint64_t success_cursor = 0U;
    for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = view.buckets.data[bucket_i];
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len);
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (bitmap_end > view.rank_payload.size) {
            throw std::out_of_range("BC family runner archive source bitmap exceeds rank payload");
        }

        const uint64_t bucket_success_offset = success_cursor;
        const uint32_t payload_start = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t aligned_payload_offset = align_up_u32(payload_start, 8U);
        bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_start);
        const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t out_bitmap_offset =
            bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        const uint64_t payload_end =
            static_cast<uint64_t>(out_bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (payload_end > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("BC family runner archive rank payload exceeds uint32");
        }

        uint32_t bucket_seen = 0U;
        uint32_t bucket_kept = 0U;
        uint32_t prefix_running = 0U;
        std::vector<uint64_t> keep_words;
        keep_words.reserve(word_count);
        const uint8_t *bitmap_words = view.rank_payload.data + bitmap_offset;
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            if ((word_i & 3U) == 0U) {
                if (prefix_running > std::numeric_limits<RankPrefix>::max()) {
                    throw std::logic_error("BC family runner archive prefix exceeds uint16");
                }
                bc_resident_append_u16_le(
                    payload.rank_payload,
                    static_cast<RankPrefix>(prefix_running));
            }
            uint64_t word = load_u64_le(
                bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            uint64_t keep_word = 0U;
            while (word != 0U) {
                const uint32_t bit = bc_resident_countr_zero64(word);
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (local_row >= cell.success_rows) {
                    throw std::out_of_range("BC family runner archive source row exceeds cell");
                }
                const uint64_t row_base = local_row * static_cast<uint64_t>(row_width);
                if (row_base + row_width > typed_value_count && values != nullptr) {
                    throw std::out_of_range("BC family runner archive row exceeds success values");
                }

                bool keep = false;
                if (row_width == 1U) {
                    const StorageT value = values != nullptr
                        ? values[static_cast<size_t>(row_base)]
                        : success_cell.read_value_typed<StorageT>(
                              static_cast<uint32_t>(local_row),
                              0U);
                    keep = value > threshold;
                } else {
                    for (uint32_t lane = 0U; lane < row_width; ++lane) {
                        const StorageT value = values != nullptr
                            ? values[static_cast<size_t>(row_base + lane)]
                            : success_cell.read_value_typed<StorageT>(
                                  static_cast<uint32_t>(local_row),
                                  lane);
                        if (value > threshold) {
                            keep = true;
                            break;
                        }
                    }
                }
                if (keep) {
                    keep_word |= (1ULL << bit);
                    if (row_width == 1U) {
                        compact_values.push_back(values != nullptr
                            ? values[static_cast<size_t>(row_base)]
                            : success_cell.read_value_typed<StorageT>(
                                  static_cast<uint32_t>(local_row),
                                  0U));
                    } else {
                        for (uint32_t lane = 0U; lane < row_width; ++lane) {
                            compact_values.push_back(values != nullptr
                                ? values[static_cast<size_t>(row_base + lane)]
                                : success_cell.read_value_typed<StorageT>(
                                      static_cast<uint32_t>(local_row),
                                      lane));
                        }
                    }
                    ++success_cursor;
                    ++bucket_kept;
                    ++metric.archive_live_rows;
                } else {
                    ++metric.threshold_pruned_rows;
                }
                ++bucket_seen;
                word &= word - 1ULL;
            }
            keep_words.push_back(keep_word);
            prefix_running += popcount64(keep_word);
        }

        const uint32_t expected_bucket_end = (bucket_i + 1U < view.buckets.size)
            ? view.buckets.data[bucket_i + 1U].success_row_offset
            : cell.success_rows;
        const uint64_t observed_bucket_end =
            static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
        if (bucket.success_row_offset > expected_bucket_end ||
            observed_bucket_end != expected_bucket_end) {
            throw std::runtime_error("BC family runner archive bucket row count mismatch");
        }

        if (bucket_kept == 0U) {
            payload.rank_payload.resize(payload_start);
            continue;
        }
        if (payload.rank_payload.size() > out_bitmap_offset) {
            throw std::logic_error("BC family runner archive prefix exceeded bitmap offset");
        }
        bc_resident_append_padding(
            payload.rank_payload,
            out_bitmap_offset - static_cast<uint32_t>(payload.rank_payload.size()));
        bc_resident_append_bitmap_le(payload.rank_payload, keep_words.data(), word_count);
        if (payload.rank_payload.size() != payload_end) {
            throw std::logic_error("BC family runner archive rank payload size mismatch");
        }
        if (bucket_success_offset > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC family runner archive success row offset exceeds uint32");
        }
        payload.buckets.push_back(BCBucketEntry{
            bucket.key,
            rank_payload_offset,
            static_cast<uint32_t>(bucket_success_offset)});
    }

    if (success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC family runner archive success rows exceed uint32");
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    if (payload.success_rows == 0U) {
        payload.buckets.clear();
        payload.rank_payload.clear();
    }
}

template <typename StorageT>
[[nodiscard]] inline BCFamilySolveRunLayerMetric bc_family_runner_archive_prune_retired(
    const BCFamilySolveRunOptions &options,
    BCFamilySolveFrontierLayer<StorageT> &retired,
    RuntimeControls::DeletionThresholdState &deletion_state,
    const BCLut &lut
) {
    BCFamilySolveRunLayerMetric metric;
    if (!retired.valid()) {
        return metric;
    }
    metric.kind = "archive";
    metric.ordinal = static_cast<uint32_t>(retired.ordinal);
    metric.layer_sum = retired.position.header().layer_sum;

    RunOptions deletion_options;
    deletion_options.deletion_threshold = options.deletion_threshold;
    deletion_options.relative_deletion_threshold = options.relative_deletion_threshold;
    deletion_options.deletion_threshold_signal_path = options.deletion_threshold_signal_path;
    deletion_state = RuntimeControls::refresh_deletion_thresholds(
        deletion_options,
        deletion_state);

    const StorageT zero_value = bc_success_zero_value_for_dtype<StorageT>(options.success_dtype);
    const StorageT terminal_value =
        bc_success_terminal_value_for_dtype<StorageT>(options.success_dtype);
    const std::filesystem::path exact_position_path = retired.position_path;
    const std::filesystem::path exact_success_path = retired.success_path;
    const std::filesystem::path archive_dir =
        options.archive_output_dir.empty() ? options.solved_output_dir : options.archive_output_dir;
    const std::filesystem::path position_target =
        bc_family_runner_position_path(archive_dir, options.prefix, metric.ordinal);
    const std::filesystem::path success_target =
        bc_family_runner_success_path(archive_dir, options.prefix, metric.ordinal);
    if (!archive_dir.empty()) {
        std::filesystem::create_directories(archive_dir);
    }

    if (!RuntimeControls::deletion_threshold_enabled(deletion_state)) {
        std::error_code ec;
        metric.position_bytes = std::filesystem::file_size(exact_position_path, ec);
        if (ec) {
            throw std::runtime_error("BC family runner exact position file_size failed: " + ec.message());
        }
        ec.clear();
        metric.success_bytes = std::filesystem::file_size(exact_success_path, ec);
        if (ec) {
            throw std::runtime_error("BC family runner exact success file_size failed: " + ec.message());
        }
        metric.live_rows = bc_family_runner_success_value_rows(retired.success);
        metric.archive_live_rows = metric.live_rows;
        if (options.compress) {
            const bool compressed_in_memory =
                bc_family_runner_try_final_compress_retired_in_memory<StorageT>(
                    options,
                    metric,
                    retired,
                    exact_position_path,
                    exact_success_path);
            retired = BCFamilySolveFrontierLayer<StorageT>();
            if (!compressed_in_memory) {
                bc_family_runner_add_final_compress_stats(
                    metric,
                    bc_family_runner_final_compression_hook(
                        options,
                        metric.ordinal,
                        exact_position_path,
                        exact_success_path,
                        true));
            }
            metric.total_seconds = metric.final_compress_seconds;
            return metric;
        }
        retired = BCFamilySolveFrontierLayer<StorageT>();
        const double publish_t0 = bc_family_solve_runner_now_seconds();
        bc_family_runner_move_or_copy_file(exact_position_path, position_target);
        bc_family_runner_move_or_copy_file(exact_success_path, success_target);
        metric.archive_prune_write_seconds =
            bc_family_solve_runner_now_seconds() - publish_t0;
        metric.total_seconds = metric.archive_prune_write_seconds;
        return metric;
    }

    const double scan_t0 = bc_family_solve_runner_now_seconds();
    const StorageT layer_max = deletion_state.relative > 0.0
        ? bc_family_runner_stream_layer_max(retired, zero_value)
        : zero_value;
    const StorageT threshold = RuntimeControls::effective_deletion_threshold<StorageT>(
        layer_max,
        zero_value,
        terminal_value,
        deletion_state);
    metric.archive_scan_seconds = bc_family_solve_runner_now_seconds() - scan_t0;
    if (!(threshold > zero_value)) {
        std::error_code ec;
        metric.position_bytes = std::filesystem::file_size(exact_position_path, ec);
        if (ec) {
            throw std::runtime_error("BC family runner exact position file_size failed: " + ec.message());
        }
        ec.clear();
        metric.success_bytes = std::filesystem::file_size(exact_success_path, ec);
        if (ec) {
            throw std::runtime_error("BC family runner exact success file_size failed: " + ec.message());
        }
        metric.live_rows = bc_family_runner_success_value_rows(retired.success);
        metric.archive_live_rows = metric.live_rows;
        if (options.compress) {
            const bool compressed_in_memory =
                bc_family_runner_try_final_compress_retired_in_memory<StorageT>(
                    options,
                    metric,
                    retired,
                    exact_position_path,
                    exact_success_path);
            retired = BCFamilySolveFrontierLayer<StorageT>();
            if (!compressed_in_memory) {
                bc_family_runner_add_final_compress_stats(
                    metric,
                    bc_family_runner_final_compression_hook(
                        options,
                        metric.ordinal,
                        exact_position_path,
                        exact_success_path,
                        true));
            }
            metric.total_seconds =
                metric.archive_scan_seconds +
                metric.final_compress_seconds;
            return metric;
        }
        retired = BCFamilySolveFrontierLayer<StorageT>();
        const double publish_t0 = bc_family_solve_runner_now_seconds();
        bc_family_runner_move_or_copy_file(exact_position_path, position_target);
        bc_family_runner_move_or_copy_file(exact_success_path, success_target);
        metric.archive_prune_write_seconds =
            bc_family_solve_runner_now_seconds() - publish_t0;
        metric.total_seconds =
            metric.archive_scan_seconds +
            metric.archive_prune_write_seconds;
        return metric;
    }

    if (options.compress) {
        auto compressed_lease = bc_family_runner_reserve_compressed_output(
            options,
            metric.ordinal,
            exact_position_path,
            exact_success_path);
        const std::filesystem::path compressed_output = NativePath::from_utf8(compressed_lease.path());
        BCCompressedResult::StreamingBuilder builder(
            retired.position,
            retired.success.row_width(),
            options.success_dtype,
            compressed_output,
            bc_family_runner_make_compress_options(options));

        const uint32_t row_width = retired.success.row_width();
        constexpr CellId kArchiveLoadBatchCells = 64U;
        std::vector<CellId> batch_cids;
        batch_cids.reserve(kArchiveLoadBatchCells);
        std::vector<StorageT> compact_values;
        FinalizedCellPayload payload;
        FinalizedCellPayload empty_payload;
        const CellId cell_count = retired.position.cell_count();
        for (CellId chunk_begin = 0U; chunk_begin < cell_count; chunk_begin += kArchiveLoadBatchCells) {
            const CellId chunk_end = std::min<CellId>(
                cell_count,
                chunk_begin + kArchiveLoadBatchCells);
            batch_cids.clear();
            for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
                const BCPositionCellDescriptor &desc = retired.position.descriptor(cid);
                if (!desc.empty() && desc.success_rows != 0U) {
                    batch_cids.push_back(cid);
                }
            }

            std::vector<BCLoadedCell> position_cells =
                batch_cids.empty()
                    ? std::vector<BCLoadedCell>{}
                    : retired.position.load_cells(batch_cids);
            std::vector<BCLoadedSuccessCell> loaded_success =
                batch_cids.empty()
                    ? std::vector<BCLoadedSuccessCell>{}
                    : retired.success.load_cells(batch_cids, nullptr, true);
            if (position_cells.size() != batch_cids.size() ||
                loaded_success.size() != batch_cids.size()) {
                throw std::logic_error("BC family runner compressed archive batch load size mismatch");
            }

            size_t loaded_index = 0U;
            for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
                const BCPositionCellDescriptor &desc = retired.position.descriptor(cid);
                if (desc.empty() || desc.success_rows == 0U) {
                    builder.write_cell(cid, empty_payload, nullptr, 0U);
                    continue;
                }
                if (loaded_index >= batch_cids.size() || batch_cids[loaded_index] != cid) {
                    throw std::logic_error("BC family runner compressed archive batch cid mismatch");
                }
                const BCLoadedCell &position_cell = position_cells[loaded_index];
                const BCLoadedSuccessCell &success_cell = loaded_success[loaded_index];
                if (position_cell.cid != cid || success_cell.cid != cid) {
                    throw std::logic_error("BC family runner compressed archive loaded cid mismatch");
                }

                payload.buckets.clear();
                payload.rank_payload.clear();
                payload.success_rows = 0U;
                bc_family_runner_compact_loaded_cell_threshold(
                    lut,
                    position_cell,
                    success_cell,
                    threshold,
                    row_width,
                    payload,
                    compact_values,
                    metric);

                if (payload.success_rows == 0U) {
                    builder.write_cell(cid, empty_payload, nullptr, 0U);
                    ++loaded_index;
                    continue;
                }
                if (payload.success_rows * static_cast<uint64_t>(row_width) !=
                    static_cast<uint64_t>(compact_values.size())) {
                    throw std::runtime_error(
                        "BC family runner compressed archive compact value mismatch");
                }
                builder.write_cell(
                    cid,
                    payload,
                    compact_values.empty() ? nullptr : compact_values.data(),
                    static_cast<uint64_t>(compact_values.size()));
                ++loaded_index;
            }
            if (loaded_index != batch_cids.size()) {
                throw std::logic_error("BC family runner compressed archive did not consume batch");
            }
        }
        BCCompressedResult::CompressStats compress_stats = builder.finish();
        metric.position_bytes = compress_stats.original_position_bytes;
        metric.success_bytes = compress_stats.original_success_bytes;
        metric.live_rows = metric.archive_live_rows;
        bc_family_runner_add_final_compress_stats(metric, compress_stats);
        compressed_lease.release();
        retired = BCFamilySolveFrontierLayer<StorageT>();
        bc_family_runner_remove_exact_sources_after_compress(
            compressed_output,
            exact_position_path,
            exact_success_path);
        metric.total_seconds =
            metric.archive_scan_seconds +
            metric.final_compress_seconds;
        return metric;
    }

    const std::filesystem::path position_tmp = position_target.string() + ".archive_tmp";
    const std::filesystem::path success_tmp = success_target.string() + ".archive_tmp";
    bc_family_runner_remove_file_quiet(position_tmp);
    bc_family_runner_remove_file_quiet(success_tmp);

    BCSingleChunkSolveStats write_stats;
    const double write_t0 = bc_family_solve_runner_now_seconds();
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    {
        std::unique_ptr<BCWritableFile> position_writer =
            bc_family_runner_make_writer(options, position_tmp);
        std::unique_ptr<BCWritableFile> success_writer =
            bc_family_runner_make_writer(options, success_tmp);
        BCFinalLayerFileStreamer<StorageT> streamer(
            retired.position,
            *position_writer,
            *success_writer,
            retired.success.row_width(),
            options.success_dtype,
            write_stats);

        const uint32_t row_width = retired.success.row_width();
        constexpr CellId kArchiveLoadBatchCells = 64U;
        std::vector<CellId> batch_cids;
        batch_cids.reserve(kArchiveLoadBatchCells);
        std::vector<StorageT> compact_values;
        FinalizedCellPayload payload;
        FinalizedCellPayload empty_payload;
        const CellId cell_count = retired.position.cell_count();
        for (CellId chunk_begin = 0U; chunk_begin < cell_count; chunk_begin += kArchiveLoadBatchCells) {
            const CellId chunk_end = std::min<CellId>(
                cell_count,
                chunk_begin + kArchiveLoadBatchCells);
            batch_cids.clear();
            for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
                const BCPositionCellDescriptor &desc = retired.position.descriptor(cid);
                if (!desc.empty() && desc.success_rows != 0U) {
                    batch_cids.push_back(cid);
                }
            }

            std::vector<BCLoadedCell> position_cells =
                batch_cids.empty()
                    ? std::vector<BCLoadedCell>{}
                    : retired.position.load_cells(batch_cids);
            std::vector<BCLoadedSuccessCell> loaded_success =
                batch_cids.empty()
                    ? std::vector<BCLoadedSuccessCell>{}
                    : retired.success.load_cells(batch_cids, nullptr, true);
            if (position_cells.size() != batch_cids.size() ||
                loaded_success.size() != batch_cids.size()) {
                throw std::logic_error("BC family runner archive batch load size mismatch");
            }

            size_t loaded_index = 0U;
            for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
                const BCPositionCellDescriptor &desc = retired.position.descriptor(cid);
                if (desc.empty() || desc.success_rows == 0U) {
                    streamer.write_cell_metadata(cid, empty_payload);
                    continue;
                }
                if (loaded_index >= batch_cids.size() || batch_cids[loaded_index] != cid) {
                    throw std::logic_error("BC family runner archive batch cid mismatch");
                }
                const BCLoadedCell &position_cell = position_cells[loaded_index];
                const BCLoadedSuccessCell &success_cell = loaded_success[loaded_index];
                if (position_cell.cid != cid || success_cell.cid != cid) {
                    throw std::logic_error("BC family runner archive loaded cid mismatch");
                }

                payload.buckets.clear();
                payload.rank_payload.clear();
                payload.success_rows = 0U;
                bc_family_runner_compact_loaded_cell_threshold(
                    lut,
                    position_cell,
                    success_cell,
                    threshold,
                    row_width,
                    payload,
                    compact_values,
                    metric);

                if (payload.success_rows == 0U) {
                    streamer.write_cell_metadata(cid, empty_payload);
                    ++loaded_index;
                    continue;
                }
                if (payload.success_rows * static_cast<uint64_t>(row_width) !=
                    static_cast<uint64_t>(compact_values.size())) {
                    throw std::runtime_error("BC family runner archive compact value mismatch");
                }
                streamer.write_cell_metadata(cid, payload);
                streamer.write_success_cell_values_raw(
                    cid,
                    compact_values.data(),
                    static_cast<uint64_t>(compact_values.size()));
                ++loaded_index;
            }
            if (loaded_index != batch_cids.size()) {
                throw std::logic_error("BC family runner archive did not consume loaded batch");
            }
        }
        BCSingleChunkSolveFileResult result = streamer.finish();
        position_bytes = result.position_bytes;
        success_bytes = result.success_bytes;
        position_writer.reset();
        success_writer.reset();
    }
    metric.archive_prune_write_seconds = bc_family_solve_runner_now_seconds() - write_t0;
    metric.live_rows = metric.archive_live_rows;
    metric.zero_pruned_rows = metric.threshold_pruned_rows;
    metric.position_bytes = position_bytes;
    metric.success_bytes = success_bytes;
    metric.position_write_seconds = write_stats.position_write_seconds;
    metric.success_write_seconds = write_stats.success_write_seconds;
    metric.output_position_write_bytes = write_stats.output_position_write.requested_bytes;
    metric.output_success_write_bytes = write_stats.output_success_write.requested_bytes;
    retired = BCFamilySolveFrontierLayer<StorageT>();
    bc_family_runner_publish_file(position_tmp, position_target);
    bc_family_runner_publish_file(success_tmp, success_target);
    if (std::filesystem::absolute(exact_position_path).lexically_normal() !=
        std::filesystem::absolute(position_target).lexically_normal()) {
        bc_family_runner_remove_file_quiet(exact_position_path);
    }
    if (std::filesystem::absolute(exact_success_path).lexically_normal() !=
        std::filesystem::absolute(success_target).lexically_normal()) {
        bc_family_runner_remove_file_quiet(exact_success_path);
    }
    if (options.direct_io && !options.keep_direct_padding) {
        std::filesystem::resize_file(position_target, position_bytes);
        std::filesystem::resize_file(success_target, success_bytes);
    }
    if (options.compress) {
        bc_family_runner_add_final_compress_stats(
            metric,
            bc_family_runner_final_compression_hook(
                options,
                metric.ordinal,
                position_target,
                success_target,
                true));
    }
    metric.total_seconds =
        metric.archive_scan_seconds +
        metric.archive_prune_write_seconds +
        metric.final_compress_seconds;
    return metric;
}

template <typename StorageT>
inline void bc_family_runner_emit_metric(
    BCFamilySolveRunResult &result,
    const BCFamilySolveLayerCallback &callback,
    const BCFamilySolveRunLayerMetric &metric
) {
    if (metric.kind.empty()) {
        return;
    }
    result.layers.push_back(metric);
    if (callback) {
        callback(result.layers.back());
    }
}

template <typename StorageT>
inline void bc_family_runner_compress_existing_archive_exact_layers(
    const BCFamilySolveRunOptions &options,
    const BCLut &lut,
    BCFamilySolveRunResult &result,
    const BCFamilySolveLayerCallback &callback
) {
    if (!options.compress) {
        return;
    }
    const std::map<uint32_t, BCFamilySolveArchiveExactLayer> layers =
        bc_family_runner_discover_archive_exact_layers(options);
    for (const auto &[ordinal, layer] : layers) {
        const double t0 = bc_family_solve_runner_now_seconds();
        BCFamilySolveRunLayerMetric metric;
        metric.kind = "archive";
        metric.solve_route = "existing_archive_compress";
        metric.ordinal = ordinal;

        if (const std::optional<std::filesystem::path> compressed_path =
                bc_family_runner_find_fresh_compressed_output(
                    options,
                    ordinal,
                    layer.position_path,
                    layer.success_path)) {
            bc_family_runner_remove_exact_sources_after_compress(
                *compressed_path,
                layer.position_path,
                layer.success_path);
            metric.total_seconds = bc_family_solve_runner_now_seconds() - t0;
            bc_family_runner_emit_metric<StorageT>(result, callback, metric);
            continue;
        }

        std::error_code ec;
        metric.position_bytes = std::filesystem::file_size(layer.position_path, ec);
        if (ec) {
            throw std::runtime_error(
                "BC family runner archive position file_size failed: " + ec.message());
        }
        ec.clear();
        metric.success_bytes = std::filesystem::file_size(layer.success_path, ec);
        if (ec) {
            throw std::runtime_error(
                "BC family runner archive success file_size failed: " + ec.message());
        }

        (void)lut;
        const BCCompressedResult::CompressStats compress_stats =
            bc_family_runner_final_compression_hook(
                options,
                ordinal,
                layer.position_path,
                layer.success_path,
                true);
        bc_family_runner_add_final_compress_stats(metric, compress_stats);
        if (compress_stats.live_rows != 0U) {
            metric.live_rows = compress_stats.live_rows;
            metric.archive_live_rows = compress_stats.live_rows;
        }
        metric.total_seconds = bc_family_solve_runner_now_seconds() - t0;
        bc_family_runner_emit_metric<StorageT>(result, callback, metric);
    }
}

template <typename StorageT>
BCFamilySolveRunResult bc_family_solve_full_run_typed(
    const BCFamilySolveRunOptions &options,
    const BCFamilySolveLayerCallback &callback = {}
) {
    if (!bc_success_dtype_matches_type<StorageT>(options.success_dtype)) {
        throw std::invalid_argument("BC family runner storage type does not match success dtype");
    }
    if (options.generated_position_dir.empty() || options.solved_output_dir.empty() ||
        options.prefix.empty()) {
        throw std::invalid_argument("BC family runner requires position/output dirs and prefix");
    }
    if (options.family_modulus == 0U ||
        options.family_modulus > kBCMaxFamilyModulusForPackedCellId) {
        throw std::invalid_argument("BC family runner family_modulus is outside 1..256");
    }
    if (options.direct_queue_depth == 0U || options.direct_io_chunk_mib == 0U ||
        options.future_reuse_max_families == 0U || options.source_words_per_item == 0U ||
        options.work_schedule_chunk == 0U) {
        throw std::invalid_argument("BC family runner numeric options must be non-zero");
    }
    if (options.optimal_branch_only) {
        throw std::invalid_argument(
            "BC family runner optimal-branch-only mode is reserved but not implemented yet"
        );
    }

    BCFamilySolveRunResult run_result;
    bc_family_runner_cleanup_archive_tmp_layers(options);
    const std::vector<uint8_t> legal_tiles =
        bc_family_runner_legal_tiles(options.target_rank);
    const BCLut lut(legal_tiles);
    const std::vector<uint8_t> success_shifts =
        options.success_shifts.empty()
            ? bc_family_runner_success_shifts()
            : options.success_shifts;
    const BCQuadrantWordSumTable word_sums = bc_family_runner_word_sums(lut);
    const std::vector<LayerSum> possible_8tile_sums =
        build_possible_8tile_sums(legal_tiles, default_2048_tile_sum_values());
    std::map<uint32_t, BCFamilySolveRunLayerFile> layers =
        bc_family_runner_discover_layers_or_empty(options);
    const std::filesystem::path archive_dir = bc_family_runner_archive_dir(options);
    const bool separate_archive_dir =
        !bc_family_runner_same_path(archive_dir, options.solved_output_dir);
    if (!options.force_restart && !options.start_ordinal &&
        (separate_archive_dir || layers.empty())) {
        bc_family_runner_compress_existing_archive_exact_layers<StorageT>(
            options,
            lut,
            run_result,
            callback);
    }
    uint32_t archive_max_ordinal = 0U;
    if (!options.force_restart && !options.start_ordinal &&
        bc_family_runner_archive_layers_are_contiguous_from_zero(
            options,
            &archive_max_ordinal)) {
        const bool generated_has_full_base =
            !layers.empty() && layers.begin()->first == 0U;
        const uint32_t generated_max_ordinal =
            layers.empty() ? 0U : layers.rbegin()->first;
        if (!generated_has_full_base || archive_max_ordinal >= generated_max_ordinal) {
            run_result.min_ordinal = 0U;
            run_result.max_ordinal = archive_max_ordinal;
            run_result.completed = true;
            return run_result;
        }
    }
    if (layers.empty()) {
        layers = bc_family_runner_discover_layers(options);
    }
    const uint32_t discovered_min_ordinal = layers.begin()->first;
    const uint32_t max_ordinal = layers.rbegin()->first;
    if (discovered_min_ordinal != 0U || max_ordinal < 2U) {
        throw std::runtime_error(
            "BC family runner expected generated ordinals from 0 with at least 3 layers"
        );
    }
    const uint32_t min_ordinal = options.min_ordinal.value_or(discovered_min_ordinal);
    if (min_ordinal < discovered_min_ordinal || min_ordinal >= max_ordinal) {
        throw std::invalid_argument("BC family runner min_ordinal is outside generated range");
    }
    run_result.min_ordinal = min_ordinal;
    run_result.max_ordinal = max_ordinal;
    std::filesystem::create_directories(options.solved_output_dir);

    RunOptions deletion_options;
    deletion_options.deletion_threshold = options.deletion_threshold;
    deletion_options.relative_deletion_threshold = options.relative_deletion_threshold;
    deletion_options.deletion_threshold_signal_path = options.deletion_threshold_signal_path;
    RuntimeControls::DeletionThresholdState deletion_state =
        RuntimeControls::current_deletion_thresholds(deletion_options);

    int64_t first_solve_ordinal = static_cast<int64_t>(max_ordinal) - 1;
    BCFamilySolveFrontierLayer<StorageT> future2;
    BCFamilySolveFrontierLayer<StorageT> future4;

    auto ensure_initial_exact_frontiers = [&]() {
        if (!bc_family_runner_exact_layer_valid<StorageT>(options, max_ordinal, lut, true)) {
            const double top_t0 = bc_family_solve_runner_now_seconds();
            const BCPositionFileReader top_position = bc_family_runner_open_position_file(
                options,
                layers.at(max_ordinal).path,
                lut);
            const uint64_t top_layer_sum = top_position.layer().header().layer_sum;
            const uint32_t top_family_unit = top_position.layer().header().family_unit;
            BCResidentSolvedLayer<StorageT> top_layer =
                bc_family_runner_make_terminal_layer<StorageT>(
                    top_position.layer(),
                    success_shifts,
                    options.success_target_rank < 0
                        ? static_cast<int>(options.target_rank)
                        : options.success_target_rank,
                    options.success_dtype,
                    options.num_threads);
            BCFamilySolveRunLayerMetric top_metric =
                bc_family_runner_write_resident_layer<StorageT>(
                    options,
                    max_ordinal,
                    top_layer,
                    "terminal");
            top_metric.total_seconds = bc_family_solve_runner_now_seconds() - top_t0;
            bc_family_runner_emit_metric<StorageT>(run_result, callback, top_metric);

            std::vector<uint8_t> empty_position_bytes =
                bc_family_runner_make_empty_position_bytes(
                    top_layer_sum + 2U,
                    top_family_unit,
                    options.family_modulus);
            BCResidentSolvedLayer<StorageT> empty_layer;
            empty_layer.open(
                std::move(empty_position_bytes),
                {},
                lut,
                1U,
                options.success_dtype);
            BCFamilySolveRunLayerMetric empty_metric =
                bc_family_runner_write_resident_layer<StorageT>(
                    options,
                    max_ordinal + 1U,
                    empty_layer,
                    "virtual_empty");
            bc_family_runner_emit_metric<StorageT>(run_result, callback, empty_metric);
            return;
        }
        if (!bc_family_runner_exact_layer_valid<StorageT>(options, max_ordinal + 1U, lut, true)) {
            const BCPositionFileReader top_position = bc_family_runner_open_position_file(
                options,
                layers.at(max_ordinal).path,
                lut);
            std::vector<uint8_t> empty_position_bytes =
                bc_family_runner_make_empty_position_bytes(
                    top_position.layer().header().layer_sum + 2U,
                    top_position.layer().header().family_unit,
                    options.family_modulus);
            BCResidentSolvedLayer<StorageT> empty_layer;
            empty_layer.open(
                std::move(empty_position_bytes),
                {},
                lut,
                1U,
                options.success_dtype);
            BCFamilySolveRunLayerMetric empty_metric =
                bc_family_runner_write_resident_layer<StorageT>(
                    options,
                    max_ordinal + 1U,
                    empty_layer,
                    "virtual_empty");
            bc_family_runner_emit_metric<StorageT>(run_result, callback, empty_metric);
        }
    };

    auto open_frontier_pair_for = [&](int64_t next_ordinal) {
        if (next_ordinal < static_cast<int64_t>(min_ordinal)) {
            return;
        }
        const uint32_t future2_ord = static_cast<uint32_t>(next_ordinal + 1);
        const uint32_t future4_ord = static_cast<uint32_t>(next_ordinal + 2);
        if (!bc_family_runner_exact_layer_valid<StorageT>(options, future2_ord, lut, true) ||
            !bc_family_runner_exact_layer_valid<StorageT>(options, future4_ord, lut, true)) {
            throw std::runtime_error(
                "BC family runner cannot resume: exact future frontier is missing");
        }
        future2 = bc_family_runner_open_frontier<StorageT>(
            options,
            static_cast<int64_t>(future2_ord),
            lut,
            possible_8tile_sums);
        future4 = bc_family_runner_open_frontier<StorageT>(
            options,
            static_cast<int64_t>(future4_ord),
            lut,
            possible_8tile_sums);
    };

    bool initialized_from_checkpoint = false;
    const std::optional<BCFamilySolveCheckpoint> checkpoint =
        (!options.force_restart && options.resume_from_checkpoint && !options.start_ordinal)
            ? bc_family_runner_read_checkpoint(options)
            : std::nullopt;
    if (checkpoint &&
        checkpoint->dtype == static_cast<uint32_t>(options.success_dtype) &&
        checkpoint->family_modulus == options.family_modulus) {
        int64_t resume_next_ordinal = checkpoint->next_ordinal;
        while (resume_next_ordinal >= static_cast<int64_t>(min_ordinal) &&
               bc_family_runner_exact_layer_valid<StorageT>(
                   options,
                   static_cast<uint32_t>(resume_next_ordinal),
                   lut,
                   true)) {
            --resume_next_ordinal;
        }
        if (resume_next_ordinal < static_cast<int64_t>(min_ordinal)) {
            bool all_requested_exact = true;
            for (uint32_t ordinal = min_ordinal; ordinal <= max_ordinal + 1U; ++ordinal) {
                if (!bc_family_runner_exact_layer_valid<StorageT>(options, ordinal, lut, true)) {
                    all_requested_exact = false;
                    break;
                }
            }
            if (all_requested_exact) {
                run_result.completed = true;
                return run_result;
            }
        } else if (resume_next_ordinal <= static_cast<int64_t>(max_ordinal) - 1 &&
                   bc_family_runner_exact_layer_valid<StorageT>(
                       options,
                       static_cast<uint32_t>(resume_next_ordinal + 1),
                       lut,
                       true) &&
                   bc_family_runner_exact_layer_valid<StorageT>(
                       options,
                       static_cast<uint32_t>(resume_next_ordinal + 2),
                       lut,
                       true)) {
            first_solve_ordinal = resume_next_ordinal;
            open_frontier_pair_for(first_solve_ordinal);
            initialized_from_checkpoint = true;
        }
    }

    if (options.start_ordinal) {
        if (*options.start_ordinal < min_ordinal || *options.start_ordinal >= max_ordinal) {
            throw std::invalid_argument(
                "BC family runner start_ordinal must be in [min_ordinal, max_ordinal)"
            );
        }
        first_solve_ordinal = static_cast<int64_t>(*options.start_ordinal);
        ensure_initial_exact_frontiers();
        open_frontier_pair_for(first_solve_ordinal);
    } else if (!initialized_from_checkpoint) {
        ensure_initial_exact_frontiers();
        if (options.force_restart) {
            first_solve_ordinal = static_cast<int64_t>(max_ordinal) - 1;
        } else {
            bool found_missing_layer = false;
            for (int64_t ordinal_signed = static_cast<int64_t>(max_ordinal) - 1;
                 ordinal_signed >= static_cast<int64_t>(min_ordinal);
                 --ordinal_signed) {
                const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
                if (bc_family_runner_exact_layer_valid<StorageT>(options, ordinal, lut, true)) {
                    BCFamilySolveRunLayerMetric skip_metric;
                    skip_metric.kind = "skip";
                    skip_metric.solve_route = "existing_exact";
                    skip_metric.ordinal = ordinal;
                    bc_family_runner_emit_metric<StorageT>(run_result, callback, skip_metric);
                    continue;
                }
                first_solve_ordinal = ordinal_signed;
                found_missing_layer = true;
                break;
            }
            if (!found_missing_layer) {
                bc_family_runner_write_checkpoint(
                    options,
                    BCFamilySolveCheckpoint{
                        -1,
                        -1,
                        -1,
                        static_cast<uint32_t>(options.success_dtype),
                        options.family_modulus
                    });
                run_result.completed = true;
                return run_result;
            }
        }
        open_frontier_pair_for(first_solve_ordinal);
        bc_family_runner_write_checkpoint(
            options,
            BCFamilySolveCheckpoint{
                first_solve_ordinal,
                first_solve_ordinal + 1,
                first_solve_ordinal + 2,
                static_cast<uint32_t>(options.success_dtype),
                options.family_modulus
            });
    }

    BCFamilySolveWorkspace<StorageT> workspace;
    BCSingleChunkSolveWorkspace<StorageT> single_workspace;
    for (int64_t ordinal_signed = first_solve_ordinal;
         ordinal_signed >= static_cast<int64_t>(min_ordinal);
         --ordinal_signed) {
        const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
        const double layer_t0 = bc_family_solve_runner_now_seconds();
        BCFamilySolveRunLayerMetric metric;
        metric.kind = "solve";
        metric.ordinal = ordinal;

        const double open_t0 = bc_family_solve_runner_now_seconds();
        double t0 = bc_family_solve_runner_now_seconds();
        BCPositionStreamingReader current =
            bc_family_runner_open_position_stream(options, layers.at(ordinal).path, lut);
        metric.open_current_position_seconds = bc_family_solve_runner_now_seconds() - t0;
        metric.open_future2_position_seconds = future2.open_position_seconds;
        metric.open_future2_success_seconds = future2.open_success_seconds;
        metric.open_future4_position_seconds = future4.open_position_seconds;
        metric.open_future4_success_seconds = future4.open_success_seconds;
        metric.open_seconds = bc_family_solve_runner_now_seconds() - open_t0;
        metric.layer_sum = current.header().layer_sum;

        t0 = bc_family_solve_runner_now_seconds();
        const uint64_t descriptor_row_count = bc_family_runner_descriptor_rows(current);
        metric.descriptor_rows_seconds = bc_family_solve_runner_now_seconds() - t0;

        const BCSolveRouteDecision route_decision =
            bc_family_runner_decide_solve_route(options, descriptor_row_count, future2, future4);
        bc_family_runner_apply_route_metric(metric, route_decision);

        const std::filesystem::path output_position =
            bc_family_runner_position_path(options.solved_output_dir, options.prefix, ordinal);
        const std::filesystem::path output_success =
            bc_family_runner_success_path(options.solved_output_dir, options.prefix, ordinal);

        std::unique_ptr<BCResidentSolvedLayer<StorageT>> produced_resident_cache;
        std::unique_ptr<BCSingleChunkFrontierLayer<StorageT>> produced_single_future4_cache;

        switch (route_decision.route) {
        case BCSolveRoute::Resident: {
            future2.single_frontier_cache.reset();
            future4.single_frontier_cache.reset();
            BCResidentSolveOptions<StorageT> solve_options =
                bc_family_runner_make_resident_solve_options<StorageT>(
                    options,
                    success_shifts,
                    word_sums);

            double current_position_read_seconds = 0.0;
            BCPositionLayerReader resident_current =
                bc_family_runner_load_current_resident_position(
                    current,
                    lut,
                    &current_position_read_seconds);
            metric.open_current_position_seconds += current_position_read_seconds;
            metric.open_seconds += current_position_read_seconds;

            double future2_position_read_seconds = 0.0;
            double future2_success_read_seconds = 0.0;
            const BCResidentSolvedLayer<StorageT> &resident_future2 =
                bc_family_runner_resident_frontier_layer<StorageT>(
                    future2,
                    lut,
                    options,
                    &future2_position_read_seconds,
                    &future2_success_read_seconds);
            metric.open_future2_position_seconds += future2_position_read_seconds;
            metric.open_future2_success_seconds += future2_success_read_seconds;
            metric.open_seconds += future2_position_read_seconds + future2_success_read_seconds;

            double future4_position_read_seconds = 0.0;
            double future4_success_read_seconds = 0.0;
            const BCResidentSolvedLayer<StorageT> &resident_future4 =
                bc_family_runner_resident_frontier_layer<StorageT>(
                    future4,
                    lut,
                    options,
                    &future4_position_read_seconds,
                    &future4_success_read_seconds);
            metric.open_future4_position_seconds += future4_position_read_seconds;
            metric.open_future4_success_seconds += future4_success_read_seconds;
            metric.open_seconds += future4_position_read_seconds + future4_success_read_seconds;

            const double solve_t0 = bc_family_solve_runner_now_seconds();
            BCResidentLayerResult<StorageT> resident_result =
                bc_resident_solve_compacted_layer<StorageT>(
                    resident_current,
                    resident_future2,
                    resident_future4,
                    solve_options);
            metric.solve_call_seconds = bc_family_solve_runner_now_seconds() - solve_t0;

            BCFamilySolveRunLayerMetric write_metric =
                bc_family_runner_write_resident_layer<StorageT>(
                    options,
                    ordinal,
                    resident_result.layer,
                    "solve");
            produced_resident_cache =
                std::make_unique<BCResidentSolvedLayer<StorageT>>(
                    std::move(resident_result.layer));
            metric.current_rows = resident_result.solve_stats.current_rows != 0U
                ? resident_result.solve_stats.current_rows
                : descriptor_row_count;
            metric.live_rows = produced_resident_cache->compact_stats.live_rows;
            metric.zero_pruned_rows = produced_resident_cache->compact_stats.zero_pruned_rows;
            metric.position_bytes = write_metric.position_bytes;
            metric.success_bytes = write_metric.success_bytes;
            metric.output_position_write_bytes = write_metric.output_position_write_bytes;
            metric.output_success_write_bytes = write_metric.output_success_write_bytes;
            metric.position_write_seconds = write_metric.position_write_seconds;
            metric.success_write_seconds = write_metric.success_write_seconds;
            metric.final_compress_seconds = write_metric.final_compress_seconds;
            metric.family_stats = BCFamilySolveStats{};
            metric.family_stats.single.current_rows = metric.current_rows;
            metric.family_stats.single.compact_live_rows = metric.live_rows;
            metric.family_stats.single.compact_zero_pruned_rows = metric.zero_pruned_rows;
            metric.family_stats.single.compact_seconds =
                produced_resident_cache->compact_stats.compact_seconds;
            metric.has_family_stats = true;
            break;
        }
        case BCSolveRoute::Single: {
            future2.resident_cache.reset();
            future4.resident_cache.reset();
            BCSingleChunkSolveOptions<StorageT> solve_options =
                bc_family_runner_make_single_solve_options<StorageT>(
                    options,
                    success_shifts,
                    word_sums);

            t0 = bc_family_solve_runner_now_seconds();
            std::unique_ptr<BCWritableFile> position_writer =
                bc_family_runner_make_writer(options, output_position);
            std::unique_ptr<BCWritableFile> success_writer =
                bc_family_runner_make_writer(options, output_success);
            metric.writer_open_seconds = bc_family_solve_runner_now_seconds() - t0;

            const double solve_t0 = bc_family_solve_runner_now_seconds();
            BCSingleChunkStrictFrontierFileResult<StorageT> solve_result =
                bc_single_chunk_solve_strict_1x_to_files_from_future4_frontier<StorageT>(
                    current,
                    future2.position,
                    future2.success,
                    bc_family_runner_take_single_frontier_layer<StorageT>(
                        future4,
                        lut,
                        options),
                    *position_writer,
                    *success_writer,
                    options.solved_output_dir /
                        (options.prefix + std::to_string(ordinal) + "_single_tmp"),
                    solve_options,
                    &single_workspace);
            metric.solve_call_seconds = bc_family_solve_runner_now_seconds() - solve_t0;
            produced_single_future4_cache =
                std::make_unique<BCSingleChunkFrontierLayer<StorageT>>(
                    std::move(solve_result.next_future4_layer));

            t0 = bc_family_solve_runner_now_seconds();
            position_writer.reset();
            success_writer.reset();
            metric.writer_close_seconds = bc_family_solve_runner_now_seconds() - t0;
            if (options.direct_io && !options.keep_direct_padding) {
                t0 = bc_family_solve_runner_now_seconds();
                std::filesystem::resize_file(output_position, solve_result.position_bytes);
                std::filesystem::resize_file(output_success, solve_result.success_bytes);
                metric.post_resize_seconds = bc_family_solve_runner_now_seconds() - t0;
            }
            const BCSingleChunkSolveStats &single_stats = solve_result.stats;
            metric.current_rows = single_stats.current_rows != 0U
                ? single_stats.current_rows
                : descriptor_row_count;
            metric.live_rows = single_stats.compact_live_rows;
            metric.zero_pruned_rows = single_stats.compact_zero_pruned_rows;
            metric.position_bytes = solve_result.position_bytes;
            metric.success_bytes = solve_result.success_bytes;
            metric.output_position_write_bytes =
                single_stats.output_position_write.requested_bytes;
            metric.output_success_write_bytes =
                single_stats.output_success_write.requested_bytes;
            metric.position_write_seconds = single_stats.position_write_seconds;
            metric.success_write_seconds = single_stats.success_write_seconds;
            metric.family_stats = BCFamilySolveStats{};
            metric.family_stats.single = single_stats;
            metric.family_stats.temp_write_seconds = single_stats.partial_write_seconds;
            metric.family_stats.temp_read_seconds = single_stats.partial_read_seconds;
            metric.has_family_stats = true;
            break;
        }
        case BCSolveRoute::Family: {
            future2.resident_cache.reset();
            future4.resident_cache.reset();
            future2.single_frontier_cache.reset();
            future4.single_frontier_cache.reset();
            t0 = bc_family_solve_runner_now_seconds();
            BCFamilyPartitionLayerMap current_partition =
                bc_family_runner_partition_for(
                    current,
                    possible_8tile_sums,
                    options.family_modulus);
            const double current_partition_seconds =
                bc_family_solve_runner_now_seconds() - t0;
            const double future2_partition_before = future2.partition_seconds;
            const double future4_partition_before = future4.partition_seconds;
            bc_family_runner_ensure_frontier_partition(
                future2,
                options,
                possible_8tile_sums);
            bc_family_runner_ensure_frontier_partition(
                future4,
                options,
                possible_8tile_sums);
            metric.partition_seconds =
                current_partition_seconds +
                (future2.partition_seconds - future2_partition_before) +
                (future4.partition_seconds - future4_partition_before);

            BCFamilySolveOptions<StorageT> solve_options;
            solve_options.solve.num_threads = options.num_threads;
            solve_options.solve.row_width = 1U;
            solve_options.solve.set_dtype(options.success_dtype);
            solve_options.solve.edge_options.canonical_batch_size = options.canonical_batch_size;
            solve_options.solve.edge_options.canonical_symm_mode = options.canonical_symm_mode;
            solve_options.solve.edge_options.spawn_rate4 = options.spawn_rate4;
            solve_options.solve.edge_options.success_target_rank =
                options.success_target_rank < 0
                    ? static_cast<int>(options.target_rank)
                    : options.success_target_rank;
            solve_options.solve.edge_options.success_shifts = &success_shifts;
            solve_options.solve.edge_options.success_check_all_cells =
                options.success_shifts.empty();
            solve_options.solve.edge_options.pattern_masks = &options.pattern_masks;
            solve_options.solve.word_sums = &word_sums;
            solve_options.source_bitmap_words_per_work_item = options.source_words_per_item;
            solve_options.source_work_schedule_chunk = options.work_schedule_chunk;
            solve_options.cell_parallel_min_work_items = options.cell_parallel_min_work_items;
            solve_options.future_reuse_max_families = options.future_reuse_max_families;
            solve_options.future_index_recycle_max_bytes = options.future_index_recycle_max_bytes;
            solve_options.final_pending_value_memory_cap_bytes =
                options.final_pending_value_memory_cap_bytes;
            solve_options.temp_direct_io = options.direct_io;
            solve_options.force_temp_buffered_io = false;
            solve_options.keep_temp_files = options.compress_temp_files;
            solve_options.compress_temp_files = options.compress_temp_files;
            solve_options.temp_direct_queue_depth = options.direct_queue_depth;
            t0 = bc_family_solve_runner_now_seconds();
            std::unique_ptr<BCWritableFile> position_writer =
                bc_family_runner_make_writer(options, output_position);
            std::unique_ptr<BCWritableFile> success_writer =
                bc_family_runner_make_writer(options, output_success);
            metric.writer_open_seconds = bc_family_solve_runner_now_seconds() - t0;

            const double solve_t0 = bc_family_solve_runner_now_seconds();
            BCFamilySolveFileResult solve_result =
                bc_family_solve_layer_to_files<StorageT>(
                    current,
                    future2.position,
                    future2.success,
                    future4.position,
                    future4.success,
                    current_partition,
                    future2.partition,
                    future4.partition,
                    *position_writer,
                    *success_writer,
                    options.solved_output_dir /
                        (options.prefix + std::to_string(ordinal) + "_family_tmp"),
                    solve_options,
                    &workspace);
            metric.solve_call_seconds = bc_family_solve_runner_now_seconds() - solve_t0;

            t0 = bc_family_solve_runner_now_seconds();
            position_writer.reset();
            success_writer.reset();
            metric.writer_close_seconds = bc_family_solve_runner_now_seconds() - t0;
            if (options.direct_io && !options.keep_direct_padding) {
                t0 = bc_family_solve_runner_now_seconds();
                std::filesystem::resize_file(output_position, solve_result.position_bytes);
                std::filesystem::resize_file(output_success, solve_result.success_bytes);
                metric.post_resize_seconds = bc_family_solve_runner_now_seconds() - t0;
            }
            const BCSingleChunkSolveStats &single_stats = solve_result.stats.single;
            metric.current_rows = single_stats.current_rows != 0U
                ? single_stats.current_rows
                : descriptor_row_count;
            metric.live_rows = single_stats.compact_live_rows;
            metric.zero_pruned_rows = single_stats.compact_zero_pruned_rows;
            metric.position_bytes = solve_result.position_bytes;
            metric.success_bytes = solve_result.success_bytes;
            metric.output_position_write_bytes =
                single_stats.output_position_write.requested_bytes;
            metric.output_success_write_bytes =
                single_stats.output_success_write.requested_bytes;
            metric.position_write_seconds = single_stats.position_write_seconds;
            metric.success_write_seconds = single_stats.success_write_seconds;
            metric.family_stats = solve_result.stats;
            metric.temp_compress_seconds = solve_result.stats.temp_compress_seconds;
            metric.temp_compressed_bytes = solve_result.stats.temp_compressed_bytes;
            metric.has_family_stats = true;
            break;
        }
        case BCSolveRoute::Auto:
            throw std::logic_error("BC family runner received unresolved auto solve route");
        }
        current = BCPositionStreamingReader();

        const double frontier_t0 = bc_family_solve_runner_now_seconds();
        BCFamilySolveFrontierLayer<StorageT> retired_future4 = std::move(future4);
        future4 = std::move(future2);
        if (produced_single_future4_cache) {
            future4.single_frontier_cache = std::move(produced_single_future4_cache);
        }
        bc_family_runner_write_checkpoint(
            options,
            BCFamilySolveCheckpoint{
                ordinal_signed - 1,
                ordinal_signed,
                future4.ordinal,
                static_cast<uint32_t>(options.success_dtype),
                options.family_modulus
            });
        bc_family_runner_remove_generated_layer_file(layers.at(ordinal));
        metric.total_seconds = bc_family_solve_runner_now_seconds() - layer_t0;
        bc_family_runner_emit_metric<StorageT>(run_result, callback, metric);
        BCFamilySolveRunLayerMetric archive_metric =
            bc_family_runner_archive_prune_retired<StorageT>(
                options,
                retired_future4,
                deletion_state,
                lut);
        if (archive_metric.position_bytes != 0U || archive_metric.success_bytes != 0U) {
            bc_family_runner_emit_metric<StorageT>(run_result, callback, archive_metric);
        }
        future2 = bc_family_runner_open_frontier<StorageT>(
            options,
            ordinal_signed,
            lut,
            possible_8tile_sums);
        if (produced_resident_cache) {
            future2.resident_cache = std::move(produced_resident_cache);
        }
        (void)frontier_t0;
    }

    if (min_ordinal == discovered_min_ordinal) {
        BCFamilySolveRunLayerMetric archive_future4 =
            bc_family_runner_archive_prune_retired<StorageT>(
                options,
                future4,
                deletion_state,
                lut);
        if (archive_future4.position_bytes != 0U || archive_future4.success_bytes != 0U) {
            bc_family_runner_emit_metric<StorageT>(run_result, callback, archive_future4);
        }
        BCFamilySolveRunLayerMetric archive_future2 =
            bc_family_runner_archive_prune_retired<StorageT>(
                options,
                future2,
                deletion_state,
                lut);
        if (archive_future2.position_bytes != 0U || archive_future2.success_bytes != 0U) {
            bc_family_runner_emit_metric<StorageT>(run_result, callback, archive_future2);
        }
        bc_family_runner_write_checkpoint(
            options,
            BCFamilySolveCheckpoint{
                -1,
                -1,
                -1,
                static_cast<uint32_t>(options.success_dtype),
                options.family_modulus
            });
        run_result.completed = true;
    } else {
        bc_family_runner_write_checkpoint(
            options,
            BCFamilySolveCheckpoint{
                static_cast<int64_t>(min_ordinal) - 1,
                future2.ordinal,
                future4.ordinal,
                static_cast<uint32_t>(options.success_dtype),
                options.family_modulus
            });
    }
    return run_result;
}

} // namespace detail

template <typename StorageT>
BCFamilySolveRunResult bc_family_solve_full_run_typed(
    const BCFamilySolveRunOptions &options,
    const BCFamilySolveLayerCallback &callback = {}
) {
    return detail::bc_family_solve_full_run_typed<StorageT>(options, callback);
}

inline BCFamilySolveRunResult bc_family_solve_full_run(
    const BCFamilySolveRunOptions &options,
    const BCFamilySolveLayerCallback &callback = {}
) {
    switch (options.success_dtype) {
        case BCSuccessDTypeMode::UInt32:
            return bc_family_solve_full_run_typed<uint32_t>(options, callback);
        case BCSuccessDTypeMode::UInt64:
            return bc_family_solve_full_run_typed<uint64_t>(options, callback);
        case BCSuccessDTypeMode::Float32:
        case BCSuccessDTypeMode::OneMinusFloat32:
            return bc_family_solve_full_run_typed<float>(options, callback);
        case BCSuccessDTypeMode::Float64:
        case BCSuccessDTypeMode::OneMinusFloat64:
            return bc_family_solve_full_run_typed<double>(options, callback);
    }
    throw std::invalid_argument("BC family runner unsupported success dtype");
}

} // namespace BC
