#include "BCDirectFileIO.h"
#include "BCFamilySolve.h"
#include "BCFamilySolveRunner.h"
#include "BCPositionScanner.h"
#include "BCSuccessIO.h"
#include "SymmetryUtils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::filesystem::path generated_position_dir = "tmp/free10_256_resident_generated_m17";
    std::filesystem::path solved_output_dir = "tmp/free10_256_family_solve_m17";
    std::filesystem::path stats_csv = "tmp/free10_256_family_solve_m17_stats.csv";
    std::filesystem::path summary_csv = "tmp/free10_256_family_solve_m17_summary.csv";
    std::string prefix = "free10_256_";
    uint32_t target_rank = 8U;
    int success_target_rank = -1;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t family_modulus = 17U;
    uint32_t future_reuse_max_families = 4U;
    uint64_t future_index_recycle_max_bytes = 0U;
    uint32_t source_words_per_item = 64U;
    uint32_t work_schedule_chunk = 1U;
    uint32_t cell_parallel_min_work_items = 4U;
    uint64_t final_pending_value_memory_cap_bytes = 0U;
    bool direct_io = false;
    bool temp_buffered = false;
    bool keep_direct_padding = true;
    BC::BCSuccessDTypeMode success_dtype = BC::BCSuccessDTypeMode::UInt32;
    uint32_t direct_queue_depth = 16U;
    std::optional<uint32_t> start_ordinal;
    std::optional<uint32_t> min_ordinal;
};

struct LayerFile {
    uint32_t ordinal = 0U;
    std::filesystem::path path;
};

struct LayerMetric {
    std::string kind = "solve";
    uint32_t ordinal = 0U;
    uint64_t layer_sum = 0U;
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t current_chunks = 0U;
    uint64_t current_cells = 0U;
    uint64_t current_work_items = 0U;
    uint64_t family_current_cells_loaded = 0U;
    uint64_t future2_batch_loads = 0U;
    uint64_t future4_batch_loads = 0U;
    uint64_t future2_cells_loaded = 0U;
    uint64_t future4_cells_loaded = 0U;
    uint64_t future2_active_cells_max = 0U;
    uint64_t future4_active_cells_max = 0U;
    uint64_t future2_position_resident_bytes = 0U;
    uint64_t future2_success_resident_bytes = 0U;
    uint64_t future4_position_resident_bytes = 0U;
    uint64_t future4_success_resident_bytes = 0U;
    uint64_t future_resident_bytes_max = 0U;
    uint64_t spawn4_passes = 0U;
    uint64_t spawn2_passes = 0U;
    uint64_t spawn4_reuse_windows = 0U;
    uint64_t spawn2_reuse_windows = 0U;
    uint64_t future_release_all_calls = 0U;
    uint64_t future_release_except_calls = 0U;
    uint64_t partial4_cells_written = 0U;
    uint64_t partial4_cells_read = 0U;
    uint64_t partial2_cells_written = 0U;
    uint64_t partial2_cells_read = 0U;
    uint64_t scratch4_cells_written = 0U;
    uint64_t scratch4_cells_read = 0U;
    uint64_t finalized_cells = 0U;
    uint64_t pending_cells_max = 0U;
    uint64_t temp_bytes_written = 0U;
    uint64_t temp_bytes_read = 0U;
    uint64_t final_stage_bytes_written = 0U;
    uint64_t final_stage_bytes_read = 0U;
    uint64_t final_stage_write_backend_ops = 0U;
    uint64_t final_stage_write_backend_bytes = 0U;
    double final_stage_write_backend_seconds = 0.0;
    uint64_t final_stage_read_backend_ops = 0U;
    uint64_t final_stage_read_backend_bytes = 0U;
    double final_stage_read_backend_seconds = 0.0;
    uint64_t temp_write_backend_ops = 0U;
    uint64_t temp_write_backend_bytes = 0U;
    double temp_write_backend_seconds = 0.0;
    uint64_t temp_read_backend_ops = 0U;
    uint64_t temp_read_backend_bytes = 0U;
    double temp_read_backend_seconds = 0.0;
    uint64_t current_position_read_bytes = 0U;
    uint64_t future2_position_read_bytes = 0U;
    uint64_t future2_success_read_bytes = 0U;
    uint64_t future4_position_read_bytes = 0U;
    uint64_t future4_success_read_bytes = 0U;
    uint64_t current_position_requested_extents = 0U;
    uint64_t current_position_coalesced_extents = 0U;
    uint64_t current_position_requested_bytes = 0U;
    uint64_t current_position_backend_ops = 0U;
    uint64_t current_position_backend_bytes = 0U;
    double current_position_backend_seconds = 0.0;
    uint64_t future2_position_requested_extents = 0U;
    uint64_t future2_position_coalesced_extents = 0U;
    uint64_t future2_position_requested_bytes = 0U;
    uint64_t future2_position_backend_ops = 0U;
    uint64_t future2_position_backend_bytes = 0U;
    double future2_position_backend_seconds = 0.0;
    uint64_t future2_success_requested_extents = 0U;
    uint64_t future2_success_coalesced_extents = 0U;
    uint64_t future2_success_requested_bytes = 0U;
    uint64_t future2_success_backend_ops = 0U;
    uint64_t future2_success_backend_bytes = 0U;
    double future2_success_backend_seconds = 0.0;
    uint64_t future4_position_requested_extents = 0U;
    uint64_t future4_position_coalesced_extents = 0U;
    uint64_t future4_position_requested_bytes = 0U;
    uint64_t future4_position_backend_ops = 0U;
    uint64_t future4_position_backend_bytes = 0U;
    double future4_position_backend_seconds = 0.0;
    uint64_t future4_success_requested_extents = 0U;
    uint64_t future4_success_coalesced_extents = 0U;
    uint64_t future4_success_requested_bytes = 0U;
    uint64_t future4_success_backend_ops = 0U;
    uint64_t future4_success_backend_bytes = 0U;
    double future4_success_backend_seconds = 0.0;
    uint64_t output_position_write_bytes = 0U;
    uint64_t output_success_write_bytes = 0U;
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
    double current_position_read_seconds = 0.0;
    double future2_position_read_seconds = 0.0;
    double future2_success_read_seconds = 0.0;
    double future2_index_seconds = 0.0;
    double future4_position_read_seconds = 0.0;
    double future4_success_read_seconds = 0.0;
    double future4_index_seconds = 0.0;
    double temp_prepare_seconds = 0.0;
    double temp_write_seconds = 0.0;
    double temp_read_prepare_seconds = 0.0;
    double temp_read_seconds = 0.0;
    double final_stage_write_seconds = 0.0;
    double final_stage_read_seconds = 0.0;
    double output_pending_seconds = 0.0;
    double compact_seconds = 0.0;
    double result_assembly_seconds = 0.0;
    double current_plan_seconds = 0.0;
    double future_release_seconds = 0.0;
    double workspace_release_seconds = 0.0;
    double family_plan_seconds = 0.0;
    double family_workspace_prepare_seconds = 0.0;
    double family_mark_empty_seconds = 0.0;
    double family_current_layout_seconds = 0.0;
    double family_future4_prepare_overhead_seconds = 0.0;
    double family_future2_prepare_overhead_seconds = 0.0;
    double family_future4_prepare_normalize_seconds = 0.0;
    double family_future2_prepare_normalize_seconds = 0.0;
    double family_future4_prepare_select_seconds = 0.0;
    double family_future2_prepare_select_seconds = 0.0;
    double family_future4_prepare_index_build_seconds = 0.0;
    double family_future2_prepare_index_build_seconds = 0.0;
    double family_future4_prepare_insert_sort_seconds = 0.0;
    double family_future2_prepare_insert_sort_seconds = 0.0;
    double family_future4_lookup_copy_seconds = 0.0;
    double family_future2_lookup_copy_seconds = 0.0;
    double family_spawn4_phase_wall_seconds = 0.0;
    double family_spawn2_phase_wall_seconds = 0.0;
    double family_spawn4_phase_untracked_seconds = 0.0;
    double family_spawn2_phase_untracked_seconds = 0.0;
    double family_spawn4_cell_compute_seconds = 0.0;
    double family_spawn2_cell_compute_seconds = 0.0;
    double family_spawn4_bucket_hit_seconds = 0.0;
    double family_spawn2_bucket_hit_seconds = 0.0;
    double family_final_dense_copy_seconds = 0.0;
    double family_compact_value_copy_seconds = 0.0;
    double family_pending_mark_seconds = 0.0;
    double family_output_finish_seconds = 0.0;
    double family_output_streamer_open_seconds = 0.0;
    double family_temp_open_seconds = 0.0;
    double family_temp_close_seconds = 0.0;
    double family_future_release_all_seconds = 0.0;
    double family_future_release_except_seconds = 0.0;
    double family_future_release_all_clear_seconds = 0.0;
    double family_future_release_except_normalize_seconds = 0.0;
    double family_future_release_except_filter_seconds = 0.0;
    double family_future_release_except_erase_seconds = 0.0;
    double family_future_release_except_ids_seconds = 0.0;
    double family_workspace_release_spawn4_partial_seconds = 0.0;
    double family_workspace_release_spawn4_scratch_seconds = 0.0;
    double family_workspace_release_spawn4_temp_values_seconds = 0.0;
    double family_workspace_release_spawn2_dense_seconds = 0.0;
    double family_workspace_release_spawn2_prefetch_seconds = 0.0;
    double family_workspace_release_spawn2_temp_values_seconds = 0.0;
    double partial_cleanup_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double writer_close_seconds = 0.0;
    double post_resize_seconds = 0.0;
    double total_seconds = 0.0;
};

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

[[nodiscard]] int parse_symm_mode(const std::string &value) {
    if (value == "identity") return static_cast<int>(SymmMode::Identity);
    if (value == "full") return static_cast<int>(SymmMode::Full);
    if (value == "diagonal") return static_cast<int>(SymmMode::Diagonal);
    if (value == "horizontal") return static_cast<int>(SymmMode::Horizontal);
    if (value == "min33") return static_cast<int>(SymmMode::Min33);
    if (value == "min24") return static_cast<int>(SymmMode::Min24);
    if (value == "min34") return static_cast<int>(SymmMode::Min34);
    if (value == "min34top") return static_cast<int>(SymmMode::Min34Top);
    return std::stoi(value);
}

[[nodiscard]] BC::BCSuccessDTypeMode parse_success_dtype(const std::string &value) {
    if (value == "uint32") return BC::BCSuccessDTypeMode::UInt32;
    if (value == "uint64") return BC::BCSuccessDTypeMode::UInt64;
    if (value == "float32") return BC::BCSuccessDTypeMode::Float32;
    if (value == "float64") return BC::BCSuccessDTypeMode::Float64;
    if (value == "one-minus-float32" || value == "1-float32") {
        return BC::BCSuccessDTypeMode::OneMinusFloat32;
    }
    if (value == "one-minus-float64" || value == "1-float64") {
        return BC::BCSuccessDTypeMode::OneMinusFloat64;
    }
    throw std::invalid_argument("unsupported success dtype: " + value);
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--position-dir" || key == "--generated-position-dir") {
            args.generated_position_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--output-dir" || key == "--solved-output-dir") {
            args.solved_output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--summary-csv") {
            args.summary_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--prefix") {
            args.prefix = require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--success-target-rank") {
            args.success_target_rank = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--success-dtype" || key == "--dtype") {
            args.success_dtype = parse_success_dtype(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--canonical-symm-mode") {
            args.canonical_symm_mode = parse_symm_mode(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--spawn-rate4") {
            args.spawn_rate4 = std::stod(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-modulus") {
            args.family_modulus = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--future-reuse-max-families") {
            args.future_reuse_max_families =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--future-index-recycle-bytes") {
            args.future_index_recycle_max_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--future-index-recycle-mib") {
            args.future_index_recycle_max_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str()))) *
                1024ULL * 1024ULL;
        } else if (key == "--source-words-per-item") {
            args.source_words_per_item =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--work-schedule-chunk") {
            args.work_schedule_chunk =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--cell-parallel-min-work-items") {
            args.cell_parallel_min_work_items =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--final-pending-value-cap-bytes") {
            args.final_pending_value_memory_cap_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--final-pending-value-cap-mib") {
            args.final_pending_value_memory_cap_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str()))) *
                1024ULL * 1024ULL;
        } else if (key == "--direct-io") {
            args.direct_io = true;
        } else if (key == "--temp-buffered") {
            args.temp_buffered = true;
        } else if (key == "--keep-direct-padding") {
            args.keep_direct_padding = true;
        } else if (key == "--trim-direct-padding") {
            args.keep_direct_padding = false;
        } else if (key == "--direct-queue-depth") {
            args.direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--start-ordinal") {
            args.start_ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--min-ordinal") {
            args.min_ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.success_target_rank < 0) {
        args.success_target_rank = static_cast<int>(args.target_rank);
    }
    if (args.family_modulus == 0U || args.direct_queue_depth == 0U ||
        args.future_reuse_max_families == 0U || args.source_words_per_item == 0U ||
        args.work_schedule_chunk == 0U) {
        throw std::invalid_argument("numeric options must be non-zero");
    }
    return args;
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("--target-rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] std::vector<uint8_t> all_board_success_shifts() {
    std::vector<uint8_t> shifts;
    shifts.reserve(16U);
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }
    return shifts;
}

[[nodiscard]] BC::BCQuadrantWordSumTable build_word_sum_table(const BC::BCLut &lut) {
    BC::BCQuadrantWordSumTable sums(BC::kBCQuadrantWordCount, 0U);
    for (uint32_t word = 0U; word < BC::kBCQuadrantWordCount; ++word) {
        const BC::BCWordDesc &desc = lut.word_desc(static_cast<uint16_t>(word));
        if (desc.valid) {
            sums[word] = desc.sum;
        }
    }
    return sums;
}

[[nodiscard]] bool board_has_target_rank(
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

[[nodiscard]] double gbps(uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1.0e9 : 0.0;
}

[[nodiscard]] std::map<uint32_t, LayerFile> discover_layers(const Args &args) {
    if (!std::filesystem::is_directory(args.generated_position_dir)) {
        throw std::runtime_error("position dir does not exist: " + args.generated_position_dir.string());
    }
    const std::regex pattern("^" + args.prefix + "([0-9]+)\\.bcpos$");
    std::map<uint32_t, LayerFile> layers;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(args.generated_position_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        const uint32_t ordinal = static_cast<uint32_t>(std::stoul(match[1].str()));
        layers.emplace(ordinal, LayerFile{ordinal, entry.path()});
    }
    if (layers.empty()) {
        throw std::runtime_error("no position files found with prefix: " + args.prefix);
    }
    uint32_t expected = layers.begin()->first;
    for (const auto &[ordinal, layer] : layers) {
        (void)layer;
        if (ordinal != expected) {
            throw std::runtime_error("position ordinals are not contiguous at " + std::to_string(expected));
        }
        ++expected;
    }
    return layers;
}

[[nodiscard]] uint64_t descriptor_rows(const BC::BCPositionStreamingReader &reader) {
    uint64_t rows = 0U;
    for (BC::CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

[[nodiscard]] std::filesystem::path position_path_for(const Args &args, uint32_t ordinal) {
    return args.solved_output_dir / (args.prefix + std::to_string(ordinal) + ".bcpos");
}

[[nodiscard]] std::filesystem::path success_path_for(const Args &args, uint32_t ordinal) {
    return args.solved_output_dir / (args.prefix + std::to_string(ordinal) + ".bcsuc");
}

[[nodiscard]] BC::BCPositionStreamingReader open_position_stream(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCLut &lut
) {
    BC::BCPositionStreamingReader reader = args.direct_io
        ? BC::BCPositionStreamingReader::open_direct_auto(
              path,
              lut,
              args.direct_queue_depth,
              args.direct_queue_depth > 1U)
        : BC::BCPositionStreamingReader::open_buffered(path, lut);
    reader.set_validate_loaded_cells(false);
    return reader;
}

template <class PositionReader>
[[nodiscard]] BC::BCSuccessStreamingReader open_success_stream(
    const Args &args,
    const std::filesystem::path &path,
    const PositionReader &position
) {
    if (args.direct_io) {
        return BC::BCSuccessStreamingReader::open_direct_auto(
            path,
            position,
            1U,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U);
    }
    return BC::BCSuccessStreamingReader::open_buffered(path, position, 1U);
}

[[nodiscard]] BC::BCPositionFileReader open_position_file(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCLut &lut
) {
    if (args.direct_io) {
        return BC::BCPositionFileReader::open_direct_auto(
            path,
            lut,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U);
    }
    return BC::BCPositionFileReader::open_buffered(path, lut);
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> make_output_writer(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size = 0U
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (args.direct_io) {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    (void)logical_size;
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

template <typename StorageT>
[[nodiscard]] BC::BCResidentSolvedLayer<StorageT> make_terminal_solved_layer(
    const BC::BCPositionLayerReader &position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank,
    BC::BCSuccessDTypeMode dtype,
    int num_threads
) {
    if (!BC::bc_success_dtype_matches_type<StorageT>(dtype)) {
        throw std::invalid_argument("terminal layer storage type does not match success dtype");
    }
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    if (offsets.back() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("terminal layer row count exceeds size_t");
    }
    const StorageT zero = BC::bc_success_zero_value_for_dtype<StorageT>(dtype);
    const StorageT terminal = BC::bc_success_terminal_value_for_dtype<StorageT>(dtype);
    std::vector<StorageT> values(static_cast<size_t>(offsets.back()), zero);
    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BC::BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                if (board_has_target_rank(entry.board, target_rank, success_shifts)) {
                    values[static_cast<size_t>(cell_base + entry.local_success_row)] =
                        terminal;
                }
            });
    }
    BC::BCResidentRawSolveResult<StorageT> raw;
    raw.values = std::move(values);
    raw.cell_value_offsets = offsets;
    return BC::bc_resident_compact_zero_in_place<StorageT>(
        position,
        raw,
        position.lut(),
        1U,
        dtype,
        zero,
        num_threads);
}

struct WriteResult {
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    double position_seconds = 0.0;
    double success_seconds = 0.0;
};

template <typename StorageT>
[[nodiscard]] WriteResult write_solved_layer_files(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentSolvedLayer<StorageT> &layer
) {
    WriteResult result;
    result.position_bytes = static_cast<uint64_t>(layer.position.bytes().size());
    {
        std::unique_ptr<BC::BCWritableFile> writer =
            make_output_writer(args, position_path_for(args, ordinal), result.position_bytes);
        BC::BCFileIOStats stats;
        const double t0 = now_seconds();
        BC::bc_single_chunk_write_position_bytes<StorageT>(*writer, layer.position.bytes(), &stats);
        writer.reset();
        result.position_seconds = now_seconds() - t0;
        if (args.direct_io && !args.keep_direct_padding) {
            std::filesystem::resize_file(position_path_for(args, ordinal), result.position_bytes);
        }
    }
    const uint64_t success_logical_size =
        BC::kBCSuccessHeaderBytes +
        static_cast<uint64_t>(layer.success_values.size()) *
            BC::bc_success_dtype_value_size(layer.dtype);
    {
        std::unique_ptr<BC::BCWritableFile> writer =
            make_output_writer(args, success_path_for(args, ordinal), success_logical_size);
        BC::BCFileIOStats stats;
        const double t0 = now_seconds();
        result.success_bytes = BC::write_success_values_to_file<StorageT>(
            *writer,
            layer.position,
            1U,
            layer.dtype,
            layer.success_values,
            &stats);
        writer.reset();
        result.success_seconds = now_seconds() - t0;
        if (args.direct_io && !args.keep_direct_padding) {
            std::filesystem::resize_file(success_path_for(args, ordinal), result.success_bytes);
        }
    }
    return result;
}

[[nodiscard]] std::vector<uint8_t> make_empty_position_bytes(
    uint64_t layer_sum,
    uint32_t family_unit,
    uint32_t family_count
) {
    std::vector<BC::FamilyCoord> coords;
    coords.reserve(family_count);
    for (uint32_t i = 0U; i < family_count; ++i) {
        coords.push_back(static_cast<BC::FamilyCoord>(i));
    }
    const BC::BCFamilyTable axis(
        layer_sum,
        static_cast<uint16_t>(family_unit),
        std::move(coords));
    const BC::BCCellMatrix matrix(axis);
    BC::BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (BC::CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        writer.mark_empty_cell(cid);
    }
    return writer.finish_layer();
}

[[nodiscard]] BC::BCFamilyPartitionLayerMap partition_for(
    const BC::BCPositionStreamingReader &reader,
    const std::vector<BC::LayerSum> &possible_8tile_sums,
    uint32_t modulus
) {
    if (reader.axis().family_count() != modulus) {
        throw std::runtime_error("position layer family_count does not match --family-modulus");
    }
    return BC::build_family_partition_layer_map(
        reader.axis(),
        possible_8tile_sums,
        BC::BCFamilyPartitionPolicy::modulo(modulus));
}

[[nodiscard]] double accounted_seconds(const LayerMetric &m) {
    return m.open_seconds +
        m.descriptor_rows_seconds +
        m.partition_seconds +
        m.writer_open_seconds +
        m.current_position_read_seconds +
        m.future2_position_read_seconds +
        m.future2_success_read_seconds +
        m.future2_index_seconds +
        m.future4_position_read_seconds +
        m.future4_success_read_seconds +
        m.future4_index_seconds +
        m.temp_prepare_seconds +
        m.temp_write_seconds +
        m.temp_read_prepare_seconds +
        m.temp_read_seconds +
        m.final_stage_write_seconds +
        m.final_stage_read_seconds +
        m.output_pending_seconds +
        m.compact_seconds +
        m.result_assembly_seconds +
        m.current_plan_seconds +
        m.future_release_seconds +
        m.workspace_release_seconds +
        m.family_plan_seconds +
        m.family_workspace_prepare_seconds +
        m.family_mark_empty_seconds +
        m.family_current_layout_seconds +
        m.family_future4_prepare_overhead_seconds +
        m.family_future2_prepare_overhead_seconds +
        m.family_future4_lookup_copy_seconds +
        m.family_future2_lookup_copy_seconds +
        m.family_spawn4_cell_compute_seconds +
        m.family_spawn2_cell_compute_seconds +
        m.family_final_dense_copy_seconds +
        m.family_compact_value_copy_seconds +
        m.family_pending_mark_seconds +
        m.family_output_finish_seconds +
        m.family_output_streamer_open_seconds +
        m.family_temp_open_seconds +
        m.family_temp_close_seconds +
        m.partial_cleanup_seconds +
        m.position_write_seconds +
        m.success_write_seconds +
        m.writer_close_seconds +
        m.post_resize_seconds;
}

void write_stats_header(std::ofstream &out) {
    out
        << "kind,ordinal,layer_sum,current_rows,live_rows,zero_pruned_rows,final_rows,"
        << "position_bytes,success_bytes,current_chunks,current_cells,current_work_items,"
        << "family_current_cells_loaded,future2_batch_loads,future4_batch_loads,"
        << "future2_cells_loaded,future4_cells_loaded,future2_active_cells_max,"
        << "future4_active_cells_max,future2_position_resident_bytes,"
        << "future2_success_resident_bytes,future4_position_resident_bytes,"
        << "future4_success_resident_bytes,future_resident_bytes_max,"
        << "spawn4_passes,spawn2_passes,"
        << "spawn4_reuse_windows,spawn2_reuse_windows,"
        << "future_release_all_calls,future_release_except_calls,"
        << "partial4_cells_written,"
        << "partial4_cells_read,partial2_cells_written,partial2_cells_read,"
        << "scratch4_cells_written,scratch4_cells_read,finalized_cells,pending_cells_max,"
        << "temp_bytes_written,temp_bytes_read,final_stage_bytes_written,"
        << "final_stage_bytes_read,final_stage_write_backend_ops,"
        << "final_stage_write_backend_bytes,final_stage_write_backend_seconds,"
        << "final_stage_read_backend_ops,final_stage_read_backend_bytes,"
        << "final_stage_read_backend_seconds,temp_write_backend_ops,"
        << "temp_write_backend_bytes,temp_write_backend_seconds,temp_read_backend_ops,"
        << "temp_read_backend_bytes,temp_read_backend_seconds,current_position_read_bytes,"
        << "future2_position_read_bytes,future2_success_read_bytes,"
        << "future4_position_read_bytes,future4_success_read_bytes,"
        << "current_position_requested_extents,current_position_coalesced_extents,"
        << "current_position_requested_bytes,current_position_backend_ops,"
        << "current_position_backend_bytes,current_position_backend_seconds,"
        << "future2_position_requested_extents,"
        << "future2_position_coalesced_extents,future2_position_requested_bytes,"
        << "future2_position_backend_ops,future2_position_backend_bytes,"
        << "future2_position_backend_seconds,"
        << "future2_success_requested_extents,future2_success_coalesced_extents,"
        << "future2_success_requested_bytes,future2_success_backend_ops,"
        << "future2_success_backend_bytes,future2_success_backend_seconds,"
        << "future4_position_requested_extents,"
        << "future4_position_coalesced_extents,future4_position_requested_bytes,"
        << "future4_position_backend_ops,future4_position_backend_bytes,"
        << "future4_position_backend_seconds,"
        << "future4_success_requested_extents,future4_success_coalesced_extents,"
        << "future4_success_requested_bytes,future4_success_backend_ops,"
        << "future4_success_backend_bytes,future4_success_backend_seconds,"
        << "output_position_write_bytes,output_success_write_bytes,open_seconds,"
        << "open_current_position_seconds,open_future2_position_seconds,"
        << "open_future2_success_seconds,open_future4_position_seconds,"
        << "open_future4_success_seconds,"
        << "descriptor_rows_seconds,partition_seconds,writer_open_seconds,"
        << "solve_call_seconds,solve_untracked_seconds,driver_untracked_seconds,"
        << "current_position_read_seconds,future2_position_read_seconds,"
        << "future2_success_read_seconds,future2_index_seconds,"
        << "future4_position_read_seconds,future4_success_read_seconds,"
        << "future4_index_seconds,temp_prepare_seconds,temp_write_seconds,"
        << "temp_read_prepare_seconds,temp_read_seconds,"
        << "final_stage_write_seconds,final_stage_read_seconds,"
        << "output_pending_seconds,compact_seconds,"
        << "result_assembly_seconds,current_plan_seconds,future_release_seconds,"
        << "workspace_release_seconds,family_plan_seconds,family_workspace_prepare_seconds,"
        << "family_mark_empty_seconds,family_current_layout_seconds,"
        << "family_future4_prepare_overhead_seconds,family_future2_prepare_overhead_seconds,"
        << "family_future4_prepare_normalize_seconds,"
        << "family_future2_prepare_normalize_seconds,"
        << "family_future4_prepare_select_seconds,"
        << "family_future2_prepare_select_seconds,"
        << "family_future4_prepare_index_build_seconds,"
        << "family_future2_prepare_index_build_seconds,"
        << "family_future4_prepare_insert_sort_seconds,"
        << "family_future2_prepare_insert_sort_seconds,"
        << "family_future4_lookup_copy_seconds,family_future2_lookup_copy_seconds,"
        << "family_spawn4_phase_wall_seconds,family_spawn2_phase_wall_seconds,"
        << "family_spawn4_phase_untracked_seconds,"
        << "family_spawn2_phase_untracked_seconds,"
        << "family_spawn4_cell_compute_seconds,family_spawn2_cell_compute_seconds,"
        << "family_spawn4_bucket_hit_seconds,family_spawn2_bucket_hit_seconds,"
        << "family_final_dense_copy_seconds,family_compact_value_copy_seconds,"
        << "family_pending_mark_seconds,family_output_finish_seconds,"
        << "family_output_streamer_open_seconds,"
        << "family_temp_open_seconds,family_temp_close_seconds,"
        << "family_future_release_all_seconds,family_future_release_except_seconds,"
        << "family_future_release_all_clear_seconds,"
        << "family_future_release_except_normalize_seconds,"
        << "family_future_release_except_filter_seconds,"
        << "family_future_release_except_erase_seconds,"
        << "family_future_release_except_ids_seconds,"
        << "family_workspace_release_spawn4_partial_seconds,"
        << "family_workspace_release_spawn4_scratch_seconds,"
        << "family_workspace_release_spawn4_temp_values_seconds,"
        << "family_workspace_release_spawn2_dense_seconds,"
        << "family_workspace_release_spawn2_prefetch_seconds,"
        << "family_workspace_release_spawn2_temp_values_seconds,"
        << "partial_cleanup_seconds,position_write_seconds,success_write_seconds,"
        << "writer_close_seconds,post_resize_seconds,accounted_seconds,"
        << "untracked_seconds,total_seconds,total_mrows_per_sec,final_mrows_per_sec,"
        << "recalc_mrows_per_sec,"
        << "current_position_gbps,future2_position_gbps,future2_success_gbps,"
        << "future4_position_gbps,future4_success_gbps,"
        << "temp_write_logical_gbps,temp_read_logical_gbps,"
        << "final_stage_write_logical_gbps,final_stage_read_logical_gbps,"
        << "output_position_gbps,output_success_gbps,current_position_backend_gbps,"
        << "future2_position_backend_gbps,future2_success_backend_gbps,"
        << "future4_position_backend_gbps,future4_success_backend_gbps,"
        << "temp_write_backend_gbps,temp_read_backend_gbps,"
        << "final_stage_write_backend_gbps,final_stage_read_backend_gbps\n";
}

void write_metric_row(std::ofstream &out, const LayerMetric &m) {
    const double accounted = accounted_seconds(m);
    const uint64_t final_rows = m.live_rows + m.zero_pruned_rows;
    const double total_mrows = m.total_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.total_seconds / 1.0e6
        : 0.0;
    const double final_mrows = m.total_seconds > 0.0
        ? static_cast<double>(final_rows) / m.total_seconds / 1.0e6
        : 0.0;
    const double compute_seconds = m.total_seconds -
        (m.open_seconds + m.current_position_read_seconds +
         m.future2_position_read_seconds + m.future2_success_read_seconds +
         m.future4_position_read_seconds + m.future4_success_read_seconds +
         m.temp_write_seconds + m.temp_read_seconds +
         m.final_stage_write_seconds + m.final_stage_read_seconds +
         m.position_write_seconds + m.success_write_seconds);
    const double recalc_mrows = compute_seconds > 0.0
        ? static_cast<double>(m.current_rows) / compute_seconds / 1.0e6
        : 0.0;
    const double solve_accounted = accounted -
        (m.open_seconds + m.descriptor_rows_seconds + m.partition_seconds +
         m.writer_open_seconds + m.writer_close_seconds + m.post_resize_seconds);
    const double solve_untracked = m.solve_call_seconds - solve_accounted;
    const double driver_tracked = m.open_seconds + m.descriptor_rows_seconds +
        m.partition_seconds + m.writer_open_seconds + m.solve_call_seconds +
        m.writer_close_seconds + m.post_resize_seconds;
    const double driver_untracked = m.total_seconds - driver_tracked;
    out
        << m.kind << ',' << m.ordinal << ',' << m.layer_sum << ','
        << m.current_rows << ',' << m.live_rows << ',' << m.zero_pruned_rows << ','
        << final_rows << ','
        << m.position_bytes << ',' << m.success_bytes << ','
        << m.current_chunks << ',' << m.current_cells << ',' << m.current_work_items << ','
        << m.family_current_cells_loaded << ','
        << m.future2_batch_loads << ',' << m.future4_batch_loads << ','
        << m.future2_cells_loaded << ',' << m.future4_cells_loaded << ','
        << m.future2_active_cells_max << ',' << m.future4_active_cells_max << ','
        << m.future2_position_resident_bytes << ','
        << m.future2_success_resident_bytes << ','
        << m.future4_position_resident_bytes << ','
        << m.future4_success_resident_bytes << ','
        << m.future_resident_bytes_max << ','
        << m.spawn4_passes << ',' << m.spawn2_passes << ','
        << m.spawn4_reuse_windows << ',' << m.spawn2_reuse_windows << ','
        << m.future_release_all_calls << ',' << m.future_release_except_calls << ','
        << m.partial4_cells_written << ',' << m.partial4_cells_read << ','
        << m.partial2_cells_written << ',' << m.partial2_cells_read << ','
        << m.scratch4_cells_written << ',' << m.scratch4_cells_read << ','
        << m.finalized_cells << ',' << m.pending_cells_max << ','
        << m.temp_bytes_written << ',' << m.temp_bytes_read << ','
        << m.final_stage_bytes_written << ',' << m.final_stage_bytes_read << ','
        << m.final_stage_write_backend_ops << ','
        << m.final_stage_write_backend_bytes << ','
        << m.final_stage_write_backend_seconds << ','
        << m.final_stage_read_backend_ops << ','
        << m.final_stage_read_backend_bytes << ','
        << m.final_stage_read_backend_seconds << ','
        << m.temp_write_backend_ops << ','
        << m.temp_write_backend_bytes << ','
        << m.temp_write_backend_seconds << ','
        << m.temp_read_backend_ops << ','
        << m.temp_read_backend_bytes << ','
        << m.temp_read_backend_seconds << ','
        << m.current_position_read_bytes << ',' << m.future2_position_read_bytes << ','
        << m.future2_success_read_bytes << ',' << m.future4_position_read_bytes << ','
        << m.future4_success_read_bytes << ','
        << m.current_position_requested_extents << ','
        << m.current_position_coalesced_extents << ','
        << m.current_position_requested_bytes << ','
        << m.current_position_backend_ops << ','
        << m.current_position_backend_bytes << ','
        << m.current_position_backend_seconds << ','
        << m.future2_position_requested_extents << ','
        << m.future2_position_coalesced_extents << ','
        << m.future2_position_requested_bytes << ','
        << m.future2_position_backend_ops << ','
        << m.future2_position_backend_bytes << ','
        << m.future2_position_backend_seconds << ','
        << m.future2_success_requested_extents << ','
        << m.future2_success_coalesced_extents << ','
        << m.future2_success_requested_bytes << ','
        << m.future2_success_backend_ops << ','
        << m.future2_success_backend_bytes << ','
        << m.future2_success_backend_seconds << ','
        << m.future4_position_requested_extents << ','
        << m.future4_position_coalesced_extents << ','
        << m.future4_position_requested_bytes << ','
        << m.future4_position_backend_ops << ','
        << m.future4_position_backend_bytes << ','
        << m.future4_position_backend_seconds << ','
        << m.future4_success_requested_extents << ','
        << m.future4_success_coalesced_extents << ','
        << m.future4_success_requested_bytes << ','
        << m.future4_success_backend_ops << ','
        << m.future4_success_backend_bytes << ','
        << m.future4_success_backend_seconds << ','
        << m.output_position_write_bytes << ','
        << m.output_success_write_bytes << ','
        << m.open_seconds << ','
        << m.open_current_position_seconds << ','
        << m.open_future2_position_seconds << ','
        << m.open_future2_success_seconds << ','
        << m.open_future4_position_seconds << ','
        << m.open_future4_success_seconds << ','
        << m.descriptor_rows_seconds << ','
        << m.partition_seconds << ',' << m.writer_open_seconds << ','
        << m.solve_call_seconds << ',' << solve_untracked << ','
        << driver_untracked << ','
        << m.current_position_read_seconds << ','
        << m.future2_position_read_seconds << ',' << m.future2_success_read_seconds << ','
        << m.future2_index_seconds << ',' << m.future4_position_read_seconds << ','
        << m.future4_success_read_seconds << ',' << m.future4_index_seconds << ','
        << m.temp_prepare_seconds << ',' << m.temp_write_seconds << ','
        << m.temp_read_prepare_seconds << ',' << m.temp_read_seconds << ','
        << m.final_stage_write_seconds << ','
        << m.final_stage_read_seconds << ',' << m.output_pending_seconds << ','
        << m.compact_seconds << ',' << m.result_assembly_seconds << ','
        << m.current_plan_seconds << ',' << m.future_release_seconds << ','
        << m.workspace_release_seconds << ',' << m.family_plan_seconds << ','
        << m.family_workspace_prepare_seconds << ',' << m.family_mark_empty_seconds << ','
        << m.family_current_layout_seconds << ','
        << m.family_future4_prepare_overhead_seconds << ','
        << m.family_future2_prepare_overhead_seconds << ','
        << m.family_future4_prepare_normalize_seconds << ','
        << m.family_future2_prepare_normalize_seconds << ','
        << m.family_future4_prepare_select_seconds << ','
        << m.family_future2_prepare_select_seconds << ','
        << m.family_future4_prepare_index_build_seconds << ','
        << m.family_future2_prepare_index_build_seconds << ','
        << m.family_future4_prepare_insert_sort_seconds << ','
        << m.family_future2_prepare_insert_sort_seconds << ','
        << m.family_future4_lookup_copy_seconds << ','
        << m.family_future2_lookup_copy_seconds << ','
        << m.family_spawn4_phase_wall_seconds << ','
        << m.family_spawn2_phase_wall_seconds << ','
        << m.family_spawn4_phase_untracked_seconds << ','
        << m.family_spawn2_phase_untracked_seconds << ','
        << m.family_spawn4_cell_compute_seconds << ','
        << m.family_spawn2_cell_compute_seconds << ','
        << m.family_spawn4_bucket_hit_seconds << ','
        << m.family_spawn2_bucket_hit_seconds << ','
        << m.family_final_dense_copy_seconds << ','
        << m.family_compact_value_copy_seconds << ','
        << m.family_pending_mark_seconds << ','
        << m.family_output_finish_seconds << ','
        << m.family_output_streamer_open_seconds << ','
        << m.family_temp_open_seconds << ',' << m.family_temp_close_seconds << ','
        << m.family_future_release_all_seconds << ','
        << m.family_future_release_except_seconds << ','
        << m.family_future_release_all_clear_seconds << ','
        << m.family_future_release_except_normalize_seconds << ','
        << m.family_future_release_except_filter_seconds << ','
        << m.family_future_release_except_erase_seconds << ','
        << m.family_future_release_except_ids_seconds << ','
        << m.family_workspace_release_spawn4_partial_seconds << ','
        << m.family_workspace_release_spawn4_scratch_seconds << ','
        << m.family_workspace_release_spawn4_temp_values_seconds << ','
        << m.family_workspace_release_spawn2_dense_seconds << ','
        << m.family_workspace_release_spawn2_prefetch_seconds << ','
        << m.family_workspace_release_spawn2_temp_values_seconds << ','
        << m.partial_cleanup_seconds << ','
        << m.position_write_seconds << ',' << m.success_write_seconds << ','
        << m.writer_close_seconds << ',' << m.post_resize_seconds << ','
        << accounted << ',' << (m.total_seconds - accounted) << ','
        << m.total_seconds << ',' << total_mrows << ',' << final_mrows << ','
        << recalc_mrows << ','
        << gbps(m.current_position_read_bytes, m.current_position_read_seconds) << ','
        << gbps(m.future2_position_read_bytes, m.future2_position_read_seconds) << ','
        << gbps(m.future2_success_read_bytes, m.future2_success_read_seconds) << ','
        << gbps(m.future4_position_read_bytes, m.future4_position_read_seconds) << ','
        << gbps(m.future4_success_read_bytes, m.future4_success_read_seconds) << ','
        << gbps(m.temp_bytes_written, m.temp_write_seconds) << ','
        << gbps(m.temp_bytes_read, m.temp_read_seconds) << ','
        << gbps(m.final_stage_bytes_written, m.final_stage_write_seconds) << ','
        << gbps(m.final_stage_bytes_read, m.final_stage_read_seconds) << ','
        << gbps(m.output_position_write_bytes, m.position_write_seconds) << ','
        << gbps(m.output_success_write_bytes, m.success_write_seconds) << ','
        << gbps(m.current_position_backend_bytes, m.current_position_backend_seconds) << ','
        << gbps(m.future2_position_backend_bytes, m.future2_position_backend_seconds) << ','
        << gbps(m.future2_success_backend_bytes, m.future2_success_backend_seconds) << ','
        << gbps(m.future4_position_backend_bytes, m.future4_position_backend_seconds) << ','
        << gbps(m.future4_success_backend_bytes, m.future4_success_backend_seconds) << ','
        << gbps(m.temp_write_backend_bytes, m.temp_write_backend_seconds) << ','
        << gbps(m.temp_read_backend_bytes, m.temp_read_backend_seconds) << ','
        << gbps(m.final_stage_write_backend_bytes, m.final_stage_write_backend_seconds) << ','
        << gbps(m.final_stage_read_backend_bytes, m.final_stage_read_backend_seconds)
        << '\n';
}

void add_to_summary(LayerMetric &dst, const LayerMetric &src) {
    dst.current_rows += src.current_rows;
    dst.live_rows += src.live_rows;
    dst.zero_pruned_rows += src.zero_pruned_rows;
    dst.position_bytes += src.position_bytes;
    dst.success_bytes += src.success_bytes;
    dst.current_chunks += src.current_chunks;
    dst.current_cells += src.current_cells;
    dst.current_work_items += src.current_work_items;
    dst.family_current_cells_loaded += src.family_current_cells_loaded;
    dst.future2_batch_loads += src.future2_batch_loads;
    dst.future4_batch_loads += src.future4_batch_loads;
    dst.future2_cells_loaded += src.future2_cells_loaded;
    dst.future4_cells_loaded += src.future4_cells_loaded;
    dst.future2_active_cells_max = std::max(dst.future2_active_cells_max, src.future2_active_cells_max);
    dst.future4_active_cells_max = std::max(dst.future4_active_cells_max, src.future4_active_cells_max);
    dst.future2_position_resident_bytes =
        std::max(dst.future2_position_resident_bytes, src.future2_position_resident_bytes);
    dst.future2_success_resident_bytes =
        std::max(dst.future2_success_resident_bytes, src.future2_success_resident_bytes);
    dst.future4_position_resident_bytes =
        std::max(dst.future4_position_resident_bytes, src.future4_position_resident_bytes);
    dst.future4_success_resident_bytes =
        std::max(dst.future4_success_resident_bytes, src.future4_success_resident_bytes);
    dst.future_resident_bytes_max =
        std::max(dst.future_resident_bytes_max, src.future_resident_bytes_max);
    dst.spawn4_passes += src.spawn4_passes;
    dst.spawn2_passes += src.spawn2_passes;
    dst.spawn4_reuse_windows += src.spawn4_reuse_windows;
    dst.spawn2_reuse_windows += src.spawn2_reuse_windows;
    dst.future_release_all_calls += src.future_release_all_calls;
    dst.future_release_except_calls += src.future_release_except_calls;
    dst.partial4_cells_written += src.partial4_cells_written;
    dst.partial4_cells_read += src.partial4_cells_read;
    dst.partial2_cells_written += src.partial2_cells_written;
    dst.partial2_cells_read += src.partial2_cells_read;
    dst.scratch4_cells_written += src.scratch4_cells_written;
    dst.scratch4_cells_read += src.scratch4_cells_read;
    dst.finalized_cells += src.finalized_cells;
    dst.pending_cells_max = std::max(dst.pending_cells_max, src.pending_cells_max);
    dst.temp_bytes_written += src.temp_bytes_written;
    dst.temp_bytes_read += src.temp_bytes_read;
    dst.final_stage_bytes_written += src.final_stage_bytes_written;
    dst.final_stage_bytes_read += src.final_stage_bytes_read;
    dst.final_stage_write_backend_ops += src.final_stage_write_backend_ops;
    dst.final_stage_write_backend_bytes += src.final_stage_write_backend_bytes;
    dst.final_stage_write_backend_seconds += src.final_stage_write_backend_seconds;
    dst.final_stage_read_backend_ops += src.final_stage_read_backend_ops;
    dst.final_stage_read_backend_bytes += src.final_stage_read_backend_bytes;
    dst.final_stage_read_backend_seconds += src.final_stage_read_backend_seconds;
    dst.temp_write_backend_ops += src.temp_write_backend_ops;
    dst.temp_write_backend_bytes += src.temp_write_backend_bytes;
    dst.temp_write_backend_seconds += src.temp_write_backend_seconds;
    dst.temp_read_backend_ops += src.temp_read_backend_ops;
    dst.temp_read_backend_bytes += src.temp_read_backend_bytes;
    dst.temp_read_backend_seconds += src.temp_read_backend_seconds;
    dst.current_position_read_bytes += src.current_position_read_bytes;
    dst.future2_position_read_bytes += src.future2_position_read_bytes;
    dst.future2_success_read_bytes += src.future2_success_read_bytes;
    dst.future4_position_read_bytes += src.future4_position_read_bytes;
    dst.future4_success_read_bytes += src.future4_success_read_bytes;
    dst.current_position_requested_extents += src.current_position_requested_extents;
    dst.current_position_coalesced_extents += src.current_position_coalesced_extents;
    dst.current_position_requested_bytes += src.current_position_requested_bytes;
    dst.current_position_backend_ops += src.current_position_backend_ops;
    dst.current_position_backend_bytes += src.current_position_backend_bytes;
    dst.current_position_backend_seconds += src.current_position_backend_seconds;
    dst.future2_position_requested_extents += src.future2_position_requested_extents;
    dst.future2_position_coalesced_extents += src.future2_position_coalesced_extents;
    dst.future2_position_requested_bytes += src.future2_position_requested_bytes;
    dst.future2_position_backend_ops += src.future2_position_backend_ops;
    dst.future2_position_backend_bytes += src.future2_position_backend_bytes;
    dst.future2_position_backend_seconds += src.future2_position_backend_seconds;
    dst.future2_success_requested_extents += src.future2_success_requested_extents;
    dst.future2_success_coalesced_extents += src.future2_success_coalesced_extents;
    dst.future2_success_requested_bytes += src.future2_success_requested_bytes;
    dst.future2_success_backend_ops += src.future2_success_backend_ops;
    dst.future2_success_backend_bytes += src.future2_success_backend_bytes;
    dst.future2_success_backend_seconds += src.future2_success_backend_seconds;
    dst.future4_position_requested_extents += src.future4_position_requested_extents;
    dst.future4_position_coalesced_extents += src.future4_position_coalesced_extents;
    dst.future4_position_requested_bytes += src.future4_position_requested_bytes;
    dst.future4_position_backend_ops += src.future4_position_backend_ops;
    dst.future4_position_backend_bytes += src.future4_position_backend_bytes;
    dst.future4_position_backend_seconds += src.future4_position_backend_seconds;
    dst.future4_success_requested_extents += src.future4_success_requested_extents;
    dst.future4_success_coalesced_extents += src.future4_success_coalesced_extents;
    dst.future4_success_requested_bytes += src.future4_success_requested_bytes;
    dst.future4_success_backend_ops += src.future4_success_backend_ops;
    dst.future4_success_backend_bytes += src.future4_success_backend_bytes;
    dst.future4_success_backend_seconds += src.future4_success_backend_seconds;
    dst.output_position_write_bytes += src.output_position_write_bytes;
    dst.output_success_write_bytes += src.output_success_write_bytes;
    dst.open_seconds += src.open_seconds;
    dst.open_current_position_seconds += src.open_current_position_seconds;
    dst.open_future2_position_seconds += src.open_future2_position_seconds;
    dst.open_future2_success_seconds += src.open_future2_success_seconds;
    dst.open_future4_position_seconds += src.open_future4_position_seconds;
    dst.open_future4_success_seconds += src.open_future4_success_seconds;
    dst.descriptor_rows_seconds += src.descriptor_rows_seconds;
    dst.partition_seconds += src.partition_seconds;
    dst.writer_open_seconds += src.writer_open_seconds;
    dst.solve_call_seconds += src.solve_call_seconds;
    dst.current_position_read_seconds += src.current_position_read_seconds;
    dst.future2_position_read_seconds += src.future2_position_read_seconds;
    dst.future2_success_read_seconds += src.future2_success_read_seconds;
    dst.future2_index_seconds += src.future2_index_seconds;
    dst.future4_position_read_seconds += src.future4_position_read_seconds;
    dst.future4_success_read_seconds += src.future4_success_read_seconds;
    dst.future4_index_seconds += src.future4_index_seconds;
    dst.temp_prepare_seconds += src.temp_prepare_seconds;
    dst.temp_write_seconds += src.temp_write_seconds;
    dst.temp_read_prepare_seconds += src.temp_read_prepare_seconds;
    dst.temp_read_seconds += src.temp_read_seconds;
    dst.final_stage_write_seconds += src.final_stage_write_seconds;
    dst.final_stage_read_seconds += src.final_stage_read_seconds;
    dst.output_pending_seconds += src.output_pending_seconds;
    dst.compact_seconds += src.compact_seconds;
    dst.result_assembly_seconds += src.result_assembly_seconds;
    dst.current_plan_seconds += src.current_plan_seconds;
    dst.future_release_seconds += src.future_release_seconds;
    dst.workspace_release_seconds += src.workspace_release_seconds;
    dst.family_plan_seconds += src.family_plan_seconds;
    dst.family_workspace_prepare_seconds += src.family_workspace_prepare_seconds;
    dst.family_mark_empty_seconds += src.family_mark_empty_seconds;
    dst.family_current_layout_seconds += src.family_current_layout_seconds;
    dst.family_future4_prepare_overhead_seconds += src.family_future4_prepare_overhead_seconds;
    dst.family_future2_prepare_overhead_seconds += src.family_future2_prepare_overhead_seconds;
    dst.family_future4_prepare_normalize_seconds += src.family_future4_prepare_normalize_seconds;
    dst.family_future2_prepare_normalize_seconds += src.family_future2_prepare_normalize_seconds;
    dst.family_future4_prepare_select_seconds += src.family_future4_prepare_select_seconds;
    dst.family_future2_prepare_select_seconds += src.family_future2_prepare_select_seconds;
    dst.family_future4_prepare_index_build_seconds +=
        src.family_future4_prepare_index_build_seconds;
    dst.family_future2_prepare_index_build_seconds +=
        src.family_future2_prepare_index_build_seconds;
    dst.family_future4_prepare_insert_sort_seconds +=
        src.family_future4_prepare_insert_sort_seconds;
    dst.family_future2_prepare_insert_sort_seconds +=
        src.family_future2_prepare_insert_sort_seconds;
    dst.family_future4_lookup_copy_seconds += src.family_future4_lookup_copy_seconds;
    dst.family_future2_lookup_copy_seconds += src.family_future2_lookup_copy_seconds;
    dst.family_spawn4_phase_wall_seconds += src.family_spawn4_phase_wall_seconds;
    dst.family_spawn2_phase_wall_seconds += src.family_spawn2_phase_wall_seconds;
    dst.family_spawn4_phase_untracked_seconds += src.family_spawn4_phase_untracked_seconds;
    dst.family_spawn2_phase_untracked_seconds += src.family_spawn2_phase_untracked_seconds;
    dst.family_spawn4_cell_compute_seconds += src.family_spawn4_cell_compute_seconds;
    dst.family_spawn2_cell_compute_seconds += src.family_spawn2_cell_compute_seconds;
    dst.family_spawn4_bucket_hit_seconds += src.family_spawn4_bucket_hit_seconds;
    dst.family_spawn2_bucket_hit_seconds += src.family_spawn2_bucket_hit_seconds;
    dst.family_final_dense_copy_seconds += src.family_final_dense_copy_seconds;
    dst.family_compact_value_copy_seconds += src.family_compact_value_copy_seconds;
    dst.family_pending_mark_seconds += src.family_pending_mark_seconds;
    dst.family_output_finish_seconds += src.family_output_finish_seconds;
    dst.family_output_streamer_open_seconds += src.family_output_streamer_open_seconds;
    dst.family_temp_open_seconds += src.family_temp_open_seconds;
    dst.family_temp_close_seconds += src.family_temp_close_seconds;
    dst.family_future_release_all_seconds += src.family_future_release_all_seconds;
    dst.family_future_release_except_seconds += src.family_future_release_except_seconds;
    dst.family_future_release_all_clear_seconds += src.family_future_release_all_clear_seconds;
    dst.family_future_release_except_normalize_seconds +=
        src.family_future_release_except_normalize_seconds;
    dst.family_future_release_except_filter_seconds +=
        src.family_future_release_except_filter_seconds;
    dst.family_future_release_except_erase_seconds +=
        src.family_future_release_except_erase_seconds;
    dst.family_future_release_except_ids_seconds +=
        src.family_future_release_except_ids_seconds;
    dst.family_workspace_release_spawn4_partial_seconds +=
        src.family_workspace_release_spawn4_partial_seconds;
    dst.family_workspace_release_spawn4_scratch_seconds +=
        src.family_workspace_release_spawn4_scratch_seconds;
    dst.family_workspace_release_spawn4_temp_values_seconds +=
        src.family_workspace_release_spawn4_temp_values_seconds;
    dst.family_workspace_release_spawn2_dense_seconds +=
        src.family_workspace_release_spawn2_dense_seconds;
    dst.family_workspace_release_spawn2_prefetch_seconds +=
        src.family_workspace_release_spawn2_prefetch_seconds;
    dst.family_workspace_release_spawn2_temp_values_seconds +=
        src.family_workspace_release_spawn2_temp_values_seconds;
    dst.partial_cleanup_seconds += src.partial_cleanup_seconds;
    dst.position_write_seconds += src.position_write_seconds;
    dst.success_write_seconds += src.success_write_seconds;
    dst.writer_close_seconds += src.writer_close_seconds;
    dst.post_resize_seconds += src.post_resize_seconds;
    dst.total_seconds += src.total_seconds;
}

template <typename StorageT>
void run_bench_typed(const Args &args) {
        if (!BC::bc_success_dtype_matches_type<StorageT>(args.success_dtype)) {
            throw std::invalid_argument("bench storage type does not match success dtype");
        }
        {
        BC::BCFamilySolveRunOptions run_options;
        run_options.generated_position_dir = args.generated_position_dir;
        run_options.solved_output_dir = args.solved_output_dir;
        run_options.prefix = args.prefix;
        run_options.target_rank = args.target_rank;
        run_options.success_target_rank = args.success_target_rank;
        run_options.canonical_symm_mode = args.canonical_symm_mode;
        run_options.spawn_rate4 = args.spawn_rate4;
        run_options.num_threads = args.num_threads;
        run_options.canonical_batch_size = args.canonical_batch_size;
        run_options.family_modulus = args.family_modulus;
        run_options.future_reuse_max_families = args.future_reuse_max_families;
        run_options.future_index_recycle_max_bytes = args.future_index_recycle_max_bytes;
        run_options.source_words_per_item = args.source_words_per_item;
        run_options.work_schedule_chunk = args.work_schedule_chunk;
        run_options.cell_parallel_min_work_items = args.cell_parallel_min_work_items;
        run_options.final_pending_value_memory_cap_bytes =
            args.final_pending_value_memory_cap_bytes;
        run_options.success_dtype = args.success_dtype;
        run_options.direct_queue_depth = args.direct_queue_depth;
        run_options.start_ordinal = args.start_ordinal;
        run_options.min_ordinal = args.min_ordinal;
        run_options.direct_io = args.direct_io;
        run_options.keep_direct_padding = args.keep_direct_padding;
        run_options.resume_from_checkpoint = false;

        std::filesystem::create_directories(args.solved_output_dir);
        if (!args.stats_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.stats_csv.parent_path());
        }
        if (!args.summary_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.summary_csv.parent_path());
        }
        std::ofstream stats(args.stats_csv);
        if (!stats) {
            throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
        }
        stats << std::setprecision(9);
        write_stats_header(stats);

        auto convert_metric = [](const BC::BCFamilySolveRunLayerMetric &src) {
            LayerMetric metric;
            metric.kind = src.kind;
            metric.ordinal = src.ordinal;
            metric.layer_sum = src.layer_sum;
            metric.current_rows = src.current_rows;
            metric.live_rows = src.live_rows;
            metric.zero_pruned_rows = src.zero_pruned_rows;
            metric.position_bytes = src.position_bytes;
            metric.success_bytes = src.success_bytes;
            metric.output_position_write_bytes = src.output_position_write_bytes;
            metric.output_success_write_bytes = src.output_success_write_bytes;
            metric.open_seconds = src.open_seconds;
            metric.open_current_position_seconds = src.open_current_position_seconds;
            metric.open_future2_position_seconds = src.open_future2_position_seconds;
            metric.open_future2_success_seconds = src.open_future2_success_seconds;
            metric.open_future4_position_seconds = src.open_future4_position_seconds;
            metric.open_future4_success_seconds = src.open_future4_success_seconds;
            metric.descriptor_rows_seconds = src.descriptor_rows_seconds;
            metric.partition_seconds = src.partition_seconds;
            metric.writer_open_seconds = src.writer_open_seconds;
            metric.solve_call_seconds = src.solve_call_seconds;
            metric.position_write_seconds = src.position_write_seconds;
            metric.success_write_seconds = src.success_write_seconds;
            metric.writer_close_seconds = src.writer_close_seconds;
            metric.post_resize_seconds = src.post_resize_seconds;
            metric.total_seconds = src.total_seconds;
            if (!src.has_family_stats) {
                return metric;
            }
            const BC::BCFamilySolveStats &fs = src.family_stats;
            const BC::BCSingleChunkSolveStats &s = fs.single;
            metric.current_chunks = s.current_chunks;
            metric.current_cells = s.current_cells;
            metric.current_work_items = s.current_work_items;
            metric.family_current_cells_loaded = fs.family_current_cells_loaded;
            metric.future2_batch_loads = s.future2_batch_loads;
            metric.future4_batch_loads = s.future4_batch_loads;
            metric.future2_cells_loaded = s.future2_cells_loaded;
            metric.future4_cells_loaded = s.future4_cells_loaded;
            metric.future2_active_cells_max = s.future2_active_cells_max;
            metric.future4_active_cells_max = s.future4_active_cells_max;
            metric.future2_position_resident_bytes = s.future2_position_resident_bytes;
            metric.future2_success_resident_bytes = s.future2_success_resident_bytes;
            metric.future4_position_resident_bytes = s.future4_position_resident_bytes;
            metric.future4_success_resident_bytes = s.future4_success_resident_bytes;
            metric.future_resident_bytes_max = s.future_resident_bytes_max;
            metric.spawn4_passes = fs.spawn4_passes;
            metric.spawn2_passes = fs.spawn2_passes;
            metric.spawn4_reuse_windows = fs.spawn4_future_reuse_groups;
            metric.spawn2_reuse_windows = fs.spawn2_future_reuse_groups;
            metric.future_release_all_calls = fs.future_release_all_calls;
            metric.future_release_except_calls = fs.future_release_except_calls;
            metric.partial4_cells_written = fs.partial4_cells_written;
            metric.partial4_cells_read = fs.partial4_cells_read;
            metric.partial2_cells_written = fs.partial2_cells_written;
            metric.partial2_cells_read = fs.partial2_cells_read;
            metric.scratch4_cells_written = fs.scratch4_cells_written;
            metric.scratch4_cells_read = fs.scratch4_cells_read;
            metric.finalized_cells = fs.finalized_cells;
            metric.pending_cells_max = fs.pending_cells_max;
            metric.temp_bytes_written = fs.temp_bytes_written;
            metric.temp_bytes_read = fs.temp_bytes_read;
            metric.final_stage_bytes_written = fs.final_stage_bytes_written;
            metric.final_stage_bytes_read = fs.final_stage_bytes_read;
            metric.final_stage_write_backend_ops = fs.final_stage_write_io.backend_io_count;
            metric.final_stage_write_backend_bytes = fs.final_stage_write_io.backend_bytes;
            metric.final_stage_write_backend_seconds = fs.final_stage_write_io.backend_seconds;
            metric.final_stage_read_backend_ops = fs.final_stage_read_io.backend_io_count;
            metric.final_stage_read_backend_bytes = fs.final_stage_read_io.backend_bytes;
            metric.final_stage_read_backend_seconds = fs.final_stage_read_io.backend_seconds;
            metric.temp_write_backend_ops = fs.temp_write_io.backend_io_count;
            metric.temp_write_backend_bytes = fs.temp_write_io.backend_bytes;
            metric.temp_write_backend_seconds = fs.temp_write_io.backend_seconds;
            metric.temp_read_backend_ops = fs.temp_read_io.backend_io_count;
            metric.temp_read_backend_bytes = fs.temp_read_io.backend_bytes;
            metric.temp_read_backend_seconds = fs.temp_read_io.backend_seconds;
            metric.current_position_read_bytes = s.current_position_load.read_bytes;
            metric.future2_position_read_bytes = s.future2_position_load.read_bytes;
            metric.future2_success_read_bytes = s.future2_success_load.read_bytes;
            metric.future4_position_read_bytes = s.future4_position_load.read_bytes;
            metric.future4_success_read_bytes = s.future4_success_load.read_bytes;
            metric.current_position_requested_extents = s.current_position_load.requested_extents;
            metric.current_position_coalesced_extents = s.current_position_load.coalesced_extents;
            metric.current_position_requested_bytes = s.current_position_load.requested_bytes;
            metric.current_position_backend_ops = s.current_position_load.backend_read_ops;
            metric.current_position_backend_bytes = s.current_position_load.backend_read_bytes;
            metric.current_position_backend_seconds = s.current_position_load.backend_read_seconds;
            metric.future2_position_requested_extents = s.future2_position_load.requested_extents;
            metric.future2_position_coalesced_extents = s.future2_position_load.coalesced_extents;
            metric.future2_position_requested_bytes = s.future2_position_load.requested_bytes;
            metric.future2_position_backend_ops = s.future2_position_load.backend_read_ops;
            metric.future2_position_backend_bytes = s.future2_position_load.backend_read_bytes;
            metric.future2_position_backend_seconds = s.future2_position_load.backend_read_seconds;
            metric.future2_success_requested_extents = s.future2_success_load.requested_extents;
            metric.future2_success_coalesced_extents = s.future2_success_load.coalesced_extents;
            metric.future2_success_requested_bytes = s.future2_success_load.requested_bytes;
            metric.future2_success_backend_ops = s.future2_success_load.backend_read_ops;
            metric.future2_success_backend_bytes = s.future2_success_load.backend_read_bytes;
            metric.future2_success_backend_seconds = s.future2_success_load.backend_read_seconds;
            metric.future4_position_requested_extents = s.future4_position_load.requested_extents;
            metric.future4_position_coalesced_extents = s.future4_position_load.coalesced_extents;
            metric.future4_position_requested_bytes = s.future4_position_load.requested_bytes;
            metric.future4_position_backend_ops = s.future4_position_load.backend_read_ops;
            metric.future4_position_backend_bytes = s.future4_position_load.backend_read_bytes;
            metric.future4_position_backend_seconds = s.future4_position_load.backend_read_seconds;
            metric.future4_success_requested_extents = s.future4_success_load.requested_extents;
            metric.future4_success_coalesced_extents = s.future4_success_load.coalesced_extents;
            metric.future4_success_requested_bytes = s.future4_success_load.requested_bytes;
            metric.future4_success_backend_ops = s.future4_success_load.backend_read_ops;
            metric.future4_success_backend_bytes = s.future4_success_load.backend_read_bytes;
            metric.future4_success_backend_seconds = s.future4_success_load.backend_read_seconds;
            metric.current_position_read_seconds = s.current_position_read_seconds;
            metric.future2_position_read_seconds = s.future2_position_read_seconds;
            metric.future2_success_read_seconds = s.future2_success_read_seconds;
            metric.future2_index_seconds = s.future2_index_seconds;
            metric.future4_position_read_seconds = s.future4_position_read_seconds;
            metric.future4_success_read_seconds = s.future4_success_read_seconds;
            metric.future4_index_seconds = s.future4_index_seconds;
            metric.temp_prepare_seconds = s.temp_prepare_seconds;
            metric.temp_write_seconds = fs.temp_write_seconds;
            metric.temp_read_prepare_seconds = fs.temp_read_prepare_seconds;
            metric.temp_read_seconds = fs.temp_read_seconds;
            metric.final_stage_write_seconds = fs.final_stage_write_seconds;
            metric.final_stage_read_seconds = fs.final_stage_read_seconds;
            metric.output_pending_seconds = fs.output_pending_seconds;
            metric.compact_seconds = s.compact_seconds;
            metric.result_assembly_seconds = s.result_assembly_seconds;
            metric.current_plan_seconds = s.current_plan_seconds;
            metric.future_release_seconds = s.future_release_seconds;
            metric.workspace_release_seconds = s.workspace_release_seconds;
            metric.family_plan_seconds = fs.plan_seconds;
            metric.family_workspace_prepare_seconds = fs.workspace_prepare_seconds;
            metric.family_mark_empty_seconds = fs.mark_empty_seconds;
            metric.family_current_layout_seconds = fs.current_layout_seconds;
            metric.family_future4_prepare_overhead_seconds = fs.future4_prepare_overhead_seconds;
            metric.family_future2_prepare_overhead_seconds = fs.future2_prepare_overhead_seconds;
            metric.family_future4_prepare_normalize_seconds = fs.future4_prepare_normalize_seconds;
            metric.family_future2_prepare_normalize_seconds = fs.future2_prepare_normalize_seconds;
            metric.family_future4_prepare_select_seconds = fs.future4_prepare_select_seconds;
            metric.family_future2_prepare_select_seconds = fs.future2_prepare_select_seconds;
            metric.family_future4_prepare_index_build_seconds =
                fs.future4_prepare_index_build_seconds;
            metric.family_future2_prepare_index_build_seconds =
                fs.future2_prepare_index_build_seconds;
            metric.family_future4_prepare_insert_sort_seconds =
                fs.future4_prepare_insert_sort_seconds;
            metric.family_future2_prepare_insert_sort_seconds =
                fs.future2_prepare_insert_sort_seconds;
            metric.family_future4_lookup_copy_seconds = fs.future4_lookup_copy_seconds;
            metric.family_future2_lookup_copy_seconds = fs.future2_lookup_copy_seconds;
            metric.family_spawn4_phase_wall_seconds = fs.spawn4_phase_wall_seconds;
            metric.family_spawn2_phase_wall_seconds = fs.spawn2_phase_wall_seconds;
            metric.family_spawn4_phase_untracked_seconds = fs.spawn4_phase_untracked_seconds;
            metric.family_spawn2_phase_untracked_seconds = fs.spawn2_phase_untracked_seconds;
            metric.family_spawn4_cell_compute_seconds = fs.spawn4_cell_compute_seconds;
            metric.family_spawn2_cell_compute_seconds = fs.spawn2_cell_compute_seconds;
            metric.family_spawn4_bucket_hit_seconds = fs.spawn4_bucket_hit_seconds;
            metric.family_spawn2_bucket_hit_seconds = fs.spawn2_bucket_hit_seconds;
            metric.family_final_dense_copy_seconds = fs.final_dense_copy_seconds;
            metric.family_compact_value_copy_seconds = fs.compact_value_copy_seconds;
            metric.family_pending_mark_seconds = fs.pending_mark_seconds;
            metric.family_output_finish_seconds = fs.output_finish_seconds;
            metric.family_output_streamer_open_seconds = fs.output_streamer_open_seconds;
            metric.family_temp_open_seconds = fs.temp_open_seconds;
            metric.family_temp_close_seconds = fs.temp_close_seconds;
            metric.family_future_release_all_seconds = fs.future_release_all_seconds;
            metric.family_future_release_except_seconds = fs.future_release_except_seconds;
            metric.family_future_release_all_clear_seconds = fs.future_release_all_clear_seconds;
            metric.family_future_release_except_normalize_seconds =
                fs.future_release_except_normalize_seconds;
            metric.family_future_release_except_filter_seconds =
                fs.future_release_except_filter_seconds;
            metric.family_future_release_except_erase_seconds =
                fs.future_release_except_erase_seconds;
            metric.family_future_release_except_ids_seconds =
                fs.future_release_except_ids_seconds;
            metric.family_workspace_release_spawn4_partial_seconds =
                fs.workspace_release_spawn4_partial_seconds;
            metric.family_workspace_release_spawn4_scratch_seconds =
                fs.workspace_release_spawn4_scratch_seconds;
            metric.family_workspace_release_spawn4_temp_values_seconds =
                fs.workspace_release_spawn4_temp_values_seconds;
            metric.family_workspace_release_spawn2_dense_seconds =
                fs.workspace_release_spawn2_dense_seconds;
            metric.family_workspace_release_spawn2_prefetch_seconds =
                fs.workspace_release_spawn2_prefetch_seconds;
            metric.family_workspace_release_spawn2_temp_values_seconds =
                fs.workspace_release_spawn2_temp_values_seconds;
            metric.partial_cleanup_seconds = s.partial_cleanup_seconds;
            return metric;
        };

        LayerMetric summary;
        summary.kind = "summary";
        const double run_t0 = now_seconds();
        BC::BCFamilySolveRunResult run_result =
            BC::bc_family_solve_full_run_typed<StorageT>(
                run_options,
                [&](const BC::BCFamilySolveRunLayerMetric &runner_metric) {
                    LayerMetric metric = convert_metric(runner_metric);
                    add_to_summary(summary, metric);
                    write_metric_row(stats, metric);
                    stats.flush();
                    std::cout << std::setprecision(9)
                        << "kind=" << metric.kind
                        << " ordinal=" << metric.ordinal
                        << " rows=" << metric.current_rows
                        << " live_rows=" << metric.live_rows
                        << " total_seconds=" << metric.total_seconds
                        << " accounted_seconds=" << accounted_seconds(metric)
                        << " untracked_seconds="
                        << (metric.total_seconds - accounted_seconds(metric))
                        << " position_bytes=" << metric.position_bytes
                        << " success_bytes=" << metric.success_bytes
                        << '\n';
                });
        summary.total_seconds = now_seconds() - run_t0;
        write_metric_row(stats, summary);
        stats.flush();

        std::ofstream summary_out(args.summary_csv);
        if (!summary_out) {
            throw std::runtime_error("failed to open summary csv: " + args.summary_csv.string());
        }
        summary_out << std::setprecision(12);
        summary_out
            << "generated_position_dir,solved_output_dir,min_ordinal,max_ordinal,"
            << "completed,total_rows,total_live_rows,total_zero_pruned_rows,"
            << "total_position_bytes,total_success_bytes,total_seconds,total_mrows_per_sec\n"
            << args.generated_position_dir.string() << ','
            << args.solved_output_dir.string() << ','
            << run_result.min_ordinal << ','
            << run_result.max_ordinal << ','
            << (run_result.completed ? 1 : 0) << ','
            << summary.current_rows << ','
            << summary.live_rows << ','
            << summary.zero_pruned_rows << ','
            << summary.position_bytes << ','
            << summary.success_bytes << ','
            << summary.total_seconds << ','
            << (summary.total_seconds > 0.0
                    ? static_cast<double>(summary.current_rows) / summary.total_seconds / 1.0e6
                    : 0.0)
            << '\n';
        return;
        }

        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        const std::vector<uint8_t> success_shifts = all_board_success_shifts();
        const BC::BCQuadrantWordSumTable word_sums = build_word_sum_table(lut);
        const std::vector<BC::LayerSum> possible_8tile_sums =
            BC::build_possible_8tile_sums(
                make_free_legal_tiles(args.target_rank),
                BC::default_2048_tile_sum_values());
        const std::map<uint32_t, LayerFile> layers = discover_layers(args);
        const uint32_t discovered_min_ordinal = layers.begin()->first;
        const uint32_t max_ordinal = layers.rbegin()->first;
        if (discovered_min_ordinal != 0U || max_ordinal < 2U) {
            throw std::runtime_error("expected generated ordinals to start at 0 and include at least 3 layers");
        }
        const uint32_t min_ordinal = args.min_ordinal.value_or(discovered_min_ordinal);
        if (min_ordinal < discovered_min_ordinal || min_ordinal >= max_ordinal) {
            throw std::invalid_argument("--min-ordinal must be within generated range and below max ordinal");
        }

        std::filesystem::create_directories(args.solved_output_dir);
        if (!args.stats_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.stats_csv.parent_path());
        }
        if (!args.summary_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.summary_csv.parent_path());
        }
        std::ofstream stats(args.stats_csv);
        if (!stats) {
            throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
        }
        stats << std::setprecision(9);
        write_stats_header(stats);

        const double all_begin = now_seconds();
        if (!args.start_ordinal.has_value()) {
            const double top_begin = now_seconds();
            const double open_begin = now_seconds();
            BC::BCPositionFileReader top_position =
                open_position_file(args, layers.at(max_ordinal).path, lut);
            const double open_seconds = now_seconds() - open_begin;
            if (top_position.layer().header().family_count != args.family_modulus) {
                throw std::runtime_error("top generated layer is not in requested family modulus");
            }
            BC::BCResidentSolvedLayer<StorageT> top_layer = make_terminal_solved_layer<StorageT>(
                top_position.layer(),
                success_shifts,
                args.success_target_rank,
                args.success_dtype,
                args.num_threads);
            WriteResult write = write_solved_layer_files<StorageT>(args, max_ordinal, top_layer);
            LayerMetric metric;
            metric.kind = "terminal";
            metric.ordinal = max_ordinal;
            metric.layer_sum = top_position.layer().header().layer_sum;
            metric.live_rows = top_layer.compact_stats.live_rows;
            metric.zero_pruned_rows = top_layer.compact_stats.zero_pruned_rows;
            metric.position_bytes = write.position_bytes;
            metric.success_bytes = write.success_bytes;
            metric.open_seconds = open_seconds;
            metric.compact_seconds = top_layer.compact_stats.compact_seconds;
            metric.position_write_seconds = write.position_seconds;
            metric.success_write_seconds = write.success_seconds;
            metric.total_seconds = now_seconds() - top_begin;
            write_metric_row(stats, metric);

            std::vector<uint8_t> empty_position_bytes = make_empty_position_bytes(
                top_position.layer().header().layer_sum + 2U,
                top_position.layer().header().family_unit,
                args.family_modulus);
            BC::BCResidentSolvedLayer<StorageT> empty_layer;
            empty_layer.open(
                empty_position_bytes,
                {},
                lut,
                1U,
                args.success_dtype);
            (void)write_solved_layer_files<StorageT>(args, max_ordinal + 1U, empty_layer);
        } else {
            if (*args.start_ordinal >= max_ordinal) {
                throw std::invalid_argument("--start-ordinal must be less than max generated ordinal");
            }
            for (uint32_t ord : {*args.start_ordinal + 1U, *args.start_ordinal + 2U}) {
                if (!std::filesystem::exists(position_path_for(args, ord)) ||
                    !std::filesystem::exists(success_path_for(args, ord))) {
                    throw std::runtime_error("--start-ordinal requires existing solved future files");
                }
            }
        }

        BC::BCFamilySolveWorkspace<StorageT> workspace;
        LayerMetric summary;
        summary.kind = "summary";
        const int64_t first_solve_ordinal = args.start_ordinal.has_value()
            ? static_cast<int64_t>(*args.start_ordinal)
            : static_cast<int64_t>(max_ordinal) - 1;
        for (int64_t ordinal_signed = first_solve_ordinal;
             ordinal_signed >= static_cast<int64_t>(min_ordinal);
             --ordinal_signed) {
            const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
            const double layer_begin = now_seconds();
            const double open_begin = now_seconds();
            double open_current_position_seconds = 0.0;
            double open_future2_position_seconds = 0.0;
            double open_future2_success_seconds = 0.0;
            double open_future4_position_seconds = 0.0;
            double open_future4_success_seconds = 0.0;
            auto current_future = std::async(
                std::launch::async,
                [&]() {
                    const double begin = now_seconds();
                    BC::BCPositionStreamingReader reader =
                        open_position_stream(args, layers.at(ordinal).path, lut);
                    return std::make_pair(std::move(reader), now_seconds() - begin);
                });
            auto future2_position_future = std::async(
                std::launch::async,
                [&]() {
                    const double begin = now_seconds();
                    BC::BCPositionStreamingReader reader =
                        open_position_stream(args, position_path_for(args, ordinal + 1U), lut);
                    return std::make_pair(std::move(reader), now_seconds() - begin);
                });
            auto future4_position_future = std::async(
                std::launch::async,
                [&]() {
                    const double begin = now_seconds();
                    BC::BCPositionStreamingReader reader =
                        open_position_stream(args, position_path_for(args, ordinal + 2U), lut);
                    return std::make_pair(std::move(reader), now_seconds() - begin);
                });
            auto current_open = current_future.get();
            auto future2_position_open = future2_position_future.get();
            auto future4_position_open = future4_position_future.get();
            BC::BCPositionStreamingReader current = std::move(current_open.first);
            BC::BCPositionStreamingReader future2_position =
                std::move(future2_position_open.first);
            BC::BCPositionStreamingReader future4_position =
                std::move(future4_position_open.first);
            open_current_position_seconds = current_open.second;
            open_future2_position_seconds = future2_position_open.second;
            open_future4_position_seconds = future4_position_open.second;

            auto future2_success_future = std::async(
                std::launch::async,
                [&]() {
                    const double begin = now_seconds();
                    BC::BCSuccessStreamingReader reader =
                        open_success_stream(
                            args,
                            success_path_for(args, ordinal + 1U),
                            future2_position);
                    return std::make_pair(std::move(reader), now_seconds() - begin);
                });
            auto future4_success_future = std::async(
                std::launch::async,
                [&]() {
                    const double begin = now_seconds();
                    BC::BCSuccessStreamingReader reader =
                        open_success_stream(
                            args,
                            success_path_for(args, ordinal + 2U),
                            future4_position);
                    return std::make_pair(std::move(reader), now_seconds() - begin);
                });
            auto future2_success_open = future2_success_future.get();
            auto future4_success_open = future4_success_future.get();
            BC::BCSuccessStreamingReader future2_success = std::move(future2_success_open.first);
            BC::BCSuccessStreamingReader future4_success = std::move(future4_success_open.first);
            open_future2_success_seconds = future2_success_open.second;
            open_future4_success_seconds = future4_success_open.second;
            const double open_seconds = now_seconds() - open_begin;
            const double descriptor_rows_t0 = now_seconds();
            const uint64_t descriptor_row_count = descriptor_rows(current);
            const double descriptor_rows_seconds = now_seconds() - descriptor_rows_t0;

            const double partition_t0 = now_seconds();
            const BC::BCFamilyPartitionLayerMap current_partition =
                partition_for(current, possible_8tile_sums, args.family_modulus);
            const BC::BCFamilyPartitionLayerMap future2_partition =
                partition_for(future2_position, possible_8tile_sums, args.family_modulus);
            const BC::BCFamilyPartitionLayerMap future4_partition =
                partition_for(future4_position, possible_8tile_sums, args.family_modulus);
            const double partition_seconds = now_seconds() - partition_t0;

            BC::BCFamilySolveOptions<StorageT> options;
            options.solve.num_threads = args.num_threads;
            options.solve.row_width = 1U;
            options.solve.set_dtype(args.success_dtype);
            options.solve.edge_options.canonical_batch_size = args.canonical_batch_size;
            options.solve.edge_options.canonical_symm_mode = args.canonical_symm_mode;
            options.solve.edge_options.spawn_rate4 = args.spawn_rate4;
            options.solve.edge_options.success_target_rank = args.success_target_rank;
            options.solve.edge_options.success_shifts = &success_shifts;
            options.solve.edge_options.success_check_all_cells = true;
            options.solve.word_sums = &word_sums;
            options.source_bitmap_words_per_work_item = args.source_words_per_item;
            options.source_work_schedule_chunk = args.work_schedule_chunk;
            options.cell_parallel_min_work_items = args.cell_parallel_min_work_items;
            options.future_reuse_max_families = args.future_reuse_max_families;
            options.future_index_recycle_max_bytes = args.future_index_recycle_max_bytes;
            options.final_pending_value_memory_cap_bytes =
                args.final_pending_value_memory_cap_bytes;
            options.temp_direct_io = args.direct_io && !args.temp_buffered;
            options.force_temp_buffered_io = args.temp_buffered;
            options.temp_direct_queue_depth = args.direct_queue_depth;

            const double writer_open_t0 = now_seconds();
            std::unique_ptr<BC::BCWritableFile> position_writer =
                make_output_writer(args, position_path_for(args, ordinal));
            std::unique_ptr<BC::BCWritableFile> success_writer =
                make_output_writer(args, success_path_for(args, ordinal));
            const double writer_open_seconds = now_seconds() - writer_open_t0;
            const double solve_call_t0 = now_seconds();
            BC::BCFamilySolveFileResult result = BC::bc_family_solve_layer_to_files<StorageT>(
                current,
                future2_position,
                future2_success,
                future4_position,
                future4_success,
                current_partition,
                future2_partition,
                future4_partition,
                *position_writer,
                *success_writer,
                args.solved_output_dir / (args.prefix + std::to_string(ordinal) + "_family_tmp"),
                options,
                &workspace);
            const double solve_call_seconds = now_seconds() - solve_call_t0;
            const double writer_close_t0 = now_seconds();
            position_writer.reset();
            success_writer.reset();
            const double writer_close_seconds = now_seconds() - writer_close_t0;
            double post_resize_seconds = 0.0;
            if (args.direct_io && !args.keep_direct_padding) {
                const double resize_t0 = now_seconds();
                std::filesystem::resize_file(position_path_for(args, ordinal), result.position_bytes);
                std::filesystem::resize_file(success_path_for(args, ordinal), result.success_bytes);
                post_resize_seconds = now_seconds() - resize_t0;
            }

            const BC::BCFamilySolveStats &fs = result.stats;
            const BC::BCSingleChunkSolveStats &s = fs.single;
            LayerMetric metric;
            metric.ordinal = ordinal;
            metric.layer_sum = current.header().layer_sum;
            metric.current_rows = s.current_rows != 0U ? s.current_rows : descriptor_row_count;
            metric.live_rows = s.compact_live_rows;
            metric.zero_pruned_rows = s.compact_zero_pruned_rows;
            metric.position_bytes = result.position_bytes;
            metric.success_bytes = result.success_bytes;
            metric.current_chunks = s.current_chunks;
            metric.current_cells = s.current_cells;
            metric.current_work_items = s.current_work_items;
            metric.family_current_cells_loaded = fs.family_current_cells_loaded;
            metric.future2_batch_loads = s.future2_batch_loads;
            metric.future4_batch_loads = s.future4_batch_loads;
            metric.future2_cells_loaded = s.future2_cells_loaded;
            metric.future4_cells_loaded = s.future4_cells_loaded;
            metric.future2_active_cells_max = s.future2_active_cells_max;
            metric.future4_active_cells_max = s.future4_active_cells_max;
            metric.future2_position_resident_bytes = s.future2_position_resident_bytes;
            metric.future2_success_resident_bytes = s.future2_success_resident_bytes;
            metric.future4_position_resident_bytes = s.future4_position_resident_bytes;
            metric.future4_success_resident_bytes = s.future4_success_resident_bytes;
            metric.future_resident_bytes_max = s.future_resident_bytes_max;
            metric.spawn4_passes = fs.spawn4_passes;
            metric.spawn2_passes = fs.spawn2_passes;
            metric.spawn4_reuse_windows = fs.spawn4_future_reuse_groups;
            metric.spawn2_reuse_windows = fs.spawn2_future_reuse_groups;
            metric.partial4_cells_written = fs.partial4_cells_written;
            metric.partial4_cells_read = fs.partial4_cells_read;
            metric.partial2_cells_written = fs.partial2_cells_written;
            metric.partial2_cells_read = fs.partial2_cells_read;
            metric.scratch4_cells_written = fs.scratch4_cells_written;
            metric.scratch4_cells_read = fs.scratch4_cells_read;
            metric.finalized_cells = fs.finalized_cells;
            metric.pending_cells_max = fs.pending_cells_max;
            metric.temp_bytes_written = fs.temp_bytes_written;
            metric.temp_bytes_read = fs.temp_bytes_read;
            metric.final_stage_bytes_written = fs.final_stage_bytes_written;
            metric.final_stage_bytes_read = fs.final_stage_bytes_read;
            metric.final_stage_write_backend_ops = fs.final_stage_write_io.backend_io_count;
            metric.final_stage_write_backend_bytes = fs.final_stage_write_io.backend_bytes;
            metric.final_stage_write_backend_seconds = fs.final_stage_write_io.backend_seconds;
            metric.final_stage_read_backend_ops = fs.final_stage_read_io.backend_io_count;
            metric.final_stage_read_backend_bytes = fs.final_stage_read_io.backend_bytes;
            metric.final_stage_read_backend_seconds = fs.final_stage_read_io.backend_seconds;
            metric.temp_write_backend_ops = fs.temp_write_io.backend_io_count;
            metric.temp_write_backend_bytes = fs.temp_write_io.backend_bytes;
            metric.temp_write_backend_seconds = fs.temp_write_io.backend_seconds;
            metric.temp_read_backend_ops = fs.temp_read_io.backend_io_count;
            metric.temp_read_backend_bytes = fs.temp_read_io.backend_bytes;
            metric.temp_read_backend_seconds = fs.temp_read_io.backend_seconds;
            metric.current_position_read_bytes = s.current_position_load.read_bytes;
            metric.future2_position_read_bytes = s.future2_position_load.read_bytes;
            metric.future2_success_read_bytes = s.future2_success_load.read_bytes;
            metric.future4_position_read_bytes = s.future4_position_load.read_bytes;
            metric.future4_success_read_bytes = s.future4_success_load.read_bytes;
            metric.current_position_requested_extents = s.current_position_load.requested_extents;
            metric.current_position_coalesced_extents = s.current_position_load.coalesced_extents;
            metric.current_position_requested_bytes = s.current_position_load.requested_bytes;
            metric.current_position_backend_ops = s.current_position_load.backend_read_ops;
            metric.current_position_backend_bytes = s.current_position_load.backend_read_bytes;
            metric.current_position_backend_seconds = s.current_position_load.backend_read_seconds;
            metric.future2_position_requested_extents = s.future2_position_load.requested_extents;
            metric.future2_position_coalesced_extents = s.future2_position_load.coalesced_extents;
            metric.future2_position_requested_bytes = s.future2_position_load.requested_bytes;
            metric.future2_position_backend_ops = s.future2_position_load.backend_read_ops;
            metric.future2_position_backend_bytes = s.future2_position_load.backend_read_bytes;
            metric.future2_position_backend_seconds = s.future2_position_load.backend_read_seconds;
            metric.future2_success_requested_extents = s.future2_success_load.requested_extents;
            metric.future2_success_coalesced_extents = s.future2_success_load.coalesced_extents;
            metric.future2_success_requested_bytes = s.future2_success_load.requested_bytes;
            metric.future2_success_backend_ops = s.future2_success_load.backend_read_ops;
            metric.future2_success_backend_bytes = s.future2_success_load.backend_read_bytes;
            metric.future2_success_backend_seconds = s.future2_success_load.backend_read_seconds;
            metric.future4_position_requested_extents = s.future4_position_load.requested_extents;
            metric.future4_position_coalesced_extents = s.future4_position_load.coalesced_extents;
            metric.future4_position_requested_bytes = s.future4_position_load.requested_bytes;
            metric.future4_position_backend_ops = s.future4_position_load.backend_read_ops;
            metric.future4_position_backend_bytes = s.future4_position_load.backend_read_bytes;
            metric.future4_position_backend_seconds = s.future4_position_load.backend_read_seconds;
            metric.future4_success_requested_extents = s.future4_success_load.requested_extents;
            metric.future4_success_coalesced_extents = s.future4_success_load.coalesced_extents;
            metric.future4_success_requested_bytes = s.future4_success_load.requested_bytes;
            metric.future4_success_backend_ops = s.future4_success_load.backend_read_ops;
            metric.future4_success_backend_bytes = s.future4_success_load.backend_read_bytes;
            metric.future4_success_backend_seconds = s.future4_success_load.backend_read_seconds;
            metric.output_position_write_bytes = s.output_position_write.requested_bytes;
            metric.output_success_write_bytes = s.output_success_write.requested_bytes;
            metric.open_seconds = open_seconds;
            metric.open_current_position_seconds = open_current_position_seconds;
            metric.open_future2_position_seconds = open_future2_position_seconds;
            metric.open_future2_success_seconds = open_future2_success_seconds;
            metric.open_future4_position_seconds = open_future4_position_seconds;
            metric.open_future4_success_seconds = open_future4_success_seconds;
            metric.descriptor_rows_seconds = descriptor_rows_seconds;
            metric.partition_seconds = partition_seconds;
            metric.writer_open_seconds = writer_open_seconds;
            metric.solve_call_seconds = solve_call_seconds;
            metric.current_position_read_seconds = s.current_position_read_seconds;
            metric.future2_position_read_seconds = s.future2_position_read_seconds;
            metric.future2_success_read_seconds = s.future2_success_read_seconds;
            metric.future2_index_seconds = s.future2_index_seconds;
            metric.future4_position_read_seconds = s.future4_position_read_seconds;
            metric.future4_success_read_seconds = s.future4_success_read_seconds;
            metric.future4_index_seconds = s.future4_index_seconds;
            metric.temp_prepare_seconds = s.temp_prepare_seconds;
            metric.temp_write_seconds = fs.temp_write_seconds;
            metric.temp_read_prepare_seconds = fs.temp_read_prepare_seconds;
            metric.temp_read_seconds = fs.temp_read_seconds;
            metric.final_stage_write_seconds = fs.final_stage_write_seconds;
            metric.final_stage_read_seconds = fs.final_stage_read_seconds;
            metric.output_pending_seconds = fs.output_pending_seconds;
            metric.compact_seconds = s.compact_seconds;
            metric.result_assembly_seconds = s.result_assembly_seconds;
            metric.current_plan_seconds = s.current_plan_seconds;
            metric.future_release_seconds = s.future_release_seconds;
            metric.workspace_release_seconds = s.workspace_release_seconds;
            metric.family_plan_seconds = fs.plan_seconds;
            metric.family_workspace_prepare_seconds = fs.workspace_prepare_seconds;
            metric.family_mark_empty_seconds = fs.mark_empty_seconds;
            metric.family_current_layout_seconds = fs.current_layout_seconds;
            metric.family_future4_prepare_overhead_seconds = fs.future4_prepare_overhead_seconds;
            metric.family_future2_prepare_overhead_seconds = fs.future2_prepare_overhead_seconds;
            metric.family_future4_prepare_normalize_seconds =
                fs.future4_prepare_normalize_seconds;
            metric.family_future2_prepare_normalize_seconds =
                fs.future2_prepare_normalize_seconds;
            metric.family_future4_prepare_select_seconds =
                fs.future4_prepare_select_seconds;
            metric.family_future2_prepare_select_seconds =
                fs.future2_prepare_select_seconds;
            metric.family_future4_prepare_index_build_seconds =
                fs.future4_prepare_index_build_seconds;
            metric.family_future2_prepare_index_build_seconds =
                fs.future2_prepare_index_build_seconds;
            metric.family_future4_prepare_insert_sort_seconds =
                fs.future4_prepare_insert_sort_seconds;
            metric.family_future2_prepare_insert_sort_seconds =
                fs.future2_prepare_insert_sort_seconds;
            metric.future_release_all_calls = fs.future_release_all_calls;
            metric.future_release_except_calls = fs.future_release_except_calls;
            metric.family_future_release_all_seconds = fs.future_release_all_seconds;
            metric.family_future_release_except_seconds = fs.future_release_except_seconds;
            metric.family_future_release_all_clear_seconds =
                fs.future_release_all_clear_seconds;
            metric.family_future_release_except_normalize_seconds =
                fs.future_release_except_normalize_seconds;
            metric.family_future_release_except_filter_seconds =
                fs.future_release_except_filter_seconds;
            metric.family_future_release_except_erase_seconds =
                fs.future_release_except_erase_seconds;
            metric.family_future_release_except_ids_seconds =
                fs.future_release_except_ids_seconds;
            metric.family_workspace_release_spawn4_partial_seconds =
                fs.workspace_release_spawn4_partial_seconds;
            metric.family_workspace_release_spawn4_scratch_seconds =
                fs.workspace_release_spawn4_scratch_seconds;
            metric.family_workspace_release_spawn4_temp_values_seconds =
                fs.workspace_release_spawn4_temp_values_seconds;
            metric.family_workspace_release_spawn2_dense_seconds =
                fs.workspace_release_spawn2_dense_seconds;
            metric.family_workspace_release_spawn2_prefetch_seconds =
                fs.workspace_release_spawn2_prefetch_seconds;
            metric.family_workspace_release_spawn2_temp_values_seconds =
                fs.workspace_release_spawn2_temp_values_seconds;
            metric.family_future4_lookup_copy_seconds = fs.future4_lookup_copy_seconds;
            metric.family_future2_lookup_copy_seconds = fs.future2_lookup_copy_seconds;
            metric.family_spawn4_phase_wall_seconds = fs.spawn4_phase_wall_seconds;
            metric.family_spawn2_phase_wall_seconds = fs.spawn2_phase_wall_seconds;
            metric.family_spawn4_phase_untracked_seconds =
                fs.spawn4_phase_untracked_seconds;
            metric.family_spawn2_phase_untracked_seconds =
                fs.spawn2_phase_untracked_seconds;
            metric.family_spawn4_cell_compute_seconds = fs.spawn4_cell_compute_seconds;
            metric.family_spawn2_cell_compute_seconds = fs.spawn2_cell_compute_seconds;
            metric.family_spawn4_bucket_hit_seconds = fs.spawn4_bucket_hit_seconds;
            metric.family_spawn2_bucket_hit_seconds = fs.spawn2_bucket_hit_seconds;
            metric.family_final_dense_copy_seconds = fs.final_dense_copy_seconds;
            metric.family_compact_value_copy_seconds = fs.compact_value_copy_seconds;
            metric.family_pending_mark_seconds = fs.pending_mark_seconds;
            metric.family_output_finish_seconds = fs.output_finish_seconds;
            metric.family_output_streamer_open_seconds = fs.output_streamer_open_seconds;
            metric.family_temp_open_seconds = fs.temp_open_seconds;
            metric.family_temp_close_seconds = fs.temp_close_seconds;
            metric.partial_cleanup_seconds = s.partial_cleanup_seconds;
            metric.position_write_seconds = s.position_write_seconds;
            metric.success_write_seconds = s.success_write_seconds;
            metric.writer_close_seconds = writer_close_seconds;
            metric.post_resize_seconds = post_resize_seconds;
            metric.total_seconds = now_seconds() - layer_begin;
            write_metric_row(stats, metric);
            stats.flush();
            add_to_summary(summary, metric);

            std::cout << std::setprecision(9)
                      << "ordinal=" << ordinal
                      << " layer_sum=" << metric.layer_sum
                      << " rows=" << metric.current_rows
                      << " live_rows=" << metric.live_rows
                      << " final_rows=" << (metric.live_rows + metric.zero_pruned_rows)
                      << " total_seconds=" << metric.total_seconds
                      << " total_mrows_per_sec="
                      << (metric.total_seconds > 0.0
                              ? static_cast<double>(metric.current_rows) / metric.total_seconds / 1.0e6
                              : 0.0)
                      << " final_mrows_per_sec="
                      << (metric.total_seconds > 0.0
                              ? static_cast<double>(metric.live_rows + metric.zero_pruned_rows) /
                                    metric.total_seconds / 1.0e6
                              : 0.0)
                      << '\n';
        }

        summary.ordinal = max_ordinal;
        summary.layer_sum = 0U;
        write_metric_row(stats, summary);

        std::ofstream summary_out(args.summary_csv);
        if (!summary_out) {
            throw std::runtime_error("failed to open summary csv: " + args.summary_csv.string());
        }
        summary_out << std::setprecision(12);
        write_stats_header(summary_out);
        write_metric_row(summary_out, summary);

        std::cout << std::setprecision(9)
                  << "family_solve_summary rows=" << summary.current_rows
                  << " live_rows=" << summary.live_rows
                  << " final_rows=" << (summary.live_rows + summary.zero_pruned_rows)
                  << " total_seconds=" << summary.total_seconds
                  << " wall_seconds=" << (now_seconds() - all_begin)
                  << " total_mrows_per_sec="
                  << (summary.total_seconds > 0.0
                          ? static_cast<double>(summary.current_rows) / summary.total_seconds / 1.0e6
                          : 0.0)
                  << " final_mrows_per_sec="
                  << (summary.total_seconds > 0.0
                          ? static_cast<double>(summary.live_rows + summary.zero_pruned_rows) /
                                summary.total_seconds / 1.0e6
                          : 0.0)
                  << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        switch (args.success_dtype) {
            case BC::BCSuccessDTypeMode::UInt32:
                run_bench_typed<uint32_t>(args);
                break;
            case BC::BCSuccessDTypeMode::UInt64:
                run_bench_typed<uint64_t>(args);
                break;
            case BC::BCSuccessDTypeMode::Float32:
            case BC::BCSuccessDTypeMode::OneMinusFloat32:
                run_bench_typed<float>(args);
                break;
            case BC::BCSuccessDTypeMode::Float64:
            case BC::BCSuccessDTypeMode::OneMinusFloat64:
                run_bench_typed<double>(args);
                break;
        }
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_solve_full_bench failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
