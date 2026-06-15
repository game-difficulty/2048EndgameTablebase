#include "BCDirectFileIO.h"
#include "BCSingleChunkSolve.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

struct Args {
    std::filesystem::path generated_position_dir = "tmp/free10_256_resident_generated";
    std::filesystem::path solved_output_dir = "tmp/free10_256_single_chunk_solved";
    std::filesystem::path stats_csv = "tmp/free10_256_single_chunk_solve_stats.csv";
    std::filesystem::path summary_csv = "tmp/free10_256_single_chunk_solve_summary.csv";
    std::string prefix = "free10_256_";
    uint32_t target_rank = 8U;
    int success_target_rank = -1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t current_chunk_rows = 128U;
    uint64_t current_chunk_max_bytes = 512ULL * 1024ULL * 1024ULL;
    std::optional<uint32_t> start_ordinal;
    std::optional<uint32_t> min_ordinal;
    bool direct_io = false;
    uint32_t direct_queue_depth = 16U;
    bool keep_direct_padding = true;
};

struct LayerFile {
    uint32_t ordinal = 0U;
    std::filesystem::path path;
};

struct LayerMetric {
    uint32_t ordinal = 0U;
    uint64_t layer_sum = 0U;
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t current_chunks = 0U;
    uint64_t current_cells = 0U;
    uint64_t current_work_items = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t tmp4_write_bytes = 0U;
    uint64_t tmp4_read_bytes = 0U;
    uint64_t tmp4_backend_write_bytes = 0U;
    uint64_t tmp4_backend_read_bytes = 0U;
    uint64_t current_position_backend_read_bytes = 0U;
    uint64_t future2_position_backend_read_bytes = 0U;
    uint64_t future2_success_backend_read_bytes = 0U;
    uint64_t future4_position_backend_read_bytes = 0U;
    uint64_t future4_success_backend_read_bytes = 0U;
    uint64_t position_backend_write_bytes = 0U;
    uint64_t success_backend_write_bytes = 0U;
    double open_seconds = 0.0;
    double current_position_read_seconds = 0.0;
    double future2_position_read_seconds = 0.0;
    double future2_success_read_seconds = 0.0;
    double future2_index_seconds = 0.0;
    double future4_position_read_seconds = 0.0;
    double future4_success_read_seconds = 0.0;
    double future4_index_seconds = 0.0;
    double current_position_backend_read_seconds = 0.0;
    double future2_position_backend_read_seconds = 0.0;
    double future4_position_backend_read_seconds = 0.0;
    double current_plan_seconds = 0.0;
    double future_release_seconds = 0.0;
    double recalc_seconds = 0.0;
    double compact_seconds = 0.0;
    double partial_write_seconds = 0.0;
    double partial_read_seconds = 0.0;
    double temp_prepare_seconds = 0.0;
    double partial_cleanup_seconds = 0.0;
    double workspace_release_seconds = 0.0;
    double result_assembly_seconds = 0.0;
    double tmp4_backend_write_seconds = 0.0;
    double tmp4_backend_read_seconds = 0.0;
    double position_backend_write_seconds = 0.0;
    double success_backend_write_seconds = 0.0;
    double position_prepare_seconds = 0.0;
    double success_prepare_seconds = 0.0;
    double success_append_seconds = 0.0;
    double success_finish_seconds = 0.0;
    double success_header_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double writer_close_seconds = 0.0;
    double frontier_move_seconds = 0.0;
    double post_resize_seconds = 0.0;
    double tmp_cleanup_seconds = 0.0;
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
    ++i;
    return argv[i];
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
            args.stats_csv = require_value(argc, argv, i, "--stats-csv");
        } else if (key == "--summary-csv") {
            args.summary_csv = require_value(argc, argv, i, "--summary-csv");
        } else if (key == "--prefix") {
            args.prefix = require_value(argc, argv, i, "--prefix");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--target-rank")));
        } else if (key == "--success-target-rank") {
            args.success_target_rank = std::stoi(require_value(argc, argv, i, "--success-target-rank"));
        } else if (key == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (key == "--batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--batch-size")));
        } else if (key == "--current-chunk-rows") {
            args.current_chunk_rows = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--current-chunk-rows")));
        } else if (key == "--current-chunk-max-bytes") {
            args.current_chunk_max_bytes = std::stoull(require_value(argc, argv, i, "--current-chunk-max-bytes"));
        } else if (key == "--start-ordinal") {
            args.start_ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--start-ordinal")));
        } else if (key == "--min-ordinal") {
            args.min_ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--min-ordinal")));
        } else if (key == "--direct-io") {
            args.direct_io = true;
        } else if (key == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--direct-queue-depth")));
        } else if (key == "--keep-direct-padding") {
            args.keep_direct_padding = true;
        } else if (key == "--trim-direct-padding") {
            args.keep_direct_padding = false;
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.success_target_rank < 0) {
        args.success_target_rank = static_cast<int>(args.target_rank);
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be non-zero");
    }
    if (args.current_chunk_rows == 0U) {
        throw std::invalid_argument("--current-chunk-rows must be non-zero");
    }
    return args;
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("--target-rank must be < 15 for free BC LUT");
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
            throw std::runtime_error("position layer ordinals are not contiguous at " + std::to_string(expected));
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
            args.direct_queue_depth > 1U
        )
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
            args.direct_queue_depth > 1U
        );
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
            args.direct_queue_depth > 1U
        );
    }
    return BC::BCPositionFileReader::open_buffered(path, lut);
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> make_output_writer(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size
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

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> make_stream_output_writer(
    const Args &args,
    const std::filesystem::path &path
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (args.direct_io) {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

[[nodiscard]] std::vector<uint8_t> make_empty_position_bytes(uint64_t layer_sum, uint32_t family_unit) {
    const BC::BCFamilyTable axis(
        layer_sum,
        static_cast<uint16_t>(family_unit),
        std::vector<BC::FamilyCoord>{0U}
    );
    BC::BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    writer.mark_empty_cell(0U);
    return writer.finish_layer();
}

[[nodiscard]] BC::BCResidentSolvedLayer<uint32_t> make_terminal_solved_layer(
    const BC::BCPositionLayerReader &position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank,
    int num_threads
) {
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    if (offsets.back() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("terminal BC layer row count exceeds size_t");
    }
    std::vector<uint32_t> values(static_cast<size_t>(offsets.back()), 0U);
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
                        max_scale_value<uint32_t>();
                }
            }
        );
    }
    BC::BCResidentRawSolveResult<uint32_t> raw;
    raw.values = std::move(values);
    raw.cell_value_offsets = offsets;
    return BC::bc_resident_compact_zero_in_place<uint32_t>(
        position,
        raw,
        position.lut(),
        1U,
        BC::BCSuccessDTypeMode::UInt32,
        0U,
        num_threads
    );
}

struct WriteResult {
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    double position_seconds = 0.0;
    double success_seconds = 0.0;
};

[[nodiscard]] WriteResult write_solved_layer_files(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentSolvedLayer<uint32_t> &layer
) {
    WriteResult result;
    const std::vector<uint8_t> &position_bytes = layer.position.bytes();
    result.position_bytes = static_cast<uint64_t>(position_bytes.size());
    const std::filesystem::path pos_path = position_path_for(args, ordinal);
    {
        std::unique_ptr<BC::BCWritableFile> writer =
            make_output_writer(args, pos_path, result.position_bytes);
        BC::BCFileIOStats stats;
        const double t0 = now_seconds();
        BC::bc_single_chunk_write_position_bytes<uint32_t>(*writer, position_bytes, &stats);
        writer.reset();
        result.position_seconds = now_seconds() - t0;
        if (args.direct_io && !args.keep_direct_padding) {
            std::filesystem::resize_file(pos_path, result.position_bytes);
        }
    }

    const uint64_t success_logical_size =
        BC::kBCSuccessHeaderBytes +
        static_cast<uint64_t>(layer.success_values.size()) * sizeof(uint32_t);
    result.success_bytes = success_logical_size;
    const std::filesystem::path suc_path = success_path_for(args, ordinal);
    {
        std::unique_ptr<BC::BCWritableFile> writer =
            make_output_writer(args, suc_path, success_logical_size);
        BC::BCFileIOStats stats;
        const double t0 = now_seconds();
        const uint64_t written = BC::write_success_values_to_file<uint32_t>(
            *writer,
            layer.position,
            1U,
            BC::BCSuccessDTypeMode::UInt32,
            layer.success_values,
            &stats
        );
        writer.reset();
        result.success_seconds = now_seconds() - t0;
        result.success_bytes = written;
        if (args.direct_io && !args.keep_direct_padding) {
            std::filesystem::resize_file(suc_path, written);
        }
    }
    return result;
}

[[nodiscard]] uint64_t process_peak_working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
            sizeof(counters)) != 0) {
        return static_cast<uint64_t>(counters.PeakWorkingSetSize);
    }
#endif
    return 0U;
}

[[nodiscard]] double metric_read_seconds(const LayerMetric &m) {
    return m.current_position_read_seconds +
        m.future2_position_read_seconds +
        m.future2_success_read_seconds +
        m.future4_position_read_seconds +
        m.future4_success_read_seconds;
}

[[nodiscard]] double metric_index_seconds(const LayerMetric &m) {
    return m.future2_index_seconds + m.future4_index_seconds;
}

[[nodiscard]] double metric_write_seconds(const LayerMetric &m) {
    return m.position_write_seconds + m.success_write_seconds;
}

[[nodiscard]] double metric_accounted_seconds(const LayerMetric &m) {
    return m.open_seconds +
        metric_read_seconds(m) +
        metric_index_seconds(m) +
        m.current_plan_seconds +
        m.future_release_seconds +
        m.recalc_seconds +
        m.compact_seconds +
        m.partial_write_seconds +
        m.partial_read_seconds +
        m.temp_prepare_seconds +
        m.partial_cleanup_seconds +
        m.workspace_release_seconds +
        m.result_assembly_seconds +
        metric_write_seconds(m) +
        m.writer_close_seconds +
        m.frontier_move_seconds +
        m.post_resize_seconds +
        m.tmp_cleanup_seconds;
}

void write_stats_header(std::ofstream &out) {
    out
        << "ordinal,layer_sum,current_rows,live_rows,zero_pruned_rows,current_chunks,"
        << "current_cells,current_work_items,position_bytes,success_bytes,"
        << "tmp4_write_bytes,tmp4_read_bytes,tmp4_backend_write_bytes,"
        << "tmp4_backend_read_bytes,current_position_backend_read_bytes,"
        << "future2_position_backend_read_bytes,future2_success_backend_read_bytes,"
        << "future4_position_backend_read_bytes,future4_success_backend_read_bytes,"
        << "position_backend_write_bytes,success_backend_write_bytes,open_seconds,"
        << "current_position_read_seconds,future2_position_read_seconds,"
        << "future2_success_read_seconds,future2_index_seconds,"
        << "future4_position_read_seconds,future4_success_read_seconds,"
        << "future4_index_seconds,current_position_backend_read_seconds,"
        << "future2_position_backend_read_seconds,future4_position_backend_read_seconds,"
        << "current_plan_seconds,future_release_seconds,"
        << "recalc_seconds,compact_seconds,tmp4_write_seconds,"
        << "tmp4_read_seconds,tmp4_backend_write_seconds,tmp4_backend_read_seconds,"
        << "temp_prepare_seconds,partial_cleanup_seconds,workspace_release_seconds,"
        << "result_assembly_seconds,"
        << "position_backend_write_seconds,position_prepare_seconds,success_prepare_seconds,"
        << "success_append_seconds,success_finish_seconds,success_header_seconds,"
        << "success_backend_write_seconds,position_write_seconds,success_write_seconds,"
        << "writer_close_seconds,frontier_move_seconds,post_resize_seconds,"
        << "tmp_cleanup_seconds,accounted_seconds,untracked_seconds,total_seconds,recalc_mrows_per_sec,"
        << "total_mrows_per_sec\n";
}

void write_metric_row(std::ofstream &out, const LayerMetric &m) {
    const double recalc_mrows = m.recalc_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.recalc_seconds / 1.0e6
        : 0.0;
    const double total_mrows = m.total_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.total_seconds / 1.0e6
        : 0.0;
    const double accounted = metric_accounted_seconds(m);
    out
        << m.ordinal << ','
        << m.layer_sum << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.current_chunks << ','
        << m.current_cells << ','
        << m.current_work_items << ','
        << m.position_bytes << ','
        << m.success_bytes << ','
        << m.tmp4_write_bytes << ','
        << m.tmp4_read_bytes << ','
        << m.tmp4_backend_write_bytes << ','
        << m.tmp4_backend_read_bytes << ','
        << m.current_position_backend_read_bytes << ','
        << m.future2_position_backend_read_bytes << ','
        << m.future2_success_backend_read_bytes << ','
        << m.future4_position_backend_read_bytes << ','
        << m.future4_success_backend_read_bytes << ','
        << m.position_backend_write_bytes << ','
        << m.success_backend_write_bytes << ','
        << m.open_seconds << ','
        << m.current_position_read_seconds << ','
        << m.future2_position_read_seconds << ','
        << m.future2_success_read_seconds << ','
        << m.future2_index_seconds << ','
        << m.future4_position_read_seconds << ','
        << m.future4_success_read_seconds << ','
        << m.future4_index_seconds << ','
        << m.current_position_backend_read_seconds << ','
        << m.future2_position_backend_read_seconds << ','
        << m.future4_position_backend_read_seconds << ','
        << m.current_plan_seconds << ','
        << m.future_release_seconds << ','
        << m.recalc_seconds << ','
        << m.compact_seconds << ','
        << m.partial_write_seconds << ','
        << m.partial_read_seconds << ','
        << m.tmp4_backend_write_seconds << ','
        << m.tmp4_backend_read_seconds << ','
        << m.temp_prepare_seconds << ','
        << m.partial_cleanup_seconds << ','
        << m.workspace_release_seconds << ','
        << m.result_assembly_seconds << ','
        << m.position_backend_write_seconds << ','
        << m.position_prepare_seconds << ','
        << m.success_prepare_seconds << ','
        << m.success_append_seconds << ','
        << m.success_finish_seconds << ','
        << m.success_header_seconds << ','
        << m.success_backend_write_seconds << ','
        << m.position_write_seconds << ','
        << m.success_write_seconds << ','
        << m.writer_close_seconds << ','
        << m.frontier_move_seconds << ','
        << m.post_resize_seconds << ','
        << m.tmp_cleanup_seconds << ','
        << accounted << ','
        << (m.total_seconds - accounted) << ','
        << m.total_seconds << ','
        << recalc_mrows << ','
        << total_mrows << '\n';
}

int run(const Args &args) {
    const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
    const std::vector<uint8_t> success_shifts = all_board_success_shifts();
    const std::map<uint32_t, LayerFile> layers = discover_layers(args);
    const uint32_t discovered_min_ordinal = layers.begin()->first;
    const uint32_t max_ordinal = layers.rbegin()->first;
    if (discovered_min_ordinal != 0U || max_ordinal < 2U) {
        throw std::runtime_error("expected generated position ordinals to start at 0 and include at least 3 layers");
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

    std::vector<LayerMetric> metrics;
    metrics.reserve(static_cast<size_t>(max_ordinal + 1U));

    const double solve_begin = now_seconds();
    const bool resume_existing_frontier = args.start_ordinal.has_value();
    if (resume_existing_frontier) {
        if (*args.start_ordinal >= max_ordinal) {
            throw std::invalid_argument("--start-ordinal must be less than the max generated ordinal");
        }
        if (!std::filesystem::exists(position_path_for(args, *args.start_ordinal + 1U)) ||
            !std::filesystem::exists(success_path_for(args, *args.start_ordinal + 1U)) ||
            !std::filesystem::exists(position_path_for(args, *args.start_ordinal + 2U)) ||
            !std::filesystem::exists(success_path_for(args, *args.start_ordinal + 2U))) {
            throw std::runtime_error("--start-ordinal requires existing solved future2/future4 files");
        }
    }
    uint64_t virtual_layer_sum = 0U;
    uint32_t virtual_family_unit = 0U;
    BC::BCSingleChunkFrontierLayer<uint32_t> future4_frontier;
    if (!resume_existing_frontier) {
        BC::BCResidentSolvedLayer<uint32_t> top_layer;
        const double top_begin = now_seconds();
        const double open_begin = now_seconds();
        BC::BCPositionFileReader top_position =
            open_position_file(args, layers.at(max_ordinal).path, lut);
        const double open_seconds = now_seconds() - open_begin;
        const uint64_t top_layer_sum = top_position.layer().header().layer_sum;
        virtual_layer_sum = top_layer_sum + 2U;
        virtual_family_unit = top_position.layer().header().family_unit;
        top_layer = make_terminal_solved_layer(
            top_position.layer(),
            success_shifts,
            args.success_target_rank,
            args.num_threads
        );
        WriteResult write = write_solved_layer_files(args, max_ordinal, top_layer);
        LayerMetric metric;
        metric.ordinal = max_ordinal;
        metric.layer_sum = top_layer_sum;
        metric.live_rows = top_layer.compact_stats.live_rows;
        metric.zero_pruned_rows = top_layer.compact_stats.zero_pruned_rows;
        metric.position_bytes = write.position_bytes;
        metric.success_bytes = write.success_bytes;
        metric.open_seconds = open_seconds;
        metric.compact_seconds = top_layer.compact_stats.compact_seconds;
        metric.position_write_seconds = write.position_seconds;
        metric.success_write_seconds = write.success_seconds;
        metric.total_seconds = now_seconds() - top_begin;
        metrics.push_back(metric);
        write_metric_row(stats, metric);
    }

    if (!resume_existing_frontier) {
        std::vector<uint8_t> virtual_empty_position_bytes =
            make_empty_position_bytes(virtual_layer_sum, virtual_family_unit);
        BC::BCResidentSolvedLayer<uint32_t> virtual_empty_write_layer;
        virtual_empty_write_layer.open(
            virtual_empty_position_bytes,
            {},
            lut,
            1U,
            BC::BCSuccessDTypeMode::UInt32
        );
        (void)write_solved_layer_files(args, max_ordinal + 1U, virtual_empty_write_layer);

        BC::BCSingleChunkFrontierLayer<uint32_t> virtual_empty_layer;
        BC::BCSuccessOwnedValues<uint32_t> empty_success_values;
        virtual_empty_layer.open(
            std::move(virtual_empty_position_bytes),
            std::move(empty_success_values),
            lut,
            1U,
            BC::BCSuccessDTypeMode::UInt32
        );
        future4_frontier = std::move(virtual_empty_layer);
    } else {
        BC::BCPositionStreamingReader future4_position =
            open_position_stream(args, position_path_for(args, *args.start_ordinal + 2U), lut);
        BC::BCSuccessStreamingReader future4_success =
            open_success_stream(args, success_path_for(args, *args.start_ordinal + 2U), future4_position);
        future4_frontier = BC::bc_single_chunk_load_frontier_layer<uint32_t>(
            future4_position,
            future4_success,
            lut,
            1U,
            BC::BCSuccessDTypeMode::UInt32
        );
    }

    BC::BCSingleChunkSolveWorkspace<uint32_t> workspace;
    const int64_t first_solve_ordinal = resume_existing_frontier
        ? static_cast<int64_t>(*args.start_ordinal)
        : static_cast<int64_t>(max_ordinal) - 1;
    for (int64_t ordinal_signed = first_solve_ordinal;
         ordinal_signed >= static_cast<int64_t>(min_ordinal);
         --ordinal_signed) {
        const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
        const double layer_begin = now_seconds();
        const double open_begin = now_seconds();
        BC::BCPositionStreamingReader current =
            open_position_stream(args, layers.at(ordinal).path, lut);
        BC::BCPositionStreamingReader future2_position =
            open_position_stream(args, position_path_for(args, ordinal + 1U), lut);
        BC::BCSuccessStreamingReader future2_success =
            open_success_stream(args, success_path_for(args, ordinal + 1U), future2_position);
        const double open_seconds = now_seconds() - open_begin;
        const uint64_t layer_sum = current.header().layer_sum;
        const uint64_t descriptor_row_count = descriptor_rows(current);

        BC::BCSingleChunkSolveOptions<uint32_t> options;
        options.solve.num_threads = args.num_threads;
        options.solve.row_width = 1U;
        options.solve.set_dtype(BC::BCSuccessDTypeMode::UInt32);
        options.solve.edge_options.canonical_batch_size = args.canonical_batch_size;
        options.solve.edge_options.success_target_rank = args.success_target_rank;
        options.solve.edge_options.success_shifts = &success_shifts;
        options.solve.edge_options.success_check_all_cells = true;
        options.solve.edge_options.future_cell_modulus = current.header().family_count;
        options.current_chunk_rows = args.current_chunk_rows;
        options.current_chunk_max_bytes = args.current_chunk_max_bytes;

        const std::filesystem::path output_pos = position_path_for(args, ordinal);
        const std::filesystem::path output_suc = success_path_for(args, ordinal);
        std::unique_ptr<BC::BCWritableFile> position_writer =
            make_stream_output_writer(args, output_pos);
        std::unique_ptr<BC::BCWritableFile> success_writer =
            make_stream_output_writer(args, output_suc);
        BC::BCSingleChunkStrictFrontierFileResult<uint32_t> result =
            BC::bc_single_chunk_solve_strict_1x_to_files_from_future4_frontier<uint32_t>(
                current,
                future2_position,
                future2_success,
                std::move(future4_frontier),
                *position_writer,
                *success_writer,
                args.solved_output_dir / (args.prefix + std::to_string(ordinal) + "_strict_tmp"),
                options,
                &workspace
            );
        const double writer_close_t0 = now_seconds();
        position_writer.reset();
        success_writer.reset();
        const double writer_close_seconds = now_seconds() - writer_close_t0;

        const double frontier_move_t0 = now_seconds();
        future4_frontier = std::move(result.next_future4_layer);
        const double frontier_move_seconds = now_seconds() - frontier_move_t0;

        double post_resize_seconds = 0.0;
        if (args.direct_io && !args.keep_direct_padding) {
            const double post_resize_t0 = now_seconds();
            std::filesystem::resize_file(output_pos, result.position_bytes);
            std::filesystem::resize_file(output_suc, result.success_bytes);
            post_resize_seconds = now_seconds() - post_resize_t0;
        }
        const double tmp_cleanup_t0 = now_seconds();
        std::error_code cleanup_ec;
        std::filesystem::remove_all(
            args.solved_output_dir / (args.prefix + std::to_string(ordinal) + "_strict_tmp"),
            cleanup_ec
        );
        const double tmp_cleanup_seconds = now_seconds() - tmp_cleanup_t0;

        LayerMetric metric;
        metric.ordinal = ordinal;
        metric.layer_sum = layer_sum;
        metric.current_rows = result.stats.current_rows != 0U
            ? result.stats.current_rows
            : descriptor_row_count;
        metric.live_rows = result.stats.compact_live_rows;
        metric.zero_pruned_rows = result.stats.compact_zero_pruned_rows;
        metric.current_chunks = result.stats.current_chunks;
        metric.current_cells = result.stats.current_cells;
        metric.current_work_items = result.stats.current_work_items;
        metric.position_bytes = result.position_bytes;
        metric.success_bytes = result.success_bytes;
        metric.tmp4_write_bytes = result.stats.partial_write_bytes;
        metric.tmp4_read_bytes = result.stats.partial_read_bytes;
        metric.tmp4_backend_write_bytes = result.stats.partial_write_io.backend_bytes;
        metric.tmp4_backend_read_bytes = result.stats.partial_read_io.backend_bytes;
        metric.current_position_backend_read_bytes =
            result.stats.current_position_load.backend_read_bytes;
        metric.future2_position_backend_read_bytes =
            result.stats.future2_position_load.backend_read_bytes;
        metric.future2_success_backend_read_bytes =
            result.stats.future2_success_load.backend_read_bytes;
        metric.future4_position_backend_read_bytes =
            result.stats.future4_position_load.backend_read_bytes;
        metric.future4_success_backend_read_bytes =
            result.stats.future4_success_load.backend_read_bytes;
        metric.position_backend_write_bytes = result.stats.output_position_write.backend_bytes;
        metric.success_backend_write_bytes = result.stats.output_success_write.backend_bytes;
        metric.open_seconds = open_seconds;
        metric.current_position_read_seconds = result.stats.current_position_read_seconds;
        metric.future2_position_read_seconds = result.stats.future2_position_read_seconds;
        metric.future2_success_read_seconds = result.stats.future2_success_read_seconds;
        metric.future2_index_seconds = result.stats.future2_index_seconds;
        metric.future4_position_read_seconds = result.stats.future4_position_read_seconds;
        metric.future4_success_read_seconds = result.stats.future4_success_read_seconds;
        metric.future4_index_seconds = result.stats.future4_index_seconds;
        metric.current_position_backend_read_seconds =
            result.stats.current_position_load.backend_read_seconds;
        metric.future2_position_backend_read_seconds =
            result.stats.future2_position_load.backend_read_seconds;
        metric.future4_position_backend_read_seconds =
            result.stats.future4_position_load.backend_read_seconds;
        metric.current_plan_seconds = result.stats.current_plan_seconds;
        metric.future_release_seconds = result.stats.future_release_seconds;
        metric.recalc_seconds = result.stats.recalc_seconds;
        metric.compact_seconds = result.stats.compact_seconds;
        metric.partial_write_seconds = result.stats.partial_write_seconds;
        metric.partial_read_seconds = result.stats.partial_read_seconds;
        metric.tmp4_backend_write_seconds = result.stats.partial_write_io.backend_seconds;
        metric.tmp4_backend_read_seconds = result.stats.partial_read_io.backend_seconds;
        metric.temp_prepare_seconds = result.stats.temp_prepare_seconds;
        metric.partial_cleanup_seconds = result.stats.partial_cleanup_seconds;
        metric.workspace_release_seconds = result.stats.workspace_release_seconds;
        metric.result_assembly_seconds = result.stats.result_assembly_seconds;
        metric.position_backend_write_seconds = result.stats.output_position_write.backend_seconds;
        metric.position_prepare_seconds = result.stats.position_prepare_seconds;
        metric.success_prepare_seconds = result.stats.success_prepare_seconds;
        metric.success_append_seconds = result.stats.success_append_seconds;
        metric.success_finish_seconds = result.stats.success_finish_seconds;
        metric.success_header_seconds = result.stats.success_header_seconds;
        metric.success_backend_write_seconds = result.stats.output_success_write.backend_seconds;
        metric.position_write_seconds = result.stats.position_write_seconds;
        metric.success_write_seconds = result.stats.success_write_seconds;
        metric.writer_close_seconds = writer_close_seconds;
        metric.frontier_move_seconds = frontier_move_seconds;
        metric.post_resize_seconds = post_resize_seconds;
        metric.tmp_cleanup_seconds = tmp_cleanup_seconds;
        metric.total_seconds = now_seconds() - layer_begin;
        metrics.push_back(metric);
        write_metric_row(stats, metric);
        stats.flush();

        std::cout << std::setprecision(9)
            << "ordinal=" << ordinal
            << " layer_sum=" << metric.layer_sum
            << " rows=" << metric.current_rows
            << " live_rows=" << metric.live_rows
            << " recalc_seconds=" << metric.recalc_seconds
            << " total_seconds=" << metric.total_seconds
            << " accounted_seconds=" << metric_accounted_seconds(metric)
            << " untracked_seconds=" << (metric.total_seconds - metric_accounted_seconds(metric))
            << " position_bytes=" << metric.position_bytes
            << " success_bytes=" << metric.success_bytes
            << '\n';

    }

    const double solve_wall_seconds = now_seconds() - solve_begin;
    uint64_t total_rows = 0U;
    uint64_t total_live_rows = 0U;
    uint64_t total_zero_pruned_rows = 0U;
    uint64_t total_position_bytes = 0U;
    uint64_t total_success_bytes = 0U;
    uint64_t max_layer_written_bytes = 0U;
    double total_open_seconds = 0.0;
    double total_current_read_seconds = 0.0;
    double total_future2_position_read_seconds = 0.0;
    double total_future2_success_read_seconds = 0.0;
    double total_future2_index_seconds = 0.0;
    double total_future4_position_read_seconds = 0.0;
    double total_future4_success_read_seconds = 0.0;
    double total_future4_index_seconds = 0.0;
    double total_current_plan_seconds = 0.0;
    double total_future_release_seconds = 0.0;
    double total_recalc_seconds = 0.0;
    double total_compact_seconds = 0.0;
    double total_partial_write_seconds = 0.0;
    double total_partial_read_seconds = 0.0;
    double total_temp_prepare_seconds = 0.0;
    double total_partial_cleanup_seconds = 0.0;
    double total_workspace_release_seconds = 0.0;
    double total_result_assembly_seconds = 0.0;
    double total_position_write_seconds = 0.0;
    double total_success_write_seconds = 0.0;
    double total_writer_close_seconds = 0.0;
    double total_frontier_move_seconds = 0.0;
    double total_post_resize_seconds = 0.0;
    double total_tmp_cleanup_seconds = 0.0;
    double total_accounted_seconds = 0.0;
    for (const LayerMetric &m : metrics) {
        total_rows += m.current_rows;
        total_live_rows += m.live_rows;
        total_zero_pruned_rows += m.zero_pruned_rows;
        total_position_bytes += m.position_bytes;
        total_success_bytes += m.success_bytes;
        max_layer_written_bytes = std::max(
            max_layer_written_bytes,
            m.position_bytes + m.success_bytes
        );
        total_open_seconds += m.open_seconds;
        total_current_read_seconds += m.current_position_read_seconds;
        total_future2_position_read_seconds += m.future2_position_read_seconds;
        total_future2_success_read_seconds += m.future2_success_read_seconds;
        total_future2_index_seconds += m.future2_index_seconds;
        total_future4_position_read_seconds += m.future4_position_read_seconds;
        total_future4_success_read_seconds += m.future4_success_read_seconds;
        total_future4_index_seconds += m.future4_index_seconds;
        total_current_plan_seconds += m.current_plan_seconds;
        total_future_release_seconds += m.future_release_seconds;
        total_recalc_seconds += m.recalc_seconds;
        total_compact_seconds += m.compact_seconds;
        total_partial_write_seconds += m.partial_write_seconds;
        total_partial_read_seconds += m.partial_read_seconds;
        total_temp_prepare_seconds += m.temp_prepare_seconds;
        total_partial_cleanup_seconds += m.partial_cleanup_seconds;
        total_workspace_release_seconds += m.workspace_release_seconds;
        total_result_assembly_seconds += m.result_assembly_seconds;
        total_position_write_seconds += m.position_write_seconds;
        total_success_write_seconds += m.success_write_seconds;
        total_writer_close_seconds += m.writer_close_seconds;
        total_frontier_move_seconds += m.frontier_move_seconds;
        total_post_resize_seconds += m.post_resize_seconds;
        total_tmp_cleanup_seconds += m.tmp_cleanup_seconds;
        total_accounted_seconds += metric_accounted_seconds(m);
    }
    const double wall_mrows = solve_wall_seconds > 0.0
        ? static_cast<double>(total_rows) / solve_wall_seconds / 1.0e6
        : 0.0;
    const double recalc_mrows = total_recalc_seconds > 0.0
        ? static_cast<double>(total_rows) / total_recalc_seconds / 1.0e6
        : 0.0;
    const uint64_t peak_working_set = process_peak_working_set_bytes();

    std::ofstream summary(args.summary_csv);
    if (!summary) {
        throw std::runtime_error("failed to open summary csv: " + args.summary_csv.string());
    }
    summary << std::setprecision(12);
    summary
        << "generated_position_dir,solved_output_dir,min_ordinal,max_ordinal,layers_solved,"
        << "total_rows,total_live_rows,total_zero_pruned_rows,total_position_bytes,"
        << "total_success_bytes,max_layer_written_bytes,total_open_seconds,"
        << "total_current_position_read_seconds,total_future2_position_read_seconds,"
        << "total_future2_success_read_seconds,total_future2_index_seconds,"
        << "total_future4_position_read_seconds,total_future4_success_read_seconds,"
        << "total_future4_index_seconds,total_current_plan_seconds,"
        << "total_future_release_seconds,total_recalc_seconds,"
        << "total_compact_seconds,total_tmp4_write_seconds,total_tmp4_read_seconds,"
        << "total_temp_prepare_seconds,total_partial_cleanup_seconds,"
        << "total_workspace_release_seconds,total_result_assembly_seconds,"
        << "total_position_write_seconds,total_success_write_seconds,"
        << "total_writer_close_seconds,total_frontier_move_seconds,total_post_resize_seconds,"
        << "total_tmp_cleanup_seconds,total_accounted_seconds,total_untracked_seconds,"
        << "solve_wall_seconds,wall_mrows_per_sec,recalc_mrows_per_sec,peak_working_set_bytes\n";
    summary
        << args.generated_position_dir.string() << ','
        << args.solved_output_dir.string() << ','
        << min_ordinal << ','
        << max_ordinal << ','
        << metrics.size() << ','
        << total_rows << ','
        << total_live_rows << ','
        << total_zero_pruned_rows << ','
        << total_position_bytes << ','
        << total_success_bytes << ','
        << max_layer_written_bytes << ','
        << total_open_seconds << ','
        << total_current_read_seconds << ','
        << total_future2_position_read_seconds << ','
        << total_future2_success_read_seconds << ','
        << total_future2_index_seconds << ','
        << total_future4_position_read_seconds << ','
        << total_future4_success_read_seconds << ','
        << total_future4_index_seconds << ','
        << total_current_plan_seconds << ','
        << total_future_release_seconds << ','
        << total_recalc_seconds << ','
        << total_compact_seconds << ','
        << total_partial_write_seconds << ','
        << total_partial_read_seconds << ','
        << total_temp_prepare_seconds << ','
        << total_partial_cleanup_seconds << ','
        << total_workspace_release_seconds << ','
        << total_result_assembly_seconds << ','
        << total_position_write_seconds << ','
        << total_success_write_seconds << ','
        << total_writer_close_seconds << ','
        << total_frontier_move_seconds << ','
        << total_post_resize_seconds << ','
        << total_tmp_cleanup_seconds << ','
        << total_accounted_seconds << ','
        << (solve_wall_seconds - total_accounted_seconds) << ','
        << solve_wall_seconds << ','
        << wall_mrows << ','
        << recalc_mrows << ','
        << peak_working_set << '\n';

    std::cout << std::setprecision(12)
        << "summary"
        << " layers_solved=" << metrics.size()
        << " total_rows=" << total_rows
        << " total_live_rows=" << total_live_rows
        << " total_recalc_seconds=" << total_recalc_seconds
        << " solve_wall_seconds=" << solve_wall_seconds
        << " total_accounted_seconds=" << total_accounted_seconds
        << " total_untracked_seconds=" << (solve_wall_seconds - total_accounted_seconds)
        << " wall_mrows_per_sec=" << wall_mrows
        << " recalc_mrows_per_sec=" << recalc_mrows
        << " peak_working_set_bytes=" << peak_working_set
        << " max_layer_written_bytes=" << max_layer_written_bytes
        << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        return run(args);
    } catch (const std::exception &ex) {
        std::cerr << "bc_single_chunk_solve_full_bench failed: " << ex.what() << '\n';
        return 1;
    }
}
