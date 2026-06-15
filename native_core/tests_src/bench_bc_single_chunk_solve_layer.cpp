#include "BCDirectFileIO.h"
#include "BCSingleChunkSolve.h"
#include "SymmetryUtils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path generated_position_dir = "tmp/free9_256_layer100_fixture/generated";
    std::filesystem::path future_solved_dir = "tmp/free9_256_layer100_fixture/solved";
    std::filesystem::path output_dir = "tmp/free9_256_single_chunk_solve_layer_output";
    std::filesystem::path stats_csv;
    std::string prefix = "free9_256_";
    uint32_t ordinal = 100U;
    uint32_t target_rank = 8U;
    int success_target_rank = 8;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t current_chunk_cells = 100000U;
    uint64_t current_chunk_max_rows = 0U;
    uint32_t current_chunk_rows = 128U;
    uint64_t current_chunk_max_bytes = 512ULL * 1024ULL * 1024ULL;
    bool restrict_future_cells = false;
    bool direct_io = false;
    bool keep_direct_padding = false;
    uint32_t direct_queue_depth = 16U;
    bool write_output = true;
    bool trace_stages = false;
    bool frontier_api = false;
    bool strict_1x = false;
    uint32_t repeats = 1U;
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

[[nodiscard]] int parse_symm_mode(const std::string &value) {
    if (value == "identity") {
        return static_cast<int>(SymmMode::Identity);
    }
    if (value == "full") {
        return static_cast<int>(SymmMode::Full);
    }
    if (value == "diagonal") {
        return static_cast<int>(SymmMode::Diagonal);
    }
    if (value == "horizontal") {
        return static_cast<int>(SymmMode::Horizontal);
    }
    if (value == "min33") {
        return static_cast<int>(SymmMode::Min33);
    }
    if (value == "min24") {
        return static_cast<int>(SymmMode::Min24);
    }
    if (value == "min34") {
        return static_cast<int>(SymmMode::Min34);
    }
    if (value == "min34top") {
        return static_cast<int>(SymmMode::Min34Top);
    }
    return std::stoi(value);
}

void print_usage(const char *exe) {
    std::cerr
        << "Usage: " << exe << " [options]\n"
        << "  --generated-position-dir DIR   current generated .bcpos directory\n"
        << "  --future-solved-dir DIR        exact solved future .bcpos/.bcsuc directory\n"
        << "  --output-dir DIR               output directory for current compacted .bcpos/.bcsuc\n"
        << "  --stats-csv PATH               write per-repeat stats CSV\n"
        << "  --ordinal N                    current layer ordinal, default 100\n"
        << "  --target-rank N                free BC LUT target rank, default 8\n"
        << "  --prefix P                     file prefix, default free9_256_\n"
        << "  --repeats N                    repeat current solve\n"
        << "  --threads N                    OpenMP thread count\n"
        << "  --canonical-batch-size N       canonical batch size\n"
        << "  --current-chunk-cells N        legacy current cells per chunk for non-strict paths\n"
        << "  --current-chunk-max-rows N     legacy current rows per chunk cap, 0 disables\n"
        << "  --current-chunk-rows N         strict row-slab rows per chunk, default 64\n"
        << "  --current-chunk-max-bytes N    strict row-slab success bytes cap, default 268435456\n"
        << "  --restrict-future-cells        prepass current chunk and only load touched future cells\n"
        << "  --trace-stages                 print per-stage progress before layer completion\n"
        << "  --no-current-board-cache       deprecated no-op; current board cache has been removed\n"
        << "  --frontier-api                 pre-load future layers and solve through production frontier API\n"
        << "  --strict-1x                    solve through strict AD-style 1+x two-pass chunk pipeline\n"
        << "  --direct-io                    use direct IO for position read and output write\n"
        << "  --keep-direct-padding          keep direct output physically padded for later direct reads\n"
        << "  --direct-queue-depth N         direct IO queue depth\n"
        << "  --no-write                     skip output writes\n";
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--generated-position-dir") {
            args.generated_position_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--future-solved-dir") {
            args.future_solved_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--output-dir") {
            args.output_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--ordinal") {
            args.ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--prefix") {
            args.prefix = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--success-target-rank") {
            args.success_target_rank = std::stoi(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--canonical-symm-mode") {
            args.canonical_symm_mode = parse_symm_mode(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--spawn-rate4") {
            args.spawn_rate4 = std::stod(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--canonical-batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--current-chunk-cells") {
            args.current_chunk_cells = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--current-chunk-max-rows") {
            args.current_chunk_max_rows = std::stoull(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--current-chunk-rows") {
            args.current_chunk_rows = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--current-chunk-max-bytes") {
            args.current_chunk_max_bytes = std::stoull(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--restrict-future-cells") {
            args.restrict_future_cells = true;
        } else if (flag == "--trace-stages") {
            args.trace_stages = true;
        } else if (flag == "--no-current-board-cache") {
            // Deprecated compatibility flag. Single solve always streams current cells.
        } else if (flag == "--frontier-api") {
            args.frontier_api = true;
        } else if (flag == "--strict-1x") {
            args.strict_1x = true;
        } else if (flag == "--direct-io") {
            args.direct_io = true;
        } else if (flag == "--keep-direct-padding") {
            args.keep_direct_padding = true;
        } else if (flag == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--no-write") {
            args.write_output = false;
        } else if (flag == "--repeats") {
            args.repeats = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (args.repeats == 0U) {
        throw std::invalid_argument("--repeats must be >= 1");
    }
    if (args.current_chunk_cells == 0U) {
        throw std::invalid_argument("--current-chunk-cells must be >= 1");
    }
    if (args.current_chunk_rows == 0U) {
        throw std::invalid_argument("--current-chunk-rows must be >= 1");
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be >= 1");
    }
    if (args.frontier_api && args.strict_1x) {
        throw std::invalid_argument("--frontier-api and --strict-1x are mutually exclusive");
    }
    if (args.strict_1x && args.restrict_future_cells) {
        throw std::invalid_argument("--strict-1x cannot use --restrict-future-cells");
    }
    if (args.strict_1x && !args.write_output) {
        throw std::invalid_argument("--strict-1x writes chunk outputs and cannot use --no-write");
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

[[nodiscard]] std::filesystem::path position_path_for(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcpos");
}

[[nodiscard]] std::filesystem::path success_path_for(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcsuc");
}

[[nodiscard]] BC::BCPositionStreamingReader open_position_stream(
    const std::filesystem::path &path,
    const BC::BCLut &lut,
    const Args &args
) {
    if (args.direct_io) {
        return BC::BCPositionStreamingReader::open_direct_auto(
            path,
            lut,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U
        );
    }
    return BC::BCPositionStreamingReader::open_buffered(path, lut);
}

template <class PositionReader>
[[nodiscard]] BC::BCSuccessStreamingReader open_success_stream(
    const std::filesystem::path &path,
    const PositionReader &position,
    const Args &args
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

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> make_output_writer(
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

template <typename T>
[[nodiscard]] BC::BCResidentSolvedLayer<T> load_frontier_layer_from_streams(
    const BC::BCPositionStreamingReader &position,
    const BC::BCSuccessStreamingReader &success,
    const BC::BCLut &lut,
    uint32_t row_width,
    BC::BCSuccessDTypeMode dtype
) {
    BC::BCSingleChunkFrontierLayer<T> frontier =
        BC::bc_single_chunk_load_frontier_layer<T>(
        position,
        success,
        lut,
        row_width,
        dtype
    );
    const T *frontier_values = frontier.success_values.data();
    std::vector<T> values;
    if (frontier_values != nullptr && frontier.success_values.size() != 0U) {
        values.assign(
            frontier_values,
            frontier_values + static_cast<std::ptrdiff_t>(frontier.success_values.size())
        );
    }
    BC::BCResidentSolvedLayer<T> out;
    out.open(std::move(frontier.position), std::move(values), row_width, dtype);
    return out;
}

struct TraceContext {
    double start_seconds = 0.0;
};

double read_seconds(const BC::BCSingleChunkSolveStats &s) {
    return s.current_position_read_seconds +
        s.future2_position_read_seconds +
        s.future2_success_read_seconds +
        s.future4_position_read_seconds +
        s.future4_success_read_seconds;
}

double index_seconds(const BC::BCSingleChunkSolveStats &s) {
    return s.future2_index_seconds + s.future4_index_seconds;
}

double write_seconds(const BC::BCSingleChunkSolveStats &s) {
    return s.position_write_seconds + s.success_write_seconds;
}

double accounted_seconds(const BC::BCSingleChunkSolveStats &s) {
    return read_seconds(s) +
        index_seconds(s) +
        s.prepass_seconds +
        s.current_plan_seconds +
        s.future_cid_select_seconds +
        s.current_board_cache_seconds +
        s.raw_alloc_seconds +
        s.future_release_seconds +
        s.recalc_seconds +
        s.compact_seconds +
        s.partial_write_seconds +
        s.partial_read_seconds +
        s.temp_prepare_seconds +
        s.partial_cleanup_seconds +
        s.workspace_release_seconds +
        s.output_release_seconds +
        s.position_build_seconds +
        s.result_assembly_seconds +
        write_seconds(s);
}

void trace_stage(
    const char *event,
    const BC::BCSingleChunkSolveStats &s,
    void *user
) {
    const auto *ctx = static_cast<const TraceContext *>(user);
    const double elapsed = ctx == nullptr ? 0.0 : now_seconds() - ctx->start_seconds;
    std::cerr << std::setprecision(9)
        << "trace event=" << event
        << " elapsed=" << elapsed
        << " chunks=" << s.current_chunks
        << " cells=" << s.current_cells
        << " work_items=" << s.current_work_items
        << " rows=" << s.current_rows
        << " future2_cells=" << s.future2_cells_loaded
        << " future4_cells=" << s.future4_cells_loaded
        << " read_seconds=" << read_seconds(s)
        << " index_seconds=" << index_seconds(s)
        << " current_plan_seconds=" << s.current_plan_seconds
        << " future_cid_select_seconds=" << s.future_cid_select_seconds
        << " current_board_cache_seconds=" << s.current_board_cache_seconds
        << " raw_alloc_seconds=" << s.raw_alloc_seconds
        << " future_release_seconds=" << s.future_release_seconds
        << " recalc_seconds=" << s.recalc_seconds
        << " compact_seconds=" << s.compact_seconds
        << " tmp4_write_seconds=" << s.partial_write_seconds
        << " tmp4_read_seconds=" << s.partial_read_seconds
        << " temp_prepare_seconds=" << s.temp_prepare_seconds
        << " partial_cleanup_seconds=" << s.partial_cleanup_seconds
        << " workspace_release_seconds=" << s.workspace_release_seconds
        << " output_release_seconds=" << s.output_release_seconds
        << " position_build_seconds=" << s.position_build_seconds
        << " result_assembly_seconds=" << s.result_assembly_seconds
        << " future_bytes_max=" << s.future_resident_bytes_max
        << " tmp4_write_bytes=" << s.partial_write_bytes
        << " tmp4_read_bytes=" << s.partial_read_bytes
        << " position_backend_write_bytes=" << s.output_position_write.backend_bytes
        << " success_backend_write_bytes=" << s.output_success_write.backend_bytes
        << '\n';
}

void write_stats_header(std::ofstream &out) {
    out
        << "iteration,current_rows,live_rows,zero_pruned_rows,current_chunks,current_cells,current_work_items,"
        << "future2_cells_loaded,future4_cells_loaded,future2_active_cells_max,future4_active_cells_max,"
        << "future_resident_bytes_max,current_board_cache_bytes,output_values,output_bytes,"
        << "tmp4_write_bytes,tmp4_read_bytes,current_position_backend_read_bytes,"
        << "future2_position_backend_read_bytes,future2_success_backend_read_bytes,"
        << "future4_position_backend_read_bytes,future4_success_backend_read_bytes,"
        << "current_position_backend_read_seconds,future2_position_backend_read_seconds,"
        << "future4_position_backend_read_seconds,"
        << "position_backend_write_bytes,success_backend_write_bytes,"
        << "position_backend_write_seconds,success_backend_write_seconds,"
        << "position_backend_write_ops,success_backend_write_ops,"
        << "current_position_read_seconds,future2_position_read_seconds,future2_success_read_seconds,"
        << "future2_index_seconds,future4_position_read_seconds,future4_success_read_seconds,"
        << "future4_index_seconds,prepass_seconds,current_plan_seconds,future_cid_select_seconds,"
        << "current_board_cache_seconds,raw_alloc_seconds,"
        << "future_release_seconds,recalc_seconds,compact_seconds,tmp4_write_seconds,"
        << "tmp4_read_seconds,temp_prepare_seconds,partial_cleanup_seconds,"
        << "workspace_release_seconds,"
        << "output_release_seconds,position_build_seconds,result_assembly_seconds,"
        << "position_write_seconds,success_write_seconds,"
        << "accounted_seconds,untracked_seconds,solve_window_seconds,total_seconds,"
        << "future_lookup_count,future_lookup_misses,encoded_queries,encode_rejects,"
        << "batch_flushes,batch_tail_flushes,batch_source_boards,canonical_flushes,"
        << "total_mrows_per_sec\n";
}

void write_stats_row(
    std::ofstream &out,
    uint32_t iteration,
    const BC::BCSingleChunkSolveStats &s,
    double total_seconds
) {
    const double total_mrows = total_seconds > 0.0
        ? static_cast<double>(s.current_rows) / total_seconds / 1.0e6
        : 0.0;
    const double accounted = accounted_seconds(s);
    out
        << iteration << ','
        << s.current_rows << ','
        << s.compact_live_rows << ','
        << s.compact_zero_pruned_rows << ','
        << s.current_chunks << ','
        << s.current_cells << ','
        << s.current_work_items << ','
        << s.future2_cells_loaded << ','
        << s.future4_cells_loaded << ','
        << s.future2_active_cells_max << ','
        << s.future4_active_cells_max << ','
        << s.future_resident_bytes_max << ','
        << s.current_board_cache_bytes << ','
        << s.output_values << ','
        << s.output_bytes << ','
        << s.partial_write_bytes << ','
        << s.partial_read_bytes << ','
        << s.current_position_load.backend_read_bytes << ','
        << s.future2_position_load.backend_read_bytes << ','
        << s.future2_success_load.backend_read_bytes << ','
        << s.future4_position_load.backend_read_bytes << ','
        << s.future4_success_load.backend_read_bytes << ','
        << s.current_position_load.backend_read_seconds << ','
        << s.future2_position_load.backend_read_seconds << ','
        << s.future4_position_load.backend_read_seconds << ','
        << s.output_position_write.backend_bytes << ','
        << s.output_success_write.backend_bytes << ','
        << s.output_position_write.backend_seconds << ','
        << s.output_success_write.backend_seconds << ','
        << s.output_position_write.backend_io_count << ','
        << s.output_success_write.backend_io_count << ','
        << s.current_position_read_seconds << ','
        << s.future2_position_read_seconds << ','
        << s.future2_success_read_seconds << ','
        << s.future2_index_seconds << ','
        << s.future4_position_read_seconds << ','
        << s.future4_success_read_seconds << ','
        << s.future4_index_seconds << ','
        << s.prepass_seconds << ','
        << s.current_plan_seconds << ','
        << s.future_cid_select_seconds << ','
        << s.current_board_cache_seconds << ','
        << s.raw_alloc_seconds << ','
        << s.future_release_seconds << ','
        << s.recalc_seconds << ','
        << s.compact_seconds << ','
        << s.partial_write_seconds << ','
        << s.partial_read_seconds << ','
        << s.temp_prepare_seconds << ','
        << s.partial_cleanup_seconds << ','
        << s.workspace_release_seconds << ','
        << s.output_release_seconds << ','
        << s.position_build_seconds << ','
        << s.result_assembly_seconds << ','
        << s.position_write_seconds << ','
        << s.success_write_seconds << ','
        << accounted << ','
        << (total_seconds - accounted) << ','
        << total_seconds << ','
        << total_seconds << ','
        << s.edge.future_lookup_count << ','
        << s.edge.future_lookup_misses << ','
        << s.edge.encoded_queries << ','
        << s.edge.encode_rejects << ','
        << s.edge.batch_flushes << ','
        << s.edge.batch_tail_flushes << ','
        << s.edge.batch_source_boards << ','
        << s.edge.canonical_flushes << ','
        << total_mrows
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        if (args.ordinal > std::numeric_limits<uint32_t>::max() - 2U) {
            throw std::invalid_argument("ordinal is too large for +2 ordinal future");
        }

        std::ofstream stats_out;
        if (!args.stats_csv.empty()) {
            if (!args.stats_csv.parent_path().empty()) {
                std::filesystem::create_directories(args.stats_csv.parent_path());
            }
            stats_out.open(args.stats_csv);
            if (!stats_out) {
                throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
            }
            stats_out << std::setprecision(9);
            write_stats_header(stats_out);
        }

        uint64_t total_rows = 0U;
        double total_wall = 0.0;
        double total_recalc = 0.0;
        double total_compact = 0.0;
        double total_position_build = 0.0;
        double total_result_assembly = 0.0;
        double total_read = 0.0;
        double total_index = 0.0;
        double total_prepass = 0.0;
        double total_current_plan = 0.0;
        double total_future_cid_select = 0.0;
        double total_current_board_cache = 0.0;
        double total_raw_alloc = 0.0;
        double total_future_release = 0.0;
        double total_workspace_release = 0.0;
        double total_output_release = 0.0;
        double total_write = 0.0;
        double total_partial_write = 0.0;
        double total_partial_read = 0.0;
        double total_temp_prepare = 0.0;
        double total_partial_cleanup = 0.0;
        double total_accounted = 0.0;
        BC::BCSingleChunkSolveWorkspace<uint32_t> workspace;

        for (uint32_t iteration = 0U; iteration < args.repeats; ++iteration) {
            const double outer_t0 = now_seconds();
            auto trace_outer = [&](const char *event) {
                if (!args.trace_stages) {
                    return;
                }
                std::cerr << std::setprecision(9)
                    << "trace event=" << event
                    << " elapsed=" << (now_seconds() - outer_t0)
                    << '\n';
            };
            trace_outer("open_current_begin");
            BC::BCPositionStreamingReader current_position = open_position_stream(
                position_path_for(args.generated_position_dir, args.prefix, args.ordinal),
                lut,
                args
            );
            trace_outer("open_current_done");
            trace_outer("open_future2_position_begin");
            BC::BCPositionStreamingReader future2_position = open_position_stream(
                position_path_for(args.future_solved_dir, args.prefix, args.ordinal + 1U),
                lut,
                args
            );
            trace_outer("open_future2_position_done");
            trace_outer("open_future4_position_begin");
            BC::BCPositionStreamingReader future4_position = open_position_stream(
                position_path_for(args.future_solved_dir, args.prefix, args.ordinal + 2U),
                lut,
                args
            );
            trace_outer("open_future4_position_done");
            trace_outer("open_future2_success_begin");
            BC::BCSuccessStreamingReader future2_success =
                open_success_stream(
                    success_path_for(args.future_solved_dir, args.prefix, args.ordinal + 1U),
                    future2_position,
                    args
                );
            trace_outer("open_future2_success_done");
            trace_outer("open_future4_success_begin");
            BC::BCSuccessStreamingReader future4_success =
                open_success_stream(
                    success_path_for(args.future_solved_dir, args.prefix, args.ordinal + 2U),
                    future4_position,
                    args
                );
            trace_outer("open_future4_success_done");

            BC::BCSingleChunkSolveOptions<uint32_t> options;
            options.solve.num_threads = args.num_threads;
            options.solve.row_width = 1U;
            options.solve.set_dtype(BC::BCSuccessDTypeMode::UInt32);
            options.solve.edge_options.canonical_batch_size = args.canonical_batch_size;
            options.solve.edge_options.canonical_symm_mode = args.canonical_symm_mode;
            options.solve.edge_options.spawn_rate4 = args.spawn_rate4;
            options.solve.edge_options.success_target_rank = args.success_target_rank;
            options.solve.edge_options.success_check_all_cells = true;
            options.solve.edge_options.future_cell_modulus = current_position.header().family_count;
            options.current_chunk_cells = args.current_chunk_cells;
            options.current_chunk_max_rows = args.current_chunk_max_rows;
            options.current_chunk_rows = args.current_chunk_rows;
            options.current_chunk_max_bytes = args.current_chunk_max_bytes;
            options.restrict_future_cells_to_current_chunk = args.restrict_future_cells;
            TraceContext trace_context{now_seconds()};
            if (args.trace_stages) {
                options.trace = trace_stage;
                options.trace_user = &trace_context;
            }

            std::optional<BC::BCResidentSolvedLayer<uint32_t>> future2_frontier;
            std::optional<BC::BCResidentSolvedLayer<uint32_t>> future4_frontier;
            if (args.frontier_api) {
                trace_outer("frontier2_load_begin");
                future2_frontier.emplace(
                    load_frontier_layer_from_streams<uint32_t>(
                        future2_position,
                        future2_success,
                        lut,
                        options.solve.row_width,
                        options.solve.dtype
                    )
                );
                trace_outer("frontier2_load_done");
                trace_outer("frontier4_load_begin");
                future4_frontier.emplace(
                    load_frontier_layer_from_streams<uint32_t>(
                        future4_position,
                        future4_success,
                        lut,
                        options.solve.row_width,
                        options.solve.dtype
                    )
                );
                trace_outer("frontier4_load_done");
            }

            BC::BCSingleChunkSolveStats stats;
            std::optional<BC::BCSingleChunkCompactedBuild<uint32_t>> held_build;
            trace_outer("solve_begin");
            const double t0 = now_seconds();
            if (args.write_output) {
                const std::filesystem::path output_pos =
                    position_path_for(args.output_dir, args.prefix, args.ordinal);
                const std::filesystem::path output_suc =
                    success_path_for(args.output_dir, args.prefix, args.ordinal);
                std::unique_ptr<BC::BCWritableFile> position_writer =
                    make_output_writer(args, output_pos);
                std::unique_ptr<BC::BCWritableFile> success_writer =
                    make_output_writer(args, output_suc);
                uint64_t result_position_bytes = 0U;
                uint64_t result_success_bytes = 0U;
                if (args.strict_1x) {
                    const std::filesystem::path temp_dir =
                        args.output_dir /
                        (args.prefix + std::to_string(args.ordinal) +
                         "_strict_tmp_" + std::to_string(iteration));
                    const auto result =
                        BC::bc_single_chunk_solve_strict_1x_to_files<uint32_t>(
                            current_position,
                            future2_position,
                            future2_success,
                            future4_position,
                            future4_success,
                            *position_writer,
                            *success_writer,
                            temp_dir,
                            options,
                            &workspace
                        );
                    stats = result.stats;
                    result_position_bytes = result.position_bytes;
                    result_success_bytes = result.success_bytes;
                    std::error_code cleanup_ec;
                    std::filesystem::remove_all(temp_dir, cleanup_ec);
                } else if (args.frontier_api) {
                    BC::BCSingleChunkCompactedBuild<uint32_t> build =
                        BC::bc_single_chunk_solve_compacted_build_from_frontier<uint32_t>(
                            current_position,
                            *future2_frontier,
                            *future4_frontier,
                            options,
                            &workspace
                        );
                    stats = build.stats;
                    const double position_t0 = now_seconds();
                    BC::bc_single_chunk_write_position_bytes<uint32_t>(
                        *position_writer,
                        build.position_bytes,
                        &stats.output_position_write
                    );
                    stats.position_write_seconds = now_seconds() - position_t0;
                    result_position_bytes = build.position_bytes.size();

                    BC::BCPositionLayerReader compact_position(build.position_bytes, current_position.lut());
                    const double success_t0 = now_seconds();
                    result_success_bytes = BC::write_success_values_to_file<uint32_t>(
                        *success_writer,
                        compact_position,
                        options.solve.row_width,
                        options.solve.dtype,
                        build.success_values,
                        &stats.output_success_write
                    );
                    stats.success_write_seconds = now_seconds() - success_t0;
                } else {
                    const auto result =
                        BC::bc_single_chunk_solve_compacted_layer_to_files<uint32_t>(
                            current_position,
                            future2_position,
                            future2_success,
                            future4_position,
                            future4_success,
                            *position_writer,
                            *success_writer,
                            options,
                            &workspace
                        );
                    stats = result.stats;
                    result_position_bytes = result.position_bytes;
                    result_success_bytes = result.success_bytes;
                }
                position_writer.reset();
                success_writer.reset();
                if (args.direct_io && !args.keep_direct_padding) {
                    std::filesystem::resize_file(output_pos, result_position_bytes);
                    std::filesystem::resize_file(output_suc, result_success_bytes);
                }
            } else {
                if (args.frontier_api) {
                    held_build.emplace(
                        BC::bc_single_chunk_solve_compacted_build_from_frontier<uint32_t>(
                            current_position,
                            *future2_frontier,
                            *future4_frontier,
                            options,
                            &workspace
                        )
                    );
                } else {
                    held_build.emplace(
                        BC::bc_single_chunk_solve_compacted_build<uint32_t>(
                            current_position,
                            future2_position,
                            future2_success,
                            future4_position,
                            future4_success,
                            options,
                            &workspace
                        )
                    );
                }
                stats = held_build->stats;
            }
            const double total_seconds = now_seconds() - t0;
            if (stats_out) {
                write_stats_row(stats_out, iteration, stats, total_seconds);
            }
            total_rows += stats.current_rows;
            total_wall += total_seconds;
            total_recalc += stats.recalc_seconds;
            total_compact += stats.compact_seconds;
            total_position_build += stats.position_build_seconds;
            total_result_assembly += stats.result_assembly_seconds;
            total_read += read_seconds(stats);
            total_index += index_seconds(stats);
            total_prepass += stats.prepass_seconds;
            total_current_plan += stats.current_plan_seconds;
            total_future_cid_select += stats.future_cid_select_seconds;
            total_current_board_cache += stats.current_board_cache_seconds;
            total_raw_alloc += stats.raw_alloc_seconds;
            total_future_release += stats.future_release_seconds;
            total_workspace_release += stats.workspace_release_seconds;
            total_output_release += stats.output_release_seconds;
            total_write += write_seconds(stats);
            total_partial_write += stats.partial_write_seconds;
            total_partial_read += stats.partial_read_seconds;
            total_temp_prepare += stats.temp_prepare_seconds;
            total_partial_cleanup += stats.partial_cleanup_seconds;
            const double accounted = accounted_seconds(stats);
            total_accounted += accounted;

            std::cout << std::setprecision(9)
                << "iteration=" << iteration
                << " rows=" << stats.current_rows
                << " chunks=" << stats.current_chunks
                << " live_rows=" << stats.compact_live_rows
                << " read_seconds=" << read_seconds(stats)
                << " index_seconds=" << index_seconds(stats)
                << " prepass_seconds=" << stats.prepass_seconds
                << " current_plan_seconds=" << stats.current_plan_seconds
                << " future_cid_select_seconds=" << stats.future_cid_select_seconds
                << " current_board_cache_seconds=" << stats.current_board_cache_seconds
                << " current_board_cache_bytes=" << stats.current_board_cache_bytes
                << " raw_alloc_seconds=" << stats.raw_alloc_seconds
                << " future_release_seconds=" << stats.future_release_seconds
                << " recalc_seconds=" << stats.recalc_seconds
                << " compact_seconds=" << stats.compact_seconds
                << " tmp4_write_seconds=" << stats.partial_write_seconds
                << " tmp4_read_seconds=" << stats.partial_read_seconds
                << " temp_prepare_seconds=" << stats.temp_prepare_seconds
                << " partial_cleanup_seconds=" << stats.partial_cleanup_seconds
                << " workspace_release_seconds=" << stats.workspace_release_seconds
                << " output_release_seconds=" << stats.output_release_seconds
                << " position_build_seconds=" << stats.position_build_seconds
                << " result_assembly_seconds=" << stats.result_assembly_seconds
                << " position_write_seconds=" << stats.position_write_seconds
                << " success_write_seconds=" << stats.success_write_seconds
                << " accounted_seconds=" << accounted
                << " untracked_seconds=" << (total_seconds - accounted)
                << " solve_window_seconds=" << total_seconds
                << " total_seconds=" << total_seconds
                << " total_mrows_per_sec="
                << (total_seconds > 0.0
                        ? static_cast<double>(stats.current_rows) / total_seconds / 1.0e6
                        : 0.0)
                << '\n';
        }

        if (stats_out) {
            stats_out.flush();
        }
        std::cout << std::setprecision(12)
            << "summary"
            << " ordinal=" << args.ordinal
            << " repeats=" << args.repeats
            << " total_rows=" << total_rows
            << " total_read_seconds=" << total_read
            << " total_index_seconds=" << total_index
            << " total_prepass_seconds=" << total_prepass
            << " total_current_plan_seconds=" << total_current_plan
            << " total_future_cid_select_seconds=" << total_future_cid_select
            << " total_current_board_cache_seconds=" << total_current_board_cache
            << " total_raw_alloc_seconds=" << total_raw_alloc
            << " total_future_release_seconds=" << total_future_release
            << " total_recalc_seconds=" << total_recalc
            << " total_compact_seconds=" << total_compact
            << " total_tmp4_write_seconds=" << total_partial_write
            << " total_tmp4_read_seconds=" << total_partial_read
            << " total_temp_prepare_seconds=" << total_temp_prepare
            << " total_partial_cleanup_seconds=" << total_partial_cleanup
            << " total_workspace_release_seconds=" << total_workspace_release
            << " total_output_release_seconds=" << total_output_release
            << " total_position_build_seconds=" << total_position_build
            << " total_result_assembly_seconds=" << total_result_assembly
            << " total_write_seconds=" << total_write
            << " total_accounted_seconds=" << total_accounted
            << " total_untracked_seconds=" << (total_wall - total_accounted)
            << " solve_window_seconds=" << total_wall
            << " total_wall_seconds=" << total_wall
            << " wall_mrows_per_sec="
            << (total_wall > 0.0 ? static_cast<double>(total_rows) / total_wall / 1.0e6 : 0.0)
            << " recalc_mrows_per_sec="
            << (total_recalc > 0.0 ? static_cast<double>(total_rows) / total_recalc / 1.0e6 : 0.0)
            << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "bc_single_chunk_solve_layer_bench failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
