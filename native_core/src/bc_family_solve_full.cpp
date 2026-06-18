#include "BCFamilySolveRunner.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    BC::BCFamilySolveRunOptions run;
    std::filesystem::path stats_csv = "tmp/bc_family_solve_full_stats.csv";
    std::filesystem::path summary_csv = "tmp/bc_family_solve_full_summary.csv";
};

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

void print_usage(std::ostream &out) {
    out
        << "bc_family_solve_full --position-dir DIR --output-dir DIR --prefix PREFIX [options]\n"
        << "  --archive-dir DIR\n"
        << "  --stats-csv PATH --summary-csv PATH\n"
        << "  --target-rank N --success-target-rank N --success-dtype MODE\n"
        << "  --family-modulus N --threads N --direct-io --direct-queue-depth N\n"
        << "  --solve-route auto|resident|single|family\n"
        << "  --available-memory-bytes N --available-memory-mib N\n"
        << "  --deletion-threshold R --relative-deletion-threshold R\n"
        << "  --compress --compress-temp-files --optimal-branch-only\n"
        << "  --start-ordinal N --min-ordinal N --restart --no-resume\n";
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    args.run.generated_position_dir = "tmp/free10_256_resident_generated_m17";
    args.run.solved_output_dir = "tmp/free10_256_family_solve_m17";
    args.run.prefix = "free10_256_";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(std::cout);
            std::exit(0);
        } else if (key == "--position-dir" || key == "--generated-position-dir") {
            args.run.generated_position_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--output-dir" || key == "--solved-output-dir") {
            args.run.solved_output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--archive-dir") {
            args.run.archive_output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--summary-csv") {
            args.summary_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--prefix") {
            args.run.prefix = require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-rank") {
            args.run.target_rank =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--success-target-rank") {
            args.run.success_target_rank = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--success-dtype" || key == "--dtype") {
            args.run.success_dtype = parse_success_dtype(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--canonical-symm-mode") {
            args.run.canonical_symm_mode = parse_symm_mode(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--spawn-rate4") {
            args.run.spawn_rate4 = std::stod(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--threads") {
            args.run.num_threads = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--batch-size") {
            args.run.canonical_batch_size =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-modulus") {
            args.run.family_modulus =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--solve-route") {
            args.run.solve_route = BC::bc_parse_solve_route(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--available-memory-bytes") {
            args.run.available_memory_override_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--available-memory-mib") {
            args.run.available_memory_override_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str()))) *
                1024ULL * 1024ULL;
        } else if (key == "--future-reuse-max-families") {
            args.run.future_reuse_max_families =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--future-index-recycle-bytes") {
            args.run.future_index_recycle_max_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--future-index-recycle-mib") {
            args.run.future_index_recycle_max_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str()))) *
                1024ULL * 1024ULL;
        } else if (key == "--source-words-per-item") {
            args.run.source_words_per_item =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--work-schedule-chunk") {
            args.run.work_schedule_chunk =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--cell-parallel-min-work-items") {
            args.run.cell_parallel_min_work_items =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--final-pending-value-cap-bytes") {
            args.run.final_pending_value_memory_cap_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--final-pending-value-cap-mib") {
            args.run.final_pending_value_memory_cap_bytes =
                static_cast<uint64_t>(std::stoull(require_value(argc, argv, i, key.c_str()))) *
                1024ULL * 1024ULL;
        } else if (key == "--direct-io") {
            args.run.direct_io = true;
        } else if (key == "--keep-direct-padding") {
            args.run.keep_direct_padding = true;
        } else if (key == "--trim-direct-padding") {
            args.run.keep_direct_padding = false;
        } else if (key == "--direct-queue-depth") {
            args.run.direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--start-ordinal") {
            args.run.start_ordinal =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--min-ordinal") {
            args.run.min_ordinal =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--restart") {
            args.run.force_restart = true;
        } else if (key == "--no-resume") {
            args.run.resume_from_checkpoint = false;
        } else if (key == "--deletion-threshold") {
            args.run.deletion_threshold = std::stod(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--relative-deletion-threshold") {
            args.run.relative_deletion_threshold =
                std::stod(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--deletion-threshold-signal") {
            args.run.deletion_threshold_signal_path = require_value(argc, argv, i, key.c_str());
        } else if (key == "--compress") {
            args.run.compress = true;
        } else if (key == "--compress-temp-files") {
            args.run.compress_temp_files = true;
        } else if (key == "--optimal-branch-only") {
            args.run.optimal_branch_only = true;
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.run.success_target_rank < 0) {
        args.run.success_target_rank = static_cast<int>(args.run.target_rank);
    }
    return args;
}

void write_stats_header(std::ostream &out) {
    out
        << "kind,ordinal,solve_route,layer_sum,current_rows,live_rows,zero_pruned_rows,"
        << "archive_live_rows,threshold_pruned_rows,position_bytes,success_bytes,"
        << "temp_compressed_bytes,"
        << "route_available_memory_bytes,route_resident_required_bytes,"
        << "route_single_required_bytes,route_required_bytes,"
        << "open_seconds,partition_seconds,solve_call_seconds,spawn4_compute_seconds,"
        << "spawn2_compute_seconds,compact_seconds,temp_write_seconds,temp_read_seconds,"
        << "temp_compress_seconds,final_stage_write_seconds,final_stage_read_seconds,"
        << "position_write_seconds,"
        << "success_write_seconds,archive_scan_seconds,archive_prune_write_seconds,"
        << "final_compress_seconds,total_seconds\n";
}

void write_stats_row(std::ostream &out, const BC::BCFamilySolveRunLayerMetric &m) {
    const BC::BCFamilySolveStats &fs = m.family_stats;
    const BC::BCSingleChunkSolveStats &s = fs.single;
    out
        << m.kind << ','
        << m.ordinal << ','
        << m.solve_route << ','
        << m.layer_sum << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.archive_live_rows << ','
        << m.threshold_pruned_rows << ','
        << m.position_bytes << ','
        << m.success_bytes << ','
        << m.temp_compressed_bytes << ','
        << m.route_available_memory_bytes << ','
        << m.route_resident_required_bytes << ','
        << m.route_single_required_bytes << ','
        << m.route_required_bytes << ','
        << m.open_seconds << ','
        << m.partition_seconds << ','
        << m.solve_call_seconds << ','
        << (m.has_family_stats ? fs.spawn4_cell_compute_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.spawn2_cell_compute_seconds : 0.0) << ','
        << (m.has_family_stats ? s.compact_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_write_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_read_seconds : 0.0) << ','
        << m.temp_compress_seconds << ','
        << (m.has_family_stats ? fs.final_stage_write_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.final_stage_read_seconds : 0.0) << ','
        << m.position_write_seconds << ','
        << m.success_write_seconds << ','
        << m.archive_scan_seconds << ','
        << m.archive_prune_write_seconds << ','
        << m.final_compress_seconds << ','
        << m.total_seconds << '\n';
}

void write_summary(
    const Args &args,
    const BC::BCFamilySolveRunResult &result,
    double wall_seconds
) {
    if (!args.summary_csv.parent_path().empty()) {
        std::filesystem::create_directories(args.summary_csv.parent_path());
    }
    std::ofstream out(args.summary_csv);
    if (!out) {
        throw std::runtime_error("failed to open summary CSV: " + args.summary_csv.string());
    }
    out << std::setprecision(12);
    uint64_t solve_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_rows = 0U;
    uint64_t archive_rows = 0U;
    uint64_t threshold_rows = 0U;
    uint64_t temp_compressed_bytes = 0U;
    double solve_seconds = 0.0;
    double archive_seconds = 0.0;
    double temp_compress_seconds = 0.0;
    double final_compress_seconds = 0.0;
    for (const BC::BCFamilySolveRunLayerMetric &m : result.layers) {
        if (m.kind == "solve") {
            solve_rows += m.current_rows;
            live_rows += m.live_rows;
            zero_rows += m.zero_pruned_rows;
            solve_seconds += m.total_seconds;
            temp_compressed_bytes += m.temp_compressed_bytes;
            temp_compress_seconds += m.temp_compress_seconds;
            final_compress_seconds += m.final_compress_seconds;
        } else if (m.kind == "archive") {
            archive_rows += m.archive_live_rows;
            threshold_rows += m.threshold_pruned_rows;
            archive_seconds += m.total_seconds;
            final_compress_seconds += m.final_compress_seconds;
        }
    }
    const double wall_mrows = wall_seconds > 0.0
        ? static_cast<double>(solve_rows) / wall_seconds / 1.0e6
        : 0.0;
    out
        << "generated_position_dir,solved_output_dir,archive_output_dir,min_ordinal,max_ordinal,"
        << "completed,solve_rows,live_rows,zero_pruned_rows,archive_live_rows,"
        << "threshold_pruned_rows,temp_compressed_bytes,solve_seconds,archive_seconds,"
        << "temp_compress_seconds,final_compress_seconds,wall_seconds,wall_mrows_per_sec\n"
        << args.run.generated_position_dir.string() << ','
        << args.run.solved_output_dir.string() << ','
        << args.run.archive_output_dir.string() << ','
        << result.min_ordinal << ','
        << result.max_ordinal << ','
        << (result.completed ? 1 : 0) << ','
        << solve_rows << ','
        << live_rows << ','
        << zero_rows << ','
        << archive_rows << ','
        << threshold_rows << ','
        << temp_compressed_bytes << ','
        << solve_seconds << ','
        << archive_seconds << ','
        << temp_compress_seconds << ','
        << final_compress_seconds << ','
        << wall_seconds << ','
        << wall_mrows << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!args.stats_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.stats_csv.parent_path());
        }
        std::ofstream stats(args.stats_csv);
        if (!stats) {
            throw std::runtime_error("failed to open stats CSV: " + args.stats_csv.string());
        }
        stats << std::setprecision(12);
        write_stats_header(stats);

        const double begin = BC::detail::bc_family_solve_runner_now_seconds();
        BC::BCFamilySolveRunResult result = BC::bc_family_solve_full_run(
            args.run,
            [&](const BC::BCFamilySolveRunLayerMetric &metric) {
                write_stats_row(stats, metric);
                stats.flush();
                std::cout << std::setprecision(9)
                    << "kind=" << metric.kind
                    << " ordinal=" << metric.ordinal
                    << " route=" << metric.solve_route
                    << " rows=" << metric.current_rows
                    << " live_rows=" << metric.live_rows
                    << " total_seconds=" << metric.total_seconds
                    << " position_bytes=" << metric.position_bytes
                    << " success_bytes=" << metric.success_bytes
                    << '\n';
            });
        const double wall_seconds = BC::detail::bc_family_solve_runner_now_seconds() - begin;
        write_summary(args, result, wall_seconds);
        std::cout << std::setprecision(12)
            << "summary"
            << " layers=" << result.layers.size()
            << " completed=" << (result.completed ? 1 : 0)
            << " wall_seconds=" << wall_seconds
            << '\n';
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_solve_full failed: " << ex.what() << '\n';
        return 1;
    }
}
