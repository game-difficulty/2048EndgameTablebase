#pragma once

#include "BCFamilySolveRunner.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace BCSolveBench {

struct Args {
    BC::BCFamilySolveRunOptions run;
    std::filesystem::path stats_csv;
    std::filesystem::path summary_csv;
};

[[nodiscard]] inline std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

[[nodiscard]] inline int parse_symm_mode(const std::string &value) {
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

[[nodiscard]] inline BC::BCSuccessDTypeMode parse_success_dtype(const std::string &value) {
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

inline void print_usage(std::ostream &out, const char *name) {
    out
        << name << " --position-dir DIR --output-dir DIR --prefix PREFIX [options]\n"
        << "  --archive-dir DIR --stats-csv PATH --summary-csv PATH\n"
        << "  --target-rank N --success-target-rank N --success-dtype MODE\n"
        << "  --family-modulus N --threads N --direct-io --direct-queue-depth N\n"
        << "  --solve-route auto|resident|single|family\n";
}

[[nodiscard]] inline Args parse_solve_runner_args(
    int argc,
    char **argv,
    const char *bench_name,
    BC::BCSolveRoute default_route,
    bool allow_route_override
) {
    Args args;
    args.run.solve_route = default_route;
    args.run.generated_position_dir = std::filesystem::path("tmp") / (std::string(bench_name) + "_generated");
    args.run.solved_output_dir = std::filesystem::path("tmp") / (std::string(bench_name) + "_solved");
    args.run.archive_output_dir = args.run.solved_output_dir;
    args.run.prefix = "free9_256_";
    args.stats_csv = std::filesystem::path("tmp") / (std::string(bench_name) + "_layers.csv");
    args.summary_csv = std::filesystem::path("tmp") / (std::string(bench_name) + "_summary.csv");

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(std::cout, bench_name);
            std::exit(0);
        } else if (key == "--position-dir" || key == "--generated-position-dir" ||
                   key == "--current-dir") {
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
        } else if (key == "--threads" || key == "--num-threads") {
            args.run.num_threads = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--batch-size") {
            args.run.canonical_batch_size =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-modulus" || key == "--cell-modulus") {
            args.run.family_modulus =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--solve-route") {
            const BC::BCSolveRoute requested =
                BC::bc_parse_solve_route(require_value(argc, argv, i, key.c_str()));
            if (!allow_route_override && requested != default_route) {
                throw std::invalid_argument("this benchmark target has a fixed production solve route");
            }
            args.run.solve_route = requested;
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
        } else if (key == "--buffered-io") {
            args.run.direct_io = false;
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
            args.run.relative_deletion_threshold = std::stod(require_value(argc, argv, i, key.c_str()));
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
    if (args.run.archive_output_dir.empty()) {
        args.run.archive_output_dir = args.run.solved_output_dir;
    }
    return args;
}

inline void write_stats_header(std::ostream &out) {
    out
        << "kind,solve_route,ordinal,layer_sum,current_rows,live_rows,zero_pruned_rows,"
        << "archive_live_rows,total_seconds,solve_call_seconds\n";
}

inline void write_stats_row(std::ostream &out, const BC::BCFamilySolveRunLayerMetric &m) {
    out
        << m.kind << ','
        << m.solve_route << ','
        << m.ordinal << ','
        << m.layer_sum << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.archive_live_rows << ','
        << m.total_seconds << ','
        << m.solve_call_seconds << '\n';
}

inline int run_solve_runner_bench(
    int argc,
    char **argv,
    const char *bench_name,
    BC::BCSolveRoute default_route,
    bool allow_route_override
) {
    try {
        Args args = parse_solve_runner_args(argc, argv, bench_name, default_route, allow_route_override);
        std::optional<std::ofstream> stats;
        if (!args.stats_csv.empty()) {
            if (!args.stats_csv.parent_path().empty()) {
                std::filesystem::create_directories(args.stats_csv.parent_path());
            }
            stats.emplace(args.stats_csv);
            if (!*stats) {
                throw std::runtime_error("failed to open stats CSV");
            }
            *stats << std::setprecision(12);
            write_stats_header(*stats);
        }
        const BC::BCFamilySolveRunResult result = BC::bc_family_solve_full_run(
            args.run,
            [&](const BC::BCFamilySolveRunLayerMetric &metric) {
                if (stats) {
                    write_stats_row(*stats, metric);
                    stats->flush();
                }
                std::cout
                    << "kind=" << metric.kind
                    << " route=" << metric.solve_route
                    << " ordinal=" << metric.ordinal
                    << " rows=" << metric.live_rows
                    << " total_seconds=" << metric.total_seconds
                    << '\n';
            });
        if (!args.summary_csv.empty()) {
            if (!args.summary_csv.parent_path().empty()) {
                std::filesystem::create_directories(args.summary_csv.parent_path());
            }
            std::ofstream summary(args.summary_csv);
            if (!summary) {
                throw std::runtime_error("failed to open summary CSV");
            }
            summary << "layers,completed,min_ordinal,max_ordinal\n"
                    << result.layers.size() << ','
                    << (result.completed ? 1 : 0) << ','
                    << result.min_ordinal << ','
                    << result.max_ordinal << '\n';
        }
        std::cout
            << "summary layers=" << result.layers.size()
            << " completed=" << (result.completed ? 1 : 0)
            << '\n';
        return result.completed ? 0 : 2;
    } catch (const std::exception &ex) {
        std::cerr << bench_name << " failed: " << ex.what() << '\n';
        return 1;
    }
}

} // namespace BCSolveBench
