#pragma once

#include "BCFamilyGenerationRunner.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace BCGenerationBench {

[[nodiscard]] inline std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

inline void print_usage(std::ostream &out, const char *name) {
    out
        << name << " --pattern freeN --target-rank N --extra-steps N [options]\n"
        << "  --output-dir DIR --stats-csv PATH --num-threads N\n"
        << "  --family-modulus N / --cell-modulus N\n"
        << "  --target-output-io buffered|direct|memory\n"
        << "  --target-direct-queue-depth N\n"
        << "  --family-route auto|resident|single|family\n";
}

inline void apply_target_extra(BC::BCFamilyGenerationRunOptions &options, uint32_t target_extra) {
    if (target_extra == 0U) {
        return;
    }
    if ((target_extra & 1U) != 0U) {
        throw std::invalid_argument("--target-extra must be even");
    }
    if (options.target_rank >= 31U) {
        throw std::invalid_argument("--target-rank is too large for --target-extra");
    }
    const uint64_t target_tile = 1ULL << options.target_rank;
    const uint64_t target_half = target_tile / 2ULL;
    const uint64_t forward_steps = target_extra / 2ULL;
    if (forward_steps + 1ULL < target_half) {
        throw std::invalid_argument("--target-extra is too small for target rank");
    }
    const uint64_t extra_steps = forward_steps - target_half + 1ULL;
    if (extra_steps > UINT32_MAX) {
        throw std::invalid_argument("--target-extra produces too many extra steps");
    }
    options.extra_steps = static_cast<uint32_t>(extra_steps);
}

[[nodiscard]] inline BC::BCFamilyGenerationRunOptions parse_generation_runner_args(
    int argc,
    char **argv,
    const char *bench_name,
    BC::BCFamilyGenerationRoute default_route,
    bool allow_route_override
) {
    BC::BCFamilyGenerationRunOptions options;
    options.family_route = default_route;
    options.output_dir = std::filesystem::path("tmp") / (std::string(bench_name) + "_generated");
    options.stats_csv = std::filesystem::path("tmp") / (std::string(bench_name) + "_generation.csv");

    uint32_t target_extra = 0U;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(std::cout, bench_name);
            std::exit(0);
        } else if (key == "--pattern") {
            options.pattern = require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-rank") {
            options.target_rank =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--extra-steps") {
            options.extra_steps =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--target-extra") {
            target_extra =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--output-dir" || key == "--file-dir") {
            options.output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--stats-csv") {
            options.stats_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--num-threads" || key == "--threads") {
            options.num_threads = std::stoi(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--batch-size") {
            options.batch_size =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--pending-buffer") {
            options.pending_buffer =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-work-schedule-chunk") {
            options.family_work_schedule_chunk =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-source-words-per-item") {
            options.family_source_words_per_item =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-reserve-buckets") {
            options.family_reserve_buckets =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-reserve-bitmap-words") {
            options.family_reserve_bitmap_words =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--warmup-extra") {
            options.warmup_extra =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-modulus" || key == "--cell-modulus") {
            options.family_modulus =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-route") {
            const BC::BCFamilyGenerationRoute requested =
                BC::bc_parse_family_route(require_value(argc, argv, i, key.c_str()));
            if (!allow_route_override && requested != default_route) {
                throw std::invalid_argument("this benchmark target has a fixed production route");
            }
            options.family_route = requested;
        } else if (key == "--family-blob") {
            options.family_blob = require_value(argc, argv, i, key.c_str());
        } else if (key == "--family-position-io" || key == "--position-io") {
            const std::string value = require_value(argc, argv, i, key.c_str());
            options.family_position_io = value == "direct" ? "direct-rank-first" : value;
        } else if (key == "--family-source-io") {
            options.family_source_io = require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-output-io") {
            const std::string value = require_value(argc, argv, i, key.c_str());
            if (value == "direct") {
                options.family_position_io = "direct-rank-first";
            } else if (value == "buffered" || value == "memory") {
                options.family_position_io = "buffered";
            } else {
                throw std::invalid_argument("--target-output-io must be buffered, direct, or memory");
            }
        } else if (key == "--target-direct-queue-depth" ||
                   key == "--position-direct-queue-depth" ||
                   key == "--direct-queue-depth") {
            options.direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-blob-checksum") {
            options.family_blob_checksum = true;
        } else if (key == "--family-memory-checkpoints") {
            options.family_memory_checkpoints = true;
        } else if (key == "--verify-layer-rows") {
            options.verify_layer_rows = true;
        } else if (key == "--no-verify-layer-rows") {
            options.verify_layer_rows = false;
        } else if (key == "--output-inspect") {
            options.output_inspect = true;
        } else if (key == "--no-output-inspect") {
            options.output_inspect = false;
        } else if (key == "--target-direct-overlapped" ||
                   key == "--file-backed" ||
                   key == "--detail-timing" ||
                   key == "--no-detail-timing") {
            // Legacy benchmark-only switches. Production runner owns timing and IO.
        } else if (key == "--seed-mode" ||
                   key == "--ex-stats-csv" ||
                   key == "--prefix" ||
                   key == "--cell-chunk-size" ||
                   key == "--cell-chunk-count" ||
                   key == "--expected-output-rows" ||
                   key == "--free9-sum-mode" ||
                   key == "--singlechunk-mode") {
            (void)require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-sum" || key == "--source" || key == "--output") {
            throw std::invalid_argument(
                std::string(bench_name) +
                " no longer implements single-layer benchmark compute; use production full-run inputs");
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }

    apply_target_extra(options, target_extra);
    return options;
}

inline int run_generation_runner_bench(
    int argc,
    char **argv,
    const char *bench_name,
    BC::BCFamilyGenerationRoute default_route,
    bool allow_route_override
) {
    try {
        BC::BCFamilyGenerationRunOptions options =
            parse_generation_runner_args(argc, argv, bench_name, default_route, allow_route_override);
        const BC::BCFamilyGenerationRunResult result = BC::bc_family_generation_full_run(
            options,
            [](const BC::BCFamilyGenerationRunLayerMetric &metric) {
                std::cout
                    << "kind=" << metric.kind
                    << " ordinal=" << metric.ordinal
                    << " layer_sum=" << metric.layer_sum
                    << " rows=" << metric.output_rows
                    << " total_seconds=" << metric.total_seconds
                    << '\n';
            });
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

} // namespace BCGenerationBench
