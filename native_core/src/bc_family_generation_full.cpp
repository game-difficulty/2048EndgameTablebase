#include "BCFamilyGenerationRunner.h"

#include <exception>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

void print_usage(std::ostream &out) {
    out
        << "bc_family_generation_full --pattern freeN --target-rank N --extra-steps N --output-dir DIR [options]\n"
        << "  Non-free BC patterns should be started through formation_core runtime so seed/mask metadata is supplied.\n"
        << "  --stats-csv PATH --num-threads N --family-modulus N\n"
        << "  --family-route auto|resident|single|family\n"
        << "  --target-direct-queue-depth N\n"
        << "  --family-blob buffered|direct\n"
        << "  --family-position-io buffered|direct-rank-first\n"
        << "  --family-source-io buffered|direct|direct-auto\n";
}

[[nodiscard]] BC::BCFamilyGenerationRunOptions parse_args(int argc, char **argv) {
    BC::BCFamilyGenerationRunOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(std::cout);
            std::exit(0);
        } else if (key == "--pattern") {
            options.pattern = require_value(argc, argv, i, key.c_str());
        } else if (key == "--target-rank") {
            options.target_rank =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--extra-steps") {
            options.extra_steps =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--output-dir") {
            options.output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--stats-csv") {
            options.stats_csv = require_value(argc, argv, i, key.c_str());
        } else if (key == "--num-threads") {
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
        } else if (key == "--family-blob") {
            options.family_blob = require_value(argc, argv, i, key.c_str());
        } else if (key == "--family-position-io") {
            options.family_position_io = require_value(argc, argv, i, key.c_str());
        } else if (key == "--family-source-io") {
            options.family_source_io = require_value(argc, argv, i, key.c_str());
        } else if (key == "--family-blob-checksum") {
            options.family_blob_checksum = true;
        } else if (key == "--family-memory-checkpoints") {
            options.family_memory_checkpoints = true;
        } else if (key == "--family-modulus") {
            options.family_modulus =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--family-route") {
            options.family_route =
                BC::bc_parse_family_route(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--target-direct-queue-depth") {
            options.direct_queue_depth =
                static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, key.c_str())));
        } else if (key == "--no-verify-layer-rows") {
            options.verify_layer_rows = false;
        } else if (key == "--no-output-inspect") {
            options.output_inspect = false;
        } else if (key == "--family-output" || key == "--generation-chain" ||
                   key == "--planner" || key == "--target-output-io" ||
                   key == "--singlechunk-mode") {
            (void)require_value(argc, argv, i, key.c_str());
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        BC::BCFamilyGenerationRunOptions options = parse_args(argc, argv);
        BC::BCFamilyGenerationRunResult result = BC::bc_family_generation_full_run(
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
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_generation_full failed: " << ex.what() << '\n';
        return 1;
    }
}
