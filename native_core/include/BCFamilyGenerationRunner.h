#pragma once

#include "BCFamilyRoutePlanner.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace BC {

struct BCFamilyGenerationRunOptions {
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    uint32_t extra_steps = 36U;
    int num_threads = 0;
    uint32_t batch_size = 8192U;
    uint32_t pending_buffer = 0U;
    uint32_t family_work_schedule_chunk = 1U;
    uint32_t family_source_words_per_item = 64U;
    uint32_t family_reserve_buckets = 0U;
    uint32_t family_reserve_bitmap_words = 0U;
    uint32_t warmup_extra = 16U;
    bool verify_layer_rows = false;
    bool output_inspect = false;
    std::string family_blob = "direct";
    std::string family_position_io = "direct-rank-first";
    std::string family_source_io = "direct-auto";
    bool family_blob_checksum = false;
    bool family_memory_checkpoints = false;
    uint32_t family_modulus = 29U;
    BCFamilyGenerationRoute family_route = BCFamilyGenerationRoute::Auto;
    uint32_t direct_queue_depth = 8U;
    std::filesystem::path output_dir;
    std::filesystem::path stats_csv;
};

struct BCFamilyGenerationRunLayerMetric {
    std::string kind = "generation";
    uint32_t ordinal = 0U;
    uint32_t layer_sum = 0U;
    uint64_t input_rows = 0U;
    uint64_t output_rows = 0U;
    uint64_t logical_size = 0U;
    uint32_t target_modulus = 0U;
    BCFamilyGenerationRoute route = BCFamilyGenerationRoute::Family;
    double total_seconds = 0.0;
    std::filesystem::path output_path;
};

struct BCFamilyGenerationRunResult {
    std::vector<BCFamilyGenerationRunLayerMetric> layers;
    bool completed = false;
};

using BCFamilyGenerationLayerCallback =
    std::function<void(const BCFamilyGenerationRunLayerMetric &)>;

BCFamilyGenerationRunResult bc_family_generation_full_run(
    const BCFamilyGenerationRunOptions &options,
    const BCFamilyGenerationLayerCallback &callback = {});

} // namespace BC
