#pragma once

#include "BCCellBuilder.h"
#include "BCFamilyMutableStore.h"
#include "BCFamilyPartitionPolicy.h"
#include "BCFamilyPositionWriter.h"
#include "BCPositionCellLoader.h"
#include "FormationRuntime.h"

#include <array>
#include <cstdint>
#include <vector>

namespace BC {

struct BCFamilyStreamingGenerationSource {
    const BCPositionStreamingReader *position = nullptr;
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;
};

struct BCFamilyMemoryCheckpoint {
    const char *label = "";
    uint64_t process_working_set_bytes = 0U;
    uint64_t process_peak_working_set_bytes = 0U;
    uint64_t process_baseline_working_set_bytes = 0U;
    uint64_t process_private_bytes = 0U;
    uint64_t process_pagefile_bytes = 0U;
    uint64_t process_peak_pagefile_bytes = 0U;

    uint64_t active_builder_bytes = 0U;
    uint64_t thread_workspace_bytes = 0U;
    uint64_t source_loaded_payload_bytes = 0U;
    uint64_t source_loaded_allocated_bytes = 0U;
    uint64_t source_reader_metadata_bytes = 0U;
    uint64_t active_index_bytes = 0U;
    uint64_t range_work_bytes = 0U;
    uint64_t pass_cache_bytes = 0U;
    uint64_t store_static_metadata_bytes = 0U;
    uint64_t store_allocated_bytes = 0U;
    uint64_t position_writer_bytes = 0U;
    uint64_t finalized_payload_bytes = 0U;
    uint64_t external_staging_bytes = 0U;
    // Cumulative bytes of cell-local builders that were destroyed earlier in
    // the layer. These bytes are no longer live BC objects, but Windows may
    // keep the underlying heap pages in the process working set.
    uint64_t released_builder_bytes_total = 0U;
    uint64_t last_release_batch_builder_bytes = 0U;

    uint64_t accounted_bytes = 0U;
    uint64_t residual_bytes = 0U;
    uint64_t baseline_adjusted_residual_bytes = 0U;
};

using BCFamilyMemoryCheckpointCallback = void (*)(
    BCFamilyMemoryCheckpoint &checkpoint,
    void *context
);

struct BCFamilyGenerationOptions {
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    // Per worker and per active target cell. FamilyChain keeps one fixed
    // contiguous buffer segment for each cell in the current fanout cross.
    uint32_t pending_insert_buffer_size = 512U;
    uint32_t source_bitmap_words_per_work_item = 64U;
    uint32_t source_work_schedule_chunk = 1U;
    uint32_t new_cell_reserve_buckets = 1024U;
    uint32_t new_cell_reserve_bitmap_words = 4096U;
    bool enable_neighbor_cell_reserve = true;
    uint32_t neighbor_reserve_radius = 2U;
    double neighbor_reserve_quantile = 0.90;
    double neighbor_reserve_scale = 1.25;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    bool enforce_three_family_window = true;
    bool collect_hot_counters = false;
    const std::array<uint32_t, 16U> *family_tile_sum_values = nullptr;
    BCFamilyPartitionPolicy family_partition_policy = BCFamilyPartitionPolicy::modulo(29U);
    // Required for the modulo partition map. The vector is used only to build
    // exact-coord groups for each dense partition family; hot encode maps
    // coords by the modulo policy directly.
    const std::vector<LayerSum> *family_possible_8tile_sums = nullptr;
    BCFamilyMemoryCheckpointCallback memory_checkpoint_callback = nullptr;
    void *memory_checkpoint_context = nullptr;
    uint64_t memory_checkpoint_external_staging_bytes = 0U;
    int success_target_rank = 0;
    const std::vector<uint8_t> *success_shifts = nullptr;
    LayerSum success_check_min_source_layer_sum = 0U;
    bool success_check_all_cells = false;
    bool keep_only_success_generated_boards = false;
    BCCellFinalizeOptions finalize_options = {};
};

struct BCFamilyGenerationStats {
    uint64_t source_families_processed = 0U;
    uint64_t source_cells_loaded = 0U;
    uint64_t source_bytes_read = 0U;
    uint64_t source_backend_read_ops = 0U;
    uint64_t source_backend_read_bytes = 0U;
    uint64_t blob_requested_extents = 0U;
    uint64_t blob_coalesced_extents = 0U;
    uint64_t blob_read_bytes = 0U;
    uint64_t blob_backend_read_ops = 0U;
    uint64_t blob_backend_read_bytes = 0U;
    uint64_t blob_append_count = 0U;
    uint64_t blob_bytes_written = 0U;
    uint64_t blob_backend_write_ops = 0U;
    uint64_t blob_backend_write_bytes = 0U;
    double blob_backend_read_seconds = 0.0;
    double blob_restore_seconds = 0.0;
    double blob_backend_write_seconds = 0.0;
    double blob_flush_seconds = 0.0;
    uint64_t writer_bucket_bytes = 0U;
    uint64_t writer_rank_payload_bytes = 0U;
    uint64_t writer_bucket_stage_flushes = 0U;
    uint64_t writer_rank_stage_flushes = 0U;
    uint64_t writer_bucket_stage_write_bytes = 0U;
    uint64_t writer_rank_stage_write_bytes = 0U;
    uint64_t source_boards_scanned = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_all_dir_calls = 0U;
    uint64_t selective_move_calls = 0U;
    uint64_t move_results_produced = 0U;
    uint64_t canonicalized_candidates = 0U;
    uint64_t encode_attempts = 0U;
    uint64_t encode_valid_candidates = 0U;
    uint64_t encoded_candidates = 0U;
    uint64_t duplicate_candidates = 0U;
    uint64_t target_window_skips = 0U;
    uint64_t family_buffer_flushes = 0U;
    uint64_t family_buffer_flush_items = 0U;
    uint64_t family_buffer_flush_max_items = 0U;
    uint64_t family_builder_bind_calls = 0U;
    uint64_t family_insert_hash_lookups = 0U;
    uint64_t family_insert_hash_probe_steps = 0U;
    uint64_t target_cells_created = 0U;
    uint64_t target_cells_reloaded = 0U;
    uint64_t target_cells_dumped = 0U;
    uint64_t target_cells_finalized = 0U;
    uint64_t target_cells_kept = 0U;
    uint64_t family_builder_hash_grows = 0U;
    uint64_t family_builder_bitmap_grows = 0U;
    uint64_t family_builder_hash_replaced_bytes = 0U;
    uint64_t family_builder_bitmap_replaced_bytes = 0U;
    uint64_t source_work_items = 0U;
    uint64_t source_bucket_ranges = 0U;
    uint64_t source_bitmap_words_scanned = 0U;
    uint64_t parallel_region_count = 0U;
    uint64_t active_cell_peak = 0U;
    uint64_t target_active_cell_peak = 0U;
    uint64_t source_loaded_cell_peak = 0U;
    uint64_t source_loaded_bytes_peak = 0U;
    uint64_t source_load_range_bytes_peak = 0U;
    uint64_t active_family_window_peak = 0U;
    uint64_t active_builder_bytes_peak = 0U;
    uint64_t thread_workspace_bytes_peak = 0U;
    uint64_t pass_cache_bytes_peak = 0U;
    uint64_t active_index_bytes_peak = 0U;
    uint64_t range_work_bytes_peak = 0U;
    uint64_t dump_buffer_bytes_peak = 0U;
    uint64_t reload_dump_bytes_peak = 0U;
    uint64_t restore_builder_bytes_peak = 0U;
    uint64_t finalize_dump_bytes_peak = 0U;
    uint64_t finalize_payload_bytes_peak = 0U;
    uint64_t store_static_metadata_bytes = 0U;
    uint64_t released_builder_bytes_total = 0U;
    uint64_t released_builder_bytes_peak = 0U;
    uint64_t release_batch_builder_bytes_peak = 0U;
    uint64_t store_allocated_bytes_peak = 0U;
    uint64_t memory_checkpoint_count = 0U;
    uint64_t memory_checkpoint_process_bytes_peak = 0U;
    uint64_t memory_checkpoint_process_peak_bytes_peak = 0U;
    uint64_t memory_checkpoint_process_private_bytes_peak = 0U;
    uint64_t memory_checkpoint_process_pagefile_bytes_peak = 0U;
    uint64_t memory_checkpoint_process_peak_pagefile_bytes_peak = 0U;
    uint64_t memory_checkpoint_accounted_bytes_peak = 0U;
    uint64_t memory_checkpoint_residual_bytes_peak = 0U;
    uint64_t memory_checkpoint_baseline_adjusted_residual_peak = 0U;
    uint32_t memory_checkpoint_residual_peak_label = 0U;
    uint32_t memory_checkpoint_baseline_adjusted_residual_peak_label = 0U;

    double source_load_seconds = 0.0;
    double build_work_seconds = 0.0;
    double parallel_seconds = 0.0;
    double dump_seconds = 0.0;
    double reload_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double generation_seconds = 0.0;
};

struct BCFamilyLoadedPassBenchmarkResult {
    BCFamilyGenerationStats stats;
    double timed_seconds = 0.0;
    uint64_t unique_encoded_candidates = 0U;
};

// Generates one target layer with FamilyChain semantics:
//   +4 source -> target mutable, no finalize
//   +2 source -> same target mutable, boundary finalize/write
//
// The source readers must be streaming readers. The target mutable store must
// be backed by cell-local mutable builders. position_writer must already be
// begun for target_axis.
BCFamilyGenerationStats generate_family_position_layer_v1(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource *source4,
    const BCFamilyStreamingGenerationSource &source2,
    BCFamilyMutableStore &target_store,
    BCFamilyPositionWriter &position_writer,
    const BCFamilyGenerationOptions &options = {}
);

// Diagnostic helper for one already-loaded source family pass. It does not
// perform source IO, dump/reload, boundary finalize, or position writes. The
// timer covers target-window prepare, source work construction, scan/unrank,
// spawn/move, canonicalize, encode, and cell-local insert for this pass only.
BCFamilyLoadedPassBenchmarkResult benchmark_loaded_family_plus_spawn_pass(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyTable &source_axis,
    FamilyId source_family_id,
    SpawnDeltaCoord delta_coord,
    uint8_t spawn_tile_rank,
    const std::vector<BCLoadedCell> &loaded_cells,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options = {}
);

} // namespace BC
