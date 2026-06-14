#pragma once

#include "BCCellBuilder.h"
#include "BCFamilyTable.h"
#include "BCPositionCellLayout.h"
#include "FormationRuntime.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace BC {

class BCLut;
class BCPositionLayerReader;
class BCPositionStreamingReader;
class BCWritableFile;

class BCResidentGenerationMutableLayer {
public:
    BCResidentGenerationMutableLayer();
    ~BCResidentGenerationMutableLayer();
    BCResidentGenerationMutableLayer(BCResidentGenerationMutableLayer &&) noexcept;
    BCResidentGenerationMutableLayer &operator=(BCResidentGenerationMutableLayer &&) noexcept;

    BCResidentGenerationMutableLayer(const BCResidentGenerationMutableLayer &) = delete;
    BCResidentGenerationMutableLayer &operator=(const BCResidentGenerationMutableLayer &) = delete;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const BCFamilyTable &axis() const;
    [[nodiscard]] BCPositionCellLayout layout() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend struct BCResidentGenerationMutableLayerAccess;
};

struct BCResidentGenerationSource {
    const BCPositionLayerReader *position = nullptr;
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;
};

struct BCResidentStreamingGenerationSource {
    const BCPositionStreamingReader *position = nullptr;
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;

    // 0 means load all cells in one batch. Non-zero values are useful for
    // SingleChunk-style source scanning and for IO coalescing benchmarks.
    uint32_t cell_chunk_size = 16U;
};

struct BCResidentGenerationOptions {
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t pending_insert_buffer_size = 1024U;
    double dynamic_reserve_factor = 1.0;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    bool collect_timing = false;
    BCCellFinalizeOptions finalize_options = {};
    const std::array<uint32_t, 16U> *tile_sum_values = nullptr;
    const std::array<uint32_t, 16U> *family_tile_sum_values = nullptr;
    int success_target_rank = 0;
    const std::vector<uint8_t> *success_shifts = nullptr;
    LayerSum success_check_min_source_layer_sum = 0U;
    bool keep_only_success_generated_boards = false;
    bool keep_only_success_secondary_generated_boards = false;
    bool collect_mutable_output_stats = true;
    bool collect_dynamic_state_stats = true;
};

struct BCResidentGenerationResult {
    std::vector<uint8_t> position_bytes;

    uint64_t source_boards_scanned = 0U;
    uint64_t output_success_rows = 0U;

    int effective_threads = 0;
    uint32_t generation_retries = 0U;
    uint32_t dynamic_hash_capacity = 0U;
    uint64_t dynamic_bucket_slots_used = 0U;
    uint64_t dynamic_bitmap_words_used = 0U;
    uint64_t dynamic_bitmap_words_allocated = 0U;
    uint64_t dynamic_bitmap_words_reserved = 0U;
    uint64_t source_position_load_requested_extents = 0U;
    uint64_t source_position_load_coalesced_extents = 0U;
    uint64_t source_position_load_requested_bytes = 0U;
    uint64_t source_position_load_read_bytes = 0U;
    uint64_t source_position_load_backend_read_ops = 0U;
    uint64_t source_position_load_backend_read_bytes = 0U;
    uint64_t target_position_file_logical_bytes = 0U;
    uint64_t target_position_write_requests = 0U;
    uint64_t target_position_write_requested_bytes = 0U;
    uint64_t target_position_write_backend_ops = 0U;
    uint64_t target_position_write_backend_bytes = 0U;

    // Wall-clock source phase time. Kept separate from scan_seconds because the
    // phase includes scan, spawn/move, canonicalize, encode, and thread-local insert.
    double generation_seconds = 0.0;
    double scan_seconds = 0.0;
    double source_position_load_seconds = 0.0;

    // Thread-accumulated hot-path seconds. These are useful for estimating CPU
    // saturation and identifying hot-path work split; they are not wall seconds.
    // Measured at source-cell granularity, so this includes scanner callback,
    // empty-cell enumeration, spawn, move_all_dir, and batch-buffer push work.
    double thread_spawn_move_seconds = 0.0;
    double thread_canonical_seconds = 0.0;
    double thread_encode_insert_seconds = 0.0;
    double spawn_move_seconds = 0.0;
    double canonical_seconds = 0.0;
    double encode_insert_seconds = 0.0;

    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double merge_seconds = 0.0;
    double finalize_seconds = 0.0;
    double cleanup_seconds = 0.0;
    double write_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;
    // FamilyChain/Resident retry attempts are included in compute_seconds.
    // This field makes failed-attempt wall time explicit instead of hiding it
    // in compute_seconds - generation_seconds.
    double retry_seconds = 0.0;

    // FamilyChain coarse stage timings. These are collected only at pass/stage
    // boundaries; hot candidate loops must not take wall-clock timestamps.
    double build_work_seconds = 0.0;
    double parallel_seconds = 0.0;
    double dump_seconds = 0.0;
    double reload_seconds = 0.0;
    uint64_t family_blob_read_bytes = 0U;
    uint64_t family_blob_backend_read_ops = 0U;
    uint64_t family_blob_backend_read_bytes = 0U;
    uint64_t family_blob_bytes_written = 0U;
    uint64_t family_blob_backend_write_ops = 0U;
    uint64_t family_blob_backend_write_bytes = 0U;
    uint64_t family_blob_staging_bytes = 0U;
    uint64_t family_writer_staging_bytes = 0U;
    double family_blob_backend_read_seconds = 0.0;
    double family_blob_restore_seconds = 0.0;
    double family_blob_backend_write_seconds = 0.0;
    double family_blob_flush_seconds = 0.0;
    uint64_t family_writer_bucket_stage_write_bytes = 0U;
    uint64_t family_writer_rank_stage_write_bytes = 0U;
    uint64_t family_writer_rank_copy_read_bytes = 0U;
    uint64_t family_writer_rank_copy_write_bytes = 0U;
    uint64_t family_active_family_window_peak = 0U;
    uint64_t family_target_active_cell_peak = 0U;
    uint64_t family_source_loaded_cell_peak = 0U;
    uint64_t family_source_loaded_bytes_peak = 0U;
    uint64_t family_source_load_range_bytes_peak = 0U;
    uint64_t family_active_builder_bytes_peak = 0U;
    uint64_t family_thread_workspace_bytes_peak = 0U;
    uint64_t family_pass_cache_bytes_peak = 0U;
    uint64_t family_active_index_bytes_peak = 0U;
    uint64_t family_range_work_bytes_peak = 0U;
    uint64_t family_dump_buffer_bytes_peak = 0U;
    uint64_t family_reload_dump_bytes_peak = 0U;
    uint64_t family_restore_builder_bytes_peak = 0U;
    uint64_t family_finalize_dump_bytes_peak = 0U;
    uint64_t family_finalize_payload_bytes_peak = 0U;
    uint64_t family_store_static_metadata_bytes = 0U;
    uint64_t family_released_builder_bytes_total = 0U;
    uint64_t family_released_builder_bytes_peak = 0U;
    uint64_t family_release_batch_builder_bytes_peak = 0U;
    uint64_t family_store_allocated_bytes_peak = 0U;
    uint64_t process_working_set_bytes = 0U;
    uint64_t process_peak_working_set_bytes = 0U;
    uint64_t process_baseline_working_set_bytes = 0U;
    uint64_t process_private_bytes = 0U;
    uint64_t process_pagefile_bytes = 0U;
    uint64_t process_peak_pagefile_bytes = 0U;
    uint64_t family_memory_checkpoint_count = 0U;
    uint64_t family_memory_checkpoint_process_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_process_peak_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_process_private_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_process_pagefile_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_process_peak_pagefile_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_accounted_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_residual_bytes_peak = 0U;
    uint64_t family_memory_checkpoint_baseline_adjusted_residual_peak = 0U;
    uint32_t family_memory_checkpoint_residual_peak_label = 0U;
    uint32_t family_memory_checkpoint_baseline_adjusted_residual_peak_label = 0U;
    uint64_t family_move_all_dir_calls = 0U;
    uint64_t family_selective_move_calls = 0U;
    uint64_t family_target_window_skips = 0U;
    uint64_t family_buffer_flushes = 0U;
    uint64_t family_buffer_flush_items = 0U;
    uint64_t family_buffer_flush_max_items = 0U;
    uint64_t family_builder_bind_calls = 0U;
    uint64_t family_insert_hash_lookups = 0U;
    uint64_t family_insert_hash_probe_steps = 0U;
    uint64_t family_target_cells_created = 0U;
    uint64_t family_target_cells_reloaded = 0U;
    uint64_t family_target_cells_dumped = 0U;
    uint64_t family_target_cells_finalized = 0U;
    uint64_t family_target_cells_kept = 0U;
    uint64_t family_builder_hash_grows = 0U;
    uint64_t family_builder_bitmap_grows = 0U;
    uint64_t family_builder_hash_replaced_bytes = 0U;
    uint64_t family_builder_bitmap_replaced_bytes = 0U;
    uint64_t family_source_work_items = 0U;
    uint64_t family_source_bucket_ranges = 0U;
    uint64_t family_source_bitmap_words_scanned = 0U;
    uint64_t family_parallel_region_count = 0U;

    [[nodiscard]] double thread_hot_seconds() const {
        return thread_spawn_move_seconds + thread_canonical_seconds + thread_encode_insert_seconds;
    }

    [[nodiscard]] double avg_hot_threads() const {
        return generation_seconds > 0.0 ? thread_hot_seconds() / generation_seconds : 0.0;
    }

    // Match EX generate stats: primary/output live if present, otherwise input/source live.
    [[nodiscard]] uint64_t ex_generate_throughput_live() const {
        return output_success_rows != 0U ? output_success_rows : source_boards_scanned;
    }

    [[nodiscard]] double throughput_mbps() const {
        return total_seconds > 0.0
            ? static_cast<double>(ex_generate_throughput_live()) / total_seconds / 1.0e6
            : 0.0;
    }

    [[nodiscard]] double compute_throughput_mbps() const {
        return compute_seconds > 0.0
            ? static_cast<double>(ex_generate_throughput_live()) / compute_seconds / 1.0e6
            : 0.0;
    }

    [[nodiscard]] double source_board_mbps() const {
        return compute_seconds > 0.0
            ? static_cast<double>(source_boards_scanned) / compute_seconds / 1.0e6
            : 0.0;
    }

    [[nodiscard]] double output_board_mbps() const {
        return compute_seconds > 0.0
            ? static_cast<double>(output_success_rows) / compute_seconds / 1.0e6
            : 0.0;
    }
};

struct BCResidentGenerationPairResult {
    BCResidentGenerationResult primary;
    BCResidentGenerationResult secondary;
    std::unique_ptr<BCResidentGenerationMutableLayer> secondary_carry;
    bool has_secondary = false;

    // Pair generation scans the current source layer once and may produce both
    // primary(+2) and secondary(+4) layers. The generation wall time is shared;
    // do not add primary.generation_seconds and secondary.generation_seconds.
    uint64_t current_boards_scanned = 0U;
    double shared_generation_seconds = 0.0;
    double total_pair_compute_seconds = 0.0;
};

struct BCResidentMutableGenerationResult {
    BCResidentGenerationResult result;
    std::unique_ptr<BCResidentGenerationMutableLayer> mutable_layer;
};

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentStreamingGenerationSource &source,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentStreamingGenerationSource &source,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source4,
    const BCResidentGenerationSource &source2,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source4,
    const BCResidentGenerationSource &source2,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCFamilyTable *secondary_axis,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    BCWritableFile *secondary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    BCWritableFile *secondary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    BCWritableFile *terminal_secondary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    BCWritableFile *terminal_secondary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationPairResult
generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentMutableGenerationResult generate_resident_mutable_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentStreamingGenerationSource &source,
    std::unique_ptr<BCResidentGenerationMutableLayer> initial_mutable,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentMutableGenerationResult generate_resident_mutable_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentStreamingGenerationSource &source,
    std::unique_ptr<BCResidentGenerationMutableLayer> initial_mutable,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult finalize_resident_mutable_layer_to_file(
    const BCLut &lut,
    std::unique_ptr<BCResidentGenerationMutableLayer> mutable_layer,
    BCWritableFile &output_file,
    BCResidentGenerationResult generation_result,
    const BCResidentGenerationOptions &options = {}
);

} // namespace BC
