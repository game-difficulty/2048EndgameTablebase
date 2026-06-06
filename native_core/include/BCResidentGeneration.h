#pragma once

#include "BCCellBuilder.h"
#include "BCFamilyTable.h"
#include "BCPositionFile.h"
#include "FormationRuntime.h"

#include <array>
#include <cstdint>
#include <vector>

namespace BC {

struct BCResidentGenerationSource {
    const BCPositionLayerReader *position = nullptr;
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;
};

struct BCResidentGenerationOptions {
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t pending_insert_buffer_size = 128U;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    bool collect_timing = true;
    BCCellFinalizeOptions finalize_options = {};
    const std::array<uint32_t, 16U> *family_tile_sum_values = nullptr;
    int success_target_rank = 0;
    const std::vector<uint8_t> *success_shifts = nullptr;
    LayerSum success_check_min_source_layer_sum = 0U;
};

struct BCResidentGenerationResult {
    std::vector<uint8_t> position_bytes;

    uint64_t source_boards_scanned = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_candidates = 0U;
    uint64_t moved_candidates = 0U;
    uint64_t encoded_candidates = 0U;
    uint64_t output_success_rows = 0U;
    uint64_t duplicate_candidates = 0U;

    uint64_t valid_candidates = 0U;
    uint64_t duplicate_candidates_possible = 0U;

    int effective_threads = 0;
    uint32_t generation_retries = 0U;
    uint32_t dynamic_hash_capacity = 0U;
    uint64_t dynamic_bucket_slots_used = 0U;
    uint64_t dynamic_bitmap_words_used = 0U;
    uint64_t dynamic_bitmap_words_allocated = 0U;
    uint64_t dynamic_bitmap_words_reserved = 0U;

    // Wall-clock source phase time. Kept separate from scan_seconds because the
    // phase includes scan, spawn/move, canonicalize, encode, and thread-local insert.
    double generation_seconds = 0.0;
    double scan_seconds = 0.0;

    // Thread-accumulated hot-path seconds. These are useful for estimating CPU
    // saturation and identifying hot-path work split; they are not wall seconds.
    // Measured at source-cell granularity, so this includes scanner callback,
    // empty-cell enumeration, spawn, move_all_dir, and batch-buffer push work.
    double thread_spawn_move_seconds = 0.0;
    double thread_canonical_seconds = 0.0;
    double thread_encode_insert_seconds = 0.0;

    // Legacy aliases kept for existing bench output.
    double spawn_move_seconds = 0.0;
    double canonical_seconds = 0.0;
    double encode_insert_seconds = 0.0;
    double merge_seconds = 0.0;
    double finalize_seconds = 0.0;
    double write_seconds = 0.0;
    double compute_seconds = 0.0;
    double total_seconds = 0.0;

    [[nodiscard]] double moved_candidate_mbps() const {
        return compute_seconds > 0.0
            ? static_cast<double>(moved_candidates) / compute_seconds / 1.0e6
            : 0.0;
    }

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
    bool has_secondary = false;

    // Pair generation scans the current source layer once and may produce both
    // primary(+2) and secondary(+4) layers. The generation wall time is shared;
    // do not add primary.generation_seconds and secondary.generation_seconds.
    uint64_t current_boards_scanned = 0U;
    double shared_generation_seconds = 0.0;
    double total_pair_compute_seconds = 0.0;
};

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
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

} // namespace BC
