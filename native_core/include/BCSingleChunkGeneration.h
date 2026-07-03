#pragma once

#include "BCResidentGeneration.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace BC {

class BCLut;
class BCFamilyTable;
class BCPositionStreamingReader;
class BCWritableFile;

struct BCSingleChunkGenerationSource {
    const BCPositionStreamingReader *position = nullptr;
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;

    // 0 means load all source cells in one chunk.
    uint32_t cell_chunk_size = 16U;
};

struct BCSingleChunkGenerationStepResult {
    BCResidentGenerationResult primary;
    BCResidentGenerationResult carry;
    std::unique_ptr<BCResidentGenerationMutableLayer> next_carry;
    bool has_carry = false;
    uint64_t current_boards_scanned = 0U;
    double total_step_compute_seconds = 0.0;
    double total_step_seconds = 0.0;
};

// SingleChunk generation streams the current/source position layer by cell
// chunk, while keeping the target mutable layer as a whole-layer BCDynamicState.
// This backend is intended for ResidentChain/SingleChunkChain generation only.
// FamilyChain must use a separate cell-local mutable store with dump/reload and
// wavefront finalization.
[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCSingleChunkGenerationSource &source,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCSingleChunkGenerationSource &source,
    const BCResidentGenerationOptions &options = {}
);

// Strict SingleChunk step with 1+x layer residency:
//   1. consume current chunks with +2 into carry_to_primary, finalize/write primary;
//   2. release primary mutable;
//   3. rescan current chunks with +4 into next_carry, without finalizing or writing it.
// At no point does this path keep both primary and secondary whole-layer mutable
// states resident.
[[nodiscard]] BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] inline BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCSingleChunkGenerationSource &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
) {
    if (current.position == nullptr) {
        throw std::invalid_argument("BC SingleChunk strict generation source position is null");
    }
    return generate_single_chunk_position_layer_strict_to_file(
        lut,
        primary_axis,
        *current.position,
        current.cell_chunk_size,
        std::move(carry_to_primary),
        secondary_axis,
        primary_output_file,
        options
    );
}

[[nodiscard]] BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
);

[[nodiscard]] inline BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCSingleChunkGenerationSource &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options = {}
) {
    if (current.position == nullptr) {
        throw std::invalid_argument("BC SingleChunk strict generation source position is null");
    }
    return generate_single_chunk_position_layer_strict_to_file(
        lut,
        primary_layout,
        *current.position,
        current.cell_chunk_size,
        std::move(carry_to_primary),
        secondary_layout,
        primary_output_file,
        options
    );
}

} // namespace BC
