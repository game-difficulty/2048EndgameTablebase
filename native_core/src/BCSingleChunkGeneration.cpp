#include "BCSingleChunkGeneration.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace BC {
namespace {

[[nodiscard]] std::vector<BCResidentStreamingGenerationSource> to_resident_streaming_sources(
    const std::vector<BCSingleChunkGenerationSource> &sources
) {
    if (sources.empty()) {
        throw std::invalid_argument("BC SingleChunk generation requires at least one source");
    }
    std::vector<BCResidentStreamingGenerationSource> out;
    out.reserve(sources.size());
    for (const BCSingleChunkGenerationSource &source : sources) {
        out.push_back(BCResidentStreamingGenerationSource{
            source.position,
            source.spawn_tile_rank,
            source.delta_coord,
            source.cell_chunk_size
        });
    }
    return out;
}

} // namespace

BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    return generate_single_chunk_position_layer(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        options
    );
}

BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source(
        lut,
        target_layout,
        to_resident_streaming_sources(sources),
        options
    );
}

BCResidentGenerationResult generate_single_chunk_position_layer_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options
) {
    return generate_single_chunk_position_layer_to_file(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        output_file,
        options
    );
}

BCResidentGenerationResult generate_single_chunk_position_layer_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCSingleChunkGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source_to_file(
        lut,
        target_layout,
        to_resident_streaming_sources(sources),
        output_file,
        options
    );
}

BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCSingleChunkGenerationSource &source,
    const BCResidentGenerationOptions &options
) {
    return generate_single_chunk_position_layer(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        std::vector<BCSingleChunkGenerationSource>{source},
        options
    );
}

BCResidentGenerationResult generate_single_chunk_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCSingleChunkGenerationSource &source,
    const BCResidentGenerationOptions &options
) {
    return generate_single_chunk_position_layer(
        lut,
        target_layout,
        std::vector<BCSingleChunkGenerationSource>{source},
        options
    );
}

BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_single_chunk_position_layer_strict_to_file(
        lut,
        primary_layout,
        current,
        current_cell_chunk_size,
        std::move(carry_to_primary),
        secondary_layout ? &*secondary_layout : nullptr,
        primary_output_file,
        options
    );
}

BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options
) {
    BCSingleChunkGenerationStepResult out;
    const BCFamilyTable &primary_axis = primary_layout.serialization_axis();

    BCResidentStreamingGenerationSource source2{
        &current,
        1U,
        1U,
        current_cell_chunk_size
    };
    BCResidentGenerationOptions primary_options = options;
    // The strict path finalizes/writes the +2 mutable state immediately below.
    // Let finalize produce the exact row count instead of scanning the bitmap
    // once in the mutable generation result and again during finalize.
    primary_options.collect_mutable_output_stats = false;
    primary_options.collect_dynamic_state_stats = false;
    BCResidentMutableGenerationResult primary_mutable =
        generate_resident_mutable_layer_from_streaming_source(
            lut,
            primary_layout,
            source2,
            std::move(carry_to_primary),
            primary_options
        );
    out.current_boards_scanned = primary_mutable.result.source_boards_scanned;
    out.primary = finalize_resident_mutable_layer_to_file(
        lut,
        std::move(primary_mutable.mutable_layer),
        primary_output_file,
        std::move(primary_mutable.result),
        options
    );

    if (secondary_layout != nullptr) {
        BCResidentGenerationOptions carry_options = options;
        carry_options.keep_only_success_generated_boards =
            options.keep_only_success_secondary_generated_boards;
        carry_options.dynamic_reserve_factor = options.dynamic_secondary_reserve_factor > 0.0
            ? options.dynamic_secondary_reserve_factor
            : options.dynamic_reserve_factor * 2.0;
        carry_options.collect_mutable_output_stats = false;
        carry_options.collect_dynamic_state_stats = false;
        BCResidentStreamingGenerationSource source4{
            &current,
            2U,
            2U,
            current_cell_chunk_size
        };
        BCResidentMutableGenerationResult carry_mutable =
            generate_resident_mutable_layer_from_streaming_source(
                lut,
                *secondary_layout,
                source4,
                nullptr,
                carry_options
            );
        out.carry = std::move(carry_mutable.result);
        out.next_carry = std::move(carry_mutable.mutable_layer);
        out.has_carry = true;
    }

    out.total_step_compute_seconds =
        out.primary.compute_seconds +
        (out.has_carry ? out.carry.compute_seconds : 0.0);
    out.total_step_seconds =
        out.primary.total_seconds +
        (out.has_carry ? out.carry.total_seconds : 0.0);
    return out;
}

} // namespace BC
