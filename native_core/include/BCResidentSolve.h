#pragma once

#include "BCPartialStore.h"
#include "BCPositionScanner.h"
#include "BCSolveEdgeKernel.h"
#include "BCSuccessIO.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace BC {

struct BCResidentSolveStats {
    uint64_t current_cells = 0U;
    uint64_t current_nonempty_cells = 0U;
    uint64_t current_empty_cells = 0U;
    uint64_t current_boards = 0U;
    uint64_t output_values = 0U;
    uint64_t output_bytes = 0U;
    BCSolveEdgeStats edge;
    BCPartialStoreStats partial;
};

template <typename StorageT>
struct BCResidentSolveOptions {
    uint32_t row_width = 1U;
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>();
    StorageT zero_value{};
    StorageT terminal_value{};
    BCDirectionMask directions = BCDirectionMask::Both;
    BCSolveTargetFamilyFilter filter2;
    BCSolveTargetFamilyFilter filter4;
    BCSolveEdgeOptions edge_options;
    const BCQuadrantWordSumTable *word_sums = nullptr;
};

template <typename StorageT>
struct BCResidentSolveResult {
    std::vector<uint8_t> success_bytes;
    BCResidentSolveStats stats;
};

template <class PositionReader, class SuccessReader>
void bc_solve_validate_success_matches_position(
    const PositionReader &position,
    const SuccessReader &success
) {
    const BCPositionHeader &p = position.header();
    const BCSuccessHeader &s = success.header();
    if (s.family_count != p.family_count ||
        s.descriptor_count != position.cell_count() ||
        s.position_key_mode != p.key_mode ||
        s.family_unit != p.family_unit ||
        s.axis_base_coord != p.axis_base_coord ||
        s.layer_sum != p.layer_sum ||
        s.position_metadata_fingerprint != bc_success_position_fingerprint_for(position)) {
        throw std::invalid_argument("BC solve future success does not match future position metadata");
    }
    const uint64_t expected_payload_bytes =
        bc_success_expected_payload_bytes_for(position, success.row_width(), success.dtype_mode());
    if (s.payload_bytes != expected_payload_bytes) {
        throw std::invalid_argument("BC solve future success payload does not match future position rows");
    }
}

template <typename StorageT, typename PrepareLookupFn, typename LookupFn>
void bc_solve_scanned_board_into_partial(
    const BCLut &lut,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    CellId current_cid,
    const BCScannedBoardEntry &entry,
    const BCResidentSolveOptions<StorageT> &options,
    PrepareLookupFn &&prepare_lookup,
    LookupFn &&lookup,
    BCPartialStore<StorageT> &partial,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    BCSolveEdgeStats &edge_stats
) {
    const BCSolveBoardQuerySummary summary =
        bc_solve_collect_board_queries<StorageT>(
            lut,
            future2_axis,
            future4_axis,
            entry.board,
            options.directions,
            options.filter2,
            options.filter4,
            workspace,
            options.edge_options,
            options.word_sums,
            &edge_stats
        );

    prepare_lookup(summary, workspace);
    for (uint32_t lane = 0U; lane < options.row_width; ++lane) {
        const StorageT value =
            bc_solve_reduce_collected_queries<StorageT>(
                summary,
                workspace,
                lookup,
                lane,
                options.zero_value,
                options.terminal_value,
                options.edge_options,
                &edge_stats,
                lane == 0U
            );
        partial.set(current_cid, entry.local_success_row, lane, value);
    }
}

template <typename StorageT>
void bc_resident_solve_validate_options(
    const BCPositionLayerReader &current_position,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCResidentSolveOptions<StorageT> &options
) {
    static_assert(
        std::is_same_v<StorageT, uint32_t> || std::is_same_v<StorageT, uint64_t> ||
        std::is_same_v<StorageT, float> || std::is_same_v<StorageT, double>,
        "unsupported BC resident solve value type"
    );
    if (options.row_width == 0U) {
        throw std::invalid_argument("BC resident solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.dtype)) {
        throw std::invalid_argument("BC resident solve dtype does not match storage type");
    }
    if (future2_success.row_width() != options.row_width ||
        future4_success.row_width() != options.row_width) {
        throw std::invalid_argument("BC resident solve future success row_width mismatch");
    }
    if (!bc_success_dtype_matches_type<StorageT>(future2_success.dtype_mode()) ||
        !bc_success_dtype_matches_type<StorageT>(future4_success.dtype_mode())) {
        throw std::invalid_argument("BC resident solve future success dtype mismatch");
    }
    bc_solve_validate_success_matches_position(future2_position, future2_success);
    bc_solve_validate_success_matches_position(future4_position, future4_success);
    if (future2_position.cell_count() == 0U || future4_position.cell_count() == 0U ||
        current_position.cell_count() == 0U) {
        throw std::invalid_argument("BC resident solve positions must be open and non-empty");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC resident solve spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.edge_options.spawn4_tile_rank);
    if (future2_position.axis().layer_sum() != expected2 ||
        future4_position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC resident solve future layer_sum does not match spawn delta");
    }
    if (future2_position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC resident solve family_unit mismatch");
    }
}

template <typename StorageT>
BCResidentSolveResult<StorageT> bc_resident_solve_success_layer(
    const BCPositionLayerReader &current_position,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCResidentSolveOptions<StorageT> &options
) {
    bc_resident_solve_validate_options(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );
    BCResidentSolveResult<StorageT> result;
    BCPartialStore<StorageT> partial(options.zero_value);
    BCSolveEdgeWorkspace<StorageT> workspace;
    BCSuccessLayerWriter writer;
    writer.begin_layer(current_position, options.row_width, options.dtype);

    const BCLut &lut = current_position.lut();
    for (CellId cid = 0U; cid < current_position.cell_count(); ++cid) {
        ++result.stats.current_cells;
        const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            ++result.stats.current_empty_cells;
            writer.mark_empty_cell(cid);
            continue;
        }

        ++result.stats.current_nonempty_cells;
        (void)partial.mutable_cell(cid, desc.success_rows, options.row_width, options.zero_value);
        BCPositionCellScanner(current_position, cid).for_each_board(
            [&](const BCScannedBoardEntry &entry) {
                ++result.stats.current_boards;
                auto prepare_lookup = [](const BCSolveBoardQuerySummary &,
                                         BCSolveEdgeWorkspace<StorageT> &) {};
                auto lookup = [&](const BCSolvePreparedQuery &query,
                                  uint32_t lane) -> BCSolveLookupResult<StorageT> {
                    const bool use_spawn2 =
                        query.spawn_tile_rank == options.edge_options.spawn2_tile_rank;
                    const BCPositionLayerReader &future_position =
                        use_spawn2 ? future2_position : future4_position;
                    const BCSuccessLayerReader &future_success =
                        use_spawn2 ? future2_success : future4_success;
                    const BCLookupResult row =
                        future_position.cold_lookup(query.cid, query.key, query.rank);
                    if (!row.found) {
                        return {};
                    }
                    return BCSolveLookupResult<StorageT>{
                        true,
                        future_success.template read_value_typed<StorageT>(
                            query.cid,
                            row.local_success_row,
                            lane
                        )
                    };
                };
                bc_solve_scanned_board_into_partial<StorageT>(
                    lut,
                    future2_position.axis(),
                    future4_position.axis(),
                    cid,
                    entry,
                    options,
                    prepare_lookup,
                    lookup,
                    partial,
                    workspace,
                    result.stats.edge
                );
            }
        );

        result.stats.output_values +=
            static_cast<uint64_t>(desc.success_rows) * options.row_width;
        std::vector<StorageT> values = partial.finalize_cell_values(cid);
        writer.write_cell_typed<StorageT>(cid, values);
    }

    result.stats.partial = partial.stats();
    result.success_bytes = writer.finish_layer();
    result.stats.output_bytes = result.success_bytes.size();
    return result;
}

} // namespace BC
