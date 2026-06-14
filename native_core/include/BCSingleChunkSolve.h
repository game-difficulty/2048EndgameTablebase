#pragma once

#include "BCLoadedCellScanner.h"
#include "BCResidentSolve.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace BC {

inline void bc_add_cell_load_stats(BCCellLoadStats &dst, const BCCellLoadStats &src) {
    dst.requested_extents += src.requested_extents;
    dst.coalesced_extents += src.coalesced_extents;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.read_bytes;
    dst.backend_read_ops += src.backend_read_ops;
    dst.backend_read_bytes += src.backend_read_bytes;
}

inline void bc_add_success_load_stats(BCSuccessLoadStats &dst, const BCSuccessLoadStats &src) {
    dst.requested_extents += src.requested_extents;
    dst.coalesced_extents += src.coalesced_extents;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.read_bytes;
    dst.backend_read_ops += src.backend_read_ops;
    dst.backend_read_bytes += src.backend_read_bytes;
}

struct BCSingleChunkSolveStats {
    uint64_t current_chunks = 0U;
    uint64_t current_cells = 0U;
    uint64_t current_nonempty_cells = 0U;
    uint64_t current_empty_cells = 0U;
    uint64_t current_boards = 0U;
    uint64_t output_values = 0U;
    uint64_t output_bytes = 0U;

    uint64_t future2_batch_loads = 0U;
    uint64_t future4_batch_loads = 0U;
    uint64_t future2_cells_loaded = 0U;
    uint64_t future4_cells_loaded = 0U;
    uint64_t future2_active_cells_max = 0U;
    uint64_t future4_active_cells_max = 0U;
    uint64_t future2_position_resident_bytes = 0U;
    uint64_t future4_position_resident_bytes = 0U;
    uint64_t future2_success_resident_bytes = 0U;
    uint64_t future4_success_resident_bytes = 0U;
    uint64_t future_resident_layers_max = 0U;
    uint64_t future_resident_bytes_max = 0U;
    uint64_t partial_spool_values = 0U;
    uint64_t partial_spool_bytes = 0U;

    BCCellLoadStats current_position_load;
    BCCellLoadStats future2_position_load;
    BCCellLoadStats future4_position_load;
    BCSuccessLoadStats future2_success_load;
    BCSuccessLoadStats future4_success_load;
    BCSolveEdgeStats edge;
    BCPartialStoreStats partial;
};

template <typename StorageT>
struct BCSingleChunkSolveOptions {
    BCResidentSolveOptions<StorageT> solve;
    uint32_t current_chunk_cells = 64U;
};

template <typename StorageT>
struct BCSingleChunkSolveResult {
    std::vector<uint8_t> success_bytes;
    BCSingleChunkSolveStats stats;
};

inline void bc_add_file_read_stats_to_cell_load(
    BCCellLoadStats &dst,
    const BCFileIOStats &src
) {
    dst.requested_extents += src.request_count;
    dst.coalesced_extents += src.backend_io_count;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.requested_bytes;
    dst.backend_read_ops += src.backend_io_count;
    dst.backend_read_bytes += src.backend_bytes;
}

inline void bc_add_file_read_stats_to_success_load(
    BCSuccessLoadStats &dst,
    const BCFileIOStats &src
) {
    dst.requested_extents += src.request_count;
    dst.coalesced_extents += src.backend_io_count;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.requested_bytes;
    dst.backend_read_ops += src.backend_io_count;
    dst.backend_read_bytes += src.backend_bytes;
}

struct BCSingleChunkResidentFutureLayer {
    std::vector<uint8_t> position_bytes;
    std::unique_ptr<BCPositionLayerReader> position;
    std::vector<uint8_t> success_bytes;
    std::unique_ptr<BCSuccessLayerReader> success;
};

template <typename StorageT>
class BCSingleChunkPartialSpool {
public:
    template <class PositionReader>
    void begin(const PositionReader &position, uint32_t row_width, StorageT initial_value) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC single chunk partial spool row_width must be non-zero");
        }
        row_width_ = row_width;
        offsets_.assign(static_cast<size_t>(position.cell_count()) + 1U, 0U);
        uint64_t cursor = 0U;
        for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
            offsets_[static_cast<size_t>(cid)] = cursor;
            const uint64_t cell_values =
                static_cast<uint64_t>(position.descriptor(cid).success_rows) *
                static_cast<uint64_t>(row_width_);
            cursor = bc_checked_add_u64(cursor, cell_values,
                "BC single chunk partial spool value count overflow");
        }
        offsets_.back() = cursor;
        if (cursor > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC single chunk partial spool exceeds size_t");
        }
        values_.assign(static_cast<size_t>(cursor), initial_value);
        written_.assign(position.cell_count(), false);
    }

    void write_empty_cell(CellId cid) {
        require_cell(cid);
        require_unwritten(cid);
        if (value_count_for_cell(cid) != 0U) {
            throw std::invalid_argument("BC single chunk partial spool empty cell has values");
        }
        written_[static_cast<size_t>(cid)] = true;
    }

    void write_cell(CellId cid, const std::vector<StorageT> &values) {
        require_cell(cid);
        require_unwritten(cid);
        const uint64_t expected = value_count_for_cell(cid);
        if (values.size() != expected) {
            throw std::invalid_argument("BC single chunk partial spool cell value count mismatch");
        }
        const uint64_t offset = offsets_[static_cast<size_t>(cid)];
        std::copy(
            values.begin(),
            values.end(),
            values_.begin() + static_cast<std::ptrdiff_t>(offset)
        );
        written_[static_cast<size_t>(cid)] = true;
    }

    [[nodiscard]] std::vector<StorageT> read_cell(CellId cid) const {
        require_cell(cid);
        if (!written_[static_cast<size_t>(cid)]) {
            throw std::logic_error("BC single chunk partial spool cell was not written");
        }
        const uint64_t offset = offsets_[static_cast<size_t>(cid)];
        const uint64_t count = value_count_for_cell(cid);
        return std::vector<StorageT>(
            values_.begin() + static_cast<std::ptrdiff_t>(offset),
            values_.begin() + static_cast<std::ptrdiff_t>(offset + count)
        );
    }

    [[nodiscard]] uint64_t value_count() const {
        return values_.size();
    }

    [[nodiscard]] uint64_t byte_size() const {
        if (values_.size() >
            static_cast<size_t>(std::numeric_limits<uint64_t>::max() / sizeof(StorageT))) {
            throw std::overflow_error("BC single chunk partial spool byte size overflow");
        }
        return static_cast<uint64_t>(values_.size()) * sizeof(StorageT);
    }

private:
    void require_cell(CellId cid) const {
        if (cid + 1U >= offsets_.size()) {
            throw std::out_of_range("BC single chunk partial spool cell id out of range");
        }
    }

    void require_unwritten(CellId cid) const {
        if (written_[static_cast<size_t>(cid)]) {
            throw std::logic_error("BC single chunk partial spool cell already written");
        }
    }

    [[nodiscard]] uint64_t value_count_for_cell(CellId cid) const {
        return offsets_[static_cast<size_t>(cid) + 1U] - offsets_[static_cast<size_t>(cid)];
    }

    uint32_t row_width_ = 0U;
    std::vector<uint64_t> offsets_;
    std::vector<StorageT> values_;
    std::vector<bool> written_;
};

inline BCSingleChunkResidentFutureLayer bc_single_chunk_materialize_future_layer(
    const BCPositionStreamingReader &position_stream,
    const BCSuccessStreamingReader &success_stream,
    uint32_t row_width,
    BCCellLoadStats &position_load_stats,
    BCSuccessLoadStats &success_load_stats,
    uint64_t &resident_loads,
    uint64_t &cells_loaded,
    uint64_t &active_cells_max,
    uint64_t &position_resident_bytes,
    uint64_t &success_resident_bytes
) {
    bc_solve_validate_success_matches_position(position_stream, success_stream);

    BCSingleChunkResidentFutureLayer layer;
    BCFileIOStats position_file_stats;
    layer.position_bytes = position_stream.read_all_bytes(&position_file_stats);
    bc_add_file_read_stats_to_cell_load(position_load_stats, position_file_stats);
    position_resident_bytes = layer.position_bytes.size();

    layer.position = std::make_unique<BCPositionLayerReader>();
    layer.position->open(layer.position_bytes, position_stream.lut());

    BCFileIOStats success_file_stats;
    layer.success_bytes = success_stream.read_all_bytes(&success_file_stats);
    bc_add_file_read_stats_to_success_load(success_load_stats, success_file_stats);
    success_resident_bytes = layer.success_bytes.size();

    layer.success = std::make_unique<BCSuccessLayerReader>();
    layer.success->open(layer.success_bytes, *layer.position, row_width);

    ++resident_loads;
    cells_loaded += position_stream.cell_count();
    active_cells_max = std::max<uint64_t>(active_cells_max, position_stream.cell_count());
    return layer;
}

template <typename StorageT>
void bc_single_chunk_solve_validate_streaming_future_options(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    if (options.current_chunk_cells == 0U) {
        throw std::invalid_argument("BC single chunk solve current_chunk_cells must be non-zero");
    }
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC single chunk solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC single chunk solve dtype does not match storage type");
    }
    if (future2_success.row_width() != options.solve.row_width ||
        future4_success.row_width() != options.solve.row_width) {
        throw std::invalid_argument("BC single chunk solve future success row_width mismatch");
    }
    if (!bc_success_dtype_matches_type<StorageT>(future2_success.dtype_mode()) ||
        !bc_success_dtype_matches_type<StorageT>(future4_success.dtype_mode())) {
        throw std::invalid_argument("BC single chunk solve future success dtype mismatch");
    }
    bc_solve_validate_success_matches_position(future2_position, future2_success);
    bc_solve_validate_success_matches_position(future4_position, future4_success);
    if (current_position.cell_count() == 0U || future2_position.cell_count() == 0U ||
        future4_position.cell_count() == 0U) {
        throw std::invalid_argument("BC single chunk solve positions must be open and non-empty");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.solve.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.solve.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC single chunk solve spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn4_tile_rank);
    if (future2_position.axis().layer_sum() != expected2 ||
        future4_position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC single chunk solve future layer_sum does not match spawn delta");
    }
    if (future2_position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC single chunk solve family_unit mismatch");
    }
}

template <typename StorageT>
void bc_single_chunk_solve_validate_options(
    const BCPositionStreamingReader &current_position,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    if (options.current_chunk_cells == 0U) {
        throw std::invalid_argument("BC single chunk solve current_chunk_cells must be non-zero");
    }
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC single chunk solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC single chunk solve dtype does not match storage type");
    }
    if (future2_success.row_width() != options.solve.row_width ||
        future4_success.row_width() != options.solve.row_width) {
        throw std::invalid_argument("BC single chunk solve future success row_width mismatch");
    }
    if (!bc_success_dtype_matches_type<StorageT>(future2_success.dtype_mode()) ||
        !bc_success_dtype_matches_type<StorageT>(future4_success.dtype_mode())) {
        throw std::invalid_argument("BC single chunk solve future success dtype mismatch");
    }
    bc_solve_validate_success_matches_position(future2_position, future2_success);
    bc_solve_validate_success_matches_position(future4_position, future4_success);
    if (current_position.cell_count() == 0U || future2_position.cell_count() == 0U ||
        future4_position.cell_count() == 0U) {
        throw std::invalid_argument("BC single chunk solve positions must be open and non-empty");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.solve.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.solve.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC single chunk solve spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn4_tile_rank);
    if (future2_position.axis().layer_sum() != expected2 ||
        future4_position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC single chunk solve future layer_sum does not match spawn delta");
    }
    if (future2_position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC single chunk solve family_unit mismatch");
    }
}

template <typename StorageT>
void bc_single_chunk_solve_phase4_partial(
    const BCPositionStreamingReader &current_position,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    BCSingleChunkPartialSpool<StorageT> &partial_spool,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveResult<StorageT> &result
) {
    BCPartialStore<StorageT> partial(options.solve.zero_value);
    BCSolveEdgeWorkspace<StorageT> workspace;
    const BCLut &lut = current_position.lut();
    const long double spawn_weight4 =
        static_cast<long double>(options.solve.edge_options.spawn_rate4);

    std::vector<CellId> chunk_cids;
    std::vector<BCLoadedCell> current_cells;
    chunk_cids.reserve(options.current_chunk_cells);
    for (CellId base = 0U; base < current_position.cell_count(); base += options.current_chunk_cells) {
        chunk_cids.clear();
        const CellId end = static_cast<CellId>(
            std::min<uint64_t>(
                static_cast<uint64_t>(current_position.cell_count()),
                static_cast<uint64_t>(base) + options.current_chunk_cells
            )
        );
        for (CellId cid = base; cid < end; ++cid) {
            chunk_cids.push_back(cid);
        }

        BCCellLoadStats current_load_stats;
        current_position.load_cells_into(chunk_cids, current_cells, &current_load_stats);
        bc_add_cell_load_stats(result.stats.current_position_load, current_load_stats);
        ++result.stats.current_chunks;

        for (const BCLoadedCell &cell : current_cells) {
            const CellId cid = cell.cid;
            ++result.stats.current_cells;
            const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
            if (desc.empty() || desc.success_rows == 0U) {
                ++result.stats.current_empty_cells;
                partial_spool.write_empty_cell(cid);
                continue;
            }
            if (cell.success_rows != desc.success_rows) {
                throw std::logic_error("BC single chunk current loaded cell row count mismatch");
            }

            ++result.stats.current_nonempty_cells;
            (void)partial.mutable_cell(
                cid,
                desc.success_rows,
                options.solve.row_width,
                options.solve.zero_value
            );

            BCLoadedCellScanner(lut, cell.view()).for_each_board(
                [&](const BCScannedBoardEntry &entry) {
                    ++result.stats.current_boards;
                    const BCSolveBoardQuerySummary summary =
                        bc_solve_collect_board_phase_queries<StorageT>(
                            lut,
                            future4_position.axis(),
                            entry.board,
                            options.solve.directions,
                            options.solve.filter4,
                            BCSolveSpawnPhase::Spawn4,
                            workspace,
                            options.solve.edge_options,
                            options.solve.word_sums,
                            &result.stats.edge
                        );
                    auto lookup = [&](const BCSolvePreparedQuery &query,
                                      uint32_t lane) -> BCSolveLookupResult<StorageT> {
                        const BCLookupResult row =
                            future4_position.cold_lookup(query.cid, query.key, query.rank);
                        if (!row.found) {
                            return {};
                        }
                        return BCSolveLookupResult<StorageT>{
                            true,
                            future4_success.template read_value_typed<StorageT>(
                                query.cid,
                                row.local_success_row,
                                lane
                            )
                        };
                    };
                    for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
                        const StorageT value =
                            bc_solve_reduce_spawn_phase_partial<StorageT>(
                                summary,
                                workspace.queries4,
                                workspace.best4,
                                lookup,
                                lane,
                                options.solve.zero_value,
                                spawn_weight4,
                                &result.stats.edge
                            );
                        partial.set(cid, entry.local_success_row, lane, value);
                    }
                }
            );

            std::vector<StorageT> values = partial.finalize_cell_values(cid);
            partial_spool.write_cell(cid, values);
        }
    }
    result.stats.partial_spool_values = partial_spool.value_count();
    result.stats.partial_spool_bytes = partial_spool.byte_size();
    result.stats.partial = partial.stats();
}

template <typename StorageT>
void bc_single_chunk_solve_phase2_finalize(
    const BCPositionStreamingReader &current_position,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCSingleChunkPartialSpool<StorageT> &partial_spool,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveResult<StorageT> &result
) {
    BCPartialStore<StorageT> partial(options.solve.zero_value);
    BCSolveEdgeWorkspace<StorageT> workspace;
    BCSuccessLayerWriter writer;
    writer.begin_layer(current_position, options.solve.row_width, options.solve.dtype);
    const BCLut &lut = current_position.lut();
    const long double spawn_weight2 =
        1.0L - static_cast<long double>(options.solve.edge_options.spawn_rate4);

    std::vector<CellId> chunk_cids;
    std::vector<BCLoadedCell> current_cells;
    chunk_cids.reserve(options.current_chunk_cells);
    for (CellId base = 0U; base < current_position.cell_count(); base += options.current_chunk_cells) {
        chunk_cids.clear();
        const CellId end = static_cast<CellId>(
            std::min<uint64_t>(
                static_cast<uint64_t>(current_position.cell_count()),
                static_cast<uint64_t>(base) + options.current_chunk_cells
            )
        );
        for (CellId cid = base; cid < end; ++cid) {
            chunk_cids.push_back(cid);
        }

        BCCellLoadStats current_load_stats;
        current_position.load_cells_into(chunk_cids, current_cells, &current_load_stats);
        bc_add_cell_load_stats(result.stats.current_position_load, current_load_stats);
        ++result.stats.current_chunks;

        for (const BCLoadedCell &cell : current_cells) {
            const CellId cid = cell.cid;
            ++result.stats.current_cells;
            const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
            if (desc.empty() || desc.success_rows == 0U) {
                ++result.stats.current_empty_cells;
                writer.mark_empty_cell(cid);
                continue;
            }
            if (cell.success_rows != desc.success_rows) {
                throw std::logic_error("BC single chunk current loaded cell row count mismatch");
            }

            ++result.stats.current_nonempty_cells;
            const std::vector<StorageT> partial4_values = partial_spool.read_cell(cid);
            (void)partial.mutable_cell(
                cid,
                desc.success_rows,
                options.solve.row_width,
                options.solve.zero_value
            );

            BCLoadedCellScanner(lut, cell.view()).for_each_board(
                [&](const BCScannedBoardEntry &entry) {
                    ++result.stats.current_boards;
                    const BCSolveBoardQuerySummary summary =
                        bc_solve_collect_board_phase_queries<StorageT>(
                            lut,
                            future2_position.axis(),
                            entry.board,
                            options.solve.directions,
                            options.solve.filter2,
                            BCSolveSpawnPhase::Spawn2,
                            workspace,
                            options.solve.edge_options,
                            options.solve.word_sums,
                            &result.stats.edge
                        );
                    auto lookup = [&](const BCSolvePreparedQuery &query,
                                      uint32_t lane) -> BCSolveLookupResult<StorageT> {
                        const BCLookupResult row =
                            future2_position.cold_lookup(query.cid, query.key, query.rank);
                        if (!row.found) {
                            return {};
                        }
                        return BCSolveLookupResult<StorageT>{
                            true,
                            future2_success.template read_value_typed<StorageT>(
                                query.cid,
                                row.local_success_row,
                                lane
                            )
                        };
                    };
                    for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
                        StorageT value = options.solve.zero_value;
                        if (summary.terminal_success) {
                            value = options.solve.terminal_value;
                        } else if (summary.empty_count != 0U) {
                            const StorageT phase2_value =
                                bc_solve_reduce_spawn_phase_partial<StorageT>(
                                    summary,
                                    workspace.queries2,
                                    workspace.best2,
                                    lookup,
                                    lane,
                                    options.solve.zero_value,
                                    spawn_weight2,
                                    &result.stats.edge
                                );
                            const uint64_t value_index =
                                static_cast<uint64_t>(entry.local_success_row) *
                                static_cast<uint64_t>(options.solve.row_width) + lane;
                            if (value_index >= partial4_values.size()) {
                                throw std::logic_error("BC single chunk partial value index out of range");
                            }
                            value = static_cast<StorageT>(
                                static_cast<long double>(partial4_values[static_cast<size_t>(value_index)]) +
                                static_cast<long double>(phase2_value)
                            );
                        }
                        partial.set(cid, entry.local_success_row, lane, value);
                    }
                    ++result.stats.edge.finalized_boards;
                }
            );

            result.stats.output_values +=
                static_cast<uint64_t>(desc.success_rows) * options.solve.row_width;
            std::vector<StorageT> values = partial.finalize_cell_values(cid);
            writer.write_cell_typed<StorageT>(cid, values);
        }
    }

    BCPartialStoreStats phase2_partial_stats = partial.stats();
    result.stats.partial.cells_created += phase2_partial_stats.cells_created;
    result.stats.partial.cells_reused += phase2_partial_stats.cells_reused;
    result.stats.partial.cells_released += phase2_partial_stats.cells_released;
    result.stats.partial.cells_finalized += phase2_partial_stats.cells_finalized;
    result.stats.partial.values_initialized += phase2_partial_stats.values_initialized;
    result.stats.partial.values_set += phase2_partial_stats.values_set;
    result.stats.partial.update_attempts += phase2_partial_stats.update_attempts;
    result.stats.partial.values_updated += phase2_partial_stats.values_updated;
    result.stats.partial.active_cells = phase2_partial_stats.active_cells;
    result.stats.partial.active_cells_max =
        std::max(result.stats.partial.active_cells_max, phase2_partial_stats.active_cells_max);
    result.stats.partial.active_bytes = phase2_partial_stats.active_bytes;
    result.stats.partial.active_bytes_max =
        std::max(result.stats.partial.active_bytes_max, phase2_partial_stats.active_bytes_max);

    result.success_bytes = writer.finish_layer();
    result.stats.output_bytes = result.success_bytes.size();
}

template <typename StorageT>
BCSingleChunkSolveResult<StorageT> bc_single_chunk_solve_success_layer(
    const BCPositionStreamingReader &current_position,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    bc_single_chunk_solve_validate_options(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );

    BCSingleChunkSolveResult<StorageT> result;
    BCPartialStore<StorageT> partial(options.solve.zero_value);
    BCSolveEdgeWorkspace<StorageT> workspace;
    BCSuccessLayerWriter writer;
    writer.begin_layer(current_position, options.solve.row_width, options.solve.dtype);

    const BCLut &lut = current_position.lut();
    std::vector<CellId> chunk_cids;
    std::vector<BCLoadedCell> current_cells;
    chunk_cids.reserve(options.current_chunk_cells);
    for (CellId base = 0U; base < current_position.cell_count(); base += options.current_chunk_cells) {
        chunk_cids.clear();
        const CellId end = static_cast<CellId>(
            std::min<uint64_t>(
                static_cast<uint64_t>(current_position.cell_count()),
                static_cast<uint64_t>(base) + options.current_chunk_cells
            )
        );
        for (CellId cid = base; cid < end; ++cid) {
            chunk_cids.push_back(cid);
        }

        BCCellLoadStats current_load_stats;
        current_position.load_cells_into(chunk_cids, current_cells, &current_load_stats);
        bc_add_cell_load_stats(result.stats.current_position_load, current_load_stats);
        ++result.stats.current_chunks;

        for (const BCLoadedCell &cell : current_cells) {
            const CellId cid = cell.cid;
            ++result.stats.current_cells;
            const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
            if (desc.empty() || desc.success_rows == 0U) {
                ++result.stats.current_empty_cells;
                writer.mark_empty_cell(cid);
                continue;
            }
            if (cell.success_rows != desc.success_rows) {
                throw std::logic_error("BC single chunk current loaded cell row count mismatch");
            }

            ++result.stats.current_nonempty_cells;
            (void)partial.mutable_cell(
                cid,
                desc.success_rows,
                options.solve.row_width,
                options.solve.zero_value
            );

            BCLoadedCellScanner(lut, cell.view()).for_each_board(
                [&](const BCScannedBoardEntry &entry) {
                    ++result.stats.current_boards;
                    auto prepare_lookup = [](const BCSolveBoardQuerySummary &,
                                             BCSolveEdgeWorkspace<StorageT> &) {};
                    auto lookup = [&](const BCSolvePreparedQuery &query,
                                      uint32_t lane) -> BCSolveLookupResult<StorageT> {
                        const bool use_spawn2 =
                            query.spawn_tile_rank == options.solve.edge_options.spawn2_tile_rank;
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
                        options.solve,
                        prepare_lookup,
                        lookup,
                        partial,
                        workspace,
                        result.stats.edge
                    );
                }
            );

            result.stats.output_values +=
                static_cast<uint64_t>(desc.success_rows) * options.solve.row_width;
            std::vector<StorageT> values = partial.finalize_cell_values(cid);
            writer.write_cell_typed<StorageT>(cid, values);
        }
    }

    result.stats.partial = partial.stats();
    result.success_bytes = writer.finish_layer();
    result.stats.output_bytes = result.success_bytes.size();
    return result;
}

template <typename StorageT>
BCSingleChunkSolveResult<StorageT> bc_single_chunk_solve_success_layer(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    bc_single_chunk_solve_validate_streaming_future_options(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );

    BCSingleChunkSolveResult<StorageT> result;
    BCSingleChunkPartialSpool<StorageT> partial_spool;
    partial_spool.begin(current_position, options.solve.row_width, options.solve.zero_value);

    {
        BCSingleChunkResidentFutureLayer future4_layer =
            bc_single_chunk_materialize_future_layer(
                future4_position,
                future4_success,
                options.solve.row_width,
                result.stats.future4_position_load,
                result.stats.future4_success_load,
                result.stats.future4_batch_loads,
                result.stats.future4_cells_loaded,
                result.stats.future4_active_cells_max,
                result.stats.future4_position_resident_bytes,
                result.stats.future4_success_resident_bytes
            );
        result.stats.future_resident_layers_max =
            std::max<uint64_t>(result.stats.future_resident_layers_max, 1U);
        result.stats.future_resident_bytes_max = std::max<uint64_t>(
            result.stats.future_resident_bytes_max,
            result.stats.future4_position_resident_bytes +
                result.stats.future4_success_resident_bytes
        );
        bc_single_chunk_solve_phase4_partial<StorageT>(
            current_position,
            *future4_layer.position,
            *future4_layer.success,
            partial_spool,
            options,
            result
        );
    }

    {
        BCSingleChunkResidentFutureLayer future2_layer =
            bc_single_chunk_materialize_future_layer(
                future2_position,
                future2_success,
                options.solve.row_width,
                result.stats.future2_position_load,
                result.stats.future2_success_load,
                result.stats.future2_batch_loads,
                result.stats.future2_cells_loaded,
                result.stats.future2_active_cells_max,
                result.stats.future2_position_resident_bytes,
                result.stats.future2_success_resident_bytes
            );
        result.stats.future_resident_layers_max =
            std::max<uint64_t>(result.stats.future_resident_layers_max, 1U);
        result.stats.future_resident_bytes_max = std::max<uint64_t>(
            result.stats.future_resident_bytes_max,
            result.stats.future2_position_resident_bytes +
                result.stats.future2_success_resident_bytes
        );
        bc_single_chunk_solve_phase2_finalize<StorageT>(
            current_position,
            *future2_layer.position,
            *future2_layer.success,
            partial_spool,
            options,
            result
        );
    }
    return result;
}

} // namespace BC
