#include "BCResidentGeneration.h"

#include "BCResidentGenerationInternal.h"
#include "BCBoardOps.h"
#include "BCCellMatrix.h"
#include "BCLoadedCellScanner.h"
#include "BCPositionCellLoader.h"
#include "BoardMover.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {
namespace ResidentGenerationInternal {
namespace {

void validate_resident_generation_source(
    const BCFamilyTable &target_axis,
    const BCResidentStreamingGenerationSource &source
) {
    if (source.position == nullptr) {
        throw std::invalid_argument("BC resident streaming generation source position is null");
    }
    if (source.spawn_tile_rank == 0U || source.spawn_tile_rank > 15U) {
        throw std::invalid_argument("BC resident streaming generation spawn_tile_rank must be in 1..15");
    }

    const BCFamilyTable &source_axis = source.position->axis();
    if (source_axis.family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC resident streaming generation source/target family_unit mismatch");
    }
    const uint32_t expected_target_total =
        static_cast<uint32_t>(source_axis.total_coord()) +
        static_cast<uint32_t>(source.delta_coord);
    if (static_cast<uint32_t>(target_axis.total_coord()) != expected_target_total) {
        throw std::invalid_argument(
            "BC resident streaming generation target total_coord must equal source total_coord + delta_coord"
        );
    }
}

[[nodiscard]] uint32_t source_bucket_count_sum(
    const std::vector<BCResidentStreamingGenerationSource> &sources
) {
    uint64_t count = 0U;
    for (const BCResidentStreamingGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        for (CellId cid = 0U; cid < source.position->cell_count(); ++cid) {
            count += source.position->descriptor(cid).bucket_count;
        }
    }
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident streaming generation source bucket count exceeds uint32");
    }
    return static_cast<uint32_t>(count);
}

[[nodiscard]] uint64_t source_success_row_count_sum(
    const std::vector<BCResidentStreamingGenerationSource> &sources
) {
    uint64_t count = 0U;
    for (const BCResidentStreamingGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        for (CellId cid = 0U; cid < source.position->cell_count(); ++cid) {
            count = bc_checked_add_u64(
                count,
                source.position->descriptor(cid).success_rows,
                "BC resident streaming generation source row count overflow"
            );
        }
    }
    return count;
}

[[nodiscard]] uint64_t source_rank_payload_word_estimate(
    const std::vector<BCResidentStreamingGenerationSource> &sources
) {
    uint64_t bytes = 0U;
    for (const BCResidentStreamingGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        for (CellId cid = 0U; cid < source.position->cell_count(); ++cid) {
            bytes += source.position->descriptor(cid).rank_payload_bytes;
        }
    }
    return bytes / sizeof(uint64_t) + 1U;
}

void add_cell_load_stats(BCCellLoadStats &dst, const BCCellLoadStats &src) {
    dst.requested_extents = bc_checked_add_u64(
        dst.requested_extents,
        src.requested_extents,
        "BC resident streaming load requested extent count overflow"
    );
    dst.coalesced_extents = bc_checked_add_u64(
        dst.coalesced_extents,
        src.coalesced_extents,
        "BC resident streaming load coalesced extent count overflow"
    );
    dst.requested_bytes = bc_checked_add_u64(
        dst.requested_bytes,
        src.requested_bytes,
        "BC resident streaming load requested byte count overflow"
    );
    dst.read_bytes = bc_checked_add_u64(
        dst.read_bytes,
        src.read_bytes,
        "BC resident streaming load read byte count overflow"
    );
    dst.backend_read_ops = bc_checked_add_u64(
        dst.backend_read_ops,
        src.backend_read_ops,
        "BC resident streaming load backend op count overflow"
    );
    dst.backend_read_bytes = bc_checked_add_u64(
        dst.backend_read_bytes,
        src.backend_read_bytes,
        "BC resident streaming load backend byte count overflow"
    );
}

void copy_cell_load_stats_to_result(
    BCResidentGenerationResult &result,
    const BCCellLoadStats &stats
) {
    result.source_position_load_requested_extents = stats.requested_extents;
    result.source_position_load_coalesced_extents = stats.coalesced_extents;
    result.source_position_load_requested_bytes = stats.requested_bytes;
    result.source_position_load_read_bytes = stats.read_bytes;
    result.source_position_load_backend_read_ops = stats.backend_read_ops;
    result.source_position_load_backend_read_bytes = stats.backend_read_bytes;
}

[[nodiscard]] bool bc_is_success_by_shifts(
    uint64_t board,
    int target_rank,
    const std::vector<uint8_t> &success_shifts
) {
    if (success_shifts.empty()) {
        return false;
    }
    const uint64_t target = static_cast<uint64_t>(target_rank);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] uint32_t countr_zero32(uint32_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC countr_zero32 requires non-zero value");
    }
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctz(value));
#else
    uint32_t count = 0U;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++count;
    }
    return count;
#endif
}

[[nodiscard]] uint32_t bc_dynamic_home_slot(CellId cid, uint64_t key, uint32_t capacity) {
    const uint64_t value = key ^ (static_cast<uint64_t>(cid) * 0x9e3779b97f4a7c15ULL);
    const uint64_t mixed = value * 11400714819323198485ULL;
    const uint32_t bits =
#if defined(__GNUC__) || defined(__clang__)
        static_cast<uint32_t>(__builtin_ctz(capacity));
#else
        [] (uint32_t v) {
            uint32_t out = 0U;
            while ((v >>= 1U) != 0U) {
                ++out;
            }
            return out;
        }(capacity);
#endif
    return static_cast<uint32_t>(mixed >> (64U - bits));
}

uint32_t bc_dynamic_acquire_words(
    BCDynamicState &state,
    BCDynamicThreadChunks &chunks,
    uint32_t words
) {
    constexpr uint32_t kChunkWords = 512U;
    if (chunks.word_next + words <= chunks.word_end) {
        const uint32_t out = chunks.word_next;
        chunks.word_next += words;
        return out;
    }
    const uint32_t chunk_words = std::max<uint32_t>(kChunkWords, words);
    const uint32_t begin = state.bitmap_cursor_words.fetch_add(chunk_words, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(begin) + chunk_words > state.reserved_bitmap_words) {
        state.overflowed.store(true, std::memory_order_release);
        return BCDynamicState::kPendingCell;
    }
    chunks.word_next = begin + words;
    chunks.word_end = begin + chunk_words;
    return begin;
}

void bc_dynamic_clear_words(BCDynamicState &state, uint32_t offset, uint32_t words) {
    for (uint32_t i = 0U; i < words; ++i) {
        state.bitmap_arena[offset + i].store(0ULL, std::memory_order_relaxed);
    }
}

uint32_t bc_dynamic_find_or_insert(
    BCDynamicState &state,
    const BCPendingEncodedCandidate &entry,
    BCDynamicThreadChunks &chunks
) {
    uint32_t slot = entry.home_slot;
    uint32_t probes = 0U;
    for (;;) {
        const uint32_t current = state.cell_array[slot].load(std::memory_order_acquire);
        if (current == entry.cid) {
            if (state.key_array[slot] == entry.key) {
                return state.bitmap_offset_array[slot];
            }
        } else if (current == BCDynamicState::kEmptyCell) {
            uint32_t expected = BCDynamicState::kEmptyCell;
            if (state.cell_array[slot].compare_exchange_strong(
                    expected,
                    BCDynamicState::kPendingCell,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                const uint32_t word_count = words_for_bits(entry.bitmap_len);
                const uint32_t offset = bc_dynamic_acquire_words(state, chunks, word_count);
                if (offset != BCDynamicState::kPendingCell) {
                    bc_dynamic_clear_words(state, offset, word_count);
                    state.key_array[slot] = entry.key;
                    state.bitmap_offset_array[slot] = offset;
                    state.cell_array[slot].store(entry.cid, std::memory_order_release);
                }
                return offset;
            }
            continue;
        } else if (current == BCDynamicState::kPendingCell) {
            if (state.overflowed.load(std::memory_order_acquire)) {
                return BCDynamicState::kPendingCell;
            }
            continue;
        }

        slot = (slot + 1U) & (state.hash_capacity - 1U);
        ++probes;
        if (probes >= state.hash_capacity) {
            state.overflowed.store(true, std::memory_order_release);
            return BCDynamicState::kPendingCell;
        }
        __builtin_prefetch(&state.cell_array[slot], 0, 1);
    }
}

void flush_pending_encoded(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &state
) {
    if (workspace.pending_encoded.empty() || state.overflowed.load(std::memory_order_acquire)) {
        return;
    }

    constexpr uint32_t kPrefetchDistance = 16U;
    const uint32_t count = static_cast<uint32_t>(workspace.pending_encoded.size());
    workspace.resolved_encoded.clear();
    workspace.resolved_encoded.reserve(count);
    const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
    for (uint32_t i = 0U; i < prefetch_count; ++i) {
        __builtin_prefetch(&state.cell_array[workspace.pending_encoded[i].home_slot], 0, 1);
    }
    for (uint32_t i = 0U; i < count; ++i) {
        if (i + kPrefetchDistance < count) {
            __builtin_prefetch(
                &state.cell_array[workspace.pending_encoded[i + kPrefetchDistance].home_slot],
                0,
                1
            );
        }
        const BCPendingEncodedCandidate &candidate = workspace.pending_encoded[i];
        const uint32_t bitmap_offset = bc_dynamic_find_or_insert(
            state,
            candidate,
            workspace.dynamic_chunks
        );
        if (bitmap_offset == BCDynamicState::kPendingCell) {
            return;
        }
        workspace.resolved_encoded.push_back(BCDynamicResolved{
            bitmap_offset,
            candidate.rank
        });
        __builtin_prefetch(
            &state.bitmap_arena[bitmap_offset + (static_cast<uint32_t>(candidate.rank) >> 6U)],
            1,
            1
        );
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const BCDynamicResolved &resolved = workspace.resolved_encoded[i];
        const uint32_t word = static_cast<uint32_t>(resolved.rank) >> 6U;
        const uint64_t mask = 1ULL << (static_cast<uint32_t>(resolved.rank) & 63U);
        std::atomic<uint64_t> &target = state.bitmap_arena[resolved.bitmap_offset + word];
        if ((target.load(std::memory_order_relaxed) & mask) == 0ULL) {
            target.fetch_or(mask, std::memory_order_relaxed);
        }
    }

    workspace.pending_encoded.clear();
}

void flush_pending_encoded_timed(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &state,
    const BCResidentGenerationOptions &options
) {
    if (workspace.pending_encoded.empty()) {
        return;
    }
    const double begin = options.collect_timing ? bc_now_seconds() : 0.0;
    flush_pending_encoded(workspace, state);
    if (options.collect_timing) {
        workspace.stats.encode_insert_seconds += bc_now_seconds() - begin;
    }
}

[[nodiscard]] BCBoardEncodedPosition encode_canonical_quadrants_position_hot(
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const BCQuadrantWords &q,
    const BCWordSumTable *word_sums
) {
    BCBoardEncodedPosition out;
    const BCFamilyTable &axis = layout.serialization_axis();

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.nw]
        : nw_desc.sum;
    const uint64_t ne_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.ne]
        : ne_desc.sum;
    const uint64_t sw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.sw]
        : sw_desc.sum;
    const uint64_t se_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.se]
        : se_desc.sum;

    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!bc_min_side_coord_u64(nw_sum + ne_sum, sw_sum + se_sum, axis.family_unit(), row_coord) ||
        !bc_min_side_coord_u64(nw_sum + sw_sum, ne_sum + se_sum, axis.family_unit(), col_coord)) {
        return out;
    }
    const FamilyId row_id = layout.try_raw_side_coord_to_physical_index(row_coord);
    const FamilyId col_id = layout.try_raw_side_coord_to_physical_index(col_coord);
    if (row_id == BCPositionCellLayout::kInvalidSideIndex ||
        col_id == BCPositionCellLayout::kInvalidSideIndex) {
        return out;
    }

    const uint32_t family_count = layout.side_coord_count();
    const uint64_t cid64 =
        static_cast<uint64_t>(row_id) * static_cast<uint64_t>(family_count) + col_id;
    if (cid64 > std::numeric_limits<CellId>::max()) {
        throw std::overflow_error("BC hot encoded cell id exceeds CellId");
    }

    const BCEncodedKeyRank encoded =
        bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return out;
    }
    out.cid = static_cast<CellId>(cid64);
    out.row_family = row_id;
    out.col_family = col_id;
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

void flush_canonical_buffer(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    BCDynamicState &dynamic_state,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums
) {
    if (workspace.canonical_buffer.empty()) {
        return;
    }

    const double canonical_begin = options.collect_timing ? bc_now_seconds() : 0.0;
    CanonicalBatch::canonicalize_inplace(
        workspace.canonical_buffer.data(),
        workspace.canonical_buffer.size(),
        options.canonical_symm_mode
    );
    if (options.collect_timing) {
        workspace.stats.canonical_seconds += bc_now_seconds() - canonical_begin;
    }
    const double encode_begin = options.collect_timing ? bc_now_seconds() : 0.0;
    for (uint64_t canonical_board : workspace.canonical_buffer) {
        if (options.keep_only_success_generated_boards) {
            if (options.success_shifts == nullptr ||
                !bc_is_success_by_shifts(
                    canonical_board,
                    options.success_target_rank,
                    *options.success_shifts
                )) {
                continue;
            }
        }
        const BCQuadrantWords q = unpack_board_to_quadrants(canonical_board);
        const BCBoardEncodedPosition encoded =
            encode_canonical_quadrants_position_hot(
                lut,
                target_layout,
                q,
                word_sums
            );
        if (!encoded.valid) {
            throw std::logic_error("BC resident generation produced candidate outside target axis");
        }
        BCEncodedKeyRank key_rank;
        key_rank.key = encoded.key;
        key_rank.rank = encoded.rank;
        key_rank.bitmap_len = encoded.bitmap_len;
        key_rank.count_ne = encoded.count_ne;
        key_rank.count_sw = encoded.count_sw;
        key_rank.count_se = encoded.count_se;
        key_rank.valid = true;

        workspace.pending_encoded.push_back(BCPendingEncodedCandidate{
            encoded.cid,
            bc_dynamic_home_slot(encoded.cid, key_rank.key, dynamic_state.hash_capacity),
            key_rank.key,
            key_rank.rank,
            key_rank.bitmap_len
        });
        if (workspace.pending_encoded.size() >= options.pending_insert_buffer_size) {
            flush_pending_encoded(workspace, dynamic_state);
        }
    }
    if (options.collect_timing) {
        workspace.stats.encode_insert_seconds += bc_now_seconds() - encode_begin;
    }
    workspace.canonical_buffer.clear();
}

void push_moved_board(
    BCThreadGenerationWorkspace &workspace,
    uint64_t spawned,
    uint64_t moved,
    const BCResidentGenerationOptions &options
) {
    if (moved == spawned) {
        return;
    }
    workspace.canonical_buffer.push_back(moved);
}

void process_source_board(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source,
    BCDynamicState &dynamic_state,
    uint64_t board,
    uint16_t source_empty_mask,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    if (skip_success_source &&
        options.success_shifts != nullptr &&
        bc_is_success_by_shifts(board, options.success_target_rank, *options.success_shifts)) {
        return;
    }
    uint32_t empty_mask = source_empty_mask;
    while (empty_mask != 0U) {
        const uint32_t cell = countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;
        const uint64_t spawned =
            board | (static_cast<uint64_t>(source.spawn_tile_rank) << (4U * cell));
        const auto moved = BoardMover::move_all_dir(spawned);
        push_moved_board(workspace, spawned, std::get<0>(moved), options);
        push_moved_board(workspace, spawned, std::get<1>(moved), options);
        push_moved_board(workspace, spawned, std::get<2>(moved), options);
        push_moved_board(workspace, spawned, std::get<3>(moved), options);
    }
    if (workspace.canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(workspace, lut, target_layout, dynamic_state, options, word_sums);
    }
}

void run_streaming_source_phase(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentStreamingGenerationSource &source,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces,
    const BCWordSumTable *word_sums,
    BCCellLoadStats &load_stats,
    double &load_seconds
) {
    validate_resident_generation_source(target_layout.serialization_axis(), source);
    const bool skip_success_source =
        bc_success_check_enabled(options, source.position->axis().layer_sum());
    const uint32_t cell_count = source.position->cell_count();
    if (cell_count == 0U) {
        return;
    }
    const uint32_t chunk_size = source.cell_chunk_size == 0U
        ? cell_count
        : source.cell_chunk_size;
    if (chunk_size == 0U) {
        throw std::invalid_argument("BC resident streaming generation chunk size resolved to zero");
    }

    const BCResidentGenerationSource board_source{
        nullptr,
        source.spawn_tile_rank,
        source.delta_coord
    };

    std::vector<CellId> cids;
    cids.reserve(chunk_size);
    for (CellId chunk_begin = 0U; chunk_begin < cell_count;) {
        const uint64_t chunk_end_u64 = std::min<uint64_t>(
            cell_count,
            static_cast<uint64_t>(chunk_begin) + static_cast<uint64_t>(chunk_size)
        );
        const CellId chunk_end = static_cast<CellId>(chunk_end_u64);
        cids.clear();
        for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
            cids.push_back(cid);
        }

        BCCellLoadStats chunk_stats;
        const double load_begin = options.collect_timing ? bc_now_seconds() : 0.0;
        std::vector<BCLoadedCell> loaded_cells = source.position->load_cells(cids, &chunk_stats);
        if (options.collect_timing) {
            load_seconds += bc_now_seconds() - load_begin;
        }
        add_cell_load_stats(load_stats, chunk_stats);
        if (loaded_cells.empty()) {
            continue;
        }

        std::exception_ptr first_exception;
#pragma omp parallel num_threads(thread_count)
        {
            try {
                const int tid = bc_omp_thread_num();
                BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
#pragma omp for schedule(dynamic, 16)
                for (int64_t cell_i = 0; cell_i < static_cast<int64_t>(loaded_cells.size()); ++cell_i) {
                    const double cell_begin = options.collect_timing ? bc_now_seconds() : 0.0;
                    const double canonical_before = workspace.stats.canonical_seconds;
                    const double encode_before = workspace.stats.encode_insert_seconds;
                    const BCLoadedCell &cell = loaded_cells[static_cast<size_t>(cell_i)];
                    BCLoadedCellScanner(lut, cell.view()).for_each_board(
                        [&](const BCScannedBoardEntry &entry) {
                            process_source_board(
                        workspace,
                        lut,
                        target_layout,
                                board_source,
                                dynamic_state,
                                entry.board,
                                entry.empty_mask,
                                options,
                                word_sums,
                                skip_success_source
                            );
                        }
                    );
                    if (options.collect_timing) {
                        const double cell_elapsed = bc_now_seconds() - cell_begin;
                        const double nested_elapsed =
                            (workspace.stats.canonical_seconds - canonical_before) +
                            (workspace.stats.encode_insert_seconds - encode_before);
                        workspace.stats.spawn_move_seconds +=
                            std::max(0.0, cell_elapsed - nested_elapsed);
                    }
                }
                flush_canonical_buffer(workspace, lut, target_layout, dynamic_state, options, word_sums);
                flush_pending_encoded_timed(workspace, dynamic_state, options);
            } catch (...) {
#pragma omp critical(BCResidentGenerationException)
                {
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
        chunk_begin = chunk_end;
    }
}

} // namespace
} // namespace ResidentGenerationInternal

using namespace ResidentGenerationInternal;

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        options
    );
}

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    if (sources.empty()) {
        throw std::invalid_argument("BC resident streaming generation requires at least one source");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident streaming generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident streaming generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident streaming generation dynamic_reserve_factor must be positive");
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    const BCCellMatrix target_matrix(target_axis);
    const uint32_t target_cell_count = target_matrix.cell_count();
    const BCWordSumTable word_sums_storage = build_word_sum_table_if_needed(
        lut,
        options.tile_sum_values != nullptr ? options.tile_sum_values : options.family_tile_sum_values
    );
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;
    uint64_t canonical_warmup = 0U;
    CanonicalBatch::canonicalize_inplace(&canonical_warmup, 1U, options.canonical_symm_mode);

    BCResidentGenerationResult result;
    result.effective_threads = thread_count;
    const double generation_begin = bc_now_seconds();
    std::vector<BCThreadGenerationWorkspace> workspaces;
    BCDynamicState dynamic_state;
    BCCellLoadStats successful_load_stats;
    double successful_load_seconds = 0.0;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double scan_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        result = BCResidentGenerationResult{};
        result.effective_threads = thread_count;
        const double prepare_begin = bc_now_seconds();
        workspaces.clear();
        workspaces.resize(static_cast<size_t>(thread_count));
        for (BCThreadGenerationWorkspace &workspace : workspaces) {
            workspace.canonical_buffer.reserve(options.canonical_batch_size);
            workspace.pending_encoded.reserve(options.pending_insert_buffer_size);
            workspace.resolved_encoded.reserve(options.pending_insert_buffer_size);
        }

        const uint64_t source_buckets = source_bucket_count_sum(sources);
        const uint64_t source_bitmap_words = source_rank_payload_word_estimate(sources);
        dynamic_state = make_bc_dynamic_state(
            target_cell_count,
            static_cast<uint64_t>(static_cast<double>(source_buckets) * reserve_factor) + 4096ULL,
            static_cast<uint64_t>(static_cast<double>(source_bitmap_words) * reserve_factor) + 512ULL * 64ULL,
            thread_count
        );
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        BCCellLoadStats attempt_load_stats;
        double attempt_load_seconds = 0.0;
        for (const BCResidentStreamingGenerationSource &source : sources) {
            const double phase_begin = bc_now_seconds();
            run_streaming_source_phase(
                lut,
                target_layout,
                source,
                options,
                thread_count,
                dynamic_state,
                workspaces,
                word_sums,
                attempt_load_stats,
                attempt_load_seconds
            );
            scan_seconds += bc_now_seconds() - phase_begin;
        }
        work_seconds += bc_now_seconds() - work_begin;
        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            successful_retry = retry;
            successful_load_stats = attempt_load_stats;
            successful_load_seconds = attempt_load_seconds;
            generated = true;
            break;
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC resident streaming generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    result.generation_seconds = generation_seconds;
    result.prepare_seconds = prepare_seconds;
    result.work_seconds = work_seconds;
    result.cleanup_seconds = 0.0;
    result.scan_seconds = scan_seconds;
    result.source_position_load_seconds = successful_load_seconds;
    copy_cell_load_stats_to_result(result, successful_load_stats);
    add_workspace_stats(result, workspaces);
    result.source_boards_scanned = source_success_row_count_sum(sources);
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(result, lut, dynamic_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(result, dynamic_state, successful_retry);
    }
    finalize_dynamic_result(
        result,
        lut,
        target_axis,
        dynamic_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        nullptr
    );
    return result;
}

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source_to_file(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        output_file,
        options
    );
}

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentStreamingGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options
) {
    if (sources.empty()) {
        throw std::invalid_argument("BC resident streaming generation requires at least one source");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident streaming generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident streaming generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident streaming generation dynamic_reserve_factor must be positive");
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    const BCCellMatrix target_matrix(target_axis);
    const uint32_t target_cell_count = target_matrix.cell_count();
    const BCWordSumTable word_sums_storage = build_word_sum_table_if_needed(
        lut,
        options.tile_sum_values != nullptr ? options.tile_sum_values : options.family_tile_sum_values
    );
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;
    uint64_t canonical_warmup = 0U;
    CanonicalBatch::canonicalize_inplace(&canonical_warmup, 1U, options.canonical_symm_mode);

    BCResidentGenerationResult result;
    result.effective_threads = thread_count;
    const double generation_begin = bc_now_seconds();
    std::vector<BCThreadGenerationWorkspace> workspaces;
    BCDynamicState dynamic_state;
    BCCellLoadStats successful_load_stats;
    double successful_load_seconds = 0.0;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double scan_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        result = BCResidentGenerationResult{};
        result.effective_threads = thread_count;
        const double prepare_begin = bc_now_seconds();
        workspaces.clear();
        workspaces.resize(static_cast<size_t>(thread_count));
        for (BCThreadGenerationWorkspace &workspace : workspaces) {
            workspace.canonical_buffer.reserve(options.canonical_batch_size);
            workspace.pending_encoded.reserve(options.pending_insert_buffer_size);
            workspace.resolved_encoded.reserve(options.pending_insert_buffer_size);
        }

        const uint64_t source_buckets = source_bucket_count_sum(sources);
        const uint64_t source_bitmap_words = source_rank_payload_word_estimate(sources);
        dynamic_state = make_bc_dynamic_state(
            target_cell_count,
            static_cast<uint64_t>(static_cast<double>(source_buckets) * reserve_factor) + 4096ULL,
            static_cast<uint64_t>(static_cast<double>(source_bitmap_words) * reserve_factor) + 512ULL * 64ULL,
            thread_count
        );
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        BCCellLoadStats attempt_load_stats;
        double attempt_load_seconds = 0.0;
        for (const BCResidentStreamingGenerationSource &source : sources) {
            const double phase_begin = bc_now_seconds();
            run_streaming_source_phase(
                lut,
                target_layout,
                source,
                options,
                thread_count,
                dynamic_state,
                workspaces,
                word_sums,
                attempt_load_stats,
                attempt_load_seconds
            );
            scan_seconds += bc_now_seconds() - phase_begin;
        }
        work_seconds += bc_now_seconds() - work_begin;
        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            successful_retry = retry;
            successful_load_stats = attempt_load_stats;
            successful_load_seconds = attempt_load_seconds;
            generated = true;
            break;
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC resident streaming generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    result.generation_seconds = generation_seconds;
    result.prepare_seconds = prepare_seconds;
    result.work_seconds = work_seconds;
    result.cleanup_seconds = 0.0;
    result.scan_seconds = scan_seconds;
    result.source_position_load_seconds = successful_load_seconds;
    copy_cell_load_stats_to_result(result, successful_load_stats);
    add_workspace_stats(result, workspaces);
    result.source_boards_scanned = source_success_row_count_sum(sources);
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(result, lut, dynamic_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(result, dynamic_state, successful_retry);
    }
    finalize_dynamic_result(
        result,
        lut,
        target_axis,
        dynamic_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        &output_file
    );
    return result;
}

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentStreamingGenerationSource &source,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source(
        lut,
        target_axis,
        std::vector<BCResidentStreamingGenerationSource>{source},
        options
    );
}

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentStreamingGenerationSource &source,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_from_streaming_source(
        lut,
        target_layout,
        std::vector<BCResidentStreamingGenerationSource>{source},
        options
    );
}

} // namespace BC
