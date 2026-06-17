#pragma once

#include "BCFamilySolvePlan.h"
#include "BCFutureFamilyWindow.h"
#include "BCSingleChunkSolve.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {

struct BCFamilySolveStats {
    BCSingleChunkSolveStats single;
    uint64_t spawn4_passes = 0U;
    uint64_t spawn2_passes = 0U;
    uint64_t family_current_cells_loaded = 0U;
    uint64_t partial4_cells_written = 0U;
    uint64_t partial4_cells_read = 0U;
    uint64_t partial2_cells_written = 0U;
    uint64_t partial2_cells_read = 0U;
    uint64_t scratch4_cells_written = 0U;
    uint64_t scratch4_cells_read = 0U;
    uint64_t finalized_cells = 0U;
    uint64_t pending_cells = 0U;
    uint64_t pending_cells_max = 0U;
    uint64_t temp_values_written = 0U;
    uint64_t temp_values_read = 0U;
    uint64_t temp_bytes_written = 0U;
    uint64_t temp_bytes_read = 0U;
    uint64_t final_stage_cells_written = 0U;
    uint64_t final_stage_cells_read = 0U;
    uint64_t final_stage_values_written = 0U;
    uint64_t final_stage_values_read = 0U;
    uint64_t final_stage_bytes_written = 0U;
    uint64_t final_stage_bytes_read = 0U;
    uint64_t partial4_profiled_values = 0U;
    uint64_t partial4_profiled_zero_values = 0U;
    uint64_t partial2_profiled_values = 0U;
    uint64_t partial2_profiled_zero_values = 0U;
    uint64_t block_current_resident_bytes_max = 0U;
    uint64_t block_scratch4_resident_bytes_max = 0U;
    uint64_t block_partial_prefetch_resident_bytes_max = 0U;
    uint64_t block_dense_resident_bytes_max = 0U;
    uint64_t block_total_resident_bytes_max = 0U;
    BCFileIOStats temp_write_io;
    BCFileIOStats temp_read_io;
    BCFileIOStats final_stage_write_io;
    BCFileIOStats final_stage_read_io;
    uint64_t spawn4_future_reuse_groups = 0U;
    uint64_t spawn2_future_reuse_groups = 0U;
    double temp_write_seconds = 0.0;
    double temp_read_prepare_seconds = 0.0;
    double temp_read_seconds = 0.0;
    double final_stage_write_seconds = 0.0;
    double final_stage_read_seconds = 0.0;
    double output_pending_seconds = 0.0;
    double plan_seconds = 0.0;
    double workspace_prepare_seconds = 0.0;
    double mark_empty_seconds = 0.0;
    double current_layout_seconds = 0.0;
    double future4_prepare_overhead_seconds = 0.0;
    double future2_prepare_overhead_seconds = 0.0;
    double future4_prepare_normalize_seconds = 0.0;
    double future2_prepare_normalize_seconds = 0.0;
    double future4_prepare_select_seconds = 0.0;
    double future2_prepare_select_seconds = 0.0;
    double future4_prepare_index_build_seconds = 0.0;
    double future2_prepare_index_build_seconds = 0.0;
    double future4_prepare_insert_sort_seconds = 0.0;
    double future2_prepare_insert_sort_seconds = 0.0;
    double future4_lookup_copy_seconds = 0.0;
    double future2_lookup_copy_seconds = 0.0;
    double spawn4_phase_wall_seconds = 0.0;
    double spawn2_phase_wall_seconds = 0.0;
    double spawn4_phase_untracked_seconds = 0.0;
    double spawn2_phase_untracked_seconds = 0.0;
    double spawn4_cell_compute_seconds = 0.0;
    double spawn2_cell_compute_seconds = 0.0;
    double spawn4_bucket_hit_seconds = 0.0;
    double spawn2_bucket_hit_seconds = 0.0;
    double spawn4_batch_candidate_thread_seconds = 0.0;
    double spawn4_batch_canonical_thread_seconds = 0.0;
    double spawn4_batch_setup_thread_seconds = 0.0;
    double spawn4_batch_reduce_thread_seconds = 0.0;
    double spawn4_batch_emit_thread_seconds = 0.0;
    double spawn2_batch_candidate_thread_seconds = 0.0;
    double spawn2_batch_canonical_thread_seconds = 0.0;
    double spawn2_batch_setup_thread_seconds = 0.0;
    double spawn2_batch_reduce_thread_seconds = 0.0;
    double spawn2_batch_emit_thread_seconds = 0.0;
    uint64_t spawn4_batch_canonical_candidates = 0U;
    uint64_t spawn4_batch_encoded_queries = 0U;
    uint64_t spawn4_batch_reduce_found = 0U;
    uint64_t spawn4_batch_entry_misses = 0U;
    uint64_t spawn4_batch_bitmap_misses = 0U;
    uint64_t spawn2_batch_canonical_candidates = 0U;
    uint64_t spawn2_batch_encoded_queries = 0U;
    uint64_t spawn2_batch_reduce_found = 0U;
    uint64_t spawn2_batch_entry_misses = 0U;
    uint64_t spawn2_batch_bitmap_misses = 0U;
    double final_dense_copy_seconds = 0.0;
    double compact_value_copy_seconds = 0.0;
    double pending_mark_seconds = 0.0;
    double output_finish_seconds = 0.0;
    double output_streamer_open_seconds = 0.0;
    double temp_open_seconds = 0.0;
    double temp_close_seconds = 0.0;
    uint64_t future_release_all_calls = 0U;
    uint64_t future_release_except_calls = 0U;
    double future_release_all_seconds = 0.0;
    double future_release_except_seconds = 0.0;
    double future_release_all_clear_seconds = 0.0;
    double future_release_except_normalize_seconds = 0.0;
    double future_release_except_filter_seconds = 0.0;
    double future_release_except_erase_seconds = 0.0;
    double future_release_except_ids_seconds = 0.0;
    double workspace_release_spawn4_partial_seconds = 0.0;
    double workspace_release_spawn4_scratch_seconds = 0.0;
    double workspace_release_spawn4_temp_values_seconds = 0.0;
    double workspace_release_spawn2_dense_seconds = 0.0;
    double workspace_release_spawn2_prefetch_seconds = 0.0;
    double workspace_release_spawn2_temp_values_seconds = 0.0;
    double workspace_release_block_state_seconds = 0.0;
};

template <typename StorageT>
struct BCFamilySolveOptions {
    BCResidentSolveOptions<StorageT> solve;
    bool keep_temp_files = false;
    uint32_t source_bitmap_words_per_work_item = 64U;
    uint32_t source_work_schedule_chunk = 1U;
    uint32_t cell_parallel_min_work_items = 4U;
    uint32_t future_reuse_max_families = 4U;
    uint64_t future_index_recycle_max_bytes = 0U;
    bool temp_direct_io = false;
    bool force_temp_buffered_io = false;
    uint32_t temp_direct_queue_depth = 16U;
    uint64_t final_pending_value_memory_cap_bytes = 0U;
    bool sparse_partial_temp = false;
    bool use_diagonal_grouped_sum = false;
    bool interleave_spawn_phases = false;
    uint32_t interleave_block_fids = 1U;
    uint64_t interleave_scratch4_memory_cap_bytes = 0U;
    bool interleave_release_future_between_phases = true;
    bool interleave_keep_future4_between_blocks = false;
    bool interleave_keep_future2_between_blocks = false;
    bool collect_edge_stats = false;
    bool collect_batch_timing = false;
    bool collect_temp_sparsity = false;
    bool collect_resident_breakdown = false;
};

struct BCFamilySolveFileResult {
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    BCFamilySolveStats stats;
};

struct BCFamilySolveCellRangeWork {
    uint32_t bucket_begin = 0U;
    uint32_t bucket_end = 0U;
    uint32_t word_begin = 0U;
    uint32_t word_end = 0U;
};

struct BCFamilyBucketSpawnTargetHits {
    uint16_t horizontal_mask = 0U;
    uint16_t vertical_mask = 0U;
};

inline constexpr std::array<uint16_t, 4U> kBCFamilyQuadrantCellMasks{
    static_cast<uint16_t>((1U << 15U) | (1U << 14U) | (1U << 11U) | (1U << 10U)),
    static_cast<uint16_t>((1U << 13U) | (1U << 12U) | (1U << 9U) | (1U << 8U)),
    static_cast<uint16_t>((1U << 7U) | (1U << 6U) | (1U << 3U) | (1U << 2U)),
    static_cast<uint16_t>((1U << 5U) | (1U << 4U) | (1U << 1U) | (1U << 0U)),
};

inline constexpr std::array<uint32_t, 4U> kBCFamilyQuadrantRepresentativeCells{
    15U,
    13U,
    7U,
    5U,
};

[[nodiscard]] inline bool bc_family_bucket_hit_horizontal(
    const BCFamilyBucketSpawnTargetHits &hits,
    uint32_t cell
) {
    return ((hits.horizontal_mask >> cell) & 1U) != 0U;
}

[[nodiscard]] inline bool bc_family_bucket_hit_vertical(
    const BCFamilyBucketSpawnTargetHits &hits,
    uint32_t cell
) {
    return ((hits.vertical_mask >> cell) & 1U) != 0U;
}

template <typename StorageT>
struct BCFamilySolveCellWorkspace {
    BCSolveEdgeWorkspace<StorageT> edge_workspace;
    BCResidentBatchWorkspace<StorageT> batch_workspace;
    std::array<uint32_t, BCResidentBatchWorkspace<StorageT>::kBatchSize> local_success_rows{};
    std::array<uint32_t, BCResidentBatchWorkspace<StorageT>::kBatchSize> local_bucket_indices{};
    std::array<uint16_t, BCResidentBatchWorkspace<StorageT>::kBatchSize> local_cell_indices{};
    std::array<uint16_t, BCResidentBatchWorkspace<StorageT>::kBatchSize> local_compact_offsets{};
    std::vector<uint16_t> compact_ref_board_slots;
    BCFamilyValueVector<StorageT> compact_best_values;
    BCFamilyValueVector<StorageT> lane_best_values;
    BCFamilyValueVector<StorageT> merged_best;
    BCSingleChunkSolveStats stats;
    double spawn4_batch_candidate_seconds = 0.0;
    double spawn4_batch_canonical_seconds = 0.0;
    double spawn4_batch_setup_seconds = 0.0;
    double spawn4_batch_reduce_seconds = 0.0;
    double spawn4_batch_emit_seconds = 0.0;
    double spawn2_batch_candidate_seconds = 0.0;
    double spawn2_batch_canonical_seconds = 0.0;
    double spawn2_batch_setup_seconds = 0.0;
    double spawn2_batch_reduce_seconds = 0.0;
    double spawn2_batch_emit_seconds = 0.0;
    uint64_t spawn4_batch_canonical_candidates = 0U;
    uint64_t spawn4_batch_encoded_queries = 0U;
    uint64_t spawn4_batch_reduce_found = 0U;
    uint64_t spawn4_batch_entry_misses = 0U;
    uint64_t spawn4_batch_bitmap_misses = 0U;
    uint64_t spawn2_batch_canonical_candidates = 0U;
    uint64_t spawn2_batch_encoded_queries = 0U;
    uint64_t spawn2_batch_reduce_found = 0U;
    uint64_t spawn2_batch_entry_misses = 0U;
    uint64_t spawn2_batch_bitmap_misses = 0U;
};

template <typename StorageT>
struct BCFamilySolveWorkspace {
    BCSolveEdgeWorkspace<StorageT> edge_workspace;
    std::vector<BCFamilySolveCellWorkspace<StorageT>> cell_workspaces;
    std::vector<BCFamilySolveCellRangeWork> cell_range_work_items;
    std::vector<BCSingleChunkLoadedWorkItem> pass_cell_work_items;
    std::vector<BCFamilyBucketSpawnTargetHits> bucket_target_hits;
    std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> pass_bucket_target_hits;
    BCSingleChunkValueBuffer<StorageT> compact_buffer;
};

namespace detail {

inline constexpr uint64_t kBCFamilySolveTempMagic = 0x31504d5446534342ULL; // "BCSF TMP1".

[[nodiscard]] inline double bc_family_positive_remainder(double elapsed, double accounted) {
    return elapsed > accounted ? elapsed - accounted : 0.0;
}

[[nodiscard]] inline uint16_t bc_family_checked_u16_size(size_t value, const char *label) {
    if (value > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<uint16_t>(value);
}

[[nodiscard]] inline uint16_t bc_family_checked_u16_u32(uint32_t value, const char *label) {
    if (value > static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<uint16_t>(value);
}

[[nodiscard]] inline uint64_t bc_family_loaded_cell_resident_bytes(const BCLoadedCell &cell) {
    return static_cast<uint64_t>(cell.buckets.capacity()) * sizeof(BCBucketEntry) +
        static_cast<uint64_t>(cell.rank_payload.capacity());
}

[[nodiscard]] inline uint64_t bc_family_loaded_cells_resident_bytes(
    const std::vector<BCLoadedCell> &cells
) {
    uint64_t bytes = static_cast<uint64_t>(cells.capacity()) * sizeof(BCLoadedCell);
    for (const BCLoadedCell &cell : cells) {
        bytes += bc_family_loaded_cell_resident_bytes(cell);
    }
    return bytes;
}

template <typename StorageT>
[[nodiscard]] uint64_t bc_family_value_vector_resident_bytes(
    const BCFamilyValueVector<StorageT> &values
) {
    return static_cast<uint64_t>(values.capacity()) * sizeof(StorageT);
}

template <typename StorageT>
[[nodiscard]] uint64_t bc_family_value_buffer_resident_bytes(
    const BCSingleChunkValueBuffer<StorageT> &values
) {
    return static_cast<uint64_t>(values.capacity()) * sizeof(StorageT);
}

template <typename StorageT>
[[nodiscard]] uint64_t bc_family_value_vectors_resident_bytes(
    const std::vector<BCFamilyValueVector<StorageT>> &values
) {
    uint64_t bytes =
        static_cast<uint64_t>(values.capacity()) * sizeof(BCFamilyValueVector<StorageT>);
    for (const BCFamilyValueVector<StorageT> &cell_values : values) {
        bytes += bc_family_value_vector_resident_bytes(cell_values);
    }
    return bytes;
}

template <typename StorageT>
void bc_family_release_value_vectors(
    std::vector<BCFamilyValueVector<StorageT>> &values,
    int threads
) {
    const size_t count = values.size();
    if (count >= 8U && threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            BCFamilyValueVector<StorageT>().swap(values[static_cast<size_t>(i)]);
        }
    } else {
        for (BCFamilyValueVector<StorageT> &value : values) {
            BCFamilyValueVector<StorageT>().swap(value);
        }
    }
    std::vector<BCFamilyValueVector<StorageT>>().swap(values);
}

template <typename StorageT>
void bc_family_release_partial_buffers(
    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> &buffers,
    int threads
) {
    const size_t count = buffers.size();
    if (count >= 8U && threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            BCFamilyValueVector<StorageT>().swap(buffers[static_cast<size_t>(i)].values());
        }
    } else {
        for (BCFamilyPartialMaxCellBuffer<StorageT> &buffer : buffers) {
            BCFamilyValueVector<StorageT>().swap(buffer.values());
        }
    }
    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>>().swap(buffers);
}

template <typename StorageT>
void bc_family_release_success_scratch(
    std::vector<BCFamilyCellSuccessScratch<StorageT>> &scratch,
    int threads
) {
    const size_t count = scratch.size();
    if (count >= 8U && threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            BCFamilyValueVector<StorageT>().swap(scratch[static_cast<size_t>(i)].values());
        }
    } else {
        for (BCFamilyCellSuccessScratch<StorageT> &cell : scratch) {
            BCFamilyValueVector<StorageT>().swap(cell.values());
        }
    }
    std::vector<BCFamilyCellSuccessScratch<StorageT>>().swap(scratch);
}

template <typename StorageT>
void bc_family_profile_partial_temp_value_vector(
    const BCFamilyValueVector<StorageT> &values,
    StorageT zero_value,
    bool spawn4,
    BCFamilySolveStats &stats
) {
    uint64_t zero = 0U;
    for (const StorageT value : values) {
        if (value == zero_value) {
            ++zero;
        }
    }
    if (spawn4) {
        stats.partial4_profiled_values = bc_checked_add_u64(
            stats.partial4_profiled_values,
            values.size(),
            "BC family partial4 sparsity stats overflow"
        );
        stats.partial4_profiled_zero_values = bc_checked_add_u64(
            stats.partial4_profiled_zero_values,
            zero,
            "BC family partial4 zero sparsity stats overflow"
        );
    } else {
        stats.partial2_profiled_values = bc_checked_add_u64(
            stats.partial2_profiled_values,
            values.size(),
            "BC family partial2 sparsity stats overflow"
        );
        stats.partial2_profiled_zero_values = bc_checked_add_u64(
            stats.partial2_profiled_zero_values,
            zero,
            "BC family partial2 zero sparsity stats overflow"
        );
    }
}

template <typename StorageT>
[[nodiscard]] bool bc_family_sparse_partial_temp_enabled(
    const BCFamilySolveOptions<StorageT> &options
) {
    return options.sparse_partial_temp && std::is_integral_v<StorageT>;
}

template <typename StorageT>
[[nodiscard]] BCFamilyValueVector<StorageT> bc_family_encode_partial_temp_values(
    const BCFamilyPartialCellLayout &layout,
    BCFamilyValueVector<StorageT> dense_values,
    StorageT zero_value,
    const BCFamilySolveOptions<StorageT> &options
) {
    if (!bc_family_sparse_partial_temp_enabled(options)) {
        return dense_values;
    }
    if (dense_values.size() != static_cast<size_t>(layout.value_count)) {
        throw std::runtime_error("BC family sparse partial encode dense count mismatch");
    }
    if constexpr (!std::is_integral_v<StorageT>) {
        return dense_values;
    } else {
        constexpr StorageT kTag = std::numeric_limits<StorageT>::max();
        BCFamilyValueVector<StorageT> encoded;
        encoded.reserve(dense_values.size() / 4U + 8U);
        encoded.push_back(kTag);
        encoded.push_back(static_cast<StorageT>(1U));
        encoded.push_back(StorageT{});
        uint64_t nonzero_rows = 0U;
        for (const BCFamilyPartialBucketLayout &bucket : layout.buckets) {
            const size_t row_values =
                static_cast<size_t>(bucket.empty_count) *
                static_cast<size_t>(layout.row_width);
            if (row_values == 0U) {
                continue;
            }
            for (uint32_t row = bucket.success_row_begin;
                 row < bucket.success_row_end();
                 ++row) {
                const uint64_t base = bucket.value_offset +
                    static_cast<uint64_t>(row - bucket.success_row_begin) *
                        static_cast<uint64_t>(row_values);
                if (base + row_values > dense_values.size()) {
                    throw std::out_of_range("BC family sparse partial encode row out of range");
                }
                const StorageT *src = dense_values.data() + static_cast<size_t>(base);
                bool any_nonzero = false;
                for (size_t i = 0U; i < row_values; ++i) {
                    if (src[i] != zero_value) {
                        any_nonzero = true;
                        break;
                    }
                }
                if (!any_nonzero) {
                    continue;
                }
                if (row == static_cast<uint32_t>(kTag)) {
                    throw std::overflow_error("BC family sparse partial row collides with tag");
                }
                encoded.push_back(static_cast<StorageT>(row));
                encoded.insert(encoded.end(), src, src + row_values);
                ++nonzero_rows;
                if (encoded.size() >= dense_values.size()) {
                    return dense_values;
                }
            }
        }
        if (nonzero_rows > static_cast<uint64_t>(std::numeric_limits<StorageT>::max())) {
            throw std::overflow_error("BC family sparse partial row count exceeds storage type");
        }
        encoded[2] = static_cast<StorageT>(nonzero_rows);
        if (encoded.size() >= dense_values.size()) {
            return dense_values;
        }
        return encoded;
    }
}

template <typename StorageT>
[[nodiscard]] BCFamilyValueVector<StorageT> bc_family_decode_partial_temp_values(
    const BCFamilyPartialCellLayout &layout,
    BCFamilyValueVector<StorageT> stored_values,
    StorageT zero_value,
    const BCFamilySolveOptions<StorageT> &options
) {
    if (stored_values.size() == static_cast<size_t>(layout.value_count)) {
        return stored_values;
    }
    if (!bc_family_sparse_partial_temp_enabled(options)) {
        throw std::runtime_error("BC family dense partial temp count mismatch");
    }
    if constexpr (!std::is_integral_v<StorageT>) {
        throw std::runtime_error("BC family sparse partial temp requires integral storage");
    } else {
        constexpr StorageT kTag = std::numeric_limits<StorageT>::max();
        if (stored_values.size() < 3U ||
            stored_values[0] != kTag ||
            stored_values[1] != static_cast<StorageT>(1U)) {
            throw std::runtime_error("BC family sparse partial temp header mismatch");
        }
        if (layout.value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family sparse partial dense count exceeds size_t");
        }
        BCFamilyValueVector<StorageT> dense(
            static_cast<size_t>(layout.value_count),
            zero_value
        );
        const uint64_t row_count = static_cast<uint64_t>(stored_values[2]);
        size_t cursor = 3U;
        size_t bucket_i = 0U;
        uint32_t previous_row = 0U;
        bool have_previous_row = false;
        for (uint64_t row_i = 0U; row_i < row_count; ++row_i) {
            if (cursor >= stored_values.size()) {
                throw std::runtime_error("BC family sparse partial temp truncated row index");
            }
            const uint32_t row = static_cast<uint32_t>(stored_values[cursor++]);
            if (have_previous_row && row <= previous_row) {
                throw std::runtime_error("BC family sparse partial temp rows are not increasing");
            }
            previous_row = row;
            have_previous_row = true;
            while (bucket_i < layout.buckets.size() &&
                   row >= layout.buckets[bucket_i].success_row_end()) {
                ++bucket_i;
            }
            if (bucket_i >= layout.buckets.size()) {
                throw std::out_of_range("BC family sparse partial decode row has no bucket");
            }
            const BCFamilyPartialBucketLayout &bucket = layout.buckets[bucket_i];
            if (row < bucket.success_row_begin || row >= bucket.success_row_end()) {
                throw std::out_of_range("BC family sparse partial decode row is outside bucket");
            }
            const size_t row_values =
                static_cast<size_t>(bucket.empty_count) *
                static_cast<size_t>(layout.row_width);
            const uint64_t base = bucket.value_offset +
                static_cast<uint64_t>(row - bucket.success_row_begin) *
                    static_cast<uint64_t>(row_values);
            if (base + row_values > dense.size()) {
                throw std::out_of_range("BC family sparse partial decode row out of range");
            }
            if (cursor + row_values > stored_values.size()) {
                throw std::runtime_error("BC family sparse partial temp truncated row values");
            }
            std::copy_n(
                stored_values.data() + cursor,
                row_values,
                dense.data() + static_cast<size_t>(base)
            );
            cursor += row_values;
        }
        if (cursor != stored_values.size()) {
            throw std::runtime_error("BC family sparse partial temp trailing values");
        }
        return dense;
    }
}

template <typename StorageT>
[[nodiscard]] BCFamilyValueVector<StorageT> bc_family_prepare_partial_temp_output(
    const BCFamilyPartialCellLayout &layout,
    BCFamilyValueVector<StorageT> dense_values,
    bool spawn4,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveStats &stats
) {
    if (options.collect_temp_sparsity) {
        bc_family_profile_partial_temp_value_vector(
            dense_values,
            options.solve.zero_value,
            spawn4,
            stats
        );
    }
    return bc_family_encode_partial_temp_values(
        layout,
        std::move(dense_values),
        options.solve.zero_value,
        options
    );
}

struct BCFamilySolveTempRecord {
    static constexpr uint64_t kUnwrittenOffset = std::numeric_limits<uint64_t>::max();

    uint64_t value_offset = kUnwrittenOffset;
    uint64_t value_count = 0U;

    [[nodiscard]] bool written() const noexcept {
        return value_offset != kUnwrittenOffset;
    }
};

static_assert(sizeof(BCFamilySolveTempRecord) == 16U, "BC family temp record should stay compact");

struct BCFamilyCurrentCellLoadResult {
    std::vector<BCLoadedCell> cells;
    BCCellLoadStats load_stats;
    double seconds = 0.0;
};

template <typename T>
class BCFamilySolveTempValueFile {
public:
    BCFamilySolveTempValueFile() = default;

    BCFamilySolveTempValueFile(
        const std::filesystem::path &path,
        uint32_t kind
    ) {
        open(path, kind);
    }

    BCFamilySolveTempValueFile(const BCFamilySolveTempValueFile &) = delete;
    BCFamilySolveTempValueFile &operator=(const BCFamilySolveTempValueFile &) = delete;

    void open(
        const std::filesystem::path &path,
        uint32_t kind,
        bool direct_io = false,
        uint32_t direct_queue_depth = 16U
    ) {
        path_ = path;
        kind_ = kind;
        direct_io_ = direct_io;
        if (!path_.parent_path().empty()) {
            std::filesystem::create_directories(path_.parent_path());
        }
        if (direct_io_) {
            direct_options_.queue_depth = direct_queue_depth == 0U ? 1U : direct_queue_depth;
            direct_options_.overlapped = direct_options_.queue_depth > 1U;
            direct_options_.preserve_unwritten_bytes = false;
            alignment_ = direct_options_.alignment;
            if ((alignment_ % sizeof(T)) != 0U) {
                throw std::invalid_argument("BC family solve temp direct alignment is not value-aligned");
            }
            payload_offset_ = alignment_;
            writer_ = std::make_unique<BCDirectFileWriter>(path_, direct_options_);
            reader_ = std::make_unique<BCDirectFileReader>(path_, direct_options_);
        } else {
            alignment_ = 1U;
            payload_offset_ = kHeaderBytes;
            writer_ = std::make_unique<BCBufferedFileWriter>(path_);
            reader_ = std::make_unique<BCBufferedFileReader>(path_);
        }
        write_header();
    }

    [[nodiscard]] uint64_t append(
        const BCFamilyValueVector<T> &values,
        BCFamilySolveStats *stats
    ) {
        return append(values.data(), values.size(), stats);
    }

    [[nodiscard]] uint64_t append(
        const T *values,
        size_t count,
        BCFamilySolveStats *stats
    ) {
        if (count != 0U && values == nullptr) {
            throw std::invalid_argument("BC family solve temp append pointer is null");
        }
        if (direct_io_) {
            align_value_cursor_for_direct();
        }
        const uint64_t offset = value_cursor_;
        const uint64_t bytes = checked_bytes(count);
        const double t0 = bc_single_chunk_now_seconds();
        if (bytes != 0U) {
            BCFileIOStats io_stats;
            writer_->write_many(
                std::vector<BCFileWriteRequest>{
                    BCFileWriteRequest{byte_offset_for_value(value_cursor_), values, bytes}
                },
                &io_stats
            );
            if (stats != nullptr) {
                bc_success_accumulate_file_stats(&stats->temp_write_io, io_stats);
            }
        }
        value_cursor_ = bc_checked_add_u64(
            value_cursor_,
            static_cast<uint64_t>(count),
            "BC family solve temp value cursor overflow"
        );
        if (stats != nullptr) {
            stats->temp_write_seconds += bc_single_chunk_now_seconds() - t0;
            stats->temp_values_written += count;
            stats->temp_bytes_written += bytes;
            stats->single.partial_write_bytes = bc_checked_add_u64(
                stats->single.partial_write_bytes,
                bytes,
                "BC family solve temp write byte stats overflow"
            );
        }
        return offset;
    }

    [[nodiscard]] std::vector<BCFamilySolveTempRecord> append_many(
        const std::vector<const BCFamilyValueVector<T> *> &value_vectors,
        BCFamilySolveStats *stats
    ) {
        std::vector<BCFamilySolveTempRecord> records(value_vectors.size());
        if (value_vectors.empty()) {
            return records;
        }
        const double t0 = bc_single_chunk_now_seconds();
        std::vector<BCFileWriteRequest> requests;
        requests.reserve(value_vectors.size());
        uint64_t total_values = 0U;
        uint64_t total_bytes = 0U;
        for (size_t i = 0U; i < value_vectors.size(); ++i) {
            const BCFamilyValueVector<T> *values = value_vectors[i];
            if (values == nullptr) {
                throw std::invalid_argument("BC family solve temp batch append values are null");
            }
            if (direct_io_) {
                align_value_cursor_for_direct();
            }
            BCFamilySolveTempRecord &record = records[i];
            record.value_offset = value_cursor_;
            record.value_count = values->size();
            const uint64_t bytes = checked_bytes(values->size());
            if (bytes != 0U) {
                requests.push_back(BCFileWriteRequest{
                    byte_offset_for_value(value_cursor_),
                    values->data(),
                    bytes
                });
            }
            value_cursor_ = bc_checked_add_u64(
                value_cursor_,
                static_cast<uint64_t>(values->size()),
                "BC family solve temp batch value cursor overflow"
            );
            total_values = bc_checked_add_u64(
                total_values,
                record.value_count,
                "BC family solve temp batch value write stats overflow"
            );
            total_bytes = bc_checked_add_u64(
                total_bytes,
                bytes,
                "BC family solve temp batch byte write stats overflow"
            );
        }
        if (!requests.empty()) {
            BCFileIOStats io_stats;
            writer_->write_many(requests, &io_stats);
            if (stats != nullptr) {
                bc_success_accumulate_file_stats(&stats->temp_write_io, io_stats);
            }
        }
        if (stats != nullptr) {
            stats->temp_write_seconds += bc_single_chunk_now_seconds() - t0;
            stats->temp_values_written += total_values;
            stats->temp_bytes_written += total_bytes;
            stats->single.partial_write_bytes = bc_checked_add_u64(
                stats->single.partial_write_bytes,
                total_bytes,
                "BC family solve temp batch write byte stats overflow"
            );
        }
        return records;
    }

    [[nodiscard]] BCFamilyValueVector<T> read(
        uint64_t value_offset,
        uint64_t value_count,
        BCFamilySolveStats *stats
    ) {
        if (value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family solve temp read count exceeds size_t");
        }
        if (value_offset > value_cursor_ || value_count > value_cursor_ - value_offset) {
            throw std::out_of_range("BC family solve temp read exceeds written values");
        }
        const size_t count = static_cast<size_t>(value_count);
        const double prepare_t0 = bc_single_chunk_now_seconds();
        BCFamilyValueVector<T> out(count);
        const uint64_t bytes = checked_bytes(count);
        if (stats != nullptr) {
            stats->temp_read_prepare_seconds += bc_single_chunk_now_seconds() - prepare_t0;
        }
        const double t0 = bc_single_chunk_now_seconds();
        if (!direct_io_ && writer_) {
            writer_->flush();
        }
        if (bytes != 0U) {
            BCFileIOStats io_stats;
            reader_->read_many(
                std::vector<BCFileReadRequest>{
                    BCFileReadRequest{byte_offset_for_value(value_offset), out.data(), bytes}
                },
                &io_stats
            );
            if (stats != nullptr) {
                bc_success_accumulate_file_stats(&stats->temp_read_io, io_stats);
            }
        }
        if (stats != nullptr) {
            stats->temp_read_seconds += bc_single_chunk_now_seconds() - t0;
            stats->temp_values_read += count;
            stats->temp_bytes_read += bytes;
            stats->single.partial_read_bytes = bc_checked_add_u64(
                stats->single.partial_read_bytes,
                bytes,
                "BC family solve temp read byte stats overflow"
            );
        }
        return out;
    }

    void read_many_records(
        const std::vector<BCFamilySolveTempRecord> &records,
        const std::vector<CellId> &cids,
        std::vector<BCFamilyValueVector<T>> &out,
        BCFamilySolveStats *stats
    ) {
        out.clear();
        out.resize(cids.size());
        if (cids.empty()) {
            return;
        }
        const double prepare_t0 = bc_single_chunk_now_seconds();
        std::vector<BCFileReadRequest> requests;
        requests.reserve(cids.size());
        uint64_t total_values = 0U;
        uint64_t total_bytes = 0U;
        for (size_t i = 0U; i < cids.size(); ++i) {
            const CellId cid = cids[i];
            if (cid >= records.size()) {
                throw std::out_of_range("BC family solve temp batch cid out of range");
            }
            const BCFamilySolveTempRecord &record = records[static_cast<size_t>(cid)];
            if (!record.written()) {
                throw std::runtime_error("BC family solve missing temp batch record");
            }
            if (record.value_offset > value_cursor_ ||
                record.value_count > value_cursor_ - record.value_offset) {
                throw std::out_of_range("BC family solve temp batch read exceeds written values");
            }
            if (record.value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC family solve temp batch read count exceeds size_t");
            }
            BCFamilyValueVector<T> &values = out[i];
            values.resize(static_cast<size_t>(record.value_count));
            const uint64_t bytes = checked_bytes(values.size());
            total_values = bc_checked_add_u64(
                total_values,
                record.value_count,
                "BC family solve temp batch value read stats overflow"
            );
            total_bytes = bc_checked_add_u64(
                total_bytes,
                bytes,
                "BC family solve temp batch byte read stats overflow"
            );
            if (bytes != 0U) {
                requests.push_back(BCFileReadRequest{
                    byte_offset_for_value(record.value_offset),
                    values.data(),
                    bytes
                });
            }
        }
        if (stats != nullptr) {
            stats->temp_read_prepare_seconds += bc_single_chunk_now_seconds() - prepare_t0;
        }
        const double t0 = bc_single_chunk_now_seconds();
        if (!direct_io_ && writer_) {
            writer_->flush();
        }
        if (!requests.empty()) {
            BCFileIOStats io_stats;
            reader_->read_many(requests, &io_stats);
            if (stats != nullptr) {
                bc_success_accumulate_file_stats(&stats->temp_read_io, io_stats);
            }
        }
        if (stats != nullptr) {
            stats->temp_read_seconds += bc_single_chunk_now_seconds() - t0;
            stats->temp_values_read += total_values;
            stats->temp_bytes_read += total_bytes;
            stats->single.partial_read_bytes = bc_checked_add_u64(
                stats->single.partial_read_bytes,
                total_bytes,
                "BC family solve temp batch read byte stats overflow"
            );
        }
    }

    void close() {
        if (writer_) {
            if (writer_->mode() != BCFileIOMode::Direct) {
                writer_->flush();
            }
            writer_.reset();
        }
        reader_.reset();
    }

    [[nodiscard]] const std::filesystem::path &path() const {
        return path_;
    }

private:
    static constexpr uint64_t kHeaderBytes = 32U;

    [[nodiscard]] static uint64_t checked_bytes(size_t count) {
        if (count > static_cast<size_t>(std::numeric_limits<uint64_t>::max() / sizeof(T))) {
            throw std::overflow_error("BC family solve temp byte count overflow");
        }
        return static_cast<uint64_t>(count) * sizeof(T);
    }

    [[nodiscard]] uint64_t byte_offset_for_value(uint64_t value_offset) const {
        if (value_offset > (std::numeric_limits<uint64_t>::max() - payload_offset_) / sizeof(T)) {
            throw std::overflow_error("BC family solve temp byte offset overflow");
        }
        return payload_offset_ + value_offset * sizeof(T);
    }

    void align_value_cursor_for_direct() {
        const uint64_t values_per_alignment = alignment_ / sizeof(T);
        if (values_per_alignment <= 1U) {
            return;
        }
        if (value_cursor_ > std::numeric_limits<uint64_t>::max() - (values_per_alignment - 1U)) {
            throw std::overflow_error("BC family solve temp value cursor align overflow");
        }
        value_cursor_ =
            (value_cursor_ + values_per_alignment - 1U) / values_per_alignment * values_per_alignment;
    }

    void write_header() {
        std::array<uint8_t, static_cast<size_t>(kHeaderBytes)> header{};
        bc_append_u64_at(header, 0U, kBCFamilySolveTempMagic);
        bc_append_u32_at(header, 8U, 1U);
        bc_append_u32_at(header, 12U, kind_);
        bc_append_u32_at(header, 16U, static_cast<uint32_t>(sizeof(T)));
        writer_->write_many(
            std::vector<BCFileWriteRequest>{
                BCFileWriteRequest{0U, header.data(), header.size()}
            }
        );
        if (!direct_io_) {
            writer_->flush();
        }
    }

    static void bc_append_u32_at(std::array<uint8_t, static_cast<size_t>(kHeaderBytes)> &out,
                                 size_t offset,
                                 uint32_t value) {
        out[offset + 0U] = static_cast<uint8_t>(value & 0xFFU);
        out[offset + 1U] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        out[offset + 2U] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        out[offset + 3U] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    }

    static void bc_append_u64_at(std::array<uint8_t, static_cast<size_t>(kHeaderBytes)> &out,
                                 size_t offset,
                                 uint64_t value) {
        for (uint32_t i = 0U; i < 8U; ++i) {
            out[offset + i] = static_cast<uint8_t>((value >> (8U * i)) & 0xFFU);
        }
    }

    std::filesystem::path path_;
    uint32_t kind_ = 0U;
    bool direct_io_ = false;
    uint32_t alignment_ = 1U;
    uint64_t payload_offset_ = kHeaderBytes;
    BCDirectFileIOOptions direct_options_;
    std::unique_ptr<BCWritableFile> writer_;
    std::unique_ptr<BCReadableFile> reader_;
    uint64_t value_cursor_ = 0U;
};

template <typename StorageT>
class BCFamilySolveTempStore {
public:
    BCFamilySolveTempStore() = default;

    BCFamilySolveTempStore(
        const std::filesystem::path &dir,
        uint32_t cell_count,
        bool direct_io = false,
        uint32_t direct_queue_depth = 16U
    ) {
        open(dir, cell_count, direct_io, direct_queue_depth);
    }

    void open(
        const std::filesystem::path &dir,
        uint32_t cell_count,
        bool direct_io = false,
        uint32_t direct_queue_depth = 16U
    ) {
        dir_ = dir;
        std::filesystem::create_directories(dir_);
        partial4_records_.assign(cell_count, {});
        partial2_records_.assign(cell_count, {});
        scratch4_records_.assign(cell_count, {});
        partial4_.open(dir_ / "family_partial4.bcfstmp", 1U, direct_io, direct_queue_depth);
        partial2_.open(dir_ / "family_partial2.bcfstmp", 2U, direct_io, direct_queue_depth);
        scratch4_.open(dir_ / "family_scratch4.bcfstmp", 3U, direct_io, direct_queue_depth);
    }

    void write_partial4(CellId cid, const BCFamilyValueVector<StorageT> &values, BCFamilySolveStats &stats) {
        wait_partial4_write(stats);
        write_record(cid, values, partial4_, partial4_records_, stats);
        ++stats.partial4_cells_written;
    }

    void write_partial2(CellId cid, const BCFamilyValueVector<StorageT> &values, BCFamilySolveStats &stats) {
        wait_partial2_write(stats);
        write_record(cid, values, partial2_, partial2_records_, stats);
        ++stats.partial2_cells_written;
    }

    void write_scratch4(CellId cid, const BCFamilyValueVector<StorageT> &values, BCFamilySolveStats &stats) {
        wait_scratch4_write(stats);
        write_record(cid, values, scratch4_, scratch4_records_, stats);
        ++stats.scratch4_cells_written;
    }

    void write_partial4_batch(
        const std::vector<CellId> &cids,
        const std::vector<BCFamilyValueVector<StorageT>> &values,
        BCFamilySolveStats &stats
    ) {
        wait_partial4_write(stats);
        write_records_batch(cids, values, partial4_, partial4_records_, stats);
        stats.partial4_cells_written += cids.size();
    }

    void write_partial2_batch(
        const std::vector<CellId> &cids,
        const std::vector<BCFamilyValueVector<StorageT>> &values,
        BCFamilySolveStats &stats
    ) {
        wait_partial2_write(stats);
        write_records_batch(cids, values, partial2_, partial2_records_, stats);
        stats.partial2_cells_written += cids.size();
    }

    void write_scratch4_batch(
        const std::vector<CellId> &cids,
        const std::vector<BCFamilyValueVector<StorageT>> &values,
        BCFamilySolveStats &stats
    ) {
        wait_scratch4_write(stats);
        write_records_batch(cids, values, scratch4_, scratch4_records_, stats);
        stats.scratch4_cells_written += cids.size();
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read_partial4(CellId cid, uint64_t expected_count, BCFamilySolveStats &stats) {
        wait_partial4_write(stats);
        ++stats.partial4_cells_read;
        return read_record(cid, expected_count, partial4_, partial4_records_, stats);
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read_partial4_any(CellId cid, BCFamilySolveStats &stats) {
        wait_partial4_write(stats);
        ++stats.partial4_cells_read;
        return read_record_any(cid, partial4_, partial4_records_, stats);
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read_partial2(CellId cid, uint64_t expected_count, BCFamilySolveStats &stats) {
        wait_partial2_write(stats);
        ++stats.partial2_cells_read;
        return read_record(cid, expected_count, partial2_, partial2_records_, stats);
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read_partial2_any(CellId cid, BCFamilySolveStats &stats) {
        wait_partial2_write(stats);
        ++stats.partial2_cells_read;
        return read_record_any(cid, partial2_, partial2_records_, stats);
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read_scratch4(CellId cid, uint64_t expected_count, BCFamilySolveStats &stats) {
        wait_scratch4_write(stats);
        ++stats.scratch4_cells_read;
        return read_record(cid, expected_count, scratch4_, scratch4_records_, stats);
    }

    void read_partial4_batch(
        const std::vector<CellId> &cids,
        std::vector<BCFamilyValueVector<StorageT>> &out,
        BCFamilySolveStats &stats
    ) {
        wait_partial4_write(stats);
        partial4_.read_many_records(partial4_records_, cids, out, &stats);
        stats.partial4_cells_read += cids.size();
    }

    void read_partial2_batch(
        const std::vector<CellId> &cids,
        std::vector<BCFamilyValueVector<StorageT>> &out,
        BCFamilySolveStats &stats
    ) {
        wait_partial2_write(stats);
        partial2_.read_many_records(partial2_records_, cids, out, &stats);
        stats.partial2_cells_read += cids.size();
    }

    void read_scratch4_batch(
        const std::vector<CellId> &cids,
        std::vector<BCFamilyValueVector<StorageT>> &out,
        BCFamilySolveStats &stats
    ) {
        wait_scratch4_write(stats);
        scratch4_.read_many_records(scratch4_records_, cids, out, &stats);
        stats.scratch4_cells_read += cids.size();
    }

    void wait_all_writes(BCFamilySolveStats &stats) {
        (void)stats;
    }

    void close() {
        partial4_.close();
        partial2_.close();
        scratch4_.close();
    }

    void cleanup() {
        close();
        std::error_code ec;
        std::filesystem::remove(partial4_.path(), ec);
        std::filesystem::remove(partial2_.path(), ec);
        std::filesystem::remove(scratch4_.path(), ec);
    }

private:
    using TempFile = BCFamilySolveTempValueFile<StorageT>;

    static void require_cid(CellId cid, const std::vector<BCFamilySolveTempRecord> &records) {
        if (cid >= records.size()) {
            throw std::out_of_range("BC family solve temp cid out of range");
        }
    }

    static void write_record(
        CellId cid,
        const BCFamilyValueVector<StorageT> &values,
        TempFile &file,
        std::vector<BCFamilySolveTempRecord> &records,
        BCFamilySolveStats &stats
    ) {
        require_cid(cid, records);
        BCFamilySolveTempRecord &record = records[static_cast<size_t>(cid)];
        if (record.written()) {
            throw std::runtime_error("BC family solve duplicate temp record");
        }
        record.value_offset = file.append(values, &stats);
        record.value_count = values.size();
    }

    static void write_records_batch(
        const std::vector<CellId> &cids,
        const std::vector<BCFamilyValueVector<StorageT>> &values,
        TempFile &file,
        std::vector<BCFamilySolveTempRecord> &records,
        BCFamilySolveStats &stats
    ) {
        if (cids.size() != values.size()) {
            throw std::invalid_argument("BC family solve temp batch write cids/value size mismatch");
        }
        if (cids.empty()) {
            return;
        }
        std::vector<const BCFamilyValueVector<StorageT> *> value_views;
        value_views.reserve(values.size());
        for (size_t i = 0U; i < cids.size(); ++i) {
            const CellId cid = cids[i];
            require_cid(cid, records);
            if (records[static_cast<size_t>(cid)].written()) {
                throw std::runtime_error("BC family solve duplicate temp batch record");
            }
            value_views.push_back(&values[i]);
        }
        std::vector<BCFamilySolveTempRecord> appended = file.append_many(value_views, &stats);
        if (appended.size() != cids.size()) {
            throw std::runtime_error("BC family solve temp append_many returned unexpected count");
        }
        for (size_t i = 0U; i < cids.size(); ++i) {
            records[static_cast<size_t>(cids[i])] = appended[i];
        }
    }

    static BCFamilyValueVector<StorageT> read_record(
        CellId cid,
        uint64_t expected_count,
        TempFile &file,
        const std::vector<BCFamilySolveTempRecord> &records,
        BCFamilySolveStats &stats
    ) {
        require_cid(cid, records);
        const BCFamilySolveTempRecord &record = records[static_cast<size_t>(cid)];
        if (!record.written()) {
            throw std::runtime_error("BC family solve missing temp record");
        }
        if (record.value_count != expected_count) {
            throw std::runtime_error("BC family solve temp value count mismatch");
        }
        return file.read(record.value_offset, record.value_count, &stats);
    }

    static BCFamilyValueVector<StorageT> read_record_any(
        CellId cid,
        TempFile &file,
        const std::vector<BCFamilySolveTempRecord> &records,
        BCFamilySolveStats &stats
    ) {
        require_cid(cid, records);
        const BCFamilySolveTempRecord &record = records[static_cast<size_t>(cid)];
        if (!record.written()) {
            throw std::runtime_error("BC family solve missing temp record");
        }
        return file.read(record.value_offset, record.value_count, &stats);
    }

    std::filesystem::path dir_;
    TempFile partial4_;
    TempFile partial2_;
    TempFile scratch4_;
    std::vector<BCFamilySolveTempRecord> partial4_records_;
    std::vector<BCFamilySolveTempRecord> partial2_records_;
    std::vector<BCFamilySolveTempRecord> scratch4_records_;

    static void wait_partial4_write(BCFamilySolveStats &stats) {
        (void)stats;
    }

    static void wait_partial2_write(BCFamilySolveStats &stats) {
        (void)stats;
    }

    static void wait_scratch4_write(BCFamilySolveStats &stats) {
        (void)stats;
    }
};

inline void bc_family_accumulate_final_stage_write_stats(
    BCFamilySolveStats &stats,
    const BCFamilySolveStats &local
) {
    stats.final_stage_values_written = bc_checked_add_u64(
        stats.final_stage_values_written,
        local.temp_values_written,
        "BC family final stage value write stats overflow"
    );
    stats.final_stage_bytes_written = bc_checked_add_u64(
        stats.final_stage_bytes_written,
        local.temp_bytes_written,
        "BC family final stage byte write stats overflow"
    );
    stats.final_stage_write_seconds += local.temp_write_seconds;
    bc_success_accumulate_file_stats(&stats.final_stage_write_io, local.temp_write_io);
}

inline void bc_family_accumulate_final_stage_read_stats(
    BCFamilySolveStats &stats,
    const BCFamilySolveStats &local
) {
    stats.final_stage_values_read = bc_checked_add_u64(
        stats.final_stage_values_read,
        local.temp_values_read,
        "BC family final stage value read stats overflow"
    );
    stats.final_stage_bytes_read = bc_checked_add_u64(
        stats.final_stage_bytes_read,
        local.temp_bytes_read,
        "BC family final stage byte read stats overflow"
    );
    stats.final_stage_read_seconds += local.temp_read_seconds;
    bc_success_accumulate_file_stats(&stats.final_stage_read_io, local.temp_read_io);
}

template <typename StorageT>
class BCFamilyFinalValueStage {
public:
    BCFamilyFinalValueStage() = default;

    BCFamilyFinalValueStage(
        const std::filesystem::path &dir,
        uint32_t cell_count,
        bool direct_io,
        uint32_t direct_queue_depth
    ) {
        open(dir, cell_count, direct_io, direct_queue_depth);
    }

    void open(
        const std::filesystem::path &dir,
        uint32_t cell_count,
        bool direct_io,
        uint32_t direct_queue_depth
    ) {
        records_.assign(cell_count, {});
        values_.open(dir / "family_final_values.bcfstmp", 4U, direct_io, direct_queue_depth);
    }

    void write(CellId cid, const BCFamilyValueVector<StorageT> &values, BCFamilySolveStats &stats) {
        require_cid(cid);
        BCFamilySolveTempRecord &record = records_[static_cast<size_t>(cid)];
        if (record.written()) {
            throw std::runtime_error("BC family final stage duplicate record");
        }
        BCFamilySolveStats local;
        record.value_offset = values_.append(values, &local);
        record.value_count = values.size();
        ++stats.final_stage_cells_written;
        bc_family_accumulate_final_stage_write_stats(stats, local);
    }

    void write_batch(
        const std::vector<CellId> &cids,
        const std::vector<const BCFamilyValueVector<StorageT> *> &values,
        BCFamilySolveStats &stats
    ) {
        if (cids.size() != values.size()) {
            throw std::invalid_argument("BC family final stage batch size mismatch");
        }
        if (cids.empty()) {
            return;
        }
        for (size_t i = 0U; i < cids.size(); ++i) {
            require_cid(cids[i]);
            if (values[i] == nullptr) {
                throw std::invalid_argument("BC family final stage batch values are null");
            }
            if (records_[static_cast<size_t>(cids[i])].written()) {
                throw std::runtime_error("BC family final stage duplicate batch record");
            }
        }
        BCFamilySolveStats local;
        std::vector<BCFamilySolveTempRecord> appended = values_.append_many(values, &local);
        if (appended.size() != cids.size()) {
            throw std::runtime_error("BC family final stage append_many returned unexpected count");
        }
        for (size_t i = 0U; i < cids.size(); ++i) {
            records_[static_cast<size_t>(cids[i])] = appended[i];
        }
        stats.final_stage_cells_written += cids.size();
        bc_family_accumulate_final_stage_write_stats(stats, local);
    }

    [[nodiscard]] bool written(CellId cid) const {
        require_cid(cid);
        return records_[static_cast<size_t>(cid)].written();
    }

    [[nodiscard]] uint64_t value_count(CellId cid) const {
        require_cid(cid);
        const BCFamilySolveTempRecord &record = records_[static_cast<size_t>(cid)];
        return record.written() ? record.value_count : 0U;
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> read(CellId cid, BCFamilySolveStats &stats) {
        require_cid(cid);
        const BCFamilySolveTempRecord &record = records_[static_cast<size_t>(cid)];
        if (!record.written()) {
            return {};
        }
        BCFamilySolveStats local;
        BCFamilyValueVector<StorageT> out = values_.read(record.value_offset, record.value_count, &local);
        ++stats.final_stage_cells_read;
        bc_family_accumulate_final_stage_read_stats(stats, local);
        return out;
    }

    void read_batch(
        const std::vector<CellId> &cids,
        std::vector<BCFamilyValueVector<StorageT>> &out,
        BCFamilySolveStats &stats
    ) {
        for (CellId cid : cids) {
            require_cid(cid);
            if (!records_[static_cast<size_t>(cid)].written()) {
                throw std::runtime_error("BC family final stage missing batch record");
            }
        }
        BCFamilySolveStats local;
        values_.read_many_records(records_, cids, out, &local);
        stats.final_stage_cells_read += cids.size();
        bc_family_accumulate_final_stage_read_stats(stats, local);
    }

    void close() {
        values_.close();
    }

    void cleanup() {
        close();
        std::error_code ec;
        std::filesystem::remove(values_.path(), ec);
    }

private:
    void require_cid(CellId cid) const {
        if (cid >= records_.size()) {
            throw std::out_of_range("BC family final stage cid out of range");
        }
    }

    BCFamilySolveTempValueFile<StorageT> values_;
    std::vector<BCFamilySolveTempRecord> records_;
};

template <typename StorageT>
struct BCFamilyPendingOutputCell {
    bool ready = false;
    bool metadata_written = false;
    BCLoadedCell source_cell;
    FinalizedCellPayload payload;
    BCFamilyValueVector<StorageT> values;
};

template <typename StorageT>
struct BCFamilyCompactedOutputCell {
    bool ready = false;
    CellId cid = 0U;
    FinalizedCellPayload payload;
    BCSingleChunkValueBuffer<StorageT> values;
    BCResidentCompactStats compact_stats;
};

template <typename StorageT>
void bc_family_release_compacted_output_values(
    std::vector<BCFamilyCompactedOutputCell<StorageT>> &compacted,
    int threads
) {
    const size_t count = compacted.size();
    if (count >= 8U && threads > 1) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            compacted[static_cast<size_t>(i)].values.reset();
        }
    } else {
        for (BCFamilyCompactedOutputCell<StorageT> &cell : compacted) {
            cell.values.reset();
        }
    }
}

template <typename StorageT>
void bc_family_update_pending_stats(
    const std::vector<BCFamilyPendingOutputCell<StorageT>> &pending,
    BCFamilySolveStats &stats
) {
    uint64_t active = 0U;
    for (const BCFamilyPendingOutputCell<StorageT> &cell : pending) {
        if (cell.ready) {
            ++active;
        }
    }
    stats.pending_cells = active;
    stats.pending_cells_max = std::max(stats.pending_cells_max, active);
}

template <typename StorageT>
void bc_family_compact_dense_cell(
    const BCLut &lut,
    const BCLoadedCell &cell,
    const BCFamilyValueVector<StorageT> &dense_values,
    uint32_t row_width,
    StorageT zero_value,
    BCFamilyCompactedOutputCell<StorageT> &out
) {
    const uint64_t expected_values =
        static_cast<uint64_t>(cell.success_rows) * static_cast<uint64_t>(row_width);
    if (dense_values.size() != expected_values) {
        throw std::runtime_error("BC family solve final dense value count mismatch");
    }
    if (expected_values > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC family solve final dense values exceed size_t");
    }

    out = {};
    out.ready = true;
    out.cid = cell.cid;
    out.values.resize_uninitialized(static_cast<size_t>(expected_values));
    for (size_t i = 0U; i < dense_values.size(); ++i) {
        out.values[i] = dense_values[i];
    }

    bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
        lut,
        cell,
        out.values,
        0U,
        row_width,
        zero_value,
        out.payload,
        out.compact_stats
    );

    const uint64_t compact_count =
        static_cast<uint64_t>(out.payload.success_rows) * static_cast<uint64_t>(row_width);
    if (compact_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC family solve compact value count exceeds size_t");
    }
    out.values.resize_uninitialized(static_cast<size_t>(compact_count));
}

template <typename StorageT>
void bc_family_flush_ready_outputs(
    std::vector<BCFamilyPendingOutputCell<StorageT>> &pending,
    uint64_t &pending_value_bytes,
    CellId &next_output_cid,
    BCSingleChunkFinalFileStreamer<StorageT> &streamer,
    BCFamilySolveStats &stats
) {
    (void)streamer;
    (void)stats;
    pending_value_bytes = 0U;
    next_output_cid = checked_u32_size(pending.size(), "BC family pending cell count exceeds uint32");
    bc_family_update_pending_stats(pending, stats);
}

template <typename StorageT>
void bc_family_mark_compacted_outputs_ready(
    std::vector<BCFamilyCompactedOutputCell<StorageT>> &compacted,
    std::vector<BCFamilyPendingOutputCell<StorageT>> &pending,
    uint64_t &pending_value_bytes,
    uint64_t pending_value_memory_cap_bytes,
    CellId next_output_cid,
    BCSingleChunkFinalFileStreamer<StorageT> &streamer,
    BCFamilySolveStats &stats
) {
    const double t0 = bc_single_chunk_now_seconds();
    const double position_write_before = stats.single.position_write_seconds;
    const double success_write_before = stats.single.success_write_seconds;
    const double assembly_before = stats.single.result_assembly_seconds;
    (void)pending;
    (void)pending_value_bytes;
    (void)pending_value_memory_cap_bytes;
    (void)next_output_cid;

    for (BCFamilyCompactedOutputCell<StorageT> &cell : compacted) {
        if (!cell.ready) {
            continue;
        }
        streamer.write_cell_metadata(cell.cid, cell.payload);
        if (cell.payload.success_rows != 0U || !cell.payload.buckets.empty()) {
            streamer.write_success_cell_values(cell.cid, cell.values);
        }
        ++stats.finalized_cells;
    }
    const double known_delta =
        (stats.single.position_write_seconds - position_write_before) +
        (stats.single.success_write_seconds - success_write_before) +
        (stats.single.result_assembly_seconds - assembly_before);
    stats.pending_mark_seconds +=
        bc_family_positive_remainder(bc_single_chunk_now_seconds() - t0, known_delta);
}

template <typename StorageT>
void bc_family_mark_empty_outputs(
    const BCPositionStreamingReader &current_position,
    std::vector<BCFamilyPendingOutputCell<StorageT>> &pending,
    BCSingleChunkFinalFileStreamer<StorageT> &streamer,
    BCFamilySolveStats &stats
) {
    for (CellId cid = 0U; cid < current_position.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
        if (!desc.empty() || desc.success_rows != 0U) {
            continue;
        }
        FinalizedCellPayload payload;
        streamer.write_cell_metadata(cid, payload);
    }
    (void)pending;
    bc_family_update_pending_stats(pending, stats);
}

template <typename StorageT>
void bc_family_compact_and_mark_ready(
    const BCLut &lut,
    const BCLoadedCell &cell,
    const BCFamilyValueVector<StorageT> &dense_values,
    uint32_t row_width,
    StorageT zero_value,
    BCFamilySolveWorkspace<StorageT> &workspace,
    std::vector<BCFamilyPendingOutputCell<StorageT>> &pending,
    BCFamilySolveStats &stats
) {
    const uint64_t expected_values =
        static_cast<uint64_t>(cell.success_rows) * static_cast<uint64_t>(row_width);
    if (dense_values.size() != expected_values) {
        throw std::runtime_error("BC family solve final dense value count mismatch");
    }
    if (expected_values > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC family solve final dense values exceed size_t");
    }
    double t0 = bc_single_chunk_now_seconds();
    workspace.compact_buffer.resize_uninitialized(static_cast<size_t>(expected_values));
    for (size_t i = 0U; i < dense_values.size(); ++i) {
        workspace.compact_buffer[i] = dense_values[i];
    }
    stats.final_dense_copy_seconds += bc_single_chunk_now_seconds() - t0;

    FinalizedCellPayload payload;
    BCResidentCompactStats compact_stats;
    t0 = bc_single_chunk_now_seconds();
    bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
        lut,
        cell,
        workspace.compact_buffer,
        0U,
        row_width,
        zero_value,
        payload,
        compact_stats
    );
    stats.single.compact_seconds += bc_single_chunk_now_seconds() - t0;
    stats.single.compact_input_rows += compact_stats.input_rows;
    stats.single.compact_live_rows += compact_stats.live_rows;
    stats.single.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
    stats.single.compact_live_cells += compact_stats.live_cells;
    stats.single.compact_empty_cells += compact_stats.empty_cells;

    BCFamilyValueVector<StorageT> compact_values;
    const uint64_t compact_count =
        static_cast<uint64_t>(payload.success_rows) * static_cast<uint64_t>(row_width);
    if (compact_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC family solve compact value count exceeds size_t");
    }
    t0 = bc_single_chunk_now_seconds();
    compact_values.reserve(static_cast<size_t>(compact_count));
    for (uint64_t i = 0U; i < compact_count; ++i) {
        compact_values.push_back(workspace.compact_buffer[static_cast<size_t>(i)]);
    }
    stats.compact_value_copy_seconds += bc_single_chunk_now_seconds() - t0;

    if (cell.cid >= pending.size()) {
        throw std::out_of_range("BC family solve pending cid out of range");
    }
    t0 = bc_single_chunk_now_seconds();
    BCFamilyPendingOutputCell<StorageT> &slot = pending[static_cast<size_t>(cell.cid)];
    if (slot.ready) {
        throw std::runtime_error("BC family solve duplicate finalized output cell");
    }
    slot.ready = true;
    slot.source_cell.cid = cell.cid;
    slot.payload = std::move(payload);
    slot.values = std::move(compact_values);
    ++stats.finalized_cells;
    bc_family_update_pending_stats(pending, stats);
    stats.pending_mark_seconds += bc_single_chunk_now_seconds() - t0;
}

template <typename StorageT>
void bc_family_merge_best_for_summary(
    const BCSolveBoardQuerySummary &summary,
    uint32_t row_width,
    const BCFamilyValueVector<StorageT> &best,
    BCFamilyValueVector<StorageT> &merged
) {
    if (best.size() != merged.size() ||
        best.size() != static_cast<size_t>(kBCBoardCellCount) * row_width) {
        throw std::invalid_argument("BC family solve best buffer size mismatch");
    }
    if (summary.terminal_success || summary.empty_count == 0U) {
        return;
    }
    uint32_t mask = summary.empty_mask;
    while (mask != 0U) {
        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            StorageT &dst = merged[static_cast<size_t>(cell) * row_width + lane];
            const StorageT src = best[static_cast<size_t>(cell) * row_width + lane];
            if (src > dst) {
                dst = src;
            }
        }
    }
}

template <typename StorageT>
void bc_family_merge_compact_best(
    uint32_t empty_count,
    uint32_t row_width,
    const StorageT *best_by_empty_slot_lane,
    StorageT *merged_by_empty_slot_lane
) {
    if (empty_count == 0U) {
        return;
    }
    if (best_by_empty_slot_lane == nullptr || merged_by_empty_slot_lane == nullptr) {
        throw std::invalid_argument("BC family solve compact merge pointer is null");
    }
    const size_t value_count =
        static_cast<size_t>(empty_count) * static_cast<size_t>(row_width);
    for (size_t i = 0U; i < value_count; ++i) {
        if (best_by_empty_slot_lane[i] > merged_by_empty_slot_lane[i]) {
            merged_by_empty_slot_lane[i] = best_by_empty_slot_lane[i];
        }
    }
}

template <typename StorageT>
[[nodiscard]] uint64_t bc_family_sum_compact_best_u64(
    const StorageT *best_by_empty_slot_lane,
    uint32_t empty_count
) {
    static_assert(
        std::is_integral_v<StorageT>,
        "BC family compact integer sum requires integral storage"
    );
    if (empty_count == 0U) {
        return 0U;
    }
    if (best_by_empty_slot_lane == nullptr) {
        throw std::invalid_argument("BC family compact best pointer is null");
    }
    uint64_t sum = 0U;
    for (uint32_t i = 0U; i < empty_count; ++i) {
        sum += static_cast<uint64_t>(best_by_empty_slot_lane[i]);
    }
    return sum;
}

template <typename StorageT>
[[nodiscard]] uint64_t bc_family_sum_merged_compact_best_u64(
    const StorageT *lhs_by_empty_slot,
    const StorageT *rhs_by_empty_slot,
    uint32_t empty_count
) {
    static_assert(
        std::is_integral_v<StorageT>,
        "BC family compact integer max sum requires integral storage"
    );
    if (empty_count == 0U) {
        return 0U;
    }
    if (lhs_by_empty_slot == nullptr || rhs_by_empty_slot == nullptr) {
        throw std::invalid_argument("BC family compact merge pointer is null");
    }
    uint64_t sum = 0U;
    for (uint32_t i = 0U; i < empty_count; ++i) {
        const StorageT value =
            lhs_by_empty_slot[i] > rhs_by_empty_slot[i]
                ? lhs_by_empty_slot[i]
                : rhs_by_empty_slot[i];
        sum += static_cast<uint64_t>(value);
    }
    return sum;
}

inline void bc_family_build_cell_range_work(
    const BCLut &lut,
    const BCLoadedCellView &cell,
    uint32_t words_per_item,
    std::vector<BCFamilySolveCellRangeWork> &out
) {
    if (words_per_item == 0U) {
        throw std::invalid_argument("BC family solve words_per_item must be non-zero");
    }
    out.clear();
    uint32_t grouped_begin = 0U;
    uint32_t grouped_words = 0U;
    auto flush_group = [&](uint32_t end_bucket) {
        if (grouped_words == 0U || grouped_begin == end_bucket) {
            return;
        }
        out.push_back(BCFamilySolveCellRangeWork{grouped_begin, end_bucket, 0U, 0U});
        grouped_words = 0U;
        grouped_begin = end_bucket;
    };
    for (uint32_t bucket_i = 0U; bucket_i < cell.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = cell.buckets.data[bucket_i];
        const BCBucketRankDecoder decoder(lut, bucket.key);
        const uint32_t word_count = words_for_bits(decoder.bitmap_len);
        if (word_count >= words_per_item) {
            flush_group(bucket_i);
            for (uint32_t begin = 0U; begin < word_count; begin += words_per_item) {
                out.push_back(BCFamilySolveCellRangeWork{
                    bucket_i,
                    bucket_i + 1U,
                    begin,
                    std::min<uint32_t>(word_count, begin + words_per_item)
                });
            }
            grouped_begin = bucket_i + 1U;
            continue;
        }
        if (grouped_words == 0U) {
            grouped_begin = bucket_i;
        } else if (grouped_words + word_count > words_per_item) {
            flush_group(bucket_i);
            grouped_begin = bucket_i;
        }
        grouped_words += word_count;
    }
    flush_group(cell.buckets.size);
}

[[nodiscard]] inline std::vector<BCFamilySolveCellRangeWork> bc_family_build_cell_range_work(
    const BCLut &lut,
    const BCLoadedCellView &cell,
    uint32_t words_per_item
) {
    std::vector<BCFamilySolveCellRangeWork> out;
    bc_family_build_cell_range_work(lut, cell, words_per_item, out);
    return out;
}

inline void bc_family_append_pass_cell_range_work(
    const BCLut &lut,
    const BCLoadedCellView &cell,
    size_t cell_index,
    uint32_t words_per_item,
    std::vector<BCSingleChunkLoadedWorkItem> &out
) {
    if (words_per_item == 0U) {
        throw std::invalid_argument("BC family solve words_per_item must be non-zero");
    }
    uint32_t grouped_begin = 0U;
    uint32_t grouped_words = 0U;
    auto flush_group = [&](uint32_t end_bucket) {
        if (grouped_words == 0U || grouped_begin == end_bucket) {
            return;
        }
        out.push_back(BCSingleChunkLoadedWorkItem{
            cell_index,
            grouped_begin,
            end_bucket,
            0U,
            0U
        });
        grouped_words = 0U;
        grouped_begin = end_bucket;
    };
    for (uint32_t bucket_i = 0U; bucket_i < cell.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = cell.buckets.data[bucket_i];
        const BCBucketRankDecoder decoder(lut, bucket.key);
        const uint32_t word_count = words_for_bits(decoder.bitmap_len);
        if (word_count >= words_per_item) {
            flush_group(bucket_i);
            for (uint32_t begin = 0U; begin < word_count; begin += words_per_item) {
                out.push_back(BCSingleChunkLoadedWorkItem{
                    cell_index,
                    bucket_i,
                    bucket_i + 1U,
                    begin,
                    std::min<uint32_t>(word_count, begin + words_per_item)
                });
            }
            grouped_begin = bucket_i + 1U;
            continue;
        }
        if (grouped_words == 0U) {
            grouped_begin = bucket_i;
        } else if (grouped_words + word_count > words_per_item) {
            flush_group(bucket_i);
            grouped_begin = bucket_i;
        }
        grouped_words += word_count;
    }
    flush_group(cell.buckets.size);
}

template <typename StorageT>
void bc_family_prepare_cell_workspace(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    uint32_t row_width,
    StorageT zero_value
) {
    (void)row_width;
    (void)zero_value;
    workspace.merged_best.clear();
    workspace.edge_workspace.clear_queries();
    workspace.stats = {};
    workspace.spawn4_batch_candidate_seconds = 0.0;
    workspace.spawn4_batch_canonical_seconds = 0.0;
    workspace.spawn4_batch_setup_seconds = 0.0;
    workspace.spawn4_batch_reduce_seconds = 0.0;
    workspace.spawn4_batch_emit_seconds = 0.0;
    workspace.spawn2_batch_candidate_seconds = 0.0;
    workspace.spawn2_batch_canonical_seconds = 0.0;
    workspace.spawn2_batch_setup_seconds = 0.0;
    workspace.spawn2_batch_reduce_seconds = 0.0;
    workspace.spawn2_batch_emit_seconds = 0.0;
    workspace.spawn4_batch_canonical_candidates = 0U;
    workspace.spawn4_batch_encoded_queries = 0U;
    workspace.spawn4_batch_reduce_found = 0U;
    workspace.spawn4_batch_entry_misses = 0U;
    workspace.spawn4_batch_bitmap_misses = 0U;
    workspace.spawn2_batch_canonical_candidates = 0U;
    workspace.spawn2_batch_encoded_queries = 0U;
    workspace.spawn2_batch_reduce_found = 0U;
    workspace.spawn2_batch_entry_misses = 0U;
    workspace.spawn2_batch_bitmap_misses = 0U;
}

template <typename StorageT>
void bc_family_accumulate_cell_workspace_batch_stats(
    BCFamilySolveStats &stats,
    const BCFamilySolveCellWorkspace<StorageT> &workspace
) {
    stats.spawn4_batch_candidate_thread_seconds += workspace.spawn4_batch_candidate_seconds;
    stats.spawn4_batch_canonical_thread_seconds += workspace.spawn4_batch_canonical_seconds;
    stats.spawn4_batch_setup_thread_seconds += workspace.spawn4_batch_setup_seconds;
    stats.spawn4_batch_reduce_thread_seconds += workspace.spawn4_batch_reduce_seconds;
    stats.spawn4_batch_emit_thread_seconds += workspace.spawn4_batch_emit_seconds;
    stats.spawn2_batch_candidate_thread_seconds += workspace.spawn2_batch_candidate_seconds;
    stats.spawn2_batch_canonical_thread_seconds += workspace.spawn2_batch_canonical_seconds;
    stats.spawn2_batch_setup_thread_seconds += workspace.spawn2_batch_setup_seconds;
    stats.spawn2_batch_reduce_thread_seconds += workspace.spawn2_batch_reduce_seconds;
    stats.spawn2_batch_emit_thread_seconds += workspace.spawn2_batch_emit_seconds;
    stats.spawn4_batch_canonical_candidates += workspace.spawn4_batch_canonical_candidates;
    stats.spawn4_batch_encoded_queries += workspace.spawn4_batch_encoded_queries;
    stats.spawn4_batch_reduce_found += workspace.spawn4_batch_reduce_found;
    stats.spawn4_batch_entry_misses += workspace.spawn4_batch_entry_misses;
    stats.spawn4_batch_bitmap_misses += workspace.spawn4_batch_bitmap_misses;
    stats.spawn2_batch_canonical_candidates += workspace.spawn2_batch_canonical_candidates;
    stats.spawn2_batch_encoded_queries += workspace.spawn2_batch_encoded_queries;
    stats.spawn2_batch_reduce_found += workspace.spawn2_batch_reduce_found;
    stats.spawn2_batch_entry_misses += workspace.spawn2_batch_entry_misses;
    stats.spawn2_batch_bitmap_misses += workspace.spawn2_batch_bitmap_misses;
}

[[nodiscard]] inline bool bc_family_can_encode_trusted_unit2_modulo(
    const BCSolvePreparedQueryEncoder &encoder
) noexcept {
    return encoder.lut != nullptr &&
        encoder.family_unit == 2U &&
        encoder.contiguous &&
        encoder.axis_base == 0U &&
        encoder.cell_modulus != 0U &&
        encoder.cell_modulus == encoder.family_count &&
        encoder.family_count != 0U &&
        static_cast<uint64_t>(encoder.family_count) * encoder.family_count <=
            static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1ULL;
}

inline void bc_family_encode_trusted_unit2_modulo(
    const BCSolvePreparedQueryEncoder &encoder,
    const BCQuadrantWords &q,
    uint16_t ref,
    BCSolvePreparedQuery &out
) {
    const BCLut &lut = *encoder.lut;
    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    const uint64_t nw_sum = nw_desc.sum;
    const uint64_t ne_sum = ne_desc.sum;
    const uint64_t sw_sum = sw_desc.sum;
    const uint64_t top_sum = nw_sum + ne_sum;
    const uint64_t left_sum = nw_sum + sw_sum;
    const uint64_t bottom_sum = encoder.layer_sum - top_sum;
    const uint64_t right_sum = encoder.layer_sum - left_sum;
    const uint64_t row_min = top_sum < bottom_sum ? top_sum : bottom_sum;
    const uint64_t col_min = left_sum < right_sum ? left_sum : right_sum;
    const uint32_t row_id =
        static_cast<uint32_t>(row_min >> 1U) % encoder.cell_modulus;
    const uint32_t col_id =
        static_cast<uint32_t>(col_min >> 1U) % encoder.cell_modulus;
    const uint32_t rank =
        (static_cast<uint32_t>(ne_desc.rank) * static_cast<uint32_t>(sw_desc.group_count) +
         static_cast<uint32_t>(sw_desc.rank)) *
            static_cast<uint32_t>(se_desc.group_count) +
        static_cast<uint32_t>(se_desc.rank);

    out = BCSolvePreparedQuery{
        (static_cast<uint64_t>(q.nw) << 48U) |
            (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
            (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
            static_cast<uint64_t>(se_desc.packed_sum_mask),
        static_cast<uint16_t>(
            static_cast<uint64_t>(row_id) * encoder.family_count + col_id
        ),
        static_cast<BucketRank>(rank),
        ref,
        0U
    };
}

[[nodiscard]] inline uint64_t bc_family_flush_canonical_boards(
    const BCSolvePreparedQueryEncoder &encoder,
    uint8_t spawn_rank,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs,
    std::vector<BCSolvePreparedQuery> &queries,
    int canonical_symm_mode
) {
    if (boards.empty()) {
        return 0U;
    }
    const size_t query_begin = queries.size();
    const bool use_trusted_unit2_modulo =
        bc_family_can_encode_trusted_unit2_modulo(encoder);
    CanonicalBatch::canonicalize_inplace(
        boards.data(),
        boards.size(),
        canonical_symm_mode
    );
    for (size_t i = 0U; i < boards.size(); ++i) {
        BCSolvePreparedQuery query;
        const BCQuadrantWords q = unpack_board_to_quadrants(boards[i]);
        if (use_trusted_unit2_modulo) {
            bc_family_encode_trusted_unit2_modulo(encoder, q, refs[i], query);
            queries.push_back(query);
            continue;
        }
        const bool encoded = encoder.encode(
                q,
                refs[i],
                spawn_rank,
                BCDirectionMask::Both,
                query
            );
        if (encoded) {
            queries.push_back(query);
        }
    }
    const uint64_t encoded_count = static_cast<uint64_t>(queries.size() - query_begin);
    boards.clear();
    refs.clear();
    return encoded_count;
}

template <typename StorageT>
inline void bc_family_record_batch_query_stats(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    bool spawn4_phase,
    uint64_t candidate_count,
    uint64_t encoded_count,
    uint64_t reduce_found,
    uint64_t entry_misses,
    uint64_t bitmap_misses
) {
    if (spawn4_phase) {
        workspace.spawn4_batch_canonical_candidates += candidate_count;
        workspace.spawn4_batch_encoded_queries += encoded_count;
        workspace.spawn4_batch_reduce_found += reduce_found;
        workspace.spawn4_batch_entry_misses += entry_misses;
        workspace.spawn4_batch_bitmap_misses += bitmap_misses;
    } else {
        workspace.spawn2_batch_canonical_candidates += candidate_count;
        workspace.spawn2_batch_encoded_queries += encoded_count;
        workspace.spawn2_batch_reduce_found += reduce_found;
        workspace.spawn2_batch_entry_misses += entry_misses;
        workspace.spawn2_batch_bitmap_misses += bitmap_misses;
    }
}

inline void bc_family_push_candidate_with_axis_hit(
    uint64_t spawned,
    uint64_t moved,
    uint16_t ref,
    bool axis_hit,
    bool filter_enabled,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs,
    BCSolveEdgeStats *stats
) {
    if (moved == spawned) {
        if (stats != nullptr) {
            ++stats->unchanged_moves;
        }
        return;
    }
    if (filter_enabled) {
        if (stats != nullptr) {
            ++stats->prefilter_checks;
        }
        if (!axis_hit) {
            if (stats != nullptr) {
                ++stats->prefilter_skips;
            }
            return;
        }
    }
    boards.push_back(moved);
    refs.push_back(ref);
}

inline void bc_family_push_moved_candidate_no_stats(
    uint64_t spawned,
    uint64_t moved,
    uint16_t ref,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs
) {
    if (moved == spawned) {
        return;
    }
    boards.push_back(moved);
    refs.push_back(ref);
}

[[nodiscard]] inline bool bc_family_is_success_board_fast(
    uint64_t board,
    const BCSolveEdgeOptions &options,
    bool success_check_all_cells,
    uint64_t success_target_pattern
) {
    if (success_check_all_cells) {
        const uint64_t diff = board ^ success_target_pattern;
        constexpr uint64_t kMask7 = 0x7777777777777777ULL;
        return (~(((diff & kMask7) + kMask7) | diff | kMask7)) != 0ULL;
    }
    return bc_solve_is_success_board(board, options);
}

[[nodiscard]] inline BCSolvePhysicalTargetFamilyHits bc_family_bucket_spawn_target_hits_for_cell(
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t nw_sum,
    uint64_t ne_sum,
    uint64_t sw_sum,
    uint64_t se_sum,
    uint32_t cell,
    uint32_t tile_sum,
    uint32_t cell_modulus
) {
    switch (cell) {
    case 15U:
    case 14U:
    case 11U:
    case 10U:
        nw_sum += tile_sum;
        break;
    case 13U:
    case 12U:
    case 9U:
    case 8U:
        ne_sum += tile_sum;
        break;
    case 7U:
    case 6U:
    case 3U:
    case 2U:
        sw_sum += tile_sum;
        break;
    default:
        se_sum += tile_sum;
        break;
    }
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return {};
    }

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!bc_min_side_coord_u64(nw_sum + ne_sum, sw_sum + se_sum, axis.family_unit(), row_coord) ||
        !bc_min_side_coord_u64(nw_sum + sw_sum, ne_sum + se_sum, axis.family_unit(), col_coord)) {
        return {};
    }

    auto coord_to_id = [&](FamilyCoord coord, FamilyId &id_out) {
        uint32_t coord_u32 = coord;
        if (cell_modulus != 0U) {
            coord_u32 %= cell_modulus;
        }
        if (coord_u32 > std::numeric_limits<FamilyCoord>::max()) {
            return false;
        }
        id_out = axis.try_coord_to_id(static_cast<FamilyCoord>(coord_u32));
        return id_out != BCFamilyTable::kInvalidFamilyId;
    };

    FamilyId row_family = BCFamilyTable::kInvalidFamilyId;
    FamilyId col_family = BCFamilyTable::kInvalidFamilyId;
    if (!coord_to_id(row_coord, row_family) || !coord_to_id(col_coord, col_family)) {
        return {};
    }
    return BCSolvePhysicalTargetFamilyHits{
        bc_solve_filter_contains_family(filter, row_family),
        bc_solve_filter_contains_family(filter, col_family)
    };
}

[[nodiscard]] inline bool bc_family_bucket_spawn_target_axis_hit_for_cell(
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t nw_sum,
    uint64_t ne_sum,
    uint64_t sw_sum,
    uint64_t se_sum,
    uint32_t cell,
    uint32_t tile_sum,
    uint32_t cell_modulus,
    BCDirectionMask direction
) {
    switch (cell) {
    case 15U:
    case 14U:
    case 11U:
    case 10U:
        nw_sum += tile_sum;
        break;
    case 13U:
    case 12U:
    case 9U:
    case 8U:
        ne_sum += tile_sum;
        break;
    case 7U:
    case 6U:
    case 3U:
    case 2U:
        sw_sum += tile_sum;
        break;
    default:
        se_sum += tile_sum;
        break;
    }
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return false;
    }

    const uint64_t side_a = direction == BCDirectionMask::Horizontal
        ? nw_sum + ne_sum
        : nw_sum + sw_sum;
    const uint64_t side_b = direction == BCDirectionMask::Horizontal
        ? sw_sum + se_sum
        : ne_sum + se_sum;
    FamilyCoord coord = 0U;
    if (!bc_min_side_coord_u64(side_a, side_b, axis.family_unit(), coord)) {
        return false;
    }

    uint32_t coord_u32 = coord;
    if (cell_modulus != 0U) {
        coord_u32 %= cell_modulus;
    }
    if (coord_u32 > std::numeric_limits<FamilyCoord>::max()) {
        return false;
    }
    const FamilyId family = axis.try_coord_to_id(static_cast<FamilyCoord>(coord_u32));
    if (family == BCFamilyTable::kInvalidFamilyId) {
        return false;
    }
    return bc_solve_filter_contains_family(filter, family);
}

inline void bc_family_build_bucket_spawn_target_hits(
    const BCLut &lut,
    const BCLoadedCellView &cell,
    const BCFamilyTable &future_axis,
    const BCSolveTargetFamilyFilter &filter,
    BCDirectionMask directions,
    uint8_t spawn_rank,
    const BCQuadrantWordSumTable *word_sums,
    uint32_t cell_modulus,
    std::vector<BCFamilyBucketSpawnTargetHits> &hits
) {
    hits.clear();
    if (!filter.enabled) {
        return;
    }
    hits.resize(cell.buckets.size);
    const uint32_t tile_sum = lut.tile_sum_value(spawn_rank);
    const bool use_word_sums = word_sums != nullptr && !word_sums->empty();
    const bool need_horizontal = bc_has_horizontal(directions);
    const bool need_vertical = bc_has_vertical(directions);
    for (uint32_t bucket_i = 0U; bucket_i < cell.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = cell.buckets.data[bucket_i];
        const BCBucketRankDecoder decoder(lut, bucket.key);
        const uint64_t nw_sum = use_word_sums
            ? (*word_sums)[decoder.nw]
            : lut.sum4_value(lut.word_desc(decoder.nw).sum_id);
        const uint64_t ne_sum = lut.sum4_value(decoder.ne_sum_id);
        const uint64_t sw_sum = lut.sum4_value(decoder.sw_sum_id);
        const uint64_t se_sum = lut.sum4_value(decoder.se_sum_id);
        BCFamilyBucketSpawnTargetHits &bucket_hits = hits[bucket_i];
        const uint16_t empty_mask = bc_bucket_empty_mask16(decoder);
        for (size_t quadrant = 0U; quadrant < kBCFamilyQuadrantCellMasks.size(); ++quadrant) {
            const uint16_t quadrant_empty_mask =
                static_cast<uint16_t>(empty_mask & kBCFamilyQuadrantCellMasks[quadrant]);
            if (quadrant_empty_mask == 0U) {
                continue;
            }
            const uint32_t representative_cell =
                kBCFamilyQuadrantRepresentativeCells[quadrant];
            if (need_horizontal && need_vertical) {
                const BCSolvePhysicalTargetFamilyHits cell_hits =
                    bc_family_bucket_spawn_target_hits_for_cell(
                        future_axis,
                        filter,
                        nw_sum,
                        ne_sum,
                        sw_sum,
                        se_sum,
                        representative_cell,
                        tile_sum,
                        cell_modulus
                    );
                if (cell_hits.horizontal) {
                    bucket_hits.horizontal_mask =
                        static_cast<uint16_t>(bucket_hits.horizontal_mask | quadrant_empty_mask);
                }
                if (cell_hits.vertical) {
                    bucket_hits.vertical_mask =
                        static_cast<uint16_t>(bucket_hits.vertical_mask | quadrant_empty_mask);
                }
                continue;
            }
            if (need_horizontal) {
                if (bc_family_bucket_spawn_target_axis_hit_for_cell(
                        future_axis,
                        filter,
                        nw_sum,
                        ne_sum,
                        sw_sum,
                        se_sum,
                        representative_cell,
                        tile_sum,
                        cell_modulus,
                        BCDirectionMask::Horizontal
                    )) {
                    bucket_hits.horizontal_mask =
                        static_cast<uint16_t>(bucket_hits.horizontal_mask | quadrant_empty_mask);
                }
            }
            if (need_vertical) {
                if (bc_family_bucket_spawn_target_axis_hit_for_cell(
                    future_axis,
                    filter,
                    nw_sum,
                    ne_sum,
                    sw_sum,
                    se_sum,
                    representative_cell,
                    tile_sum,
                    cell_modulus,
                    BCDirectionMask::Vertical
                )) {
                    bucket_hits.vertical_mask =
                        static_cast<uint16_t>(bucket_hits.vertical_mask | quadrant_empty_mask);
                }
            }
        }
    }
}

template <typename StorageT, bool RecordCompactRefBoardSlots>
uint32_t bc_family_collect_phase_batch_candidates(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits = nullptr
) {
    static_assert(
        static_cast<size_t>(BCResidentBatchWorkspace<StorageT>::kBatchSize) *
            static_cast<size_t>(kBCBoardCellCount) <=
            static_cast<size_t>(std::numeric_limits<uint16_t>::max()),
        "BC family compact refs require uint16 range"
    );
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    const bool success_check_all_cells =
        success_check_enabled && options.solve.edge_options.success_check_all_cells;
    const uint64_t success_target_pattern =
        success_check_all_cells
            ? static_cast<uint64_t>(options.solve.edge_options.success_target_rank) *
                0x1111111111111111ULL
            : 0U;
    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    BCSolveEdgeStats *edge_stats =
        options.collect_edge_stats ? &workspace.stats.edge : nullptr;
    const bool horizontal = bc_has_horizontal(directions);
    const bool vertical = bc_has_vertical(directions);

    if constexpr (RecordCompactRefBoardSlots) {
        workspace.compact_ref_board_slots.clear();
    }
    uint32_t compact_slot_count = 0U;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = batch.boards[board_slot];
        batch.empty_counts[board_slot] = 0U;
        batch.terminal[board_slot] = 0U;
        workspace.local_compact_offsets[board_slot] = bc_family_checked_u16_u32(
            compact_slot_count,
            "BC family compact offset exceeds uint16"
        );
        if (edge_stats != nullptr) {
            ++edge_stats->source_boards;
        }
        if (success_check_enabled &&
            bc_family_is_success_board_fast(
                board,
                options.solve.edge_options,
                success_check_all_cells,
                success_target_pattern
            )) {
            batch.terminal[board_slot] = 1U;
            if (edge_stats != nullptr) {
                ++edge_stats->terminal_success_boards;
            }
            continue;
        }

        uint32_t empty_mask = batch.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>(compact_slot_count++);
            if constexpr (RecordCompactRefBoardSlots) {
                workspace.compact_ref_board_slots.push_back(static_cast<uint16_t>(board_slot));
            }
            const uint32_t shift = 4U * cell;
            ++batch.empty_counts[board_slot];
            if (edge_stats != nullptr) {
                ++edge_stats->empty_slots;
                ++edge_stats->spawned_boards;
            }

            const uint64_t spawned =
                board | (static_cast<uint64_t>(spawn_rank) << shift);
            if (directions == BCDirectionMask::Both) {
                if (edge_stats != nullptr) {
                    edge_stats->selective_move_calls += 4U;
                }
                BCSolvePhysicalTargetFamilyHits target_hits;
                if (filter.enabled && bucket_target_hits != nullptr) {
                    const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                    const BCFamilyBucketSpawnTargetHits &hits =
                        (*bucket_target_hits)[static_cast<size_t>(bucket_i)];
                    target_hits.horizontal = bc_family_bucket_hit_horizontal(hits, cell);
                    target_hits.vertical = bc_family_bucket_hit_vertical(hits, cell);
                } else {
                    target_hits = bc_solve_spawned_target_family_hits(
                            lut,
                            future_axis,
                            filter,
                            spawned,
                            options.solve.word_sums,
                            options.solve.edge_options.future_cell_modulus
                        );
                }
                if (!filter.enabled || target_hits.horizontal || edge_stats != nullptr) {
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hits.horizontal,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hits.horizontal,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
                if (!filter.enabled || target_hits.vertical || edge_stats != nullptr) {
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hits.vertical,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hits.vertical,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
            } else {
                if (horizontal) {
                    if (edge_stats != nullptr) {
                        edge_stats->selective_move_calls += 2U;
                    }
                    bool target_hit = true;
                    if (filter.enabled && bucket_target_hits != nullptr) {
                        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                        target_hit =
                            bc_family_bucket_hit_horizontal(
                                (*bucket_target_hits)[static_cast<size_t>(bucket_i)],
                                cell
                            );
                    } else {
                        target_hit = bc_solve_spawned_target_family_axis_hit(
                            lut,
                            future_axis,
                            filter,
                            spawned,
                            BCDirectionMask::Horizontal,
                            options.solve.word_sums,
                            options.solve.edge_options.future_cell_modulus
                        );
                    }
                    if (filter.enabled && !target_hit && edge_stats == nullptr) {
                        continue;
                    }
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
                if (vertical) {
                    if (edge_stats != nullptr) {
                        edge_stats->selective_move_calls += 2U;
                    }
                    bool target_hit = true;
                    if (filter.enabled && bucket_target_hits != nullptr) {
                        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                        target_hit =
                            bc_family_bucket_hit_vertical(
                                (*bucket_target_hits)[static_cast<size_t>(bucket_i)],
                                cell
                            );
                    } else {
                        target_hit = bc_solve_spawned_target_family_axis_hit(
                            lut,
                            future_axis,
                            filter,
                            spawned,
                            BCDirectionMask::Vertical,
                            options.solve.word_sums,
                            options.solve.edge_options.future_cell_modulus
                        );
                    }
                    if (filter.enabled && !target_hit && edge_stats == nullptr) {
                        continue;
                    }
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
            }
        }
    }
    return compact_slot_count;
}

template <typename StorageT, class Emit>
void bc_family_flush_phase_batch(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCSolvePreparedQueryEncoder &encoder,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits,
    Emit &&emit
) {
    static_assert(
        static_cast<size_t>(BCResidentBatchWorkspace<StorageT>::kBatchSize) *
            static_cast<size_t>(kBCBoardCellCount) <=
            static_cast<size_t>(std::numeric_limits<uint16_t>::max()),
        "BC family compact refs require uint16 range"
    );
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    if (count == 0U) {
        return;
    }

    batch.canonical2_boards.clear();
    batch.canonical2_refs.clear();
    batch.queries2.clear();

    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;

    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const bool collect_batch_timing = options.collect_batch_timing;
    const double candidate_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint32_t compact_slot_count =
        bc_family_collect_phase_batch_candidates<StorageT, false>(
            workspace,
            lut,
            future_axis,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits
        );
    if (collect_batch_timing) {
        const double candidate_seconds = bc_single_chunk_now_seconds() - candidate_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_candidate_seconds += candidate_seconds;
        } else {
            workspace.spawn2_batch_candidate_seconds += candidate_seconds;
        }
    }

    const double canonical_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint64_t candidate_count =
        static_cast<uint64_t>(batch.canonical2_boards.size());
    const uint64_t encoded_count = bc_family_flush_canonical_boards(
        encoder,
        spawn_rank,
        batch.canonical2_boards,
        batch.canonical2_refs,
        batch.queries2,
        options.solve.edge_options.canonical_symm_mode
    );
    if (collect_batch_timing) {
        const double canonical_seconds = bc_single_chunk_now_seconds() - canonical_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_canonical_seconds += canonical_seconds;
        } else {
            workspace.spawn2_batch_canonical_seconds += canonical_seconds;
        }
    }

    const double setup_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const size_t best_per_lane = static_cast<size_t>(compact_slot_count);
    const size_t best_values =
        best_per_lane * static_cast<size_t>(options.solve.row_width);
    workspace.compact_best_values.resize(best_values);
    std::fill(
        workspace.compact_best_values.begin(),
        workspace.compact_best_values.end(),
        options.solve.zero_value
    );
    if (collect_batch_timing) {
        const double setup_seconds = bc_single_chunk_now_seconds() - setup_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_setup_seconds += setup_seconds;
        } else {
            workspace.spawn2_batch_setup_seconds += setup_seconds;
        }
    }

    const double reduce_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    BCSolveEdgeStats *edge_stats =
        options.collect_edge_stats ? &workspace.stats.edge : nullptr;
    typename BCFutureSuccessLookupView<StorageT>::BatchLookupStats lookup_stats;
    auto *lookup_stats_ptr = collect_batch_timing ? &lookup_stats : nullptr;
    uint64_t reduce_found = 0U;
    if (options.solve.row_width == 1U) {
        reduce_found += future_lookup.reduce_max_queries(
            batch.queries2,
            workspace.compact_best_values.data(),
            best_per_lane,
            0U,
            edge_stats,
            true,
            lookup_stats_ptr
        );
    } else {
        workspace.lane_best_values.resize(best_per_lane);
        for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
            std::fill(
                workspace.lane_best_values.begin(),
                workspace.lane_best_values.end(),
                options.solve.zero_value
            );
            reduce_found += future_lookup.reduce_max_queries(
                batch.queries2,
                workspace.lane_best_values.data(),
                workspace.lane_best_values.size(),
                lane,
                edge_stats,
                true,
                lookup_stats_ptr
            );
            for (size_t ref = 0U; ref < best_per_lane; ++ref) {
                workspace.compact_best_values[ref * options.solve.row_width + lane] =
                    workspace.lane_best_values[ref];
            }
        }
    }
    if (collect_batch_timing) {
        const double reduce_seconds = bc_single_chunk_now_seconds() - reduce_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_reduce_seconds += reduce_seconds;
        } else {
            workspace.spawn2_batch_reduce_seconds += reduce_seconds;
        }
    }
    bc_family_record_batch_query_stats(
        workspace,
        spawn4_phase,
        candidate_count,
        encoded_count,
        reduce_found,
        lookup_stats.entry_misses,
        lookup_stats.bitmap_misses
    );

    const double emit_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        BCSolveBoardQuerySummary summary;
        summary.terminal_success = batch.terminal[board_slot] != 0U;
        summary.empty_mask = batch.empty_masks[board_slot];
        summary.empty_count = batch.empty_counts[board_slot];
        const StorageT *best = summary.empty_count == 0U
            ? nullptr
            : workspace.compact_best_values.data() +
                static_cast<size_t>(workspace.local_compact_offsets[board_slot]) *
                    options.solve.row_width;
        emit(
            workspace.local_success_rows[board_slot],
            workspace.local_bucket_indices[board_slot],
            summary,
            best,
            workspace
        );
    }
    if (collect_batch_timing) {
        const double emit_seconds = bc_single_chunk_now_seconds() - emit_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_emit_seconds += emit_seconds;
        } else {
            workspace.spawn2_batch_emit_seconds += emit_seconds;
        }
    }
    batch.clear_batch();
}

template <typename StorageT, bool RecordCompactRefBoardSlots, bool HorizontalAxis>
uint32_t bc_family_collect_phase_batch_candidates_multi_cell_precomputed_axis(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    BCSolveSpawnPhase phase,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> &bucket_target_hits_by_cell
) {
    static_assert(
        static_cast<size_t>(BCResidentBatchWorkspace<StorageT>::kBatchSize) *
            static_cast<size_t>(kBCBoardCellCount) <=
            static_cast<size_t>(std::numeric_limits<uint16_t>::max()),
        "BC family compact refs require uint16 range"
    );
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    const bool success_check_all_cells =
        success_check_enabled && options.solve.edge_options.success_check_all_cells;
    const uint64_t success_target_pattern =
        success_check_all_cells
            ? static_cast<uint64_t>(options.solve.edge_options.success_target_rank) *
                0x1111111111111111ULL
            : 0U;
    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;

    if constexpr (RecordCompactRefBoardSlots) {
        workspace.compact_ref_board_slots.clear();
    }
    uint32_t compact_slot_count = 0U;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = batch.boards[board_slot];
        batch.empty_counts[board_slot] = 0U;
        batch.terminal[board_slot] = 0U;
        workspace.local_compact_offsets[board_slot] = bc_family_checked_u16_u32(
            compact_slot_count,
            "BC family compact offset exceeds uint16"
        );
        if (success_check_enabled &&
            bc_family_is_success_board_fast(
                board,
                options.solve.edge_options,
                success_check_all_cells,
                success_target_pattern
            )) {
            batch.terminal[board_slot] = 1U;
            continue;
        }

        const uint32_t cell_index = workspace.local_cell_indices[board_slot];
        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
        const BCFamilyBucketSpawnTargetHits &hits =
            bucket_target_hits_by_cell[static_cast<size_t>(cell_index)]
                [static_cast<size_t>(bucket_i)];
        uint32_t empty_mask = batch.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>(compact_slot_count++);
            if constexpr (RecordCompactRefBoardSlots) {
                workspace.compact_ref_board_slots.push_back(static_cast<uint16_t>(board_slot));
            }
            ++batch.empty_counts[board_slot];
            const bool target_hit = HorizontalAxis
                ? bc_family_bucket_hit_horizontal(hits, cell)
                : bc_family_bucket_hit_vertical(hits, cell);
            if (!target_hit) {
                continue;
            }

            const uint64_t spawned =
                board | (static_cast<uint64_t>(spawn_rank) << (4U * cell));
            if constexpr (HorizontalAxis) {
                const auto moved = BoardMover::move_horizontal_pair(spawned);
                bc_family_push_moved_candidate_no_stats(
                    spawned,
                    moved.first,
                    ref,
                    batch.canonical2_boards,
                    batch.canonical2_refs
                );
                bc_family_push_moved_candidate_no_stats(
                    spawned,
                    moved.second,
                    ref,
                    batch.canonical2_boards,
                    batch.canonical2_refs
                );
            } else {
                const auto moved = BoardMover::move_vertical_pair(spawned);
                bc_family_push_moved_candidate_no_stats(
                    spawned,
                    moved.first,
                    ref,
                    batch.canonical2_boards,
                    batch.canonical2_refs
                );
                bc_family_push_moved_candidate_no_stats(
                    spawned,
                    moved.second,
                    ref,
                    batch.canonical2_boards,
                    batch.canonical2_refs
                );
            }
        }
    }
    return compact_slot_count;
}

template <typename StorageT, bool RecordCompactRefBoardSlots>
uint32_t bc_family_collect_phase_batch_candidates_multi_cell(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> *bucket_target_hits_by_cell
) {
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    const bool success_check_all_cells =
        success_check_enabled && options.solve.edge_options.success_check_all_cells;
    const uint64_t success_target_pattern =
        success_check_all_cells
            ? static_cast<uint64_t>(options.solve.edge_options.success_target_rank) *
                0x1111111111111111ULL
            : 0U;
    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    BCSolveEdgeStats *edge_stats =
        options.collect_edge_stats ? &workspace.stats.edge : nullptr;
    const bool horizontal = bc_has_horizontal(directions);
    const bool vertical = bc_has_vertical(directions);
    const bool fast_precomputed_hits =
        edge_stats == nullptr && filter.enabled && bucket_target_hits_by_cell != nullptr;
    if (fast_precomputed_hits && directions == BCDirectionMask::Horizontal) {
        return bc_family_collect_phase_batch_candidates_multi_cell_precomputed_axis<
            StorageT,
            RecordCompactRefBoardSlots,
            true
        >(workspace, phase, options, *bucket_target_hits_by_cell);
    }
    if (fast_precomputed_hits && directions == BCDirectionMask::Vertical) {
        return bc_family_collect_phase_batch_candidates_multi_cell_precomputed_axis<
            StorageT,
            RecordCompactRefBoardSlots,
            false
        >(workspace, phase, options, *bucket_target_hits_by_cell);
    }

    if constexpr (RecordCompactRefBoardSlots) {
        workspace.compact_ref_board_slots.clear();
    }
    uint32_t compact_slot_count = 0U;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = batch.boards[board_slot];
        batch.empty_counts[board_slot] = 0U;
        batch.terminal[board_slot] = 0U;
        workspace.local_compact_offsets[board_slot] = bc_family_checked_u16_u32(
            compact_slot_count,
            "BC family compact offset exceeds uint16"
        );
        if (edge_stats != nullptr) {
            ++edge_stats->source_boards;
        }
        if (success_check_enabled &&
            bc_family_is_success_board_fast(
                board,
                options.solve.edge_options,
                success_check_all_cells,
                success_target_pattern
            )) {
            batch.terminal[board_slot] = 1U;
            if (edge_stats != nullptr) {
                ++edge_stats->terminal_success_boards;
            }
            continue;
        }

        const uint32_t cell_index = workspace.local_cell_indices[board_slot];
        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
        const BCFamilyBucketSpawnTargetHits *precomputed_hits = nullptr;
        if (fast_precomputed_hits) {
            precomputed_hits =
                &(*bucket_target_hits_by_cell)[static_cast<size_t>(cell_index)]
                    [static_cast<size_t>(bucket_i)];
        }

        uint32_t empty_mask = batch.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>(compact_slot_count++);
            if constexpr (RecordCompactRefBoardSlots) {
                workspace.compact_ref_board_slots.push_back(static_cast<uint16_t>(board_slot));
            }
            const uint32_t shift = 4U * cell;
            ++batch.empty_counts[board_slot];
            if (edge_stats != nullptr) {
                ++edge_stats->empty_slots;
                ++edge_stats->spawned_boards;
            }

            const uint64_t spawned =
                board | (static_cast<uint64_t>(spawn_rank) << shift);
            if (precomputed_hits != nullptr) {
                const BCFamilyBucketSpawnTargetHits &hits = *precomputed_hits;
                if (directions == BCDirectionMask::Both) {
                    if (bc_family_bucket_hit_horizontal(hits, cell)) {
                        const auto moved = BoardMover::move_horizontal_pair(spawned);
                        bc_family_push_moved_candidate_no_stats(
                            spawned, moved.first, ref, batch.canonical2_boards, batch.canonical2_refs);
                        bc_family_push_moved_candidate_no_stats(
                            spawned, moved.second, ref, batch.canonical2_boards, batch.canonical2_refs);
                    }
                    if (bc_family_bucket_hit_vertical(hits, cell)) {
                        const auto moved = BoardMover::move_vertical_pair(spawned);
                        bc_family_push_moved_candidate_no_stats(
                            spawned, moved.first, ref, batch.canonical2_boards, batch.canonical2_refs);
                        bc_family_push_moved_candidate_no_stats(
                            spawned, moved.second, ref, batch.canonical2_boards, batch.canonical2_refs);
                    }
                    continue;
                }
                if (horizontal) {
                    if (!bc_family_bucket_hit_horizontal(hits, cell)) {
                        continue;
                    }
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_family_push_moved_candidate_no_stats(
                        spawned, moved.first, ref, batch.canonical2_boards, batch.canonical2_refs);
                    bc_family_push_moved_candidate_no_stats(
                        spawned, moved.second, ref, batch.canonical2_boards, batch.canonical2_refs);
                    continue;
                }
                if (vertical) {
                    if (!bc_family_bucket_hit_vertical(hits, cell)) {
                        continue;
                    }
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_family_push_moved_candidate_no_stats(
                        spawned, moved.first, ref, batch.canonical2_boards, batch.canonical2_refs);
                    bc_family_push_moved_candidate_no_stats(
                        spawned, moved.second, ref, batch.canonical2_boards, batch.canonical2_refs);
                    continue;
                }
            }
            if (directions == BCDirectionMask::Both) {
                if (edge_stats != nullptr) {
                    edge_stats->selective_move_calls += 4U;
                }
                BCSolvePhysicalTargetFamilyHits target_hits;
                if (filter.enabled && bucket_target_hits_by_cell != nullptr) {
                    const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                    const BCFamilyBucketSpawnTargetHits &hits =
                        (*bucket_target_hits_by_cell)[static_cast<size_t>(cell_index)]
                            [static_cast<size_t>(bucket_i)];
                    target_hits.horizontal = bc_family_bucket_hit_horizontal(hits, cell);
                    target_hits.vertical = bc_family_bucket_hit_vertical(hits, cell);
                } else {
                    target_hits = bc_solve_spawned_target_family_hits(
                        lut,
                        future_axis,
                        filter,
                        spawned,
                        options.solve.word_sums,
                        options.solve.edge_options.future_cell_modulus
                    );
                }
                if (!filter.enabled || target_hits.horizontal || edge_stats != nullptr) {
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hits.horizontal,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hits.horizontal,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
                if (!filter.enabled || target_hits.vertical || edge_stats != nullptr) {
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hits.vertical,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hits.vertical,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
            } else {
                if (horizontal) {
                    if (edge_stats != nullptr) {
                        edge_stats->selective_move_calls += 2U;
                    }
                    bool target_hit = true;
                    if (filter.enabled && bucket_target_hits_by_cell != nullptr) {
                        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                        target_hit =
                            bc_family_bucket_hit_horizontal(
                                (*bucket_target_hits_by_cell)[static_cast<size_t>(cell_index)]
                                    [static_cast<size_t>(bucket_i)],
                                cell
                            );
                    } else {
                        target_hit = bc_solve_spawned_target_family_axis_hit(
                            lut,
                            future_axis,
                            filter,
                            spawned,
                            BCDirectionMask::Horizontal,
                            options.solve.word_sums,
                            options.solve.edge_options.future_cell_modulus
                        );
                    }
                    if (filter.enabled && !target_hit && edge_stats == nullptr) {
                        continue;
                    }
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
                if (vertical) {
                    if (edge_stats != nullptr) {
                        edge_stats->selective_move_calls += 2U;
                    }
                    bool target_hit = true;
                    if (filter.enabled && bucket_target_hits_by_cell != nullptr) {
                        const uint32_t bucket_i = workspace.local_bucket_indices[board_slot];
                        target_hit =
                            bc_family_bucket_hit_vertical(
                                (*bucket_target_hits_by_cell)[static_cast<size_t>(cell_index)]
                                    [static_cast<size_t>(bucket_i)],
                                cell
                            );
                    } else {
                        target_hit = bc_solve_spawned_target_family_axis_hit(
                            lut,
                            future_axis,
                            filter,
                            spawned,
                            BCDirectionMask::Vertical,
                            options.solve.word_sums,
                            options.solve.edge_options.future_cell_modulus
                        );
                    }
                    if (filter.enabled && !target_hit && edge_stats == nullptr) {
                        continue;
                    }
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.first,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                    bc_family_push_candidate_with_axis_hit(
                        spawned,
                        moved.second,
                        ref,
                        target_hit,
                        filter.enabled,
                        batch.canonical2_boards,
                        batch.canonical2_refs,
                        edge_stats
                    );
                }
            }
        }
    }
    return compact_slot_count;
}

template <typename StorageT>
void bc_family_flush_first_direction_batch_multi_cell(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCSolvePreparedQueryEncoder &encoder,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<BCFamilyPartialCellLayout> &layouts,
    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> &partials,
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> *bucket_target_hits_by_cell
) {
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    if (count == 0U) {
        return;
    }

    batch.canonical2_boards.clear();
    batch.canonical2_refs.clear();
    batch.queries2.clear();

    const bool collect_batch_timing = options.collect_batch_timing;
    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const double candidate_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint32_t compact_slot_count =
        bc_family_collect_phase_batch_candidates_multi_cell<StorageT, false>(
            workspace,
            lut,
            future_axis,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits_by_cell
        );
    if (collect_batch_timing) {
        const double candidate_seconds = bc_single_chunk_now_seconds() - candidate_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_candidate_seconds += candidate_seconds;
        } else {
            workspace.spawn2_batch_candidate_seconds += candidate_seconds;
        }
    }

    const double canonical_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint8_t spawn_rank = spawn4_phase
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    const uint64_t candidate_count =
        static_cast<uint64_t>(batch.canonical2_boards.size());
    const uint64_t encoded_count = bc_family_flush_canonical_boards(
        encoder,
        spawn_rank,
        batch.canonical2_boards,
        batch.canonical2_refs,
        batch.queries2,
        options.solve.edge_options.canonical_symm_mode
    );
    if (collect_batch_timing) {
        const double canonical_seconds = bc_single_chunk_now_seconds() - canonical_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_canonical_seconds += canonical_seconds;
        } else {
            workspace.spawn2_batch_canonical_seconds += canonical_seconds;
        }
    }

    const double setup_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const size_t best_per_lane = static_cast<size_t>(compact_slot_count);
    const size_t best_values =
        best_per_lane * static_cast<size_t>(options.solve.row_width);
    workspace.compact_best_values.resize(best_values);
    std::fill(
        workspace.compact_best_values.begin(),
        workspace.compact_best_values.end(),
        options.solve.zero_value
    );
    if (collect_batch_timing) {
        const double setup_seconds = bc_single_chunk_now_seconds() - setup_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_setup_seconds += setup_seconds;
        } else {
            workspace.spawn2_batch_setup_seconds += setup_seconds;
        }
    }

    const double reduce_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    BCSolveEdgeStats *edge_stats =
        options.collect_edge_stats ? &workspace.stats.edge : nullptr;
    typename BCFutureSuccessLookupView<StorageT>::BatchLookupStats lookup_stats;
    auto *lookup_stats_ptr = collect_batch_timing ? &lookup_stats : nullptr;
    uint64_t reduce_found = 0U;
    if (options.solve.row_width == 1U) {
        reduce_found += future_lookup.reduce_max_queries(
            batch.queries2,
            workspace.compact_best_values.data(),
            best_per_lane,
            0U,
            edge_stats,
            true,
            lookup_stats_ptr
        );
    } else {
        workspace.lane_best_values.resize(best_per_lane);
        for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
            std::fill(
                workspace.lane_best_values.begin(),
                workspace.lane_best_values.end(),
                options.solve.zero_value
            );
            reduce_found += future_lookup.reduce_max_queries(
                batch.queries2,
                workspace.lane_best_values.data(),
                workspace.lane_best_values.size(),
                lane,
                edge_stats,
                true,
                lookup_stats_ptr
            );
            for (size_t ref = 0U; ref < best_per_lane; ++ref) {
                workspace.compact_best_values[ref * options.solve.row_width + lane] =
                    workspace.lane_best_values[ref];
            }
        }
    }
    if (collect_batch_timing) {
        const double reduce_seconds = bc_single_chunk_now_seconds() - reduce_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_reduce_seconds += reduce_seconds;
        } else {
            workspace.spawn2_batch_reduce_seconds += reduce_seconds;
        }
    }
    bc_family_record_batch_query_stats(
        workspace,
        spawn4_phase,
        candidate_count,
        encoded_count,
        reduce_found,
        lookup_stats.entry_misses,
        lookup_stats.bitmap_misses
    );

    const double emit_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        BCSolveBoardQuerySummary summary;
        summary.terminal_success = batch.terminal[board_slot] != 0U;
        summary.empty_mask = batch.empty_masks[board_slot];
        summary.empty_count = batch.empty_counts[board_slot];
        if (summary.terminal_success || summary.empty_count == 0U) {
            continue;
        }
        const uint32_t cell_index = workspace.local_cell_indices[board_slot];
        const BCFamilyPartialCellLayout &layout = layouts[static_cast<size_t>(cell_index)];
        const uint32_t local_bucket_index = workspace.local_bucket_indices[board_slot];
        const BCFamilyPartialBucketLayout &partial_bucket =
            layout.buckets[static_cast<size_t>(local_bucket_index)];
        if (summary.empty_count != partial_bucket.empty_count) {
            throw std::logic_error("BC family spawn4 multi-cell partial empty count mismatch");
        }
        const StorageT *best =
            workspace.compact_best_values.data() +
            static_cast<size_t>(workspace.local_compact_offsets[board_slot]) *
                options.solve.row_width;
        partials[static_cast<size_t>(cell_index)].write_compact_success_row(
            layout,
            partial_bucket,
            workspace.local_success_rows[board_slot],
            best
        );
    }
    if (collect_batch_timing) {
        const double emit_seconds = bc_single_chunk_now_seconds() - emit_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_emit_seconds += emit_seconds;
        } else {
            workspace.spawn2_batch_emit_seconds += emit_seconds;
        }
    }
    batch.clear_batch();
}

template <typename StorageT, class Emit>
void bc_family_flush_phase_partial_sum_batch_multi_cell(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCSolvePreparedQueryEncoder &encoder,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<BCFamilyPartialCellLayout> &layouts,
    const std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> &partials,
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> *bucket_target_hits_by_cell,
    Emit &&emit
) {
    static_assert(
        std::is_integral_v<StorageT>,
        "BC family multi-cell partial grouped sum path requires integral storage"
    );
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    if (count == 0U) {
        return;
    }
    if (options.solve.row_width != 1U) {
        throw std::logic_error("BC family multi-cell partial grouped sum path requires row_width=1");
    }

    batch.canonical2_boards.clear();
    batch.canonical2_refs.clear();
    batch.queries2.clear();

    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const bool collect_batch_timing = options.collect_batch_timing;

    const double candidate_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint32_t compact_slot_count =
        bc_family_collect_phase_batch_candidates_multi_cell<StorageT, true>(
            workspace,
            lut,
            future_axis,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits_by_cell
        );
    if (collect_batch_timing) {
        const double candidate_seconds = bc_single_chunk_now_seconds() - candidate_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_candidate_seconds += candidate_seconds;
        } else {
            workspace.spawn2_batch_candidate_seconds += candidate_seconds;
        }
    }

    const double canonical_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint64_t candidate_count =
        static_cast<uint64_t>(batch.canonical2_boards.size());
    const uint64_t encoded_count = bc_family_flush_canonical_boards(
        encoder,
        spawn_rank,
        batch.canonical2_boards,
        batch.canonical2_refs,
        batch.queries2,
        options.solve.edge_options.canonical_symm_mode
    );
    if (collect_batch_timing) {
        const double canonical_seconds = bc_single_chunk_now_seconds() - canonical_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_canonical_seconds += canonical_seconds;
        } else {
            workspace.spawn2_batch_canonical_seconds += canonical_seconds;
        }
    }

    const double setup_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    workspace.compact_best_values.resize(compact_slot_count);
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        batch.integer_sum2[board_slot] = 0U;
        if (batch.terminal[board_slot] != 0U || batch.empty_counts[board_slot] == 0U) {
            continue;
        }
        const uint32_t cell_index = workspace.local_cell_indices[board_slot];
        const BCFamilyPartialCellLayout &layout = layouts[static_cast<size_t>(cell_index)];
        const BCFamilyPartialMaxCellBuffer<StorageT> &partial =
            partials[static_cast<size_t>(cell_index)];
        const BCFamilyPartialBucketLayout &partial_bucket =
            layout.buckets[static_cast<size_t>(workspace.local_bucket_indices[board_slot])];
        if (batch.empty_counts[board_slot] != partial_bucket.empty_count) {
            throw std::logic_error("BC family multi-cell partial grouped sum empty count mismatch");
        }
        const StorageT *previous =
            partial.compact_success_row_data(
                layout,
                partial_bucket,
                workspace.local_success_rows[board_slot]
            );
        const uint32_t ref_begin = workspace.local_compact_offsets[board_slot];
        uint64_t sum = 0U;
        for (uint32_t slot = 0U; slot < batch.empty_counts[board_slot]; ++slot) {
            const StorageT value = previous[slot];
            workspace.compact_best_values[static_cast<size_t>(ref_begin) + slot] = value;
            sum += static_cast<uint64_t>(value);
        }
        batch.integer_sum2[board_slot] = sum;
    }
    if (collect_batch_timing) {
        const double setup_seconds = bc_single_chunk_now_seconds() - setup_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_setup_seconds += setup_seconds;
        } else {
            workspace.spawn2_batch_setup_seconds += setup_seconds;
        }
    }

    const double reduce_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    typename BCFutureSuccessLookupView<StorageT>::BatchLookupStats lookup_stats;
    auto *lookup_stats_ptr = collect_batch_timing ? &lookup_stats : nullptr;
    const uint64_t reduce_found = future_lookup.reduce_grouped_compact_query_max_deltas(
        batch.queries2,
        workspace.compact_ref_board_slots.data(),
        workspace.compact_best_values.data(),
        compact_slot_count,
        batch.integer_sum2.data(),
        count,
        0U,
        true,
        lookup_stats_ptr
    );
    if (collect_batch_timing) {
        const double reduce_seconds = bc_single_chunk_now_seconds() - reduce_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_reduce_seconds += reduce_seconds;
        } else {
            workspace.spawn2_batch_reduce_seconds += reduce_seconds;
        }
    }
    bc_family_record_batch_query_stats(
        workspace,
        spawn4_phase,
        candidate_count,
        encoded_count,
        reduce_found,
        lookup_stats.entry_misses,
        lookup_stats.bitmap_misses
    );

    const double emit_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        BCSolveBoardQuerySummary summary;
        summary.terminal_success = batch.terminal[board_slot] != 0U;
        summary.empty_mask = batch.empty_masks[board_slot];
        summary.empty_count = batch.empty_counts[board_slot];
        emit(
            workspace.local_cell_indices[board_slot],
            workspace.local_success_rows[board_slot],
            workspace.local_bucket_indices[board_slot],
            summary,
            batch.integer_sum2[board_slot],
            workspace
        );
    }
    if (collect_batch_timing) {
        const double emit_seconds = bc_single_chunk_now_seconds() - emit_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_emit_seconds += emit_seconds;
        } else {
            workspace.spawn2_batch_emit_seconds += emit_seconds;
        }
    }
    batch.clear_batch();
}

template <typename StorageT, class Emit>
void bc_family_flush_phase_sum_batch(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCSolvePreparedQueryEncoder &encoder,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits,
    Emit &&emit
) {
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    if (count == 0U) {
        return;
    }
    if (options.solve.row_width != 1U) {
        throw std::logic_error("BC family compact grouped sum path requires row_width=1");
    }

    batch.canonical2_boards.clear();
    batch.canonical2_refs.clear();
    batch.queries2.clear();

    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;

    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const bool collect_batch_timing = options.collect_batch_timing;
    const double candidate_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint32_t compact_slot_count =
        bc_family_collect_phase_batch_candidates<StorageT, true>(
            workspace,
            lut,
            future_axis,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits
        );
    if (collect_batch_timing) {
        const double candidate_seconds = bc_single_chunk_now_seconds() - candidate_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_candidate_seconds += candidate_seconds;
        } else {
            workspace.spawn2_batch_candidate_seconds += candidate_seconds;
        }
    }

    const double canonical_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint64_t candidate_count =
        static_cast<uint64_t>(batch.canonical2_boards.size());
    const uint64_t encoded_count = bc_family_flush_canonical_boards(
        encoder,
        spawn_rank,
        batch.canonical2_boards,
        batch.canonical2_refs,
        batch.queries2,
        options.solve.edge_options.canonical_symm_mode
    );
    if (collect_batch_timing) {
        const double canonical_seconds = bc_single_chunk_now_seconds() - canonical_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_canonical_seconds += canonical_seconds;
        } else {
            workspace.spawn2_batch_canonical_seconds += canonical_seconds;
        }
    }

    const double setup_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint64_t zero = static_cast<uint64_t>(options.solve.zero_value);
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        batch.integer_sum2[board_slot] =
            static_cast<uint64_t>(batch.empty_counts[board_slot]) * zero;
    }
    if (collect_batch_timing) {
        const double setup_seconds = bc_single_chunk_now_seconds() - setup_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_setup_seconds += setup_seconds;
        } else {
            workspace.spawn2_batch_setup_seconds += setup_seconds;
        }
    }

    const double reduce_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    typename BCFutureSuccessLookupView<StorageT>::BatchLookupStats lookup_stats;
    auto *lookup_stats_ptr = collect_batch_timing ? &lookup_stats : nullptr;
    const uint64_t reduce_found = future_lookup.reduce_grouped_compact_query_sums(
        batch.queries2,
        workspace.compact_ref_board_slots.data(),
        compact_slot_count,
        batch.integer_sum2.data(),
        count,
        0U,
        options.solve.zero_value,
        true,
        lookup_stats_ptr
    );
    if (collect_batch_timing) {
        const double reduce_seconds = bc_single_chunk_now_seconds() - reduce_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_reduce_seconds += reduce_seconds;
        } else {
            workspace.spawn2_batch_reduce_seconds += reduce_seconds;
        }
    }
    bc_family_record_batch_query_stats(
        workspace,
        spawn4_phase,
        candidate_count,
        encoded_count,
        reduce_found,
        lookup_stats.entry_misses,
        lookup_stats.bitmap_misses
    );

    const double emit_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        BCSolveBoardQuerySummary summary;
        summary.terminal_success = batch.terminal[board_slot] != 0U;
        summary.empty_mask = batch.empty_masks[board_slot];
        summary.empty_count = batch.empty_counts[board_slot];
        emit(
            workspace.local_success_rows[board_slot],
            workspace.local_bucket_indices[board_slot],
            summary,
            batch.integer_sum2[board_slot],
            workspace
        );
    }
    if (collect_batch_timing) {
        const double emit_seconds = bc_single_chunk_now_seconds() - emit_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_emit_seconds += emit_seconds;
        } else {
            workspace.spawn2_batch_emit_seconds += emit_seconds;
        }
    }
    batch.clear_batch();
}

template <typename StorageT, class Emit>
void bc_family_flush_phase_partial_sum_batch(
    BCFamilySolveCellWorkspace<StorageT> &workspace,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCSolvePreparedQueryEncoder &encoder,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    const BCFamilyPartialCellLayout &layout,
    const BCFamilyPartialMaxCellBuffer<StorageT> &partial,
    const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits,
    Emit &&emit
) {
    static_assert(
        std::is_integral_v<StorageT>,
        "BC family partial grouped sum path requires integral storage"
    );
    BCResidentBatchWorkspace<StorageT> &batch = workspace.batch_workspace;
    const uint32_t count = batch.count;
    if (count == 0U) {
        return;
    }
    if (options.solve.row_width != 1U) {
        throw std::logic_error("BC family partial grouped sum path requires row_width=1");
    }

    batch.canonical2_boards.clear();
    batch.canonical2_refs.clear();
    batch.queries2.clear();

    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;

    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const bool collect_batch_timing = options.collect_batch_timing;
    const double candidate_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint32_t compact_slot_count =
        bc_family_collect_phase_batch_candidates<StorageT, true>(
            workspace,
            lut,
            future_axis,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits
        );
    if (collect_batch_timing) {
        const double candidate_seconds = bc_single_chunk_now_seconds() - candidate_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_candidate_seconds += candidate_seconds;
        } else {
            workspace.spawn2_batch_candidate_seconds += candidate_seconds;
        }
    }

    const double canonical_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    const uint64_t candidate_count =
        static_cast<uint64_t>(batch.canonical2_boards.size());
    const uint64_t encoded_count = bc_family_flush_canonical_boards(
        encoder,
        spawn_rank,
        batch.canonical2_boards,
        batch.canonical2_refs,
        batch.queries2,
        options.solve.edge_options.canonical_symm_mode
    );
    if (collect_batch_timing) {
        const double canonical_seconds = bc_single_chunk_now_seconds() - canonical_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_canonical_seconds += canonical_seconds;
        } else {
            workspace.spawn2_batch_canonical_seconds += canonical_seconds;
        }
    }

    const double setup_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    workspace.compact_best_values.resize(compact_slot_count);
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        batch.integer_sum2[board_slot] = 0U;
        if (batch.terminal[board_slot] != 0U || batch.empty_counts[board_slot] == 0U) {
            continue;
        }
        const BCFamilyPartialBucketLayout &partial_bucket =
            layout.buckets[static_cast<size_t>(workspace.local_bucket_indices[board_slot])];
        if (batch.empty_counts[board_slot] != partial_bucket.empty_count) {
            throw std::logic_error("BC family partial grouped sum empty count mismatch");
        }
        const StorageT *previous =
            partial.compact_success_row_data(
                layout,
                partial_bucket,
                workspace.local_success_rows[board_slot]
            );
        const uint32_t ref_begin = workspace.local_compact_offsets[board_slot];
        uint64_t sum = 0U;
        for (uint32_t slot = 0U; slot < batch.empty_counts[board_slot]; ++slot) {
            const StorageT value = previous[slot];
            workspace.compact_best_values[static_cast<size_t>(ref_begin) + slot] = value;
            sum += static_cast<uint64_t>(value);
        }
        batch.integer_sum2[board_slot] = sum;
    }
    if (collect_batch_timing) {
        const double setup_seconds = bc_single_chunk_now_seconds() - setup_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_setup_seconds += setup_seconds;
        } else {
            workspace.spawn2_batch_setup_seconds += setup_seconds;
        }
    }

    const double reduce_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    typename BCFutureSuccessLookupView<StorageT>::BatchLookupStats lookup_stats;
    auto *lookup_stats_ptr = collect_batch_timing ? &lookup_stats : nullptr;
    const uint64_t reduce_found = future_lookup.reduce_grouped_compact_query_max_deltas(
        batch.queries2,
        workspace.compact_ref_board_slots.data(),
        workspace.compact_best_values.data(),
        compact_slot_count,
        batch.integer_sum2.data(),
        count,
        0U,
        true,
        lookup_stats_ptr
    );
    if (collect_batch_timing) {
        const double reduce_seconds = bc_single_chunk_now_seconds() - reduce_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_reduce_seconds += reduce_seconds;
        } else {
            workspace.spawn2_batch_reduce_seconds += reduce_seconds;
        }
    }
    bc_family_record_batch_query_stats(
        workspace,
        spawn4_phase,
        candidate_count,
        encoded_count,
        reduce_found,
        lookup_stats.entry_misses,
        lookup_stats.bitmap_misses
    );

    const double emit_t0 = collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        BCSolveBoardQuerySummary summary;
        summary.terminal_success = batch.terminal[board_slot] != 0U;
        summary.empty_mask = batch.empty_masks[board_slot];
        summary.empty_count = batch.empty_counts[board_slot];
        emit(
            workspace.local_success_rows[board_slot],
            workspace.local_bucket_indices[board_slot],
            summary,
            batch.integer_sum2[board_slot],
            workspace
        );
    }
    if (collect_batch_timing) {
        const double emit_seconds = bc_single_chunk_now_seconds() - emit_t0;
        if (spawn4_phase) {
            workspace.spawn4_batch_emit_seconds += emit_seconds;
        } else {
            workspace.spawn2_batch_emit_seconds += emit_seconds;
        }
    }
    batch.clear_batch();
}

template <typename StorageT, class Flush>
void bc_family_scan_cell_phase_batches_with_flush(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Flush &&flush
) {
    std::vector<BCFamilySolveCellRangeWork> &work_items = workspace.cell_range_work_items;
    bc_family_build_cell_range_work(
        lut,
        cell.view(),
        options.source_bitmap_words_per_work_item,
        work_items
    );
    stats.single.current_work_items = bc_checked_add_u64(
        stats.single.current_work_items,
        work_items.size(),
        "BC family solve current work item stats overflow"
    );
    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    const bool use_parallel =
        threads > 1 &&
        work_items.size() >= static_cast<size_t>(std::max<uint32_t>(
            2U,
            options.cell_parallel_min_work_items
        ));
    const uint8_t spawn_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    std::vector<BCFamilyBucketSpawnTargetHits> &bucket_target_hits =
        workspace.bucket_target_hits;
    const double bucket_hit_t0 =
        options.collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
    bc_family_build_bucket_spawn_target_hits(
        lut,
        cell.view(),
        future_axis,
        filter,
        directions,
        spawn_rank,
        options.solve.word_sums,
        options.solve.edge_options.future_cell_modulus,
        bucket_target_hits
    );
    if (options.collect_batch_timing) {
        const double bucket_hit_seconds = bc_single_chunk_now_seconds() - bucket_hit_t0;
        if (phase == BCSolveSpawnPhase::Spawn4) {
            stats.spawn4_bucket_hit_seconds += bucket_hit_seconds;
        } else {
            stats.spawn2_bucket_hit_seconds += bucket_hit_seconds;
        }
    }
    const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits_ptr =
        bucket_target_hits.empty() ? nullptr : &bucket_target_hits;
    const double t0 = bc_single_chunk_now_seconds();

    auto push_entry = [](BCFamilySolveCellWorkspace<StorageT> &local,
                         uint32_t bucket_i,
                         const BCScannedBoardEntry &entry) {
        BCResidentBatchWorkspace<StorageT> &batch = local.batch_workspace;
        const uint32_t slot = batch.count;
        batch.boards[slot] = entry.board;
        batch.empty_masks[slot] = entry.empty_mask;
        local.local_success_rows[slot] = entry.local_success_row;
        local.local_bucket_indices[slot] = bucket_i;
        ++batch.count;
    };

    if (!use_parallel) {
        if (workspace.cell_workspaces.empty()) {
            workspace.cell_workspaces.resize(1U);
        }
        BCFamilySolveCellWorkspace<StorageT> &local = workspace.cell_workspaces.front();
        bc_family_prepare_cell_workspace(local, options.solve.row_width, options.solve.zero_value);
        local.batch_workspace.clear_batch();
        BCLoadedCellScanner scanner(lut, cell.view());
        for (const BCFamilySolveCellRangeWork &work : work_items) {
            for (uint32_t bucket_i = work.bucket_begin; bucket_i < work.bucket_end; ++bucket_i) {
                const bool ranged_bucket =
                    work.bucket_end == work.bucket_begin + 1U && work.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                if (bucket_i + 1U < work.bucket_end) {
                    __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                }
#endif
                scanner.for_each_bucket_word_range_board(
                    bucket_i,
                    ranged_bucket ? work.word_begin : 0U,
                    ranged_bucket ? work.word_end : 0U,
                    [&](const BCScannedBoardEntry &entry) {
                        push_entry(local, bucket_i, entry);
                        if (local.batch_workspace.count ==
                            BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                            flush(local, bucket_target_hits_ptr);
                        }
                    }
                );
            }
        }
        flush(local, bucket_target_hits_ptr);
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(stats.single.edge, local.stats.edge);
        }
        bc_family_accumulate_cell_workspace_batch_stats(stats, local);
        stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
        return;
    }

    if (workspace.cell_workspaces.size() < static_cast<size_t>(threads)) {
        workspace.cell_workspaces.resize(static_cast<size_t>(threads));
    }
    for (int i = 0; i < threads; ++i) {
        bc_family_prepare_cell_workspace(
            workspace.cell_workspaces[static_cast<size_t>(i)],
            options.solve.row_width,
            options.solve.zero_value
        );
        workspace.cell_workspaces[static_cast<size_t>(i)].batch_workspace.clear_batch();
    }

    std::exception_ptr first_exception;
    const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
        1U,
        options.source_work_schedule_chunk
    ));
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCFamilySolveCellWorkspace<StorageT> &local =
            workspace.cell_workspaces[static_cast<size_t>(tid)];
        BCLoadedCellScanner scanner(lut, cell.view());
        try {
#pragma omp for schedule(dynamic, schedule_chunk)
            for (int64_t work_i = 0; work_i < static_cast<int64_t>(work_items.size()); ++work_i) {
                const BCFamilySolveCellRangeWork &work =
                    work_items[static_cast<size_t>(work_i)];
                for (uint32_t bucket_i = work.bucket_begin; bucket_i < work.bucket_end; ++bucket_i) {
                    const bool ranged_bucket =
                        work.bucket_end == work.bucket_begin + 1U && work.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                    if (bucket_i + 1U < work.bucket_end) {
                        __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                    }
#endif
                    scanner.for_each_bucket_word_range_board(
                        bucket_i,
                        ranged_bucket ? work.word_begin : 0U,
                        ranged_bucket ? work.word_end : 0U,
                        [&](const BCScannedBoardEntry &entry) {
                            push_entry(local, bucket_i, entry);
                            if (local.batch_workspace.count ==
                                BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                                flush(local, bucket_target_hits_ptr);
                            }
                        }
                    );
                }
            }
            flush(local, bucket_target_hits_ptr);
        } catch (...) {
#pragma omp critical(BCFamilySolveCellPhaseBatchException)
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
    for (int i = 0; i < threads; ++i) {
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(
                stats.single.edge,
                workspace.cell_workspaces[static_cast<size_t>(i)].stats.edge
            );
        }
        bc_family_accumulate_cell_workspace_batch_stats(
            stats,
            workspace.cell_workspaces[static_cast<size_t>(i)]
        );
    }
    stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
}

template <typename StorageT>
void bc_family_first_direction_cells_batch(
    const BCLut &lut,
    const std::vector<BCLoadedCell> &current_cells,
    const std::vector<size_t> &cell_indices,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const std::vector<BCFamilyPartialCellLayout> &layouts,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> &partials
) {
    if (cell_indices.empty()) {
        return;
    }
    if (options.solve.row_width == 0U) {
        throw std::logic_error("BC family first-direction batch row_width is zero");
    }

    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const uint8_t spawn_rank = spawn4_phase
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    std::vector<BCSingleChunkLoadedWorkItem> &work_items = workspace.pass_cell_work_items;
    work_items.clear();
    std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> &bucket_hits_by_cell =
        workspace.pass_bucket_target_hits;
    bucket_hits_by_cell.clear();
    bucket_hits_by_cell.resize(current_cells.size());

    for (size_t cell_index : cell_indices) {
        const BCLoadedCell &cell = current_cells[cell_index];
        if (cell.success_rows == 0U || cell.buckets.empty()) {
            continue;
        }
        partials[cell_index].reset(layouts[cell_index], options.solve.zero_value);
        bc_family_append_pass_cell_range_work(
            lut,
            cell.view(),
            cell_index,
            options.source_bitmap_words_per_work_item,
            work_items
        );
        const double bucket_hit_t0 =
            options.collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
        bc_family_build_bucket_spawn_target_hits(
            lut,
            cell.view(),
            future_axis,
            filter,
            directions,
            spawn_rank,
            options.solve.word_sums,
            options.solve.edge_options.future_cell_modulus,
            bucket_hits_by_cell[cell_index]
        );
        if (options.collect_batch_timing) {
            const double bucket_hit_seconds = bc_single_chunk_now_seconds() - bucket_hit_t0;
            if (spawn4_phase) {
                stats.spawn4_bucket_hit_seconds += bucket_hit_seconds;
            } else {
                stats.spawn2_bucket_hit_seconds += bucket_hit_seconds;
            }
        }
    }
    stats.single.current_work_items = bc_checked_add_u64(
        stats.single.current_work_items,
        work_items.size(),
        "BC family solve pass work item stats overflow"
    );
    if (work_items.empty()) {
        return;
    }

    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    if (workspace.cell_workspaces.size() < static_cast<size_t>(threads)) {
        workspace.cell_workspaces.resize(static_cast<size_t>(threads));
    }
    for (int i = 0; i < threads; ++i) {
        bc_family_prepare_cell_workspace(
            workspace.cell_workspaces[static_cast<size_t>(i)],
            options.solve.row_width,
            options.solve.zero_value
        );
        workspace.cell_workspaces[static_cast<size_t>(i)].batch_workspace.clear_batch();
    }

    const BCSolvePreparedQueryEncoder encoder(
        lut,
        future_axis,
        options.solve.edge_options.future_cell_modulus
    );
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> *bucket_hits_ptr =
        filter.enabled ? &bucket_hits_by_cell : nullptr;
    const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
        1U,
        options.source_work_schedule_chunk
    ));
    const double t0 = bc_single_chunk_now_seconds();
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCFamilySolveCellWorkspace<StorageT> &local =
            workspace.cell_workspaces[static_cast<size_t>(tid)];
        auto flush = [&]() {
            bc_family_flush_first_direction_batch_multi_cell<StorageT>(
                local,
                lut,
                future_axis,
                encoder,
                future_lookup,
                directions,
                phase,
                filter,
                options,
                layouts,
                partials,
                bucket_hits_ptr
            );
        };
        try {
#pragma omp for schedule(dynamic, schedule_chunk)
            for (int64_t item_i_signed = 0;
                 item_i_signed < static_cast<int64_t>(work_items.size());
                 ++item_i_signed) {
                const BCSingleChunkLoadedWorkItem &item =
                    work_items[static_cast<size_t>(item_i_signed)];
                const BCLoadedCell &cell = current_cells[item.cell_index];
                BCLoadedCellScanner scanner(lut, cell.view());
                for (uint32_t bucket_i = item.bucket_begin;
                     bucket_i < item.bucket_end;
                     ++bucket_i) {
                    const bool ranged_bucket =
                        item.bucket_end == item.bucket_begin + 1U && item.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                    if (bucket_i + 1U < item.bucket_end) {
                        __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                    }
#endif
                    scanner.for_each_bucket_word_range_board(
                        bucket_i,
                        ranged_bucket ? item.word_begin : 0U,
                        ranged_bucket ? item.word_end : 0U,
                        [&](const BCScannedBoardEntry &entry) {
                            BCResidentBatchWorkspace<StorageT> &batch = local.batch_workspace;
                            const uint32_t slot = batch.count;
                            batch.boards[slot] = entry.board;
                            batch.empty_masks[slot] = entry.empty_mask;
                            local.local_success_rows[slot] = entry.local_success_row;
                            local.local_bucket_indices[slot] = bucket_i;
                            local.local_cell_indices[slot] =
                                bc_family_checked_u16_size(
                                    item.cell_index,
                                    "BC family local cell index exceeds uint16"
                                );
                            ++batch.count;
                            if (batch.count == BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                                flush();
                            }
                        }
                    );
                }
            }
            flush();
        } catch (...) {
#pragma omp critical(BCFamilySolvePassBatchException)
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
    for (int i = 0; i < threads; ++i) {
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(
                stats.single.edge,
                workspace.cell_workspaces[static_cast<size_t>(i)].stats.edge
            );
        }
        bc_family_accumulate_cell_workspace_batch_stats(
            stats,
            workspace.cell_workspaces[static_cast<size_t>(i)]
        );
    }
    stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
}

template <typename StorageT, class Emit>
void bc_family_partial_sum_cells_batch(
    const BCLut &lut,
    const std::vector<BCLoadedCell> &current_cells,
    const std::vector<size_t> &cell_indices,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const std::vector<BCFamilyPartialCellLayout> &layouts,
    const std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> &partials,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Emit &&emit
) {
    static_assert(
        std::is_integral_v<StorageT>,
        "BC family multi-cell partial sum path requires integral storage"
    );
    if (cell_indices.empty()) {
        return;
    }
    if (options.solve.row_width != 1U) {
        throw std::logic_error("BC family multi-cell partial sum path requires row_width=1");
    }

    const bool spawn4_phase = phase == BCSolveSpawnPhase::Spawn4;
    const uint8_t spawn_rank = spawn4_phase
        ? options.solve.edge_options.spawn4_tile_rank
        : options.solve.edge_options.spawn2_tile_rank;
    std::vector<BCSingleChunkLoadedWorkItem> &work_items = workspace.pass_cell_work_items;
    work_items.clear();
    std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> &bucket_hits_by_cell =
        workspace.pass_bucket_target_hits;
    bucket_hits_by_cell.clear();
    bucket_hits_by_cell.resize(current_cells.size());

    for (size_t cell_index : cell_indices) {
        const BCLoadedCell &cell = current_cells[cell_index];
        if (cell.success_rows == 0U || cell.buckets.empty()) {
            continue;
        }
        if (partials[cell_index].values().size() !=
            static_cast<size_t>(layouts[cell_index].value_count)) {
            throw std::logic_error("BC family multi-cell partial sum partial count mismatch");
        }
        bc_family_append_pass_cell_range_work(
            lut,
            cell.view(),
            cell_index,
            options.source_bitmap_words_per_work_item,
            work_items
        );
        const double bucket_hit_t0 =
            options.collect_batch_timing ? bc_single_chunk_now_seconds() : 0.0;
        bc_family_build_bucket_spawn_target_hits(
            lut,
            cell.view(),
            future_axis,
            filter,
            directions,
            spawn_rank,
            options.solve.word_sums,
            options.solve.edge_options.future_cell_modulus,
            bucket_hits_by_cell[cell_index]
        );
        if (options.collect_batch_timing) {
            const double bucket_hit_seconds = bc_single_chunk_now_seconds() - bucket_hit_t0;
            if (spawn4_phase) {
                stats.spawn4_bucket_hit_seconds += bucket_hit_seconds;
            } else {
                stats.spawn2_bucket_hit_seconds += bucket_hit_seconds;
            }
        }
    }
    stats.single.current_work_items = bc_checked_add_u64(
        stats.single.current_work_items,
        work_items.size(),
        "BC family solve pass work item stats overflow"
    );
    if (work_items.empty()) {
        return;
    }

    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    if (workspace.cell_workspaces.size() < static_cast<size_t>(threads)) {
        workspace.cell_workspaces.resize(static_cast<size_t>(threads));
    }
    for (int i = 0; i < threads; ++i) {
        bc_family_prepare_cell_workspace(
            workspace.cell_workspaces[static_cast<size_t>(i)],
            options.solve.row_width,
            options.solve.zero_value
        );
        workspace.cell_workspaces[static_cast<size_t>(i)].batch_workspace.clear_batch();
    }

    const BCSolvePreparedQueryEncoder encoder(
        lut,
        future_axis,
        options.solve.edge_options.future_cell_modulus
    );
    const std::vector<std::vector<BCFamilyBucketSpawnTargetHits>> *bucket_hits_ptr =
        filter.enabled ? &bucket_hits_by_cell : nullptr;
    const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
        1U,
        options.source_work_schedule_chunk
    ));
    const double t0 = bc_single_chunk_now_seconds();
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCFamilySolveCellWorkspace<StorageT> &local =
            workspace.cell_workspaces[static_cast<size_t>(tid)];
        auto flush = [&]() {
            bc_family_flush_phase_partial_sum_batch_multi_cell<StorageT>(
                local,
                lut,
                future_axis,
                encoder,
                future_lookup,
                directions,
                phase,
                filter,
                options,
                layouts,
                partials,
                bucket_hits_ptr,
                emit
            );
        };
        try {
#pragma omp for schedule(dynamic, schedule_chunk)
            for (int64_t item_i_signed = 0;
                 item_i_signed < static_cast<int64_t>(work_items.size());
                 ++item_i_signed) {
                const BCSingleChunkLoadedWorkItem &item =
                    work_items[static_cast<size_t>(item_i_signed)];
                const BCLoadedCell &cell = current_cells[item.cell_index];
                BCLoadedCellScanner scanner(lut, cell.view());
                for (uint32_t bucket_i = item.bucket_begin;
                     bucket_i < item.bucket_end;
                     ++bucket_i) {
                    const bool ranged_bucket =
                        item.bucket_end == item.bucket_begin + 1U && item.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                    if (bucket_i + 1U < item.bucket_end) {
                        __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                    }
#endif
                    scanner.for_each_bucket_word_range_board(
                        bucket_i,
                        ranged_bucket ? item.word_begin : 0U,
                        ranged_bucket ? item.word_end : 0U,
                        [&](const BCScannedBoardEntry &entry) {
                            BCResidentBatchWorkspace<StorageT> &batch = local.batch_workspace;
                            const uint32_t slot = batch.count;
                            batch.boards[slot] = entry.board;
                            batch.empty_masks[slot] = entry.empty_mask;
                            local.local_success_rows[slot] = entry.local_success_row;
                            local.local_bucket_indices[slot] = bucket_i;
                            local.local_cell_indices[slot] =
                                bc_family_checked_u16_size(
                                    item.cell_index,
                                    "BC family local cell index exceeds uint16"
                                );
                            ++batch.count;
                            if (batch.count == BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                                flush();
                            }
                        }
                    );
                }
            }
            flush();
        } catch (...) {
#pragma omp critical(BCFamilySolvePassPartialSumBatchException)
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
    for (int i = 0; i < threads; ++i) {
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(
                stats.single.edge,
                workspace.cell_workspaces[static_cast<size_t>(i)].stats.edge
            );
        }
        bc_family_accumulate_cell_workspace_batch_stats(
            stats,
            workspace.cell_workspaces[static_cast<size_t>(i)]
        );
    }
    stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
}

template <typename StorageT, class Emit>
void bc_family_scan_cell_phase_batches(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Emit &&emit
) {
    const BCSolvePreparedQueryEncoder encoder(
        lut,
        future_axis,
        options.solve.edge_options.future_cell_modulus
    );
    auto flush = [&](BCFamilySolveCellWorkspace<StorageT> &local,
                     const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits) {
        bc_family_flush_phase_batch<StorageT>(
            local,
            lut,
            future_axis,
            encoder,
            future_lookup,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits,
            emit
        );
    };
    bc_family_scan_cell_phase_batches_with_flush<StorageT>(
        lut,
        cell,
        directions,
        phase,
        filter,
        future_axis,
        future_lookup,
        options,
        workspace,
        stats,
        flush
    );
}

template <typename StorageT, class Emit>
void bc_family_scan_cell_phase_sum_batches(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Emit &&emit
) {
    const BCSolvePreparedQueryEncoder encoder(
        lut,
        future_axis,
        options.solve.edge_options.future_cell_modulus
    );
    auto flush = [&](BCFamilySolveCellWorkspace<StorageT> &local,
                     const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits) {
        bc_family_flush_phase_sum_batch<StorageT>(
            local,
            lut,
            future_axis,
            encoder,
            future_lookup,
            directions,
            phase,
            filter,
            options,
            bucket_target_hits,
            emit
        );
    };
    bc_family_scan_cell_phase_batches_with_flush<StorageT>(
        lut,
        cell,
        directions,
        phase,
        filter,
        future_axis,
        future_lookup,
        options,
        workspace,
        stats,
        flush
    );
}

template <typename StorageT, class Emit>
void bc_family_scan_cell_phase_partial_sum_batches(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCSolveSpawnPhase phase,
    const BCSolveTargetFamilyFilter &filter,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCFamilySolveOptions<StorageT> &options,
    const BCFamilyPartialCellLayout &layout,
    const BCFamilyPartialMaxCellBuffer<StorageT> &partial,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Emit &&emit
) {
    const BCSolvePreparedQueryEncoder encoder(
        lut,
        future_axis,
        options.solve.edge_options.future_cell_modulus
    );
    auto flush = [&](BCFamilySolveCellWorkspace<StorageT> &local,
                     const std::vector<BCFamilyBucketSpawnTargetHits> *bucket_target_hits) {
        bc_family_flush_phase_partial_sum_batch<StorageT>(
            local,
            lut,
            future_axis,
            encoder,
            future_lookup,
            directions,
            phase,
            filter,
            options,
            layout,
            partial,
            bucket_target_hits,
            emit
        );
    };
    bc_family_scan_cell_phase_batches_with_flush<StorageT>(
        lut,
        cell,
        directions,
        phase,
        filter,
        future_axis,
        future_lookup,
        options,
        workspace,
        stats,
        flush
    );
}

template <typename StorageT, class Emit>
void bc_family_scan_cell_entries(
    const BCLut &lut,
    const BCLoadedCell &cell,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    Emit &&emit
) {
    std::vector<BCFamilySolveCellRangeWork> &work_items = workspace.cell_range_work_items;
    bc_family_build_cell_range_work(
        lut,
        cell.view(),
        options.source_bitmap_words_per_work_item,
        work_items
    );
    stats.single.current_work_items = bc_checked_add_u64(
        stats.single.current_work_items,
        work_items.size(),
        "BC family solve current work item stats overflow"
    );
    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    const bool use_parallel =
        threads > 1 &&
        work_items.size() >= static_cast<size_t>(std::max<uint32_t>(
            2U,
            options.cell_parallel_min_work_items
        ));
    const double t0 = bc_single_chunk_now_seconds();

    if (!use_parallel) {
        if (workspace.cell_workspaces.empty()) {
            workspace.cell_workspaces.resize(1U);
        }
        BCFamilySolveCellWorkspace<StorageT> &local = workspace.cell_workspaces.front();
        bc_family_prepare_cell_workspace(local, options.solve.row_width, options.solve.zero_value);
        BCLoadedCellScanner scanner(lut, cell.view());
        for (const BCFamilySolveCellRangeWork &work : work_items) {
            for (uint32_t bucket_i = work.bucket_begin; bucket_i < work.bucket_end; ++bucket_i) {
                const bool ranged_bucket =
                    work.bucket_end == work.bucket_begin + 1U && work.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                if (bucket_i + 1U < work.bucket_end) {
                    __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                }
#endif
                scanner.for_each_bucket_word_range_board(
                    bucket_i,
                    ranged_bucket ? work.word_begin : 0U,
                    ranged_bucket ? work.word_end : 0U,
                    [&](const BCScannedBoardEntry &entry) {
                        emit(entry, local);
                    }
                );
            }
        }
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(stats.single.edge, local.stats.edge);
        }
        stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
        return;
    }

    if (workspace.cell_workspaces.size() < static_cast<size_t>(threads)) {
        workspace.cell_workspaces.resize(static_cast<size_t>(threads));
    }
    for (int i = 0; i < threads; ++i) {
        bc_family_prepare_cell_workspace(
            workspace.cell_workspaces[static_cast<size_t>(i)],
            options.solve.row_width,
            options.solve.zero_value
        );
    }

    std::exception_ptr first_exception;
    const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
        1U,
        options.source_work_schedule_chunk
    ));
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCFamilySolveCellWorkspace<StorageT> &local =
            workspace.cell_workspaces[static_cast<size_t>(tid)];
        BCLoadedCellScanner scanner(lut, cell.view());
        try {
#pragma omp for schedule(dynamic, schedule_chunk)
            for (int64_t work_i = 0; work_i < static_cast<int64_t>(work_items.size()); ++work_i) {
                const BCFamilySolveCellRangeWork &work =
                    work_items[static_cast<size_t>(work_i)];
                for (uint32_t bucket_i = work.bucket_begin; bucket_i < work.bucket_end; ++bucket_i) {
                    const bool ranged_bucket =
                        work.bucket_end == work.bucket_begin + 1U && work.word_end != 0U;
#if defined(__GNUC__) || defined(__clang__)
                    if (bucket_i + 1U < work.bucket_end) {
                        __builtin_prefetch(&cell.buckets[static_cast<size_t>(bucket_i + 1U)], 0, 1);
                    }
#endif
                    scanner.for_each_bucket_word_range_board(
                        bucket_i,
                        ranged_bucket ? work.word_begin : 0U,
                        ranged_bucket ? work.word_end : 0U,
                        [&](const BCScannedBoardEntry &entry) {
                            emit(entry, local);
                        }
                    );
                }
            }
        } catch (...) {
#pragma omp critical(BCFamilySolveCellException)
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
    for (int i = 0; i < threads; ++i) {
        if (options.collect_edge_stats) {
            bc_resident_solve_accumulate_edge_stats(
                stats.single.edge,
                workspace.cell_workspaces[static_cast<size_t>(i)].stats.edge
            );
        }
    }
    stats.single.recalc_seconds += bc_single_chunk_now_seconds() - t0;
}

} // namespace detail

template <typename StorageT>
void bc_family_validate_solve_inputs(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCFamilyPartitionLayerMap &current_partition,
    const BCFamilyPartitionLayerMap &future2_partition,
    const BCFamilyPartitionLayerMap &future4_partition,
    const BCFamilySolveOptions<StorageT> &options
) {
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC family solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC family solve dtype does not match storage type");
    }
    if (future2_success.row_width() != options.solve.row_width ||
        future4_success.row_width() != options.solve.row_width) {
        throw std::invalid_argument("BC family solve future success row_width mismatch");
    }
    if (future2_success.dtype_mode() != options.solve.dtype ||
        future4_success.dtype_mode() != options.solve.dtype) {
        throw std::invalid_argument("BC family solve future success dtype mismatch");
    }
    if (current_position.axis().family_unit() != future2_position.axis().family_unit() ||
        current_position.axis().family_unit() != future4_position.axis().family_unit()) {
        throw std::invalid_argument("BC family solve family_unit mismatch");
    }
    if (current_partition.family_count() != current_position.axis().family_count() ||
        future2_partition.family_count() != future2_position.axis().family_count() ||
        future4_partition.family_count() != future4_position.axis().family_count()) {
        throw std::invalid_argument("BC family solve partition/axis count mismatch");
    }
    const bool any_mod =
        current_partition.policy.kind == BCFamilyPartitionKind::ModuloCoord ||
        future2_partition.policy.kind == BCFamilyPartitionKind::ModuloCoord ||
        future4_partition.policy.kind == BCFamilyPartitionKind::ModuloCoord;
    if (any_mod) {
        if (current_partition.policy.kind != BCFamilyPartitionKind::ModuloCoord ||
            future2_partition.policy.kind != BCFamilyPartitionKind::ModuloCoord ||
            future4_partition.policy.kind != BCFamilyPartitionKind::ModuloCoord ||
            current_partition.policy.modulus != future2_partition.policy.modulus ||
            current_partition.policy.modulus != future4_partition.policy.modulus) {
            throw std::invalid_argument("BC family solve v1 requires matching modulo partitions");
        }
    }
}

[[nodiscard]] inline SpawnDeltaCoord bc_family_solve_delta_for_spawn_rank(
    const BCLut &lut,
    const BCFamilyTable &current_axis,
    uint8_t spawn_rank
) {
    const uint32_t tile_sum = lut.tile_sum_value(spawn_rank);
    const uint32_t unit = current_axis.family_unit();
    if (unit == 0U || (tile_sum % unit) != 0U) {
        throw std::invalid_argument("BC family solve spawn delta is not divisible by family_unit");
    }
    const uint32_t delta = tile_sum / unit;
    if (delta > std::numeric_limits<SpawnDeltaCoord>::max()) {
        throw std::overflow_error("BC family solve spawn delta exceeds SpawnDeltaCoord");
    }
    return static_cast<SpawnDeltaCoord>(delta);
}

template <typename StorageT>
[[nodiscard]] BCFamilySolveOptions<StorageT> bc_family_effective_options(
    const BCLut &lut,
    const BCFamilyPartitionLayerMap &current_partition,
    BCFamilySolveOptions<StorageT> options
) {
    if (current_partition.policy.kind == BCFamilyPartitionKind::ModuloCoord) {
        options.solve.edge_options.future_cell_modulus = current_partition.policy.modulus;
    } else {
        options.solve.edge_options.future_cell_modulus = 0U;
    }
    if (options.temp_direct_queue_depth == 0U) {
        options.temp_direct_queue_depth = 1U;
    }
    if (options.interleave_block_fids == 0U) {
        options.interleave_block_fids = 1U;
    }
    if (options.interleave_block_fids != 1U) {
        throw std::invalid_argument(
            "BC family solve requires interleave_block_fids == 1"
        );
    }
    const int target_rank = options.solve.edge_options.success_target_rank;
    if (target_rank > 0 && target_rank < 16) {
        const uint32_t target_sum =
            lut.tile_sum_value(static_cast<uint8_t>(target_rank));
        if (current_partition.layer_sum < target_sum) {
            options.solve.edge_options.success_target_rank = 0;
            options.solve.edge_options.success_shifts = nullptr;
            options.solve.edge_options.success_check_all_cells = false;
        }
    }
    return options;
}

struct BCFamilySolveCachedPass {
    BCFamilySolvePassPlan pass;
    std::vector<CellId> current_cids;
    std::vector<CellId> keep_cids;
    std::vector<FamilyId> keep_families;
};

[[nodiscard]] inline std::vector<BCFamilySolvePassPlan> bc_family_build_passes(
    const BCFamilySolvePlanner &planner,
    BCSolveSpawnPhase phase,
    SpawnDeltaCoord delta_coord,
    uint32_t family_count
) {
    std::vector<BCFamilySolvePassPlan> passes;
    passes.reserve(family_count);
    for (FamilyId fid = 0U; fid < family_count; ++fid) {
        passes.push_back(planner.make_pass(fid, phase, delta_coord));
    }
    return passes;
}

inline void bc_family_add_unique_cells(
    std::vector<CellId> &dst,
    const std::vector<CellId> &src
) {
    dst.insert(dst.end(), src.begin(), src.end());
}

inline void bc_family_sort_unique_cells(std::vector<CellId> &cells) {
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}

inline void bc_family_add_pass_families_unbounded(
    std::vector<FamilyId> &families,
    const FamilyIdList3 &pass_families
) {
    for (FamilyId family : pass_families) {
        if (std::find(families.begin(), families.end(), family) == families.end()) {
            families.push_back(family);
        }
    }
}

[[nodiscard]] inline bool bc_family_try_add_pass_families(
    std::vector<FamilyId> &families,
    const FamilyIdList3 &pass_families,
    uint32_t max_families
) {
    std::vector<FamilyId> trial = families;
    bc_family_add_pass_families_unbounded(trial, pass_families);
    if (max_families != 0U && trial.size() > max_families) {
        return false;
    }
    families = std::move(trial);
    return true;
}

[[nodiscard]] inline std::vector<BCFamilySolveCachedPass> bc_family_build_cached_passes(
    const std::vector<BCFamilySolvePassPlan> &passes,
    uint32_t max_future_families
) {
    std::vector<BCFamilySolveCachedPass> cached;
    cached.reserve(passes.size());
    for (const BCFamilySolvePassPlan &pass : passes) {
        if (max_future_families != 0U && pass.future_families.size() > max_future_families) {
            throw std::logic_error("BC family solve pass fanout exceeds configured future family window");
        }
        BCFamilySolveCachedPass entry;
        entry.pass = pass;
        entry.current_cids.reserve(pass.current_cells.size());
        for (const BCFamilySolveCellWork &work : pass.current_cells) {
            entry.current_cids.push_back(work.cid);
        }
        cached.push_back(std::move(entry));
    }

    if (max_future_families == 0U) {
        return cached;
    }

    for (size_t i = 0U; i < cached.size(); ++i) {
        std::vector<FamilyId> keep_families;
        for (size_t future_i = i + 1U; future_i < cached.size(); ++future_i) {
            std::vector<FamilyId> trial = keep_families;
            if (!bc_family_try_add_pass_families(
                    trial,
                    cached[future_i].pass.future_families,
                    max_future_families)) {
                break;
            }
            keep_families = std::move(trial);
            bc_family_add_unique_cells(cached[i].keep_cids, cached[future_i].pass.future_cids);
        }
        cached[i].keep_families = std::move(keep_families);
        bc_family_sort_unique_cells(cached[i].keep_cids);
    }
    return cached;
}

inline void bc_family_add_future_window_stats(
    BCSingleChunkSolveStats &dst,
    const BCFutureFamilyWindowStats &src,
    bool spawn4
) {
    BCCellLoadStats &position = spawn4 ? dst.future4_position_load : dst.future2_position_load;
    BCSuccessLoadStats &success = spawn4 ? dst.future4_success_load : dst.future2_success_load;
    uint64_t &batch_loads = spawn4 ? dst.future4_batch_loads : dst.future2_batch_loads;
    uint64_t &cells_loaded = spawn4 ? dst.future4_cells_loaded : dst.future2_cells_loaded;
    uint64_t &active_cells_max = spawn4 ? dst.future4_active_cells_max : dst.future2_active_cells_max;
    double &position_seconds = spawn4 ? dst.future4_position_read_seconds : dst.future2_position_read_seconds;
    double &success_seconds = spawn4 ? dst.future4_success_read_seconds : dst.future2_success_read_seconds;

    batch_loads += src.future_views_loaded;
    cells_loaded += src.future_cells_loaded;
    active_cells_max = std::max(active_cells_max, src.active_cells_max);

    position.requested_extents += src.position_requested_extents;
    position.coalesced_extents += src.position_coalesced_extents;
    position.requested_bytes += src.position_requested_bytes;
    position.read_bytes += src.position_read_bytes;
    position.backend_read_ops += src.position_backend_read_ops;
    position.backend_read_bytes += src.position_backend_read_bytes;
    position.backend_read_seconds += src.position_backend_read_seconds;

    success.requested_extents += src.success_requested_extents;
    success.coalesced_extents += src.success_coalesced_extents;
    success.requested_bytes += src.success_requested_bytes;
    success.read_bytes += src.success_read_bytes;
    success.backend_read_ops += src.success_backend_read_ops;
    success.backend_read_bytes += src.success_backend_read_bytes;
    success.backend_read_seconds += src.success_backend_read_seconds;

    position_seconds += src.position_read_seconds;
    success_seconds += src.success_read_seconds;
}

template <typename StorageT>
[[nodiscard]] BCFutureSuccessLookupView<StorageT> bc_family_make_empty_future_lookup(
    const BCLut &lut,
    uint32_t cell_count,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    BCFutureSuccessLookupView<StorageT> lookup;
    std::vector<BCLoadedCell> position_cells;
    const std::vector<BCLoadedSuccessCell> success_cells;
    lookup.open_loaded(
        lut,
        cell_count,
        std::move(position_cells),
        success_cells,
        row_width,
        dtype
    );
    return lookup;
}

template <typename StorageT>
void bc_family_spawn4_cell(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCFamilySolveCellVisitKind visit,
    const BCFamilyPartialCellLayout &layout,
    const BCFamilySolveOptions<StorageT> &options,
    const BCFamilyTable &future_axis,
    const BCSolveTargetFamilyFilter &filter,
    const BCFutureSuccessLookupView<StorageT> &lookup,
    detail::BCFamilySolveTempStore<StorageT> &temp,
    BCFamilySolveWorkspace<StorageT> &workspace,
    BCFamilySolveStats &stats,
    BCFamilyValueVector<StorageT> *prefetched_partial = nullptr,
    BCFamilyValueVector<StorageT> *temp_output = nullptr
) {
    if (cell.success_rows == 0U || cell.buckets.empty()) {
        return;
    }
    BCFamilyPartialMaxCellBuffer<StorageT> partial;
    BCFamilyCellSuccessScratch<StorageT> scratch;
    const bool first = visit == BCFamilySolveCellVisitKind::FirstDirection;
    const bool second = visit == BCFamilySolveCellVisitKind::SecondDirection;
    if (first) {
        partial.reset(layout, options.solve.zero_value);
    } else if (second) {
        BCFamilyValueVector<StorageT> stored_partial;
        if (prefetched_partial != nullptr) {
            stored_partial = std::move(*prefetched_partial);
        } else {
            stored_partial = detail::bc_family_sparse_partial_temp_enabled(options)
                ? temp.read_partial4_any(cell.cid, stats)
                : temp.read_partial4(cell.cid, layout.value_count, stats);
        }
        partial.values() = detail::bc_family_decode_partial_temp_values(
            layout,
            std::move(stored_partial),
            options.solve.zero_value,
            options
        );
        if (partial.values().size() != layout.value_count) {
            throw std::runtime_error("BC family spawn4 prefetched partial count mismatch");
        }
        scratch.reset(cell.cid, cell.success_rows, options.solve.row_width, options.solve.zero_value);
    } else {
        scratch.reset(cell.cid, cell.success_rows, options.solve.row_width, options.solve.zero_value);
    }

    if constexpr (std::is_integral_v<StorageT>) {
        if (second && options.solve.row_width == 1U) {
            detail::bc_family_scan_cell_phase_partial_sum_batches<StorageT>(
                lut,
                cell,
                directions,
                BCSolveSpawnPhase::Spawn4,
                filter,
                future_axis,
                lookup,
                options,
                layout,
                partial,
                workspace,
                stats,
                [&](uint32_t local_success_row,
                    uint32_t /*local_bucket_index*/,
                    const BCSolveBoardQuerySummary &summary,
                    uint64_t sum,
                    BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
                if (summary.terminal_success || summary.empty_count == 0U) {
                    scratch.write_zero_row(local_success_row, options.solve.zero_value);
                } else {
                    scratch.write_spawn4_sum_contribution(
                        local_success_row,
                        sum,
                        summary.empty_count,
                        options.solve.edge_options.spawn_rate4,
                        options.solve.zero_value
                    );
                }
            });
            if (temp_output != nullptr) {
                *temp_output = std::move(scratch.values());
            } else {
                temp.write_scratch4(cell.cid, scratch.values(), stats);
            }
            return;
        }
    }

    if (options.use_diagonal_grouped_sum && !first && !second && options.solve.row_width == 1U) {
        detail::bc_family_scan_cell_phase_sum_batches<StorageT>(
            lut,
            cell,
            directions,
            BCSolveSpawnPhase::Spawn4,
            filter,
            future_axis,
            lookup,
            options,
            workspace,
            stats,
            [&](uint32_t local_success_row,
                uint32_t /*local_bucket_index*/,
                const BCSolveBoardQuerySummary &summary,
                uint64_t sum,
                BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
            if (summary.terminal_success || summary.empty_count == 0U) {
                scratch.write_zero_row(local_success_row, options.solve.zero_value);
            } else {
                scratch.write_spawn4_sum_contribution(
                    local_success_row,
                    sum,
                    summary.empty_count,
                    options.solve.edge_options.spawn_rate4,
                    options.solve.zero_value
                );
            }
        });
    } else {
        detail::bc_family_scan_cell_phase_batches<StorageT>(
            lut,
            cell,
            directions,
            BCSolveSpawnPhase::Spawn4,
            filter,
            future_axis,
            lookup,
            options,
            workspace,
            stats,
            [&](uint32_t local_success_row,
                uint32_t local_bucket_index,
                const BCSolveBoardQuerySummary &summary,
                const StorageT *best,
                BCFamilySolveCellWorkspace<StorageT> &local) {
            const BCFamilyPartialBucketLayout &partial_bucket =
                layout.buckets[static_cast<size_t>(local_bucket_index)];
            if (first) {
                if (summary.terminal_success || summary.empty_count == 0U) {
                    return;
                }
                if (summary.empty_count != partial_bucket.empty_count) {
                    throw std::logic_error("BC family spawn4 compact partial empty count mismatch");
                }
                partial.write_compact_success_row(layout, partial_bucket, local_success_row, best);
                return;
            }
            if (second) {
                if (summary.terminal_success || summary.empty_count == 0U) {
                    scratch.write_zero_row(local_success_row, options.solve.zero_value);
                } else {
                    if (summary.empty_count != partial_bucket.empty_count) {
                        throw std::logic_error("BC family spawn4 compact merge empty count mismatch");
                    }
                    if constexpr (std::is_integral_v<StorageT>) {
                        if (options.solve.row_width == 1U) {
                            const StorageT *previous =
                                partial.compact_success_row_data(
                                    layout,
                                    partial_bucket,
                                    local_success_row
                                );
                            const uint64_t sum =
                                detail::bc_family_sum_merged_compact_best_u64(
                                    best,
                                    previous,
                                    summary.empty_count
                                );
                            scratch.write_spawn4_sum_contribution(
                                local_success_row,
                                sum,
                                summary.empty_count,
                                options.solve.edge_options.spawn_rate4,
                                options.solve.zero_value
                            );
                            return;
                        }
                    }
                    const size_t compact_values =
                        static_cast<size_t>(summary.empty_count) * options.solve.row_width;
                    local.merged_best.resize(compact_values);
                    partial.read_compact_success_row(
                        layout,
                        partial_bucket,
                        local_success_row,
                        local.merged_best.data(),
                        options.solve.zero_value
                    );
                    detail::bc_family_merge_compact_best(
                        summary.empty_count,
                        options.solve.row_width,
                        best,
                        local.merged_best.data()
                    );
                    scratch.write_spawn4_compact_contribution(
                        local_success_row,
                        local.merged_best.data(),
                        summary.empty_count,
                        options.solve.edge_options.spawn_rate4,
                        options.solve.zero_value
                    );
                }
                return;
            }
            if (summary.terminal_success || summary.empty_count == 0U) {
                scratch.write_zero_row(local_success_row, options.solve.zero_value);
            } else {
                if constexpr (std::is_integral_v<StorageT>) {
                    if (options.solve.row_width == 1U) {
                        const uint64_t sum =
                            detail::bc_family_sum_compact_best_u64(
                                best,
                                summary.empty_count
                            );
                        scratch.write_spawn4_sum_contribution(
                            local_success_row,
                            sum,
                            summary.empty_count,
                            options.solve.edge_options.spawn_rate4,
                            options.solve.zero_value
                        );
                        return;
                    }
                }
                scratch.write_spawn4_compact_contribution(
                    local_success_row,
                    best,
                    summary.empty_count,
                    options.solve.edge_options.spawn_rate4,
                    options.solve.zero_value
                );
            }
        });
    }

    if (first) {
        if (temp_output != nullptr) {
            *temp_output = std::move(partial.values());
        } else {
            temp.write_partial4(cell.cid, partial.values(), stats);
        }
    } else {
        if (temp_output != nullptr) {
            *temp_output = std::move(scratch.values());
        } else {
            temp.write_scratch4(cell.cid, scratch.values(), stats);
        }
    }
}

template <typename StorageT>
void bc_family_spawn2_cell(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCDirectionMask directions,
    BCFamilySolveCellVisitKind visit,
    const BCFamilyPartialCellLayout &layout,
    const BCFamilySolveOptions<StorageT> &options,
    const BCFamilyTable &future_axis,
    const BCSolveTargetFamilyFilter &filter,
    const BCFutureSuccessLookupView<StorageT> &lookup,
    detail::BCFamilySolveTempStore<StorageT> &temp,
    BCFamilySolveWorkspace<StorageT> &workspace,
    std::vector<detail::BCFamilyPendingOutputCell<StorageT>> &pending,
    BCFamilySolveStats &stats,
    BCFamilyValueVector<StorageT> *dense_output = nullptr,
    BCFamilyValueVector<StorageT> *prefetched_partial = nullptr,
    BCFamilyValueVector<StorageT> *prefetched_scratch4 = nullptr,
    BCFamilyValueVector<StorageT> *temp_output = nullptr
) {
    if (cell.success_rows == 0U || cell.buckets.empty()) {
        return;
    }
    BCFamilyPartialMaxCellBuffer<StorageT> partial;
    BCFamilyCellSuccessScratch<StorageT> final_values;
    const bool first = visit == BCFamilySolveCellVisitKind::FirstDirection;
    const bool second = visit == BCFamilySolveCellVisitKind::SecondDirection;
    if (first) {
        partial.reset(layout, options.solve.zero_value);
    } else if (second) {
        BCFamilyValueVector<StorageT> stored_partial;
        if (prefetched_partial != nullptr) {
            stored_partial = std::move(*prefetched_partial);
        } else {
            stored_partial = detail::bc_family_sparse_partial_temp_enabled(options)
                ? temp.read_partial2_any(cell.cid, stats)
                : temp.read_partial2(cell.cid, layout.value_count, stats);
        }
        partial.values() = detail::bc_family_decode_partial_temp_values(
            layout,
            std::move(stored_partial),
            options.solve.zero_value,
            options
        );
        if (partial.values().size() != layout.value_count) {
            throw std::runtime_error("BC family spawn2 prefetched partial count mismatch");
        }
        if (prefetched_scratch4 != nullptr) {
            final_values.adopt(
                cell.cid,
                cell.success_rows,
                options.solve.row_width,
                std::move(*prefetched_scratch4)
            );
        } else {
            final_values.adopt(
                cell.cid,
                cell.success_rows,
                options.solve.row_width,
                temp.read_scratch4(
                    cell.cid,
                    static_cast<uint64_t>(cell.success_rows) * options.solve.row_width,
                    stats
                )
            );
        }
    } else {
        if (prefetched_scratch4 != nullptr) {
            final_values.adopt(
                cell.cid,
                cell.success_rows,
                options.solve.row_width,
                std::move(*prefetched_scratch4)
            );
        } else {
            final_values.adopt(
                cell.cid,
                cell.success_rows,
                options.solve.row_width,
                temp.read_scratch4(
                    cell.cid,
                    static_cast<uint64_t>(cell.success_rows) * options.solve.row_width,
                    stats
                )
            );
        }
    }

    bool used_partial_sum_path = false;
    if constexpr (std::is_integral_v<StorageT>) {
        if (second && options.solve.row_width == 1U) {
            detail::bc_family_scan_cell_phase_partial_sum_batches<StorageT>(
                lut,
                cell,
                directions,
                BCSolveSpawnPhase::Spawn2,
                filter,
                future_axis,
                lookup,
                options,
                layout,
                partial,
                workspace,
                stats,
                [&](uint32_t local_success_row,
                    uint32_t /*local_bucket_index*/,
                    const BCSolveBoardQuerySummary &summary,
                    uint64_t sum,
                    BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
                if (summary.terminal_success) {
                    final_values.write_terminal_row(local_success_row, options.solve.terminal_value);
                } else if (summary.empty_count == 0U) {
                    final_values.write_zero_row(local_success_row, options.solve.zero_value);
                } else {
                    final_values.finalize_spawn2_sum_row(
                        local_success_row,
                        sum,
                        summary.empty_count,
                        options.solve.edge_options.spawn_rate4,
                        options.solve.zero_value
                    );
                }
            });
            used_partial_sum_path = true;
        }
    }

    if (!used_partial_sum_path &&
        options.use_diagonal_grouped_sum && !first && !second && options.solve.row_width == 1U) {
        detail::bc_family_scan_cell_phase_sum_batches<StorageT>(
            lut,
            cell,
            directions,
            BCSolveSpawnPhase::Spawn2,
            filter,
            future_axis,
            lookup,
            options,
            workspace,
            stats,
            [&](uint32_t local_success_row,
                uint32_t /*local_bucket_index*/,
                const BCSolveBoardQuerySummary &summary,
                uint64_t sum,
                BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
            if (summary.terminal_success) {
                final_values.write_terminal_row(local_success_row, options.solve.terminal_value);
            } else if (summary.empty_count == 0U) {
                final_values.write_zero_row(local_success_row, options.solve.zero_value);
            } else {
                final_values.finalize_spawn2_sum_row(
                    local_success_row,
                    sum,
                    summary.empty_count,
                    options.solve.edge_options.spawn_rate4,
                    options.solve.zero_value
                );
            }
        });
    } else if (!used_partial_sum_path) {
        detail::bc_family_scan_cell_phase_batches<StorageT>(
            lut,
            cell,
            directions,
            BCSolveSpawnPhase::Spawn2,
            filter,
            future_axis,
            lookup,
            options,
            workspace,
            stats,
            [&](uint32_t local_success_row,
                uint32_t local_bucket_index,
                const BCSolveBoardQuerySummary &summary,
                const StorageT *best,
                BCFamilySolveCellWorkspace<StorageT> &local) {
            const BCFamilyPartialBucketLayout &partial_bucket =
                layout.buckets[static_cast<size_t>(local_bucket_index)];
            if (first) {
                if (summary.terminal_success || summary.empty_count == 0U) {
                    return;
                }
                if (summary.empty_count != partial_bucket.empty_count) {
                    throw std::logic_error("BC family spawn2 compact partial empty count mismatch");
                }
                partial.write_compact_success_row(layout, partial_bucket, local_success_row, best);
                return;
            }
            const StorageT *best_for_final = best;
            if (second) {
                if (!summary.terminal_success && summary.empty_count != 0U) {
                    if (summary.empty_count != partial_bucket.empty_count) {
                        throw std::logic_error("BC family spawn2 compact merge empty count mismatch");
                    }
                    if constexpr (std::is_integral_v<StorageT>) {
                        if (options.solve.row_width == 1U) {
                            const StorageT *previous =
                                partial.compact_success_row_data(
                                    layout,
                                    partial_bucket,
                                    local_success_row
                                );
                            const uint64_t sum =
                                detail::bc_family_sum_merged_compact_best_u64(
                                    best,
                                    previous,
                                    summary.empty_count
                                );
                            final_values.finalize_spawn2_sum_row(
                                local_success_row,
                                sum,
                                summary.empty_count,
                                options.solve.edge_options.spawn_rate4,
                                options.solve.zero_value
                            );
                            return;
                        }
                    }
                    const size_t compact_values =
                        static_cast<size_t>(summary.empty_count) * options.solve.row_width;
                    local.merged_best.resize(compact_values);
                    partial.read_compact_success_row(
                        layout,
                        partial_bucket,
                        local_success_row,
                        local.merged_best.data(),
                        options.solve.zero_value
                    );
                    detail::bc_family_merge_compact_best(
                        summary.empty_count,
                        options.solve.row_width,
                        best,
                        local.merged_best.data()
                    );
                    best_for_final = local.merged_best.data();
                }
            }
            if (summary.terminal_success) {
                final_values.write_terminal_row(local_success_row, options.solve.terminal_value);
            } else if (summary.empty_count == 0U) {
                final_values.write_zero_row(local_success_row, options.solve.zero_value);
            } else {
                if constexpr (std::is_integral_v<StorageT>) {
                    if (options.solve.row_width == 1U) {
                        const uint64_t sum =
                            detail::bc_family_sum_compact_best_u64(
                                best_for_final,
                                summary.empty_count
                            );
                        final_values.finalize_spawn2_sum_row(
                            local_success_row,
                            sum,
                            summary.empty_count,
                            options.solve.edge_options.spawn_rate4,
                            options.solve.zero_value
                        );
                        return;
                    }
                }
                final_values.finalize_spawn2_compact_row(
                    local_success_row,
                    best_for_final,
                    summary.empty_count,
                    options.solve.edge_options.spawn_rate4,
                    options.solve.zero_value
                );
            }
        });
    }

    if (first) {
        if (temp_output != nullptr) {
            *temp_output = std::move(partial.values());
        } else {
            temp.write_partial2(cell.cid, partial.values(), stats);
        }
        return;
    }
    if (dense_output != nullptr) {
        *dense_output = std::move(final_values.values());
        return;
    }
    detail::bc_family_compact_and_mark_ready(
        lut,
        cell,
        final_values.values(),
        options.solve.row_width,
        options.solve.zero_value,
        workspace,
        pending,
        stats
    );
}

template <typename StorageT>
BCFamilySolveFileResult bc_family_solve_layer_to_files(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCFamilyPartitionLayerMap &current_partition,
    const BCFamilyPartitionLayerMap &future2_partition,
    const BCFamilyPartitionLayerMap &future4_partition,
    BCWritableFile &position_file,
    BCWritableFile &success_file,
    const std::filesystem::path &temp_dir,
    const BCFamilySolveOptions<StorageT> &input_options,
    BCFamilySolveWorkspace<StorageT> *workspace = nullptr
) {
    BCFamilySolveOptions<StorageT> options =
        bc_family_effective_options(current_position.lut(), current_partition, input_options);
    bc_family_validate_solve_inputs<StorageT>(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        current_partition,
        future2_partition,
        future4_partition,
        options
    );

    BCFamilySolveStats stats;
    const double temp_prepare_t0 = bc_single_chunk_now_seconds();
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_dir, cleanup_ec);
    std::filesystem::create_directories(temp_dir);
    stats.single.temp_prepare_seconds += bc_single_chunk_now_seconds() - temp_prepare_t0;

    const double plan_t0 = bc_single_chunk_now_seconds();
    const BCLut &lut = current_position.lut();
    const SpawnDeltaCoord delta2 = bc_family_solve_delta_for_spawn_rank(
        lut,
        current_position.axis(),
        options.solve.edge_options.spawn2_tile_rank
    );
    const SpawnDeltaCoord delta4 = bc_family_solve_delta_for_spawn_rank(
        lut,
        current_position.axis(),
        options.solve.edge_options.spawn4_tile_rank
    );
    BCFamilySolvePlanner planner2(
        current_position.axis(),
        current_partition,
        future2_position.axis(),
        future2_partition
    );
    BCFamilySolvePlanner planner4(
        current_position.axis(),
        current_partition,
        future4_position.axis(),
        future4_partition
    );
    const uint32_t current_family_count = current_position.axis().family_count();
    const std::vector<BCFamilySolvePassPlan> spawn4_passes =
        bc_family_build_passes(planner4, BCSolveSpawnPhase::Spawn4, delta4, current_family_count);
    const std::vector<BCFamilySolveCachedPass> spawn4_cached =
        bc_family_build_cached_passes(spawn4_passes, options.future_reuse_max_families);
    const std::vector<BCFamilySolvePassPlan> spawn2_passes =
        bc_family_build_passes(planner2, BCSolveSpawnPhase::Spawn2, delta2, current_family_count);
    const std::vector<BCFamilySolveCachedPass> spawn2_cached =
        bc_family_build_cached_passes(spawn2_passes, options.future_reuse_max_families);
    stats.plan_seconds += bc_single_chunk_now_seconds() - plan_t0;

    const double workspace_t0 = bc_single_chunk_now_seconds();
    BCFamilySolveWorkspace<StorageT> local_workspace;
    BCFamilySolveWorkspace<StorageT> &scratch =
        workspace == nullptr ? local_workspace : *workspace;
    stats.workspace_prepare_seconds += bc_single_chunk_now_seconds() - workspace_t0;

    const double temp_open_t0 = bc_single_chunk_now_seconds();
    const bool temp_direct_io =
        !options.force_temp_buffered_io &&
        (options.temp_direct_io ||
         position_file.mode() == BCFileIOMode::Direct ||
         success_file.mode() == BCFileIOMode::Direct);
    detail::BCFamilySolveTempStore<StorageT> temp(
        temp_dir,
        current_position.cell_count(),
        temp_direct_io,
        options.temp_direct_queue_depth
    );
    stats.temp_open_seconds += bc_single_chunk_now_seconds() - temp_open_t0;
    std::vector<detail::BCFamilyPendingOutputCell<StorageT>> pending(current_position.cell_count());
    uint64_t pending_value_bytes = 0U;

    if (options.interleave_spawn_phases) {
        if (spawn4_cached.size() != spawn2_cached.size()) {
            throw std::logic_error("BC family interleaved solve pass count mismatch");
        }

        const double output_streamer_t0 = bc_single_chunk_now_seconds();
        const double position_write_before = stats.single.position_write_seconds;
        const double success_write_before = stats.single.success_write_seconds;
        BCSingleChunkFinalFileStreamer<StorageT> output_streamer(
            current_position,
            position_file,
            success_file,
            options.solve.row_width,
            options.solve.dtype,
            stats.single
        );
        stats.output_streamer_open_seconds += detail::bc_family_positive_remainder(
            bc_single_chunk_now_seconds() - output_streamer_t0,
            (stats.single.position_write_seconds - position_write_before) +
                (stats.single.success_write_seconds - success_write_before)
        );
        const double mark_empty_t0 = bc_single_chunk_now_seconds();
        detail::bc_family_mark_empty_outputs(current_position, pending, output_streamer, stats);
        stats.mark_empty_seconds += bc_single_chunk_now_seconds() - mark_empty_t0;
        CellId next_output_cid = 0U;
        detail::bc_family_flush_ready_outputs(
            pending,
            pending_value_bytes,
            next_output_cid,
            output_streamer,
            stats
        );

        BCFutureFamilyWindowOptions future_window_options;
        future_window_options.max_recycled_index_bytes =
            options.future_index_recycle_max_bytes;
        future_window_options.release_threads =
            static_cast<uint32_t>(bc_resident_solve_effective_threads(options.solve.num_threads));
        BCFutureFamilyWindow<StorageT> future4_window(
            future4_position,
            future4_success,
            future_window_options
        );
        BCFutureFamilyWindow<StorageT> future2_window(
            future2_position,
            future2_success,
            future_window_options
        );
        auto open_interleaved_lookup = [&](
            BCFutureFamilyWindow<StorageT> &window,
            const BCPositionStreamingReader &future_position,
            const BCFamilySolvePassPlan &pass,
            bool spawn4
        ) {
            BCFutureSuccessLookupView<StorageT> lookup =
                bc_family_make_empty_future_lookup<StorageT>(
                    future_position.lut(),
                    future_position.cell_count(),
                    options.solve.row_width,
                    options.solve.dtype
                );
            if (pass.future_cids.empty()) {
                return lookup;
            }

            if (spawn4) {
                ++stats.spawn4_future_reuse_groups;
            } else {
                ++stats.spawn2_future_reuse_groups;
            }
            const BCFutureFamilyWindowStats before_prepare = window.stats();
            const double prepare_t0 = bc_single_chunk_now_seconds();
            window.prepare_cells(pass.future_cids);
            const double prepare_elapsed = bc_single_chunk_now_seconds() - prepare_t0;
            const BCFutureFamilyWindowStats after_prepare = window.stats();
            const double prepare_io_seconds =
                (after_prepare.position_read_seconds - before_prepare.position_read_seconds) +
                (after_prepare.success_read_seconds - before_prepare.success_read_seconds);
            if (spawn4) {
                stats.future4_prepare_overhead_seconds +=
                    detail::bc_family_positive_remainder(prepare_elapsed, prepare_io_seconds);
                stats.future4_prepare_normalize_seconds +=
                    after_prepare.prepare_normalize_seconds -
                    before_prepare.prepare_normalize_seconds;
                stats.future4_prepare_select_seconds +=
                    after_prepare.prepare_select_seconds -
                    before_prepare.prepare_select_seconds;
                stats.future4_prepare_index_build_seconds +=
                    after_prepare.prepare_index_build_seconds -
                    before_prepare.prepare_index_build_seconds;
                stats.future4_prepare_insert_sort_seconds +=
                    after_prepare.prepare_insert_sort_seconds -
                    before_prepare.prepare_insert_sort_seconds;
            } else {
                stats.future2_prepare_overhead_seconds +=
                    detail::bc_family_positive_remainder(prepare_elapsed, prepare_io_seconds);
                stats.future2_prepare_normalize_seconds +=
                    after_prepare.prepare_normalize_seconds -
                    before_prepare.prepare_normalize_seconds;
                stats.future2_prepare_select_seconds +=
                    after_prepare.prepare_select_seconds -
                    before_prepare.prepare_select_seconds;
                stats.future2_prepare_index_build_seconds +=
                    after_prepare.prepare_index_build_seconds -
                    before_prepare.prepare_index_build_seconds;
                stats.future2_prepare_insert_sort_seconds +=
                    after_prepare.prepare_insert_sort_seconds -
                    before_prepare.prepare_insert_sort_seconds;
            }

            uint64_t position_bytes = 0U;
            uint64_t success_bytes = 0U;
            double &index_seconds = spawn4
                ? stats.single.future4_index_seconds
                : stats.single.future2_index_seconds;
            const double index_before = index_seconds;
            const double lookup_t0 = bc_single_chunk_now_seconds();
            lookup = window.open_success_lookup(
                options.solve.row_width,
                options.solve.dtype,
                index_seconds,
                position_bytes,
                success_bytes
            );
            const double lookup_copy = detail::bc_family_positive_remainder(
                bc_single_chunk_now_seconds() - lookup_t0,
                index_seconds - index_before
            );
            if (spawn4) {
                stats.future4_lookup_copy_seconds += lookup_copy;
            } else {
                stats.future2_lookup_copy_seconds += lookup_copy;
            }

            const uint64_t resident_position_bytes =
                window.active_position_resident_bytes() +
                window.recycled_index_resident_bytes() +
                position_bytes;
            const uint64_t resident_success_bytes =
                window.active_success_resident_bytes() + success_bytes;
            if (spawn4) {
                stats.single.future4_position_resident_bytes = std::max(
                    stats.single.future4_position_resident_bytes,
                    resident_position_bytes
                );
                stats.single.future4_success_resident_bytes = std::max(
                    stats.single.future4_success_resident_bytes,
                    resident_success_bytes
                );
            } else {
                stats.single.future2_position_resident_bytes = std::max(
                    stats.single.future2_position_resident_bytes,
                    resident_position_bytes
                );
                stats.single.future2_success_resident_bytes = std::max(
                    stats.single.future2_success_resident_bytes,
                    resident_success_bytes
                );
            }
            stats.single.future_resident_layers_max = std::max<uint64_t>(
                stats.single.future_resident_layers_max,
                1U
            );
            stats.single.future_resident_bytes_max = std::max<uint64_t>(
                stats.single.future_resident_bytes_max,
                resident_position_bytes + resident_success_bytes
            );
            return lookup;
        };

        auto release_future_window_all = [&](BCFutureFamilyWindow<StorageT> &window) {
            const BCFutureFamilyWindowStats before = window.stats();
            const double release_t0 = bc_single_chunk_now_seconds();
            window.release_all();
            const double elapsed = bc_single_chunk_now_seconds() - release_t0;
            const BCFutureFamilyWindowStats after = window.stats();
            stats.single.future_release_seconds += elapsed;
            stats.future_release_all_seconds += elapsed;
            stats.future_release_all_clear_seconds +=
                after.release_all_clear_seconds - before.release_all_clear_seconds;
            ++stats.future_release_all_calls;
        };
        auto release_future_window_except = [&](
            BCFutureFamilyWindow<StorageT> &window,
            const std::vector<CellId> &keep_cids
        ) {
            const BCFutureFamilyWindowStats before = window.stats();
            const double release_t0 = bc_single_chunk_now_seconds();
            window.release_except(keep_cids);
            const double elapsed = bc_single_chunk_now_seconds() - release_t0;
            const BCFutureFamilyWindowStats after = window.stats();
            stats.single.future_release_seconds += elapsed;
            stats.future_release_except_seconds += elapsed;
            stats.future_release_except_normalize_seconds +=
                after.release_except_normalize_seconds -
                before.release_except_normalize_seconds;
            stats.future_release_except_filter_seconds +=
                after.release_except_filter_seconds -
                before.release_except_filter_seconds;
            stats.future_release_except_erase_seconds +=
                after.release_except_erase_seconds -
                before.release_except_erase_seconds;
            stats.future_release_except_ids_seconds +=
                after.release_except_ids_seconds -
                before.release_except_ids_seconds;
            ++stats.future_release_except_calls;
        };
        auto interleaved_phase_accounted_seconds = [&](bool spawn4) {
            double total =
                stats.single.current_position_read_seconds +
                stats.temp_write_seconds +
                stats.temp_read_prepare_seconds +
                stats.temp_read_seconds +
                stats.single.future_release_seconds +
                stats.single.workspace_release_seconds +
                stats.current_layout_seconds;
            if (spawn4) {
                const BCFutureFamilyWindowStats future_window_stats = future4_window.stats();
                total +=
                    future_window_stats.position_read_seconds +
                    future_window_stats.success_read_seconds +
                    stats.single.future4_index_seconds +
                    stats.future4_prepare_overhead_seconds +
                    stats.future4_lookup_copy_seconds +
                    stats.spawn4_cell_compute_seconds;
            } else {
                const BCFutureFamilyWindowStats future_window_stats = future2_window.stats();
                total +=
                    future_window_stats.position_read_seconds +
                    future_window_stats.success_read_seconds +
                    stats.single.future2_index_seconds +
                    stats.future2_prepare_overhead_seconds +
                    stats.future2_lookup_copy_seconds +
                    stats.spawn2_cell_compute_seconds +
                    stats.single.compact_seconds +
                    stats.single.result_assembly_seconds +
                    stats.output_pending_seconds +
                    stats.final_stage_write_seconds +
                    stats.final_stage_read_seconds +
                    stats.single.position_write_seconds +
                    stats.single.success_write_seconds +
                    stats.final_dense_copy_seconds +
                    stats.compact_value_copy_seconds +
                    stats.pending_mark_seconds +
                    stats.output_finish_seconds;
            }
            return total;
        };

        struct InterleavedBlockPassState {
            std::vector<BCLoadedCell> current_cells;
            std::vector<BCFamilyPartialCellLayout> current_layouts;
            std::vector<uint8_t> current_layout_ready;
            std::vector<CellId> local_scratch4_cids;
            std::vector<BCFamilyValueVector<StorageT>> local_scratch4_values;
            bool local_scratch4_spilled = false;
        };

        const size_t interleave_block_size = std::max<size_t>(
            1U,
            static_cast<size_t>(options.interleave_block_fids)
        );

        if (interleave_block_size >= 1U) {
            std::vector<InterleavedBlockPassState> block_states;
            uint64_t interleaved_block_base_resident_bytes = 0U;
            const bool collect_resident_breakdown = options.collect_resident_breakdown;
            const int release_threads =
                bc_resident_solve_effective_threads(options.solve.num_threads);
            auto scratch4_state_resident_bytes =
                [](const InterleavedBlockPassState &state) -> uint64_t {
                return static_cast<uint64_t>(state.local_scratch4_cids.capacity()) *
                        sizeof(CellId) +
                    detail::bc_family_value_vectors_resident_bytes(
                        state.local_scratch4_values
                    );
            };
            auto update_interleaved_block_resident_bytes = [&]() -> uint64_t {
                if (!collect_resident_breakdown) {
                    interleaved_block_base_resident_bytes = 0U;
                    return 0U;
                }
                uint64_t current_bytes =
                    static_cast<uint64_t>(block_states.capacity()) *
                    sizeof(InterleavedBlockPassState);
                uint64_t scratch4_bytes = 0U;
                for (const InterleavedBlockPassState &state : block_states) {
                    current_bytes += detail::bc_family_loaded_cells_resident_bytes(
                        state.current_cells
                    );
                    current_bytes +=
                        static_cast<uint64_t>(state.current_layouts.capacity()) *
                        sizeof(BCFamilyPartialCellLayout);
                    current_bytes +=
                        static_cast<uint64_t>(state.current_layout_ready.capacity()) *
                        sizeof(uint8_t);
                    scratch4_bytes += scratch4_state_resident_bytes(state);
                }
                stats.block_current_resident_bytes_max = std::max(
                    stats.block_current_resident_bytes_max,
                    current_bytes
                );
                stats.block_scratch4_resident_bytes_max = std::max(
                    stats.block_scratch4_resident_bytes_max,
                    scratch4_bytes
                );
                stats.block_total_resident_bytes_max = std::max(
                    stats.block_total_resident_bytes_max,
                    current_bytes + scratch4_bytes
                );
                interleaved_block_base_resident_bytes = current_bytes + scratch4_bytes;
                return interleaved_block_base_resident_bytes;
            };
            auto current_interleaved_scratch4_resident_bytes = [&]() -> uint64_t {
                uint64_t bytes = 0U;
                for (const InterleavedBlockPassState &state : block_states) {
                    bytes += scratch4_state_resident_bytes(state);
                }
                return bytes;
            };
            auto spill_interleaved_scratch4_state =
                [&](InterleavedBlockPassState &state) {
                if (state.local_scratch4_spilled || state.local_scratch4_values.empty()) {
                    return;
                }
                temp.write_scratch4_batch(
                    state.local_scratch4_cids,
                    state.local_scratch4_values,
                    stats
                );
                {
                    const double release_t0 = bc_single_chunk_now_seconds();
                    detail::bc_family_release_value_vectors(
                        state.local_scratch4_values,
                        release_threads
                    );
                    const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                    stats.single.workspace_release_seconds += elapsed;
                    stats.workspace_release_spawn4_temp_values_seconds += elapsed;
                }
                state.local_scratch4_spilled = true;
            };
            auto enforce_interleaved_scratch4_memory_cap = [&](uint64_t target_bytes) {
                const uint64_t cap = options.interleave_scratch4_memory_cap_bytes;
                if (cap == 0U) {
                    return;
                }
                if (target_bytes > cap) {
                    target_bytes = cap;
                }
                uint64_t bytes = current_interleaved_scratch4_resident_bytes();
                if (bytes <= target_bytes) {
                    return;
                }
                for (InterleavedBlockPassState &state : block_states) {
                    if (bytes <= target_bytes) {
                        break;
                    }
                    const uint64_t before = scratch4_state_resident_bytes(state);
                    if (before == 0U || state.local_scratch4_spilled) {
                        continue;
                    }
                    spill_interleaved_scratch4_state(state);
                    const uint64_t after = scratch4_state_resident_bytes(state);
                    bytes = bytes >= before - after ? bytes - (before - after) : 0U;
                }
            };
            for (size_t block_begin = 0U;
                 block_begin < spawn4_cached.size();
                 block_begin += interleave_block_size) {
                const size_t block_end = std::min(
                    spawn4_cached.size(),
                    block_begin + interleave_block_size
                );
                const size_t block_count = block_end - block_begin;
                if (block_states.size() < block_count) {
                    block_states.resize(block_count);
                }
                update_interleaved_block_resident_bytes();

                for (size_t pass_i = block_begin; pass_i < block_end; ++pass_i) {
                    if (options.interleave_scratch4_memory_cap_bytes != 0U &&
                        pass_i > block_begin) {
                        enforce_interleaved_scratch4_memory_cap(
                            options.interleave_scratch4_memory_cap_bytes / 2U
                        );
                        update_interleaved_block_resident_bytes();
                    }
                    const BCFamilySolveCachedPass &cached4 = spawn4_cached[pass_i];
                    const BCFamilySolveCachedPass &cached2 = spawn2_cached[pass_i];
                    const BCFamilySolvePassPlan &pass4 = cached4.pass;
                    const BCFamilySolvePassPlan &pass2 = cached2.pass;
                    if (pass4.fid != pass2.fid ||
                        cached4.current_cids != cached2.current_cids ||
                        pass4.current_cells.size() != pass2.current_cells.size()) {
                        throw std::logic_error("BC family interleaved solve pass alignment mismatch");
                    }
                    for (size_t i = 0U; i < pass4.current_cells.size(); ++i) {
                        const BCFamilySolveCellWork &lhs = pass4.current_cells[i];
                        const BCFamilySolveCellWork &rhs = pass2.current_cells[i];
                        if (lhs.cid != rhs.cid ||
                            lhs.directions != rhs.directions ||
                            lhs.visit != rhs.visit) {
                            throw std::logic_error("BC family interleaved solve cell alignment mismatch");
                        }
                    }

                    const double spawn4_phase_t0 = bc_single_chunk_now_seconds();
                    const double spawn4_phase_accounted_before =
                        interleaved_phase_accounted_seconds(true);
                    ++stats.spawn4_passes;
                    std::vector<size_t> spawn4_second_indices;
                    std::vector<CellId> spawn4_second_cids;
                    std::vector<int64_t> spawn4_scratch_slot_by_cell(
                        pass4.current_cells.size(),
                        -1
                    );
                    spawn4_second_indices.reserve(pass4.current_cells.size());
                    spawn4_second_cids.reserve(pass4.current_cells.size());
                    InterleavedBlockPassState &state = block_states[pass_i - block_begin];
                    state.local_scratch4_cids.clear();
                    state.local_scratch4_values.clear();
                    state.local_scratch4_cids.reserve(pass4.current_cells.size());
                    state.local_scratch4_values.reserve(pass4.current_cells.size());
                    for (size_t i = 0U; i < pass4.current_cells.size(); ++i) {
                        const BCFamilySolveCellWork &work = pass4.current_cells[i];
                        if (work.visit == BCFamilySolveCellVisitKind::FirstDirection) {
                            continue;
                        }
                        if (current_position.descriptor(work.cid).empty()) {
                            continue;
                        }
                        spawn4_scratch_slot_by_cell[i] =
                            static_cast<int64_t>(state.local_scratch4_cids.size());
                        state.local_scratch4_cids.push_back(work.cid);
                        state.local_scratch4_values.emplace_back();
                        if (work.visit == BCFamilySolveCellVisitKind::SecondDirection) {
                            spawn4_second_indices.push_back(i);
                            spawn4_second_cids.push_back(work.cid);
                        }
                    }
                    std::vector<uint8_t> spawn4_scratch_ready(
                        state.local_scratch4_values.size(),
                        0U
                    );
                    std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_values;
                    temp.read_partial4_batch(spawn4_second_cids, spawn4_partial_values, stats);
                    uint64_t spawn4_partial_prefetch_bytes = 0U;
                    if (collect_resident_breakdown) {
                        spawn4_partial_prefetch_bytes =
                            detail::bc_family_value_vectors_resident_bytes(
                                spawn4_partial_values
                            );
                        stats.block_partial_prefetch_resident_bytes_max = std::max(
                            stats.block_partial_prefetch_resident_bytes_max,
                            spawn4_partial_prefetch_bytes
                        );
                        stats.block_total_resident_bytes_max = std::max(
                            stats.block_total_resident_bytes_max,
                            interleaved_block_base_resident_bytes +
                                spawn4_partial_prefetch_bytes
                        );
                    }
                    state.local_scratch4_spilled = false;
                    BCCellLoadStats current_load_stats;
                    double current_load_seconds = 0.0;
                    {
                        const double current_t0 = bc_single_chunk_now_seconds();
                        current_position.load_cells_into(
                            cached4.current_cids,
                            state.current_cells,
                            &current_load_stats
                        );
                        current_load_seconds = bc_single_chunk_now_seconds() - current_t0;
                    }
                    BCFutureSuccessLookupView<StorageT> lookup4 =
                        open_interleaved_lookup(future4_window, future4_position, pass4, true);

                    stats.single.current_position_read_seconds += current_load_seconds;
                    bc_single_chunk_add_cell_load_stats(
                        stats.single.current_position_load,
                        current_load_stats
                    );
                    stats.family_current_cells_loaded += state.current_cells.size();
                    ++stats.single.current_chunks;

                    std::vector<int64_t> spawn4_prefetch_index(state.current_cells.size(), -1);
                    for (size_t i = 0U; i < spawn4_second_indices.size(); ++i) {
                        spawn4_prefetch_index[spawn4_second_indices[i]] = static_cast<int64_t>(i);
                    }

                    state.current_layouts.resize(state.current_cells.size());
                    state.current_layout_ready.assign(state.current_cells.size(), 0U);
                    std::vector<CellId> spawn4_partial_write_cids;
                    std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_write_values;
                    spawn4_partial_write_cids.reserve(state.current_cells.size());
                    spawn4_partial_write_values.reserve(state.current_cells.size());
                    state.local_scratch4_cids.reserve(state.current_cells.size());
                    state.local_scratch4_values.reserve(state.current_cells.size());
                    const BCSolveTargetFamilyFilter spawn4_filter =
                        bc_family_make_solve_filter(pass4.future_families);
                    const bool spawn4_batch_first_enabled = options.solve.row_width == 1U;
                    std::vector<size_t> spawn4_first_horizontal_indices;
                    std::vector<size_t> spawn4_first_vertical_indices;
                    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> spawn4_first_partials;
                    const bool spawn4_batch_second_enabled =
                        std::is_integral_v<StorageT> && options.solve.row_width == 1U;
                    std::vector<size_t> spawn4_second_horizontal_indices;
                    std::vector<size_t> spawn4_second_vertical_indices;
                    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> spawn4_second_partials;
                    std::vector<BCFamilyCellSuccessScratch<StorageT>> spawn4_second_scratch;
                    if (spawn4_batch_first_enabled) {
                        spawn4_first_horizontal_indices.reserve(state.current_cells.size());
                        spawn4_first_vertical_indices.reserve(state.current_cells.size());
                        spawn4_first_partials.resize(state.current_cells.size());
                    }
                    if (spawn4_batch_second_enabled) {
                        spawn4_second_horizontal_indices.reserve(state.current_cells.size());
                        spawn4_second_vertical_indices.reserve(state.current_cells.size());
                        spawn4_second_partials.resize(state.current_cells.size());
                        spawn4_second_scratch.resize(state.current_cells.size());
                    }
                    update_interleaved_block_resident_bytes();
                    if (collect_resident_breakdown) {
                        stats.block_total_resident_bytes_max = std::max(
                            stats.block_total_resident_bytes_max,
                            interleaved_block_base_resident_bytes +
                                spawn4_partial_prefetch_bytes
                        );
                        const uint64_t spawn4_scratch_dense_bytes =
                            detail::bc_family_value_vectors_resident_bytes(
                                state.local_scratch4_values
                            );
                        stats.block_partial_prefetch_resident_bytes_max = std::max(
                            stats.block_partial_prefetch_resident_bytes_max,
                            spawn4_scratch_dense_bytes
                        );
                        stats.block_total_resident_bytes_max = std::max(
                            stats.block_total_resident_bytes_max,
                            interleaved_block_base_resident_bytes +
                                spawn4_partial_prefetch_bytes +
                                spawn4_scratch_dense_bytes
                        );
                    }
                    for (size_t i = 0U; i < state.current_cells.size(); ++i) {
                        const BCLoadedCell &cell = state.current_cells[i];
                        if (cell.success_rows == 0U || cell.buckets.empty()) {
                            ++stats.single.current_empty_cells;
                            continue;
                        }
                        ++stats.single.current_nonempty_cells;
                        stats.single.current_cells += 1U;
                        stats.single.current_rows += cell.success_rows;
                        stats.single.current_boards += cell.success_rows;
                        const double layout_t0 = bc_single_chunk_now_seconds();
                        state.current_layouts[i] = bc_family_make_partial_cell_layout(
                            lut,
                            cell.view(),
                            options.solve.row_width
                        );
                        state.current_layout_ready[i] = 1U;
                        stats.current_layout_seconds += bc_single_chunk_now_seconds() - layout_t0;
                        const BCFamilySolveCellWork &cell_work = pass4.current_cells[i];
                        if (spawn4_batch_first_enabled &&
                            cell_work.visit == BCFamilySolveCellVisitKind::FirstDirection) {
                            if (cell_work.directions == BCDirectionMask::Horizontal) {
                                spawn4_first_horizontal_indices.push_back(i);
                                continue;
                            }
                            if (cell_work.directions == BCDirectionMask::Vertical) {
                                spawn4_first_vertical_indices.push_back(i);
                                continue;
                            }
                        }
                        if (spawn4_batch_second_enabled &&
                            cell_work.visit == BCFamilySolveCellVisitKind::SecondDirection &&
                            (cell_work.directions == BCDirectionMask::Horizontal ||
                             cell_work.directions == BCDirectionMask::Vertical)) {
                            if (spawn4_prefetch_index[i] < 0) {
                                throw std::logic_error("BC family interleaved missing spawn4 partial prefetch");
                            }
                            BCFamilyValueVector<StorageT> stored_partial = std::move(
                                spawn4_partial_values[
                                    static_cast<size_t>(spawn4_prefetch_index[i])
                                ]
                            );
                            spawn4_second_partials[i].values() =
                                detail::bc_family_decode_partial_temp_values(
                                    state.current_layouts[i],
                                    std::move(stored_partial),
                                    options.solve.zero_value,
                                    options
                                );
                            if (spawn4_second_partials[i].values().size() !=
                                static_cast<size_t>(state.current_layouts[i].value_count)) {
                                throw std::runtime_error(
                                    "BC family spawn4 batched partial count mismatch"
                                );
                            }
                            spawn4_second_scratch[i].reset(
                                cell.cid,
                                cell.success_rows,
                                options.solve.row_width,
                                options.solve.zero_value
                            );
                            if (cell_work.directions == BCDirectionMask::Horizontal) {
                                spawn4_second_horizontal_indices.push_back(i);
                            } else {
                                spawn4_second_vertical_indices.push_back(i);
                            }
                            continue;
                        }
                        const double temp_read_before = stats.temp_read_seconds;
                        const double temp_write_before = stats.temp_write_seconds;
                        const double cell_t0 = bc_single_chunk_now_seconds();
                        BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
                        if (spawn4_prefetch_index[i] >= 0) {
                            prefetched_partial = &spawn4_partial_values[
                                static_cast<size_t>(spawn4_prefetch_index[i])
                            ];
                        }
                        BCFamilyValueVector<StorageT> temp_output_values;
                        bc_family_spawn4_cell(
                            lut,
                            cell,
                            cell_work.directions,
                            cell_work.visit,
                            state.current_layouts[i],
                            options,
                            future4_position.axis(),
                            spawn4_filter,
                            lookup4,
                            temp,
                            scratch,
                            stats,
                            prefetched_partial,
                            &temp_output_values
                        );
                        const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
                        const double temp_delta =
                            (stats.temp_read_seconds - temp_read_before) +
                            (stats.temp_write_seconds - temp_write_before);
                        stats.spawn4_cell_compute_seconds +=
                            detail::bc_family_positive_remainder(cell_elapsed, temp_delta);
                        if (cell_work.visit == BCFamilySolveCellVisitKind::FirstDirection) {
                            temp_output_values = detail::bc_family_prepare_partial_temp_output(
                                state.current_layouts[i],
                                std::move(temp_output_values),
                                true,
                                options,
                                stats
                            );
                            spawn4_partial_write_cids.push_back(cell.cid);
                            spawn4_partial_write_values.push_back(std::move(temp_output_values));
                        } else {
                            const int64_t slot = spawn4_scratch_slot_by_cell[i];
                            if (slot < 0) {
                                throw std::logic_error("BC family interleaved missing scratch4 slot");
                            }
                            state.local_scratch4_values[static_cast<size_t>(slot)] =
                                std::move(temp_output_values);
                            spawn4_scratch_ready[static_cast<size_t>(slot)] = 1U;
                        }
                    }
                    auto append_spawn4_first_batch = [&](const std::vector<size_t> &indices,
                                                         BCDirectionMask directions) {
                        if (indices.empty()) {
                            return;
                        }
                        const double batch_t0 = bc_single_chunk_now_seconds();
                        detail::bc_family_first_direction_cells_batch<StorageT>(
                            lut,
                            state.current_cells,
                            indices,
                            directions,
                            BCSolveSpawnPhase::Spawn4,
                            state.current_layouts,
                            future4_position.axis(),
                            lookup4,
                            spawn4_filter,
                            options,
                            scratch,
                            stats,
                            spawn4_first_partials
                        );
                        stats.spawn4_cell_compute_seconds +=
                            bc_single_chunk_now_seconds() - batch_t0;
                        for (size_t cell_index : indices) {
                            const BCLoadedCell &cell = state.current_cells[cell_index];
                            if (cell.success_rows == 0U || cell.buckets.empty()) {
                                continue;
                            }
                            BCFamilyValueVector<StorageT> temp_output_values =
                                detail::bc_family_prepare_partial_temp_output(
                                    state.current_layouts[cell_index],
                                    std::move(spawn4_first_partials[cell_index].values()),
                                    true,
                                    options,
                                    stats
                                );
                            spawn4_partial_write_cids.push_back(cell.cid);
                            spawn4_partial_write_values.push_back(std::move(temp_output_values));
                        }
                    };
                    append_spawn4_first_batch(
                        spawn4_first_horizontal_indices,
                        BCDirectionMask::Horizontal
                    );
                    append_spawn4_first_batch(
                        spawn4_first_vertical_indices,
                        BCDirectionMask::Vertical
                    );
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        detail::bc_family_release_partial_buffers(
                            spawn4_first_partials,
                            release_threads
                        );
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn4_partial_seconds += elapsed;
                    }
                    auto append_spawn4_second_batch = [&](const std::vector<size_t> &indices,
                                                          BCDirectionMask directions) {
                        if (indices.empty()) {
                            return;
                        }
                        if constexpr (std::is_integral_v<StorageT>) {
                            const double batch_t0 = bc_single_chunk_now_seconds();
                            detail::bc_family_partial_sum_cells_batch<StorageT>(
                                lut,
                                state.current_cells,
                                indices,
                                directions,
                                BCSolveSpawnPhase::Spawn4,
                                state.current_layouts,
                                spawn4_second_partials,
                                future4_position.axis(),
                                lookup4,
                                spawn4_filter,
                                options,
                                scratch,
                                stats,
                                [&](uint32_t cell_index,
                                    uint32_t local_success_row,
                                    uint32_t /*local_bucket_index*/,
                                    const BCSolveBoardQuerySummary &summary,
                                    uint64_t sum,
                                    BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
                                    BCFamilyCellSuccessScratch<StorageT> &cell_scratch =
                                        spawn4_second_scratch[static_cast<size_t>(cell_index)];
                                    if (summary.terminal_success || summary.empty_count == 0U) {
                                        cell_scratch.write_zero_row(
                                            local_success_row,
                                            options.solve.zero_value
                                        );
                                    } else {
                                        cell_scratch.write_spawn4_sum_contribution(
                                            local_success_row,
                                            sum,
                                            summary.empty_count,
                                            options.solve.edge_options.spawn_rate4,
                                            options.solve.zero_value
                                        );
                                    }
                                }
                            );
                            stats.spawn4_cell_compute_seconds +=
                                bc_single_chunk_now_seconds() - batch_t0;
                            for (size_t cell_index : indices) {
                                const int64_t slot = spawn4_scratch_slot_by_cell[cell_index];
                                if (slot < 0) {
                                    throw std::logic_error("BC family interleaved missing scratch4 slot");
                                }
                                state.local_scratch4_values[static_cast<size_t>(slot)] =
                                    std::move(spawn4_second_scratch[cell_index].values());
                                spawn4_scratch_ready[static_cast<size_t>(slot)] = 1U;
                            }
                        } else {
                            (void)directions;
                        }
                    };
                    append_spawn4_second_batch(
                        spawn4_second_horizontal_indices,
                        BCDirectionMask::Horizontal
                    );
                    append_spawn4_second_batch(
                        spawn4_second_vertical_indices,
                        BCDirectionMask::Vertical
                    );
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        detail::bc_family_release_partial_buffers(
                            spawn4_second_partials,
                            release_threads
                        );
                        detail::bc_family_release_success_scratch(
                            spawn4_second_scratch,
                            release_threads
                        );
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn4_scratch_seconds += elapsed;
                    }
                    for (uint8_t ready : spawn4_scratch_ready) {
                        if (ready == 0U) {
                            throw std::logic_error("BC family interleaved missing scratch4 output");
                        }
                    }
                    enforce_interleaved_scratch4_memory_cap(
                        options.interleave_scratch4_memory_cap_bytes
                    );
                    update_interleaved_block_resident_bytes();
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        std::vector<uint8_t>().swap(spawn4_scratch_ready);
                        detail::bc_family_release_value_vectors(
                            spawn4_partial_values,
                            release_threads
                        );
                        std::vector<int64_t>().swap(spawn4_prefetch_index);
                        std::vector<int64_t>().swap(spawn4_scratch_slot_by_cell);
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn4_temp_values_seconds += elapsed;
                    }
                    temp.write_partial4_batch(
                        spawn4_partial_write_cids,
                        spawn4_partial_write_values,
                        stats
                    );
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        std::vector<CellId>().swap(spawn4_partial_write_cids);
                        detail::bc_family_release_value_vectors(
                            spawn4_partial_write_values,
                            release_threads
                        );
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn4_temp_values_seconds += elapsed;
                    }
                    if (pass_i + 1U < block_end) {
                        release_future_window_except(future4_window, cached4.keep_cids);
                    } else if (options.interleave_keep_future4_between_blocks &&
                               pass_i + 1U < spawn4_cached.size()) {
                        release_future_window_except(future4_window, cached4.keep_cids);
                    } else {
                        release_future_window_all(future4_window);
                    }
                    const double spawn4_phase_elapsed =
                        bc_single_chunk_now_seconds() - spawn4_phase_t0;
                    stats.spawn4_phase_wall_seconds += spawn4_phase_elapsed;
                    stats.spawn4_phase_untracked_seconds +=
                        detail::bc_family_positive_remainder(
                            spawn4_phase_elapsed,
                            interleaved_phase_accounted_seconds(true) -
                                spawn4_phase_accounted_before
                        );
                }

                for (size_t pass_i = block_begin; pass_i < block_end; ++pass_i) {
                    const BCFamilySolveCachedPass &cached2 = spawn2_cached[pass_i];
                    const BCFamilySolvePassPlan &pass2 = cached2.pass;
                    InterleavedBlockPassState &state = block_states[pass_i - block_begin];
                    std::vector<BCLoadedCell> &current_cells = state.current_cells;

                    const double spawn2_phase_t0 = bc_single_chunk_now_seconds();
                    const double spawn2_phase_accounted_before =
                        interleaved_phase_accounted_seconds(false);
                    ++stats.spawn2_passes;
                    std::vector<size_t> spawn2_partial_indices;
                    std::vector<CellId> spawn2_partial_cids;
                    spawn2_partial_indices.reserve(pass2.current_cells.size());
                    spawn2_partial_cids.reserve(pass2.current_cells.size());
                    for (size_t i = 0U; i < pass2.current_cells.size(); ++i) {
                        const BCFamilySolveCellWork &work = pass2.current_cells[i];
                        if (current_position.descriptor(work.cid).empty()) {
                            continue;
                        }
                        if (work.visit == BCFamilySolveCellVisitKind::SecondDirection) {
                            spawn2_partial_indices.push_back(i);
                            spawn2_partial_cids.push_back(work.cid);
                        }
                    }
                    std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_values;
                    temp.read_partial2_batch(spawn2_partial_cids, spawn2_partial_values, stats);
                    uint64_t spawn2_partial_prefetch_bytes = 0U;
                    if (collect_resident_breakdown) {
                        spawn2_partial_prefetch_bytes =
                            detail::bc_family_value_vectors_resident_bytes(
                                spawn2_partial_values
                            );
                        stats.block_partial_prefetch_resident_bytes_max = std::max(
                            stats.block_partial_prefetch_resident_bytes_max,
                            spawn2_partial_prefetch_bytes
                        );
                        stats.block_total_resident_bytes_max = std::max(
                            stats.block_total_resident_bytes_max,
                            interleaved_block_base_resident_bytes +
                                spawn2_partial_prefetch_bytes
                        );
                    }
                    BCFutureSuccessLookupView<StorageT> lookup2 =
                        open_interleaved_lookup(future2_window, future2_position, pass2, false);
                    std::vector<int64_t> spawn2_partial_prefetch_index(current_cells.size(), -1);
                    for (size_t i = 0U; i < spawn2_partial_indices.size(); ++i) {
                        spawn2_partial_prefetch_index[spawn2_partial_indices[i]] =
                            static_cast<int64_t>(i);
                    }
                    std::vector<int64_t> spawn2_scratch_prefetch_index(current_cells.size(), -1);
                    size_t scratch_index = 0U;
                    for (size_t i = 0U; i < pass2.current_cells.size(); ++i) {
                        const BCFamilySolveCellWork &work = pass2.current_cells[i];
                        if (work.visit == BCFamilySolveCellVisitKind::FirstDirection ||
                            current_position.descriptor(work.cid).empty()) {
                            continue;
                        }
                        if (scratch_index >= state.local_scratch4_cids.size() ||
                            state.local_scratch4_cids[scratch_index] != work.cid) {
                            throw std::logic_error("BC family interleaved scratch4 cid mismatch");
                        }
                        spawn2_scratch_prefetch_index[i] = static_cast<int64_t>(scratch_index++);
                    }
                    if (scratch_index != state.local_scratch4_cids.size()) {
                        throw std::logic_error("BC family interleaved scratch4 count mismatch");
                    }
                    std::vector<BCFamilyValueVector<StorageT>> spawn2_spilled_scratch4_values;
                    std::vector<BCFamilyValueVector<StorageT>> *spawn2_scratch4_values =
                        &state.local_scratch4_values;
                    if (state.local_scratch4_spilled) {
                        temp.read_scratch4_batch(
                            state.local_scratch4_cids,
                            spawn2_spilled_scratch4_values,
                            stats
                        );
                        spawn2_scratch4_values = &spawn2_spilled_scratch4_values;
                    }
                    if (scratch_index != spawn2_scratch4_values->size()) {
                        throw std::logic_error("BC family interleaved scratch4 value count mismatch");
                    }

                    struct DenseFinalizeCell {
                        size_t cell_index = 0U;
                        BCFamilyValueVector<StorageT> values;
                    };
                    const int compact_threads =
                        bc_resident_solve_effective_threads(options.solve.num_threads);
                    const size_t compact_batch_limit =
                        std::max<size_t>(1U, static_cast<size_t>(compact_threads));
                    std::vector<DenseFinalizeCell> dense_batch;
                    dense_batch.reserve(compact_batch_limit);
                    std::vector<detail::BCFamilyCompactedOutputCell<StorageT>> compacted;
                    std::vector<BCResidentCompactStats> per_compact_thread(
                        static_cast<size_t>(compact_threads)
                    );
                    std::vector<CellId> spawn2_partial_write_cids;
                    std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_write_values;
                    spawn2_partial_write_cids.reserve(current_cells.size());
                    spawn2_partial_write_values.reserve(current_cells.size());
                    const BCSolveTargetFamilyFilter spawn2_filter =
                        bc_family_make_solve_filter(pass2.future_families);
                    const bool spawn2_batch_first_enabled = options.solve.row_width == 1U;
                    std::vector<size_t> spawn2_first_horizontal_indices;
                    std::vector<size_t> spawn2_first_vertical_indices;
                    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> spawn2_first_partials;
                    const bool spawn2_batch_second_enabled =
                        std::is_integral_v<StorageT> && options.solve.row_width == 1U;
                    std::vector<size_t> spawn2_second_horizontal_indices;
                    std::vector<size_t> spawn2_second_vertical_indices;
                    std::vector<BCFamilyPartialMaxCellBuffer<StorageT>> spawn2_second_partials;
                    std::vector<BCFamilyCellSuccessScratch<StorageT>> spawn2_second_final;
                    if (spawn2_batch_first_enabled) {
                        spawn2_first_horizontal_indices.reserve(current_cells.size());
                        spawn2_first_vertical_indices.reserve(current_cells.size());
                        spawn2_first_partials.resize(current_cells.size());
                    }
                    if (spawn2_batch_second_enabled) {
                        spawn2_second_horizontal_indices.reserve(current_cells.size());
                        spawn2_second_vertical_indices.reserve(current_cells.size());
                        spawn2_second_partials.resize(current_cells.size());
                        spawn2_second_final.resize(current_cells.size());
                    }
                    update_interleaved_block_resident_bytes();
                    if (collect_resident_breakdown) {
                        stats.block_total_resident_bytes_max = std::max(
                            stats.block_total_resident_bytes_max,
                            interleaved_block_base_resident_bytes +
                                spawn2_partial_prefetch_bytes
                        );
                    }
                    auto flush_dense_batch = [&]() {
                        if (dense_batch.empty()) {
                            return;
                        }
                        if (collect_resident_breakdown) {
                            uint64_t dense_bytes =
                                static_cast<uint64_t>(dense_batch.capacity()) *
                                sizeof(DenseFinalizeCell);
                            for (const DenseFinalizeCell &cell : dense_batch) {
                                dense_bytes += detail::bc_family_value_vector_resident_bytes(
                                    cell.values
                                );
                            }
                            dense_bytes +=
                                static_cast<uint64_t>(compacted.capacity()) *
                                sizeof(detail::BCFamilyCompactedOutputCell<StorageT>);
                            for (const detail::BCFamilyCompactedOutputCell<StorageT> &cell : compacted) {
                                dense_bytes += detail::bc_family_value_buffer_resident_bytes(
                                    cell.values
                                );
                            }
                            stats.block_dense_resident_bytes_max = std::max(
                                stats.block_dense_resident_bytes_max,
                                dense_bytes
                            );
                            stats.block_total_resident_bytes_max = std::max(
                                stats.block_total_resident_bytes_max,
                                interleaved_block_base_resident_bytes +
                                    spawn2_partial_prefetch_bytes +
                                    dense_bytes
                            );
                        }
                        compacted.clear();
                        compacted.resize(dense_batch.size());
                        std::fill(
                            per_compact_thread.begin(),
                            per_compact_thread.end(),
                            BCResidentCompactStats{}
                        );
                        std::exception_ptr first_compact_exception;
                        const double compact_t0 = bc_single_chunk_now_seconds();
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
                        for (int64_t i_signed = 0;
                             i_signed < static_cast<int64_t>(dense_batch.size());
                             ++i_signed) {
                            const size_t i = static_cast<size_t>(i_signed);
#if defined(_OPENMP)
                            const int tid = omp_get_thread_num();
#else
                            const int tid = 0;
#endif
                            try {
                                detail::bc_family_compact_dense_cell<StorageT>(
                                    lut,
                                    current_cells[dense_batch[i].cell_index],
                                    dense_batch[i].values,
                                    options.solve.row_width,
                                    options.solve.zero_value,
                                    compacted[i]
                                );
                                per_compact_thread[static_cast<size_t>(tid)].input_rows +=
                                    compacted[i].compact_stats.input_rows;
                                per_compact_thread[static_cast<size_t>(tid)].live_rows +=
                                    compacted[i].compact_stats.live_rows;
                                per_compact_thread[static_cast<size_t>(tid)].zero_pruned_rows +=
                                    compacted[i].compact_stats.zero_pruned_rows;
                                per_compact_thread[static_cast<size_t>(tid)].live_cells +=
                                    compacted[i].compact_stats.live_cells;
                                per_compact_thread[static_cast<size_t>(tid)].empty_cells +=
                                    compacted[i].compact_stats.empty_cells;
                            } catch (...) {
#pragma omp critical(BCFamilySolveCompactException)
                                {
                                    if (!first_compact_exception) {
                                        first_compact_exception = std::current_exception();
                                    }
                                }
                            }
                        }
                        stats.single.compact_seconds +=
                            bc_single_chunk_now_seconds() - compact_t0;
                        if (first_compact_exception) {
                            std::rethrow_exception(first_compact_exception);
                        }
                        for (const BCResidentCompactStats &compact_stats : per_compact_thread) {
                            stats.single.compact_input_rows += compact_stats.input_rows;
                            stats.single.compact_live_rows += compact_stats.live_rows;
                            stats.single.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
                            stats.single.compact_live_cells += compact_stats.live_cells;
                            stats.single.compact_empty_cells += compact_stats.empty_cells;
                        }
                        detail::bc_family_mark_compacted_outputs_ready(
                            compacted,
                            pending,
                            pending_value_bytes,
                            options.final_pending_value_memory_cap_bytes,
                            next_output_cid,
                            output_streamer,
                            stats
                        );
                        detail::bc_family_update_pending_stats(pending, stats);
                        {
                            const double release_t0 = bc_single_chunk_now_seconds();
                            if (dense_batch.size() >= 8U && release_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(release_threads)
                                for (int64_t i = 0;
                                     i < static_cast<int64_t>(dense_batch.size());
                                     ++i) {
                                    BCFamilyValueVector<StorageT>().swap(
                                        dense_batch[static_cast<size_t>(i)].values
                                    );
                                }
                            } else {
                                for (DenseFinalizeCell &cell : dense_batch) {
                                    BCFamilyValueVector<StorageT>().swap(cell.values);
                                }
                            }
                            detail::bc_family_release_compacted_output_values(
                                compacted,
                                release_threads
                            );
                            dense_batch.clear();
                            compacted.clear();
                            const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                            stats.single.workspace_release_seconds += elapsed;
                            stats.workspace_release_spawn2_dense_seconds += elapsed;
                        }
                        detail::bc_family_flush_ready_outputs(
                            pending,
                            pending_value_bytes,
                            next_output_cid,
                            output_streamer,
                            stats
                        );
                    };

                    for (size_t i = 0U; i < current_cells.size(); ++i) {
                        const BCLoadedCell &cell = current_cells[i];
                        if (cell.success_rows == 0U || cell.buckets.empty()) {
                            continue;
                        }
                        if (state.current_layout_ready[i] == 0U) {
                            throw std::logic_error("BC family interleaved missing current cell layout");
                        }
                        const BCFamilySolveCellWork &cell_work = pass2.current_cells[i];
                        if (spawn2_batch_first_enabled &&
                            cell_work.visit == BCFamilySolveCellVisitKind::FirstDirection) {
                            if (cell_work.directions == BCDirectionMask::Horizontal) {
                                spawn2_first_horizontal_indices.push_back(i);
                                continue;
                            }
                            if (cell_work.directions == BCDirectionMask::Vertical) {
                                spawn2_first_vertical_indices.push_back(i);
                                continue;
                            }
                        }
                        if (spawn2_batch_second_enabled &&
                            cell_work.visit == BCFamilySolveCellVisitKind::SecondDirection &&
                            (cell_work.directions == BCDirectionMask::Horizontal ||
                             cell_work.directions == BCDirectionMask::Vertical)) {
                            if (spawn2_partial_prefetch_index[i] < 0 ||
                                spawn2_scratch_prefetch_index[i] < 0) {
                                throw std::logic_error(
                                    "BC family interleaved missing spawn2 second prefetch"
                                );
                            }
                            BCFamilyValueVector<StorageT> stored_partial = std::move(
                                spawn2_partial_values[
                                    static_cast<size_t>(spawn2_partial_prefetch_index[i])
                                ]
                            );
                            spawn2_second_partials[i].values() =
                                detail::bc_family_decode_partial_temp_values(
                                    state.current_layouts[i],
                                    std::move(stored_partial),
                                    options.solve.zero_value,
                                    options
                                );
                            if (spawn2_second_partials[i].values().size() !=
                                static_cast<size_t>(state.current_layouts[i].value_count)) {
                                throw std::runtime_error(
                                    "BC family spawn2 batched partial count mismatch"
                                );
                            }
                            spawn2_second_final[i].adopt(
                                cell.cid,
                                cell.success_rows,
                                options.solve.row_width,
                                    std::move(
                                    (*spawn2_scratch4_values)[
                                        static_cast<size_t>(spawn2_scratch_prefetch_index[i])
                                    ]
                                )
                            );
                            if (cell_work.directions == BCDirectionMask::Horizontal) {
                                spawn2_second_horizontal_indices.push_back(i);
                            } else {
                                spawn2_second_vertical_indices.push_back(i);
                            }
                            continue;
                        }
                        BCFamilyValueVector<StorageT> dense_final_values;
                        const double temp_read_before = stats.temp_read_seconds;
                        const double temp_write_before = stats.temp_write_seconds;
                        const double cell_t0 = bc_single_chunk_now_seconds();
                        const bool first_visit =
                            cell_work.visit == BCFamilySolveCellVisitKind::FirstDirection;
                        BCFamilyValueVector<StorageT> temp_output_values;
                        BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
                        if (spawn2_partial_prefetch_index[i] >= 0) {
                            prefetched_partial = &spawn2_partial_values[
                                static_cast<size_t>(spawn2_partial_prefetch_index[i])
                            ];
                        }
                        BCFamilyValueVector<StorageT> *prefetched_scratch4 = nullptr;
                        if (spawn2_scratch_prefetch_index[i] >= 0) {
                            prefetched_scratch4 = &(*spawn2_scratch4_values)[
                                static_cast<size_t>(spawn2_scratch_prefetch_index[i])
                            ];
                        }
                        bc_family_spawn2_cell(
                            lut,
                            cell,
                            cell_work.directions,
                            cell_work.visit,
                            state.current_layouts[i],
                            options,
                            future2_position.axis(),
                            spawn2_filter,
                            lookup2,
                            temp,
                            scratch,
                            pending,
                            stats,
                            &dense_final_values,
                            prefetched_partial,
                            prefetched_scratch4,
                            first_visit ? &temp_output_values : nullptr
                        );
                        const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
                        const double known_delta =
                            (stats.temp_read_seconds - temp_read_before) +
                            (stats.temp_write_seconds - temp_write_before);
                        stats.spawn2_cell_compute_seconds +=
                            detail::bc_family_positive_remainder(cell_elapsed, known_delta);
                        if (first_visit) {
                            temp_output_values = detail::bc_family_prepare_partial_temp_output(
                                state.current_layouts[i],
                                std::move(temp_output_values),
                                false,
                                options,
                                stats
                            );
                            spawn2_partial_write_cids.push_back(cell.cid);
                            spawn2_partial_write_values.push_back(std::move(temp_output_values));
                        }
                        if (!dense_final_values.empty()) {
                            dense_batch.push_back(DenseFinalizeCell{i, std::move(dense_final_values)});
                            if (dense_batch.size() >= compact_batch_limit) {
                                flush_dense_batch();
                            }
                        }
                    }
                    auto append_spawn2_first_batch = [&](const std::vector<size_t> &indices,
                                                         BCDirectionMask directions) {
                        if (indices.empty()) {
                            return;
                        }
                        const double batch_t0 = bc_single_chunk_now_seconds();
                        detail::bc_family_first_direction_cells_batch<StorageT>(
                            lut,
                            current_cells,
                            indices,
                            directions,
                            BCSolveSpawnPhase::Spawn2,
                            state.current_layouts,
                            future2_position.axis(),
                            lookup2,
                            spawn2_filter,
                            options,
                            scratch,
                            stats,
                            spawn2_first_partials
                        );
                        stats.spawn2_cell_compute_seconds +=
                            bc_single_chunk_now_seconds() - batch_t0;
                        for (size_t cell_index : indices) {
                            const BCLoadedCell &cell = current_cells[cell_index];
                            if (cell.success_rows == 0U || cell.buckets.empty()) {
                                continue;
                            }
                            BCFamilyValueVector<StorageT> temp_output_values =
                                detail::bc_family_prepare_partial_temp_output(
                                    state.current_layouts[cell_index],
                                    std::move(spawn2_first_partials[cell_index].values()),
                                    false,
                                    options,
                                    stats
                                );
                            spawn2_partial_write_cids.push_back(cell.cid);
                            spawn2_partial_write_values.push_back(std::move(temp_output_values));
                        }
                    };
                    append_spawn2_first_batch(
                        spawn2_first_horizontal_indices,
                        BCDirectionMask::Horizontal
                    );
                    append_spawn2_first_batch(
                        spawn2_first_vertical_indices,
                        BCDirectionMask::Vertical
                    );
                    detail::bc_family_release_partial_buffers(
                        spawn2_first_partials,
                        release_threads
                    );
                    auto append_spawn2_second_batch = [&](const std::vector<size_t> &indices,
                                                          BCDirectionMask directions) {
                        if (indices.empty()) {
                            return;
                        }
                        if constexpr (std::is_integral_v<StorageT>) {
                            const double batch_t0 = bc_single_chunk_now_seconds();
                            detail::bc_family_partial_sum_cells_batch<StorageT>(
                                lut,
                                current_cells,
                                indices,
                                directions,
                                BCSolveSpawnPhase::Spawn2,
                                state.current_layouts,
                                spawn2_second_partials,
                                future2_position.axis(),
                                lookup2,
                                spawn2_filter,
                                options,
                                scratch,
                                stats,
                                [&](uint32_t cell_index,
                                    uint32_t local_success_row,
                                    uint32_t /*local_bucket_index*/,
                                    const BCSolveBoardQuerySummary &summary,
                                    uint64_t sum,
                                    BCFamilySolveCellWorkspace<StorageT> &/*local*/) {
                                    BCFamilyCellSuccessScratch<StorageT> &final_values =
                                        spawn2_second_final[static_cast<size_t>(cell_index)];
                                    if (summary.terminal_success) {
                                        final_values.write_terminal_row(
                                            local_success_row,
                                            options.solve.terminal_value
                                        );
                                    } else if (summary.empty_count == 0U) {
                                        final_values.write_zero_row(
                                            local_success_row,
                                            options.solve.zero_value
                                        );
                                    } else {
                                        final_values.finalize_spawn2_sum_row(
                                            local_success_row,
                                            sum,
                                            summary.empty_count,
                                            options.solve.edge_options.spawn_rate4,
                                            options.solve.zero_value
                                        );
                                    }
                                }
                            );
                            stats.spawn2_cell_compute_seconds +=
                                bc_single_chunk_now_seconds() - batch_t0;
                            for (size_t cell_index : indices) {
                                BCFamilyValueVector<StorageT> values =
                                    std::move(spawn2_second_final[cell_index].values());
                                if (values.empty()) {
                                    continue;
                                }
                                dense_batch.push_back(DenseFinalizeCell{
                                    cell_index,
                                    std::move(values)
                                });
                                if (dense_batch.size() >= compact_batch_limit) {
                                    flush_dense_batch();
                                }
                            }
                        } else {
                            (void)directions;
                        }
                    };
                    append_spawn2_second_batch(
                        spawn2_second_horizontal_indices,
                        BCDirectionMask::Horizontal
                    );
                    append_spawn2_second_batch(
                        spawn2_second_vertical_indices,
                        BCDirectionMask::Vertical
                    );
                    detail::bc_family_release_partial_buffers(
                        spawn2_second_partials,
                        release_threads
                    );
                    detail::bc_family_release_success_scratch(
                        spawn2_second_final,
                        release_threads
                    );
                    flush_dense_batch();
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        compacted.clear();
                        std::vector<BCResidentCompactStats>().swap(per_compact_thread);
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn2_dense_seconds += elapsed;
                    }
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        detail::bc_family_release_value_vectors(
                            spawn2_partial_values,
                            release_threads
                        );
                        std::vector<int64_t>().swap(spawn2_partial_prefetch_index);
                        if (state.local_scratch4_spilled) {
                            detail::bc_family_release_value_vectors(
                                spawn2_spilled_scratch4_values,
                                release_threads
                            );
                        } else {
                            detail::bc_family_release_value_vectors(
                                state.local_scratch4_values,
                                release_threads
                            );
                        }
                        std::vector<int64_t>().swap(spawn2_scratch_prefetch_index);
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn2_prefetch_seconds += elapsed;
                    }
                    temp.write_partial2_batch(
                        spawn2_partial_write_cids,
                        spawn2_partial_write_values,
                        stats
                    );
                    {
                        const double release_t0 = bc_single_chunk_now_seconds();
                        std::vector<CellId>().swap(spawn2_partial_write_cids);
                        detail::bc_family_release_value_vectors(
                            spawn2_partial_write_values,
                            release_threads
                        );
                        state.current_layout_ready.clear();
                        std::vector<CellId>().swap(state.local_scratch4_cids);
                        state.local_scratch4_spilled = false;
                        const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                        stats.single.workspace_release_seconds += elapsed;
                        stats.workspace_release_spawn2_temp_values_seconds += elapsed;
                    }
                    if (pass_i + 1U < block_end) {
                        release_future_window_except(future2_window, cached2.keep_cids);
                    } else if (options.interleave_keep_future2_between_blocks &&
                               pass_i + 1U < spawn2_cached.size()) {
                        release_future_window_except(future2_window, cached2.keep_cids);
                    } else {
                        release_future_window_all(future2_window);
                    }
                    const double spawn2_phase_elapsed =
                        bc_single_chunk_now_seconds() - spawn2_phase_t0;
                    stats.spawn2_phase_wall_seconds += spawn2_phase_elapsed;
                    stats.spawn2_phase_untracked_seconds +=
                        detail::bc_family_positive_remainder(
                            spawn2_phase_elapsed,
                            interleaved_phase_accounted_seconds(false) -
                                spawn2_phase_accounted_before
                        );
                }
            }
            {
                const double release_t0 = bc_single_chunk_now_seconds();
                std::vector<InterleavedBlockPassState>().swap(block_states);
                const double elapsed = bc_single_chunk_now_seconds() - release_t0;
                stats.single.workspace_release_seconds += elapsed;
                stats.workspace_release_block_state_seconds += elapsed;
            }
        } else {
        for (size_t pass_i = 0U; pass_i < spawn4_cached.size(); ++pass_i) {
            const BCFamilySolveCachedPass &cached4 = spawn4_cached[pass_i];
            const BCFamilySolveCachedPass &cached2 = spawn2_cached[pass_i];
            const BCFamilySolvePassPlan &pass4 = cached4.pass;
            const BCFamilySolvePassPlan &pass2 = cached2.pass;
            if (pass4.fid != pass2.fid ||
                cached4.current_cids != cached2.current_cids ||
                pass4.current_cells.size() != pass2.current_cells.size()) {
                throw std::logic_error("BC family interleaved solve pass alignment mismatch");
            }
            for (size_t i = 0U; i < pass4.current_cells.size(); ++i) {
                const BCFamilySolveCellWork &lhs = pass4.current_cells[i];
                const BCFamilySolveCellWork &rhs = pass2.current_cells[i];
                if (lhs.cid != rhs.cid ||
                    lhs.directions != rhs.directions ||
                    lhs.visit != rhs.visit) {
                    throw std::logic_error("BC family interleaved solve cell alignment mismatch");
                }
            }

            ++stats.spawn4_passes;
            std::vector<size_t> spawn4_second_indices;
            std::vector<CellId> spawn4_second_cids;
            spawn4_second_indices.reserve(pass4.current_cells.size());
            spawn4_second_cids.reserve(pass4.current_cells.size());
            for (size_t i = 0U; i < pass4.current_cells.size(); ++i) {
                const BCFamilySolveCellWork &work = pass4.current_cells[i];
                if (work.visit != BCFamilySolveCellVisitKind::SecondDirection) {
                    continue;
                }
                if (current_position.descriptor(work.cid).empty()) {
                    continue;
                }
                spawn4_second_indices.push_back(i);
                spawn4_second_cids.push_back(work.cid);
            }
            std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_values;
            temp.read_partial4_batch(spawn4_second_cids, spawn4_partial_values, stats);
            detail::BCFamilyCurrentCellLoadResult current_result;
            {
                const double current_t0 = bc_single_chunk_now_seconds();
                current_result.cells =
                    current_position.load_cells(cached4.current_cids, &current_result.load_stats);
                current_result.seconds = bc_single_chunk_now_seconds() - current_t0;
            }
            BCFutureSuccessLookupView<StorageT> lookup4 =
                open_interleaved_lookup(future4_window, future4_position, pass4, true);

            std::vector<BCLoadedCell> current_cells = std::move(current_result.cells);
            stats.single.current_position_read_seconds += current_result.seconds;
            bc_single_chunk_add_cell_load_stats(
                stats.single.current_position_load,
                current_result.load_stats
            );
            stats.family_current_cells_loaded += current_cells.size();
            ++stats.single.current_chunks;

            std::vector<int64_t> spawn4_prefetch_index(current_cells.size(), -1);
            for (size_t i = 0U; i < spawn4_second_indices.size(); ++i) {
                spawn4_prefetch_index[spawn4_second_indices[i]] = static_cast<int64_t>(i);
            }

            std::vector<BCFamilyPartialCellLayout> current_layouts(current_cells.size());
            std::vector<uint8_t> current_layout_ready(current_cells.size(), 0U);
            std::vector<CellId> spawn4_partial_write_cids;
            std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_write_values;
            std::vector<CellId> local_scratch4_cids;
            std::vector<BCFamilyValueVector<StorageT>> local_scratch4_values;
            spawn4_partial_write_cids.reserve(current_cells.size());
            spawn4_partial_write_values.reserve(current_cells.size());
            local_scratch4_cids.reserve(current_cells.size());
            local_scratch4_values.reserve(current_cells.size());
            for (size_t i = 0U; i < current_cells.size(); ++i) {
                const BCLoadedCell &cell = current_cells[i];
                if (cell.success_rows == 0U || cell.buckets.empty()) {
                    ++stats.single.current_empty_cells;
                    continue;
                }
                ++stats.single.current_nonempty_cells;
                stats.single.current_cells += 1U;
                stats.single.current_rows += cell.success_rows;
                stats.single.current_boards += cell.success_rows;
                const double layout_t0 = bc_single_chunk_now_seconds();
                current_layouts[i] =
                    bc_family_make_partial_cell_layout(lut, cell.view(), options.solve.row_width);
                current_layout_ready[i] = 1U;
                stats.current_layout_seconds += bc_single_chunk_now_seconds() - layout_t0;
                const double temp_read_before = stats.temp_read_seconds;
                const double temp_write_before = stats.temp_write_seconds;
                const double cell_t0 = bc_single_chunk_now_seconds();
                const BCSolveTargetFamilyFilter filter =
                    bc_family_make_solve_filter(pass4.future_families);
                BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
                if (spawn4_prefetch_index[i] >= 0) {
                    prefetched_partial =
                        &spawn4_partial_values[static_cast<size_t>(spawn4_prefetch_index[i])];
                }
                BCFamilyValueVector<StorageT> temp_output_values;
                bc_family_spawn4_cell(
                    lut,
                    cell,
                    pass4.current_cells[i].directions,
                    pass4.current_cells[i].visit,
                    current_layouts[i],
                    options,
                    future4_position.axis(),
                    filter,
                    lookup4,
                    temp,
                    scratch,
                    stats,
                    prefetched_partial,
                    &temp_output_values
                );
                const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
                const double temp_delta =
                    (stats.temp_read_seconds - temp_read_before) +
                    (stats.temp_write_seconds - temp_write_before);
                stats.spawn4_cell_compute_seconds +=
                    detail::bc_family_positive_remainder(cell_elapsed, temp_delta);
                if (pass4.current_cells[i].visit == BCFamilySolveCellVisitKind::FirstDirection) {
                    temp_output_values = detail::bc_family_prepare_partial_temp_output(
                        current_layouts[i],
                        std::move(temp_output_values),
                        true,
                        options,
                        stats
                    );
                    spawn4_partial_write_cids.push_back(cell.cid);
                    spawn4_partial_write_values.push_back(std::move(temp_output_values));
                } else {
                    local_scratch4_cids.push_back(cell.cid);
                    local_scratch4_values.push_back(std::move(temp_output_values));
                }
            }
            std::vector<BCFamilyValueVector<StorageT>>().swap(spawn4_partial_values);
            std::vector<int64_t>().swap(spawn4_prefetch_index);
            temp.write_partial4_batch(
                spawn4_partial_write_cids,
                spawn4_partial_write_values,
                stats
            );
            std::vector<CellId>().swap(spawn4_partial_write_cids);
            std::vector<BCFamilyValueVector<StorageT>>().swap(spawn4_partial_write_values);
            if (options.interleave_release_future_between_phases) {
                release_future_window_all(future4_window);
            } else {
                release_future_window_except(future4_window, cached4.keep_cids);
            }

            ++stats.spawn2_passes;
            std::vector<size_t> spawn2_partial_indices;
            std::vector<CellId> spawn2_partial_cids;
            spawn2_partial_indices.reserve(pass2.current_cells.size());
            spawn2_partial_cids.reserve(pass2.current_cells.size());
            for (size_t i = 0U; i < pass2.current_cells.size(); ++i) {
                const BCFamilySolveCellWork &work = pass2.current_cells[i];
                if (current_position.descriptor(work.cid).empty()) {
                    continue;
                }
                if (work.visit == BCFamilySolveCellVisitKind::SecondDirection) {
                    spawn2_partial_indices.push_back(i);
                    spawn2_partial_cids.push_back(work.cid);
                }
            }
            std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_values;
            temp.read_partial2_batch(spawn2_partial_cids, spawn2_partial_values, stats);
            BCFutureSuccessLookupView<StorageT> lookup2 =
                open_interleaved_lookup(future2_window, future2_position, pass2, false);
            std::vector<int64_t> spawn2_partial_prefetch_index(current_cells.size(), -1);
            for (size_t i = 0U; i < spawn2_partial_indices.size(); ++i) {
                spawn2_partial_prefetch_index[spawn2_partial_indices[i]] =
                    static_cast<int64_t>(i);
            }
            std::vector<int64_t> spawn2_scratch_prefetch_index(current_cells.size(), -1);
            size_t scratch_index = 0U;
            for (size_t i = 0U; i < pass2.current_cells.size(); ++i) {
                const BCFamilySolveCellWork &work = pass2.current_cells[i];
                if (work.visit == BCFamilySolveCellVisitKind::FirstDirection ||
                    current_position.descriptor(work.cid).empty()) {
                    continue;
                }
                if (scratch_index >= local_scratch4_cids.size() ||
                    local_scratch4_cids[scratch_index] != work.cid) {
                    throw std::logic_error("BC family interleaved scratch4 cid mismatch");
                }
                spawn2_scratch_prefetch_index[i] = static_cast<int64_t>(scratch_index++);
            }
            if (scratch_index != local_scratch4_values.size()) {
                throw std::logic_error("BC family interleaved scratch4 count mismatch");
            }

            struct DenseFinalizeCell {
                size_t cell_index = 0U;
                BCFamilyValueVector<StorageT> values;
            };
            const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
            const size_t compact_batch_limit = std::max<size_t>(1U, static_cast<size_t>(compact_threads));
            std::vector<DenseFinalizeCell> dense_batch;
            dense_batch.reserve(compact_batch_limit);
            std::vector<CellId> spawn2_partial_write_cids;
            std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_write_values;
            spawn2_partial_write_cids.reserve(current_cells.size());
            spawn2_partial_write_values.reserve(current_cells.size());
            auto flush_dense_batch = [&]() {
                if (dense_batch.empty()) {
                    return;
                }
                std::vector<detail::BCFamilyCompactedOutputCell<StorageT>> compacted(dense_batch.size());
                std::vector<BCResidentCompactStats> per_compact_thread(
                    static_cast<size_t>(compact_threads)
                );
                std::exception_ptr first_compact_exception;
                const double compact_t0 = bc_single_chunk_now_seconds();
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
                for (int64_t i_signed = 0;
                     i_signed < static_cast<int64_t>(dense_batch.size());
                     ++i_signed) {
                    const size_t i = static_cast<size_t>(i_signed);
#if defined(_OPENMP)
                    const int tid = omp_get_thread_num();
#else
                    const int tid = 0;
#endif
                    try {
                        detail::bc_family_compact_dense_cell<StorageT>(
                            lut,
                            current_cells[dense_batch[i].cell_index],
                            dense_batch[i].values,
                            options.solve.row_width,
                            options.solve.zero_value,
                            compacted[i]
                        );
                        per_compact_thread[static_cast<size_t>(tid)].input_rows +=
                            compacted[i].compact_stats.input_rows;
                        per_compact_thread[static_cast<size_t>(tid)].live_rows +=
                            compacted[i].compact_stats.live_rows;
                        per_compact_thread[static_cast<size_t>(tid)].zero_pruned_rows +=
                            compacted[i].compact_stats.zero_pruned_rows;
                        per_compact_thread[static_cast<size_t>(tid)].live_cells +=
                            compacted[i].compact_stats.live_cells;
                        per_compact_thread[static_cast<size_t>(tid)].empty_cells +=
                            compacted[i].compact_stats.empty_cells;
                    } catch (...) {
#pragma omp critical(BCFamilySolveCompactException)
                        {
                            if (!first_compact_exception) {
                                first_compact_exception = std::current_exception();
                            }
                        }
                    }
                }
                stats.single.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
                if (first_compact_exception) {
                    std::rethrow_exception(first_compact_exception);
                }
                for (const BCResidentCompactStats &compact_stats : per_compact_thread) {
                    stats.single.compact_input_rows += compact_stats.input_rows;
                    stats.single.compact_live_rows += compact_stats.live_rows;
                    stats.single.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
                    stats.single.compact_live_cells += compact_stats.live_cells;
                    stats.single.compact_empty_cells += compact_stats.empty_cells;
                }
                detail::bc_family_mark_compacted_outputs_ready(
                    compacted,
                    pending,
                    pending_value_bytes,
                    options.final_pending_value_memory_cap_bytes,
                    next_output_cid,
                    output_streamer,
                    stats
                );
                detail::bc_family_update_pending_stats(pending, stats);
                dense_batch.clear();
                detail::bc_family_flush_ready_outputs(
                    pending,
                    pending_value_bytes,
                    next_output_cid,
                    output_streamer,
                    stats
                );
            };

            for (size_t i = 0U; i < current_cells.size(); ++i) {
                const BCLoadedCell &cell = current_cells[i];
                if (cell.success_rows == 0U || cell.buckets.empty()) {
                    continue;
                }
                if (current_layout_ready[i] == 0U) {
                    throw std::logic_error("BC family interleaved missing current cell layout");
                }
                BCFamilyValueVector<StorageT> dense_final_values;
                const double temp_read_before = stats.temp_read_seconds;
                const double temp_write_before = stats.temp_write_seconds;
                const double cell_t0 = bc_single_chunk_now_seconds();
                const BCSolveTargetFamilyFilter filter =
                    bc_family_make_solve_filter(pass2.future_families);
                const bool first_visit =
                    pass2.current_cells[i].visit == BCFamilySolveCellVisitKind::FirstDirection;
                BCFamilyValueVector<StorageT> temp_output_values;
                BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
                if (spawn2_partial_prefetch_index[i] >= 0) {
                    prefetched_partial =
                        &spawn2_partial_values[static_cast<size_t>(spawn2_partial_prefetch_index[i])];
                }
                BCFamilyValueVector<StorageT> *prefetched_scratch4 = nullptr;
                if (spawn2_scratch_prefetch_index[i] >= 0) {
                    prefetched_scratch4 =
                        &local_scratch4_values[static_cast<size_t>(spawn2_scratch_prefetch_index[i])];
                }
                bc_family_spawn2_cell(
                    lut,
                    cell,
                    pass2.current_cells[i].directions,
                    pass2.current_cells[i].visit,
                    current_layouts[i],
                    options,
                    future2_position.axis(),
                    filter,
                    lookup2,
                    temp,
                    scratch,
                    pending,
                    stats,
                    &dense_final_values,
                    prefetched_partial,
                    prefetched_scratch4,
                    first_visit ? &temp_output_values : nullptr
                );
                const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
                const double known_delta =
                    (stats.temp_read_seconds - temp_read_before) +
                    (stats.temp_write_seconds - temp_write_before);
                stats.spawn2_cell_compute_seconds +=
                    detail::bc_family_positive_remainder(cell_elapsed, known_delta);
                if (first_visit) {
                    temp_output_values = detail::bc_family_prepare_partial_temp_output(
                        current_layouts[i],
                        std::move(temp_output_values),
                        false,
                        options,
                        stats
                    );
                    spawn2_partial_write_cids.push_back(cell.cid);
                    spawn2_partial_write_values.push_back(std::move(temp_output_values));
                }
                if (!dense_final_values.empty()) {
                    dense_batch.push_back(DenseFinalizeCell{i, std::move(dense_final_values)});
                    if (dense_batch.size() >= compact_batch_limit) {
                        flush_dense_batch();
                    }
                }
            }
            flush_dense_batch();
            std::vector<BCFamilyValueVector<StorageT>>().swap(spawn2_partial_values);
            std::vector<int64_t>().swap(spawn2_partial_prefetch_index);
            std::vector<BCFamilyValueVector<StorageT>>().swap(local_scratch4_values);
            std::vector<int64_t>().swap(spawn2_scratch_prefetch_index);
            temp.write_partial2_batch(
                spawn2_partial_write_cids,
                spawn2_partial_write_values,
                stats
            );
            std::vector<CellId>().swap(spawn2_partial_write_cids);
            std::vector<BCFamilyValueVector<StorageT>>().swap(spawn2_partial_write_values);
            if (options.interleave_release_future_between_phases) {
                release_future_window_all(future2_window);
            } else {
                release_future_window_except(future2_window, cached2.keep_cids);
            }
        }
        }

        bc_family_add_future_window_stats(stats.single, future4_window.stats(), true);
        bc_family_add_future_window_stats(stats.single, future2_window.stats(), false);

        if (next_output_cid != current_position.cell_count()) {
            detail::bc_family_flush_ready_outputs(
                pending,
                pending_value_bytes,
                next_output_cid,
                output_streamer,
                stats
            );
        }
        if (next_output_cid != current_position.cell_count()) {
            throw std::runtime_error("BC family solve did not finalize all output cells");
        }

        const double output_finish_t0 = bc_single_chunk_now_seconds();
        BCSingleChunkSolveFileResult single_result = output_streamer.finish();
        stats.output_finish_seconds += bc_single_chunk_now_seconds() - output_finish_t0;
        const double temp_close_t0 = bc_single_chunk_now_seconds();
        temp.wait_all_writes(stats);
        temp.close();
        stats.temp_close_seconds += bc_single_chunk_now_seconds() - temp_close_t0;
        if (!options.keep_temp_files) {
            const double cleanup_t0 = bc_single_chunk_now_seconds();
            temp.cleanup();
            std::filesystem::remove_all(temp_dir, cleanup_ec);
            stats.single.partial_cleanup_seconds += bc_single_chunk_now_seconds() - cleanup_t0;
        }

        BCFamilySolveFileResult result;
        result.position_bytes = single_result.position_bytes;
        result.success_bytes = single_result.success_bytes;
        stats.single.output_values = single_result.stats.output_values;
        stats.single.output_bytes = single_result.stats.output_bytes;
        result.stats = std::move(stats);
        return result;
    }

    BCFutureFamilyWindowOptions future_window_options;
    future_window_options.max_recycled_index_bytes =
        options.future_index_recycle_max_bytes;
    future_window_options.release_threads =
        static_cast<uint32_t>(bc_resident_solve_effective_threads(options.solve.num_threads));
    BCFutureFamilyWindow<StorageT> future4_window(
        future4_position,
        future4_success,
        future_window_options
    );
    for (const BCFamilySolveCachedPass &cached : spawn4_cached) {
        const BCFamilySolvePassPlan &pass = cached.pass;
        ++stats.spawn4_passes;

        std::vector<size_t> spawn4_second_indices;
        std::vector<CellId> spawn4_second_cids;
        spawn4_second_indices.reserve(pass.current_cells.size());
        spawn4_second_cids.reserve(pass.current_cells.size());
        for (size_t i = 0U; i < pass.current_cells.size(); ++i) {
            const BCFamilySolveCellWork &work = pass.current_cells[i];
            if (work.visit != BCFamilySolveCellVisitKind::SecondDirection) {
                continue;
            }
            if (current_position.descriptor(work.cid).empty()) {
                continue;
            }
            spawn4_second_indices.push_back(i);
            spawn4_second_cids.push_back(work.cid);
        }
        std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_values;
        temp.read_partial4_batch(spawn4_second_cids, spawn4_partial_values, stats);
        detail::BCFamilyCurrentCellLoadResult spawn4_current_result;
        {
            const double current_t0 = bc_single_chunk_now_seconds();
            spawn4_current_result.cells =
                current_position.load_cells(cached.current_cids, &spawn4_current_result.load_stats);
            spawn4_current_result.seconds = bc_single_chunk_now_seconds() - current_t0;
        }

        BCFutureSuccessLookupView<StorageT> lookup =
            bc_family_make_empty_future_lookup<StorageT>(
                future4_position.lut(),
                future4_position.cell_count(),
                options.solve.row_width,
                options.solve.dtype
            );
        if (!pass.future_cids.empty()) {
            ++stats.spawn4_future_reuse_groups;
            const BCFutureFamilyWindowStats before_prepare = future4_window.stats();
            const double prepare_t0 = bc_single_chunk_now_seconds();
            future4_window.prepare_cells(pass.future_cids);
            const double prepare_elapsed = bc_single_chunk_now_seconds() - prepare_t0;
            const BCFutureFamilyWindowStats after_prepare = future4_window.stats();
            const double prepare_io_seconds =
                (after_prepare.position_read_seconds - before_prepare.position_read_seconds) +
                (after_prepare.success_read_seconds - before_prepare.success_read_seconds);
            stats.future4_prepare_overhead_seconds +=
                detail::bc_family_positive_remainder(prepare_elapsed, prepare_io_seconds);
            uint64_t position_bytes = 0U;
            uint64_t success_bytes = 0U;
            const double index_before = stats.single.future4_index_seconds;
            const double lookup_t0 = bc_single_chunk_now_seconds();
            lookup = future4_window.open_success_lookup(
                options.solve.row_width,
                options.solve.dtype,
                stats.single.future4_index_seconds,
                position_bytes,
                success_bytes
            );
            stats.future4_lookup_copy_seconds += detail::bc_family_positive_remainder(
                bc_single_chunk_now_seconds() - lookup_t0,
                stats.single.future4_index_seconds - index_before
            );
            const uint64_t resident_position_bytes =
                future4_window.active_position_resident_bytes() +
                future4_window.recycled_index_resident_bytes() +
                position_bytes;
            const uint64_t resident_success_bytes =
                future4_window.active_success_resident_bytes() + success_bytes;
            stats.single.future4_position_resident_bytes = std::max(
                stats.single.future4_position_resident_bytes,
                resident_position_bytes
            );
            stats.single.future4_success_resident_bytes = std::max(
                stats.single.future4_success_resident_bytes,
                resident_success_bytes
            );
            stats.single.future_resident_layers_max = std::max<uint64_t>(
                stats.single.future_resident_layers_max,
                1U
            );
            stats.single.future_resident_bytes_max = std::max<uint64_t>(
                stats.single.future_resident_bytes_max,
                resident_position_bytes + resident_success_bytes
            );
        }

        std::vector<BCLoadedCell> current_cells = std::move(spawn4_current_result.cells);
        stats.single.current_position_read_seconds += spawn4_current_result.seconds;
        bc_single_chunk_add_cell_load_stats(
            stats.single.current_position_load,
            spawn4_current_result.load_stats
        );
        stats.family_current_cells_loaded += current_cells.size();
        ++stats.single.current_chunks;

        std::vector<int64_t> spawn4_prefetch_index(current_cells.size(), -1);
        for (size_t i = 0U; i < spawn4_second_indices.size(); ++i) {
            spawn4_prefetch_index[spawn4_second_indices[i]] = static_cast<int64_t>(i);
        }

        std::vector<CellId> spawn4_partial_write_cids;
        std::vector<BCFamilyValueVector<StorageT>> spawn4_partial_write_values;
        std::vector<CellId> spawn4_scratch_write_cids;
        std::vector<BCFamilyValueVector<StorageT>> spawn4_scratch_write_values;
        spawn4_partial_write_cids.reserve(current_cells.size());
        spawn4_partial_write_values.reserve(current_cells.size());
        spawn4_scratch_write_cids.reserve(current_cells.size());
        spawn4_scratch_write_values.reserve(current_cells.size());
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                ++stats.single.current_empty_cells;
                continue;
            }
            ++stats.single.current_nonempty_cells;
            stats.single.current_cells += 1U;
            stats.single.current_rows += cell.success_rows;
            stats.single.current_boards += cell.success_rows;
            const double layout_t0 = bc_single_chunk_now_seconds();
            const BCFamilyPartialCellLayout layout =
                bc_family_make_partial_cell_layout(lut, cell.view(), options.solve.row_width);
            stats.current_layout_seconds += bc_single_chunk_now_seconds() - layout_t0;
            const double temp_read_before = stats.temp_read_seconds;
            const double temp_write_before = stats.temp_write_seconds;
            const double cell_t0 = bc_single_chunk_now_seconds();
            const BCSolveTargetFamilyFilter filter =
                bc_family_make_solve_filter(pass.future_families);
            BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
            if (spawn4_prefetch_index[i] >= 0) {
                prefetched_partial =
                    &spawn4_partial_values[static_cast<size_t>(spawn4_prefetch_index[i])];
            }
            BCFamilyValueVector<StorageT> temp_output_values;
            bc_family_spawn4_cell(
                lut,
                cell,
                pass.current_cells[i].directions,
                pass.current_cells[i].visit,
                layout,
                options,
                future4_position.axis(),
                filter,
                lookup,
                temp,
                scratch,
                stats,
                prefetched_partial,
                &temp_output_values
            );
            const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
            const double temp_delta =
                (stats.temp_read_seconds - temp_read_before) +
                (stats.temp_write_seconds - temp_write_before);
            stats.spawn4_cell_compute_seconds +=
                detail::bc_family_positive_remainder(cell_elapsed, temp_delta);
            if (pass.current_cells[i].visit == BCFamilySolveCellVisitKind::FirstDirection) {
                temp_output_values = detail::bc_family_prepare_partial_temp_output(
                    layout,
                    std::move(temp_output_values),
                    true,
                    options,
                    stats
                );
                spawn4_partial_write_cids.push_back(cell.cid);
                spawn4_partial_write_values.push_back(std::move(temp_output_values));
            } else {
                spawn4_scratch_write_cids.push_back(cell.cid);
                spawn4_scratch_write_values.push_back(std::move(temp_output_values));
            }
        }
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn4_partial_values);
        std::vector<int64_t>().swap(spawn4_prefetch_index);
        temp.write_partial4_batch(
            spawn4_partial_write_cids,
            spawn4_partial_write_values,
            stats
        );
        temp.write_scratch4_batch(
            spawn4_scratch_write_cids,
            spawn4_scratch_write_values,
            stats
        );
        std::vector<CellId>().swap(spawn4_partial_write_cids);
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn4_partial_write_values);
        std::vector<CellId>().swap(spawn4_scratch_write_cids);
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn4_scratch_write_values);
        const double release_t0 = bc_single_chunk_now_seconds();
        future4_window.release_except(cached.keep_cids);
        stats.single.future_release_seconds += bc_single_chunk_now_seconds() - release_t0;
    }
    bc_family_add_future_window_stats(stats.single, future4_window.stats(), true);
    const double future4_release_all_t0 = bc_single_chunk_now_seconds();
    future4_window.release_all();
    stats.single.future_release_seconds += bc_single_chunk_now_seconds() - future4_release_all_t0;

    const double output_streamer_t0 = bc_single_chunk_now_seconds();
    const double position_write_before = stats.single.position_write_seconds;
    const double success_write_before = stats.single.success_write_seconds;
    BCSingleChunkFinalFileStreamer<StorageT> output_streamer(
        current_position,
        position_file,
        success_file,
        options.solve.row_width,
        options.solve.dtype,
        stats.single
    );
    stats.output_streamer_open_seconds += detail::bc_family_positive_remainder(
        bc_single_chunk_now_seconds() - output_streamer_t0,
        (stats.single.position_write_seconds - position_write_before) +
            (stats.single.success_write_seconds - success_write_before)
    );
    {
        const double mark_empty_t0 = bc_single_chunk_now_seconds();
        detail::bc_family_mark_empty_outputs(current_position, pending, output_streamer, stats);
        stats.mark_empty_seconds += bc_single_chunk_now_seconds() - mark_empty_t0;
    }
    CellId next_output_cid = 0U;
    detail::bc_family_flush_ready_outputs(
        pending,
        pending_value_bytes,
        next_output_cid,
        output_streamer,
        stats
    );

    BCFutureFamilyWindow<StorageT> future2_window(
        future2_position,
        future2_success,
        future_window_options
    );
    for (const BCFamilySolveCachedPass &cached : spawn2_cached) {
        const BCFamilySolvePassPlan &pass = cached.pass;
        ++stats.spawn2_passes;

        std::vector<size_t> spawn2_partial_indices;
        std::vector<CellId> spawn2_partial_cids;
        std::vector<size_t> spawn2_scratch_indices;
        std::vector<CellId> spawn2_scratch_cids;
        spawn2_partial_indices.reserve(pass.current_cells.size());
        spawn2_partial_cids.reserve(pass.current_cells.size());
        spawn2_scratch_indices.reserve(pass.current_cells.size());
        spawn2_scratch_cids.reserve(pass.current_cells.size());
        for (size_t i = 0U; i < pass.current_cells.size(); ++i) {
            const BCFamilySolveCellWork &work = pass.current_cells[i];
            if (current_position.descriptor(work.cid).empty()) {
                continue;
            }
            if (work.visit == BCFamilySolveCellVisitKind::SecondDirection) {
                spawn2_partial_indices.push_back(i);
                spawn2_partial_cids.push_back(work.cid);
            }
            if (work.visit != BCFamilySolveCellVisitKind::FirstDirection) {
                spawn2_scratch_indices.push_back(i);
                spawn2_scratch_cids.push_back(work.cid);
            }
        }
        std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_values;
        std::vector<BCFamilyValueVector<StorageT>> spawn2_scratch_values;
        temp.read_partial2_batch(spawn2_partial_cids, spawn2_partial_values, stats);
        temp.read_scratch4_batch(spawn2_scratch_cids, spawn2_scratch_values, stats);
        detail::BCFamilyCurrentCellLoadResult spawn2_current_result;
        {
            const double current_t0 = bc_single_chunk_now_seconds();
            spawn2_current_result.cells =
                current_position.load_cells(cached.current_cids, &spawn2_current_result.load_stats);
            spawn2_current_result.seconds = bc_single_chunk_now_seconds() - current_t0;
        }

        BCFutureSuccessLookupView<StorageT> lookup =
            bc_family_make_empty_future_lookup<StorageT>(
                future2_position.lut(),
                future2_position.cell_count(),
                options.solve.row_width,
                options.solve.dtype
            );
        if (!pass.future_cids.empty()) {
            ++stats.spawn2_future_reuse_groups;
            const BCFutureFamilyWindowStats before_prepare = future2_window.stats();
            const double prepare_t0 = bc_single_chunk_now_seconds();
            future2_window.prepare_cells(pass.future_cids);
            const double prepare_elapsed = bc_single_chunk_now_seconds() - prepare_t0;
            const BCFutureFamilyWindowStats after_prepare = future2_window.stats();
            const double prepare_io_seconds =
                (after_prepare.position_read_seconds - before_prepare.position_read_seconds) +
                (after_prepare.success_read_seconds - before_prepare.success_read_seconds);
            stats.future2_prepare_overhead_seconds +=
                detail::bc_family_positive_remainder(prepare_elapsed, prepare_io_seconds);
            uint64_t position_bytes = 0U;
            uint64_t success_bytes = 0U;
            const double index_before = stats.single.future2_index_seconds;
            const double lookup_t0 = bc_single_chunk_now_seconds();
            lookup = future2_window.open_success_lookup(
                options.solve.row_width,
                options.solve.dtype,
                stats.single.future2_index_seconds,
                position_bytes,
                success_bytes
            );
            stats.future2_lookup_copy_seconds += detail::bc_family_positive_remainder(
                bc_single_chunk_now_seconds() - lookup_t0,
                stats.single.future2_index_seconds - index_before
            );
            const uint64_t resident_position_bytes =
                future2_window.active_position_resident_bytes() +
                future2_window.recycled_index_resident_bytes() +
                position_bytes;
            const uint64_t resident_success_bytes =
                future2_window.active_success_resident_bytes() + success_bytes;
            stats.single.future2_position_resident_bytes = std::max(
                stats.single.future2_position_resident_bytes,
                resident_position_bytes
            );
            stats.single.future2_success_resident_bytes = std::max(
                stats.single.future2_success_resident_bytes,
                resident_success_bytes
            );
            stats.single.future_resident_layers_max = std::max<uint64_t>(
                stats.single.future_resident_layers_max,
                1U
            );
            stats.single.future_resident_bytes_max = std::max<uint64_t>(
                stats.single.future_resident_bytes_max,
                resident_position_bytes + resident_success_bytes
            );
        }

        std::vector<BCLoadedCell> current_cells = std::move(spawn2_current_result.cells);
        stats.single.current_position_read_seconds += spawn2_current_result.seconds;
        bc_single_chunk_add_cell_load_stats(
            stats.single.current_position_load,
            spawn2_current_result.load_stats
        );
        stats.family_current_cells_loaded += current_cells.size();
        ++stats.single.current_chunks;

        std::vector<int64_t> spawn2_partial_prefetch_index(current_cells.size(), -1);
        for (size_t i = 0U; i < spawn2_partial_indices.size(); ++i) {
            spawn2_partial_prefetch_index[spawn2_partial_indices[i]] = static_cast<int64_t>(i);
        }
        std::vector<int64_t> spawn2_scratch_prefetch_index(current_cells.size(), -1);
        for (size_t i = 0U; i < spawn2_scratch_indices.size(); ++i) {
            spawn2_scratch_prefetch_index[spawn2_scratch_indices[i]] = static_cast<int64_t>(i);
        }

        struct DenseFinalizeCell {
            size_t cell_index = 0U;
            BCFamilyValueVector<StorageT> values;
        };
        const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
        const size_t compact_batch_limit = std::max<size_t>(1U, static_cast<size_t>(compact_threads));
        std::vector<DenseFinalizeCell> dense_batch;
        dense_batch.reserve(compact_batch_limit);
        std::vector<CellId> spawn2_partial_write_cids;
        std::vector<BCFamilyValueVector<StorageT>> spawn2_partial_write_values;
        spawn2_partial_write_cids.reserve(current_cells.size());
        spawn2_partial_write_values.reserve(current_cells.size());
        auto flush_dense_batch = [&]() {
            if (dense_batch.empty()) {
                return;
            }
            std::vector<detail::BCFamilyCompactedOutputCell<StorageT>> compacted(dense_batch.size());
            std::vector<BCResidentCompactStats> per_compact_thread(
                static_cast<size_t>(compact_threads)
            );
            std::exception_ptr first_compact_exception;
            const double compact_t0 = bc_single_chunk_now_seconds();
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
            for (int64_t i_signed = 0;
                 i_signed < static_cast<int64_t>(dense_batch.size());
                 ++i_signed) {
                const size_t i = static_cast<size_t>(i_signed);
#if defined(_OPENMP)
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                try {
                    detail::bc_family_compact_dense_cell<StorageT>(
                        lut,
                        current_cells[dense_batch[i].cell_index],
                        dense_batch[i].values,
                        options.solve.row_width,
                        options.solve.zero_value,
                        compacted[i]
                    );
                    per_compact_thread[static_cast<size_t>(tid)].input_rows +=
                        compacted[i].compact_stats.input_rows;
                    per_compact_thread[static_cast<size_t>(tid)].live_rows +=
                        compacted[i].compact_stats.live_rows;
                    per_compact_thread[static_cast<size_t>(tid)].zero_pruned_rows +=
                        compacted[i].compact_stats.zero_pruned_rows;
                    per_compact_thread[static_cast<size_t>(tid)].live_cells +=
                        compacted[i].compact_stats.live_cells;
                    per_compact_thread[static_cast<size_t>(tid)].empty_cells +=
                        compacted[i].compact_stats.empty_cells;
                } catch (...) {
#pragma omp critical(BCFamilySolveCompactException)
                    {
                        if (!first_compact_exception) {
                            first_compact_exception = std::current_exception();
                        }
                    }
                }
            }
            stats.single.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
            if (first_compact_exception) {
                std::rethrow_exception(first_compact_exception);
            }
            for (const BCResidentCompactStats &compact_stats : per_compact_thread) {
                stats.single.compact_input_rows += compact_stats.input_rows;
                stats.single.compact_live_rows += compact_stats.live_rows;
                stats.single.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
                stats.single.compact_live_cells += compact_stats.live_cells;
                stats.single.compact_empty_cells += compact_stats.empty_cells;
            }
            detail::bc_family_mark_compacted_outputs_ready(
                compacted,
                pending,
                pending_value_bytes,
                options.final_pending_value_memory_cap_bytes,
                next_output_cid,
                output_streamer,
                stats
            );
            detail::bc_family_update_pending_stats(pending, stats);
            dense_batch.clear();
            detail::bc_family_flush_ready_outputs(
                pending,
                pending_value_bytes,
                next_output_cid,
                output_streamer,
                stats
            );
        };

        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                continue;
            }
            const double layout_t0 = bc_single_chunk_now_seconds();
            const BCFamilyPartialCellLayout layout =
                bc_family_make_partial_cell_layout(lut, cell.view(), options.solve.row_width);
            stats.current_layout_seconds += bc_single_chunk_now_seconds() - layout_t0;
            BCFamilyValueVector<StorageT> dense_final_values;
            const double temp_read_before = stats.temp_read_seconds;
            const double temp_write_before = stats.temp_write_seconds;
            const double cell_t0 = bc_single_chunk_now_seconds();
            const BCSolveTargetFamilyFilter filter =
                bc_family_make_solve_filter(pass.future_families);
            const bool first_visit =
                pass.current_cells[i].visit == BCFamilySolveCellVisitKind::FirstDirection;
            BCFamilyValueVector<StorageT> temp_output_values;
            BCFamilyValueVector<StorageT> *prefetched_partial = nullptr;
            if (spawn2_partial_prefetch_index[i] >= 0) {
                prefetched_partial =
                    &spawn2_partial_values[static_cast<size_t>(spawn2_partial_prefetch_index[i])];
            }
            BCFamilyValueVector<StorageT> *prefetched_scratch4 = nullptr;
            if (spawn2_scratch_prefetch_index[i] >= 0) {
                prefetched_scratch4 =
                    &spawn2_scratch_values[static_cast<size_t>(spawn2_scratch_prefetch_index[i])];
            }
            bc_family_spawn2_cell(
                lut,
                cell,
                pass.current_cells[i].directions,
                pass.current_cells[i].visit,
                layout,
                options,
                future2_position.axis(),
                filter,
                lookup,
                temp,
                scratch,
                pending,
                stats,
                &dense_final_values,
                prefetched_partial,
                prefetched_scratch4,
                first_visit ? &temp_output_values : nullptr
            );
            const double cell_elapsed = bc_single_chunk_now_seconds() - cell_t0;
            const double known_delta =
                (stats.temp_read_seconds - temp_read_before) +
                (stats.temp_write_seconds - temp_write_before);
            stats.spawn2_cell_compute_seconds +=
                detail::bc_family_positive_remainder(cell_elapsed, known_delta);
            if (first_visit) {
                temp_output_values = detail::bc_family_prepare_partial_temp_output(
                    layout,
                    std::move(temp_output_values),
                    false,
                    options,
                    stats
                );
                spawn2_partial_write_cids.push_back(cell.cid);
                spawn2_partial_write_values.push_back(std::move(temp_output_values));
            }
            if (!dense_final_values.empty()) {
                dense_batch.push_back(DenseFinalizeCell{i, std::move(dense_final_values)});
                if (dense_batch.size() >= compact_batch_limit) {
                    flush_dense_batch();
                }
            }
        }
        flush_dense_batch();
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn2_partial_values);
        std::vector<int64_t>().swap(spawn2_partial_prefetch_index);
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn2_scratch_values);
        std::vector<int64_t>().swap(spawn2_scratch_prefetch_index);
        temp.write_partial2_batch(
            spawn2_partial_write_cids,
            spawn2_partial_write_values,
            stats
        );
        std::vector<CellId>().swap(spawn2_partial_write_cids);
        std::vector<BCFamilyValueVector<StorageT>>().swap(spawn2_partial_write_values);
        const double release_t0 = bc_single_chunk_now_seconds();
        future2_window.release_except(cached.keep_cids);
        stats.single.future_release_seconds += bc_single_chunk_now_seconds() - release_t0;
    }
    bc_family_add_future_window_stats(stats.single, future2_window.stats(), false);
    const double future2_release_all_t0 = bc_single_chunk_now_seconds();
    future2_window.release_all();
    stats.single.future_release_seconds += bc_single_chunk_now_seconds() - future2_release_all_t0;

    if (next_output_cid != current_position.cell_count()) {
        detail::bc_family_flush_ready_outputs(
            pending,
            pending_value_bytes,
            next_output_cid,
            output_streamer,
            stats
        );
    }
    if (next_output_cid != current_position.cell_count()) {
        throw std::runtime_error("BC family solve did not finalize all output cells");
    }

    const double output_finish_t0 = bc_single_chunk_now_seconds();
    BCSingleChunkSolveFileResult single_result = output_streamer.finish();
    stats.output_finish_seconds += bc_single_chunk_now_seconds() - output_finish_t0;
    const double temp_close_t0 = bc_single_chunk_now_seconds();
    temp.wait_all_writes(stats);
    temp.close();
    stats.temp_close_seconds += bc_single_chunk_now_seconds() - temp_close_t0;
    if (!options.keep_temp_files) {
        const double cleanup_t0 = bc_single_chunk_now_seconds();
        temp.cleanup();
        std::filesystem::remove_all(temp_dir, cleanup_ec);
        stats.single.partial_cleanup_seconds += bc_single_chunk_now_seconds() - cleanup_t0;
    }

    BCFamilySolveFileResult result;
    result.position_bytes = single_result.position_bytes;
    result.success_bytes = single_result.success_bytes;
    stats.single.output_values = single_result.stats.output_values;
    stats.single.output_bytes = single_result.stats.output_bytes;
    result.stats = std::move(stats);
    return result;
}

template <typename StorageT>
BCFamilySolveFileResult bc_family_solve_exact_layer_to_files(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    BCWritableFile &position_file,
    BCWritableFile &success_file,
    const std::filesystem::path &temp_dir,
    const BCFamilySolveOptions<StorageT> &options,
    BCFamilySolveWorkspace<StorageT> *workspace = nullptr
) {
    const BCFamilyPartitionLayerMap current_partition =
        build_family_partition_layer_map_from_axis(current_position.axis());
    const BCFamilyPartitionLayerMap future2_partition =
        build_family_partition_layer_map_from_axis(future2_position.axis());
    const BCFamilyPartitionLayerMap future4_partition =
        build_family_partition_layer_map_from_axis(future4_position.axis());
    return bc_family_solve_layer_to_files<StorageT>(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        current_partition,
        future2_partition,
        future4_partition,
        position_file,
        success_file,
        temp_dir,
        options,
        workspace
    );
}

} // namespace BC
