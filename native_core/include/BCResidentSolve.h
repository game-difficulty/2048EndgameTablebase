#pragma once

#include "BCFutureSuccessLookup.h"
#include "BCPartialStore.h"
#include "BCPositionScanner.h"
#include "BCSolveEdgeKernel.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace BC {

inline constexpr int kBCResidentCellDynamicChunk = 1;

struct BCResidentSolveStats {
    uint64_t current_cells = 0U;
    uint64_t current_nonempty_cells = 0U;
    uint64_t current_empty_cells = 0U;
    uint64_t current_rows = 0U;
    uint64_t current_boards = 0U;
    uint64_t output_values = 0U;
    uint64_t output_bytes = 0U;
    uint64_t queries2 = 0U;
    uint64_t queries4 = 0U;
    uint64_t found2 = 0U;
    uint64_t found4 = 0U;
    uint64_t terminal_success_rows = 0U;
    double future_index_seconds = 0.0;
    double recalc_seconds = 0.0;
    double write_seconds = 0.0;
    BCSolveEdgeStats edge;
};

template <typename StorageT>
struct BCResidentSolveOptions {
    uint32_t row_width = 1U;
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>();
    StorageT zero_value = bc_success_zero_value_for_dtype<StorageT>(dtype);
    StorageT terminal_value = bc_success_terminal_value_for_dtype<StorageT>(dtype);
    BCDirectionMask directions = BCDirectionMask::Both;
    BCSolveTargetFamilyFilter filter2;
    BCSolveTargetFamilyFilter filter4;
    BCSolveEdgeOptions edge_options;
    const BCQuadrantWordSumTable *word_sums = nullptr;
    int num_threads = 0;

    void set_dtype(BCSuccessDTypeMode mode) {
        if (!bc_success_dtype_matches_type<StorageT>(mode)) {
            throw std::invalid_argument("BC resident solve option dtype does not match storage type");
        }
        dtype = mode;
        zero_value = bc_success_zero_value_for_dtype<StorageT>(dtype);
        terminal_value = bc_success_terminal_value_for_dtype<StorageT>(dtype);
    }
};

struct BCResidentCompactStats {
    uint64_t input_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t live_cells = 0U;
    uint64_t empty_cells = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    double compact_seconds = 0.0;
};

struct BCResidentArchivePruneResult {
    bool pruned = false;
    BCResidentCompactStats stats;
};

template <typename StorageT>
struct BCResidentRawSolveResult {
    std::vector<StorageT> values;
    std::vector<uint64_t> cell_value_offsets;
    BCResidentSolveStats stats;
};

template <typename StorageT>
struct BCResidentSolvedLayer {
    static_assert(
        std::is_same_v<StorageT, uint32_t> || std::is_same_v<StorageT, uint64_t> ||
        std::is_same_v<StorageT, float> || std::is_same_v<StorageT, double>,
        "unsupported BC resident solved layer value type"
    );

    BCPositionLayerReader position;
    std::vector<StorageT> success_values;
    std::vector<uint8_t> keep_rows;
    BCFutureSuccessLookupView<StorageT> lookup;
    BCResidentCompactStats compact_stats;
    uint32_t row_width = 1U;
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>();

    BCResidentSolvedLayer() = default;

    BCResidentSolvedLayer(BCResidentSolvedLayer &&other) noexcept {
        *this = std::move(other);
    }

    BCResidentSolvedLayer &operator=(BCResidentSolvedLayer &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        position = std::move(other.position);
        success_values = std::move(other.success_values);
        keep_rows = std::move(other.keep_rows);
        lookup = std::move(other.lookup);
        compact_stats = other.compact_stats;
        row_width = other.row_width;
        dtype = other.dtype;
        refresh_lookup_view_after_move();
        return *this;
    }

    BCResidentSolvedLayer(const BCResidentSolvedLayer &) = delete;
    BCResidentSolvedLayer &operator=(const BCResidentSolvedLayer &) = delete;

    void open(
        std::vector<uint8_t> position_bytes,
        std::vector<StorageT> values,
        const BCLut &lut,
        uint32_t row_width = 1U,
        BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>(),
        std::vector<uint8_t> keep = {}
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC resident solved layer row_width must be non-zero");
        }
        if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
            throw std::invalid_argument("BC resident solved layer dtype does not match storage type");
        }
        position.open(std::move(position_bytes), lut);
        this->row_width = row_width;
        this->dtype = dtype;
        success_values = std::move(values);
        keep_rows = std::move(keep);
        lookup.open_flat(
            lut,
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size(),
            row_width,
            keep_rows.empty() ? nullptr : keep_rows.data(),
            keep_rows.size()
        );
    }

    void open(
        BCPositionLayerReader position_reader,
        std::vector<StorageT> values,
        uint32_t row_width = 1U,
        BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>(),
        std::vector<uint8_t> keep = {}
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC resident solved layer row_width must be non-zero");
        }
        if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
            throw std::invalid_argument("BC resident solved layer dtype does not match storage type");
        }
        position = std::move(position_reader);
        this->row_width = row_width;
        this->dtype = dtype;
        success_values = std::move(values);
        keep_rows = std::move(keep);
        lookup.open_flat(
            position.lut(),
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size(),
            row_width,
            keep_rows.empty() ? nullptr : keep_rows.data(),
            keep_rows.size()
        );
    }

    void refresh_lookup_view() {
        lookup.rebind_flat_values(
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size(),
            keep_rows.empty() ? nullptr : keep_rows.data(),
            keep_rows.size()
        );
    }

private:
    void refresh_lookup_view_after_move() {
        if (position.cell_count() == 0U) {
            return;
        }
        lookup.rebind_flat_values(
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size(),
            keep_rows.empty() ? nullptr : keep_rows.data(),
            keep_rows.size()
        );
    }
};

template <typename StorageT>
struct BCResidentLayerResult {
    BCResidentSolvedLayer<StorageT> layer;
    BCResidentSolveStats solve_stats;
};

using BCResidentUInt32RawSolveResult = BCResidentRawSolveResult<uint32_t>;
using BCResidentUInt32SolvedLayer = BCResidentSolvedLayer<uint32_t>;

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

[[nodiscard]] inline double bc_resident_solve_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline int bc_resident_solve_effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

inline void bc_resident_solve_accumulate_edge_stats(
    BCSolveEdgeStats &dst,
    const BCSolveEdgeStats &src
) {
    dst.source_boards += src.source_boards;
    dst.terminal_success_boards += src.terminal_success_boards;
    dst.empty_slots += src.empty_slots;
    dst.spawned_boards += src.spawned_boards;
    dst.move_all_dir_calls += src.move_all_dir_calls;
    dst.selective_move_calls += src.selective_move_calls;
    dst.unchanged_moves += src.unchanged_moves;
    dst.prefilter_checks += src.prefilter_checks;
    dst.prefilter_skips += src.prefilter_skips;
    dst.canonicalized_candidates += src.canonicalized_candidates;
    dst.encoded_queries += src.encoded_queries;
    dst.encode_rejects += src.encode_rejects;
    dst.future_lookup_count += src.future_lookup_count;
    dst.future_lookup_misses += src.future_lookup_misses;
    dst.finalized_boards += src.finalized_boards;
    dst.batch_flushes += src.batch_flushes;
    dst.batch_tail_flushes += src.batch_tail_flushes;
    dst.batch_source_boards += src.batch_source_boards;
    dst.canonical_flushes += src.canonical_flushes;
}

inline void bc_resident_solve_accumulate_stats(
    BCResidentSolveStats &dst,
    const BCResidentSolveStats &src
) {
    dst.current_cells += src.current_cells;
    dst.current_nonempty_cells += src.current_nonempty_cells;
    dst.current_empty_cells += src.current_empty_cells;
    dst.current_rows += src.current_rows;
    dst.current_boards += src.current_boards;
    dst.output_values += src.output_values;
    dst.output_bytes += src.output_bytes;
    dst.queries2 += src.queries2;
    dst.queries4 += src.queries4;
    dst.found2 += src.found2;
    dst.found4 += src.found4;
    dst.terminal_success_rows += src.terminal_success_rows;
    dst.future_index_seconds += src.future_index_seconds;
    dst.recalc_seconds += src.recalc_seconds;
    dst.write_seconds += src.write_seconds;
    bc_resident_solve_accumulate_edge_stats(dst.edge, src.edge);
}

[[nodiscard]] inline uint64_t bc_resident_position_total_rows(const BCPositionLayerReader &position) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        rows = bc_checked_add_u64(
            rows,
            position.descriptor(cid).success_rows,
            "BC resident total row count overflow"
        );
    }
    return rows;
}

[[nodiscard]] inline std::vector<uint64_t> bc_resident_cell_value_offsets(
    const BCPositionLayerReader &position
) {
    std::vector<uint64_t> offsets(static_cast<size_t>(position.cell_count()) + 1U, 0U);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        offsets[static_cast<size_t>(cid) + 1U] = bc_checked_add_u64(
            offsets[static_cast<size_t>(cid)],
            position.descriptor(cid).success_rows,
            "BC resident cell value offset overflow"
        );
    }
    return offsets;
}

struct BCResidentSolveWorkItem {
    CellId cid = 0U;
    uint32_t bucket_begin = 0U;
    uint32_t bucket_end = 0U;
    uint32_t word_begin = 0U;
    uint32_t word_end = 0U;
};

struct BCResidentSolveWorkPlan {
    std::vector<BCResidentSolveWorkItem> items;
    BCResidentSolveStats stats;
};

inline constexpr uint32_t kBCResidentLargeBucketMinWords = 512U;
inline constexpr uint32_t kBCResidentLargeBucketChunkWords = 256U;

[[nodiscard]] inline BCResidentSolveWorkPlan bc_resident_build_solve_work_plan(
    const BCPositionLayerReader &position,
    uint32_t row_width = 1U
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC resident work plan row_width must be non-zero");
    }
    BCResidentSolveWorkPlan plan;
    plan.items.reserve(position.cell_count());
    const BCLut &lut = position.lut();
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        ++plan.stats.current_cells;
        const BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            ++plan.stats.current_empty_cells;
            continue;
        }
        ++plan.stats.current_nonempty_cells;
        plan.stats.output_values = bc_checked_add_u64(
            plan.stats.output_values,
            static_cast<uint64_t>(desc.success_rows) * row_width,
            "BC resident work plan output value count overflow"
        );
        const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
        uint32_t small_begin = 0U;
        uint32_t small_end = 0U;
        auto flush_small = [&]() {
            if (small_begin == small_end) {
                return;
            }
            plan.items.push_back(BCResidentSolveWorkItem{
                cid,
                small_begin,
                small_end,
                0U,
                0U
            });
            small_begin = small_end;
        };
        for (uint32_t bucket_i = 0U; bucket_i < buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = buckets.data[bucket_i];
            const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            if (word_count >= kBCResidentLargeBucketMinWords) {
                flush_small();
                for (uint32_t word_begin = 0U; word_begin < word_count;) {
                    const uint32_t word_end = std::min<uint32_t>(
                        word_count,
                        word_begin + kBCResidentLargeBucketChunkWords
                    );
                    plan.items.push_back(BCResidentSolveWorkItem{
                        cid,
                        bucket_i,
                        bucket_i + 1U,
                        word_begin,
                        word_end
                    });
                    word_begin = word_end;
                }
                small_begin = bucket_i + 1U;
                small_end = small_begin;
            } else {
                if (small_begin == small_end) {
                    small_begin = bucket_i;
                }
                small_end = bucket_i + 1U;
            }
        }
        flush_small();
    }
    return plan;
}

template <typename StorageT>
struct BCResidentBatchWorkspace {
    static constexpr uint32_t kBatchSize = 512U;
    static constexpr size_t kBestCount = static_cast<size_t>(kBatchSize) * kBCBoardCellCount;

    std::array<uint64_t, kBatchSize> boards{};
    std::array<uint64_t, kBatchSize> output_indices{};
    std::array<uint16_t, kBatchSize> empty_masks{};
    std::array<uint8_t, kBatchSize> terminal{};
    std::array<uint32_t, kBatchSize> empty_counts{};
    std::array<StorageT, kBestCount> best2{};
    std::array<StorageT, kBestCount> best4{};
    std::vector<uint64_t> canonical2_boards;
    std::vector<uint64_t> canonical4_boards;
    std::vector<uint16_t> canonical2_refs;
    std::vector<uint16_t> canonical4_refs;
    std::vector<BCSolvePreparedQuery> queries2;
    std::vector<BCSolvePreparedQuery> queries4;
    uint32_t count = 0U;

    BCResidentBatchWorkspace() {
        constexpr size_t max_candidates =
            static_cast<size_t>(kBatchSize) * kBCBoardCellCount * 4U;
        canonical2_boards.reserve(max_candidates);
        canonical4_boards.reserve(max_candidates);
        canonical2_refs.reserve(max_candidates);
        canonical4_refs.reserve(max_candidates);
        queries2.reserve(max_candidates);
        queries4.reserve(max_candidates);
    }

    void clear_batch() {
        count = 0U;
        canonical2_boards.clear();
        canonical4_boards.clear();
        canonical2_refs.clear();
        canonical4_refs.clear();
        queries2.clear();
        queries4.clear();
    }
};

using BCResidentUInt32BatchWorkspace = BCResidentBatchWorkspace<uint32_t>;

template <typename StorageT>
[[nodiscard]] inline StorageT bc_resident_reduce_weighted(
    long double sum2,
    long double sum4,
    uint32_t empty_count,
    double spawn_rate4,
    StorageT zero_value
) {
    if (empty_count == 0U) {
        return zero_value;
    }
    if constexpr (std::is_same_v<StorageT, uint32_t>) {
        if (spawn_rate4 == 0.1) {
            const uint64_t numerator =
                9ULL * static_cast<uint64_t>(sum2) + static_cast<uint64_t>(sum4);
            const uint64_t denominator = 10ULL * static_cast<uint64_t>(empty_count);
            return static_cast<uint32_t>(numerator / denominator);
        }
    }
    const long double p4 = static_cast<long double>(spawn_rate4);
    const long double p2 = 1.0L - p4;
    return static_cast<StorageT>((sum2 * p2 + sum4 * p4) / static_cast<long double>(empty_count));
}

template <typename StorageT>
[[nodiscard]] inline uint8_t bc_resident_push_candidate(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t spawned,
    uint64_t moved,
    uint16_t ref,
    BCDirectionMask move_axis,
    const BCResidentSolveOptions<StorageT> &options,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs
) {
    if (moved == spawned) {
        return 1U;
    }
    if (filter.enabled) {
        if (!bc_solve_physical_target_family_may_hit(
                lut,
                axis,
                filter,
                moved,
                move_axis,
                options.word_sums)) {
            return 2U;
        }
    }
    boards.push_back(moved);
    refs.push_back(ref);
    return 0U;
}

template <typename StorageT>
inline void bc_resident_flush_canonical(
    const BCSolvePreparedQueryEncoder &encoder,
    uint8_t spawn_tile_rank,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs,
    std::vector<BCSolvePreparedQuery> &queries,
    const BCResidentSolveOptions<StorageT> &options,
    BCResidentSolveStats &stats
) {
    (void)stats;
    if (boards.empty()) {
        return;
    }
    CanonicalBatch::canonicalize_inplace(
        boards.data(),
        boards.size(),
        options.edge_options.canonical_symm_mode
    );
    for (size_t i = 0U; i < boards.size(); ++i) {
        BCSolvePreparedQuery query;
        const bool encoded = encoder.encode(
            unpack_board_to_quadrants(boards[i]),
            refs[i],
            spawn_tile_rank,
            BCDirectionMask::Both,
            query
        );
        if (!encoded) {
            continue;
        }
        queries.push_back(query);
    }
    boards.clear();
    refs.clear();
}

template <typename StorageT>
inline void bc_resident_solve_batch(
    BCResidentBatchWorkspace<StorageT> &workspace,
    std::vector<StorageT> &out_values,
    const BCLut &lut,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    const BCFutureSuccessLookupView<StorageT> &future2_lookup,
    const BCFutureSuccessLookupView<StorageT> &future4_lookup,
    const BCResidentSolveOptions<StorageT> &options,
    BCResidentSolveStats &stats
) {
    const uint32_t count = workspace.count;
    if (count == 0U) {
        return;
    }
    (void)stats;
    workspace.canonical2_boards.clear();
    workspace.canonical4_boards.clear();
    workspace.canonical2_refs.clear();
    workspace.canonical4_refs.clear();
    workspace.queries2.clear();
    workspace.queries4.clear();

    const bool success_check_enabled = bc_solve_success_check_enabled(options.edge_options);
    const bool fast_unfiltered_both =
        options.directions == BCDirectionMask::Both &&
        !options.filter2.enabled &&
        !options.filter4.enabled;
    auto push_moved_unfiltered = [&](
        uint64_t spawned,
        uint64_t moved,
        uint16_t ref,
        std::vector<uint64_t> &canonical_boards,
        std::vector<uint16_t> &canonical_refs
    ) {
        if (moved == spawned) {
            return;
        }
        canonical_boards.push_back(moved);
        canonical_refs.push_back(ref);
    };
    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = workspace.boards[board_slot];
        workspace.empty_counts[board_slot] = 0U;
        workspace.empty_masks[board_slot] = 0U;
        if (success_check_enabled) {
            workspace.terminal[board_slot] =
                bc_solve_is_success_board(board, options.edge_options) ? 1U : 0U;
            if (workspace.terminal[board_slot] != 0U) {
                continue;
            }
        }
        uint32_t empty_mask = bc_zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            ++workspace.empty_counts[board_slot];

            if (fast_unfiltered_both) {
                const uint64_t spawned2 =
                    set_board_tile_unchecked(board, cell, options.edge_options.spawn2_tile_rank);
                const auto moved2_horizontal = BoardMover::move_horizontal_pair(spawned2);
                const auto moved2_vertical = BoardMover::move_vertical_pair(spawned2);
                push_moved_unfiltered(
                    spawned2, moved2_horizontal.first, ref,
                    workspace.canonical2_boards, workspace.canonical2_refs);
                push_moved_unfiltered(
                    spawned2, moved2_horizontal.second, ref,
                    workspace.canonical2_boards, workspace.canonical2_refs);
                push_moved_unfiltered(
                    spawned2, moved2_vertical.first, ref,
                    workspace.canonical2_boards, workspace.canonical2_refs);
                push_moved_unfiltered(
                    spawned2, moved2_vertical.second, ref,
                    workspace.canonical2_boards, workspace.canonical2_refs);

                const uint64_t spawned4 =
                    set_board_tile_unchecked(board, cell, options.edge_options.spawn4_tile_rank);
                const auto moved4_horizontal = BoardMover::move_horizontal_pair(spawned4);
                const auto moved4_vertical = BoardMover::move_vertical_pair(spawned4);
                push_moved_unfiltered(
                    spawned4, moved4_horizontal.first, ref,
                    workspace.canonical4_boards, workspace.canonical4_refs);
                push_moved_unfiltered(
                    spawned4, moved4_horizontal.second, ref,
                    workspace.canonical4_boards, workspace.canonical4_refs);
                push_moved_unfiltered(
                    spawned4, moved4_vertical.first, ref,
                    workspace.canonical4_boards, workspace.canonical4_refs);
                push_moved_unfiltered(
                    spawned4, moved4_vertical.second, ref,
                    workspace.canonical4_boards, workspace.canonical4_refs);
                continue;
            }

            auto push_spawn = [&](uint8_t spawn_rank,
                                  const BCFamilyTable &axis,
                                  const BCSolveTargetFamilyFilter &filter,
                                  std::vector<uint64_t> &canonical_boards,
                                  std::vector<uint16_t> &canonical_refs) {
                const uint64_t spawned = set_board_tile_unchecked(board, cell, spawn_rank);
                if (options.directions == BCDirectionMask::Both) {
                    const auto moved = BoardMover::move_all_dir(spawned);
                    (void)bc_resident_push_candidate(
                        lut, axis, filter, spawned, std::get<0>(moved), ref,
                        BCDirectionMask::Horizontal, options, canonical_boards, canonical_refs);
                    (void)bc_resident_push_candidate(
                        lut, axis, filter, spawned, std::get<1>(moved), ref,
                        BCDirectionMask::Horizontal, options, canonical_boards, canonical_refs);
                    (void)bc_resident_push_candidate(
                        lut, axis, filter, spawned, std::get<2>(moved), ref,
                        BCDirectionMask::Vertical, options, canonical_boards, canonical_refs);
                    (void)bc_resident_push_candidate(
                        lut, axis, filter, spawned, std::get<3>(moved), ref,
                        BCDirectionMask::Vertical, options, canonical_boards, canonical_refs);
                } else {
                    if (bc_has_horizontal(options.directions)) {
                        const auto moved = BoardMover::move_horizontal_pair(spawned);
                        (void)bc_resident_push_candidate(
                            lut, axis, filter, spawned, moved.first, ref,
                            BCDirectionMask::Horizontal, options, canonical_boards, canonical_refs);
                        (void)bc_resident_push_candidate(
                            lut, axis, filter, spawned, moved.second, ref,
                            BCDirectionMask::Horizontal, options, canonical_boards, canonical_refs);
                    }
                    if (bc_has_vertical(options.directions)) {
                        const auto moved = BoardMover::move_vertical_pair(spawned);
                        (void)bc_resident_push_candidate(
                            lut, axis, filter, spawned, moved.first, ref,
                            BCDirectionMask::Vertical, options, canonical_boards, canonical_refs);
                        (void)bc_resident_push_candidate(
                            lut, axis, filter, spawned, moved.second, ref,
                            BCDirectionMask::Vertical, options, canonical_boards, canonical_refs);
                    }
                }
            };

            push_spawn(
                options.edge_options.spawn2_tile_rank,
                future2_axis,
                options.filter2,
                workspace.canonical2_boards,
                workspace.canonical2_refs
            );
            push_spawn(
                options.edge_options.spawn4_tile_rank,
                future4_axis,
                options.filter4,
                workspace.canonical4_boards,
                workspace.canonical4_refs
            );
        }
    }
    const BCSolvePreparedQueryEncoder future2_encoder(
        lut,
        future2_axis,
        options.edge_options.future_cell_modulus
    );
    const BCSolvePreparedQueryEncoder future4_encoder(
        lut,
        future4_axis,
        options.edge_options.future_cell_modulus
    );
    bc_resident_flush_canonical(
        future2_encoder,
        options.edge_options.spawn2_tile_rank,
        workspace.canonical2_boards,
        workspace.canonical2_refs,
        workspace.queries2,
        options,
        stats
    );
    bc_resident_flush_canonical(
        future4_encoder,
        options.edge_options.spawn4_tile_rank,
        workspace.canonical4_boards,
        workspace.canonical4_refs,
        workspace.queries4,
        options,
        stats
    );

    for (uint32_t lane = 0U; lane < options.row_width; ++lane) {
        std::fill_n(
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            options.zero_value
        );
        std::fill_n(
            workspace.best4.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            options.zero_value
        );
        (void)future2_lookup.reduce_max_queries(
            workspace.queries2,
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            lane,
            nullptr,
            true
        );
        (void)future4_lookup.reduce_max_queries(
            workspace.queries4,
            workspace.best4.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            lane,
            nullptr,
            true
        );

        for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
            StorageT value = options.zero_value;
            if (success_check_enabled && workspace.terminal[board_slot] != 0U) {
                value = options.terminal_value;
            } else if (workspace.empty_counts[board_slot] != 0U) {
                if constexpr (std::is_same_v<StorageT, uint32_t>) {
                    uint32_t mask = workspace.empty_masks[board_slot];
                    uint64_t sum2 = 0U;
                    uint64_t sum4 = 0U;
                    while (mask != 0U) {
                        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                        const size_t index = static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                        sum2 += workspace.best2[index];
                        sum4 += workspace.best4[index];
                    }
                    if (options.edge_options.spawn_rate4 == 0.1) {
                        const uint64_t numerator = 9ULL * sum2 + sum4;
                        const uint64_t denominator =
                            10ULL * static_cast<uint64_t>(workspace.empty_counts[board_slot]);
                        value = static_cast<uint32_t>(numerator / denominator);
                    } else {
                        value = bc_resident_reduce_weighted<StorageT>(
                            static_cast<long double>(sum2),
                            static_cast<long double>(sum4),
                            workspace.empty_counts[board_slot],
                            options.edge_options.spawn_rate4,
                            options.zero_value
                        );
                    }
                } else {
                    uint32_t mask = workspace.empty_masks[board_slot];
                    long double sum2 = 0.0L;
                    long double sum4 = 0.0L;
                    while (mask != 0U) {
                        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                        const size_t index = static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                        sum2 += static_cast<long double>(workspace.best2[index]);
                        sum4 += static_cast<long double>(workspace.best4[index]);
                    }
                    value = bc_resident_reduce_weighted<StorageT>(
                        sum2,
                        sum4,
                        workspace.empty_counts[board_slot],
                        options.edge_options.spawn_rate4,
                        options.zero_value
                    );
                }
            }
            const uint64_t row_index = workspace.output_indices[board_slot];
            const uint64_t value_index =
                row_index * static_cast<uint64_t>(options.row_width) + lane;
            if (value_index >= out_values.size()) {
                throw std::out_of_range("BC resident batch output value index out of range");
            }
            out_values[static_cast<size_t>(value_index)] = value;
        }
    }
    workspace.clear_batch();
}

template <typename StorageT>
inline BCResidentRawSolveResult<StorageT> bc_resident_solve_raw_values(
    const BCPositionLayerReader &current_position,
    const BCResidentSolvedLayer<StorageT> &future2,
    const BCResidentSolvedLayer<StorageT> &future4,
    const BCResidentSolveOptions<StorageT> &options
) {
    static_assert(
        std::is_same_v<StorageT, uint32_t> || std::is_same_v<StorageT, uint64_t> ||
        std::is_same_v<StorageT, float> || std::is_same_v<StorageT, double>,
        "unsupported BC resident raw solve value type"
    );
    if (options.row_width == 0U) {
        throw std::invalid_argument("BC resident solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.dtype)) {
        throw std::invalid_argument("BC resident solve dtype does not match storage type");
    }
    if (future2.row_width != options.row_width || future4.row_width != options.row_width) {
        throw std::invalid_argument("BC resident solve future row_width mismatch");
    }
    if (future2.position.cell_count() == 0U || future4.position.cell_count() == 0U) {
        throw std::invalid_argument("BC resident solve future layers must be open");
    }
    const uint64_t current_rows = bc_resident_position_total_rows(current_position);
    if (current_rows > std::numeric_limits<uint64_t>::max() / options.row_width ||
        current_rows * options.row_width > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC resident row value count exceeds size_t");
    }
    BCResidentRawSolveResult<StorageT> result;
    result.cell_value_offsets = bc_resident_cell_value_offsets(current_position);
    result.values.assign(
        static_cast<size_t>(current_rows * options.row_width),
        options.zero_value
    );
    const BCResidentSolveWorkPlan work_plan =
        bc_resident_build_solve_work_plan(current_position, options.row_width);

    const int threads = bc_resident_solve_effective_threads(options.num_threads);
    std::vector<BCResidentSolveStats> per_thread(static_cast<size_t>(threads));
    const double recalc_t0 = bc_resident_solve_now_seconds();
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCResidentSolveStats &thread_stats = per_thread[static_cast<size_t>(tid)];
        BCResidentBatchWorkspace<StorageT> workspace;
        auto flush = [&]() {
            bc_resident_solve_batch<StorageT>(
                workspace,
                result.values,
                current_position.lut(),
                future2.position.axis(),
                future4.position.axis(),
                future2.lookup,
                future4.lookup,
                options,
                thread_stats
            );
        };
#pragma omp for schedule(dynamic, 1)
        for (int64_t item_signed = 0;
             item_signed < static_cast<int64_t>(work_plan.items.size());
             ++item_signed) {
            const BCResidentSolveWorkItem &item =
                work_plan.items[static_cast<size_t>(item_signed)];
            const CellId cid = item.cid;
            const BCPositionCellDescriptor &desc = current_position.descriptor(cid);
            if (desc.empty() || desc.success_rows == 0U) {
                continue;
            }
            const uint64_t cell_base = result.cell_value_offsets[static_cast<size_t>(cid)];
            BCPositionCellScanner scanner(current_position, cid);
            for (uint32_t bucket_i = item.bucket_begin; bucket_i < item.bucket_end; ++bucket_i) {
                const bool ranged_bucket =
                    item.bucket_end == item.bucket_begin + 1U && item.word_end != 0U;
                scanner.for_each_bucket_word_range_board(
                    bucket_i,
                    ranged_bucket ? item.word_begin : 0U,
                    ranged_bucket ? item.word_end : 0U,
                    [&](const BCScannedBoardEntry &entry) {
                        workspace.boards[workspace.count] = entry.board;
                        workspace.output_indices[workspace.count] = cell_base + entry.local_success_row;
                        ++workspace.count;
                        if (workspace.count == BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                            flush();
                        }
                    }
                );
            }
        }
        flush();
    }
    const double recalc_t1 = bc_resident_solve_now_seconds();
    result.stats = work_plan.stats;
    result.stats.current_rows = current_rows;
    result.stats.current_boards = current_rows;
    result.stats.recalc_seconds = recalc_t1 - recalc_t0;
    for (const BCResidentSolveStats &stats : per_thread) {
        bc_resident_solve_accumulate_stats(result.stats, stats);
    }
    return result;
}

inline void bc_resident_append_u16_le(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

inline void bc_resident_append_u64_le(std::vector<uint8_t> &out, uint64_t value) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        out.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
    }
}

inline void bc_resident_store_u32_le(uint8_t *out, uint32_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    out[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    out[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

inline void bc_resident_store_u16_le(uint8_t *out, uint16_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
}

inline void bc_resident_store_u64_le(uint8_t *out, uint64_t value) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        out[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU);
    }
}

inline void bc_resident_store_position_header(uint8_t *out, const BCPositionHeader &header) {
    bc_resident_store_u32_le(out + 0U, header.magic);
    bc_resident_store_u32_le(out + 4U, header.format_version);
    bc_resident_store_u32_le(out + 8U, header.header_bytes);
    bc_resident_store_u32_le(out + 12U, header.key_mode);
    bc_resident_store_u32_le(out + 16U, header.rank_prefix_bits);
    bc_resident_store_u32_le(out + 20U, header.rank_prefix_type);
    bc_resident_store_u32_le(out + 24U, header.rank_payload_align);
    bc_resident_store_u32_le(out + 28U, header.family_unit);
    bc_resident_store_u32_le(out + 32U, header.axis_base_coord);
    bc_resident_store_u32_le(out + 36U, header.family_count);
    bc_resident_store_u64_le(out + 40U, header.layer_sum);
    bc_resident_store_u64_le(out + 48U, header.descriptor_count);
    bc_resident_store_u64_le(out + 56U, header.descriptor_table_offset);
    bc_resident_store_u64_le(out + 64U, header.descriptor_table_bytes);
    bc_resident_store_u64_le(out + 72U, header.bucket_meta_offset);
    bc_resident_store_u64_le(out + 80U, header.bucket_meta_bytes);
    bc_resident_store_u64_le(out + 88U, header.rank_payload_offset);
    bc_resident_store_u64_le(out + 96U, header.rank_payload_bytes);
    bc_resident_store_u64_le(out + 104U, header.axis_coord_table_bytes);
}

inline void bc_resident_store_axis_coord_table(uint8_t *out, const BCFamilyTable &axis) {
    const std::vector<FamilyCoord> &coords = axis.coords();
    if (coords.size() != axis.family_count()) {
        throw std::logic_error("BC resident position axis coord table size mismatch");
    }
    for (uint32_t i = 0U; i < axis.family_count(); ++i) {
        bc_resident_store_u32_le(out + static_cast<size_t>(i) * sizeof(uint32_t), coords[i]);
    }
}

inline void bc_resident_store_cell_descriptor(
    uint8_t *out,
    const BCPositionCellDescriptor &descriptor
) {
    bc_resident_store_u32_le(out + 0U, descriptor.bucket_count);
    bc_resident_store_u32_le(out + 4U, descriptor.success_rows);
    bc_resident_store_u64_le(out + 8U, descriptor.bucket_meta_offset);
    bc_resident_store_u64_le(out + 16U, descriptor.rank_payload_offset);
    bc_resident_store_u64_le(out + 24U, descriptor.rank_payload_bytes);
    bc_resident_store_u32_le(out + 32U, descriptor.reserved0);
    bc_resident_store_u32_le(out + 36U, descriptor.flags_or_padding);
}

inline void bc_resident_store_bucket_entry(uint8_t *out, const BCBucketEntry &entry) {
    bc_resident_store_u64_le(out + 0U, entry.key);
    bc_resident_store_u32_le(out + 8U, entry.rank_payload_offset);
    bc_resident_store_u32_le(out + 12U, entry.success_row_offset);
}

inline void bc_resident_append_padding(std::vector<uint8_t> &out, uint32_t bytes) {
    out.insert(out.end(), bytes, 0U);
}

inline void bc_resident_append_prefix256_le(
    std::vector<uint8_t> &out,
    const uint64_t *bitmap,
    uint32_t word_count,
    uint32_t bitmap_len
) {
    uint32_t running = 0U;
    const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
    for (uint32_t block = 0U; block < prefix_count; ++block) {
        if (running > std::numeric_limits<RankPrefix>::max()) {
            throw std::logic_error("BC resident compact prefix running popcount exceeds uint16");
        }
        bc_resident_append_u16_le(out, static_cast<RankPrefix>(running));
        const uint32_t first_bit = block * kBCRankPrefixBits;
        const uint32_t last_bit = std::min<uint32_t>(bitmap_len, first_bit + kBCRankPrefixBits);
        const uint32_t first_word = first_bit / kBCBitmapWordBits;
        const uint32_t last_word = words_for_bits(last_bit);
        if (last_word > word_count) {
            throw std::invalid_argument("BC resident compact bitmap is shorter than bitmap_len");
        }
        for (uint32_t word = first_word; word < last_word; ++word) {
            uint64_t value = bitmap[word];
            if (word + 1U == last_word && (last_bit & 63U) != 0U) {
                value &= (1ULL << (last_bit & 63U)) - 1ULL;
            }
            running += popcount64(value);
        }
    }
}

inline void bc_resident_append_bitmap_le(
    std::vector<uint8_t> &out,
    const uint64_t *bitmap,
    uint32_t word_count
) {
    for (uint32_t i = 0U; i < word_count; ++i) {
        bc_resident_append_u64_le(out, bitmap[i]);
    }
}

[[nodiscard]] inline std::vector<uint8_t> bc_resident_build_position_bytes_from_payloads(
    const BCFamilyTable &axis,
    const std::vector<FinalizedCellPayload> &payloads
) {
    if (axis.family_count() == 0U) {
        throw std::invalid_argument("BC resident position build requires non-empty axis");
    }
    const BCCellMatrix matrix(axis);
    if (payloads.size() != matrix.cell_count()) {
        throw std::invalid_argument("BC resident position build payload count mismatch");
    }

    const uint64_t descriptor_count = payloads.size();
    if (descriptor_count >
        std::numeric_limits<uint64_t>::max() / kBCPositionCellDescriptorBytes) {
        throw std::overflow_error("BC resident position descriptor bytes overflow");
    }
    const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
    const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis.family_count());
    const uint64_t descriptor_offset = bc_checked_add_u64(
        kBCPositionHeaderBytes,
        axis_coord_bytes,
        "BC resident position descriptor offset overflow"
    );
    const uint64_t bucket_offset = bc_checked_add_u64(
        descriptor_offset,
        descriptor_bytes,
        "BC resident position bucket offset overflow"
    );

    std::vector<BCPositionCellDescriptor> descriptors(payloads.size());
    uint64_t bucket_cursor = 0U;
    uint64_t rank_cursor = 0U;
    for (CellId cid = 0U; cid < payloads.size(); ++cid) {
        const FinalizedCellPayload &payload = payloads[static_cast<size_t>(cid)];
        if (payload.buckets.empty()) {
            if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
                throw std::invalid_argument("BC resident empty payload has metadata");
            }
            BCPositionCellDescriptor descriptor;
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
            descriptors[static_cast<size_t>(cid)] = descriptor;
            continue;
        }
        if (payload.buckets.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC resident position bucket_count exceeds uint32");
        }
        for (size_t i = 1U; i < payload.buckets.size(); ++i) {
            if (payload.buckets[i - 1U].key >= payload.buckets[i].key) {
                throw std::invalid_argument("BC resident position buckets must be sorted");
            }
        }

        const uint64_t bucket_bytes =
            static_cast<uint64_t>(payload.buckets.size()) * kBCPositionBucketEntryBytes;
        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = static_cast<uint32_t>(payload.buckets.size());
        descriptor.success_rows = payload.success_rows;
        descriptor.bucket_meta_offset = bucket_cursor;
        descriptor.rank_payload_offset = rank_cursor;
        descriptor.rank_payload_bytes = payload.rank_payload.size();
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;
        descriptors[static_cast<size_t>(cid)] = descriptor;
        bucket_cursor = bc_checked_add_u64(
            bucket_cursor,
            bucket_bytes,
            "BC resident position bucket bytes overflow"
        );
        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            payload.rank_payload.size(),
            "BC resident position rank bytes overflow"
        );
    }

    const uint64_t rank_offset = bc_checked_add_u64(
        bucket_offset,
        bucket_cursor,
        "BC resident position rank offset overflow"
    );
    const uint64_t logical_size = bc_checked_add_u64(
        rank_offset,
        rank_cursor,
        "BC resident position logical size overflow"
    );
    if (logical_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC resident position logical size exceeds size_t");
    }

    BCPositionHeader header;
    header.family_unit = axis.family_unit();
    header.axis_base_coord = axis.axis_base_coord();
    header.family_count = axis.family_count();
    header.layer_sum = axis.layer_sum();
    header.axis_coord_table_bytes = axis_coord_bytes;
    header.descriptor_count = descriptor_count;
    header.descriptor_table_offset = descriptor_offset;
    header.descriptor_table_bytes = descriptor_bytes;
    header.bucket_meta_offset = bucket_offset;
    header.bucket_meta_bytes = bucket_cursor;
    header.rank_payload_offset = rank_offset;
    header.rank_payload_bytes = rank_cursor;

    std::vector<uint8_t> out(static_cast<size_t>(logical_size));
    bc_resident_store_position_header(out.data(), header);
    bc_resident_store_axis_coord_table(out.data() + kBCPositionHeaderBytes, axis);

    uint8_t *descriptor_out = out.data() + static_cast<size_t>(descriptor_offset);
    for (const BCPositionCellDescriptor &descriptor : descriptors) {
        bc_resident_store_cell_descriptor(descriptor_out, descriptor);
        descriptor_out += kBCPositionCellDescriptorBytes;
    }

    uint8_t *bucket_out = out.data() + static_cast<size_t>(bucket_offset);
    for (const FinalizedCellPayload &payload : payloads) {
        for (const BCBucketEntry &bucket : payload.buckets) {
            bc_resident_store_bucket_entry(bucket_out, bucket);
            bucket_out += kBCPositionBucketEntryBytes;
        }
    }

    uint8_t *rank_out = out.data() + static_cast<size_t>(rank_offset);
    for (const FinalizedCellPayload &payload : payloads) {
        if (!payload.rank_payload.empty()) {
            std::memcpy(rank_out, payload.rank_payload.data(), payload.rank_payload.size());
            rank_out += payload.rank_payload.size();
        }
    }
    if (descriptor_out != out.data() + static_cast<size_t>(bucket_offset) ||
        bucket_out != out.data() + static_cast<size_t>(rank_offset) ||
        rank_out != out.data() + static_cast<size_t>(logical_size)) {
        throw std::logic_error("BC resident position build size mismatch");
    }
    return out;
}

[[nodiscard]] inline uint32_t bc_resident_countr_zero64(uint64_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC resident countr_zero64 requires non-zero value");
    }
#if defined(_MSC_VER)
    unsigned long index = 0U;
#if defined(_M_X64) || defined(_M_ARM64)
    _BitScanForward64(&index, value);
    return static_cast<uint32_t>(index);
#else
    const uint32_t lo = static_cast<uint32_t>(value);
    if (lo != 0U) {
        _BitScanForward(&index, lo);
        return static_cast<uint32_t>(index);
    }
    _BitScanForward(&index, static_cast<uint32_t>(value >> 32U));
    return 32U + static_cast<uint32_t>(index);
#endif
#else
    return static_cast<uint32_t>(__builtin_ctzll(value));
#endif
}

template <typename StorageT>
inline void bc_resident_compact_cell(
    const BCPositionLayerReader &position,
    CellId cid,
    const std::vector<StorageT> &raw_values,
    uint64_t cell_value_offset,
    uint32_t row_width,
    StorageT zero_value,
    FinalizedCellPayload &payload,
    std::vector<StorageT> &cell_success_values,
    BCResidentCompactStats &stats
) {
    const BCPositionCellDescriptor &desc = position.descriptor(cid);
    if (desc.empty() || desc.success_rows == 0U) {
        ++stats.empty_cells;
        return;
    }
    stats.input_rows += desc.success_rows;
    const BCLut &lut = position.lut();
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
    const BCRankPayloadView rank_payload = position.rank_payload_for_cell(cid);
    uint64_t success_cursor = 0U;
    for (uint32_t bucket_i = 0U; bucket_i < buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = buckets.data[bucket_i];
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (bitmap_end > rank_payload.size) {
            throw std::out_of_range("BC resident compact source bitmap exceeds rank payload");
        }
        std::vector<uint64_t> keep_bitmap(word_count, 0U);
        uint32_t bucket_seen = 0U;
        uint32_t bucket_kept = 0U;
        const uint8_t *bitmap_words = rank_payload.data + bitmap_offset;
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = bc_resident_countr_zero64(word);
                const uint32_t rank = word_i * kBCBitmapWordBits + bit;
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (rank >= bitmap_len || local_row >= desc.success_rows) {
                    throw std::out_of_range("BC resident compact source row exceeds descriptor");
                }
                const uint64_t row_base =
                    (cell_value_offset + local_row) * static_cast<uint64_t>(row_width);
                if (row_base + row_width > raw_values.size()) {
                    throw std::out_of_range("BC resident compact source row exceeds raw values");
                }
                bool keep_row = false;
                for (uint32_t lane = 0U; lane < row_width; ++lane) {
                    if (raw_values[static_cast<size_t>(row_base + lane)] != zero_value) {
                        keep_row = true;
                        break;
                    }
                }
                if (keep_row) {
                    keep_bitmap[word_i] |= (1ULL << bit);
                    for (uint32_t lane = 0U; lane < row_width; ++lane) {
                        cell_success_values.push_back(
                            raw_values[static_cast<size_t>(row_base + lane)]
                        );
                    }
                    ++bucket_kept;
                    ++stats.live_rows;
                } else {
                    ++stats.zero_pruned_rows;
                }
                ++bucket_seen;
                word &= word - 1ULL;
            }
        }
        if (bucket_kept == 0U) {
            continue;
        }
        const uint32_t payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
        bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_offset);
        const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        bc_resident_append_prefix256_le(payload.rank_payload, keep_bitmap.data(), word_count, bitmap_len);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        bc_resident_append_padding(
            payload.rank_payload,
            out_bitmap_offset - static_cast<uint32_t>(payload.rank_payload.size())
        );
        bc_resident_append_bitmap_le(payload.rank_payload, keep_bitmap.data(), word_count);
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC resident compact success row offset exceeds uint32");
        }
        payload.buckets.push_back(BCBucketEntry{
            bucket.key,
            rank_payload_offset,
            static_cast<uint32_t>(success_cursor)
        });
        success_cursor += bucket_kept;
    }
    if (success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident compact success rows exceed uint32");
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    if (payload.success_rows == 0U) {
        payload.buckets.clear();
        payload.rank_payload.clear();
        ++stats.empty_cells;
    } else {
        ++stats.live_cells;
    }
}

template <typename StorageT, typename KeepRowFn>
inline void bc_resident_compact_cell_in_place_if(
    const BCPositionLayerReader &position,
    CellId cid,
    std::vector<StorageT> &raw_values,
    uint64_t cell_value_offset,
    uint32_t row_width,
    FinalizedCellPayload &payload,
    BCResidentCompactStats &stats,
    KeepRowFn &&keep_row_fn
) {
    const BCPositionCellDescriptor &desc = position.descriptor(cid);
    if (desc.empty() || desc.success_rows == 0U) {
        ++stats.empty_cells;
        return;
    }
    stats.input_rows += desc.success_rows;
    const BCLut &lut = position.lut();
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
    const BCRankPayloadView rank_payload = position.rank_payload_for_cell(cid);
    uint64_t success_cursor = 0U;
    for (uint32_t bucket_i = 0U; bucket_i < buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = buckets.data[bucket_i];
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (bitmap_end > rank_payload.size) {
            throw std::out_of_range("BC resident in-place compact source bitmap exceeds rank payload");
        }
        std::vector<uint64_t> keep_bitmap(word_count, 0U);
        const uint64_t bucket_success_offset = success_cursor;
        uint32_t bucket_seen = 0U;
        uint32_t bucket_kept = 0U;
        const uint8_t *bitmap_words = rank_payload.data + bitmap_offset;
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = bc_resident_countr_zero64(word);
                const uint32_t rank = word_i * kBCBitmapWordBits + bit;
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (rank >= bitmap_len || local_row >= desc.success_rows) {
                    throw std::out_of_range("BC resident in-place compact source row exceeds descriptor");
                }
                const uint64_t row_base =
                    (cell_value_offset + local_row) * static_cast<uint64_t>(row_width);
                if (row_base + row_width > raw_values.size()) {
                    throw std::out_of_range("BC resident in-place compact source row exceeds raw values");
                }
                const bool keep_row =
                    keep_row_fn(raw_values.data() + static_cast<size_t>(row_base), row_width);
                if (keep_row) {
                    keep_bitmap[word_i] |= (1ULL << bit);
                    const uint64_t dst_base =
                        (cell_value_offset + success_cursor) * static_cast<uint64_t>(row_width);
                    if (dst_base + row_width > raw_values.size()) {
                        throw std::out_of_range("BC resident in-place compact destination exceeds raw values");
                    }
                    if (dst_base != row_base) {
                        for (uint32_t lane = 0U; lane < row_width; ++lane) {
                            raw_values[static_cast<size_t>(dst_base + lane)] =
                                raw_values[static_cast<size_t>(row_base + lane)];
                        }
                    }
                    ++success_cursor;
                    ++bucket_kept;
                    ++stats.live_rows;
                } else {
                    ++stats.zero_pruned_rows;
                }
                ++bucket_seen;
                word &= word - 1ULL;
            }
        }
        if (bucket_kept == 0U) {
            continue;
        }
        const uint32_t payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
        bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_offset);
        const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        bc_resident_append_prefix256_le(payload.rank_payload, keep_bitmap.data(), word_count, bitmap_len);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        bc_resident_append_padding(
            payload.rank_payload,
            out_bitmap_offset - static_cast<uint32_t>(payload.rank_payload.size())
        );
        bc_resident_append_bitmap_le(payload.rank_payload, keep_bitmap.data(), word_count);
        if (bucket_success_offset > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC resident in-place compact success row offset exceeds uint32");
        }
        payload.buckets.push_back(BCBucketEntry{
            bucket.key,
            rank_payload_offset,
            static_cast<uint32_t>(bucket_success_offset)
        });
    }
    if (success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident in-place compact success rows exceed uint32");
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    if (payload.success_rows == 0U) {
        payload.buckets.clear();
        payload.rank_payload.clear();
        ++stats.empty_cells;
    } else {
        ++stats.live_cells;
    }
}

template <typename StorageT>
inline void bc_resident_compact_zero_cell_in_place(
    const BCPositionLayerReader &position,
    CellId cid,
    std::vector<StorageT> &raw_values,
    uint64_t cell_value_offset,
    uint32_t row_width,
    StorageT zero_value,
    FinalizedCellPayload &payload,
    BCResidentCompactStats &stats
) {
    if (row_width == 1U) {
        const BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            ++stats.empty_cells;
            return;
        }
        stats.input_rows += desc.success_rows;
        const BCLut &lut = position.lut();
        const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
        const BCRankPayloadView rank_payload = position.rank_payload_for_cell(cid);
        payload.buckets.reserve(buckets.size);
        payload.rank_payload.reserve(rank_payload.size);
        uint64_t success_cursor = 0U;
        for (uint32_t bucket_i = 0U; bucket_i < buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = buckets.data[bucket_i];
            const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
                bucket.rank_payload_offset,
                bitmap_len
            );
            const uint64_t bitmap_end =
                static_cast<uint64_t>(bitmap_offset) +
                static_cast<uint64_t>(word_count) * sizeof(uint64_t);
            if (bitmap_end > rank_payload.size) {
                throw std::out_of_range("BC resident in-place compact source bitmap exceeds rank payload");
            }
            const uint64_t bucket_success_offset = success_cursor;
            const uint32_t payload_start = static_cast<uint32_t>(payload.rank_payload.size());
            const uint32_t aligned_payload_offset = align_up_u32(payload_start, 8U);
            bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_start);
            const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
            const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
            const uint64_t payload_end =
                static_cast<uint64_t>(out_bitmap_offset) +
                static_cast<uint64_t>(word_count) * sizeof(uint64_t);
            if (payload_end > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::overflow_error("BC resident compact rank payload exceeds uint32");
            }
            payload.rank_payload.resize(static_cast<size_t>(payload_end), 0U);

            uint32_t bucket_seen = 0U;
            uint32_t bucket_kept = 0U;
            uint32_t prefix_running = 0U;
            const uint8_t *bitmap_words = rank_payload.data + bitmap_offset;
            for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
                if ((word_i & 3U) == 0U) {
                    if (prefix_running > std::numeric_limits<RankPrefix>::max()) {
                        throw std::logic_error("BC resident compact prefix running popcount exceeds uint16");
                    }
                    bc_resident_store_u16_le(
                        payload.rank_payload.data() +
                            static_cast<size_t>(rank_payload_offset) +
                            static_cast<size_t>(word_i / 4U) * sizeof(RankPrefix),
                        static_cast<RankPrefix>(prefix_running)
                    );
                }
                uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
                if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                    word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
                }
                uint64_t keep_word = 0U;
                while (word != 0U) {
                    const uint32_t bit = bc_resident_countr_zero64(word);
                    const uint64_t local_row =
                        static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                    const uint64_t row_index = cell_value_offset + local_row;
                    if (row_index >= raw_values.size()) {
                        throw std::out_of_range("BC resident in-place compact source row exceeds raw values");
                    }
                    if (raw_values[static_cast<size_t>(row_index)] != zero_value) {
                        keep_word |= (1ULL << bit);
                        const uint64_t dst_index = cell_value_offset + success_cursor;
                        if (dst_index >= raw_values.size()) {
                            throw std::out_of_range("BC resident in-place compact destination exceeds raw values");
                        }
                        if (dst_index != row_index) {
                            raw_values[static_cast<size_t>(dst_index)] =
                                raw_values[static_cast<size_t>(row_index)];
                        }
                        ++success_cursor;
                        ++bucket_kept;
                        ++stats.live_rows;
                    } else {
                        ++stats.zero_pruned_rows;
                    }
                    ++bucket_seen;
                    word &= word - 1ULL;
                }
                bc_resident_store_u64_le(
                    payload.rank_payload.data() +
                        static_cast<size_t>(out_bitmap_offset) +
                        static_cast<size_t>(word_i) * sizeof(uint64_t),
                    keep_word
                );
                prefix_running += popcount64(keep_word);
            }
            if (bucket_kept == 0U) {
                payload.rank_payload.resize(payload_start);
                continue;
            }
            if (bucket_success_offset > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC resident in-place compact success row offset exceeds uint32");
            }
            payload.buckets.push_back(BCBucketEntry{
                bucket.key,
                rank_payload_offset,
                static_cast<uint32_t>(bucket_success_offset)
            });
        }
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC resident in-place compact success rows exceed uint32");
        }
        payload.success_rows = static_cast<uint32_t>(success_cursor);
        if (payload.success_rows == 0U) {
            payload.buckets.clear();
            payload.rank_payload.clear();
            ++stats.empty_cells;
        } else {
            ++stats.live_cells;
        }
        return;
    }

    bc_resident_compact_cell_in_place_if<StorageT>(
        position,
        cid,
        raw_values,
        cell_value_offset,
        row_width,
        payload,
        stats,
        [zero_value](const StorageT *row, uint32_t width) {
            for (uint32_t lane = 0U; lane < width; ++lane) {
                if (row[lane] != zero_value) {
                    return true;
                }
            }
            return false;
        }
    );
}

template <typename StorageT>
inline BCResidentSolvedLayer<StorageT> bc_resident_compact_layer(
    const BCPositionLayerReader &position,
    const std::vector<StorageT> &raw_values,
    const std::vector<uint64_t> &cell_value_offsets,
    const BCLut &lut,
    uint32_t row_width = 1U,
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>(),
    StorageT zero_value = StorageT{},
    int num_threads = 0
) {
    const double compact_t0 = bc_resident_solve_now_seconds();
    if (row_width == 0U) {
        throw std::invalid_argument("BC resident compact row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
        throw std::invalid_argument("BC resident compact dtype does not match storage type");
    }
    const uint32_t cell_count = position.cell_count();
    if (cell_value_offsets.size() != static_cast<size_t>(cell_count) + 1U ||
        cell_value_offsets.back() > std::numeric_limits<uint64_t>::max() / row_width ||
        cell_value_offsets.back() * row_width != raw_values.size()) {
        throw std::invalid_argument("BC resident compact cell offsets do not match raw values");
    }
    std::vector<FinalizedCellPayload> payloads(cell_count);
    std::vector<std::vector<StorageT>> cell_success_values(cell_count);
    const int threads = bc_resident_solve_effective_threads(num_threads);
    std::vector<BCResidentCompactStats> per_thread(static_cast<size_t>(threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(threads)
    for (int64_t cid_signed = 0; cid_signed < static_cast<int64_t>(cell_count); ++cid_signed) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const CellId cid = static_cast<CellId>(cid_signed);
        bc_resident_compact_cell<StorageT>(
            position,
            cid,
            raw_values,
            cell_value_offsets[static_cast<size_t>(cid)],
            row_width,
            zero_value,
            payloads[static_cast<size_t>(cid)],
            cell_success_values[static_cast<size_t>(cid)],
            per_thread[static_cast<size_t>(tid)]
        );
    }

    BCResidentSolvedLayer<StorageT> out;
    for (const BCResidentCompactStats &stats : per_thread) {
        out.compact_stats.input_rows += stats.input_rows;
        out.compact_stats.live_rows += stats.live_rows;
        out.compact_stats.zero_pruned_rows += stats.zero_pruned_rows;
        out.compact_stats.live_cells += stats.live_cells;
        out.compact_stats.empty_cells += stats.empty_cells;
    }
    if (out.compact_stats.live_rows > std::numeric_limits<uint64_t>::max() / row_width ||
        out.compact_stats.live_rows * row_width > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC resident compact value count exceeds size_t");
    }
    std::vector<StorageT> compact_values;
    compact_values.reserve(static_cast<size_t>(out.compact_stats.live_rows * row_width));
    for (std::vector<StorageT> &values : cell_success_values) {
        compact_values.insert(compact_values.end(), values.begin(), values.end());
        std::vector<StorageT>().swap(values);
    }

    std::vector<uint8_t> position_bytes =
        bc_resident_build_position_bytes_from_payloads(position.axis(), payloads);
    out.compact_stats.position_bytes = position_bytes.size();
    out.compact_stats.success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(compact_values.size()) * bc_success_dtype_value_size(dtype);
    out.open(std::move(position_bytes), std::move(compact_values), lut, row_width, dtype);
    out.compact_stats.compact_seconds = bc_resident_solve_now_seconds() - compact_t0;
    return out;
}

template <typename StorageT>
inline BCResidentSolvedLayer<StorageT> bc_resident_compact_zero_in_place(
    const BCPositionLayerReader &position,
    BCResidentRawSolveResult<StorageT> &raw,
    const BCLut &lut,
    uint32_t row_width = 1U,
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>(),
    StorageT zero_value = StorageT{},
    int num_threads = 0
) {
    const double compact_t0 = bc_resident_solve_now_seconds();
    if (row_width == 0U) {
        throw std::invalid_argument("BC resident in-place compact row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
        throw std::invalid_argument("BC resident in-place compact dtype does not match storage type");
    }
    const uint32_t cell_count = position.cell_count();
    if (raw.cell_value_offsets.size() != static_cast<size_t>(cell_count) + 1U ||
        raw.cell_value_offsets.back() > std::numeric_limits<uint64_t>::max() / row_width ||
        raw.cell_value_offsets.back() * row_width != raw.values.size()) {
        throw std::invalid_argument("BC resident in-place compact cell offsets do not match raw values");
    }

    std::vector<FinalizedCellPayload> payloads(cell_count);
    const int threads = bc_resident_solve_effective_threads(num_threads);
    std::vector<BCResidentCompactStats> per_thread(static_cast<size_t>(threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(threads)
    for (int64_t cid_signed = 0; cid_signed < static_cast<int64_t>(cell_count); ++cid_signed) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const CellId cid = static_cast<CellId>(cid_signed);
        bc_resident_compact_zero_cell_in_place<StorageT>(
            position,
            cid,
            raw.values,
            raw.cell_value_offsets[static_cast<size_t>(cid)],
            row_width,
            zero_value,
            payloads[static_cast<size_t>(cid)],
            per_thread[static_cast<size_t>(tid)]
        );
    }

    BCResidentSolvedLayer<StorageT> out;
    for (const BCResidentCompactStats &stats : per_thread) {
        out.compact_stats.input_rows += stats.input_rows;
        out.compact_stats.live_rows += stats.live_rows;
        out.compact_stats.zero_pruned_rows += stats.zero_pruned_rows;
        out.compact_stats.live_cells += stats.live_cells;
        out.compact_stats.empty_cells += stats.empty_cells;
    }
    if (out.compact_stats.live_rows > std::numeric_limits<uint64_t>::max() / row_width ||
        out.compact_stats.live_rows * row_width > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC resident in-place compact value count exceeds size_t");
    }

    std::vector<uint64_t> compact_cell_offsets(static_cast<size_t>(cell_count) + 1U, 0U);
    for (CellId cid = 0U; cid < cell_count; ++cid) {
        compact_cell_offsets[static_cast<size_t>(cid) + 1U] = bc_checked_add_u64(
            compact_cell_offsets[static_cast<size_t>(cid)],
            payloads[static_cast<size_t>(cid)].success_rows,
            "BC resident in-place compact cell offset overflow"
        );
    }
    if (compact_cell_offsets.back() != out.compact_stats.live_rows) {
        throw std::logic_error("BC resident in-place compact live row count mismatch");
    }

    for (CellId cid = 0U; cid < cell_count; ++cid) {
        const uint64_t row_count = payloads[static_cast<size_t>(cid)].success_rows;
        if (row_count == 0U) {
            continue;
        }
        const uint64_t src_value =
            raw.cell_value_offsets[static_cast<size_t>(cid)] * static_cast<uint64_t>(row_width);
        const uint64_t dst_value =
            compact_cell_offsets[static_cast<size_t>(cid)] * static_cast<uint64_t>(row_width);
        const uint64_t value_count = row_count * static_cast<uint64_t>(row_width);
        if (src_value + value_count > raw.values.size() ||
            dst_value + value_count > raw.values.size()) {
            throw std::out_of_range("BC resident in-place compact final move exceeds raw values");
        }
        if (src_value != dst_value) {
            std::memmove(
                raw.values.data() + static_cast<size_t>(dst_value),
                raw.values.data() + static_cast<size_t>(src_value),
                static_cast<size_t>(value_count) * sizeof(StorageT)
            );
        }
    }

    const uint64_t compact_value_count =
        out.compact_stats.live_rows * static_cast<uint64_t>(row_width);
    raw.values.resize(static_cast<size_t>(compact_value_count));
    if (raw.values.capacity() != raw.values.size()) {
        std::vector<StorageT>(raw.values.begin(), raw.values.end()).swap(raw.values);
    }
    raw.cell_value_offsets.clear();
    raw.cell_value_offsets.shrink_to_fit();

    std::vector<uint8_t> position_bytes =
        bc_resident_build_position_bytes_from_payloads(position.axis(), payloads);
    out.compact_stats.position_bytes = position_bytes.size();
    out.compact_stats.success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(raw.values.size()) * bc_success_dtype_value_size(dtype);
    out.open(std::move(position_bytes), std::move(raw.values), lut, row_width, dtype);
    out.compact_stats.compact_seconds = bc_resident_solve_now_seconds() - compact_t0;
    return out;
}

template <typename StorageT>
inline BCResidentCompactStats bc_resident_prune_below_threshold_for_archive_in_place(
    BCResidentSolvedLayer<StorageT> &layer,
    StorageT threshold,
    int num_threads = 0
) {
    const double compact_t0 = bc_resident_solve_now_seconds();
    const uint32_t row_width = layer.row_width;
    if (row_width == 0U) {
        throw std::invalid_argument("BC resident archive prune row_width must be non-zero");
    }
    const uint32_t cell_count = layer.position.cell_count();
    std::vector<uint64_t> cell_value_offsets = bc_resident_cell_value_offsets(layer.position);
    if (cell_value_offsets.size() != static_cast<size_t>(cell_count) + 1U ||
        cell_value_offsets.back() > std::numeric_limits<uint64_t>::max() / row_width ||
        cell_value_offsets.back() * row_width != layer.success_values.size()) {
        throw std::invalid_argument("BC resident archive prune offsets do not match success values");
    }

    std::vector<FinalizedCellPayload> payloads(cell_count);
    const int threads = bc_resident_solve_effective_threads(num_threads);
    std::vector<BCResidentCompactStats> per_thread(static_cast<size_t>(threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(threads)
    for (int64_t cid_signed = 0; cid_signed < static_cast<int64_t>(cell_count); ++cid_signed) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const CellId cid = static_cast<CellId>(cid_signed);
        bc_resident_compact_cell_in_place_if<StorageT>(
            layer.position,
            cid,
            layer.success_values,
            cell_value_offsets[static_cast<size_t>(cid)],
            row_width,
            payloads[static_cast<size_t>(cid)],
            per_thread[static_cast<size_t>(tid)],
            [threshold](const StorageT *row, uint32_t width) {
                for (uint32_t lane = 0U; lane < width; ++lane) {
                    if (row[lane] > threshold) {
                        return true;
                    }
                }
                return false;
            }
        );
    }

    BCResidentCompactStats stats;
    for (const BCResidentCompactStats &part : per_thread) {
        stats.input_rows += part.input_rows;
        stats.live_rows += part.live_rows;
        stats.zero_pruned_rows += part.zero_pruned_rows;
        stats.live_cells += part.live_cells;
        stats.empty_cells += part.empty_cells;
    }
    if (stats.live_rows > std::numeric_limits<uint64_t>::max() / row_width ||
        stats.live_rows * row_width > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC resident archive prune value count exceeds size_t");
    }

    std::vector<uint64_t> compact_cell_offsets(static_cast<size_t>(cell_count) + 1U, 0U);
    for (CellId cid = 0U; cid < cell_count; ++cid) {
        compact_cell_offsets[static_cast<size_t>(cid) + 1U] = bc_checked_add_u64(
            compact_cell_offsets[static_cast<size_t>(cid)],
            payloads[static_cast<size_t>(cid)].success_rows,
            "BC resident archive prune cell offset overflow"
        );
    }
    if (compact_cell_offsets.back() != stats.live_rows) {
        throw std::logic_error("BC resident archive prune live row count mismatch");
    }

    for (CellId cid = 0U; cid < cell_count; ++cid) {
        const uint64_t row_count = payloads[static_cast<size_t>(cid)].success_rows;
        if (row_count == 0U) {
            continue;
        }
        const uint64_t src_value =
            cell_value_offsets[static_cast<size_t>(cid)] * static_cast<uint64_t>(row_width);
        const uint64_t dst_value =
            compact_cell_offsets[static_cast<size_t>(cid)] * static_cast<uint64_t>(row_width);
        const uint64_t value_count = row_count * static_cast<uint64_t>(row_width);
        if (src_value + value_count > layer.success_values.size() ||
            dst_value + value_count > layer.success_values.size()) {
            throw std::out_of_range("BC resident archive prune final move exceeds success values");
        }
        if (src_value != dst_value) {
            std::memmove(
                layer.success_values.data() + static_cast<size_t>(dst_value),
                layer.success_values.data() + static_cast<size_t>(src_value),
                static_cast<size_t>(value_count) * sizeof(StorageT)
            );
        }
    }

    const uint64_t compact_value_count = stats.live_rows * static_cast<uint64_t>(row_width);
    layer.success_values.resize(static_cast<size_t>(compact_value_count));

    const BCLut &lut = layer.position.lut();
    const uint32_t old_row_width = layer.row_width;
    const BCSuccessDTypeMode old_dtype = layer.dtype;
    std::vector<uint8_t> position_bytes =
        bc_resident_build_position_bytes_from_payloads(layer.position.axis(), payloads);
    stats.position_bytes = position_bytes.size();
    stats.success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(layer.success_values.size()) * bc_success_dtype_value_size(old_dtype);
    std::vector<StorageT> compact_values = std::move(layer.success_values);
    layer.open(std::move(position_bytes), std::move(compact_values), lut, old_row_width, old_dtype);
    stats.compact_seconds = bc_resident_solve_now_seconds() - compact_t0;
    layer.compact_stats = stats;
    return stats;
}

template <typename StorageT>
inline BCResidentArchivePruneResult bc_resident_archive_prune_if_threshold_enabled_in_place(
    BCResidentSolvedLayer<StorageT> &layer,
    StorageT threshold,
    int num_threads = 0
) {
    BCResidentArchivePruneResult result;
    const StorageT zero_value = bc_success_zero_value_for_dtype<StorageT>(layer.dtype);
    if (!(threshold > zero_value)) {
        return result;
    }
    result.stats = bc_resident_prune_below_threshold_for_archive_in_place<StorageT>(
        layer,
        threshold,
        num_threads
    );
    result.pruned = true;
    return result;
}

template <typename StorageT>
inline BCResidentLayerResult<StorageT> bc_resident_solve_compacted_layer(
    const BCPositionLayerReader &current_position,
    const BCResidentSolvedLayer<StorageT> &future2,
    const BCResidentSolvedLayer<StorageT> &future4,
    const BCResidentSolveOptions<StorageT> &options
) {
    if (options.row_width == 0U) {
        throw std::invalid_argument("BC resident solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.dtype)) {
        throw std::invalid_argument("BC resident solve dtype does not match storage type");
    }
    if (future2.position.cell_count() == 0U || future4.position.cell_count() == 0U) {
        throw std::invalid_argument("BC resident solve future layers must be open");
    }
    if (future2.row_width != options.row_width || future4.row_width != options.row_width) {
        throw std::invalid_argument("BC resident solve future row_width mismatch");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        current_position.lut().tile_sum_value(options.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        current_position.lut().tile_sum_value(options.edge_options.spawn4_tile_rank);
    if (future2.position.axis().layer_sum() != expected2 ||
        future4.position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC resident solve future layer_sum mismatch");
    }
    BCResidentRawSolveResult<StorageT> raw = bc_resident_solve_raw_values<StorageT>(
        current_position,
        future2,
        future4,
        options
    );
    BCResidentLayerResult<StorageT> result;
    result.solve_stats = raw.stats;
    result.layer = bc_resident_compact_zero_in_place<StorageT>(
        current_position,
        raw,
        current_position.lut(),
        options.row_width,
        options.dtype,
        options.zero_value,
        options.num_threads
    );
    result.solve_stats.output_values =
        result.layer.compact_stats.live_rows * static_cast<uint64_t>(options.row_width);
    result.solve_stats.output_bytes = result.layer.compact_stats.success_bytes;
    return result;
}

// Legacy scaffold kept for SingleChunkSolve experiments. Production resident solve
// uses bc_resident_solve_compacted_layer() and batch direct future lookup instead.
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

} // namespace BC
