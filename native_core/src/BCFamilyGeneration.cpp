#include "BCFamilyGeneration.h"

#include "BCBoardOps.h"
#include "BCLoadedCellScanner.h"
#include "BoardMover.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {
namespace {

[[nodiscard]] double family_now_seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

[[nodiscard]] bool family_stage_timing_enabled() {
    const char *value = std::getenv("BC_FAMILY_STAGE_TIMING");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

[[nodiscard]] bool family_progress_enabled() {
    const char *value = std::getenv("BC_FAMILY_PROGRESS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool family_validate_enabled() {
    const char *value = std::getenv("BC_FAMILY_VALIDATE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void validate_family_store_if_enabled(
    const BCFamilyMutableStore &store,
    std::string_view label
) {
    if (!family_validate_enabled()) {
        return;
    }
    try {
        store.validate_internal_for_testing();
    } catch (const std::exception &ex) {
        throw std::logic_error(
            std::string("BC Family store validation failed at ") +
            std::string(label) +
            ": " +
            ex.what()
        );
    }
}

[[nodiscard]] const char *family_pass_stats_csv_path() {
    const char *value = std::getenv("BC_FAMILY_PASS_STATS_CSV");
    if (value == nullptr || value[0] == '\0') {
        return nullptr;
    }
    return value;
}

[[nodiscard]] bool family_pass_stats_enabled() {
    return family_pass_stats_csv_path() != nullptr;
}

void append_family_pass_stats_csv(
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource &source,
    const BCFamilyGenerationPass &pass,
    bool finalize_boundaries,
    bool source_has_rows,
    size_t source_cell_count,
    size_t active_cell_count,
    size_t keep_cell_count,
    size_t boundary_cell_count,
    size_t loaded_cell_count,
    uint64_t loaded_bytes,
    uint64_t load_read_bytes,
    size_t range_work_count,
    const BCFamilyGenerationStats &before,
    const BCFamilyGenerationStats &after
) {
    const char *path = family_pass_stats_csv_path();
    if (path == nullptr) {
        return;
    }
    static bool header_written = false;
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open BC_FAMILY_PASS_STATS_CSV");
    }
    if (!header_written) {
        out
            << "source_layer,target_layer,finalize_phase,source_id,source_coord,"
            << "target_family0,target_family1,source_has_rows,source_cells,active_cells,"
            << "keep_cells,boundary_cells,loaded_cells,loaded_bytes,source_load_read_bytes,"
            << "range_work,source_work_items,source_bucket_ranges,source_bitmap_words_scanned,"
            << "source_boards_scanned,spawned_boards,move_results_produced,"
            << "canonicalized_candidates,encode_attempts,encode_valid_candidates,"
            << "encoded_candidates,duplicate_candidates,parallel_seconds,"
            << "load_seconds,reload_seconds,build_work_seconds,dump_seconds,finalize_seconds,"
            << "write_seconds,buffer_flushes,buffer_flush_items,buffer_flush_max_items,"
            << "builder_bind_calls,hash_lookups,hash_probe_steps,target_window_skips,"
            << "move_all_dir_calls,selective_move_calls,"
            << "output_mboards_per_parallel_second\n";
        header_written = true;
    }
    const auto delta_u64 = [](uint64_t lhs, uint64_t rhs) -> uint64_t {
        return lhs >= rhs ? lhs - rhs : 0U;
    };
    const auto delta_d = [](double lhs, double rhs) -> double {
        return lhs >= rhs ? lhs - rhs : 0.0;
    };
    const uint64_t encoded = delta_u64(after.encoded_candidates, before.encoded_candidates);
    const double parallel_seconds = delta_d(after.parallel_seconds, before.parallel_seconds);
    const double output_mbps =
        parallel_seconds > 0.0 ? static_cast<double>(encoded) / parallel_seconds / 1.0e6 : 0.0;
    const int32_t target_family0 = pass.target_families.size() >= 1U
        ? static_cast<int32_t>(pass.target_families[0])
        : -1;
    const int32_t target_family1 = pass.target_families.size() >= 2U
        ? static_cast<int32_t>(pass.target_families[1])
        : -1;
    out
        << source.position->axis().layer_sum() << ','
        << target_axis.layer_sum() << ','
        << (finalize_boundaries ? 1U : 0U) << ','
        << static_cast<uint32_t>(pass.source_id) << ','
        << static_cast<uint32_t>(pass.source_coord) << ','
        << target_family0 << ','
        << target_family1 << ','
        << (source_has_rows ? 1U : 0U) << ','
        << source_cell_count << ','
        << active_cell_count << ','
        << keep_cell_count << ','
        << boundary_cell_count << ','
        << loaded_cell_count << ','
        << loaded_bytes << ','
        << load_read_bytes << ','
        << range_work_count << ','
        << delta_u64(after.source_work_items, before.source_work_items) << ','
        << delta_u64(after.source_bucket_ranges, before.source_bucket_ranges) << ','
        << delta_u64(after.source_bitmap_words_scanned, before.source_bitmap_words_scanned) << ','
        << delta_u64(after.source_boards_scanned, before.source_boards_scanned) << ','
        << delta_u64(after.spawned_boards, before.spawned_boards) << ','
        << delta_u64(after.move_results_produced, before.move_results_produced) << ','
        << delta_u64(after.canonicalized_candidates, before.canonicalized_candidates) << ','
        << delta_u64(after.encode_attempts, before.encode_attempts) << ','
        << delta_u64(after.encode_valid_candidates, before.encode_valid_candidates) << ','
        << encoded << ','
        << delta_u64(after.duplicate_candidates, before.duplicate_candidates) << ','
        << parallel_seconds << ','
        << delta_d(after.source_load_seconds, before.source_load_seconds) << ','
        << delta_d(after.reload_seconds, before.reload_seconds) << ','
        << delta_d(after.build_work_seconds, before.build_work_seconds) << ','
        << delta_d(after.dump_seconds, before.dump_seconds) << ','
        << delta_d(after.finalize_seconds, before.finalize_seconds) << ','
        << delta_d(after.write_seconds, before.write_seconds) << ','
        << delta_u64(after.family_buffer_flushes, before.family_buffer_flushes) << ','
        << delta_u64(after.family_buffer_flush_items, before.family_buffer_flush_items) << ','
        << delta_u64(after.family_buffer_flush_max_items, before.family_buffer_flush_max_items) << ','
        << delta_u64(after.family_builder_bind_calls, before.family_builder_bind_calls) << ','
        << delta_u64(after.family_insert_hash_lookups, before.family_insert_hash_lookups) << ','
        << delta_u64(after.family_insert_hash_probe_steps, before.family_insert_hash_probe_steps) << ','
        << delta_u64(after.target_window_skips, before.target_window_skips) << ','
        << delta_u64(after.move_all_dir_calls, before.move_all_dir_calls) << ','
        << delta_u64(after.selective_move_calls, before.selective_move_calls) << ','
        << output_mbps
        << '\n';
}

[[nodiscard]] int family_effective_threads(int requested) {
#if defined(_OPENMP)
    if (requested > 0) {
        return requested;
    }
    return omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

[[nodiscard]] int family_thread_num() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void add_load_stats(BCFamilyGenerationStats &stats, const BCCellLoadStats &load) {
    stats.source_bytes_read += load.read_bytes;
    stats.source_backend_read_ops += load.backend_read_ops;
    stats.source_backend_read_bytes += load.backend_read_bytes;
}

[[nodiscard]] uint64_t loaded_cells_payload_bytes(const std::vector<BCLoadedCell> &loaded) {
    uint64_t total = 0U;
    for (const BCLoadedCell &cell : loaded) {
        total = bc_checked_add_u64(
            total,
            static_cast<uint64_t>(cell.buckets.size()) * sizeof(BCBucketEntry),
            "BC Family loaded bucket byte count overflow"
        );
        total = bc_checked_add_u64(
            total,
            cell.rank_payload.size(),
            "BC Family loaded rank payload byte count overflow"
        );
    }
    return total;
}

[[nodiscard]] uint64_t loaded_cells_allocated_bytes(const std::vector<BCLoadedCell> &loaded) {
    uint64_t total = static_cast<uint64_t>(loaded.capacity()) * sizeof(BCLoadedCell);
    for (const BCLoadedCell &cell : loaded) {
        total = bc_checked_add_u64(
            total,
            static_cast<uint64_t>(cell.buckets.capacity()) * sizeof(BCBucketEntry),
            "BC Family loaded bucket capacity byte count overflow"
        );
        total = bc_checked_add_u64(
            total,
            cell.rank_payload.capacity(),
            "BC Family loaded rank payload capacity byte count overflow"
        );
    }
    return total;
}

[[nodiscard]] uint64_t finalized_payloads_allocated_bytes(
    const std::vector<FinalizedCellPayload> &payloads
) {
    uint64_t total = static_cast<uint64_t>(payloads.capacity()) * sizeof(FinalizedCellPayload);
    for (const FinalizedCellPayload &payload : payloads) {
        total = bc_checked_add_u64(
            total,
            static_cast<uint64_t>(payload.buckets.capacity()) * sizeof(BCBucketEntry),
            "BC Family finalized bucket capacity byte count overflow"
        );
        total = bc_checked_add_u64(
            total,
            payload.rank_payload.capacity(),
            "BC Family finalized rank payload capacity byte count overflow"
        );
    }
    return total;
}

[[nodiscard]] uint64_t finalized_payload_allocated_bytes(
    const FinalizedCellPayload &payload
) {
    uint64_t total = sizeof(FinalizedCellPayload);
    total = bc_checked_add_u64(
        total,
        static_cast<uint64_t>(payload.buckets.capacity()) * sizeof(BCBucketEntry),
        "BC Family finalized single bucket capacity byte count overflow"
    );
    total = bc_checked_add_u64(
        total,
        payload.rank_payload.capacity(),
        "BC Family finalized single rank payload capacity byte count overflow"
    );
    return total;
}

[[nodiscard]] uint32_t family_memory_checkpoint_label_code(const char *label) {
    if (label == nullptr) {
        return 0U;
    }
    const std::string_view view(label);
    if (view == "after_pass_cache") {
        return 1U;
    }
    if (view == "before_pass_cache") {
        return 7U;
    }
    if (view == "after_source_activity_scan") {
        return 8U;
    }
    if (view == "pre_parallel") {
        return 2U;
    }
    if (view == "post_parallel") {
        return 3U;
    }
    if (view == "post_release") {
        return 4U;
    }
    if (view == "finalized_payloads") {
        return 5U;
    }
    if (view == "empty_pass_post_release") {
        return 6U;
    }
    if (view == "after_plus4_phase") {
        return 9U;
    }
    if (view == "after_plus2_phase") {
        return 10U;
    }
    return 255U;
}

void validate_family_source(
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource &source
) {
    if (source.position == nullptr) {
        throw std::invalid_argument("BC Family generation source position is null");
    }
    if (source.spawn_tile_rank == 0U || source.spawn_tile_rank > 15U) {
        throw std::invalid_argument("BC Family generation spawn tile rank is invalid");
    }
    if (source.position->axis().family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC Family generation source/target family unit mismatch");
    }
    const uint64_t expected_total =
        static_cast<uint64_t>(source.position->axis().total_coord()) +
        static_cast<uint64_t>(source.delta_coord);
    if (static_cast<uint64_t>(target_axis.total_coord()) != expected_total) {
        throw std::invalid_argument("BC Family generation source/target total coord mismatch");
    }
}

[[nodiscard]] FamilyId nearest_source_family_id(
    const BCFamilyTable &source_axis,
    FamilyCoord target_coord
) {
    if (source_axis.family_count() == 0U) {
        throw std::invalid_argument("BC Family reserve source axis is empty");
    }
    const std::vector<FamilyCoord> &coords = source_axis.coords();
    if (target_coord <= coords.front()) {
        return 0U;
    }
    if (target_coord >= coords.back()) {
        return static_cast<FamilyId>(source_axis.family_count() - 1U);
    }
    const auto it = std::lower_bound(coords.begin(), coords.end(), target_coord);
    if (it == coords.end()) {
        return static_cast<FamilyId>(source_axis.family_count() - 1U);
    }
    if (*it == target_coord || it == coords.begin()) {
        return static_cast<FamilyId>(it - coords.begin());
    }
    const auto prev = it - 1;
    const uint64_t next_delta = static_cast<uint64_t>(*it) - target_coord;
    const uint64_t prev_delta = static_cast<uint64_t>(target_coord) - *prev;
    return static_cast<FamilyId>(
        prev_delta <= next_delta ? (prev - coords.begin()) : (it - coords.begin())
    );
}

void append_source_neighbor_reserve_samples(
    const BCPositionStreamingReader &source,
    FamilyCoord target_row_coord,
    FamilyCoord target_col_coord,
    uint32_t radius,
    std::vector<uint32_t> &bucket_samples,
    std::vector<uint32_t> &bitmap_word_samples
) {
    const BCFamilyTable &source_axis = source.axis();
    if (source_axis.family_count() == 0U) {
        return;
    }
    const BCCellMatrix source_matrix(source_axis);
    const int32_t center_row = static_cast<int32_t>(
        nearest_source_family_id(source_axis, target_row_coord)
    );
    const int32_t center_col = static_cast<int32_t>(
        nearest_source_family_id(source_axis, target_col_coord)
    );
    const int32_t first = 0;
    const int32_t last = static_cast<int32_t>(source_axis.family_count()) - 1;
    const int32_t r = static_cast<int32_t>(std::min<uint32_t>(
        radius,
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
    ));
    for (int32_t row = std::max(first, center_row - r); row <= std::min(last, center_row + r); ++row) {
        for (int32_t col = std::max(first, center_col - r); col <= std::min(last, center_col + r); ++col) {
            const CellId source_cid = source_matrix.cid(
                static_cast<FamilyId>(row),
                static_cast<FamilyId>(col)
            );
            const BCPositionCellDescriptor &desc = source.descriptor(source_cid);
            if (desc.empty() || desc.bucket_count == 0U || desc.rank_payload_bytes == 0U) {
                continue;
            }
            bucket_samples.push_back(desc.bucket_count);
            const uint64_t rank_words64 = (desc.rank_payload_bytes + 7U) / 8U;
            bitmap_word_samples.push_back(static_cast<uint32_t>(std::min<uint64_t>(
                rank_words64,
                std::numeric_limits<uint32_t>::max()
            )));
        }
    }
}

[[nodiscard]] uint32_t reserve_quantile_u32(std::vector<uint32_t> values, double quantile) {
    if (values.empty()) {
        return 0U;
    }
    if (!(quantile >= 0.0)) {
        quantile = 0.0;
    }
    if (quantile > 1.0) {
        quantile = 1.0;
    }
    const double scaled_index = quantile * static_cast<double>(values.size() - 1U);
    const size_t index = static_cast<size_t>(std::ceil(scaled_index));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

[[nodiscard]] uint32_t scaled_reserve_u32(uint32_t value, double scale) {
    if (value == 0U) {
        return 0U;
    }
    if (!(scale > 0.0)) {
        scale = 1.0;
    }
    const double scaled = std::ceil(static_cast<double>(value) * scale);
    if (scaled >= static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(scaled);
}

[[nodiscard]] std::vector<BCCellMutableReserveHint> build_family_neighbor_reserve_hints(
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource *source4,
    const BCFamilyStreamingGenerationSource &source2,
    const BCFamilyGenerationOptions &options
) {
    const BCCellMatrix target_matrix(target_axis);
    std::vector<BCCellMutableReserveHint> hints(target_matrix.cell_count());
    if (!options.enable_neighbor_cell_reserve) {
        return hints;
    }
    const uint32_t radius = options.neighbor_reserve_radius;
    std::vector<uint32_t> bucket_samples;
    std::vector<uint32_t> bitmap_word_samples;
    bucket_samples.reserve(
        static_cast<size_t>(2U) * static_cast<size_t>(2U * radius + 1U) *
        static_cast<size_t>(2U * radius + 1U)
    );
    bitmap_word_samples.reserve(bucket_samples.capacity());
    for (FamilyId row = 0U; row < target_axis.family_count(); ++row) {
        const FamilyCoord row_coord = target_axis.id_to_coord(row);
        for (FamilyId col = 0U; col < target_axis.family_count(); ++col) {
            const FamilyCoord col_coord = target_axis.id_to_coord(col);
            bucket_samples.clear();
            bitmap_word_samples.clear();
            append_source_neighbor_reserve_samples(
                *source2.position,
                row_coord,
                col_coord,
                radius,
                bucket_samples,
                bitmap_word_samples
            );
            if (source4 != nullptr) {
                append_source_neighbor_reserve_samples(
                    *source4->position,
                    row_coord,
                    col_coord,
                    radius,
                    bucket_samples,
                    bitmap_word_samples
                );
            }
            const uint32_t bucket_q = reserve_quantile_u32(
                bucket_samples,
                options.neighbor_reserve_quantile
            );
            const uint32_t bitmap_q = reserve_quantile_u32(
                bitmap_word_samples,
                options.neighbor_reserve_quantile
            );
            hints[target_matrix.cid(row, col)] = BCCellMutableReserveHint{
                scaled_reserve_u32(bucket_q, options.neighbor_reserve_scale),
                scaled_reserve_u32(bitmap_q, options.neighbor_reserve_scale)
            };
        }
    }
    return hints;
}

[[nodiscard]] bool source_family_has_success_rows(
    const BCPositionStreamingReader &position,
    const std::vector<CellId> &cids
) {
    for (CellId cid : cids) {
        if (position.descriptor(cid).success_rows != 0U) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool family_success_check_enabled(
    const BCFamilyGenerationOptions &options,
    LayerSum source_layer_sum
) {
    return options.success_target_rank > 0 &&
           options.success_shifts != nullptr &&
           !options.success_shifts->empty() &&
           options.success_check_min_source_layer_sum != 0U &&
           source_layer_sum >= options.success_check_min_source_layer_sum;
}

[[nodiscard]] bool family_is_success_by_shifts(
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

[[nodiscard]] bool family_is_success_board(
    uint64_t board,
    const BCFamilyGenerationOptions &options
) {
    if (options.success_shifts == nullptr || options.success_shifts->empty()) {
        return false;
    }
    if (options.success_check_all_cells) {
        const uint64_t target =
            static_cast<uint64_t>(options.success_target_rank) * 0x1111111111111111ULL;
        const uint64_t diff = board ^ target;
        constexpr uint64_t kMask7 = 0x7777777777777777ULL;
        return (~(((diff & kMask7) + kMask7) | diff | kMask7)) != 0ULL;
    }
    return family_is_success_by_shifts(
        board,
        options.success_target_rank,
        *options.success_shifts
    );
}

struct FamilyThreadStats {
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
    uint64_t source_work_items = 0U;
    uint64_t source_bucket_ranges = 0U;
    uint64_t source_bitmap_words_scanned = 0U;
};

inline constexpr uint32_t kInvalidFamilyActiveIndex = std::numeric_limits<uint32_t>::max();

struct FamilyEncodedCandidate {
    uint32_t active_index = kInvalidFamilyActiveIndex;
    uint64_t key = 0U;
    BucketRank rank = 0U;
    BucketBitmapLen bitmap_len = 0U;
};

struct FamilyEncodeHotContext {
    const BCFamilyTable *axis = nullptr;
    LayerSum layer_sum = 0U;
    uint32_t family_count = 0U;
    FamilyId active_family0 = 0U;
    FamilyId active_family1 = 0U;
    uint32_t active_family_count = 0U;
    const BCFamilyPartitionLayerMap *partition = nullptr;
};

struct FamilyActiveCellBuffer {
    CellId cid = 0U;
    BCCellMutableBuilder *builder = nullptr;
    uint32_t offset = 0U;
    uint32_t size = 0U;
    uint32_t capacity = 0U;
    bool queued_for_flush = false;
    BCCellBitmapThreadChunk bitmap_chunk = {};

    void reset_for_cell(CellId next_cid, uint32_t next_offset, uint32_t next_capacity) {
        cid = next_cid;
        builder = nullptr;
        offset = next_offset;
        size = 0U;
        capacity = next_capacity;
        queued_for_flush = false;
        bitmap_chunk = {};
    }
};

[[nodiscard]] uint32_t family_active_index_from_row_col(
    FamilyId row,
    FamilyId col,
    const FamilyIdList2 &target_families,
    uint32_t family_count
) {
    const uint32_t active_family_count = static_cast<uint32_t>(target_families.size());
    if (active_family_count == 0U || family_count == 0U || active_family_count > 2U) {
        return kInvalidFamilyActiveIndex;
    }

    const uint32_t g0 = static_cast<uint32_t>(target_families[0]);
    if (row == g0) {
        return static_cast<uint32_t>(col);
    }
    if (col == g0) {
        return family_count + (static_cast<uint32_t>(row) < g0
            ? static_cast<uint32_t>(row)
            : static_cast<uint32_t>(row) - 1U);
    }
    if (active_family_count < 2U) {
        return kInvalidFamilyActiveIndex;
    }

    const uint32_t g1 = static_cast<uint32_t>(target_families[1]);
    const uint32_t base1 = 2U * family_count - 1U;
    if (row == g1) {
        const uint32_t col_u32 = static_cast<uint32_t>(col);
        if (col_u32 == g0) {
            return kInvalidFamilyActiveIndex;
        }
        return base1 + col_u32 - (col_u32 > g0 ? 1U : 0U);
    }
    if (col == g1) {
        const uint32_t row_u32 = static_cast<uint32_t>(row);
        if (row_u32 == g0 || row_u32 == g1) {
            return kInvalidFamilyActiveIndex;
        }
        const uint32_t before =
            (row_u32 > g0 ? 1U : 0U) +
            (row_u32 > g1 ? 1U : 0U);
        return base1 + (family_count - 1U) + row_u32 - before;
    }
    return kInvalidFamilyActiveIndex;
}

[[nodiscard]] uint32_t family_active_index_from_row_col_fast(
    FamilyId row,
    FamilyId col,
    FamilyId g0,
    FamilyId g1,
    uint32_t active_family_count,
    uint32_t family_count
) {
    if (active_family_count == 0U || family_count == 0U || active_family_count > 2U) {
        return kInvalidFamilyActiveIndex;
    }
    if (row == g0) {
        return static_cast<uint32_t>(col);
    }
    if (col == g0) {
        return family_count + (static_cast<uint32_t>(row) < static_cast<uint32_t>(g0)
            ? static_cast<uint32_t>(row)
            : static_cast<uint32_t>(row) - 1U);
    }
    if (active_family_count < 2U) {
        return kInvalidFamilyActiveIndex;
    }

    const uint32_t base1 = 2U * family_count - 1U;
    if (row == g1) {
        const uint32_t col_u32 = static_cast<uint32_t>(col);
        if (col_u32 == static_cast<uint32_t>(g0)) {
            return kInvalidFamilyActiveIndex;
        }
        return base1 + col_u32 - (col_u32 > static_cast<uint32_t>(g0) ? 1U : 0U);
    }
    if (col == g1) {
        const uint32_t row_u32 = static_cast<uint32_t>(row);
        if (row_u32 == static_cast<uint32_t>(g0) || row_u32 == static_cast<uint32_t>(g1)) {
            return kInvalidFamilyActiveIndex;
        }
        const uint32_t before =
            (row_u32 > static_cast<uint32_t>(g0) ? 1U : 0U) +
            (row_u32 > static_cast<uint32_t>(g1) ? 1U : 0U);
        return base1 + (family_count - 1U) + row_u32 - before;
    }
    return kInvalidFamilyActiveIndex;
}

[[nodiscard]] FamilyEncodedCandidate encode_family_candidate_compact(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const FamilyIdList2 &target_families,
    const BCQuadrantWords &q,
    const BCFamilyPartitionLayerMap *partition = nullptr
) {
    FamilyEncodedCandidate out;

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = nw_desc.sum;
    const uint64_t ne_sum = ne_desc.sum;
    const uint64_t sw_sum = sw_desc.sum;
    const uint64_t se_sum = se_desc.sum;
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    const uint16_t family_unit = axis.family_unit();
    auto min_side_coord_fast = [family_unit](uint64_t first_sum, uint64_t second_sum, FamilyCoord &coord_out) {
        const uint64_t min_sum = std::min(first_sum, second_sum);
        uint64_t coord = 0U;
        if (family_unit == 2U) {
            if ((min_sum & 1ULL) != 0ULL) {
                return false;
            }
            coord = min_sum >> 1U;
        } else {
            if (family_unit == 0U || (min_sum % family_unit) != 0U) {
                return false;
            }
            coord = min_sum / family_unit;
        }
        if (coord > std::numeric_limits<FamilyCoord>::max()) {
            return false;
        }
        coord_out = static_cast<FamilyCoord>(coord);
        return true;
    };

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!min_side_coord_fast(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
        !min_side_coord_fast(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
        return out;
    }

    const uint32_t family_count = axis.family_count();
    const FamilyId row_id = partition != nullptr
        ? partition->try_coord_to_family_id(row_coord)
        : axis.try_coord_to_id(row_coord);
    const FamilyId col_id = partition != nullptr
        ? partition->try_coord_to_family_id(col_coord)
        : axis.try_coord_to_id(col_coord);
    if (row_id == BCFamilyTable::kInvalidFamilyId ||
        col_id == BCFamilyTable::kInvalidFamilyId) {
        return out;
    }

    const uint32_t count_ne = ne_desc.group_count;
    const uint32_t count_sw = sw_desc.group_count;
    const uint32_t count_se = se_desc.group_count;
    const uint32_t bitmap_len = count_ne * count_sw * count_se;
    if (bitmap_len == 0U ||
        bitmap_len > kBCMaxBucketBitmapLen ||
        bitmap_len > std::numeric_limits<BucketBitmapLen>::max()) {
        throw std::logic_error("BC Family compact encode bitmap length is invalid");
    }
    const uint32_t rank =
        (static_cast<uint32_t>(ne_desc.rank) * count_sw + sw_desc.rank) * count_se + se_desc.rank;
    if (rank >= bitmap_len || rank > std::numeric_limits<BucketRank>::max()) {
        throw std::logic_error("BC Family compact encode rank is invalid");
    }
    const uint32_t active_index = family_active_index_from_row_col(
        row_id,
        col_id,
        target_families,
        family_count
    );
    if (active_index == kInvalidFamilyActiveIndex) {
        return out;
    }
    out.active_index = active_index;
    out.key =
        (static_cast<uint64_t>(q.nw) << 48U) |
        (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
        (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
        static_cast<uint64_t>(se_desc.packed_sum_mask);
    out.rank = static_cast<BucketRank>(rank);
    out.bitmap_len = static_cast<BucketBitmapLen>(bitmap_len);
    return out;
}

[[nodiscard]] FamilyEncodedCandidate encode_family_candidate_compact_unit2(
    const BCLut &lut,
    const FamilyEncodeHotContext &ctx,
    const BCQuadrantWords &q
) {
    FamilyEncodedCandidate out;
    if (ctx.active_family_count == 0U) {
        return out;
    }

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);

    // Family generation inputs come from validated position files plus legal
    // spawn/move/canonicalize, so quadrant validity and total layer sum are
    // invariants here. Public board encode helpers keep those checks.
    if (ctx.axis == nullptr) {
        return out;
    }
    const uint64_t nw_sum = nw_desc.sum;
    const uint64_t ne_sum = ne_desc.sum;
    const uint64_t sw_sum = sw_desc.sum;
    const uint64_t se_sum = se_desc.sum;
    const uint64_t top_sum = nw_sum + ne_sum;
    const uint64_t left_sum = nw_sum + sw_sum;
    if (top_sum > ctx.layer_sum || left_sum > ctx.layer_sum) {
        return out;
    }
    const uint64_t bottom_sum = ctx.layer_sum - top_sum;
    const uint64_t right_sum = ctx.layer_sum - left_sum;
    const uint64_t row_min = top_sum < bottom_sum ? top_sum : bottom_sum;
    const uint64_t col_min = left_sum < right_sum ? left_sum : right_sum;
    const uint64_t row_coord64 = row_min >> 1U;
    const uint64_t col_coord64 = col_min >> 1U;
    if (row_coord64 > std::numeric_limits<FamilyCoord>::max() ||
        col_coord64 > std::numeric_limits<FamilyCoord>::max()) {
        return out;
    }
    const FamilyCoord row_coord = static_cast<FamilyCoord>(row_coord64);
    const FamilyCoord col_coord = static_cast<FamilyCoord>(col_coord64);
    const FamilyId row_id = ctx.partition != nullptr
        ? ctx.partition->coord_to_family_id_trusted(row_coord)
        : ctx.axis->try_coord_to_id(row_coord);
    const FamilyId col_id = ctx.partition != nullptr
        ? ctx.partition->coord_to_family_id_trusted(col_coord)
        : ctx.axis->try_coord_to_id(col_coord);
    if (row_id == BCFamilyTable::kInvalidFamilyId ||
        col_id == BCFamilyTable::kInvalidFamilyId) {
        return out;
    }
    const uint32_t active_index = family_active_index_from_row_col_fast(
        row_id,
        col_id,
        ctx.active_family0,
        ctx.active_family1,
        ctx.active_family_count,
        ctx.family_count
    );
    if (active_index == kInvalidFamilyActiveIndex) {
        return out;
    }

    const uint32_t count_ne = ne_desc.group_count;
    const uint32_t count_sw = sw_desc.group_count;
    const uint32_t count_se = se_desc.group_count;
    const uint32_t bitmap_len = count_ne * count_sw * count_se;
    if (bitmap_len == 0U ||
        bitmap_len > kBCMaxBucketBitmapLen ||
        bitmap_len > std::numeric_limits<BucketBitmapLen>::max()) {
        throw std::logic_error("BC Family compact fast encode bitmap length is invalid");
    }
    const uint32_t rank =
        (static_cast<uint32_t>(ne_desc.rank) * count_sw + sw_desc.rank) * count_se + se_desc.rank;
    if (rank >= bitmap_len || rank > std::numeric_limits<BucketRank>::max()) {
        throw std::logic_error("BC Family compact fast encode rank is invalid");
    }

    out.active_index = active_index;
    out.key =
        (static_cast<uint64_t>(q.nw) << 48U) |
        (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
        (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
        static_cast<uint64_t>(se_desc.packed_sum_mask);
    out.rank = static_cast<BucketRank>(rank);
    out.bitmap_len = static_cast<BucketBitmapLen>(bitmap_len);
    return out;
}

struct FamilyThreadWorkspace {
    static constexpr CellId kInvalidCachedCell = std::numeric_limits<CellId>::max();

    std::vector<uint64_t> canonical_buffer;
    std::vector<BCCellEncodedInsert> active_encoded_storage;
    std::vector<BCCellResolvedInsert> resolved_batch;
    std::vector<FamilyActiveCellBuffer> active_buffers;
    std::vector<uint32_t> touched_buffer_indices;
    FamilyId active_family0 = 0U;
    FamilyId active_family1 = 0U;
    FamilyIdList2 active_target_families;
    uint32_t active_family_count = 0U;
    uint32_t active_axis_family_count = 0U;
    FamilyThreadStats stats;

    FamilyThreadWorkspace() {
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        uint64_t bytes = 0U;
        bytes += static_cast<uint64_t>(canonical_buffer.capacity()) * sizeof(uint64_t);
        bytes += static_cast<uint64_t>(active_encoded_storage.capacity()) * sizeof(BCCellEncodedInsert);
        bytes += static_cast<uint64_t>(resolved_batch.capacity()) * sizeof(BCCellResolvedInsert);
        bytes += static_cast<uint64_t>(active_buffers.capacity()) * sizeof(FamilyActiveCellBuffer);
        bytes += static_cast<uint64_t>(touched_buffer_indices.capacity()) * sizeof(uint32_t);
        return bytes;
    }

    void prepare_active_window(
        const std::vector<CellId> &cells,
        const FamilyIdList2 &target_families,
        uint32_t axis_family_count,
        uint32_t cell_count,
        uint32_t pending_insert_buffer_size
    ) {
        active_axis_family_count = axis_family_count;
        active_family_count = static_cast<uint32_t>(target_families.size());
        if (active_family_count > 2U) {
            throw std::logic_error("BC Family active window cannot map more than two target families");
        }
        active_target_families = FamilyIdList2{};
        for (FamilyId family : target_families) {
            active_target_families.push_back(family);
        }
        active_family0 = active_family_count >= 1U ? target_families[0] : 0U;
        active_family1 = active_family_count >= 2U ? target_families[1] : 0U;

        touched_buffer_indices.clear();
        active_buffers.resize(cells.size());
        const uint64_t total_buffer_slots =
            static_cast<uint64_t>(cells.size()) * pending_insert_buffer_size;
        if (total_buffer_slots > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("BC Family per-thread active cell buffer slots exceed uint32");
        }
        if (active_encoded_storage.size() < static_cast<size_t>(total_buffer_slots)) {
            active_encoded_storage.resize(static_cast<size_t>(total_buffer_slots));
        }
        for (uint32_t i = 0U; i < cells.size(); ++i) {
            const CellId cid = cells[i];
            if (cid >= cell_count) {
                throw std::out_of_range("BC Family active cell is outside dense matrix");
            }
            active_buffers[i].reset_for_cell(
                cid,
                static_cast<uint32_t>(
                    static_cast<uint64_t>(i) * pending_insert_buffer_size
                ),
                pending_insert_buffer_size
            );
        }
    }

    [[nodiscard]] uint32_t active_index_from_row_col(FamilyId row, FamilyId col) const {
        const uint32_t f = active_axis_family_count;
        if (active_family_count == 0U || f == 0U) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint32_t g0 = active_family0;
        if (row == g0) {
            return static_cast<uint32_t>(col);
        }
        if (col == g0) {
            return f + (static_cast<uint32_t>(row) < g0
                ? static_cast<uint32_t>(row)
                : static_cast<uint32_t>(row) - 1U);
        }
        if (active_family_count < 2U) {
            return std::numeric_limits<uint32_t>::max();
        }

        const uint32_t g1 = active_family1;
        const uint32_t base1 = 2U * f - 1U;
        if (row == g1) {
            const uint32_t col_u32 = static_cast<uint32_t>(col);
            if (col_u32 == g0) {
                return std::numeric_limits<uint32_t>::max();
            }
            return base1 + col_u32 - (col_u32 > g0 ? 1U : 0U);
        }
        if (col == g1) {
            const uint32_t row_u32 = static_cast<uint32_t>(row);
            if (row_u32 == g0 || row_u32 == g1) {
                return std::numeric_limits<uint32_t>::max();
            }
            const uint32_t before =
                (row_u32 > g0 ? 1U : 0U) +
                (row_u32 > g1 ? 1U : 0U);
            return base1 + (f - 1U) + row_u32 - before;
        }
        return std::numeric_limits<uint32_t>::max();
    }
};

[[nodiscard]] uint64_t family_thread_workspace_bytes(
    const std::vector<FamilyThreadWorkspace> &workspaces
) {
    uint64_t bytes = 0U;
    for (const FamilyThreadWorkspace &workspace : workspaces) {
        bytes += workspace.allocated_bytes();
    }
    return bytes;
}

void record_family_memory_checkpoint(
    BCFamilyGenerationStats &stats,
    const BCFamilyGenerationOptions &options,
    const char *label,
    const BCFamilyMutableStore &target_store,
    const BCFamilyPositionWriter &position_writer,
    const BCFamilyStreamingGenerationSource *source,
    const std::vector<FamilyThreadWorkspace> *workspaces,
    uint64_t source_loaded_payload_bytes,
    uint64_t source_loaded_allocated_bytes,
    uint64_t active_index_bytes,
    uint64_t range_work_bytes,
    uint64_t pass_cache_bytes,
    uint64_t finalized_payload_bytes
) {
    if (options.memory_checkpoint_callback == nullptr) {
        return;
    }

    BCFamilyMemoryCheckpoint checkpoint;
    checkpoint.label = label;
    checkpoint.active_builder_bytes = target_store.active_builder_bytes();
    checkpoint.thread_workspace_bytes =
        workspaces == nullptr ? 0U : family_thread_workspace_bytes(*workspaces);
    checkpoint.source_loaded_payload_bytes = source_loaded_payload_bytes;
    checkpoint.source_loaded_allocated_bytes = source_loaded_allocated_bytes;
    checkpoint.source_reader_metadata_bytes =
        source == nullptr || source->position == nullptr ? 0U : source->position->allocated_bytes();
    checkpoint.active_index_bytes = active_index_bytes;
    checkpoint.range_work_bytes = range_work_bytes;
    checkpoint.pass_cache_bytes = pass_cache_bytes;
    checkpoint.store_static_metadata_bytes = target_store.stats().static_metadata_bytes;
    checkpoint.store_allocated_bytes = target_store.allocated_bytes();
    checkpoint.position_writer_bytes = position_writer.allocated_bytes();
    checkpoint.finalized_payload_bytes = finalized_payload_bytes;
    checkpoint.external_staging_bytes = options.memory_checkpoint_external_staging_bytes;
    checkpoint.released_builder_bytes_total =
        target_store.stats().released_builder_bytes_total;
    checkpoint.last_release_batch_builder_bytes =
        target_store.last_release_batch_builder_bytes();

    uint64_t accounted = 0U;
    const auto add = [&](uint64_t bytes, const char *message) {
        accounted = bc_checked_add_u64(accounted, bytes, message);
    };
    add(checkpoint.active_builder_bytes, "BC Family memory checkpoint active builder overflow");
    add(checkpoint.thread_workspace_bytes, "BC Family memory checkpoint workspace overflow");
    add(checkpoint.source_loaded_allocated_bytes, "BC Family memory checkpoint source loaded overflow");
    add(checkpoint.source_reader_metadata_bytes, "BC Family memory checkpoint source reader overflow");
    add(checkpoint.active_index_bytes, "BC Family memory checkpoint active index overflow");
    add(checkpoint.range_work_bytes, "BC Family memory checkpoint range work overflow");
    add(checkpoint.pass_cache_bytes, "BC Family memory checkpoint pass cache overflow");
    add(checkpoint.store_allocated_bytes, "BC Family memory checkpoint store allocated overflow");
    add(checkpoint.position_writer_bytes, "BC Family memory checkpoint position writer overflow");
    add(checkpoint.finalized_payload_bytes, "BC Family memory checkpoint finalized payload overflow");
    add(checkpoint.external_staging_bytes, "BC Family memory checkpoint external staging overflow");
    checkpoint.accounted_bytes = accounted;

    options.memory_checkpoint_callback(checkpoint, options.memory_checkpoint_context);
    if (checkpoint.process_working_set_bytes > checkpoint.accounted_bytes) {
        checkpoint.residual_bytes =
            checkpoint.process_working_set_bytes - checkpoint.accounted_bytes;
    }
    const uint64_t baseline_plus_accounted = bc_checked_add_u64(
        checkpoint.process_baseline_working_set_bytes,
        checkpoint.accounted_bytes,
        "BC Family memory checkpoint baseline/accounted overflow"
    );
    if (checkpoint.process_working_set_bytes > baseline_plus_accounted) {
        checkpoint.baseline_adjusted_residual_bytes =
            checkpoint.process_working_set_bytes - baseline_plus_accounted;
    }
    const uint32_t label_code = family_memory_checkpoint_label_code(label);

    ++stats.memory_checkpoint_count;
    stats.memory_checkpoint_process_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_process_bytes_peak,
        checkpoint.process_working_set_bytes
    );
    stats.memory_checkpoint_process_peak_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_process_peak_bytes_peak,
        checkpoint.process_peak_working_set_bytes
    );
    stats.memory_checkpoint_process_private_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_process_private_bytes_peak,
        checkpoint.process_private_bytes
    );
    stats.memory_checkpoint_process_pagefile_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_process_pagefile_bytes_peak,
        checkpoint.process_pagefile_bytes
    );
    stats.memory_checkpoint_process_peak_pagefile_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_process_peak_pagefile_bytes_peak,
        checkpoint.process_peak_pagefile_bytes
    );
    stats.memory_checkpoint_accounted_bytes_peak = std::max<uint64_t>(
        stats.memory_checkpoint_accounted_bytes_peak,
        checkpoint.accounted_bytes
    );
    stats.store_allocated_bytes_peak = std::max<uint64_t>(
        stats.store_allocated_bytes_peak,
        checkpoint.store_allocated_bytes
    );
    if (checkpoint.residual_bytes > stats.memory_checkpoint_residual_bytes_peak) {
        stats.memory_checkpoint_residual_bytes_peak = checkpoint.residual_bytes;
        stats.memory_checkpoint_residual_peak_label = label_code;
    }
    if (checkpoint.baseline_adjusted_residual_bytes >
        stats.memory_checkpoint_baseline_adjusted_residual_peak) {
        stats.memory_checkpoint_baseline_adjusted_residual_peak =
            checkpoint.baseline_adjusted_residual_bytes;
        stats.memory_checkpoint_baseline_adjusted_residual_peak_label = label_code;
    }
}

void add_thread_stats(BCFamilyGenerationStats &dst, const FamilyThreadStats &src) {
    dst.source_boards_scanned += src.source_boards_scanned;
    dst.spawned_boards += src.spawned_boards;
    dst.move_all_dir_calls += src.move_all_dir_calls;
    dst.selective_move_calls += src.selective_move_calls;
    dst.move_results_produced += src.move_results_produced;
    dst.canonicalized_candidates += src.canonicalized_candidates;
    dst.encode_attempts += src.encode_attempts;
    dst.encode_valid_candidates += src.encode_valid_candidates;
    dst.encoded_candidates += src.encoded_candidates;
    dst.duplicate_candidates += src.duplicate_candidates;
    dst.target_window_skips += src.target_window_skips;
    dst.family_buffer_flushes += src.family_buffer_flushes;
    dst.family_buffer_flush_items += src.family_buffer_flush_items;
    dst.family_buffer_flush_max_items = std::max(dst.family_buffer_flush_max_items, src.family_buffer_flush_max_items);
    dst.family_builder_bind_calls += src.family_builder_bind_calls;
    dst.family_insert_hash_lookups += src.family_insert_hash_lookups;
    dst.family_insert_hash_probe_steps += src.family_insert_hash_probe_steps;
    dst.source_work_items += src.source_work_items;
    dst.source_bucket_ranges += src.source_bucket_ranges;
    dst.source_bitmap_words_scanned += src.source_bitmap_words_scanned;
}

void flush_family_cell_buffer(
    FamilyThreadWorkspace &workspace,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options,
    uint32_t active_index,
    bool clear_queue_flag
) {
    if (target_store.builder_overflowed()) {
        return;
    }
    if (active_index >= workspace.active_buffers.size()) {
        throw std::out_of_range("BC Family pending buffer active index out of range");
    }
    FamilyActiveCellBuffer &buffer = workspace.active_buffers[active_index];
    if (buffer.size == 0U) {
        return;
    }
    if (buffer.builder == nullptr) {
        ++workspace.stats.family_builder_bind_calls;
        buffer.builder = &target_store.get_or_create(buffer.cid);
    }
    BCCellEncodedInsert *encoded_begin = workspace.active_encoded_storage.data() + buffer.offset;
    constexpr uint32_t kMinBatchForBitmapChunk = 1U;
    BCCellBitmapThreadChunk *bitmap_chunk =
        buffer.size >= kMinBatchForBitmapChunk
            ? &buffer.bitmap_chunk
            : nullptr;
    uint64_t probe_steps = 0U;
    const BCCellInsertBatchResult batch =
        buffer.builder->resolve_and_apply_compact_batch(
        encoded_begin,
        nullptr,
        buffer.size,
        workspace.resolved_batch,
        bitmap_chunk,
        false,
        options.collect_hot_counters ? &probe_steps : nullptr
    );
    if (buffer.builder->overflowed() ||
        workspace.resolved_batch.size() != buffer.size) {
        target_store.mark_builder_overflow();
        buffer.size = 0U;
        if (clear_queue_flag) {
            buffer.queued_for_flush = false;
        }
        return;
    }
    ++workspace.stats.family_buffer_flushes;
    workspace.stats.family_buffer_flush_items += buffer.size;
    workspace.stats.family_buffer_flush_max_items =
        std::max<uint64_t>(workspace.stats.family_buffer_flush_max_items, buffer.size);
    workspace.stats.family_insert_hash_lookups += buffer.size;
    if (options.collect_hot_counters) {
        workspace.stats.family_insert_hash_probe_steps += probe_steps;
    }
    workspace.stats.encoded_candidates += buffer.size;
    workspace.stats.duplicate_candidates += batch.duplicate_ranks;
    if (batch.new_ranks + batch.duplicate_ranks != buffer.size) {
        throw std::logic_error("BC Family active pending insert result count mismatch");
    }
    buffer.size = 0U;
    if (clear_queue_flag) {
        buffer.queued_for_flush = false;
    }
    (void)options;
}

void flush_family_pending_encoded(
    FamilyThreadWorkspace &workspace,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options
) {
    if (target_store.builder_overflowed()) {
        return;
    }
    for (uint32_t active_index : workspace.touched_buffer_indices) {
        flush_family_cell_buffer(workspace, target_store, options, active_index, true);
        if (target_store.builder_overflowed()) {
            return;
        }
    }
    workspace.touched_buffer_indices.clear();
}

void enqueue_family_encoded_candidate(
    FamilyThreadWorkspace &workspace,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options,
    const FamilyEncodedCandidate &encoded
) {
    const uint32_t active_index = encoded.active_index;
    if (active_index >= workspace.active_buffers.size()) {
        ++workspace.stats.target_window_skips;
        return;
    }
    FamilyActiveCellBuffer &buffer = workspace.active_buffers[active_index];
    const CellId cid = buffer.cid;
    if (buffer.builder == nullptr) {
        ++workspace.stats.family_builder_bind_calls;
        buffer.builder = &target_store.get_or_create(cid);
    }
    if (!buffer.queued_for_flush) {
        workspace.touched_buffer_indices.push_back(active_index);
        buffer.queued_for_flush = true;
    }
    if (buffer.size >= buffer.capacity) {
        flush_family_cell_buffer(workspace, target_store, options, active_index, false);
        if (target_store.builder_overflowed()) {
            return;
        }
    }
    if (buffer.size >= buffer.capacity) {
        throw std::logic_error("BC Family active cell buffer did not flush before enqueue");
    }
    const uint32_t write_index = buffer.offset + buffer.size;
    const uint32_t home_slot = buffer.builder->fixed_capacity_mode()
        ? buffer.builder->home_slot_for_key(encoded.key)
        : BCCellEncodedInsert::kInvalidHomeSlot;
    workspace.active_encoded_storage[write_index] = BCCellEncodedInsert{
        encoded.key,
        encoded.rank,
        encoded.bitmap_len,
        home_slot
    };
    ++buffer.size;
    if (buffer.size >= options.pending_insert_buffer_size) {
        flush_family_cell_buffer(workspace, target_store, options, active_index, false);
    }
}

void flush_family_canonical_buffer(
    FamilyThreadWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyPartitionLayerMap *target_partition,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options
) {
    if (workspace.canonical_buffer.empty()) {
        return;
    }
    CanonicalBatch::canonicalize_inplace(
        workspace.canonical_buffer.data(),
        workspace.canonical_buffer.size(),
        options.canonical_symm_mode
    );
    workspace.stats.canonicalized_candidates += workspace.canonical_buffer.size();

    const bool use_fast_unit2 =
        target_axis.family_unit() == 2U &&
        workspace.active_family_count != 0U;
    const FamilyEncodeHotContext hot_context{
        &target_axis,
        target_axis.layer_sum(),
        target_axis.family_count(),
        workspace.active_family0,
        workspace.active_family1,
        workspace.active_family_count,
        target_partition
    };

    for (uint64_t canonical : workspace.canonical_buffer) {
        ++workspace.stats.encode_attempts;
        const BCQuadrantWords q = unpack_board_to_quadrants(canonical);
        const FamilyEncodedCandidate encoded =
            use_fast_unit2
                ? encode_family_candidate_compact_unit2(
                    lut,
                    hot_context,
                    q
                )
                : encode_family_candidate_compact(
                    lut,
                    target_axis,
                    workspace.active_target_families,
                    q,
                    target_partition
                );
        if (encoded.active_index == kInvalidFamilyActiveIndex) {
            ++workspace.stats.target_window_skips;
            continue;
        }
        ++workspace.stats.encode_valid_candidates;
        enqueue_family_encoded_candidate(
            workspace,
            target_store,
            options,
            encoded
        );
    }
    workspace.canonical_buffer.clear();
}

void push_family_moved_board(
    FamilyThreadWorkspace &workspace,
    uint64_t spawned,
    uint64_t moved,
    const BCFamilyGenerationOptions &options
) {
    if (moved == spawned) {
        return;
    }
    ++workspace.stats.move_results_produced;
    if (options.keep_only_success_generated_boards &&
        !family_is_success_board(moved, options)) {
        return;
    }
    workspace.canonical_buffer.push_back(moved);
}

[[nodiscard]] uint32_t family_countr_zero32(uint32_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC family countr_zero32 requires non-zero value");
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

void process_family_source_board(
    FamilyThreadWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyPartitionLayerMap *target_partition,
    BCDirectionMask directions,
    uint8_t spawn_tile_rank,
    uint64_t board,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options,
    bool skip_success_source
) {
    if (skip_success_source && family_is_success_board(board, options)) {
        return;
    }
    uint32_t empty_mask = bc_zero_cell_mask16(board);
    while (empty_mask != 0U) {
        const uint32_t cell = family_countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;
        const uint64_t spawned =
            board | (static_cast<uint64_t>(spawn_tile_rank) << (4U * cell));
        ++workspace.stats.spawned_boards;
        if (directions == BCDirectionMask::Both) {
            ++workspace.stats.move_all_dir_calls;
            const auto moved = BoardMover::move_all_dir(spawned);
            push_family_moved_board(workspace, spawned, std::get<0>(moved), options);
            push_family_moved_board(workspace, spawned, std::get<1>(moved), options);
            push_family_moved_board(workspace, spawned, std::get<2>(moved), options);
            push_family_moved_board(workspace, spawned, std::get<3>(moved), options);
        } else {
            if (bc_has_horizontal(directions)) {
                workspace.stats.selective_move_calls += 2U;
                const auto moved = BoardMover::move_horizontal_pair(spawned);
                push_family_moved_board(
                    workspace,
                    spawned,
                    moved.first,
                    options
                );
                push_family_moved_board(
                    workspace,
                    spawned,
                    moved.second,
                    options
                );
            }
            if (bc_has_vertical(directions)) {
                workspace.stats.selective_move_calls += 2U;
                const auto moved = BoardMover::move_vertical_pair(spawned);
                push_family_moved_board(
                    workspace,
                    spawned,
                    moved.first,
                    options
                );
                push_family_moved_board(
                    workspace,
                    spawned,
                    moved.second,
                    options
                );
            }
        }
        if (workspace.canonical_buffer.size() >= options.canonical_batch_size) {
            flush_family_canonical_buffer(
                workspace,
                lut,
                target_axis,
                target_partition,
                target_store,
                options
            );
        }
    }
}

struct FamilySourceRangeWork {
    uint32_t loaded_index = 0U;
    uint32_t bucket_begin = 0U;
    uint32_t bucket_end = 0U;
    uint32_t word_begin = 0U;
    uint32_t word_end = 0U;
    BCDirectionMask directions = BCDirectionMask::None;
};

struct FamilyCachedPass {
    BCFamilyGenerationPass pass;
    bool has_source_rows = false;
    std::vector<BCSourceCellWork> source_work;
    std::vector<CellId> source_cids;
    std::vector<CellId> target_need_cells;
    std::vector<CellId> keep_cells;
    std::vector<CellId> boundary_cells;
};

[[nodiscard]] uint64_t family_cached_pass_bytes(const std::vector<FamilyCachedPass> &passes) {
    uint64_t bytes = static_cast<uint64_t>(passes.capacity()) * sizeof(FamilyCachedPass);
    for (const FamilyCachedPass &pass : passes) {
        bytes += static_cast<uint64_t>(pass.source_work.capacity()) * sizeof(BCSourceCellWork);
        bytes += static_cast<uint64_t>(pass.source_cids.capacity()) * sizeof(CellId);
        bytes += static_cast<uint64_t>(pass.target_need_cells.capacity()) * sizeof(CellId);
        bytes += static_cast<uint64_t>(pass.keep_cells.capacity()) * sizeof(CellId);
        bytes += static_cast<uint64_t>(pass.boundary_cells.capacity()) * sizeof(CellId);
    }
    return bytes;
}

[[nodiscard]] bool add_family_to_small_set(std::vector<FamilyId> &families, FamilyId family) {
    if (std::find(families.begin(), families.end(), family) != families.end()) {
        return true;
    }
    if (families.size() >= 2U) {
        return false;
    }
    families.push_back(family);
    return true;
}

[[nodiscard]] bool add_fanout_to_small_set(std::vector<FamilyId> &families, const FamilyIdList2 &fanout) {
    for (FamilyId family : fanout) {
        if (!add_family_to_small_set(families, family)) {
            return false;
        }
    }
    return true;
}

void append_unique_cells(std::vector<CellId> &dst, const std::vector<CellId> &src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

void sort_unique_cells(std::vector<CellId> &cells) {
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}

[[nodiscard]] uint32_t family_countr_zero64(uint64_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC family countr_zero64 requires non-zero value");
    }
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctzll(value));
#else
    uint32_t count = 0U;
    while ((value & 1ULL) == 0ULL) {
        value >>= 1U;
        ++count;
    }
    return count;
#endif
}

[[nodiscard]] BCFamilyPartitionLayerMap make_family_partition_layer_for_generation(
    const BCFamilyTable &axis,
    const BCFamilyGenerationOptions &options
) {
    if (options.family_partition_policy.is_exact()) {
        return build_family_partition_layer_map_from_axis(axis);
    }
    if (options.family_possible_8tile_sums == nullptr) {
        throw std::invalid_argument(
            "BC modulo FamilyChain generation requires family_possible_8tile_sums"
        );
    }
    return build_family_partition_layer_map(
        axis,
        *options.family_possible_8tile_sums,
        options.family_partition_policy
    );
}

[[nodiscard]] std::vector<FamilyCachedPass> build_family_phase_pass_cache(
    BCFamilyGenerationScheduler &scheduler,
    const BCFamilyPartitionLayerMap &source_partition,
    const BCFamilyPartitionLayerMap &target_partition,
    uint32_t source_family_count,
    SpawnDeltaCoord delta_coord,
    uint8_t spawn_tile_rank,
    const std::vector<uint8_t> &source_family_has_rows,
    bool finalize_boundaries,
    BCFamilyGenerationScheduler *next_keep_scheduler,
    SpawnDeltaCoord next_keep_delta,
    const BCFamilyTable &target_axis,
    bool reverse_execution_order = false,
    bool use_last_producer_boundary = true
) {
    if (finalize_boundaries && reverse_execution_order) {
        throw std::invalid_argument("BC Family boundary-finalize phase cannot run in reverse order");
    }
    std::vector<FamilyCachedPass> cached;
    cached.reserve(source_family_count);
    if (!source_family_has_rows.empty() &&
        source_family_has_rows.size() != source_family_count) {
        throw std::invalid_argument("BC Family source-family activity map size mismatch");
    }

    auto make_base_entry = [&](FamilyId source_id) {
        FamilyCachedPass entry;
        entry.has_source_rows =
            source_family_has_rows.empty() || source_family_has_rows[source_id] != 0U;
        entry.pass.spawn_tile_rank = spawn_tile_rank;
        entry.pass.delta_coord = delta_coord;
        entry.pass.source_id = source_id;
        entry.pass.source_coord = scheduler.source_axis().id_to_coord(source_id);
        entry.source_work = scheduler.source_cells_for_family(source_id);
        std::sort(
            entry.source_work.begin(),
            entry.source_work.end(),
            [](const BCSourceCellWork &lhs, const BCSourceCellWork &rhs) {
                return lhs.cid < rhs.cid;
            }
        );
        entry.source_cids.clear();
        entry.source_cids.reserve(entry.source_work.size());
        for (const BCSourceCellWork &work : entry.source_work) {
            entry.source_cids.push_back(work.cid);
        }
        return entry;
    };

    for (FamilyId source_id = 0U; source_id < source_family_count; ++source_id) {
        FamilyCachedPass base = make_base_entry(source_id);
        if (!base.has_source_rows) {
            cached.push_back(std::move(base));
            continue;
        }
        std::vector<FamilyId> mapped =
            map_partition_source_family_to_target_families(
                source_partition,
                target_axis,
                target_partition,
                source_id,
                delta_coord
            );
        if (mapped.empty()) {
            base.has_source_rows = false;
            cached.push_back(std::move(base));
            continue;
        }
        for (size_t begin = 0U; begin < mapped.size(); begin += 2U) {
            FamilyCachedPass entry = base;
            entry.pass = scheduler.make_pass_existing_targets(source_id, delta_coord, spawn_tile_rank);
            entry.pass.target_families = FamilyIdList2{};
            entry.pass.target_families.push_back(mapped[begin]);
            if (begin + 1U < mapped.size()) {
                entry.pass.target_families.push_back(mapped[begin + 1U]);
            }
            entry.target_need_cells = scheduler.target_need_cells(entry.pass);
            cached.push_back(std::move(entry));
        }
    }

    std::vector<uint32_t> execution_order;
    execution_order.reserve(cached.size());
    if (reverse_execution_order) {
        for (uint32_t i = static_cast<uint32_t>(cached.size()); i != 0U; --i) {
            execution_order.push_back(i - 1U);
        }
    } else {
        for (uint32_t i = 0U; i < cached.size(); ++i) {
            execution_order.push_back(i);
        }
    }

    if (finalize_boundaries && use_last_producer_boundary && !execution_order.empty()) {
        const uint32_t cell_count = scheduler.target_matrix().cell_count();
        std::vector<uint32_t> last(cell_count, std::numeric_limits<uint32_t>::max());
        for (uint32_t order_i = 0U; order_i < execution_order.size(); ++order_i) {
            const FamilyCachedPass &entry = cached[execution_order[order_i]];
            if (!entry.has_source_rows) {
                continue;
            }
            for (CellId cid : entry.target_need_cells) {
                if (cid >= cell_count) {
                    throw std::out_of_range("BC Family boundary target cell out of range");
                }
                last[cid] = order_i;
            }
        }
        const uint32_t final_order = static_cast<uint32_t>(execution_order.size() - 1U);
        for (CellId cid = 0U; cid < cell_count; ++cid) {
            const uint32_t order_i =
                last[cid] == std::numeric_limits<uint32_t>::max()
                    ? final_order
                    : last[cid];
            cached[execution_order[order_i]].boundary_cells.push_back(cid);
        }
        for (FamilyCachedPass &entry : cached) {
            if (!entry.boundary_cells.empty()) {
                std::sort(entry.boundary_cells.begin(), entry.boundary_cells.end());
            }
        }
    } else if (finalize_boundaries) {
        bool has_boundary = false;
        FamilyCoord prev_boundary = 0U;
        for (uint32_t pass_index : execution_order) {
            FamilyCachedPass &entry = cached[pass_index];
            entry.boundary_cells = scheduler.boundary_cells_for_advance(
                has_boundary,
                prev_boundary,
                entry.pass.source_coord
            );
            has_boundary = true;
            prev_boundary = entry.pass.source_coord;
        }
    }

    for (uint32_t order_i = 0U; order_i < execution_order.size(); ++order_i) {
        const uint32_t pass_index = execution_order[order_i];
        FamilyCachedPass &entry = cached[pass_index];
        std::vector<FamilyId> keep_families;
        for (uint32_t future_i = order_i + 1U; future_i < execution_order.size(); ++future_i) {
            const uint32_t future_index = execution_order[future_i];
            const FamilyCachedPass &future = cached[future_index];
            std::vector<FamilyId> trial = keep_families;
            if (!add_fanout_to_small_set(trial, future.pass.target_families)) {
                break;
            }
            keep_families = std::move(trial);
            append_unique_cells(entry.keep_cells, future.target_need_cells);
        }
        if (entry.keep_cells.empty() &&
            !finalize_boundaries &&
            next_keep_scheduler != nullptr &&
            next_keep_scheduler->source_axis().family_count() != 0U) {
            entry.keep_cells =
                next_keep_scheduler->target_need_cells_for_existing_source_targets(0U, next_keep_delta);
        }
        if (!entry.keep_cells.empty()) {
            sort_unique_cells(entry.keep_cells);
        }
    }

    if (finalize_boundaries && !cached.empty()) {
        const FamilyCoord target_last = target_axis.id_to_coord(
            static_cast<FamilyId>(target_axis.family_count() - 1U)
        );
        cached.back().keep_cells.clear();
        (void)target_last;
    }
    if (!reverse_execution_order) {
        return cached;
    }
    std::vector<FamilyCachedPass> ordered;
    ordered.reserve(cached.size());
    for (uint32_t pass_index : execution_order) {
        ordered.push_back(std::move(cached[pass_index]));
    }
    return ordered;
}

[[nodiscard]] std::vector<FamilySourceRangeWork> build_family_source_range_work(
    const BCLut &lut,
    const std::vector<BCLoadedCell> &loaded,
    const std::vector<BCSourceCellWork> &source_work,
    uint32_t words_per_item
) {
    if (words_per_item == 0U) {
        throw std::invalid_argument("BC Family generation words_per_item must be non-zero");
    }
    if (loaded.size() != source_work.size()) {
        throw std::logic_error("BC Family generation range work source size mismatch");
    }
    std::vector<FamilySourceRangeWork> out;
    for (uint32_t loaded_i = 0U; loaded_i < loaded.size(); ++loaded_i) {
        const BCLoadedCellView view = loaded[loaded_i].view();
        uint32_t grouped_begin = 0U;
        uint32_t grouped_words = 0U;
        auto flush_group = [&](uint32_t end_bucket) {
            if (grouped_words == 0U || grouped_begin == end_bucket) {
                return;
            }
            out.push_back(FamilySourceRangeWork{
                loaded_i,
                grouped_begin,
                end_bucket,
                0U,
                0U,
                source_work[loaded_i].directions
            });
            grouped_words = 0U;
            grouped_begin = end_bucket;
        };
        for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = view.buckets.data[bucket_i];
            const BCBucketRankDecoder decoder(lut, bucket.key);
            const uint32_t word_count = words_for_bits(decoder.bitmap_len);
            if (word_count >= words_per_item) {
                flush_group(bucket_i);
                for (uint32_t begin = 0U; begin < word_count; begin += words_per_item) {
                    out.push_back(FamilySourceRangeWork{
                        loaded_i,
                        bucket_i,
                        bucket_i + 1U,
                        begin,
                        std::min<uint32_t>(word_count, begin + words_per_item),
                        source_work[loaded_i].directions
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
        flush_group(static_cast<uint32_t>(view.buckets.size));
    }
    return out;
}

void process_family_source_bucket_words(
    FamilyThreadWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyPartitionLayerMap *target_partition,
    uint8_t spawn_tile_rank,
    const BCLoadedCellView &view,
    const BCBucketEntry &bucket,
    uint32_t word_begin,
    uint32_t word_end,
    BCDirectionMask directions,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options,
    bool skip_success_source
) {
    const BCBucketBoardDecoder decoder(lut, bucket.key);
    const uint32_t bitmap_len = decoder.bitmap_len();
    const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
    const uint32_t effective_word_end = word_end == 0U ? bitmap_word_count : word_end;
    if (word_begin > effective_word_end || effective_word_end > bitmap_word_count) {
        throw std::out_of_range("BC Family generation source range word range out of bounds");
    }
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len
    );
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_end > view.rank_payload.size) {
        throw std::out_of_range("BC Family generation source range bitmap exceeds rank payload");
    }
    const uint8_t *bitmap_words = view.rank_payload.data + bitmap_offset;
    workspace.stats.source_bitmap_words_scanned += effective_word_end - word_begin;

    const auto emit_rank = [&](uint32_t rank_u32, uint64_t board) {
        if (rank_u32 >= bitmap_len || rank_u32 > std::numeric_limits<BucketRank>::max()) {
            throw std::logic_error("BC Family generation source range computed invalid rank");
        }
        process_family_source_board(
            workspace,
            lut,
            target_axis,
            target_partition,
            directions,
            spawn_tile_rank,
            board,
            target_store,
            options,
            skip_success_source
        );
        ++workspace.stats.source_boards_scanned;
    };

    const uint32_t count_se = decoder.rank_decoder.count_se;
    if (count_se == 0U) {
        throw std::logic_error("BC Family generation source bucket has an empty SE word group");
    }
    if (bitmap_len == 0U || word_begin == effective_word_end) {
        return;
    }
    const uint32_t rank_begin = word_begin * kBCBitmapWordBits;
    const uint32_t rank_end = std::min<uint32_t>(
        effective_word_end * kBCBitmapWordBits,
        bitmap_len
    );
    if (rank_begin >= rank_end) {
        return;
    }
    const uint32_t tmp_begin = rank_begin / count_se;
    const uint32_t tmp_end = (rank_end + count_se - 1U) / count_se;
    for (uint32_t tmp = tmp_begin; tmp < tmp_end; ++tmp) {
        const uint32_t block_begin = tmp * count_se;
        const uint32_t block_end = std::min<uint32_t>(block_begin + count_se, bitmap_len);
        const uint32_t local_begin = rank_begin > block_begin ? rank_begin - block_begin : 0U;
        const uint32_t local_end =
            rank_end < block_end ? rank_end - block_begin : block_end - block_begin;
        if (local_begin >= local_end) {
            continue;
        }
        const uint64_t base_bits = decoder.base_bits_for_tmp(tmp);
        for (uint32_t rank_se = local_begin; rank_se < local_end; ) {
            const uint32_t rank = block_begin + rank_se;
            const uint32_t word_i = rank / kBCBitmapWordBits;
            const uint32_t bit_i = rank & (kBCBitmapWordBits - 1U);
            uint64_t word =
                load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            word >>= bit_i;
            const uint32_t bits_in_word = std::min<uint32_t>(
                kBCBitmapWordBits - bit_i,
                local_end - rank_se
            );
            if (bits_in_word < kBCBitmapWordBits) {
                word &= (1ULL << bits_in_word) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = family_countr_zero64(word);
                const uint32_t se = rank_se + bit;
                emit_rank(
                    block_begin + se,
                    decoder.board_from_base_and_se(base_bits, se)
                );
                word &= word - 1U;
            }
            rank_se += bits_in_word;
        }
    }
}

void process_family_source_range(
    FamilyThreadWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyPartitionLayerMap *target_partition,
    uint8_t spawn_tile_rank,
    const BCLoadedCell &cell,
    const FamilySourceRangeWork &work,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options,
    bool skip_success_source
) {
    const BCLoadedCellView view = cell.view();
    if (work.bucket_begin > work.bucket_end || work.bucket_end > view.buckets.size) {
        throw std::out_of_range("BC Family generation source range bucket index out of range");
    }
    ++workspace.stats.source_work_items;
    ++workspace.stats.source_bucket_ranges;

    for (uint32_t bucket_i = work.bucket_begin; bucket_i < work.bucket_end; ++bucket_i) {
        const BCBucketEntry &bucket = view.buckets.data[bucket_i];
        const bool single_bucket = work.bucket_end == work.bucket_begin + 1U;
        process_family_source_bucket_words(
            workspace,
            lut,
            target_axis,
            target_partition,
            spawn_tile_rank,
            view,
            bucket,
            single_bucket ? work.word_begin : 0U,
            single_bucket ? work.word_end : 0U,
            work.directions,
            target_store,
            options,
            skip_success_source
        );
    }
}

void write_finalized_boundary_cells(
    BCFamilyMutableStore &target_store,
    BCFamilyPositionWriter &position_writer,
    const std::vector<CellId> &cells,
    const BCFamilyGenerationOptions &generation_options,
    const BCCellFinalizeOptions &finalize_options,
    BCFamilyGenerationStats &stats,
    bool stage_timing,
    const BCFamilyStreamingGenerationSource *source_for_accounting,
    const std::vector<FamilyThreadWorkspace> *workspaces_for_accounting,
    uint64_t pass_cache_bytes_for_accounting
) {
    if (family_progress_enabled()) {
        std::cerr
            << "BC_FAMILY_PROGRESS finalize_begin cells="
            << cells.size();
        if (!cells.empty()) {
            std::cerr
                << " first=" << cells.front()
                << " last=" << cells.back();
        }
        std::cerr << '\n';
    }
    BCCellMutableBuilder::FinalizeScratch finalize_scratch;
    validate_family_store_if_enabled(target_store, "before_boundary_finalize");
    for (CellId cid : cells) {
        const double finalize_begin = stage_timing ? family_now_seconds() : 0.0;
        double streamed_write_seconds = 0.0;
        if (family_progress_enabled()) {
            std::cerr
                << "BC_FAMILY_PROGRESS finalize_cell cid=" << cid
                << " state=" << static_cast<int>(target_store.state(cid))
                << '\n';
        }
        auto write_cell = [&](uint32_t bucket_count,
                              uint32_t success_rows,
                              uint64_t rank_payload_bytes,
                              auto &&emit_payload) {
            const double write_begin = stage_timing ? family_now_seconds() : 0.0;
            position_writer.write_finalized_cell_streamed(
                cid,
                bucket_count,
                success_rows,
                rank_payload_bytes,
                std::forward<decltype(emit_payload)>(emit_payload)
            );
            if (stage_timing) {
                streamed_write_seconds += family_now_seconds() - write_begin;
            }
        };
        target_store.finalize_cell_streamed_into(
            cid,
            finalize_scratch,
            write_cell,
            finalize_options
        );
        if (stage_timing) {
            const double elapsed = family_now_seconds() - finalize_begin;
            stats.finalize_seconds += elapsed > streamed_write_seconds
                ? elapsed - streamed_write_seconds
                : 0.0;
            stats.write_seconds += streamed_write_seconds;
        }
        if (family_progress_enabled()) {
            std::cerr << "BC_FAMILY_PROGRESS finalize_cell_done cid=" << cid << '\n';
        }
        const uint64_t finalized_payload_bytes = finalize_scratch.allocated_bytes();
        stats.finalize_payload_bytes_peak = std::max<uint64_t>(
            stats.finalize_payload_bytes_peak,
            finalized_payload_bytes
        );
        record_family_memory_checkpoint(
            stats,
            generation_options,
            "finalized_payloads",
            target_store,
            position_writer,
            source_for_accounting,
            workspaces_for_accounting,
            0U,
            0U,
            0U,
            0U,
            pass_cache_bytes_for_accounting,
            finalized_payload_bytes
        );
    }
    validate_family_store_if_enabled(target_store, "after_boundary_finalize");
    if (family_progress_enabled()) {
        std::cerr << "BC_FAMILY_PROGRESS finalize_done cells=" << cells.size() << '\n';
    }
    if (family_progress_enabled()) {
        std::cerr << "BC_FAMILY_PROGRESS write_done cells=" << cells.size() << '\n';
    }
}

void run_family_phase(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource &source,
    BCFamilyMutableStore &target_store,
    BCFamilyPositionWriter &position_writer,
    const BCFamilyGenerationOptions &options,
    bool finalize_boundaries,
    BCFamilyGenerationStats &stats,
    BCFamilyGenerationScheduler *next_keep_scheduler,
    SpawnDeltaCoord next_keep_delta
) {
    validate_family_source(target_axis, source);
    BCFamilyGenerationScheduler scheduler(source.position->axis(), target_axis);
    const BCFamilyPartitionLayerMap source_partition =
        make_family_partition_layer_for_generation(source.position->axis(), options);
    const BCFamilyPartitionLayerMap target_partition =
        make_family_partition_layer_for_generation(target_axis, options);
    const int thread_count = family_effective_threads(options.num_threads);
    const bool stage_timing = family_stage_timing_enabled();
    std::vector<FamilyThreadWorkspace> workspaces(static_cast<size_t>(thread_count));
    for (FamilyThreadWorkspace &workspace : workspaces) {
        workspace.canonical_buffer.reserve(options.canonical_batch_size);
    }

    bool has_boundary = false;
    FamilyCoord prev_boundary = 0U;
    const uint32_t source_family_count = source.position->axis().family_count();
    const bool skip_success_source =
        family_success_check_enabled(options, source.position->axis().layer_sum());
    const bool progress_enabled = family_progress_enabled();
    std::vector<BCSourceCellWork> source_work;
    std::vector<CellId> cids;
    std::vector<BCLoadedCell> loaded;
    std::vector<FamilySourceRangeWork> range_work;
    std::vector<CellId> active_cells;
    BCFamilyGenerationPass pass;
    uint64_t phase_pass_cache_bytes = 0U;
    bool pass_has_work = false;

    {
        record_family_memory_checkpoint(
            stats,
            options,
            "before_pass_cache",
            target_store,
            position_writer,
            &source,
            &workspaces,
            0U,
            0U,
            0U,
            0U,
            0U,
            0U
        );
        std::vector<uint8_t> source_family_has_rows(source_family_count, 0U);
        for (FamilyId source_id = 0U; source_id < source_family_count; ++source_id) {
            std::vector<BCSourceCellWork> family_work =
                scheduler.source_cells_for_family(source_id);
            std::vector<CellId> family_cids;
            family_cids.reserve(family_work.size());
            for (const BCSourceCellWork &work : family_work) {
                family_cids.push_back(work.cid);
            }
            source_family_has_rows[source_id] =
                source_family_has_success_rows(*source.position, family_cids) ? 1U : 0U;
        }
        record_family_memory_checkpoint(
            stats,
            options,
            "after_source_activity_scan",
            target_store,
            position_writer,
            &source,
            &workspaces,
            0U,
            0U,
            0U,
            0U,
            0U,
            0U
        );
        std::vector<FamilyCachedPass> cached_passes = build_family_phase_pass_cache(
            scheduler,
            source_partition,
            target_partition,
            source_family_count,
            source.delta_coord,
            source.spawn_tile_rank,
            source_family_has_rows,
            finalize_boundaries,
            next_keep_scheduler,
            next_keep_delta,
            target_axis,
            !finalize_boundaries && next_keep_scheduler != nullptr,
            true
        );
        phase_pass_cache_bytes = family_cached_pass_bytes(cached_passes);
        stats.pass_cache_bytes_peak = std::max<uint64_t>(
            stats.pass_cache_bytes_peak,
            phase_pass_cache_bytes
        );
        record_family_memory_checkpoint(
            stats,
            options,
            "after_pass_cache",
            target_store,
            position_writer,
            &source,
            &workspaces,
            0U,
            0U,
            0U,
            0U,
            phase_pass_cache_bytes,
            0U
        );
        for (FamilyId source_id = 0U; source_id < source_family_count; ++source_id) {
            const FamilyCachedPass &cached = cached_passes[source_id];
            const BCFamilyGenerationStats pass_before = stats;
            uint64_t pass_loaded_bytes = 0U;
            uint64_t pass_load_read_bytes = 0U;
            size_t pass_loaded_cells = 0U;
            size_t pass_range_work_count = 0U;
            if (progress_enabled) {
                std::cerr
                    << "BC_FAMILY_PROGRESS source_layer="
                    << source.position->axis().layer_sum()
                    << " target_layer=" << target_axis.layer_sum()
                    << " finalize=" << (finalize_boundaries ? 1 : 0)
                    << " source_id=" << cached.pass.source_id
                    << " target_families=";
                for (FamilyId family : cached.pass.target_families) {
                    std::cerr << static_cast<uint32_t>(family) << ';';
                }
                std::cerr << '\n';
            }
            const std::vector<BCSourceCellWork> &source_work_ref = cached.source_work;
            const std::vector<CellId> &active_cells_ref = cached.target_need_cells;
            pass = cached.pass;
            if (options.enforce_three_family_window && pass.target_families.size() > 2U) {
                throw std::logic_error("BC Family generation target fanout exceeds two families");
            }
            if (!cached.has_source_rows) {
                ++stats.source_families_processed;
                if (finalize_boundaries) {
                    write_finalized_boundary_cells(
                        target_store,
                        position_writer,
                        cached.boundary_cells,
                        options,
                        options.finalize_options,
                        stats,
                        stage_timing,
                        &source,
                        &workspaces,
                        phase_pass_cache_bytes
                    );
                    has_boundary = true;
                    prev_boundary = pass.source_coord;
                }
                const double dump_begin = stage_timing ? family_now_seconds() : 0.0;
                target_store.release_except(cached.keep_cells);
                validate_family_store_if_enabled(target_store, "after_empty_pass_release_except");
                if (stage_timing) {
                    stats.dump_seconds += family_now_seconds() - dump_begin;
                }
                stats.active_cell_peak = std::max<uint64_t>(
                    stats.active_cell_peak,
                    target_store.resident_cell_count()
                );
                stats.active_builder_bytes_peak = std::max<uint64_t>(
                    stats.active_builder_bytes_peak,
                    target_store.active_builder_bytes()
                );
                record_family_memory_checkpoint(
                    stats,
                    options,
                    "empty_pass_post_release",
                    target_store,
                    position_writer,
                    &source,
                    &workspaces,
                    0U,
                    0U,
                    0U,
                    0U,
                    phase_pass_cache_bytes,
                    0U
                );
                if (family_pass_stats_enabled()) {
                    append_family_pass_stats_csv(
                        target_axis,
                        source,
                        pass,
                        finalize_boundaries,
                        false,
                        cached.source_cids.size(),
                        cached.target_need_cells.size(),
                        cached.keep_cells.size(),
                        cached.boundary_cells.size(),
                        0U,
                        0U,
                        0U,
                        0U,
                        pass_before,
                        stats
                    );
                }
                continue;
            }
            const double prepare_begin = stage_timing ? family_now_seconds() : 0.0;
            target_store.prepare_target_window_for_cached_cells(pass.target_families, cached.target_need_cells);
            validate_family_store_if_enabled(target_store, "after_prepare_target_window");
            const uint64_t active_index_bytes = 0U;
            stats.active_index_bytes_peak = std::max<uint64_t>(
                stats.active_index_bytes_peak,
                active_index_bytes
            );
            if (stage_timing) {
                stats.reload_seconds += family_now_seconds() - prepare_begin;
            }
            stats.target_active_cell_peak = std::max<uint64_t>(
                stats.target_active_cell_peak,
                target_store.active_cell_count()
            );
            stats.active_family_window_peak = std::max<uint64_t>(
                stats.active_family_window_peak,
                1U + static_cast<uint64_t>(pass.target_families.size())
            );

            if (options.enforce_three_family_window &&
                source_work_ref.size() > static_cast<size_t>(2U) * source.position->axis().family_count() - 1U) {
                throw std::logic_error("BC Family generation source family view exceeds 2F-1 cells");
            }
            BCCellLoadStats load_stats;
            const double load_begin = stage_timing ? family_now_seconds() : 0.0;
            source.position->load_cells_into(cached.source_cids, loaded, &load_stats);
            if (stage_timing) {
                stats.source_load_seconds += family_now_seconds() - load_begin;
            }
            add_load_stats(stats, load_stats);
            stats.source_cells_loaded += loaded.size();
            pass_loaded_cells = loaded.size();
            pass_loaded_bytes = loaded_cells_payload_bytes(loaded);
            const uint64_t pass_loaded_allocated_bytes = loaded_cells_allocated_bytes(loaded);
            pass_load_read_bytes = load_stats.read_bytes;
            stats.source_loaded_cell_peak = std::max<uint64_t>(stats.source_loaded_cell_peak, loaded.size());
            stats.source_loaded_bytes_peak = std::max<uint64_t>(
                stats.source_loaded_bytes_peak,
                pass_loaded_bytes
            );
            stats.source_load_range_bytes_peak = std::max<uint64_t>(
                stats.source_load_range_bytes_peak,
                load_stats.read_bytes
            );
            ++stats.source_families_processed;
            if (loaded.size() != source_work_ref.size()) {
                throw std::logic_error("BC Family generation loaded source cell count mismatch");
            }
            if (options.enforce_three_family_window &&
                (1U + static_cast<uint64_t>(pass.target_families.size())) > 3U) {
                throw std::logic_error("BC Family generation active family window exceeds three families");
            }

            const double build_work_begin = stage_timing ? family_now_seconds() : 0.0;
            range_work = build_family_source_range_work(
                lut,
                loaded,
                source_work_ref,
                options.source_bitmap_words_per_work_item
            );
            pass_range_work_count = range_work.size();
            const uint64_t range_work_bytes =
                static_cast<uint64_t>(range_work.capacity()) * sizeof(FamilySourceRangeWork);
            stats.range_work_bytes_peak = std::max<uint64_t>(
                stats.range_work_bytes_peak,
                range_work_bytes
            );
            if (stage_timing) {
                stats.build_work_seconds += family_now_seconds() - build_work_begin;
            }
            record_family_memory_checkpoint(
                stats,
                options,
                "pre_parallel",
                target_store,
                position_writer,
                &source,
                &workspaces,
                pass_loaded_bytes,
                pass_loaded_allocated_bytes,
                active_index_bytes,
                range_work_bytes,
                phase_pass_cache_bytes,
                0U
            );

            const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
                1U,
                options.source_work_schedule_chunk
            ));
            std::exception_ptr first_exception;
            ++stats.parallel_region_count;
            const double parallel_begin = stage_timing ? family_now_seconds() : 0.0;
#pragma omp parallel num_threads(thread_count)
            {
                const int tid = family_thread_num();
                FamilyThreadWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
                try {
                    workspace.prepare_active_window(
                        active_cells_ref,
                        pass.target_families,
                        target_axis.family_count(),
                        target_store.cell_count(),
                        options.pending_insert_buffer_size
                    );
#pragma omp for schedule(dynamic, schedule_chunk) nowait
                    for (int64_t i = 0; i < static_cast<int64_t>(range_work.size()); ++i) {
                        const FamilySourceRangeWork &work = range_work[static_cast<size_t>(i)];
                        process_family_source_range(
                            workspace,
                            lut,
                            target_axis,
                            &target_partition,
                            source.spawn_tile_rank,
                            loaded[work.loaded_index],
                            work,
                            target_store,
                            options,
                            skip_success_source
                        );
                    }
                    flush_family_canonical_buffer(
                        workspace,
                        lut,
                        target_axis,
                        &target_partition,
                        target_store,
                        options
                    );
                    flush_family_pending_encoded(
                        workspace,
                        target_store,
                        options
                    );
                } catch (...) {
#pragma omp critical
                    {
                        if (!first_exception) {
                            first_exception = std::current_exception();
                        }
                    }
                }
            }
            if (stage_timing) {
                stats.parallel_seconds += family_now_seconds() - parallel_begin;
            }
            if (first_exception) {
                std::rethrow_exception(first_exception);
            }
            validate_family_store_if_enabled(target_store, "after_parallel_insert");
            if (target_store.builder_overflowed()) {
                throw BCCellMutableBuilderOverflow("BC Family generation cell builder capacity overflow");
            }
            for (FamilyThreadWorkspace &ws : workspaces) {
                add_thread_stats(stats, ws.stats);
                ws.stats = {};
            }
            stats.thread_workspace_bytes_peak = std::max<uint64_t>(
                stats.thread_workspace_bytes_peak,
                family_thread_workspace_bytes(workspaces)
            );
            record_family_memory_checkpoint(
                stats,
                options,
                "post_parallel",
                target_store,
                position_writer,
                &source,
                &workspaces,
                pass_loaded_bytes,
                pass_loaded_allocated_bytes,
                active_index_bytes,
                range_work_bytes,
                phase_pass_cache_bytes,
                0U
            );
            loaded.clear();
            std::vector<FamilySourceRangeWork>().swap(range_work);

            if (finalize_boundaries) {
                write_finalized_boundary_cells(
                    target_store,
                    position_writer,
                    cached.boundary_cells,
                    options,
                    options.finalize_options,
                    stats,
                    stage_timing,
                    &source,
                    &workspaces,
                    phase_pass_cache_bytes
                );
                has_boundary = true;
                prev_boundary = pass.source_coord;
            }

            const double dump_begin = stage_timing ? family_now_seconds() : 0.0;
            target_store.release_except(cached.keep_cells);
            validate_family_store_if_enabled(target_store, "after_release_except");
            if (stage_timing) {
                stats.dump_seconds += family_now_seconds() - dump_begin;
            }
            if (options.enforce_three_family_window) {
                for (CellId resident : target_store.resident_cells_for_testing()) {
                    if (std::find(cached.keep_cells.begin(), cached.keep_cells.end(), resident) == cached.keep_cells.end()) {
                        throw std::logic_error("BC Family generation retained resident cell outside keep window");
                    }
                }
            }
            stats.active_cell_peak = std::max<uint64_t>(
                stats.active_cell_peak,
                target_store.resident_cell_count()
            );
            stats.active_builder_bytes_peak = std::max<uint64_t>(
                stats.active_builder_bytes_peak,
                target_store.active_builder_bytes()
            );
            record_family_memory_checkpoint(
                stats,
                options,
                "post_release",
                target_store,
                position_writer,
                &source,
                &workspaces,
                0U,
                loaded_cells_allocated_bytes(loaded),
                active_index_bytes,
                0U,
                phase_pass_cache_bytes,
                0U
            );
            if (family_pass_stats_enabled()) {
                append_family_pass_stats_csv(
                    target_axis,
                    source,
                    pass,
                    finalize_boundaries,
                    true,
                    cached.source_cids.size(),
                    cached.target_need_cells.size(),
                    cached.keep_cells.size(),
                    cached.boundary_cells.size(),
                    pass_loaded_cells,
                    pass_loaded_bytes,
                    pass_load_read_bytes,
                    pass_range_work_count,
                    pass_before,
                    stats
                );
            }
        }
    }

    (void)has_boundary;
    (void)prev_boundary;
}

} // namespace

BCFamilyGenerationStats generate_family_position_layer_v1(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyStreamingGenerationSource *source4,
    const BCFamilyStreamingGenerationSource &source2,
    BCFamilyMutableStore &target_store,
    BCFamilyPositionWriter &position_writer,
    const BCFamilyGenerationOptions &options
) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC Family generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC Family generation pending_insert_buffer_size must be non-zero");
    }
    if (options.source_bitmap_words_per_work_item == 0U) {
        throw std::invalid_argument("BC Family generation source_bitmap_words_per_work_item must be non-zero");
    }
    const bool stage_timing = family_stage_timing_enabled();
    const double generation_begin = stage_timing ? family_now_seconds() : 0.0;
    BCFamilyGenerationStats stats;
    validate_family_source(target_axis, source2);
    if (source4 != nullptr) {
        validate_family_source(target_axis, *source4);
    }
    target_store.set_default_builder_reserve(
        options.new_cell_reserve_buckets,
        options.new_cell_reserve_bitmap_words
    );
    target_store.set_builder_reserve_hints(
        build_family_neighbor_reserve_hints(target_axis, source4, source2, options)
    );
    std::unique_ptr<BCFamilyGenerationScheduler> keep_scheduler;
    if (source4 != nullptr) {
        keep_scheduler = std::make_unique<BCFamilyGenerationScheduler>(source2.position->axis(), target_axis);
        run_family_phase(
            lut,
            target_axis,
            *source4,
            target_store,
            position_writer,
            options,
            false,
            stats,
            keep_scheduler.get(),
            source2.delta_coord
        );
        record_family_memory_checkpoint(
            stats,
            options,
            "after_plus4_phase",
            target_store,
            position_writer,
            source4,
            nullptr,
            0U,
            0U,
            0U,
            0U,
            0U,
            0U
        );
    }
    run_family_phase(
        lut,
        target_axis,
        source2,
        target_store,
        position_writer,
        options,
        true,
        stats,
        nullptr,
        0U
    );
    record_family_memory_checkpoint(
        stats,
        options,
        "after_plus2_phase",
        target_store,
        position_writer,
        &source2,
        nullptr,
        0U,
        0U,
        0U,
        0U,
        0U,
        0U
    );
    const BCFamilyMutableStoreStats &store_stats = target_store.stats();
    stats.target_cells_created = store_stats.created_builders;
    stats.target_cells_reloaded = store_stats.reloaded_builders;
    stats.target_cells_dumped = store_stats.dumped_builders;
    stats.target_cells_finalized = store_stats.finalized_cells;
    stats.target_cells_kept = store_stats.kept_resident_cells;
    stats.family_builder_hash_grows = store_stats.builder_hash_grows;
    stats.family_builder_bitmap_grows = store_stats.builder_bitmap_grows;
    stats.family_builder_hash_replaced_bytes = store_stats.builder_hash_replaced_bytes;
    stats.family_builder_bitmap_replaced_bytes = store_stats.builder_bitmap_replaced_bytes;
    stats.dump_buffer_bytes_peak = store_stats.dump_buffer_bytes_peak;
    stats.reload_dump_bytes_peak = store_stats.reload_dump_bytes_peak;
    stats.restore_builder_bytes_peak = store_stats.restore_builder_bytes_peak;
    stats.finalize_dump_bytes_peak = store_stats.finalize_dump_bytes_peak;
    stats.store_static_metadata_bytes = store_stats.static_metadata_bytes;
    stats.released_builder_bytes_total = store_stats.released_builder_bytes_total;
    stats.released_builder_bytes_peak = store_stats.released_builder_bytes_peak;
    stats.release_batch_builder_bytes_peak = store_stats.release_batch_builder_bytes_peak;
    const BCGenerationBlobIOStats &blob_stats = target_store.blob_read_stats();
    stats.blob_requested_extents = blob_stats.requested_extents;
    stats.blob_coalesced_extents = blob_stats.coalesced_extents;
    stats.blob_read_bytes = blob_stats.read_bytes;
    stats.blob_backend_read_ops = blob_stats.backend_read_ops;
    stats.blob_backend_read_bytes = blob_stats.backend_read_bytes;
    stats.blob_backend_read_seconds = blob_stats.backend_read_seconds;
    stats.blob_restore_seconds = blob_stats.restore_seconds;
    const BCGenerationBlobIOStats append_stats = target_store.blob_append_stats();
    stats.blob_append_count = append_stats.append_count;
    stats.blob_bytes_written = append_stats.bytes_written;
    stats.blob_backend_write_ops = append_stats.backend_write_ops;
    stats.blob_backend_write_bytes = append_stats.backend_write_bytes;
    stats.blob_backend_write_seconds = append_stats.backend_write_seconds;
    stats.blob_flush_seconds = append_stats.flush_seconds;
    const BCFamilyPositionWriterStats &writer_stats = position_writer.stats();
    stats.writer_bucket_bytes = writer_stats.bucket_bytes;
    stats.writer_rank_payload_bytes = writer_stats.rank_payload_bytes;
    stats.writer_bucket_stage_flushes = writer_stats.bucket_stage_flushes;
    stats.writer_rank_stage_flushes = writer_stats.rank_stage_flushes;
    stats.writer_bucket_stage_write_bytes = writer_stats.bucket_stage_write_bytes;
    stats.writer_rank_stage_write_bytes = writer_stats.rank_stage_write_bytes;
    if (stage_timing) {
        stats.generation_seconds = family_now_seconds() - generation_begin;
    }
    return stats;
}

BCFamilyLoadedPassBenchmarkResult benchmark_loaded_family_plus_spawn_pass(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCFamilyTable &source_axis,
    FamilyId source_family_id,
    SpawnDeltaCoord delta_coord,
    uint8_t spawn_tile_rank,
    const std::vector<BCLoadedCell> &loaded_cells,
    BCFamilyMutableStore &target_store,
    const BCFamilyGenerationOptions &options
) {
    if (source_family_id >= source_axis.family_count()) {
        throw std::out_of_range("BC Family loaded pass benchmark source family id out of range");
    }
    if (spawn_tile_rank == 0U || spawn_tile_rank > 15U) {
        throw std::invalid_argument("BC Family loaded pass benchmark spawn tile rank is invalid");
    }
    if (source_axis.family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC Family loaded pass benchmark source/target family unit mismatch");
    }
    const uint32_t expected_total =
        static_cast<uint32_t>(source_axis.total_coord()) + static_cast<uint32_t>(delta_coord);
    if (target_axis.total_coord() != expected_total) {
        throw std::invalid_argument("BC Family loaded pass benchmark source/target total coord mismatch");
    }
    if (options.canonical_batch_size == 0U ||
        options.pending_insert_buffer_size == 0U ||
        options.source_bitmap_words_per_work_item == 0U) {
        throw std::invalid_argument("BC Family loaded pass benchmark options contain zero batch size");
    }

    const double begin = family_now_seconds();

    BCFamilyGenerationScheduler scheduler(source_axis, target_axis);
    const BCFamilyPartitionLayerMap source_partition =
        make_family_partition_layer_for_generation(source_axis, options);
    const BCFamilyPartitionLayerMap target_partition =
        make_family_partition_layer_for_generation(target_axis, options);
    BCFamilyGenerationPass pass =
        scheduler.make_pass_existing_targets(source_family_id, delta_coord, spawn_tile_rank);
    pass.target_families = checked_partition_fanout2(
        map_partition_source_family_to_target_families(
            source_partition,
            target_axis,
            target_partition,
            source_family_id,
            delta_coord
        )
    );
    if (options.enforce_three_family_window && pass.target_families.size() > 2U) {
        throw std::logic_error("BC Family loaded pass benchmark target fanout exceeds two families");
    }
    const std::vector<BCSourceCellWork> source_work = scheduler.source_cells(pass);
    if (loaded_cells.size() != source_work.size()) {
        throw std::invalid_argument("BC Family loaded pass benchmark loaded cell count mismatch");
    }
    if (options.enforce_three_family_window &&
        source_work.size() > static_cast<size_t>(2U) * source_axis.family_count() - 1U) {
        throw std::logic_error("BC Family loaded pass benchmark source family view exceeds 2F-1 cells");
    }

    target_store.set_default_builder_reserve(
        options.new_cell_reserve_buckets,
        options.new_cell_reserve_bitmap_words
    );
    const double prepare_begin = family_now_seconds();
    target_store.prepare_target_window_for_families(pass.target_families);
    const double prepare_seconds = family_now_seconds() - prepare_begin;
    const std::vector<CellId> active_cells = target_store.active_cells();
    const double build_work_begin = family_now_seconds();
    const std::vector<FamilySourceRangeWork> range_work =
        build_family_source_range_work(
            lut,
            loaded_cells,
            source_work,
            options.source_bitmap_words_per_work_item
        );
    const double build_work_seconds = family_now_seconds() - build_work_begin;

    const int thread_count = family_effective_threads(options.num_threads);
    std::vector<FamilyThreadWorkspace> workspaces(static_cast<size_t>(thread_count));
    for (FamilyThreadWorkspace &workspace : workspaces) {
        workspace.canonical_buffer.reserve(options.canonical_batch_size);
    }

    std::exception_ptr first_exception;
    const int schedule_chunk = static_cast<int>(std::max<uint32_t>(
        1U,
        options.source_work_schedule_chunk
    ));
    const double parallel_begin = family_now_seconds();
#pragma omp parallel num_threads(thread_count)
    {
        const int tid = family_thread_num();
        FamilyThreadWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
        try {
            workspace.prepare_active_window(
                active_cells,
                pass.target_families,
                target_axis.family_count(),
                target_store.cell_count(),
                options.pending_insert_buffer_size
            );
#pragma omp for schedule(dynamic, schedule_chunk) nowait
            for (int64_t i = 0; i < static_cast<int64_t>(range_work.size()); ++i) {
                const FamilySourceRangeWork &work = range_work[static_cast<size_t>(i)];
                process_family_source_range(
                    workspace,
                    lut,
                    target_axis,
                    &target_partition,
                    spawn_tile_rank,
                    loaded_cells[work.loaded_index],
                    work,
                    target_store,
                    options,
                    false
                );
            }
            flush_family_canonical_buffer(
                workspace,
                lut,
                target_axis,
                &target_partition,
                target_store,
                options
            );
            flush_family_pending_encoded(
                workspace,
                target_store,
                options
            );
        } catch (...) {
#pragma omp critical
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    const double parallel_seconds = family_now_seconds() - parallel_begin;
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    if (target_store.builder_overflowed()) {
        throw BCCellMutableBuilderOverflow("BC Family loaded pass benchmark cell builder capacity overflow");
    }

    BCFamilyLoadedPassBenchmarkResult result;
    result.timed_seconds = family_now_seconds() - begin;
    result.stats.reload_seconds = prepare_seconds;
    result.stats.build_work_seconds = build_work_seconds;
    result.stats.parallel_seconds = parallel_seconds;
    result.stats.source_families_processed = 1U;
    result.stats.source_cells_loaded = loaded_cells.size();
    result.stats.source_loaded_cell_peak = loaded_cells.size();
    result.stats.source_loaded_bytes_peak = loaded_cells_payload_bytes(loaded_cells);
    result.stats.source_load_range_bytes_peak = result.stats.source_loaded_bytes_peak;
    result.stats.target_active_cell_peak = target_store.active_cell_count();
    result.stats.active_cell_peak = target_store.resident_cell_count();
    result.stats.active_family_window_peak = 1U + static_cast<uint64_t>(pass.target_families.size());
    result.stats.active_builder_bytes_peak = target_store.active_builder_bytes();
    result.stats.source_work_items = range_work.size();
    result.stats.parallel_region_count = 1U;
    for (const FamilyThreadWorkspace &workspace : workspaces) {
        add_thread_stats(result.stats, workspace.stats);
    }
    result.stats.generation_seconds = result.timed_seconds;
    result.unique_encoded_candidates =
        result.stats.encoded_candidates >= result.stats.duplicate_candidates
            ? result.stats.encoded_candidates - result.stats.duplicate_candidates
            : 0U;
    return result;
}


} // namespace BC
