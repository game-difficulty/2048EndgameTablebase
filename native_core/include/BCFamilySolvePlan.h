#pragma once

#include "BCBoardCodec.h"
#include "BCCellMatrix.h"
#include "BCFamilyGenerationScheduler.h"
#include "BCFamilyPartitionPolicy.h"
#include "BCFutureSuccessLookup.h"
#include "BCPositionCellLoader.h"
#include "BCSolveEdgeKernel.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace BC {

template <typename T>
struct BCFamilyNoInitAllocator : std::allocator<T> {
    using value_type = T;

    BCFamilyNoInitAllocator() noexcept = default;

    template <typename U>
    BCFamilyNoInitAllocator(const BCFamilyNoInitAllocator<U> &) noexcept {}

    template <typename U>
    struct rebind {
        using other = BCFamilyNoInitAllocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U *ptr, Args &&...args) {
        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<U>) {
            ::new (static_cast<void *>(ptr)) U;
        } else {
            ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...);
        }
    }
};

template <typename T, typename U>
inline bool operator==(
    const BCFamilyNoInitAllocator<T> &,
    const BCFamilyNoInitAllocator<U> &
) noexcept {
    return true;
}

template <typename T, typename U>
inline bool operator!=(
    const BCFamilyNoInitAllocator<T> &,
    const BCFamilyNoInitAllocator<U> &
) noexcept {
    return false;
}

template <typename T>
using BCFamilyValueVector = std::vector<T, BCFamilyNoInitAllocator<T>>;

enum class BCFamilySolveCellVisitKind : uint8_t {
    FirstDirection,
    SecondDirection,
    DiagonalBoth,
};

struct BCFamilySolveCellWork {
    CellId cid = 0U;
    BCDirectionMask directions = BCDirectionMask::None;
    BCFamilySolveCellVisitKind visit = BCFamilySolveCellVisitKind::FirstDirection;
};

struct BCFamilySolvePassPlan {
    FamilyId fid = 0U;
    BCSolveSpawnPhase phase = BCSolveSpawnPhase::Spawn4;
    SpawnDeltaCoord delta_coord = 0U;
    FamilyIdList3 future_families;
    std::vector<CellId> future_cids;
    std::vector<BCFamilySolveCellWork> current_cells;
};

[[nodiscard]] inline BCSolveTargetFamilyFilter bc_family_make_solve_filter(
    const FamilyIdList3 &families
) {
    BCSolveTargetFamilyFilter filter;
    filter.enabled = true;
    filter.families = families;
    return filter;
}

class BCFamilySolvePlanner {
public:
    BCFamilySolvePlanner(
        const BCFamilyTable &current_axis,
        const BCFamilyPartitionLayerMap &current_partition,
        const BCFamilyTable &future_axis,
        const BCFamilyPartitionLayerMap &future_partition
    ) : current_axis_(&current_axis),
        current_partition_(&current_partition),
        future_axis_(&future_axis),
        future_partition_(&future_partition),
        scheduler_(current_axis, future_axis) {
        if (current_partition.family_count() != current_axis.family_count()) {
            throw std::invalid_argument("BC family solve current partition/axis count mismatch");
        }
        if (future_partition.family_count() != future_axis.family_count()) {
            throw std::invalid_argument("BC family solve future partition/axis count mismatch");
        }
    }

    [[nodiscard]] BCFamilySolvePassPlan make_pass(
        FamilyId fid,
        BCSolveSpawnPhase phase,
        SpawnDeltaCoord delta_coord
    ) const {
        if (fid >= current_axis_->family_count()) {
            throw std::out_of_range("BC family solve fid out of range");
        }

        std::vector<FamilyId> mapped = map_partition_source_family_to_target_families(
            *current_partition_,
            *future_axis_,
            *future_partition_,
            fid,
            delta_coord
        );

        BCFamilySolvePassPlan plan;
        plan.fid = fid;
        plan.phase = phase;
        plan.delta_coord = delta_coord;
        plan.future_families = checked_partition_fanout3(mapped);
        plan.future_cids = scheduler_.target_need_cells_for_families(plan.future_families);
        std::sort(plan.future_cids.begin(), plan.future_cids.end());

        std::vector<BCSourceCellWork> source = scheduler_.source_cells_for_family(fid);
        plan.current_cells.reserve(source.size());
        const BCCellMatrix &matrix = scheduler_.source_matrix();
        for (const BCSourceCellWork &work : source) {
            const FamilyId row = matrix.row(work.cid);
            const FamilyId col = matrix.col(work.cid);
            BCFamilySolveCellVisitKind visit = BCFamilySolveCellVisitKind::FirstDirection;
            if (row == fid && col == fid) {
                visit = BCFamilySolveCellVisitKind::DiagonalBoth;
            } else if (bc_has_horizontal(work.directions)) {
                if (row != fid) {
                    throw std::logic_error("BC family solve horizontal work does not belong to row fid");
                }
                visit = fid < col
                    ? BCFamilySolveCellVisitKind::FirstDirection
                    : BCFamilySolveCellVisitKind::SecondDirection;
            } else if (bc_has_vertical(work.directions)) {
                if (col != fid) {
                    throw std::logic_error("BC family solve vertical work does not belong to col fid");
                }
                visit = fid < row
                    ? BCFamilySolveCellVisitKind::FirstDirection
                    : BCFamilySolveCellVisitKind::SecondDirection;
            } else {
                throw std::logic_error("BC family solve source work has no direction");
            }
            plan.current_cells.push_back(BCFamilySolveCellWork{
                work.cid,
                work.directions,
                visit
            });
        }
        std::sort(
            plan.current_cells.begin(),
            plan.current_cells.end(),
            [](const BCFamilySolveCellWork &lhs, const BCFamilySolveCellWork &rhs) {
                if (lhs.cid != rhs.cid) {
                    return lhs.cid < rhs.cid;
                }
                return static_cast<uint8_t>(lhs.directions) < static_cast<uint8_t>(rhs.directions);
            }
        );
        return plan;
    }

private:
    const BCFamilyTable *current_axis_ = nullptr;
    const BCFamilyPartitionLayerMap *current_partition_ = nullptr;
    const BCFamilyTable *future_axis_ = nullptr;
    const BCFamilyPartitionLayerMap *future_partition_ = nullptr;
    mutable BCFamilyGenerationScheduler scheduler_;
};

struct BCFamilyPartialBucketLayout {
    uint32_t success_row_begin = 0U;
    uint32_t success_row_end = 0U;
    uint16_t empty_mask = 0U;
    uint32_t empty_count = 0U;
    uint64_t value_offset = 0U;
    std::array<uint8_t, kBCBoardCellCount> empty_slot_ordinals{};

    [[nodiscard]] uint32_t live_rows() const {
        return success_row_end - success_row_begin;
    }
};

struct BCFamilyPartialCellLayout {
    CellId cid = 0U;
    uint32_t success_rows = 0U;
    uint32_t row_width = 0U;
    uint64_t value_count = 0U;
    std::vector<BCFamilyPartialBucketLayout> buckets;

    [[nodiscard]] bool empty() const {
        return success_rows == 0U || buckets.empty();
    }

    [[nodiscard]] const BCFamilyPartialBucketLayout &bucket_for_success_row(
        uint32_t success_row
    ) const {
        if (success_row >= success_rows) {
            throw std::out_of_range("BC family partial success row out of range");
        }
        const auto it = std::upper_bound(
            buckets.begin(),
            buckets.end(),
            success_row,
            [](uint32_t row, const BCFamilyPartialBucketLayout &bucket) {
                return row < bucket.success_row_end;
            }
        );
        if (it == buckets.end() ||
            success_row < it->success_row_begin ||
            success_row >= it->success_row_end) {
            throw std::logic_error("BC family partial row is not covered by any bucket");
        }
        return *it;
    }

    [[nodiscard]] uint64_t value_offset_for(
        uint32_t success_row,
        uint32_t empty_cell,
        uint32_t lane
    ) const {
        return value_offset_for(bucket_for_success_row(success_row), success_row, empty_cell, lane);
    }

    [[nodiscard]] uint64_t value_offset_for(
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        uint32_t empty_cell,
        uint32_t lane
    ) const {
        if (lane >= row_width) {
            throw std::out_of_range("BC family partial lane out of range");
        }
        if (empty_cell >= kBCBoardCellCount) {
            throw std::out_of_range("BC family partial empty cell out of range");
        }
        if (success_row < bucket.success_row_begin || success_row >= bucket.success_row_end) {
            throw std::out_of_range("BC family partial row is outside supplied bucket");
        }
        const uint32_t empty_bit = 1U << empty_cell;
        if ((bucket.empty_mask & empty_bit) == 0U) {
            throw std::invalid_argument("BC family partial cell is not empty in this bucket");
        }
        const uint32_t slot_ordinal = bucket.empty_slot_ordinals[empty_cell];
        if (slot_ordinal >= bucket.empty_count) {
            throw std::logic_error("BC family partial empty slot ordinal is invalid");
        }
        const uint64_t row_ordinal =
            static_cast<uint64_t>(success_row - bucket.success_row_begin);
        const uint64_t slot_offset =
            (row_ordinal * bucket.empty_count + slot_ordinal) *
                static_cast<uint64_t>(row_width) +
            lane;
        return bucket.value_offset + slot_offset;
    }
};

[[nodiscard]] inline uint64_t bc_family_checked_mul_u64(
    uint64_t lhs,
    uint64_t rhs,
    const char *label
) {
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::overflow_error(label);
    }
    return lhs * rhs;
}

[[nodiscard]] inline BCFamilyPartialCellLayout bc_family_make_partial_cell_layout(
    const BCLut &lut,
    const BCLoadedCellView &cell,
    uint32_t row_width
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC family partial layout row_width must be non-zero");
    }
    if (cell.buckets.size != 0U && cell.buckets.data == nullptr) {
        throw std::invalid_argument("BC family partial layout bucket pointer is null");
    }

    BCFamilyPartialCellLayout out;
    out.cid = cell.cid;
    out.success_rows = cell.success_rows;
    out.row_width = row_width;
    out.buckets.reserve(cell.buckets.size);

    uint64_t value_cursor = 0U;
    for (uint32_t i = 0U; i < cell.buckets.size; ++i) {
        const BCBucketEntry &bucket = cell.buckets.data[i];
        const uint32_t begin = bucket.success_row_offset;
        const uint32_t end = i + 1U < cell.buckets.size
            ? cell.buckets.data[i + 1U].success_row_offset
            : cell.success_rows;
        if (begin > end || end > cell.success_rows) {
            throw std::runtime_error("BC family partial bucket success rows are not monotonic");
        }

        const BCBucketBoardDecoder decoder(lut, bucket.key);
        const uint16_t empty_mask = bc_bucket_empty_mask16(decoder.rank_decoder);
        const uint32_t empty_count = popcount64(static_cast<uint64_t>(empty_mask));
        const uint64_t rows = static_cast<uint64_t>(end - begin);
        const uint64_t values_for_bucket = bc_family_checked_mul_u64(
            bc_family_checked_mul_u64(rows, empty_count, "BC family partial bucket row/slot overflow"),
            row_width,
            "BC family partial bucket value overflow"
        );
        BCFamilyPartialBucketLayout partial_bucket;
        partial_bucket.success_row_begin = begin;
        partial_bucket.success_row_end = end;
        partial_bucket.empty_mask = empty_mask;
        partial_bucket.empty_count = empty_count;
        partial_bucket.value_offset = value_cursor;
        partial_bucket.empty_slot_ordinals.fill(0xFFU);
        uint32_t slot_ordinal = 0U;
        uint32_t slot_mask = empty_mask;
        while (slot_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(slot_mask);
            partial_bucket.empty_slot_ordinals[cell] = static_cast<uint8_t>(slot_ordinal++);
        }
        out.buckets.push_back(partial_bucket);
        value_cursor = bc_checked_add_u64(
            value_cursor,
            values_for_bucket,
            "BC family partial layout value count overflow"
        );
    }
    out.value_count = value_cursor;
    return out;
}

template <typename StorageT>
class BCFamilyPartialMaxCellBuffer {
public:
    void reset(const BCFamilyPartialCellLayout &layout, StorageT zero_value) {
        if (layout.value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family partial buffer value count exceeds size_t");
        }
        values_.assign(static_cast<size_t>(layout.value_count), zero_value);
    }

    [[nodiscard]] const BCFamilyValueVector<StorageT> &values() const {
        return values_;
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> &values() {
        return values_;
    }

    void write_success_row(
        const BCFamilyPartialCellLayout &layout,
        uint32_t success_row,
        const StorageT *best_by_cell_lane
    ) {
        update_success_row(layout, success_row, best_by_cell_lane, false);
    }

    void write_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_cell_lane
    ) {
        update_success_row(layout, bucket, success_row, best_by_cell_lane, false);
    }

    void write_compact_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane
    ) {
        copy_compact_success_row(layout, bucket, success_row, best_by_empty_slot_lane, false);
    }

    void merge_success_row(
        const BCFamilyPartialCellLayout &layout,
        uint32_t success_row,
        const StorageT *best_by_cell_lane
    ) {
        update_success_row(layout, success_row, best_by_cell_lane, true);
    }

    void merge_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_cell_lane
    ) {
        update_success_row(layout, bucket, success_row, best_by_cell_lane, true);
    }

    void merge_compact_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane
    ) {
        copy_compact_success_row(layout, bucket, success_row, best_by_empty_slot_lane, true);
    }

    void read_success_row(
        const BCFamilyPartialCellLayout &layout,
        uint32_t success_row,
        StorageT *best_by_cell_lane,
        StorageT zero_value
    ) const {
        const BCFamilyPartialBucketLayout &bucket = layout.bucket_for_success_row(success_row);
        read_success_row(layout, bucket, success_row, best_by_cell_lane, zero_value);
    }

    void read_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        StorageT *best_by_cell_lane,
        StorageT zero_value
    ) const {
        if (best_by_cell_lane == nullptr) {
            throw std::invalid_argument("BC family partial read destination is null");
        }
        std::fill(
            best_by_cell_lane,
            best_by_cell_lane +
                static_cast<size_t>(kBCBoardCellCount) * static_cast<size_t>(layout.row_width),
            zero_value
        );
        uint32_t mask = bucket.empty_mask;
        const uint64_t row_base = bucket.value_offset +
            static_cast<uint64_t>(success_row - bucket.success_row_begin) *
                static_cast<uint64_t>(bucket.empty_count) *
                static_cast<uint64_t>(layout.row_width);
        while (mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
            const uint64_t cell_base = row_base +
                static_cast<uint64_t>(bucket.empty_slot_ordinals[cell]) *
                    static_cast<uint64_t>(layout.row_width);
            if (layout.row_width == 1U) {
                best_by_cell_lane[cell] = values_[static_cast<size_t>(cell_base)];
                continue;
            }
            for (uint32_t lane = 0U; lane < layout.row_width; ++lane) {
                best_by_cell_lane[static_cast<size_t>(cell) * layout.row_width + lane] =
                    values_[static_cast<size_t>(cell_base + lane)];
            }
        }
    }

    void read_compact_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        StorageT *best_by_empty_slot_lane,
        StorageT zero_value
    ) const {
        if (best_by_empty_slot_lane == nullptr) {
            throw std::invalid_argument("BC family partial compact read destination is null");
        }
        const size_t value_count =
            static_cast<size_t>(bucket.empty_count) * static_cast<size_t>(layout.row_width);
        if (value_count == 0U) {
            return;
        }
        const uint64_t base = compact_row_base(layout, bucket, success_row);
        if (base + value_count > values_.size()) {
            throw std::out_of_range("BC family partial compact row read is out of range");
        }
        (void)zero_value;
        std::copy_n(
            values_.data() + static_cast<size_t>(base),
            value_count,
            best_by_empty_slot_lane
        );
    }

    [[nodiscard]] const StorageT *compact_success_row_data(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row
    ) const {
        const size_t value_count =
            static_cast<size_t>(bucket.empty_count) * static_cast<size_t>(layout.row_width);
        if (value_count == 0U) {
            return nullptr;
        }
        const uint64_t base = compact_row_base(layout, bucket, success_row);
        if (base + value_count > values_.size()) {
            throw std::out_of_range("BC family partial compact row data is out of range");
        }
        return values_.data() + static_cast<size_t>(base);
    }

private:
    [[nodiscard]] static uint64_t compact_row_base(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row
    ) {
        if (success_row < bucket.success_row_begin || success_row >= bucket.success_row_end) {
            throw std::out_of_range("BC family partial compact row is outside supplied bucket");
        }
        return bucket.value_offset +
            static_cast<uint64_t>(success_row - bucket.success_row_begin) *
                static_cast<uint64_t>(bucket.empty_count) *
                static_cast<uint64_t>(layout.row_width);
    }

    void copy_compact_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane,
        bool merge
    ) {
        if (best_by_empty_slot_lane == nullptr) {
            throw std::invalid_argument("BC family partial compact write source is null");
        }
        const size_t value_count =
            static_cast<size_t>(bucket.empty_count) * static_cast<size_t>(layout.row_width);
        if (value_count == 0U) {
            return;
        }
        const uint64_t base = compact_row_base(layout, bucket, success_row);
        if (base + value_count > values_.size()) {
            throw std::out_of_range("BC family partial compact row write is out of range");
        }
        StorageT *dst = values_.data() + static_cast<size_t>(base);
        if (!merge) {
            std::copy_n(best_by_empty_slot_lane, value_count, dst);
            return;
        }
        for (size_t i = 0U; i < value_count; ++i) {
            if (best_by_empty_slot_lane[i] > dst[i]) {
                dst[i] = best_by_empty_slot_lane[i];
            }
        }
    }

    void update_success_row(
        const BCFamilyPartialCellLayout &layout,
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        bool merge
    ) {
        update_success_row(
            layout,
            layout.bucket_for_success_row(success_row),
            success_row,
            best_by_cell_lane,
            merge
        );
    }

    void update_success_row(
        const BCFamilyPartialCellLayout &layout,
        const BCFamilyPartialBucketLayout &bucket,
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        bool merge
    ) {
        if (best_by_cell_lane == nullptr) {
            throw std::invalid_argument("BC family partial write source is null");
        }
        uint32_t mask = bucket.empty_mask;
        const uint64_t row_base = bucket.value_offset +
            static_cast<uint64_t>(success_row - bucket.success_row_begin) *
                static_cast<uint64_t>(bucket.empty_count) *
                static_cast<uint64_t>(layout.row_width);
        while (mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
            const uint64_t cell_base = row_base +
                static_cast<uint64_t>(bucket.empty_slot_ordinals[cell]) *
                    static_cast<uint64_t>(layout.row_width);
            if (layout.row_width == 1U) {
                StorageT &dst = values_[static_cast<size_t>(cell_base)];
                const StorageT src = best_by_cell_lane[cell];
                if (!merge || src > dst) {
                    dst = src;
                }
                continue;
            }
            for (uint32_t lane = 0U; lane < layout.row_width; ++lane) {
                StorageT &dst = values_[static_cast<size_t>(cell_base + lane)];
                const StorageT src =
                    best_by_cell_lane[static_cast<size_t>(cell) * layout.row_width + lane];
                if (!merge || src > dst) {
                    dst = src;
                }
            }
        }
    }

    BCFamilyValueVector<StorageT> values_;
};

template <typename StorageT>
class BCFamilyCellSuccessScratch {
public:
    void reset(CellId cid, uint32_t success_rows, uint32_t row_width, StorageT zero_value) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC family success scratch row_width must be non-zero");
        }
        const uint64_t value_count =
            static_cast<uint64_t>(success_rows) * static_cast<uint64_t>(row_width);
        if (value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family success scratch value count exceeds size_t");
        }
        cid_ = cid;
        success_rows_ = success_rows;
        row_width_ = row_width;
        values_.assign(static_cast<size_t>(value_count), zero_value);
    }

    void adopt(
        CellId cid,
        uint32_t success_rows,
        uint32_t row_width,
        BCFamilyValueVector<StorageT> values
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC family success scratch row_width must be non-zero");
        }
        const uint64_t value_count =
            static_cast<uint64_t>(success_rows) * static_cast<uint64_t>(row_width);
        if (value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family success scratch value count exceeds size_t");
        }
        if (values.size() != static_cast<size_t>(value_count)) {
            throw std::runtime_error("BC family success scratch adopted value count mismatch");
        }
        cid_ = cid;
        success_rows_ = success_rows;
        row_width_ = row_width;
        values_ = std::move(values);
    }

    [[nodiscard]] CellId cid() const {
        return cid_;
    }

    [[nodiscard]] uint32_t success_rows() const {
        return success_rows_;
    }

    [[nodiscard]] uint32_t row_width() const {
        return row_width_;
    }

    [[nodiscard]] const BCFamilyValueVector<StorageT> &values() const {
        return values_;
    }

    [[nodiscard]] BCFamilyValueVector<StorageT> &values() {
        return values_;
    }

    void write_zero_row(uint32_t success_row, StorageT zero_value) {
        check_row(success_row);
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            values_[value_index(success_row, lane)] = zero_value;
        }
    }

    void write_terminal_row(uint32_t success_row, StorageT terminal_value) {
        check_row(success_row);
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            values_[value_index(success_row, lane)] = terminal_value;
        }
    }

    void write_spawn4_contribution(
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        uint16_t empty_mask,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                write_weighted_average_row_ratio(
                    success_row,
                    best_by_cell_lane,
                    empty_mask,
                    empty_count,
                    1U,
                    10U,
                    zero_value,
                    false
                );
                return;
            }
        }
        write_weighted_average_row(
            success_row,
            best_by_cell_lane,
            empty_mask,
            empty_count,
            static_cast<long double>(spawn_rate4),
            zero_value,
            false
        );
    }

    void write_spawn4_compact_contribution(
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                write_weighted_average_compact_row_ratio(
                    success_row,
                    best_by_empty_slot_lane,
                    empty_count,
                    1U,
                    10U,
                    zero_value,
                    false
                );
                return;
            }
        }
        write_weighted_average_compact_row(
            success_row,
            best_by_empty_slot_lane,
            empty_count,
            static_cast<long double>(spawn_rate4),
            zero_value,
            false
        );
    }

    void write_spawn4_sum_contribution(
        uint32_t success_row,
        uint64_t sum,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if (row_width_ != 1U) {
            throw std::logic_error("BC family success scratch sum contribution requires row_width=1");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        StorageT contribution = zero_value;
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                contribution = static_cast<uint32_t>(
                    sum / (10ULL * static_cast<uint64_t>(empty_count))
                );
            } else {
                contribution = static_cast<uint32_t>(
                    (static_cast<long double>(sum) * static_cast<long double>(spawn_rate4)) /
                    static_cast<long double>(empty_count)
                );
            }
        } else {
            contribution = static_cast<StorageT>(
                (static_cast<long double>(sum) * static_cast<long double>(spawn_rate4)) /
                static_cast<long double>(empty_count)
            );
        }
        values_[value_index(success_row, 0U)] = contribution;
    }

    void finalize_spawn2_row(
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        uint16_t empty_mask,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                write_weighted_average_row_ratio(
                    success_row,
                    best_by_cell_lane,
                    empty_mask,
                    empty_count,
                    9U,
                    10U,
                    zero_value,
                    true
                );
                return;
            }
        }
        write_weighted_average_row(
            success_row,
            best_by_cell_lane,
            empty_mask,
            empty_count,
            1.0L - static_cast<long double>(spawn_rate4),
            zero_value,
            true
        );
    }

    void finalize_spawn2_compact_row(
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                write_weighted_average_compact_row_ratio(
                    success_row,
                    best_by_empty_slot_lane,
                    empty_count,
                    9U,
                    10U,
                    zero_value,
                    true
                );
                return;
            }
        }
        write_weighted_average_compact_row(
            success_row,
            best_by_empty_slot_lane,
            empty_count,
            1.0L - static_cast<long double>(spawn_rate4),
            zero_value,
            true
        );
    }

    void finalize_spawn2_sum_row(
        uint32_t success_row,
        uint64_t sum,
        uint32_t empty_count,
        double spawn_rate4,
        StorageT zero_value
    ) {
        if (row_width_ != 1U) {
            throw std::logic_error("BC family success scratch sum finalize requires row_width=1");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        StorageT contribution = zero_value;
        if constexpr (std::is_same_v<StorageT, uint32_t>) {
            if (spawn_rate4 == 0.1) {
                contribution = static_cast<uint32_t>(
                    (9ULL * sum) / (10ULL * static_cast<uint64_t>(empty_count))
                );
            } else {
                contribution = static_cast<uint32_t>(
                    (static_cast<long double>(sum) *
                        (1.0L - static_cast<long double>(spawn_rate4))) /
                    static_cast<long double>(empty_count)
                );
            }
        } else {
            contribution = static_cast<StorageT>(
                (static_cast<long double>(sum) *
                    (1.0L - static_cast<long double>(spawn_rate4))) /
                static_cast<long double>(empty_count)
            );
        }
        StorageT &dst = values_[value_index(success_row, 0U)];
        dst = static_cast<StorageT>(dst + contribution);
    }

private:
    [[nodiscard]] size_t value_index(uint32_t success_row, uint32_t lane) const {
        return static_cast<size_t>(success_row) * row_width_ + lane;
    }

    void check_row(uint32_t success_row) const {
        if (success_row >= success_rows_) {
            throw std::out_of_range("BC family success scratch row out of range");
        }
    }

    void write_weighted_average_row(
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        uint16_t empty_mask,
        uint32_t empty_count,
        long double weight,
        StorageT zero_value,
        bool add_to_existing
    ) {
        check_row(success_row);
        if (best_by_cell_lane == nullptr) {
            throw std::invalid_argument("BC family success scratch best pointer is null");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            long double sum = 0.0L;
            uint32_t mask = empty_mask;
            while (mask != 0U) {
                const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                sum += static_cast<long double>(
                    best_by_cell_lane[static_cast<size_t>(cell) * row_width_ + lane]
                );
            }
            const StorageT contribution =
                static_cast<StorageT>((sum * weight) / static_cast<long double>(empty_count));
            StorageT &dst = values_[value_index(success_row, lane)];
            dst = add_to_existing ? static_cast<StorageT>(dst + contribution) : contribution;
        }
    }

    void write_weighted_average_row_ratio(
        uint32_t success_row,
        const StorageT *best_by_cell_lane,
        uint16_t empty_mask,
        uint32_t empty_count,
        uint32_t numerator,
        uint32_t denominator,
        StorageT zero_value,
        bool add_to_existing
    ) {
        check_row(success_row);
        if (best_by_cell_lane == nullptr) {
            throw std::invalid_argument("BC family success scratch best pointer is null");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        if (denominator == 0U) {
            throw std::invalid_argument("BC family success scratch denominator is zero");
        }
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            uint64_t sum = 0U;
            uint32_t mask = empty_mask;
            while (mask != 0U) {
                const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                sum += static_cast<uint64_t>(
                    best_by_cell_lane[static_cast<size_t>(cell) * row_width_ + lane]
                );
            }
            const uint64_t divisor =
                static_cast<uint64_t>(denominator) * static_cast<uint64_t>(empty_count);
            const StorageT contribution =
                static_cast<StorageT>((sum * numerator) / divisor);
            StorageT &dst = values_[value_index(success_row, lane)];
            dst = add_to_existing ? static_cast<StorageT>(dst + contribution) : contribution;
        }
    }

    void write_weighted_average_compact_row(
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane,
        uint32_t empty_count,
        long double weight,
        StorageT zero_value,
        bool add_to_existing
    ) {
        check_row(success_row);
        if (best_by_empty_slot_lane == nullptr) {
            throw std::invalid_argument("BC family success scratch compact best pointer is null");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            long double sum = 0.0L;
            for (uint32_t slot = 0U; slot < empty_count; ++slot) {
                sum += static_cast<long double>(
                    best_by_empty_slot_lane[
                        static_cast<size_t>(slot) * row_width_ + lane
                    ]
                );
            }
            const StorageT contribution =
                static_cast<StorageT>((sum * weight) / static_cast<long double>(empty_count));
            StorageT &dst = values_[value_index(success_row, lane)];
            dst = add_to_existing ? static_cast<StorageT>(dst + contribution) : contribution;
        }
    }

    void write_weighted_average_compact_row_ratio(
        uint32_t success_row,
        const StorageT *best_by_empty_slot_lane,
        uint32_t empty_count,
        uint32_t numerator,
        uint32_t denominator,
        StorageT zero_value,
        bool add_to_existing
    ) {
        check_row(success_row);
        if (best_by_empty_slot_lane == nullptr) {
            throw std::invalid_argument("BC family success scratch compact best pointer is null");
        }
        if (empty_count == 0U) {
            write_zero_row(success_row, zero_value);
            return;
        }
        if (denominator == 0U) {
            throw std::invalid_argument("BC family success scratch compact denominator is zero");
        }
        for (uint32_t lane = 0U; lane < row_width_; ++lane) {
            uint64_t sum = 0U;
            for (uint32_t slot = 0U; slot < empty_count; ++slot) {
                sum += static_cast<uint64_t>(
                    best_by_empty_slot_lane[
                        static_cast<size_t>(slot) * row_width_ + lane
                    ]
                );
            }
            const uint64_t divisor =
                static_cast<uint64_t>(denominator) * static_cast<uint64_t>(empty_count);
            const StorageT contribution =
                static_cast<StorageT>((sum * numerator) / divisor);
            StorageT &dst = values_[value_index(success_row, lane)];
            dst = add_to_existing ? static_cast<StorageT>(dst + contribution) : contribution;
        }
    }

    CellId cid_ = 0U;
    uint32_t success_rows_ = 0U;
    uint32_t row_width_ = 0U;
    BCFamilyValueVector<StorageT> values_;
};

template <typename StorageT>
[[nodiscard]] inline BCFutureSuccessLookupView<StorageT> bc_family_open_loaded_future_lookup(
    const BCLut &lut,
    uint32_t cell_count,
    std::vector<BCLoadedCell> position_cells,
    const std::vector<BCLoadedSuccessCell> &success_cells,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    BCFutureSuccessLookupView<StorageT> lookup;
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

struct BCFamilyDirectionalPhaseStats {
    uint64_t queries = 0U;
};

template <typename StorageT, typename Mover = BoardMover>
BCSolveBoardQuerySummary bc_family_fill_directional_phase_best(
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    uint64_t source_board,
    BCDirectionMask directions,
    const BCSolveTargetFamilyFilter &filter,
    BCSolveSpawnPhase phase,
    const BCFutureSuccessLookupView<StorageT> &lookup,
    uint32_t row_width,
    StorageT zero_value,
    StorageT *best_by_cell_lane,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    const BCSolveEdgeOptions &options = {},
    const BCQuadrantWordSumTable *word_sums = nullptr,
    BCSolveEdgeStats *edge_stats = nullptr,
    BCFamilyDirectionalPhaseStats *phase_stats = nullptr,
    bool trusted_queries = false
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC family directional phase row_width must be non-zero");
    }
    if (lookup.row_width() != row_width) {
        throw std::invalid_argument("BC family directional phase lookup row_width mismatch");
    }
    if (best_by_cell_lane == nullptr) {
        throw std::invalid_argument("BC family directional phase best pointer is null");
    }
    const size_t best_count =
        static_cast<size_t>(kBCBoardCellCount) * static_cast<size_t>(row_width);
    std::fill(best_by_cell_lane, best_by_cell_lane + best_count, zero_value);

    const BCSolveBoardQuerySummary summary =
        bc_solve_collect_board_phase_queries<StorageT, Mover>(
            lut,
            future_axis,
            source_board,
            directions,
            filter,
            phase,
            workspace,
            options,
            word_sums,
            edge_stats
        );
    const std::vector<BCSolvePreparedQuery> &queries =
        phase == BCSolveSpawnPhase::Spawn4 ? workspace.queries4 : workspace.queries2;
    if (phase_stats != nullptr) {
        phase_stats->queries += queries.size();
    }
    if (summary.terminal_success || summary.empty_count == 0U || queries.empty()) {
        return summary;
    }

    std::array<StorageT, kBCBoardCellCount> lane_best;
    for (uint32_t lane = 0U; lane < row_width; ++lane) {
        lane_best.fill(zero_value);
        (void)lookup.reduce_max_queries(
            queries,
            lane_best.data(),
            lane_best.size(),
            lane,
            edge_stats,
            trusted_queries
        );
        uint32_t mask = summary.empty_mask;
        while (mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
            best_by_cell_lane[static_cast<size_t>(cell) * row_width + lane] =
                lane_best[cell];
        }
    }
    return summary;
}

} // namespace BC
