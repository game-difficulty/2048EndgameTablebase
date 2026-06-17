#pragma once

#include "BCCellMatrix.h"
#include "BCFutureSuccessLookup.h"
#include "BCPositionCellLoader.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {

struct BCFutureFamilyWindowStats {
    uint64_t future_views_loaded = 0U;
    uint64_t requested_view_cells = 0U;
    uint64_t future_cells_loaded = 0U;
    uint64_t future_cells_retained = 0U;
    uint64_t future_cells_released = 0U;
    uint64_t active_cells = 0U;
    uint64_t active_cells_max = 0U;

    uint64_t position_requested_extents = 0U;
    uint64_t position_coalesced_extents = 0U;
    uint64_t position_requested_bytes = 0U;
    uint64_t position_read_bytes = 0U;
    uint64_t position_backend_read_ops = 0U;
    uint64_t position_backend_read_bytes = 0U;
    double position_backend_read_seconds = 0.0;

    uint64_t success_requested_extents = 0U;
    uint64_t success_coalesced_extents = 0U;
    uint64_t success_requested_bytes = 0U;
    uint64_t success_read_bytes = 0U;
    uint64_t success_backend_read_ops = 0U;
    uint64_t success_backend_read_bytes = 0U;
    double success_backend_read_seconds = 0.0;

    uint64_t lookup_count = 0U;
    uint64_t lookup_miss_count = 0U;
    uint64_t batch_lookup_count = 0U;
    uint64_t recycled_index_bytes = 0U;
    uint64_t recycled_index_bytes_max = 0U;
    uint64_t recycled_index_cells = 0U;
    uint64_t recycled_index_cells_max = 0U;
    uint64_t recycled_index_hits = 0U;

    double position_read_seconds = 0.0;
    double success_read_seconds = 0.0;
    double lookup_seconds = 0.0;
    double prepare_normalize_seconds = 0.0;
    double prepare_select_seconds = 0.0;
    double prepare_index_build_seconds = 0.0;
    double prepare_insert_sort_seconds = 0.0;
    double release_all_clear_seconds = 0.0;
    double release_except_normalize_seconds = 0.0;
    double release_except_filter_seconds = 0.0;
    double release_except_erase_seconds = 0.0;
    double release_except_ids_seconds = 0.0;
};

template <typename T>
struct BCFutureLookupResult {
    bool found = false;
    uint32_t local_success_row = 0U;
    T value{};
};

struct BCFutureLookupQuery {
    CellId cid = 0U;
    uint64_t key = 0U;
    BucketRank rank = 0U;
    uint32_t lane = 0U;
    uint32_t ref = 0U;
    bool valid = true;
};

template <typename T>
struct BCFutureBatchLookupResult {
    uint32_t ref = 0U;
    BCFutureLookupResult<T> result;
};

struct BCFutureFamilyWindowOptions {
    bool measure_scalar_lookup_seconds = false;
    uint64_t max_recycled_index_bytes = 0U;
    uint32_t release_threads = 1U;
};

[[nodiscard]] inline std::vector<CellId> bc_future_family_view_cells(
    const BCFamilyTable &axis,
    FamilyId family_id
) {
    const BCCellMatrix matrix(axis);
    const uint32_t family_count = matrix.family_count();
    if (family_id >= family_count) {
        throw std::out_of_range("BC future family window family id out of range");
    }

    std::vector<CellId> cells;
    cells.reserve(static_cast<size_t>(family_count) * 2U - 1U);
    for (uint32_t x = 0U; x < family_count; ++x) {
        cells.push_back(matrix.cid(family_id, static_cast<FamilyId>(x)));
    }
    for (uint32_t x = 0U; x < family_count; ++x) {
        if (x == family_id) {
            continue;
        }
        cells.push_back(matrix.cid(static_cast<FamilyId>(x), family_id));
    }
    std::sort(cells.begin(), cells.end());
    return cells;
}

template <typename SuccessT>
class BCFutureFamilyWindow {
public:
    BCFutureFamilyWindow(
        const BCPositionStreamingReader &position_reader,
        const BCSuccessStreamingReader &success_reader,
        BCFutureFamilyWindowOptions options = {}
    )
        : position_reader_(&position_reader),
          success_reader_(&success_reader),
          options_(options) {
        validate_readers();
    }

    void reset_stats() {
        stats_ = {};
    }

    [[nodiscard]] const BCFutureFamilyWindowStats &stats() const {
        return stats_;
    }

    [[nodiscard]] const std::vector<CellId> &active_cell_ids() const {
        return active_cell_ids_;
    }

    [[nodiscard]] bool contains(CellId cid) const {
        return find_active(cid) != active_.end();
    }

    [[nodiscard]] uint32_t active_cell_count() const {
        return checked_u32_size(active_.size(), "BC future family active cell count exceeds uint32");
    }

    void prepare_cells(const std::vector<CellId> &need_cells) {
        std::vector<CellId> next_cells = active_cell_ids_;
        next_cells.insert(next_cells.end(), need_cells.begin(), need_cells.end());
        load_cells(std::move(next_cells));
    }

    void release_all() {
        stats_.future_cells_released += active_.size();
        const auto clear_begin = std::chrono::steady_clock::now();
        release_success_payloads(0U, active_.size());
        for (ActiveCell &cell : active_) {
            recycle_released_cell(std::move(cell), true);
        }
        active_.clear();
        active_cell_ids_.clear();
        stats_.release_all_clear_seconds += seconds_since(clear_begin);
        update_active_stats();
    }

    void release_except(std::vector<CellId> keep_cells) {
        require_open();
        const auto normalize_begin = std::chrono::steady_clock::now();
        normalize_cell_list(keep_cells);
        stats_.release_except_normalize_seconds += seconds_since(normalize_begin);

        const size_t old_active_size = active_.size();
        size_t write_index = 0U;
        size_t active_index = 0U;
        const auto filter_begin = std::chrono::steady_clock::now();
        for (CellId cid : keep_cells) {
            while (active_index < active_.size() && active_[active_index].cid < cid) {
                ++active_index;
            }
            if (active_index == active_.size()) {
                break;
            }
            if (active_[active_index].cid != cid) {
                continue;
            }
            if (write_index != active_index) {
                std::swap(active_[write_index], active_[active_index]);
            }
            ++write_index;
            ++active_index;
        }
        stats_.release_except_filter_seconds += seconds_since(filter_begin);

        stats_.future_cells_retained += write_index;
        stats_.future_cells_released +=
            old_active_size >= write_index ? old_active_size - write_index : 0U;
        const auto erase_begin = std::chrono::steady_clock::now();
        release_success_payloads(write_index, active_.size());
        for (size_t i = write_index; i < active_.size(); ++i) {
            recycle_released_cell(std::move(active_[i]), true);
        }
        active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(write_index), active_.end());
        stats_.release_except_erase_seconds += seconds_since(erase_begin);
        const auto ids_begin = std::chrono::steady_clock::now();
        active_cell_ids_.clear();
        active_cell_ids_.reserve(active_.size());
        for (const ActiveCell &cell : active_) {
            active_cell_ids_.push_back(cell.cid);
        }
        update_active_stats();
        stats_.release_except_ids_seconds += seconds_since(ids_begin);
    }

    void load_family(FamilyId family_id) {
        load_cells(bc_future_family_view_cells(position_reader_->axis(), family_id));
    }

    void load_cells(std::vector<CellId> next_cells) {
        require_open();
        const auto normalize_begin = std::chrono::steady_clock::now();
        normalize_cell_list(next_cells);
        stats_.prepare_normalize_seconds += seconds_since(normalize_begin);

        ++stats_.future_views_loaded;
        stats_.requested_view_cells += next_cells.size();

        std::vector<CellId> missing;
        missing.reserve(next_cells.size());

        const size_t old_active_size = active_.size();
        size_t write_index = 0U;
        size_t active_index = 0U;
        const auto select_begin = std::chrono::steady_clock::now();
        for (CellId cid : next_cells) {
            while (active_index < active_.size() && active_[active_index].cid < cid) {
                ++active_index;
            }
            if (active_index < active_.size() && active_[active_index].cid == cid) {
                if (write_index != active_index) {
                    std::swap(active_[write_index], active_[active_index]);
                }
                ++write_index;
                ++active_index;
            } else {
                missing.push_back(cid);
            }
        }

        const uint64_t retained_count = write_index;
        const uint64_t released_count =
            old_active_size >= write_index ? old_active_size - write_index : 0U;
        stats_.future_cells_retained += retained_count;
        stats_.future_cells_released += released_count;

        release_success_payloads(write_index, active_.size());
        for (size_t i = write_index; i < active_.size(); ++i) {
            recycle_released_cell(std::move(active_[i]), true);
        }
        active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(write_index), active_.end());
        stats_.prepare_select_seconds += seconds_since(select_begin);
        std::vector<ActiveCell> loaded = load_missing_cells(missing);
        const auto insert_begin = std::chrono::steady_clock::now();
        active_.insert(
            active_.end(),
            std::make_move_iterator(loaded.begin()),
            std::make_move_iterator(loaded.end())
        );
        std::sort(
            active_.begin(),
            active_.end(),
            [](const ActiveCell &lhs, const ActiveCell &rhs) {
                return lhs.cid < rhs.cid;
            }
        );
        active_cell_ids_ = std::move(next_cells);
        update_active_stats();
        stats_.prepare_insert_sort_seconds += seconds_since(insert_begin);
    }

    [[nodiscard]] BCFutureLookupResult<SuccessT> lookup(
        CellId cid,
        uint64_t key,
        BucketRank rank,
        uint32_t lane = 0U
    ) {
        if (options_.measure_scalar_lookup_seconds) {
            const auto begin = std::chrono::steady_clock::now();
            BCFutureLookupResult<SuccessT> result = lookup_impl(cid, key, rank, lane);
            stats_.lookup_seconds += seconds_since(begin);
            return result;
        }
        return lookup_impl(cid, key, rank, lane);
    }

    [[nodiscard]] std::vector<BCFutureBatchLookupResult<SuccessT>> lookup_batch(
        const std::vector<BCFutureLookupQuery> &queries
    ) {
        const auto begin = std::chrono::steady_clock::now();
        ++stats_.batch_lookup_count;
        std::vector<BCFutureBatchLookupResult<SuccessT>> out;
        out.reserve(queries.size());
        for (const BCFutureLookupQuery &query : queries) {
            BCFutureBatchLookupResult<SuccessT> item;
            item.ref = query.ref;
            if (query.valid) {
                item.result = lookup_impl(query.cid, query.key, query.rank, query.lane);
            }
            out.push_back(item);
        }
        stats_.lookup_seconds += seconds_since(begin);
        return out;
    }

    [[nodiscard]] uint64_t active_position_resident_bytes() const {
        uint64_t bytes = 0U;
        for (const ActiveCell &cell : active_) {
            bytes += static_cast<uint64_t>(cell.position.buckets.capacity()) * sizeof(BCBucketEntry);
            bytes += static_cast<uint64_t>(cell.position.rank_payload.capacity());
            bytes += static_cast<uint64_t>(cell.index.entries.capacity()) *
                sizeof(typename BCFutureSuccessLookupView<SuccessT>::DirectEntry);
            bytes += static_cast<uint64_t>(cell.index.word_rank_bases.capacity()) *
                sizeof(uint32_t);
        }
        return bytes;
    }

    [[nodiscard]] uint64_t active_success_resident_bytes() const {
        uint64_t bytes = 0U;
        std::vector<const detail::BCAlignedBuffer *> external_buffers;
        external_buffers.reserve(active_.size());
        for (const ActiveCell &cell : active_) {
            bytes += static_cast<uint64_t>(cell.success.raw_bytes.capacity());
            bytes += static_cast<uint64_t>(cell.success.values.capacity()) * sizeof(uint32_t);
            if (cell.success.external_value_bytes) {
                external_buffers.push_back(cell.success.external_value_bytes.get());
            }
        }
        std::sort(external_buffers.begin(), external_buffers.end());
        external_buffers.erase(
            std::unique(external_buffers.begin(), external_buffers.end()),
            external_buffers.end()
        );
        for (const detail::BCAlignedBuffer *buffer : external_buffers) {
            bytes += static_cast<uint64_t>(buffer->size());
        }
        return bytes;
    }

    [[nodiscard]] uint64_t recycled_index_resident_bytes() const {
        return recycled_index_bytes_;
    }

    [[nodiscard]] BCFutureSuccessLookupView<SuccessT> open_success_lookup(
        uint32_t row_width,
        BCSuccessDTypeMode dtype,
        double &index_seconds,
        uint64_t &resident_position_bytes,
        uint64_t &resident_success_bytes
    ) const {
        require_open();
        resident_position_bytes = 0U;
        resident_success_bytes = 0U;

        const auto begin = std::chrono::steady_clock::now();
        BCFutureSuccessLookupView<SuccessT> lookup;
        if constexpr (std::is_same_v<SuccessT, uint32_t>) {
            std::vector<typename BCFutureSuccessLookupView<SuccessT>::CachedCellRef> refs;
            refs.reserve(active_.size());
            for (const ActiveCell &cell : active_) {
                typename BCFutureSuccessLookupView<SuccessT>::CachedCellRef ref;
                ref.cid = cell.cid;
                ref.rank_payload = cell.position.view().rank_payload;
                ref.index = &cell.index;
                ref.value_count = cell.success.uint32_value_count();
                ref.value_ptr = cell.success.uint32_values_data();
                refs.push_back(ref);
            }
            lookup.open_loaded_cached_refs(
                position_reader_->lut(),
                position_reader_->cell_count(),
                refs,
                row_width,
                dtype
            );
        } else {
            std::vector<BCLoadedCell> position_cells;
            position_cells.reserve(active_.size());
            std::vector<BCLoadedSuccessCell> success_cells;
            success_cells.reserve(active_.size());
            for (const ActiveCell &cell : active_) {
                position_cells.push_back(cell.position);
                success_cells.push_back(cell.success);
            }
            for (const BCLoadedCell &cell : position_cells) {
                resident_position_bytes +=
                    static_cast<uint64_t>(cell.buckets.capacity()) * sizeof(BCBucketEntry) +
                    static_cast<uint64_t>(cell.rank_payload.capacity());
            }
            for (const BCLoadedSuccessCell &cell : success_cells) {
                resident_success_bytes += static_cast<uint64_t>(cell.raw_bytes.capacity()) +
                    static_cast<uint64_t>(cell.values.capacity()) * sizeof(uint32_t);
                if (cell.external_value_bytes) {
                    resident_success_bytes +=
                        static_cast<uint64_t>(cell.external_value_bytes->size());
                }
            }
            lookup.open_loaded(
                position_reader_->lut(),
                position_reader_->cell_count(),
                std::move(position_cells),
                success_cells,
                row_width,
                dtype
            );
        }
        index_seconds += seconds_since(begin);
        return lookup;
    }

private:
    struct ActiveCell {
        CellId cid = 0U;
        BCLoadedCell position;
        BCLoadedSuccessCell success;
        typename BCFutureSuccessLookupView<SuccessT>::CachedCellIndex index;
    };

    using ActiveIterator = typename std::vector<ActiveCell>::iterator;
    using ConstActiveIterator = typename std::vector<ActiveCell>::const_iterator;

    void require_open() const {
        if (position_reader_ == nullptr || success_reader_ == nullptr) {
            throw std::logic_error("BC future family window is not open");
        }
    }

    void validate_readers() const {
        require_open();
        if (position_reader_->cell_count() != success_reader_->cell_count()) {
            throw std::invalid_argument("BC future family position/success cell_count mismatch");
        }
        if (!bc_success_dtype_matches_type<SuccessT>(success_reader_->dtype_mode())) {
            throw std::invalid_argument("BC future family success type does not match dtype");
        }
    }

    void normalize_cell_list(std::vector<CellId> &cells) const {
        const uint32_t cell_count = position_reader_->cell_count();
        for (CellId cid : cells) {
            if (cid >= cell_count) {
                throw std::out_of_range("BC future family window cell id out of range");
            }
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    }

    [[nodiscard]] static uint64_t index_resident_bytes(
        const typename BCFutureSuccessLookupView<SuccessT>::CachedCellIndex &index
    ) {
        return static_cast<uint64_t>(index.entries.capacity()) *
                sizeof(typename BCFutureSuccessLookupView<SuccessT>::DirectEntry) +
            static_cast<uint64_t>(index.word_rank_bases.capacity()) * sizeof(uint32_t);
    }

    [[nodiscard]] static uint64_t position_resident_bytes(const BCLoadedCell &position) {
        return static_cast<uint64_t>(position.buckets.capacity()) * sizeof(BCBucketEntry) +
            static_cast<uint64_t>(position.rank_payload.capacity());
    }

    [[nodiscard]] static uint64_t recycled_cell_resident_bytes(const ActiveCell &cell) {
        return index_resident_bytes(cell.index) + position_resident_bytes(cell.position);
    }

    void update_recycled_index_stats() {
        stats_.recycled_index_bytes = recycled_index_bytes_;
        stats_.recycled_index_cells = recycled_cells_.size();
        stats_.recycled_index_bytes_max = std::max(
            stats_.recycled_index_bytes_max,
            recycled_index_bytes_
        );
        stats_.recycled_index_cells_max = std::max<uint64_t>(
            stats_.recycled_index_cells_max,
            recycled_cells_.size()
        );
    }

    [[nodiscard]] ActiveCell acquire_recycled_cell() {
        if (recycled_cells_.empty()) {
            return {};
        }
        ActiveCell cell = std::move(recycled_cells_.back());
        recycled_cells_.pop_back();
        const uint64_t bytes = recycled_cell_resident_bytes(cell);
        recycled_index_bytes_ = recycled_index_bytes_ >= bytes
            ? recycled_index_bytes_ - bytes
            : 0U;
        ++stats_.recycled_index_hits;
        update_recycled_index_stats();
        return cell;
    }

    static void release_success_payload(BCLoadedSuccessCell &success) {
        std::vector<uint8_t>().swap(success.raw_bytes);
        std::vector<uint32_t>().swap(success.values);
        success.external_value_bytes.reset();
        success.external_values = nullptr;
        success.external_value_count = 0U;
        success.cid = 0U;
        success.dtype = kBCSuccessDTypeUint32;
        success.row_width = 0U;
        success.success_rows = 0U;
    }

    void release_success_payloads(size_t begin, size_t end) {
        if (begin >= end || begin >= active_.size()) {
            return;
        }
        end = std::min(end, active_.size());
        const size_t count = end - begin;
        const int threads = static_cast<int>(std::max<uint32_t>(1U, options_.release_threads));
        if (count >= 8U && threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t i = static_cast<int64_t>(begin);
                 i < static_cast<int64_t>(end);
                 ++i) {
                release_success_payload(active_[static_cast<size_t>(i)].success);
            }
#else
            for (size_t i = begin; i < end; ++i) {
                release_success_payload(active_[i].success);
            }
#endif
        } else {
            for (size_t i = begin; i < end; ++i) {
                release_success_payload(active_[i].success);
            }
        }
    }

    void recycle_released_cell(ActiveCell &&cell, bool success_already_released = false) {
        cell.position.cid = 0U;
        cell.position.success_rows = 0U;
        cell.position.buckets.clear();
        cell.position.rank_payload.clear();
        const uint64_t bytes = recycled_cell_resident_bytes(cell);
        cell.cid = 0U;
        if (!success_already_released) {
            cell.success = {};
        }
        if (bytes == 0U || options_.max_recycled_index_bytes == 0U ||
            bytes > options_.max_recycled_index_bytes ||
            recycled_index_bytes_ > options_.max_recycled_index_bytes - bytes) {
            update_recycled_index_stats();
            return;
        }
        recycled_index_bytes_ += bytes;
        recycled_cells_.push_back(std::move(cell));
        update_recycled_index_stats();
    }

    [[nodiscard]] std::vector<ActiveCell> load_missing_cells(const std::vector<CellId> &missing) {
        std::vector<ActiveCell> loaded;
        loaded.reserve(missing.size());
        if (missing.empty()) {
            return loaded;
        }

        BCCellLoadStats position_stats;
        const auto position_begin = std::chrono::steady_clock::now();
        for (CellId cid : missing) {
            ActiveCell cell = acquire_recycled_cell();
            cell.cid = cid;
            loaded.push_back(std::move(cell));
        }
        std::vector<BCLoadedCell> position_cells;
        position_cells.reserve(missing.size());
        for (ActiveCell &cell : loaded) {
            position_cells.push_back(std::move(cell.position));
        }
        position_reader_->load_cells_into(missing, position_cells, &position_stats);
        stats_.position_read_seconds += seconds_since(position_begin);

        BCSuccessLoadStats success_stats;
        const auto success_begin = std::chrono::steady_clock::now();
        std::vector<BCLoadedSuccessCell> success_cells =
            success_reader_->load_cells(
                missing,
                &success_stats,
                std::is_same_v<SuccessT, uint32_t>
            );
        stats_.success_read_seconds += seconds_since(success_begin);

        if (position_cells.size() != missing.size() || success_cells.size() != missing.size()) {
            throw std::logic_error("BC future family window loaded cell count mismatch");
        }

        add_position_stats(position_stats);
        add_success_stats(success_stats);
        stats_.future_cells_loaded += missing.size();

        for (size_t i = 0U; i < missing.size(); ++i) {
            if (position_cells[i].cid != missing[i] || success_cells[i].cid != missing[i]) {
                throw std::logic_error("BC future family window loaded cell order mismatch");
            }
            if (position_cells[i].success_rows != success_cells[i].success_rows) {
                throw std::logic_error("BC future family window position/success row mismatch");
            }
            ActiveCell &cell = loaded[i];
            cell.position = std::move(position_cells[i]);
            cell.success = std::move(success_cells[i]);
            const auto index_begin = std::chrono::steady_clock::now();
            BCFutureSuccessLookupView<SuccessT>::build_loaded_cell_index(
                position_reader_->lut(),
                cell.position,
                cell.index
            );
            stats_.prepare_index_build_seconds += seconds_since(index_begin);
        }
        return loaded;
    }

    void add_position_stats(const BCCellLoadStats &stats) {
        stats_.position_requested_extents += stats.requested_extents;
        stats_.position_coalesced_extents += stats.coalesced_extents;
        stats_.position_requested_bytes += stats.requested_bytes;
        stats_.position_read_bytes += stats.read_bytes;
        stats_.position_backend_read_ops += stats.backend_read_ops;
        stats_.position_backend_read_bytes += stats.backend_read_bytes;
        stats_.position_backend_read_seconds += stats.backend_read_seconds;
    }

    void add_success_stats(const BCSuccessLoadStats &stats) {
        stats_.success_requested_extents += stats.requested_extents;
        stats_.success_coalesced_extents += stats.coalesced_extents;
        stats_.success_requested_bytes += stats.requested_bytes;
        stats_.success_read_bytes += stats.read_bytes;
        stats_.success_backend_read_ops += stats.backend_read_ops;
        stats_.success_backend_read_bytes += stats.backend_read_bytes;
        stats_.success_backend_read_seconds += stats.backend_read_seconds;
    }

    void update_active_stats() {
        stats_.active_cells = active_.size();
        stats_.active_cells_max = std::max<uint64_t>(stats_.active_cells_max, active_.size());
    }

    [[nodiscard]] ActiveIterator find_active(CellId cid) {
        return std::lower_bound(
            active_.begin(),
            active_.end(),
            cid,
            [](const ActiveCell &cell, CellId target) {
                return cell.cid < target;
            }
        );
    }

    [[nodiscard]] ConstActiveIterator find_active(CellId cid) const {
        return std::lower_bound(
            active_.begin(),
            active_.end(),
            cid,
            [](const ActiveCell &cell, CellId target) {
                return cell.cid < target;
            }
        );
    }

    [[nodiscard]] BCFutureLookupResult<SuccessT> lookup_impl(
        CellId cid,
        uint64_t key,
        BucketRank rank,
        uint32_t lane
    ) {
        ++stats_.lookup_count;
        const ActiveIterator it = find_active(cid);
        if (it == active_.end() || it->cid != cid) {
            ++stats_.lookup_miss_count;
            return {};
        }
        const BCLookupResult row = it->position.lookup(position_reader_->lut(), key, rank);
        if (!row.found) {
            ++stats_.lookup_miss_count;
            return {};
        }
        BCFutureLookupResult<SuccessT> out;
        out.found = true;
        out.local_success_row = row.local_success_row;
        out.value = it->success.template read_value_typed<SuccessT>(row.local_success_row, lane);
        return out;
    }

    [[nodiscard]] static double seconds_since(std::chrono::steady_clock::time_point begin) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }

    const BCPositionStreamingReader *position_reader_ = nullptr;
    const BCSuccessStreamingReader *success_reader_ = nullptr;
    BCFutureFamilyWindowOptions options_;
    std::vector<ActiveCell> active_;
    std::vector<ActiveCell> recycled_cells_;
    std::vector<CellId> active_cell_ids_;
    uint64_t recycled_index_bytes_ = 0U;
    BCFutureFamilyWindowStats stats_;
};

} // namespace BC
