#pragma once

#include "BCCellMatrix.h"
#include "BCPositionCellLoader.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

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

    uint64_t success_requested_extents = 0U;
    uint64_t success_coalesced_extents = 0U;
    uint64_t success_requested_bytes = 0U;
    uint64_t success_read_bytes = 0U;
    uint64_t success_backend_read_ops = 0U;
    uint64_t success_backend_read_bytes = 0U;

    uint64_t lookup_count = 0U;
    uint64_t lookup_miss_count = 0U;
    uint64_t batch_lookup_count = 0U;

    double position_read_seconds = 0.0;
    double success_read_seconds = 0.0;
    double lookup_seconds = 0.0;
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

    void release_all() {
        stats_.future_cells_released += active_.size();
        active_.clear();
        active_cell_ids_.clear();
        update_active_stats();
    }

    void load_family(FamilyId family_id) {
        load_cells(bc_future_family_view_cells(position_reader_->axis(), family_id));
    }

    void load_cells(std::vector<CellId> next_cells) {
        require_open();
        normalize_cell_list(next_cells);

        ++stats_.future_views_loaded;
        stats_.requested_view_cells += next_cells.size();

        std::vector<ActiveCell> retained;
        retained.reserve(std::min(active_.size(), next_cells.size()));
        std::vector<CellId> missing;
        missing.reserve(next_cells.size());

        size_t active_index = 0U;
        for (CellId cid : next_cells) {
            while (active_index < active_.size() && active_[active_index].cid < cid) {
                ++active_index;
            }
            if (active_index < active_.size() && active_[active_index].cid == cid) {
                retained.push_back(std::move(active_[active_index]));
                ++active_index;
            } else {
                missing.push_back(cid);
            }
        }

        const uint64_t retained_count = retained.size();
        const uint64_t released_count =
            active_.size() >= retained.size() ? active_.size() - retained.size() : 0U;
        stats_.future_cells_retained += retained_count;
        stats_.future_cells_released += released_count;

        std::vector<ActiveCell> loaded = load_missing_cells(missing);
        active_ = std::move(retained);
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

private:
    struct ActiveCell {
        CellId cid = 0U;
        BCLoadedCell position;
        BCLoadedSuccessCell success;
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

    [[nodiscard]] std::vector<ActiveCell> load_missing_cells(const std::vector<CellId> &missing) {
        std::vector<ActiveCell> loaded;
        loaded.reserve(missing.size());
        if (missing.empty()) {
            return loaded;
        }

        BCCellLoadStats position_stats;
        const auto position_begin = std::chrono::steady_clock::now();
        std::vector<BCLoadedCell> position_cells =
            position_reader_->load_cells(missing, &position_stats);
        stats_.position_read_seconds += seconds_since(position_begin);

        BCSuccessLoadStats success_stats;
        const auto success_begin = std::chrono::steady_clock::now();
        std::vector<BCLoadedSuccessCell> success_cells =
            success_reader_->load_cells(missing, &success_stats);
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
            ActiveCell cell;
            cell.cid = missing[i];
            cell.position = std::move(position_cells[i]);
            cell.success = std::move(success_cells[i]);
            loaded.push_back(std::move(cell));
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
    }

    void add_success_stats(const BCSuccessLoadStats &stats) {
        stats_.success_requested_extents += stats.requested_extents;
        stats_.success_coalesced_extents += stats.coalesced_extents;
        stats_.success_requested_bytes += stats.requested_bytes;
        stats_.success_read_bytes += stats.read_bytes;
        stats_.success_backend_read_ops += stats.backend_read_ops;
        stats_.success_backend_read_bytes += stats.backend_read_bytes;
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
    std::vector<CellId> active_cell_ids_;
    BCFutureFamilyWindowStats stats_;
};

} // namespace BC
