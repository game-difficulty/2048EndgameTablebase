#pragma once

#include "BCTypes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace BC {

[[nodiscard]] inline uint32_t bc_partial_checked_u32_size(size_t value, const char *label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(label);
    }
    return static_cast<uint32_t>(value);
}

struct BCPartialStoreStats {
    uint64_t cells_created = 0U;
    uint64_t cells_reused = 0U;
    uint64_t cells_released = 0U;
    uint64_t cells_finalized = 0U;
    uint64_t values_initialized = 0U;
    uint64_t values_set = 0U;
    uint64_t update_attempts = 0U;
    uint64_t values_updated = 0U;
    uint64_t active_cells = 0U;
    uint64_t active_cells_max = 0U;
    uint64_t active_bytes = 0U;
    uint64_t active_bytes_max = 0U;
};

template <typename StorageT>
class BCPartialCellBlock {
public:
    static_assert(
        std::is_same_v<StorageT, uint32_t> || std::is_same_v<StorageT, uint64_t> ||
        std::is_same_v<StorageT, float> || std::is_same_v<StorageT, double>,
        "unsupported BC partial store value type"
    );

    void reset(
        CellId cid,
        uint32_t success_rows,
        uint32_t row_width,
        StorageT initial_value
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC partial cell row_width must be non-zero");
        }
        const uint64_t value_count =
            static_cast<uint64_t>(success_rows) * static_cast<uint64_t>(row_width);
        if (value_count > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("BC partial cell value count exceeds size_t");
        }
        cid_ = cid;
        success_rows_ = success_rows;
        row_width_ = row_width;
        values_.assign(static_cast<size_t>(value_count), initial_value);
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

    [[nodiscard]] size_t value_count() const {
        return values_.size();
    }

    [[nodiscard]] uint64_t byte_size() const {
        return static_cast<uint64_t>(values_.size()) * sizeof(StorageT);
    }

    [[nodiscard]] const std::vector<StorageT> &values() const {
        return values_;
    }

    [[nodiscard]] std::vector<StorageT> copy_values() const {
        return values_;
    }

    [[nodiscard]] std::vector<StorageT> take_values() {
        success_rows_ = 0U;
        row_width_ = 0U;
        return std::move(values_);
    }

    void set(uint32_t row, uint32_t lane, StorageT value) {
        values_[index(row, lane)] = value;
    }

    bool update_best(uint32_t row, uint32_t lane, StorageT value) {
        StorageT &slot = values_[index(row, lane)];
        if (value <= slot) {
            return false;
        }
        slot = value;
        return true;
    }

private:
    [[nodiscard]] size_t index(uint32_t row, uint32_t lane) const {
        if (row >= success_rows_) {
            throw std::out_of_range("BC partial cell row out of range");
        }
        if (lane >= row_width_) {
            throw std::out_of_range("BC partial cell lane out of range");
        }
        const uint64_t offset =
            static_cast<uint64_t>(row) * static_cast<uint64_t>(row_width_) + lane;
        if (offset >= values_.size()) {
            throw std::logic_error("BC partial cell index exceeds value block");
        }
        return static_cast<size_t>(offset);
    }

    CellId cid_ = 0U;
    uint32_t success_rows_ = 0U;
    uint32_t row_width_ = 0U;
    std::vector<StorageT> values_;
};

template <typename StorageT>
class BCPartialStore {
public:
    explicit BCPartialStore(StorageT initial_value = StorageT{})
        : initial_value_(initial_value) {}

    void reset_stats() {
        stats_ = {};
        update_active_stats();
    }

    [[nodiscard]] const BCPartialStoreStats &stats() const {
        return stats_;
    }

    [[nodiscard]] uint32_t active_cell_count() const {
        return bc_partial_checked_u32_size(
            active_.size(),
            "BC partial active cell count exceeds uint32"
        );
    }

    [[nodiscard]] bool contains(CellId cid) const {
        return find_cell(cid) != active_.end();
    }

    BCPartialCellBlock<StorageT> &mutable_cell(
        CellId cid,
        uint32_t success_rows,
        uint32_t row_width
    ) {
        return mutable_cell(cid, success_rows, row_width, initial_value_);
    }

    BCPartialCellBlock<StorageT> &mutable_cell(
        CellId cid,
        uint32_t success_rows,
        uint32_t row_width,
        StorageT initial_value
    ) {
        auto it = find_cell(cid);
        if (it != active_.end()) {
            if (it->block.success_rows() != success_rows ||
                it->block.row_width() != row_width) {
                throw std::invalid_argument("BC partial cell dimensions changed while active");
            }
            ++stats_.cells_reused;
            return it->block;
        }

        ActiveCell cell;
        cell.cid = cid;
        cell.block.reset(cid, success_rows, row_width, initial_value);
        active_.push_back(std::move(cell));
        std::sort(
            active_.begin(),
            active_.end(),
            [](const ActiveCell &lhs, const ActiveCell &rhs) {
                return lhs.cid < rhs.cid;
            }
        );
        ++stats_.cells_created;
        stats_.values_initialized +=
            static_cast<uint64_t>(success_rows) * static_cast<uint64_t>(row_width);
        update_active_stats();
        return find_cell(cid)->block;
    }

    void set(CellId cid, uint32_t row, uint32_t lane, StorageT value) {
        auto it = require_cell(cid);
        it->block.set(row, lane, value);
        ++stats_.values_set;
    }

    bool update_best(CellId cid, uint32_t row, uint32_t lane, StorageT value) {
        auto it = require_cell(cid);
        ++stats_.update_attempts;
        const bool updated = it->block.update_best(row, lane, value);
        if (updated) {
            ++stats_.values_updated;
        }
        return updated;
    }

    [[nodiscard]] std::vector<StorageT> finalize_cell_values(CellId cid) {
        auto it = require_cell(cid);
        std::vector<StorageT> values = it->block.take_values();
        active_.erase(it);
        ++stats_.cells_finalized;
        update_active_stats();
        return values;
    }

    void release_cell(CellId cid) {
        auto it = find_cell(cid);
        if (it == active_.end()) {
            return;
        }
        active_.erase(it);
        ++stats_.cells_released;
        update_active_stats();
    }

    void release_except(std::vector<CellId> keep_cells) {
        std::sort(keep_cells.begin(), keep_cells.end());
        keep_cells.erase(std::unique(keep_cells.begin(), keep_cells.end()), keep_cells.end());
        const size_t before = active_.size();
        active_.erase(
            std::remove_if(
                active_.begin(),
                active_.end(),
                [&keep_cells](const ActiveCell &cell) {
                    return !std::binary_search(keep_cells.begin(), keep_cells.end(), cell.cid);
                }
            ),
            active_.end()
        );
        stats_.cells_released += before - active_.size();
        update_active_stats();
    }

    void release_all() {
        stats_.cells_released += active_.size();
        active_.clear();
        update_active_stats();
    }

private:
    struct ActiveCell {
        CellId cid = 0U;
        BCPartialCellBlock<StorageT> block;
    };

    using Iterator = typename std::vector<ActiveCell>::iterator;
    using ConstIterator = typename std::vector<ActiveCell>::const_iterator;

    [[nodiscard]] Iterator find_cell(CellId cid) {
        return std::find_if(
            active_.begin(),
            active_.end(),
            [cid](const ActiveCell &cell) {
                return cell.cid == cid;
            }
        );
    }

    [[nodiscard]] ConstIterator find_cell(CellId cid) const {
        return std::find_if(
            active_.begin(),
            active_.end(),
            [cid](const ActiveCell &cell) {
                return cell.cid == cid;
            }
        );
    }

    [[nodiscard]] Iterator require_cell(CellId cid) {
        auto it = find_cell(cid);
        if (it == active_.end()) {
            throw std::out_of_range("BC partial cell is not active");
        }
        return it;
    }

    void update_active_stats() {
        stats_.active_cells = active_.size();
        stats_.active_cells_max = std::max(stats_.active_cells_max, stats_.active_cells);
        uint64_t bytes = 0U;
        for (const ActiveCell &cell : active_) {
            bytes += cell.block.byte_size();
        }
        stats_.active_bytes = bytes;
        stats_.active_bytes_max = std::max(stats_.active_bytes_max, bytes);
    }

    StorageT initial_value_{};
    std::vector<ActiveCell> active_;
    BCPartialStoreStats stats_;
};

} // namespace BC
