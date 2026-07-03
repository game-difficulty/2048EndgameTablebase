#pragma once

#include "BCFamilyTable.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

inline constexpr uint32_t kMaxDenseCellSetBuilderCells = 10'000'000U;

class BCCellMatrix {
public:
    explicit BCCellMatrix(const BCFamilyTable &axis) : axis_(&axis) {
        if (axis.family_count() == 0U) {
            throw std::invalid_argument("BC cell matrix requires a non-empty family axis");
        }
    }

    [[nodiscard]] const BCFamilyTable &axis() const {
        return *axis_;
    }

    [[nodiscard]] uint32_t family_count() const {
        return axis_->family_count();
    }

    [[nodiscard]] uint32_t cell_count() const {
        const uint32_t f = family_count();
        return f * f;
    }

    [[nodiscard]] CellId cid(FamilyId row, FamilyId col) const {
        const uint32_t f = family_count();
        if (row >= f || col >= f) {
            throw std::out_of_range("BC cell matrix family id out of range");
        }
        return static_cast<CellId>(static_cast<uint32_t>(row) * f + static_cast<uint32_t>(col));
    }

    [[nodiscard]] FamilyId row(CellId cid_value) const {
        if (cid_value >= cell_count()) {
            throw std::out_of_range("BC cell id out of range");
        }
        return static_cast<FamilyId>(cid_value / family_count());
    }

    [[nodiscard]] FamilyId col(CellId cid_value) const {
        if (cid_value >= cell_count()) {
            throw std::out_of_range("BC cell id out of range");
        }
        return static_cast<FamilyId>(cid_value % family_count());
    }

private:
    const BCFamilyTable *axis_ = nullptr;
};

class BCCellSetBuilder {
public:
    explicit BCCellSetBuilder(const BCCellMatrix &matrix)
        : matrix_(&matrix) {
        const uint32_t cells = matrix.cell_count();
        if (cells > kMaxDenseCellSetBuilderCells) {
            throw std::invalid_argument(
                "BC cell set builder refuses dense matrix with " +
                std::to_string(cells) +
                " cells; practical guard is " +
                std::to_string(kMaxDenseCellSetBuilderCells)
            );
        }
        marks_.assign(cells, 0U);
    }

    void begin_epoch() {
        cells_.clear();
        if (++epoch_ == 0U) {
            std::fill(marks_.begin(), marks_.end(), 0U);
            epoch_ = 1U;
        }
    }

    void add_cell(CellId cid_value) {
        if (cid_value >= marks_.size()) {
            throw std::out_of_range("BC cell set cid out of range");
        }
        uint32_t &mark = marks_[static_cast<size_t>(cid_value)];
        if (mark == epoch_) {
            return;
        }
        mark = epoch_;
        cells_.push_back(cid_value);
    }

    void add_family_cross(FamilyId g) {
        const uint32_t f = matrix_->family_count();
        if (g >= f) {
            throw std::out_of_range("BC family cross id out of range");
        }
        for (uint32_t x = 0; x < f; ++x) {
            add_cell(matrix_->cid(g, static_cast<FamilyId>(x)));
        }
        for (uint32_t x = 0; x < f; ++x) {
            if (x == g) {
                continue;
            }
            add_cell(matrix_->cid(static_cast<FamilyId>(x), g));
        }
    }

    void add_family_crosses(const std::vector<FamilyId> &family_ids) {
        for (FamilyId id : family_ids) {
            add_family_cross(id);
        }
    }

    template <typename FamilyIdRange>
    void add_family_crosses(const FamilyIdRange &family_ids) {
        for (FamilyId id : family_ids) {
            add_family_cross(id);
        }
    }

    [[nodiscard]] const std::vector<CellId> &cells() const {
        return cells_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return static_cast<uint64_t>(marks_.capacity()) * sizeof(uint32_t) +
            static_cast<uint64_t>(cells_.capacity()) * sizeof(CellId);
    }

private:
    const BCCellMatrix *matrix_ = nullptr;
    std::vector<uint32_t> marks_;
    uint32_t epoch_ = 1U;
    std::vector<CellId> cells_;
};

inline void collect_boundary_cells(
    const BCCellMatrix &matrix,
    FamilyCoord prev_boundary_coord,
    FamilyCoord current_boundary_coord,
    std::vector<CellId> &out
) {
    out.clear();
    if (current_boundary_coord <= prev_boundary_coord) {
        return;
    }

    const BCFamilyTable &axis = matrix.axis();
    uint64_t reserve_count = 0U;
    for (FamilyId id = 0U; id < axis.family_count(); ++id) {
        const FamilyCoord coord = axis.id_to_coord(id);
        if (coord <= prev_boundary_coord || coord > current_boundary_coord) {
            continue;
        }
        reserve_count += static_cast<uint64_t>(2U) * id + 1U;
    }
    if (reserve_count > out.max_size()) {
        throw std::length_error("BC boundary cell reserve exceeds vector max_size");
    }
    out.reserve(static_cast<size_t>(reserve_count));

    for (FamilyId boundary_id = 0U; boundary_id < axis.family_count(); ++boundary_id) {
        const FamilyCoord coord = axis.id_to_coord(boundary_id);
        if (coord <= prev_boundary_coord || coord > current_boundary_coord) {
            continue;
        }
        for (uint32_t x = 0; x <= boundary_id; ++x) {
            out.push_back(matrix.cid(boundary_id, static_cast<FamilyId>(x)));
        }
        for (uint32_t x = 0; x < boundary_id; ++x) {
            out.push_back(matrix.cid(static_cast<FamilyId>(x), boundary_id));
        }
    }
}

[[nodiscard]] inline std::vector<CellId> collect_boundary_cells(
    const BCCellMatrix &matrix,
    FamilyCoord prev_boundary_coord,
    FamilyCoord current_boundary_coord
) {
    std::vector<CellId> out;
    collect_boundary_cells(matrix, prev_boundary_coord, current_boundary_coord, out);
    return out;
}

} // namespace BC
