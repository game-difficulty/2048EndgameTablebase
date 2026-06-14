#pragma once

#include "BCFamilyPartitionPolicy.h"
#include "BCFamilyTable.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace BC {

using BCPositionSideCoord = uint32_t;
using BCPositionSideIndex = uint16_t;

// Dense physical cell layout for Resident/SingleChunk generation.
// Bucket keys keep true quadrant sums. Cell ids are only physical partitions:
// raw side coordinates are mapped to the serialized physical axis, usually by
// coord % modulus.
class BCPositionCellLayout {
public:
    static constexpr BCPositionSideIndex kInvalidSideIndex =
        std::numeric_limits<BCPositionSideIndex>::max();

    BCPositionCellLayout() = default;

    explicit BCPositionCellLayout(BCFamilyTable serialization_axis)
        : serialization_axis_(std::move(serialization_axis)),
          dense_side_coords_(axis_is_dense_range(serialization_axis_)),
          raw_coord_modulus_(0U) {}

    BCPositionCellLayout(BCFamilyTable serialization_axis, uint32_t raw_coord_modulus)
        : serialization_axis_(std::move(serialization_axis)),
          dense_side_coords_(axis_is_dense_range(serialization_axis_)),
          raw_coord_modulus_(raw_coord_modulus) {
        if (raw_coord_modulus_ == 0U ||
            raw_coord_modulus_ > std::numeric_limits<BCPositionSideIndex>::max()) {
            throw std::invalid_argument("BC position cell layout raw coordinate modulus is invalid");
        }
        build_raw_coord_to_physical_index_lut();
    }

    static BCPositionCellLayout from_serialized_axis(BCFamilyTable axis) {
        return BCPositionCellLayout(std::move(axis));
    }

    static BCPositionCellLayout from_modulo_axis(
        BCFamilyTable serialization_axis,
        uint32_t raw_coord_modulus
    ) {
        return BCPositionCellLayout(std::move(serialization_axis), raw_coord_modulus);
    }

    [[nodiscard]] const BCFamilyTable &serialization_axis() const {
        return serialization_axis_;
    }

    [[nodiscard]] LayerSum layer_sum() const {
        return serialization_axis_.layer_sum();
    }

    [[nodiscard]] uint16_t side_unit() const {
        return serialization_axis_.family_unit();
    }

    [[nodiscard]] uint32_t total_coord() const {
        return serialization_axis_.total_coord();
    }

    [[nodiscard]] uint32_t axis_base_coord() const {
        return serialization_axis_.axis_base_coord();
    }

    [[nodiscard]] uint32_t side_coord_count() const {
        return serialization_axis_.family_count();
    }

    [[nodiscard]] const std::vector<BCPositionSideCoord> &side_coords() const {
        return serialization_axis_.coords();
    }

    [[nodiscard]] bool side_coords_are_dense_range() const noexcept {
        return dense_side_coords_;
    }

    [[nodiscard]] bool contains_side_coord(BCPositionSideCoord coord) const {
        return serialization_axis_.contains_coord(coord);
    }

    [[nodiscard]] BCPositionSideIndex try_side_coord_to_index(BCPositionSideCoord coord) const {
        return serialization_axis_.try_coord_to_id(coord);
    }

    [[nodiscard]] BCPositionSideIndex side_coord_to_index(BCPositionSideCoord coord) const {
        return serialization_axis_.coord_to_id(coord);
    }

    [[nodiscard]] BCPositionSideIndex side_coord_to_index_trusted(BCPositionSideCoord coord) const noexcept {
        return serialization_axis_.coord_to_id_trusted(coord);
    }

    [[nodiscard]] bool uses_modulo_cells() const noexcept {
        return raw_coord_modulus_ != 0U;
    }

    [[nodiscard]] uint32_t raw_coord_modulus() const noexcept {
        return raw_coord_modulus_;
    }

    [[nodiscard]] BCPositionSideIndex try_raw_side_coord_to_physical_index(
        BCPositionSideCoord coord
    ) const {
        if (raw_coord_modulus_ == 0U) {
            return serialization_axis_.try_coord_to_id(coord);
        }
        return coord < raw_coord_to_physical_index_lut_.size()
            ? raw_coord_to_physical_index_lut_[coord]
            : kInvalidSideIndex;
    }

    [[nodiscard]] BCPositionSideIndex raw_side_coord_to_physical_index(
        BCPositionSideCoord coord
    ) const {
        const BCPositionSideIndex id = try_raw_side_coord_to_physical_index(coord);
        if (id == kInvalidSideIndex) {
            throw std::out_of_range("BC raw side coord is outside this position cell layout");
        }
        return id;
    }

    [[nodiscard]] BCPositionSideCoord side_index_to_coord(BCPositionSideIndex id) const {
        return serialization_axis_.id_to_coord(id);
    }

    [[nodiscard]] uint32_t cell_count() const {
        const uint32_t side_count = side_coord_count();
        return side_count * side_count;
    }

    [[nodiscard]] CellId cell_id(BCPositionSideIndex row_index, BCPositionSideIndex col_index) const {
        const uint32_t side_count = side_coord_count();
        if (row_index >= side_count || col_index >= side_count) {
            throw std::out_of_range("BC position cell layout side index out of range");
        }
        return static_cast<CellId>(
            static_cast<uint32_t>(row_index) * side_count + static_cast<uint32_t>(col_index)
        );
    }

    [[nodiscard]] BCPositionSideIndex row_index(CellId cid) const {
        if (cid >= cell_count()) {
            throw std::out_of_range("BC position cell id out of range");
        }
        return static_cast<BCPositionSideIndex>(cid / side_coord_count());
    }

    [[nodiscard]] BCPositionSideIndex col_index(CellId cid) const {
        if (cid >= cell_count()) {
            throw std::out_of_range("BC position cell id out of range");
        }
        return static_cast<BCPositionSideIndex>(cid % side_coord_count());
    }

    [[nodiscard]] bool equivalent_to(const BCPositionCellLayout &other) const {
        return layer_sum() == other.layer_sum() &&
            side_unit() == other.side_unit() &&
            side_coords() == other.side_coords() &&
            raw_coord_modulus_ == other.raw_coord_modulus_;
    }

    [[nodiscard]] bool equivalent_to_axis(const BCFamilyTable &axis) const {
        return layer_sum() == axis.layer_sum() &&
            side_unit() == axis.family_unit() &&
            side_coords() == axis.coords();
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return serialization_axis_.allocated_bytes() +
            static_cast<uint64_t>(raw_coord_to_physical_index_lut_.capacity()) *
                sizeof(BCPositionSideIndex);
    }

private:
    [[nodiscard]] static bool axis_is_dense_range(const BCFamilyTable &axis) {
        const std::vector<FamilyCoord> &coords = axis.coords();
        if (coords.empty()) {
            return false;
        }
        const uint32_t span =
            static_cast<uint32_t>(coords.back()) - static_cast<uint32_t>(coords.front()) + 1U;
        return span == coords.size();
    }

    void build_raw_coord_to_physical_index_lut() {
        raw_coord_to_physical_index_lut_.assign(
            static_cast<size_t>(serialization_axis_.total_coord() / 2U) + 1U,
            kInvalidSideIndex
        );
        for (uint32_t coord = 0U; coord < raw_coord_to_physical_index_lut_.size(); ++coord) {
            const FamilyCoord physical_coord =
                static_cast<FamilyCoord>(coord % raw_coord_modulus_);
            raw_coord_to_physical_index_lut_[coord] =
                serialization_axis_.try_coord_to_id(physical_coord);
        }
    }

    BCFamilyTable serialization_axis_;
    bool dense_side_coords_ = false;
    uint32_t raw_coord_modulus_ = 0U;
    std::vector<BCPositionSideIndex> raw_coord_to_physical_index_lut_;
};

[[nodiscard]] inline BCPositionCellLayout build_modulo_position_cell_layout_for_layer(
    LayerSum layer_sum,
    uint16_t side_unit,
    const std::vector<LayerSum> &possible_8tile_sums,
    uint32_t modulus
) {
    (void)possible_8tile_sums;
    BCFamilyTable physical_axis =
        build_family_partition_axis_for_layer(
            layer_sum,
            side_unit,
            std::vector<LayerSum>{0U},
            BCFamilyPartitionPolicy::modulo(modulus)
        );
    return BCPositionCellLayout::from_modulo_axis(
        std::move(physical_axis),
        modulus
    );
}

} // namespace BC
