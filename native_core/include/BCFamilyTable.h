#pragma once

#include "BCTypes.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

class BCFamilyTable {
public:
    BCFamilyTable() = default;

    BCFamilyTable(
        LayerSum layer_sum,
        uint16_t family_unit,
        const std::vector<FamilyCoord> &axis_coords
    ) {
        reset(layer_sum, family_unit, axis_coords);
    }

    static BCFamilyTable from_range(
        LayerSum layer_sum,
        uint16_t family_unit,
        FamilyCoord first_coord,
        FamilyCoord last_coord
    ) {
        if (last_coord < first_coord) {
            throw std::invalid_argument("BC family axis range is empty or reversed");
        }
        const uint32_t count =
            static_cast<uint32_t>(last_coord) - static_cast<uint32_t>(first_coord) + 1U;
        if (count > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument("BC family axis range exceeds uint16 count");
        }
        std::vector<FamilyCoord> coords;
        coords.reserve(count);
        for (uint32_t offset = 0; offset < count; ++offset) {
            coords.push_back(static_cast<FamilyCoord>(static_cast<uint32_t>(first_coord) + offset));
        }
        return BCFamilyTable(layer_sum, family_unit, coords);
    }

    void reset(
        LayerSum layer_sum,
        uint16_t family_unit,
        const std::vector<FamilyCoord> &axis_coords
    ) {
        if (family_unit == 0U) {
            throw std::invalid_argument("BC family_unit must be non-zero");
        }
        if ((layer_sum % family_unit) != 0U) {
            throw std::invalid_argument("BC layer_sum must be divisible by family_unit");
        }
        if (axis_coords.empty()) {
            throw std::invalid_argument("BC family axis must not be empty");
        }
        if (axis_coords.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument("BC family axis exceeds uint16 count");
        }

        const uint32_t total = layer_sum / family_unit;
        if (total > std::numeric_limits<FamilyCoord>::max()) {
            throw std::invalid_argument("BC total family coord exceeds FamilyCoord range");
        }

        for (size_t i = 0; i < axis_coords.size(); ++i) {
            if (static_cast<uint32_t>(axis_coords[i]) > total / 2U) {
                throw std::invalid_argument("BC family coord exceeds normalized half-sum bound");
            }
            if (i == 0U) {
                continue;
            }
            if (axis_coords[i] <= axis_coords[i - 1U]) {
                throw std::invalid_argument("BC family axis coords must be strictly ascending");
            }
            if (static_cast<uint32_t>(axis_coords[i]) !=
                static_cast<uint32_t>(axis_coords[i - 1U]) + 1U) {
                throw std::invalid_argument("BC family axis coords must be continuous");
            }
        }

        layer_sum_ = layer_sum;
        family_unit_ = family_unit;
        total_coord_ = static_cast<FamilyCoord>(total);
        axis_base_coord_ = axis_coords.front();
        family_count_ = static_cast<uint16_t>(axis_coords.size());

        for (FamilyId id = 0; id < family_count_; ++id) {
            const FamilyCoord coord = id_to_coord(id);
            if (coord_to_id(coord) != id) {
                throw std::logic_error("BC family coord/id roundtrip failed during construction");
            }
        }
    }

    [[nodiscard]] LayerSum layer_sum() const {
        return layer_sum_;
    }

    [[nodiscard]] FamilyCoord total_coord() const {
        return total_coord_;
    }

    [[nodiscard]] uint16_t family_unit() const {
        return family_unit_;
    }

    [[nodiscard]] FamilyCoord axis_base_coord() const {
        return axis_base_coord_;
    }

    [[nodiscard]] uint16_t family_count() const {
        return family_count_;
    }

    [[nodiscard]] bool contains_coord(FamilyCoord coord) const {
        if (family_count_ == 0U || coord < axis_base_coord_) {
            return false;
        }
        const uint32_t offset =
            static_cast<uint32_t>(coord) - static_cast<uint32_t>(axis_base_coord_);
        return offset < family_count_;
    }

    [[nodiscard]] FamilyId coord_to_id(FamilyCoord coord) const {
        if (!contains_coord(coord)) {
            throw std::out_of_range(
                "BC family coord is outside this layer axis: " + std::to_string(coord)
            );
        }
        return static_cast<FamilyId>(
            static_cast<uint32_t>(coord) - static_cast<uint32_t>(axis_base_coord_)
        );
    }

    [[nodiscard]] FamilyCoord id_to_coord(FamilyId id) const {
        if (id >= family_count_) {
            throw std::out_of_range(
                "BC family id is outside this layer axis: " + std::to_string(id)
            );
        }
        return static_cast<FamilyCoord>(static_cast<uint32_t>(axis_base_coord_) + id);
    }

private:
    LayerSum layer_sum_ = 0U;
    uint16_t family_unit_ = 1U;
    FamilyCoord total_coord_ = 0U;
    FamilyCoord axis_base_coord_ = 0U;
    uint16_t family_count_ = 0U;
};

inline FamilyIdList2 map_source_family_to_target_families(
    const BCFamilyTable &source_axis,
    const BCFamilyTable &target_axis,
    FamilyId source_id,
    SpawnDeltaCoord delta_coord
) {
    if (source_axis.family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC source/target family_unit mismatch");
    }
    const uint32_t expected_target_total =
        static_cast<uint32_t>(source_axis.total_coord()) + static_cast<uint32_t>(delta_coord);
    if (static_cast<uint32_t>(target_axis.total_coord()) != expected_target_total) {
        throw std::invalid_argument(
            "BC target total_coord must equal source total_coord + delta_coord"
        );
    }

    const FamilyCoord a = source_axis.id_to_coord(source_id);
    const FamilyCoord n = source_axis.total_coord();
    if (static_cast<uint32_t>(a) > static_cast<uint32_t>(n)) {
        throw std::logic_error("BC source family coord exceeds source total coord");
    }
    const FamilyCoord b = static_cast<FamilyCoord>(static_cast<uint32_t>(n) - a);
    if (a > b) {
        throw std::logic_error("BC source family coord is not side-normalized");
    }

    const uint32_t a_plus_delta = static_cast<uint32_t>(a) + static_cast<uint32_t>(delta_coord);
    const FamilyCoord target0 = a;
    const FamilyCoord target1 = static_cast<FamilyCoord>(
        std::min<uint32_t>(a_plus_delta, static_cast<uint32_t>(b))
    );

    FamilyIdList2 result;
    auto add_target = [&](FamilyCoord coord) {
        if (!target_axis.contains_coord(coord)) {
            throw std::out_of_range(
                "BC target family coord is outside target layer axis: " +
                std::to_string(coord)
            );
        }
        const FamilyId id = target_axis.coord_to_id(coord);
        for (FamilyId existing : result) {
            if (existing == id) {
                return;
            }
        }
        result.push_back(id);
    };

    add_target(target0);
    add_target(target1);
    return result;
}

} // namespace BC
