#pragma once

#include "BCTypes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace BC {

class BCFamilyTable {
public:
    static constexpr FamilyId kInvalidFamilyId = std::numeric_limits<FamilyId>::max();

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

        const uint64_t total = layer_sum / family_unit;
        if (total > std::numeric_limits<FamilyCoord>::max()) {
            throw std::invalid_argument("BC total family coord exceeds FamilyCoord range");
        }

        for (size_t i = 0; i < axis_coords.size(); ++i) {
            if (i == 0U) {
                continue;
            }
            if (axis_coords[i] <= axis_coords[i - 1U]) {
                throw std::invalid_argument("BC family axis coords must be strictly ascending");
            }
        }

        layer_sum_ = layer_sum;
        family_unit_ = family_unit;
        total_coord_ = static_cast<FamilyCoord>(total);
        coords_ = axis_coords;
        axis_base_coord_ = coords_.front();
        family_count_ = static_cast<uint16_t>(coords_.size());
        coord_to_id_lut_.assign(
            static_cast<size_t>(coords_.back()) + 1U,
            kInvalidFamilyId
        );
        for (FamilyId id = 0; id < family_count_; ++id) {
            coord_to_id_lut_[coords_[id]] = id;
        }

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
        return coord < coord_to_id_lut_.size() &&
            coord_to_id_lut_[coord] != kInvalidFamilyId;
    }

    [[nodiscard]] FamilyId try_coord_to_id(FamilyCoord coord) const {
        return coord < coord_to_id_lut_.size()
            ? coord_to_id_lut_[coord]
            : kInvalidFamilyId;
    }

    [[nodiscard]] FamilyId coord_to_id_trusted(FamilyCoord coord) const noexcept {
        return coord_to_id_lut_[coord];
    }

    [[nodiscard]] FamilyId coord_to_id(FamilyCoord coord) const {
        if (coord >= coord_to_id_lut_.size()) {
            throw std::out_of_range(
                "BC family coord is outside this layer axis: " + std::to_string(coord)
            );
        }
        const FamilyId id = coord_to_id_lut_[coord];
        if (id == kInvalidFamilyId) {
            throw std::out_of_range(
                "BC family coord is outside this layer axis: " + std::to_string(coord)
            );
        }
        return id;
    }

    [[nodiscard]] FamilyCoord id_to_coord(FamilyId id) const {
        if (id >= family_count_) {
            throw std::out_of_range(
                "BC family id is outside this layer axis: " + std::to_string(id)
            );
        }
        return coords_[id];
    }

    [[nodiscard]] const std::vector<FamilyCoord> &coords() const {
        return coords_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return static_cast<uint64_t>(coords_.capacity()) * sizeof(FamilyCoord) +
            static_cast<uint64_t>(coord_to_id_lut_.capacity()) * sizeof(FamilyId);
    }

private:
    LayerSum layer_sum_ = 0U;
    uint16_t family_unit_ = 1U;
    FamilyCoord total_coord_ = 0U;
    FamilyCoord axis_base_coord_ = 0U;
    uint16_t family_count_ = 0U;
    std::vector<FamilyCoord> coords_;
    std::vector<FamilyId> coord_to_id_lut_;
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
    const uint64_t expected_target_total =
        static_cast<uint64_t>(source_axis.total_coord()) + static_cast<uint64_t>(delta_coord);
    if (static_cast<uint64_t>(target_axis.total_coord()) != expected_target_total) {
        throw std::invalid_argument(
            "BC target total_coord must equal source total_coord + delta_coord"
        );
    }

    const FamilyCoord a = source_axis.id_to_coord(source_id);
    const FamilyCoord n = source_axis.total_coord();
    if (static_cast<uint64_t>(a) > static_cast<uint64_t>(n)) {
        throw std::logic_error("BC source family coord exceeds source total coord");
    }
    const FamilyCoord b = static_cast<FamilyCoord>(static_cast<uint64_t>(n) - a);
    if (a > b) {
        throw std::logic_error("BC source family coord is not side-normalized");
    }

    const uint64_t a_plus_delta = static_cast<uint64_t>(a) + static_cast<uint64_t>(delta_coord);
    const FamilyCoord target0 = a;
    const FamilyCoord target1 = static_cast<FamilyCoord>(
        std::min<uint64_t>(a_plus_delta, static_cast<uint64_t>(b))
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

[[nodiscard]] inline std::vector<LayerSum> build_possible_8tile_sums(
    const std::vector<uint8_t> &legal_tiles,
    const std::array<uint32_t, 16U> &tile_sum_values
) {
    if (legal_tiles.empty()) {
        throw std::invalid_argument("BC possible sum builder requires non-empty legal tile alphabet");
    }
    std::unordered_set<LayerSum> current;
    current.insert(0U);
    for (uint32_t depth = 0U; depth < 8U; ++depth) {
        std::unordered_set<LayerSum> next;
        next.reserve(current.size() * legal_tiles.size());
        for (LayerSum base : current) {
            for (uint8_t tile : legal_tiles) {
                if (tile >= tile_sum_values.size()) {
                    throw std::out_of_range("BC legal tile exceeds tile sum table");
                }
                const LayerSum value = tile_sum_values[tile];
                if (base > std::numeric_limits<LayerSum>::max() - value) {
                    throw std::overflow_error("BC possible 8-tile sum overflow");
                }
                next.insert(base + value);
            }
        }
        current.swap(next);
    }
    std::vector<LayerSum> sums(current.begin(), current.end());
    std::sort(sums.begin(), sums.end());
    return sums;
}

[[nodiscard]] inline BCFamilyTable build_family_axis_for_layer(
    LayerSum layer_sum,
    uint16_t family_unit,
    const std::vector<LayerSum> &possible_8tile_sums
) {
    if (family_unit == 0U) {
        throw std::invalid_argument("BC family_unit must be non-zero");
    }
    if ((layer_sum % family_unit) != 0U) {
        throw std::invalid_argument("BC layer_sum must be divisible by family_unit");
    }
    if (possible_8tile_sums.empty()) {
        throw std::invalid_argument("BC theory axis requires non-empty possible 8-tile sums");
    }
    if (!std::is_sorted(possible_8tile_sums.begin(), possible_8tile_sums.end())) {
        throw std::invalid_argument("BC possible 8-tile sums must be sorted");
    }

    std::vector<FamilyCoord> coords;
    for (LayerSum h : possible_8tile_sums) {
        if (h > layer_sum) {
            break;
        }
        const LayerSum other = layer_sum - h;
        if (!std::binary_search(possible_8tile_sums.begin(), possible_8tile_sums.end(), other)) {
            continue;
        }
        const LayerSum min_side = std::min(h, other);
        if ((min_side % family_unit) != 0U) {
            continue;
        }
        const LayerSum coord64 = min_side / family_unit;
        if (coord64 > std::numeric_limits<FamilyCoord>::max()) {
            throw std::overflow_error("BC theory axis coord exceeds FamilyCoord");
        }
        coords.push_back(static_cast<FamilyCoord>(coord64));
    }
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    if (coords.empty()) {
        throw std::invalid_argument("BC theory axis is empty for layer sum");
    }
    return BCFamilyTable(layer_sum, family_unit, coords);
}

} // namespace BC
