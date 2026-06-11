#pragma once

#include "BCFamilyTable.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

enum class BCFamilyPartitionKind {
    ExactCoord,
    ModuloCoord,
};

struct BCFamilyPartitionPolicy {
    BCFamilyPartitionKind kind = BCFamilyPartitionKind::ExactCoord;
    uint32_t modulus = 0U;

    [[nodiscard]] static BCFamilyPartitionPolicy exact() {
        return BCFamilyPartitionPolicy{BCFamilyPartitionKind::ExactCoord, 0U};
    }

    [[nodiscard]] static BCFamilyPartitionPolicy modulo(uint32_t value) {
        if (value == 0U || value > std::numeric_limits<FamilyId>::max()) {
            throw std::invalid_argument("BC modulo family partition requires 1..65535 modulus");
        }
        return BCFamilyPartitionPolicy{BCFamilyPartitionKind::ModuloCoord, value};
    }

    [[nodiscard]] bool is_exact() const {
        return kind == BCFamilyPartitionKind::ExactCoord;
    }

    [[nodiscard]] std::string name() const {
        if (is_exact()) {
            return "exact";
        }
        return "mod" + std::to_string(modulus);
    }
};

struct BCFamilyPartitionLayerMap {
    LayerSum layer_sum = 0U;
    uint16_t family_unit = 1U;
    FamilyCoord total_coord = 0U;
    BCFamilyPartitionPolicy policy = BCFamilyPartitionPolicy::exact();
    std::vector<std::vector<FamilyCoord>> exact_coords_by_family;
    std::vector<uint8_t> active_family;
    std::vector<FamilyId> coord_to_family_lut;

    [[nodiscard]] uint32_t family_count() const {
        return static_cast<uint32_t>(exact_coords_by_family.size());
    }

    [[nodiscard]] bool empty() const {
        return exact_coords_by_family.empty();
    }

    [[nodiscard]] FamilyId try_coord_to_family_id(FamilyCoord coord) const noexcept {
        return coord < coord_to_family_lut.size()
            ? coord_to_family_lut[coord]
            : BCFamilyTable::kInvalidFamilyId;
    }

    [[nodiscard]] FamilyId coord_to_family_id(FamilyCoord coord) const {
        const FamilyId id = try_coord_to_family_id(coord);
        if (id == BCFamilyTable::kInvalidFamilyId) {
            throw std::out_of_range("BC partition coord is outside this layer family map");
        }
        return id;
    }
};

[[nodiscard]] inline BCFamilyPartitionLayerMap build_family_partition_layer_map_from_axis(
    const BCFamilyTable &axis
) {
    BCFamilyPartitionLayerMap out;
    out.layer_sum = axis.layer_sum();
    out.family_unit = axis.family_unit();
    out.total_coord = axis.total_coord();
    out.policy = BCFamilyPartitionPolicy::exact();
    out.exact_coords_by_family.resize(axis.family_count());
    out.active_family.assign(axis.family_count(), 1U);
    out.coord_to_family_lut.assign(
        static_cast<size_t>(axis.total_coord() / 2U) + 1U,
        BCFamilyTable::kInvalidFamilyId
    );
    for (FamilyId id = 0U; id < axis.family_count(); ++id) {
        const FamilyCoord coord = axis.id_to_coord(id);
        out.exact_coords_by_family[id].push_back(coord);
        if (coord < out.coord_to_family_lut.size()) {
            out.coord_to_family_lut[coord] = id;
        }
    }
    return out;
}

[[nodiscard]] inline BCFamilyPartitionLayerMap build_family_partition_layer_map(
    const BCFamilyTable &axis,
    const std::vector<LayerSum> &possible_8tile_sums,
    const BCFamilyPartitionPolicy &policy
) {
    if (policy.is_exact()) {
        return build_family_partition_layer_map_from_axis(axis);
    }
    if (possible_8tile_sums.empty()) {
        throw std::invalid_argument("BC modulo family partition requires possible 8-tile sums");
    }
    if (!std::is_sorted(possible_8tile_sums.begin(), possible_8tile_sums.end())) {
        throw std::invalid_argument("BC possible 8-tile sums must be sorted");
    }
    if (axis.family_count() != policy.modulus) {
        throw std::invalid_argument("BC modulo axis family_count must equal partition modulus");
    }

    BCFamilyPartitionLayerMap out;
    out.layer_sum = axis.layer_sum();
    out.family_unit = axis.family_unit();
    out.total_coord = axis.total_coord();
    out.policy = policy;
    out.exact_coords_by_family.resize(policy.modulus);
    out.active_family.assign(policy.modulus, 0U);
    out.coord_to_family_lut.assign(
        static_cast<size_t>(axis.total_coord() / 2U) + 1U,
        BCFamilyTable::kInvalidFamilyId
    );
    for (uint32_t coord = 0U; coord < out.coord_to_family_lut.size(); ++coord) {
        out.coord_to_family_lut[coord] = static_cast<FamilyId>(coord % policy.modulus);
    }

    const BCFamilyTable exact_axis =
        build_family_axis_for_layer(axis.layer_sum(), axis.family_unit(), possible_8tile_sums);
    for (FamilyCoord coord : exact_axis.coords()) {
        const FamilyId id = static_cast<FamilyId>(coord % policy.modulus);
        out.exact_coords_by_family[id].push_back(coord);
        out.active_family[id] = 1U;
    }
    return out;
}

[[nodiscard]] inline BCFamilyTable build_family_partition_axis_for_layer(
    LayerSum layer_sum,
    uint16_t family_unit,
    const std::vector<LayerSum> &possible_8tile_sums,
    const BCFamilyPartitionPolicy &policy
) {
    if (policy.is_exact()) {
        return build_family_axis_for_layer(layer_sum, family_unit, possible_8tile_sums);
    }
    std::vector<FamilyCoord> coords;
    coords.reserve(policy.modulus);
    for (uint32_t id = 0U; id < policy.modulus; ++id) {
        coords.push_back(static_cast<FamilyCoord>(id));
    }
    return BCFamilyTable(layer_sum, family_unit, coords);
}

[[nodiscard]] inline std::vector<FamilyId> map_partition_source_family_to_target_families(
    const BCFamilyPartitionLayerMap &source,
    const BCFamilyTable &target_axis,
    const BCFamilyPartitionLayerMap &target,
    FamilyId source_id,
    SpawnDeltaCoord delta_coord
) {
    if (source.family_unit != target.family_unit) {
        throw std::invalid_argument("BC partition source/target family_unit mismatch");
    }
    const uint64_t expected_total =
        static_cast<uint64_t>(source.total_coord) + static_cast<uint64_t>(delta_coord);
    if (static_cast<uint64_t>(target.total_coord) != expected_total) {
        throw std::invalid_argument("BC partition target total must equal source total + delta");
    }
    if (source_id >= source.exact_coords_by_family.size()) {
        throw std::out_of_range("BC partition source family id out of range");
    }

    std::vector<FamilyId> out;
    const std::vector<FamilyCoord> &coords = source.exact_coords_by_family[source_id];
    for (FamilyCoord a : coords) {
        const FamilyCoord n = source.total_coord;
        if (a > n) {
            throw std::logic_error("BC partition source coord exceeds source total");
        }
        const FamilyCoord b = static_cast<FamilyCoord>(
            static_cast<uint64_t>(n) - static_cast<uint64_t>(a)
        );
        if (a > b) {
            throw std::logic_error("BC partition source coord is not side-normalized");
        }
        const FamilyCoord target_coords[2] = {
            a,
            static_cast<FamilyCoord>(std::min<uint64_t>(
                static_cast<uint64_t>(a) + static_cast<uint64_t>(delta_coord),
                static_cast<uint64_t>(b)
            ))
        };
        for (FamilyCoord coord : target_coords) {
            const FamilyId id = target.try_coord_to_family_id(coord);
            if (id == BCFamilyTable::kInvalidFamilyId) {
                continue;
            }
            if (std::find(out.begin(), out.end(), id) == out.end()) {
                out.push_back(id);
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

[[nodiscard]] inline FamilyIdList2 checked_partition_fanout2(
    const std::vector<FamilyId> &families
) {
    if (families.size() > 2U) {
        throw std::logic_error(
            "BC FamilyChain current backend supports at most two target fanout families"
        );
    }
    FamilyIdList2 out;
    for (FamilyId family : families) {
        out.push_back(family);
    }
    return out;
}

} // namespace BC
