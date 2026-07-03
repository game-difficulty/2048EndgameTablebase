#pragma once

#include "BCFamilyPartitionPolicy.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

struct BCExactFamilyCoordSet {
    LayerSum layer_sum = 0U;
    uint16_t family_unit = 1U;
    FamilyCoord total_coord = 0U;
    std::vector<FamilyCoord> coords;
    std::vector<uint64_t> ordered_side_pair_count;
};

struct BCFamilyPartitionLayer {
    LayerSum layer_sum = 0U;
    uint16_t family_unit = 1U;
    FamilyCoord total_coord = 0U;
    BCFamilyPartitionPolicy policy = BCFamilyPartitionPolicy::exact();
    std::vector<FamilyCoord> exact_coords;
    std::vector<uint64_t> exact_coord_weights;
    std::vector<FamilyId> exact_coord_family;
    std::vector<std::vector<FamilyCoord>> family_exact_coords;
    std::vector<uint64_t> family_weight;
    uint32_t active_family_count = 0U;
};

struct BCFamilyPartitionAnalysis {
    BCFamilyPartitionPolicy policy = BCFamilyPartitionPolicy::exact();
    LayerSum source_layer_sum = 0U;
    LayerSum target_layer_sum = 0U;
    uint16_t family_unit = 1U;
    SpawnDeltaCoord delta_coord = 0U;

    uint32_t source_family_count = 0U;
    uint32_t target_family_count = 0U;
    uint32_t source_active_family_count = 0U;
    uint32_t target_active_family_count = 0U;
    uint32_t source_exact_coord_count = 0U;
    uint32_t target_exact_coord_count = 0U;

    uint32_t max_exact_coords_per_source_family = 0U;
    uint32_t max_exact_coords_per_target_family = 0U;
    uint64_t max_source_family_weight = 0U;
    uint64_t max_target_family_weight = 0U;

    uint32_t max_target_fanout = 0U;
    double avg_target_fanout = 0.0;
    uint32_t source_families_with_fanout_gt2 = 0U;
    uint32_t source_families_with_missing_exact_target = 0U;
    uint64_t total_target_fanout_edges = 0U;

    uint64_t source_cells_per_pass = 0U;
    uint64_t max_target_need_cells_per_pass = 0U;
    uint64_t total_source_cell_visits = 0U;
    uint64_t total_target_need_cell_visits = 0U;
};

[[nodiscard]] inline BCExactFamilyCoordSet build_exact_family_coord_set_for_layer(
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
        throw std::invalid_argument("BC partition analysis requires non-empty possible sums");
    }
    if (!std::is_sorted(possible_8tile_sums.begin(), possible_8tile_sums.end())) {
        throw std::invalid_argument("BC possible sums must be sorted");
    }

    const LayerSum total_coord64 = layer_sum / family_unit;
    if (total_coord64 > std::numeric_limits<FamilyCoord>::max()) {
        throw std::overflow_error("BC exact coord total exceeds FamilyCoord");
    }

    std::vector<FamilyCoord> coords;
    std::vector<uint64_t> weights;
    for (LayerSum h : possible_8tile_sums) {
        if (h > layer_sum) {
            break;
        }
        const LayerSum other = layer_sum - h;
        if (!std::binary_search(possible_8tile_sums.begin(), possible_8tile_sums.end(), other)) {
            continue;
        }
        const LayerSum small = std::min(h, other);
        if ((small % family_unit) != 0U) {
            continue;
        }
        const LayerSum coord64 = small / family_unit;
        if (coord64 > std::numeric_limits<FamilyCoord>::max()) {
            throw std::overflow_error("BC exact family coord exceeds FamilyCoord");
        }
        coords.push_back(static_cast<FamilyCoord>(coord64));
        weights.push_back(1U);
    }

    std::vector<size_t> order(coords.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return coords[lhs] < coords[rhs];
    });

    BCExactFamilyCoordSet out;
    out.layer_sum = layer_sum;
    out.family_unit = family_unit;
    out.total_coord = static_cast<FamilyCoord>(total_coord64);
    for (size_t index : order) {
        const FamilyCoord coord = coords[index];
        if (!out.coords.empty() && out.coords.back() == coord) {
            out.ordered_side_pair_count.back() += weights[index];
        } else {
            out.coords.push_back(coord);
            out.ordered_side_pair_count.push_back(weights[index]);
        }
    }
    if (out.coords.empty()) {
        throw std::invalid_argument("BC exact family coord set is empty for layer");
    }
    return out;
}

[[nodiscard]] inline BCFamilyPartitionLayer build_family_partition_layer(
    const BCExactFamilyCoordSet &exact,
    const BCFamilyPartitionPolicy &policy
) {
    BCFamilyPartitionLayer out;
    out.layer_sum = exact.layer_sum;
    out.family_unit = exact.family_unit;
    out.total_coord = exact.total_coord;
    out.policy = policy;
    out.exact_coords = exact.coords;
    out.exact_coord_weights = exact.ordered_side_pair_count;

    uint32_t family_count = 0U;
    if (policy.kind == BCFamilyPartitionKind::ExactCoord) {
        if (exact.coords.size() > std::numeric_limits<FamilyId>::max()) {
            throw std::invalid_argument("BC exact partition exceeds FamilyId range");
        }
        family_count = static_cast<uint32_t>(exact.coords.size());
    } else {
        family_count = policy.modulus;
    }
    if (family_count == 0U || family_count > std::numeric_limits<FamilyId>::max()) {
        throw std::invalid_argument("BC partition family count exceeds FamilyId range");
    }

    out.exact_coord_family.resize(exact.coords.size(), BCFamilyTable::kInvalidFamilyId);
    out.family_exact_coords.resize(family_count);
    out.family_weight.assign(family_count, 0U);
    std::vector<uint8_t> seen(family_count, 0U);

    for (size_t i = 0; i < exact.coords.size(); ++i) {
        FamilyId id = 0U;
        if (policy.kind == BCFamilyPartitionKind::ExactCoord) {
            id = static_cast<FamilyId>(i);
        } else {
            id = static_cast<FamilyId>(exact.coords[i] % policy.modulus);
        }
        out.exact_coord_family[i] = id;
        out.family_exact_coords[id].push_back(exact.coords[i]);
        out.family_weight[id] += exact.ordered_side_pair_count[i];
        if (seen[id] == 0U) {
            seen[id] = 1U;
            ++out.active_family_count;
        }
    }
    return out;
}

[[nodiscard]] inline uint64_t bc_need_cells_for_fanout(uint32_t family_count, uint32_t fanout_count) {
    if (fanout_count > family_count) {
        throw std::logic_error("BC fanout count cannot exceed family count");
    }
    return 2ULL * static_cast<uint64_t>(fanout_count) * family_count -
        static_cast<uint64_t>(fanout_count) * fanout_count;
}

[[nodiscard]] inline BCFamilyPartitionAnalysis analyze_family_partition_transition(
    const BCFamilyPartitionLayer &source,
    const BCFamilyPartitionLayer &target,
    SpawnDeltaCoord delta_coord
) {
    if (source.family_unit != target.family_unit) {
        throw std::invalid_argument("BC partition analysis source/target family_unit mismatch");
    }
    if (target.total_coord != static_cast<FamilyCoord>(
        static_cast<uint64_t>(source.total_coord) + static_cast<uint64_t>(delta_coord)
    )) {
        throw std::invalid_argument("BC partition analysis target total != source total + delta");
    }
    if (source.policy.kind != target.policy.kind ||
        source.policy.modulus != target.policy.modulus) {
        throw std::invalid_argument("BC partition analysis source/target policy mismatch");
    }

    BCFamilyPartitionAnalysis out;
    out.policy = source.policy;
    out.source_layer_sum = source.layer_sum;
    out.target_layer_sum = target.layer_sum;
    out.family_unit = source.family_unit;
    out.delta_coord = delta_coord;
    out.source_family_count = static_cast<uint32_t>(source.family_exact_coords.size());
    out.target_family_count = static_cast<uint32_t>(target.family_exact_coords.size());
    out.source_active_family_count = source.active_family_count;
    out.target_active_family_count = target.active_family_count;
    out.source_exact_coord_count = static_cast<uint32_t>(source.exact_coords.size());
    out.target_exact_coord_count = static_cast<uint32_t>(target.exact_coords.size());
    out.source_cells_per_pass = 2ULL * out.source_family_count - 1ULL;

    const auto max_vector_size = [](const std::vector<std::vector<FamilyCoord>> &groups) {
        uint32_t max_value = 0U;
        for (const std::vector<FamilyCoord> &group : groups) {
            max_value = std::max<uint32_t>(max_value, static_cast<uint32_t>(group.size()));
        }
        return max_value;
    };
    const auto max_weight = [](const std::vector<uint64_t> &weights) {
        return weights.empty() ? 0ULL : *std::max_element(weights.begin(), weights.end());
    };
    out.max_exact_coords_per_source_family = max_vector_size(source.family_exact_coords);
    out.max_exact_coords_per_target_family = max_vector_size(target.family_exact_coords);
    out.max_source_family_weight = max_weight(source.family_weight);
    out.max_target_family_weight = max_weight(target.family_weight);

    std::vector<uint8_t> target_exact_coord_present;
    if (!target.exact_coords.empty()) {
        target_exact_coord_present.assign(static_cast<size_t>(target.exact_coords.back()) + 1U, 0U);
        for (FamilyCoord coord : target.exact_coords) {
            target_exact_coord_present[coord] = 1U;
        }
    }

    std::vector<uint32_t> target_coord_to_family(
        target_exact_coord_present.size(),
        std::numeric_limits<uint32_t>::max()
    );
    for (size_t i = 0; i < target.exact_coords.size(); ++i) {
        target_coord_to_family[target.exact_coords[i]] = target.exact_coord_family[i];
    }

    std::vector<uint32_t> fanout_marks(out.target_family_count, 0U);
    uint32_t epoch = 1U;
    for (uint32_t source_family = 0U; source_family < out.source_family_count; ++source_family) {
        const std::vector<FamilyCoord> &coords = source.family_exact_coords[source_family];
        if (coords.empty()) {
            continue;
        }
        uint32_t fanout = 0U;
        bool missing_exact_target = false;
        for (FamilyCoord a : coords) {
            const FamilyCoord n = source.total_coord;
            if (a > n) {
                throw std::logic_error("BC source exact coord exceeds total coord");
            }
            const FamilyCoord b = static_cast<FamilyCoord>(static_cast<uint64_t>(n) - a);
            if (a > b) {
                throw std::logic_error("BC source exact coord is not side-normalized");
            }
            const FamilyCoord targets[2] = {
                a,
                static_cast<FamilyCoord>(std::min<uint64_t>(
                    static_cast<uint64_t>(a) + delta_coord,
                    static_cast<uint64_t>(b)
                ))
            };
            for (FamilyCoord coord : targets) {
                uint32_t family = std::numeric_limits<uint32_t>::max();
                if (coord < target_coord_to_family.size() &&
                    target_coord_to_family[coord] != std::numeric_limits<uint32_t>::max()) {
                    family = target_coord_to_family[coord];
                } else {
                    missing_exact_target = true;
                    if (target.policy.kind == BCFamilyPartitionKind::ModuloCoord) {
                        family = coord % target.policy.modulus;
                    }
                }
                if (family == std::numeric_limits<uint32_t>::max()) {
                    continue;
                }
                if (fanout_marks[family] != epoch) {
                    fanout_marks[family] = epoch;
                    ++fanout;
                }
            }
        }
        ++epoch;
        if (epoch == 0U) {
            std::fill(fanout_marks.begin(), fanout_marks.end(), 0U);
            epoch = 1U;
        }

        out.max_target_fanout = std::max(out.max_target_fanout, fanout);
        out.total_target_fanout_edges += fanout;
        if (fanout > 2U) {
            ++out.source_families_with_fanout_gt2;
        }
        if (missing_exact_target) {
            ++out.source_families_with_missing_exact_target;
        }
        const uint64_t need_cells = bc_need_cells_for_fanout(out.target_family_count, fanout);
        out.max_target_need_cells_per_pass = std::max(out.max_target_need_cells_per_pass, need_cells);
        out.total_source_cell_visits += out.source_cells_per_pass;
        out.total_target_need_cell_visits += need_cells;
    }

    if (out.source_active_family_count != 0U) {
        out.avg_target_fanout =
            static_cast<double>(out.total_target_fanout_edges) /
            static_cast<double>(out.source_active_family_count);
    }
    return out;
}

[[nodiscard]] inline BCFamilyPartitionAnalysis analyze_family_partition_transition(
    LayerSum source_layer_sum,
    uint16_t family_unit,
    SpawnDeltaCoord delta_coord,
    const std::vector<LayerSum> &possible_8tile_sums,
    const BCFamilyPartitionPolicy &policy
) {
    const LayerSum target_layer_sum =
        source_layer_sum + static_cast<LayerSum>(delta_coord) * family_unit;
    const BCExactFamilyCoordSet source_exact =
        build_exact_family_coord_set_for_layer(source_layer_sum, family_unit, possible_8tile_sums);
    const BCExactFamilyCoordSet target_exact =
        build_exact_family_coord_set_for_layer(target_layer_sum, family_unit, possible_8tile_sums);
    return analyze_family_partition_transition(
        build_family_partition_layer(source_exact, policy),
        build_family_partition_layer(target_exact, policy),
        delta_coord
    );
}

} // namespace BC
