#pragma once

#include "BCCellMatrix.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace BC {

enum class BCDirectionMask : uint8_t {
    None = 0U,
    Horizontal = 1U,
    Vertical = 2U,
    Both = 3U,
};

[[nodiscard]] inline bool bc_has_horizontal(BCDirectionMask mask) {
    return (static_cast<uint8_t>(mask) & static_cast<uint8_t>(BCDirectionMask::Horizontal)) != 0U;
}

[[nodiscard]] inline bool bc_has_vertical(BCDirectionMask mask) {
    return (static_cast<uint8_t>(mask) & static_cast<uint8_t>(BCDirectionMask::Vertical)) != 0U;
}

struct BCFamilyGenerationPass {
    uint8_t spawn_tile_rank = 0U;
    SpawnDeltaCoord delta_coord = 0U;
    FamilyId source_id = 0U;
    FamilyCoord source_coord = 0U;
    FamilyIdList3 target_families;
};

struct BCSourceCellWork {
    CellId cid = 0U;
    BCDirectionMask directions = BCDirectionMask::None;
};

class BCFamilyGenerationScheduler {
public:
    BCFamilyGenerationScheduler(
        const BCFamilyTable &source_axis,
        const BCFamilyTable &target_axis
    ) : source_axis_(&source_axis),
        target_axis_(&target_axis),
        source_matrix_(source_axis),
        target_matrix_(target_axis),
        target_cell_set_(target_matrix_) {}

    [[nodiscard]] const BCFamilyTable &source_axis() const {
        return *source_axis_;
    }

    [[nodiscard]] const BCFamilyTable &target_axis() const {
        return *target_axis_;
    }

    [[nodiscard]] const BCCellMatrix &source_matrix() const {
        return source_matrix_;
    }

    [[nodiscard]] const BCCellMatrix &target_matrix() const {
        return target_matrix_;
    }

    [[nodiscard]] BCFamilyGenerationPass make_pass(
        FamilyId source_id,
        SpawnDeltaCoord delta_coord,
        uint8_t spawn_tile_rank
    ) const {
        if (spawn_tile_rank == 0U || spawn_tile_rank > 15U) {
            throw std::invalid_argument("BC family generation spawn tile rank is out of range");
        }
        FamilyIdList3 target_families;
        const FamilyIdList2 mapped =
            map_source_family_to_target_families(
                *source_axis_,
                *target_axis_,
                source_id,
                delta_coord
            );
        for (FamilyId family : mapped) {
            target_families.push_back(family);
        }
        return BCFamilyGenerationPass{
            spawn_tile_rank,
            delta_coord,
            source_id,
            source_axis_->id_to_coord(source_id),
            target_families
        };
    }

    [[nodiscard]] BCFamilyGenerationPass make_pass_existing_targets(
        FamilyId source_id,
        SpawnDeltaCoord delta_coord,
        uint8_t spawn_tile_rank
    ) const {
        if (spawn_tile_rank == 0U || spawn_tile_rank > 15U) {
            throw std::invalid_argument("BC family generation spawn tile rank is out of range");
        }
        const FamilyCoord source_coord = source_axis_->id_to_coord(source_id);
        const FamilyCoord n = source_axis_->total_coord();
        if (source_coord > n) {
            throw std::logic_error("BC source family coord exceeds source total coord");
        }
        const FamilyCoord b = static_cast<FamilyCoord>(
            static_cast<uint64_t>(n) - static_cast<uint64_t>(source_coord)
        );
        if (source_coord > b) {
            throw std::logic_error("BC source family coord is not side-normalized");
        }

        const FamilyCoord target0 = source_coord;
        const FamilyCoord target1 = static_cast<FamilyCoord>(std::min<uint64_t>(
            static_cast<uint64_t>(source_coord) + static_cast<uint64_t>(delta_coord),
            static_cast<uint64_t>(b)
        ));

        FamilyIdList3 target_families;
        auto add_if_present = [&](FamilyCoord coord) {
            if (!target_axis_->contains_coord(coord)) {
                return;
            }
            const FamilyId id = target_axis_->coord_to_id(coord);
            for (FamilyId existing : target_families) {
                if (existing == id) {
                    return;
                }
            }
            target_families.push_back(id);
        };
        add_if_present(target0);
        add_if_present(target1);

        return BCFamilyGenerationPass{
            spawn_tile_rank,
            delta_coord,
            source_id,
            source_coord,
            target_families
        };
    }

    [[nodiscard]] std::vector<BCSourceCellWork> source_cells(
        const BCFamilyGenerationPass &pass
    ) const {
        return source_cells_for_family(pass.source_id);
    }

    [[nodiscard]] std::vector<BCSourceCellWork> source_cells_for_family(
        FamilyId source_id
    ) const {
        const uint32_t f = source_matrix_.family_count();
        if (source_id >= f) {
            throw std::out_of_range("BC family generation source id out of range");
        }
        std::vector<BCSourceCellWork> out;
        out.reserve(static_cast<size_t>(2U) * f - 1U);
        for (uint32_t x = 0U; x < f; ++x) {
            const FamilyId id = static_cast<FamilyId>(x);
            out.push_back(BCSourceCellWork{
                source_matrix_.cid(source_id, id),
                id == source_id ? BCDirectionMask::Both : BCDirectionMask::Horizontal
            });
        }
        for (uint32_t x = 0U; x < f; ++x) {
            const FamilyId id = static_cast<FamilyId>(x);
            if (id == source_id) {
                continue;
            }
            out.push_back(BCSourceCellWork{
                source_matrix_.cid(id, source_id),
                BCDirectionMask::Vertical
            });
        }
        return out;
    }

    [[nodiscard]] std::vector<CellId> target_need_cells(
        const BCFamilyGenerationPass &pass
    ) {
        return target_need_cells_for_families(pass.target_families);
    }

    [[nodiscard]] std::vector<CellId> target_need_cells_for_source(
        FamilyId source_id,
        SpawnDeltaCoord delta_coord
    ) {
        return target_need_cells(make_pass(source_id, delta_coord, 1U));
    }

    [[nodiscard]] std::vector<CellId> target_need_cells_for_existing_source_targets(
        FamilyId source_id,
        SpawnDeltaCoord delta_coord
    ) {
        return target_need_cells(make_pass_existing_targets(source_id, delta_coord, 1U));
    }

    [[nodiscard]] std::vector<CellId> target_need_cells_for_families(
        const FamilyIdList3 &families
    ) {
        target_cell_set_.begin_epoch();
        target_cell_set_.add_family_crosses(families);
        return target_cell_set_.cells();
    }

    [[nodiscard]] std::vector<CellId> boundary_cells_for_advance(
        bool has_previous_boundary,
        FamilyCoord previous_boundary_coord,
        FamilyCoord current_boundary_coord
    ) const {
        std::vector<CellId> out;
        if (has_previous_boundary) {
            collect_boundary_cells(
                target_matrix_,
                previous_boundary_coord,
                current_boundary_coord,
                out
            );
            return out;
        }
        const BCFamilyTable &axis = *target_axis_;
        const uint32_t f = axis.family_count();
        for (uint32_t id_u32 = 0U; id_u32 < f; ++id_u32) {
            const FamilyId boundary_id = static_cast<FamilyId>(id_u32);
            const FamilyCoord coord = axis.id_to_coord(boundary_id);
            if (coord > current_boundary_coord) {
                break;
            }
            for (uint32_t x = 0U; x <= boundary_id; ++x) {
                out.push_back(target_matrix_.cid(boundary_id, static_cast<FamilyId>(x)));
            }
            for (uint32_t x = 0U; x < boundary_id; ++x) {
                out.push_back(target_matrix_.cid(static_cast<FamilyId>(x), boundary_id));
            }
        }
        return out;
    }

private:
    const BCFamilyTable *source_axis_ = nullptr;
    const BCFamilyTable *target_axis_ = nullptr;
    BCCellMatrix source_matrix_;
    BCCellMatrix target_matrix_;
    BCCellSetBuilder target_cell_set_;
};

} // namespace BC
