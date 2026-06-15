#pragma once

#include "BCBoardOps.h"
#include "BCFamilyGenerationScheduler.h"
#include "BoardMover.h"
#include "CanonicalBatch.h"
#include "FormationRuntime.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace BC {

using BCQuadrantWordSumTable = std::vector<uint32_t>;

struct BCSolvePreparedQuery {
    uint64_t key = 0U;
    uint16_t cid = 0U;
    BucketRank rank = 0U;
    uint16_t ref = 0U;
    uint16_t reserved = 0U;
};

static_assert(sizeof(BCSolvePreparedQuery) == 16U, "BC solve prepared query should stay 16 bytes");

template <typename StorageT>
struct BCSolveLookupResult {
    bool found = false;
    StorageT value{};
};

struct BCSolveTargetFamilyFilter {
    bool enabled = false;
    FamilyIdList2 families;
};

struct BCSolveEdgeOptions {
    uint32_t canonical_batch_size = 8192U;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    uint8_t spawn2_tile_rank = 1U;
    uint8_t spawn4_tile_rank = 2U;
    double spawn_rate4 = 0.1;
    int success_target_rank = 0;
    const std::vector<uint8_t> *success_shifts = nullptr;
    bool success_check_all_cells = false;
    uint32_t future_cell_modulus = 0U;
};

struct BCSolveEdgeStats {
    uint64_t source_boards = 0U;
    uint64_t terminal_success_boards = 0U;
    uint64_t empty_slots = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_all_dir_calls = 0U;
    uint64_t selective_move_calls = 0U;
    uint64_t unchanged_moves = 0U;
    uint64_t prefilter_checks = 0U;
    uint64_t prefilter_skips = 0U;
    uint64_t canonicalized_candidates = 0U;
    uint64_t encoded_queries = 0U;
    uint64_t encode_rejects = 0U;
    uint64_t future_lookup_count = 0U;
    uint64_t future_lookup_misses = 0U;
    uint64_t finalized_boards = 0U;
    uint64_t batch_flushes = 0U;
    uint64_t batch_tail_flushes = 0U;
    uint64_t batch_source_boards = 0U;
    uint64_t canonical_flushes = 0U;
};

struct BCSolveBoardQuerySummary {
    bool terminal_success = false;
    uint16_t empty_mask = 0U;
    uint32_t empty_count = 0U;
};

enum class BCSolveSpawnPhase : uint8_t {
    Spawn2,
    Spawn4,
};

struct BCSolveCanonicalCandidate {
    uint64_t board = 0U;
    uint16_t ref = 0U;
    BCDirectionMask move_axis = BCDirectionMask::None;
};

template <typename StorageT>
struct BCSolveEdgeWorkspace {
    std::vector<BCSolveCanonicalCandidate> canonical2;
    std::vector<BCSolveCanonicalCandidate> canonical4;
    std::vector<BCSolvePreparedQuery> queries2;
    std::vector<BCSolvePreparedQuery> queries4;
    std::array<StorageT, kBCBoardCellCount> best2;
    std::array<StorageT, kBCBoardCellCount> best4;

    void clear_queries() {
        canonical2.clear();
        canonical4.clear();
        queries2.clear();
        queries4.clear();
    }
};

[[nodiscard]] inline uint32_t bc_solve_countr_zero32(uint32_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0UL;
    _BitScanForward(&index, value);
    return static_cast<uint32_t>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctz(value));
#else
    uint32_t index = 0U;
    while (((value >> index) & 1U) == 0U) {
        ++index;
    }
    return index;
#endif
}

[[nodiscard]] inline uint32_t bc_solve_pop_lowest_set_bit_index(uint32_t &mask) {
    const uint32_t index = bc_solve_countr_zero32(mask);
    mask &= mask - 1U;
    return index;
}

[[nodiscard]] inline bool bc_solve_success_check_enabled(const BCSolveEdgeOptions &options) {
    return options.success_target_rank > 0 &&
        (options.success_check_all_cells ||
         (options.success_shifts != nullptr && !options.success_shifts->empty()));
}

[[nodiscard]] inline bool bc_solve_is_success_board(
    uint64_t board,
    const BCSolveEdgeOptions &options
) {
    if (!bc_solve_success_check_enabled(options)) {
        return false;
    }
    if (options.success_check_all_cells) {
        const uint64_t target =
            static_cast<uint64_t>(options.success_target_rank) * 0x1111111111111111ULL;
        const uint64_t diff = board ^ target;
        constexpr uint64_t kMask7 = 0x7777777777777777ULL;
        return (~(((diff & kMask7) + kMask7) | diff | kMask7)) != 0ULL;
    }
    const uint64_t target = static_cast<uint64_t>(options.success_target_rank);
    for (uint8_t shift : *options.success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool bc_solve_filter_contains_family(
    const BCSolveTargetFamilyFilter &filter,
    FamilyId family
) {
    if (!filter.enabled) {
        return true;
    }
    for (FamilyId candidate : filter.families) {
        if (candidate == family) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool bc_solve_physical_target_family_may_hit(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t board,
    BCDirectionMask direction,
    const BCQuadrantWordSumTable *word_sums
) {
    if (!filter.enabled) {
        return true;
    }

    const BCQuadrantWords q = unpack_board_to_quadrants(board);
    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return true;
    }

    const bool use_word_sums = word_sums != nullptr && !word_sums->empty();
    const uint64_t nw_sum = use_word_sums ? (*word_sums)[q.nw] : lut.sum4_value(nw_desc.sum_id);
    const uint64_t ne_sum = use_word_sums ? (*word_sums)[q.ne] : lut.sum4_value(ne_desc.sum_id);
    const uint64_t sw_sum = use_word_sums ? (*word_sums)[q.sw] : lut.sum4_value(sw_desc.sum_id);
    const uint64_t se_sum = use_word_sums ? (*word_sums)[q.se] : lut.sum4_value(se_desc.sum_id);
    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return true;
    }

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!bc_min_side_coord_u64(nw_sum + ne_sum, sw_sum + se_sum, axis.family_unit(), row_coord) ||
        !bc_min_side_coord_u64(nw_sum + sw_sum, ne_sum + se_sum, axis.family_unit(), col_coord)) {
        return true;
    }
    if (!axis.contains_coord(row_coord) || !axis.contains_coord(col_coord)) {
        return true;
    }

    const FamilyId row_family = axis.coord_to_id(row_coord);
    const FamilyId col_family = axis.coord_to_id(col_coord);
    if (direction == BCDirectionMask::Horizontal) {
        return bc_solve_filter_contains_family(filter, row_family);
    }
    if (direction == BCDirectionMask::Vertical) {
        return bc_solve_filter_contains_family(filter, col_family);
    }
    return bc_solve_filter_contains_family(filter, row_family) ||
        bc_solve_filter_contains_family(filter, col_family);
}

struct BCSolvePreparedQueryEncoder {
    const BCLut *lut = nullptr;
    const BCFamilyTable *axis = nullptr;
    LayerSum layer_sum = 0U;
    uint16_t family_unit = 0U;
    uint32_t family_count = 0U;
    uint32_t axis_base = 0U;
    bool contiguous = false;
    uint32_t cell_modulus = 0U;

    BCSolvePreparedQueryEncoder() = default;

    BCSolvePreparedQueryEncoder(
        const BCLut &lut_in,
        const BCFamilyTable &axis_in,
        uint32_t cell_modulus_in = 0U
    )
        : lut(&lut_in),
          axis(&axis_in),
          layer_sum(axis_in.layer_sum()),
          family_unit(axis_in.family_unit()),
          family_count(axis_in.family_count()),
          axis_base(axis_in.axis_base_coord()),
          contiguous(axis_in.is_contiguous_range()),
          cell_modulus(cell_modulus_in) {
        if (cell_modulus != 0U && cell_modulus > std::numeric_limits<FamilyCoord>::max()) {
            throw std::invalid_argument("BC solve future cell modulus exceeds FamilyCoord");
        }
        if (static_cast<uint64_t>(family_count) * static_cast<uint64_t>(family_count) >
            static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1ULL) {
            throw std::invalid_argument("BC solve prepared query requires <= 65536 future cells");
        }
    }

    [[nodiscard]] bool encode(
        const BCQuadrantWords &q,
        uint16_t ref,
        uint8_t spawn_tile_rank,
        BCDirectionMask move_axis,
        BCSolvePreparedQuery &out
    ) const {
        (void)spawn_tile_rank;
        (void)move_axis;
        const BCWordDesc &nw_desc = lut->word_desc(q.nw);
        const BCWordDesc &ne_desc = lut->word_desc(q.ne);
        const BCWordDesc &sw_desc = lut->word_desc(q.sw);
        const BCWordDesc &se_desc = lut->word_desc(q.se);
        if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
            return false;
        }

        const uint64_t nw_sum = nw_desc.sum;
        const uint64_t ne_sum = ne_desc.sum;
        const uint64_t sw_sum = sw_desc.sum;
        const uint64_t se_sum = se_desc.sum;
        if (nw_sum + ne_sum + sw_sum + se_sum != layer_sum) {
            return false;
        }

        auto min_side_coord_fast = [this](uint64_t first_sum, uint64_t second_sum, FamilyCoord &coord_out) {
            const uint64_t min_sum = std::min(first_sum, second_sum);
            uint64_t coord = 0U;
            if (family_unit == 2U) {
                if ((min_sum & 1ULL) != 0ULL) {
                    return false;
                }
                coord = min_sum >> 1U;
            } else {
                if (family_unit == 0U || (min_sum % family_unit) != 0U) {
                    return false;
                }
                coord = min_sum / family_unit;
            }
            if (coord > std::numeric_limits<FamilyCoord>::max()) {
                return false;
            }
            coord_out = static_cast<FamilyCoord>(coord);
            return true;
        };

        FamilyCoord row_coord = 0U;
        FamilyCoord col_coord = 0U;
        if (!min_side_coord_fast(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
            !min_side_coord_fast(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
            return false;
        }

        auto coord_to_id = [this](FamilyCoord coord, FamilyId &id_out) {
            uint32_t coord_u32 = coord;
            if (cell_modulus != 0U) {
                coord_u32 %= cell_modulus;
            }
            if (contiguous) {
                if (coord_u32 < axis_base) {
                    return false;
                }
                const uint32_t offset = coord_u32 - axis_base;
                if (offset >= family_count) {
                    return false;
                }
                id_out = static_cast<FamilyId>(offset);
                return true;
            }
            if (coord_u32 > std::numeric_limits<FamilyCoord>::max()) {
                return false;
            }
            id_out = axis->try_coord_to_id(static_cast<FamilyCoord>(coord_u32));
            return id_out != BCFamilyTable::kInvalidFamilyId;
        };

        FamilyId row_id = BCFamilyTable::kInvalidFamilyId;
        FamilyId col_id = BCFamilyTable::kInvalidFamilyId;
        if (!coord_to_id(row_coord, row_id) || !coord_to_id(col_coord, col_id)) {
                return false;
        }

        const uint16_t count_ne = ne_desc.group_count;
        const uint16_t count_sw = sw_desc.group_count;
        const uint16_t count_se = se_desc.group_count;
        const uint32_t bitmap_len =
            static_cast<uint32_t>(count_ne) *
            static_cast<uint32_t>(count_sw) *
            static_cast<uint32_t>(count_se);
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen ||
            bitmap_len > std::numeric_limits<BucketBitmapLen>::max()) {
            throw std::logic_error("BC solve encoded bitmap length is outside uint16 bounds");
        }

        const uint32_t rank =
            (static_cast<uint32_t>(ne_desc.rank) * count_sw + sw_desc.rank) * count_se + se_desc.rank;
        if (rank >= bitmap_len || rank > std::numeric_limits<BucketRank>::max()) {
            throw std::logic_error("BC solve encoded rank is outside bitmap length");
        }

        const uint64_t cid =
            static_cast<uint64_t>(row_id) * family_count + static_cast<uint32_t>(col_id);

        out = BCSolvePreparedQuery{
            (static_cast<uint64_t>(q.nw) << 48U) |
                (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
                (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
                static_cast<uint64_t>(se_desc.packed_sum_mask),
            static_cast<uint16_t>(cid),
            static_cast<BucketRank>(rank),
            ref,
            0U
        };
        return true;
    }
};

[[nodiscard]] inline bool bc_solve_encode_prepared_query_hot(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q,
    uint16_t ref,
    uint8_t spawn_tile_rank,
    BCDirectionMask move_axis,
    BCSolvePreparedQuery &out,
    uint32_t cell_modulus = 0U
) {
    const BCSolvePreparedQueryEncoder encoder(lut, axis, cell_modulus);
    return encoder.encode(q, ref, spawn_tile_rank, move_axis, out);
}

template <typename StorageT>
void bc_solve_prepare_best_arrays(
    BCSolveEdgeWorkspace<StorageT> &workspace,
    StorageT zero_value
) {
    workspace.best2.fill(zero_value);
    workspace.best4.fill(zero_value);
}

template <typename StorageT>
void bc_solve_flush_canonical_candidates(
    const BCLut &lut,
    const BCFamilyTable &axis,
    uint8_t spawn_tile_rank,
    std::vector<BCSolveCanonicalCandidate> &canonical,
    std::vector<BCSolvePreparedQuery> &queries,
    const BCSolveEdgeOptions &options,
    const BCQuadrantWordSumTable *word_sums,
    BCSolveEdgeStats *stats
) {
    if (canonical.empty()) {
        return;
    }

    std::vector<uint64_t> boards;
    boards.reserve(canonical.size());
    for (const BCSolveCanonicalCandidate &candidate : canonical) {
        boards.push_back(candidate.board);
    }
    CanonicalBatch::canonicalize_inplace(boards.data(), boards.size(), options.canonical_symm_mode);

    if (stats != nullptr) {
        stats->canonicalized_candidates += boards.size();
    }
    for (size_t i = 0U; i < boards.size(); ++i) {
        (void)word_sums;
        BCSolvePreparedQuery query;
        const bool encoded = bc_solve_encode_prepared_query_hot(
            lut,
            axis,
            unpack_board_to_quadrants(boards[i]),
            canonical[i].ref,
            spawn_tile_rank,
            canonical[i].move_axis,
            query,
            options.future_cell_modulus
        );
        if (!encoded) {
            if (stats != nullptr) {
                ++stats->encode_rejects;
            }
            continue;
        }
        queries.push_back(query);
        if (stats != nullptr) {
            ++stats->encoded_queries;
        }
    }
    canonical.clear();
}

template <typename StorageT>
void bc_solve_push_moved_candidate(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t spawned,
    uint64_t moved,
    uint16_t ref,
    BCDirectionMask move_axis,
    std::vector<BCSolveCanonicalCandidate> &canonical,
    const BCSolveEdgeOptions &options,
    const BCQuadrantWordSumTable *word_sums,
    BCSolveEdgeStats *stats
) {
    if (moved == spawned) {
        if (stats != nullptr) {
            ++stats->unchanged_moves;
        }
        return;
    }
    if (filter.enabled) {
        if (stats != nullptr) {
            ++stats->prefilter_checks;
        }
        if (!bc_solve_physical_target_family_may_hit(lut, axis, filter, moved, move_axis, word_sums)) {
            if (stats != nullptr) {
                ++stats->prefilter_skips;
            }
            return;
        }
    }
    canonical.push_back(BCSolveCanonicalCandidate{
        moved,
        ref,
        move_axis
    });
    (void)options;
}

template <typename StorageT, typename Mover = BoardMover>
BCSolveBoardQuerySummary bc_solve_collect_board_queries(
    const BCLut &lut,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    uint64_t source_board,
    BCDirectionMask directions,
    const BCSolveTargetFamilyFilter &filter2,
    const BCSolveTargetFamilyFilter &filter4,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    const BCSolveEdgeOptions &options = {},
    const BCQuadrantWordSumTable *word_sums = nullptr,
    BCSolveEdgeStats *stats = nullptr
) {
    if (options.spawn2_tile_rank == 0U || options.spawn2_tile_rank > 15U ||
        options.spawn4_tile_rank == 0U || options.spawn4_tile_rank > 15U) {
        throw std::invalid_argument("BC solve edge spawn tile rank is out of range");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC solve edge canonical_batch_size must be non-zero");
    }
    workspace.clear_queries();
    if (stats != nullptr) {
        ++stats->source_boards;
    }

    BCSolveBoardQuerySummary summary;
    if (bc_solve_is_success_board(source_board, options)) {
        summary.terminal_success = true;
        if (stats != nullptr) {
            ++stats->terminal_success_boards;
        }
        return summary;
    }

    uint32_t empty_mask = bc_zero_cell_mask16(source_board);
    summary.empty_mask = static_cast<uint16_t>(empty_mask);
    while (empty_mask != 0U) {
        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
        const uint16_t ref = static_cast<uint16_t>(cell);
        ++summary.empty_count;
        if (stats != nullptr) {
            ++stats->empty_slots;
            stats->spawned_boards += 2U;
        }

        const uint64_t spawned2 =
            set_board_tile_unchecked(source_board, cell, options.spawn2_tile_rank);
        const uint64_t spawned4 =
            set_board_tile_unchecked(source_board, cell, options.spawn4_tile_rank);

        auto push_pair = [&](uint64_t spawned, uint8_t spawn_tile_rank,
                             const BCFamilyTable &axis,
                             const BCSolveTargetFamilyFilter &filter,
                             std::vector<BCSolveCanonicalCandidate> &canonical) {
            if (directions == BCDirectionMask::Both) {
                if (stats != nullptr) {
                    ++stats->move_all_dir_calls;
                }
                const auto moved = Mover::move_all_dir(spawned);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, axis, filter, spawned, std::get<0>(moved), ref,
                    BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, axis, filter, spawned, std::get<1>(moved), ref,
                    BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, axis, filter, spawned, std::get<2>(moved), ref,
                    BCDirectionMask::Vertical, canonical, options, word_sums, stats);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, axis, filter, spawned, std::get<3>(moved), ref,
                    BCDirectionMask::Vertical, canonical, options, word_sums, stats);
            } else {
                if (bc_has_horizontal(directions)) {
                    if (stats != nullptr) {
                        stats->selective_move_calls += 2U;
                    }
                    const auto moved = Mover::move_horizontal_pair(spawned);
                    bc_solve_push_moved_candidate<StorageT>(
                        lut, axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
                    bc_solve_push_moved_candidate<StorageT>(
                        lut, axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
                }
                if (bc_has_vertical(directions)) {
                    if (stats != nullptr) {
                        stats->selective_move_calls += 2U;
                    }
                    const auto moved = Mover::move_vertical_pair(spawned);
                    bc_solve_push_moved_candidate<StorageT>(
                        lut, axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Vertical, canonical, options, word_sums, stats);
                    bc_solve_push_moved_candidate<StorageT>(
                        lut, axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Vertical, canonical, options, word_sums, stats);
                }
            }
            if (canonical.size() >= options.canonical_batch_size) {
                auto &queries = spawn_tile_rank == options.spawn2_tile_rank
                    ? workspace.queries2
                    : workspace.queries4;
                bc_solve_flush_canonical_candidates<StorageT>(
                    lut, axis, spawn_tile_rank, canonical, queries, options, word_sums, stats);
            }
        };

        push_pair(spawned2, options.spawn2_tile_rank, future2_axis, filter2, workspace.canonical2);
        push_pair(spawned4, options.spawn4_tile_rank, future4_axis, filter4, workspace.canonical4);
    }

    bc_solve_flush_canonical_candidates<StorageT>(
        lut,
        future2_axis,
        options.spawn2_tile_rank,
        workspace.canonical2,
        workspace.queries2,
        options,
        word_sums,
        stats
    );
    bc_solve_flush_canonical_candidates<StorageT>(
        lut,
        future4_axis,
        options.spawn4_tile_rank,
        workspace.canonical4,
        workspace.queries4,
        options,
        word_sums,
        stats
    );
    return summary;
}

template <typename StorageT, typename Mover = BoardMover>
BCSolveBoardQuerySummary bc_solve_collect_board_phase_queries(
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    uint64_t source_board,
    BCDirectionMask directions,
    const BCSolveTargetFamilyFilter &filter,
    BCSolveSpawnPhase phase,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    const BCSolveEdgeOptions &options = {},
    const BCQuadrantWordSumTable *word_sums = nullptr,
    BCSolveEdgeStats *stats = nullptr
) {
    if (options.spawn2_tile_rank == 0U || options.spawn2_tile_rank > 15U ||
        options.spawn4_tile_rank == 0U || options.spawn4_tile_rank > 15U) {
        throw std::invalid_argument("BC solve edge spawn tile rank is out of range");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC solve edge canonical_batch_size must be non-zero");
    }
    workspace.clear_queries();
    if (stats != nullptr) {
        ++stats->source_boards;
    }

    BCSolveBoardQuerySummary summary;
    if (bc_solve_is_success_board(source_board, options)) {
        summary.terminal_success = true;
        if (stats != nullptr) {
            ++stats->terminal_success_boards;
        }
        return summary;
    }

    const uint8_t spawn_tile_rank = phase == BCSolveSpawnPhase::Spawn4
        ? options.spawn4_tile_rank
        : options.spawn2_tile_rank;
    std::vector<BCSolveCanonicalCandidate> &canonical =
        phase == BCSolveSpawnPhase::Spawn4 ? workspace.canonical4 : workspace.canonical2;
    std::vector<BCSolvePreparedQuery> &queries =
        phase == BCSolveSpawnPhase::Spawn4 ? workspace.queries4 : workspace.queries2;

    uint32_t empty_mask = bc_zero_cell_mask16(source_board);
    summary.empty_mask = static_cast<uint16_t>(empty_mask);
    while (empty_mask != 0U) {
        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
        const uint16_t ref = static_cast<uint16_t>(cell);
        ++summary.empty_count;
        if (stats != nullptr) {
            ++stats->empty_slots;
            ++stats->spawned_boards;
        }

        const uint64_t spawned =
            set_board_tile_unchecked(source_board, cell, spawn_tile_rank);
        if (directions == BCDirectionMask::Both) {
            if (stats != nullptr) {
                ++stats->move_all_dir_calls;
            }
            const auto moved = Mover::move_all_dir(spawned);
            bc_solve_push_moved_candidate<StorageT>(
                lut, future_axis, filter, spawned, std::get<0>(moved), ref,
                BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
            bc_solve_push_moved_candidate<StorageT>(
                lut, future_axis, filter, spawned, std::get<1>(moved), ref,
                BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
            bc_solve_push_moved_candidate<StorageT>(
                lut, future_axis, filter, spawned, std::get<2>(moved), ref,
                BCDirectionMask::Vertical, canonical, options, word_sums, stats);
            bc_solve_push_moved_candidate<StorageT>(
                lut, future_axis, filter, spawned, std::get<3>(moved), ref,
                BCDirectionMask::Vertical, canonical, options, word_sums, stats);
        } else {
            if (bc_has_horizontal(directions)) {
                if (stats != nullptr) {
                    stats->selective_move_calls += 2U;
                }
                const auto moved = Mover::move_horizontal_pair(spawned);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved.first, ref,
                    BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved.second, ref,
                    BCDirectionMask::Horizontal, canonical, options, word_sums, stats);
            }
            if (bc_has_vertical(directions)) {
                if (stats != nullptr) {
                    stats->selective_move_calls += 2U;
                }
                const auto moved = Mover::move_vertical_pair(spawned);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved.first, ref,
                    BCDirectionMask::Vertical, canonical, options, word_sums, stats);
                bc_solve_push_moved_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved.second, ref,
                    BCDirectionMask::Vertical, canonical, options, word_sums, stats);
            }
        }
        if (canonical.size() >= options.canonical_batch_size) {
            bc_solve_flush_canonical_candidates<StorageT>(
                lut, future_axis, spawn_tile_rank, canonical, queries, options, word_sums, stats);
        }
    }

    bc_solve_flush_canonical_candidates<StorageT>(
        lut,
        future_axis,
        spawn_tile_rank,
        canonical,
        queries,
        options,
        word_sums,
        stats
    );
    return summary;
}

template <typename StorageT>
[[nodiscard]] StorageT bc_solve_weighted_average(
    StorageT best2,
    StorageT best4,
    double spawn_rate4
) {
    const long double p4 = static_cast<long double>(spawn_rate4);
    const long double p2 = 1.0L - p4;
    const long double value =
        static_cast<long double>(best2) * p2 +
        static_cast<long double>(best4) * p4;
    return static_cast<StorageT>(value);
}

template <typename StorageT>
[[nodiscard]] StorageT bc_solve_divide_success_sum(long double sum, uint32_t count) {
    if (count == 0U) {
        return StorageT{};
    }
    return static_cast<StorageT>(sum / static_cast<long double>(count));
}

template <typename StorageT, typename LookupFn>
StorageT bc_solve_reduce_spawn_phase_partial(
    const BCSolveBoardQuerySummary &summary,
    const std::vector<BCSolvePreparedQuery> &queries,
    std::array<StorageT, kBCBoardCellCount> &best,
    LookupFn &&lookup,
    uint32_t lane,
    StorageT zero_value,
    long double spawn_weight,
    BCSolveEdgeStats *stats = nullptr
) {
    if (summary.terminal_success || summary.empty_count == 0U) {
        return zero_value;
    }

    best.fill(zero_value);
    for (const BCSolvePreparedQuery &query : queries) {
        if (query.ref >= best.size()) {
            continue;
        }
        if (stats != nullptr) {
            ++stats->future_lookup_count;
        }
        const BCSolveLookupResult<StorageT> result = lookup(query, lane);
        if (!result.found) {
            if (stats != nullptr) {
                ++stats->future_lookup_misses;
            }
            continue;
        }
        if (result.value > best[query.ref]) {
            best[query.ref] = result.value;
        }
    }

    long double sum = 0.0L;
    uint32_t empty_mask = summary.empty_mask;
    while (empty_mask != 0U) {
        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
        sum += static_cast<long double>(best[cell]);
    }
    return static_cast<StorageT>(
        (sum * spawn_weight) / static_cast<long double>(summary.empty_count)
    );
}

template <typename StorageT, typename LookupFn>
StorageT bc_solve_reduce_collected_queries(
    const BCSolveBoardQuerySummary &summary,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    LookupFn &&lookup,
    uint32_t lane,
    StorageT zero_value,
    StorageT terminal_value,
    const BCSolveEdgeOptions &options = {},
    BCSolveEdgeStats *stats = nullptr,
    bool count_finalized_board = true
) {
    if (summary.terminal_success) {
        if (stats != nullptr && count_finalized_board) {
            ++stats->finalized_boards;
        }
        return terminal_value;
    }
    if (summary.empty_count == 0U) {
        if (stats != nullptr && count_finalized_board) {
            ++stats->finalized_boards;
        }
        return zero_value;
    }

    bc_solve_prepare_best_arrays(workspace, zero_value);
    auto reduce = [&](const std::vector<BCSolvePreparedQuery> &queries,
                      std::array<StorageT, kBCBoardCellCount> &best) {
        for (const BCSolvePreparedQuery &query : queries) {
            if (query.ref >= best.size()) {
                continue;
            }
            if (stats != nullptr) {
                ++stats->future_lookup_count;
            }
            const BCSolveLookupResult<StorageT> result = lookup(query, lane);
            if (!result.found) {
                if (stats != nullptr) {
                    ++stats->future_lookup_misses;
                }
                continue;
            }
            if (result.value > best[query.ref]) {
                best[query.ref] = result.value;
            }
        }
    };
    reduce(workspace.queries2, workspace.best2);
    reduce(workspace.queries4, workspace.best4);

    long double sum2 = 0.0L;
    long double sum4 = 0.0L;
    uint32_t empty_mask = summary.empty_mask;
    while (empty_mask != 0U) {
        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
        sum2 += static_cast<long double>(workspace.best2[cell]);
        sum4 += static_cast<long double>(workspace.best4[cell]);
    }
    if (stats != nullptr && count_finalized_board) {
        ++stats->finalized_boards;
    }
    const long double inv_empty = 1.0L / static_cast<long double>(summary.empty_count);
    const long double p4 = static_cast<long double>(options.spawn_rate4);
    const long double p2 = 1.0L - p4;
    return static_cast<StorageT>(
        (sum2 * p2 + sum4 * p4) * inv_empty
    );
}

template <typename StorageT, typename LookupFn, typename Mover = BoardMover>
StorageT bc_solve_board_value(
    const BCLut &lut,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    uint64_t source_board,
    BCDirectionMask directions,
    const BCSolveTargetFamilyFilter &filter2,
    const BCSolveTargetFamilyFilter &filter4,
    LookupFn &&lookup,
    StorageT zero_value,
    StorageT terminal_value,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    const BCSolveEdgeOptions &options = {},
    const BCQuadrantWordSumTable *word_sums = nullptr,
    BCSolveEdgeStats *stats = nullptr
) {
    const BCSolveBoardQuerySummary summary =
        bc_solve_collect_board_queries<StorageT, Mover>(
            lut,
            future2_axis,
            future4_axis,
            source_board,
            directions,
            filter2,
            filter4,
            workspace,
            options,
            word_sums,
            stats
        );
    auto lane0_lookup = [&lookup](const BCSolvePreparedQuery &query, uint32_t) {
        return lookup(query);
    };
    return bc_solve_reduce_collected_queries<StorageT>(
        summary,
        workspace,
        lane0_lookup,
        0U,
        zero_value,
        terminal_value,
        options,
        stats,
        true
    );
}

template <typename StorageT, typename LookupFn, typename Mover = BoardMover>
StorageT bc_solve_board_value(
    const BCLut &lut,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    uint64_t source_board,
    LookupFn &&lookup,
    StorageT zero_value,
    StorageT terminal_value,
    BCSolveEdgeWorkspace<StorageT> &workspace,
    const BCSolveEdgeOptions &options = {},
    const BCQuadrantWordSumTable *word_sums = nullptr,
    BCSolveEdgeStats *stats = nullptr
) {
    return bc_solve_board_value<StorageT, LookupFn, Mover>(
        lut,
        future2_axis,
        future4_axis,
        source_board,
        BCDirectionMask::Both,
        BCSolveTargetFamilyFilter{},
        BCSolveTargetFamilyFilter{},
        std::forward<LookupFn>(lookup),
        zero_value,
        terminal_value,
        workspace,
        options,
        word_sums,
        stats
    );
}

} // namespace BC
