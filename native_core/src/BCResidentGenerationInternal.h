#pragma once

#include "BCFamilyGenerationScheduler.h"
#include "BCResidentGeneration.h"
#include "BoardMover.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace BC {
namespace ResidentGenerationInternal {

[[nodiscard]] double bc_now_seconds();
[[nodiscard]] int bc_omp_thread_num();
[[nodiscard]] int effective_thread_count(const BCResidentGenerationOptions &options);
[[nodiscard]] bool bc_success_check_enabled(
    const BCResidentGenerationOptions &options,
    LayerSum source_layer_sum
);

template <class Emit, class Mover = BoardMover>
inline void bc_generate_spawn_move_candidates(
    uint64_t board,
    uint16_t empty_mask16,
    uint8_t spawn_tile_rank,
    BCDirectionMask directions,
    Emit &&emit
) {
    uint32_t empty_mask = empty_mask16;
    while (empty_mask != 0U) {
#if defined(__GNUC__) || defined(__clang__)
        const uint32_t cell = static_cast<uint32_t>(__builtin_ctz(empty_mask));
#else
        uint32_t cell = 0U;
        uint32_t probe = empty_mask;
        while ((probe & 1U) == 0U) {
            probe >>= 1U;
            ++cell;
        }
#endif
        empty_mask &= empty_mask - 1U;
        const uint64_t spawned =
            board | (static_cast<uint64_t>(spawn_tile_rank) << (4U * cell));
        if (directions == BCDirectionMask::Both) {
            const auto moved = Mover::move_all_dir(spawned);
            emit(spawned, std::get<0>(moved));
            emit(spawned, std::get<1>(moved));
            emit(spawned, std::get<2>(moved));
            emit(spawned, std::get<3>(moved));
            continue;
        }
        if (bc_has_horizontal(directions)) {
            const auto moved = Mover::move_horizontal_pair(spawned);
            emit(spawned, moved.first);
            emit(spawned, moved.second);
        }
        if (bc_has_vertical(directions)) {
            const auto moved = Mover::move_vertical_pair(spawned);
            emit(spawned, moved.first);
            emit(spawned, moved.second);
        }
    }
}

struct BCThreadGenerationStats {
    double spawn_move_seconds = 0.0;
    double canonical_seconds = 0.0;
    double encode_insert_seconds = 0.0;
};

struct BCPendingEncodedCandidate {
    CellId cid = 0U;
    uint32_t home_slot = 0U;
    uint64_t key = 0U;
    BucketRank rank = 0U;
    BucketBitmapLen bitmap_len = 0U;
};

struct BCDynamicState {
    static constexpr uint32_t kEmptyCell = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kPendingCell = std::numeric_limits<uint32_t>::max() - 1U;

    uint32_t cell_count = 0U;
    uint32_t hash_capacity = 0U;
    uint64_t reserved_bitmap_words = 0U;
    std::unique_ptr<std::atomic<uint32_t>[]> cell_array;
    std::unique_ptr<uint64_t[]> key_array;
    std::unique_ptr<uint32_t[]> bitmap_offset_array;
    std::unique_ptr<std::atomic<uint64_t>[]> bitmap_arena;
    std::atomic<uint32_t> bitmap_cursor_words{0U};
    std::atomic<bool> overflowed{false};

    BCDynamicState() = default;
    BCDynamicState(const BCDynamicState &) = delete;
    BCDynamicState &operator=(const BCDynamicState &) = delete;
    BCDynamicState(BCDynamicState &&other) noexcept
        : cell_count(other.cell_count),
          hash_capacity(other.hash_capacity),
          reserved_bitmap_words(other.reserved_bitmap_words),
          cell_array(std::move(other.cell_array)),
          key_array(std::move(other.key_array)),
          bitmap_offset_array(std::move(other.bitmap_offset_array)),
          bitmap_arena(std::move(other.bitmap_arena)),
          bitmap_cursor_words(other.bitmap_cursor_words.load(std::memory_order_relaxed)),
          overflowed(other.overflowed.load(std::memory_order_relaxed)) {}
    BCDynamicState &operator=(BCDynamicState &&other) noexcept {
        if (this != &other) {
            cell_count = other.cell_count;
            hash_capacity = other.hash_capacity;
            reserved_bitmap_words = other.reserved_bitmap_words;
            cell_array = std::move(other.cell_array);
            key_array = std::move(other.key_array);
            bitmap_offset_array = std::move(other.bitmap_offset_array);
            bitmap_arena = std::move(other.bitmap_arena);
            bitmap_cursor_words.store(
                other.bitmap_cursor_words.load(std::memory_order_relaxed),
                std::memory_order_relaxed
            );
            overflowed.store(other.overflowed.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

struct BCDynamicThreadChunks {
    uint32_t word_next = 0U;
    uint32_t word_end = 0U;
};

struct BCDynamicResolved {
    uint32_t bitmap_offset = 0U;
    BucketRank rank = 0U;
};

struct BCThreadGenerationWorkspace {
    std::vector<uint64_t> canonical_buffer;
    std::vector<BCPendingEncodedCandidate> pending_encoded;
    std::vector<BCDynamicResolved> resolved_encoded;
    BCDynamicThreadChunks dynamic_chunks;
    BCThreadGenerationStats stats;
};

struct BCDynamicSlotRef {
    uint64_t key = 0U;
    uint32_t slot = 0U;
    // Packed as low 16 bits = bitmap_len, high 16 bits = live_count.
    // Both are bounded by the BC bucket maximum bitmap_len (<= 46656).
    uint32_t bitmap_len_live = 0U;
};

struct BCDynamicGroupedSlotRefs {
    std::vector<BCDynamicSlotRef> refs;
    std::vector<uint32_t> cell_begin;
};

using BCWordSumTable = std::vector<uint32_t>;

[[nodiscard]] BCWordSumTable build_word_sum_table(
    const std::array<uint32_t, 16U> *tile_sum_values
);
[[nodiscard]] bool tile_sum_values_match_lut(
    const BCLut &lut,
    const std::array<uint32_t, 16U> *tile_sum_values
);
[[nodiscard]] BCWordSumTable build_word_sum_table_if_needed(
    const BCLut &lut,
    const std::array<uint32_t, 16U> *tile_sum_values
);

[[nodiscard]] BCDynamicState make_bc_dynamic_state(
    uint32_t cell_count,
    uint64_t bucket_estimate,
    uint64_t bitmap_word_estimate,
    int thread_count = 1
);

void add_workspace_stats(
    BCResidentGenerationResult &result,
    const std::vector<BCThreadGenerationWorkspace> &workspaces
);

void set_dynamic_stats(
    BCResidentGenerationResult &result,
    const BCLut &lut,
    const BCDynamicState &state,
    uint32_t generation_retries
);

void set_dynamic_capacity_stats(
    BCResidentGenerationResult &result,
    const BCDynamicState &state,
    uint32_t generation_retries
);

void finalize_dynamic_result(
    BCResidentGenerationResult &result,
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCDynamicState &dynamic_state,
    const BCResidentGenerationOptions &options,
    int thread_count,
    double total_begin,
    double generation_seconds,
    BCWritableFile *output_file
);

} // namespace ResidentGenerationInternal
} // namespace BC
