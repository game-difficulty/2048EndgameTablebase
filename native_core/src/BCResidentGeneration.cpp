#include "BCResidentGeneration.h"

#include "BCBoardOps.h"
#include "BCCellMatrix.h"
#include "BCPositionScanner.h"
#include "BoardMover.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {
namespace {

[[nodiscard]] double bc_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] int bc_omp_max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

[[nodiscard]] int bc_omp_thread_num() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

[[nodiscard]] int effective_thread_count(const BCResidentGenerationOptions &options) {
    const int requested = options.num_threads == 0 ? bc_omp_max_threads() : options.num_threads;
    if (requested <= 0) {
        throw std::invalid_argument("BC resident generation num_threads must be positive or zero");
    }
    return requested;
}

[[nodiscard]] bool bc_success_check_enabled(
    const BCResidentGenerationOptions &options,
    LayerSum source_layer_sum
) {
    return options.success_target_rank > 0 &&
           options.success_shifts != nullptr &&
           !options.success_shifts->empty() &&
           options.success_check_min_source_layer_sum != 0U &&
           source_layer_sum >= options.success_check_min_source_layer_sum;
}

[[nodiscard]] bool bc_is_success_by_shifts(
    uint64_t board,
    int target_rank,
    const std::vector<uint8_t> &success_shifts
) {
    if (success_shifts.empty()) {
        return false;
    }
    const uint64_t target = static_cast<uint64_t>(target_rank);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] uint32_t countr_zero32(uint32_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC countr_zero32 requires non-zero value");
    }
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctz(value));
#else
    uint32_t count = 0U;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++count;
    }
    return count;
#endif
}

[[nodiscard]] uint32_t countr_zero64(uint64_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC countr_zero64 requires non-zero value");
    }
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctzll(value));
#else
    uint32_t count = 0U;
    while ((value & 1ULL) == 0ULL) {
        value >>= 1U;
        ++count;
    }
    return count;
#endif
}

[[nodiscard]] uint32_t zero_cell_mask16(uint64_t board) {
    constexpr uint64_t kNibbleLsbMask = 0x1111111111111111ULL;
    const uint64_t nonzero_lsb =
        (board | (board >> 1U) | (board >> 2U) | (board >> 3U)) & kNibbleLsbMask;
    const uint64_t zero_lsb = (~nonzero_lsb) & kNibbleLsbMask;
#if defined(__BMI2__)
    return static_cast<uint32_t>(_pext_u64(zero_lsb, kNibbleLsbMask));
#else
    uint32_t mask = 0U;
    for (uint32_t cell = 0U; cell < kBCBoardCellCount; ++cell) {
        if (((zero_lsb >> (4U * cell)) & 1ULL) != 0ULL) {
            mask |= 1U << cell;
        }
    }
    return mask;
#endif
}

void validate_resident_generation_source(
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source
) {
    if (source.position == nullptr) {
        throw std::invalid_argument("BC resident generation source position is null");
    }
    if (source.spawn_tile_rank == 0U || source.spawn_tile_rank > 15U) {
        throw std::invalid_argument("BC resident generation spawn_tile_rank must be in 1..15");
    }

    const BCFamilyTable &source_axis = source.position->axis();
    if (source_axis.family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC resident generation source/target family_unit mismatch");
    }
    const uint32_t expected_target_total =
        static_cast<uint32_t>(source_axis.total_coord()) +
        static_cast<uint32_t>(source.delta_coord);
    if (static_cast<uint32_t>(target_axis.total_coord()) != expected_target_total) {
        throw std::invalid_argument(
            "BC resident generation target total_coord must equal source total_coord + delta_coord"
        );
    }
}

struct BCThreadGenerationStats {
    uint64_t source_boards_scanned = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t move_candidates = 0U;
    uint64_t moved_candidates = 0U;
    uint64_t encoded_candidates = 0U;
    uint64_t thread_local_duplicate_candidates = 0U;

    double spawn_move_seconds = 0.0;
    double canonical_seconds = 0.0;
    double encode_insert_seconds = 0.0;
};

struct BCPendingEncodedCandidate {
    CellId cid = 0U;
    BCEncodedKeyRank encoded;
    uint32_t word_count = 0U;
    uint32_t home_slot = 0U;
};

struct BCDynamicState {
    static constexpr uint32_t kEmptyCell = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kPendingCell = std::numeric_limits<uint32_t>::max() - 1U;

    uint32_t cell_count = 0U;
    uint32_t hash_capacity = 0U;
    uint64_t reserved_bitmap_words = 0U;
    std::unique_ptr<std::atomic<uint32_t>[]> cell_array;
    std::unique_ptr<uint64_t[]> key_array;
    std::unique_ptr<BucketBitmapLen[]> bitmap_len_array;
    std::unique_ptr<uint32_t[]> word_count_array;
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
          bitmap_len_array(std::move(other.bitmap_len_array)),
          word_count_array(std::move(other.word_count_array)),
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
            bitmap_len_array = std::move(other.bitmap_len_array);
            word_count_array = std::move(other.word_count_array);
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
    uint32_t word_count = 0U;
    BucketRank rank = 0U;
};

struct BCThreadGenerationWorkspace {
    std::vector<std::unique_ptr<BCCellBuilder>> builders;
    std::vector<uint64_t> canonical_buffer;
    std::vector<BCPendingEncodedCandidate> pending_encoded;
    std::vector<BCDynamicResolved> resolved_encoded;
    BCDynamicThreadChunks dynamic_chunks;
    BCThreadGenerationStats stats;
};

struct BCSourceBucketWorkItem {
    CellId cid = 0U;
    uint32_t bucket_index = 0U;
    uint32_t word_begin = 0U;
    uint32_t word_end = 0U;
};

struct BCDynamicSlotRef {
    CellId cid = 0U;
    uint64_t key = 0U;
    uint32_t slot = 0U;
};

using BCWordSumTable = std::vector<uint32_t>;

[[nodiscard]] std::vector<FinalizedCellPayload> finalize_dynamic_state(
    const BCDynamicState &state,
    int thread_count
);

[[nodiscard]] BCCellBuilder &ensure_cell_builder(
    std::vector<std::unique_ptr<BCCellBuilder>> &builders,
    CellId cid,
    const BCLut &lut
) {
    if (cid >= builders.size()) {
        throw std::logic_error("BC resident generation encoded cell id exceeds target matrix");
    }
    std::unique_ptr<BCCellBuilder> &builder = builders[static_cast<size_t>(cid)];
    if (!builder) {
        builder = std::make_unique<BCCellBuilder>(lut);
    }
    return *builder;
}

[[nodiscard]] BCWordSumTable build_word_sum_table(
    const std::array<uint32_t, 16U> *tile_sum_values
) {
    BCWordSumTable sums;
    if (tile_sum_values == nullptr) {
        return sums;
    }
    sums.resize(kBCQuadrantWordCount);
    for (uint32_t word = 0U; word < kBCQuadrantWordCount; ++word) {
        sums[word] =
            (*tile_sum_values)[word & 0xFU] +
            (*tile_sum_values)[(word >> 4U) & 0xFU] +
            (*tile_sum_values)[(word >> 8U) & 0xFU] +
            (*tile_sum_values)[(word >> 12U) & 0xFU];
    }
    return sums;
}

[[nodiscard]] BCBoardEncodedPosition encode_canonical_quadrants_position_hot(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCQuadrantWords &q,
    const BCWordSumTable *word_sums
) {
    BCBoardEncodedPosition out;

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.nw]
        : lut.sum4_value(nw_desc.sum_id);
    const uint64_t ne_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.ne]
        : lut.sum4_value(ne_desc.sum_id);
    const uint64_t sw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.sw]
        : lut.sum4_value(sw_desc.sum_id);
    const uint64_t se_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.se]
        : lut.sum4_value(se_desc.sum_id);

    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    FamilyCoord row_coord = 0U;
    FamilyCoord col_coord = 0U;
    if (!bc_min_side_coord_u64(nw_sum + ne_sum, sw_sum + se_sum, axis.family_unit(), row_coord) ||
        !bc_min_side_coord_u64(nw_sum + sw_sum, ne_sum + se_sum, axis.family_unit(), col_coord)) {
        return out;
    }
    if (!axis.contains_coord(row_coord) || !axis.contains_coord(col_coord)) {
        return out;
    }

    const FamilyId row_id = axis.coord_to_id(row_coord);
    const FamilyId col_id = axis.coord_to_id(col_coord);
    const uint32_t family_count = axis.family_count();
    const uint64_t cid64 =
        static_cast<uint64_t>(row_id) * static_cast<uint64_t>(family_count) + col_id;
    if (cid64 > std::numeric_limits<CellId>::max()) {
        throw std::overflow_error("BC hot encoded cell id exceeds CellId");
    }

    const BCEncodedKeyRank encoded =
        bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return out;
    }
    out.cid = static_cast<CellId>(cid64);
    out.row_family = row_id;
    out.col_family = col_id;
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

[[nodiscard]] std::vector<BCSourceBucketWorkItem> collect_source_bucket_work_items(
    const BCPositionLayerReader &position,
    const BCLut &lut
) {
    (void)lut;
    uint64_t item_count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        item_count += position.descriptor(cid).bucket_count;
    }
    if (item_count > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident generation source bucket work list exceeds uint32 items");
    }
    std::vector<BCSourceBucketWorkItem> items;
    items.reserve(static_cast<size_t>(item_count));
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
        for (uint32_t bucket = 0U; bucket < buckets.size; ++bucket) {
            items.push_back(BCSourceBucketWorkItem{
                cid,
                bucket,
                0U,
                0U
            });
        }
    }
    return items;
}

[[nodiscard]] uint64_t bc_mix_u64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

[[nodiscard]] uint32_t bc_next_power_of_two_u32(uint64_t value) {
    if (value > (1ULL << 31U)) {
        throw std::overflow_error("BC resident dynamic hash capacity exceeds uint32 power-of-two range");
    }
    uint32_t out = 1U;
    while (out < value) {
        out <<= 1U;
    }
    return std::max<uint32_t>(1024U, out);
}

[[nodiscard]] uint32_t choose_bc_dynamic_capacity(uint64_t bucket_estimate) {
    constexpr uint64_t kLoadNumerator = 40U;
    constexpr uint64_t kLoadDenominator = 100U;
    const uint64_t required =
        (std::max<uint64_t>(bucket_estimate, 1U) * kLoadDenominator + (kLoadNumerator - 1U)) /
        kLoadNumerator;
    return bc_next_power_of_two_u32(required);
}

[[nodiscard]] uint32_t bc_dynamic_home_slot(CellId cid, uint64_t key, uint32_t capacity) {
    const uint64_t mixed = bc_mix_u64(key ^ (static_cast<uint64_t>(cid) * 0x9e3779b97f4a7c15ULL));
    return static_cast<uint32_t>(mixed & (capacity - 1U));
}

[[nodiscard]] uint32_t source_bucket_count_sum(const std::vector<BCResidentGenerationSource> &sources) {
    uint64_t count = 0U;
    for (const BCResidentGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        for (CellId cid = 0U; cid < source.position->cell_count(); ++cid) {
            count += source.position->descriptor(cid).bucket_count;
        }
    }
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident generation source bucket count exceeds uint32");
    }
    return static_cast<uint32_t>(count);
}

[[nodiscard]] uint64_t position_bucket_count(const BCPositionLayerReader &position) {
    uint64_t count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        count += position.descriptor(cid).bucket_count;
    }
    return count;
}

[[nodiscard]] uint64_t source_rank_payload_word_estimate(const std::vector<BCResidentGenerationSource> &sources) {
    uint64_t bytes = 0U;
    for (const BCResidentGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        for (CellId cid = 0U; cid < source.position->cell_count(); ++cid) {
            bytes += source.position->descriptor(cid).rank_payload_bytes;
        }
    }
    return bytes / sizeof(uint64_t) + 1U;
}

[[nodiscard]] uint64_t position_rank_payload_word_estimate(const BCPositionLayerReader &position) {
    uint64_t bytes = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        bytes += position.descriptor(cid).rank_payload_bytes;
    }
    return bytes / sizeof(uint64_t) + 1U;
}

BCDynamicState make_bc_dynamic_state(
    uint32_t cell_count,
    uint64_t bucket_estimate,
    uint64_t bitmap_word_estimate
) {
    constexpr uint64_t kMinBitmapWords = 512U * 64U;
    BCDynamicState state;
    state.cell_count = cell_count;
    state.hash_capacity = choose_bc_dynamic_capacity(bucket_estimate);
    state.reserved_bitmap_words = std::max<uint64_t>(bitmap_word_estimate, kMinBitmapWords);
    if (state.reserved_bitmap_words > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident dynamic bitmap arena exceeds uint32 words");
    }
    state.cell_array = std::make_unique<std::atomic<uint32_t>[]>(state.hash_capacity);
    state.key_array = std::make_unique<uint64_t[]>(state.hash_capacity);
    state.bitmap_len_array = std::make_unique<BucketBitmapLen[]>(state.hash_capacity);
    state.word_count_array = std::make_unique<uint32_t[]>(state.hash_capacity);
    state.bitmap_offset_array = std::make_unique<uint32_t[]>(state.hash_capacity);
    for (uint32_t i = 0U; i < state.hash_capacity; ++i) {
        state.cell_array[i].store(BCDynamicState::kEmptyCell, std::memory_order_relaxed);
        state.key_array[i] = 0U;
        state.bitmap_len_array[i] = 0U;
        state.word_count_array[i] = 0U;
        state.bitmap_offset_array[i] = 0U;
    }
    state.bitmap_arena = std::make_unique<std::atomic<uint64_t>[]>(
        static_cast<size_t>(state.reserved_bitmap_words)
    );
    return state;
}

uint32_t bc_dynamic_acquire_words(
    BCDynamicState &state,
    BCDynamicThreadChunks &chunks,
    uint32_t words
) {
    constexpr uint32_t kChunkWords = 512U;
    if (chunks.word_next + words <= chunks.word_end) {
        const uint32_t out = chunks.word_next;
        chunks.word_next += words;
        return out;
    }
    const uint32_t chunk_words = std::max<uint32_t>(kChunkWords, words);
    const uint32_t begin = state.bitmap_cursor_words.fetch_add(chunk_words, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(begin) + chunk_words > state.reserved_bitmap_words) {
        state.overflowed.store(true, std::memory_order_release);
        return BCDynamicState::kPendingCell;
    }
    chunks.word_next = begin + words;
    chunks.word_end = begin + chunk_words;
    return begin;
}

void bc_dynamic_clear_words(BCDynamicState &state, uint32_t offset, uint32_t words) {
    for (uint32_t i = 0U; i < words; ++i) {
        state.bitmap_arena[offset + i].store(0ULL, std::memory_order_relaxed);
    }
}

uint32_t bc_dynamic_find_or_insert(
    BCDynamicState &state,
    const BCPendingEncodedCandidate &entry,
    BCDynamicThreadChunks &chunks
) {
    uint32_t slot = entry.home_slot;
    uint32_t probes = 0U;
    for (;;) {
        const uint32_t current = state.cell_array[slot].load(std::memory_order_acquire);
        if (current == entry.cid) {
            if (state.key_array[slot] == entry.encoded.key) {
                return state.bitmap_offset_array[slot];
            }
        } else if (current == BCDynamicState::kEmptyCell) {
            uint32_t expected = BCDynamicState::kEmptyCell;
            if (state.cell_array[slot].compare_exchange_strong(
                    expected,
                    BCDynamicState::kPendingCell,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                const uint32_t offset = bc_dynamic_acquire_words(state, chunks, entry.word_count);
                if (offset != BCDynamicState::kPendingCell) {
                    bc_dynamic_clear_words(state, offset, entry.word_count);
                    state.key_array[slot] = entry.encoded.key;
                    state.bitmap_len_array[slot] = entry.encoded.bitmap_len;
                    state.word_count_array[slot] = entry.word_count;
                    state.bitmap_offset_array[slot] = offset;
                    state.cell_array[slot].store(entry.cid, std::memory_order_release);
                }
                return offset;
            }
            continue;
        } else if (current == BCDynamicState::kPendingCell) {
            if (state.overflowed.load(std::memory_order_acquire)) {
                return BCDynamicState::kPendingCell;
            }
            continue;
        }

        slot = (slot + 1U) & (state.hash_capacity - 1U);
        ++probes;
        if (probes >= state.hash_capacity) {
            state.overflowed.store(true, std::memory_order_release);
            return BCDynamicState::kPendingCell;
        }
        __builtin_prefetch(&state.cell_array[slot], 0, 1);
    }
}

void flush_pending_encoded(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &state
) {
    if (workspace.pending_encoded.empty() || state.overflowed.load(std::memory_order_acquire)) {
        return;
    }

    workspace.resolved_encoded.resize(workspace.pending_encoded.size());
    constexpr uint32_t kPrefetchDistance = 16U;
    const uint32_t count = static_cast<uint32_t>(workspace.pending_encoded.size());
    const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
    for (uint32_t i = 0U; i < prefetch_count; ++i) {
        __builtin_prefetch(&state.cell_array[workspace.pending_encoded[i].home_slot], 0, 1);
    }
    for (uint32_t i = 0U; i < count; ++i) {
        if (i + kPrefetchDistance < count) {
            __builtin_prefetch(
                &state.cell_array[workspace.pending_encoded[i + kPrefetchDistance].home_slot],
                0,
                1
            );
        }
        const BCPendingEncodedCandidate &candidate = workspace.pending_encoded[i];
        BCDynamicResolved &resolved = workspace.resolved_encoded[i];
        resolved.bitmap_offset = bc_dynamic_find_or_insert(state, candidate, workspace.dynamic_chunks);
        if (resolved.bitmap_offset == BCDynamicState::kPendingCell) {
            return;
        }
        resolved.word_count = candidate.word_count;
        resolved.rank = candidate.encoded.rank;
        __builtin_prefetch(
            &state.bitmap_arena[resolved.bitmap_offset + (static_cast<uint32_t>(resolved.rank) >> 6U)],
            1,
            1
        );
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const BCDynamicResolved &resolved = workspace.resolved_encoded[i];
        const uint32_t word = static_cast<uint32_t>(resolved.rank) >> 6U;
        const uint64_t mask = 1ULL << (static_cast<uint32_t>(resolved.rank) & 63U);
        std::atomic<uint64_t> &target = state.bitmap_arena[resolved.bitmap_offset + word];
        if ((target.load(std::memory_order_relaxed) & mask) != 0ULL) {
            ++workspace.stats.thread_local_duplicate_candidates;
            continue;
        }
        const uint64_t old = target.fetch_or(mask, std::memory_order_relaxed);
        if ((old & mask) != 0ULL) {
            ++workspace.stats.thread_local_duplicate_candidates;
        }
    }

    workspace.pending_encoded.clear();
}

void flush_pending_encoded_timed(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &state,
    const BCResidentGenerationOptions &options
) {
    if (workspace.pending_encoded.empty()) {
        return;
    }
    const double begin = options.collect_timing ? bc_now_seconds() : 0.0;
    flush_pending_encoded(workspace, state);
    if (options.collect_timing) {
        workspace.stats.encode_insert_seconds += bc_now_seconds() - begin;
    }
}

void flush_canonical_buffer(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    BCDynamicState &dynamic_state,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums
) {
    if (workspace.canonical_buffer.empty()) {
        return;
    }

    const double canonical_begin = options.collect_timing ? bc_now_seconds() : 0.0;
    CanonicalBatch::canonicalize_inplace(
        workspace.canonical_buffer.data(),
        workspace.canonical_buffer.size(),
        options.canonical_symm_mode
    );
    if (options.collect_timing) {
        workspace.stats.canonical_seconds += bc_now_seconds() - canonical_begin;
    }

    const double encode_begin = options.collect_timing ? bc_now_seconds() : 0.0;
    for (uint64_t canonical_board : workspace.canonical_buffer) {
        const BCQuadrantWords q = unpack_board_to_quadrants(canonical_board);
        const BCBoardEncodedPosition encoded =
            encode_canonical_quadrants_position_hot(
                lut,
                target_axis,
                q,
                word_sums
            );
        if (!encoded.valid) {
            throw std::logic_error("BC resident generation produced candidate outside target axis");
        }
        BCEncodedKeyRank key_rank;
        key_rank.key = encoded.key;
        key_rank.rank = encoded.rank;
        key_rank.bitmap_len = encoded.bitmap_len;
        key_rank.count_ne = encoded.count_ne;
        key_rank.count_sw = encoded.count_sw;
        key_rank.count_se = encoded.count_se;
        key_rank.valid = true;

        workspace.pending_encoded.push_back(BCPendingEncodedCandidate{
            encoded.cid,
            key_rank,
            words_for_bits(key_rank.bitmap_len),
            bc_dynamic_home_slot(encoded.cid, key_rank.key, dynamic_state.hash_capacity)
        });
        ++workspace.stats.encoded_candidates;
        if (workspace.pending_encoded.size() >= options.pending_insert_buffer_size) {
            flush_pending_encoded(workspace, dynamic_state);
        }
    }
    if (options.collect_timing) {
        workspace.stats.encode_insert_seconds += bc_now_seconds() - encode_begin;
    }
    workspace.canonical_buffer.clear();
}

void push_pending_encoded_candidate(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &dynamic_state,
    CellId cid,
    const BCEncodedKeyRank &key_rank,
    const BCResidentGenerationOptions &options
) {
    workspace.pending_encoded.push_back(BCPendingEncodedCandidate{
        cid,
        key_rank,
        words_for_bits(key_rank.bitmap_len),
        bc_dynamic_home_slot(cid, key_rank.key, dynamic_state.hash_capacity)
    });
    ++workspace.stats.encoded_candidates;
    if (workspace.pending_encoded.size() >= options.pending_insert_buffer_size) {
        flush_pending_encoded(workspace, dynamic_state);
    }
}

void push_moved_candidate(
    BCThreadGenerationWorkspace &workspace,
    uint64_t spawned,
    uint64_t moved
) {
    ++workspace.stats.move_candidates;
    if (moved == spawned) {
        return;
    }
    ++workspace.stats.moved_candidates;
    workspace.canonical_buffer.push_back(moved);
}

void process_source_board_pair(
    BCThreadGenerationWorkspace &primary_workspace,
    BCThreadGenerationWorkspace *secondary_workspace,
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCFamilyTable *secondary_axis,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    uint64_t board,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    ++primary_workspace.stats.source_boards_scanned;
    if (skip_success_source &&
        options.success_shifts != nullptr &&
        bc_is_success_by_shifts(board, options.success_target_rank, *options.success_shifts)) {
        return;
    }
    uint32_t empty_mask = zero_cell_mask16(board);
    while (empty_mask != 0U) {
        const uint32_t cell = countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;

        const uint64_t spawn2 = board | (1ULL << (4U * cell));
        ++primary_workspace.stats.spawned_boards;
        const auto moved2 = BoardMover::move_all_dir(spawn2);
        push_moved_candidate(primary_workspace, spawn2, std::get<0>(moved2));
        push_moved_candidate(primary_workspace, spawn2, std::get<1>(moved2));
        push_moved_candidate(primary_workspace, spawn2, std::get<2>(moved2));
        push_moved_candidate(primary_workspace, spawn2, std::get<3>(moved2));

        if (secondary_workspace != nullptr && secondary_axis != nullptr && secondary_state != nullptr) {
            const uint64_t spawn4 = board | (2ULL << (4U * cell));
            ++secondary_workspace->stats.spawned_boards;
            const auto moved4 = BoardMover::move_all_dir(spawn4);
            push_moved_candidate(*secondary_workspace, spawn4, std::get<0>(moved4));
            push_moved_candidate(*secondary_workspace, spawn4, std::get<1>(moved4));
            push_moved_candidate(*secondary_workspace, spawn4, std::get<2>(moved4));
            push_moved_candidate(*secondary_workspace, spawn4, std::get<3>(moved4));
        }
    }
    if (primary_workspace.canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(
            primary_workspace,
            lut,
            primary_axis,
            primary_state,
            options,
            word_sums
        );
    }
    if (secondary_workspace != nullptr &&
        secondary_axis != nullptr &&
        secondary_state != nullptr &&
        secondary_workspace->canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(
            *secondary_workspace,
            lut,
            *secondary_axis,
            *secondary_state,
            options,
            word_sums
        );
    }
}

void process_source_board(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source,
    BCDynamicState &dynamic_state,
    uint64_t board,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    ++workspace.stats.source_boards_scanned;
    if (skip_success_source &&
        options.success_shifts != nullptr &&
        bc_is_success_by_shifts(board, options.success_target_rank, *options.success_shifts)) {
        return;
    }
    uint32_t empty_mask = zero_cell_mask16(board);
    while (empty_mask != 0U) {
        const uint32_t cell = countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;
        const uint64_t spawned =
            board | (static_cast<uint64_t>(source.spawn_tile_rank) << (4U * cell));
        ++workspace.stats.spawned_boards;

        const auto moved = BoardMover::move_all_dir(spawned);
        push_moved_candidate(workspace, spawned, std::get<0>(moved));
        push_moved_candidate(workspace, spawned, std::get<1>(moved));
        push_moved_candidate(workspace, spawned, std::get<2>(moved));
        push_moved_candidate(workspace, spawned, std::get<3>(moved));
    }
    if (workspace.canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(workspace, lut, target_axis, dynamic_state, options, word_sums);
    }
}

void process_source_bucket_pair(
    BCThreadGenerationWorkspace &primary_workspace,
    BCThreadGenerationWorkspace *secondary_workspace,
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCFamilyTable *secondary_axis,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    const BCPositionLayerReader &position,
    const BCSourceBucketWorkItem &item,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCPositionCellDescriptor &desc = position.descriptor(item.cid);
    if (item.bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation pair bucket work item is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(item.cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(item.cid);
    if (item.bucket_index >= buckets.size) {
        throw std::out_of_range("BC resident generation pair bucket view is shorter than descriptor");
    }

    const BCBucketEntry &bucket = buckets.data[item.bucket_index];
    const BCBucketRankDecoder decoder(lut, bucket.key);
    const uint32_t bitmap_len = decoder.bitmap_len;
    const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len
    );
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_end > payload.size) {
        throw std::out_of_range("BC resident generation pair bucket bitmap exceeds rank payload");
    }
    const uint32_t word_end = item.word_end == 0U ? bitmap_word_count : item.word_end;
    if (item.word_begin > word_end || word_end > bitmap_word_count) {
        throw std::out_of_range("BC resident generation pair bucket word range is invalid");
    }

    const uint8_t *bitmap_words = payload.data + bitmap_offset;
    for (uint32_t word_i = item.word_begin; word_i < word_end; ++word_i) {
        uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        while (word != 0U) {
            const uint32_t bit = countr_zero64(word);
            const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
            if (rank_u32 >= bitmap_len ||
                rank_u32 > std::numeric_limits<BucketRank>::max()) {
                throw std::logic_error("BC resident generation pair computed invalid source rank");
            }
            const BCQuadrantWords words = decoder.unrank(lut, static_cast<BucketRank>(rank_u32));
            process_source_board_pair(
                primary_workspace,
                secondary_workspace,
                lut,
                primary_axis,
                secondary_axis,
                primary_state,
                secondary_state,
                pack_quadrants_to_board(words),
                options,
                word_sums,
                skip_success_source
            );
            word &= word - 1U;
        }
    }
}

void insert_position_bucket_into_dynamic(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    BCDynamicState &dynamic_state,
    const BCPositionLayerReader &position,
    const BCSourceBucketWorkItem &item,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellDescriptor &desc = position.descriptor(item.cid);
    if (item.bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation carry bucket work item is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(item.cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(item.cid);
    if (item.bucket_index >= buckets.size) {
        throw std::out_of_range("BC resident generation carry bucket view is shorter than descriptor");
    }

    const BCBucketEntry &bucket = buckets.data[item.bucket_index];
    const BCBucketRankDecoder decoder(lut, bucket.key);
    const uint32_t bitmap_len = decoder.bitmap_len;
    const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len
    );
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_end > payload.size) {
        throw std::out_of_range("BC resident generation carry bucket bitmap exceeds rank payload");
    }
    const uint32_t word_end = item.word_end == 0U ? bitmap_word_count : item.word_end;
    if (item.word_begin > word_end || word_end > bitmap_word_count) {
        throw std::out_of_range("BC resident generation carry bucket word range is invalid");
    }

    const uint8_t *bitmap_words = payload.data + bitmap_offset;
    for (uint32_t word_i = item.word_begin; word_i < word_end; ++word_i) {
        uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        while (word != 0U) {
            const uint32_t bit = countr_zero64(word);
            const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
            if (rank_u32 >= bitmap_len ||
                rank_u32 > std::numeric_limits<BucketRank>::max()) {
                throw std::logic_error("BC resident generation carry computed invalid rank");
            }
            BCEncodedKeyRank key_rank;
            key_rank.key = bucket.key;
            key_rank.rank = static_cast<BucketRank>(rank_u32);
            key_rank.bitmap_len = static_cast<BucketBitmapLen>(bitmap_len);
            key_rank.count_ne = decoder.count_ne;
            key_rank.count_sw = decoder.count_sw;
            key_rank.count_se = decoder.count_se;
            key_rank.valid = true;
            push_pending_encoded_candidate(
                workspace,
                dynamic_state,
                item.cid,
                key_rank,
                options
            );
            word &= word - 1U;
        }
    }
}

void process_source_bucket(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source,
    BCDynamicState &dynamic_state,
    const BCSourceBucketWorkItem &item,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCPositionLayerReader &position = *source.position;
    const BCPositionCellDescriptor &desc = position.descriptor(item.cid);
    if (item.bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation bucket work item is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(item.cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(item.cid);
    if (item.bucket_index >= buckets.size) {
        throw std::out_of_range("BC resident generation bucket view is shorter than descriptor");
    }

    const BCBucketEntry &bucket = buckets.data[item.bucket_index];
    const BCBucketRankDecoder decoder(lut, bucket.key);
    const uint32_t bitmap_len = decoder.bitmap_len;
    const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len
    );
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_end > payload.size) {
        throw std::out_of_range("BC resident generation bucket bitmap exceeds rank payload");
    }
    const uint32_t word_end = item.word_end == 0U ? bitmap_word_count : item.word_end;
    if (item.word_begin > word_end || word_end > bitmap_word_count) {
        throw std::out_of_range("BC resident generation bucket word range is invalid");
    }

    const uint8_t *bitmap_words = payload.data + bitmap_offset;
    for (uint32_t word_i = item.word_begin; word_i < word_end; ++word_i) {
        uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        while (word != 0U) {
            const uint32_t bit = countr_zero64(word);
            const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
            if (rank_u32 >= bitmap_len ||
                rank_u32 > std::numeric_limits<BucketRank>::max()) {
                throw std::logic_error("BC resident generation computed invalid source rank");
            }
            const BCQuadrantWords words = decoder.unrank(lut, static_cast<BucketRank>(rank_u32));
            process_source_board(
                workspace,
                lut,
                target_axis,
                source,
                dynamic_state,
                pack_quadrants_to_board(words),
                options,
                word_sums,
                skip_success_source
            );
            word &= word - 1U;
        }
    }
}

void run_source_phase(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces,
    const BCWordSumTable *word_sums
) {
    validate_resident_generation_source(target_axis, source);
    const bool skip_success_source =
        bc_success_check_enabled(options, source.position->axis().layer_sum());
    const std::vector<BCSourceBucketWorkItem> work_items =
        collect_source_bucket_work_items(*source.position, lut);
    if (work_items.empty()) {
        return;
    }
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(thread_count)
    {
        try {
            const int tid = bc_omp_thread_num();
            BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
#pragma omp for schedule(dynamic, 16)
            for (int64_t item_i = 0; item_i < static_cast<int64_t>(work_items.size()); ++item_i) {
                const double bucket_begin = options.collect_timing ? bc_now_seconds() : 0.0;
                const double canonical_before = workspace.stats.canonical_seconds;
                const double encode_before = workspace.stats.encode_insert_seconds;
                process_source_bucket(
                    workspace,
                    lut,
                    target_axis,
                    source,
                    dynamic_state,
                    work_items[static_cast<size_t>(item_i)],
                    options,
                    word_sums,
                    skip_success_source
                );
                if (options.collect_timing) {
                    const double bucket_elapsed = bc_now_seconds() - bucket_begin;
                    const double nested_elapsed =
                        (workspace.stats.canonical_seconds - canonical_before) +
                        (workspace.stats.encode_insert_seconds - encode_before);
                    workspace.stats.spawn_move_seconds +=
                        std::max(0.0, bucket_elapsed - nested_elapsed);
                }
            }
            flush_canonical_buffer(workspace, lut, target_axis, dynamic_state, options, word_sums);
            flush_pending_encoded_timed(workspace, dynamic_state, options);
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

void run_insert_position_layer_phase(
    const BCLut &lut,
    const BCPositionLayerReader &position,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces
) {
    const std::vector<BCSourceBucketWorkItem> work_items =
        collect_source_bucket_work_items(position, lut);
    if (work_items.empty()) {
        return;
    }
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(thread_count)
    {
        try {
            const int tid = bc_omp_thread_num();
            BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
#pragma omp for schedule(dynamic, 16)
            for (int64_t item_i = 0; item_i < static_cast<int64_t>(work_items.size()); ++item_i) {
                const double begin = options.collect_timing ? bc_now_seconds() : 0.0;
                insert_position_bucket_into_dynamic(
                    workspace,
                    lut,
                    dynamic_state,
                    position,
                    work_items[static_cast<size_t>(item_i)],
                    options
                );
                if (options.collect_timing) {
                    workspace.stats.encode_insert_seconds += bc_now_seconds() - begin;
                }
            }
            flush_pending_encoded_timed(workspace, dynamic_state, options);
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

void run_source_pair_phase(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCFamilyTable *secondary_axis,
    const BCPositionLayerReader &current,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    std::vector<BCThreadGenerationWorkspace> &primary_workspaces,
    std::vector<BCThreadGenerationWorkspace> *secondary_workspaces,
    const BCWordSumTable *word_sums
) {
    const bool skip_success_source =
        bc_success_check_enabled(options, current.axis().layer_sum());
    const std::vector<BCSourceBucketWorkItem> work_items =
        collect_source_bucket_work_items(current, lut);
    if (work_items.empty()) {
        return;
    }
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(thread_count)
    {
        try {
            const int tid = bc_omp_thread_num();
            BCThreadGenerationWorkspace &primary_workspace =
                primary_workspaces[static_cast<size_t>(tid)];
            BCThreadGenerationWorkspace *secondary_workspace =
                secondary_workspaces != nullptr
                    ? &(*secondary_workspaces)[static_cast<size_t>(tid)]
                    : nullptr;
#pragma omp for schedule(dynamic, 16)
            for (int64_t item_i = 0; item_i < static_cast<int64_t>(work_items.size()); ++item_i) {
                const double bucket_begin = options.collect_timing ? bc_now_seconds() : 0.0;
                const double primary_canonical_before = primary_workspace.stats.canonical_seconds;
                const double primary_encode_before = primary_workspace.stats.encode_insert_seconds;
                const double secondary_canonical_before =
                    secondary_workspace != nullptr ? secondary_workspace->stats.canonical_seconds : 0.0;
                const double secondary_encode_before =
                    secondary_workspace != nullptr ? secondary_workspace->stats.encode_insert_seconds : 0.0;
                process_source_bucket_pair(
                    primary_workspace,
                    secondary_workspace,
                    lut,
                    primary_axis,
                    secondary_axis,
                    primary_state,
                    secondary_state,
                    current,
                    work_items[static_cast<size_t>(item_i)],
                    options,
                    word_sums,
                    skip_success_source
                );
                if (options.collect_timing) {
                    const double bucket_elapsed = bc_now_seconds() - bucket_begin;
                    const double primary_nested =
                        (primary_workspace.stats.canonical_seconds - primary_canonical_before) +
                        (primary_workspace.stats.encode_insert_seconds - primary_encode_before);
                    const double secondary_nested =
                        secondary_workspace != nullptr
                            ? (secondary_workspace->stats.canonical_seconds - secondary_canonical_before) +
                              (secondary_workspace->stats.encode_insert_seconds - secondary_encode_before)
                            : 0.0;
                    primary_workspace.stats.spawn_move_seconds +=
                        std::max(0.0, bucket_elapsed - primary_nested - secondary_nested);
                }
            }
            flush_canonical_buffer(
                primary_workspace,
                lut,
                primary_axis,
                primary_state,
                options,
                word_sums
            );
            flush_pending_encoded_timed(primary_workspace, primary_state, options);
            if (secondary_workspace != nullptr &&
                secondary_axis != nullptr &&
                secondary_state != nullptr) {
                flush_canonical_buffer(
                    *secondary_workspace,
                    lut,
                    *secondary_axis,
                    *secondary_state,
                    options,
                    word_sums
                );
                flush_pending_encoded_timed(*secondary_workspace, *secondary_state, options);
            }
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

void add_workspace_stats(
    BCResidentGenerationResult &result,
    const std::vector<BCThreadGenerationWorkspace> &workspaces
) {
    for (const BCThreadGenerationWorkspace &workspace : workspaces) {
        result.source_boards_scanned += workspace.stats.source_boards_scanned;
        result.spawned_boards += workspace.stats.spawned_boards;
        result.move_candidates += workspace.stats.move_candidates;
        result.moved_candidates += workspace.stats.moved_candidates;
        result.encoded_candidates += workspace.stats.encoded_candidates;
        result.thread_spawn_move_seconds += workspace.stats.spawn_move_seconds;
        result.thread_canonical_seconds += workspace.stats.canonical_seconds;
        result.thread_encode_insert_seconds += workspace.stats.encode_insert_seconds;
    }
    result.spawn_move_seconds = result.thread_spawn_move_seconds;
    result.canonical_seconds = result.thread_canonical_seconds;
    result.encode_insert_seconds = result.thread_encode_insert_seconds;
}

void finalize_dynamic_result(
    BCResidentGenerationResult &result,
    const BCFamilyTable &axis,
    const BCDynamicState &dynamic_state,
    const BCResidentGenerationOptions &options,
    int thread_count,
    double total_begin,
    double generation_seconds
) {
    result.generation_seconds = generation_seconds;
    result.merge_seconds = 0.0;

    const double finalize_begin = bc_now_seconds();
    std::vector<FinalizedCellPayload> payloads = finalize_dynamic_state(dynamic_state, thread_count);
    result.finalize_seconds = bc_now_seconds() - finalize_begin;

    for (const FinalizedCellPayload &payload : payloads) {
        result.output_success_rows += payload.success_rows;
    }
    result.duplicate_candidates =
        result.encoded_candidates >= result.output_success_rows
            ? result.encoded_candidates - result.output_success_rows
            : 0U;
    result.valid_candidates = result.encoded_candidates;
    result.duplicate_candidates_possible = result.duplicate_candidates;

    const double write_begin = bc_now_seconds();
    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < dynamic_state.cell_count; ++cid) {
        const FinalizedCellPayload &payload = payloads[static_cast<size_t>(cid)];
        if (payload.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, payload);
    }
    result.position_bytes = writer.finish_layer();
    result.write_seconds = bc_now_seconds() - write_begin;

    result.compute_seconds = generation_seconds + result.merge_seconds + result.finalize_seconds;
    (void)options;
    result.total_seconds = bc_now_seconds() - total_begin;
}

void bc_append_padding(std::vector<uint8_t> &buffer, uint32_t bytes) {
    buffer.insert(buffer.end(), bytes, 0U);
}

void bc_append_u16_le(std::vector<uint8_t> &buffer, uint16_t value) {
    buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void bc_append_u64_le(std::vector<uint8_t> &buffer, uint64_t value) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        buffer.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
    }
}

void bc_append_prefix256_le(
    std::vector<uint8_t> &buffer,
    const uint64_t *bitmap,
    uint32_t word_count,
    uint32_t bitmap_len
) {
    uint32_t running = 0U;
    const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
    for (uint32_t block = 0U; block < prefix_count; ++block) {
        if (running > std::numeric_limits<RankPrefix>::max()) {
            throw std::logic_error("BC dynamic prefix running popcount exceeds uint16");
        }
        bc_append_u16_le(buffer, static_cast<RankPrefix>(running));
        const uint32_t block_first_bit = block * kBCRankPrefixBits;
        const uint32_t block_last_bit = std::min<uint32_t>(
            bitmap_len,
            block_first_bit + kBCRankPrefixBits
        );
        const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
        const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
        if (last_word_exclusive > word_count) {
            throw std::logic_error("BC dynamic prefix bitmap is shorter than bitmap_len");
        }
        for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
            uint64_t value = bitmap[word];
            if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
            }
            running += popcount64(value);
        }
    }
}

void bc_append_bitmap_le(std::vector<uint8_t> &buffer, const uint64_t *bitmap, uint32_t word_count) {
    for (uint32_t i = 0U; i < word_count; ++i) {
        bc_append_u64_le(buffer, bitmap[i]);
    }
}

[[nodiscard]] uint32_t bc_dynamic_count_live(const uint64_t *bitmap, uint32_t word_count, uint32_t bitmap_len) {
    uint32_t live = 0U;
    for (uint32_t word = 0U; word < word_count; ++word) {
        uint64_t value = bitmap[word];
        if (word + 1U == word_count && (bitmap_len & 63U) != 0U) {
            value &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        live += popcount64(value);
    }
    return live;
}

[[nodiscard]] std::vector<BCDynamicSlotRef> collect_dynamic_slot_refs(const BCDynamicState &state) {
    std::vector<BCDynamicSlotRef> refs;
    refs.reserve(state.hash_capacity / 4U);
    for (uint32_t slot = 0U; slot < state.hash_capacity; ++slot) {
        const uint32_t cid = state.cell_array[slot].load(std::memory_order_acquire);
        if (cid == BCDynamicState::kEmptyCell) {
            continue;
        }
        if (cid == BCDynamicState::kPendingCell) {
            throw std::runtime_error("BC dynamic finalize saw pending cell slot");
        }
        refs.push_back(BCDynamicSlotRef{
            static_cast<CellId>(cid),
            state.key_array[slot],
            slot
        });
    }
    std::sort(
        refs.begin(),
        refs.end(),
        [](const BCDynamicSlotRef &lhs, const BCDynamicSlotRef &rhs) {
            if (lhs.cid != rhs.cid) {
                return lhs.cid < rhs.cid;
            }
            return lhs.key < rhs.key;
        }
    );
    return refs;
}

[[nodiscard]] std::vector<FinalizedCellPayload> finalize_dynamic_state(
    const BCDynamicState &state,
    int thread_count
) {
    const std::vector<BCDynamicSlotRef> refs = collect_dynamic_slot_refs(state);
    std::vector<FinalizedCellPayload> payloads(state.cell_count);
    std::vector<uint32_t> cell_begin(state.cell_count + 1U, 0U);
    uint32_t cursor = 0U;
    for (uint32_t cid = 0U; cid < state.cell_count; ++cid) {
        cell_begin[cid] = cursor;
        while (cursor < refs.size() && refs[cursor].cid == cid) {
            ++cursor;
        }
    }
    cell_begin[state.cell_count] = cursor;

    std::exception_ptr first_exception;
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(state.cell_count); ++cid_i) {
        try {
            const uint32_t cid = static_cast<uint32_t>(cid_i);
            const uint32_t begin = cell_begin[cid];
            const uint32_t end = cell_begin[cid + 1U];
            if (begin == end) {
                continue;
            }

            FinalizedCellPayload payload;
            payload.buckets.reserve(end - begin);
            uint64_t success_cursor = 0U;
            std::vector<uint64_t> bitmap;
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t slot = refs[i].slot;
                const uint32_t bitmap_len = state.bitmap_len_array[slot];
                const uint32_t word_count = state.word_count_array[slot];
                const uint32_t bitmap_offset = state.bitmap_offset_array[slot];
                if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
                    throw std::out_of_range("BC dynamic finalize bitmap offset exceeds arena");
                }

                bitmap.resize(word_count);
                for (uint32_t word = 0U; word < word_count; ++word) {
                    bitmap[word] = state.bitmap_arena[bitmap_offset + word].load(std::memory_order_relaxed);
                }
                const uint32_t live = bc_dynamic_count_live(bitmap.data(), word_count, bitmap_len);
                if (live == 0U) {
                    continue;
                }

                const uint32_t payload_offset =
                    checked_u32_size(payload.rank_payload.size(), "BC dynamic rank payload offset exceeds uint32");
                const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
                bc_append_padding(payload.rank_payload, aligned_payload_offset - payload_offset);
                const uint32_t rank_payload_offset =
                    checked_u32_size(payload.rank_payload.size(), "BC dynamic aligned rank payload offset exceeds uint32");

                bc_append_prefix256_le(payload.rank_payload, bitmap.data(), word_count, bitmap_len);
                const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
                bc_append_padding(
                    payload.rank_payload,
                    out_bitmap_offset - checked_u32_size(
                        payload.rank_payload.size(),
                        "BC dynamic prefix payload end exceeds uint32"
                    )
                );
                bc_append_bitmap_le(payload.rank_payload, bitmap.data(), word_count);
                payload.buckets.push_back(BCBucketEntry{
                    refs[i].key,
                    rank_payload_offset,
                    checked_u32_size(success_cursor, "BC dynamic success row offset exceeds uint32")
                });
                success_cursor += live;
                if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("BC dynamic finalized cell success_rows exceeds uint32");
                }
            }
            payload.success_rows = static_cast<uint32_t>(success_cursor);
            payloads[cid] = std::move(payload);
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return payloads;
}

[[nodiscard]] std::vector<std::unique_ptr<BCCellBuilder>> merge_thread_local_builders(
    const BCLut &lut,
    uint32_t target_cell_count,
    int thread_count,
    std::vector<BCThreadGenerationWorkspace> &workspaces
) {
    std::vector<std::unique_ptr<BCCellBuilder>> merged(target_cell_count);
    std::exception_ptr first_exception;

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(target_cell_count); ++cid_i) {
        try {
            std::unique_ptr<BCCellBuilder> cell_builder;
            size_t total_buckets = 0U;
            size_t total_bitmap_words = 0U;
            for (const BCThreadGenerationWorkspace &workspace : workspaces) {
                const std::unique_ptr<BCCellBuilder> &local =
                    workspace.builders[static_cast<size_t>(cid_i)];
                if (!local) {
                    continue;
                }
                if (total_buckets > std::numeric_limits<size_t>::max() - local->bucket_count()) {
                    throw std::overflow_error("BC resident generation merge bucket reserve overflow");
                }
                total_buckets += local->bucket_count();
                if (total_bitmap_words > std::numeric_limits<size_t>::max() - local->live_bitmap_words()) {
                    throw std::overflow_error("BC resident generation merge bitmap reserve overflow");
                }
                total_bitmap_words += local->live_bitmap_words();
            }
            if (total_buckets != 0U) {
                cell_builder = std::make_unique<BCCellBuilder>(lut);
                cell_builder->reserve_buckets_and_bitmap_words(total_buckets, total_bitmap_words);
            }
            for (BCThreadGenerationWorkspace &workspace : workspaces) {
                const std::unique_ptr<BCCellBuilder> &local =
                    workspace.builders[static_cast<size_t>(cid_i)];
                if (!local) {
                    continue;
                }
                if (!cell_builder) {
                    continue;
                }
                (void)cell_builder->merge_from(*local);
            }
            merged[static_cast<size_t>(cid_i)] = std::move(cell_builder);
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return merged;
}

[[nodiscard]] std::vector<FinalizedCellPayload> finalize_cells(
    const std::vector<std::unique_ptr<BCCellBuilder>> &builders,
    const BCResidentGenerationOptions &options,
    int thread_count
) {
    std::vector<FinalizedCellPayload> payloads(builders.size());
    std::exception_ptr first_exception;

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(builders.size()); ++cid_i) {
        try {
            const std::unique_ptr<BCCellBuilder> &builder = builders[static_cast<size_t>(cid_i)];
            if (builder) {
                payloads[static_cast<size_t>(cid_i)] = builder->finalize(options.finalize_options);
            }
        } catch (...) {
#pragma omp critical(BCResidentGenerationException)
            {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return payloads;
}

} // namespace

BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    if (sources.empty()) {
        throw std::invalid_argument("BC resident generation requires at least one source");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident generation pending_insert_buffer_size must be non-zero");
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCCellMatrix target_matrix(target_axis);
    const uint32_t target_cell_count = target_matrix.cell_count();
    const BCWordSumTable word_sums_storage = build_word_sum_table(options.family_tile_sum_values);
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;

    BCResidentGenerationResult result;
    result.effective_threads = thread_count;
    const double generation_begin = bc_now_seconds();
    std::vector<BCThreadGenerationWorkspace> workspaces;
    BCDynamicState dynamic_state;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = 1.0;
    bool generated = false;
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        result = BCResidentGenerationResult{};
        result.effective_threads = thread_count;
        workspaces.clear();
        workspaces.resize(static_cast<size_t>(thread_count));
        for (BCThreadGenerationWorkspace &workspace : workspaces) {
            workspace.canonical_buffer.reserve(options.canonical_batch_size);
            workspace.pending_encoded.reserve(options.pending_insert_buffer_size);
            workspace.resolved_encoded.reserve(options.pending_insert_buffer_size);
        }

        const uint64_t source_buckets = source_bucket_count_sum(sources);
        const uint64_t source_bitmap_words = source_rank_payload_word_estimate(sources);
        dynamic_state = make_bc_dynamic_state(
            target_cell_count,
            static_cast<uint64_t>(static_cast<double>(source_buckets) * reserve_factor) + 4096ULL,
            static_cast<uint64_t>(static_cast<double>(source_bitmap_words) * reserve_factor) + 512ULL * 64ULL
        );

        for (const BCResidentGenerationSource &source : sources) {
            const double phase_begin = bc_now_seconds();
            run_source_phase(
                lut,
                target_axis,
                source,
                options,
                thread_count,
                dynamic_state,
                workspaces,
                word_sums
            );
            result.scan_seconds += bc_now_seconds() - phase_begin;
        }
        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            generated = true;
            break;
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC resident generation dynamic state exceeded retry limit");
    }
    const double generation_seconds = bc_now_seconds() - generation_begin;
    result.generation_seconds = generation_seconds;

    for (const BCThreadGenerationWorkspace &workspace : workspaces) {
        result.source_boards_scanned += workspace.stats.source_boards_scanned;
        result.spawned_boards += workspace.stats.spawned_boards;
        result.move_candidates += workspace.stats.move_candidates;
        result.moved_candidates += workspace.stats.moved_candidates;
        result.encoded_candidates += workspace.stats.encoded_candidates;
        result.thread_spawn_move_seconds += workspace.stats.spawn_move_seconds;
        result.thread_canonical_seconds += workspace.stats.canonical_seconds;
        result.thread_encode_insert_seconds += workspace.stats.encode_insert_seconds;
    }
    result.spawn_move_seconds = result.thread_spawn_move_seconds;
    result.canonical_seconds = result.thread_canonical_seconds;
    result.encode_insert_seconds = result.thread_encode_insert_seconds;

    result.merge_seconds = 0.0;

    const double finalize_begin = bc_now_seconds();
    std::vector<FinalizedCellPayload> payloads = finalize_dynamic_state(dynamic_state, thread_count);
    result.finalize_seconds = bc_now_seconds() - finalize_begin;

    for (const FinalizedCellPayload &payload : payloads) {
        result.output_success_rows += payload.success_rows;
    }
    result.duplicate_candidates =
        result.encoded_candidates >= result.output_success_rows
            ? result.encoded_candidates - result.output_success_rows
            : 0U;
    result.valid_candidates = result.encoded_candidates;
    result.duplicate_candidates_possible = result.duplicate_candidates;

    const double write_begin = bc_now_seconds();
    BCPositionLayerWriter writer;
    writer.begin_layer(target_axis);
    for (CellId cid = 0U; cid < target_cell_count; ++cid) {
        const FinalizedCellPayload &payload = payloads[static_cast<size_t>(cid)];
        if (payload.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, payload);
    }
    result.position_bytes = writer.finish_layer();
    result.write_seconds = bc_now_seconds() - write_begin;

    result.compute_seconds = generation_seconds + result.merge_seconds + result.finalize_seconds;
    result.total_seconds = bc_now_seconds() - total_begin;
    return result;
}

BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentGenerationSource &source4,
    const BCResidentGenerationSource &source2,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer(
        lut,
        target_axis,
        std::vector<BCResidentGenerationSource>{source4, source2},
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCFamilyTable *secondary_axis,
    const BCResidentGenerationOptions &options
) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident pair generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident pair generation pending_insert_buffer_size must be non-zero");
    }

    const BCFamilyTable &current_axis = current.axis();
    if (current_axis.family_unit() != primary_axis.family_unit()) {
        throw std::invalid_argument("BC resident pair current/primary family_unit mismatch");
    }
    if (static_cast<uint32_t>(primary_axis.total_coord()) !=
        static_cast<uint32_t>(current_axis.total_coord()) + 1U) {
        throw std::invalid_argument("BC resident pair primary total_coord must equal current + 1");
    }
    if (carry_to_primary != nullptr) {
        const BCFamilyTable &carry_axis = carry_to_primary->axis();
        if (carry_axis.layer_sum() != primary_axis.layer_sum() ||
            carry_axis.family_unit() != primary_axis.family_unit() ||
            carry_axis.axis_base_coord() != primary_axis.axis_base_coord() ||
            carry_axis.family_count() != primary_axis.family_count()) {
            throw std::invalid_argument("BC resident pair carry axis must match primary axis");
        }
    }
    const bool has_secondary = secondary_axis != nullptr;
    if (has_secondary) {
        if (current_axis.family_unit() != secondary_axis->family_unit()) {
            throw std::invalid_argument("BC resident pair current/secondary family_unit mismatch");
        }
        if (static_cast<uint32_t>(secondary_axis->total_coord()) !=
            static_cast<uint32_t>(current_axis.total_coord()) + 2U) {
            throw std::invalid_argument("BC resident pair secondary total_coord must equal current + 2");
        }
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCCellMatrix primary_matrix(primary_axis);
    const BCCellMatrix secondary_matrix = has_secondary
        ? BCCellMatrix(*secondary_axis)
        : BCCellMatrix(primary_axis);
    const BCWordSumTable word_sums_storage = build_word_sum_table(options.family_tile_sum_values);
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;

    BCResidentGenerationPairResult pair;
    pair.has_secondary = has_secondary;
    pair.primary.effective_threads = thread_count;
    pair.secondary.effective_threads = thread_count;

    std::vector<BCThreadGenerationWorkspace> primary_workspaces;
    std::vector<BCThreadGenerationWorkspace> secondary_workspaces;
    BCDynamicState primary_state;
    BCDynamicState secondary_state;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = 1.0;
    bool generated = false;

    auto prepare_workspaces = [&](std::vector<BCThreadGenerationWorkspace> &workspaces) {
        workspaces.clear();
        workspaces.resize(static_cast<size_t>(thread_count));
        for (BCThreadGenerationWorkspace &workspace : workspaces) {
            workspace.canonical_buffer.reserve(options.canonical_batch_size);
            workspace.pending_encoded.reserve(options.pending_insert_buffer_size);
            workspace.resolved_encoded.reserve(options.pending_insert_buffer_size);
        }
    };

    const uint64_t current_buckets = position_bucket_count(current);
    const uint64_t current_bitmap_words = position_rank_payload_word_estimate(current);
    const uint64_t carry_buckets =
        carry_to_primary != nullptr ? position_bucket_count(*carry_to_primary) : 0U;
    const uint64_t carry_bitmap_words =
        carry_to_primary != nullptr ? position_rank_payload_word_estimate(*carry_to_primary) : 0U;

    const double generation_begin = bc_now_seconds();
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        pair.primary = BCResidentGenerationResult{};
        pair.secondary = BCResidentGenerationResult{};
        pair.primary.effective_threads = thread_count;
        pair.secondary.effective_threads = thread_count;
        prepare_workspaces(primary_workspaces);
        if (has_secondary) {
            prepare_workspaces(secondary_workspaces);
        } else {
            secondary_workspaces.clear();
        }

        primary_state = make_bc_dynamic_state(
            primary_matrix.cell_count(),
            static_cast<uint64_t>(static_cast<double>(current_buckets + carry_buckets) * reserve_factor) + 4096ULL,
            static_cast<uint64_t>(static_cast<double>(current_bitmap_words + carry_bitmap_words) * reserve_factor) +
                512ULL * 64ULL
        );
        if (has_secondary) {
            secondary_state = make_bc_dynamic_state(
                secondary_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor) +
                    512ULL * 64ULL
            );
        }

        if (carry_to_primary != nullptr) {
            run_insert_position_layer_phase(
                lut,
                *carry_to_primary,
                options,
                thread_count,
                primary_state,
                primary_workspaces
            );
        }
        run_source_pair_phase(
            lut,
            primary_axis,
            secondary_axis,
            current,
            options,
            thread_count,
            primary_state,
            has_secondary ? &secondary_state : nullptr,
            primary_workspaces,
            has_secondary ? &secondary_workspaces : nullptr,
            word_sums
        );

        const bool primary_overflow = primary_state.overflowed.load(std::memory_order_acquire);
        const bool secondary_overflow =
            has_secondary && secondary_state.overflowed.load(std::memory_order_acquire);
        if (!primary_overflow && !secondary_overflow) {
            generated = true;
            break;
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC resident pair generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    pair.primary.scan_seconds = generation_seconds;
    pair.secondary.scan_seconds = has_secondary ? generation_seconds : 0.0;
    add_workspace_stats(pair.primary, primary_workspaces);
    if (has_secondary) {
        add_workspace_stats(pair.secondary, secondary_workspaces);
    }

    finalize_dynamic_result(
        pair.primary,
        primary_axis,
        primary_state,
        options,
        thread_count,
        total_begin,
        generation_seconds
    );
    if (has_secondary) {
        finalize_dynamic_result(
            pair.secondary,
            *secondary_axis,
            secondary_state,
            options,
            thread_count,
            total_begin,
            generation_seconds
        );
    }
    return pair;
}

} // namespace BC
