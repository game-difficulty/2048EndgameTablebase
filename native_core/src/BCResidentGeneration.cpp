#include "BCResidentGeneration.h"

#include "BCResidentGenerationInternal.h"
#include "BCBoardOps.h"
#include "BCCellMatrix.h"
#include "BCFileIO.h"
#include "BCLoadedCellScanner.h"
#include "BCPositionCellLoader.h"
#include "BCPositionScanner.h"
#include "BCSortUtils.h"
#include "BoardMover.h"
#include "CanonicalBatch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {
namespace ResidentGenerationInternal {

[[nodiscard]] double bc_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] static int bc_omp_max_threads() {
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

[[nodiscard]] static const std::array<uint32_t, 16U> *resident_tile_sum_values(
    const BCResidentGenerationOptions &options
) {
    return options.tile_sum_values != nullptr
        ? options.tile_sum_values
        : options.family_tile_sum_values;
}

[[nodiscard]] static bool bc_is_success_by_shifts(
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

[[nodiscard]] static bool bc_family_axes_equal(
    const BCFamilyTable &lhs,
    const BCFamilyTable &rhs
) {
    return lhs.layer_sum() == rhs.layer_sum() &&
           lhs.family_unit() == rhs.family_unit() &&
           lhs.coords() == rhs.coords();
}

[[nodiscard]] static uint32_t countr_zero32(uint32_t value) {
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

[[nodiscard]] static uint32_t countr_zero64(uint64_t value) {
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

static void validate_resident_generation_source(
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

[[nodiscard]] static std::vector<FinalizedCellPayload> finalize_dynamic_state(
    const BCLut &lut,
    const BCDynamicState &state,
    int thread_count
);

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

[[nodiscard]] bool tile_sum_values_match_lut(
    const BCLut &lut,
    const std::array<uint32_t, 16U> *tile_sum_values
) {
    if (tile_sum_values == nullptr) {
        return true;
    }
    for (uint32_t tile = 0U; tile < tile_sum_values->size(); ++tile) {
        if ((*tile_sum_values)[tile] != lut.tile_sum_value(static_cast<uint8_t>(tile))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BCWordSumTable build_word_sum_table_if_needed(
    const BCLut &lut,
    const std::array<uint32_t, 16U> *tile_sum_values
) {
    return tile_sum_values_match_lut(lut, tile_sum_values)
        ? BCWordSumTable{}
        : build_word_sum_table(tile_sum_values);
}

[[nodiscard]] static BCBoardEncodedPosition encode_canonical_quadrants_position_hot(
    const BCLut &lut,
    const BCPositionCellLayout &layout,
    const BCQuadrantWords &q,
    const BCWordSumTable *word_sums
) {
    BCBoardEncodedPosition out;
    const BCFamilyTable &axis = layout.serialization_axis();

    const BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return out;
    }

    const uint64_t nw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.nw]
        : nw_desc.sum;
    const uint64_t ne_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.ne]
        : ne_desc.sum;
    const uint64_t sw_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.sw]
        : sw_desc.sum;
    const uint64_t se_sum = word_sums != nullptr && !word_sums->empty()
        ? (*word_sums)[q.se]
        : se_desc.sum;

    if (nw_sum + ne_sum + sw_sum + se_sum != axis.layer_sum()) {
        return out;
    }

    const uint16_t family_unit = axis.family_unit();
    auto min_side_coord = [family_unit](uint64_t first_sum, uint64_t second_sum, FamilyCoord &coord_out) {
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
    if (!min_side_coord(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
        !min_side_coord(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
        return out;
    }

    const FamilyId row_id = layout.try_raw_side_coord_to_physical_index(row_coord);
    const FamilyId col_id = layout.try_raw_side_coord_to_physical_index(col_coord);
    if (row_id == BCPositionCellLayout::kInvalidSideIndex ||
        col_id == BCPositionCellLayout::kInvalidSideIndex) {
        return out;
    }

    const uint32_t family_count = layout.side_coord_count();
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

[[nodiscard]] static uint32_t bc_next_power_of_two_u32(uint64_t value) {
    if (value > (1ULL << 31U)) {
        throw std::overflow_error("BC resident dynamic hash capacity exceeds uint32 power-of-two range");
    }
    uint32_t out = 1U;
    while (out < value) {
        out <<= 1U;
    }
    return std::max<uint32_t>(1024U, out);
}

[[nodiscard]] static uint32_t choose_bc_dynamic_capacity(uint64_t bucket_estimate) {
    constexpr uint64_t kLoadNumerator = 60U;
    constexpr uint64_t kLoadDenominator = 100U;
    const uint64_t required =
        (std::max<uint64_t>(bucket_estimate, 1U) * kLoadDenominator + (kLoadNumerator - 1U)) /
        kLoadNumerator;
    return bc_next_power_of_two_u32(required);
}

[[nodiscard]] static uint32_t bc_dynamic_home_slot(CellId cid, uint64_t key, uint32_t capacity) {
    const uint64_t value = key ^ (static_cast<uint64_t>(cid) * 0x9e3779b97f4a7c15ULL);
    const uint64_t mixed = value * 11400714819323198485ULL;
    const uint32_t bits =
#if defined(__GNUC__) || defined(__clang__)
        static_cast<uint32_t>(__builtin_ctz(capacity));
#else
        [] (uint32_t v) {
            uint32_t out = 0U;
            while ((v >>= 1U) != 0U) {
                ++out;
            }
            return out;
        }(capacity);
#endif
    return static_cast<uint32_t>(mixed >> (64U - bits));
}

[[nodiscard]] static uint32_t source_bucket_count_sum(const std::vector<BCResidentGenerationSource> &sources) {
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

[[nodiscard]] static uint64_t position_success_row_count(const BCPositionLayerReader &position) {
    uint64_t count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        count += position.descriptor(cid).success_rows;
    }
    return count;
}

[[nodiscard]] static uint64_t position_success_row_count(const BCPositionStreamingReader &position) {
    uint64_t count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        count += position.descriptor(cid).success_rows;
    }
    return count;
}

[[nodiscard]] static uint64_t source_success_row_count_sum(
    const std::vector<BCResidentGenerationSource> &sources
) {
    uint64_t count = 0U;
    for (const BCResidentGenerationSource &source : sources) {
        if (source.position == nullptr) {
            continue;
        }
        count = bc_checked_add_u64(
            count,
            position_success_row_count(*source.position),
            "BC resident generation source row count overflow"
        );
    }
    return count;
}

[[nodiscard]] static uint64_t position_bucket_count(const BCPositionLayerReader &position) {
    uint64_t count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        count += position.descriptor(cid).bucket_count;
    }
    return count;
}

[[nodiscard]] static uint64_t position_bucket_count(const BCPositionStreamingReader &position) {
    uint64_t count = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        count += position.descriptor(cid).bucket_count;
    }
    return count;
}

[[nodiscard]] static uint64_t source_rank_payload_word_estimate(const std::vector<BCResidentGenerationSource> &sources) {
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

[[nodiscard]] static uint64_t position_rank_payload_word_estimate(const BCPositionLayerReader &position) {
    uint64_t bytes = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        bytes += position.descriptor(cid).rank_payload_bytes;
    }
    return bytes / sizeof(uint64_t) + 1U;
}

[[nodiscard]] static uint64_t position_rank_payload_word_estimate(const BCPositionStreamingReader &position) {
    uint64_t bytes = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        bytes += position.descriptor(cid).rank_payload_bytes;
    }
    return bytes / sizeof(uint64_t) + 1U;
}

static void add_cell_load_stats(BCCellLoadStats &dst, const BCCellLoadStats &src) {
    dst.requested_extents = bc_checked_add_u64(
        dst.requested_extents,
        src.requested_extents,
        "BC resident streaming pair load requested extent count overflow"
    );
    dst.coalesced_extents = bc_checked_add_u64(
        dst.coalesced_extents,
        src.coalesced_extents,
        "BC resident streaming pair load coalesced extent count overflow"
    );
    dst.requested_bytes = bc_checked_add_u64(
        dst.requested_bytes,
        src.requested_bytes,
        "BC resident streaming pair load requested byte count overflow"
    );
    dst.read_bytes = bc_checked_add_u64(
        dst.read_bytes,
        src.read_bytes,
        "BC resident streaming pair load read byte count overflow"
    );
    dst.backend_read_ops = bc_checked_add_u64(
        dst.backend_read_ops,
        src.backend_read_ops,
        "BC resident streaming pair load backend op count overflow"
    );
    dst.backend_read_bytes = bc_checked_add_u64(
        dst.backend_read_bytes,
        src.backend_read_bytes,
        "BC resident streaming pair load backend byte count overflow"
    );
}

struct BCSourceBucketIndex {
    std::vector<uint64_t> cell_bucket_begin;
    uint64_t bucket_count = 0U;
};

[[nodiscard]] static BCSourceBucketIndex collect_source_bucket_index(
    const BCPositionLayerReader &position
) {
    BCSourceBucketIndex index;
    index.cell_bucket_begin.resize(static_cast<size_t>(position.cell_count()) + 1U, 0U);
    uint64_t cursor = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        index.cell_bucket_begin[cid] = cursor;
        cursor += position.descriptor(cid).bucket_count;
    }
    index.cell_bucket_begin[position.cell_count()] = cursor;
    index.bucket_count = cursor;
    return index;
}

static void resolve_source_bucket_index(
    const BCSourceBucketIndex &index,
    uint64_t linear_bucket,
    CellId &cid,
    uint32_t &bucket_index
) {
    if (linear_bucket >= index.bucket_count) {
        throw std::out_of_range("BC resident generation linear bucket index is out of range");
    }
    const auto begin = index.cell_bucket_begin.begin();
    const auto it = std::upper_bound(begin, index.cell_bucket_begin.end(), linear_bucket);
    if (it == begin) {
        throw std::logic_error("BC resident generation bucket prefix lookup underflow");
    }
    const uint64_t cid64 = static_cast<uint64_t>((it - begin) - 1);
    if (cid64 > std::numeric_limits<CellId>::max()) {
        throw std::overflow_error("BC resident generation resolved cell id exceeds CellId");
    }
    cid = static_cast<CellId>(cid64);
    const uint64_t local = linear_bucket - index.cell_bucket_begin[cid];
    if (local > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident generation local bucket index exceeds uint32");
    }
    bucket_index = static_cast<uint32_t>(local);
}

BCDynamicState make_bc_dynamic_state(
    uint32_t cell_count,
    uint64_t bucket_estimate,
    uint64_t bitmap_word_estimate,
    int thread_count
) {
    constexpr uint64_t kMinBitmapWords = 512U * 64U;
    BCDynamicState state;
    state.cell_count = cell_count;
    state.hash_capacity = choose_bc_dynamic_capacity(bucket_estimate);
    state.reserved_bitmap_words = std::max<uint64_t>(bitmap_word_estimate, kMinBitmapWords);
    if (state.reserved_bitmap_words > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC resident dynamic bitmap arena exceeds uint32 words");
    }
    state.cell_array = std::unique_ptr<std::atomic<uint32_t>[]>(
        new std::atomic<uint32_t>[state.hash_capacity]
    );
    state.key_array = std::unique_ptr<uint64_t[]>(new uint64_t[state.hash_capacity]);
    state.bitmap_offset_array = std::unique_ptr<uint32_t[]>(new uint32_t[state.hash_capacity]);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(thread_count) if(state.hash_capacity >= 16384U)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(state.hash_capacity); ++i) {
        state.cell_array[static_cast<uint32_t>(i)].store(
            BCDynamicState::kEmptyCell,
            std::memory_order_relaxed
        );
    }
    state.bitmap_arena = std::unique_ptr<std::atomic<uint64_t>[]>(
        new std::atomic<uint64_t>[static_cast<size_t>(state.reserved_bitmap_words)]
    );
    return state;
}

static uint32_t bc_dynamic_acquire_words(
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

static void bc_dynamic_clear_words(BCDynamicState &state, uint32_t offset, uint32_t words) {
    for (uint32_t i = 0U; i < words; ++i) {
        state.bitmap_arena[offset + i].store(0ULL, std::memory_order_relaxed);
    }
}

static uint32_t bc_dynamic_find_or_insert(
    BCDynamicState &state,
    const BCPendingEncodedCandidate &entry,
    BCDynamicThreadChunks &chunks
) {
    uint32_t slot = entry.home_slot;
    uint32_t probes = 0U;
    for (;;) {
        const uint32_t current = state.cell_array[slot].load(std::memory_order_acquire);
        if (current == entry.cid) {
            if (state.key_array[slot] == entry.key) {
                return state.bitmap_offset_array[slot];
            }
        } else if (current == BCDynamicState::kEmptyCell) {
            uint32_t expected = BCDynamicState::kEmptyCell;
            if (state.cell_array[slot].compare_exchange_strong(
                    expected,
                    BCDynamicState::kPendingCell,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                const uint32_t word_count = words_for_bits(entry.bitmap_len);
                const uint32_t offset = bc_dynamic_acquire_words(state, chunks, word_count);
                if (offset != BCDynamicState::kPendingCell) {
                    bc_dynamic_clear_words(state, offset, word_count);
                    state.key_array[slot] = entry.key;
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

static void flush_pending_encoded(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &state
) {
    if (workspace.pending_encoded.empty() || state.overflowed.load(std::memory_order_acquire)) {
        return;
    }

    constexpr uint32_t kPrefetchDistance = 16U;
    const uint32_t count = static_cast<uint32_t>(workspace.pending_encoded.size());
    workspace.resolved_encoded.clear();
    workspace.resolved_encoded.reserve(count);
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
        const uint32_t bitmap_offset = bc_dynamic_find_or_insert(
            state,
            candidate,
            workspace.dynamic_chunks
        );
        if (bitmap_offset == BCDynamicState::kPendingCell) {
            return;
        }
        workspace.resolved_encoded.push_back(BCDynamicResolved{
            bitmap_offset,
            candidate.rank
        });
        __builtin_prefetch(
            &state.bitmap_arena[bitmap_offset + (static_cast<uint32_t>(candidate.rank) >> 6U)],
            1,
            1
        );
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const BCDynamicResolved &resolved = workspace.resolved_encoded[i];
        const uint32_t word = static_cast<uint32_t>(resolved.rank) >> 6U;
        const uint64_t mask = 1ULL << (static_cast<uint32_t>(resolved.rank) & 63U);
        std::atomic<uint64_t> &target = state.bitmap_arena[resolved.bitmap_offset + word];
        if ((target.load(std::memory_order_relaxed) & mask) == 0ULL) {
            target.fetch_or(mask, std::memory_order_relaxed);
        }
    }

    workspace.pending_encoded.clear();
}

static void flush_pending_encoded_timed(
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

static void flush_canonical_buffer(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    BCDynamicState &dynamic_state,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool keep_only_success
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
        if (keep_only_success) {
            if (options.success_shifts == nullptr ||
                !bc_is_success_by_shifts(
                    canonical_board,
                    options.success_target_rank,
                    *options.success_shifts
                )) {
                continue;
            }
        }
        const BCQuadrantWords q = unpack_board_to_quadrants(canonical_board);
        const BCBoardEncodedPosition encoded =
            encode_canonical_quadrants_position_hot(
                lut,
                target_layout,
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
            bc_dynamic_home_slot(encoded.cid, key_rank.key, dynamic_state.hash_capacity),
            key_rank.key,
            key_rank.rank,
            key_rank.bitmap_len
        });
        if (workspace.pending_encoded.size() >= options.pending_insert_buffer_size) {
            flush_pending_encoded(workspace, dynamic_state);
        }
    }
    if (options.collect_timing) {
        workspace.stats.encode_insert_seconds += bc_now_seconds() - encode_begin;
    }
    workspace.canonical_buffer.clear();
}

static void push_pending_encoded_candidate(
    BCThreadGenerationWorkspace &workspace,
    BCDynamicState &dynamic_state,
    CellId cid,
    const BCEncodedKeyRank &key_rank,
    const BCResidentGenerationOptions &options
) {
    workspace.pending_encoded.push_back(BCPendingEncodedCandidate{
        cid,
        bc_dynamic_home_slot(cid, key_rank.key, dynamic_state.hash_capacity),
        key_rank.key,
        key_rank.rank,
        key_rank.bitmap_len
    });
    if (workspace.pending_encoded.size() >= options.pending_insert_buffer_size) {
        flush_pending_encoded(workspace, dynamic_state);
    }
}

static void push_moved_board(
    BCThreadGenerationWorkspace &workspace,
    uint64_t spawned,
    uint64_t moved
) {
    if (moved == spawned) {
        return;
    }
    workspace.canonical_buffer.push_back(moved);
}

static void process_source_board_pair(
    BCThreadGenerationWorkspace &primary_workspace,
    BCThreadGenerationWorkspace *secondary_workspace,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout *secondary_layout,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    uint64_t board,
    uint16_t source_empty_mask,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    if (skip_success_source &&
        options.success_shifts != nullptr &&
        bc_is_success_by_shifts(board, options.success_target_rank, *options.success_shifts)) {
        return;
    }
    uint32_t empty_mask = source_empty_mask;
    while (empty_mask != 0U) {
        const uint32_t cell = countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;

        const uint64_t spawn2 = board | (1ULL << (4U * cell));
        const auto moved2 = BoardMover::move_all_dir(spawn2);
        push_moved_board(primary_workspace, spawn2, std::get<0>(moved2));
        push_moved_board(primary_workspace, spawn2, std::get<1>(moved2));
        push_moved_board(primary_workspace, spawn2, std::get<2>(moved2));
        push_moved_board(primary_workspace, spawn2, std::get<3>(moved2));

        if (secondary_workspace != nullptr && secondary_layout != nullptr && secondary_state != nullptr) {
            const uint64_t spawn4 = board | (2ULL << (4U * cell));
            const auto moved4 = BoardMover::move_all_dir(spawn4);
            push_moved_board(*secondary_workspace, spawn4, std::get<0>(moved4));
            push_moved_board(*secondary_workspace, spawn4, std::get<1>(moved4));
            push_moved_board(*secondary_workspace, spawn4, std::get<2>(moved4));
            push_moved_board(*secondary_workspace, spawn4, std::get<3>(moved4));
        }
    }
    if (primary_workspace.canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(
            primary_workspace,
            lut,
            primary_layout,
            primary_state,
            options,
            word_sums,
            options.keep_only_success_generated_boards
        );
    }
    if (secondary_workspace != nullptr &&
        secondary_layout != nullptr &&
        secondary_state != nullptr &&
        secondary_workspace->canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(
            *secondary_workspace,
            lut,
            *secondary_layout,
            *secondary_state,
            options,
            word_sums,
            options.keep_only_success_secondary_generated_boards
        );
    }
}

static void process_source_board(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source,
    BCDynamicState &dynamic_state,
    uint64_t board,
    uint16_t source_empty_mask,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    if (skip_success_source &&
        options.success_shifts != nullptr &&
        bc_is_success_by_shifts(board, options.success_target_rank, *options.success_shifts)) {
        return;
    }
    uint32_t empty_mask = source_empty_mask;
    while (empty_mask != 0U) {
        const uint32_t cell = countr_zero32(empty_mask);
        empty_mask &= empty_mask - 1U;
        const uint64_t spawned =
            board | (static_cast<uint64_t>(source.spawn_tile_rank) << (4U * cell));
        const auto moved = BoardMover::move_all_dir(spawned);
        push_moved_board(workspace, spawned, std::get<0>(moved));
        push_moved_board(workspace, spawned, std::get<1>(moved));
        push_moved_board(workspace, spawned, std::get<2>(moved));
        push_moved_board(workspace, spawned, std::get<3>(moved));
    }
    if (workspace.canonical_buffer.size() >= options.canonical_batch_size) {
        flush_canonical_buffer(
            workspace,
            lut,
            target_layout,
            dynamic_state,
            options,
            word_sums,
            options.keep_only_success_generated_boards
        );
    }
}

template <class Fn>
static void for_each_bucket_rank(
    const BCLut &lut,
    const BCBucketEntry &bucket,
    BCRankPayloadView payload,
    Fn &&fn
) {
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

    const uint8_t *bitmap_words = payload.data + bitmap_offset;
    for (uint32_t word_i = 0U; word_i < bitmap_word_count; ++word_i) {
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
            fn(decoder, static_cast<BucketRank>(rank_u32));
            word &= word - 1U;
        }
    }
}

template <class Fn>
static void for_each_bucket_board(
    const BCLut &lut,
    const BCBucketEntry &bucket,
    BCRankPayloadView payload,
    Fn &&fn
) {
    const BCBucketBoardDecoder decoder(lut, bucket.key);
    const uint32_t bitmap_len = decoder.bitmap_len();
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

    const uint8_t *bitmap_words = payload.data + bitmap_offset;
    uint32_t bucket_seen = 0U;
    bc_scan_bucket_board_entries(
        bucket,
        decoder,
        bitmap_words,
        bitmap_word_count,
        bitmap_len,
        0U,
        bitmap_word_count,
        std::numeric_limits<uint64_t>::max(),
        bucket_seen,
        [&](BucketRank, uint32_t, uint64_t board, uint16_t empty_mask) {
            fn(board, empty_mask);
        }
    );
}

static void process_source_bucket_pair(
    BCThreadGenerationWorkspace &primary_workspace,
    BCThreadGenerationWorkspace *secondary_workspace,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout *secondary_layout,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    const BCPositionLayerReader &position,
    CellId cid,
    uint32_t bucket_index,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCPositionCellDescriptor &desc = position.descriptor(cid);
    if (bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation pair bucket index is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(cid);
    if (buckets.size != desc.bucket_count) {
        throw std::out_of_range("BC resident generation pair bucket view is shorter than descriptor");
    }
    const BCBucketEntry &bucket = buckets.data[bucket_index];
    for_each_bucket_board(
        lut,
        bucket,
        payload,
        [&](uint64_t board, uint16_t empty_mask) {
            process_source_board_pair(
                primary_workspace,
                secondary_workspace,
                lut,
                primary_layout,
                secondary_layout,
                primary_state,
                secondary_state,
                board,
                empty_mask,
                options,
                word_sums,
                skip_success_source
            );
        }
    );
}

struct BCLoadedBucketRef {
    uint32_t loaded_cell_index = 0U;
    uint32_t bucket_index = 0U;
};

static void process_loaded_cell_bucket_pair(
    BCThreadGenerationWorkspace &primary_workspace,
    BCThreadGenerationWorkspace *secondary_workspace,
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout *secondary_layout,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    const BCLoadedCell &cell,
    uint32_t bucket_index,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCLoadedCellView view = cell.view();
    if (bucket_index >= view.buckets.size) {
        throw std::out_of_range("BC SingleChunk loaded bucket index is out of range");
    }
    const BCBucketEntry &bucket = view.buckets.data[bucket_index];
    for_each_bucket_board(
        lut,
        bucket,
        view.rank_payload,
        [&](uint64_t board, uint16_t empty_mask) {
            process_source_board_pair(
                primary_workspace,
                secondary_workspace,
                lut,
                primary_layout,
                secondary_layout,
                primary_state,
                secondary_state,
                board,
                empty_mask,
                options,
                word_sums,
                skip_success_source
            );
        }
    );
}

static void process_loaded_cell_bucket_delta(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    BCDynamicState &dynamic_state,
    const BCResidentGenerationSource &source,
    const BCLoadedCell &cell,
    uint32_t bucket_index,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCLoadedCellView view = cell.view();
    if (bucket_index >= view.buckets.size) {
        throw std::out_of_range("BC SingleChunk loaded delta bucket index is out of range");
    }
    const BCBucketEntry &bucket = view.buckets.data[bucket_index];
    for_each_bucket_board(
        lut,
        bucket,
        view.rank_payload,
        [&](uint64_t board, uint16_t empty_mask) {
            process_source_board(
                workspace,
                lut,
                target_layout,
                source,
                dynamic_state,
                board,
                empty_mask,
                options,
                word_sums,
                skip_success_source
            );
        }
    );
}

static void insert_position_bucket_into_dynamic(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    BCDynamicState &dynamic_state,
    const BCPositionLayerReader &position,
    CellId cid,
    uint32_t bucket_index,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellDescriptor &desc = position.descriptor(cid);
    if (bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation carry bucket index is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(cid);
    if (buckets.size != desc.bucket_count) {
        throw std::out_of_range("BC resident generation carry bucket view is shorter than descriptor");
    }
    const BCBucketEntry &bucket = buckets.data[bucket_index];
    for_each_bucket_rank(
        lut,
        bucket,
        payload,
        [&](const BCBucketRankDecoder &decoder, BucketRank rank) {
            BCEncodedKeyRank key_rank;
            key_rank.key = bucket.key;
            key_rank.rank = rank;
            key_rank.bitmap_len = static_cast<BucketBitmapLen>(decoder.bitmap_len);
            key_rank.count_ne = decoder.count_ne;
            key_rank.count_sw = decoder.count_sw;
            key_rank.count_se = decoder.count_se;
            key_rank.valid = true;
            push_pending_encoded_candidate(
                workspace,
                dynamic_state,
                cid,
                key_rank,
                options
            );
        }
    );
}

static void process_source_bucket(
    BCThreadGenerationWorkspace &workspace,
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source,
    BCDynamicState &dynamic_state,
    CellId cid,
    uint32_t bucket_index,
    const BCResidentGenerationOptions &options,
    const BCWordSumTable *word_sums,
    bool skip_success_source
) {
    const BCPositionLayerReader &position = *source.position;
    const BCPositionCellDescriptor &desc = position.descriptor(cid);
    if (bucket_index >= desc.bucket_count) {
        throw std::out_of_range("BC resident generation bucket index is out of range");
    }
    const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
    const BCRankPayloadView payload = position.rank_payload_for_cell(cid);
    if (buckets.size != desc.bucket_count) {
        throw std::out_of_range("BC resident generation bucket view is shorter than descriptor");
    }
    const BCBucketEntry &bucket = buckets.data[bucket_index];
    for_each_bucket_board(
        lut,
        bucket,
        payload,
        [&](uint64_t board, uint16_t empty_mask) {
            process_source_board(
                workspace,
                lut,
                target_layout,
                source,
                dynamic_state,
                board,
                empty_mask,
                options,
                word_sums,
                skip_success_source
            );
        }
    );
}

static void run_source_phase(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces,
    const BCWordSumTable *word_sums
) {
    validate_resident_generation_source(target_layout.serialization_axis(), source);
    const bool skip_success_source =
        bc_success_check_enabled(options, source.position->axis().layer_sum());
    const BCSourceBucketIndex bucket_index = collect_source_bucket_index(*source.position);
    if (bucket_index.bucket_count == 0U) {
        return;
    }
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(thread_count)
    {
        try {
            const int tid = bc_omp_thread_num();
            BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
            const double phase_begin = options.collect_timing ? bc_now_seconds() : 0.0;
            const double canonical_before = workspace.stats.canonical_seconds;
            const double encode_before = workspace.stats.encode_insert_seconds;
#pragma omp for schedule(dynamic, 16) nowait
            for (int64_t bucket_i = 0; bucket_i < static_cast<int64_t>(bucket_index.bucket_count); ++bucket_i) {
                CellId cid = 0U;
                uint32_t local_bucket = 0U;
                resolve_source_bucket_index(bucket_index, static_cast<uint64_t>(bucket_i), cid, local_bucket);
                process_source_bucket(
                    workspace,
                    lut,
                    target_layout,
                    source,
                    dynamic_state,
                    cid,
                    local_bucket,
                    options,
                    word_sums,
                    skip_success_source
                );
            }
            flush_canonical_buffer(
                workspace,
                lut,
                target_layout,
                dynamic_state,
                options,
                word_sums,
                options.keep_only_success_generated_boards
            );
            flush_pending_encoded_timed(workspace, dynamic_state, options);
            if (options.collect_timing) {
                const double phase_elapsed = bc_now_seconds() - phase_begin;
                const double nested_elapsed =
                    (workspace.stats.canonical_seconds - canonical_before) +
                    (workspace.stats.encode_insert_seconds - encode_before);
                workspace.stats.spawn_move_seconds += std::max(0.0, phase_elapsed - nested_elapsed);
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

static void run_insert_position_layer_phase(
    const BCLut &lut,
    const BCPositionLayerReader &position,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces
) {
    const BCSourceBucketIndex bucket_index = collect_source_bucket_index(position);
    if (bucket_index.bucket_count == 0U) {
        return;
    }
    std::exception_ptr first_exception;

#pragma omp parallel num_threads(thread_count)
    {
        try {
            const int tid = bc_omp_thread_num();
            BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
            const double phase_begin = options.collect_timing ? bc_now_seconds() : 0.0;
            const double encode_before = workspace.stats.encode_insert_seconds;
#pragma omp for schedule(dynamic, 16) nowait
            for (int64_t bucket_i = 0; bucket_i < static_cast<int64_t>(bucket_index.bucket_count); ++bucket_i) {
                CellId cid = 0U;
                uint32_t local_bucket = 0U;
                resolve_source_bucket_index(bucket_index, static_cast<uint64_t>(bucket_i), cid, local_bucket);
                insert_position_bucket_into_dynamic(
                    workspace,
                    lut,
                    dynamic_state,
                    position,
                    cid,
                    local_bucket,
                    options
                );
            }
            flush_pending_encoded_timed(workspace, dynamic_state, options);
            if (options.collect_timing) {
                const double phase_elapsed = bc_now_seconds() - phase_begin;
                const double nested_elapsed = workspace.stats.encode_insert_seconds - encode_before;
                workspace.stats.encode_insert_seconds += std::max(0.0, phase_elapsed - nested_elapsed);
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

static void run_source_pair_phase(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout *secondary_layout,
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
    const BCSourceBucketIndex bucket_index = collect_source_bucket_index(current);
    if (bucket_index.bucket_count == 0U) {
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
            const double phase_begin = options.collect_timing ? bc_now_seconds() : 0.0;
            const double primary_canonical_before = primary_workspace.stats.canonical_seconds;
            const double primary_encode_before = primary_workspace.stats.encode_insert_seconds;
            const double secondary_canonical_before =
                secondary_workspace != nullptr ? secondary_workspace->stats.canonical_seconds : 0.0;
            const double secondary_encode_before =
                secondary_workspace != nullptr ? secondary_workspace->stats.encode_insert_seconds : 0.0;
#pragma omp for schedule(dynamic, 16) nowait
            for (int64_t bucket_i = 0; bucket_i < static_cast<int64_t>(bucket_index.bucket_count); ++bucket_i) {
                CellId cid = 0U;
                uint32_t local_bucket = 0U;
                resolve_source_bucket_index(bucket_index, static_cast<uint64_t>(bucket_i), cid, local_bucket);
                process_source_bucket_pair(
                    primary_workspace,
                    secondary_workspace,
                    lut,
                    primary_layout,
                    secondary_layout,
                    primary_state,
                    secondary_state,
                    current,
                    cid,
                    local_bucket,
                    options,
                    word_sums,
                    skip_success_source
                );
            }
            flush_canonical_buffer(
                primary_workspace,
                lut,
                primary_layout,
                primary_state,
                options,
                word_sums,
                options.keep_only_success_generated_boards
            );
            flush_pending_encoded_timed(primary_workspace, primary_state, options);
            if (secondary_workspace != nullptr &&
                secondary_layout != nullptr &&
                secondary_state != nullptr) {
                flush_canonical_buffer(
                    *secondary_workspace,
                    lut,
                    *secondary_layout,
                    *secondary_state,
                    options,
                    word_sums,
                    options.keep_only_success_secondary_generated_boards
                );
                flush_pending_encoded_timed(*secondary_workspace, *secondary_state, options);
            }
            if (options.collect_timing) {
                const double phase_elapsed = bc_now_seconds() - phase_begin;
                const double primary_nested =
                    (primary_workspace.stats.canonical_seconds - primary_canonical_before) +
                    (primary_workspace.stats.encode_insert_seconds - primary_encode_before);
                const double secondary_nested =
                    secondary_workspace != nullptr
                        ? (secondary_workspace->stats.canonical_seconds - secondary_canonical_before) +
                          (secondary_workspace->stats.encode_insert_seconds - secondary_encode_before)
                        : 0.0;
                primary_workspace.stats.spawn_move_seconds +=
                    std::max(0.0, phase_elapsed - primary_nested - secondary_nested);
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

static void run_streaming_source_pair_phase(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionCellLayout *secondary_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &primary_state,
    BCDynamicState *secondary_state,
    std::vector<BCThreadGenerationWorkspace> &primary_workspaces,
    std::vector<BCThreadGenerationWorkspace> *secondary_workspaces,
    const BCWordSumTable *word_sums,
    BCCellLoadStats &load_stats,
    double &load_seconds
) {
    const bool skip_success_source =
        bc_success_check_enabled(options, current.axis().layer_sum());
    const uint32_t cell_count = current.cell_count();
    if (cell_count == 0U) {
        return;
    }
    const uint32_t chunk_size = current_cell_chunk_size == 0U
        ? cell_count
        : current_cell_chunk_size;
    if (chunk_size == 0U) {
        throw std::invalid_argument("BC SingleChunk streaming pair chunk size resolved to zero");
    }

    std::vector<CellId> cids;
    cids.reserve(chunk_size);
    for (CellId chunk_begin = 0U; chunk_begin < cell_count;) {
        const uint64_t chunk_end_u64 = std::min<uint64_t>(
            cell_count,
            static_cast<uint64_t>(chunk_begin) + static_cast<uint64_t>(chunk_size)
        );
        const CellId chunk_end = static_cast<CellId>(chunk_end_u64);
        cids.clear();
        for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
            cids.push_back(cid);
        }

        BCCellLoadStats chunk_stats;
        const double load_begin = options.collect_timing ? bc_now_seconds() : 0.0;
        std::vector<BCLoadedCell> loaded_cells = current.load_cells(cids, &chunk_stats);
        if (options.collect_timing) {
            load_seconds += bc_now_seconds() - load_begin;
        }
        add_cell_load_stats(load_stats, chunk_stats);
        if (loaded_cells.empty()) {
            chunk_begin = chunk_end;
            continue;
        }

        std::vector<BCLoadedBucketRef> bucket_refs;
        uint64_t bucket_ref_count = 0U;
        for (const BCLoadedCell &cell : loaded_cells) {
            bucket_ref_count += cell.view().buckets.size;
        }
        if (bucket_ref_count > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC SingleChunk loaded bucket ref count exceeds uint32");
        }
        bucket_refs.reserve(static_cast<size_t>(bucket_ref_count));
        for (uint32_t cell_i = 0U; cell_i < static_cast<uint32_t>(loaded_cells.size()); ++cell_i) {
            const BCLoadedCellView view = loaded_cells[static_cast<size_t>(cell_i)].view();
            for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
                bucket_refs.push_back(BCLoadedBucketRef{cell_i, bucket_i});
            }
        }
        if (bucket_refs.empty()) {
            chunk_begin = chunk_end;
            continue;
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
                const double phase_begin = options.collect_timing ? bc_now_seconds() : 0.0;
                const double primary_canonical_before = primary_workspace.stats.canonical_seconds;
                const double primary_encode_before = primary_workspace.stats.encode_insert_seconds;
                const double secondary_canonical_before =
                    secondary_workspace != nullptr ? secondary_workspace->stats.canonical_seconds : 0.0;
                const double secondary_encode_before =
                    secondary_workspace != nullptr ? secondary_workspace->stats.encode_insert_seconds : 0.0;
#pragma omp for schedule(dynamic, 64)
                for (int64_t ref_i = 0; ref_i < static_cast<int64_t>(bucket_refs.size()); ++ref_i) {
                    const BCLoadedBucketRef ref = bucket_refs[static_cast<size_t>(ref_i)];
                    process_loaded_cell_bucket_pair(
                        primary_workspace,
                        secondary_workspace,
                        lut,
                        primary_layout,
                        secondary_layout,
                        primary_state,
                        secondary_state,
                        loaded_cells[static_cast<size_t>(ref.loaded_cell_index)],
                        ref.bucket_index,
                        options,
                        word_sums,
                        skip_success_source
                    );
                }
                if (options.collect_timing) {
                    const double phase_elapsed = bc_now_seconds() - phase_begin;
                    const double primary_nested =
                        (primary_workspace.stats.canonical_seconds - primary_canonical_before) +
                        (primary_workspace.stats.encode_insert_seconds - primary_encode_before);
                    double secondary_nested = 0.0;
                    if (secondary_workspace != nullptr) {
                        secondary_nested =
                            (secondary_workspace->stats.canonical_seconds - secondary_canonical_before) +
                            (secondary_workspace->stats.encode_insert_seconds - secondary_encode_before);
                    }
                    primary_workspace.stats.spawn_move_seconds +=
                        std::max(0.0, phase_elapsed - primary_nested - secondary_nested);
                }
                flush_canonical_buffer(
                    primary_workspace,
                    lut,
                    primary_layout,
                    primary_state,
                    options,
                    word_sums,
                    options.keep_only_success_generated_boards
                );
                flush_pending_encoded_timed(primary_workspace, primary_state, options);
                if (secondary_workspace != nullptr &&
                    secondary_layout != nullptr &&
                    secondary_state != nullptr) {
                    flush_canonical_buffer(
                        *secondary_workspace,
                        lut,
                        *secondary_layout,
                        *secondary_state,
                        options,
                        word_sums,
                        options.keep_only_success_secondary_generated_boards
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
        chunk_begin = chunk_end;
    }
}

static void run_streaming_source_delta_phase(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    uint8_t spawn_tile_rank,
    SpawnDeltaCoord delta_coord,
    const BCResidentGenerationOptions &options,
    int thread_count,
    BCDynamicState &dynamic_state,
    std::vector<BCThreadGenerationWorkspace> &workspaces,
    const BCWordSumTable *word_sums,
    BCCellLoadStats &load_stats,
    double &load_seconds
) {
    if (spawn_tile_rank == 0U || spawn_tile_rank > 15U) {
        throw std::invalid_argument("BC SingleChunk delta spawn_tile_rank must be in 1..15");
    }
    const BCFamilyTable &current_axis = current.axis();
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    if (current_axis.family_unit() != target_axis.family_unit()) {
        throw std::invalid_argument("BC SingleChunk delta current/target family_unit mismatch");
    }
    if (static_cast<uint32_t>(target_axis.total_coord()) !=
        static_cast<uint32_t>(current_axis.total_coord()) + static_cast<uint32_t>(delta_coord)) {
        throw std::invalid_argument("BC SingleChunk delta target total_coord mismatch");
    }

    const bool skip_success_source =
        bc_success_check_enabled(options, current.axis().layer_sum());
    const uint32_t cell_count = current.cell_count();
    if (cell_count == 0U) {
        return;
    }
    const uint32_t chunk_size = current_cell_chunk_size == 0U
        ? cell_count
        : current_cell_chunk_size;
    if (chunk_size == 0U) {
        throw std::invalid_argument("BC SingleChunk delta chunk size resolved to zero");
    }

    const BCResidentGenerationSource board_source{
        nullptr,
        spawn_tile_rank,
        delta_coord
    };

    std::vector<CellId> cids;
    cids.reserve(chunk_size);
    for (CellId chunk_begin = 0U; chunk_begin < cell_count;) {
        const uint64_t chunk_end_u64 = std::min<uint64_t>(
            cell_count,
            static_cast<uint64_t>(chunk_begin) + static_cast<uint64_t>(chunk_size)
        );
        const CellId chunk_end = static_cast<CellId>(chunk_end_u64);
        cids.clear();
        for (CellId cid = chunk_begin; cid < chunk_end; ++cid) {
            cids.push_back(cid);
        }

        BCCellLoadStats chunk_stats;
        const double load_begin = options.collect_timing ? bc_now_seconds() : 0.0;
        std::vector<BCLoadedCell> loaded_cells = current.load_cells(cids, &chunk_stats);
        if (options.collect_timing) {
            load_seconds += bc_now_seconds() - load_begin;
        }
        add_cell_load_stats(load_stats, chunk_stats);
        if (loaded_cells.empty()) {
            chunk_begin = chunk_end;
            continue;
        }

        std::vector<BCLoadedBucketRef> bucket_refs;
        uint64_t bucket_ref_count = 0U;
        for (const BCLoadedCell &cell : loaded_cells) {
            bucket_ref_count += cell.view().buckets.size;
        }
        if (bucket_ref_count > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC SingleChunk delta loaded bucket ref count exceeds uint32");
        }
        bucket_refs.reserve(static_cast<size_t>(bucket_ref_count));
        for (uint32_t cell_i = 0U; cell_i < static_cast<uint32_t>(loaded_cells.size()); ++cell_i) {
            const BCLoadedCellView view = loaded_cells[static_cast<size_t>(cell_i)].view();
            for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
                bucket_refs.push_back(BCLoadedBucketRef{cell_i, bucket_i});
            }
        }
        if (bucket_refs.empty()) {
            chunk_begin = chunk_end;
            continue;
        }

        std::exception_ptr first_exception;
#pragma omp parallel num_threads(thread_count)
        {
            try {
                const int tid = bc_omp_thread_num();
                BCThreadGenerationWorkspace &workspace = workspaces[static_cast<size_t>(tid)];
                const double phase_begin = options.collect_timing ? bc_now_seconds() : 0.0;
                const double canonical_before = workspace.stats.canonical_seconds;
                const double encode_before = workspace.stats.encode_insert_seconds;
#pragma omp for schedule(dynamic, 64)
                for (int64_t ref_i = 0; ref_i < static_cast<int64_t>(bucket_refs.size()); ++ref_i) {
                    const BCLoadedBucketRef ref = bucket_refs[static_cast<size_t>(ref_i)];
                    process_loaded_cell_bucket_delta(
                        workspace,
                        lut,
                        target_layout,
                        dynamic_state,
                        board_source,
                        loaded_cells[static_cast<size_t>(ref.loaded_cell_index)],
                        ref.bucket_index,
                        options,
                        word_sums,
                        skip_success_source
                    );
                }
                if (options.collect_timing) {
                    const double phase_elapsed = bc_now_seconds() - phase_begin;
                    const double nested =
                        (workspace.stats.canonical_seconds - canonical_before) +
                        (workspace.stats.encode_insert_seconds - encode_before);
                    workspace.stats.spawn_move_seconds += std::max(0.0, phase_elapsed - nested);
                }
                flush_canonical_buffer(
                    workspace,
                    lut,
                    target_layout,
                    dynamic_state,
                    options,
                    word_sums,
                    options.keep_only_success_generated_boards
                );
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
        chunk_begin = chunk_end;
    }
}

void add_workspace_stats(
    BCResidentGenerationResult &result,
    const std::vector<BCThreadGenerationWorkspace> &workspaces
) {
    for (const BCThreadGenerationWorkspace &workspace : workspaces) {
        result.thread_spawn_move_seconds += workspace.stats.spawn_move_seconds;
        result.thread_canonical_seconds += workspace.stats.canonical_seconds;
        result.thread_encode_insert_seconds += workspace.stats.encode_insert_seconds;
    }
}

[[nodiscard]] static size_t bc_checked_size_t_u64(uint64_t value, const char *label) {
    if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<size_t>(value);
}

static void bc_store_u16_le(uint8_t *out, uint16_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
}

static void bc_store_u32_le(uint8_t *out, uint32_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    out[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    out[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

static void bc_store_u64_le(uint8_t *out, uint64_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    out[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    out[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    out[4] = static_cast<uint8_t>((value >> 32U) & 0xFFU);
    out[5] = static_cast<uint8_t>((value >> 40U) & 0xFFU);
    out[6] = static_cast<uint8_t>((value >> 48U) & 0xFFU);
    out[7] = static_cast<uint8_t>((value >> 56U) & 0xFFU);
}

static void bc_store_cell_descriptor_le(uint8_t *out, const BCPositionCellDescriptor &descriptor) {
    bc_store_u32_le(out + 0U, descriptor.bucket_count);
    bc_store_u32_le(out + 4U, descriptor.success_rows);
    bc_store_u64_le(out + 8U, descriptor.bucket_meta_offset);
    bc_store_u64_le(out + 16U, descriptor.rank_payload_offset);
    bc_store_u64_le(out + 24U, descriptor.rank_payload_bytes);
    bc_store_u32_le(out + 32U, descriptor.reserved0);
    bc_store_u32_le(out + 36U, descriptor.flags_or_padding);
}

static void bc_store_bucket_entry_le(uint8_t *out, const BCBucketEntry &entry) {
    bc_store_u64_le(out + 0U, entry.key);
    bc_store_u32_le(out + 8U, entry.rank_payload_offset);
    bc_store_u32_le(out + 12U, entry.success_row_offset);
}

[[nodiscard]] static std::vector<uint8_t> build_position_layer_bytes_from_payloads(
    const BCFamilyTable &axis,
    const std::vector<FinalizedCellPayload> &payloads
) {
    const BCCellMatrix matrix(axis);
    if (payloads.size() != matrix.cell_count()) {
        throw std::invalid_argument("BC position payload count does not match axis cell count");
    }

    std::vector<BCPositionCellDescriptor> descriptors(payloads.size());
    uint64_t bucket_cursor = 0U;
    uint64_t rank_cursor = 0U;
    for (CellId cid = 0U; cid < payloads.size(); ++cid) {
        const FinalizedCellPayload &payload = payloads[static_cast<size_t>(cid)];
        if (payload.buckets.empty()) {
            if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
                throw std::invalid_argument("BC empty finalized payload has non-empty metadata");
            }
            BCPositionCellDescriptor descriptor;
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
            descriptors[static_cast<size_t>(cid)] = descriptor;
            continue;
        }
        if (payload.buckets.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC position cell bucket_count exceeds uint32");
        }
        for (size_t i = 1U; i < payload.buckets.size(); ++i) {
            if (payload.buckets[i - 1U].key >= payload.buckets[i].key) {
                throw std::invalid_argument("BC position writer requires sorted unique bucket keys");
            }
        }

        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = static_cast<uint32_t>(payload.buckets.size());
        descriptor.success_rows = payload.success_rows;
        descriptor.bucket_meta_offset = bucket_cursor;
        descriptor.rank_payload_offset = rank_cursor;
        descriptor.rank_payload_bytes = payload.rank_payload.size();
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;
        descriptors[static_cast<size_t>(cid)] = descriptor;

        bucket_cursor = bc_checked_add_u64(
            bucket_cursor,
            static_cast<uint64_t>(payload.buckets.size()) * kBCPositionBucketEntryBytes,
            "BC position payload bucket byte count overflow"
        );
        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            static_cast<uint64_t>(payload.rank_payload.size()),
            "BC position payload rank byte count overflow"
        );
    }

    const uint64_t descriptor_count = descriptors.size();
    const uint64_t descriptor_bytes =
        descriptor_count * static_cast<uint64_t>(kBCPositionCellDescriptorBytes);
    const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis.family_count());
    const uint64_t descriptor_offset = bc_checked_add_u64(
        kBCPositionHeaderBytes,
        axis_coord_bytes,
        "BC position descriptor table offset overflow"
    );
    const uint64_t bucket_offset = bc_checked_add_u64(
        descriptor_offset,
        descriptor_bytes,
        "BC position bucket stream offset overflow"
    );
    const uint64_t rank_offset = bc_checked_add_u64(
        bucket_offset,
        bucket_cursor,
        "BC position rank stream offset overflow"
    );
    const uint64_t logical_size = bc_checked_add_u64(
        rank_offset,
        rank_cursor,
        "BC position layer logical size overflow"
    );

    BCPositionHeader header;
    header.family_unit = axis.family_unit();
    header.axis_base_coord = axis.axis_base_coord();
    header.family_count = axis.family_count();
    header.layer_sum = axis.layer_sum();
    header.axis_coord_table_bytes = axis_coord_bytes;
    header.descriptor_count = descriptor_count;
    header.descriptor_table_offset = descriptor_offset;
    header.descriptor_table_bytes = descriptor_bytes;
    header.bucket_meta_offset = bucket_offset;
    header.bucket_meta_bytes = bucket_cursor;
    header.rank_payload_offset = rank_offset;
    header.rank_payload_bytes = rank_cursor;

    std::vector<uint8_t> out(bc_checked_size_t_u64(logical_size, "BC position layer exceeds size_t"));
    std::vector<uint8_t> scratch;
    scratch.reserve(static_cast<size_t>(std::max<uint64_t>(kBCPositionHeaderBytes, axis_coord_bytes)));
    bc_append_header(scratch, header);
    std::memcpy(
        out.data(),
        scratch.data(),
        scratch.size()
    );
    scratch.clear();
    bc_append_axis_coord_table(scratch, axis);
    std::memcpy(
        out.data() + bc_checked_size_t_u64(kBCPositionHeaderBytes, "BC position header offset exceeds size_t"),
        scratch.data(),
        scratch.size()
    );

    uint8_t *descriptor_cursor =
        out.data() + bc_checked_size_t_u64(descriptor_offset, "BC descriptor offset exceeds size_t");
    for (const BCPositionCellDescriptor &descriptor : descriptors) {
        bc_store_cell_descriptor_le(descriptor_cursor, descriptor);
        descriptor_cursor += kBCPositionCellDescriptorBytes;
    }

    for (CellId cid = 0U; cid < payloads.size(); ++cid) {
        const FinalizedCellPayload &payload = payloads[static_cast<size_t>(cid)];
        if (payload.buckets.empty()) {
            continue;
        }
        const BCPositionCellDescriptor &descriptor = descriptors[static_cast<size_t>(cid)];
        uint8_t *bucket_cursor_ptr =
            out.data() + bc_checked_size_t_u64(
                bucket_offset + descriptor.bucket_meta_offset,
                "BC bucket payload offset exceeds size_t"
            );
        for (const BCBucketEntry &bucket : payload.buckets) {
            bc_store_bucket_entry_le(bucket_cursor_ptr, bucket);
            bucket_cursor_ptr += kBCPositionBucketEntryBytes;
        }
        if (!payload.rank_payload.empty()) {
            std::memcpy(
                out.data() + bc_checked_size_t_u64(
                    rank_offset + descriptor.rank_payload_offset,
                    "BC rank payload offset exceeds size_t"
                ),
                payload.rank_payload.data(),
                payload.rank_payload.size()
            );
        }
    }
    return out;
}

[[nodiscard]] static uint64_t write_dynamic_state_to_file_streaming(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCDynamicState &state,
    int thread_count,
    BCWritableFile &file,
    BCFileIOStats *stats,
    uint64_t &success_rows_out,
    uint64_t &bucket_slots_used_out,
    uint64_t &bitmap_words_used_out,
    double *write_seconds_out
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
) {
    result.generation_seconds = generation_seconds;

    if (output_file != nullptr) {
        BCFileIOStats write_stats;
        uint64_t output_rows = 0U;
        uint64_t bucket_slots_used = 0U;
        uint64_t bitmap_words_used = 0U;
        double backend_write_seconds = 0.0;
        const double finalize_begin = bc_now_seconds();
        result.target_position_file_logical_bytes =
            write_dynamic_state_to_file_streaming(
                lut,
                axis,
                dynamic_state,
                thread_count,
                *output_file,
                &write_stats,
                output_rows,
                bucket_slots_used,
                bitmap_words_used,
                &backend_write_seconds
            );
        const double finalize_write_seconds = bc_now_seconds() - finalize_begin;
        result.output_success_rows = output_rows;
        result.dynamic_bucket_slots_used = bucket_slots_used;
        result.dynamic_bitmap_words_used = bitmap_words_used;
        result.target_position_write_requests = write_stats.request_count;
        result.target_position_write_requested_bytes = write_stats.requested_bytes;
        result.target_position_write_backend_ops = write_stats.backend_io_count;
        result.target_position_write_backend_bytes = write_stats.backend_bytes;
        result.finalize_seconds = std::max(0.0, finalize_write_seconds - backend_write_seconds);
        result.write_seconds = backend_write_seconds;
        result.position_bytes.clear();
    } else {
        const double finalize_begin = bc_now_seconds();
        std::vector<FinalizedCellPayload> payloads =
            finalize_dynamic_state(lut, dynamic_state, thread_count);
        result.finalize_seconds = bc_now_seconds() - finalize_begin;

        for (const FinalizedCellPayload &payload : payloads) {
            result.output_success_rows += payload.success_rows;
        }
        const double write_begin = bc_now_seconds();
        result.position_bytes = build_position_layer_bytes_from_payloads(axis, payloads);
        result.target_position_file_logical_bytes =
            static_cast<uint64_t>(result.position_bytes.size());
        result.target_position_write_requested_bytes =
            static_cast<uint64_t>(result.position_bytes.size());
        result.write_seconds = bc_now_seconds() - write_begin;
    }

    result.compute_seconds = generation_seconds + result.finalize_seconds;
    (void)options;
    result.total_seconds = bc_now_seconds() - total_begin;
}

static void bc_append_padding(std::vector<uint8_t> &buffer, uint32_t bytes) {
    buffer.insert(buffer.end(), bytes, 0U);
}

static void bc_append_u16_le(std::vector<uint8_t> &buffer, uint16_t value) {
    const size_t offset = buffer.size();
    buffer.resize(offset + 2U);
    uint8_t *p = buffer.data() + offset;
    p[0] = static_cast<uint8_t>(value & 0xFFU);
    p[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
}

static void bc_append_u64_le(std::vector<uint8_t> &buffer, uint64_t value) {
    const size_t offset = buffer.size();
    buffer.resize(offset + 8U);
    uint8_t *p = buffer.data() + offset;
    p[0] = static_cast<uint8_t>(value & 0xFFU);
    p[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    p[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    p[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    p[4] = static_cast<uint8_t>((value >> 32U) & 0xFFU);
    p[5] = static_cast<uint8_t>((value >> 40U) & 0xFFU);
    p[6] = static_cast<uint8_t>((value >> 48U) & 0xFFU);
    p[7] = static_cast<uint8_t>((value >> 56U) & 0xFFU);
}

static void bc_append_prefix256_le(
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

static void bc_append_bitmap_le(std::vector<uint8_t> &buffer, const uint64_t *bitmap, uint32_t word_count) {
    const size_t offset = buffer.size();
    buffer.resize(offset + static_cast<size_t>(word_count) * sizeof(uint64_t));
    uint8_t *p = buffer.data() + offset;
    for (uint32_t i = 0U; i < word_count; ++i) {
        const uint64_t value = bitmap[i];
        p[0] = static_cast<uint8_t>(value & 0xFFU);
        p[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        p[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        p[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
        p[4] = static_cast<uint8_t>((value >> 32U) & 0xFFU);
        p[5] = static_cast<uint8_t>((value >> 40U) & 0xFFU);
        p[6] = static_cast<uint8_t>((value >> 48U) & 0xFFU);
        p[7] = static_cast<uint8_t>((value >> 56U) & 0xFFU);
        p += sizeof(uint64_t);
    }
}

static void bc_append_rank_payload_le_from_atomic_arena(
    std::vector<uint8_t> &buffer,
    const BCDynamicState &state,
    uint32_t bitmap_offset,
    uint32_t word_count,
    uint32_t bitmap_len,
    uint32_t rank_payload_offset
) {
    if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
        throw std::out_of_range("BC dynamic rank bitmap offset exceeds arena");
    }
    if (checked_u32_size(buffer.size(), "BC dynamic rank payload buffer offset exceeds uint32") !=
        rank_payload_offset) {
        throw std::logic_error("BC dynamic rank payload buffer is not positioned at payload offset");
    }

    const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
    const uint32_t prefix_bytes = prefix_count * static_cast<uint32_t>(sizeof(RankPrefix));
    const uint32_t prefix_end = checked_u32_add(
        rank_payload_offset,
        prefix_bytes,
        "BC dynamic rank prefix payload end overflow"
    );
    const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
    const uint64_t payload_end = bc_checked_add_u64(
        out_bitmap_offset,
        static_cast<uint64_t>(word_count) * sizeof(uint64_t),
        "BC dynamic rank payload byte count overflow"
    );
    if (payload_end > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC dynamic rank payload exceeds size_t");
    }
    buffer.resize(static_cast<size_t>(payload_end));

    uint8_t *prefix_out = buffer.data() + rank_payload_offset;
    uint8_t *bitmap_out = buffer.data() + out_bitmap_offset;
    if (prefix_end < out_bitmap_offset) {
        std::memset(
            buffer.data() + prefix_end,
            0,
            static_cast<size_t>(out_bitmap_offset - prefix_end)
        );
    }

    uint32_t running = 0U;
    for (uint32_t block = 0U; block < prefix_count; ++block) {
        if (running > std::numeric_limits<RankPrefix>::max()) {
            throw std::logic_error("BC dynamic rank prefix running popcount exceeds uint16");
        }
        bc_store_u16_le(
            prefix_out + static_cast<size_t>(block) * sizeof(RankPrefix),
            static_cast<RankPrefix>(running)
        );
        const uint32_t block_first_bit = block * kBCRankPrefixBits;
        const uint32_t block_last_bit = std::min<uint32_t>(
            bitmap_len,
            block_first_bit + kBCRankPrefixBits
        );
        const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
        const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
        if (last_word_exclusive > word_count) {
            throw std::logic_error("BC dynamic rank bitmap is shorter than bitmap_len");
        }
        for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
            uint64_t value =
                state.bitmap_arena[bitmap_offset + word].load(std::memory_order_relaxed);
            if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
            }
            bc_store_u64_le(
                bitmap_out + static_cast<size_t>(word) * sizeof(uint64_t),
                value
            );
            running += popcount64(value);
        }
    }
    if (checked_u32_size(buffer.size(), "BC dynamic rank payload end exceeds uint32") <
        out_bitmap_offset) {
        throw std::logic_error("BC dynamic rank payload append did not reach bitmap offset");
    }
}

[[nodiscard]] static uint32_t bc_dynamic_count_live(const uint64_t *bitmap, uint32_t word_count, uint32_t bitmap_len) {
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

[[nodiscard]] static BCDynamicGroupedSlotRefs collect_dynamic_slot_refs_by_cell(
    const BCDynamicState &state,
    int thread_count
) {
    BCDynamicGroupedSlotRefs out;
    out.cell_begin.assign(static_cast<size_t>(state.cell_count) + 1U, 0U);

    const int effective_threads = std::max(1, thread_count);
    constexpr uint32_t kParallelCollectMinSlots = 65536U;
    constexpr uint32_t kParallelCollectMinCells = 512U;
#if defined(_OPENMP)
    if (
        effective_threads > 1 &&
        state.hash_capacity >= kParallelCollectMinSlots &&
        state.cell_count >= kParallelCollectMinCells
    ) {
        const uint32_t cell_stride = state.cell_count + 1U;
        std::vector<uint32_t> local_counts(
            static_cast<size_t>(effective_threads) * static_cast<size_t>(cell_stride),
            0U
        );
        std::atomic<int> error_code{0};

#pragma omp parallel num_threads(effective_threads)
        {
            const int tid = omp_get_thread_num();
            uint32_t *counts =
                local_counts.data() + static_cast<size_t>(tid) * static_cast<size_t>(cell_stride);
#pragma omp for schedule(static)
            for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(state.hash_capacity); ++slot_i) {
                const uint32_t slot = static_cast<uint32_t>(slot_i);
                const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
                if (cid == BCDynamicState::kEmptyCell) {
                    continue;
                }
                if (cid == BCDynamicState::kPendingCell) {
                    int expected = 0;
                    (void)error_code.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
                    continue;
                }
                if (cid >= state.cell_count) {
                    int expected = 0;
                    (void)error_code.compare_exchange_strong(expected, 2, std::memory_order_relaxed);
                    continue;
                }
                ++counts[static_cast<size_t>(cid) + 1U];
            }
        }

        const int count_error = error_code.load(std::memory_order_relaxed);
        if (count_error == 1) {
            throw std::runtime_error("BC dynamic finalize saw pending cell slot");
        }
        if (count_error == 2) {
            throw std::runtime_error("BC dynamic finalize saw cell id outside target matrix");
        }

#pragma omp parallel for schedule(static) num_threads(effective_threads) if(state.cell_count >= 4096U)
        for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(state.cell_count); ++cid_i) {
            const uint32_t cid = static_cast<uint32_t>(cid_i);
            uint64_t count = 0U;
            for (int tid = 0; tid < effective_threads; ++tid) {
                count += local_counts[
                    static_cast<size_t>(tid) * static_cast<size_t>(cell_stride) +
                    static_cast<size_t>(cid) + 1U
                ];
            }
            if (count > std::numeric_limits<uint32_t>::max()) {
                error_code.store(3, std::memory_order_relaxed);
                count = std::numeric_limits<uint32_t>::max();
            }
            out.cell_begin[static_cast<size_t>(cid) + 1U] = static_cast<uint32_t>(count);
        }
        if (error_code.load(std::memory_order_relaxed) == 3) {
            throw std::overflow_error("BC dynamic finalize per-cell bucket count overflow");
        }

        for (uint32_t cid = 0U; cid < state.cell_count; ++cid) {
            const uint64_t next =
                static_cast<uint64_t>(out.cell_begin[cid]) +
                static_cast<uint64_t>(out.cell_begin[static_cast<size_t>(cid) + 1U]);
            if (next > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC dynamic finalize grouped refs exceed uint32");
            }
            out.cell_begin[static_cast<size_t>(cid) + 1U] = static_cast<uint32_t>(next);
        }

        out.refs.resize(out.cell_begin[state.cell_count]);

#pragma omp parallel for schedule(static) num_threads(effective_threads) if(state.cell_count >= 4096U)
        for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(state.cell_count); ++cid_i) {
            const uint32_t cid = static_cast<uint32_t>(cid_i);
            uint32_t cursor = out.cell_begin[cid];
            for (int tid = 0; tid < effective_threads; ++tid) {
                const size_t base = static_cast<size_t>(tid) * static_cast<size_t>(cell_stride);
                const uint32_t count = local_counts[base + static_cast<size_t>(cid) + 1U];
                local_counts[base + static_cast<size_t>(cid)] = cursor;
                cursor += count;
            }
            if (cursor != out.cell_begin[static_cast<size_t>(cid) + 1U]) {
                error_code.store(4, std::memory_order_relaxed);
            }
        }
        if (error_code.load(std::memory_order_relaxed) == 4) {
            throw std::runtime_error("BC dynamic finalize grouped cell range underfilled");
        }

#pragma omp parallel num_threads(effective_threads)
        {
            const int tid = omp_get_thread_num();
            uint32_t *cursors =
                local_counts.data() + static_cast<size_t>(tid) * static_cast<size_t>(cell_stride);
#pragma omp for schedule(static)
            for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(state.hash_capacity); ++slot_i) {
                const uint32_t slot = static_cast<uint32_t>(slot_i);
                const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
                if (cid == BCDynamicState::kEmptyCell) {
                    continue;
                }
                if (cid == BCDynamicState::kPendingCell || cid >= state.cell_count) {
                    int expected = 0;
                    (void)error_code.compare_exchange_strong(expected, 5, std::memory_order_relaxed);
                    continue;
                }
                const uint32_t index = cursors[cid]++;
                if (index >= out.cell_begin[static_cast<size_t>(cid) + 1U]) {
                    int expected = 0;
                    (void)error_code.compare_exchange_strong(expected, 6, std::memory_order_relaxed);
                    continue;
                }
                out.refs[index] = BCDynamicSlotRef{
                    state.key_array[slot],
                    slot,
                    0U
                };
            }
        }

        const int fill_error = error_code.load(std::memory_order_relaxed);
        if (fill_error == 5) {
            throw std::runtime_error("BC dynamic finalize saw invalid cell slot during grouping");
        }
        if (fill_error == 6) {
            throw std::runtime_error("BC dynamic finalize grouped cell range overflow");
        }
        return out;
    }
#else
    (void)effective_threads;
#endif

    for (uint32_t slot = 0U; slot < state.hash_capacity; ++slot) {
        const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
        if (cid == BCDynamicState::kEmptyCell) {
            continue;
        }
        if (cid == BCDynamicState::kPendingCell) {
            throw std::runtime_error("BC dynamic finalize saw pending cell slot");
        }
        if (cid >= state.cell_count) {
            throw std::runtime_error("BC dynamic finalize saw cell id outside target matrix");
        }
        if (out.cell_begin[static_cast<size_t>(cid) + 1U] == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic finalize per-cell bucket count overflow");
        }
        ++out.cell_begin[static_cast<size_t>(cid) + 1U];
    }

    for (uint32_t cid = 0U; cid < state.cell_count; ++cid) {
        const uint64_t next =
            static_cast<uint64_t>(out.cell_begin[cid]) +
            static_cast<uint64_t>(out.cell_begin[static_cast<size_t>(cid) + 1U]);
        if (next > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic finalize grouped refs exceed uint32");
        }
        out.cell_begin[static_cast<size_t>(cid) + 1U] = static_cast<uint32_t>(next);
    }

    out.refs.resize(out.cell_begin[state.cell_count]);
    std::vector<uint32_t> write_cursor = out.cell_begin;
    for (uint32_t slot = 0U; slot < state.hash_capacity; ++slot) {
        const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
        if (cid == BCDynamicState::kEmptyCell) {
            continue;
        }
        if (cid == BCDynamicState::kPendingCell) {
            throw std::runtime_error("BC dynamic finalize saw pending cell slot during grouping");
        }
        if (cid >= state.cell_count) {
            throw std::runtime_error("BC dynamic finalize saw cell id outside target matrix during grouping");
        }
        const uint32_t index = write_cursor[cid]++;
        if (index >= out.cell_begin[static_cast<size_t>(cid) + 1U]) {
            throw std::runtime_error("BC dynamic finalize grouped cell range overflow");
        }
        out.refs[index] = BCDynamicSlotRef{
            state.key_array[slot],
            slot,
            0U
        };
    }
    for (uint32_t cid = 0U; cid < state.cell_count; ++cid) {
        if (write_cursor[cid] != out.cell_begin[static_cast<size_t>(cid) + 1U]) {
            throw std::runtime_error("BC dynamic finalize grouped cell range underfilled");
        }
    }
    return out;
}

static void sort_dynamic_slot_refs_by_key(
    std::vector<BCDynamicSlotRef> &refs,
    uint32_t begin,
    uint32_t end
) {
    if (end - begin < 2U) {
        return;
    }

    constexpr size_t kNativeKeyValueSortMinBuckets = 10000U;
    const size_t count = static_cast<size_t>(end - begin);
    if (count >= kNativeKeyValueSortMinBuckets) {
        if (auto keyvalue_sort = BC::detail::bc_resolve_keyvalue_sort_uint64_uint32(); keyvalue_sort != nullptr) {
            thread_local std::vector<uint64_t> sort_keys;
            thread_local std::vector<uint32_t> sort_slots;
            sort_keys.resize(count);
            sort_slots.resize(count);
            for (size_t i = 0U; i < count; ++i) {
                const BCDynamicSlotRef &ref = refs[static_cast<size_t>(begin) + i];
                sort_keys[i] = ref.key;
                sort_slots[i] = ref.slot;
            }
            keyvalue_sort(sort_keys.data(), sort_slots.data(), count, false);
            for (size_t i = 0U; i < count; ++i) {
                refs[static_cast<size_t>(begin) + i] = BCDynamicSlotRef{
                    sort_keys[i],
                    sort_slots[i],
                    0U
                };
            }
            return;
        }
    }

    std::sort(
        refs.begin() + static_cast<std::ptrdiff_t>(begin),
        refs.begin() + static_cast<std::ptrdiff_t>(end),
        [](const BCDynamicSlotRef &lhs, const BCDynamicSlotRef &rhs) {
            return lhs.key < rhs.key;
        }
    );
}

void set_dynamic_stats(
    BCResidentGenerationResult &result,
    const BCLut &lut,
    const BCDynamicState &state,
    uint32_t generation_retries
) {
    result.generation_retries = generation_retries;
    result.dynamic_hash_capacity = state.hash_capacity;
    result.dynamic_bitmap_words_reserved = state.reserved_bitmap_words;
    result.dynamic_bitmap_words_allocated =
        std::min<uint64_t>(
            state.bitmap_cursor_words.load(std::memory_order_relaxed),
            state.reserved_bitmap_words
        );

    uint64_t used_buckets = 0U;
    uint64_t used_words = 0U;
    for (uint32_t slot = 0U; slot < state.hash_capacity; ++slot) {
        const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
        if (cid == BCDynamicState::kEmptyCell) {
            continue;
        }
        if (cid == BCDynamicState::kPendingCell) {
            throw std::runtime_error("BC dynamic stats saw pending cell slot");
        }
        ++used_buckets;
        used_words += words_for_bits(bitmap_len_from_trusted_key(lut, state.key_array[slot]));
    }
    result.dynamic_bucket_slots_used = used_buckets;
    result.dynamic_bitmap_words_used = used_words;
}

void set_dynamic_capacity_stats(
    BCResidentGenerationResult &result,
    const BCDynamicState &state,
    uint32_t generation_retries
) {
    result.generation_retries = generation_retries;
    result.dynamic_hash_capacity = state.hash_capacity;
    result.dynamic_bitmap_words_reserved = state.reserved_bitmap_words;
    result.dynamic_bitmap_words_allocated =
        std::min<uint64_t>(
            state.bitmap_cursor_words.load(std::memory_order_relaxed),
            state.reserved_bitmap_words
        );
    result.dynamic_bucket_slots_used = 0U;
    result.dynamic_bitmap_words_used = 0U;
}

[[nodiscard]] static uint64_t count_dynamic_live_bits(const BCLut &lut, const BCDynamicState &state) {
    uint64_t live = 0U;
    for (uint32_t slot = 0U; slot < state.hash_capacity; ++slot) {
        const uint32_t cid = state.cell_array[slot].load(std::memory_order_relaxed);
        if (cid == BCDynamicState::kEmptyCell) {
            continue;
        }
        if (cid == BCDynamicState::kPendingCell) {
            throw std::runtime_error("BC dynamic live count saw pending cell slot");
        }
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, state.key_array[slot]);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t offset = state.bitmap_offset_array[slot];
        for (uint32_t word = 0U; word < word_count; ++word) {
            uint64_t bits = state.bitmap_arena[offset + word].load(std::memory_order_relaxed);
            if (word + 1U == word_count) {
                const uint32_t tail = bitmap_len & 63U;
                if (tail != 0U) {
                    bits &= (1ULL << tail) - 1ULL;
                }
            }
            live += static_cast<uint64_t>(__builtin_popcountll(bits));
        }
    }
    return live;
}

[[nodiscard]] static std::vector<FinalizedCellPayload> finalize_dynamic_state(
    const BCLut &lut,
    const BCDynamicState &state,
    int thread_count
) {
    BCDynamicGroupedSlotRefs grouped = collect_dynamic_slot_refs_by_cell(state, thread_count);
    std::vector<BCDynamicSlotRef> refs = std::move(grouped.refs);
    std::vector<uint32_t> cell_begin = std::move(grouped.cell_begin);

    std::vector<FinalizedCellPayload> payloads(state.cell_count);

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
            // Finalized cell lookup binary-searches bucket metadata by key, so
            // only the buckets within this cell need sorted order. There is no
            // ordering requirement across different cells.
            sort_dynamic_slot_refs_by_key(refs, begin, end);

            FinalizedCellPayload payload;
            payload.buckets.reserve(end - begin);
            uint64_t success_cursor = 0U;
            std::vector<uint64_t> bitmap;
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t slot = refs[i].slot;
                const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, refs[i].key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                const uint32_t bitmap_offset = state.bitmap_offset_array[slot];

                bitmap.resize(word_count);
                if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
                    throw std::out_of_range("BC dynamic finalize bitmap offset exceeds arena");
                }
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

struct BCDynamicCellMeasure {
    uint32_t bucket_count = 0U;
    uint32_t success_rows = 0U;
    uint64_t bitmap_words = 0U;
    uint64_t rank_payload_bytes = 0U;
};

[[nodiscard]] static uint32_t pack_dynamic_len_live(uint32_t bitmap_len, uint32_t live) {
    if (bitmap_len > std::numeric_limits<uint16_t>::max() ||
        live > std::numeric_limits<uint16_t>::max()) {
        throw std::overflow_error("BC dynamic bucket bitmap_len/live exceeds packed uint16");
    }
    return bitmap_len | (live << 16U);
}

[[nodiscard]] static uint32_t dynamic_ref_bitmap_len(const BCDynamicSlotRef &ref) noexcept {
    return ref.bitmap_len_live & 0xFFFFU;
}

[[nodiscard]] static uint32_t dynamic_ref_live_count(const BCDynamicSlotRef &ref) noexcept {
    return ref.bitmap_len_live >> 16U;
}

[[nodiscard]] static BCDynamicCellMeasure measure_dynamic_cell_payload(
    const BCLut &lut,
    const BCDynamicState &state,
    std::vector<BCDynamicSlotRef> &refs,
    uint32_t begin,
    uint32_t end
) {
    BCDynamicCellMeasure out;
    uint64_t rank_cursor = 0U;
    uint64_t success_cursor = 0U;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t slot = refs[i].slot;
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, refs[i].key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = state.bitmap_offset_array[slot];
        if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
            throw std::out_of_range("BC dynamic streaming measure bitmap offset exceeds arena");
        }
        uint32_t live = 0U;
        for (uint32_t word = 0U; word < word_count; ++word) {
            uint64_t bits = state.bitmap_arena[bitmap_offset + word].load(std::memory_order_relaxed);
            if (word + 1U == word_count) {
                const uint32_t tail = bitmap_len & 63U;
                if (tail != 0U) {
                    bits &= (1ULL << tail) - 1ULL;
                }
            }
            live += static_cast<uint32_t>(__builtin_popcountll(bits));
        }
        if (live == 0U) {
            refs[i].bitmap_len_live = pack_dynamic_len_live(bitmap_len, 0U);
            continue;
        }
        refs[i].bitmap_len_live = pack_dynamic_len_live(bitmap_len, live);

        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            (8U - (rank_cursor & 7U)) & 7U,
            "BC dynamic streaming rank alignment overflow"
        );
        if (rank_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming rank payload offset exceeds uint32");
        }
        const uint32_t rank_payload_offset = static_cast<uint32_t>(rank_cursor);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        rank_cursor = bc_checked_add_u64(
            out_bitmap_offset,
            static_cast<uint64_t>(word_count) * sizeof(uint64_t),
            "BC dynamic streaming rank payload byte count overflow"
        );
        ++out.bucket_count;
        out.bitmap_words = bc_checked_add_u64(
            out.bitmap_words,
            word_count,
            "BC dynamic streaming bitmap word count overflow"
        );
        success_cursor += live;
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming cell success_rows exceeds uint32");
        }
    }
    out.success_rows = static_cast<uint32_t>(success_cursor);
    out.rank_payload_bytes = rank_cursor;
    return out;
}

[[nodiscard]] static FinalizedCellPayload finalize_dynamic_cell_payload(
    const BCLut &lut,
    const BCDynamicState &state,
    const std::vector<BCDynamicSlotRef> &refs,
    uint32_t begin,
    uint32_t end
) {
    FinalizedCellPayload payload;
    if (begin == end) {
        return payload;
    }
    payload.buckets.reserve(end - begin);
    uint64_t success_cursor = 0U;
    std::vector<uint64_t> bitmap;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t slot = refs[i].slot;
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, refs[i].key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = state.bitmap_offset_array[slot];

        bitmap.resize(word_count);
        if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
            throw std::out_of_range("BC dynamic streaming finalize bitmap offset exceeds arena");
        }
        for (uint32_t word = 0U; word < word_count; ++word) {
            bitmap[word] = state.bitmap_arena[bitmap_offset + word].load(std::memory_order_relaxed);
        }
        const uint32_t live = bc_dynamic_count_live(bitmap.data(), word_count, bitmap_len);
        if (live == 0U) {
            continue;
        }

        const uint32_t payload_offset =
            checked_u32_size(payload.rank_payload.size(), "BC dynamic streaming rank payload offset exceeds uint32");
        const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
        bc_append_padding(payload.rank_payload, aligned_payload_offset - payload_offset);
        const uint32_t rank_payload_offset =
            checked_u32_size(payload.rank_payload.size(), "BC dynamic streaming aligned rank payload offset exceeds uint32");

        bc_append_prefix256_le(payload.rank_payload, bitmap.data(), word_count, bitmap_len);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        bc_append_padding(
            payload.rank_payload,
            out_bitmap_offset - checked_u32_size(
                payload.rank_payload.size(),
                "BC dynamic streaming prefix payload end exceeds uint32"
            )
        );
        bc_append_bitmap_le(payload.rank_payload, bitmap.data(), word_count);
        payload.buckets.push_back(BCBucketEntry{
            refs[i].key,
            rank_payload_offset,
            checked_u32_size(success_cursor, "BC dynamic streaming success row offset exceeds uint32")
        });
        success_cursor += live;
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming finalized cell success_rows exceeds uint32");
        }
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    return payload;
}

static void append_dynamic_cell_bucket_entries_le(
    const BCLut &lut,
    const BCDynamicState &state,
    const std::vector<BCDynamicSlotRef> &refs,
    uint32_t begin,
    uint32_t end,
    const BCPositionCellDescriptor &descriptor,
    std::vector<uint8_t> &out
) {
    out.clear();
    out.reserve(static_cast<size_t>(descriptor.bucket_count) * kBCPositionBucketEntryBytes);
    uint64_t rank_cursor = 0U;
    uint64_t success_cursor = 0U;
    uint32_t bucket_count = 0U;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t live = dynamic_ref_live_count(refs[i]);
        if (live == 0U) {
            continue;
        }
        const uint32_t bitmap_len = dynamic_ref_bitmap_len(refs[i]);
        const uint32_t word_count = words_for_bits(bitmap_len);

        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            (8U - (rank_cursor & 7U)) & 7U,
            "BC dynamic streaming bucket rank alignment overflow"
        );
        if (rank_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming bucket rank payload offset exceeds uint32");
        }
        const uint32_t rank_payload_offset = static_cast<uint32_t>(rank_cursor);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        rank_cursor = bc_checked_add_u64(
            out_bitmap_offset,
            static_cast<uint64_t>(word_count) * sizeof(uint64_t),
            "BC dynamic streaming bucket rank payload byte count overflow"
        );
        bc_append_bucket_entry(out, BCBucketEntry{
            refs[i].key,
            rank_payload_offset,
            checked_u32_size(success_cursor, "BC dynamic streaming bucket success row offset exceeds uint32")
        });
        success_cursor += live;
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming bucket success_rows exceeds uint32");
        }
        ++bucket_count;
    }
    if (bucket_count != descriptor.bucket_count ||
        success_cursor != descriptor.success_rows ||
        rank_cursor != descriptor.rank_payload_bytes) {
        throw std::logic_error("BC dynamic streaming bucket descriptor mismatch");
    }
}

static void append_dynamic_cell_rank_payload(
    const BCLut &lut,
    const BCDynamicState &state,
    const std::vector<BCDynamicSlotRef> &refs,
    uint32_t begin,
    uint32_t end,
    const BCPositionCellDescriptor &descriptor,
    std::vector<uint8_t> &out
) {
    out.clear();
    out.reserve(static_cast<size_t>(descriptor.rank_payload_bytes));
    uint32_t bucket_count = 0U;
    uint64_t success_cursor = 0U;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t slot = refs[i].slot;
        const uint32_t live = dynamic_ref_live_count(refs[i]);
        if (live == 0U) {
            continue;
        }
        const uint32_t bitmap_len = dynamic_ref_bitmap_len(refs[i]);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = state.bitmap_offset_array[slot];
        if (static_cast<uint64_t>(bitmap_offset) + word_count > state.reserved_bitmap_words) {
            throw std::out_of_range("BC dynamic streaming rank bitmap offset exceeds arena");
        }
        const uint32_t payload_offset =
            checked_u32_size(out.size(), "BC dynamic streaming rank payload offset exceeds uint32");
        const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
        bc_append_padding(out, aligned_payload_offset - payload_offset);
        const uint32_t rank_payload_offset =
            checked_u32_size(out.size(), "BC dynamic streaming rank aligned payload offset exceeds uint32");
        bc_append_rank_payload_le_from_atomic_arena(
            out,
            state,
            bitmap_offset,
            word_count,
            bitmap_len,
            rank_payload_offset
        );
        success_cursor += live;
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC dynamic streaming rank success_rows exceeds uint32");
        }
        ++bucket_count;
    }
    if (bucket_count != descriptor.bucket_count ||
        success_cursor != descriptor.success_rows ||
        out.size() != descriptor.rank_payload_bytes) {
        throw std::logic_error("BC dynamic streaming rank descriptor mismatch");
    }
}

class BCResidentAlignedWriteBuffer {
public:
    BCResidentAlignedWriteBuffer() = default;
    ~BCResidentAlignedWriteBuffer() {
        release();
    }

    BCResidentAlignedWriteBuffer(const BCResidentAlignedWriteBuffer &) = delete;
    BCResidentAlignedWriteBuffer &operator=(const BCResidentAlignedWriteBuffer &) = delete;

    void reset(uint64_t capacity, uint32_t alignment) {
        release();
        if (capacity == 0U) {
            return;
        }
        alignment_ = std::max<uint32_t>(
            alignment,
            static_cast<uint32_t>(alignof(std::max_align_t))
        );
#if defined(_WIN32)
        data_ = static_cast<uint8_t *>(_aligned_malloc(static_cast<size_t>(capacity), alignment_));
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
#else
        void *ptr = nullptr;
        if (posix_memalign(&ptr, alignment_, static_cast<size_t>(capacity)) != 0) {
            throw std::bad_alloc();
        }
        data_ = static_cast<uint8_t *>(ptr);
#endif
        capacity_ = capacity;
        size_ = 0U;
    }

    void release() noexcept {
        if (data_ != nullptr) {
#if defined(_WIN32)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
            data_ = nullptr;
        }
        capacity_ = 0U;
        size_ = 0U;
        alignment_ = 1U;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0U;
    }

    [[nodiscard]] uint64_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] uint64_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]] uint8_t *data() noexcept {
        return data_;
    }

    void clear() noexcept {
        size_ = 0U;
    }

    void append(const uint8_t *src, uint64_t bytes) {
        if (bytes > capacity_ - size_) {
            throw std::overflow_error("BC aligned write staging buffer append exceeds capacity");
        }
        std::memcpy(data_ + static_cast<size_t>(size_), src, static_cast<size_t>(bytes));
        size_ += bytes;
    }

private:
    uint8_t *data_ = nullptr;
    uint64_t capacity_ = 0U;
    uint64_t size_ = 0U;
    uint32_t alignment_ = 1U;
};

class BCDynamicSequentialWriteStager {
public:
    BCDynamicSequentialWriteStager(BCWritableFile &file, BCFileIOStats *stats)
        : file_(file), stats_(stats) {
        const uint32_t alignment = file_.preferred_write_alignment();
        for (BCResidentAlignedWriteBuffer &buffer : buffers_) {
            buffer.reset(kStagingBytes, alignment);
        }
    }

    void append(const void *data, uint64_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC dynamic streaming write append pointer is null");
        }
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            BCResidentAlignedWriteBuffer &buffer = active_buffer();
            const uint64_t available = kStagingBytes - buffer.size();
            if (available == 0U) {
                enqueue_active_buffer();
                continue;
            }
            const uint64_t take = std::min<uint64_t>(remaining, available);
            buffer.append(cursor, take);
            cursor += take;
            remaining -= take;
            if (buffer.size() == kStagingBytes) {
                enqueue_active_buffer();
            }
        }
    }

    void finish() {
        if (!active_buffer().empty()) {
            enqueue_active_buffer();
        }
        flush_batch();
    }

    [[nodiscard]] double write_seconds() const noexcept {
        return write_seconds_;
    }

private:
    static constexpr uint64_t kStagingBytes = 16ULL * 1024ULL * 1024ULL;
    static constexpr uint32_t kMaxQueuedWrites = 8U;

    void add_stats(const BCFileIOStats &src) {
        if (stats_ == nullptr) {
            return;
        }
        stats_->request_count += src.request_count;
        stats_->requested_bytes += src.requested_bytes;
        stats_->backend_io_count += src.backend_io_count;
        stats_->backend_bytes += src.backend_bytes;
    }

    [[nodiscard]] BCResidentAlignedWriteBuffer &active_buffer() {
        if (pending_count_ == kMaxQueuedWrites) {
            flush_batch();
        }
        return buffers_[pending_count_];
    }

    void enqueue_active_buffer() {
        BCResidentAlignedWriteBuffer &buffer = active_buffer();
        if (buffer.empty()) {
            return;
        }
        pending_requests_.push_back(BCFileWriteRequest{
            cursor_,
            buffer.data(),
            buffer.size()
        });
        cursor_ = bc_checked_add_u64(
            cursor_,
            buffer.size(),
            "BC dynamic streaming write cursor overflow"
        );
        ++pending_count_;
        if (pending_count_ == kMaxQueuedWrites) {
            flush_batch();
        }
    }

    void flush_batch() {
        if (pending_count_ == 0U) {
            return;
        }
        BCFileIOStats local_stats;
        const double write_begin = bc_now_seconds();
        file_.write_many(pending_requests_, stats_ == nullptr ? nullptr : &local_stats);
        write_seconds_ += bc_now_seconds() - write_begin;
        add_stats(local_stats);
        for (uint32_t i = 0U; i < pending_count_; ++i) {
            buffers_[i].clear();
        }
        pending_requests_.clear();
        pending_count_ = 0U;
    }

    BCWritableFile &file_;
    BCFileIOStats *stats_ = nullptr;
    std::array<BCResidentAlignedWriteBuffer, kMaxQueuedWrites> buffers_;
    std::vector<BCFileWriteRequest> pending_requests_;
    uint32_t pending_count_ = 0U;
    uint64_t cursor_ = 0U;
    double write_seconds_ = 0.0;
};

static void append_dynamic_bucket_stream_parallel(
    const BCLut &lut,
    const BCDynamicState &state,
    const std::vector<BCDynamicSlotRef> &refs,
    const std::vector<uint32_t> &cell_begin,
    const std::vector<BCPositionCellDescriptor> &descriptors,
    int thread_count,
    BCDynamicSequentialWriteStager &stager
) {
    constexpr uint64_t kMaxBatchBucketBytes = 64ULL * 1024ULL * 1024ULL;
    constexpr uint32_t kMaxBatchCells = 512U;
    const uint32_t cell_count =
        checked_u32_size(descriptors.size(), "BC dynamic streaming bucket descriptor count exceeds uint32");
    std::exception_ptr first_exception;
    for (CellId batch_begin = 0U; batch_begin < cell_count;) {
        uint64_t batch_bytes = 0U;
        CellId batch_end = batch_begin;
        while (batch_end < cell_count && batch_end - batch_begin < kMaxBatchCells) {
            const BCPositionCellDescriptor &descriptor = descriptors[batch_end];
            const uint64_t next_bytes = bc_checked_add_u64(
                batch_bytes,
                static_cast<uint64_t>(descriptor.bucket_count) * kBCPositionBucketEntryBytes,
                "BC dynamic streaming bucket batch byte count overflow"
            );
            if (batch_end != batch_begin && next_bytes > kMaxBatchBucketBytes) {
                break;
            }
            batch_bytes = next_bytes;
            ++batch_end;
        }
        if (batch_end == batch_begin) {
            ++batch_end;
        }

        const uint32_t batch_cells = batch_end - batch_begin;
        std::vector<std::vector<uint8_t>> batch_payloads(batch_cells);
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
        for (int64_t local_i = 0; local_i < static_cast<int64_t>(batch_cells); ++local_i) {
            try {
                const CellId cid = batch_begin + static_cast<CellId>(local_i);
                const BCPositionCellDescriptor &descriptor = descriptors[cid];
                if (descriptor.bucket_count == 0U) {
                    continue;
                }
                append_dynamic_cell_bucket_entries_le(
                    lut,
                    state,
                    refs,
                    cell_begin[cid],
                    cell_begin[cid + 1U],
                    descriptor,
                    batch_payloads[static_cast<size_t>(local_i)]
                );
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

        for (uint32_t local_i = 0U; local_i < batch_cells; ++local_i) {
            const std::vector<uint8_t> &payload = batch_payloads[local_i];
            if (!payload.empty()) {
                stager.append(payload.data(), payload.size());
            }
        }
        batch_begin = batch_end;
    }
}

static void append_dynamic_rank_payload_stream_parallel(
    const BCLut &lut,
    const BCDynamicState &state,
    const std::vector<BCDynamicSlotRef> &refs,
    const std::vector<uint32_t> &cell_begin,
    const std::vector<BCPositionCellDescriptor> &descriptors,
    int thread_count,
    BCDynamicSequentialWriteStager &stager
) {
    constexpr uint64_t kMaxBatchRankBytes = 64ULL * 1024ULL * 1024ULL;
    constexpr uint32_t kMaxBatchCells = 256U;
    const uint32_t cell_count =
        checked_u32_size(descriptors.size(), "BC dynamic streaming descriptor count exceeds uint32");
    std::exception_ptr first_exception;
    for (CellId batch_begin = 0U; batch_begin < cell_count;) {
        uint64_t batch_bytes = 0U;
        CellId batch_end = batch_begin;
        while (batch_end < cell_count && batch_end - batch_begin < kMaxBatchCells) {
            const BCPositionCellDescriptor &descriptor = descriptors[batch_end];
            const uint64_t next_bytes = bc_checked_add_u64(
                batch_bytes,
                descriptor.rank_payload_bytes,
                "BC dynamic streaming rank batch byte count overflow"
            );
            if (batch_end != batch_begin && next_bytes > kMaxBatchRankBytes) {
                break;
            }
            batch_bytes = next_bytes;
            ++batch_end;
        }
        if (batch_end == batch_begin) {
            ++batch_end;
        }

        const uint32_t batch_cells = batch_end - batch_begin;
        std::vector<std::vector<uint8_t>> batch_payloads(batch_cells);
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
        for (int64_t local_i = 0; local_i < static_cast<int64_t>(batch_cells); ++local_i) {
            try {
                const CellId cid = batch_begin + static_cast<CellId>(local_i);
                const BCPositionCellDescriptor &descriptor = descriptors[cid];
                if (descriptor.bucket_count == 0U) {
                    continue;
                }
                append_dynamic_cell_rank_payload(
                    lut,
                    state,
                    refs,
                    cell_begin[cid],
                    cell_begin[cid + 1U],
                    descriptor,
                    batch_payloads[static_cast<size_t>(local_i)]
                );
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

        for (uint32_t local_i = 0U; local_i < batch_cells; ++local_i) {
            const std::vector<uint8_t> &payload = batch_payloads[local_i];
            if (!payload.empty()) {
                stager.append(payload.data(), payload.size());
            }
        }
        batch_begin = batch_end;
    }
}

[[nodiscard]] static uint64_t write_dynamic_state_to_file_streaming(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCDynamicState &state,
    int thread_count,
    BCWritableFile &file,
    BCFileIOStats *stats,
    uint64_t &success_rows_out,
    uint64_t &bucket_slots_used_out,
    uint64_t &bitmap_words_used_out,
    double *write_seconds_out
) {
    BCDynamicGroupedSlotRefs grouped = collect_dynamic_slot_refs_by_cell(state, thread_count);
    std::vector<BCDynamicSlotRef> refs = std::move(grouped.refs);
    std::vector<uint32_t> cell_begin = std::move(grouped.cell_begin);

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
            // Cell-local key order only; descriptor offsets define inter-cell
            // layout, and keys from different cells are unrelated.
            sort_dynamic_slot_refs_by_key(refs, begin, end);
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

    std::vector<BCDynamicCellMeasure> measures(state.cell_count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t cid_i = 0; cid_i < static_cast<int64_t>(state.cell_count); ++cid_i) {
        try {
            const uint32_t cid = static_cast<uint32_t>(cid_i);
            const uint32_t begin = cell_begin[cid];
            const uint32_t end = cell_begin[cid + 1U];
            if (begin == end) {
                continue;
            }
            measures[cid] = measure_dynamic_cell_payload(lut, state, refs, begin, end);
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

    const uint64_t descriptor_count = state.cell_count;
    const uint64_t descriptor_bytes =
        descriptor_count * static_cast<uint64_t>(kBCPositionCellDescriptorBytes);
    const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis.family_count());
    const uint64_t descriptor_offset = bc_checked_add_u64(
        kBCPositionHeaderBytes,
        axis_coord_bytes,
        "BC dynamic streaming descriptor table offset overflow"
    );
    const uint64_t bucket_offset = bc_checked_add_u64(
        descriptor_offset,
        descriptor_bytes,
        "BC dynamic streaming bucket stream offset overflow"
    );

    std::vector<BCPositionCellDescriptor> descriptors(state.cell_count);
    uint64_t bucket_cursor = 0U;
    uint64_t rank_cursor = 0U;
    success_rows_out = 0U;
    bucket_slots_used_out = 0U;
    bitmap_words_used_out = 0U;
    for (CellId cid = 0U; cid < state.cell_count; ++cid) {
        const uint32_t begin = cell_begin[cid];
        const uint32_t end = cell_begin[cid + 1U];
        if (begin == end) {
            BCPositionCellDescriptor descriptor;
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
            descriptors[cid] = descriptor;
            continue;
        }
        const BCDynamicCellMeasure measure = measures[cid];
        if (measure.bucket_count == 0U) {
            BCPositionCellDescriptor descriptor;
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
            descriptors[cid] = descriptor;
            continue;
        }
        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = measure.bucket_count;
        descriptor.success_rows = measure.success_rows;
        descriptor.bucket_meta_offset = bucket_cursor;
        descriptor.rank_payload_offset = rank_cursor;
        descriptor.rank_payload_bytes = measure.rank_payload_bytes;
        descriptor.flags_or_padding = 0U;
        descriptors[cid] = descriptor;
        bucket_cursor = bc_checked_add_u64(
            bucket_cursor,
            static_cast<uint64_t>(measure.bucket_count) * kBCPositionBucketEntryBytes,
            "BC dynamic streaming bucket byte count overflow"
        );
        bucket_slots_used_out = bc_checked_add_u64(
            bucket_slots_used_out,
            measure.bucket_count,
            "BC dynamic streaming bucket slot count overflow"
        );
        bitmap_words_used_out = bc_checked_add_u64(
            bitmap_words_used_out,
            measure.bitmap_words,
            "BC dynamic streaming bitmap word count overflow"
        );
        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            measure.rank_payload_bytes,
            "BC dynamic streaming rank byte count overflow"
        );
        success_rows_out = bc_checked_add_u64(
            success_rows_out,
            measure.success_rows,
            "BC dynamic streaming success row count overflow"
        );
    }

    const uint64_t rank_offset = bc_checked_add_u64(
        bucket_offset,
        bucket_cursor,
        "BC dynamic streaming rank stream offset overflow"
    );
    const uint64_t logical_size = bc_checked_add_u64(
        rank_offset,
        rank_cursor,
        "BC dynamic streaming logical file size overflow"
    );

    BCPositionHeader header;
    header.family_unit = axis.family_unit();
    header.axis_base_coord = axis.axis_base_coord();
    header.family_count = axis.family_count();
    header.layer_sum = axis.layer_sum();
    header.axis_coord_table_bytes = axis_coord_bytes;
    header.descriptor_count = descriptor_count;
    header.descriptor_table_offset = descriptor_offset;
    header.descriptor_table_bytes = descriptor_bytes;
    header.bucket_meta_offset = bucket_offset;
    header.bucket_meta_bytes = bucket_cursor;
    header.rank_payload_offset = rank_offset;
    header.rank_payload_bytes = rank_cursor;

    std::vector<uint8_t> header_bytes;
    header_bytes.reserve(kBCPositionHeaderBytes);
    bc_append_header(header_bytes, header);
    std::vector<uint8_t> axis_coord_table;
    axis_coord_table.reserve(static_cast<size_t>(axis_coord_bytes));
    bc_append_axis_coord_table(axis_coord_table, axis);

    std::vector<uint8_t> descriptor_table;
    descriptor_table.reserve(static_cast<size_t>(descriptor_bytes));
    for (const BCPositionCellDescriptor &descriptor : descriptors) {
        bc_append_cell_descriptor(descriptor_table, descriptor);
    }

    if (stats != nullptr) {
        *stats = {};
    }
    file.prepare_full_overwrite(logical_size);
    BCDynamicSequentialWriteStager stager(file, stats);
    stager.append(header_bytes.data(), header_bytes.size());
    stager.append(axis_coord_table.data(), axis_coord_table.size());
    stager.append(descriptor_table.data(), descriptor_table.size());

    append_dynamic_bucket_stream_parallel(
        lut,
        state,
        refs,
        cell_begin,
        descriptors,
        thread_count,
        stager
    );

    append_dynamic_rank_payload_stream_parallel(
        lut,
        state,
        refs,
        cell_begin,
        descriptors,
        thread_count,
        stager
    );
    stager.finish();
    double write_seconds = stager.write_seconds();
    const double flush_begin = bc_now_seconds();
    file.flush();
    write_seconds += bc_now_seconds() - flush_begin;
    if (write_seconds_out != nullptr) {
        *write_seconds_out = write_seconds;
    }
    return logical_size;
}

} // namespace ResidentGenerationInternal

using namespace ResidentGenerationInternal;

struct BCResidentGenerationMutableLayer::Impl {
    BCFamilyTable axis;
    BCDynamicState state;

    Impl(BCFamilyTable axis_in, BCDynamicState state_in)
        : axis(std::move(axis_in)), state(std::move(state_in)) {}
};

struct BCResidentGenerationMutableLayerAccess {
    [[nodiscard]] static bool has_impl(const BCResidentGenerationMutableLayer &layer) noexcept {
        return static_cast<bool>(layer.impl_);
    }

    [[nodiscard]] static const BCFamilyTable &axis(const BCResidentGenerationMutableLayer &layer) {
        if (!layer.impl_) {
            throw std::logic_error("BC mutable generation layer is empty");
        }
        return layer.impl_->axis;
    }

    [[nodiscard]] static BCDynamicState &state(BCResidentGenerationMutableLayer &layer) {
        if (!layer.impl_) {
            throw std::logic_error("BC mutable generation layer is empty");
        }
        return layer.impl_->state;
    }

    [[nodiscard]] static std::unique_ptr<BCResidentGenerationMutableLayer> make(
        const BCFamilyTable &axis,
        BCDynamicState state
    ) {
        auto layer = std::make_unique<BCResidentGenerationMutableLayer>();
        layer->impl_ = std::make_unique<BCResidentGenerationMutableLayer::Impl>(
            axis,
            std::move(state)
        );
        return layer;
    }
};

BCResidentGenerationMutableLayer::BCResidentGenerationMutableLayer() = default;
BCResidentGenerationMutableLayer::~BCResidentGenerationMutableLayer() = default;
BCResidentGenerationMutableLayer::BCResidentGenerationMutableLayer(
    BCResidentGenerationMutableLayer &&
) noexcept = default;
BCResidentGenerationMutableLayer &BCResidentGenerationMutableLayer::operator=(
    BCResidentGenerationMutableLayer &&
) noexcept = default;

bool BCResidentGenerationMutableLayer::empty() const noexcept {
    return !impl_;
}

const BCFamilyTable &BCResidentGenerationMutableLayer::axis() const {
    return BCResidentGenerationMutableLayerAccess::axis(*this);
}

BCPositionCellLayout BCResidentGenerationMutableLayer::layout() const {
    return BCPositionCellLayout::from_serialized_axis(axis());
}

BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        options
    );
}

BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
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
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident generation dynamic_reserve_factor must be positive");
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    const BCCellMatrix target_matrix(target_axis);
    const uint32_t target_cell_count = target_matrix.cell_count();
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;

    BCResidentGenerationResult result;
    result.effective_threads = thread_count;
    const double generation_begin = bc_now_seconds();
    std::vector<BCThreadGenerationWorkspace> workspaces;
    BCDynamicState dynamic_state;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double scan_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        result = BCResidentGenerationResult{};
        result.effective_threads = thread_count;
        const double prepare_begin = bc_now_seconds();
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
            static_cast<uint64_t>(static_cast<double>(source_bitmap_words) * reserve_factor) + 512ULL * 64ULL,
            thread_count
        );
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        for (const BCResidentGenerationSource &source : sources) {
            const double phase_begin = bc_now_seconds();
            run_source_phase(
                lut,
                target_layout,
                source,
                options,
                thread_count,
                dynamic_state,
                workspaces,
                word_sums
            );
            scan_seconds += bc_now_seconds() - phase_begin;
        }
        work_seconds += bc_now_seconds() - work_begin;
        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            successful_retry = retry;
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
    result.prepare_seconds = prepare_seconds;
    result.work_seconds = work_seconds;
    result.cleanup_seconds = 0.0;
    result.scan_seconds = scan_seconds;

    add_workspace_stats(result, workspaces);
    result.source_boards_scanned = source_success_row_count_sum(sources);
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(result, lut, dynamic_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(result, dynamic_state, successful_retry);
    }
    finalize_dynamic_result(
        result,
        lut,
        target_axis,
        dynamic_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        nullptr
    );
    return result;
}

BCResidentGenerationResult generate_resident_position_layer_to_file(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const std::vector<BCResidentGenerationSource> &sources,
    BCWritableFile &output_file,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_to_file(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        sources,
        output_file,
        options
    );
}

BCResidentGenerationResult generate_resident_position_layer_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const std::vector<BCResidentGenerationSource> &sources,
    BCWritableFile &output_file,
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
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident generation dynamic_reserve_factor must be positive");
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    const BCCellMatrix target_matrix(target_axis);
    const uint32_t target_cell_count = target_matrix.cell_count();
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;

    BCResidentGenerationResult result;
    result.effective_threads = thread_count;
    const double generation_begin = bc_now_seconds();
    std::vector<BCThreadGenerationWorkspace> workspaces;
    BCDynamicState dynamic_state;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    double scan_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        result = BCResidentGenerationResult{};
        result.effective_threads = thread_count;
        const double prepare_begin = bc_now_seconds();
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
            static_cast<uint64_t>(static_cast<double>(source_bitmap_words) * reserve_factor) + 512ULL * 64ULL,
            thread_count
        );
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        for (const BCResidentGenerationSource &source : sources) {
            const double phase_begin = bc_now_seconds();
            run_source_phase(
                lut,
                target_layout,
                source,
                options,
                thread_count,
                dynamic_state,
                workspaces,
                word_sums
            );
            scan_seconds += bc_now_seconds() - phase_begin;
        }
        work_seconds += bc_now_seconds() - work_begin;
        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            successful_retry = retry;
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
    result.prepare_seconds = prepare_seconds;
    result.work_seconds = work_seconds;
    result.cleanup_seconds = 0.0;
    result.scan_seconds = scan_seconds;

    add_workspace_stats(result, workspaces);
    result.source_boards_scanned = source_success_row_count_sum(sources);
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(result, lut, dynamic_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(result, dynamic_state, successful_retry);
    }
    finalize_dynamic_result(
        result,
        lut,
        target_axis,
        dynamic_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        &output_file
    );
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

BCResidentGenerationResult generate_resident_position_layer(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentGenerationSource &source4,
    const BCResidentGenerationSource &source2,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer(
        lut,
        target_layout,
        std::vector<BCResidentGenerationSource>{source4, source2},
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_impl(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile *primary_output_file,
    BCWritableFile *secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident pair generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident pair generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident pair generation dynamic_reserve_factor must be positive");
    }

    const BCFamilyTable &primary_axis = primary_layout.serialization_axis();
    const BCFamilyTable *secondary_axis =
        secondary_layout != nullptr ? &secondary_layout->serialization_axis() : nullptr;
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
        if (!bc_family_axes_equal(carry_axis, primary_axis)) {
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
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
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
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;

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
        const double prepare_begin = bc_now_seconds();
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
                512ULL * 64ULL,
            thread_count
        );
        if (has_secondary) {
            secondary_state = make_bc_dynamic_state(
                secondary_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor) +
                    512ULL * 64ULL,
                thread_count
            );
        }
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
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
            primary_layout,
            secondary_layout,
            current,
            options,
            thread_count,
            primary_state,
            has_secondary ? &secondary_state : nullptr,
            primary_workspaces,
            has_secondary ? &secondary_workspaces : nullptr,
            word_sums
        );
        work_seconds += bc_now_seconds() - work_begin;

        const bool primary_overflow = primary_state.overflowed.load(std::memory_order_acquire);
        const bool secondary_overflow =
            has_secondary && secondary_state.overflowed.load(std::memory_order_acquire);
        if (!primary_overflow && !secondary_overflow) {
            successful_retry = retry;
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
    pair.primary.prepare_seconds = prepare_seconds;
    pair.primary.work_seconds = work_seconds;
    pair.primary.cleanup_seconds = 0.0;
    if (has_secondary) {
        pair.secondary.prepare_seconds = prepare_seconds;
        pair.secondary.work_seconds = work_seconds;
        pair.secondary.cleanup_seconds = 0.0;
    }
    add_workspace_stats(pair.primary, primary_workspaces);
    if (has_secondary) {
        add_workspace_stats(pair.secondary, secondary_workspaces);
    }
    pair.primary.source_boards_scanned = position_success_row_count(current);
    pair.current_boards_scanned = pair.primary.source_boards_scanned;
    pair.shared_generation_seconds = generation_seconds;
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(pair.primary, lut, primary_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(pair.primary, primary_state, successful_retry);
    }
    if (has_secondary) {
        if (options.collect_dynamic_state_stats) {
            set_dynamic_stats(pair.secondary, lut, secondary_state, successful_retry);
        } else {
            set_dynamic_capacity_stats(pair.secondary, secondary_state, successful_retry);
        }
    }

    finalize_dynamic_result(
        pair.primary,
        lut,
        primary_axis,
        primary_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        primary_output_file
    );
    if (has_secondary) {
        finalize_dynamic_result(
            pair.secondary,
            lut,
            *secondary_axis,
            secondary_state,
            options,
            thread_count,
            total_begin,
            generation_seconds,
            secondary_output_file
        );
    }
    pair.total_pair_compute_seconds =
        generation_seconds +
        pair.primary.finalize_seconds +
        (has_secondary ? pair.secondary.finalize_seconds : 0.0);
    return pair;
}

BCResidentGenerationPairResult generate_resident_position_layer_pair(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCFamilyTable *secondary_axis,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_resident_position_layer_pair_impl(
        lut,
        primary_layout,
        current,
        carry_to_primary,
        secondary_layout ? &*secondary_layout : nullptr,
        nullptr,
        nullptr,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_pair_impl(
        lut,
        primary_layout,
        current,
        carry_to_primary,
        secondary_layout,
        nullptr,
        nullptr,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    BCWritableFile *secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    if (secondary_axis != nullptr && secondary_output_file == nullptr) {
        throw std::invalid_argument("BC resident pair file output requires secondary_output_file");
    }
    if (secondary_axis == nullptr && secondary_output_file != nullptr) {
        throw std::invalid_argument("BC resident pair file output got secondary file without secondary axis");
    }
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_resident_position_layer_pair_impl(
        lut,
        primary_layout,
        current,
        carry_to_primary,
        secondary_layout ? &*secondary_layout : nullptr,
        &primary_output_file,
        secondary_output_file,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader *carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    BCWritableFile *secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    if (secondary_layout != nullptr && secondary_output_file == nullptr) {
        throw std::invalid_argument("BC resident pair file output requires secondary_output_file");
    }
    if (secondary_layout == nullptr && secondary_output_file != nullptr) {
        throw std::invalid_argument("BC resident pair file output got secondary file without secondary axis");
    }
    return generate_resident_position_layer_pair_impl(
        lut,
        primary_layout,
        current,
        carry_to_primary,
        secondary_layout,
        &primary_output_file,
        secondary_output_file,
        options
    );
}

static BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_impl(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile *primary_output_file,
    BCWritableFile *terminal_secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC resident mutable pair generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC resident mutable pair generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC resident mutable pair generation dynamic_reserve_factor must be positive");
    }
    if (secondary_layout == nullptr && terminal_secondary_output_file != nullptr) {
        throw std::invalid_argument("BC resident mutable pair got secondary file without secondary axis");
    }

    const BCFamilyTable &primary_axis = primary_layout.serialization_axis();
    const BCFamilyTable *secondary_axis =
        secondary_layout != nullptr ? &secondary_layout->serialization_axis() : nullptr;
    const BCFamilyTable &current_axis = current.axis();
    if (current_axis.family_unit() != primary_axis.family_unit()) {
        throw std::invalid_argument("BC resident mutable pair current/primary family_unit mismatch");
    }
    if (static_cast<uint32_t>(primary_axis.total_coord()) !=
        static_cast<uint32_t>(current_axis.total_coord()) + 1U) {
        throw std::invalid_argument("BC resident mutable pair primary total_coord must equal current + 1");
    }
    if (carry_to_primary && !carry_to_primary->empty()) {
        const BCFamilyTable &carry_axis = carry_to_primary->axis();
        if (!bc_family_axes_equal(carry_axis, primary_axis)) {
            throw std::invalid_argument("BC resident mutable carry axis must match primary axis");
        }
    }

    const bool has_secondary = secondary_axis != nullptr;
    if (has_secondary) {
        if (current_axis.family_unit() != secondary_axis->family_unit()) {
            throw std::invalid_argument("BC resident mutable pair current/secondary family_unit mismatch");
        }
        if (static_cast<uint32_t>(secondary_axis->total_coord()) !=
            static_cast<uint32_t>(current_axis.total_coord()) + 2U) {
            throw std::invalid_argument("BC resident mutable pair secondary total_coord must equal current + 2");
        }
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCCellMatrix primary_matrix(primary_axis);
    const BCCellMatrix secondary_matrix = has_secondary
        ? BCCellMatrix(*secondary_axis)
        : BCCellMatrix(primary_axis);
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
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
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    const bool has_mutable_carry = carry_to_primary && !carry_to_primary->empty();

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
    const double generation_begin = bc_now_seconds();
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        pair.primary = BCResidentGenerationResult{};
        pair.secondary = BCResidentGenerationResult{};
        pair.primary.effective_threads = thread_count;
        pair.secondary.effective_threads = thread_count;

        const double prepare_begin = bc_now_seconds();
        prepare_workspaces(primary_workspaces);
        if (has_secondary) {
            prepare_workspaces(secondary_workspaces);
        } else {
            secondary_workspaces.clear();
        }

        if (has_mutable_carry) {
            primary_state = std::move(BCResidentGenerationMutableLayerAccess::state(*carry_to_primary));
            if (primary_state.cell_count != primary_matrix.cell_count()) {
                throw std::invalid_argument("BC resident mutable carry cell count mismatch");
            }
        } else {
            primary_state = make_bc_dynamic_state(
                primary_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor) +
                    512ULL * 64ULL,
                thread_count
            );
        }

        if (has_secondary) {
            const uint64_t future_bucket_estimate =
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor * 2.0) + 4096ULL;
            const uint64_t future_bitmap_estimate =
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor * 2.0) +
                512ULL * 64ULL;
            secondary_state = make_bc_dynamic_state(
                secondary_matrix.cell_count(),
                future_bucket_estimate,
                future_bitmap_estimate,
                thread_count
            );
        }
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        run_source_pair_phase(
            lut,
            primary_layout,
            secondary_layout,
            current,
            options,
            thread_count,
            primary_state,
            has_secondary ? &secondary_state : nullptr,
            primary_workspaces,
            has_secondary ? &secondary_workspaces : nullptr,
            word_sums
        );
        work_seconds += bc_now_seconds() - work_begin;

        const bool primary_overflow = primary_state.overflowed.load(std::memory_order_acquire);
        const bool secondary_overflow =
            has_secondary && secondary_state.overflowed.load(std::memory_order_acquire);
        if (!primary_overflow && !secondary_overflow) {
            successful_retry = retry;
            generated = true;
            break;
        }
        if (has_mutable_carry) {
            throw std::runtime_error(
                "BC resident mutable carry generation overflowed; increase dynamic_reserve_factor"
            );
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC resident mutable pair generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    pair.primary.scan_seconds = generation_seconds;
    pair.primary.prepare_seconds = prepare_seconds;
    pair.primary.work_seconds = work_seconds;
    pair.primary.cleanup_seconds = 0.0;
    if (has_secondary) {
        pair.secondary.scan_seconds = generation_seconds;
        pair.secondary.prepare_seconds = prepare_seconds;
        pair.secondary.work_seconds = work_seconds;
        pair.secondary.cleanup_seconds = 0.0;
    }
    add_workspace_stats(pair.primary, primary_workspaces);
    if (has_secondary) {
        add_workspace_stats(pair.secondary, secondary_workspaces);
    }
    pair.primary.source_boards_scanned = position_success_row_count(current);
    pair.current_boards_scanned = pair.primary.source_boards_scanned;
    pair.shared_generation_seconds = generation_seconds;
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(pair.primary, lut, primary_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(pair.primary, primary_state, successful_retry);
    }
    if (has_secondary) {
        if (options.collect_dynamic_state_stats) {
            set_dynamic_stats(pair.secondary, lut, secondary_state, successful_retry);
        } else {
            set_dynamic_capacity_stats(pair.secondary, secondary_state, successful_retry);
        }
    }

    finalize_dynamic_result(
        pair.primary,
        lut,
        primary_axis,
        primary_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        primary_output_file
    );
    if (has_secondary) {
        if (terminal_secondary_output_file != nullptr) {
            finalize_dynamic_result(
                pair.secondary,
                lut,
                *secondary_axis,
                secondary_state,
                options,
                thread_count,
                total_begin,
                generation_seconds,
                terminal_secondary_output_file
            );
        } else {
            if (options.collect_mutable_output_stats) {
                pair.secondary.output_success_rows = count_dynamic_live_bits(lut, secondary_state);
            }
            pair.secondary.generation_seconds = generation_seconds;
            pair.secondary.compute_seconds = generation_seconds;
            pair.secondary.total_seconds = bc_now_seconds() - total_begin;
            pair.secondary_carry = BCResidentGenerationMutableLayerAccess::make(
                *secondary_axis,
                std::move(secondary_state)
            );
        }
    }
    pair.total_pair_compute_seconds =
        generation_seconds +
        pair.primary.finalize_seconds +
        (has_secondary ? pair.secondary.finalize_seconds : 0.0);
    return pair;
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_resident_position_layer_pair_with_mutable_carry_impl(
        lut,
        primary_layout,
        current,
        std::move(carry_to_primary),
        secondary_layout ? &*secondary_layout : nullptr,
        nullptr,
        nullptr,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_pair_with_mutable_carry_impl(
        lut,
        primary_layout,
        current,
        std::move(carry_to_primary),
        secondary_layout,
        nullptr,
        nullptr,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    BCWritableFile *terminal_secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_resident_position_layer_pair_with_mutable_carry_impl(
        lut,
        primary_layout,
        current,
        std::move(carry_to_primary),
        secondary_layout ? &*secondary_layout : nullptr,
        &primary_output_file,
        terminal_secondary_output_file,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionLayerReader &current,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    BCWritableFile *terminal_secondary_output_file,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_position_layer_pair_with_mutable_carry_impl(
        lut,
        primary_layout,
        current,
        std::move(carry_to_primary),
        secondary_layout,
        &primary_output_file,
        terminal_secondary_output_file,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCFamilyTable &primary_axis,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable *secondary_axis,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options
) {
    const BCPositionCellLayout primary_layout =
        BCPositionCellLayout::from_serialized_axis(primary_axis);
    const std::optional<BCPositionCellLayout> secondary_layout =
        secondary_axis != nullptr
            ? std::optional<BCPositionCellLayout>(
                  BCPositionCellLayout::from_serialized_axis(*secondary_axis)
              )
            : std::nullopt;
    return generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(
        lut,
        primary_layout,
        current,
        current_cell_chunk_size,
        std::move(carry_to_primary),
        secondary_layout ? &*secondary_layout : nullptr,
        primary_output_file,
        options
    );
}

BCResidentGenerationPairResult generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(
    const BCLut &lut,
    const BCPositionCellLayout &primary_layout,
    const BCPositionStreamingReader &current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCPositionCellLayout *secondary_layout,
    BCWritableFile &primary_output_file,
    const BCResidentGenerationOptions &options
) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC SingleChunk pair generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC SingleChunk pair generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC SingleChunk pair generation dynamic_reserve_factor must be positive");
    }

    const BCFamilyTable &primary_axis = primary_layout.serialization_axis();
    const BCFamilyTable *secondary_axis =
        secondary_layout != nullptr ? &secondary_layout->serialization_axis() : nullptr;
    const BCFamilyTable &current_axis = current.axis();
    if (current_axis.family_unit() != primary_axis.family_unit()) {
        throw std::invalid_argument("BC SingleChunk pair current/primary family_unit mismatch");
    }
    if (static_cast<uint32_t>(primary_axis.total_coord()) !=
        static_cast<uint32_t>(current_axis.total_coord()) + 1U) {
        throw std::invalid_argument("BC SingleChunk pair primary total_coord must equal current + 1");
    }
    if (carry_to_primary && !carry_to_primary->empty()) {
        const BCFamilyTable &carry_axis = carry_to_primary->axis();
        if (!bc_family_axes_equal(carry_axis, primary_axis)) {
            throw std::invalid_argument("BC SingleChunk mutable carry axis must match primary axis");
        }
    }

    const bool has_secondary = secondary_axis != nullptr;
    if (has_secondary) {
        if (current_axis.family_unit() != secondary_axis->family_unit()) {
            throw std::invalid_argument("BC SingleChunk pair current/secondary family_unit mismatch");
        }
        if (static_cast<uint32_t>(secondary_axis->total_coord()) !=
            static_cast<uint32_t>(current_axis.total_coord()) + 2U) {
            throw std::invalid_argument("BC SingleChunk pair secondary total_coord must equal current + 2");
        }
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCCellMatrix primary_matrix(primary_axis);
    const BCCellMatrix secondary_matrix = has_secondary
        ? BCCellMatrix(*secondary_axis)
        : BCCellMatrix(primary_axis);
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;
    uint64_t canonical_warmup = 0U;
    CanonicalBatch::canonicalize_inplace(&canonical_warmup, 1U, options.canonical_symm_mode);

    BCResidentGenerationPairResult pair;
    pair.has_secondary = has_secondary;
    pair.primary.effective_threads = thread_count;
    pair.secondary.effective_threads = thread_count;

    std::vector<BCThreadGenerationWorkspace> primary_workspaces;
    std::vector<BCThreadGenerationWorkspace> secondary_workspaces;
    BCDynamicState primary_state;
    BCDynamicState secondary_state;
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    BCCellLoadStats successful_load_stats;
    double successful_load_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    const bool has_mutable_carry = carry_to_primary && !carry_to_primary->empty();

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
    const double generation_begin = bc_now_seconds();
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        pair.primary = BCResidentGenerationResult{};
        pair.secondary = BCResidentGenerationResult{};
        pair.primary.effective_threads = thread_count;
        pair.secondary.effective_threads = thread_count;

        const double prepare_begin = bc_now_seconds();
        prepare_workspaces(primary_workspaces);
        if (has_secondary) {
            prepare_workspaces(secondary_workspaces);
        } else {
            secondary_workspaces.clear();
        }

        if (has_mutable_carry) {
            primary_state = std::move(BCResidentGenerationMutableLayerAccess::state(*carry_to_primary));
            if (primary_state.cell_count != primary_matrix.cell_count()) {
                throw std::invalid_argument("BC SingleChunk mutable carry cell count mismatch");
            }
        } else {
            primary_state = make_bc_dynamic_state(
                primary_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor) +
                    512ULL * 64ULL,
                thread_count
            );
        }

        if (has_secondary) {
            secondary_state = make_bc_dynamic_state(
                secondary_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor * 2.0) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor * 2.0) +
                    512ULL * 64ULL,
                thread_count
            );
        }
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        BCCellLoadStats attempt_load_stats;
        double attempt_load_seconds = 0.0;
        run_streaming_source_pair_phase(
            lut,
            primary_layout,
            secondary_layout,
            current,
            current_cell_chunk_size,
            options,
            thread_count,
            primary_state,
            has_secondary ? &secondary_state : nullptr,
            primary_workspaces,
            has_secondary ? &secondary_workspaces : nullptr,
            word_sums,
            attempt_load_stats,
            attempt_load_seconds
        );
        work_seconds += bc_now_seconds() - work_begin;

        const bool primary_overflow = primary_state.overflowed.load(std::memory_order_acquire);
        const bool secondary_overflow =
            has_secondary && secondary_state.overflowed.load(std::memory_order_acquire);
        if (!primary_overflow && !secondary_overflow) {
            successful_retry = retry;
            successful_load_stats = attempt_load_stats;
            successful_load_seconds = attempt_load_seconds;
            generated = true;
            break;
        }
        if (has_mutable_carry) {
            throw std::runtime_error(
                "BC SingleChunk mutable carry generation overflowed; increase dynamic_reserve_factor"
            );
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC SingleChunk pair generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    pair.primary.scan_seconds = generation_seconds;
    pair.primary.prepare_seconds = prepare_seconds;
    pair.primary.work_seconds = work_seconds;
    pair.primary.cleanup_seconds = 0.0;
    pair.primary.source_position_load_seconds = successful_load_seconds;
    pair.primary.source_position_load_requested_extents = successful_load_stats.requested_extents;
    pair.primary.source_position_load_coalesced_extents = successful_load_stats.coalesced_extents;
    pair.primary.source_position_load_requested_bytes = successful_load_stats.requested_bytes;
    pair.primary.source_position_load_read_bytes = successful_load_stats.read_bytes;
    pair.primary.source_position_load_backend_read_ops = successful_load_stats.backend_read_ops;
    pair.primary.source_position_load_backend_read_bytes = successful_load_stats.backend_read_bytes;
    if (has_secondary) {
        pair.secondary.scan_seconds = generation_seconds;
        pair.secondary.prepare_seconds = prepare_seconds;
        pair.secondary.work_seconds = work_seconds;
        pair.secondary.cleanup_seconds = 0.0;
    }
    add_workspace_stats(pair.primary, primary_workspaces);
    if (has_secondary) {
        add_workspace_stats(pair.secondary, secondary_workspaces);
    }
    pair.primary.source_boards_scanned = position_success_row_count(current);
    pair.current_boards_scanned = pair.primary.source_boards_scanned;
    pair.shared_generation_seconds = generation_seconds;
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(pair.primary, lut, primary_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(pair.primary, primary_state, successful_retry);
    }
    if (has_secondary) {
        if (options.collect_dynamic_state_stats) {
            set_dynamic_stats(pair.secondary, lut, secondary_state, successful_retry);
        } else {
            set_dynamic_capacity_stats(pair.secondary, secondary_state, successful_retry);
        }
    }

    finalize_dynamic_result(
        pair.primary,
        lut,
        primary_axis,
        primary_state,
        options,
        thread_count,
        total_begin,
        generation_seconds,
        &primary_output_file
    );
    if (has_secondary) {
        if (options.collect_mutable_output_stats) {
            pair.secondary.output_success_rows = count_dynamic_live_bits(lut, secondary_state);
        }
        pair.secondary.generation_seconds = generation_seconds;
        pair.secondary.compute_seconds = generation_seconds;
        pair.secondary.total_seconds = bc_now_seconds() - total_begin;
        pair.secondary_carry = BCResidentGenerationMutableLayerAccess::make(
            *secondary_axis,
            std::move(secondary_state)
        );
    }
    pair.total_pair_compute_seconds = generation_seconds + pair.primary.finalize_seconds;
    return pair;
}

BCResidentMutableGenerationResult generate_resident_mutable_layer_from_streaming_source(
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    const BCResidentStreamingGenerationSource &source,
    std::unique_ptr<BCResidentGenerationMutableLayer> initial_mutable,
    const BCResidentGenerationOptions &options
) {
    return generate_resident_mutable_layer_from_streaming_source(
        lut,
        BCPositionCellLayout::from_serialized_axis(target_axis),
        source,
        std::move(initial_mutable),
        options
    );
}

BCResidentMutableGenerationResult generate_resident_mutable_layer_from_streaming_source(
    const BCLut &lut,
    const BCPositionCellLayout &target_layout,
    const BCResidentStreamingGenerationSource &source,
    std::unique_ptr<BCResidentGenerationMutableLayer> initial_mutable,
    const BCResidentGenerationOptions &options
) {
    if (source.position == nullptr) {
        throw std::invalid_argument("BC mutable streaming generation source position is null");
    }
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC mutable streaming generation canonical_batch_size must be non-zero");
    }
    if (options.pending_insert_buffer_size == 0U) {
        throw std::invalid_argument("BC mutable streaming generation pending_insert_buffer_size must be non-zero");
    }
    if (!(options.dynamic_reserve_factor > 0.0)) {
        throw std::invalid_argument("BC mutable streaming generation dynamic_reserve_factor must be positive");
    }
    const BCFamilyTable &target_axis = target_layout.serialization_axis();
    if (initial_mutable && !initial_mutable->empty()) {
        const BCFamilyTable &initial_axis = initial_mutable->axis();
        if (!bc_family_axes_equal(initial_axis, target_axis)) {
            throw std::invalid_argument("BC mutable streaming generation initial mutable axis mismatch");
        }
    }

    const double total_begin = bc_now_seconds();
    const int thread_count = effective_thread_count(options);
    const BCCellMatrix target_matrix(target_axis);
    const BCWordSumTable word_sums_storage =
        build_word_sum_table_if_needed(lut, resident_tile_sum_values(options));
    const BCWordSumTable *word_sums =
        word_sums_storage.empty() ? nullptr : &word_sums_storage;
    uint64_t canonical_warmup = 0U;
    CanonicalBatch::canonicalize_inplace(&canonical_warmup, 1U, options.canonical_symm_mode);

    BCResidentMutableGenerationResult out;
    out.result.effective_threads = thread_count;

    auto prepare_workspaces = [&](std::vector<BCThreadGenerationWorkspace> &workspaces) {
        workspaces.clear();
        workspaces.resize(static_cast<size_t>(thread_count));
        for (BCThreadGenerationWorkspace &workspace : workspaces) {
            workspace.canonical_buffer.reserve(options.canonical_batch_size);
            workspace.pending_encoded.reserve(options.pending_insert_buffer_size);
            workspace.resolved_encoded.reserve(options.pending_insert_buffer_size);
        }
    };

    const bool has_initial_mutable = initial_mutable && !initial_mutable->empty();
    const uint64_t current_buckets = position_bucket_count(*source.position);
    const uint64_t current_bitmap_words = position_rank_payload_word_estimate(*source.position);
    constexpr uint32_t kMaxGenerationRetries = 6U;
    double reserve_factor = options.dynamic_reserve_factor;
    double prepare_seconds = 0.0;
    double work_seconds = 0.0;
    BCCellLoadStats successful_load_stats;
    double successful_load_seconds = 0.0;
    bool generated = false;
    uint32_t successful_retry = 0U;
    BCDynamicState dynamic_state;
    std::vector<BCThreadGenerationWorkspace> workspaces;

    const double generation_begin = bc_now_seconds();
    for (uint32_t retry = 0U; retry <= kMaxGenerationRetries; ++retry) {
        out.result = BCResidentGenerationResult{};
        out.result.effective_threads = thread_count;
        const double prepare_begin = bc_now_seconds();
        prepare_workspaces(workspaces);
        if (has_initial_mutable) {
            dynamic_state = std::move(BCResidentGenerationMutableLayerAccess::state(*initial_mutable));
            if (dynamic_state.cell_count != target_matrix.cell_count()) {
                throw std::invalid_argument("BC mutable streaming generation cell count mismatch");
            }
        } else {
            dynamic_state = make_bc_dynamic_state(
                target_matrix.cell_count(),
                static_cast<uint64_t>(static_cast<double>(current_buckets) * reserve_factor) + 4096ULL,
                static_cast<uint64_t>(static_cast<double>(current_bitmap_words) * reserve_factor) +
                    512ULL * 64ULL,
                thread_count
            );
        }
        prepare_seconds += bc_now_seconds() - prepare_begin;

        const double work_begin = bc_now_seconds();
        BCCellLoadStats attempt_load_stats;
        double attempt_load_seconds = 0.0;
        run_streaming_source_delta_phase(
            lut,
            target_layout,
            *source.position,
            source.cell_chunk_size,
            source.spawn_tile_rank,
            source.delta_coord,
            options,
            thread_count,
            dynamic_state,
            workspaces,
            word_sums,
            attempt_load_stats,
            attempt_load_seconds
        );
        work_seconds += bc_now_seconds() - work_begin;

        if (!dynamic_state.overflowed.load(std::memory_order_acquire)) {
            successful_retry = retry;
            successful_load_stats = attempt_load_stats;
            successful_load_seconds = attempt_load_seconds;
            generated = true;
            break;
        }
        if (has_initial_mutable) {
            throw std::runtime_error(
                "BC mutable streaming generation overflowed existing mutable state; increase carry reserve"
            );
        }
        reserve_factor *= 2.0;
    }
    if (!generated) {
        throw std::runtime_error("BC mutable streaming generation dynamic state exceeded retry limit");
    }

    const double generation_seconds = bc_now_seconds() - generation_begin;
    out.result.generation_seconds = generation_seconds;
    out.result.scan_seconds = generation_seconds;
    out.result.prepare_seconds = prepare_seconds;
    out.result.work_seconds = work_seconds;
    out.result.cleanup_seconds = 0.0;
    out.result.source_position_load_seconds = successful_load_seconds;
    out.result.source_position_load_requested_extents = successful_load_stats.requested_extents;
    out.result.source_position_load_coalesced_extents = successful_load_stats.coalesced_extents;
    out.result.source_position_load_requested_bytes = successful_load_stats.requested_bytes;
    out.result.source_position_load_read_bytes = successful_load_stats.read_bytes;
    out.result.source_position_load_backend_read_ops = successful_load_stats.backend_read_ops;
    out.result.source_position_load_backend_read_bytes = successful_load_stats.backend_read_bytes;
    add_workspace_stats(out.result, workspaces);
    out.result.source_boards_scanned = position_success_row_count(*source.position);
    if (options.collect_dynamic_state_stats) {
        set_dynamic_stats(out.result, lut, dynamic_state, successful_retry);
    } else {
        set_dynamic_capacity_stats(out.result, dynamic_state, successful_retry);
    }
    if (options.collect_mutable_output_stats) {
        out.result.output_success_rows = count_dynamic_live_bits(lut, dynamic_state);
    }
    out.result.compute_seconds = generation_seconds;
    out.result.total_seconds = bc_now_seconds() - total_begin;
    out.mutable_layer = BCResidentGenerationMutableLayerAccess::make(target_axis, std::move(dynamic_state));
    return out;
}

BCResidentGenerationResult finalize_resident_mutable_layer_to_file(
    const BCLut &lut,
    std::unique_ptr<BCResidentGenerationMutableLayer> mutable_layer,
    BCWritableFile &output_file,
    BCResidentGenerationResult generation_result,
    const BCResidentGenerationOptions &options
) {
    if (!mutable_layer || mutable_layer->empty()) {
        throw std::invalid_argument("BC mutable finalize requires a non-empty mutable layer");
    }
    const BCFamilyTable axis = mutable_layer->axis();
    BCDynamicState state = std::move(BCResidentGenerationMutableLayerAccess::state(*mutable_layer));
    const int thread_count =
        generation_result.effective_threads != 0
            ? generation_result.effective_threads
            : effective_thread_count(options);
    generation_result.output_success_rows = 0U;
    const double total_begin = bc_now_seconds() - generation_result.generation_seconds;
    finalize_dynamic_result(
        generation_result,
        lut,
        axis,
        state,
        options,
        thread_count,
        total_begin,
        generation_result.generation_seconds,
        &output_file
    );
    return generation_result;
}

} // namespace BC
