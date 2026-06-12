#pragma once

#include "BCCellBuilder.h"
#include "BCTypes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
#include <intrin.h>
#endif
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace BC {

inline std::atomic<uint32_t> g_bc_cell_finalize_debug_cid{0U};
inline std::atomic<uint32_t> g_bc_cell_finalize_debug_bucket_count{0U};
inline std::atomic<uint32_t> g_bc_cell_finalize_debug_sorted_size{0U};
inline std::atomic<uint32_t> g_bc_cell_finalize_debug_stage{0U};
inline constexpr uint32_t kBCMutableBatchPrefetchDistance = 16U;

inline void bc_cell_mutable_spin_pause() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    _mm_pause();
#elif defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
    YieldProcessor();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

struct BCCellBuilderDumpView {
    CellId cid = 0U;
    uint32_t dump_generation = 0U;
    const uint8_t *data = nullptr;
    uint32_t size = 0U;
};

struct BCCellBuilderDump {
    CellId cid = 0U;
    uint32_t dump_generation = 0U;
    std::vector<uint8_t> bytes;

    [[nodiscard]] BCCellBuilderDumpView view() const {
        if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable cell dump exceeds uint32 bytes");
        }
        return BCCellBuilderDumpView{
            cid,
            dump_generation,
            bytes.data(),
            static_cast<uint32_t>(bytes.size())
        };
    }
};

struct BCCellBuilderDumpStreamLayout {
    CellId cid = 0U;
    uint32_t dump_generation = 0U;
    uint32_t metadata_bytes = 0U;
    uint32_t bitmap_words = 0U;
    uint32_t bitmap_capacity_words = 0U;
    uint32_t total_bytes = 0U;
};

struct BCCellBuilderDumpStreamScratch {
    std::vector<uint8_t> metadata;
    std::array<uint8_t, sizeof(uint64_t)> word = {};
};

enum class BCCellMutableSlotState : uint8_t {
    Empty = 0U,
    Pending = 1U,
    Occupied = 2U,
};

struct BCCellResolvedInsert {
    uint64_t mask = 0U;
    uint32_t word_offset = 0U;
    bool new_bucket = false;
};

static_assert(sizeof(BCCellResolvedInsert) <= 16U, "BC resolved insert must stay cache-compact");

struct BCCellEncodedInsert {
    static constexpr uint32_t kInvalidHomeSlot = std::numeric_limits<uint32_t>::max();

    uint64_t key = 0U;
    BucketRank rank = 0U;
    BucketBitmapLen bitmap_len = 0U;
    uint32_t home_slot = kInvalidHomeSlot;
};

struct BCCellTrustedKeyRankInsert {
    uint64_t key = 0U;
    BucketRank rank = 0U;
};

static_assert(sizeof(BCCellTrustedKeyRankInsert) <= 16U, "BC trusted insert must stay cache-compact");

struct BCCellInsertBatchResult {
    uint64_t new_buckets = 0U;
    uint64_t new_ranks = 0U;
    uint64_t duplicate_ranks = 0U;
};

struct BCCellBitmapThreadChunk {
    uint32_t word_next = 0U;
    uint32_t word_end = 0U;
};

struct BCCellAlignedBitmapDeleter {
    void operator()(uint64_t *ptr) const noexcept {
#ifdef _WIN32
        if (ptr != nullptr) {
            VirtualFree(ptr, 0, MEM_RELEASE);
        }
#else
        ::operator delete[](ptr, std::align_val_t{4096U});
#endif
    }
};

using BCCellBitmapArenaPtr = std::unique_ptr<uint64_t[], BCCellAlignedBitmapDeleter>;

struct BCCellAlignedByteDeleter {
    void operator()(uint8_t *ptr) const noexcept {
#ifdef _WIN32
        if (ptr != nullptr) {
            VirtualFree(ptr, 0, MEM_RELEASE);
        }
#else
        ::operator delete[](ptr, std::align_val_t{4096U});
#endif
    }
};

using BCCellAlignedBytePtr = std::unique_ptr<uint8_t[], BCCellAlignedByteDeleter>;

[[nodiscard]] inline BCCellBitmapArenaPtr bc_allocate_cell_bitmap_arena(uint32_t words) {
    if (words == 0U) {
        return BCCellBitmapArenaPtr{};
    }
    if (words > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        throw std::overflow_error("BC mutable bitmap arena allocation size overflow");
    }
#ifdef _WIN32
    void *ptr = VirtualAlloc(
        nullptr,
        static_cast<size_t>(words) * sizeof(uint64_t),
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE
    );
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
#else
    void *ptr = ::operator new[](
        static_cast<size_t>(words) * sizeof(uint64_t),
        std::align_val_t{4096U}
    );
#endif
    return BCCellBitmapArenaPtr(static_cast<uint64_t *>(ptr));
}

[[nodiscard]] inline BCCellAlignedBytePtr bc_allocate_cell_aligned_bytes(uint64_t bytes) {
    if (bytes == 0U) {
        return BCCellAlignedBytePtr{};
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC mutable aligned byte allocation exceeds size_t");
    }
#ifdef _WIN32
    void *ptr = VirtualAlloc(
        nullptr,
        static_cast<size_t>(bytes),
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE
    );
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
#else
    void *ptr = ::operator new[](
        static_cast<size_t>(bytes),
        std::align_val_t{4096U}
    );
#endif
    return BCCellAlignedBytePtr(static_cast<uint8_t *>(ptr));
}

class BCCellMutableBuilderOverflow : public std::runtime_error {
public:
    explicit BCCellMutableBuilderOverflow(const char *message)
        : std::runtime_error(message) {}
};

class BCCellMutableBuilder {
public:
    struct SortedBucketRef {
        uint64_t key = 0U;
        uint32_t slot = 0U;
        uint32_t bitmap_offset = 0U;
        BucketBitmapLen bitmap_len = 0U;
        uint32_t word_count = 0U;
        uint32_t live_count = 0U;
    };

    struct DumpBucketRef {
        uint64_t key = 0U;
        uint32_t bitmap_offset = 0U;
        BucketBitmapLen bitmap_len = 0U;
        uint32_t word_count = 0U;
        uint32_t live_count = 0U;
    };

    struct FinalizeScratch {
        FinalizedCellPayload payload;
        std::vector<SortedBucketRef> sorted;
        std::vector<SortedBucketRef> sorted_reordered;
        std::vector<DumpBucketRef> dump_sorted;
        std::vector<DumpBucketRef> dump_reordered;
        std::vector<uint64_t> sort_keys;
        std::vector<uint32_t> sort_indices;
        std::vector<uint8_t> rank_chunk;

        void clear_payload_keep_capacity() {
            payload.buckets.clear();
            payload.rank_payload.clear();
            payload.success_rows = 0U;
        }

        [[nodiscard]] uint64_t allocated_bytes() const {
            uint64_t bytes = sizeof(FinalizeScratch);
            bytes += static_cast<uint64_t>(payload.buckets.capacity()) * sizeof(BCBucketEntry);
            bytes += static_cast<uint64_t>(payload.rank_payload.capacity());
            bytes += static_cast<uint64_t>(sorted.capacity()) * sizeof(SortedBucketRef);
            bytes += static_cast<uint64_t>(sorted_reordered.capacity()) * sizeof(SortedBucketRef);
            bytes += static_cast<uint64_t>(dump_sorted.capacity()) * sizeof(DumpBucketRef);
            bytes += static_cast<uint64_t>(dump_reordered.capacity()) * sizeof(DumpBucketRef);
            bytes += static_cast<uint64_t>(sort_keys.capacity()) * sizeof(uint64_t);
            bytes += static_cast<uint64_t>(sort_indices.capacity()) * sizeof(uint32_t);
            bytes += static_cast<uint64_t>(rank_chunk.capacity());
            return bytes;
        }
    };

    struct StreamedFinalizePlan {
        uint32_t bucket_count = 0U;
        uint32_t success_rows = 0U;
        uint32_t rank_payload_bytes = 0U;
    };

    BCCellMutableBuilder(const BCLut &lut, CellId cid)
        : lut_(&lut), cid_(cid) {
        initialize_empty_table(kInitialCapacity);
    }

    BCCellMutableBuilder(const BCCellMutableBuilder &) = delete;
    BCCellMutableBuilder &operator=(const BCCellMutableBuilder &) = delete;

    [[nodiscard]] CellId cid() const {
        return cid_;
    }

    [[nodiscard]] bool dirty() const {
        return dirty_.load(std::memory_order_acquire);
    }

    void mark_clean() {
        dirty_.store(false, std::memory_order_release);
    }

    void reserve(uint32_t buckets, uint32_t bitmap_words) {
        if (buckets != 0U) {
            grow_hash_table(capacity_for_bucket_count(buckets));
        }
        if (bitmap_words != 0U) {
            reserve_bitmap_words(bitmap_words);
        }
    }

    [[nodiscard]] bool overflowed() const {
        return overflowed_.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint32_t home_slot_for_key(uint64_t key) const {
        if (capacity_ == 0U) {
            throw std::logic_error("BC mutable cell builder has zero hash capacity");
        }
        return slot_for_key_with_shift(key, hash_shift_);
    }

    [[nodiscard]] uint32_t compact_home_slot_or_compute(
        const BCCellEncodedInsert &encoded,
        const uint32_t *home_slots,
        uint32_t index
    ) const {
        if (home_slots != nullptr) {
            return home_slots[index];
        }
        return encoded.home_slot != BCCellEncodedInsert::kInvalidHomeSlot
            ? encoded.home_slot
            : home_slot_for_key(encoded.key);
    }

    [[nodiscard]] BCInsertResult insert(uint64_t key, BucketRank rank) {
        const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
        return insert_impl(key, rank, bitmap_len);
    }

    [[nodiscard]] BCInsertResult insert_encoded(const BCEncodedKeyRank &encoded) {
        if (!encoded.valid) {
            throw std::invalid_argument("BC mutable cell builder cannot insert invalid encoded key/rank");
        }
        return insert_impl(encoded.key, encoded.rank, encoded.bitmap_len);
    }

    void prefetch_home_slot_index(uint32_t home_slot) const {
        prefetch_slot_index(home_slot);
    }

    [[nodiscard]] bool resolve_encoded_with_home_slot(
        const BCEncodedKeyRank &encoded,
        uint32_t home_slot,
        BCCellResolvedInsert &resolved,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr
    ) {
        if (!encoded.valid) {
            throw std::invalid_argument("BC mutable cell builder cannot resolve invalid encoded key/rank");
        }
        if (bucket_count_.load(std::memory_order_acquire) > max_buckets_for_capacity(capacity_)) {
            mark_capacity_overflow();
            return false;
        }
        bool inserted = false;
        ReadGuard guard(*this);
        if (!resolve_one_guarded_at_slot(
                encoded.key,
                encoded.rank,
                encoded.bitmap_len,
                home_slot,
                inserted,
                resolved,
                bitmap_chunk)) {
            return false;
        }
        return true;
    }

    [[nodiscard]] BCInsertResult apply_resolved_insert(
        const BCCellResolvedInsert &resolved
    ) {
        ReadGuard guard(*this);
        uint64_t *base = bitmap_base();
        if (base == nullptr) {
            throw std::invalid_argument("BC mutable resolved insert has null bitmap arena");
        }
        uint64_t *word = base + resolved.word_offset;
        if ((atomic_load_bitmap_word(word) & resolved.mask) != 0ULL) {
            return BCInsertResult{resolved.new_bucket, false};
        }
        const uint64_t old = atomic_fetch_or_bitmap_word(word, resolved.mask);
        const bool new_rank = (old & resolved.mask) == 0U;
        if (new_rank) {
            mark_dirty_once();
        }
        return BCInsertResult{resolved.new_bucket, new_rank};
    }

    void resolve_batch(
        const std::vector<BCEncodedKeyRank> &encoded,
        std::vector<BCCellResolvedInsert> &resolved
    ) {
        resolved.clear();
        resolved.reserve(encoded.size());
        if (encoded.empty()) {
            return;
        }

        for (;;) {
            ensure_batch_capacity(static_cast<uint32_t>(encoded.size()));
            bool need_grow = false;
            uint32_t grow_bitmap_words = 0U;
            {
                ReadGuard guard(*this);
                resolved.clear();
                resolved.resize(encoded.size());
                constexpr uint32_t kPrefetchDistance = kBCMutableBatchPrefetchDistance;
                const uint32_t count = static_cast<uint32_t>(encoded.size());
                const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
                for (uint32_t i = 0U; i < prefetch_count; ++i) {
                    prefetch_home_slot(encoded[i].key);
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    if (i + kPrefetchDistance < count) {
                        prefetch_home_slot(encoded[i + kPrefetchDistance].key);
                    }
                    const BCEncodedKeyRank &item = encoded[i];
                    if (!item.valid) {
                        throw std::invalid_argument("BC mutable cell builder cannot resolve invalid encoded key/rank");
                    }
                    bool inserted = false;
                    if (!resolve_one_guarded(item.key, item.rank, item.bitmap_len, inserted, resolved[i])) {
                        grow_bitmap_words = std::max<uint32_t>(grow_bitmap_words, words_for_bits(item.bitmap_len));
                        need_grow = true;
                        break;
                    }
#if defined(__GNUC__) || defined(__clang__)
                    if (uint64_t *base = bitmap_base()) {
                        __builtin_prefetch(base + resolved[i].word_offset, 1, 1);
                    }
#endif
                }
                if (!need_grow) {
                    return;
                }
            }
            grow_hash_table(capacity_for_bucket_count(
                bucket_count_.load(std::memory_order_acquire) + static_cast<uint32_t>(encoded.size())
            ));
            if (grow_bitmap_words != 0U) {
                const uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
                const uint32_t current_capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
                const uint32_t doubled = current_capacity > std::numeric_limits<uint32_t>::max() / 2U
                    ? std::numeric_limits<uint32_t>::max()
                    : current_capacity * 2U;
                reserve_bitmap_words_stop_world(std::max<uint32_t>(used + grow_bitmap_words, doubled));
            }
        }
    }

    [[nodiscard]] BCCellInsertBatchResult resolve_and_apply_compact_batch(
        const BCCellEncodedInsert *encoded,
        uint32_t *home_slots,
        uint32_t count,
        std::vector<BCCellResolvedInsert> &resolved,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        bool count_new_buckets = true,
        uint64_t *probe_steps_out = nullptr
    ) {
        if (count != 0U && encoded == nullptr) {
            throw std::invalid_argument("BC mutable cell compact resolve/apply batch pointer is null");
        }
        resolved.clear();
        resolved.reserve(count);
        if (probe_steps_out != nullptr) {
            *probe_steps_out = 0U;
        }
        if (count == 0U) {
            return {};
        }
        for (;;) {
            ensure_batch_capacity(count);
            bool need_grow = false;
            uint64_t probe_steps = 0U;
            uint32_t grow_bitmap_words = 0U;
            {
                resolved.clear();
                resolved.resize(count);
                ReadGuard guard(*this);
                uint64_t *bitmap_base_ptr = bitmap_base();
                if (home_slots != nullptr) {
                    for (uint32_t i = 0U; i < count; ++i) {
                        home_slots[i] =
                            encoded[i].home_slot != BCCellEncodedInsert::kInvalidHomeSlot
                                ? encoded[i].home_slot
                                : home_slot_for_key(encoded[i].key);
                    }
                }
                constexpr uint32_t kPrefetchDistance = kBCMutableBatchPrefetchDistance;
                const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
                for (uint32_t i = 0U; i < prefetch_count; ++i) {
                    if (home_slots != nullptr) {
                        prefetch_slot_index(home_slots[i]);
                    } else {
                        prefetch_slot_index(compact_home_slot_or_compute(encoded[i], home_slots, i));
                    }
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    if (i + kPrefetchDistance < count) {
                        if (home_slots != nullptr) {
                            prefetch_slot_index(home_slots[i + kPrefetchDistance]);
                        } else {
                            prefetch_slot_index(compact_home_slot_or_compute(
                                encoded[i + kPrefetchDistance],
                                home_slots,
                                i + kPrefetchDistance
                            ));
                        }
                    }
                    const BCCellEncodedInsert &item = encoded[i];
                    const uint32_t home_slot =
                        compact_home_slot_or_compute(item, home_slots, i);
                    bool inserted = false;
                    uint32_t item_probe_steps = 0U;
                    if (!resolve_one_compact_guarded_at_slot(
                            item.key,
                            item.rank,
                            item.bitmap_len,
                            home_slot,
                            inserted,
                            resolved[i],
                            bitmap_chunk,
                            probe_steps_out != nullptr ? &item_probe_steps : nullptr)) {
                        probe_steps += item_probe_steps;
                        grow_bitmap_words = std::max<uint32_t>(
                            grow_bitmap_words,
                            words_for_bits(item.bitmap_len)
                        );
                        need_grow = true;
                        break;
                    }
                    probe_steps += item_probe_steps;
#if defined(__GNUC__) || defined(__clang__)
                    if (bitmap_base_ptr != nullptr) {
                        __builtin_prefetch(bitmap_base_ptr + resolved[i].word_offset, 1, 1);
                    }
#endif
                }
                if (!need_grow) {
                    if (probe_steps_out != nullptr) {
                        *probe_steps_out = probe_steps;
                    }
                    return apply_resolved_batch_guarded(resolved, count_new_buckets);
                }
            }
            grow_hash_table(capacity_for_bucket_count(
                bucket_count_.load(std::memory_order_acquire) + count
            ));
            if (grow_bitmap_words != 0U) {
                const uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
                const uint32_t current_capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
                const uint32_t doubled = current_capacity > std::numeric_limits<uint32_t>::max() / 2U
                    ? std::numeric_limits<uint32_t>::max()
                    : current_capacity * 2U;
                reserve_bitmap_words_stop_world(std::max<uint32_t>(used + grow_bitmap_words, doubled));
            }
        }
    }

    [[nodiscard]] BCCellInsertBatchResult resolve_and_apply_trusted_key_rank_batch(
        const BCCellTrustedKeyRankInsert *encoded,
        uint32_t count,
        std::vector<BCCellResolvedInsert> &resolved,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        bool count_new_buckets = true,
        uint64_t *probe_steps_out = nullptr
    ) {
        if (count != 0U && encoded == nullptr) {
            throw std::invalid_argument("BC mutable trusted key/rank batch pointer is null");
        }
        resolved.clear();
        resolved.reserve(count);
        if (probe_steps_out != nullptr) {
            *probe_steps_out = 0U;
        }
        if (count == 0U) {
            return {};
        }

        for (;;) {
            ensure_batch_capacity(count);
            bool need_grow = false;
            uint64_t probe_steps = 0U;
            uint32_t grow_bitmap_words = 0U;
            {
                resolved.clear();
                resolved.resize(count);
                ReadGuard guard(*this);
                uint64_t *bitmap_base_ptr = bitmap_base();
                constexpr uint32_t kPrefetchDistance = kBCMutableBatchPrefetchDistance;
                const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
                for (uint32_t i = 0U; i < prefetch_count; ++i) {
                    prefetch_home_slot(encoded[i].key);
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    if (i + kPrefetchDistance < count) {
                        prefetch_home_slot(encoded[i + kPrefetchDistance].key);
                    }
                    const BCCellTrustedKeyRankInsert &item = encoded[i];
                    bool inserted = false;
                    uint32_t item_probe_steps = 0U;
                    if (!resolve_one_trusted_key_rank_guarded_at_slot(
                            item.key,
                            item.rank,
                            home_slot_for_key(item.key),
                            inserted,
                            resolved[i],
                            bitmap_chunk,
                            probe_steps_out != nullptr ? &item_probe_steps : nullptr)) {
                        probe_steps += item_probe_steps;
                        const BucketBitmapLen bitmap_len =
                            bitmap_len_from_trusted_key(*lut_, item.key);
                        grow_bitmap_words = std::max<uint32_t>(
                            grow_bitmap_words,
                            words_for_bits(bitmap_len)
                        );
                        need_grow = true;
                        break;
                    }
                    probe_steps += item_probe_steps;
#if defined(__GNUC__) || defined(__clang__)
                    if (bitmap_base_ptr != nullptr) {
                        __builtin_prefetch(bitmap_base_ptr + resolved[i].word_offset, 1, 1);
                    }
#endif
                }
                if (!need_grow) {
                    if (probe_steps_out != nullptr) {
                        *probe_steps_out = probe_steps;
                    }
                    return apply_resolved_batch_guarded(resolved, count_new_buckets);
                }
            }
            grow_hash_table(capacity_for_bucket_count(
                bucket_count_.load(std::memory_order_acquire) + count
            ));
            if (grow_bitmap_words != 0U) {
                const uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
                const uint32_t current_capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
                const uint32_t doubled = current_capacity > std::numeric_limits<uint32_t>::max() / 2U
                    ? std::numeric_limits<uint32_t>::max()
                    : current_capacity * 2U;
                reserve_bitmap_words_stop_world(std::max<uint32_t>(used + grow_bitmap_words, doubled));
            }
        }
    }

    [[nodiscard]] BCCellInsertBatchResult apply_resolved_batch(
        const std::vector<BCCellResolvedInsert> &resolved,
        bool count_new_buckets = true
    ) {
        ReadGuard guard(*this);
        return apply_resolved_batch_guarded(resolved, count_new_buckets);
    }

    [[nodiscard]] BCCellInsertBatchResult apply_resolved_batch_guarded(
        const std::vector<BCCellResolvedInsert> &resolved,
        bool count_new_buckets = true
    ) {
        BCCellInsertBatchResult result;
        constexpr uint32_t kPrefetchDistance = kBCMutableBatchPrefetchDistance;
        const uint32_t count = static_cast<uint32_t>(resolved.size());
        bool any_new_rank = false;
        uint64_t *base = bitmap_base();
        if (count != 0U && base == nullptr) {
            throw std::invalid_argument("BC mutable resolved batch has null bitmap arena");
        }
#ifndef NDEBUG
        const uint32_t debug_bitmap_capacity =
            bitmap_capacity_words_.load(std::memory_order_acquire);
#endif
        const uint32_t prefetch_count = std::min<uint32_t>(count, kPrefetchDistance);
        for (uint32_t i = 0U; i < prefetch_count; ++i) {
#ifndef NDEBUG
            if (resolved[i].word_offset >= debug_bitmap_capacity) {
                throw std::logic_error("BC mutable resolved prefetch word offset exceeds bitmap arena");
            }
#endif
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(base + resolved[i].word_offset, 1, 1);
#endif
        }
        for (uint32_t i = 0U; i < count; ++i) {
            if (i + kPrefetchDistance < count) {
#ifndef NDEBUG
                if (resolved[i + kPrefetchDistance].word_offset >= debug_bitmap_capacity) {
                    throw std::logic_error("BC mutable resolved lookahead word offset exceeds bitmap arena");
                }
#endif
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(base + resolved[i + kPrefetchDistance].word_offset, 1, 1);
#endif
            }
            const BCCellResolvedInsert &item = resolved[i];
#ifndef NDEBUG
            if (item.word_offset >= debug_bitmap_capacity) {
                throw std::logic_error("BC mutable resolved apply word offset exceeds bitmap arena");
            }
#endif
            uint64_t *word = base + item.word_offset;
            if (count_new_buckets) {
                result.new_buckets += item.new_bucket ? 1U : 0U;
            }
            if ((atomic_load_bitmap_word(word) & item.mask) != 0ULL) {
                if (count_new_buckets) {
                    ++result.duplicate_ranks;
                }
                continue;
            }
            const uint64_t old = atomic_fetch_or_bitmap_word(word, item.mask);
            if ((old & item.mask) == 0U) {
                if (count_new_buckets) {
                    ++result.new_ranks;
                }
                any_new_rank = true;
            } else if (count_new_buckets) {
                ++result.duplicate_ranks;
            }
        }
        if (any_new_rank) {
            mark_dirty_once();
        }
        return result;
    }

    [[nodiscard]] BCCellInsertBatchResult insert_resolved_batch(
        const std::vector<BCEncodedKeyRank> &encoded
    ) {
        std::vector<BCCellResolvedInsert> resolved;
        resolve_batch(encoded, resolved);
        return apply_resolved_batch(resolved);
    }

    [[nodiscard]] bool contains(uint64_t key, BucketRank rank) const {
        const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
        if (rank >= bitmap_len) {
            return false;
        }
        ReadGuard guard(*this);
        const uint32_t slot = find_existing_bucket_no_create(key);
        if (slot == kInvalidSlot) {
            return false;
        }
        if (slot_bitmap_len(slot) != bitmap_len) {
            return false;
        }
        const uint32_t word = static_cast<uint32_t>(rank) / kBCBitmapWordBits;
        const uint32_t bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
        uint64_t *bitmap_word = slot_bitmap_word(slot, word);
        if (bitmap_word == nullptr) {
            return false;
        }
        return (atomic_load_bitmap_word(bitmap_word) & (1ULL << bit)) != 0U;
    }

    [[nodiscard]] uint64_t live_rows_estimate() const {
        ReadGuard guard(*this);
        uint64_t rows = 0U;
        const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t i = occupied_slots_[index];
            if (i >= capacity_ || slot_state(i) != kOccupiedState) {
                continue;
            }
            rows += live_count(i);
        }
        return rows;
    }

    [[nodiscard]] uint64_t bucket_count_estimate() const {
        return bucket_count_.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        const uint64_t table_bytes = static_cast<uint64_t>(capacity_) * slot_metadata_bytes_per_entry();
        const uint64_t occupied_bytes = static_cast<uint64_t>(capacity_) * sizeof(uint32_t);
        const uint64_t bitmap_bytes =
            static_cast<uint64_t>(bitmap_capacity_words_.load(std::memory_order_acquire)) * sizeof(uint64_t);
        return sizeof(BCCellMutableBuilder) + table_bytes + occupied_bytes + bitmap_bytes;
    }

    [[nodiscard]] uint32_t hash_capacity() const {
        return capacity_;
    }

    [[nodiscard]] uint32_t max_bucket_capacity() const {
        return max_buckets_for_capacity(capacity_);
    }

    [[nodiscard]] uint32_t bitmap_used_words() const {
        return bitmap_used_words_.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint32_t bitmap_capacity_words() const {
        return bitmap_capacity_words_.load(std::memory_order_acquire);
    }

    void validate_internal_for_testing() const {
        ReadGuard guard(*this);
        if (capacity_ == 0U || (capacity_ & (capacity_ - 1U)) != 0U) {
            throw std::logic_error("BC mutable cell validation saw invalid hash capacity");
        }
        if (occupied_slots_ == nullptr || occupied_slots_capacity_ < capacity_) {
            throw std::logic_error("BC mutable cell validation saw invalid occupied slot storage");
        }
        const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
        if (buckets > max_buckets_for_capacity(capacity_) || buckets > occupied_slots_capacity_) {
            throw std::logic_error("BC mutable cell validation saw bucket count exceed capacity");
        }
        const uint32_t bitmap_used = bitmap_used_words_.load(std::memory_order_acquire);
        const uint32_t bitmap_capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (bitmap_used > bitmap_capacity) {
            throw std::logic_error("BC mutable cell validation saw bitmap usage exceed capacity");
        }
        if (bitmap_used != 0U && bitmap_base() == nullptr) {
            throw std::logic_error("BC mutable cell validation saw missing bitmap arena");
        }

        std::vector<uint8_t> seen(capacity_, 0U);
        uint32_t listed = 0U;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot >= capacity_) {
                throw std::logic_error("BC mutable cell validation saw occupied slot out of range");
            }
            if (seen[slot] != 0U) {
                throw std::logic_error("BC mutable cell validation saw duplicate occupied slot");
            }
            seen[slot] = 1U;
            if (slot_state(slot) != kOccupiedState) {
                throw std::logic_error("BC mutable cell validation saw non-occupied listed slot");
            }
            const uint64_t key = slots_.keys[slot];
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            const uint32_t offset = slots_.bitmap_offsets[slot];
            if (bitmap_len == 0U ||
                bitmap_len > kBCMaxBucketBitmapLen ||
                word_count == 0U ||
                offset > bitmap_used ||
                word_count > bitmap_used - offset ||
                offset > bitmap_capacity ||
                word_count > bitmap_capacity - offset) {
                throw std::logic_error("BC mutable cell validation saw invalid bitmap range");
            }
            ++listed;
        }

        uint32_t scanned = 0U;
        for (uint32_t slot = 0U; slot < capacity_; ++slot) {
            const uint8_t state = slot_state(slot);
            if (state == kOccupiedState) {
                ++scanned;
                if (seen[slot] == 0U) {
                    throw std::logic_error("BC mutable cell validation saw unlisted occupied slot");
                }
            } else if (state != kEmptyState) {
                throw std::logic_error("BC mutable cell validation saw pending slot outside insert");
            }
        }
        if (scanned != listed || listed != buckets) {
            throw std::logic_error("BC mutable cell validation occupied count mismatch");
        }
    }

    [[nodiscard]] uint64_t hash_grow_count() const {
        return hash_grow_count_.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t bitmap_grow_count() const {
        return bitmap_grow_count_.load(std::memory_order_acquire);
    }

    void mark_growth_baseline() const {
        hash_grow_baseline_ = hash_grow_count_.load(std::memory_order_acquire);
        bitmap_grow_baseline_ = bitmap_grow_count_.load(std::memory_order_acquire);
        hash_replaced_bytes_baseline_ =
            hash_replaced_bytes_total_.load(std::memory_order_acquire);
        bitmap_replaced_bytes_baseline_ =
            bitmap_replaced_bytes_total_.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t runtime_hash_grow_count() const {
        const uint64_t total = hash_grow_count_.load(std::memory_order_acquire);
        return total >= hash_grow_baseline_ ? total - hash_grow_baseline_ : 0U;
    }

    [[nodiscard]] uint64_t runtime_bitmap_grow_count() const {
        const uint64_t total = bitmap_grow_count_.load(std::memory_order_acquire);
        return total >= bitmap_grow_baseline_ ? total - bitmap_grow_baseline_ : 0U;
    }

    [[nodiscard]] uint64_t runtime_hash_replaced_bytes() const {
        const uint64_t total = hash_replaced_bytes_total_.load(std::memory_order_acquire);
        return total >= hash_replaced_bytes_baseline_ ? total - hash_replaced_bytes_baseline_ : 0U;
    }

    [[nodiscard]] uint64_t runtime_bitmap_replaced_bytes() const {
        const uint64_t total = bitmap_replaced_bytes_total_.load(std::memory_order_acquire);
        return total >= bitmap_replaced_bytes_baseline_ ? total - bitmap_replaced_bytes_baseline_ : 0U;
    }

    [[nodiscard]] uint64_t finalize_count_for_testing() const {
        return finalize_count_;
    }

    void dump_into(uint32_t dump_generation, BCCellBuilderDump &dump) const {
        StopWorldGuard stop(*this);
        dump.cid = cid_;
        dump.dump_generation = dump_generation;
        dump.bytes.clear();
        const uint32_t buckets = bucket_count_.load(std::memory_order_relaxed);
        uint32_t occupied_bucket_count = 0U;
        uint64_t bitmap_words64 = 0U;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot >= capacity_) {
                throw std::logic_error("BC mutable cell dump occupied slot index is out of range");
            }
            if (slot_state(slot) == kOccupiedState) {
                const uint64_t key = slots_.keys[slot];
                const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                ++occupied_bucket_count;
                bitmap_words64 += word_count;
                if (bitmap_words64 > std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("BC mutable cell dump bitmap word count exceeds uint32");
                }
            }
        }
        const uint32_t bitmap_words = static_cast<uint32_t>(bitmap_words64);
        const uint64_t slots_bytes64 = static_cast<uint64_t>(occupied_bucket_count) * sizeof(uint32_t);
        const uint64_t keys_bytes64 = static_cast<uint64_t>(occupied_bucket_count) * sizeof(uint64_t);
        const uint64_t offsets_bytes64 = 0U;
        const uint64_t metadata_bytes64 =
            static_cast<uint64_t>(kDumpHeaderBytes) +
            slots_bytes64 +
            keys_bytes64 +
            offsets_bytes64;
        const uint64_t total_bytes64 =
            metadata_bytes64 +
            static_cast<uint64_t>(bitmap_words) * sizeof(uint64_t);
        if (total_bytes64 > std::numeric_limits<uint32_t>::max() ||
            metadata_bytes64 > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable cell raw dump exceeds uint32 bytes");
        }
        const uint32_t metadata_bytes = static_cast<uint32_t>(metadata_bytes64);
        dump.bytes.resize(static_cast<size_t>(total_bytes64));
        uint8_t *header = dump.bytes.data();
        store_u32_le(header + 0U, kDumpMagic);
        store_u16_le(header + 4U, kDumpVersion);
        store_u16_le(header + 6U, 0U);
        store_u32_le(header + 8U, cid_);
        store_u32_le(header + 12U, dump_generation);
        store_u32_le(header + 16U, capacity_);
        store_u32_le(header + 20U, occupied_bucket_count);
        store_u32_le(header + 24U, bitmap_words);
        store_u32_le(header + 28U, bitmap_capacity_words_.load(std::memory_order_acquire));
        store_u32_le(header + 32U, metadata_bytes);
        store_u32_le(header + 36U, checked_u32(slots_bytes64, "BC mutable raw dump slots bytes overflow"));
        store_u32_le(header + 40U, checked_u32(keys_bytes64, "BC mutable raw dump keys bytes overflow"));
        store_u32_le(header + 44U, checked_u32(offsets_bytes64, "BC mutable raw dump offsets bytes overflow"));
        store_u32_le(header + 48U, 0U);
        store_u32_le(header + 52U, checked_u32(total_bytes64, "BC mutable raw dump total bytes overflow"));
        store_u32_le(header + 56U, 0U);
        store_u32_le(header + 60U, 0U);

        uint8_t *cursor = dump.bytes.data() + kDumpHeaderBytes;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            store_u32_le(cursor, slot);
            cursor += sizeof(uint32_t);
        }
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            store_u64_le(cursor, slots_.keys[slot]);
            cursor += sizeof(uint64_t);
        }
        if (cursor != dump.bytes.data() + metadata_bytes) {
            throw std::logic_error("BC mutable raw dump metadata size accounting mismatch");
        }
        uint64_t *base = bitmap_base();
        if (bitmap_words != 0U && base == nullptr) {
            throw std::logic_error("BC mutable cell dump missing bitmap arena");
        }
        uint8_t *bitmap = dump.bytes.data() + metadata_bytes;
        uint32_t bitmap_cursor = 0U;
        if (host_is_little_endian()) {
            uint32_t group_source = 0U;
            uint32_t group_target = 0U;
            uint32_t group_words = 0U;
            bool have_group = false;
            const auto flush_group = [&]() {
                if (!have_group || group_words == 0U) {
                    return;
                }
                std::memcpy(
                    bitmap + static_cast<size_t>(group_target) * sizeof(uint64_t),
                    base + group_source,
                    static_cast<size_t>(group_words) * sizeof(uint64_t)
                );
                have_group = false;
                group_words = 0U;
            };
            for (uint32_t index = 0U; index < buckets; ++index) {
                const uint32_t slot = occupied_slots_[index];
                if (slot_state(slot) != kOccupiedState) {
                    continue;
                }
                const uint64_t key = slots_.keys[slot];
                const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                const uint32_t source_offset = slots_.bitmap_offsets[slot];
                const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
                if (source_offset > capacity || word_count > capacity - source_offset) {
                    throw std::logic_error("BC mutable cell dump bitmap range exceeds arena");
                }
                if (word_count == 0U) {
                    continue;
                }
                if (!have_group) {
                    group_source = source_offset;
                    group_target = bitmap_cursor;
                    group_words = word_count;
                    have_group = true;
                } else if (source_offset == group_source + group_words) {
                    group_words += word_count;
                } else {
                    flush_group();
                    group_source = source_offset;
                    group_target = bitmap_cursor;
                    group_words = word_count;
                    have_group = true;
                }
                bitmap_cursor += word_count;
            }
            flush_group();

            bitmap_cursor = 0U;
            for (uint32_t index = 0U; index < buckets; ++index) {
                const uint32_t slot = occupied_slots_[index];
                if (slot_state(slot) != kOccupiedState) {
                    continue;
                }
                const uint64_t key = slots_.keys[slot];
                const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                if (word_count != 0U) {
                    uint64_t last = 0U;
                    uint8_t *last_bytes =
                        bitmap + static_cast<size_t>(bitmap_cursor + word_count - 1U) * sizeof(uint64_t);
                    std::memcpy(&last, last_bytes, sizeof(uint64_t));
                    last &= tail_mask_for_bits(bitmap_len, word_count - 1U);
                    std::memcpy(last_bytes, &last, sizeof(uint64_t));
                }
                bitmap_cursor += word_count;
            }
        } else {
            for (uint32_t index = 0U; index < buckets; ++index) {
                const uint32_t slot = occupied_slots_[index];
                if (slot_state(slot) != kOccupiedState) {
                    continue;
                }
                const uint64_t key = slots_.keys[slot];
                const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                uint8_t *target = bitmap + static_cast<size_t>(bitmap_cursor) * sizeof(uint64_t);
                for (uint32_t word = 0U; word < word_count; ++word) {
                    store_u64_le(target + static_cast<size_t>(word) * sizeof(uint64_t),
                                 masked_bitmap_word_stopped(slot, word));
                }
                bitmap_cursor += word_count;
            }
        }
    }

    [[nodiscard]] BCCellBuilderDump dump(uint32_t dump_generation) const {
        BCCellBuilderDump dump;
        dump_into(dump_generation, dump);
        return dump;
    }

    template <class EmitMetadataFn, class EmitBitmapFn>
    [[nodiscard]] BCCellBuilderDumpStreamLayout dump_streamed_into(
        uint32_t dump_generation,
        BCCellBuilderDumpStreamScratch &scratch,
        EmitMetadataFn &&emit_metadata,
        EmitBitmapFn &&emit_bitmap
    ) const {
        StopWorldGuard stop(*this);
        const uint32_t buckets = bucket_count_.load(std::memory_order_relaxed);
        uint32_t occupied_bucket_count = 0U;
        uint64_t bitmap_words64 = 0U;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot >= capacity_) {
                throw std::logic_error("BC mutable cell streamed dump occupied slot index is out of range");
            }
            if (slot_state(slot) == kOccupiedState) {
                const uint64_t key = slots_.keys[slot];
                const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
                const uint32_t word_count = words_for_bits(bitmap_len);
                ++occupied_bucket_count;
                bitmap_words64 += word_count;
                if (bitmap_words64 > std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("BC mutable cell streamed dump bitmap word count exceeds uint32");
                }
            }
        }

        const uint32_t bitmap_words = static_cast<uint32_t>(bitmap_words64);
        const uint64_t slots_bytes64 = static_cast<uint64_t>(occupied_bucket_count) * sizeof(uint32_t);
        const uint64_t keys_bytes64 = static_cast<uint64_t>(occupied_bucket_count) * sizeof(uint64_t);
        const uint64_t offsets_bytes64 = 0U;
        const uint64_t metadata_bytes64 =
            static_cast<uint64_t>(kDumpHeaderBytes) +
            slots_bytes64 +
            keys_bytes64 +
            offsets_bytes64;
        const uint64_t total_bytes64 =
            metadata_bytes64 +
            static_cast<uint64_t>(bitmap_words) * sizeof(uint64_t);
        if (total_bytes64 > std::numeric_limits<uint32_t>::max() ||
            metadata_bytes64 > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable streamed dump exceeds uint32 bytes");
        }
        const uint32_t metadata_bytes = static_cast<uint32_t>(metadata_bytes64);
        scratch.metadata.resize(metadata_bytes);
        uint8_t *header = scratch.metadata.data();
        store_u32_le(header + 0U, kDumpMagic);
        store_u16_le(header + 4U, kDumpVersion);
        store_u16_le(header + 6U, 0U);
        store_u32_le(header + 8U, cid_);
        store_u32_le(header + 12U, dump_generation);
        store_u32_le(header + 16U, capacity_);
        store_u32_le(header + 20U, occupied_bucket_count);
        store_u32_le(header + 24U, bitmap_words);
        store_u32_le(header + 28U, bitmap_capacity_words_.load(std::memory_order_acquire));
        store_u32_le(header + 32U, metadata_bytes);
        store_u32_le(header + 36U, checked_u32(slots_bytes64, "BC mutable streamed dump slots bytes overflow"));
        store_u32_le(header + 40U, checked_u32(keys_bytes64, "BC mutable streamed dump keys bytes overflow"));
        store_u32_le(header + 44U, checked_u32(offsets_bytes64, "BC mutable streamed dump offsets bytes overflow"));
        store_u32_le(header + 48U, 0U);
        store_u32_le(header + 52U, checked_u32(total_bytes64, "BC mutable streamed dump total bytes overflow"));
        store_u32_le(header + 56U, 0U);
        store_u32_le(header + 60U, 0U);

        uint8_t *cursor = scratch.metadata.data() + kDumpHeaderBytes;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            store_u32_le(cursor, slot);
            cursor += sizeof(uint32_t);
        }
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            store_u64_le(cursor, slots_.keys[slot]);
            cursor += sizeof(uint64_t);
        }
        if (cursor != scratch.metadata.data() + metadata_bytes) {
            throw std::logic_error("BC mutable streamed dump metadata size accounting mismatch");
        }

        emit_metadata(scratch.metadata.data(), metadata_bytes);

        uint64_t *base = bitmap_base();
        if (bitmap_words != 0U && base == nullptr) {
            throw std::logic_error("BC mutable cell streamed dump missing bitmap arena");
        }
        uint32_t bitmap_cursor = 0U;
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            const uint64_t key = slots_.keys[slot];
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            const uint32_t source_offset = slots_.bitmap_offsets[slot];
            const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
            if (source_offset > capacity || word_count > capacity - source_offset) {
                throw std::logic_error("BC mutable cell streamed dump bitmap range exceeds arena");
            }
            if (word_count == 0U) {
                continue;
            }
            const uint64_t tail_mask = tail_mask_for_bits(bitmap_len, word_count - 1U);
            if (host_is_little_endian()) {
                const uint32_t direct_words = tail_mask == ~0ULL ? word_count : word_count - 1U;
                if (direct_words != 0U) {
                    emit_bitmap(base + source_offset, static_cast<uint64_t>(direct_words) * sizeof(uint64_t));
                }
                if (direct_words != word_count) {
                    const uint64_t last = base[source_offset + word_count - 1U] & tail_mask;
                    store_u64_le(scratch.word.data(), last);
                    emit_bitmap(scratch.word.data(), sizeof(uint64_t));
                }
            } else {
                for (uint32_t word = 0U; word < word_count; ++word) {
                    store_u64_le(scratch.word.data(), masked_bitmap_word_stopped(slot, word));
                    emit_bitmap(scratch.word.data(), sizeof(uint64_t));
                }
            }
            bitmap_cursor += word_count;
        }
        if (bitmap_cursor != bitmap_words) {
            throw std::logic_error("BC mutable streamed dump bitmap word accounting mismatch");
        }
        return BCCellBuilderDumpStreamLayout{
            cid_,
            dump_generation,
            metadata_bytes,
            bitmap_words,
            bitmap_capacity_words_.load(std::memory_order_acquire),
            static_cast<uint32_t>(total_bytes64)
        };
    }

    struct CompactDumpHeader {
        uint32_t capacity = 0U;
        uint32_t bucket_count = 0U;
        uint32_t bitmap_words = 0U;
        uint32_t bitmap_capacity_words = 0U;
        uint32_t header_bytes = 0U;
        uint32_t states_bytes = 0U;
        uint32_t keys_bytes = 0U;
        uint32_t offsets_bytes = 0U;
        uint32_t occupied_bytes = 0U;
        uint32_t metadata_bytes = 0U;
        uint32_t total_bytes = 0U;
        uint16_t version = 0U;
    };

    [[nodiscard]] static bool host_is_little_endian() {
        const uint16_t value = 1U;
        return *reinterpret_cast<const uint8_t *>(&value) == 1U;
    }

    [[nodiscard]] static constexpr uint32_t compact_dump_header_bytes() {
        return kDumpHeaderBytes;
    }

    [[nodiscard]] static CompactDumpHeader parse_compact_dump_header(
        CellId cid,
        uint32_t dump_generation,
        const uint8_t *header,
        uint32_t header_bytes
    ) {
        if (header == nullptr || header_bytes < kDumpHeaderBytes) {
            throw std::invalid_argument("BC mutable compact dump header is truncated");
        }
        const uint32_t magic = load_u32_le_unchecked(header + 0U);
        const uint16_t version = load_u16_le(header + 4U);
        if (magic != kDumpMagic || version != kDumpVersion) {
            throw std::invalid_argument("BC mutable compact dump header is invalid");
        }
        const uint32_t stored_cid = load_u32_le_unchecked(header + 8U);
        const uint32_t stored_generation = load_u32_le_unchecked(header + 12U);
        const uint32_t stored_capacity = load_u32_le_unchecked(header + 16U);
        const uint32_t stored_bucket_count = load_u32_le_unchecked(header + 20U);
        const uint32_t stored_bitmap_words = load_u32_le_unchecked(header + 24U);
        const uint32_t stored_bitmap_capacity = load_u32_le_unchecked(header + 28U);
        if (stored_cid != cid || stored_generation != dump_generation) {
            throw std::invalid_argument("BC mutable compact dump metadata mismatch");
        }
        if (stored_capacity < kInitialCapacity ||
            (stored_capacity & (stored_capacity - 1U)) != 0U ||
            stored_bucket_count > max_buckets_for_capacity(stored_capacity)) {
            throw std::invalid_argument("BC mutable compact dump flat table capacity is invalid");
        }
        if (stored_bitmap_capacity < stored_bitmap_words) {
            throw std::invalid_argument("BC mutable compact dump bitmap capacity is invalid");
        }
        if (header_bytes < kDumpHeaderBytes) {
            throw std::invalid_argument("BC mutable raw dump header is truncated");
        }
        const uint32_t stored_metadata_bytes = load_u32_le_unchecked(header + 32U);
        const uint32_t stored_states_bytes = load_u32_le_unchecked(header + 36U);
        const uint32_t stored_keys_bytes = load_u32_le_unchecked(header + 40U);
        const uint32_t stored_offsets_bytes = load_u32_le_unchecked(header + 44U);
        const uint32_t stored_occupied_bytes = load_u32_le_unchecked(header + 48U);
        const uint32_t stored_total_bytes = load_u32_le_unchecked(header + 52U);
        const uint64_t expected_states_bytes = static_cast<uint64_t>(stored_bucket_count) * sizeof(uint32_t);
        const uint64_t expected_keys_bytes = static_cast<uint64_t>(stored_bucket_count) * sizeof(uint64_t);
        const uint64_t expected_offsets_bytes = 0U;
        const uint64_t expected_occupied_bytes = 0U;
        const uint64_t expected_metadata_bytes =
            static_cast<uint64_t>(kDumpHeaderBytes) +
            expected_states_bytes +
            expected_keys_bytes +
            expected_offsets_bytes +
            expected_occupied_bytes;
        const uint64_t expected_total_bytes =
            expected_metadata_bytes + static_cast<uint64_t>(stored_bitmap_words) * sizeof(uint64_t);
        if (expected_total_bytes > std::numeric_limits<uint32_t>::max() ||
            expected_metadata_bytes > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable raw dump size exceeds uint32");
        }
        if (stored_states_bytes != expected_states_bytes ||
            stored_keys_bytes != expected_keys_bytes ||
            stored_offsets_bytes != expected_offsets_bytes ||
            stored_occupied_bytes != expected_occupied_bytes ||
            stored_metadata_bytes != expected_metadata_bytes ||
            stored_total_bytes != expected_total_bytes) {
            throw std::invalid_argument("BC mutable raw dump layout is invalid");
        }
        return CompactDumpHeader{
            stored_capacity,
            stored_bucket_count,
            stored_bitmap_words,
            stored_bitmap_capacity,
            kDumpHeaderBytes,
            stored_states_bytes,
            stored_keys_bytes,
            stored_offsets_bytes,
            stored_occupied_bytes,
            stored_metadata_bytes,
            stored_total_bytes,
            version
        };
    }

    [[nodiscard]] static std::unique_ptr<BCCellMutableBuilder> restore_compact_metadata_for_direct_bitmap(
        const BCLut &lut,
        CellId cid,
        uint32_t dump_generation,
        const uint8_t *metadata,
        uint32_t metadata_bytes,
        bool allocate_bitmap_arena = true,
        bool validate_metadata = true
    ) {
        if (!host_is_little_endian()) {
            throw std::runtime_error("BC mutable direct compact restore requires little-endian host");
        }
        const CompactDumpHeader header =
            parse_compact_dump_header(cid, dump_generation, metadata, metadata_bytes);
        if (metadata_bytes != header.metadata_bytes) {
            throw std::invalid_argument("BC mutable compact dump metadata size mismatch");
        }
        std::unique_ptr<BCCellMutableBuilder> builder(
            new BCCellMutableBuilder(lut, cid, RestoreNoInitTag{})
        );
        builder->slots_ = allocate_slot_table(header.capacity, false);
        builder->capacity_ = header.capacity;
        builder->capacity_mask_ = header.capacity - 1U;
        builder->hash_shift_ = hash_shift_for_capacity(header.capacity);
        builder->allocate_occupied_slots(header.capacity);
        if (allocate_bitmap_arena && header.bitmap_capacity_words != 0U) {
            const uint64_t bitmap_bytes = static_cast<uint64_t>(header.bitmap_words) * sizeof(uint64_t);
            const uint64_t padded_bytes = (bitmap_bytes + 4095ULL) & ~4095ULL;
            if (padded_bytes / sizeof(uint64_t) > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable compact direct restore padded bitmap exceeds uint32 words");
            }
            const uint32_t padded_words = static_cast<uint32_t>(padded_bytes / sizeof(uint64_t));
            const uint32_t restore_capacity = std::max<uint32_t>(header.bitmap_capacity_words, padded_words);
            builder->bitmap_arena_ = bc_allocate_cell_bitmap_arena(restore_capacity);
            builder->bitmap_capacity_words_.store(restore_capacity, std::memory_order_release);
        }
        const uint8_t *cursor = metadata + kDumpHeaderBytes;
        const uint8_t *slot_bytes = cursor;
        cursor += header.states_bytes;
        const uint8_t *key_bytes = cursor;
        cursor += header.keys_bytes;
        cursor += header.offsets_bytes;
        if (cursor != metadata + header.metadata_bytes) {
            throw std::logic_error("BC mutable raw dump restore metadata accounting mismatch");
        }
        for (uint32_t index = 0U; index < header.bucket_count; ++index) {
            const uint32_t slot =
                load_u32_le_unchecked(slot_bytes + static_cast<size_t>(index) * sizeof(uint32_t));
            const uint64_t key =
                load_u64_le(key_bytes + static_cast<size_t>(index) * sizeof(uint64_t));
            const uint32_t bitmap_offset = builder->bitmap_used_words_.load(std::memory_order_relaxed);
            if (slot >= header.capacity) {
                throw std::invalid_argument("BC mutable raw dump occupied slot is invalid");
            }
            if (builder->slots_.states[slot].load(std::memory_order_relaxed) != kEmptyState) {
                throw std::invalid_argument("BC mutable raw dump contains duplicate slot");
            }
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(lut, key);
            const uint32_t words = words_for_bits(bitmap_len);
            if (validate_metadata) {
                if (bitmap_len == 0U ||
                    bitmap_len > kBCMaxBucketBitmapLen ||
                    words == 0U ||
                    bitmap_offset > header.bitmap_words ||
                    words > header.bitmap_words - bitmap_offset) {
                    throw std::invalid_argument("BC mutable raw dump bucket bitmap range is invalid");
                }
            }
            builder->slots_.keys[slot] = key;
            builder->slots_.bitmap_offsets[slot] = bitmap_offset;
            builder->slots_.states[slot].store(kOccupiedState, std::memory_order_release);
            builder->record_occupied_slot(index, slot);
            builder->bitmap_used_words_.store(
                bitmap_offset + words,
                std::memory_order_relaxed
            );
        }
        if (builder->bitmap_used_words_.load(std::memory_order_relaxed) != header.bitmap_words) {
            throw std::invalid_argument("BC mutable raw dump bitmap word accounting mismatch");
        }
        builder->bitmap_used_words_.store(header.bitmap_words, std::memory_order_release);
        builder->bucket_count_.store(header.bucket_count, std::memory_order_release);
        builder->dirty_.store(false, std::memory_order_release);
        return builder;
    }

    [[nodiscard]] uint8_t *direct_restore_bitmap_bytes(uint32_t expected_words, uint64_t read_bytes = 0U) {
        if (expected_words != bitmap_used_words_.load(std::memory_order_acquire)) {
            throw std::invalid_argument("BC mutable compact direct restore bitmap word mismatch");
        }
        if (expected_words == 0U) {
            return nullptr;
        }
        if (bitmap_base() == nullptr) {
            throw std::logic_error("BC mutable compact direct restore has no bitmap arena");
        }
        const uint64_t capacity_bytes =
            static_cast<uint64_t>(bitmap_capacity_words_.load(std::memory_order_acquire)) * sizeof(uint64_t);
        const uint64_t required_bytes =
            read_bytes == 0U ? static_cast<uint64_t>(expected_words) * sizeof(uint64_t) : read_bytes;
        if (required_bytes > capacity_bytes) {
            throw std::out_of_range("BC mutable compact direct restore bitmap read exceeds arena capacity");
        }
        return reinterpret_cast<uint8_t *>(bitmap_base());
    }

    void finish_direct_bitmap_restore(bool mask_tail_bits = true) {
        if (mask_tail_bits) {
            const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
            for (uint32_t index = 0U; index < buckets; ++index) {
                const uint32_t slot = occupied_slots_[index];
                if (slot >= capacity_) {
                    throw std::logic_error("BC mutable compact direct restore occupied slot out of range");
                }
                if (slot_state(slot) != kOccupiedState) {
                    continue;
                }
                const BucketBitmapLen bitmap_len = slot_bitmap_len(slot);
                const uint32_t words = words_for_bits(bitmap_len);
                if (words == 0U) {
                    throw std::logic_error("BC mutable compact direct restore saw empty bucket");
                }
                uint64_t *last = slot_bitmap_word(slot, words - 1U);
                if (last == nullptr) {
                    throw std::logic_error("BC mutable compact direct restore missing last bitmap word");
                }
                *last &= tail_mask_for_bits(bitmap_len, words - 1U);
            }
        }
        dirty_.store(false, std::memory_order_release);
    }

    [[nodiscard]] static std::unique_ptr<BCCellMutableBuilder> restore(
        const BCLut &lut,
        CellId cid,
        BCCellBuilderDumpView dump
    ) {
        if (dump.cid != cid) {
            throw std::invalid_argument("BC mutable cell dump cid mismatch");
        }
        if (dump.data == nullptr && dump.size != 0U) {
            throw std::invalid_argument("BC mutable cell dump pointer is null");
        }
        if (dump.size < kDumpHeaderBytes) {
            throw std::invalid_argument("BC mutable cell dump is smaller than header");
        }
        const CompactDumpHeader header =
            parse_compact_dump_header(cid, dump.dump_generation, dump.data, dump.size);
        if (header.total_bytes != dump.size) {
            throw std::invalid_argument("BC mutable cell dump size mismatch");
        }
        std::unique_ptr<BCCellMutableBuilder> builder =
            restore_compact_metadata_for_direct_bitmap(
                lut,
                cid,
                dump.dump_generation,
                dump.data,
                header.metadata_bytes,
                true,
                true
            );
        const uint8_t *bitmap = dump.data + header.metadata_bytes;
        if (header.bitmap_words != 0U) {
            uint8_t *target = builder->direct_restore_bitmap_bytes(header.bitmap_words);
            if (host_is_little_endian()) {
                std::memcpy(
                    target,
                    bitmap,
                    static_cast<size_t>(header.bitmap_words) * sizeof(uint64_t)
                );
            } else {
                uint64_t *words = reinterpret_cast<uint64_t *>(target);
                for (uint32_t word = 0U; word < header.bitmap_words; ++word) {
                    words[word] =
                        load_u64_le(bitmap + static_cast<size_t>(word) * sizeof(uint64_t));
                }
            }
        }
        builder->finish_direct_bitmap_restore(true);
        return builder;
    }

    static void finalize_dump_into(
        const BCLut &lut,
        CellId cid,
        BCCellBuilderDumpView dump,
        FinalizeScratch &scratch,
        BCCellFinalizeOptions options = {}
    ) {
        if (dump.cid != cid) {
            throw std::invalid_argument("BC mutable finalize dump cid mismatch");
        }
        if (dump.data == nullptr && dump.size != 0U) {
            throw std::invalid_argument("BC mutable finalize dump pointer is null");
        }
        const CompactDumpHeader header =
            parse_compact_dump_header(cid, dump.dump_generation, dump.data, dump.size);
        if (header.total_bytes != dump.size) {
            throw std::invalid_argument("BC mutable finalize dump size mismatch");
        }
        const uint8_t *cursor = dump.data + kDumpHeaderBytes;
        const uint8_t *slot_bytes = cursor;
        cursor += header.states_bytes;
        const uint8_t *key_bytes = cursor;
        cursor += header.keys_bytes;
        cursor += header.offsets_bytes;
        if (cursor != dump.data + header.metadata_bytes) {
            throw std::logic_error("BC mutable finalize dump metadata accounting mismatch");
        }
        const uint8_t *bitmap = dump.data + header.metadata_bytes;

        scratch.clear_payload_keep_capacity();
        scratch.dump_sorted.clear();
        scratch.dump_sorted.reserve(header.bucket_count);
        uint32_t bitmap_cursor = 0U;
        for (uint32_t index = 0U; index < header.bucket_count; ++index) {
            const uint32_t slot =
                load_u32_le_unchecked(slot_bytes + static_cast<size_t>(index) * sizeof(uint32_t));
            const uint64_t key =
                load_u64_le(key_bytes + static_cast<size_t>(index) * sizeof(uint64_t));
            const uint32_t bitmap_offset = bitmap_cursor;
            if (slot >= header.capacity) {
                throw std::invalid_argument("BC mutable finalize dump slot is invalid");
            }
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(lut, key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            if (bitmap_len == 0U ||
                bitmap_len > kBCMaxBucketBitmapLen ||
                word_count == 0U ||
                bitmap_offset > header.bitmap_words ||
                word_count > header.bitmap_words - bitmap_offset) {
                throw std::invalid_argument("BC mutable finalize dump bitmap range is invalid");
            }
            uint32_t live = 0U;
            for (uint32_t word = 0U; word < word_count; ++word) {
                const uint64_t value =
                    load_u64_le(bitmap + static_cast<size_t>(bitmap_offset + word) * sizeof(uint64_t)) &
                    tail_mask_for_bits(bitmap_len, word);
                live += popcount64(value);
            }
            if (live != 0U) {
                scratch.dump_sorted.push_back(DumpBucketRef{
                    key,
                    bitmap_offset,
                    bitmap_len,
                    word_count,
                    live
                });
            }
            bitmap_cursor += word_count;
        }
        if (bitmap_cursor != header.bitmap_words) {
            throw std::invalid_argument("BC mutable finalize dump bitmap word accounting mismatch");
        }
        const bool use_keyvalue_sort =
            options.keyvalue_sort != nullptr &&
            scratch.dump_sorted.size() >= options.simd_sort_min_bucket_count;
        if (use_keyvalue_sort) {
            scratch.sort_keys.clear();
            scratch.sort_indices.clear();
            scratch.sort_keys.reserve(scratch.dump_sorted.size());
            scratch.sort_indices.reserve(scratch.dump_sorted.size());
            for (uint32_t i = 0U; i < scratch.dump_sorted.size(); ++i) {
                scratch.sort_keys.push_back(scratch.dump_sorted[i].key);
                scratch.sort_indices.push_back(i);
            }
            options.keyvalue_sort(
                scratch.sort_keys.data(),
                scratch.sort_indices.data(),
                scratch.sort_keys.size(),
                false
            );
            scratch.dump_reordered.clear();
            scratch.dump_reordered.reserve(scratch.dump_sorted.size());
            for (uint32_t index : scratch.sort_indices) {
                if (index >= scratch.dump_sorted.size()) {
                    throw std::logic_error("BC mutable finalize dump sorter returned invalid index");
                }
                DumpBucketRef ref = scratch.dump_sorted[index];
                ref.key = scratch.sort_keys[scratch.dump_reordered.size()];
                scratch.dump_reordered.push_back(ref);
            }
            scratch.dump_sorted.swap(scratch.dump_reordered);
        } else {
            scratch.sort_indices.clear();
            scratch.sort_indices.reserve(scratch.dump_sorted.size());
            for (uint32_t i = 0U; i < scratch.dump_sorted.size(); ++i) {
                scratch.sort_indices.push_back(i);
            }
            std::sort(
                scratch.sort_indices.begin(),
                scratch.sort_indices.end(),
                [&scratch](uint32_t lhs, uint32_t rhs) {
                    return scratch.dump_sorted[lhs].key < scratch.dump_sorted[rhs].key;
                }
            );
            scratch.dump_reordered.clear();
            scratch.dump_reordered.reserve(scratch.dump_sorted.size());
            for (uint32_t index : scratch.sort_indices) {
                if (index >= scratch.dump_sorted.size()) {
                    throw std::logic_error("BC mutable finalize dump fallback sorter returned invalid index");
                }
                scratch.dump_reordered.push_back(scratch.dump_sorted[index]);
            }
            scratch.dump_sorted.swap(scratch.dump_reordered);
        }

        FinalizedCellPayload &out = scratch.payload;
        out.buckets.reserve(scratch.dump_sorted.size());
        uint32_t reserve_bytes = 0U;
        for (const DumpBucketRef &ref : scratch.dump_sorted) {
            reserve_bytes = align_up_u32(reserve_bytes, 8U);
            const uint32_t bitmap_offset =
                bc_rank_payload_bitmap_offset(reserve_bytes, ref.bitmap_len);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(ref.word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable finalize dump bitmap bytes overflow");
            }
            reserve_bytes = checked_u32_add(
                bitmap_offset,
                static_cast<uint32_t>(bitmap_bytes64),
                "BC mutable finalize dump rank payload reserve overflow"
            );
        }
        out.rank_payload.reserve(reserve_bytes);

        uint64_t success_cursor = 0U;
        for (const DumpBucketRef &ref : scratch.dump_sorted) {
            const uint32_t payload_offset =
                checked_u32(out.rank_payload.size(), "BC mutable finalize dump rank payload offset");
            const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
            append_padding(out.rank_payload, aligned_payload_offset - payload_offset);
            const uint32_t rank_payload_offset =
                checked_u32(out.rank_payload.size(), "BC mutable finalize dump aligned rank payload offset");

            uint32_t running = 0U;
            const uint32_t prefix_count = prefix_count_for_bits(ref.bitmap_len);
            for (uint32_t block = 0U; block < prefix_count; ++block) {
                if (running > std::numeric_limits<RankPrefix>::max()) {
                    throw std::logic_error("BC mutable finalize dump prefix exceeds uint16");
                }
                append_u16_le(out.rank_payload, static_cast<RankPrefix>(running));
                const uint32_t block_first_bit = block * kBCRankPrefixBits;
                const uint32_t block_last_bit = std::min<uint32_t>(
                    ref.bitmap_len,
                    block_first_bit + kBCRankPrefixBits
                );
                const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
                const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
                for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
                    uint64_t value =
                        load_u64_le(bitmap + static_cast<size_t>(ref.bitmap_offset + word) * sizeof(uint64_t)) &
                        tail_mask_for_bits(ref.bitmap_len, word);
                    if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                        value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
                    }
                    running += popcount64(value);
                }
            }

            const uint32_t bitmap_payload_offset =
                bc_rank_payload_bitmap_offset(rank_payload_offset, ref.bitmap_len);
            append_padding(
                out.rank_payload,
                bitmap_payload_offset - checked_u32(out.rank_payload.size(), "BC mutable finalize dump prefix end")
            );
            if (host_is_little_endian()) {
                const size_t old_size = out.rank_payload.size();
                const size_t bytes = static_cast<size_t>(ref.word_count) * sizeof(uint64_t);
                out.rank_payload.resize(old_size + bytes);
                std::memcpy(
                    out.rank_payload.data() + old_size,
                    bitmap + static_cast<size_t>(ref.bitmap_offset) * sizeof(uint64_t),
                    bytes
                );
                if (ref.word_count != 0U) {
                    uint64_t last = 0U;
                    uint8_t *last_bytes =
                        out.rank_payload.data() + old_size +
                        static_cast<size_t>(ref.word_count - 1U) * sizeof(uint64_t);
                    std::memcpy(&last, last_bytes, sizeof(uint64_t));
                    last &= tail_mask_for_bits(ref.bitmap_len, ref.word_count - 1U);
                    std::memcpy(last_bytes, &last, sizeof(uint64_t));
                }
            } else {
                for (uint32_t word = 0U; word < ref.word_count; ++word) {
                    append_u64_le(
                        out.rank_payload,
                        load_u64_le(bitmap + static_cast<size_t>(ref.bitmap_offset + word) * sizeof(uint64_t)) &
                            tail_mask_for_bits(ref.bitmap_len, word)
                    );
                }
            }

            out.buckets.push_back(BCBucketEntry{
                ref.key,
                rank_payload_offset,
                checked_u32(success_cursor, "BC mutable finalize dump success row offset")
            });
            success_cursor += ref.live_count;
            if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable finalize dump success rows exceed uint32");
            }
        }
        out.success_rows = static_cast<uint32_t>(success_cursor);
    }

    template <class WriteCellFn>
    static void finalize_dump_streamed_into(
        const BCLut &lut,
        CellId cid,
        BCCellBuilderDumpView dump,
        FinalizeScratch &scratch,
        WriteCellFn &&write_cell,
        BCCellFinalizeOptions options = {}
    ) {
        if (dump.cid != cid) {
            throw std::invalid_argument("BC mutable streamed finalize dump cid mismatch");
        }
        if (dump.data == nullptr && dump.size != 0U) {
            throw std::invalid_argument("BC mutable streamed finalize dump pointer is null");
        }
        const CompactDumpHeader header =
            parse_compact_dump_header(cid, dump.dump_generation, dump.data, dump.size);
        if (header.total_bytes != dump.size) {
            throw std::invalid_argument("BC mutable streamed finalize dump size mismatch");
        }
        const uint8_t *cursor = dump.data + kDumpHeaderBytes;
        const uint8_t *slot_bytes = cursor;
        cursor += header.states_bytes;
        const uint8_t *key_bytes = cursor;
        cursor += header.keys_bytes;
        cursor += header.offsets_bytes;
        if (cursor != dump.data + header.metadata_bytes) {
            throw std::logic_error("BC mutable streamed finalize dump metadata accounting mismatch");
        }
        const uint8_t *bitmap = dump.data + header.metadata_bytes;

        scratch.clear_payload_keep_capacity();
        scratch.dump_sorted.clear();
        scratch.dump_sorted.reserve(header.bucket_count);
        uint32_t bitmap_cursor = 0U;
        for (uint32_t index = 0U; index < header.bucket_count; ++index) {
            const uint32_t slot =
                load_u32_le_unchecked(slot_bytes + static_cast<size_t>(index) * sizeof(uint32_t));
            const uint64_t key =
                load_u64_le(key_bytes + static_cast<size_t>(index) * sizeof(uint64_t));
            const uint32_t bitmap_offset = bitmap_cursor;
            if (slot >= header.capacity) {
                throw std::invalid_argument("BC mutable streamed finalize dump slot is invalid");
            }
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(lut, key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            if (bitmap_len == 0U ||
                bitmap_len > kBCMaxBucketBitmapLen ||
                word_count == 0U ||
                bitmap_offset > header.bitmap_words ||
                word_count > header.bitmap_words - bitmap_offset) {
                throw std::invalid_argument("BC mutable streamed finalize dump bitmap range is invalid");
            }
            uint32_t live = 0U;
            for (uint32_t word = 0U; word < word_count; ++word) {
                const uint64_t value =
                    load_u64_le(bitmap + static_cast<size_t>(bitmap_offset + word) * sizeof(uint64_t)) &
                    tail_mask_for_bits(bitmap_len, word);
                live += popcount64(value);
            }
            if (live != 0U) {
                scratch.dump_sorted.push_back(DumpBucketRef{
                    key,
                    bitmap_offset,
                    bitmap_len,
                    word_count,
                    live
                });
            }
            bitmap_cursor += word_count;
        }
        if (bitmap_cursor != header.bitmap_words) {
            throw std::invalid_argument("BC mutable streamed finalize dump bitmap word accounting mismatch");
        }
        sort_dump_refs_for_finalize(scratch, options);
        const StreamedFinalizePlan plan = measure_dump_streamed_plan(scratch.dump_sorted);
        auto emit_payload = [&](auto &&emit_bucket, auto &&emit_rank) {
            emit_dump_streamed_payload(
                bitmap,
                scratch.dump_sorted,
                scratch,
                emit_bucket,
                emit_rank
            );
        };
        write_cell(
            plan.bucket_count,
            plan.success_rows,
            plan.rank_payload_bytes,
            emit_payload
        );
    }

    [[nodiscard]] static FinalizedCellPayload finalize_dump(
        const BCLut &lut,
        CellId cid,
        BCCellBuilderDumpView dump,
        BCCellFinalizeOptions options = {}
    ) {
        FinalizeScratch scratch;
        finalize_dump_into(lut, cid, dump, scratch, options);
        return std::move(scratch.payload);
    }

    void finalize_into(FinalizeScratch &scratch, BCCellFinalizeOptions options = {}) const {
        StopWorldGuard stop(*this);
        ++finalize_count_;
        scratch.clear_payload_keep_capacity();
        sorted_bucket_refs_into(options, scratch.sorted, scratch.sort_keys, scratch.sort_indices, scratch.sorted_reordered);
        FinalizedCellPayload &out = scratch.payload;
        out.buckets.reserve(scratch.sorted.size());
        out.rank_payload.reserve(estimate_rank_payload_bytes(scratch.sorted));
        uint64_t success_cursor = 0U;
        for (const SortedBucketRef &ref : scratch.sorted) {
            if (slot_state(ref.slot) != kOccupiedState) {
                throw std::logic_error("BC mutable cell finalize saw unoccupied bucket");
            }
            const uint32_t bucket_live = ref.live_count;
            if (bucket_live == 0U) {
                continue;
            }
            const uint32_t payload_offset =
                checked_u32(out.rank_payload.size(), "BC mutable rank payload offset");
            const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
            append_padding(out.rank_payload, aligned_payload_offset - payload_offset);
            const uint32_t rank_payload_offset =
                checked_u32(out.rank_payload.size(), "BC mutable aligned rank payload offset");

            append_prefix256_le(out.rank_payload, ref.slot, ref.bitmap_len, ref.word_count);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, ref.bitmap_len);
            append_padding(
                out.rank_payload,
                bitmap_offset - checked_u32(out.rank_payload.size(), "BC mutable prefix payload end")
            );
            append_bitmap_le(
                out.rank_payload,
                ref.slot,
                ref.bitmap_len,
                ref.word_count,
                ref.bitmap_offset
            );
            out.buckets.push_back(BCBucketEntry{
                ref.key,
                rank_payload_offset,
                checked_u32(success_cursor, "BC mutable success row offset")
            });
            success_cursor += bucket_live;
            if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable finalized cell success_rows exceeds uint32");
            }
        }
        out.success_rows = static_cast<uint32_t>(success_cursor);
        dirty_.store(false, std::memory_order_release);
    }

    template <class WriteCellFn>
    void finalize_streamed_into(
        FinalizeScratch &scratch,
        WriteCellFn &&write_cell,
        BCCellFinalizeOptions options = {}
    ) const {
        StopWorldGuard stop(*this);
        ++finalize_count_;
        scratch.clear_payload_keep_capacity();
        sorted_bucket_refs_into(
            options,
            scratch.sorted,
            scratch.sort_keys,
            scratch.sort_indices,
            scratch.sorted_reordered
        );
        const StreamedFinalizePlan plan = measure_sorted_streamed_plan(scratch.sorted);
        auto emit_payload = [&](auto &&emit_bucket, auto &&emit_rank) {
            emit_sorted_streamed_payload(
                scratch.sorted,
                scratch,
                emit_bucket,
                emit_rank
            );
        };
        write_cell(
            plan.bucket_count,
            plan.success_rows,
            plan.rank_payload_bytes,
            emit_payload
        );
        dirty_.store(false, std::memory_order_release);
    }

    [[nodiscard]] FinalizedCellPayload finalize(BCCellFinalizeOptions options = {}) const {
        FinalizeScratch scratch;
        finalize_into(scratch, options);
        return std::move(scratch.payload);
    }

private:
    struct RestoreNoInitTag {};

    BCCellMutableBuilder(const BCLut &lut, CellId cid, RestoreNoInitTag)
        : lut_(&lut), cid_(cid) {}

    struct SlotArrays {
        std::unique_ptr<std::atomic<uint8_t>[]> states;
        BCCellAlignedBytePtr metadata_storage;
        uint64_t *keys = nullptr;
        uint32_t *bitmap_offsets = nullptr;
        uint32_t capacity = 0U;
    };

    struct BitmapAllocation {
        uint32_t offset = 0U;
        bool ok = false;
    };

    class ReadGuard {
    public:
        explicit ReadGuard(const BCCellMutableBuilder &builder) : builder_(&builder) {
            for (;;) {
                while (builder_->resizing_.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                builder_->active_inserts_.fetch_add(1U, std::memory_order_acq_rel);
                active_ = true;
                if (!builder_->resizing_.load(std::memory_order_acquire)) {
                    return;
                }
                release();
            }
        }

        ReadGuard(const ReadGuard &) = delete;
        ReadGuard &operator=(const ReadGuard &) = delete;

        ~ReadGuard() {
            release();
        }

        void release() {
            if (active_) {
                builder_->active_inserts_.fetch_sub(1U, std::memory_order_acq_rel);
                active_ = false;
            }
        }

    private:
        const BCCellMutableBuilder *builder_ = nullptr;
        bool active_ = false;
    };

    class StopWorldGuard {
    public:
        explicit StopWorldGuard(const BCCellMutableBuilder &builder)
            : builder_(&builder), lock_(builder.resize_mutex_) {
            builder_->resizing_.store(true, std::memory_order_release);
            while (builder_->active_inserts_.load(std::memory_order_acquire) != 0U) {
                std::this_thread::yield();
            }
        }

        StopWorldGuard(const StopWorldGuard &) = delete;
        StopWorldGuard &operator=(const StopWorldGuard &) = delete;

        ~StopWorldGuard() {
            builder_->resizing_.store(false, std::memory_order_release);
        }

    private:
        const BCCellMutableBuilder *builder_ = nullptr;
        std::unique_lock<std::mutex> lock_;
    };

    [[nodiscard]] uint8_t slot_state(uint32_t slot) const {
        return slots_.states[slot].load(std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t *bitmap_base() const {
        return bitmap_arena_.get();
    }

    [[nodiscard]] uint64_t *slot_bitmap_base(uint32_t slot) const {
        uint64_t *base = bitmap_base();
        if (base == nullptr) {
            return nullptr;
        }
        const uint32_t offset = slots_.bitmap_offsets[slot];
        if (offset > bitmap_capacity_words_.load(std::memory_order_acquire)) {
            return nullptr;
        }
        return base + offset;
    }

    [[nodiscard]] BucketBitmapLen slot_bitmap_len(uint32_t slot) const {
        return bitmap_len_from_key(*lut_, slots_.keys[slot]);
    }

    [[nodiscard]] uint32_t slot_word_count(uint32_t slot) const {
        return words_for_bits(slot_bitmap_len(slot));
    }

    [[nodiscard]] uint64_t *slot_bitmap_word(uint32_t slot, uint32_t word) const {
        if (word >= slot_word_count(slot)) {
            return nullptr;
        }
        const uint32_t offset = slots_.bitmap_offsets[slot];
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (offset >= capacity || word >= capacity - offset) {
            return nullptr;
        }
        uint64_t *base = slot_bitmap_base(slot);
        return base == nullptr ? nullptr : base + word;
    }

    [[nodiscard]] BCInsertResult insert_impl(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len
    ) {
        BCCellResolvedInsert resolved = resolve_one(key, rank, bitmap_len);
        return apply_resolved_insert(resolved);
    }

    [[nodiscard]] BCCellResolvedInsert resolve_one(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len
    ) {
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
            throw std::logic_error("BC mutable cell insert saw invalid bitmap_len");
        }
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC mutable cell insert rank exceeds bitmap_len");
        }
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t rank_word = static_cast<uint32_t>(rank) / kBCBitmapWordBits;
        const uint32_t rank_bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
        if (rank_word >= word_count) {
            throw std::logic_error("BC mutable cell rank word outside bitmap");
        }
        for (;;) {
            ReadGuard guard(*this);
            BCCellResolvedInsert out;
            bool inserted = false;
            if (!resolve_one_guarded(key, rank, bitmap_len, inserted, out)) {
                guard.release();
                grow_hash_table(capacity_for_bucket_count(bucket_count_.load(std::memory_order_acquire) + 1U));
                const uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
                reserve_bitmap_words_stop_world(used + words_for_bits(bitmap_len));
                continue;
            }
            (void)word_count;
            (void)rank_word;
            (void)rank_bit;
            return out;
        }
    }

    [[nodiscard]] bool resolve_one_guarded(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len,
        bool &inserted,
        BCCellResolvedInsert &out,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        return resolve_one_guarded_at_slot(
            key,
            rank,
            bitmap_len,
            home_slot_for_key(key),
            inserted,
            out,
            bitmap_chunk,
            probe_steps_out
        );
    }

    [[nodiscard]] bool resolve_one_guarded_at_slot(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len,
        uint32_t home_slot,
        bool &inserted,
        BCCellResolvedInsert &out,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
            throw std::logic_error("BC mutable cell insert saw invalid bitmap_len");
        }
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC mutable cell insert rank exceeds bitmap_len");
        }
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t rank_word = static_cast<uint32_t>(rank) / kBCBitmapWordBits;
        const uint32_t rank_bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
        if (rank_word >= word_count) {
            throw std::logic_error("BC mutable cell rank word outside bitmap");
        }
        inserted = false;
        const uint32_t bucket_slot = find_or_create_bucket_guarded_at_slot(
            key,
            bitmap_len,
            word_count,
            home_slot,
            inserted,
            bitmap_chunk,
            probe_steps_out
        );
        if (bucket_slot == kInvalidSlot) {
            return false;
        }
        const uint32_t offset = slots_.bitmap_offsets[bucket_slot];
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (offset >= capacity || rank_word >= capacity - offset) {
            throw std::logic_error("BC mutable cell bucket has null bitmap");
        }
        uint64_t *base = bitmap_base();
        if (base == nullptr) {
            throw std::logic_error("BC mutable cell bucket has null bitmap");
        }
        (void)base;
        out = BCCellResolvedInsert{
            1ULL << rank_bit,
            offset + rank_word,
            inserted
        };
        return true;
    }

    [[nodiscard]] bool resolve_one_compact_guarded_at_slot(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len,
        uint32_t home_slot,
        bool &inserted,
        BCCellResolvedInsert &out,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
            throw std::logic_error("BC mutable compact insert saw invalid bitmap_len");
        }
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC mutable compact insert rank exceeds bitmap_len");
        }
        inserted = false;
        const uint32_t bucket_slot = find_or_create_bucket_compact_at_slot(
            key,
            bitmap_len,
            home_slot,
            inserted,
            bitmap_chunk,
            probe_steps_out
        );
        if (bucket_slot == kInvalidSlot) {
            return false;
        }
        const uint32_t rank_word = static_cast<uint32_t>(rank) >> 6U;
        const uint32_t rank_bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
        const uint32_t offset = slots_.bitmap_offsets[bucket_slot];
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (offset >= capacity || rank_word >= capacity - offset) {
            throw std::logic_error("BC mutable compact bucket bitmap word exceeds arena");
        }
        out = BCCellResolvedInsert{
            1ULL << rank_bit,
            offset + rank_word,
            inserted
        };
        return true;
    }

    [[nodiscard]] bool resolve_one_trusted_key_rank_guarded_at_slot(
        uint64_t key,
        BucketRank rank,
        uint32_t home_slot,
        bool &inserted,
        BCCellResolvedInsert &out,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        inserted = false;
        const uint32_t bucket_slot = find_or_create_bucket_trusted_key_rank_at_slot(
            key,
            rank,
            home_slot,
            inserted,
            bitmap_chunk,
            probe_steps_out
        );
        if (bucket_slot == kInvalidSlot) {
            return false;
        }
#ifndef NDEBUG
        const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(*lut_, key);
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
            throw std::logic_error("BC mutable trusted insert saw invalid bitmap_len");
        }
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC mutable trusted insert rank exceeds bitmap_len");
        }
#endif
        const uint32_t rank_word = static_cast<uint32_t>(rank) >> 6U;
        const uint32_t rank_bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
        const uint32_t offset = slots_.bitmap_offsets[bucket_slot];
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (offset >= capacity || rank_word >= capacity - offset) {
            throw std::logic_error("BC mutable trusted bucket bitmap word exceeds arena");
        }
        out = BCCellResolvedInsert{
            1ULL << rank_bit,
            offset + rank_word,
            inserted
        };
        return true;
    }

    [[nodiscard]] uint32_t find_or_create_bucket_guarded(
        uint64_t key,
        BucketBitmapLen bitmap_len,
        uint32_t word_count,
        bool &inserted,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        return find_or_create_bucket_guarded_at_slot(
            key,
            bitmap_len,
            word_count,
            home_slot_for_key(key),
            inserted,
            bitmap_chunk,
            probe_steps_out
        );
    }

    [[nodiscard]] uint32_t find_or_create_bucket_guarded_at_slot(
        uint64_t key,
        BucketBitmapLen bitmap_len,
        uint32_t word_count,
        uint32_t home_slot,
        bool &inserted,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        inserted = false;

        const uint32_t mask = capacity_mask_;
        uint32_t slot_index = home_slot & mask;
        for (uint32_t probe = 0U; probe < capacity_; ++probe) {
            if (probe_steps_out != nullptr) {
                *probe_steps_out = probe + 1U;
            }
            uint8_t state = slots_.states[slot_index].load(std::memory_order_acquire);
            if (state == kPendingState) {
                while ((state = slots_.states[slot_index].load(std::memory_order_acquire)) == kPendingState) {
                    bc_cell_mutable_spin_pause();
                }
            }
            if (state == kOccupiedState) {
                if (slots_.keys[slot_index] == key) {
                    return slot_index;
                }
            } else if (state == kEmptyState) {
                uint8_t expected = kEmptyState;
                if (slots_.states[slot_index].compare_exchange_strong(
                        expected,
                        kPendingState,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    const BitmapAllocation allocation = allocate_bitmap_words(word_count, bitmap_chunk);
                    if (!allocation.ok) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    uint32_t occupied_index = 0U;
                    if (!try_reserve_occupied_index(occupied_index)) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    slots_.keys[slot_index] = key;
                    slots_.bitmap_offsets[slot_index] = allocation.offset;
                    record_occupied_slot(occupied_index, slot_index);
                    slots_.states[slot_index].store(kOccupiedState, std::memory_order_release);
                    inserted = true;
                    return slot_index;
                }
                continue;
            }
            slot_index = (slot_index + 1U) & mask;
            prefetch_slot_index(slot_index);
        }
        return kInvalidSlot;
    }

    [[nodiscard]] uint32_t find_or_create_bucket_compact_at_slot(
        uint64_t key,
        BucketBitmapLen bitmap_len,
        uint32_t home_slot,
        bool &inserted,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        inserted = false;

        const uint32_t mask = capacity_mask_;
        uint32_t slot_index = home_slot & mask;
        for (uint32_t probe = 0U; probe < capacity_; ++probe) {
            if (probe_steps_out != nullptr) {
                *probe_steps_out = probe + 1U;
            }
            uint8_t state = slots_.states[slot_index].load(std::memory_order_acquire);
            if (state == kPendingState) {
                while ((state = slots_.states[slot_index].load(std::memory_order_acquire)) == kPendingState) {
                    bc_cell_mutable_spin_pause();
                }
            }
            if (state == kOccupiedState) {
                if (slots_.keys[slot_index] == key) {
                    return slot_index;
                }
            } else if (state == kEmptyState) {
                uint8_t expected = kEmptyState;
                if (slots_.states[slot_index].compare_exchange_strong(
                        expected,
                        kPendingState,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    const uint32_t word_count = words_for_bits(bitmap_len);
                    const BitmapAllocation allocation = allocate_bitmap_words(word_count, bitmap_chunk);
                    if (!allocation.ok) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    uint32_t occupied_index = 0U;
                    if (!try_reserve_occupied_index(occupied_index)) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    slots_.keys[slot_index] = key;
                    slots_.bitmap_offsets[slot_index] = allocation.offset;
                    record_occupied_slot(occupied_index, slot_index);
                    slots_.states[slot_index].store(kOccupiedState, std::memory_order_release);
                    inserted = true;
                    return slot_index;
                }
                continue;
            }
            slot_index = (slot_index + 1U) & mask;
            prefetch_slot_index(slot_index);
        }
        return kInvalidSlot;
    }

    [[nodiscard]] uint32_t find_or_create_bucket_trusted_key_rank_at_slot(
        uint64_t key,
        BucketRank rank,
        uint32_t home_slot,
        bool &inserted,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr,
        uint32_t *probe_steps_out = nullptr
    ) {
        inserted = false;

        const uint32_t mask = capacity_mask_;
        uint32_t slot_index = home_slot & mask;
        for (uint32_t probe = 0U; probe < capacity_; ++probe) {
            if (probe_steps_out != nullptr) {
                *probe_steps_out = probe + 1U;
            }
            uint8_t state = slots_.states[slot_index].load(std::memory_order_acquire);
            if (state == kPendingState) {
                while ((state = slots_.states[slot_index].load(std::memory_order_acquire)) == kPendingState) {
                    bc_cell_mutable_spin_pause();
                }
            }
            if (state == kOccupiedState) {
                if (slots_.keys[slot_index] == key) {
                    return slot_index;
                }
            } else if (state == kEmptyState) {
                uint8_t expected = kEmptyState;
                if (slots_.states[slot_index].compare_exchange_strong(
                        expected,
                        kPendingState,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(*lut_, key);
                    if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        throw std::logic_error("BC mutable trusted bucket computed invalid bitmap_len");
                    }
                    if (rank >= bitmap_len) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        throw std::out_of_range("BC mutable trusted insert rank exceeds bitmap_len");
                    }
                    const uint32_t word_count = words_for_bits(bitmap_len);
                    const BitmapAllocation allocation = allocate_bitmap_words(word_count, bitmap_chunk);
                    if (!allocation.ok) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    uint32_t occupied_index = 0U;
                    if (!try_reserve_occupied_index(occupied_index)) {
                        slots_.states[slot_index].store(kEmptyState, std::memory_order_release);
                        return kInvalidSlot;
                    }
                    slots_.keys[slot_index] = key;
                    slots_.bitmap_offsets[slot_index] = allocation.offset;
                    record_occupied_slot(occupied_index, slot_index);
                    slots_.states[slot_index].store(kOccupiedState, std::memory_order_release);
                    inserted = true;
                    return slot_index;
                }
                continue;
            }
            slot_index = (slot_index + 1U) & mask;
            prefetch_slot_index(slot_index);
        }
        return kInvalidSlot;
    }

    void ensure_batch_capacity(uint32_t incoming) const {
        if (incoming == 0U) {
            return;
        }
        const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
        if (buckets <= max_buckets_for_capacity(capacity_) &&
            incoming <= max_buckets_for_capacity(capacity_) - buckets) {
            return;
        }
        grow_hash_table(capacity_for_bucket_count(buckets + incoming));
    }

    void mark_capacity_overflow() const {
        overflowed_.store(true, std::memory_order_release);
    }

    void mark_dirty_once() {
        if (!dirty_.load(std::memory_order_relaxed)) {
            dirty_.store(true, std::memory_order_release);
        }
    }

    void prefetch_home_slot(uint64_t key) const {
#if defined(__GNUC__) || defined(__clang__)
        prefetch_slot_index(home_slot_for_key(key));
#else
        (void)key;
#endif
    }

    void prefetch_slot_index(uint32_t slot) const {
#if defined(__GNUC__) || defined(__clang__)
        if (capacity_ == 0U || !slots_.states) {
            return;
        }
        const uint32_t index = slot & capacity_mask_;
        __builtin_prefetch(&slots_.states[index], 0, 1);
        __builtin_prefetch(&slots_.keys[index], 0, 1);
        __builtin_prefetch(&slots_.bitmap_offsets[index], 0, 1);
#else
        (void)slot;
#endif
    }

    [[nodiscard]] uint32_t find_existing_bucket_no_create(uint64_t key) const {
        if (capacity_ == 0U || !slots_.states) {
            return kInvalidSlot;
        }
        const uint32_t mask = capacity_mask_;
        uint32_t slot_index = slot_for_key_with_shift(key, hash_shift_);
        for (uint32_t probe = 0U; probe < capacity_; ++probe) {
            uint8_t state = slots_.states[slot_index].load(std::memory_order_acquire);
            if (state == kPendingState) {
                while ((state = slots_.states[slot_index].load(std::memory_order_acquire)) == kPendingState) {
                    bc_cell_mutable_spin_pause();
                }
            }
            if (state == kEmptyState) {
                return kInvalidSlot;
            }
            if (state == kOccupiedState && slots_.keys[slot_index] == key) {
                return slot_index;
            }
            slot_index = (slot_index + 1U) & mask;
        }
        return kInvalidSlot;
    }

    [[nodiscard]] uint32_t restore_bucket_slot(uint64_t key, BucketBitmapLen bitmap_len, uint32_t word_count) {
        for (;;) {
            bool inserted = false;
            const uint32_t slot = find_or_create_bucket_guarded(key, bitmap_len, word_count, inserted);
            if (slot != kInvalidSlot) {
                if (!inserted) {
                    throw std::invalid_argument("BC mutable cell dump contains duplicate key");
                }
                return slot;
            }
            grow_hash_table(capacity_for_bucket_count(bucket_count_.load(std::memory_order_acquire) + 1U));
        }
    }

    void grow_hash_table(uint32_t requested_capacity) const {
        if (requested_capacity < kInitialCapacity) {
            requested_capacity = kInitialCapacity;
        }
        requested_capacity = next_power_of_two_at_least(requested_capacity);
        std::unique_lock<std::mutex> lock(resize_mutex_);
        const uint32_t current_buckets = bucket_count_.load(std::memory_order_acquire);
        const uint32_t required_capacity = capacity_for_bucket_count(current_buckets + 1U);
        requested_capacity = std::max(requested_capacity, required_capacity);
        if (capacity_ >= requested_capacity &&
            current_buckets <= max_buckets_for_capacity(capacity_)) {
            return;
        }
        hash_grow_count_.fetch_add(1U, std::memory_order_relaxed);
        resizing_.store(true, std::memory_order_release);
        while (active_inserts_.load(std::memory_order_acquire) != 0U) {
            std::this_thread::yield();
        }
        const uint64_t replaced_bytes =
            static_cast<uint64_t>(capacity_) * slot_metadata_bytes_per_entry() +
            static_cast<uint64_t>(occupied_slots_capacity_) * sizeof(uint32_t);
        SlotArrays new_slots = allocate_slot_table(requested_capacity);
        const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t i = occupied_slots_[index];
            if (slot_state(i) != kOccupiedState) {
                continue;
            }
            place_existing_bucket(new_slots, requested_capacity, i);
        }
        slots_ = std::move(new_slots);
        capacity_ = requested_capacity;
        capacity_mask_ = requested_capacity - 1U;
        hash_shift_ = hash_shift_for_capacity(requested_capacity);
        rebuild_occupied_slots_from_table();
        hash_replaced_bytes_total_.fetch_add(replaced_bytes, std::memory_order_relaxed);
        resizing_.store(false, std::memory_order_release);
    }

    void initialize_empty_table(uint32_t capacity) {
        capacity_ = next_power_of_two_at_least(std::max<uint32_t>(capacity, kInitialCapacity));
        capacity_mask_ = capacity_ - 1U;
        hash_shift_ = hash_shift_for_capacity(capacity_);
        slots_ = allocate_slot_table(capacity_);
        allocate_occupied_slots(capacity_);
    }

    [[nodiscard]] static SlotArrays allocate_slot_table(uint32_t capacity, bool initialize_metadata = true) {
        if ((capacity & (capacity - 1U)) != 0U) {
            throw std::invalid_argument("BC mutable cell flat table capacity must be power of two");
        }
        static_assert(
            sizeof(std::atomic<uint8_t>) == sizeof(uint8_t),
            "BC mutable cell state fast clear expects byte-sized atomic state"
        );
        SlotArrays slots;
        slots.capacity = capacity;
        slots.states = std::unique_ptr<std::atomic<uint8_t>[]>(new std::atomic<uint8_t>[capacity]);
        auto align_up = [](uint64_t value, uint64_t alignment) -> uint64_t {
            return (value + alignment - 1U) & ~(alignment - 1U);
        };
        uint64_t cursor = 0U;
        cursor = align_up(cursor, alignof(uint64_t));
        const uint64_t keys_offset = cursor;
        cursor += static_cast<uint64_t>(capacity) * sizeof(uint64_t);
        cursor = align_up(cursor, alignof(uint32_t));
        const uint64_t offsets_offset = cursor;
        cursor += static_cast<uint64_t>(capacity) * sizeof(uint32_t);
        slots.metadata_storage = bc_allocate_cell_aligned_bytes(cursor);
        uint8_t *base = slots.metadata_storage.get();
        slots.keys = reinterpret_cast<uint64_t *>(base + keys_offset);
        slots.bitmap_offsets = reinterpret_cast<uint32_t *>(base + offsets_offset);
        for (uint32_t i = 0U; i < capacity; ++i) {
            slots.states[i].store(kEmptyState, std::memory_order_relaxed);
        }
        if (!initialize_metadata) {
            return slots;
        }
        for (uint32_t i = 0U; i < capacity; ++i) {
            slots.keys[i] = 0U;
            slots.bitmap_offsets[i] = 0U;
        }
        return slots;
    }

    void place_existing_bucket(SlotArrays &table, uint32_t capacity, uint32_t source_slot) const {
        const uint32_t mask = capacity - 1U;
        uint32_t slot_index = slot_for_key(slots_.keys[source_slot], capacity);
        while (table.states[slot_index].load(std::memory_order_relaxed) == kOccupiedState) {
            slot_index = (slot_index + 1U) & mask;
        }
        table.keys[slot_index] = slots_.keys[source_slot];
        table.bitmap_offsets[slot_index] = slots_.bitmap_offsets[source_slot];
        table.states[slot_index].store(kOccupiedState, std::memory_order_release);
    }

    void allocate_occupied_slots(uint32_t capacity) const {
        occupied_slots_.reset(new uint32_t[capacity]);
        occupied_slots_capacity_ = capacity;
    }

    void record_occupied_slot(uint32_t index, uint32_t slot) const {
        if (index >= occupied_slots_capacity_ || occupied_slots_ == nullptr) {
            mark_capacity_overflow();
            throw BCCellMutableBuilderOverflow("BC mutable cell occupied slot list overflow");
        }
        occupied_slots_[index] = slot;
    }

    void rebuild_occupied_slots_from_table() const {
        if (occupied_slots_capacity_ < capacity_ || occupied_slots_ == nullptr) {
            allocate_occupied_slots(capacity_);
        }
        uint32_t out = 0U;
        for (uint32_t slot = 0U; slot < capacity_; ++slot) {
            if (slot_state(slot) == kOccupiedState) {
                occupied_slots_[out++] = slot;
            }
        }
        if (out != bucket_count_.load(std::memory_order_acquire)) {
            throw std::logic_error("BC mutable cell occupied slot list rebuild count mismatch");
        }
    }

    [[nodiscard]] bool should_grow_for_one_more() const {
        const uint32_t buckets = bucket_count_.load(std::memory_order_acquire);
        return buckets + 1U > max_buckets_for_capacity(capacity_);
    }

    [[nodiscard]] bool try_reserve_occupied_index(uint32_t &index) const {
        const uint32_t max_buckets = std::min(
            max_buckets_for_capacity(capacity_),
            occupied_slots_capacity_
        );
        uint32_t current = bucket_count_.load(std::memory_order_acquire);
        while (current < max_buckets) {
            if (bucket_count_.compare_exchange_weak(
                    current,
                    current + 1U,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                index = current;
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static uint32_t max_buckets_for_capacity(uint32_t capacity) {
        return static_cast<uint32_t>((static_cast<uint64_t>(capacity) * kLoadNumerator) / kLoadDenominator);
    }

    [[nodiscard]] static uint32_t capacity_for_bucket_count(uint32_t buckets) {
        if (buckets == 0U) {
            return kInitialCapacity;
        }
        const uint64_t needed =
            (static_cast<uint64_t>(buckets) * kLoadDenominator + kLoadNumerator - 1U) / kLoadNumerator;
        if (needed > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable cell flat table capacity overflow");
        }
        return next_power_of_two_at_least(static_cast<uint32_t>(std::max<uint64_t>(needed, kInitialCapacity)));
    }

    [[nodiscard]] static uint32_t next_power_of_two_at_least(uint32_t value) {
        if (value <= 1U) {
            return 1U;
        }
        --value;
        value |= value >> 1U;
        value |= value >> 2U;
        value |= value >> 4U;
        value |= value >> 8U;
        value |= value >> 16U;
        if (value == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC mutable cell next power-of-two overflow");
        }
        return value + 1U;
    }

    [[nodiscard]] BitmapAllocation allocate_bitmap_words(
        uint32_t word_count,
        BCCellBitmapThreadChunk *bitmap_chunk = nullptr
    ) {
        if (word_count == 0U) {
            throw std::invalid_argument("BC mutable cell cannot allocate empty bitmap");
        }
        if (bitmap_chunk != nullptr && bitmap_chunk->word_next <= bitmap_chunk->word_end &&
            word_count <= bitmap_chunk->word_end - bitmap_chunk->word_next) {
            const uint32_t offset = bitmap_chunk->word_next;
            bitmap_chunk->word_next += word_count;
            clear_bitmap_words(offset, word_count);
            return BitmapAllocation{offset, true};
        }
        if (bitmap_chunk != nullptr) {
            constexpr uint32_t kChunkWords = 128U;
            const uint32_t chunk_words = std::max<uint32_t>(kChunkWords, word_count);
            const BitmapAllocation chunk = acquire_bitmap_words_from_cursor(chunk_words);
            if (!chunk.ok) {
                return chunk;
            }
            bitmap_chunk->word_next = chunk.offset + word_count;
            bitmap_chunk->word_end = chunk.offset + chunk_words;
            clear_bitmap_words(chunk.offset, word_count);
            return BitmapAllocation{chunk.offset, true};
        }
        return acquire_bitmap_words_exact(word_count);
    }

    [[nodiscard]] BitmapAllocation acquire_bitmap_words_from_cursor(uint32_t word_count) {
        uint32_t used = bitmap_used_words_.load(std::memory_order_relaxed);
        uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        while (used <= capacity && word_count <= capacity - used) {
            if (bitmap_used_words_.compare_exchange_weak(
                    used,
                    used + word_count,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return BitmapAllocation{used, true};
            }
            capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        }

        return BitmapAllocation{0U, false};
    }

    [[nodiscard]] BitmapAllocation acquire_bitmap_words_exact(uint32_t word_count) {
        const BitmapAllocation out = acquire_bitmap_words_from_cursor(word_count);
        if (out.ok) {
            clear_bitmap_words(out.offset, word_count);
        }
        return out;
    }

    [[nodiscard]] BitmapAllocation acquire_bitmap_words_exact_locked(uint32_t word_count) {
        std::lock_guard<std::mutex> lock(bitmap_alloc_mutex_);
        uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
        uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (used > capacity || word_count > capacity - used) {
            return BitmapAllocation{0U, false};
        }
        used = bitmap_used_words_.load(std::memory_order_relaxed);
        while (used <= capacity && word_count <= capacity - used) {
            if (bitmap_used_words_.compare_exchange_weak(
                    used,
                    used + word_count,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return BitmapAllocation{used, true};
            }
        }
        mark_capacity_overflow();
        return BitmapAllocation{0U, false};
    }

    void reserve_bitmap_words(uint32_t words) {
        std::lock_guard<std::mutex> lock(bitmap_alloc_mutex_);
        if (bitmap_capacity_words_.load(std::memory_order_acquire) >= words) {
            return;
        }
        if (!grow_bitmap_arena_locked(words)) {
            throw std::overflow_error("BC mutable bitmap arena reserve overflow");
        }
    }

    void reserve_bitmap_words_stop_world(uint32_t words) const {
        if (bitmap_capacity_words_.load(std::memory_order_acquire) >= words) {
            return;
        }
        std::unique_lock<std::mutex> resize_lock(resize_mutex_);
        resizing_.store(true, std::memory_order_release);
        while (active_inserts_.load(std::memory_order_acquire) != 0U) {
            std::this_thread::yield();
        }
        std::lock_guard<std::mutex> bitmap_lock(bitmap_alloc_mutex_);
        if (bitmap_capacity_words_.load(std::memory_order_acquire) < words &&
            !const_cast<BCCellMutableBuilder *>(this)->grow_bitmap_arena_locked(words)) {
            resizing_.store(false, std::memory_order_release);
            throw std::overflow_error("BC mutable bitmap arena stop-world reserve overflow");
        }
        resizing_.store(false, std::memory_order_release);
    }

    [[nodiscard]] bool grow_bitmap_arena_locked(uint32_t required_words) {
        const uint32_t current_capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (current_capacity >= required_words) {
            return true;
        }
        bitmap_grow_count_.fetch_add(1U, std::memory_order_relaxed);
        const uint32_t doubled_capacity =
            current_capacity > std::numeric_limits<uint32_t>::max() / 2U
                ? std::numeric_limits<uint32_t>::max()
                : current_capacity * 2U;
        uint32_t target = std::max<uint32_t>(
            required_words,
            std::max<uint32_t>(kInitialBitmapArenaWords, doubled_capacity)
        );
        if (target < current_capacity) {
            return false;
        }
        while (target < required_words) {
            if (target > std::numeric_limits<uint32_t>::max() / 2U) {
                target = required_words;
                break;
            }
            target *= 2U;
        }
        auto next = bc_allocate_cell_bitmap_arena(target);
        const uint32_t used = bitmap_used_words_.load(std::memory_order_acquire);
        if (used > target) {
            return false;
        }
        const uint64_t *old_base = bitmap_base();
        const uint64_t replaced_bytes =
            static_cast<uint64_t>(current_capacity) * sizeof(uint64_t);
        for (uint32_t i = 0U; i < used; ++i) {
            next[i] = old_base != nullptr ? old_base[i] : 0U;
        }
        bitmap_arena_ = std::move(next);
        bitmap_capacity_words_.store(target, std::memory_order_release);
        bitmap_replaced_bytes_total_.fetch_add(replaced_bytes, std::memory_order_relaxed);
        return true;
    }

    void clear_bitmap_words(uint32_t offset, uint32_t word_count) {
        uint64_t *base = bitmap_base();
        if (base == nullptr) {
            throw std::logic_error("BC mutable bitmap arena is not allocated");
        }
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (offset > capacity || word_count > capacity - offset) {
            throw std::logic_error("BC mutable bitmap clear range exceeds arena");
        }
        for (uint32_t i = 0U; i < word_count; ++i) {
            base[offset + i] = 0U;
        }
    }

    [[nodiscard]] static uint32_t slot_for_key(uint64_t key, uint32_t capacity) {
        if (capacity == 0U || (capacity & (capacity - 1U)) != 0U) {
            throw std::logic_error("BC mutable cell flat table capacity must be non-zero power of two");
        }
        const uint64_t mixed = key * 11400714819323198485ULL;
        const uint32_t shift = hash_shift_for_capacity(capacity);
        return shift >= 64U ? 0U : static_cast<uint32_t>(mixed >> shift);
    }

    [[nodiscard]] static uint32_t hash_shift_for_capacity(uint32_t capacity) {
        if (capacity == 0U || (capacity & (capacity - 1U)) != 0U) {
            throw std::logic_error("BC mutable cell flat table capacity must be non-zero power of two");
        }
        const uint32_t bits =
#if defined(__GNUC__) || defined(__clang__)
            static_cast<uint32_t>(__builtin_ctz(capacity));
#else
            [] (uint32_t value) {
                uint32_t out = 0U;
                while ((value >>= 1U) != 0U) {
                    ++out;
                }
                return out;
            }(capacity);
#endif
        return bits == 0U ? 64U : 64U - bits;
    }

    [[nodiscard]] static uint32_t slot_for_key_with_shift(uint64_t key, uint32_t shift) {
        const uint64_t mixed = key * 11400714819323198485ULL;
        return shift >= 64U ? 0U : static_cast<uint32_t>(mixed >> shift);
    }

    [[nodiscard]] static uint16_t checked_u16(uint32_t value, const char *label) {
        if (value > std::numeric_limits<uint16_t>::max()) {
            throw std::overflow_error(label);
        }
        return static_cast<uint16_t>(value);
    }

    [[nodiscard]] static uint32_t checked_u32(uint64_t value, const char *label) {
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(label);
        }
        return static_cast<uint32_t>(value);
    }

    [[nodiscard]] static constexpr uint64_t slot_metadata_bytes_per_entry() {
        return sizeof(std::atomic<uint8_t>) +
            sizeof(uint64_t) +
            sizeof(uint32_t);
    }

    static void append_padding(std::vector<uint8_t> &buffer, uint32_t bytes) {
        buffer.insert(buffer.end(), bytes, 0U);
    }

    static void append_u16_le(std::vector<uint8_t> &buffer, uint16_t value) {
        buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
        buffer.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    }

    static void append_u32_le(std::vector<uint8_t> &buffer, uint32_t value) {
        for (uint32_t i = 0U; i < 4U; ++i) {
            buffer.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
        }
    }

    static void append_u64_le(std::vector<uint8_t> &buffer, uint64_t value) {
        for (uint32_t i = 0U; i < 8U; ++i) {
            buffer.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
        }
    }

    static void store_u16_le(uint8_t *data, uint16_t value) {
        data[0] = static_cast<uint8_t>(value & 0xFFU);
        data[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    }

    static void store_u32_le(uint8_t *data, uint32_t value) {
        data[0] = static_cast<uint8_t>(value & 0xFFU);
        data[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        data[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        data[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    }

    static void store_u64_le(uint8_t *data, uint64_t value) {
        data[0] = static_cast<uint8_t>(value & 0xFFU);
        data[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        data[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        data[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
        data[4] = static_cast<uint8_t>((value >> 32U) & 0xFFU);
        data[5] = static_cast<uint8_t>((value >> 40U) & 0xFFU);
        data[6] = static_cast<uint8_t>((value >> 48U) & 0xFFU);
        data[7] = static_cast<uint8_t>((value >> 56U) & 0xFFU);
    }

    [[nodiscard]] static uint32_t load_u32_le_unchecked(const uint8_t *data) {
        return
            static_cast<uint32_t>(data[0]) |
            (static_cast<uint32_t>(data[1]) << 8U) |
            (static_cast<uint32_t>(data[2]) << 16U) |
            (static_cast<uint32_t>(data[3]) << 24U);
    }

    [[nodiscard]] static uint64_t atomic_load_bitmap_word(const uint64_t *word) {
#if defined(__GNUC__) || defined(__clang__)
        return __atomic_load_n(word, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
        return static_cast<uint64_t>(
            _InterlockedCompareExchange64(
                reinterpret_cast<volatile long long *>(const_cast<uint64_t *>(word)),
                0,
                0
            )
        );
#else
#error "BCCellMutableBuilder requires a 64-bit relaxed atomic load implementation"
#endif
    }

    [[nodiscard]] static uint64_t atomic_fetch_or_bitmap_word(uint64_t *word, uint64_t mask) {
#if defined(__GNUC__) || defined(__clang__)
        return __atomic_fetch_or(word, mask, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
        return static_cast<uint64_t>(
            _InterlockedOr64(
                reinterpret_cast<volatile long long *>(word),
                static_cast<long long>(mask)
            )
        );
#else
#error "BCCellMutableBuilder requires a 64-bit relaxed atomic fetch_or implementation"
#endif
    }

    [[nodiscard]] static uint64_t align_up_u64_local(uint64_t value, uint64_t alignment) {
        if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
            throw std::invalid_argument("BC mutable dump alignment must be power-of-two");
        }
        if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
            throw std::overflow_error("BC mutable dump align_up overflow");
        }
        return (value + alignment - 1U) & ~(alignment - 1U);
    }

    [[nodiscard]] static uint64_t tail_mask_for_bits(uint32_t bitmap_len, uint32_t word) {
        const uint32_t word_count = words_for_bits(bitmap_len);
        if (word + 1U != word_count) {
            return ~0ULL;
        }
        const uint32_t tail = bitmap_len & (kBCBitmapWordBits - 1U);
        return tail == 0U ? ~0ULL : ((1ULL << tail) - 1ULL);
    }

    [[nodiscard]] uint64_t masked_bitmap_word(uint32_t slot, uint32_t word) const {
        uint64_t *bitmap_word = slot_bitmap_word(slot, word);
        if (bitmap_word == nullptr) {
            throw std::logic_error("BC mutable cell bitmap word is out of range");
        }
        return atomic_load_bitmap_word(bitmap_word) &
            tail_mask_for_bits(slot_bitmap_len(slot), word);
    }

    [[nodiscard]] uint64_t masked_bitmap_word_stopped(uint32_t slot, uint32_t word) const {
        uint64_t *bitmap_word = slot_bitmap_word(slot, word);
        if (bitmap_word == nullptr) {
            throw std::logic_error("BC mutable cell bitmap word is out of range");
        }
        return *bitmap_word & tail_mask_for_bits(slot_bitmap_len(slot), word);
    }

    [[nodiscard]] uint32_t live_count(uint32_t slot) const {
        uint32_t live = 0U;
        const uint32_t word_count = slot_word_count(slot);
        for (uint32_t word = 0U; word < word_count; ++word) {
            live += popcount64(masked_bitmap_word(slot, word));
        }
        return live;
    }

    [[nodiscard]] uint32_t live_count_stopped(uint32_t slot) const {
        uint32_t live = 0U;
        const uint32_t word_count = slot_word_count(slot);
        for (uint32_t word = 0U; word < word_count; ++word) {
            live += popcount64(masked_bitmap_word_stopped(slot, word));
        }
        return live;
    }

    void append_prefix256_le(std::vector<uint8_t> &buffer, uint32_t slot) const {
        const BucketBitmapLen bitmap_len = slot_bitmap_len(slot);
        append_prefix256_le(buffer, slot, bitmap_len, words_for_bits(bitmap_len));
    }

    void append_prefix256_le(
        std::vector<uint8_t> &buffer,
        uint32_t slot,
        BucketBitmapLen bitmap_len,
        uint32_t word_count
    ) const {
        uint32_t running = 0U;
        const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
        for (uint32_t block = 0U; block < prefix_count; ++block) {
            if (running > std::numeric_limits<RankPrefix>::max()) {
                throw std::logic_error("BC mutable prefix running popcount exceeds uint16");
            }
            append_u16_le(buffer, static_cast<RankPrefix>(running));
            const uint32_t block_first_bit = block * kBCRankPrefixBits;
            const uint32_t block_last_bit = std::min<uint32_t>(
                bitmap_len,
                block_first_bit + kBCRankPrefixBits
            );
            const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
            const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
            if (last_word_exclusive > word_count) {
                throw std::invalid_argument("BC mutable prefix bitmap is shorter than bitmap_len");
            }
            for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
                uint64_t value = masked_bitmap_word_stopped(slot, word);
                if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                    value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
                }
                running += popcount64(value);
            }
        }
    }

    void append_bitmap_le(std::vector<uint8_t> &buffer, uint32_t slot) const {
        const BucketBitmapLen bitmap_len = slot_bitmap_len(slot);
        const uint32_t word_count = words_for_bits(bitmap_len);
        append_bitmap_le(buffer, slot, bitmap_len, word_count, slots_.bitmap_offsets[slot]);
    }

    void append_bitmap_le(
        std::vector<uint8_t> &buffer,
        uint32_t slot,
        BucketBitmapLen bitmap_len,
        uint32_t word_count,
        uint32_t source_offset
    ) const {
        if (host_is_little_endian()) {
            uint64_t *base = bitmap_base();
            const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
            if (base == nullptr ||
                source_offset > capacity ||
                word_count > capacity - source_offset) {
                throw std::logic_error("BC mutable cell finalize bitmap range exceeds arena");
            }
            const size_t old_size = buffer.size();
            const size_t bytes = static_cast<size_t>(word_count) * sizeof(uint64_t);
            buffer.resize(old_size + bytes);
            std::memcpy(buffer.data() + old_size, base + source_offset, bytes);
            if (word_count != 0U) {
                uint64_t last = 0U;
                uint8_t *last_bytes =
                    buffer.data() + old_size + static_cast<size_t>(word_count - 1U) * sizeof(uint64_t);
                std::memcpy(&last, last_bytes, sizeof(uint64_t));
                last &= tail_mask_for_bits(bitmap_len, word_count - 1U);
                std::memcpy(last_bytes, &last, sizeof(uint64_t));
            }
        } else {
            for (uint32_t word = 0U; word < word_count; ++word) {
                append_u64_le(buffer, masked_bitmap_word_stopped(slot, word));
            }
        }
    }

    void sorted_bucket_refs_into(
        BCCellFinalizeOptions options,
        std::vector<SortedBucketRef> &sorted,
        std::vector<uint64_t> &keys,
        std::vector<uint32_t> &indices,
        std::vector<SortedBucketRef> &reordered
    ) const {
        g_bc_cell_finalize_debug_cid.store(cid_, std::memory_order_relaxed);
        g_bc_cell_finalize_debug_bucket_count.store(
            bucket_count_.load(std::memory_order_relaxed),
            std::memory_order_relaxed
        );
        g_bc_cell_finalize_debug_sorted_size.store(0U, std::memory_order_relaxed);
        g_bc_cell_finalize_debug_stage.store(1U, std::memory_order_relaxed);
        sorted.clear();
        sorted.reserve(bucket_count_.load(std::memory_order_relaxed));
        const uint32_t buckets = bucket_count_.load(std::memory_order_relaxed);
        uint64_t *base = bitmap_base();
        for (uint32_t index = 0U; index < buckets; ++index) {
            const uint32_t slot = occupied_slots_[index];
            if (slot_state(slot) != kOccupiedState) {
                continue;
            }
            const uint64_t key = slots_.keys[slot];
            const BucketBitmapLen bitmap_len = bitmap_len_from_key(*lut_, key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            const uint32_t bitmap_offset = slots_.bitmap_offsets[slot];
            if (base == nullptr && word_count != 0U) {
                throw std::logic_error("BC mutable cell sorted refs missing bitmap arena");
            }
            const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
            if (bitmap_offset > capacity || word_count > capacity - bitmap_offset) {
                throw std::logic_error("BC mutable cell sorted refs bitmap range exceeds arena");
            }
            uint32_t bucket_live = 0U;
            for (uint32_t word = 0U; word < word_count; ++word) {
                bucket_live += popcount64(
                    base[bitmap_offset + word] & tail_mask_for_bits(bitmap_len, word)
                );
            }
            if (bucket_live == 0U) {
                continue;
            }
            sorted.push_back(SortedBucketRef{key, slot, bitmap_offset, bitmap_len, word_count, bucket_live});
        }
        g_bc_cell_finalize_debug_sorted_size.store(
            static_cast<uint32_t>(std::min<size_t>(sorted.size(), std::numeric_limits<uint32_t>::max())),
            std::memory_order_relaxed
        );
        g_bc_cell_finalize_debug_stage.store(2U, std::memory_order_relaxed);

        const bool use_keyvalue_sort =
            options.keyvalue_sort != nullptr &&
            sorted.size() >= options.simd_sort_min_bucket_count;
        if (use_keyvalue_sort) {
            g_bc_cell_finalize_debug_stage.store(3U, std::memory_order_relaxed);
            keys.clear();
            indices.clear();
            keys.reserve(sorted.size());
            indices.reserve(sorted.size());
            for (uint32_t i = 0U; i < sorted.size(); ++i) {
                keys.push_back(sorted[i].key);
                indices.push_back(i);
            }
            options.keyvalue_sort(keys.data(), indices.data(), keys.size(), false);
            reordered.clear();
            reordered.reserve(keys.size());
            for (size_t i = 0U; i < indices.size(); ++i) {
                const uint32_t index = indices[i];
                if (index >= sorted.size()) {
                    throw std::logic_error("BC mutable keyvalue sorter returned out-of-range index");
                }
                SortedBucketRef ref = sorted[index];
                ref.key = keys[i];
                reordered.push_back(ref);
            }
            sorted.swap(reordered);
        } else {
            g_bc_cell_finalize_debug_stage.store(4U, std::memory_order_relaxed);
            indices.clear();
            indices.reserve(sorted.size());
            for (uint32_t i = 0U; i < sorted.size(); ++i) {
                indices.push_back(i);
            }
            std::sort(
                indices.begin(),
                indices.end(),
                [&sorted](uint32_t lhs, uint32_t rhs) {
                    return sorted[lhs].key < sorted[rhs].key;
                }
            );
            reordered.clear();
            reordered.reserve(sorted.size());
            for (uint32_t index : indices) {
                if (index >= sorted.size()) {
                    throw std::logic_error("BC mutable fallback sorter returned invalid index");
                }
                reordered.push_back(sorted[index]);
            }
            sorted.swap(reordered);
        }
        g_bc_cell_finalize_debug_stage.store(5U, std::memory_order_relaxed);
    }

    [[nodiscard]] uint32_t estimate_rank_payload_bytes(const std::vector<SortedBucketRef> &sorted) const {
        uint32_t cursor = 0U;
        for (const SortedBucketRef &ref : sorted) {
            cursor = align_up_u32(cursor, 8U);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(cursor, ref.bitmap_len);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(ref.word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable rank payload bitmap bytes overflow");
            }
            cursor = checked_u32_add(
                bitmap_offset,
                static_cast<uint32_t>(bitmap_bytes64),
                "BC mutable rank payload reserve overflow"
            );
        }
        return cursor;
    }

    [[nodiscard]] static uint32_t estimate_dump_rank_payload_bytes(
        const std::vector<DumpBucketRef> &sorted
    ) {
        uint32_t cursor = 0U;
        for (const DumpBucketRef &ref : sorted) {
            cursor = align_up_u32(cursor, 8U);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(cursor, ref.bitmap_len);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(ref.word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable dump rank payload bitmap bytes overflow");
            }
            cursor = checked_u32_add(
                bitmap_offset,
                static_cast<uint32_t>(bitmap_bytes64),
                "BC mutable dump rank payload reserve overflow"
            );
        }
        return cursor;
    }

    [[nodiscard]] StreamedFinalizePlan measure_sorted_streamed_plan(
        const std::vector<SortedBucketRef> &sorted
    ) const {
        StreamedFinalizePlan plan;
        plan.bucket_count = checked_u32(sorted.size(), "BC mutable streamed bucket count exceeds uint32");
        plan.rank_payload_bytes = estimate_rank_payload_bytes(sorted);
        uint64_t rows = 0U;
        for (const SortedBucketRef &ref : sorted) {
            rows += ref.live_count;
            if (rows > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed success rows exceed uint32");
            }
        }
        plan.success_rows = static_cast<uint32_t>(rows);
        return plan;
    }

    [[nodiscard]] static StreamedFinalizePlan measure_dump_streamed_plan(
        const std::vector<DumpBucketRef> &sorted
    ) {
        StreamedFinalizePlan plan;
        plan.bucket_count = checked_u32(sorted.size(), "BC mutable streamed dump bucket count exceeds uint32");
        plan.rank_payload_bytes = estimate_dump_rank_payload_bytes(sorted);
        uint64_t rows = 0U;
        for (const DumpBucketRef &ref : sorted) {
            rows += ref.live_count;
            if (rows > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed dump success rows exceed uint32");
            }
        }
        plan.success_rows = static_cast<uint32_t>(rows);
        return plan;
    }

    static void sort_dump_refs_for_finalize(
        FinalizeScratch &scratch,
        BCCellFinalizeOptions options
    ) {
        const bool use_keyvalue_sort =
            options.keyvalue_sort != nullptr &&
            scratch.dump_sorted.size() >= options.simd_sort_min_bucket_count;
        if (use_keyvalue_sort) {
            scratch.sort_keys.clear();
            scratch.sort_indices.clear();
            scratch.sort_keys.reserve(scratch.dump_sorted.size());
            scratch.sort_indices.reserve(scratch.dump_sorted.size());
            for (uint32_t i = 0U; i < scratch.dump_sorted.size(); ++i) {
                scratch.sort_keys.push_back(scratch.dump_sorted[i].key);
                scratch.sort_indices.push_back(i);
            }
            options.keyvalue_sort(
                scratch.sort_keys.data(),
                scratch.sort_indices.data(),
                scratch.sort_keys.size(),
                false
            );
            scratch.dump_reordered.clear();
            scratch.dump_reordered.reserve(scratch.dump_sorted.size());
            for (uint32_t index : scratch.sort_indices) {
                if (index >= scratch.dump_sorted.size()) {
                    throw std::logic_error("BC mutable dump sorter returned invalid index");
                }
                DumpBucketRef ref = scratch.dump_sorted[index];
                ref.key = scratch.sort_keys[scratch.dump_reordered.size()];
                scratch.dump_reordered.push_back(ref);
            }
            scratch.dump_sorted.swap(scratch.dump_reordered);
        } else {
            scratch.sort_indices.clear();
            scratch.sort_indices.reserve(scratch.dump_sorted.size());
            for (uint32_t i = 0U; i < scratch.dump_sorted.size(); ++i) {
                scratch.sort_indices.push_back(i);
            }
            std::sort(
                scratch.sort_indices.begin(),
                scratch.sort_indices.end(),
                [&scratch](uint32_t lhs, uint32_t rhs) {
                    return scratch.dump_sorted[lhs].key < scratch.dump_sorted[rhs].key;
                }
            );
            scratch.dump_reordered.clear();
            scratch.dump_reordered.reserve(scratch.dump_sorted.size());
            for (uint32_t index : scratch.sort_indices) {
                if (index >= scratch.dump_sorted.size()) {
                    throw std::logic_error("BC mutable dump fallback sorter returned invalid index");
                }
                scratch.dump_reordered.push_back(scratch.dump_sorted[index]);
            }
            scratch.dump_sorted.swap(scratch.dump_reordered);
        }
    }

    template <class EmitRank>
    static void emit_rank_padding(EmitRank &&emit_rank, uint32_t bytes) {
        if (bytes == 0U) {
            return;
        }
        const uint64_t zeros = 0U;
        while (bytes != 0U) {
            const uint32_t chunk = std::min<uint32_t>(bytes, sizeof(zeros));
            emit_rank(&zeros, chunk);
            bytes -= chunk;
        }
    }

    template <class EmitRank>
    void emit_resident_prefix_chunk(
        const SortedBucketRef &ref,
        FinalizeScratch &scratch,
        EmitRank &&emit_rank
    ) const {
        scratch.rank_chunk.clear();
        scratch.rank_chunk.reserve(static_cast<size_t>(prefix_count_for_bits(ref.bitmap_len)) * sizeof(RankPrefix));
        uint32_t running = 0U;
        const uint32_t prefix_count = prefix_count_for_bits(ref.bitmap_len);
        for (uint32_t block = 0U; block < prefix_count; ++block) {
            if (running > std::numeric_limits<RankPrefix>::max()) {
                throw std::logic_error("BC mutable streamed prefix running popcount exceeds uint16");
            }
            append_u16_le(scratch.rank_chunk, static_cast<RankPrefix>(running));
            const uint32_t block_first_bit = block * kBCRankPrefixBits;
            const uint32_t block_last_bit = std::min<uint32_t>(
                ref.bitmap_len,
                block_first_bit + kBCRankPrefixBits
            );
            const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
            const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
            if (last_word_exclusive > ref.word_count) {
                throw std::invalid_argument("BC mutable streamed prefix bitmap is shorter than bitmap_len");
            }
            for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
                uint64_t value = masked_bitmap_word_stopped(ref.slot, word);
                if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                    value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
                }
                running += popcount64(value);
            }
        }
        if (!scratch.rank_chunk.empty()) {
            emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
        }
    }

    template <class EmitRank>
    static void emit_dump_prefix_chunk(
        const uint8_t *bitmap,
        const DumpBucketRef &ref,
        FinalizeScratch &scratch,
        EmitRank &&emit_rank
    ) {
        scratch.rank_chunk.clear();
        scratch.rank_chunk.reserve(static_cast<size_t>(prefix_count_for_bits(ref.bitmap_len)) * sizeof(RankPrefix));
        uint32_t running = 0U;
        const uint32_t prefix_count = prefix_count_for_bits(ref.bitmap_len);
        for (uint32_t block = 0U; block < prefix_count; ++block) {
            if (running > std::numeric_limits<RankPrefix>::max()) {
                throw std::logic_error("BC mutable streamed dump prefix running popcount exceeds uint16");
            }
            append_u16_le(scratch.rank_chunk, static_cast<RankPrefix>(running));
            const uint32_t block_first_bit = block * kBCRankPrefixBits;
            const uint32_t block_last_bit = std::min<uint32_t>(
                ref.bitmap_len,
                block_first_bit + kBCRankPrefixBits
            );
            const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
            const uint32_t last_word_exclusive = words_for_bits(block_last_bit);
            if (last_word_exclusive > ref.word_count) {
                throw std::invalid_argument("BC mutable streamed dump prefix bitmap is shorter than bitmap_len");
            }
            for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
                uint64_t value =
                    load_u64_le(bitmap + static_cast<size_t>(ref.bitmap_offset + word) * sizeof(uint64_t)) &
                    tail_mask_for_bits(ref.bitmap_len, word);
                if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                    value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
                }
                running += popcount64(value);
            }
        }
        if (!scratch.rank_chunk.empty()) {
            emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
        }
    }

    template <class EmitRank>
    void emit_resident_bitmap_chunk(
        const SortedBucketRef &ref,
        FinalizeScratch &scratch,
        EmitRank &&emit_rank
    ) const {
        uint64_t *base = bitmap_base();
        const uint32_t capacity = bitmap_capacity_words_.load(std::memory_order_acquire);
        if (base == nullptr ||
            ref.bitmap_offset > capacity ||
            ref.word_count > capacity - ref.bitmap_offset) {
            throw std::logic_error("BC mutable streamed bitmap range exceeds arena");
        }
        if (ref.word_count == 0U) {
            return;
        }
        const uint64_t tail_mask = tail_mask_for_bits(ref.bitmap_len, ref.word_count - 1U);
        if (host_is_little_endian()) {
            const uint32_t direct_words =
                tail_mask == ~0ULL ? ref.word_count : ref.word_count - 1U;
            if (direct_words != 0U) {
                emit_rank(
                    reinterpret_cast<const uint8_t *>(base + ref.bitmap_offset),
                    static_cast<uint64_t>(direct_words) * sizeof(uint64_t)
                );
            }
            if (direct_words != ref.word_count) {
                scratch.rank_chunk.resize(sizeof(uint64_t));
                uint64_t last = base[ref.bitmap_offset + ref.word_count - 1U] & tail_mask;
                store_u64_le(scratch.rank_chunk.data(), last);
                emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
            }
        } else {
            scratch.rank_chunk.clear();
            scratch.rank_chunk.reserve(static_cast<size_t>(ref.word_count) * sizeof(uint64_t));
            for (uint32_t word = 0U; word < ref.word_count; ++word) {
                const uint64_t value =
                    base[ref.bitmap_offset + word] & tail_mask_for_bits(ref.bitmap_len, word);
                append_u64_le(scratch.rank_chunk, value);
            }
            if (!scratch.rank_chunk.empty()) {
                emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
            }
        }
    }

    template <class EmitRank>
    static void emit_dump_bitmap_chunk(
        const uint8_t *bitmap,
        const DumpBucketRef &ref,
        FinalizeScratch &scratch,
        EmitRank &&emit_rank
    ) {
        if (ref.word_count == 0U) {
            return;
        }
        const uint64_t tail_mask = tail_mask_for_bits(ref.bitmap_len, ref.word_count - 1U);
        const uint8_t *source = bitmap + static_cast<size_t>(ref.bitmap_offset) * sizeof(uint64_t);
        if (host_is_little_endian()) {
            const uint32_t direct_words =
                tail_mask == ~0ULL ? ref.word_count : ref.word_count - 1U;
            if (direct_words != 0U) {
                emit_rank(source, static_cast<uint64_t>(direct_words) * sizeof(uint64_t));
            }
            if (direct_words != ref.word_count) {
                const uint64_t last =
                    load_u64_le(source + static_cast<size_t>(ref.word_count - 1U) * sizeof(uint64_t)) &
                    tail_mask;
                scratch.rank_chunk.resize(sizeof(uint64_t));
                store_u64_le(scratch.rank_chunk.data(), last);
                emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
            }
        } else {
            scratch.rank_chunk.clear();
            scratch.rank_chunk.reserve(static_cast<size_t>(ref.word_count) * sizeof(uint64_t));
            for (uint32_t word = 0U; word < ref.word_count; ++word) {
                const uint64_t value =
                    load_u64_le(source + static_cast<size_t>(word) * sizeof(uint64_t)) &
                    tail_mask_for_bits(ref.bitmap_len, word);
                append_u64_le(scratch.rank_chunk, value);
            }
            if (!scratch.rank_chunk.empty()) {
                emit_rank(scratch.rank_chunk.data(), scratch.rank_chunk.size());
            }
        }
    }

    template <class EmitBucket, class EmitRank>
    void emit_sorted_streamed_payload(
        const std::vector<SortedBucketRef> &sorted,
        FinalizeScratch &scratch,
        EmitBucket &&emit_bucket,
        EmitRank &&emit_rank
    ) const {
        uint32_t rank_cursor = 0U;
        uint64_t success_cursor = 0U;
        for (const SortedBucketRef &ref : sorted) {
            if (slot_state(ref.slot) != kOccupiedState) {
                throw std::logic_error("BC mutable streamed finalize saw unoccupied bucket");
            }
            rank_cursor = emit_rank_alignment(rank_cursor, emit_rank);
            const uint32_t rank_payload_offset = rank_cursor;
            emit_resident_prefix_chunk(ref, scratch, emit_rank);
            rank_cursor = checked_u32_add(
                rank_cursor,
                static_cast<uint32_t>(prefix_count_for_bits(ref.bitmap_len) * sizeof(RankPrefix)),
                "BC mutable streamed prefix cursor overflow"
            );
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, ref.bitmap_len);
            if (bitmap_offset < rank_cursor) {
                throw std::logic_error("BC mutable streamed bitmap offset precedes prefix end");
            }
            emit_rank_padding(emit_rank, bitmap_offset - rank_cursor);
            rank_cursor = bitmap_offset;
            emit_resident_bitmap_chunk(ref, scratch, emit_rank);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(ref.word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed bitmap byte count exceeds uint32");
            }
            rank_cursor = checked_u32_add(
                rank_cursor,
                static_cast<uint32_t>(bitmap_bytes64),
                "BC mutable streamed bitmap cursor overflow"
            );
            emit_bucket(BCBucketEntry{
                ref.key,
                rank_payload_offset,
                checked_u32(success_cursor, "BC mutable streamed success row offset")
            });
            success_cursor += ref.live_count;
            if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed success rows exceed uint32");
            }
        }
        const StreamedFinalizePlan plan = measure_sorted_streamed_plan(sorted);
        if (rank_cursor != plan.rank_payload_bytes ||
            success_cursor != plan.success_rows) {
            throw std::logic_error("BC mutable streamed finalize emitted byte/count mismatch");
        }
    }

    template <class EmitBucket, class EmitRank>
    static void emit_dump_streamed_payload(
        const uint8_t *bitmap,
        const std::vector<DumpBucketRef> &sorted,
        FinalizeScratch &scratch,
        EmitBucket &&emit_bucket,
        EmitRank &&emit_rank
    ) {
        uint32_t rank_cursor = 0U;
        uint64_t success_cursor = 0U;
        for (const DumpBucketRef &ref : sorted) {
            rank_cursor = emit_rank_alignment(rank_cursor, emit_rank);
            const uint32_t rank_payload_offset = rank_cursor;
            emit_dump_prefix_chunk(bitmap, ref, scratch, emit_rank);
            rank_cursor = checked_u32_add(
                rank_cursor,
                static_cast<uint32_t>(prefix_count_for_bits(ref.bitmap_len) * sizeof(RankPrefix)),
                "BC mutable streamed dump prefix cursor overflow"
            );
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, ref.bitmap_len);
            if (bitmap_offset < rank_cursor) {
                throw std::logic_error("BC mutable streamed dump bitmap offset precedes prefix end");
            }
            emit_rank_padding(emit_rank, bitmap_offset - rank_cursor);
            rank_cursor = bitmap_offset;
            emit_dump_bitmap_chunk(bitmap, ref, scratch, emit_rank);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(ref.word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed dump bitmap byte count exceeds uint32");
            }
            rank_cursor = checked_u32_add(
                rank_cursor,
                static_cast<uint32_t>(bitmap_bytes64),
                "BC mutable streamed dump bitmap cursor overflow"
            );
            emit_bucket(BCBucketEntry{
                ref.key,
                rank_payload_offset,
                checked_u32(success_cursor, "BC mutable streamed dump success row offset")
            });
            success_cursor += ref.live_count;
            if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC mutable streamed dump success rows exceed uint32");
            }
        }
        const StreamedFinalizePlan plan = measure_dump_streamed_plan(sorted);
        if (rank_cursor != plan.rank_payload_bytes ||
            success_cursor != plan.success_rows) {
            throw std::logic_error("BC mutable streamed dump emitted byte/count mismatch");
        }
    }

    template <class EmitRank>
    static uint32_t emit_rank_alignment(uint32_t rank_cursor, EmitRank &&emit_rank) {
        const uint32_t aligned = align_up_u32(rank_cursor, 8U);
        emit_rank_padding(emit_rank, aligned - rank_cursor);
        return aligned;
    }

    static constexpr uint8_t kEmptyState = static_cast<uint8_t>(BCCellMutableSlotState::Empty);
    static constexpr uint8_t kPendingState = static_cast<uint8_t>(BCCellMutableSlotState::Pending);
    static constexpr uint8_t kOccupiedState = static_cast<uint8_t>(BCCellMutableSlotState::Occupied);
    static constexpr uint32_t kInitialCapacity = 8U;
    static constexpr uint32_t kInitialBitmapArenaWords = 4096U;
    static constexpr uint32_t kCreationLockCount = 64U;
    static constexpr uint32_t kLoadNumerator = 3U;
    static constexpr uint32_t kLoadDenominator = 4U;
    static constexpr uint32_t kDumpHeaderBytes = 64U;
    static constexpr uint32_t kDumpMagic = 0x4D434342U; // "BCCM" little-endian.
    static constexpr uint16_t kDumpVersion = 1U;
    static constexpr uint32_t kInvalidSlot = std::numeric_limits<uint32_t>::max();
    const BCLut *lut_ = nullptr;
    CellId cid_ = 0U;
    mutable SlotArrays slots_;
    mutable uint32_t capacity_ = 0U;
    mutable uint32_t capacity_mask_ = 0U;
    mutable uint32_t hash_shift_ = 64U;
    mutable std::unique_ptr<uint32_t[]> occupied_slots_;
    mutable uint32_t occupied_slots_capacity_ = 0U;
    mutable std::atomic<uint32_t> bucket_count_{0U};
    mutable std::atomic<bool> resizing_{false};
    mutable std::atomic<uint32_t> active_inserts_{0U};
    mutable std::mutex resize_mutex_;
    mutable std::array<std::mutex, kCreationLockCount> creation_locks_;
    mutable std::mutex bitmap_alloc_mutex_;
    mutable BCCellBitmapArenaPtr bitmap_arena_;
    mutable std::atomic<uint32_t> bitmap_used_words_{0U};
    mutable std::atomic<uint32_t> bitmap_capacity_words_{0U};
    mutable std::atomic<uint64_t> hash_grow_count_{0U};
    mutable std::atomic<uint64_t> bitmap_grow_count_{0U};
    mutable std::atomic<uint64_t> hash_replaced_bytes_total_{0U};
    mutable std::atomic<uint64_t> bitmap_replaced_bytes_total_{0U};
    mutable uint64_t hash_grow_baseline_ = 0U;
    mutable uint64_t bitmap_grow_baseline_ = 0U;
    mutable uint64_t hash_replaced_bytes_baseline_ = 0U;
    mutable uint64_t bitmap_replaced_bytes_baseline_ = 0U;
    mutable std::atomic<bool> dirty_{false};
    mutable std::atomic<bool> overflowed_{false};
    mutable uint64_t finalize_count_ = 0U;
};

} // namespace BC
