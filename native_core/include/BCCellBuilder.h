#pragma once

#include "BCKeyRank.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {

struct BCBucketEntry {
    uint64_t key = 0U;
    uint32_t rank_payload_offset = 0U;
    uint32_t success_row_offset = 0U;
};

static_assert(sizeof(BCBucketEntry) == 16U, "BCBucketEntry must stay 16 bytes");

struct BCLookupResult {
    bool found = false;
    uint32_t local_success_row = 0U;
};

[[nodiscard]] inline uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC alignment must be a non-zero power of two");
    }
    const uint32_t mask = alignment - 1U;
    if (value > std::numeric_limits<uint32_t>::max() - mask) {
        throw std::overflow_error("BC align_up_u32 overflow");
    }
    return (value + mask) & ~mask;
}

[[nodiscard]] inline uint32_t checked_u32_size(size_t value, const char *label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(label);
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] inline uint32_t checked_u32_add(uint32_t lhs, uint32_t rhs, const char *label) {
    if (lhs > std::numeric_limits<uint32_t>::max() - rhs) {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

[[nodiscard]] inline uint32_t bc_rank_payload_bitmap_offset(uint32_t bucket_payload_offset, uint32_t bitmap_len) {
    const uint32_t prefix_bytes = prefix_count_for_bits(bitmap_len) * static_cast<uint32_t>(sizeof(RankPrefix));
    return align_up_u32(checked_u32_add(bucket_payload_offset, prefix_bytes, "BC rank payload prefix offset overflow"), 8U);
}

struct BCRankPayloadView {
    const uint8_t *data = nullptr;
    uint32_t size = 0U;
};

struct BCBucketEntryView {
    const BCBucketEntry *data = nullptr;
    uint32_t size = 0U;
};

[[nodiscard]] inline BCLookupResult lookup_finalized_cell(
    const BCLut &lut,
    BCBucketEntryView buckets,
    BCRankPayloadView payload,
    uint64_t key,
    BucketRank rank
) {
    if (buckets.size != 0U && buckets.data == nullptr) {
        throw std::invalid_argument("BC finalized cell bucket view pointer is null");
    }
    if (payload.size != 0U && payload.data == nullptr) {
        throw std::invalid_argument("BC finalized cell rank payload pointer is null");
    }
    if (buckets.size == 0U) {
        return {};
    }

    const BCBucketEntry *begin = buckets.data;
    const BCBucketEntry *end = buckets.data + buckets.size;
    const auto it = std::lower_bound(
        begin,
        end,
        key,
        [](const BCBucketEntry &entry, uint64_t target) {
            return entry.key < target;
        }
    );
    if (it == end || it->key != key) {
        return {};
    }

    const uint32_t bitmap_len = bitmap_len_from_key(lut, key);
    if (rank >= bitmap_len) {
        return {};
    }
    const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
    const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
    const uint32_t prefix_offset = it->rank_payload_offset;
    const uint32_t prefix_bytes = prefix_count * static_cast<uint32_t>(sizeof(RankPrefix));
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(prefix_offset, bitmap_len);
    const uint64_t prefix_end =
        static_cast<uint64_t>(prefix_offset) + static_cast<uint64_t>(prefix_bytes);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > payload.size || bitmap_end > payload.size) {
        throw std::out_of_range("BC finalized cell rank payload is truncated");
    }

    const BCBitmapRankResult rank_result = bitmap_test_and_rank_le_bytes(
        payload.data + prefix_offset,
        prefix_count,
        payload.data + bitmap_offset,
        bitmap_word_count,
        rank
    );
    if (!rank_result.found) {
        return {};
    }
    if (it->success_row_offset > std::numeric_limits<uint32_t>::max() - rank_result.rank_before) {
        throw std::overflow_error("BC finalized cell local_success_row overflow");
    }
    return BCLookupResult{
        true,
        it->success_row_offset + static_cast<uint32_t>(rank_result.rank_before)
    };
}

[[nodiscard]] inline BCLookupResult lookup_finalized_cell(
    const BCLut &lut,
    const std::vector<BCBucketEntry> &buckets,
    BCRankPayloadView payload,
    uint64_t key,
    BucketRank rank
) {
    return lookup_finalized_cell(
        lut,
        BCBucketEntryView{
            buckets.data(),
            checked_u32_size(buckets.size(), "BC finalized cell bucket count exceeds uint32")
        },
        payload,
        key,
        rank
    );
}

struct FinalizedCellPayload {
    std::vector<BCBucketEntry> buckets;
    std::vector<uint8_t> rank_payload;
    uint32_t success_rows = 0U;

    [[nodiscard]] BCLookupResult lookup(
        const BCLut &lut,
        uint64_t key,
        BucketRank rank
    ) const {
        return lookup_finalized_cell(
            lut,
            buckets,
            BCRankPayloadView{
                rank_payload.data(),
                checked_u32_size(rank_payload.size(), "BC finalized cell rank payload exceeds uint32")
            },
            key,
            rank
        );
    }
};

using BCKeyValueSortUint64Uint32Fn = void (*)(uint64_t *keys, uint32_t *values, size_t count, bool descending);

[[nodiscard]] inline bool bc_finalize_keyvalue_sort_allowed() {
#if defined(_OPENMP)
    return omp_in_parallel() == 0;
#else
    return true;
#endif
}

struct BCCellFinalizeOptions {
    BCKeyValueSortUint64Uint32Fn keyvalue_sort = nullptr;
    size_t simd_sort_min_bucket_count = 2U;
};

struct BCInsertResult {
    bool new_bucket = false;
    bool new_rank = false;
};

struct BCMergeResult {
    uint64_t new_buckets = 0U;
    uint64_t new_ranks = 0U;
    uint64_t duplicate_ranks = 0U;
};

class BCCellBuilder {
    struct BucketBuildState {
        uint64_t key = 0U;
        BucketBitmapLen bitmap_len = 0U;
        uint32_t word_count = 0U;
        uint32_t live_count = 0U;
        uint32_t bitmap_offset = 0U;
        bool occupied = false;
    };

    struct SortedBucketRef {
        uint64_t key = 0U;
        uint32_t slot = 0U;
    };

public:
    explicit BCCellBuilder(const BCLut &lut) : lut_(&lut) {}

    void insert(uint64_t key, BucketRank rank) {
        (void)insert_and_report(key, rank);
    }

    [[nodiscard]] BCInsertResult insert_and_report(uint64_t key, BucketRank rank) {
        return insert_impl(key, rank, bitmap_len_from_key(*lut_, key));
    }

    [[nodiscard]] BCInsertResult insert_encoded_and_report(const BCEncodedKeyRank &encoded) {
        if (!encoded.valid) {
            throw std::invalid_argument("BC cell builder cannot insert invalid encoded key/rank");
        }
        return insert_impl(encoded.key, encoded.rank, encoded.bitmap_len);
    }

    [[nodiscard]] bool contains(uint64_t key, BucketRank rank) const {
        const uint32_t bitmap_len = bitmap_len_from_key(*lut_, key);
        if (rank >= bitmap_len) {
            return false;
        }
        const size_t slot = find_existing_slot(key);
        if (slot == kNoSlot) {
            return false;
        }
        const BucketBuildState &bucket = table_[slot];
        return bitmap_test(bitmap_words(bucket), bucket.word_count, rank);
    }

    [[nodiscard]] size_t bucket_count() const {
        return bucket_count_;
    }

    [[nodiscard]] uint64_t live_rows() const {
        return live_rows_;
    }

    [[nodiscard]] size_t live_bitmap_words() const {
        return live_bitmap_word_count();
    }

    void reserve_buckets(size_t count) {
        reserve_buckets_and_bitmap_words(count, count * kDefaultReserveWordsPerBucket);
    }

    void reserve_buckets_and_bitmap_words(size_t bucket_count, size_t bitmap_words) {
        ensure_table_capacity(bucket_count);
        if (bitmap_words > bitmap_arena_.capacity()) {
            bitmap_arena_.reserve(bitmap_words);
        }
    }

    [[nodiscard]] BCMergeResult merge_from(const BCCellBuilder &other) {
        BCMergeResult result;
        if (other.bucket_count_ == 0U) {
            return result;
        }
        ensure_table_capacity(checked_size_add(
            bucket_count_,
            other.bucket_count_,
            "BC cell builder merge bucket reserve overflow"
        ));
        const size_t target_bitmap_words = checked_size_add(
            bitmap_arena_.size(),
            other.live_bitmap_word_count(),
            "BC cell builder merge bitmap reserve overflow"
        );
        if (target_bitmap_words > bitmap_arena_.capacity()) {
            bitmap_arena_.reserve(target_bitmap_words);
        }
        for (const BucketBuildState &source : other.table_) {
            if (!source.occupied) {
                continue;
            }
            if (source.live_count == 0U) {
                continue;
            }
            bool inserted = false;
            const uint64_t *source_bitmap = other.bitmap_words(source);
            const size_t slot = find_or_create_bucket_from_bitmap(
                source.key,
                source.bitmap_len,
                source.word_count,
                source.live_count,
                source_bitmap,
                inserted
            );
            BucketBuildState &target = table_[slot];
            uint64_t *target_bitmap = bitmap_words(target);
            if (inserted) {
                live_rows_ += source.live_count;
                ++result.new_buckets;
                result.new_ranks += source.live_count;
                continue;
            }
            if (target.bitmap_len != source.bitmap_len ||
                target.word_count != source.word_count) {
                throw std::logic_error("BC cell builder merge saw incompatible bucket state");
            }
            uint32_t new_bits = 0U;
            for (uint32_t word = 0; word < source.word_count; ++word) {
                const uint64_t source_word = source_bitmap[word];
                const uint64_t added = source_word & ~target_bitmap[word];
                if (added != 0U) {
                    new_bits += popcount64(added);
                    target_bitmap[word] |= source_word;
                }
            }
            if (new_bits > source.live_count) {
                throw std::logic_error("BC cell builder merge counted too many new bits");
            }
            target.live_count += new_bits;
            live_rows_ += new_bits;
            result.new_ranks += new_bits;
            result.duplicate_ranks += static_cast<uint64_t>(source.live_count) - new_bits;
        }
        return result;
    }

    [[nodiscard]] FinalizedCellPayload finalize(BCCellFinalizeOptions options = {}) const {
        const std::vector<SortedBucketRef> sorted = sorted_bucket_refs(options);
        FinalizedCellPayload out;
        out.buckets.reserve(sorted.size());
        out.rank_payload.reserve(estimate_rank_payload_bytes(sorted));
        uint64_t success_cursor = 0U;
        for (const SortedBucketRef &sorted_bucket : sorted) {
            if (sorted_bucket.slot >= table_.size()) {
                throw std::logic_error("BC cell builder finalize saw invalid bucket slot");
            }
            const BucketBuildState &bucket = table_[sorted_bucket.slot];
            if (!bucket.occupied) {
                throw std::logic_error("BC cell builder finalize saw unoccupied bucket slot");
            }
            const uint32_t bitmap_len = bucket.bitmap_len;
            const uint32_t word_count = bucket.word_count;
            const uint32_t bucket_live = bucket.live_count;
            if (bucket_live == 0U) {
                continue;
            }
            const uint64_t *bitmap = bitmap_words(bucket);

            const uint32_t payload_offset = checked_u32(out.rank_payload.size(), "BC rank payload offset");
            const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
            append_padding(out.rank_payload, aligned_payload_offset - payload_offset);
            const uint32_t rank_payload_offset =
                checked_u32(out.rank_payload.size(), "BC aligned rank payload offset");

            append_prefix256_le(out.rank_payload, bitmap, word_count, bitmap_len);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
            append_padding(out.rank_payload, bitmap_offset - checked_u32(out.rank_payload.size(), "BC prefix payload end"));
            append_bitmap_le(out.rank_payload, bitmap, word_count);

            out.buckets.push_back(BCBucketEntry{
                sorted_bucket.key,
                rank_payload_offset,
                checked_u32(success_cursor, "BC success row offset")
            });
            success_cursor += bucket_live;
            if (success_cursor > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC finalized cell success_rows exceeds uint32");
            }
        }
        out.success_rows = static_cast<uint32_t>(success_cursor);
        return out;
    }

private:
    [[nodiscard]] BCInsertResult insert_impl(
        uint64_t key,
        BucketRank rank,
        BucketBitmapLen bitmap_len
    ) {
        if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen) {
            throw std::logic_error("BC cell builder insert saw invalid bitmap_len");
        }
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC cell builder insert rank exceeds bitmap_len");
        }

        const uint32_t word_count = words_for_bits(bitmap_len);
        bool inserted = false;
        const size_t slot = find_or_create_bucket(key, bitmap_len, word_count, inserted);
        BucketBuildState &bucket = table_[slot];
        if (bucket.bitmap_len != bitmap_len ||
            bucket.word_count != word_count) {
            throw std::logic_error("BC cell builder saw inconsistent bitmap_len for key");
        }
        uint64_t *bitmap = bitmap_words(bucket);

        const uint32_t rank_u32 = rank;
        const uint32_t word = rank_u32 / kBCBitmapWordBits;
        const uint32_t bit = rank_u32 & (kBCBitmapWordBits - 1U);
        const uint64_t mask = 1ULL << bit;
        const bool was_set = (bitmap[word] & mask) != 0U;
        if (!was_set) {
            bitmap[word] |= mask;
            ++bucket.live_count;
            ++live_rows_;
        }
        return BCInsertResult{inserted, !was_set};
    }

    [[nodiscard]] static uint32_t checked_u32(uint64_t value, const char *label) {
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(label);
        }
        return static_cast<uint32_t>(value);
    }

    static void append_padding(std::vector<uint8_t> &buffer, uint32_t bytes) {
        buffer.insert(buffer.end(), bytes, 0U);
    }

    static void append_u16_le(std::vector<uint8_t> &buffer, uint16_t value) {
        buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
        buffer.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    }

    static void append_u64_le(std::vector<uint8_t> &buffer, uint64_t value) {
        for (uint32_t i = 0; i < 8U; ++i) {
            buffer.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
        }
    }

    static void append_prefix256_le(
        std::vector<uint8_t> &buffer,
        const uint64_t *bitmap,
        uint32_t word_count,
        uint32_t bitmap_len
    ) {
        if (bitmap == nullptr && word_count != 0U) {
            throw std::invalid_argument("BC append_prefix256_le bitmap pointer is null");
        }
        uint32_t running = 0U;
        const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
        for (uint32_t block = 0; block < prefix_count; ++block) {
            if (running > std::numeric_limits<RankPrefix>::max()) {
                throw std::logic_error("BC prefix running popcount exceeds uint16");
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
                throw std::invalid_argument("BC append_prefix256_le bitmap is shorter than bitmap_len");
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

    static void append_bitmap_le(std::vector<uint8_t> &buffer, const uint64_t *bitmap, uint32_t word_count) {
        if (bitmap == nullptr && word_count != 0U) {
            throw std::invalid_argument("BC append_bitmap_le bitmap pointer is null");
        }
        for (uint32_t i = 0; i < word_count; ++i) {
            append_u64_le(buffer, bitmap[i]);
        }
    }

    [[nodiscard]] std::vector<SortedBucketRef> sorted_bucket_refs(BCCellFinalizeOptions options) const {
        if (table_.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC cell builder bucket count exceeds uint32 sort index");
        }
        std::vector<uint64_t> keys;
        std::vector<uint32_t> slots;
        keys.reserve(bucket_count_);
        slots.reserve(bucket_count_);
        for (uint32_t slot = 0U; slot < table_.size(); ++slot) {
            const BucketBuildState &bucket = table_[slot];
            if (!bucket.occupied) {
                continue;
            }
            keys.push_back(bucket.key);
            slots.push_back(slot);
        }

        const bool use_keyvalue_sort =
            bc_finalize_keyvalue_sort_allowed() &&
            options.keyvalue_sort != nullptr &&
            keys.size() >= options.simd_sort_min_bucket_count;
        if (use_keyvalue_sort) {
            options.keyvalue_sort(keys.data(), slots.data(), keys.size(), false);
        } else {
            std::vector<uint32_t> order(slots.size());
            for (uint32_t i = 0U; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(
                order.begin(),
                order.end(),
                [&keys](uint32_t lhs, uint32_t rhs) {
                    return keys[lhs] < keys[rhs];
                }
            );
            std::vector<uint64_t> sorted_keys;
            std::vector<uint32_t> sorted_slots;
            sorted_keys.reserve(order.size());
            sorted_slots.reserve(order.size());
            for (uint32_t index : order) {
                sorted_keys.push_back(keys[index]);
                sorted_slots.push_back(slots[index]);
            }
            keys.swap(sorted_keys);
            slots.swap(sorted_slots);
        }

        std::vector<SortedBucketRef> sorted;
        sorted.reserve(keys.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            const uint32_t slot = slots[i];
            if (slot >= table_.size() || !table_[slot].occupied) {
                throw std::logic_error("BC keyvalue sorter returned out-of-range bucket index");
            }
            sorted.push_back(SortedBucketRef{
                keys[i],
                slot
            });
        }
        return sorted;
    }

    [[nodiscard]] uint32_t estimate_rank_payload_bytes(
        const std::vector<SortedBucketRef> &sorted
    ) const {
        uint32_t cursor = 0U;
        for (const SortedBucketRef &bucket : sorted) {
            if (bucket.slot >= table_.size() || !table_[bucket.slot].occupied) {
                throw std::logic_error("BC cell builder estimate saw invalid bucket slot");
            }
            const BucketBuildState &state = table_[bucket.slot];
            if (state.live_count == 0U) {
                continue;
            }
            const uint32_t bitmap_len = state.bitmap_len;
            const uint32_t word_count = state.word_count;
            cursor = align_up_u32(cursor, 8U);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(cursor, bitmap_len);
            const uint64_t bitmap_bytes64 = static_cast<uint64_t>(word_count) * sizeof(uint64_t);
            if (bitmap_bytes64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC rank payload bitmap bytes overflow");
            }
            const uint32_t bitmap_bytes = static_cast<uint32_t>(bitmap_bytes64);
            cursor = checked_u32_add(bitmap_offset, bitmap_bytes, "BC rank payload reserve overflow");
        }
        return cursor;
    }

    [[nodiscard]] static uint64_t mix_u64(uint64_t value) {
        value ^= value >> 30U;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27U;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31U;
        return value;
    }

    [[nodiscard]] static size_t next_power_of_two(size_t value) {
        size_t out = 1U;
        while (out < value) {
            if (out > (std::numeric_limits<size_t>::max() >> 1U)) {
                throw std::overflow_error("BC cell builder hash table capacity overflow");
            }
            out <<= 1U;
        }
        return out;
    }

    [[nodiscard]] static size_t checked_size_add(size_t lhs, size_t rhs, const char *label) {
        if (lhs > std::numeric_limits<size_t>::max() - rhs) {
            throw std::overflow_error(label);
        }
        return lhs + rhs;
    }

    [[nodiscard]] size_t live_bitmap_word_count() const {
        size_t words = 0U;
        for (const BucketBuildState &bucket : table_) {
            if (!bucket.occupied || bucket.live_count == 0U) {
                continue;
            }
            if (words > std::numeric_limits<size_t>::max() - bucket.word_count) {
                throw std::overflow_error("BC cell builder bitmap word reserve overflow");
            }
            words += bucket.word_count;
        }
        return words;
    }

    void invalidate_cache() {
        cached_valid_ = false;
    }

    void ensure_table_capacity(size_t desired_buckets) {
        constexpr size_t kMinCapacity = 8U;
        if (desired_buckets == 0U) {
            return;
        }
        if (!table_.empty() && desired_buckets * 10U <= table_.size() * 7U) {
            return;
        }
        const size_t target = next_power_of_two(std::max(kMinCapacity, desired_buckets * 2U));
        rehash(target);
    }

    void rehash(size_t new_capacity) {
        if ((new_capacity & (new_capacity - 1U)) != 0U) {
            throw std::invalid_argument("BC cell builder hash capacity must be power of two");
        }
        std::vector<BucketBuildState> old = std::move(table_);
        table_.assign(new_capacity, BucketBuildState{});
        bucket_count_ = 0U;
        invalidate_cache();
        for (const BucketBuildState &bucket : old) {
            if (!bucket.occupied) {
                continue;
            }
            const size_t slot = find_empty_slot_for_rehash(bucket.key);
            table_[slot] = bucket;
            ++bucket_count_;
        }
    }

    [[nodiscard]] size_t find_empty_slot_for_rehash(uint64_t key) const {
        const size_t mask = table_.size() - 1U;
        size_t slot = static_cast<size_t>(mix_u64(key)) & mask;
        while (table_[slot].occupied) {
            slot = (slot + 1U) & mask;
        }
        return slot;
    }

    [[nodiscard]] size_t find_existing_slot(uint64_t key) const {
        if (table_.empty()) {
            return kNoSlot;
        }
        const size_t mask = table_.size() - 1U;
        size_t slot = static_cast<size_t>(mix_u64(key)) & mask;
        while (table_[slot].occupied) {
            if (table_[slot].key == key) {
                return slot;
            }
            slot = (slot + 1U) & mask;
        }
        return kNoSlot;
    }

    [[nodiscard]] size_t find_or_create_bucket(
        uint64_t key,
        BucketBitmapLen bitmap_len,
        uint32_t word_count,
        bool &inserted
    ) {
        if (cached_valid_ &&
            cached_key_ == key &&
            cached_slot_ < table_.size() &&
            table_[cached_slot_].occupied &&
            table_[cached_slot_].key == key) {
            inserted = false;
            return cached_slot_;
        }

        ensure_table_capacity(bucket_count_ + 1U);
        const size_t mask = table_.size() - 1U;
        size_t slot = static_cast<size_t>(mix_u64(key)) & mask;
        while (table_[slot].occupied) {
            if (table_[slot].key == key) {
                inserted = false;
                cached_valid_ = true;
                cached_key_ = key;
                cached_slot_ = slot;
                return slot;
            }
            slot = (slot + 1U) & mask;
        }

        const uint32_t bitmap_offset =
            checked_u32(bitmap_arena_.size(), "BC cell builder bitmap arena exceeds uint32 words");
        bitmap_arena_.resize(bitmap_arena_.size() + word_count, 0U);
        table_[slot] = BucketBuildState{
            key,
            bitmap_len,
            word_count,
            0U,
            bitmap_offset,
            true
        };
        ++bucket_count_;
        inserted = true;
        cached_valid_ = true;
        cached_key_ = key;
        cached_slot_ = slot;
        return slot;
    }

    [[nodiscard]] size_t find_or_create_bucket_from_bitmap(
        uint64_t key,
        BucketBitmapLen bitmap_len,
        uint32_t word_count,
        uint32_t live_count,
        const uint64_t *source_bitmap,
        bool &inserted
    ) {
        if (source_bitmap == nullptr && word_count != 0U) {
            throw std::invalid_argument("BC cell builder merge source bitmap pointer is null");
        }
        if (cached_valid_ &&
            cached_key_ == key &&
            cached_slot_ < table_.size() &&
            table_[cached_slot_].occupied &&
            table_[cached_slot_].key == key) {
            inserted = false;
            return cached_slot_;
        }

        ensure_table_capacity(bucket_count_ + 1U);
        const size_t mask = table_.size() - 1U;
        size_t slot = static_cast<size_t>(mix_u64(key)) & mask;
        while (table_[slot].occupied) {
            if (table_[slot].key == key) {
                inserted = false;
                cached_valid_ = true;
                cached_key_ = key;
                cached_slot_ = slot;
                return slot;
            }
            slot = (slot + 1U) & mask;
        }

        const uint32_t bitmap_offset =
            checked_u32(bitmap_arena_.size(), "BC cell builder bitmap arena exceeds uint32 words");
        bitmap_arena_.insert(bitmap_arena_.end(), source_bitmap, source_bitmap + word_count);
        table_[slot] = BucketBuildState{
            key,
            bitmap_len,
            word_count,
            live_count,
            bitmap_offset,
            true
        };
        ++bucket_count_;
        inserted = true;
        cached_valid_ = true;
        cached_key_ = key;
        cached_slot_ = slot;
        return slot;
    }

    [[nodiscard]] uint64_t *bitmap_words(BucketBuildState &bucket) {
        const uint64_t end =
            static_cast<uint64_t>(bucket.bitmap_offset) + static_cast<uint64_t>(bucket.word_count);
        if (end > bitmap_arena_.size()) {
            throw std::logic_error("BC cell builder bitmap arena offset is out of range");
        }
        return bitmap_arena_.data() + bucket.bitmap_offset;
    }

    [[nodiscard]] const uint64_t *bitmap_words(const BucketBuildState &bucket) const {
        const uint64_t end =
            static_cast<uint64_t>(bucket.bitmap_offset) + static_cast<uint64_t>(bucket.word_count);
        if (end > bitmap_arena_.size()) {
            throw std::logic_error("BC cell builder bitmap arena offset is out of range");
        }
        return bitmap_arena_.data() + bucket.bitmap_offset;
    }

    static constexpr size_t kNoSlot = std::numeric_limits<size_t>::max();
    static constexpr size_t kDefaultReserveWordsPerBucket = 4U;

    const BCLut *lut_ = nullptr;
    std::vector<BucketBuildState> table_;
    size_t bucket_count_ = 0U;
    std::vector<uint64_t> bitmap_arena_;
    uint64_t live_rows_ = 0U;
    uint64_t cached_key_ = 0U;
    size_t cached_slot_ = 0U;
    bool cached_valid_ = false;
};

} // namespace BC
