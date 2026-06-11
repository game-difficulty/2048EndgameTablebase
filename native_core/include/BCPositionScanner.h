#pragma once

#include "BCBoardCodec.h"
#include "BCPositionFile.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace BC {

struct BCScannedPositionEntry {
    uint64_t key = 0U;
    BucketRank rank = 0U;
    uint32_t local_success_row = 0U;
    uint16_t nw = 0U;
    uint16_t ne = 0U;
    uint16_t sw = 0U;
    uint16_t se = 0U;
};

struct BCScannedBoardEntry {
    uint64_t key = 0U;
    BucketRank rank = 0U;
    uint32_t local_success_row = 0U;
    uint64_t board = 0U;
};

class BCPositionCellScanner {
public:
    BCPositionCellScanner(const BCPositionLayerReader &position, CellId cid)
        : position_(&position), cid_(cid) {}

    template <class Fn>
    void for_each(Fn &&fn) const {
        scan_impl(
            fn,
            [](Fn &callback, const BCBucketEntry &bucket, BucketRank rank, uint32_t local_row,
               const BCQuadrantWords &words) {
                callback(BCScannedPositionEntry{
                    bucket.key,
                    rank,
                    local_row,
                    words.nw,
                    words.ne,
                    words.sw,
                    words.se
                });
            }
        );
    }

    template <class Fn>
    void for_each_board(Fn &&fn) const {
        scan_board_impl(fn);
    }

    [[nodiscard]] std::vector<BCScannedPositionEntry> scan() const {
        const BCPositionCellDescriptor &desc = position_->descriptor(cid_);
        std::vector<BCScannedPositionEntry> out;
        out.reserve(desc.success_rows);
        for_each([&out](const BCScannedPositionEntry &entry) {
            out.push_back(entry);
        });
        return out;
    }

private:
    template <class Fn, class Emit>
    void scan_impl(Fn &fn, Emit &&emit) const {
        const BCPositionCellDescriptor &desc = position_->descriptor(cid_);
        if (desc.empty()) {
            return;
        }

        const BCLut &lut = position_->lut();
        const BCBucketEntryView buckets = position_->bucket_entries_for_cell(cid_);
        const BCRankPayloadView payload = position_->rank_payload_for_cell(cid_);
        uint64_t total_seen = 0U;
        for (uint32_t bucket_i = 0; bucket_i < buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = buckets.data[bucket_i];
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
                throw std::out_of_range("BC position scanner bucket bitmap exceeds rank payload");
            }

            uint32_t bucket_seen = 0U;
            const uint8_t *bitmap_words = payload.data + bitmap_offset;
            for (uint32_t word_i = 0; word_i < bitmap_word_count; ++word_i) {
                uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
                if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
                    word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
                }
                while (word != 0U) {
                    const uint32_t bit = countr_zero64(word);
                    const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
                    if (rank_u32 >= bitmap_len ||
                        rank_u32 > std::numeric_limits<BucketRank>::max()) {
                        throw std::logic_error("BC position scanner computed invalid rank");
                    }
                    const uint64_t local_row =
                        static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                    if (local_row >= desc.success_rows ||
                        local_row > std::numeric_limits<uint32_t>::max()) {
                        throw std::out_of_range("BC position scanner bucket success row exceeds descriptor");
                    }
                    const BucketRank rank = static_cast<BucketRank>(rank_u32);
                    const BCQuadrantWords words = decoder.unrank(lut, rank);
                    emit(fn, bucket, rank, static_cast<uint32_t>(local_row), words);
                    ++bucket_seen;
                    ++total_seen;
                    if (total_seen > desc.success_rows) {
                        throw std::out_of_range("BC position scanner emitted too many rows");
                    }
                    word &= word - 1U;
                }
            }
            const uint64_t bucket_end_row =
                static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
            if (bucket_end_row > desc.success_rows) {
                throw std::out_of_range("BC position scanner bucket row range exceeds descriptor");
            }
        }
        if (total_seen != desc.success_rows) {
            throw std::runtime_error("BC position scanner success row count mismatch");
        }
    }

    template <class Fn>
    void scan_board_impl(Fn &fn) const {
        const BCPositionCellDescriptor &desc = position_->descriptor(cid_);
        if (desc.empty()) {
            return;
        }

        const BCLut &lut = position_->lut();
        const BCBucketEntryView buckets = position_->bucket_entries_for_cell(cid_);
        const BCRankPayloadView payload = position_->rank_payload_for_cell(cid_);
        uint64_t total_seen = 0U;
        for (uint32_t bucket_i = 0; bucket_i < buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = buckets.data[bucket_i];
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
                throw std::out_of_range("BC position scanner bucket bitmap exceeds rank payload");
            }

            uint32_t bucket_seen = 0U;
            const uint8_t *bitmap_words = payload.data + bitmap_offset;
            for (uint32_t word_i = 0; word_i < bitmap_word_count; ++word_i) {
                uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
                if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
                    word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
                }
                while (word != 0U) {
                    const uint32_t bit = countr_zero64(word);
                    const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
                    if (rank_u32 >= bitmap_len ||
                        rank_u32 > std::numeric_limits<BucketRank>::max()) {
                        throw std::logic_error("BC position scanner computed invalid rank");
                    }
                    const uint64_t local_row =
                        static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                    if (local_row >= desc.success_rows ||
                        local_row > std::numeric_limits<uint32_t>::max()) {
                        throw std::out_of_range("BC position scanner bucket success row exceeds descriptor");
                    }
                    const BucketRank rank = static_cast<BucketRank>(rank_u32);
                    fn(BCScannedBoardEntry{
                        bucket.key,
                        rank,
                        static_cast<uint32_t>(local_row),
                        decoder.board(rank)
                    });
                    ++bucket_seen;
                    ++total_seen;
                    if (total_seen > desc.success_rows) {
                        throw std::out_of_range("BC position scanner emitted too many rows");
                    }
                    word &= word - 1U;
                }
            }
            const uint64_t bucket_end_row =
                static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
            if (bucket_end_row > desc.success_rows) {
                throw std::out_of_range("BC position scanner bucket row range exceeds descriptor");
            }
        }
        if (total_seen != desc.success_rows) {
            throw std::runtime_error("BC position scanner success row count mismatch");
        }
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

    const BCPositionLayerReader *position_ = nullptr;
    CellId cid_ = 0U;
};

} // namespace BC
