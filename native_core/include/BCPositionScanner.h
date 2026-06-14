#pragma once

#include "BCBoardCodec.h"
#include "BCPositionFile.h"

#include <algorithm>
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

[[nodiscard]] inline uint32_t bc_scanner_countr_zero64(uint64_t value) {
    if (value == 0U) {
        throw std::invalid_argument("BC scanner countr_zero64 requires non-zero value");
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

template <class Emit>
inline void bc_scan_bucket_board_se_blocks(
    const BCBucketEntry &bucket,
    const BCBucketBoardDecoder &decoder,
    const uint8_t *bitmap_words,
    uint32_t bitmap_word_count,
    uint32_t bitmap_len,
    uint32_t word_begin,
    uint32_t word_end,
    uint64_t success_rows,
    uint32_t &bucket_seen,
    Emit &&emit
) {
    if (bitmap_words == nullptr && bitmap_word_count != 0U) {
        throw std::invalid_argument("BC scanner bitmap words pointer is null");
    }
    if (word_begin > word_end || word_end > bitmap_word_count) {
        throw std::out_of_range("BC scanner bucket word range is invalid");
    }
    if (decoder.rank_decoder.count_se == 0U) {
        throw std::logic_error("BC scanner bucket has an empty SE word group");
    }
    if (bitmap_len == 0U || word_begin == word_end) {
        return;
    }

    const uint32_t count_se = decoder.rank_decoder.count_se;
    const uint32_t rank_begin = word_begin * kBCBitmapWordBits;
    const uint64_t raw_rank_end =
        static_cast<uint64_t>(word_end) * static_cast<uint64_t>(kBCBitmapWordBits);
    const uint32_t rank_end =
        static_cast<uint32_t>(std::min<uint64_t>(raw_rank_end, bitmap_len));
    if (rank_begin >= rank_end) {
        return;
    }

    const uint32_t tmp_begin = rank_begin / count_se;
    const uint32_t tmp_end = (rank_end + count_se - 1U) / count_se;
    for (uint32_t tmp = tmp_begin; tmp < tmp_end; ++tmp) {
        const uint32_t block_begin = tmp * count_se;
        const uint32_t block_end = std::min<uint32_t>(block_begin + count_se, bitmap_len);
        const uint32_t local_begin =
            rank_begin > block_begin ? rank_begin - block_begin : 0U;
        const uint32_t local_end =
            rank_end < block_end ? rank_end - block_begin : block_end - block_begin;
        if (local_begin >= local_end) {
            continue;
        }

        const uint64_t base_bits = decoder.base_bits_for_tmp(tmp);
        for (uint32_t rank_se = local_begin; rank_se < local_end; ) {
            const uint32_t rank = block_begin + rank_se;
            const uint32_t word_i = rank / kBCBitmapWordBits;
            const uint32_t bit_i = rank & (kBCBitmapWordBits - 1U);
            const uint32_t bits_in_word =
                std::min<uint32_t>(kBCBitmapWordBits - bit_i, local_end - rank_se);
            uint64_t word =
                load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            word >>= bit_i;
            if (bits_in_word < kBCBitmapWordBits) {
                word &= (1ULL << bits_in_word) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = bc_scanner_countr_zero64(word);
                const uint32_t rank_u32 = block_begin + rank_se + bit;
                if (rank_u32 >= bitmap_len ||
                    rank_u32 > std::numeric_limits<BucketRank>::max()) {
                    throw std::logic_error("BC scanner computed invalid rank");
                }
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (local_row >= success_rows ||
                    local_row > std::numeric_limits<uint32_t>::max()) {
                    throw std::out_of_range("BC scanner bucket success row exceeds descriptor");
                }
                emit(
                    static_cast<BucketRank>(rank_u32),
                    static_cast<uint32_t>(local_row),
                    decoder.board_from_base_and_se(base_bits, rank_se + bit)
                );
                ++bucket_seen;
                word &= word - 1U;
            }
            rank_se += bits_in_word;
        }
    }
}

template <class Emit>
inline void bc_scan_bucket_board_entries(
    const BCBucketEntry &bucket,
    const BCBucketBoardDecoder &decoder,
    const uint8_t *bitmap_words,
    uint32_t bitmap_word_count,
    uint32_t bitmap_len,
    uint32_t word_begin,
    uint32_t word_end,
    uint64_t success_rows,
    uint32_t &bucket_seen,
    Emit &&emit
) {
    bc_scan_bucket_board_se_blocks(
        bucket,
        decoder,
        bitmap_words,
        bitmap_word_count,
        bitmap_len,
        word_begin,
        word_end,
        success_rows,
        bucket_seen,
        std::forward<Emit>(emit)
    );
}

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

    template <class Fn>
    void for_each_bucket_word_range_board(
        uint32_t bucket_index,
        uint32_t word_begin,
        uint32_t word_end,
        Fn &&fn
    ) const {
        scan_bucket_word_range_board_impl(fn, bucket_index, word_begin, word_end);
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
            bc_scan_bucket_board_entries(
                bucket,
                decoder,
                bitmap_words,
                bitmap_word_count,
                bitmap_len,
                0U,
                bitmap_word_count,
                desc.success_rows,
                bucket_seen,
                [&](BucketRank rank, uint32_t local_row, uint64_t board) {
                    fn(BCScannedBoardEntry{bucket.key, rank, local_row, board});
                    ++total_seen;
                    if (total_seen > desc.success_rows) {
                        throw std::out_of_range("BC position scanner emitted too many rows");
                    }
                }
            );
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
    void scan_bucket_word_range_board_impl(
        Fn &fn,
        uint32_t bucket_index,
        uint32_t word_begin,
        uint32_t word_end
    ) const {
        const BCPositionCellDescriptor &desc = position_->descriptor(cid_);
        if (desc.empty()) {
            return;
        }
        if (bucket_index >= desc.bucket_count) {
            throw std::out_of_range("BC position scanner bucket word range bucket index out of range");
        }

        const BCLut &lut = position_->lut();
        const BCBucketEntryView buckets = position_->bucket_entries_for_cell(cid_);
        const BCRankPayloadView payload = position_->rank_payload_for_cell(cid_);
        if (bucket_index >= buckets.size) {
            throw std::out_of_range("BC position scanner bucket word range bucket view is short");
        }
        const BCBucketEntry &bucket = buckets.data[bucket_index];
        const BCBucketBoardDecoder decoder(lut, bucket.key);
        const uint32_t bitmap_len = decoder.bitmap_len();
        const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
        const uint32_t effective_word_end = word_end == 0U ? bitmap_word_count : word_end;
        if (word_begin > effective_word_end || effective_word_end > bitmap_word_count) {
            throw std::out_of_range("BC position scanner bucket word range is out of bounds");
        }
        if (word_begin == effective_word_end) {
            return;
        }

        const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t prefix_end =
            static_cast<uint64_t>(bucket.rank_payload_offset) +
            static_cast<uint64_t>(prefix_count) * sizeof(RankPrefix);
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
        if (prefix_end > payload.size || bitmap_end > payload.size) {
            throw std::out_of_range("BC position scanner bucket word range exceeds rank payload");
        }

        const uint8_t *prefix = payload.data + bucket.rank_payload_offset;
        const uint8_t *bitmap_words = payload.data + bitmap_offset;
        uint32_t bucket_seen = rank_before_word_index_le_bytes(
            prefix,
            prefix_count,
            bitmap_words,
            bitmap_word_count,
            bitmap_len,
            word_begin
        );
        bc_scan_bucket_board_entries(
            bucket,
            decoder,
            bitmap_words,
            bitmap_word_count,
            bitmap_len,
            word_begin,
            effective_word_end,
            desc.success_rows,
            bucket_seen,
            [&](BucketRank rank, uint32_t local_row, uint64_t board) {
                fn(BCScannedBoardEntry{bucket.key, rank, local_row, board});
            }
        );
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
