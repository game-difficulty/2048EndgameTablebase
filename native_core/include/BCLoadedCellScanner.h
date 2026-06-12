#pragma once

#include "BCPositionCellLoader.h"
#include "BCPositionScanner.h"

#include <algorithm>

namespace BC {

class BCLoadedCellScanner {
public:
    BCLoadedCellScanner(const BCLut &lut, BCLoadedCellView cell)
        : lut_(&lut), cell_(cell) {}

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
    void for_each_bucket_word_range(
        uint32_t bucket_index,
        uint32_t word_begin,
        uint32_t word_end,
        Fn &&fn
    ) const {
        scan_bucket_word_range_impl(
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
            },
            bucket_index,
            word_begin,
            word_end
        );
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
        std::vector<BCScannedPositionEntry> out;
        out.reserve(cell_.success_rows);
        for_each([&out](const BCScannedPositionEntry &entry) {
            out.push_back(entry);
        });
        return out;
    }

private:
    template <class Fn, class Emit>
    void scan_bucket_word_range_impl(
        Fn &fn,
        Emit &&emit,
        uint32_t bucket_index,
        uint32_t word_begin,
        uint32_t word_end
    ) const {
        require_valid_views();
        if (bucket_index >= cell_.buckets.size) {
            throw std::out_of_range("BC loaded cell scanner bucket word range bucket index out of range");
        }

        const BCBucketEntry &bucket = cell_.buckets.data[bucket_index];
        const BCBucketRankDecoder decoder(*lut_, bucket.key);
        const uint32_t bitmap_len = decoder.bitmap_len;
        const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
        const uint32_t effective_word_end = word_end == 0U ? bitmap_word_count : word_end;
        if (word_begin > effective_word_end || effective_word_end > bitmap_word_count) {
            throw std::out_of_range("BC loaded cell scanner bucket word range is out of bounds");
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
        if (prefix_end > cell_.rank_payload.size || bitmap_end > cell_.rank_payload.size) {
            throw std::out_of_range("BC loaded cell scanner bucket word range exceeds rank payload");
        }

        const uint8_t *prefix = cell_.rank_payload.data + bucket.rank_payload_offset;
        const uint8_t *bitmap_words = cell_.rank_payload.data + bitmap_offset;
        uint32_t bucket_seen = rank_before_word_index_le_bytes(
            prefix,
            prefix_count,
            bitmap_words,
            bitmap_word_count,
            bitmap_len,
            word_begin
        );

        for (uint32_t word_i = word_begin; word_i < effective_word_end; ++word_i) {
            uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = countr_zero64(word);
                const uint32_t rank_u32 = word_i * kBCBitmapWordBits + bit;
                if (rank_u32 >= bitmap_len ||
                    rank_u32 > std::numeric_limits<BucketRank>::max()) {
                    throw std::logic_error("BC loaded cell scanner range computed invalid rank");
                }
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (local_row >= cell_.success_rows ||
                    local_row > std::numeric_limits<uint32_t>::max()) {
                    throw std::out_of_range("BC loaded cell scanner range success row exceeds descriptor");
                }
                const BucketRank rank = static_cast<BucketRank>(rank_u32);
                const BCQuadrantWords words = decoder.unrank(*lut_, rank);
                emit(fn, bucket, rank, static_cast<uint32_t>(local_row), words);
                ++bucket_seen;
                word &= word - 1U;
            }
        }
    }

    template <class Fn>
    void scan_bucket_word_range_board_impl(
        Fn &fn,
        uint32_t bucket_index,
        uint32_t word_begin,
        uint32_t word_end
    ) const {
        require_valid_views();
        if (bucket_index >= cell_.buckets.size) {
            throw std::out_of_range("BC loaded cell scanner bucket word range bucket index out of range");
        }

        const BCBucketEntry &bucket = cell_.buckets.data[bucket_index];
        const BCBucketBoardDecoder decoder(*lut_, bucket.key);
        const uint32_t bitmap_len = decoder.bitmap_len();
        const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
        const uint32_t effective_word_end = word_end == 0U ? bitmap_word_count : word_end;
        if (word_begin > effective_word_end || effective_word_end > bitmap_word_count) {
            throw std::out_of_range("BC loaded cell scanner bucket word range is out of bounds");
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
        if (prefix_end > cell_.rank_payload.size || bitmap_end > cell_.rank_payload.size) {
            throw std::out_of_range("BC loaded cell scanner bucket word range exceeds rank payload");
        }

        const uint8_t *prefix = cell_.rank_payload.data + bucket.rank_payload_offset;
        const uint8_t *bitmap_words = cell_.rank_payload.data + bitmap_offset;
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
            cell_.success_rows,
            bucket_seen,
            [&](BucketRank rank, uint32_t local_row, uint64_t board) {
                fn(BCScannedBoardEntry{bucket.key, rank, local_row, board});
            }
        );
    }

    template <class Fn, class Emit>
    void scan_impl(Fn &fn, Emit &&emit) const {
        require_valid_views();
        if (cell_.buckets.size == 0U) {
            if (cell_.success_rows != 0U) {
                throw std::runtime_error("BC loaded cell scanner empty bucket view has non-zero rows");
            }
            return;
        }

        uint64_t total_seen = 0U;
        for (uint32_t bucket_i = 0; bucket_i < cell_.buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = cell_.buckets.data[bucket_i];
            const BCBucketRankDecoder decoder(*lut_, bucket.key);
            const uint32_t bitmap_len = decoder.bitmap_len;
            const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
                bucket.rank_payload_offset,
                bitmap_len
            );
            const uint64_t bitmap_end =
                static_cast<uint64_t>(bitmap_offset) +
                static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
            if (bitmap_end > cell_.rank_payload.size) {
                throw std::out_of_range("BC loaded cell scanner bucket bitmap exceeds rank payload");
            }

            uint32_t bucket_seen = 0U;
            const uint8_t *bitmap_words = cell_.rank_payload.data + bitmap_offset;
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
                        throw std::logic_error("BC loaded cell scanner computed invalid rank");
                    }
                    const uint64_t local_row =
                        static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                    if (local_row >= cell_.success_rows ||
                        local_row > std::numeric_limits<uint32_t>::max()) {
                        throw std::out_of_range("BC loaded cell scanner bucket success row exceeds descriptor");
                    }
                    const BucketRank rank = static_cast<BucketRank>(rank_u32);
                    const BCQuadrantWords words = decoder.unrank(*lut_, rank);
                    emit(fn, bucket, rank, static_cast<uint32_t>(local_row), words);
                    ++bucket_seen;
                    ++total_seen;
                    if (total_seen > cell_.success_rows) {
                        throw std::out_of_range("BC loaded cell scanner emitted too many rows");
                    }
                    word &= word - 1U;
                }
            }
            const uint64_t bucket_end_row =
                static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
            if (bucket_end_row > cell_.success_rows) {
                throw std::out_of_range("BC loaded cell scanner bucket row range exceeds descriptor");
            }
        }
        if (total_seen != cell_.success_rows) {
            throw std::runtime_error("BC loaded cell scanner success row count mismatch");
        }
    }

    template <class Fn>
    void scan_board_impl(Fn &fn) const {
        require_valid_views();
        if (cell_.buckets.size == 0U) {
            if (cell_.success_rows != 0U) {
                throw std::runtime_error("BC loaded cell scanner empty bucket view has non-zero rows");
            }
            return;
        }

        uint64_t total_seen = 0U;
        for (uint32_t bucket_i = 0; bucket_i < cell_.buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = cell_.buckets.data[bucket_i];
            const BCBucketBoardDecoder decoder(*lut_, bucket.key);
            const uint32_t bitmap_len = decoder.bitmap_len();
            const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
            const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
                bucket.rank_payload_offset,
                bitmap_len
            );
            const uint64_t bitmap_end =
                static_cast<uint64_t>(bitmap_offset) +
                static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
            if (bitmap_end > cell_.rank_payload.size) {
                throw std::out_of_range("BC loaded cell scanner bucket bitmap exceeds rank payload");
            }

            uint32_t bucket_seen = 0U;
            const uint8_t *bitmap_words = cell_.rank_payload.data + bitmap_offset;
            bc_scan_bucket_board_entries(
                bucket,
                decoder,
                bitmap_words,
                bitmap_word_count,
                bitmap_len,
                0U,
                bitmap_word_count,
                cell_.success_rows,
                bucket_seen,
                [&](BucketRank rank, uint32_t local_row, uint64_t board) {
                    fn(BCScannedBoardEntry{bucket.key, rank, local_row, board});
                    ++total_seen;
                    if (total_seen > cell_.success_rows) {
                        throw std::out_of_range("BC loaded cell scanner emitted too many rows");
                    }
                }
            );
            const uint64_t bucket_end_row =
                static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
            if (bucket_end_row > cell_.success_rows) {
                throw std::out_of_range("BC loaded cell scanner bucket row range exceeds descriptor");
            }
        }
        if (total_seen != cell_.success_rows) {
            throw std::runtime_error("BC loaded cell scanner success row count mismatch");
        }
    }

    void require_valid_views() const {
        if (lut_ == nullptr) {
            throw std::invalid_argument("BC loaded cell scanner LUT pointer is null");
        }
        if (cell_.buckets.size != 0U && cell_.buckets.data == nullptr) {
            throw std::invalid_argument("BC loaded cell scanner bucket view pointer is null");
        }
        if (cell_.rank_payload.size != 0U && cell_.rank_payload.data == nullptr) {
            throw std::invalid_argument("BC loaded cell scanner rank payload pointer is null");
        }
    }

    [[nodiscard]] static uint32_t countr_zero64(uint64_t value) {
        if (value == 0U) {
            throw std::invalid_argument("BC loaded cell countr_zero64 requires non-zero value");
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

    const BCLut *lut_ = nullptr;
    BCLoadedCellView cell_;
};

} // namespace BC
