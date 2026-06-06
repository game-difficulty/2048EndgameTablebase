#pragma once

#include "BCLut.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace BC {

struct BCEncodedKeyRank {
    uint64_t key = 0U;
    BucketRank rank = 0U;
    BucketBitmapLen bitmap_len = 0U;
    uint16_t count_ne = 0U;
    uint16_t count_sw = 0U;
    uint16_t count_se = 0U;
    bool valid = false;
};

struct BCBitmapRankResult {
    bool found = false;
    RankPrefix rank_before = 0U;
};

struct BCQuadrantWords {
    uint16_t nw = 0U;
    uint16_t ne = 0U;
    uint16_t sw = 0U;
    uint16_t se = 0U;
};

[[nodiscard]] inline uint32_t words_for_bits(uint32_t bits) {
    return (bits + kBCBitmapWordBits - 1U) / kBCBitmapWordBits;
}

[[nodiscard]] inline uint32_t prefix_count_for_bits(uint32_t bits) {
    return (bits + kBCRankPrefixBits - 1U) / kBCRankPrefixBits;
}

[[nodiscard]] inline uint32_t popcount64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
#endif
}

[[nodiscard]] inline uint16_t load_u16_le(const uint8_t *data) {
    if (data == nullptr) {
        throw std::invalid_argument("BC load_u16_le pointer is null");
    }
    return static_cast<uint16_t>(
        static_cast<uint16_t>(data[0]) |
        (static_cast<uint16_t>(data[1]) << 8U)
    );
}

[[nodiscard]] inline uint64_t load_u64_le(const uint8_t *data) {
    if (data == nullptr) {
        throw std::invalid_argument("BC load_u64_le pointer is null");
    }
    return
        static_cast<uint64_t>(data[0]) |
        (static_cast<uint64_t>(data[1]) << 8U) |
        (static_cast<uint64_t>(data[2]) << 16U) |
        (static_cast<uint64_t>(data[3]) << 24U) |
        (static_cast<uint64_t>(data[4]) << 32U) |
        (static_cast<uint64_t>(data[5]) << 40U) |
        (static_cast<uint64_t>(data[6]) << 48U) |
        (static_cast<uint64_t>(data[7]) << 56U);
}

[[nodiscard]] inline BCEncodedKeyRank encode_key_and_rank(
    const BCLut &lut,
    uint16_t nw,
    uint16_t ne,
    uint16_t sw,
    uint16_t se
) {
    const BCWordDesc &nw_desc = lut.word_desc(nw);
    const BCWordDesc &ne_desc = lut.word_desc(ne);
    const BCWordDesc &sw_desc = lut.word_desc(sw);
    const BCWordDesc &se_desc = lut.word_desc(se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return {};
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
        throw std::logic_error("BC encoded bitmap length is outside uint16 bounds");
    }

    const uint32_t rank =
        (static_cast<uint32_t>(ne_desc.rank) * count_sw + sw_desc.rank) * count_se + se_desc.rank;
    if (rank >= bitmap_len || rank > std::numeric_limits<BucketRank>::max()) {
        throw std::logic_error("BC encoded rank is outside bitmap length");
    }

    const uint64_t key =
        (static_cast<uint64_t>(nw) << 48U) |
        (static_cast<uint64_t>(ne_desc.packed_sum_mask) << 32U) |
        (static_cast<uint64_t>(sw_desc.packed_sum_mask) << 16U) |
        static_cast<uint64_t>(se_desc.packed_sum_mask);

    return BCEncodedKeyRank{
        key,
        static_cast<BucketRank>(rank),
        static_cast<BucketBitmapLen>(bitmap_len),
        count_ne,
        count_sw,
        count_se,
        true
    };
}

[[nodiscard]] inline BucketBitmapLen bitmap_len_from_key(const BCLut &lut, uint64_t key) {
    const uint16_t ne = static_cast<uint16_t>((key >> 32U) & 0xFFFFU);
    const uint16_t sw = static_cast<uint16_t>((key >> 16U) & 0xFFFFU);
    const uint16_t se = static_cast<uint16_t>(key & 0xFFFFU);
    const uint16_t count_ne = lut.count4(packed_sum_id(ne), packed_empty_mask(ne));
    const uint16_t count_sw = lut.count4(packed_sum_id(sw), packed_empty_mask(sw));
    const uint16_t count_se = lut.count4(packed_sum_id(se), packed_empty_mask(se));
    const uint32_t bitmap_len =
        static_cast<uint32_t>(count_ne) *
        static_cast<uint32_t>(count_sw) *
        static_cast<uint32_t>(count_se);
    if (bitmap_len == 0U || bitmap_len > kBCMaxBucketBitmapLen ||
        bitmap_len > std::numeric_limits<BucketBitmapLen>::max()) {
        throw std::logic_error("BC bitmap_len_from_key computed invalid bitmap length");
    }
    return static_cast<BucketBitmapLen>(bitmap_len);
}

struct BCBucketRankDecoder {
    uint64_t key = 0U;
    uint16_t nw = 0U;
    uint16_t ne_sum_id = 0U;
    uint16_t sw_sum_id = 0U;
    uint16_t se_sum_id = 0U;
    uint8_t ne_empty_mask = 0U;
    uint8_t sw_empty_mask = 0U;
    uint8_t se_empty_mask = 0U;
    uint16_t count_ne = 0U;
    uint16_t count_sw = 0U;
    uint16_t count_se = 0U;
    BucketBitmapLen bitmap_len = 0U;
    BCWordGroupView ne_group;
    BCWordGroupView sw_group;
    BCWordGroupView se_group;

    BCBucketRankDecoder() = default;

    BCBucketRankDecoder(const BCLut &lut, uint64_t bucket_key) {
        reset(lut, bucket_key);
    }

    void reset(const BCLut &lut, uint64_t bucket_key) {
        key = bucket_key;
        nw = static_cast<uint16_t>((key >> 48U) & 0xFFFFU);
        const uint16_t ne = static_cast<uint16_t>((key >> 32U) & 0xFFFFU);
        const uint16_t sw = static_cast<uint16_t>((key >> 16U) & 0xFFFFU);
        const uint16_t se = static_cast<uint16_t>(key & 0xFFFFU);
        if (!lut.word_desc(nw).valid) {
            throw std::invalid_argument("BC bucket decoder key has invalid NW exact word");
        }

        ne_sum_id = packed_sum_id(ne);
        sw_sum_id = packed_sum_id(sw);
        se_sum_id = packed_sum_id(se);
        ne_empty_mask = packed_empty_mask(ne);
        sw_empty_mask = packed_empty_mask(sw);
        se_empty_mask = packed_empty_mask(se);
        ne_group = lut.word_group(ne_sum_id, ne_empty_mask);
        sw_group = lut.word_group(sw_sum_id, sw_empty_mask);
        se_group = lut.word_group(se_sum_id, se_empty_mask);
        count_ne = ne_group.count;
        count_sw = sw_group.count;
        count_se = se_group.count;
        const uint32_t len =
            static_cast<uint32_t>(count_ne) *
            static_cast<uint32_t>(count_sw) *
            static_cast<uint32_t>(count_se);
        if (count_ne == 0U || count_sw == 0U || count_se == 0U ||
            len == 0U || len > kBCMaxBucketBitmapLen ||
            len > std::numeric_limits<BucketBitmapLen>::max()) {
            throw std::out_of_range("BC bucket decoder computed invalid bitmap length");
        }
        bitmap_len = static_cast<BucketBitmapLen>(len);
    }

    [[nodiscard]] BCQuadrantWords unrank(const BCLut &lut, BucketRank rank) const {
        if (rank >= bitmap_len) {
            throw std::out_of_range("BC bucket decoder rank is outside bucket bitmap");
        }
        const uint32_t rank_se = static_cast<uint32_t>(rank) % count_se;
        const uint32_t tmp = static_cast<uint32_t>(rank) / count_se;
        const uint32_t rank_sw = tmp % count_sw;
        const uint32_t rank_ne = tmp / count_sw;
        if (rank_ne >= count_ne) {
            throw std::logic_error("BC bucket decoder computed NE rank outside count");
        }

        return BCQuadrantWords{
            nw,
            ne_group.words[rank_ne],
            sw_group.words[rank_sw],
            se_group.words[rank_se]
        };
    }
};

[[nodiscard]] inline BCQuadrantWords unrank_key_rank_to_quadrants(
    const BCLut &lut,
    uint64_t key,
    BucketRank rank
) {
    return BCBucketRankDecoder(lut, key).unrank(lut, rank);
}

inline void build_prefix256_into(
    const uint64_t *bitmap_words,
    uint32_t bitmap_word_count,
    uint32_t bitmap_len,
    RankPrefix *prefix_out,
    uint32_t prefix_count
) {
    if (bitmap_len > kBCMaxBucketBitmapLen) {
        throw std::invalid_argument("BC prefix bitmap_len exceeds BC bucket maximum");
    }
    const uint32_t needed_words = words_for_bits(bitmap_len);
    if (bitmap_word_count < needed_words) {
        throw std::invalid_argument("BC prefix bitmap_words is shorter than bitmap_len");
    }
    const uint32_t needed_prefix_count = prefix_count_for_bits(bitmap_len);
    if (prefix_count < needed_prefix_count) {
        throw std::invalid_argument("BC prefix output is shorter than bitmap_len");
    }
    if (needed_words != 0U && bitmap_words == nullptr) {
        throw std::invalid_argument("BC prefix bitmap_words pointer is null");
    }
    if (needed_prefix_count != 0U && prefix_out == nullptr) {
        throw std::invalid_argument("BC prefix output pointer is null");
    }

    uint32_t running = 0U;
    for (uint32_t block = 0; block < needed_prefix_count; ++block) {
        if (running > std::numeric_limits<RankPrefix>::max()) {
            throw std::logic_error("BC prefix running popcount exceeds uint16");
        }
        prefix_out[block] = static_cast<RankPrefix>(running);
        const uint32_t block_first_bit = block * kBCRankPrefixBits;
        const uint32_t block_last_bit = std::min<uint32_t>(
            bitmap_len,
            block_first_bit + kBCRankPrefixBits
        );
        const uint32_t first_word = block_first_bit / kBCBitmapWordBits;
        const uint32_t last_word_exclusive =
            words_for_bits(block_last_bit);
        for (uint32_t word = first_word; word < last_word_exclusive; ++word) {
            uint64_t value = bitmap_words[word];
            if (word + 1U == last_word_exclusive && (block_last_bit & 63U) != 0U) {
                value &= (1ULL << (block_last_bit & 63U)) - 1ULL;
            }
            running += popcount64(value);
        }
    }
}

[[nodiscard]] inline std::vector<RankPrefix> build_prefix256(
    const std::vector<uint64_t> &bitmap_words,
    uint32_t bitmap_len
) {
    std::vector<RankPrefix> prefix(prefix_count_for_bits(bitmap_len), 0U);
    build_prefix256_into(
        bitmap_words.data(),
        static_cast<uint32_t>(bitmap_words.size()),
        bitmap_len,
        prefix.data(),
        static_cast<uint32_t>(prefix.size())
    );
    return prefix;
}

[[nodiscard]] inline RankPrefix rank_before(
    const RankPrefix *prefix,
    uint32_t prefix_count,
    const uint64_t *bitmap_words,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const uint32_t rank_u32 = rank;
    const uint32_t block = rank_u32 / kBCRankPrefixBits;
    if (block >= prefix_count) {
        throw std::out_of_range("BC rank_before rank exceeds prefix coverage");
    }
    if (prefix == nullptr) {
        throw std::invalid_argument("BC rank_before prefix pointer is null");
    }
    const uint32_t block_first_word = block * (kBCRankPrefixBits / kBCBitmapWordBits);
    const uint32_t word_in_block = (rank_u32 & (kBCRankPrefixBits - 1U)) / kBCBitmapWordBits;
    const uint32_t bit_in_word = rank_u32 & (kBCBitmapWordBits - 1U);
    const uint32_t target_word = block_first_word + word_in_block;
    if (target_word >= bitmap_word_count) {
        throw std::out_of_range("BC rank_before rank exceeds bitmap word coverage");
    }
    if (bitmap_words == nullptr) {
        throw std::invalid_argument("BC rank_before bitmap_words pointer is null");
    }

    uint32_t count = prefix[block];
    for (uint32_t offset = 0; offset < word_in_block; ++offset) {
        count += popcount64(bitmap_words[block_first_word + offset]);
    }
    if (bit_in_word != 0U) {
        count += popcount64(bitmap_words[target_word] & ((1ULL << bit_in_word) - 1ULL));
    }
    if (count > std::numeric_limits<RankPrefix>::max()) {
        throw std::logic_error("BC rank_before result exceeds uint16");
    }
    return static_cast<RankPrefix>(count);
}

[[nodiscard]] inline RankPrefix rank_before(
    const std::vector<RankPrefix> &prefix,
    const std::vector<uint64_t> &bitmap_words,
    BucketRank rank
) {
    return rank_before(
        prefix.data(),
        static_cast<uint32_t>(prefix.size()),
        bitmap_words.data(),
        static_cast<uint32_t>(bitmap_words.size()),
        rank
    );
}

[[nodiscard]] inline RankPrefix rank_before_le_bytes(
    const uint8_t *prefix_bytes,
    uint32_t prefix_count,
    const uint8_t *bitmap_word_bytes,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const uint32_t rank_u32 = rank;
    const uint32_t block = rank_u32 / kBCRankPrefixBits;
    if (block >= prefix_count) {
        throw std::out_of_range("BC rank_before_le_bytes rank exceeds prefix coverage");
    }
    if (prefix_bytes == nullptr) {
        throw std::invalid_argument("BC rank_before_le_bytes prefix pointer is null");
    }
    const uint32_t block_first_word = block * (kBCRankPrefixBits / kBCBitmapWordBits);
    const uint32_t word_in_block = (rank_u32 & (kBCRankPrefixBits - 1U)) / kBCBitmapWordBits;
    const uint32_t bit_in_word = rank_u32 & (kBCBitmapWordBits - 1U);
    const uint32_t target_word = block_first_word + word_in_block;
    if (target_word >= bitmap_word_count) {
        throw std::out_of_range("BC rank_before_le_bytes rank exceeds bitmap word coverage");
    }
    if (bitmap_word_bytes == nullptr) {
        throw std::invalid_argument("BC rank_before_le_bytes bitmap pointer is null");
    }

    uint32_t count = load_u16_le(prefix_bytes + static_cast<size_t>(block) * sizeof(RankPrefix));
    for (uint32_t offset = 0; offset < word_in_block; ++offset) {
        const uint32_t word = block_first_word + offset;
        count += popcount64(load_u64_le(bitmap_word_bytes + static_cast<size_t>(word) * sizeof(uint64_t)));
    }
    if (bit_in_word != 0U) {
        const uint64_t target = load_u64_le(bitmap_word_bytes + static_cast<size_t>(target_word) * sizeof(uint64_t));
        count += popcount64(target & ((1ULL << bit_in_word) - 1ULL));
    }
    if (count > std::numeric_limits<RankPrefix>::max()) {
        throw std::logic_error("BC rank_before_le_bytes result exceeds uint16");
    }
    return static_cast<RankPrefix>(count);
}

[[nodiscard]] inline bool bitmap_test(
    const uint64_t *bitmap_words,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const uint32_t word = static_cast<uint32_t>(rank) / kBCBitmapWordBits;
    const uint32_t bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
    if (word >= bitmap_word_count) {
        throw std::out_of_range("BC bitmap_test rank exceeds bitmap word coverage");
    }
    if (bitmap_words == nullptr) {
        throw std::invalid_argument("BC bitmap_test bitmap_words pointer is null");
    }
    return ((bitmap_words[word] >> bit) & 1ULL) != 0ULL;
}

[[nodiscard]] inline bool bitmap_test_le_bytes(
    const uint8_t *bitmap_word_bytes,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const uint32_t word = static_cast<uint32_t>(rank) / kBCBitmapWordBits;
    const uint32_t bit = static_cast<uint32_t>(rank) & (kBCBitmapWordBits - 1U);
    if (word >= bitmap_word_count) {
        throw std::out_of_range("BC bitmap_test_le_bytes rank exceeds bitmap word coverage");
    }
    if (bitmap_word_bytes == nullptr) {
        throw std::invalid_argument("BC bitmap_test_le_bytes bitmap pointer is null");
    }
    return ((load_u64_le(bitmap_word_bytes + static_cast<size_t>(word) * sizeof(uint64_t)) >> bit) & 1ULL) != 0ULL;
}

[[nodiscard]] inline BCBitmapRankResult bitmap_test_and_rank(
    const RankPrefix *prefix,
    uint32_t prefix_count,
    const uint64_t *bitmap_words,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const RankPrefix before = rank_before(prefix, prefix_count, bitmap_words, bitmap_word_count, rank);
    return BCBitmapRankResult{bitmap_test(bitmap_words, bitmap_word_count, rank), before};
}

[[nodiscard]] inline BCBitmapRankResult bitmap_test_and_rank_le_bytes(
    const uint8_t *prefix_bytes,
    uint32_t prefix_count,
    const uint8_t *bitmap_word_bytes,
    uint32_t bitmap_word_count,
    BucketRank rank
) {
    const RankPrefix before = rank_before_le_bytes(
        prefix_bytes,
        prefix_count,
        bitmap_word_bytes,
        bitmap_word_count,
        rank
    );
    return BCBitmapRankResult{
        bitmap_test_le_bytes(bitmap_word_bytes, bitmap_word_count, rank),
        before
    };
}

[[nodiscard]] inline bool bitmap_test(const std::vector<uint64_t> &bitmap_words, BucketRank rank) {
    return bitmap_test(bitmap_words.data(), static_cast<uint32_t>(bitmap_words.size()), rank);
}

[[nodiscard]] inline BCBitmapRankResult bitmap_test_and_rank(
    const std::vector<RankPrefix> &prefix,
    const std::vector<uint64_t> &bitmap_words,
    BucketRank rank
) {
    return bitmap_test_and_rank(
        prefix.data(),
        static_cast<uint32_t>(prefix.size()),
        bitmap_words.data(),
        static_cast<uint32_t>(bitmap_words.size()),
        rank
    );
}

} // namespace BC
