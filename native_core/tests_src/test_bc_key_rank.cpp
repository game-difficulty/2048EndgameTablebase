#include "BCKeyRank.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using BC::BCLut;
using BC::BCEncodedKeyRank;
using BC::BCWordDesc;
using BC::BucketRank;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throws(Fn &&fn, const char *message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

bool brute_word_legal(uint16_t word, const std::vector<uint8_t> &alphabet) {
    for (uint32_t i = 0; i < 4U; ++i) {
        const uint8_t tile = BC::word_tile(word, i);
        if (std::find(alphabet.begin(), alphabet.end(), tile) == alphabet.end()) {
            return false;
        }
    }
    return true;
}

uint8_t brute_empty_mask(uint16_t word) {
    uint8_t mask = 0U;
    for (uint32_t i = 0; i < 4U; ++i) {
        if (BC::word_tile(word, i) == 0U) {
            mask |= static_cast<uint8_t>(1U << i);
        }
    }
    return mask;
}

uint32_t naive_popcount_before(const std::vector<uint64_t> &words, uint32_t rank) {
    uint32_t count = 0U;
    const uint32_t full_words = rank / 64U;
    const uint32_t bit = rank & 63U;
    for (uint32_t i = 0; i < full_words; ++i) {
        count += BC::popcount64(words[i]);
    }
    if (bit != 0U) {
        count += BC::popcount64(words[full_words] & ((1ULL << bit) - 1ULL));
    }
    return count;
}

bool naive_test_bit(const std::vector<uint64_t> &words, uint32_t rank) {
    return ((words[rank / 64U] >> (rank & 63U)) & 1ULL) != 0ULL;
}

void append_u16_le(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void append_u64_le(std::vector<uint8_t> &out, uint64_t value) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        out.push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
    }
}

std::vector<uint8_t> prefix_to_le_bytes(const std::vector<BC::RankPrefix> &prefix) {
    std::vector<uint8_t> out;
    out.reserve(prefix.size() * sizeof(BC::RankPrefix));
    for (BC::RankPrefix value : prefix) {
        append_u16_le(out, value);
    }
    return out;
}

std::vector<uint8_t> bitmap_to_le_bytes(const std::vector<uint64_t> &words) {
    std::vector<uint8_t> out;
    out.reserve(words.size() * sizeof(uint64_t));
    for (uint64_t value : words) {
        append_u64_le(out, value);
    }
    return out;
}

std::vector<uint16_t> collect_valid_words(const BCLut &lut) {
    std::vector<uint16_t> words;
    for (uint32_t word = 0; word < BC::kBCQuadrantWordCount; ++word) {
        if (lut.word_desc(static_cast<uint16_t>(word)).valid) {
            words.push_back(static_cast<uint16_t>(word));
        }
    }
    return words;
}

void test_word_desc_roundtrip() {
    const std::vector<uint8_t> alphabet = test_alphabet();
    const BCLut lut(alphabet);
    uint32_t valid_count = 0U;
    uint16_t max_group_count = 0U;
    for (uint32_t word = 0; word < BC::kBCQuadrantWordCount; ++word) {
        const uint16_t w = static_cast<uint16_t>(word);
        const BCWordDesc &desc = lut.word_desc(w);
        const bool expected_valid = brute_word_legal(w, alphabet);
        check(desc.valid == expected_valid, "word_desc valid flag mismatch");
        if (!desc.valid) {
            continue;
        }
        ++valid_count;
        check(desc.empty_mask == brute_empty_mask(w), "word_desc empty_mask mismatch");
        check(desc.sum_id < lut.sum_count(), "word_desc sum_id out of range");
        const uint16_t count = lut.count4(desc.sum_id, desc.empty_mask);
        max_group_count = std::max(max_group_count, count);
        check(count <= BC::kBCMaxWordsPerSumMask, "count4 exceeded 36");
        check(desc.rank < count, "word_desc rank outside count4");
        check(lut.unrank_word(desc.sum_id, desc.empty_mask, desc.rank) == w, "unrank roundtrip failed");
        check(
            BC::packed_sum_id(lut.pack_sum_mask_for_word(w)) == desc.sum_id,
            "packed sum_id mismatch"
        );
        check(
            BC::packed_empty_mask(lut.pack_sum_mask_for_word(w)) == desc.empty_mask,
            "packed empty_mask mismatch"
        );
    }
    check(valid_count > 0U, "test LUT should have valid words");
    check(max_group_count > 0U, "test LUT should have nonempty groups");
}

void test_group_counts() {
    const BCLut lut(test_alphabet());
    for (uint32_t sum_id = 0; sum_id < lut.sum_count(); ++sum_id) {
        for (uint32_t mask = 0; mask < 16U; ++mask) {
            const uint16_t count = lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            const BC::BCWordGroupView group =
                lut.word_group(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            check(group.count == count, "word_group count mismatch");
            check((group.words != nullptr) == (count != 0U), "word_group pointer/count mismatch");
            check(count <= BC::kBCMaxWordsPerSumMask, "count4 group exceeds 36");
            for (uint32_t rank = 0; rank < count; ++rank) {
                const uint16_t word =
                    lut.unrank_word(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask), static_cast<BucketRank>(rank));
                check(group.words[rank] == word, "word_group word mismatch");
                const BCWordDesc &desc = lut.word_desc(word);
                check(desc.valid, "unrank returned invalid word");
                check(desc.sum_id == sum_id, "unrank sum_id mismatch");
                check(desc.empty_mask == mask, "unrank empty_mask mismatch");
                check(desc.rank == rank, "unrank rank mismatch");
            }
        }
    }

    const std::array<uint32_t, 16U> bad_values{};
    expect_throws(
        [&] {
            (void)BCLut(std::vector<uint8_t>{0U, 1U, 2U, 3U}, bad_values);
        },
        "LUT should reject sum+mask groups larger than 36"
    );
}

void test_key_rank_mixed_radix() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    check(words.size() >= 4U, "not enough valid words for key/rank test");

    std::mt19937 rng(123456789U);
    std::uniform_int_distribution<size_t> dist(0U, words.size() - 1U);
    for (uint32_t i = 0; i < 10000U; ++i) {
        const uint16_t nw = words[dist(rng)];
        const uint16_t ne = words[dist(rng)];
        const uint16_t sw = words[dist(rng)];
        const uint16_t se = words[dist(rng)];
        const BCEncodedKeyRank encoded = BC::encode_key_and_rank(lut, nw, ne, sw, se);
        check(encoded.valid, "valid quadrants should encode");

        const BCWordDesc &ne_desc = lut.word_desc(ne);
        const BCWordDesc &sw_desc = lut.word_desc(sw);
        const BCWordDesc &se_desc = lut.word_desc(se);
        const uint32_t expected_rank =
            (static_cast<uint32_t>(ne_desc.rank) * encoded.count_sw + sw_desc.rank) *
            encoded.count_se + se_desc.rank;
        const uint32_t expected_len =
            static_cast<uint32_t>(encoded.count_ne) *
            static_cast<uint32_t>(encoded.count_sw) *
            static_cast<uint32_t>(encoded.count_se);

        check(encoded.rank == expected_rank, "mixed-radix rank mismatch");
        check(encoded.bitmap_len == expected_len, "bitmap_len mismatch");
        check(encoded.rank < encoded.bitmap_len, "rank must be inside bitmap_len");
        check(encoded.bitmap_len <= BC::kBCMaxBucketBitmapLen, "bitmap_len exceeds BC max");
        check(BC::bitmap_len_from_key(lut, encoded.key) == encoded.bitmap_len, "bitmap_len_from_key mismatch");

        const uint16_t key_nw = static_cast<uint16_t>(encoded.key >> 48U);
        const uint16_t key_ne = static_cast<uint16_t>((encoded.key >> 32U) & 0xFFFFU);
        const uint16_t key_sw = static_cast<uint16_t>((encoded.key >> 16U) & 0xFFFFU);
        const uint16_t key_se = static_cast<uint16_t>(encoded.key & 0xFFFFU);
        check(key_nw == nw, "key should store exact NW in bits 63:48");
        check(BC::packed_sum_id(key_ne) == ne_desc.sum_id, "key NE sum_id mismatch");
        check(BC::packed_empty_mask(key_ne) == ne_desc.empty_mask, "key NE empty_mask mismatch");
        check(BC::packed_sum_id(key_sw) == sw_desc.sum_id, "key SW sum_id mismatch");
        check(BC::packed_empty_mask(key_sw) == sw_desc.empty_mask, "key SW empty_mask mismatch");
        check(BC::packed_sum_id(key_se) == se_desc.sum_id, "key SE sum_id mismatch");
        check(BC::packed_empty_mask(key_se) == se_desc.empty_mask, "key SE empty_mask mismatch");
    }

    const BCEncodedKeyRank invalid = BC::encode_key_and_rank(lut, 0xEEEEU, words[0], words[1], words[2]);
    check(!invalid.valid, "invalid quadrant word should produce invalid encode result");
}

void test_near_max_lut_bitmap_len() {
    const BCLut lut(test_alphabet());
    uint16_t best_sum_id = 0U;
    uint8_t best_mask = 0U;
    uint16_t best_count = 0U;
    for (uint32_t sum_id = 0; sum_id < lut.sum_count(); ++sum_id) {
        for (uint32_t mask = 0; mask < 16U; ++mask) {
            const uint16_t count = lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            if (count > best_count) {
                best_count = count;
                best_sum_id = static_cast<uint16_t>(sum_id);
                best_mask = static_cast<uint8_t>(mask);
            }
        }
    }
    check(best_count > 1U, "expected a multi-word LUT group");
    const uint16_t nw = lut.unrank_word(best_sum_id, best_mask, 0U);
    const uint16_t q = lut.unrank_word(best_sum_id, best_mask, static_cast<BucketRank>(best_count - 1U));
    const BCEncodedKeyRank encoded = BC::encode_key_and_rank(lut, nw, q, q, q);
    check(encoded.valid, "near-max encode should be valid");
    check(encoded.count_ne == best_count, "near-max NE count mismatch");
    check(encoded.count_sw == best_count, "near-max SW count mismatch");
    check(encoded.count_se == best_count, "near-max SE count mismatch");
    check(
        encoded.bitmap_len == static_cast<uint32_t>(best_count) * best_count * best_count,
        "near-max bitmap_len mismatch"
    );
    check(encoded.bitmap_len <= BC::kBCMaxBucketBitmapLen, "near-max bitmap_len exceeds hard bound");
}

void test_prefix256_for_len(uint32_t bitmap_len, bool exhaustive) {
    std::vector<uint64_t> words(BC::words_for_bits(bitmap_len), 0U);
    for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
        if ((rank % 7U) == 0U || (rank % 13U) == 0U || (rank % 257U) == 3U) {
            words[rank / 64U] |= 1ULL << (rank & 63U);
        }
    }
    const std::vector<BC::RankPrefix> prefix = BC::build_prefix256(words, bitmap_len);
    std::vector<BC::RankPrefix> pointer_prefix(prefix.size(), 0U);
    BC::build_prefix256_into(
        words.data(),
        static_cast<uint32_t>(words.size()),
        bitmap_len,
        pointer_prefix.data(),
        static_cast<uint32_t>(pointer_prefix.size())
    );
    check(pointer_prefix == prefix, "pointer prefix build differs from vector wrapper");
    check(prefix.size() == BC::prefix_count_for_bits(bitmap_len), "prefix count mismatch");
    const std::vector<uint8_t> prefix_bytes = prefix_to_le_bytes(prefix);
    const std::vector<uint8_t> bitmap_bytes = bitmap_to_le_bytes(words);

    std::vector<uint32_t> boundary_bits = {
        0U,
        1U,
        63U,
        64U,
        65U,
        255U,
        256U,
        257U,
        511U,
        512U,
        bitmap_len
    };
    std::sort(boundary_bits.begin(), boundary_bits.end());
    boundary_bits.erase(std::unique(boundary_bits.begin(), boundary_bits.end()), boundary_bits.end());
    for (uint32_t bit_index : boundary_bits) {
        if (bit_index > bitmap_len) {
            continue;
        }
        const uint32_t expected = naive_popcount_before(words, bit_index);
        const BC::RankPrefix got = BC::rank_before_bit_index(
            prefix.data(),
            static_cast<uint32_t>(prefix.size()),
            words.data(),
            static_cast<uint32_t>(words.size()),
            bitmap_len,
            bit_index
        );
        const BC::RankPrefix got_le = BC::rank_before_bit_index_le_bytes(
            prefix_bytes.data(),
            static_cast<uint32_t>(prefix.size()),
            bitmap_bytes.data(),
            static_cast<uint32_t>(words.size()),
            bitmap_len,
            bit_index
        );
        check(got == expected, "rank_before_bit_index boundary mismatch");
        check(got_le == expected, "rank_before_bit_index_le_bytes boundary mismatch");
    }

    for (uint32_t word_index = 0U; word_index <= words.size(); ++word_index) {
        const uint32_t bit_index = std::min<uint32_t>(bitmap_len, word_index * BC::kBCBitmapWordBits);
        const uint32_t expected = naive_popcount_before(words, bit_index);
        const BC::RankPrefix got = BC::rank_before_word_index(
            prefix.data(),
            static_cast<uint32_t>(prefix.size()),
            words.data(),
            static_cast<uint32_t>(words.size()),
            bitmap_len,
            word_index
        );
        const BC::RankPrefix got_le = BC::rank_before_word_index_le_bytes(
            prefix_bytes.data(),
            static_cast<uint32_t>(prefix.size()),
            bitmap_bytes.data(),
            static_cast<uint32_t>(words.size()),
            bitmap_len,
            word_index
        );
        check(got == expected, "rank_before_word_index mismatch");
        check(got_le == expected, "rank_before_word_index_le_bytes mismatch");
    }

    if (exhaustive) {
        for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
            const auto result = BC::bitmap_test_and_rank(prefix, words, static_cast<BucketRank>(rank));
            const auto pointer_result = BC::bitmap_test_and_rank(
                pointer_prefix.data(),
                static_cast<uint32_t>(pointer_prefix.size()),
                words.data(),
                static_cast<uint32_t>(words.size()),
                static_cast<BucketRank>(rank)
            );
            check(result.found == naive_test_bit(words, rank), "bitmap_test mismatch");
            check(result.rank_before == naive_popcount_before(words, rank), "prefix rank_before mismatch");
            check(pointer_result.found == result.found, "pointer bitmap_test mismatch");
            check(pointer_result.rank_before == result.rank_before, "pointer rank_before mismatch");
        }
    } else {
        std::mt19937 rng(987654321U);
        std::uniform_int_distribution<uint32_t> dist(0U, bitmap_len - 1U);
        for (uint32_t i = 0; i < 10000U; ++i) {
            const uint32_t rank = dist(rng);
            const auto result = BC::bitmap_test_and_rank(prefix, words, static_cast<BucketRank>(rank));
            const auto pointer_result = BC::bitmap_test_and_rank(
                pointer_prefix.data(),
                static_cast<uint32_t>(pointer_prefix.size()),
                words.data(),
                static_cast<uint32_t>(words.size()),
                static_cast<BucketRank>(rank)
            );
            check(result.found == naive_test_bit(words, rank), "random bitmap_test mismatch");
            check(result.rank_before == naive_popcount_before(words, rank), "random prefix rank_before mismatch");
            check(pointer_result.found == result.found, "random pointer bitmap_test mismatch");
            check(pointer_result.rank_before == result.rank_before, "random pointer rank_before mismatch");
        }
    }
}

void test_prefix256() {
    test_prefix256_for_len(4096U, true);
    test_prefix256_for_len(BC::kBCMaxBucketBitmapLen, true);
    expect_throws(
        [] {
            const std::vector<uint64_t> empty;
            (void)BC::build_prefix256(empty, 65U);
        },
        "build_prefix256 should reject short bitmap_words"
    );
    expect_throws(
        [] {
            uint64_t word = 0U;
            BC::RankPrefix prefix = 0U;
            BC::build_prefix256_into(&word, 1U, 65U, &prefix, 0U);
        },
        "build_prefix256_into should reject short prefix output"
    );
}

} // namespace

int main() {
    try {
        std::cerr << "test_word_desc_roundtrip\n";
        test_word_desc_roundtrip();
        std::cerr << "test_group_counts\n";
        test_group_counts();
        std::cerr << "test_key_rank_mixed_radix\n";
        test_key_rank_mixed_radix();
        std::cerr << "test_near_max_lut_bitmap_len\n";
        test_near_max_lut_bitmap_len();
        std::cerr << "test_prefix256\n";
        test_prefix256();
    } catch (const std::exception &ex) {
        std::cerr << "bc_key_rank_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_key_rank_test passed\n";
    return 0;
}
