#include "BCCellBuilder.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCLut;
using BC::BCEncodedKeyRank;
using BC::BCWordDesc;
using BC::BucketRank;
using BC::CellId;
using BC::FinalizedCellPayload;

uint32_t g_sort_hook_calls = 0U;
size_t g_sort_hook_last_count = 0U;
bool g_sort_hook_last_descending = true;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void tracking_keyvalue_sort_uint64_uint32(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
) {
    ++g_sort_hook_calls;
    g_sort_hook_last_count = count;
    g_sort_hook_last_descending = descending;
    std::vector<std::pair<uint64_t, uint32_t>> items;
    items.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        items.push_back({keys[i], values[i]});
    }
    std::sort(
        items.begin(),
        items.end(),
        [descending](const auto &lhs, const auto &rhs) {
            return descending ? lhs.first > rhs.first : lhs.first < rhs.first;
        }
    );
    for (size_t i = 0; i < count; ++i) {
        keys[i] = items[i].first;
        values[i] = items[i].second;
    }
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
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

struct GroupChoice {
    uint16_t sum_id = 0U;
    uint8_t empty_mask = 0U;
    uint16_t count = 0U;
};

GroupChoice find_group_with_count(const BCLut &lut, uint16_t min_count) {
    GroupChoice best;
    for (uint32_t sum_id = 0; sum_id < lut.sum_count(); ++sum_id) {
        for (uint32_t mask = 0; mask < 16U; ++mask) {
            const uint16_t count = lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            if (count >= min_count && count > best.count) {
                best = GroupChoice{static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask), count};
            }
        }
    }
    check(best.count >= min_count, "failed to find enough same-key ranks");
    return best;
}

BCEncodedKeyRank encode_from_group(
    const BCLut &lut,
    uint16_t nw,
    const GroupChoice &group,
    BucketRank ne_rank,
    BucketRank sw_rank,
    BucketRank se_rank
) {
    return BC::encode_key_and_rank(
        lut,
        nw,
        lut.unrank_word(group.sum_id, group.empty_mask, ne_rank),
        lut.unrank_word(group.sum_id, group.empty_mask, sw_rank),
        lut.unrank_word(group.sum_id, group.empty_mask, se_rank)
    );
}

BCEncodedKeyRank encode_from_group_mixed_rank(
    const BCLut &lut,
    uint16_t nw,
    const GroupChoice &group,
    BucketRank rank
) {
    const uint32_t count = group.count;
    const uint32_t rank_u32 = rank;
    check(rank_u32 < count * count * count, "mixed rank exceeds same-group bitmap length");
    const BucketRank ne_rank = static_cast<BucketRank>(rank_u32 / (count * count));
    const uint32_t rem = rank_u32 % (count * count);
    const BucketRank sw_rank = static_cast<BucketRank>(rem / count);
    const BucketRank se_rank = static_cast<BucketRank>(rem % count);
    return encode_from_group(lut, nw, group, ne_rank, sw_rank, se_rank);
}

uint32_t naive_rank_before(const std::set<BucketRank> &ranks, BucketRank rank) {
    return static_cast<uint32_t>(std::distance(ranks.begin(), ranks.lower_bound(rank)));
}

std::set<BucketRank> oracle_ranks_for_key(
    const std::set<std::pair<uint64_t, BucketRank>> &oracle,
    uint64_t key
) {
    std::set<BucketRank> out;
    for (const auto &[item_key, rank] : oracle) {
        if (item_key == key) {
            out.insert(rank);
        }
    }
    return out;
}

void verify_payload_layout_and_prefix(
    const BCLut &lut,
    const FinalizedCellPayload &payload,
    const std::set<std::pair<uint64_t, BucketRank>> &oracle
) {
    uint32_t expected_success_offset = 0U;
    uint64_t previous_key = 0U;
    bool first_key = true;
    for (const auto &bucket : payload.buckets) {
        if (!first_key) {
            check(previous_key < bucket.key, "finalized bucket keys must be strictly sorted");
        }
        first_key = false;
        previous_key = bucket.key;
        check(bucket.rank_payload_offset % 8U == 0U, "rank_payload_offset must be 8-byte aligned");
        check(bucket.success_row_offset == expected_success_offset, "success_row_offset mismatch");

        const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
        const uint32_t prefix_count = BC::prefix_count_for_bits(bitmap_len);
        const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
        const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
        check(bitmap_offset % 8U == 0U, "bitmap words must be 8-byte aligned");
        check(
            static_cast<uint64_t>(bitmap_offset) + static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t) <=
                payload.rank_payload.size(),
            "bucket rank payload is out of bounds"
        );

        const uint8_t *prefix = payload.rank_payload.data() + bucket.rank_payload_offset;
        const uint8_t *bitmap_words = payload.rank_payload.data() + bitmap_offset;
        const std::set<BucketRank> ranks = oracle_ranks_for_key(oracle, bucket.key);
        check(!ranks.empty(), "oracle should have ranks for every finalized bucket");
        for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
            const auto result = BC::bitmap_test_and_rank_le_bytes(
                prefix,
                prefix_count,
                bitmap_words,
                bitmap_word_count,
                static_cast<BucketRank>(rank)
            );
            const bool expected_found = ranks.find(static_cast<BucketRank>(rank)) != ranks.end();
            check(result.found == expected_found, "payload bitmap bit mismatch");
            check(
                result.rank_before == naive_rank_before(ranks, static_cast<BucketRank>(rank)),
                "payload prefix rank mismatch"
            );
        }
        expected_success_offset += static_cast<uint32_t>(ranks.size());
    }
    check(payload.success_rows == expected_success_offset, "payload success_rows mismatch");
}

void test_single_key_multiple_ranks() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const uint16_t nw = words.front();
    const GroupChoice group = find_group_with_count(lut, 4U);
    const std::vector<BCEncodedKeyRank> encoded = {
        encode_from_group(lut, nw, group, 0U, 0U, 0U),
        encode_from_group(lut, nw, group, 0U, 0U, 1U),
        encode_from_group(lut, nw, group, 0U, 1U, 0U),
        encode_from_group(lut, nw, group, 1U, 0U, 0U),
    };

    BCCellBuilder builder(lut);
    std::set<std::pair<uint64_t, BucketRank>> oracle;
    for (const BCEncodedKeyRank &item : encoded) {
        check(item.valid, "encoded item should be valid");
        builder.insert(item.key, item.rank);
        oracle.insert({item.key, item.rank});
    }
    builder.insert(encoded[1].key, encoded[1].rank);
    builder.insert(encoded[1].key, encoded[1].rank);

    for (const BCEncodedKeyRank &item : encoded) {
        check(builder.contains(item.key, item.rank), "builder should contain inserted rank");
    }

    const FinalizedCellPayload payload = builder.finalize();
    check(payload.buckets.size() == 1U, "single-key finalize should produce one bucket");
    check(payload.success_rows == oracle.size(), "duplicate insert should not increase success rows");
    const std::set<BucketRank> sorted_ranks = oracle_ranks_for_key(oracle, encoded.front().key);
    for (BucketRank rank : sorted_ranks) {
        const auto lookup = payload.lookup(lut, encoded.front().key, rank);
        check(lookup.found, "lookup should find inserted rank");
        check(
            lookup.local_success_row == naive_rank_before(sorted_ranks, rank),
            "single-key local_success_row should equal rank order"
        );
        const auto raw_lookup = BC::lookup_finalized_cell(
            lut,
            BC::BCBucketEntryView{
                payload.buckets.data(),
                static_cast<uint32_t>(payload.buckets.size())
            },
            BC::BCRankPayloadView{
                payload.rank_payload.data(),
                static_cast<uint32_t>(payload.rank_payload.size())
            },
            encoded.front().key,
            rank
        );
        check(raw_lookup.found, "raw finalized lookup should find inserted rank");
        check(
            raw_lookup.local_success_row == lookup.local_success_row,
            "raw finalized lookup row should match vector lookup"
        );
    }
    verify_payload_layout_and_prefix(lut, payload, oracle);
}

void test_cross_prefix256_block() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const uint16_t nw = words.front();
    const GroupChoice group = find_group_with_count(lut, 7U);
    const uint32_t bitmap_len =
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count);
    check(bitmap_len > 256U, "cross-prefix test requires bitmap_len > 256");
    const std::vector<BucketRank> ranks = {
        0U,
        1U,
        255U,
        256U,
        257U,
        static_cast<BucketRank>(bitmap_len - 1U)
    };

    BCCellBuilder builder(lut);
    std::set<std::pair<uint64_t, BucketRank>> oracle;
    uint64_t key = 0U;
    bool have_key = false;
    for (BucketRank rank : ranks) {
        const BCEncodedKeyRank encoded = encode_from_group_mixed_rank(lut, nw, group, rank);
        check(encoded.valid, "cross-prefix encoded item should be valid");
        check(encoded.bitmap_len == bitmap_len, "cross-prefix bitmap_len mismatch");
        if (!have_key) {
            key = encoded.key;
            have_key = true;
        }
        check(encoded.key == key, "cross-prefix ranks should share one key");
        check(encoded.rank == rank, "cross-prefix encoded rank mismatch");
        builder.insert(encoded.key, encoded.rank);
        oracle.insert({encoded.key, encoded.rank});
    }

    const FinalizedCellPayload payload = builder.finalize();
    check(payload.buckets.size() == 1U, "cross-prefix test should produce one bucket");
    check(payload.success_rows == ranks.size(), "cross-prefix success_rows mismatch");
    const std::set<BucketRank> sorted_ranks = oracle_ranks_for_key(oracle, key);
    for (BucketRank rank : ranks) {
        const auto lookup = payload.lookup(lut, key, rank);
        check(lookup.found, "cross-prefix lookup missed inserted rank");
        check(
            lookup.local_success_row == naive_rank_before(sorted_ranks, rank),
            "cross-prefix local_success_row mismatch"
        );
    }
    verify_payload_layout_and_prefix(lut, payload, oracle);
}

void test_multiple_keys_sorted() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 3U);
    std::vector<BCEncodedKeyRank> encoded = {
        encode_from_group(lut, words[7], group, 0U, 0U, 0U),
        encode_from_group(lut, words[3], group, 0U, 0U, 1U),
        encode_from_group(lut, words[5], group, 0U, 1U, 0U),
    };

    BCCellBuilder builder(lut);
    std::set<std::pair<uint64_t, BucketRank>> oracle;
    for (const auto &item : encoded) {
        builder.insert(item.key, item.rank);
        oracle.insert({item.key, item.rank});
    }
    const FinalizedCellPayload payload = builder.finalize();
    check(payload.buckets.size() == 3U, "expected three distinct exact-NW keys");
    for (size_t i = 1; i < payload.buckets.size(); ++i) {
        check(payload.buckets[i - 1U].key < payload.buckets[i].key, "bucket keys not sorted");
    }
    verify_payload_layout_and_prefix(lut, payload, oracle);
}

void test_finalize_keyvalue_sort_hook() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 3U);
    const std::vector<BCEncodedKeyRank> encoded = {
        encode_from_group(lut, words[11], group, 0U, 0U, 0U),
        encode_from_group(lut, words[2], group, 0U, 0U, 1U),
        encode_from_group(lut, words[8], group, 0U, 1U, 0U),
        encode_from_group(lut, words[5], group, 1U, 0U, 0U),
    };

    BCCellBuilder builder(lut);
    std::set<std::pair<uint64_t, BucketRank>> oracle;
    for (const auto &item : encoded) {
        builder.insert(item.key, item.rank);
        oracle.insert({item.key, item.rank});
    }

    g_sort_hook_calls = 0U;
    g_sort_hook_last_count = 0U;
    g_sort_hook_last_descending = true;
    BC::BCCellFinalizeOptions options;
    options.keyvalue_sort = tracking_keyvalue_sort_uint64_uint32;
    options.simd_sort_min_bucket_count = 2U;
    const FinalizedCellPayload payload = builder.finalize(options);
    check(g_sort_hook_calls == 1U, "keyvalue sort hook should be called once");
    check(g_sort_hook_last_count == payload.buckets.size(), "keyvalue sort hook count mismatch");
    check(!g_sort_hook_last_descending, "keyvalue sort hook should sort ascending");
    for (size_t i = 1; i < payload.buckets.size(); ++i) {
        check(payload.buckets[i - 1U].key < payload.buckets[i].key, "sort hook bucket keys not sorted");
    }
    verify_payload_layout_and_prefix(lut, payload, oracle);
}

void test_insert_report_and_encoded_insert() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 4U);
    const BCEncodedKeyRank encoded0 = encode_from_group(lut, words.front(), group, 0U, 0U, 0U);
    const BCEncodedKeyRank encoded1 = encode_from_group(lut, words.front(), group, 0U, 0U, 1U);
    check(encoded0.valid && encoded1.valid, "insert report encoded values should be valid");

    BCCellBuilder reported(lut);
    const auto first = reported.insert_and_report(encoded0.key, encoded0.rank);
    check(first.new_bucket, "first insert should create a bucket");
    check(first.new_rank, "first insert should create a rank");
    check(reported.live_rows() == 1U, "live_rows should track first insert");
    const auto duplicate = reported.insert_and_report(encoded0.key, encoded0.rank);
    check(!duplicate.new_bucket, "duplicate insert should not create a bucket");
    check(!duplicate.new_rank, "duplicate insert should not create a rank");
    check(reported.live_rows() == 1U, "live_rows should ignore duplicate insert");
    const auto second = reported.insert_encoded_and_report(encoded1);
    check(!second.new_bucket, "same-key encoded insert should not create a bucket");
    check(second.new_rank, "same-key encoded insert should create a new rank");
    check(reported.live_rows() == 2U, "live_rows should track encoded insert");

    BCCellBuilder baseline(lut);
    baseline.insert(encoded0.key, encoded0.rank);
    baseline.insert(encoded0.key, encoded0.rank);
    baseline.insert(encoded1.key, encoded1.rank);

    const FinalizedCellPayload reported_payload = reported.finalize();
    const FinalizedCellPayload baseline_payload = baseline.finalize();
    check(reported_payload.success_rows == baseline_payload.success_rows, "encoded insert success_rows mismatch");
    check(reported_payload.buckets.size() == baseline_payload.buckets.size(), "encoded insert bucket count mismatch");
    for (const BCEncodedKeyRank &encoded : {encoded0, encoded1}) {
        const auto lhs = reported_payload.lookup(lut, encoded.key, encoded.rank);
        const auto rhs = baseline_payload.lookup(lut, encoded.key, encoded.rank);
        check(lhs.found && rhs.found, "encoded insert lookup should find rank");
        check(lhs.local_success_row == rhs.local_success_row, "encoded insert local row mismatch");
    }
}

void test_merge_from_dedups_and_ors() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 4U);
    const BCEncodedKeyRank r0 = encode_from_group(lut, words.front(), group, 0U, 0U, 0U);
    const BCEncodedKeyRank r1 = encode_from_group(lut, words.front(), group, 0U, 0U, 1U);
    const BCEncodedKeyRank r2 = encode_from_group(lut, words.front(), group, 0U, 1U, 0U);
    check(r0.valid && r1.valid && r2.valid, "merge encoded values should be valid");

    BCCellBuilder target(lut);
    (void)target.insert_encoded_and_report(r0);
    (void)target.insert_encoded_and_report(r1);

    BCCellBuilder source(lut);
    (void)source.insert_encoded_and_report(r1);
    (void)source.insert_encoded_and_report(r2);

    const auto merge = target.merge_from(source);
    check(merge.new_buckets == 0U, "merge should not create a new bucket for same key");
    check(merge.new_ranks == 1U, "merge should add exactly one new rank");
    check(merge.duplicate_ranks == 1U, "merge should report one duplicate rank");
    check(target.live_rows() == 3U, "merge live_rows mismatch");

    const FinalizedCellPayload payload = target.finalize();
    check(payload.success_rows == 3U, "merge finalize should use live_count");
    for (const BCEncodedKeyRank &encoded : {r0, r1, r2}) {
        check(payload.lookup(lut, encoded.key, encoded.rank).found, "merge payload should contain inserted rank");
    }
}

void test_random_oracle() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    BCCellBuilder builder(lut);
    std::set<std::pair<uint64_t, BucketRank>> oracle;
    std::mt19937 rng(123456789U);
    std::uniform_int_distribution<size_t> word_dist(0U, words.size() - 1U);
    for (uint32_t i = 0; i < 2000U; ++i) {
        const BCEncodedKeyRank encoded = BC::encode_key_and_rank(
            lut,
            words[word_dist(rng)],
            words[word_dist(rng)],
            words[word_dist(rng)],
            words[word_dist(rng)]
        );
        if (!encoded.valid) {
            continue;
        }
        builder.insert(encoded.key, encoded.rank);
        oracle.insert({encoded.key, encoded.rank});
        if ((i % 7U) == 0U) {
            builder.insert(encoded.key, encoded.rank);
        }
    }

    const FinalizedCellPayload payload = builder.finalize();
    check(payload.success_rows == oracle.size(), "random oracle success_rows mismatch");
    for (const auto &[key, rank] : oracle) {
        const auto ranks = oracle_ranks_for_key(oracle, key);
        const auto lookup = payload.lookup(lut, key, rank);
        check(lookup.found, "random oracle lookup missed inserted rank");
        check(
            lookup.local_success_row ==
                payload.buckets[static_cast<size_t>(
                    std::lower_bound(
                        payload.buckets.begin(),
                        payload.buckets.end(),
                        key,
                        [](const BC::BCBucketEntry &entry, uint64_t target) { return entry.key < target; }
                    ) - payload.buckets.begin()
                )].success_row_offset + naive_rank_before(ranks, rank),
            "random oracle local_success_row mismatch"
        );
    }

    uint32_t negative_checks = 0U;
    for (const auto &bucket : payload.buckets) {
        const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
        const std::set<BucketRank> ranks = oracle_ranks_for_key(oracle, bucket.key);
        for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
            if (ranks.find(static_cast<BucketRank>(rank)) != ranks.end()) {
                continue;
            }
            const auto lookup = payload.lookup(lut, bucket.key, static_cast<BucketRank>(rank));
            check(!lookup.found, "lookup should miss rank not inserted for existing key");
            if (++negative_checks >= 200U) {
                break;
            }
        }
        if (negative_checks >= 200U) {
            break;
        }
    }
    check(negative_checks > 0U, "negative lookup checks did not run");
    verify_payload_layout_and_prefix(lut, payload, oracle);
}

} // namespace

int main() {
    try {
        std::cerr << "test_single_key_multiple_ranks\n";
        test_single_key_multiple_ranks();
        std::cerr << "test_cross_prefix256_block\n";
        test_cross_prefix256_block();
        std::cerr << "test_multiple_keys_sorted\n";
        test_multiple_keys_sorted();
        std::cerr << "test_finalize_keyvalue_sort_hook\n";
        test_finalize_keyvalue_sort_hook();
        std::cerr << "test_insert_report_and_encoded_insert\n";
        test_insert_report_and_encoded_insert();
        std::cerr << "test_merge_from_dedups_and_ors\n";
        test_merge_from_dedups_and_ors();
        std::cerr << "test_random_oracle\n";
        test_random_oracle();
    } catch (const std::exception &ex) {
        std::cerr << "bc_cell_builder_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_cell_builder_test passed\n";
    return 0;
}
