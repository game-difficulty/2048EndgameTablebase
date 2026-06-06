#include "BCPositionFile.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyTable;
using BC::BCLookupResult;
using BC::BCLut;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BucketRank;
using BC::CellId;
using BC::FinalizedCellPayload;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
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
    check(best.count >= min_count, "failed to find required LUT group");
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
    check(rank_u32 < count * count * count, "mixed rank exceeds group bitmap length");
    const BucketRank ne_rank = static_cast<BucketRank>(rank_u32 / (count * count));
    const uint32_t rem = rank_u32 % (count * count);
    const BucketRank sw_rank = static_cast<BucketRank>(rem / count);
    const BucketRank se_rank = static_cast<BucketRank>(rem % count);
    return encode_from_group(lut, nw, group, ne_rank, sw_rank, se_rank);
}

struct CellFixture {
    CellId cid = 0U;
    FinalizedCellPayload payload;
    std::set<std::pair<uint64_t, BucketRank>> oracle;
};

CellFixture build_cell(
    const BCLut &lut,
    CellId cid,
    const std::vector<BCEncodedKeyRank> &items
) {
    BCCellBuilder builder(lut);
    CellFixture fixture;
    fixture.cid = cid;
    for (const BCEncodedKeyRank &item : items) {
        check(item.valid, "encoded fixture item should be valid");
        builder.insert(item.key, item.rank);
        fixture.oracle.insert({item.key, item.rank});
    }
    fixture.payload = builder.finalize();
    return fixture;
}

std::vector<CellFixture> make_cell_fixtures(
    const BCLut &lut,
    const BCCellMatrix &matrix
) {
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice small_group = find_group_with_count(lut, 4U);
    const GroupChoice cross_group = find_group_with_count(lut, 7U);
    const uint16_t nw0 = words.front();

    std::vector<CellFixture> fixtures;
    fixtures.push_back(build_cell(
        lut,
        matrix.cid(0U, 1U),
        {
            encode_from_group(lut, nw0, small_group, 0U, 0U, 0U),
            encode_from_group(lut, nw0, small_group, 0U, 0U, 1U),
            encode_from_group(lut, nw0, small_group, 1U, 0U, 0U),
        }
    ));
    fixtures.push_back(build_cell(
        lut,
        matrix.cid(1U, 2U),
        {
            encode_from_group(lut, words[3], small_group, 0U, 0U, 0U),
            encode_from_group(lut, words[9], small_group, 0U, 0U, 1U),
            encode_from_group(lut, words[5], small_group, 0U, 1U, 0U),
        }
    ));

    const uint32_t cross_bitmap_len =
        static_cast<uint32_t>(cross_group.count) *
        static_cast<uint32_t>(cross_group.count) *
        static_cast<uint32_t>(cross_group.count);
    check(cross_bitmap_len > 256U, "cross-prefix fixture requires bitmap_len > 256");
    fixtures.push_back(build_cell(
        lut,
        matrix.cid(3U, 3U),
        {
            encode_from_group_mixed_rank(lut, nw0, cross_group, 0U),
            encode_from_group_mixed_rank(lut, nw0, cross_group, 1U),
            encode_from_group_mixed_rank(lut, nw0, cross_group, 255U),
            encode_from_group_mixed_rank(lut, nw0, cross_group, 256U),
            encode_from_group_mixed_rank(lut, nw0, cross_group, 257U),
            encode_from_group_mixed_rank(
                lut,
                nw0,
                cross_group,
                static_cast<BucketRank>(cross_bitmap_len - 1U)
            ),
        }
    ));

    return fixtures;
}

std::vector<uint8_t> write_synthetic_layer(
    const BCFamilyTable &axis,
    const std::vector<CellFixture> &fixtures
) {
    const BCCellMatrix matrix(axis);
    std::map<CellId, const CellFixture *> by_cid;
    for (const CellFixture &fixture : fixtures) {
        by_cid[fixture.cid] = &fixture;
    }

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0; cid < matrix.cell_count(); ++cid) {
        const auto it = by_cid.find(cid);
        if (it == by_cid.end()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, it->second->payload);
    }
    return writer.finish_layer();
}

void check_reader_header(
    const BCPositionLayerReader &reader,
    const BCFamilyTable &axis
) {
    check(reader.axis().layer_sum() == axis.layer_sum(), "reader layer_sum mismatch");
    check(reader.axis().family_unit() == axis.family_unit(), "reader family_unit mismatch");
    check(reader.axis().axis_base_coord() == axis.axis_base_coord(), "reader axis_base mismatch");
    check(reader.axis().family_count() == axis.family_count(), "reader family_count mismatch");
    check(reader.header().key_mode == BC::kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256,
        "reader key_mode mismatch");
    check(reader.header().rank_prefix_bits == BC::kBCRankPrefixBits, "reader rank_prefix_bits mismatch");
    check(reader.header().rank_prefix_type == BC::kBCPositionRankPrefixTypeUint16,
        "reader rank_prefix_type mismatch");
    check(reader.header().rank_payload_align == 8U, "reader rank_payload_align mismatch");
}

void check_descriptor_ranges(const BCPositionLayerReader &reader) {
    const BC::BCPositionHeader &header = reader.header();
    for (CellId cid = 0; cid < reader.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = reader.descriptor(cid);
        const uint64_t bucket_bytes =
            static_cast<uint64_t>(desc.bucket_count) * BC::kBCPositionBucketEntryBytes;
        check(desc.bucket_meta_offset + bucket_bytes <= header.bucket_meta_bytes,
            "descriptor bucket range exceeds bucket stream");
        check(desc.rank_payload_offset + desc.rank_payload_bytes <= header.rank_payload_bytes,
            "descriptor rank payload range exceeds rank stream");
        if (desc.empty()) {
            check(desc.bucket_count == 0U, "empty descriptor should have no buckets");
            check(desc.success_rows == 0U, "empty descriptor should have no success rows");
            check(desc.rank_payload_bytes == 0U, "empty descriptor should have no rank payload");
        }
    }
}

void check_cold_lookups(
    const BCLut &lut,
    const BCPositionLayerReader &reader,
    const std::vector<CellFixture> &fixtures
) {
    for (const CellFixture &fixture : fixtures) {
        for (const auto &[key, rank] : fixture.oracle) {
            const BCLookupResult expected = fixture.payload.lookup(lut, key, rank);
            const BCLookupResult got = reader.cold_lookup(fixture.cid, key, rank);
            check(expected.found, "fixture payload should find inserted key/rank");
            check(got.found, "position cold lookup missed inserted key/rank");
            check(got.local_success_row == expected.local_success_row,
                "position cold lookup success row mismatch");
            check(reader.descriptor(fixture.cid).success_rows == fixture.payload.success_rows,
                "position descriptor success_rows mismatch");
        }

        for (const BC::BCBucketEntry &bucket : fixture.payload.buckets) {
            const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
            bool checked_miss = false;
            for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
                const auto item = std::make_pair(bucket.key, static_cast<BucketRank>(rank));
                if (fixture.oracle.find(item) != fixture.oracle.end()) {
                    continue;
                }
                const BCLookupResult miss = reader.cold_lookup(
                    fixture.cid,
                    bucket.key,
                    static_cast<BucketRank>(rank)
                );
                check(!miss.found, "position cold lookup should miss absent rank");
                checked_miss = true;
                break;
            }
            check(checked_miss, "miss lookup did not run for bucket");
        }
    }
}

void store_u32_le(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
    check(offset + 4U <= bytes.size(), "store_u32_le out of range");
    bytes[offset + 0U] = static_cast<uint8_t>(value & 0xFFU);
    bytes[offset + 1U] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    bytes[offset + 2U] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    bytes[offset + 3U] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

void store_u64_le(std::vector<uint8_t> &bytes, size_t offset, uint64_t value) {
    check(offset + 8U <= bytes.size(), "store_u64_le out of range");
    for (uint32_t i = 0; i < 8U; ++i) {
        bytes[offset + i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU);
    }
}

void expect_open_failure(
    const std::vector<uint8_t> &bytes,
    const BCLut &lut,
    const char *message
) {
    bool threw = false;
    try {
        BCPositionLayerReader reader(bytes, lut);
        (void)reader;
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}

void test_position_roundtrip_and_lookup() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_synthetic_layer(axis, fixtures);

    BCPositionLayerReader reader(bytes, lut);
    check_reader_header(reader, axis);
    check(reader.cell_count() == matrix.cell_count(), "reader cell_count mismatch");
    check_descriptor_ranges(reader);
    check(reader.descriptor(matrix.cid(0U, 0U)).empty(), "cid 0 should be an empty cell");
    check_cold_lookups(lut, reader, fixtures);
}

void test_header_validation_errors() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_synthetic_layer(axis, fixtures);

    std::vector<uint8_t> bad_version = bytes;
    store_u32_le(bad_version, 4U, 999U);
    expect_open_failure(bad_version, lut, "bad format version should fail");

    std::vector<uint8_t> bad_key_mode = bytes;
    store_u32_le(bad_key_mode, 12U, 999U);
    expect_open_failure(bad_key_mode, lut, "bad key mode should fail");
}

void test_descriptor_validation_errors() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_synthetic_layer(axis, fixtures);

    std::vector<uint8_t> bad_descriptor = bytes;
    const size_t first_descriptor_offset = BC::kBCPositionHeaderBytes;
    const size_t rank_payload_bytes_offset = first_descriptor_offset + 24U;
    store_u64_le(bad_descriptor, rank_payload_bytes_offset, std::numeric_limits<uint64_t>::max());
    expect_open_failure(bad_descriptor, lut, "bad descriptor rank payload range should fail");
}

} // namespace

int main() {
    try {
        std::cerr << "test_position_roundtrip_and_lookup\n";
        test_position_roundtrip_and_lookup();
        std::cerr << "test_header_validation_errors\n";
        test_header_validation_errors();
        std::cerr << "test_descriptor_validation_errors\n";
        test_descriptor_validation_errors();
    } catch (const std::exception &ex) {
        std::cerr << "bc_position_file_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_position_file_test passed\n";
    return 0;
}
