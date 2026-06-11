#include "BCDirectFileIO.h"
#include "BCPositionCellLoader.h"
#include "BCPositionFile.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
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
using BC::BCPositionStreamingReader;
using BC::BucketRank;
using BC::CellId;
using BC::FinalizedCellPayload;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("bc_position_file_test_" + std::to_string(static_cast<long long>(stamp)));
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

const BCLut &shared_test_lut() {
    static const BCLut *lut = new BCLut(test_alphabet());
    return *lut;
}

bool direct_io_tests_enabled() {
#if defined(_WIN32) || defined(__linux__)
    const char *value = std::getenv("BC_RUN_DIRECT_IO_TESTS");
    return value != nullptr && std::string(value) == "1";
#else
    return false;
#endif
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
    check(reader.axis().coords() == axis.coords(), "reader axis coord table mismatch");
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

void check_streaming_cold_lookups(
    const BCLut &lut,
    const BCPositionStreamingReader &reader,
    const std::vector<CellFixture> &fixtures
) {
    for (const CellFixture &fixture : fixtures) {
        for (const auto &[key, rank] : fixture.oracle) {
            const BCLookupResult expected = fixture.payload.lookup(lut, key, rank);
            const BCLookupResult got = reader.cold_lookup(fixture.cid, key, rank);
            check(expected.found, "fixture payload should find inserted key/rank");
            check(got.found, "streaming position cold lookup missed inserted key/rank");
            check(got.local_success_row == expected.local_success_row,
                "streaming position cold lookup success row mismatch");
            check(reader.descriptor(fixture.cid).success_rows == fixture.payload.success_rows,
                "streaming position descriptor success_rows mismatch");
        }
    }
}

std::vector<const FinalizedCellPayload *> make_payload_views(
    uint32_t cell_count,
    const std::vector<CellFixture> &fixtures
) {
    std::vector<const FinalizedCellPayload *> payloads(cell_count, nullptr);
    for (const CellFixture &fixture : fixtures) {
        payloads[fixture.cid] = &fixture.payload;
    }
    return payloads;
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
    const BCLut &lut = shared_test_lut();
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

void test_position_noncontiguous_axis_roundtrip() {
    const BCLut &lut = shared_test_lut();
    const BCFamilyTable axis(40U, 2U, std::vector<BC::FamilyCoord>{0U, 2U, 5U, 9U});
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_synthetic_layer(axis, fixtures);

    const BCPositionLayerReader reader(bytes, lut);
    check_reader_header(reader, axis);
    check(reader.header().axis_coord_table_bytes == axis.family_count() * sizeof(uint32_t),
        "noncontiguous position axis coord byte count mismatch");
    check(reader.header().descriptor_table_offset ==
            BC::kBCPositionHeaderBytes + reader.header().axis_coord_table_bytes,
        "noncontiguous position descriptor table offset mismatch");
    check(!reader.axis().contains_coord(1U), "noncontiguous position reader should preserve gaps");
    check_cold_lookups(lut, reader, fixtures);
}

void test_header_validation_errors() {
    const BCLut &lut = shared_test_lut();
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
    const BCLut &lut = shared_test_lut();
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_synthetic_layer(axis, fixtures);

    std::vector<uint8_t> bad_descriptor = bytes;
    const size_t first_descriptor_offset =
        static_cast<size_t>(BC::kBCPositionHeaderBytes + BC::bc_axis_coord_table_bytes(axis.family_count()));
    const size_t rank_payload_bytes_offset = first_descriptor_offset + 24U;
    store_u64_le(bad_descriptor, rank_payload_bytes_offset, std::numeric_limits<uint64_t>::max());
    expect_open_failure(bad_descriptor, lut, "bad descriptor rank payload range should fail");
}

void test_position_streaming_writer_buffered_and_direct() {
    const BCLut &lut = shared_test_lut();
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> memory_bytes = write_synthetic_layer(axis, fixtures);
    const std::vector<const FinalizedCellPayload *> payloads =
        make_payload_views(matrix.cell_count(), fixtures);
    TempDir tmp;

    const std::filesystem::path buffered_path = tmp.path / "buffered.bcpos";
    BC::BCFileIOStats buffered_stats;
    {
        BC::BCBufferedFileWriter writer(buffered_path);
        const uint64_t logical_size =
            BC::write_position_payloads_to_file(writer, axis, payloads, &buffered_stats);
        check(logical_size == memory_bytes.size(), "buffered streaming position logical size mismatch");
    }
    check(std::filesystem::file_size(buffered_path) == memory_bytes.size(),
        "buffered streaming position file size mismatch");
    check(buffered_stats.requested_bytes == memory_bytes.size(),
        "buffered streaming writer requested byte count mismatch");
    check(buffered_stats.backend_bytes == memory_bytes.size(),
        "buffered streaming writer backend byte count mismatch");
    const std::vector<uint8_t> buffered_bytes = BC::read_position_layer_from_file(buffered_path);
    check(buffered_bytes == memory_bytes, "buffered streaming position bytes mismatch");
    const BCPositionLayerReader buffered_reader(buffered_bytes, lut);
    check_cold_lookups(lut, buffered_reader, fixtures);

#if defined(_WIN32) || defined(__linux__)
    if (!direct_io_tests_enabled()) {
        std::cerr << "skip direct position streaming writer test; set BC_RUN_DIRECT_IO_TESTS=1 to enable\n";
        return;
    }
    const std::filesystem::path direct_path = tmp.path / "direct.bcpos";
    BC::BCFileIOStats direct_stats;
    uint64_t direct_logical_size = 0U;
    {
        BC::BCDirectFileWriter writer(direct_path);
        direct_logical_size =
            BC::write_position_payloads_to_file(writer, axis, payloads, &direct_stats);
    }
    check(direct_logical_size == memory_bytes.size(), "direct streaming position logical size mismatch");
    const uint64_t direct_physical_size = std::filesystem::file_size(direct_path);
    check(direct_physical_size >= direct_logical_size, "direct streaming position physical size too small");
    check((direct_physical_size % 4096U) == 0U, "direct streaming position physical size should be aligned");
    check(direct_stats.requested_bytes == memory_bytes.size(),
        "direct streaming writer requested byte count mismatch");
    check(direct_stats.backend_bytes >= memory_bytes.size(),
        "direct streaming writer backend bytes should cover logical file");

    std::vector<uint8_t> direct_prefix(static_cast<size_t>(direct_logical_size));
    BC::BCBufferedFileReader raw_reader(direct_path);
    raw_reader.read_at(0U, direct_prefix.data(), direct_prefix.size());
    check(direct_prefix == memory_bytes, "direct streaming position logical bytes mismatch");

    BC::BCDirectFileIOOptions direct_options;
    direct_options.logical_size = direct_logical_size;
    direct_options.overlapped = false;
    BCPositionStreamingReader direct_reader(
        std::make_unique<BC::BCDirectFileReader>(direct_path, direct_options),
        lut
    );
    check(direct_reader.file_size() == direct_logical_size,
        "direct streaming reader should expose logical size");
    check_streaming_cold_lookups(lut, direct_reader, fixtures);
#endif
}

} // namespace

int main() {
    try {
        std::cerr << "test_position_roundtrip_and_lookup\n";
        test_position_roundtrip_and_lookup();
        std::cerr << "test_position_noncontiguous_axis_roundtrip\n";
        test_position_noncontiguous_axis_roundtrip();
        std::cerr << "test_header_validation_errors\n";
        test_header_validation_errors();
        std::cerr << "test_descriptor_validation_errors\n";
        test_descriptor_validation_errors();
        std::cerr << "test_position_streaming_writer_buffered_and_direct\n";
        test_position_streaming_writer_buffered_and_direct();
    } catch (const std::exception &ex) {
        std::cerr << "bc_position_file_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_position_file_test passed\n";
    return 0;
}
