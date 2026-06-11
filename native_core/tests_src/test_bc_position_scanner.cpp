#include "BCLoadedCellScanner.h"
#include "BCPositionScanner.h"

#include <chrono>
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
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCLoadedCellScanner;
using BC::BCPositionStreamingReader;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCScannedPositionEntry;
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
            ("bc_position_scanner_test_" + std::to_string(static_cast<long long>(stamp)));
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
    const uint32_t cross_bitmap_len =
        static_cast<uint32_t>(cross_group.count) *
        static_cast<uint32_t>(cross_group.count) *
        static_cast<uint32_t>(cross_group.count);
    check(cross_bitmap_len > 256U, "cross-prefix fixture requires bitmap_len > 256");

    return {
        build_cell(
            lut,
            matrix.cid(0U, 1U),
            {
                encode_from_group(lut, nw0, small_group, 0U, 0U, 0U),
                encode_from_group(lut, nw0, small_group, 0U, 0U, 1U),
                encode_from_group(lut, nw0, small_group, 1U, 0U, 0U),
            }
        ),
        build_cell(
            lut,
            matrix.cid(1U, 2U),
            {
                encode_from_group(lut, words[3], small_group, 0U, 0U, 0U),
                encode_from_group(lut, words[9], small_group, 0U, 0U, 1U),
                encode_from_group(lut, words[5], small_group, 0U, 1U, 0U),
            }
        ),
        build_cell(
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
        ),
    };
}

std::vector<uint8_t> write_position_layer(
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

bool same_entry(const BCScannedPositionEntry &lhs, const BCScannedPositionEntry &rhs) {
    return lhs.key == rhs.key &&
        lhs.rank == rhs.rank &&
        lhs.local_success_row == rhs.local_success_row &&
        lhs.nw == rhs.nw &&
        lhs.ne == rhs.ne &&
        lhs.sw == rhs.sw &&
        lhs.se == rhs.se;
}

bool same_entries(
    const std::vector<BCScannedPositionEntry> &lhs,
    const std::vector<BCScannedPositionEntry> &rhs
) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!same_entry(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

bool same_board_entry(const BC::BCScannedBoardEntry &lhs, const BC::BCScannedBoardEntry &rhs) {
    return lhs.key == rhs.key &&
        lhs.rank == rhs.rank &&
        lhs.local_success_row == rhs.local_success_row &&
        lhs.board == rhs.board;
}

bool same_board_entries(
    const std::vector<BC::BCScannedBoardEntry> &lhs,
    const std::vector<BC::BCScannedBoardEntry> &rhs
) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!same_board_entry(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

void check_same_quadrants(const BC::BCQuadrantWords &lhs, const BC::BCQuadrantWords &rhs) {
    check(lhs.nw == rhs.nw, "quadrant NW mismatch");
    check(lhs.ne == rhs.ne, "quadrant NE mismatch");
    check(lhs.sw == rhs.sw, "quadrant SW mismatch");
    check(lhs.se == rhs.se, "quadrant SE mismatch");
}

void test_key_rank_unrank_roundtrip() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 7U);
    const std::vector<BucketRank> ranks = {0U, 1U, 17U, 255U, 256U, 257U};
    for (BucketRank rank : ranks) {
        const BCEncodedKeyRank encoded = encode_from_group_mixed_rank(lut, words.front(), group, rank);
        const BC::BCQuadrantWords q = BC::unrank_key_rank_to_quadrants(lut, encoded.key, encoded.rank);
        const BCEncodedKeyRank roundtrip = BC::encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
        check(roundtrip.valid, "unranked quadrants should re-encode");
        check(roundtrip.key == encoded.key, "unrank roundtrip key mismatch");
        check(roundtrip.rank == encoded.rank, "unrank roundtrip rank mismatch");
    }
}

void test_bucket_decoder_matches_helper() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 7U);
    const std::vector<BucketRank> ranks = {0U, 1U, 17U, 255U, 256U, 257U};
    for (BucketRank rank : ranks) {
        const BCEncodedKeyRank encoded = encode_from_group_mixed_rank(lut, words.front(), group, rank);
        const BC::BCBucketRankDecoder decoder(lut, encoded.key);
        check(decoder.key == encoded.key, "bucket decoder key mismatch");
        check(decoder.bitmap_len == encoded.bitmap_len, "bucket decoder bitmap_len mismatch");
        const BC::BCQuadrantWords from_decoder = decoder.unrank(lut, encoded.rank);
        const BC::BCQuadrantWords from_helper =
            BC::unrank_key_rank_to_quadrants(lut, encoded.key, encoded.rank);
        check_same_quadrants(from_decoder, from_helper);
        const BCEncodedKeyRank roundtrip = BC::encode_key_and_rank(
            lut,
            from_decoder.nw,
            from_decoder.ne,
            from_decoder.sw,
            from_decoder.se
        );
        check(roundtrip.valid, "decoder quadrants should re-encode");
        check(roundtrip.key == encoded.key, "decoder roundtrip key mismatch");
        check(roundtrip.rank == encoded.rank, "decoder roundtrip rank mismatch");
    }
}

void test_position_cell_scanner() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 3U);
    const BCCellMatrix matrix(axis);
    const std::vector<CellFixture> fixtures = make_cell_fixtures(lut, matrix);
    const std::vector<uint8_t> bytes = write_position_layer(axis, fixtures);
    const BCPositionLayerReader reader(bytes, lut);
    TempDir tmp;
    const std::filesystem::path path = tmp.path / "layer.bcpos";
    BC::write_position_layer_to_file(path, bytes);
    const BCPositionStreamingReader streaming_reader =
        BCPositionStreamingReader::open_buffered(path, lut);

    const std::vector<BCScannedPositionEntry> empty_scan =
        BCPositionCellScanner(reader, matrix.cid(0U, 0U)).scan();
    check(empty_scan.empty(), "empty cell scanner should return no entries");
    const BC::BCLoadedCell empty_loaded = streaming_reader.load_cell(matrix.cid(0U, 0U));
    check(
        BCLoadedCellScanner(lut, empty_loaded.view()).scan().empty(),
        "loaded empty cell scanner should return no entries"
    );

    for (const CellFixture &fixture : fixtures) {
        std::vector<BCScannedPositionEntry> streamed;
        BCPositionCellScanner(reader, fixture.cid).for_each(
            [&streamed](const BCScannedPositionEntry &entry) {
                streamed.push_back(entry);
            }
        );
        const std::vector<BCScannedPositionEntry> scanned =
            BCPositionCellScanner(reader, fixture.cid).scan();
        check(same_entries(streamed, scanned), "scanner for_each output should match scan output");

        const BC::BCLoadedCell loaded = streaming_reader.load_cell(fixture.cid);
        const std::vector<BCScannedPositionEntry> loaded_scanned =
            BCLoadedCellScanner(lut, loaded.view()).scan();
        check(same_entries(scanned, loaded_scanned), "loaded cell scanner output should match memory scanner");

        std::vector<BC::BCScannedBoardEntry> memory_boards;
        std::vector<BC::BCScannedBoardEntry> loaded_boards;
        BCPositionCellScanner(reader, fixture.cid).for_each_board(
            [&memory_boards](const BC::BCScannedBoardEntry &entry) {
                memory_boards.push_back(entry);
            }
        );
        BCLoadedCellScanner(lut, loaded.view()).for_each_board(
            [&loaded_boards](const BC::BCScannedBoardEntry &entry) {
                loaded_boards.push_back(entry);
            }
        );
        check(same_board_entries(memory_boards, loaded_boards),
            "loaded cell board scanner output should match memory scanner");

        check(scanned.size() == fixture.oracle.size(), "scanner entry count mismatch");
        uint32_t previous_row = 0U;
        bool first = true;
        for (const BCScannedPositionEntry &entry : scanned) {
            const auto key_rank = std::make_pair(entry.key, entry.rank);
            check(fixture.oracle.find(key_rank) != fixture.oracle.end(), "scanner produced unknown key/rank");

            const BCEncodedKeyRank roundtrip =
                BC::encode_key_and_rank(lut, entry.nw, entry.ne, entry.sw, entry.se);
            check(roundtrip.valid, "scanner quadrants should re-encode");
            check(roundtrip.key == entry.key, "scanner quadrant roundtrip key mismatch");
            check(roundtrip.rank == entry.rank, "scanner quadrant roundtrip rank mismatch");

            const auto expected_lookup = fixture.payload.lookup(lut, entry.key, entry.rank);
            check(expected_lookup.found, "payload lookup should find scanner entry");
            check(
                expected_lookup.local_success_row == entry.local_success_row,
                "scanner local_success_row mismatch"
            );
            if (!first) {
                check(previous_row < entry.local_success_row, "scanner local_success_row should increase");
            }
            first = false;
            previous_row = entry.local_success_row;
        }
    }
}

} // namespace

int main() {
    try {
        std::cerr << "test_key_rank_unrank_roundtrip\n";
        test_key_rank_unrank_roundtrip();
        std::cerr << "test_bucket_decoder_matches_helper\n";
        test_bucket_decoder_matches_helper();
        std::cerr << "test_position_cell_scanner\n";
        test_position_cell_scanner();
    } catch (const std::exception &ex) {
        std::cerr << "bc_position_scanner_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_position_scanner_test passed\n";
    return 0;
}
