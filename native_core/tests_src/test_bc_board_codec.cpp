#include "BCBoardCodec.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCFamilyTable.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionCellScanner;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCQuadrantWords;
using BC::BCScannedBoardEntry;
using BC::BCScannedPositionEntry;
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
            const uint16_t count =
                lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
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

void check_same_quadrants(const BCQuadrantWords &lhs, const BCQuadrantWords &rhs) {
    check(lhs.nw == rhs.nw, "NW word mismatch");
    check(lhs.ne == rhs.ne, "NE word mismatch");
    check(lhs.sw == rhs.sw, "SW word mismatch");
    check(lhs.se == rhs.se, "SE word mismatch");
}

void check_codec_variants(const BCQuadrantWords &q) {
    const uint64_t board = BC::pack_quadrants_to_board(q);
    const uint64_t by_parts =
        BC::pack_nw_quadrant_to_board_bits(q.nw) |
        BC::pack_ne_quadrant_to_board_bits(q.ne) |
        BC::pack_sw_quadrant_to_board_bits(q.sw) |
        BC::pack_se_quadrant_to_board_bits(q.se);
    check(by_parts == board, "quadrant board bit helpers should match full pack");
    check_same_quadrants(BC::unpack_board_to_quadrants(board), q);
}

void test_manual_board_layout() {
    const uint64_t board = 0x123456789ABCDEF0ULL;
    const BCQuadrantWords q = BC::unpack_board_to_quadrants(board);

    check(q.nw == 0x6521U, "manual NW quadrant layout mismatch");
    check(q.ne == 0x8743U, "manual NE quadrant layout mismatch");
    check(q.sw == 0xEDA9U, "manual SW quadrant layout mismatch");
    check(q.se == 0x0FCBU, "manual SE quadrant layout mismatch");
    check(BC::pack_quadrants_to_board(q) == board, "manual pack should reconstruct board");

    check(BC::board_tile(board, 15U) == 0x1U, "A4 board tile mismatch");
    check(BC::board_tile(board, 14U) == 0x2U, "B4 board tile mismatch");
    check(BC::board_tile(board, 0U) == 0x0U, "D1 board tile mismatch");
    check(BC::set_board_tile(0U, 15U, 0xAU) == 0xA000000000000000ULL, "set A4 mismatch");
    check_codec_variants(q);
}

void test_pack_unpack_roundtrip() {
    const std::vector<BCQuadrantWords> fixed = {
        BCQuadrantWords{},
        BCQuadrantWords{0xFFFFU, 0xEEEEU, 0xDDDDU, 0xCCCCU},
        BCQuadrantWords{0x0123U, 0x4567U, 0x89ABU, 0xCDEFU},
        BCQuadrantWords{0x6521U, 0x8743U, 0xEDA9U, 0x0FCBU},
    };
    for (const BCQuadrantWords &q : fixed) {
        check_codec_variants(q);
    }

    std::mt19937 rng(12345U);
    std::uniform_int_distribution<uint32_t> dist(0U, 0xFFFFU);
    for (uint32_t i = 0; i < 10000U; ++i) {
        const BCQuadrantWords q{
            static_cast<uint16_t>(dist(rng)),
            static_cast<uint16_t>(dist(rng)),
            static_cast<uint16_t>(dist(rng)),
            static_cast<uint16_t>(dist(rng))
        };
        check_codec_variants(q);
    }
}

void test_encode_board_adapter() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    check(words.size() >= 4U, "not enough valid words for board adapter test");

    std::mt19937 rng(45678U);
    std::uniform_int_distribution<size_t> dist(0U, words.size() - 1U);
    for (uint32_t i = 0; i < 5000U; ++i) {
        const BCQuadrantWords q{
            words[dist(rng)],
            words[dist(rng)],
            words[dist(rng)],
            words[dist(rng)]
        };
        const BCEncodedKeyRank direct = BC::encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
        const uint64_t board = BC::pack_quadrants_to_board(q);
        const BCEncodedKeyRank from_board = BC::encode_board_to_key_rank(lut, board);
        const BCEncodedKeyRank from_canonical = BC::encode_canonical_board_to_key_rank(lut, board);
        check(direct.valid, "direct quadrant encode should be valid");
        check(from_board.valid, "board adapter encode should be valid");
        check(from_board.key == direct.key, "board adapter key mismatch");
        check(from_board.rank == direct.rank, "board adapter rank mismatch");
        check(from_board.bitmap_len == direct.bitmap_len, "board adapter bitmap_len mismatch");
        check(from_canonical.key == direct.key, "canonical board adapter key mismatch");
        check(from_canonical.rank == direct.rank, "canonical board adapter rank mismatch");
    }
}

void test_bucket_board_decoder() {
    const BCLut lut(test_alphabet());
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 7U);
    const uint32_t bitmap_len =
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count);
    const std::vector<BucketRank> ranks = {
        0U,
        1U,
        7U,
        63U,
        64U,
        255U,
        256U,
        static_cast<BucketRank>(bitmap_len - 1U)
    };

    for (uint16_t nw : {words.front(), words[words.size() / 2U], words.back()}) {
        const BCEncodedKeyRank encoded = encode_from_group_mixed_rank(lut, nw, group, 0U);
        check(encoded.valid, "bucket board decoder fixture key should encode");
        const BC::BCBucketRankDecoder rank_decoder(lut, encoded.key);
        const BC::BCBucketBoardDecoder board_decoder(lut, encoded.key);
        check(board_decoder.bitmap_len() == rank_decoder.bitmap_len, "bucket board decoder bitmap_len mismatch");
        for (BucketRank rank : ranks) {
            if (rank >= rank_decoder.bitmap_len) {
                continue;
            }
            const BCQuadrantWords q = rank_decoder.unrank(lut, rank);
            const uint64_t expected = BC::pack_quadrants_to_board(q);
            const uint64_t fast = board_decoder.board(rank);
            check(fast == expected, "bucket board decoder board mismatch");
            check_same_quadrants(BC::unpack_board_to_quadrants(fast), q);
        }
    }
}

void test_scanner_entries_board_roundtrip() {
    const BCLut lut(test_alphabet());
    const BCFamilyTable axis = BCFamilyTable::from_range(6U, 1U, 0U, 2U);
    const BCCellMatrix matrix(axis);
    const std::vector<uint16_t> words = collect_valid_words(lut);
    const GroupChoice group = find_group_with_count(lut, 7U);
    const uint32_t bitmap_len =
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count);
    check(bitmap_len > 256U, "scanner board roundtrip fixture needs cross-prefix bucket");

    const CellFixture fixture = build_cell(
        lut,
        matrix.cid(1U, 2U),
        {
            encode_from_group(lut, words.front(), group, 0U, 0U, 0U),
            encode_from_group_mixed_rank(lut, words.front(), group, 1U),
            encode_from_group_mixed_rank(lut, words.front(), group, 255U),
            encode_from_group_mixed_rank(lut, words.front(), group, 256U),
            encode_from_group_mixed_rank(lut, words.front(), group, 257U),
            encode_from_group_mixed_rank(
                lut,
                words.front(),
                group,
                static_cast<BucketRank>(bitmap_len - 1U)
            ),
        }
    );
    const std::vector<uint8_t> bytes = write_position_layer(axis, {fixture});
    const BCPositionLayerReader reader(bytes, lut);

    const std::vector<BCScannedPositionEntry> scanned =
        BCPositionCellScanner(reader, fixture.cid).scan();
    check(scanned.size() == fixture.oracle.size(), "scanner board roundtrip entry count mismatch");

    std::vector<BCScannedBoardEntry> board_entries;
    BCPositionCellScanner(reader, fixture.cid).for_each_board(
        [&board_entries](const BCScannedBoardEntry &entry) {
            board_entries.push_back(entry);
        }
    );
    check(board_entries.size() == scanned.size(), "for_each_board entry count mismatch");

    for (size_t i = 0; i < scanned.size(); ++i) {
        const BCScannedPositionEntry &entry = scanned[i];
        const BCScannedBoardEntry &board_entry = board_entries[i];
        check(board_entry.key == entry.key, "for_each_board key mismatch");
        check(board_entry.rank == entry.rank, "for_each_board rank mismatch");
        check(board_entry.local_success_row == entry.local_success_row, "for_each_board row mismatch");
        check(
            fixture.oracle.find({entry.key, entry.rank}) != fixture.oracle.end(),
            "scanner board roundtrip produced unknown key/rank"
        );
        const BCQuadrantWords q{entry.nw, entry.ne, entry.sw, entry.se};
        const uint64_t board = BC::pack_quadrants_to_board(q);
        check(board_entry.board == board, "for_each_board board mismatch");
        check_same_quadrants(BC::unpack_board_to_quadrants(board), q);
        const BCEncodedKeyRank from_board = BC::encode_board_to_key_rank(lut, board);
        check(from_board.valid, "scanner board should encode");
        check(from_board.key == entry.key, "scanner board roundtrip key mismatch");
        check(from_board.rank == entry.rank, "scanner board roundtrip rank mismatch");
        const auto lookup = fixture.payload.lookup(lut, entry.key, entry.rank);
        check(lookup.found, "scanner board roundtrip payload lookup should find entry");
        check(lookup.local_success_row == entry.local_success_row, "scanner board roundtrip local_success_row mismatch");
    }
}

} // namespace

int main() {
    try {
        std::cerr << "test_manual_board_layout\n";
        test_manual_board_layout();
        std::cerr << "test_pack_unpack_roundtrip\n";
        test_pack_unpack_roundtrip();
        std::cerr << "test_encode_board_adapter\n";
        test_encode_board_adapter();
        std::cerr << "test_bucket_board_decoder\n";
        test_bucket_board_decoder();
        std::cerr << "test_scanner_entries_board_roundtrip\n";
        test_scanner_entries_board_roundtrip();
    } catch (const std::exception &ex) {
        std::cerr << "bc_board_codec_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_board_codec_test passed\n";
    return 0;
}
