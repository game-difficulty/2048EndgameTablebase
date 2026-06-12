#include "BCDirectFileIO.h"
#include "BCLoadedCellScanner.h"
#include "BCPositionCellLoader.h"
#include "BCPositionFamilyRemapReader.h"

#include <array>
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

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("bc_position_cell_loader_test_" + std::to_string(static_cast<long long>(stamp)));
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

uint16_t find_word_with_sum(const BCLut &lut, uint32_t sum) {
    for (uint32_t word = 0; word < BC::kBCQuadrantWordCount; ++word) {
        const BC::BCWordDesc &desc = lut.word_desc(static_cast<uint16_t>(word));
        if (desc.valid && desc.sum == sum) {
            return static_cast<uint16_t>(word);
        }
    }
    throw std::runtime_error("failed to find valid word with requested sum");
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

struct FixtureLayer {
    TempDir tmp;
    const BCLut &lut;
    BCFamilyTable axis;
    BCCellMatrix matrix;
    std::vector<CellFixture> fixtures;
    std::vector<uint8_t> bytes;
    std::filesystem::path path;
    BCPositionLayerReader memory_reader;
    BCPositionStreamingReader streaming_reader;

    FixtureLayer()
        : lut(shared_test_lut()),
          axis(BCFamilyTable::from_range(6U, 1U, 0U, 3U)),
          matrix(axis),
          fixtures(make_cell_fixtures(lut, matrix)),
          bytes(write_synthetic_layer(axis, fixtures)),
          path(tmp.path / "layer.bcpos"),
          memory_reader(bytes, lut) {
        BC::write_position_layer_to_file(path, bytes);
        streaming_reader = BCPositionStreamingReader::open_buffered(path, lut);
    }
};

const CellFixture &fixture_by_cid(const std::vector<CellFixture> &fixtures, CellId cid) {
    for (const CellFixture &fixture : fixtures) {
        if (fixture.cid == cid) {
            return fixture;
        }
    }
    throw std::runtime_error("fixture cid not found");
}

void check_header_and_descriptors(const FixtureLayer &fixture) {
    const BCPositionStreamingReader &reader = fixture.streaming_reader;
    check(reader.file_size() == fixture.bytes.size(), "streaming reader file size mismatch");
    check(reader.axis().layer_sum() == fixture.axis.layer_sum(), "streaming reader layer_sum mismatch");
    check(reader.axis().family_unit() == fixture.axis.family_unit(), "streaming reader family_unit mismatch");
    check(reader.axis().axis_base_coord() == fixture.axis.axis_base_coord(), "streaming reader axis_base mismatch");
    check(reader.axis().family_count() == fixture.axis.family_count(), "streaming reader family_count mismatch");
    check(reader.axis().coords() == fixture.axis.coords(), "streaming reader axis coords mismatch");
    check(reader.cell_count() == fixture.matrix.cell_count(), "streaming reader cell_count mismatch");
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        const auto &expected = fixture.memory_reader.descriptor(cid);
        const auto &got = reader.descriptor(cid);
        check(got.bucket_count == expected.bucket_count, "streaming descriptor bucket_count mismatch");
        check(got.success_rows == expected.success_rows, "streaming descriptor success_rows mismatch");
        check(got.bucket_meta_offset == expected.bucket_meta_offset, "streaming descriptor bucket offset mismatch");
        check(got.rank_payload_offset == expected.rank_payload_offset, "streaming descriptor rank offset mismatch");
        check(got.rank_payload_bytes == expected.rank_payload_bytes, "streaming descriptor rank bytes mismatch");
        check(got.flags_or_padding == expected.flags_or_padding, "streaming descriptor flags mismatch");
    }
}

void check_loaded_cell_matches_fixture(
    const FixtureLayer &fixture,
    const BC::BCLoadedCell &cell
) {
    const CellFixture &expected = fixture_by_cid(fixture.fixtures, cell.cid);
    check(cell.success_rows == expected.payload.success_rows, "loaded cell success rows mismatch");
    check(cell.buckets.size() == expected.payload.buckets.size(), "loaded cell bucket count mismatch");
    for (size_t i = 0U; i < cell.buckets.size(); ++i) {
        check(cell.buckets[i].key == expected.payload.buckets[i].key, "loaded cell bucket key mismatch");
        check(
            cell.buckets[i].rank_payload_offset == expected.payload.buckets[i].rank_payload_offset,
            "loaded cell bucket rank payload offset mismatch"
        );
        check(
            cell.buckets[i].success_row_offset == expected.payload.buckets[i].success_row_offset,
            "loaded cell bucket success row offset mismatch"
        );
    }
    check(cell.rank_payload == expected.payload.rank_payload, "loaded cell rank payload mismatch");
    for (const auto &[key, rank] : expected.oracle) {
        const BCLookupResult expected_lookup = expected.payload.lookup(fixture.lut, key, rank);
        const BCLookupResult got_lookup = cell.lookup(fixture.lut, key, rank);
        const BCLookupResult streaming_lookup =
            fixture.streaming_reader.cold_lookup(cell.cid, key, rank);
        check(expected_lookup.found && got_lookup.found && streaming_lookup.found,
            "loaded cell lookup missed inserted item");
        check(got_lookup.local_success_row == expected_lookup.local_success_row,
            "loaded cell lookup local row mismatch");
        check(streaming_lookup.local_success_row == expected_lookup.local_success_row,
            "streaming cold lookup local row mismatch");
    }
}

void test_open_reads_header_and_descriptors_only() {
    FixtureLayer fixture;
    check_header_and_descriptors(fixture);
}

void test_load_empty_and_single_cell() {
    FixtureLayer fixture;
    const CellId empty_cid = fixture.matrix.cid(0U, 0U);
    BC::BCCellLoadStats empty_stats;
    const BC::BCLoadedCell empty = fixture.streaming_reader.load_cell(empty_cid, &empty_stats);
    check(empty.cid == empty_cid, "empty loaded cell cid mismatch");
    check(empty.view().empty(), "empty loaded cell should be empty");
    check(empty.success_rows == 0U, "empty loaded cell success_rows should be zero");
    check(empty_stats.requested_extents == 0U, "empty cell should not request extents");
    check(empty_stats.coalesced_extents == 0U, "empty cell should not read extents");

    BC::BCCellLoadStats stats;
    const BC::BCLoadedCell cell = fixture.streaming_reader.load_cell(fixture.fixtures.front().cid, &stats);
    check_loaded_cell_matches_fixture(fixture, cell);
    check(stats.requested_extents == 2U, "single non-empty cell should request bucket and rank extents");
    check(stats.coalesced_extents <= stats.requested_extents, "single cell coalesced extent count invalid");
    check(stats.read_bytes >= stats.requested_bytes, "single cell read bytes should cover requested bytes");
    check(stats.backend_read_ops == stats.coalesced_extents, "buffered backend read ops should match coalesced extents");
    check(stats.backend_read_bytes == stats.read_bytes, "buffered backend read bytes should match coalesced bytes");
}

void test_loaded_cell_word_range_scanner_uses_prefix() {
    FixtureLayer fixture;
    const BC::BCLoadedCell cell = fixture.streaming_reader.load_cell(fixture.fixtures[2].cid);
    const BC::BCLoadedCellView view = cell.view();
    check(view.buckets.size == 1U, "word range scanner fixture should have one bucket");

    std::vector<BC::BCScannedPositionEntry> full;
    BC::BCLoadedCellScanner(fixture.lut, view).for_each(
        [&full](const BC::BCScannedPositionEntry &entry) {
            full.push_back(entry);
        }
    );

    std::vector<BC::BCScannedPositionEntry> ranged;
    for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
        const uint32_t bitmap_len = BC::bitmap_len_from_key(fixture.lut, view.buckets.data[bucket_i].key);
        const uint32_t word_count = BC::words_for_bits(bitmap_len);
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            BC::BCLoadedCellScanner(fixture.lut, view).for_each_bucket_word_range(
                bucket_i,
                word_i,
                word_i + 1U,
                [&ranged](const BC::BCScannedPositionEntry &entry) {
                    ranged.push_back(entry);
                }
            );
        }
    }

    check(ranged.size() == full.size(), "word range scanner emitted different entry count");
    for (size_t i = 0U; i < full.size(); ++i) {
        check(ranged[i].key == full[i].key, "word range scanner key mismatch");
        check(ranged[i].rank == full[i].rank, "word range scanner rank mismatch");
        check(ranged[i].local_success_row == full[i].local_success_row,
            "word range scanner local row mismatch");
        check(ranged[i].nw == full[i].nw, "word range scanner NW mismatch");
        check(ranged[i].ne == full[i].ne, "word range scanner NE mismatch");
        check(ranged[i].sw == full[i].sw, "word range scanner SW mismatch");
        check(ranged[i].se == full[i].se, "word range scanner SE mismatch");
    }

    std::vector<BC::BCScannedPositionEntry> from_256;
    BC::BCLoadedCellScanner(fixture.lut, view).for_each_bucket_word_range(
        0U,
        BC::kBCRankPrefixBits / BC::kBCBitmapWordBits,
        0U,
        [&from_256](const BC::BCScannedPositionEntry &entry) {
            from_256.push_back(entry);
        }
    );
    check(!from_256.empty(), "word range scanner expected entries from 256-bit block onward");
    check(from_256.front().rank >= BC::kBCRankPrefixBits,
        "word range scanner 256-bit start rank mismatch");
    check(from_256.front().local_success_row == 3U,
        "word range scanner should use prefix count before 256-bit block");
}

void test_load_cells_coalesces_and_preserves_order() {
    FixtureLayer fixture;
    std::vector<CellId> cids;
    cids.push_back(fixture.fixtures[2].cid);
    cids.push_back(fixture.matrix.cid(0U, 0U));
    cids.push_back(fixture.fixtures[0].cid);
    cids.push_back(fixture.fixtures[1].cid);

    BC::BCCellLoadStats stats;
    const std::vector<BC::BCLoadedCell> cells = fixture.streaming_reader.load_cells(cids, &stats);
    check(cells.size() == cids.size(), "loaded cell vector size mismatch");
    for (size_t i = 0U; i < cids.size(); ++i) {
        check(cells[i].cid == cids[i], "load_cells should preserve input order");
    }
    check(cells[1].view().empty(), "load_cells empty cell should stay empty");
    check_loaded_cell_matches_fixture(fixture, cells[0]);
    check_loaded_cell_matches_fixture(fixture, cells[2]);
    check_loaded_cell_matches_fixture(fixture, cells[3]);
    check(stats.requested_extents == 6U, "three non-empty cells should request six extents");
    check(stats.coalesced_extents < stats.requested_extents, "batch cell load should coalesce adjacent extents");
    check(stats.read_bytes >= stats.requested_bytes, "batch read bytes should cover requested bytes");
    check(stats.backend_read_ops == stats.coalesced_extents, "batch backend read ops should match coalesced extents");
    check(stats.backend_read_bytes == stats.read_bytes, "batch backend read bytes should match coalesced bytes");
}

void test_direct_streaming_reader_matches_buffered() {
#if defined(_WIN32) || defined(__linux__)
    if (!direct_io_tests_enabled()) {
        std::cerr << "skip direct position streaming reader test; set BC_RUN_DIRECT_IO_TESTS=1 to enable\n";
        return;
    }
    FixtureLayer fixture;
    const std::filesystem::path direct_path = fixture.tmp.path / "layer_direct.bcpos";
    {
        BC::BCBufferedFileWriter writer(direct_path);
        writer.resize(BC::bc_direct_align_up(static_cast<uint64_t>(fixture.bytes.size()), 4096U));
        writer.write_at(0U, fixture.bytes.data(), static_cast<uint64_t>(fixture.bytes.size()));
        writer.flush();
    }

    BC::BCDirectFileIOOptions options;
    options.logical_size = fixture.bytes.size();
    options.overlapped = false;
    BCPositionStreamingReader direct_reader(
        std::make_unique<BC::BCDirectFileReader>(direct_path, options),
        fixture.lut
    );
    check(direct_reader.file_size() == fixture.bytes.size(), "direct streaming reader logical size mismatch");

    std::vector<CellId> cids;
    cids.push_back(fixture.fixtures[2].cid);
    cids.push_back(fixture.matrix.cid(0U, 0U));
    cids.push_back(fixture.fixtures[0].cid);
    cids.push_back(fixture.fixtures[1].cid);
    BC::BCCellLoadStats direct_stats;
    const std::vector<BC::BCLoadedCell> direct_cells = direct_reader.load_cells(cids, &direct_stats);
    BC::BCCellLoadStats buffered_stats;
    const std::vector<BC::BCLoadedCell> buffered_cells =
        fixture.streaming_reader.load_cells(cids, &buffered_stats);
    check(direct_cells.size() == buffered_cells.size(), "direct loaded cell count mismatch");
    for (size_t i = 0U; i < direct_cells.size(); ++i) {
        check(direct_cells[i].cid == buffered_cells[i].cid, "direct loaded cid mismatch");
        check(direct_cells[i].success_rows == buffered_cells[i].success_rows,
            "direct loaded success rows mismatch");
        check(direct_cells[i].buckets.size() == buffered_cells[i].buckets.size(),
            "direct loaded bucket count mismatch");
        for (size_t bucket_i = 0U; bucket_i < direct_cells[i].buckets.size(); ++bucket_i) {
            check(direct_cells[i].buckets[bucket_i].key == buffered_cells[i].buckets[bucket_i].key,
                "direct loaded bucket key mismatch");
            check(
                direct_cells[i].buckets[bucket_i].rank_payload_offset ==
                    buffered_cells[i].buckets[bucket_i].rank_payload_offset,
                "direct loaded bucket rank payload offset mismatch"
            );
            check(
                direct_cells[i].buckets[bucket_i].success_row_offset ==
                    buffered_cells[i].buckets[bucket_i].success_row_offset,
                "direct loaded bucket success row offset mismatch"
            );
        }
        check(direct_cells[i].rank_payload == buffered_cells[i].rank_payload,
            "direct loaded rank payload mismatch");
    }
    check(direct_stats.requested_extents == buffered_stats.requested_extents,
        "direct requested extent count mismatch");
    check(direct_stats.coalesced_extents == buffered_stats.coalesced_extents,
        "direct coalesced extent count mismatch");
    check(direct_stats.backend_read_ops <= direct_stats.coalesced_extents,
        "direct backend ops should be coalesced physical reads");
    check(direct_stats.backend_read_bytes >= direct_stats.read_bytes,
        "direct backend bytes should include physical alignment padding");
#endif
}

void test_family_remap_reader_filters_by_bucket_key() {
    TempDir tmp;
    const BCLut &lut = shared_test_lut();
    const std::vector<BC::LayerSum> possible_8tile_sums{
        0U, 2U, 4U, 6U, 8U
    };
    const BCFamilyTable physical_axis = BCFamilyTable::from_range(8U, 2U, 0U, 0U);
    const BCFamilyTable logical_axis = BCFamilyTable::from_range(8U, 2U, 0U, 2U);
    const BCCellMatrix physical_matrix(physical_axis);
    const BCCellMatrix logical_matrix(logical_axis);

    const uint16_t sum0 = find_word_with_sum(lut, 0U);
    const uint16_t sum2 = find_word_with_sum(lut, 2U);
    const uint16_t sum4 = find_word_with_sum(lut, 4U);
    const uint16_t sum6 = find_word_with_sum(lut, 6U);
    const uint16_t sum8 = find_word_with_sum(lut, 8U);
    const BCEncodedKeyRank coord00 =
        BC::encode_key_and_rank(lut, sum0, sum0, sum0, sum8);
    const BCEncodedKeyRank coord11 =
        BC::encode_key_and_rank(lut, sum2, sum0, sum0, sum6);
    const BCEncodedKeyRank coord22 =
        BC::encode_key_and_rank(lut, sum4, sum0, sum0, sum4);
    check(coord00.valid && coord11.valid && coord22.valid, "remap encoded fixture should be valid");

    const CellFixture mixed = build_cell(
        lut,
        physical_matrix.cid(0U, 0U),
        {coord00, coord11, coord22}
    );
    const std::vector<uint8_t> bytes = write_synthetic_layer(physical_axis, {mixed});
    const std::filesystem::path path = tmp.path / "mixed_old_modulus.bcpos";
    BC::write_position_layer_to_file(path, bytes);
    BCPositionStreamingReader source = BCPositionStreamingReader::open_buffered(path, lut);
    BC::BCPositionFamilyRemapReader remap(source, logical_axis, possible_8tile_sums);
    check(!remap.direct(), "different modulus should use remap path");

    std::vector<CellId> cids{
        logical_matrix.cid(0U, 0U),
        logical_matrix.cid(1U, 1U),
        logical_matrix.cid(2U, 2U),
        logical_matrix.cid(0U, 1U)
    };
    BC::BCCellLoadStats stats;
    std::vector<BC::BCLoadedCell> cells;
    remap.load_cells_into(cids, cells, &stats);
    check(cells.size() == cids.size(), "remap load should preserve requested cell count");
    check(remap.has_success_rows(std::vector<CellId>{cids[0]}), "remap should see target success rows");

    const std::array<BCEncodedKeyRank, 3U> expected{coord00, coord11, coord22};
    for (size_t i = 0U; i < expected.size(); ++i) {
        check(cells[i].cid == cids[i], "remapped cell cid mismatch");
        check(cells[i].success_rows == 1U, "remapped target cell should keep one row");
        check(cells[i].buckets.size() == 1U, "remapped target cell should keep one bucket");
        const BCLookupResult lookup = cells[i].lookup(lut, expected[i].key, expected[i].rank);
        check(lookup.found, "remapped target lookup should find expected row");
        for (size_t j = 0U; j < expected.size(); ++j) {
            if (i == j) {
                continue;
            }
            const BCLookupResult other = cells[i].lookup(lut, expected[j].key, expected[j].rank);
            check(!other.found, "remapped target cell should not keep another logical cell row");
        }
    }
    check(cells[3].cid == cids[3], "remapped empty cell cid mismatch");
    check(cells[3].success_rows == 0U, "non-target remapped cell should be empty");
    check(cells[3].buckets.empty(), "non-target remapped cell should keep no buckets");
    check(stats.requested_extents != 0U, "remap should read old physical cells");
    const BC::BCPositionFamilyRemapStats &remap_stats = remap.stats();
    check(remap_stats.physical_cells_loaded >= 4U, "remap should load over-approx physical cells per logical cell");
    check(remap_stats.physical_bytes_loaded > 0U, "remap should account physical bytes loaded");
    check(remap_stats.remapped_bytes_kept > 0U, "remap should account kept bytes");
    check(
        remap_stats.discarded_bytes <= remap_stats.physical_bytes_loaded,
        "remap discarded byte count should be bounded by physical bytes"
    );
}

void test_cell_extents_and_error_paths() {
    FixtureLayer fixture;
    const CellId empty_cid = fixture.matrix.cid(0U, 0U);
    check(fixture.streaming_reader.cell_extents(empty_cid).empty(), "empty cell extents should be empty");
    check(fixture.streaming_reader.cell_extents(fixture.fixtures.front().cid).size() == 2U,
        "non-empty cell should expose bucket and rank extents");
    expect_throws(
        [&]() {
            (void)fixture.streaming_reader.load_cell(fixture.streaming_reader.cell_count());
        },
        "loading out-of-range cell should throw"
    );

    const std::filesystem::path truncated_path = fixture.tmp.path / "truncated.bcpos";
    std::vector<uint8_t> truncated = fixture.bytes;
    truncated.pop_back();
    BC::write_position_layer_to_file(truncated_path, truncated);
    expect_throws(
        [&]() {
            (void)BCPositionStreamingReader::open_buffered(truncated_path, fixture.lut);
        },
        "opening truncated streaming position file should throw"
    );
}

} // namespace

int main() {
    try {
        std::cerr << "test_open_reads_header_and_descriptors_only\n";
        test_open_reads_header_and_descriptors_only();
        std::cerr << "test_load_empty_and_single_cell\n";
        test_load_empty_and_single_cell();
        std::cerr << "test_loaded_cell_word_range_scanner_uses_prefix\n";
        test_loaded_cell_word_range_scanner_uses_prefix();
        std::cerr << "test_load_cells_coalesces_and_preserves_order\n";
        test_load_cells_coalesces_and_preserves_order();
        std::cerr << "test_direct_streaming_reader_matches_buffered\n";
        test_direct_streaming_reader_matches_buffered();
        std::cerr << "test_family_remap_reader_filters_by_bucket_key\n";
        test_family_remap_reader_filters_by_bucket_key();
        std::cerr << "test_cell_extents_and_error_paths\n";
        test_cell_extents_and_error_paths();
    } catch (const std::exception &ex) {
        std::cerr << "bc_position_cell_loader_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_position_cell_loader_test passed\n";
    return 0;
}
