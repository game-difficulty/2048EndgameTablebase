#include "BCFileIO.h"
#include "BCDirectFileIO.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
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
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BucketRank;
using BC::CellId;
using BC::FinalizedCellPayload;

uint64_t g_last_position_file_size = 0U;
uint64_t g_last_success_file_size = 0U;

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
            ("bc_file_io_test_" + std::to_string(static_cast<long long>(stamp)));
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

struct SyntheticPosition {
    const BCLut &lut;
    BCFamilyTable axis;
    BCCellMatrix matrix;
    std::vector<CellFixture> fixtures;
    std::vector<uint8_t> bytes;
    BCPositionLayerReader reader;

    SyntheticPosition()
        : lut(shared_test_lut()),
          axis(BCFamilyTable::from_range(6U, 1U, 0U, 3U)),
          matrix(axis) {
        const std::vector<uint16_t> words = collect_valid_words(lut);
        const GroupChoice group = find_group_with_count(lut, 4U);
        fixtures = {
            build_cell(
                lut,
                matrix.cid(0U, 1U),
                {
                    encode_from_group(lut, words.front(), group, 0U, 0U, 0U),
                    encode_from_group(lut, words.front(), group, 0U, 0U, 1U),
                    encode_from_group(lut, words.front(), group, 1U, 0U, 0U),
                }
            ),
            build_cell(
                lut,
                matrix.cid(2U, 3U),
                {
                    encode_from_group(lut, words[3], group, 0U, 0U, 0U),
                    encode_from_group(lut, words[7], group, 0U, 1U, 0U),
                }
            ),
        };

        std::map<CellId, const CellFixture *> by_cid;
        for (const CellFixture &fixture : fixtures) {
            by_cid[fixture.cid] = &fixture;
        }

        BCPositionLayerWriter writer;
        writer.begin_layer(axis);
        for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
            const auto it = by_cid.find(cid);
            if (it == by_cid.end()) {
                writer.mark_empty_cell(cid);
            } else {
                writer.write_cell(cid, it->second->payload);
            }
        }
        bytes = writer.finish_layer();
        reader.open(bytes, lut);
    }
};

std::vector<uint32_t> expected_success_values(CellId cid, uint32_t rows, uint32_t row_width) {
    std::vector<uint32_t> values;
    values.reserve(static_cast<size_t>(rows) * row_width);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t lane = 0; lane < row_width; ++lane) {
            values.push_back(static_cast<uint32_t>(cid) * 1000000U + row * 10U + lane);
        }
    }
    return values;
}

std::vector<uint8_t> write_success_bytes(
    const BCPositionLayerReader &position,
    uint32_t row_width
) {
    BCSuccessLayerWriter writer;
    writer.begin_layer(position, row_width);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, expected_success_values(cid, desc.success_rows, row_width));
    }
    return writer.finish_layer();
}

void check_position_reader_matches_fixture(
    const BCLut &lut,
    const BCPositionLayerReader &reader,
    const std::vector<CellFixture> &fixtures
) {
    for (const CellFixture &fixture : fixtures) {
        check(
            reader.descriptor(fixture.cid).success_rows == fixture.payload.success_rows,
            "file position descriptor success_rows mismatch"
        );
        for (const auto &[key, rank] : fixture.oracle) {
            const BCLookupResult expected = fixture.payload.lookup(lut, key, rank);
            const BCLookupResult got = reader.cold_lookup(fixture.cid, key, rank);
            check(expected.found && got.found, "file position cold lookup missed inserted rank");
            check(
                expected.local_success_row == got.local_success_row,
                "file position cold lookup local row mismatch"
            );
        }
    }
}

void verify_success_reader(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &success,
    uint32_t row_width
) {
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        const std::vector<uint32_t> expected =
            expected_success_values(cid, desc.success_rows, row_width);
        check(success.read_cell(cid) == expected, "file success read_cell mismatch");
        for (uint32_t row = 0U; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0U; lane < row_width; ++lane) {
                check(
                    success.read_value(cid, row, lane) == expected[row * row_width + lane],
                    "file success read_value mismatch"
                );
            }
        }
    }
}

void test_buffered_positioned_io() {
    TempDir tmp;
    const std::filesystem::path path = tmp.path / "positioned.bin";
    const std::array<uint8_t, 4U> first = {1U, 2U, 3U, 4U};
    const std::array<uint8_t, 4U> second = {5U, 6U, 7U, 8U};

    {
        BC::BCBufferedFileWriter writer(path);
        writer.resize(8192U);
        writer.write_at(4096U, second.data(), second.size());
        writer.write_at(0U, first.data(), first.size());
        writer.resize(4100U);
        writer.flush();
    }

    check(std::filesystem::file_size(path) == 4100U, "buffered file resize size mismatch");
    BC::BCBufferedFileReader reader(path);
    check(reader.size() == 4100U, "buffered file reader size mismatch");

    std::array<uint8_t, 4U> got_first = {};
    std::array<uint8_t, 4U> got_second = {};
    reader.read_at(0U, got_first.data(), got_first.size());
    reader.read_at(4096U, got_second.data(), got_second.size());
    check(got_first == first, "buffered read_at offset 0 mismatch");
    check(got_second == second, "buffered read_at non-sequential offset mismatch");
    expect_throws(
        [&]() {
            uint8_t byte = 0U;
            reader.read_at(4100U, &byte, 1U);
        },
        "buffered read beyond EOF should throw"
    );
}

void test_buffered_batch_io() {
    TempDir tmp;
    const std::filesystem::path path = tmp.path / "batch.bin";
    const std::array<uint8_t, 4U> first = {11U, 12U, 13U, 14U};
    const std::array<uint8_t, 4U> second = {21U, 22U, 23U, 24U};
    const std::array<uint8_t, 4U> third = {31U, 32U, 33U, 34U};

    {
        BC::BCBufferedFileWriter writer(path);
        writer.resize(12292U);
        BC::BCFileIOStats stats;
        writer.write_many(
            {
                BC::BCFileWriteRequest{8192U, second.data(), second.size()},
                BC::BCFileWriteRequest{0U, first.data(), first.size()},
                BC::BCFileWriteRequest{12288U, third.data(), third.size()},
            },
            &stats
        );
        writer.flush();
        check(stats.request_count == 3U, "buffered write_many request count mismatch");
        check(stats.requested_bytes == 12U, "buffered write_many requested bytes mismatch");
        check(stats.backend_io_count == 3U, "buffered write_many backend count mismatch");
        check(stats.backend_bytes == 12U, "buffered write_many backend bytes mismatch");
    }

    BC::BCBufferedFileReader reader(path);
    std::array<uint8_t, 4U> got_first = {};
    std::array<uint8_t, 4U> got_second = {};
    std::array<uint8_t, 4U> got_third = {};
    BC::BCFileIOStats stats;
    reader.read_many(
        {
            BC::BCFileReadRequest{12288U, got_third.data(), got_third.size()},
            BC::BCFileReadRequest{0U, got_first.data(), got_first.size()},
            BC::BCFileReadRequest{8192U, got_second.data(), got_second.size()},
        },
        &stats
    );
    check(got_first == first, "buffered read_many first mismatch");
    check(got_second == second, "buffered read_many second mismatch");
    check(got_third == third, "buffered read_many third mismatch");
    check(stats.request_count == 3U, "buffered read_many request count mismatch");
    check(stats.requested_bytes == 12U, "buffered read_many requested bytes mismatch");
    check(stats.backend_io_count == 3U, "buffered read_many backend count mismatch");
    check(stats.backend_bytes == 12U, "buffered read_many backend bytes mismatch");
}

void test_direct_batch_io() {
#if defined(_WIN32) || defined(__linux__)
    if (!direct_io_tests_enabled()) {
        std::cerr << "skip direct file IO batch test; set BC_RUN_DIRECT_IO_TESTS=1 to enable\n";
        return;
    }
    TempDir tmp;
    const std::filesystem::path path = tmp.path / "direct_batch.bin";
    std::vector<uint8_t> expected(9000U, 0U);
    const std::array<uint8_t, 7U> first = {1U, 2U, 3U, 4U, 5U, 6U, 7U};
    const std::array<uint8_t, 9U> second = {21U, 22U, 23U, 24U, 25U, 26U, 27U, 28U, 29U};
    const std::array<uint8_t, 11U> third = {41U, 42U, 43U, 44U, 45U, 46U, 47U, 48U, 49U, 50U, 51U};
    std::copy(first.begin(), first.end(), expected.begin() + 3U);
    std::copy(second.begin(), second.end(), expected.begin() + 4093U);
    std::copy(third.begin(), third.end(), expected.begin() + 8188U);

    {
        BC::BCDirectFileWriter writer(path);
        writer.resize(expected.size());
        BC::BCFileIOStats stats;
        writer.write_many(
            {
                BC::BCFileWriteRequest{4093U, second.data(), second.size()},
                BC::BCFileWriteRequest{3U, first.data(), first.size()},
                BC::BCFileWriteRequest{8188U, third.data(), third.size()},
            },
            &stats
        );
        writer.flush();
        check(stats.request_count == 3U, "direct write_many request count mismatch");
        check(stats.requested_bytes == first.size() + second.size() + third.size(),
            "direct write_many requested bytes mismatch");
        check(stats.backend_io_count >= 1U, "direct write_many should report backend ops");
        check(stats.backend_bytes >= 4096U, "direct write_many should report aligned backend bytes");
    }

    const uint64_t physical_size = std::filesystem::file_size(path);
    check((physical_size % 4096U) == 0U, "direct writer physical size should be aligned");
    check(physical_size >= expected.size(), "direct writer physical size should cover logical bytes");

    BC::BCDirectFileIOOptions read_options;
    read_options.logical_size = expected.size();
    read_options.overlapped = false;
    read_options.queue_depth = 2U;
    BC::BCDirectFileReader reader(path, read_options);
    check(reader.size() == expected.size(), "direct reader should report logical size");
    std::vector<uint8_t> actual(expected.size(), 0U);
    BC::BCFileIOStats stats;
    reader.read_many(
        {
            BC::BCFileReadRequest{8188U, actual.data() + 8188U, third.size()},
            BC::BCFileReadRequest{0U, actual.data(), 16U},
            BC::BCFileReadRequest{4088U, actual.data() + 4088U, 24U},
        },
        &stats
    );
    for (uint32_t i = 0U; i < 16U; ++i) {
        check(actual[i] == expected[i], "direct read_many first range mismatch");
    }
    for (uint32_t i = 4088U; i < 4112U; ++i) {
        check(actual[i] == expected[i], "direct read_many cross-boundary range mismatch");
    }
    for (uint32_t i = 8188U; i < 8199U; ++i) {
        check(actual[i] == expected[i], "direct read_many tail range mismatch");
    }
    check(stats.request_count == 3U, "direct read_many request count mismatch");
    check(stats.requested_bytes == 16U + 24U + third.size(), "direct read_many requested bytes mismatch");
    check(stats.backend_io_count > 0U, "direct read_many backend ops should be reported");
    check(stats.backend_bytes >= 4096U, "direct read_many backend bytes should be aligned");
#endif
}

void test_windows_direct_io_chunk_limits() {
#if defined(_WIN32)
    constexpr uint32_t kAlignment = 4096U;
    const uint64_t max_chunk = BC::detail::windows_max_io_chunk_bytes(kAlignment);
    check(max_chunk != 0U, "Windows direct max chunk should be non-zero");
    check(max_chunk <= 0xFFFFFFFFULL, "Windows direct max chunk should fit DWORD");
    check((max_chunk % kAlignment) == 0U, "Windows direct max chunk should be aligned");
    check(
        BC::detail::windows_io_chunk_count(max_chunk + kAlignment, kAlignment) == 2U,
        "Windows direct chunk count should split over-DWORD requests"
    );

    std::vector<BC::detail::BCPhysicalRange> ranges;
    ranges.push_back(BC::detail::BCPhysicalRange{0U, max_chunk + 2U * kAlignment, {}});
    BC::detail::split_physical_ranges_for_windows_io(ranges, kAlignment);
    check(ranges.size() == 2U, "Windows direct physical range should split");
    check(ranges[0].bytes == max_chunk, "Windows direct first split chunk mismatch");
    check(ranges[1].bytes == 2U * kAlignment, "Windows direct second split chunk mismatch");
    for (const BC::detail::BCPhysicalRange &range : ranges) {
        check(range.bytes <= max_chunk, "Windows direct split chunk exceeds max");
        check((range.bytes % kAlignment) == 0U, "Windows direct split chunk should be aligned");
    }
#endif
}

void test_position_file_path_roundtrip() {
    TempDir tmp;
    SyntheticPosition position;
    const std::filesystem::path path = tmp.path / "layer.bcpos";

    BC::write_position_layer_to_file(path, position.bytes);
    g_last_position_file_size = std::filesystem::file_size(path);
    check(g_last_position_file_size == position.bytes.size(), "position file size mismatch");

    const std::vector<uint8_t> read_bytes = BC::read_position_layer_from_file(path);
    check(read_bytes == position.bytes, "position file read bytes mismatch");

    BCPositionLayerReader reader(read_bytes, position.lut);
    check_position_reader_matches_fixture(position.lut, reader, position.fixtures);

    const BC::BCPositionFileReader file_reader =
        BC::BCPositionFileReader::open_buffered(path, position.lut);
    check(file_reader.bytes() == position.bytes, "position file reader bytes mismatch");
    check_position_reader_matches_fixture(position.lut, file_reader.layer(), position.fixtures);
}

void test_success_file_path_roundtrip() {
    TempDir tmp;
    SyntheticPosition position;
    const uint32_t row_width = 3U;
    const std::vector<uint8_t> success_bytes = write_success_bytes(position.reader, row_width);
    const std::filesystem::path position_path = tmp.path / "layer.bcpos";
    const std::filesystem::path success_path = tmp.path / "layer.bcsuc";

    BC::write_position_layer_to_file(position_path, position.bytes);
    const std::vector<uint8_t> position_bytes = BC::read_position_layer_from_file(position_path);
    BCPositionLayerReader position_reader(position_bytes, position.lut);

    BC::write_success_layer_to_file(success_path, success_bytes);
    g_last_success_file_size = std::filesystem::file_size(success_path);
    check(g_last_success_file_size == success_bytes.size(), "success file size mismatch");

    const std::vector<uint8_t> read_success = BC::read_success_layer_from_file(success_path);
    check(read_success == success_bytes, "success file read bytes mismatch");

    BCSuccessLayerReader success_reader(read_success, position_reader, row_width);
    verify_success_reader(position_reader, success_reader, row_width);

    const BC::BCSuccessFileReader file_reader =
        BC::BCSuccessFileReader::open_buffered(success_path, position_reader, row_width);
    check(file_reader.bytes() == success_bytes, "success file reader bytes mismatch");
    verify_success_reader(position_reader, file_reader.reader(), row_width);
}

} // namespace

int main() {
    try {
        std::cerr << "test_buffered_positioned_io\n";
        test_buffered_positioned_io();
        std::cerr << "test_buffered_batch_io\n";
        test_buffered_batch_io();
        std::cerr << "test_direct_batch_io\n";
        test_direct_batch_io();
        std::cerr << "test_windows_direct_io_chunk_limits\n";
        test_windows_direct_io_chunk_limits();
        std::cerr << "test_position_file_path_roundtrip\n";
        test_position_file_path_roundtrip();
        std::cerr << "test_success_file_path_roundtrip\n";
        test_success_file_path_roundtrip();
    } catch (const std::exception &ex) {
        std::cerr << "bc_file_io_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout
        << "bc_file_io_test passed"
        << " position_file_bytes=" << g_last_position_file_size
        << " success_file_bytes=" << g_last_success_file_size
        << "\n";
    return 0;
}
