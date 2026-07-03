#include "BCDirectFileIO.h"
#include "BCSuccessIO.h"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
using BC::BCSuccessStreamingReader;
using BC::BCSuccessDTypeMode;
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
            ("bc_success_io_test_" + std::to_string(static_cast<long long>(stamp)));
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
};

CellFixture build_cell(
    const BCLut &lut,
    CellId cid,
    const std::vector<BCEncodedKeyRank> &items
) {
    BCCellBuilder builder(lut);
    for (const BCEncodedKeyRank &item : items) {
        check(item.valid, "encoded fixture item should be valid");
        builder.insert(item.key, item.rank);
    }
    return CellFixture{cid, builder.finalize()};
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

struct SyntheticLayer {
    const BCLut &lut;
    BCFamilyTable axis;
    std::vector<CellFixture> fixtures;
    std::vector<uint8_t> position_bytes;
    BCPositionLayerReader position;

    SyntheticLayer()
        : lut(shared_test_lut()),
          axis(BCFamilyTable::from_range(6U, 1U, 0U, 3U)) {
        const BCCellMatrix matrix(axis);
        fixtures = make_cell_fixtures(lut, matrix);
        position_bytes = write_position_layer(axis, fixtures);
        position.open(position_bytes, lut);
    }
};

std::vector<uint32_t> expected_values(CellId cid, uint32_t success_rows, uint32_t row_width) {
    std::vector<uint32_t> values;
    values.reserve(static_cast<size_t>(success_rows) * row_width);
    for (uint32_t row = 0; row < success_rows; ++row) {
        for (uint32_t lane = 0; lane < row_width; ++lane) {
            if (row_width == 1U) {
                values.push_back(static_cast<uint32_t>(cid) * 100000U + row);
            } else {
                values.push_back(static_cast<uint32_t>(cid) * 1000000U + row * 10U + lane);
            }
        }
    }
    return values;
}

template <typename T>
std::vector<T> expected_typed_values(
    CellId cid,
    uint32_t success_rows,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    std::vector<T> values;
    values.reserve(static_cast<size_t>(success_rows) * row_width);
    for (uint32_t row = 0; row < success_rows; ++row) {
        for (uint32_t lane = 0; lane < row_width; ++lane) {
            const uint32_t seed = static_cast<uint32_t>(cid) * 1000U + row * 10U + lane;
            if constexpr (std::is_same_v<T, uint32_t>) {
                values.push_back(seed);
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                values.push_back(static_cast<uint64_t>(seed) * 1000000000ULL + 17ULL);
            } else if constexpr (std::is_same_v<T, float>) {
                const float unit = static_cast<float>((seed % 97U) + 1U) / 128.0f;
                if (dtype == BCSuccessDTypeMode::OneMinusFloat32) {
                    values.push_back(unit - 1.0f);
                } else {
                    values.push_back(unit);
                }
            } else {
                const double unit = static_cast<double>((seed % 193U) + 1U) / 256.0;
                if (dtype == BCSuccessDTypeMode::OneMinusFloat64) {
                    values.push_back(unit - 1.0);
                } else {
                    values.push_back(unit);
                }
            }
        }
    }
    return values;
}

std::vector<uint8_t> write_success_layer(
    const BCPositionLayerReader &position,
    uint32_t row_width
) {
    BCSuccessLayerWriter writer;
    writer.begin_layer(position, row_width);
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, expected_values(cid, desc.success_rows, row_width));
    }
    return writer.finish_layer();
}

template <typename T>
std::vector<uint8_t> write_success_layer_typed(
    const BCPositionLayerReader &position,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    BCSuccessLayerWriter writer;
    writer.begin_layer(position, row_width, dtype);
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell_typed<T>(
            cid,
            expected_typed_values<T>(cid, desc.success_rows, row_width, dtype)
        );
    }
    return writer.finish_layer();
}

void expect_reader_failure(
    const std::vector<uint8_t> &bytes,
    const BCPositionLayerReader &position,
    uint32_t expected_row_width,
    const char *message
) {
    bool threw = false;
    try {
        BCSuccessLayerReader reader(bytes, position, expected_row_width);
        (void)reader;
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}

void expect_writer_failure(const char *message, const std::function<void()> &fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
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

CellId first_non_empty_cell(const BCPositionLayerReader &position) {
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        if (position.descriptor(cid).success_rows != 0U) {
            return cid;
        }
    }
    throw std::runtime_error("synthetic position has no non-empty cells");
}

CellId first_empty_cell(const BCPositionLayerReader &position) {
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        if (position.descriptor(cid).success_rows == 0U) {
            return cid;
        }
    }
    throw std::runtime_error("synthetic position has no empty cells");
}

void verify_success_reader(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &reader,
    uint32_t row_width
) {
    check(reader.row_width() == row_width, "success reader row_width mismatch");
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        const std::vector<uint32_t> got = reader.read_cell(cid);
        const std::vector<uint32_t> expected =
            expected_values(cid, desc.success_rows, row_width);
        check(got == expected, "success read_cell values mismatch");
        if (desc.success_rows == 0U) {
            check(got.empty(), "empty success cell should return empty vector");
            continue;
        }
        for (uint32_t row = 0; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0; lane < row_width; ++lane) {
                check(
                    reader.read_value(cid, row, lane) == expected[row * row_width + lane],
                    "success read_value mismatch"
                );
            }
        }
        bool row_threw = false;
        try {
            (void)reader.read_value(cid, desc.success_rows, 0U);
        } catch (const std::out_of_range &) {
            row_threw = true;
        }
        check(row_threw, "success read_value row out of range should throw");

        bool lane_threw = false;
        try {
            (void)reader.read_value(cid, 0U, row_width);
        } catch (const std::out_of_range &) {
            lane_threw = true;
        }
        check(lane_threw, "success read_value lane out of range should throw");
    }
}

void verify_success_streaming_reader(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &memory_reader,
    const BCSuccessStreamingReader &streaming_reader,
    uint32_t row_width
) {
    check(streaming_reader.row_width() == row_width, "success streaming row_width mismatch");
    check(streaming_reader.cell_count() == position.cell_count(), "success streaming cell_count mismatch");
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        const BC::BCLoadedSuccessCell cell = streaming_reader.load_cell(cid);
        check(cell.cid == cid, "success streaming loaded cell cid mismatch");
        check(cell.row_width == row_width, "success streaming loaded cell row_width mismatch");
        check(cell.success_rows == desc.success_rows, "success streaming loaded cell row count mismatch");
        check(cell.values == memory_reader.read_cell(cid), "success streaming load_cell values mismatch");
        if (desc.success_rows == 0U) {
            check(cell.values.empty(), "success streaming empty cell values should be empty");
            continue;
        }
        for (uint32_t row = 0; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0; lane < row_width; ++lane) {
                const uint32_t expected = memory_reader.read_value(cid, row, lane);
                check(cell.read_value(row, lane) == expected, "loaded success cell read_value mismatch");
                check(streaming_reader.read_value(cid, row, lane) == expected,
                    "success streaming read_value mismatch");
            }
        }
        bool row_threw = false;
        try {
            (void)streaming_reader.read_value(cid, desc.success_rows, 0U);
        } catch (const std::out_of_range &) {
            row_threw = true;
        }
        check(row_threw, "success streaming read_value row out of range should throw");
    }
}

template <typename T>
void verify_success_reader_typed(
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &reader,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    check(reader.row_width() == row_width, "typed success reader row_width mismatch");
    check(reader.dtype_mode() == dtype, "typed success reader dtype mismatch");
    check(reader.header().payload_bytes ==
        BC::bc_success_expected_payload_bytes(position, row_width, dtype),
        "typed success reader payload byte count mismatch");
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        const std::vector<T> got = reader.read_cell_typed<T>(cid);
        const std::vector<T> expected =
            expected_typed_values<T>(cid, desc.success_rows, row_width, dtype);
        check(got == expected, "typed success read_cell values mismatch");
        const std::vector<uint8_t> raw = reader.read_cell_raw(cid);
        check(raw.size() == expected.size() * sizeof(T), "typed success raw cell byte count mismatch");
        if (desc.success_rows == 0U) {
            check(got.empty(), "typed empty success cell should return empty vector");
            continue;
        }
        for (uint32_t row = 0; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0; lane < row_width; ++lane) {
                check(
                    reader.read_value_typed<T>(cid, row, lane) == expected[row * row_width + lane],
                    "typed success read_value mismatch"
                );
            }
        }
    }
}

template <typename T>
void verify_success_streaming_reader_typed(
    const BCPositionLayerReader &position,
    const BCSuccessStreamingReader &streaming_reader,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    check(streaming_reader.row_width() == row_width, "typed streaming row_width mismatch");
    check(streaming_reader.dtype_mode() == dtype, "typed streaming dtype mismatch");
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        const BC::BCLoadedSuccessCell cell = streaming_reader.load_cell(cid);
        const std::vector<T> expected =
            expected_typed_values<T>(cid, desc.success_rows, row_width, dtype);
        check(cell.cid == cid, "typed streaming loaded cell cid mismatch");
        check(cell.dtype_mode() == dtype, "typed streaming loaded cell dtype mismatch");
        check(cell.raw_bytes.size() == expected.size() * sizeof(T),
            "typed streaming loaded raw byte count mismatch");
        if constexpr (!std::is_same_v<T, uint32_t>) {
            check(cell.values.empty(), "non-u32 streaming cell should not populate u32 values");
        }
        if (desc.success_rows == 0U) {
            continue;
        }
        for (uint32_t row = 0; row < desc.success_rows; ++row) {
            for (uint32_t lane = 0; lane < row_width; ++lane) {
                const T expected_value = expected[row * row_width + lane];
                check(cell.read_value_typed<T>(row, lane) == expected_value,
                    "typed loaded success cell read_value mismatch");
                check(streaming_reader.read_value_typed<T>(cid, row, lane) == expected_value,
                    "typed streaming read_value mismatch");
            }
        }
    }
}

void test_success_roundtrip_row_width_1() {
    SyntheticLayer layer;
    const uint32_t row_width = 1U;
    const std::vector<uint8_t> success_bytes = write_success_layer(layer.position, row_width);
    BCSuccessLayerReader reader(success_bytes, layer.position, row_width);
    verify_success_reader(layer.position, reader, row_width);
}

void test_success_roundtrip_row_width_3() {
    SyntheticLayer layer;
    const uint32_t row_width = 3U;
    const std::vector<uint8_t> success_bytes = write_success_layer(layer.position, row_width);
    BCSuccessLayerReader reader(success_bytes, layer.position, row_width);
    verify_success_reader(layer.position, reader, row_width);
}

template <typename T>
void run_typed_success_roundtrip(
    BCSuccessDTypeMode dtype,
    const char *file_stem,
    bool verify_streaming
) {
    SyntheticLayer layer;
    const uint32_t row_width = 2U;
    const std::vector<uint8_t> success_bytes =
        write_success_layer_typed<T>(layer.position, row_width, dtype);
    BCSuccessLayerReader reader(success_bytes, layer.position, row_width);
    verify_success_reader_typed<T>(layer.position, reader, row_width, dtype);

    if (!verify_streaming) {
        return;
    }

    TempDir tmp;
    const std::filesystem::path success_path =
        tmp.path / (std::string(file_stem) + ".bcsuc");
    BC::write_success_layer_to_file(success_path, success_bytes);
    const BCSuccessStreamingReader streaming_reader =
        BCSuccessStreamingReader::open_buffered(success_path, layer.position, row_width);
    verify_success_streaming_reader_typed<T>(layer.position, streaming_reader, row_width, dtype);

    const BCCellMatrix matrix(layer.axis);
    const std::vector<CellId> cids = {
        matrix.cid(3U, 3U),
        matrix.cid(0U, 1U),
        matrix.cid(1U, 2U),
    };
    BC::BCSuccessLoadStats stats;
    const std::vector<BC::BCLoadedSuccessCell> cells = streaming_reader.load_cells(cids, &stats);
    check(cells.size() == cids.size(), "typed streaming batch result size mismatch");
    check(stats.requested_extents == 3U, "typed streaming batch should request three extents");
    check(stats.requested_bytes > 0U, "typed streaming batch should request bytes");
    for (size_t i = 0U; i < cids.size(); ++i) {
        const auto &desc = layer.position.descriptor(cids[i]);
        const std::vector<T> expected =
            expected_typed_values<T>(cids[i], desc.success_rows, row_width, dtype);
        check(cells[i].raw_bytes.size() == expected.size() * sizeof(T),
            "typed streaming batch raw byte count mismatch");
    }
}

void test_success_dtype_roundtrips() {
    run_typed_success_roundtrip<uint32_t>(BCSuccessDTypeMode::UInt32, "u32", false);
    run_typed_success_roundtrip<uint64_t>(BCSuccessDTypeMode::UInt64, "u64", true);
    run_typed_success_roundtrip<float>(BCSuccessDTypeMode::Float32, "f32", false);
    run_typed_success_roundtrip<double>(BCSuccessDTypeMode::Float64, "f64", false);
    run_typed_success_roundtrip<float>(BCSuccessDTypeMode::OneMinusFloat32, "omf32", true);
    run_typed_success_roundtrip<double>(BCSuccessDTypeMode::OneMinusFloat64, "omf64", false);
}

void test_success_streaming_reader() {
    SyntheticLayer layer;
    const uint32_t row_width = 3U;
    const std::vector<uint8_t> success_bytes = write_success_layer(layer.position, row_width);
    const BCSuccessLayerReader memory_reader(success_bytes, layer.position, row_width);
    TempDir tmp;
    const std::filesystem::path position_path = tmp.path / "layer.bcpos";
    const std::filesystem::path success_path = tmp.path / "layer.bcsuc";
    BC::write_position_layer_to_file(position_path, layer.position_bytes);
    BC::write_success_layer_to_file(success_path, success_bytes);

    const BCSuccessStreamingReader streaming_reader =
        BCSuccessStreamingReader::open_buffered(success_path, layer.position, row_width);
    verify_success_streaming_reader(layer.position, memory_reader, streaming_reader, row_width);

    const BCPositionStreamingReader streaming_position =
        BCPositionStreamingReader::open_buffered(position_path, layer.lut);
    const BCSuccessStreamingReader streaming_reader_from_streaming_position =
        BCSuccessStreamingReader::open_buffered(success_path, streaming_position, row_width);
    verify_success_streaming_reader(
        layer.position,
        memory_reader,
        streaming_reader_from_streaming_position,
        row_width
    );

    const BCCellMatrix matrix(layer.axis);
    const std::vector<CellId> cids = {
        matrix.cid(3U, 3U),
        matrix.cid(0U, 0U),
        matrix.cid(0U, 1U),
        matrix.cid(1U, 2U),
    };
    BC::BCSuccessLoadStats stats;
    const std::vector<BC::BCLoadedSuccessCell> cells = streaming_reader.load_cells(cids, &stats);
    check(cells.size() == cids.size(), "success streaming load_cells result size mismatch");
    for (size_t i = 0U; i < cids.size(); ++i) {
        check(cells[i].cid == cids[i], "success streaming load_cells should preserve input order");
        check(cells[i].values == memory_reader.read_cell(cids[i]),
            "success streaming load_cells values mismatch");
    }
    check(cells[1].empty(), "success streaming empty loaded cell should be empty");
    check(stats.requested_extents == 3U, "success streaming batch should request three non-empty extents");
    check(stats.coalesced_extents < stats.requested_extents,
        "success streaming batch should coalesce adjacent payload extents");
    check(stats.read_bytes >= stats.requested_bytes,
        "success streaming batch read bytes should cover requested bytes");
    check(stats.backend_read_ops == stats.coalesced_extents,
        "success streaming buffered backend ops should match coalesced extents");
    check(stats.backend_read_bytes == stats.read_bytes,
        "success streaming buffered backend bytes should match coalesced bytes");

#if defined(_WIN32) || defined(__linux__)
    if (!direct_io_tests_enabled()) {
        std::cerr << "skip direct success streaming reader test; set BC_RUN_DIRECT_IO_TESTS=1 to enable\n";
        return;
    }
    const std::filesystem::path direct_success_path = tmp.path / "layer_direct.bcsuc";
    {
        BC::BCBufferedFileWriter writer(direct_success_path);
        writer.resize(BC::bc_direct_align_up(static_cast<uint64_t>(success_bytes.size()), 4096U));
        writer.write_at(0U, success_bytes.data(), static_cast<uint64_t>(success_bytes.size()));
        writer.flush();
    }
    BC::BCDirectFileIOOptions direct_options;
    direct_options.logical_size = success_bytes.size();
    direct_options.overlapped = false;
    const BCSuccessStreamingReader direct_success_reader(
        std::make_unique<BC::BCDirectFileReader>(direct_success_path, direct_options),
        layer.position,
        row_width
    );
    verify_success_streaming_reader(layer.position, memory_reader, direct_success_reader, row_width);
    BC::BCSuccessLoadStats direct_stats;
    const std::vector<BC::BCLoadedSuccessCell> direct_cells =
        direct_success_reader.load_cells(cids, &direct_stats);
    check(direct_cells.size() == cells.size(), "direct success streaming cell count mismatch");
    for (size_t i = 0U; i < direct_cells.size(); ++i) {
        check(direct_cells[i].values == cells[i].values, "direct success streaming values mismatch");
    }
    check(direct_stats.requested_extents == stats.requested_extents,
        "direct success streaming requested extents mismatch");
    check(direct_stats.coalesced_extents == stats.coalesced_extents,
        "direct success streaming coalesced extents mismatch");
    check(direct_stats.backend_read_ops <= direct_stats.coalesced_extents,
        "direct success streaming backend ops should be coalesced physical reads");
    check(direct_stats.backend_read_bytes >= direct_stats.read_bytes,
        "direct success streaming backend bytes should include alignment padding");
#endif
}

void test_success_header_errors() {
    SyntheticLayer layer;
    const std::vector<uint8_t> success_bytes = write_success_layer(layer.position, 1U);

    std::vector<uint8_t> bad_version = success_bytes;
    store_u32_le(bad_version, 4U, 999U);
    expect_reader_failure(bad_version, layer.position, 1U, "bad success version should fail");

    std::vector<uint8_t> bad_dtype = success_bytes;
    store_u32_le(bad_dtype, 12U, 999U);
    expect_reader_failure(bad_dtype, layer.position, 1U, "bad success dtype should fail");

    expect_reader_failure(success_bytes, layer.position, 2U, "row_width mismatch should fail");

    std::vector<uint8_t> bad_family_count = success_bytes;
    store_u32_le(bad_family_count, 20U, layer.position.header().family_count + 1U);
    expect_reader_failure(bad_family_count, layer.position, 1U, "family_count mismatch should fail");

    std::vector<uint8_t> bad_descriptor_count = success_bytes;
    store_u64_le(bad_descriptor_count, 24U, layer.position.cell_count() + 1U);
    expect_reader_failure(bad_descriptor_count, layer.position, 1U, "descriptor_count mismatch should fail");
}

void test_success_payload_errors() {
    SyntheticLayer layer;
    const std::vector<uint8_t> success_bytes = write_success_layer(layer.position, 1U);

    std::vector<uint8_t> missing = success_bytes;
    missing.pop_back();
    expect_reader_failure(missing, layer.position, 1U, "missing success payload should fail");

    std::vector<uint8_t> extra = success_bytes;
    extra.push_back(0U);
    expect_reader_failure(extra, layer.position, 1U, "extra success payload bytes should fail");
}

void test_success_writer_errors() {
    SyntheticLayer layer;
    const CellId non_empty = first_non_empty_cell(layer.position);
    const CellId empty = first_empty_cell(layer.position);
    const auto &non_empty_desc = layer.position.descriptor(non_empty);

    expect_writer_failure("non-empty short write should fail", [&]() {
        BCSuccessLayerWriter writer;
        writer.begin_layer(layer.position, 1U);
        writer.write_cell(non_empty, {});
    });

    expect_writer_failure("non-empty long write should fail", [&]() {
        BCSuccessLayerWriter writer;
        writer.begin_layer(layer.position, 1U);
        std::vector<uint32_t> values =
            expected_values(non_empty, non_empty_desc.success_rows, 1U);
        values.push_back(123U);
        writer.write_cell(non_empty, values);
    });

    expect_writer_failure("empty non-empty write should fail", [&]() {
        BCSuccessLayerWriter writer;
        writer.begin_layer(layer.position, 1U);
        writer.write_cell(empty, {1U});
    });

    expect_writer_failure("duplicate cell write should fail", [&]() {
        BCSuccessLayerWriter writer;
        writer.begin_layer(layer.position, 1U);
        writer.write_cell(non_empty, expected_values(non_empty, non_empty_desc.success_rows, 1U));
        writer.write_cell(non_empty, expected_values(non_empty, non_empty_desc.success_rows, 1U));
    });

    expect_writer_failure("finish with unwritten cell should fail", [&]() {
        BCSuccessLayerWriter writer;
        writer.begin_layer(layer.position, 1U);
        (void)writer.finish_layer();
    });
}

} // namespace

int main() {
    try {
        std::cerr << "test_success_roundtrip_row_width_1\n";
        test_success_roundtrip_row_width_1();
        std::cerr << "test_success_roundtrip_row_width_3\n";
        test_success_roundtrip_row_width_3();
        std::cerr << "test_success_dtype_roundtrips\n";
        test_success_dtype_roundtrips();
        std::cerr << "test_success_streaming_reader\n";
        test_success_streaming_reader();
        std::cerr << "test_success_header_errors\n";
        test_success_header_errors();
        std::cerr << "test_success_payload_errors\n";
        test_success_payload_errors();
        std::cerr << "test_success_writer_errors\n";
        test_success_writer_errors();
    } catch (const std::exception &ex) {
        std::cerr << "bc_success_io_test failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "bc_success_io_test passed\n";
    return 0;
}
