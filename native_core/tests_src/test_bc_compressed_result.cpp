#include "BCCompressedResult.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCEncodedKeyRank;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCSuccessDTypeMode;
using BC::BCSuccessLayerReader;
using BC::BCSuccessLayerWriter;
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
            ("bc_compressed_result_test_" + std::to_string(static_cast<long long>(stamp)));
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_bytes(const std::filesystem::path &path, const std::vector<uint8_t> &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open test file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
}

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U};
}

const BCLut &shared_lut() {
    static const BCLut *lut = new BCLut(test_alphabet());
    return *lut;
}

struct GroupChoice {
    uint16_t sum_id = 0U;
    uint8_t empty_mask = 0U;
    uint16_t count = 0U;
};

GroupChoice find_group_with_count(const BCLut &lut, uint16_t min_count) {
    GroupChoice best;
    for (uint32_t sum_id = 0U; sum_id < lut.sum_count(); ++sum_id) {
        for (uint32_t mask = 0U; mask < 16U; ++mask) {
            const uint16_t count =
                lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            if (count >= min_count && count > best.count) {
                best = GroupChoice{static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask), count};
            }
        }
    }
    check(best.count >= min_count, "failed to find test LUT group");
    return best;
}

uint16_t first_valid_word(const BCLut &lut) {
    for (uint32_t word = 0U; word < BC::kBCQuadrantWordCount; ++word) {
        if (lut.word_desc(static_cast<uint16_t>(word)).valid) {
            return static_cast<uint16_t>(word);
        }
    }
    throw std::runtime_error("no valid BC test word");
}

BCEncodedKeyRank encode_from_group(
    const BCLut &lut,
    uint16_t nw,
    const GroupChoice &group,
    BucketRank rank
) {
    const uint32_t count = group.count;
    const uint32_t rank_u32 = rank;
    const BucketRank ne_rank = static_cast<BucketRank>(rank_u32 / (count * count));
    const uint32_t rem = rank_u32 % (count * count);
    const BucketRank sw_rank = static_cast<BucketRank>(rem / count);
    const BucketRank se_rank = static_cast<BucketRank>(rem % count);
    return BC::encode_key_and_rank(
        lut,
        nw,
        lut.unrank_word(group.sum_id, group.empty_mask, ne_rank),
        lut.unrank_word(group.sum_id, group.empty_mask, sw_rank),
        lut.unrank_word(group.sum_id, group.empty_mask, se_rank));
}

struct QueryCase {
    uint64_t board = 0U;
    CellId cid = 0U;
    uint64_t key = 0U;
    BucketRank rank = 0U;
};

struct LayerFixture {
    const BCLut &lut;
    BCFamilyTable axis;
    std::map<CellId, std::vector<BCEncodedKeyRank>> items_by_cid;
    std::vector<QueryCase> queries;
    std::vector<uint8_t> position_bytes;
    BCPositionLayerReader position;

    LayerFixture()
        : lut(shared_lut()),
          axis(make_axis()) {
        build_items();
        position_bytes = write_position_layer();
        position.open(position_bytes, lut);
    }

    static BCFamilyTable make_axis() {
        const BCLut &lut = shared_lut();
        const uint16_t nw = first_valid_word(lut);
        const GroupChoice group = find_group_with_count(lut, 4U);
        const uint32_t layer_sum =
            lut.word_desc(nw).sum + 3U * lut.sum4_value(group.sum_id);
        return BCFamilyTable::from_range(static_cast<BC::LayerSum>(layer_sum), 1U, 0U,
                                         static_cast<BC::FamilyCoord>(layer_sum));
    }

    void build_items() {
        const uint16_t nw = first_valid_word(lut);
        const GroupChoice group = find_group_with_count(lut, 4U);
        const uint32_t bitmap_len =
            static_cast<uint32_t>(group.count) * group.count * group.count;
        const std::vector<BucketRank> ranks = {
            0U,
            1U,
            2U,
            static_cast<BucketRank>(bitmap_len - 1U)
        };
        for (BucketRank rank : ranks) {
            const BCEncodedKeyRank encoded = encode_from_group(lut, nw, group, rank);
            check(encoded.valid, "test encoded item invalid");
            const BC::BCBucketBoardDecoder decoder(lut, encoded.key);
            const uint64_t board = decoder.board(encoded.rank);
            const BC::BCBoardEncodedPosition pos =
                BC::encode_spawned_canonical_board(lut, axis, board);
            check(pos.valid, "test board should encode to position");
            check(pos.key == encoded.key && pos.rank == encoded.rank, "test key/rank roundtrip mismatch");
            items_by_cid[pos.cid].push_back(encoded);
            queries.push_back(QueryCase{board, pos.cid, pos.key, pos.rank});
        }
    }

    std::vector<uint8_t> write_position_layer() const {
        const BCCellMatrix matrix(axis);
        BCPositionLayerWriter writer;
        writer.begin_layer(axis);
        for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
            const auto it = items_by_cid.find(cid);
            if (it == items_by_cid.end()) {
                writer.mark_empty_cell(cid);
                continue;
            }
            BCCellBuilder builder(lut);
            for (const BCEncodedKeyRank &item : it->second) {
                builder.insert(item.key, item.rank);
            }
            writer.write_cell(cid, builder.finalize());
        }
        return writer.finish_layer();
    }
};

template <typename T>
T expected_value(CellId cid, uint32_t row, uint32_t lane, BCSuccessDTypeMode dtype) {
    const uint32_t seed = static_cast<uint32_t>(cid) * 1000U + row * 10U + lane;
    if constexpr (std::is_same_v<T, uint32_t>) {
        return seed;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return static_cast<uint64_t>(seed) * 1000000000ULL + 17ULL;
    } else if constexpr (std::is_same_v<T, float>) {
        const float unit = static_cast<float>((seed % 97U) + 1U) / 128.0f;
        return dtype == BCSuccessDTypeMode::OneMinusFloat32 ? unit - 1.0f : unit;
    } else {
        const double unit = static_cast<double>((seed % 193U) + 1U) / 256.0;
        return dtype == BCSuccessDTypeMode::OneMinusFloat64 ? unit - 1.0 : unit;
    }
}

template <typename T>
std::vector<T> expected_cell_values(
    CellId cid,
    uint32_t rows,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    std::vector<T> values;
    values.reserve(static_cast<size_t>(rows) * row_width);
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            values.push_back(expected_value<T>(cid, row, lane, dtype));
        }
    }
    return values;
}

template <typename T>
uint64_t raw_bits(T value) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return value;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return value;
    } else if constexpr (std::is_same_v<T, float>) {
        uint32_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(value));
        return bits;
    } else {
        uint64_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(value));
        return bits;
    }
}

template <typename T>
std::vector<uint8_t> write_success_layer(
    const BCPositionLayerReader &position,
    uint32_t row_width,
    BCSuccessDTypeMode dtype
) {
    BCSuccessLayerWriter writer;
    writer.begin_layer(position, row_width, dtype);
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const auto &desc = position.descriptor(cid);
        if (desc.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell_typed<T>(
            cid,
            expected_cell_values<T>(cid, desc.success_rows, row_width, dtype));
    }
    return writer.finish_layer();
}

template <typename T>
void verify_dtype(
    const LayerFixture &fixture,
    const TempDir &temp,
    BCSuccessDTypeMode dtype,
    const std::string &name
) {
    constexpr uint32_t row_width = 2U;
    const std::vector<uint8_t> success_bytes =
        write_success_layer<T>(fixture.position, row_width, dtype);
    const std::filesystem::path pos_path = temp.path / ("layer_" + name + ".bcpos");
    const std::filesystem::path suc_path = temp.path / ("layer_" + name + ".bcsuc");
    const std::filesystem::path cmp_path = temp.path / ("layer_" + name + ".bccmp");
    write_bytes(pos_path, fixture.position_bytes);
    write_bytes(suc_path, success_bytes);

    BCCompressedResult::CompressOptions options;
    options.worker_count = 2U;
    const auto stats = BCCompressedResult::compress_exact_layer_to_result(
        pos_path,
        suc_path,
        fixture.lut,
        cmp_path,
        options);
    check(stats.bucket_blocks != 0U, "compressed test should emit bucket blocks");
    check(stats.value_blocks != 0U, "compressed test should emit value blocks");

    BCCompressedResult::PointReader reader(cmp_path, fixture.lut);
    check(reader.row_width() == row_width, "compressed reader row_width mismatch");
    check(reader.dtype_mode() == dtype, "compressed reader dtype mismatch");
    for (const QueryCase &query : fixture.queries) {
        const BC::BCLookupResult exact =
            fixture.position.cold_lookup(query.cid, query.key, query.rank);
        check(exact.found, "exact position lookup should find query");
        for (uint32_t lane = 0U; lane < row_width; ++lane) {
            const auto result = reader.lookup(query.board, lane);
            check(result.found, "compressed lookup should find query");
            check(result.cid == query.cid, "compressed lookup cid mismatch");
            check(result.local_success_row == exact.local_success_row,
                  "compressed lookup row mismatch");
            const T expected = expected_value<T>(query.cid, exact.local_success_row, lane, dtype);
            check(result.raw_value_bits == raw_bits(expected), "compressed lookup raw value mismatch");
            check(result.bucket_block_raw_bytes <= options.bucket_block_raw_hard_cap_bytes,
                  "compressed lookup bucket block exceeds cap");
            check(result.value_block_raw_bytes <= options.value_block_raw_hard_cap_bytes,
                  "compressed lookup value block exceeds cap");
        }
    }

    const auto miss = reader.lookup(0U, 0U);
    check(!miss.found, "compressed lookup should miss wrong layer board");

    uint64_t sampled_board = 0U;
    uint64_t sampled_raw = 0U;
    double sampled_numeric = 0.0;
    check(
        reader.sample(sampled_board, sampled_raw, sampled_numeric, 0U),
        "compressed reader sample should return a board");
    const auto sampled_lookup = reader.lookup(sampled_board, 0U);
    check(sampled_lookup.found, "sampled board should be lookup-able");
    check(sampled_lookup.raw_value_bits == sampled_raw, "sampled raw value mismatch");

    uint64_t sampled_board_cold = 0U;
    uint64_t sampled_raw_cold = 0U;
    double sampled_numeric_cold = 0.0;
    check(
        BCCompressedResult::sample_cold(
            cmp_path,
            fixture.lut,
            sampled_board_cold,
            sampled_raw_cold,
            sampled_numeric_cold,
            0U),
        "compressed sample_cold should return a board");
    const auto sampled_cold_lookup = reader.lookup(sampled_board_cold, 0U);
    check(sampled_cold_lookup.found, "sample_cold board should be lookup-able");
    check(sampled_cold_lookup.raw_value_bits == sampled_raw_cold,
          "sample_cold raw value mismatch");
}

void verify_in_memory_path(const LayerFixture &fixture, const TempDir &temp) {
    constexpr uint32_t row_width = 1U;
    const std::vector<uint8_t> success_bytes =
        write_success_layer<uint32_t>(fixture.position, row_width, BCSuccessDTypeMode::UInt32);
    BCSuccessLayerReader success(success_bytes, fixture.position, row_width);
    const std::filesystem::path cmp_path = temp.path / "in_memory.bccmp";
    BCCompressedResult::CompressOptions options;
    options.worker_count = 2U;
    const auto stats = BCCompressedResult::compress_in_memory_layer_to_result(
        fixture.position,
        success,
        cmp_path,
        options);
    check(stats.output_bytes != 0U, "in-memory compression should write output");
    BCCompressedResult::PointReader reader(cmp_path, fixture.lut);
    const QueryCase &query = fixture.queries.front();
    const auto exact = fixture.position.cold_lookup(query.cid, query.key, query.rank);
    const auto got = reader.lookup(query.board, 0U);
    check(got.found, "in-memory compressed lookup should find query");
    check(got.raw_value_bits ==
              expected_value<uint32_t>(query.cid, exact.local_success_row, 0U, BCSuccessDTypeMode::UInt32),
          "in-memory compressed lookup value mismatch");
}

} // namespace

int main() {
    try {
        TempDir temp;
        LayerFixture fixture;
        verify_dtype<uint32_t>(fixture, temp, BCSuccessDTypeMode::UInt32, "u32");
        verify_dtype<uint64_t>(fixture, temp, BCSuccessDTypeMode::UInt64, "u64");
        verify_dtype<float>(fixture, temp, BCSuccessDTypeMode::Float32, "f32");
        verify_dtype<double>(fixture, temp, BCSuccessDTypeMode::Float64, "f64");
        verify_dtype<float>(fixture, temp, BCSuccessDTypeMode::OneMinusFloat32, "omf32");
        verify_dtype<double>(fixture, temp, BCSuccessDTypeMode::OneMinusFloat64, "omf64");
        verify_in_memory_path(fixture, temp);
        std::cout << "BC compressed result tests passed\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "BC compressed result test failed: " << ex.what() << "\n";
        return 1;
    }
}
