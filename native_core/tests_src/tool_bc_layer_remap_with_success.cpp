#include "BCDirectFileIO.h"
#include "BCFileIO.h"
#include "BCPositionFamilyRemapReader.h"
#include "BCPositionFile.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path source_position;
    std::filesystem::path source_success;
    std::filesystem::path output_position;
    std::filesystem::path output_success;
    uint32_t target_rank = 8U;
    uint32_t target_modulus = 0U;
    bool direct_io = false;
    uint32_t direct_queue_depth = 16U;
};

struct Timings {
    double position_remap_seconds = 0.0;
    double source_load_seconds = 0.0;
    double success_remap_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
};

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--source-position") {
            args.source_position = require_value(argc, argv, i, "--source-position");
        } else if (key == "--source-success") {
            args.source_success = require_value(argc, argv, i, "--source-success");
        } else if (key == "--output-position") {
            args.output_position = require_value(argc, argv, i, "--output-position");
        } else if (key == "--output-success") {
            args.output_success = require_value(argc, argv, i, "--output-success");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-rank"))
            );
        } else if (key == "--target-modulus") {
            args.target_modulus = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-modulus"))
            );
        } else if (key == "--direct-io") {
            args.direct_io = true;
        } else if (key == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--direct-queue-depth"))
            );
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.source_position.empty() || args.output_position.empty()) {
        throw std::invalid_argument("--source-position and --output-position are required");
    }
    if (args.source_success.empty() != args.output_success.empty()) {
        throw std::invalid_argument("--source-success and --output-success must be supplied together");
    }
    if (args.target_rank >= 15U) {
        throw std::invalid_argument("--target-rank must be < 15");
    }
    if (args.target_modulus == 0U) {
        throw std::invalid_argument("--target-modulus must be non-zero");
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be non-zero");
    }
    return args;
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] BC::BCLut make_free_lut(uint32_t target_rank) {
    return BC::BCLut(make_free_legal_tiles(target_rank));
}

[[nodiscard]] BC::BCPositionStreamingReader open_position(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCLut &lut
) {
    BC::BCPositionStreamingReader reader = args.direct_io
        ? BC::BCPositionStreamingReader::open_direct_auto(
              path,
              lut,
              args.direct_queue_depth,
              args.direct_queue_depth > 1U)
        : BC::BCPositionStreamingReader::open_buffered(path, lut);
    reader.set_validate_loaded_cells(false);
    return reader;
}

[[nodiscard]] BC::BCSuccessStreamingReader open_success(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCPositionStreamingReader &position
) {
    return args.direct_io
        ? BC::BCSuccessStreamingReader::open_direct_auto(
              path,
              position,
              1U,
              args.direct_queue_depth,
              args.direct_queue_depth > 1U)
        : BC::BCSuccessStreamingReader::open_buffered(path, position, 1U);
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> open_writer(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size = 0U
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (args.direct_io) {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    (void)logical_size;
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

[[nodiscard]] uint32_t bucket_live_rows(
    const BC::BCLut &lut,
    const BC::BCRankPayloadView &payload,
    const BC::BCBucketEntry &bucket
) {
    const BC::BucketBitmapLen bitmap_len = BC::bitmap_len_from_trusted_key(lut, bucket.key);
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len
    );
    const uint32_t word_count = BC::words_for_bits(bitmap_len);
    uint32_t rows = 0U;
    for (uint32_t i = 0U; i < word_count; ++i) {
        uint64_t word = BC::load_u64_le(
            payload.data + bitmap_offset + static_cast<size_t>(i) * sizeof(uint64_t)
        );
        if (i + 1U == word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        rows += BC::popcount64(word);
    }
    return rows;
}

[[nodiscard]] BC::BCFamilyPartitionLayerMap partition_for_axis(
    const BC::BCFamilyTable &axis,
    const std::vector<BC::LayerSum> &possible_8tile_sums
) {
    return BC::build_family_partition_layer_map(
        axis,
        possible_8tile_sums,
        BC::BCFamilyPartitionPolicy::modulo(axis.family_count())
    );
}

[[nodiscard]] BC::CellId key_source_cid(
    const BC::BCLut &lut,
    uint64_t key,
    const BC::BCFamilyTable &source_axis,
    const BC::BCFamilyPartitionLayerMap &source_partition,
    const BC::BCCellMatrix &source_matrix
) {
    const uint16_t nw = static_cast<uint16_t>((key >> 48U) & 0xFFFFU);
    const uint16_t ne = static_cast<uint16_t>((key >> 32U) & 0xFFFFU);
    const uint16_t sw = static_cast<uint16_t>((key >> 16U) & 0xFFFFU);
    const uint16_t se = static_cast<uint16_t>(key & 0xFFFFU);
    const BC::BCWordDesc &nw_desc = lut.word_desc(nw);
    if (!nw_desc.valid) {
        throw std::runtime_error("BC success remap source key has invalid NW word");
    }
    const uint64_t nw_sum = nw_desc.sum;
    const uint64_t ne_sum = lut.sum4_value(BC::packed_sum_id(ne));
    const uint64_t sw_sum = lut.sum4_value(BC::packed_sum_id(sw));
    const uint64_t se_sum = lut.sum4_value(BC::packed_sum_id(se));
    const uint16_t family_unit = source_axis.family_unit();
    auto coord_from_pair = [family_unit](uint64_t lhs, uint64_t rhs, BC::FamilyCoord &out) {
        const uint64_t min_sum = std::min<uint64_t>(lhs, rhs);
        if (family_unit == 0U || (min_sum % family_unit) != 0U) {
            return false;
        }
        const uint64_t coord = min_sum / family_unit;
        if (coord > std::numeric_limits<BC::FamilyCoord>::max()) {
            return false;
        }
        out = static_cast<BC::FamilyCoord>(coord);
        return true;
    };

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!coord_from_pair(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
        !coord_from_pair(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
        throw std::runtime_error("BC success remap key does not map to source coord");
    }
    const BC::FamilyId row_id = source_partition.try_coord_to_family_id(row_coord);
    const BC::FamilyId col_id = source_partition.try_coord_to_family_id(col_coord);
    if (row_id == BC::BCFamilyTable::kInvalidFamilyId ||
        col_id == BC::BCFamilyTable::kInvalidFamilyId) {
        throw std::runtime_error("BC success remap key does not map to source family");
    }
    return source_matrix.cid(row_id, col_id);
}

[[nodiscard]] const BC::BCBucketEntry &find_source_bucket(
    const BC::BCLoadedCell &source_cell,
    uint64_t key
) {
    const auto it = std::lower_bound(
        source_cell.buckets.begin(),
        source_cell.buckets.end(),
        key,
        [](const BC::BCBucketEntry &entry, uint64_t needle) {
            return entry.key < needle;
        }
    );
    if (it == source_cell.buckets.end() || it->key != key) {
        throw std::runtime_error("BC success remap source bucket key not found");
    }
    return *it;
}

[[nodiscard]] std::vector<uint64_t> cell_value_offsets(
    const BC::BCPositionStreamingReader &position,
    uint32_t row_width
) {
    std::vector<uint64_t> offsets(position.cell_count(), 0U);
    uint64_t cursor = 0U;
    for (BC::CellId cid = 0; cid < position.cell_count(); ++cid) {
        offsets[static_cast<size_t>(cid)] = cursor;
        const uint64_t count =
            static_cast<uint64_t>(position.descriptor(cid).success_rows) * row_width;
        cursor = BC::bc_checked_add_u64(cursor, count, "BC success remap offset overflow");
    }
    return offsets;
}

[[nodiscard]] std::vector<uint64_t> cell_value_offsets_for_loaded_cells(
    const std::vector<BC::BCLoadedCell> &cells
) {
    std::vector<uint64_t> offsets(cells.size(), 0U);
    uint64_t cursor = 0U;
    for (size_t cid = 0U; cid < cells.size(); ++cid) {
        offsets[cid] = cursor;
        cursor = BC::bc_checked_add_u64(
            cursor,
            cells[cid].success_rows,
            "BC success remap loaded-cell offset overflow"
        );
    }
    return offsets;
}

uint64_t write_success_values_streaming(
    BC::BCWritableFile &file,
    const BC::BCPositionStreamingReader &position,
    const std::vector<uint32_t> &values,
    BC::BCFileIOStats *stats = nullptr
) {
    const uint32_t row_width = 1U;
    const BC::BCSuccessDTypeMode dtype = BC::BCSuccessDTypeMode::UInt32;
    const uint64_t expected_values = BC::bc_success_total_values_for(position, row_width);
    if (expected_values != static_cast<uint64_t>(values.size())) {
        throw std::invalid_argument("BC success remap output value count mismatch");
    }
    const uint64_t payload_bytes = expected_values * sizeof(uint32_t);
    const std::vector<uint64_t> offsets = cell_value_offsets(position, row_width);

    BC::BCSuccessHeader header;
    header.dtype = static_cast<uint32_t>(dtype);
    header.row_width = row_width;
    header.family_count = position.header().family_count;
    header.descriptor_count = position.cell_count();
    header.cell_value_offsets_offset = BC::kBCSuccessHeaderBytes;
    header.payload_offset = BC::bc_checked_add_u64(
        header.cell_value_offsets_offset,
        BC::bc_success_cell_value_offsets_bytes(header.descriptor_count),
        "BC success remap payload offset overflow"
    );
    header.payload_bytes = payload_bytes;
    header.position_key_mode = position.header().key_mode;
    header.family_unit = position.header().family_unit;
    header.axis_base_coord = position.header().axis_base_coord;
    header.layer_sum = position.header().layer_sum;
    header.position_metadata_fingerprint = BC::bc_success_position_fingerprint_for(position);

    std::vector<uint8_t> header_bytes;
    header_bytes.reserve(BC::kBCSuccessHeaderBytes);
    BC::bc_append_success_header(header_bytes, header);
    std::vector<uint8_t> offset_bytes;
    offset_bytes.reserve(static_cast<size_t>(
        BC::bc_success_cell_value_offsets_bytes(header.descriptor_count)
    ));
    BC::bc_append_success_cell_value_offsets(offset_bytes, offsets);

    const uint64_t logical_size = BC::bc_success_logical_size(header);
    file.prepare_full_overwrite(logical_size);
    BC::BCSequentialSuccessWriteStager stager(file, stats);
    stager.append(header_bytes.data(), header_bytes.size());
    stager.append(offset_bytes.data(), offset_bytes.size());
    stager.append_values(values);
    stager.finish();
    return logical_size;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const double all_begin = now_seconds();
        const BC::BCLut lut = make_free_lut(args.target_rank);
        const std::vector<BC::LayerSum> possible_8tile_sums =
            BC::build_possible_8tile_sums(
                make_free_legal_tiles(args.target_rank),
                BC::default_2048_tile_sum_values()
            );

        BC::BCPositionStreamingReader source =
            open_position(args, args.source_position, lut);
        const BC::BCFamilyTable target_axis =
            BC::build_family_partition_axis_for_layer(
                source.header().layer_sum,
                source.header().family_unit,
                possible_8tile_sums,
                BC::BCFamilyPartitionPolicy::modulo(args.target_modulus)
            );
        BC::BCPositionFamilyRemapReader remap(source, target_axis, possible_8tile_sums);
        const BC::BCCellMatrix target_matrix(target_axis);
        std::vector<BC::CellId> target_cids(target_matrix.cell_count());
        for (BC::CellId cid = 0U; cid < target_matrix.cell_count(); ++cid) {
            target_cids[static_cast<size_t>(cid)] = cid;
        }

        Timings timings;
        const double remap_t0 = now_seconds();
        std::vector<BC::BCLoadedCell> target_cells;
        remap.load_cells_into(target_cids, target_cells, nullptr);
        timings.position_remap_seconds = now_seconds() - remap_t0;

        std::vector<BC::FinalizedCellPayload> payloads(target_cells.size());
        for (size_t i = 0U; i < target_cells.size(); ++i) {
            payloads[i].buckets = target_cells[i].buckets;
            payloads[i].rank_payload = target_cells[i].rank_payload;
            payloads[i].success_rows = target_cells[i].success_rows;
        }

        uint64_t position_bytes = 0U;
        {
            const double t0 = now_seconds();
            std::unique_ptr<BC::BCWritableFile> writer = open_writer(args, args.output_position);
            position_bytes = BC::write_position_payloads_to_file(*writer, target_axis, payloads, nullptr);
            writer.reset();
            timings.position_write_seconds = now_seconds() - t0;
        }

        uint64_t success_bytes = 0U;
        uint64_t max_raw = 0U;
        if (!args.source_success.empty()) {
            const double source_t0 = now_seconds();
            BC::BCSuccessStreamingReader source_success =
                open_success(args, args.source_success, source);
            std::vector<uint32_t> source_values =
                source_success.read_all_values_typed<uint32_t>();
            const std::vector<uint64_t> source_offsets = cell_value_offsets(source, 1U);
            std::vector<BC::CellId> source_cids(source.cell_count());
            for (BC::CellId cid = 0U; cid < source.cell_count(); ++cid) {
                source_cids[static_cast<size_t>(cid)] = cid;
            }
            std::vector<BC::BCLoadedCell> source_cells;
            source.load_cells_into(source_cids, source_cells, nullptr);
            const BC::BCFamilyPartitionLayerMap source_partition =
                partition_for_axis(source.axis(), possible_8tile_sums);
            const BC::BCCellMatrix source_matrix(source.axis());
            timings.source_load_seconds = now_seconds() - source_t0;

            const double success_t0 = now_seconds();
            const std::vector<uint64_t> target_offsets =
                cell_value_offsets_for_loaded_cells(target_cells);
            const uint64_t target_value_count =
                target_offsets.empty()
                    ? 0U
                    : target_offsets.back() +
                        static_cast<uint64_t>(target_cells.back().success_rows);
            if (target_value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC success remap target value count exceeds size_t");
            }
            std::vector<uint32_t> target_values(static_cast<size_t>(target_value_count), 0U);
            for (BC::CellId target_cid = 0U; target_cid < target_cells.size(); ++target_cid) {
                const BC::BCLoadedCell &target_cell = target_cells[static_cast<size_t>(target_cid)];
                const BC::BCLoadedCellView target_view = target_cell.view();
                for (uint32_t i = 0U; i < target_view.buckets.size; ++i) {
                    const BC::BCBucketEntry &target_bucket = target_view.buckets.data[i];
                    const BC::CellId source_cid = key_source_cid(
                        lut,
                        target_bucket.key,
                        source.axis(),
                        source_partition,
                        source_matrix
                    );
                    const BC::BCLoadedCell &source_cell = source_cells[static_cast<size_t>(source_cid)];
                    const BC::BCBucketEntry &source_bucket =
                        find_source_bucket(source_cell, target_bucket.key);
                    const uint32_t target_rows =
                        bucket_live_rows(lut, target_view.rank_payload, target_bucket);
                    const uint32_t source_rows =
                        bucket_live_rows(lut, source_cell.view().rank_payload, source_bucket);
                    if (target_rows != source_rows) {
                        throw std::runtime_error("BC success remap bucket row count mismatch");
                    }
                    const uint64_t source_begin =
                        source_offsets[static_cast<size_t>(source_cid)] +
                        source_bucket.success_row_offset;
                    const uint64_t target_begin =
                        target_offsets[static_cast<size_t>(target_cid)] +
                        target_bucket.success_row_offset;
                    if (source_begin > source_values.size() ||
                        target_begin > target_values.size() ||
                        target_rows > source_values.size() - source_begin ||
                        target_rows > target_values.size() - target_begin) {
                        throw std::runtime_error("BC success remap copy range out of bounds");
                    }
                    std::copy_n(
                        source_values.data() + static_cast<size_t>(source_begin),
                        static_cast<size_t>(target_rows),
                        target_values.data() + static_cast<size_t>(target_begin)
                    );
                }
            }
            max_raw = target_values.empty()
                ? 0U
                : *std::max_element(target_values.begin(), target_values.end());
            timings.success_remap_seconds = now_seconds() - success_t0;

            const double write_t0 = now_seconds();
            BC::BCPositionStreamingReader target_position =
                open_position(args, args.output_position, lut);
            BC::BCSuccessHeader output_header;
            output_header.dtype = BC::kBCSuccessDTypeUint32;
            output_header.row_width = 1U;
            output_header.family_count = target_position.header().family_count;
            output_header.descriptor_count = target_position.cell_count();
            output_header.cell_value_offsets_offset = BC::kBCSuccessHeaderBytes;
            output_header.payload_offset = BC::bc_checked_add_u64(
                output_header.cell_value_offsets_offset,
                BC::bc_success_cell_value_offsets_bytes(output_header.descriptor_count),
                "BC success remap logical size overflow"
            );
            output_header.payload_bytes =
                static_cast<uint64_t>(target_values.size()) * sizeof(uint32_t);
            output_header.position_key_mode = target_position.header().key_mode;
            output_header.family_unit = target_position.header().family_unit;
            output_header.axis_base_coord = target_position.header().axis_base_coord;
            output_header.layer_sum = target_position.header().layer_sum;
            output_header.position_metadata_fingerprint =
                BC::bc_success_position_fingerprint_for(target_position);
            const uint64_t logical_size = BC::bc_success_logical_size(output_header);
            std::unique_ptr<BC::BCWritableFile> writer =
                open_writer(args, args.output_success, logical_size);
            success_bytes = write_success_values_streaming(
                *writer,
                target_position,
                target_values,
                nullptr
            );
            writer.reset();
            timings.success_write_seconds = now_seconds() - write_t0;
        }

        uint64_t rows = 0U;
        uint64_t buckets = 0U;
        uint64_t rank_bytes = 0U;
        for (const BC::FinalizedCellPayload &payload : payloads) {
            rows += payload.success_rows;
            buckets += payload.buckets.size();
            rank_bytes += payload.rank_payload.size();
        }
        const double total_seconds = now_seconds() - all_begin;
        std::cout << std::setprecision(9)
                  << "layer_sum=" << source.header().layer_sum
                  << " source_family_count=" << source.axis().family_count()
                  << " target_family_count=" << target_axis.family_count()
                  << " rows=" << rows
                  << " buckets=" << buckets
                  << " rank_payload_bytes=" << rank_bytes
                  << " position_bytes=" << position_bytes
                  << " success_bytes=" << success_bytes
                  << " max_raw=" << max_raw
                  << " position_remap_seconds=" << timings.position_remap_seconds
                  << " source_load_seconds=" << timings.source_load_seconds
                  << " success_remap_seconds=" << timings.success_remap_seconds
                  << " position_write_seconds=" << timings.position_write_seconds
                  << " success_write_seconds=" << timings.success_write_seconds
                  << " total_seconds=" << total_seconds
                  << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "bc_layer_remap_with_success failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
