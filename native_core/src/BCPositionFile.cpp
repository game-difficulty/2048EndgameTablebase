#include "BCPositionFile.h"

#include "BCDirectFileIO.h"
#include "BCFileIO.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace BC {
namespace {

inline constexpr uint64_t kBCPositionStreamingWriteStagingBytes = 16ULL * 1024ULL * 1024ULL;

void bc_accumulate_file_stats(BCFileIOStats *dst, const BCFileIOStats &src) {
    if (dst == nullptr) {
        return;
    }
    if (dst->request_count > std::numeric_limits<uint64_t>::max() - src.request_count ||
        dst->requested_bytes > std::numeric_limits<uint64_t>::max() - src.requested_bytes ||
        dst->backend_io_count > std::numeric_limits<uint64_t>::max() - src.backend_io_count ||
        dst->backend_bytes > std::numeric_limits<uint64_t>::max() - src.backend_bytes) {
        throw std::overflow_error("BC position streaming write stats overflow");
    }
    dst->request_count += src.request_count;
    dst->requested_bytes += src.requested_bytes;
    dst->backend_io_count += src.backend_io_count;
    dst->backend_bytes += src.backend_bytes;
}

void bc_store_u32_le(uint8_t *out, uint32_t value) {
    out[0] = static_cast<uint8_t>(value & 0xFFU);
    out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    out[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    out[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

void bc_store_u64_le(uint8_t *out, uint64_t value) {
    for (uint32_t i = 0U; i < 8U; ++i) {
        out[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU);
    }
}

[[nodiscard]] std::array<uint8_t, kBCPositionBucketEntryBytes> bc_serialize_bucket_entry(
    const BCBucketEntry &entry
) {
    std::array<uint8_t, kBCPositionBucketEntryBytes> out = {};
    bc_store_u64_le(out.data() + 0U, entry.key);
    bc_store_u32_le(out.data() + 8U, entry.rank_payload_offset);
    bc_store_u32_le(out.data() + 12U, entry.success_row_offset);
    return out;
}

class BCSequentialPositionWriteStager {
public:
    BCSequentialPositionWriteStager(BCWritableFile &file, BCFileIOStats *stats)
        : file_(file), stats_(stats) {
        buffer_.reserve(static_cast<size_t>(kBCPositionStreamingWriteStagingBytes));
    }

    void append(const void *data, uint64_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC position streaming write append pointer is null");
        }
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining > 0U) {
            const uint64_t available =
                kBCPositionStreamingWriteStagingBytes - static_cast<uint64_t>(buffer_.size());
            if (available == 0U) {
                flush_buffer();
                continue;
            }
            const uint64_t take = std::min<uint64_t>(remaining, available);
            const size_t old_size = buffer_.size();
            buffer_.resize(old_size + static_cast<size_t>(take));
            std::memcpy(buffer_.data() + old_size, cursor, static_cast<size_t>(take));
            cursor += take;
            remaining -= take;
            if (buffer_.size() == kBCPositionStreamingWriteStagingBytes) {
                flush_buffer();
            }
        }
    }

    void finish() {
        flush_buffer();
    }

private:
    void flush_buffer() {
        if (buffer_.empty()) {
            return;
        }
        BCFileIOStats local_stats;
        std::vector<BCFileWriteRequest> request{
            BCFileWriteRequest{
                file_cursor_,
                buffer_.data(),
                static_cast<uint64_t>(buffer_.size())
            }
        };
        file_.write_many(request, stats_ == nullptr ? nullptr : &local_stats);
        bc_accumulate_file_stats(stats_, local_stats);
        file_cursor_ = bc_checked_add_u64(
            file_cursor_,
            buffer_.size(),
            "BC position streaming write cursor overflow"
        );
        buffer_.clear();
    }

    BCWritableFile &file_;
    BCFileIOStats *stats_ = nullptr;
    std::vector<uint8_t> buffer_;
    uint64_t file_cursor_ = 0U;
};

} // namespace

void bc_validate_position_payload_for_file(const FinalizedCellPayload &payload) {
    if (payload.buckets.empty()) {
        if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
            throw std::invalid_argument("BC empty finalized payload has non-empty metadata");
        }
        return;
    }
    if (payload.buckets.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC position cell bucket_count exceeds uint32");
    }
    for (size_t i = 1U; i < payload.buckets.size(); ++i) {
        if (payload.buckets[i - 1U].key >= payload.buckets[i].key) {
            throw std::invalid_argument("BC position writer requires sorted unique bucket keys");
        }
    }
}

uint64_t write_position_payloads_to_file(
    BCWritableFile &file,
    const BCFamilyTable &axis,
    const std::vector<const FinalizedCellPayload *> &cell_payloads,
    BCFileIOStats *stats
) {
    if (axis.family_count() == 0U) {
        throw std::invalid_argument("BC position streaming writer requires non-empty family axis");
    }
    const BCCellMatrix matrix(axis);
    if (cell_payloads.size() != matrix.cell_count()) {
        throw std::invalid_argument("BC position streaming writer payload count does not match cell count");
    }

    const uint64_t descriptor_count = cell_payloads.size();
    if (descriptor_count >
        std::numeric_limits<uint64_t>::max() / kBCPositionCellDescriptorBytes) {
        throw std::overflow_error("BC position streaming descriptor byte size overflow");
    }
    const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
    const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis.family_count());
    const uint64_t descriptor_offset = bc_checked_add_u64(
        kBCPositionHeaderBytes,
        axis_coord_bytes,
        "BC position streaming descriptor table offset overflow"
    );
    const uint64_t bucket_offset = bc_checked_add_u64(
        descriptor_offset,
        descriptor_bytes,
        "BC position streaming bucket stream offset overflow"
    );

    std::vector<BCPositionCellDescriptor> descriptors(cell_payloads.size());
    uint64_t bucket_cursor = 0U;
    uint64_t rank_cursor = 0U;
    for (CellId cid = 0U; cid < cell_payloads.size(); ++cid) {
        const FinalizedCellPayload *payload = cell_payloads[static_cast<size_t>(cid)];
        if (payload == nullptr || payload->buckets.empty()) {
            if (payload != nullptr) {
                bc_validate_position_payload_for_file(*payload);
            }
            BCPositionCellDescriptor descriptor;
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
            descriptors[static_cast<size_t>(cid)] = descriptor;
            continue;
        }

        bc_validate_position_payload_for_file(*payload);
        const uint64_t bucket_bytes =
            static_cast<uint64_t>(payload->buckets.size()) * kBCPositionBucketEntryBytes;
        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = static_cast<uint32_t>(payload->buckets.size());
        descriptor.success_rows = payload->success_rows;
        descriptor.bucket_meta_offset = bucket_cursor;
        descriptor.rank_payload_offset = rank_cursor;
        descriptor.rank_payload_bytes = payload->rank_payload.size();
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;
        descriptors[static_cast<size_t>(cid)] = descriptor;
        bucket_cursor = bc_checked_add_u64(
            bucket_cursor,
            bucket_bytes,
            "BC position streaming bucket stream byte count overflow"
        );
        rank_cursor = bc_checked_add_u64(
            rank_cursor,
            payload->rank_payload.size(),
            "BC position streaming rank stream byte count overflow"
        );
    }

    const uint64_t rank_offset = bc_checked_add_u64(
        bucket_offset,
        bucket_cursor,
        "BC position streaming rank stream offset overflow"
    );
    const uint64_t logical_size = bc_checked_add_u64(
        rank_offset,
        rank_cursor,
        "BC position streaming logical file size overflow"
    );

    BCPositionHeader header;
    header.family_unit = axis.family_unit();
    header.axis_base_coord = axis.axis_base_coord();
    header.family_count = axis.family_count();
    header.layer_sum = axis.layer_sum();
    header.axis_coord_table_bytes = axis_coord_bytes;
    header.descriptor_count = descriptor_count;
    header.descriptor_table_offset = descriptor_offset;
    header.descriptor_table_bytes = descriptor_bytes;
    header.bucket_meta_offset = bucket_offset;
    header.bucket_meta_bytes = bucket_cursor;
    header.rank_payload_offset = rank_offset;
    header.rank_payload_bytes = rank_cursor;

    std::vector<uint8_t> header_bytes;
    header_bytes.reserve(kBCPositionHeaderBytes);
    bc_append_header(header_bytes, header);
    std::vector<uint8_t> axis_coord_table;
    axis_coord_table.reserve(static_cast<size_t>(axis_coord_bytes));
    bc_append_axis_coord_table(axis_coord_table, axis);

    if (descriptor_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC position streaming descriptor table exceeds size_t");
    }
    std::vector<uint8_t> descriptor_table;
    descriptor_table.reserve(static_cast<size_t>(descriptor_bytes));
    for (const BCPositionCellDescriptor &descriptor : descriptors) {
        bc_append_cell_descriptor(descriptor_table, descriptor);
    }

    if (stats != nullptr) {
        *stats = {};
    }
    file.prepare_full_overwrite(logical_size);
    BCSequentialPositionWriteStager stager(file, stats);
    stager.append(header_bytes.data(), static_cast<uint64_t>(header_bytes.size()));
    stager.append(axis_coord_table.data(), static_cast<uint64_t>(axis_coord_table.size()));
    stager.append(descriptor_table.data(), static_cast<uint64_t>(descriptor_table.size()));

    uint64_t bucket_bytes_written = 0U;
    for (const FinalizedCellPayload *payload : cell_payloads) {
        if (payload == nullptr || payload->buckets.empty()) {
            continue;
        }
        for (const BCBucketEntry &bucket : payload->buckets) {
            const auto bucket_bytes = bc_serialize_bucket_entry(bucket);
            stager.append(bucket_bytes.data(), bucket_bytes.size());
            bucket_bytes_written = bc_checked_add_u64(
                bucket_bytes_written,
                bucket_bytes.size(),
                "BC position streaming bucket bytes written overflow"
            );
        }
    }
    if (bucket_bytes_written != bucket_cursor) {
        throw std::logic_error("BC position streaming bucket byte count mismatch");
    }

    uint64_t rank_bytes_written = 0U;
    for (const FinalizedCellPayload *payload : cell_payloads) {
        if (payload == nullptr || payload->buckets.empty() || payload->rank_payload.empty()) {
            continue;
        }
        stager.append(payload->rank_payload.data(), static_cast<uint64_t>(payload->rank_payload.size()));
        rank_bytes_written = bc_checked_add_u64(
            rank_bytes_written,
            payload->rank_payload.size(),
            "BC position streaming rank bytes written overflow"
        );
    }
    if (rank_bytes_written != rank_cursor) {
        throw std::logic_error("BC position streaming rank byte count mismatch");
    }
    stager.finish();
    if (file.mode() != BCFileIOMode::Direct) {
        file.flush();
    }
    return logical_size;
}

uint64_t write_position_payloads_to_file(
    BCWritableFile &file,
    const BCFamilyTable &axis,
    const std::vector<FinalizedCellPayload> &cell_payloads,
    BCFileIOStats *stats
) {
    std::vector<const FinalizedCellPayload *> payload_views;
    payload_views.reserve(cell_payloads.size());
    for (const FinalizedCellPayload &payload : cell_payloads) {
        payload_views.push_back(&payload);
    }
    return write_position_payloads_to_file(file, axis, payload_views, stats);
}

void write_position_layer_to_file(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    write_bytes_to_buffered_file(path, bytes);
}

std::vector<uint8_t> read_position_layer_from_file(const std::filesystem::path &path) {
    return read_bytes_from_buffered_file(path);
}

BCPositionFileReader::BCPositionFileReader(std::unique_ptr<BCReadableFile> file, const BCLut &lut) {
    open(std::move(file), lut);
}

BCPositionFileReader::~BCPositionFileReader() = default;
BCPositionFileReader::BCPositionFileReader(BCPositionFileReader &&) noexcept = default;
BCPositionFileReader &BCPositionFileReader::operator=(BCPositionFileReader &&) noexcept = default;

BCPositionFileReader BCPositionFileReader::open_buffered(
    const std::filesystem::path &path,
    const BCLut &lut
) {
    return BCPositionFileReader(std::make_unique<BCBufferedFileReader>(path), lut);
}

BCPositionFileReader BCPositionFileReader::open_direct_auto(
    const std::filesystem::path &path,
    const BCLut &lut,
    uint32_t queue_depth,
    bool overlapped
) {
    BCBufferedFileReader probe(path);
    std::vector<uint8_t> header_bytes(kBCPositionHeaderBytes);
    probe.read_at(0U, header_bytes.data(), header_bytes.size());
    const BCPositionHeader header = bc_read_header(header_bytes);
    const uint64_t logical_size = bc_position_logical_size_from_header(header);
    BCDirectFileIOOptions options;
    options.queue_depth = queue_depth;
    options.overlapped = overlapped || queue_depth > 1U;
    options.logical_size = logical_size;
    const uint64_t required_physical = bc_direct_align_up(logical_size, options.alignment);
    if (probe.size() >= required_physical) {
        return BCPositionFileReader(std::make_unique<BCDirectFileReader>(path, options), lut);
    }
    return BCPositionFileReader(std::make_unique<BCBufferedFileReader>(path), lut);
}

void BCPositionFileReader::open(std::unique_ptr<BCReadableFile> file, const BCLut &lut) {
    if (!file) {
        throw std::invalid_argument("BC position file reader file is null");
    }
    file_ = std::move(file);
    const uint64_t byte_count = file_->size();
    if (byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC position file exceeds addressable memory vector size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count), 0U);
    if (!bytes.empty()) {
        if (file_->mode() == BCFileIOMode::Direct) {
            constexpr uint64_t kDirectReadChunkBytes = 16ULL * 1024ULL * 1024ULL;
            std::vector<BCFileReadRequest> requests;
            requests.reserve(
                static_cast<size_t>((byte_count + kDirectReadChunkBytes - 1U) / kDirectReadChunkBytes)
            );
            uint64_t offset = 0U;
            while (offset < byte_count) {
                const uint64_t take = std::min<uint64_t>(kDirectReadChunkBytes, byte_count - offset);
                requests.push_back(BCFileReadRequest{
                    offset,
                    bytes.data() + static_cast<size_t>(offset),
                    take
                });
                offset += take;
            }
            file_->read_many(requests);
        } else {
            file_->read_at(0U, bytes.data(), byte_count);
        }
    }
    layer_.open(std::move(bytes), lut);
}

const BCPositionLayerReader &BCPositionFileReader::layer() const {
    return layer_;
}

const std::vector<uint8_t> &BCPositionFileReader::bytes() const {
    return layer_.bytes();
}

BCPositionLayerReader BCPositionFileReader::take_layer() {
    return std::move(layer_);
}

} // namespace BC
