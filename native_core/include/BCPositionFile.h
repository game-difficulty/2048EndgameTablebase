#pragma once

#include "BCCellBuilder.h"
#include "BCCellMatrix.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace BC {

class BCReadableFile;
class BCWritableFile;
struct BCFileIOStats;

inline constexpr uint32_t kBCPositionMagic = 0x46504342U; // "BCPF" little-endian.
inline constexpr uint32_t kBCPositionFormatVersion = 2U;
inline constexpr uint32_t kBCPositionHeaderBytes = 112U;
inline constexpr uint32_t kBCPositionCellDescriptorBytes = 40U;
inline constexpr uint32_t kBCPositionBucketEntryBytes = 16U;
inline constexpr uint32_t kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256 = 1U;
inline constexpr uint32_t kBCPositionRankPrefixTypeUint16 = 1U;
inline constexpr uint32_t kBCPositionCellFlagEmpty = 1U;

struct BCPositionHeader {
    uint32_t magic = kBCPositionMagic;
    uint32_t format_version = kBCPositionFormatVersion;
    uint32_t header_bytes = kBCPositionHeaderBytes;
    uint32_t key_mode = kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256;
    uint32_t rank_prefix_bits = kBCRankPrefixBits;
    uint32_t rank_prefix_type = kBCPositionRankPrefixTypeUint16;
    uint32_t rank_payload_align = 8U;
    uint32_t family_unit = 0U;
    uint32_t axis_base_coord = 0U;
    uint32_t family_count = 0U;
    uint64_t layer_sum = 0U;
    uint64_t descriptor_count = 0U;
    uint64_t descriptor_table_offset = 0U;
    uint64_t descriptor_table_bytes = 0U;
    uint64_t bucket_meta_offset = 0U;
    uint64_t bucket_meta_bytes = 0U;
    uint64_t rank_payload_offset = 0U;
    uint64_t rank_payload_bytes = 0U;
    uint64_t axis_coord_table_bytes = 0U;
};

struct BCPositionCellDescriptor {
    uint32_t bucket_count = 0U;
    uint32_t success_rows = 0U;
    uint64_t bucket_meta_offset = 0U;
    uint64_t rank_payload_offset = 0U;
    uint64_t rank_payload_bytes = 0U;
    uint32_t reserved0 = 0U;
    uint32_t flags_or_padding = kBCPositionCellFlagEmpty;

    [[nodiscard]] bool empty() const {
        return (flags_or_padding & kBCPositionCellFlagEmpty) != 0U || bucket_count == 0U;
    }
};

[[nodiscard]] inline uint32_t bc_checked_u32(uint64_t value, const char *label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(label);
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] inline uint64_t bc_checked_add_u64(uint64_t lhs, uint64_t rhs, const char *label) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

inline void bc_append_u32_le(std::vector<uint8_t> &out, uint32_t value) {
    const size_t offset = out.size();
    out.resize(offset + 4U);
    uint8_t *p = out.data() + offset;
    p[0] = static_cast<uint8_t>(value & 0xFFU);
    p[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    p[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    p[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

inline void bc_append_u64_le(std::vector<uint8_t> &out, uint64_t value) {
    const size_t offset = out.size();
    out.resize(offset + 8U);
    uint8_t *p = out.data() + offset;
    p[0] = static_cast<uint8_t>(value & 0xFFU);
    p[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    p[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    p[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    p[4] = static_cast<uint8_t>((value >> 32U) & 0xFFU);
    p[5] = static_cast<uint8_t>((value >> 40U) & 0xFFU);
    p[6] = static_cast<uint8_t>((value >> 48U) & 0xFFU);
    p[7] = static_cast<uint8_t>((value >> 56U) & 0xFFU);
}

[[nodiscard]] inline uint32_t bc_load_u32_le(const uint8_t *data) {
    if (data == nullptr) {
        throw std::invalid_argument("BC position load_u32_le pointer is null");
    }
    return
        static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8U) |
        (static_cast<uint32_t>(data[2]) << 16U) |
        (static_cast<uint32_t>(data[3]) << 24U);
}

inline void bc_require_bytes(
    const std::vector<uint8_t> &bytes,
    uint64_t offset,
    uint64_t count,
    const char *label
) {
    const uint64_t end = bc_checked_add_u64(offset, count, label);
    if (end > bytes.size()) {
        throw std::out_of_range(label);
    }
}

inline void bc_append_header(std::vector<uint8_t> &out, const BCPositionHeader &header) {
    const size_t begin = out.size();
    bc_append_u32_le(out, header.magic);
    bc_append_u32_le(out, header.format_version);
    bc_append_u32_le(out, header.header_bytes);
    bc_append_u32_le(out, header.key_mode);
    bc_append_u32_le(out, header.rank_prefix_bits);
    bc_append_u32_le(out, header.rank_prefix_type);
    bc_append_u32_le(out, header.rank_payload_align);
    bc_append_u32_le(out, header.family_unit);
    bc_append_u32_le(out, header.axis_base_coord);
    bc_append_u32_le(out, header.family_count);
    bc_append_u64_le(out, header.layer_sum);
    bc_append_u64_le(out, header.descriptor_count);
    bc_append_u64_le(out, header.descriptor_table_offset);
    bc_append_u64_le(out, header.descriptor_table_bytes);
    bc_append_u64_le(out, header.bucket_meta_offset);
    bc_append_u64_le(out, header.bucket_meta_bytes);
    bc_append_u64_le(out, header.rank_payload_offset);
    bc_append_u64_le(out, header.rank_payload_bytes);
    bc_append_u64_le(out, header.axis_coord_table_bytes);
    if (out.size() - begin != kBCPositionHeaderBytes) {
        throw std::logic_error("BC position header serialized size mismatch");
    }
}

[[nodiscard]] inline BCPositionHeader bc_read_header(const std::vector<uint8_t> &bytes) {
    bc_require_bytes(bytes, 0U, kBCPositionHeaderBytes, "BC position file is smaller than header");
    BCPositionHeader header;
    const uint8_t *p = bytes.data();
    header.magic = bc_load_u32_le(p + 0U);
    header.format_version = bc_load_u32_le(p + 4U);
    header.header_bytes = bc_load_u32_le(p + 8U);
    header.key_mode = bc_load_u32_le(p + 12U);
    header.rank_prefix_bits = bc_load_u32_le(p + 16U);
    header.rank_prefix_type = bc_load_u32_le(p + 20U);
    header.rank_payload_align = bc_load_u32_le(p + 24U);
    header.family_unit = bc_load_u32_le(p + 28U);
    header.axis_base_coord = bc_load_u32_le(p + 32U);
    header.family_count = bc_load_u32_le(p + 36U);
    header.layer_sum = load_u64_le(p + 40U);
    header.descriptor_count = load_u64_le(p + 48U);
    header.descriptor_table_offset = load_u64_le(p + 56U);
    header.descriptor_table_bytes = load_u64_le(p + 64U);
    header.bucket_meta_offset = load_u64_le(p + 72U);
    header.bucket_meta_bytes = load_u64_le(p + 80U);
    header.rank_payload_offset = load_u64_le(p + 88U);
    header.rank_payload_bytes = load_u64_le(p + 96U);
    header.axis_coord_table_bytes = load_u64_le(p + 104U);
    return header;
}

[[nodiscard]] inline uint64_t bc_position_logical_size_from_header(
    const BCPositionHeader &header
) {
    const uint64_t bucket_end = bc_checked_add_u64(
        header.bucket_meta_offset,
        header.bucket_meta_bytes,
        "BC position bucket logical end overflow"
    );
    const uint64_t rank_end = bc_checked_add_u64(
        header.rank_payload_offset,
        header.rank_payload_bytes,
        "BC position rank logical end overflow"
    );
    return std::max(bucket_end, rank_end);
}

[[nodiscard]] inline uint64_t bc_axis_coord_table_bytes(uint32_t family_count) {
    return static_cast<uint64_t>(family_count) * sizeof(uint32_t);
}

inline void bc_append_axis_coord_table(
    std::vector<uint8_t> &out,
    const BCFamilyTable &axis
) {
    const std::vector<FamilyCoord> &coords = axis.coords();
    if (coords.size() != axis.family_count()) {
        throw std::logic_error("BC position axis coord table size mismatch");
    }
    for (FamilyCoord coord : coords) {
        bc_append_u32_le(out, coord);
    }
}

[[nodiscard]] inline std::vector<FamilyCoord> bc_read_axis_coord_table(
    const std::vector<uint8_t> &bytes,
    const BCPositionHeader &header
) {
    const uint64_t table_bytes = bc_axis_coord_table_bytes(header.family_count);
    if (header.axis_coord_table_bytes != table_bytes) {
        throw std::runtime_error("BC position axis coord table byte size mismatch");
    }
    bc_require_bytes(
        bytes,
        kBCPositionHeaderBytes,
        table_bytes,
        "BC position axis coord table exceeds file"
    );
    std::vector<FamilyCoord> coords;
    coords.reserve(static_cast<size_t>(header.family_count));
    const uint8_t *base = bytes.data() + kBCPositionHeaderBytes;
    for (uint32_t i = 0U; i < header.family_count; ++i) {
        coords.push_back(static_cast<FamilyCoord>(bc_load_u32_le(base + i * sizeof(uint32_t))));
    }
    return coords;
}

inline void bc_append_cell_descriptor(
    std::vector<uint8_t> &out,
    const BCPositionCellDescriptor &descriptor
) {
    const size_t begin = out.size();
    bc_append_u32_le(out, descriptor.bucket_count);
    bc_append_u32_le(out, descriptor.success_rows);
    bc_append_u64_le(out, descriptor.bucket_meta_offset);
    bc_append_u64_le(out, descriptor.rank_payload_offset);
    bc_append_u64_le(out, descriptor.rank_payload_bytes);
    bc_append_u32_le(out, descriptor.reserved0);
    bc_append_u32_le(out, descriptor.flags_or_padding);
    if (out.size() - begin != kBCPositionCellDescriptorBytes) {
        throw std::logic_error("BC position descriptor serialized size mismatch");
    }
}

[[nodiscard]] inline BCPositionCellDescriptor bc_read_cell_descriptor(
    const uint8_t *data,
    uint64_t remaining
) {
    if (data == nullptr) {
        throw std::invalid_argument("BC position descriptor pointer is null");
    }
    if (remaining < kBCPositionCellDescriptorBytes) {
        throw std::out_of_range("BC position descriptor is truncated");
    }
    BCPositionCellDescriptor descriptor;
    descriptor.bucket_count = bc_load_u32_le(data + 0U);
    descriptor.success_rows = bc_load_u32_le(data + 4U);
    descriptor.bucket_meta_offset = load_u64_le(data + 8U);
    descriptor.rank_payload_offset = load_u64_le(data + 16U);
    descriptor.rank_payload_bytes = load_u64_le(data + 24U);
    descriptor.reserved0 = bc_load_u32_le(data + 32U);
    descriptor.flags_or_padding = bc_load_u32_le(data + 36U);
    return descriptor;
}

inline void bc_append_bucket_entry(std::vector<uint8_t> &out, const BCBucketEntry &entry) {
    const size_t begin = out.size();
    bc_append_u64_le(out, entry.key);
    bc_append_u32_le(out, entry.rank_payload_offset);
    bc_append_u32_le(out, entry.success_row_offset);
    if (out.size() - begin != kBCPositionBucketEntryBytes) {
        throw std::logic_error("BC position bucket entry serialized size mismatch");
    }
}

[[nodiscard]] inline BCBucketEntry bc_read_bucket_entry(const uint8_t *data) {
    if (data == nullptr) {
        throw std::invalid_argument("BC position bucket entry pointer is null");
    }
    BCBucketEntry entry;
    entry.key = load_u64_le(data + 0U);
    entry.rank_payload_offset = bc_load_u32_le(data + 8U);
    entry.success_row_offset = bc_load_u32_le(data + 12U);
    return entry;
}

class BCPositionLayerWriter {
public:
    BCPositionLayerWriter() = default;

    void begin_layer(const BCFamilyTable &axis) {
        if (axis.family_count() == 0U) {
            throw std::invalid_argument("BC position writer requires non-empty family axis");
        }
        axis_ = axis;
        const BCCellMatrix matrix(axis_);
        descriptors_.assign(matrix.cell_count(), BCPositionCellDescriptor{});
        written_.assign(matrix.cell_count(), false);
        bucket_meta_stream_.clear();
        rank_payload_stream_.clear();
        begun_ = true;
    }

    void write_cell(CellId cid, const FinalizedCellPayload &payload) {
        require_begun();
        require_cell_for_write(cid);
        if (payload.buckets.empty()) {
            if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
                throw std::invalid_argument("BC empty finalized payload has non-empty metadata");
            }
            mark_empty_cell(cid);
            return;
        }
        if (payload.buckets.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC position cell bucket_count exceeds uint32");
        }
        for (size_t i = 1; i < payload.buckets.size(); ++i) {
            if (payload.buckets[i - 1U].key >= payload.buckets[i].key) {
                throw std::invalid_argument("BC position writer requires sorted unique bucket keys");
            }
        }

        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = static_cast<uint32_t>(payload.buckets.size());
        descriptor.success_rows = payload.success_rows;
        descriptor.bucket_meta_offset = bucket_meta_stream_.size();
        descriptor.rank_payload_offset = rank_payload_stream_.size();
        descriptor.rank_payload_bytes = payload.rank_payload.size();
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;

        for (const BCBucketEntry &bucket : payload.buckets) {
            bc_append_bucket_entry(bucket_meta_stream_, bucket);
        }
        rank_payload_stream_.insert(
            rank_payload_stream_.end(),
            payload.rank_payload.begin(),
            payload.rank_payload.end()
        );
        descriptors_[static_cast<size_t>(cid)] = descriptor;
        written_[static_cast<size_t>(cid)] = true;
    }

    void mark_empty_cell(CellId cid) {
        require_begun();
        require_cell_for_write(cid);
        BCPositionCellDescriptor descriptor;
        descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
        descriptors_[static_cast<size_t>(cid)] = descriptor;
        written_[static_cast<size_t>(cid)] = true;
    }

    [[nodiscard]] std::vector<uint8_t> finish_layer() const {
        require_begun();
        for (bool written : written_) {
            if (!written) {
                throw std::logic_error("BC position writer cannot finish with unwritten cells");
            }
        }

        const uint64_t descriptor_count = descriptors_.size();
        const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
        const uint64_t bucket_bytes = bucket_meta_stream_.size();
        const uint64_t rank_bytes = rank_payload_stream_.size();
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis_.family_count());
        const uint64_t descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC position descriptor table offset overflow"
        );
        const uint64_t bucket_offset = bc_checked_add_u64(
            descriptor_offset,
            descriptor_bytes,
            "BC position bucket stream offset overflow"
        );
        const uint64_t rank_offset = bc_checked_add_u64(
            bucket_offset,
            bucket_bytes,
            "BC position rank stream offset overflow"
        );

        BCPositionHeader header;
        header.family_unit = axis_.family_unit();
        header.axis_base_coord = axis_.axis_base_coord();
        header.family_count = axis_.family_count();
        header.layer_sum = axis_.layer_sum();
        header.axis_coord_table_bytes = axis_coord_bytes;
        header.descriptor_count = descriptor_count;
        header.descriptor_table_offset = descriptor_offset;
        header.descriptor_table_bytes = descriptor_bytes;
        header.bucket_meta_offset = bucket_offset;
        header.bucket_meta_bytes = bucket_bytes;
        header.rank_payload_offset = rank_offset;
        header.rank_payload_bytes = rank_bytes;

        std::vector<uint8_t> out;
        out.reserve(static_cast<size_t>(rank_offset + rank_bytes));
        bc_append_header(out, header);
        bc_append_axis_coord_table(out, axis_);
        for (const BCPositionCellDescriptor &descriptor : descriptors_) {
            bc_append_cell_descriptor(out, descriptor);
        }
        out.insert(out.end(), bucket_meta_stream_.begin(), bucket_meta_stream_.end());
        out.insert(out.end(), rank_payload_stream_.begin(), rank_payload_stream_.end());
        return out;
    }

private:
    void require_begun() const {
        if (!begun_) {
            throw std::logic_error("BC position writer layer has not begun");
        }
    }

    void require_cell_for_write(CellId cid) const {
        if (cid >= descriptors_.size()) {
            throw std::out_of_range("BC position writer cell id out of range");
        }
        if (written_[static_cast<size_t>(cid)]) {
            throw std::logic_error("BC position writer cell already written");
        }
    }

    BCFamilyTable axis_;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<bool> written_;
    std::vector<uint8_t> bucket_meta_stream_;
    std::vector<uint8_t> rank_payload_stream_;
    bool begun_ = false;
};

void bc_validate_position_payload_for_file(const FinalizedCellPayload &payload);

[[nodiscard]] uint64_t write_position_payloads_to_file(
    BCWritableFile &file,
    const BCFamilyTable &axis,
    const std::vector<const FinalizedCellPayload *> &cell_payloads,
    BCFileIOStats *stats = nullptr
);

[[nodiscard]] uint64_t write_position_payloads_to_file(
    BCWritableFile &file,
    const BCFamilyTable &axis,
    const std::vector<FinalizedCellPayload> &cell_payloads,
    BCFileIOStats *stats = nullptr
);
class BCPositionLayerReader {
public:
    BCPositionLayerReader() = default;

    BCPositionLayerReader(const std::vector<uint8_t> &bytes, const BCLut &lut) {
        open(bytes, lut);
    }

    BCPositionLayerReader(std::vector<uint8_t> &&bytes, const BCLut &lut) {
        open(std::move(bytes), lut);
    }

    void open(const std::vector<uint8_t> &bytes, const BCLut &lut) {
        bytes_ = bytes;
        lut_ = &lut;
        header_ = bc_read_header(bytes_);
        validate_header();
        read_axis();
        read_descriptors();
        read_bucket_entries();
        validate_descriptors();
    }

    void open(std::vector<uint8_t> &&bytes, const BCLut &lut) {
        bytes_ = std::move(bytes);
        lut_ = &lut;
        header_ = bc_read_header(bytes_);
        validate_header();
        read_axis();
        read_descriptors();
        read_bucket_entries();
        validate_descriptors();
    }

    [[nodiscard]] const BCPositionHeader &header() const {
        return header_;
    }

    [[nodiscard]] const BCFamilyTable &axis() const {
        return axis_;
    }

    [[nodiscard]] const BCLut &lut() const {
        if (lut_ == nullptr) {
            throw std::logic_error("BC position reader is not open");
        }
        return *lut_;
    }

    [[nodiscard]] const std::vector<uint8_t> &bytes() const {
        return bytes_;
    }

    [[nodiscard]] uint32_t cell_count() const {
        return static_cast<uint32_t>(descriptors_.size());
    }

    [[nodiscard]] const BCPositionCellDescriptor &descriptor(CellId cid) const {
        if (cid >= descriptors_.size()) {
            throw std::out_of_range("BC position reader cell id out of range");
        }
        return descriptors_[static_cast<size_t>(cid)];
    }

    [[nodiscard]] BCBucketEntryView bucket_entries_for_cell(CellId cid) const {
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.empty()) {
            return {};
        }
        const uint64_t bucket_index64 = desc.bucket_meta_offset / kBCPositionBucketEntryBytes;
        if (bucket_index64 > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC position cell bucket index exceeds uint32");
        }
        const size_t bucket_index = static_cast<size_t>(bucket_index64);
        if (bucket_index + desc.bucket_count > bucket_entries_.size()) {
            throw std::out_of_range("BC position cell bucket view exceeds bucket stream");
        }
        return BCBucketEntryView{
            bucket_entries_.data() + bucket_index,
            desc.bucket_count
        };
    }

    [[nodiscard]] BCRankPayloadView rank_payload_for_cell(CellId cid) const {
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.empty()) {
            return {};
        }
        if (desc.rank_payload_bytes > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC position cell rank payload exceeds uint32 lookup view");
        }
        const uint64_t payload_file_offset = header_.rank_payload_offset + desc.rank_payload_offset;
        bc_require_bytes(
            bytes_,
            payload_file_offset,
            desc.rank_payload_bytes,
            "BC position cell rank payload view exceeds file"
        );
        return BCRankPayloadView{
            bytes_.data() + static_cast<size_t>(payload_file_offset),
            static_cast<uint32_t>(desc.rank_payload_bytes)
        };
    }

    [[nodiscard]] BCLookupResult cold_lookup(CellId cid, uint64_t key, BucketRank rank) const {
        if (lut_ == nullptr) {
            throw std::logic_error("BC position reader is not open");
        }
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.empty()) {
            return {};
        }
        return lookup_finalized_cell(
            *lut_,
            bucket_entries_for_cell(cid),
            rank_payload_for_cell(cid),
            key,
            rank
        );
    }

private:
    void validate_header() const {
        if (header_.magic != kBCPositionMagic) {
            throw std::runtime_error("BC position file magic mismatch");
        }
        if (header_.format_version != kBCPositionFormatVersion) {
            throw std::runtime_error("BC position file format version mismatch");
        }
        if (header_.header_bytes != kBCPositionHeaderBytes) {
            throw std::runtime_error("BC position file header size mismatch");
        }
        if (header_.key_mode != kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256) {
            throw std::runtime_error("BC position file key mode mismatch");
        }
        if (header_.rank_prefix_bits != kBCRankPrefixBits) {
            throw std::runtime_error("BC position file rank prefix bits mismatch");
        }
        if (header_.rank_prefix_type != kBCPositionRankPrefixTypeUint16) {
            throw std::runtime_error("BC position file rank prefix type mismatch");
        }
        if (header_.rank_payload_align != 8U) {
            throw std::runtime_error("BC position file rank payload align mismatch");
        }
        if (header_.family_unit == 0U ||
            header_.family_unit > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("BC position file invalid family_unit");
        }
        if (header_.axis_base_coord > std::numeric_limits<FamilyCoord>::max() ||
            header_.family_count == 0U ||
            header_.family_count > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("BC position file invalid family axis");
        }
        if (header_.layer_sum > std::numeric_limits<LayerSum>::max()) {
            throw std::runtime_error("BC position file layer_sum exceeds LayerSum");
        }
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(header_.family_count);
        if (header_.axis_coord_table_bytes != axis_coord_bytes) {
            throw std::runtime_error("BC position file axis coord table byte size mismatch");
        }

        const uint64_t expected_desc_bytes =
            header_.descriptor_count * kBCPositionCellDescriptorBytes;
        if (header_.descriptor_table_bytes != expected_desc_bytes) {
            throw std::runtime_error("BC position file descriptor table byte size mismatch");
        }
        if ((header_.bucket_meta_bytes % kBCPositionBucketEntryBytes) != 0U) {
            throw std::runtime_error("BC position file bucket metadata stream is not entry-aligned");
        }
        const uint64_t expected_descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC position expected descriptor offset overflow"
        );
        if (header_.descriptor_table_offset != expected_descriptor_offset) {
            throw std::runtime_error("BC position file descriptor table offset mismatch");
        }
        const uint64_t metadata_end = bc_checked_add_u64(
            header_.descriptor_table_offset,
            header_.descriptor_table_bytes,
            "BC position metadata end overflow"
        );
        const uint64_t bucket_end = bc_checked_add_u64(
            header_.bucket_meta_offset,
            header_.bucket_meta_bytes,
            "BC position bucket metadata end overflow"
        );
        const uint64_t rank_end = bc_checked_add_u64(
            header_.rank_payload_offset,
            header_.rank_payload_bytes,
            "BC position rank payload end overflow"
        );
        if ((header_.bucket_meta_bytes != 0U && header_.bucket_meta_offset < metadata_end) ||
            (header_.rank_payload_bytes != 0U && header_.rank_payload_offset < metadata_end)) {
            throw std::runtime_error("BC position data stream overlaps metadata");
        }
        if (header_.bucket_meta_bytes != 0U && header_.rank_payload_bytes != 0U &&
            header_.bucket_meta_offset < rank_end && header_.rank_payload_offset < bucket_end) {
            throw std::runtime_error("BC position bucket and rank streams overlap");
        }
        const uint64_t data_end = std::max(bucket_end, rank_end);
        bc_require_bytes(bytes_, kBCPositionHeaderBytes, axis_coord_bytes,
            "BC position axis coord table exceeds file");
        bc_require_bytes(bytes_, header_.descriptor_table_offset, header_.descriptor_table_bytes,
            "BC position descriptor table exceeds file");
        bc_require_bytes(bytes_, header_.bucket_meta_offset, header_.bucket_meta_bytes,
            "BC position bucket metadata exceeds file");
        bc_require_bytes(bytes_, header_.rank_payload_offset, header_.rank_payload_bytes,
            "BC position rank payload exceeds file");
        if (data_end > bytes_.size()) {
            throw std::runtime_error("BC position file has trailing or missing bytes");
        }
        if (bytes_.size() - data_end >= 4096U) {
            throw std::runtime_error("BC position file has excessive trailing padding");
        }
    }

    void read_axis() {
        axis_ = BCFamilyTable(
            static_cast<LayerSum>(header_.layer_sum),
            static_cast<uint16_t>(header_.family_unit),
            bc_read_axis_coord_table(bytes_, header_)
        );
        if (axis_.axis_base_coord() != header_.axis_base_coord) {
            throw std::runtime_error("BC position file axis_base_coord does not match coord table");
        }
        const BCCellMatrix matrix(axis_);
        if (header_.descriptor_count != matrix.cell_count()) {
            throw std::runtime_error("BC position descriptor count does not match dense cell matrix");
        }
    }

    void read_descriptors() {
        descriptors_.clear();
        descriptors_.reserve(static_cast<size_t>(header_.descriptor_count));
        const size_t base = static_cast<size_t>(header_.descriptor_table_offset);
        for (uint64_t i = 0; i < header_.descriptor_count; ++i) {
            const size_t offset = base + static_cast<size_t>(i * kBCPositionCellDescriptorBytes);
            descriptors_.push_back(
                bc_read_cell_descriptor(
                    bytes_.data() + offset,
                    bytes_.size() - offset
                )
            );
        }
    }

    void read_bucket_entries() {
        bucket_entries_.clear();
        const uint64_t bucket_count = header_.bucket_meta_bytes / kBCPositionBucketEntryBytes;
        if (bucket_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("BC position bucket metadata stream has too many entries");
        }
        bucket_entries_.reserve(static_cast<size_t>(bucket_count));
        const size_t base = static_cast<size_t>(header_.bucket_meta_offset);
        for (uint64_t i = 0; i < bucket_count; ++i) {
            const size_t offset = base + static_cast<size_t>(i * kBCPositionBucketEntryBytes);
            bucket_entries_.push_back(bc_read_bucket_entry(bytes_.data() + offset));
        }
    }

    void validate_descriptors() const {
        for (const BCPositionCellDescriptor &desc : descriptors_) {
            if (desc.reserved0 != 0U) {
                throw std::runtime_error("BC position descriptor reserved field is non-zero");
            }
            if ((desc.flags_or_padding & ~kBCPositionCellFlagEmpty) != 0U) {
                throw std::runtime_error("BC position descriptor has unknown flags");
            }
            const uint64_t bucket_bytes =
                static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes;
            if ((desc.bucket_meta_offset % kBCPositionBucketEntryBytes) != 0U ||
                bc_checked_add_u64(desc.bucket_meta_offset, bucket_bytes,
                    "BC position descriptor bucket range overflow") > header_.bucket_meta_bytes) {
                throw std::runtime_error("BC position descriptor bucket range exceeds stream");
            }
            if (bc_checked_add_u64(desc.rank_payload_offset, desc.rank_payload_bytes,
                    "BC position descriptor rank payload range overflow") > header_.rank_payload_bytes) {
                throw std::runtime_error("BC position descriptor rank payload range exceeds stream");
            }
            if (desc.rank_payload_bytes > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("BC position descriptor rank payload exceeds cell-local uint32");
            }
            if (desc.empty()) {
                if (desc.bucket_count != 0U || desc.success_rows != 0U || desc.rank_payload_bytes != 0U) {
                    throw std::runtime_error("BC position empty descriptor carries non-empty data");
                }
                continue;
            }
            if (desc.bucket_count == 0U || desc.success_rows == 0U) {
                throw std::runtime_error("BC position non-empty descriptor has no bucket or success rows");
            }
        }
    }

    const BCLut *lut_ = nullptr;
    BCPositionHeader header_;
    BCFamilyTable axis_;
    std::vector<uint8_t> bytes_;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<BCBucketEntry> bucket_entries_;
};

void write_position_layer_to_file(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
);

[[nodiscard]] std::vector<uint8_t> read_position_layer_from_file(
    const std::filesystem::path &path
);

// Buffered file correctness wrapper. It currently reads the serialized position
// file into memory and reuses BCPositionLayerReader. Production large-layer
// loading should replace this with descriptor/cell streaming over BCReadableFile.
class BCPositionFileReader {
public:
    BCPositionFileReader(std::unique_ptr<BCReadableFile> file, const BCLut &lut);
    ~BCPositionFileReader();

    BCPositionFileReader(BCPositionFileReader &&) noexcept;
    BCPositionFileReader &operator=(BCPositionFileReader &&) noexcept;
    BCPositionFileReader(const BCPositionFileReader &) = delete;
    BCPositionFileReader &operator=(const BCPositionFileReader &) = delete;

    static BCPositionFileReader open_buffered(
        const std::filesystem::path &path,
        const BCLut &lut
    );

    static BCPositionFileReader open_direct_auto(
        const std::filesystem::path &path,
        const BCLut &lut,
        uint32_t queue_depth = 8U,
        bool overlapped = true
    );

    void open(std::unique_ptr<BCReadableFile> file, const BCLut &lut);

    [[nodiscard]] const BCPositionLayerReader &layer() const;
    [[nodiscard]] const std::vector<uint8_t> &bytes() const;
    [[nodiscard]] BCPositionLayerReader take_layer();

private:
    std::unique_ptr<BCReadableFile> file_;
    BCPositionLayerReader layer_;
};
} // namespace BC
