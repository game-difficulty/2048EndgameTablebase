#pragma once

#include "BCPositionFile.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace BC {

inline constexpr uint32_t kBCSuccessMagic = 0x46534342U; // "BCSF" little-endian.
inline constexpr uint32_t kBCSuccessFormatVersion = 1U;
inline constexpr uint32_t kBCSuccessHeaderBytes = 88U;
inline constexpr uint32_t kBCSuccessDTypeUint32 = 1U;

struct BCSuccessHeader {
    uint32_t magic = kBCSuccessMagic;
    uint32_t format_version = kBCSuccessFormatVersion;
    uint32_t header_bytes = kBCSuccessHeaderBytes;
    uint32_t dtype = kBCSuccessDTypeUint32;
    uint32_t row_width = 0U;
    uint32_t family_count = 0U;
    uint64_t descriptor_count = 0U;
    uint64_t payload_offset = kBCSuccessHeaderBytes;
    uint64_t payload_bytes = 0U;
    uint32_t position_key_mode = 0U;
    uint32_t family_unit = 0U;
    uint32_t axis_base_coord = 0U;
    uint32_t reserved32 = 0U;
    uint64_t layer_sum = 0U;
    uint64_t position_metadata_fingerprint = 0U;
    uint64_t reserved64 = 0U;
};

inline void bc_append_success_header(std::vector<uint8_t> &out, const BCSuccessHeader &header) {
    const size_t begin = out.size();
    bc_append_u32_le(out, header.magic);
    bc_append_u32_le(out, header.format_version);
    bc_append_u32_le(out, header.header_bytes);
    bc_append_u32_le(out, header.dtype);
    bc_append_u32_le(out, header.row_width);
    bc_append_u32_le(out, header.family_count);
    bc_append_u64_le(out, header.descriptor_count);
    bc_append_u64_le(out, header.payload_offset);
    bc_append_u64_le(out, header.payload_bytes);
    bc_append_u32_le(out, header.position_key_mode);
    bc_append_u32_le(out, header.family_unit);
    bc_append_u32_le(out, header.axis_base_coord);
    bc_append_u32_le(out, header.reserved32);
    bc_append_u64_le(out, header.layer_sum);
    bc_append_u64_le(out, header.position_metadata_fingerprint);
    bc_append_u64_le(out, header.reserved64);
    if (out.size() - begin != kBCSuccessHeaderBytes) {
        throw std::logic_error("BC success header serialized size mismatch");
    }
}

[[nodiscard]] inline BCSuccessHeader bc_read_success_header(const std::vector<uint8_t> &bytes) {
    bc_require_bytes(bytes, 0U, kBCSuccessHeaderBytes, "BC success file is smaller than header");
    const uint8_t *p = bytes.data();
    BCSuccessHeader header;
    header.magic = bc_load_u32_le(p + 0U);
    header.format_version = bc_load_u32_le(p + 4U);
    header.header_bytes = bc_load_u32_le(p + 8U);
    header.dtype = bc_load_u32_le(p + 12U);
    header.row_width = bc_load_u32_le(p + 16U);
    header.family_count = bc_load_u32_le(p + 20U);
    header.descriptor_count = load_u64_le(p + 24U);
    header.payload_offset = load_u64_le(p + 32U);
    header.payload_bytes = load_u64_le(p + 40U);
    header.position_key_mode = bc_load_u32_le(p + 48U);
    header.family_unit = bc_load_u32_le(p + 52U);
    header.axis_base_coord = bc_load_u32_le(p + 56U);
    header.reserved32 = bc_load_u32_le(p + 60U);
    header.layer_sum = load_u64_le(p + 64U);
    header.position_metadata_fingerprint = load_u64_le(p + 72U);
    header.reserved64 = load_u64_le(p + 80U);
    return header;
}

[[nodiscard]] inline uint64_t bc_success_position_fingerprint(const BCPositionLayerReader &position) {
    // Placeholder compatibility fingerprint for the memory-format stage.
    // It is intentionally simple and only guards against obvious mismatches.
    const BCPositionHeader &p = position.header();
    uint64_t value = 0x9E3779B97F4A7C15ULL;
    auto mix = [&value](uint64_t x) {
        value ^= x + 0x9E3779B97F4A7C15ULL + (value << 6U) + (value >> 2U);
    };
    mix(p.key_mode);
    mix(p.family_unit);
    mix(p.axis_base_coord);
    mix(p.family_count);
    mix(p.layer_sum);
    mix(position.cell_count());
    return value;
}

[[nodiscard]] inline uint64_t bc_success_total_values(
    const BCPositionLayerReader &position,
    uint32_t row_width
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC success row_width must be non-zero");
    }
    uint64_t rows = 0U;
    for (CellId cid = 0; cid < position.cell_count(); ++cid) {
        rows = bc_checked_add_u64(
            rows,
            position.descriptor(cid).success_rows,
            "BC success total rows overflow"
        );
    }
    if (rows > std::numeric_limits<uint64_t>::max() / row_width) {
        throw std::overflow_error("BC success total value count overflow");
    }
    return rows * row_width;
}

[[nodiscard]] inline uint64_t bc_success_expected_payload_bytes(
    const BCPositionLayerReader &position,
    uint32_t row_width
) {
    const uint64_t values = bc_success_total_values(position, row_width);
    if (values > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) {
        throw std::overflow_error("BC success payload byte count overflow");
    }
    return values * sizeof(uint32_t);
}

class BCSuccessLayerWriter {
public:
    BCSuccessLayerWriter() = default;

    void begin_layer(const BCPositionLayerReader &position, uint32_t row_width) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC success writer row_width must be non-zero");
        }
        position_ = &position;
        row_width_ = row_width;
        cell_values_.assign(position.cell_count(), {});
        written_.assign(position.cell_count(), false);
        begun_ = true;
    }

    void write_cell(CellId cid, const std::vector<uint32_t> &values) {
        require_begun();
        require_cell_for_write(cid);
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            throw std::invalid_argument("BC success writer cannot write values for an empty cell");
        }
        const uint64_t expected = expected_values_for_cell(desc);
        if (values.size() != expected) {
            throw std::invalid_argument("BC success writer cell value count mismatch");
        }
        cell_values_[static_cast<size_t>(cid)] = values;
        written_[static_cast<size_t>(cid)] = true;
    }

    void mark_empty_cell(CellId cid) {
        require_begun();
        require_cell_for_write(cid);
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        if (desc.success_rows != 0U || !desc.empty()) {
            throw std::invalid_argument("BC success writer can only mark empty position cells as empty");
        }
        written_[static_cast<size_t>(cid)] = true;
    }

    [[nodiscard]] std::vector<uint8_t> finish_layer() const {
        require_begun();
        for (bool written : written_) {
            if (!written) {
                throw std::logic_error("BC success writer cannot finish with unwritten cells");
            }
        }
        const uint64_t payload_bytes = bc_success_expected_payload_bytes(*position_, row_width_);
        BCSuccessHeader header;
        header.row_width = row_width_;
        header.family_count = position_->header().family_count;
        header.descriptor_count = position_->cell_count();
        header.payload_offset = kBCSuccessHeaderBytes;
        header.payload_bytes = payload_bytes;
        header.position_key_mode = position_->header().key_mode;
        header.family_unit = position_->header().family_unit;
        header.axis_base_coord = position_->header().axis_base_coord;
        header.layer_sum = position_->header().layer_sum;
        header.position_metadata_fingerprint = bc_success_position_fingerprint(*position_);

        std::vector<uint8_t> out;
        out.reserve(static_cast<size_t>(kBCSuccessHeaderBytes + payload_bytes));
        bc_append_success_header(out, header);
        for (CellId cid = 0; cid < position_->cell_count(); ++cid) {
            const BCPositionCellDescriptor &desc = position_->descriptor(cid);
            if (desc.success_rows == 0U) {
                continue;
            }
            const std::vector<uint32_t> &values = cell_values_[static_cast<size_t>(cid)];
            if (values.size() != expected_values_for_cell(desc)) {
                throw std::logic_error("BC success writer stored malformed cell values");
            }
            for (uint32_t value : values) {
                bc_append_u32_le(out, value);
            }
        }
        if (out.size() != kBCSuccessHeaderBytes + payload_bytes) {
            throw std::logic_error("BC success writer emitted unexpected byte size");
        }
        return out;
    }

private:
    void require_begun() const {
        if (!begun_ || position_ == nullptr) {
            throw std::logic_error("BC success writer layer has not begun");
        }
    }

    void require_cell_for_write(CellId cid) const {
        if (cid >= written_.size()) {
            throw std::out_of_range("BC success writer cell id out of range");
        }
        if (written_[static_cast<size_t>(cid)]) {
            throw std::logic_error("BC success writer cell already written");
        }
    }

    [[nodiscard]] uint64_t expected_values_for_cell(const BCPositionCellDescriptor &desc) const {
        return static_cast<uint64_t>(desc.success_rows) * row_width_;
    }

    const BCPositionLayerReader *position_ = nullptr;
    uint32_t row_width_ = 0U;
    std::vector<std::vector<uint32_t>> cell_values_;
    std::vector<bool> written_;
    bool begun_ = false;
};

class BCSuccessLayerReader {
public:
    BCSuccessLayerReader() = default;

    BCSuccessLayerReader(
        const std::vector<uint8_t> &bytes,
        const BCPositionLayerReader &position,
        uint32_t expected_row_width
    ) {
        open(bytes, position, expected_row_width);
    }

    void open(
        const std::vector<uint8_t> &bytes,
        const BCPositionLayerReader &position,
        uint32_t expected_row_width
    ) {
        bytes_ = bytes;
        position_ = &position;
        expected_row_width_ = expected_row_width;
        header_ = bc_read_success_header(bytes_);
        validate_header();
        build_cell_value_offsets();
    }

    [[nodiscard]] const BCSuccessHeader &header() const {
        return header_;
    }

    [[nodiscard]] uint32_t row_width() const {
        return header_.row_width;
    }

    [[nodiscard]] std::vector<uint32_t> read_cell(CellId cid) const {
        require_open();
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        const uint64_t value_count = static_cast<uint64_t>(desc.success_rows) * header_.row_width;
        if (value_count > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("BC success read_cell value count exceeds size_t");
        }
        std::vector<uint32_t> values;
        values.reserve(static_cast<size_t>(value_count));
        const uint64_t byte_offset = value_byte_offset(cid);
        for (uint64_t i = 0; i < value_count; ++i) {
            values.push_back(bc_load_u32_le(bytes_.data() + static_cast<size_t>(byte_offset + i * sizeof(uint32_t))));
        }
        return values;
    }

    [[nodiscard]] uint32_t read_value(CellId cid, uint32_t row, uint32_t lane = 0U) const {
        require_open();
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        if (row >= desc.success_rows) {
            throw std::out_of_range("BC success read_value row out of range");
        }
        if (lane >= header_.row_width) {
            throw std::out_of_range("BC success read_value lane out of range");
        }
        const uint64_t value_index =
            static_cast<uint64_t>(row) * header_.row_width + lane;
        const uint64_t byte_offset = value_byte_offset(cid) + value_index * sizeof(uint32_t);
        return bc_load_u32_le(bytes_.data() + static_cast<size_t>(byte_offset));
    }

private:
    void require_open() const {
        if (position_ == nullptr) {
            throw std::logic_error("BC success reader is not open");
        }
    }

    void validate_header() const {
        require_open();
        if (expected_row_width_ == 0U) {
            throw std::invalid_argument("BC success reader expected row_width must be non-zero");
        }
        if (header_.magic != kBCSuccessMagic) {
            throw std::runtime_error("BC success file magic mismatch");
        }
        if (header_.format_version != kBCSuccessFormatVersion) {
            throw std::runtime_error("BC success file format version mismatch");
        }
        if (header_.header_bytes != kBCSuccessHeaderBytes) {
            throw std::runtime_error("BC success file header size mismatch");
        }
        if (header_.dtype != kBCSuccessDTypeUint32) {
            throw std::runtime_error("BC success file dtype mismatch");
        }
        if (header_.row_width != expected_row_width_) {
            throw std::runtime_error("BC success file row_width mismatch");
        }
        if (header_.family_count != position_->header().family_count) {
            throw std::runtime_error("BC success file family_count mismatch");
        }
        if (header_.descriptor_count != position_->cell_count()) {
            throw std::runtime_error("BC success file descriptor_count mismatch");
        }
        if (header_.payload_offset != kBCSuccessHeaderBytes) {
            throw std::runtime_error("BC success file payload offset mismatch");
        }
        if (header_.position_key_mode != position_->header().key_mode ||
            header_.family_unit != position_->header().family_unit ||
            header_.axis_base_coord != position_->header().axis_base_coord ||
            header_.layer_sum != position_->header().layer_sum ||
            header_.position_metadata_fingerprint != bc_success_position_fingerprint(*position_)) {
            throw std::runtime_error("BC success file position metadata mismatch");
        }
        if (header_.reserved32 != 0U || header_.reserved64 != 0U) {
            throw std::runtime_error("BC success file reserved field is non-zero");
        }
        const uint64_t expected_payload_bytes =
            bc_success_expected_payload_bytes(*position_, header_.row_width);
        if (header_.payload_bytes != expected_payload_bytes) {
            throw std::runtime_error("BC success file payload byte size mismatch");
        }
        bc_require_bytes(bytes_, header_.payload_offset, header_.payload_bytes,
            "BC success payload exceeds file");
        const uint64_t expected_file_size = bc_checked_add_u64(
            header_.payload_offset,
            header_.payload_bytes,
            "BC success expected file size overflow"
        );
        if (expected_file_size != bytes_.size()) {
            throw std::runtime_error("BC success file has trailing or missing bytes");
        }
    }

    void build_cell_value_offsets() {
        cell_value_offsets_.assign(position_->cell_count() + 1U, 0U);
        uint64_t cursor = 0U;
        for (CellId cid = 0; cid < position_->cell_count(); ++cid) {
            cell_value_offsets_[static_cast<size_t>(cid)] = cursor;
            const uint64_t values =
                static_cast<uint64_t>(position_->descriptor(cid).success_rows) *
                header_.row_width;
            cursor = bc_checked_add_u64(cursor, values, "BC success cell value offset overflow");
        }
        cell_value_offsets_.back() = cursor;
        if (cursor > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
            cursor * sizeof(uint32_t) != header_.payload_bytes) {
            throw std::runtime_error("BC success derived cell offsets do not match payload bytes");
        }
    }

    [[nodiscard]] uint64_t value_byte_offset(CellId cid) const {
        if (cid >= position_->cell_count()) {
            throw std::out_of_range("BC success reader cell id out of range");
        }
        const uint64_t value_offset = cell_value_offsets_[static_cast<size_t>(cid)];
        if (value_offset > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) {
            throw std::overflow_error("BC success value byte offset overflow");
        }
        const uint64_t byte_offset = bc_checked_add_u64(
            header_.payload_offset,
            value_offset * sizeof(uint32_t),
            "BC success value byte offset overflow"
        );
        const uint64_t cell_values =
            static_cast<uint64_t>(position_->descriptor(cid).success_rows) *
            header_.row_width;
        if (cell_values > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) {
            throw std::overflow_error("BC success cell byte count overflow");
        }
        const uint64_t cell_bytes = cell_values * sizeof(uint32_t);
        bc_require_bytes(bytes_, byte_offset, cell_bytes, "BC success cell payload exceeds file");
        return byte_offset;
    }

    const BCPositionLayerReader *position_ = nullptr;
    uint32_t expected_row_width_ = 0U;
    BCSuccessHeader header_;
    std::vector<uint8_t> bytes_;
    std::vector<uint64_t> cell_value_offsets_;
};

inline void write_success_layer_to_file(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    write_bytes_to_buffered_file(path, bytes);
}

[[nodiscard]] inline std::vector<uint8_t> read_success_layer_from_file(
    const std::filesystem::path &path
) {
    return read_bytes_from_buffered_file(path);
}

// Buffered file correctness wrapper. It currently reads the serialized success
// file into memory and reuses BCSuccessLayerReader. Production solve hot paths
// should use a streaming/cell-view reader backed by BCReadableFile.
class BCSuccessFileReader {
public:
    BCSuccessFileReader(
        std::unique_ptr<BCReadableFile> file,
        const BCPositionLayerReader &position,
        uint32_t expected_row_width
    ) {
        open(std::move(file), position, expected_row_width);
    }

    static BCSuccessFileReader open_buffered(
        const std::filesystem::path &path,
        const BCPositionLayerReader &position,
        uint32_t expected_row_width
    ) {
        return BCSuccessFileReader(
            std::make_unique<BCBufferedFileReader>(path),
            position,
            expected_row_width
        );
    }

    void open(
        std::unique_ptr<BCReadableFile> file,
        const BCPositionLayerReader &position,
        uint32_t expected_row_width
    ) {
        if (!file) {
            throw std::invalid_argument("BC success file reader file is null");
        }
        file_ = std::move(file);
        const uint64_t byte_count = file_->size();
        if (byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC success file exceeds addressable memory vector size");
        }
        bytes_.assign(static_cast<size_t>(byte_count), 0U);
        if (!bytes_.empty()) {
            file_->read_at(0U, bytes_.data(), byte_count);
        }
        reader_.open(bytes_, position, expected_row_width);
    }

    [[nodiscard]] const BCSuccessLayerReader &reader() const {
        return reader_;
    }

    [[nodiscard]] const std::vector<uint8_t> &bytes() const {
        return bytes_;
    }

private:
    std::unique_ptr<BCReadableFile> file_;
    std::vector<uint8_t> bytes_;
    BCSuccessLayerReader reader_;
};

} // namespace BC
