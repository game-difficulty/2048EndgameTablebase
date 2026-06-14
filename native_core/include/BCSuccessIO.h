#pragma once

#include "BCFileIO.h"
#include "BCPositionCellLoader.h"
#include "BCPositionFile.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace BC {

inline constexpr uint32_t kBCSuccessMagic = 0x46534342U; // "BCSF" little-endian.
inline constexpr uint32_t kBCSuccessFormatVersion = 1U;
inline constexpr uint32_t kBCSuccessHeaderBytes = 88U;
inline constexpr uint32_t kBCSuccessDTypeUint32 = 1U;

enum class BCSuccessDTypeMode : uint32_t {
    UInt32 = 1U,
    UInt64 = 2U,
    Float32 = 3U,
    Float64 = 4U,
    OneMinusFloat32 = 5U,
    OneMinusFloat64 = 6U,
};

[[nodiscard]] inline BCSuccessDTypeMode bc_success_dtype_from_u32(uint32_t value) {
    switch (static_cast<BCSuccessDTypeMode>(value)) {
        case BCSuccessDTypeMode::UInt32:
        case BCSuccessDTypeMode::UInt64:
        case BCSuccessDTypeMode::Float32:
        case BCSuccessDTypeMode::Float64:
        case BCSuccessDTypeMode::OneMinusFloat32:
        case BCSuccessDTypeMode::OneMinusFloat64:
            return static_cast<BCSuccessDTypeMode>(value);
    }
    throw std::runtime_error("BC success unsupported dtype mode");
}

[[nodiscard]] inline uint32_t bc_success_dtype_value_size(BCSuccessDTypeMode mode) {
    switch (mode) {
        case BCSuccessDTypeMode::UInt64:
        case BCSuccessDTypeMode::Float64:
        case BCSuccessDTypeMode::OneMinusFloat64:
            return 8U;
        case BCSuccessDTypeMode::UInt32:
        case BCSuccessDTypeMode::Float32:
        case BCSuccessDTypeMode::OneMinusFloat32:
            return 4U;
    }
    throw std::runtime_error("BC success unsupported dtype mode");
}

[[nodiscard]] inline uint32_t bc_success_dtype_value_size(uint32_t mode) {
    return bc_success_dtype_value_size(bc_success_dtype_from_u32(mode));
}

template <typename T>
[[nodiscard]] inline BCSuccessDTypeMode bc_success_default_dtype_for_type() {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return BCSuccessDTypeMode::UInt32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return BCSuccessDTypeMode::UInt64;
    } else if constexpr (std::is_same_v<T, float>) {
        return BCSuccessDTypeMode::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return BCSuccessDTypeMode::Float64;
    } else {
        static_assert(
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "unsupported BC success value type"
        );
    }
}

template <typename T>
[[nodiscard]] inline bool bc_success_dtype_matches_type(BCSuccessDTypeMode mode) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return mode == BCSuccessDTypeMode::UInt32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return mode == BCSuccessDTypeMode::UInt64;
    } else if constexpr (std::is_same_v<T, float>) {
        return mode == BCSuccessDTypeMode::Float32 ||
               mode == BCSuccessDTypeMode::OneMinusFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        return mode == BCSuccessDTypeMode::Float64 ||
               mode == BCSuccessDTypeMode::OneMinusFloat64;
    } else {
        return false;
    }
}

template <typename T>
inline void bc_append_success_value_le(std::vector<uint8_t> &out, T value) {
    static_assert(
        std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
        std::is_same_v<T, float> || std::is_same_v<T, double>,
        "unsupported BC success value type"
    );
    if constexpr (std::is_same_v<T, uint32_t>) {
        bc_append_u32_le(out, value);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        bc_append_u64_le(out, value);
    } else if constexpr (std::is_same_v<T, float>) {
        uint32_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(value));
        bc_append_u32_le(out, bits);
    } else if constexpr (std::is_same_v<T, double>) {
        uint64_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(value));
        bc_append_u64_le(out, bits);
    }
}

template <typename T>
[[nodiscard]] inline T bc_load_success_value_le(const uint8_t *data) {
    static_assert(
        std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
        std::is_same_v<T, float> || std::is_same_v<T, double>,
        "unsupported BC success value type"
    );
    if constexpr (std::is_same_v<T, uint32_t>) {
        return bc_load_u32_le(data);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return load_u64_le(data);
    } else if constexpr (std::is_same_v<T, float>) {
        const uint32_t bits = bc_load_u32_le(data);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    } else {
        const uint64_t bits = load_u64_le(data);
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
}

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

template <class PositionReader>
[[nodiscard]] inline uint64_t bc_success_position_fingerprint_for(const PositionReader &position) {
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
    mix(p.axis_coord_table_bytes);
    mix(p.layer_sum);
    mix(position.cell_count());
    for (FamilyCoord coord : position.axis().coords()) {
        mix(coord);
    }
    return value;
}

[[nodiscard]] inline uint64_t bc_success_position_fingerprint(const BCPositionLayerReader &position) {
    return bc_success_position_fingerprint_for(position);
}

template <class PositionReader>
[[nodiscard]] inline uint64_t bc_success_total_values_for(
    const PositionReader &position,
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
    uint32_t row_width,
    BCSuccessDTypeMode dtype = BCSuccessDTypeMode::UInt32
) {
    const uint64_t values = bc_success_total_values_for(position, row_width);
    const uint32_t value_size = bc_success_dtype_value_size(dtype);
    if (values > std::numeric_limits<uint64_t>::max() / value_size) {
        throw std::overflow_error("BC success payload byte count overflow");
    }
    return values * value_size;
}

template <class PositionReader>
[[nodiscard]] inline uint64_t bc_success_expected_payload_bytes_for(
    const PositionReader &position,
    uint32_t row_width,
    BCSuccessDTypeMode dtype = BCSuccessDTypeMode::UInt32
) {
    const uint64_t values = bc_success_total_values_for(position, row_width);
    const uint32_t value_size = bc_success_dtype_value_size(dtype);
    if (values > std::numeric_limits<uint64_t>::max() / value_size) {
        throw std::overflow_error("BC success payload byte count overflow");
    }
    return values * value_size;
}

class BCSuccessLayerWriter {
public:
    BCSuccessLayerWriter() = default;

    void begin_layer(const BCPositionLayerReader &position, uint32_t row_width) {
        begin_layer(position, row_width, BCSuccessDTypeMode::UInt32);
    }

    void begin_layer(const BCPositionStreamingReader &position, uint32_t row_width) {
        begin_layer(position, row_width, BCSuccessDTypeMode::UInt32);
    }

    void begin_layer(
        const BCPositionLayerReader &position,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        begin_layer_impl(position, row_width, dtype);
    }

    void begin_layer(
        const BCPositionStreamingReader &position,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        begin_layer_impl(position, row_width, dtype);
    }

    template <class PositionReader>
    void begin_layer_impl(
        const PositionReader &position,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC success writer row_width must be non-zero");
        }
        (void)bc_success_dtype_value_size(dtype);
        position_header_ = position.header();
        position_fingerprint_ = bc_success_position_fingerprint_for(position);
        row_width_ = row_width;
        dtype_ = dtype;
        const uint32_t cell_count = position.cell_count();
        descriptors_.assign(cell_count, BCPositionCellDescriptor{});
        for (CellId cid = 0U; cid < cell_count; ++cid) {
            descriptors_[static_cast<size_t>(cid)] = position.descriptor(cid);
        }
        payload_bytes_ = bc_success_expected_payload_bytes_for(position, row_width, dtype);
        cell_bytes_.assign(cell_count, {});
        written_.assign(cell_count, false);
        begun_ = true;
    }

    void write_cell(CellId cid, const std::vector<uint32_t> &values) {
        write_cell_typed<uint32_t>(cid, values);
    }

    template <typename T>
    void write_cell_typed(CellId cid, const std::vector<T> &values) {
        require_begun();
        require_cell_for_write(cid);
        if (!bc_success_dtype_matches_type<T>(dtype_)) {
            throw std::invalid_argument("BC success writer value type does not match dtype");
        }
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            throw std::invalid_argument("BC success writer cannot write values for an empty cell");
        }
        const uint64_t expected = expected_values_for_cell(desc);
        if (values.size() != expected) {
            throw std::invalid_argument("BC success writer cell value count mismatch");
        }
        std::vector<uint8_t> bytes;
        const uint32_t value_size = bc_success_dtype_value_size(dtype_);
        if (expected > std::numeric_limits<size_t>::max() / value_size) {
            throw std::overflow_error("BC success writer cell byte count exceeds size_t");
        }
        bytes.reserve(static_cast<size_t>(expected) * value_size);
        for (T value : values) {
            bc_append_success_value_le(bytes, value);
        }
        cell_bytes_[static_cast<size_t>(cid)] = std::move(bytes);
        written_[static_cast<size_t>(cid)] = true;
    }

    void write_cell_raw(CellId cid, const std::vector<uint8_t> &bytes) {
        require_begun();
        require_cell_for_write(cid);
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            throw std::invalid_argument("BC success writer cannot write raw values for an empty cell");
        }
        const uint64_t expected = expected_bytes_for_cell(desc);
        if (bytes.size() != expected) {
            throw std::invalid_argument("BC success writer raw cell byte count mismatch");
        }
        cell_bytes_[static_cast<size_t>(cid)] = bytes;
        written_[static_cast<size_t>(cid)] = true;
    }

    void mark_empty_cell(CellId cid) {
        require_begun();
        require_cell_for_write(cid);
        const BCPositionCellDescriptor &desc = descriptor(cid);
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
        BCSuccessHeader header;
        header.dtype = static_cast<uint32_t>(dtype_);
        header.row_width = row_width_;
        header.family_count = position_header_.family_count;
        header.descriptor_count = descriptors_.size();
        header.payload_offset = kBCSuccessHeaderBytes;
        header.payload_bytes = payload_bytes_;
        header.position_key_mode = position_header_.key_mode;
        header.family_unit = position_header_.family_unit;
        header.axis_base_coord = position_header_.axis_base_coord;
        header.layer_sum = position_header_.layer_sum;
        header.position_metadata_fingerprint = position_fingerprint_;

        std::vector<uint8_t> out;
        out.reserve(static_cast<size_t>(kBCSuccessHeaderBytes + payload_bytes_));
        bc_append_success_header(out, header);
        for (CellId cid = 0; cid < descriptors_.size(); ++cid) {
            const BCPositionCellDescriptor &desc = descriptors_[static_cast<size_t>(cid)];
            if (desc.success_rows == 0U) {
                continue;
            }
            const std::vector<uint8_t> &bytes = cell_bytes_[static_cast<size_t>(cid)];
            if (bytes.size() != expected_bytes_for_cell(desc)) {
                throw std::logic_error("BC success writer stored malformed cell values");
            }
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
        if (out.size() != kBCSuccessHeaderBytes + payload_bytes_) {
            throw std::logic_error("BC success writer emitted unexpected byte size");
        }
        return out;
    }

private:
    void require_begun() const {
        if (!begun_) {
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

    [[nodiscard]] const BCPositionCellDescriptor &descriptor(CellId cid) const {
        if (cid >= descriptors_.size()) {
            throw std::out_of_range("BC success writer descriptor cell id out of range");
        }
        return descriptors_[static_cast<size_t>(cid)];
    }

    [[nodiscard]] uint64_t expected_values_for_cell(const BCPositionCellDescriptor &desc) const {
        return static_cast<uint64_t>(desc.success_rows) * row_width_;
    }

    [[nodiscard]] uint64_t expected_bytes_for_cell(const BCPositionCellDescriptor &desc) const {
        const uint64_t values = expected_values_for_cell(desc);
        const uint32_t value_size = bc_success_dtype_value_size(dtype_);
        if (values > std::numeric_limits<uint64_t>::max() / value_size) {
            throw std::overflow_error("BC success writer cell byte count overflow");
        }
        return values * value_size;
    }

    BCPositionHeader position_header_;
    uint64_t position_fingerprint_ = 0U;
    uint64_t payload_bytes_ = 0U;
    uint32_t row_width_ = 0U;
    BCSuccessDTypeMode dtype_ = BCSuccessDTypeMode::UInt32;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<std::vector<uint8_t>> cell_bytes_;
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

    [[nodiscard]] BCSuccessDTypeMode dtype_mode() const {
        return bc_success_dtype_from_u32(header_.dtype);
    }

    [[nodiscard]] uint32_t value_size() const {
        return bc_success_dtype_value_size(dtype_mode());
    }

    [[nodiscard]] std::vector<uint32_t> read_cell(CellId cid) const {
        require_dtype_type<uint32_t>();
        return read_cell_typed<uint32_t>(cid);
    }

    template <typename T>
    [[nodiscard]] std::vector<T> read_cell_typed(CellId cid) const {
        require_open();
        require_dtype_type<T>();
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        const uint64_t value_count = static_cast<uint64_t>(desc.success_rows) * header_.row_width;
        if (value_count > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("BC success read_cell value count exceeds size_t");
        }
        std::vector<T> values;
        values.reserve(static_cast<size_t>(value_count));
        const uint64_t byte_offset = value_byte_offset(cid);
        for (uint64_t i = 0; i < value_count; ++i) {
            values.push_back(
                bc_load_success_value_le<T>(
                    bytes_.data() + static_cast<size_t>(byte_offset + i * sizeof(T))
                )
            );
        }
        return values;
    }

    [[nodiscard]] std::vector<uint8_t> read_cell_raw(CellId cid) const {
        require_open();
        const uint64_t offset = value_byte_offset(cid);
        const uint64_t bytes = cell_byte_count(cid);
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC success raw cell byte count exceeds size_t");
        }
        return std::vector<uint8_t>(
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset + bytes)
        );
    }

    [[nodiscard]] uint32_t read_value(CellId cid, uint32_t row, uint32_t lane = 0U) const {
        require_dtype_type<uint32_t>();
        return read_value_typed<uint32_t>(cid, row, lane);
    }

    template <typename T>
    [[nodiscard]] T read_value_typed(CellId cid, uint32_t row, uint32_t lane = 0U) const {
        require_open();
        require_dtype_type<T>();
        const BCPositionCellDescriptor &desc = position_->descriptor(cid);
        if (row >= desc.success_rows) {
            throw std::out_of_range("BC success read_value row out of range");
        }
        if (lane >= header_.row_width) {
            throw std::out_of_range("BC success read_value lane out of range");
        }
        const uint64_t value_index =
            static_cast<uint64_t>(row) * header_.row_width + lane;
        const uint32_t value_size = bc_success_dtype_value_size(dtype_mode());
        const uint64_t byte_offset = value_byte_offset(cid) + value_index * value_size;
        return bc_load_success_value_le<T>(bytes_.data() + static_cast<size_t>(byte_offset));
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
        (void)bc_success_dtype_from_u32(header_.dtype);
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
            bc_success_expected_payload_bytes(*position_, header_.row_width, dtype_mode());
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
        const uint32_t value_size = bc_success_dtype_value_size(dtype_mode());
        if (cursor > std::numeric_limits<uint64_t>::max() / value_size ||
            cursor * value_size != header_.payload_bytes) {
            throw std::runtime_error("BC success derived cell offsets do not match payload bytes");
        }
    }

    [[nodiscard]] uint64_t value_byte_offset(CellId cid) const {
        if (cid >= position_->cell_count()) {
            throw std::out_of_range("BC success reader cell id out of range");
        }
        const uint64_t value_offset = cell_value_offsets_[static_cast<size_t>(cid)];
        const uint32_t value_size = bc_success_dtype_value_size(dtype_mode());
        if (value_offset > std::numeric_limits<uint64_t>::max() / value_size) {
            throw std::overflow_error("BC success value byte offset overflow");
        }
        const uint64_t byte_offset = bc_checked_add_u64(
            header_.payload_offset,
            value_offset * value_size,
            "BC success value byte offset overflow"
        );
        const uint64_t cell_bytes = cell_byte_count(cid);
        bc_require_bytes(bytes_, byte_offset, cell_bytes, "BC success cell payload exceeds file");
        return byte_offset;
    }

    [[nodiscard]] uint64_t cell_byte_count(CellId cid) const {
        const uint64_t cell_values =
            static_cast<uint64_t>(position_->descriptor(cid).success_rows) *
            header_.row_width;
        const uint32_t value_size = bc_success_dtype_value_size(dtype_mode());
        if (cell_values > std::numeric_limits<uint64_t>::max() / value_size) {
            throw std::overflow_error("BC success cell byte count overflow");
        }
        return cell_values * value_size;
    }

    template <typename T>
    void require_dtype_type() const {
        if (!bc_success_dtype_matches_type<T>(dtype_mode())) {
            throw std::runtime_error("BC success reader value type does not match dtype");
        }
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

inline void bc_success_accumulate_file_stats(BCFileIOStats *dst, const BCFileIOStats &src) {
    if (dst == nullptr) {
        return;
    }
    if (dst->request_count > std::numeric_limits<uint64_t>::max() - src.request_count ||
        dst->requested_bytes > std::numeric_limits<uint64_t>::max() - src.requested_bytes ||
        dst->backend_io_count > std::numeric_limits<uint64_t>::max() - src.backend_io_count ||
        dst->backend_bytes > std::numeric_limits<uint64_t>::max() - src.backend_bytes) {
        throw std::overflow_error("BC success streaming write stats overflow");
    }
    dst->request_count += src.request_count;
    dst->requested_bytes += src.requested_bytes;
    dst->backend_io_count += src.backend_io_count;
    dst->backend_bytes += src.backend_bytes;
}

class BCSequentialSuccessWriteStager {
public:
    explicit BCSequentialSuccessWriteStager(BCWritableFile &file, BCFileIOStats *stats = nullptr)
        : file_(file), stats_(stats) {
        direct_mode_ = file_.mode() == BCFileIOMode::Direct;
        const uint32_t preferred_alignment = file_.preferred_write_alignment();
        alignment_ = preferred_alignment == 0U ? 1U : preferred_alignment;
        stage_bytes_ = kTargetStageBytes - (kTargetStageBytes % alignment_);
        if (stage_bytes_ == 0U) {
            stage_bytes_ = alignment_;
        }
        max_pending_chunks_ = direct_mode_ ? kDirectPendingChunksPerGroup : 1U;
        group_count_ = direct_mode_ ? kDirectPipelineGroups : 1U;
        for (uint32_t group = 0U; group < group_count_; ++group) {
            groups_[group].buffers.resize(max_pending_chunks_);
        }
        if (stats_ != nullptr) {
            *stats_ = {};
        }
    }

    void append(const void *data, uint64_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC success streaming write append pointer is null");
        }
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            ensure_active_buffer();
            const uint64_t available = stage_bytes_ - active_bytes_;
            if (available == 0U) {
                finalize_active_buffer(true);
                continue;
            }
            const uint64_t take = std::min<uint64_t>(available, remaining);
            PendingGroup &group = groups_[active_group_];
            StageBuffer &buffer = group.buffers[group.pending_count];
            std::memcpy(buffer.data + static_cast<size_t>(active_bytes_), cursor, static_cast<size_t>(take));
            active_bytes_ += take;
            cursor += take;
            remaining -= take;
            if (active_bytes_ == stage_bytes_ && remaining != 0U) {
                finalize_active_buffer(true);
            }
        }
    }

    template <typename T>
    void append_values(const std::vector<T> &values) {
        static_assert(
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "unsupported BC success streaming write type"
        );
        if (values.empty()) {
            return;
        }
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        append(values.data(), static_cast<uint64_t>(values.size()) * sizeof(T));
#else
        for (T value : values) {
            std::vector<uint8_t> bytes;
            bytes.reserve(sizeof(T));
            bc_append_success_value_le(bytes, value);
            append(bytes.data(), bytes.size());
        }
#endif
    }

    void finish() {
        if (active_bytes_ != 0U) {
            finalize_active_buffer(false);
        }
        flush_active_group(false);
        wait_for_in_flight();
        if (!direct_mode_) {
            file_.flush();
        }
    }

private:
    static constexpr uint64_t kTargetStageBytes = 16ULL * 1024ULL * 1024ULL;
    static constexpr uint32_t kDirectPipelineGroups = 2U;
    static constexpr uint32_t kDirectPendingChunksPerGroup = 2U;

    struct StageBuffer {
        std::vector<uint8_t> storage;
        uint8_t *data = nullptr;
    };

    struct PendingGroup {
        std::vector<StageBuffer> buffers;
        std::array<uint64_t, kDirectPendingChunksPerGroup> offsets{};
        std::array<uint64_t, kDirectPendingChunksPerGroup> bytes{};
        uint32_t pending_count = 0U;
    };

    static uint8_t *align_pointer(uint8_t *ptr, uint64_t alignment) {
        if (alignment <= 1U) {
            return ptr;
        }
        const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t rem = raw % alignment;
        if (rem == 0U) {
            return ptr;
        }
        return reinterpret_cast<uint8_t *>(raw + (alignment - rem));
    }

    void ensure_active_buffer() {
        PendingGroup &group = groups_[active_group_];
        if (group.pending_count >= max_pending_chunks_) {
            flush_active_group(true);
        }
        PendingGroup &active = groups_[active_group_];
        StageBuffer &buffer = active.buffers[active.pending_count];
        if (buffer.data != nullptr) {
            return;
        }
        if (stage_bytes_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - alignment_) {
            throw std::overflow_error("BC success streaming write stage buffer size exceeds size_t");
        }
        buffer.storage.resize(static_cast<size_t>(stage_bytes_ + alignment_));
        buffer.data = align_pointer(buffer.storage.data(), alignment_);
    }

    void finalize_active_buffer(bool allow_async) {
        if (active_bytes_ == 0U) {
            return;
        }
        ensure_active_buffer();
        PendingGroup &group = groups_[active_group_];
        group.offsets[group.pending_count] = file_cursor_;
        group.bytes[group.pending_count] = active_bytes_;
        file_cursor_ = bc_checked_add_u64(
            file_cursor_,
            active_bytes_,
            "BC success streaming write cursor overflow"
        );
        active_bytes_ = 0U;
        ++group.pending_count;
        if (group.pending_count >= max_pending_chunks_) {
            flush_active_group(allow_async);
        }
    }

    [[nodiscard]] std::vector<BCFileWriteRequest> build_group_requests(PendingGroup &group) {
        std::vector<BCFileWriteRequest> requests;
        requests.reserve(group.pending_count);
        for (uint32_t i = 0U; i < group.pending_count; ++i) {
            requests.push_back(BCFileWriteRequest{
                group.offsets[i],
                group.buffers[i].data,
                group.bytes[i]
            });
        }
        return requests;
    }

    void flush_active_group(bool allow_async) {
        PendingGroup &group = groups_[active_group_];
        if (group.pending_count == 0U) {
            return;
        }
        std::vector<BCFileWriteRequest> requests = build_group_requests(group);
        group.pending_count = 0U;
        if (!direct_mode_) {
            BCFileIOStats local;
            file_.write_many(requests, stats_ == nullptr ? nullptr : &local);
            bc_success_accumulate_file_stats(stats_, local);
            return;
        }

        wait_for_in_flight();
        if (!allow_async) {
            BCFileIOStats local;
            file_.write_many(requests, stats_ == nullptr ? nullptr : &local);
            bc_success_accumulate_file_stats(stats_, local);
            return;
        }

        in_flight_ = std::async(
            std::launch::async,
            [this, requests = std::move(requests)]() mutable {
                BCFileIOStats local;
                file_.write_many(requests, &local);
                return local;
            }
        );
        active_group_ = (active_group_ + 1U) % group_count_;
    }

    void wait_for_in_flight() {
        if (!in_flight_.valid()) {
            return;
        }
        BCFileIOStats local = in_flight_.get();
        bc_success_accumulate_file_stats(stats_, local);
    }

    BCWritableFile &file_;
    BCFileIOStats *stats_ = nullptr;
    std::array<PendingGroup, kDirectPipelineGroups> groups_{};
    std::future<BCFileIOStats> in_flight_;
    uint64_t stage_bytes_ = kTargetStageBytes;
    uint64_t active_bytes_ = 0U;
    uint64_t file_cursor_ = 0U;
    uint64_t alignment_ = 1U;
    uint32_t max_pending_chunks_ = 1U;
    uint32_t group_count_ = 1U;
    uint32_t active_group_ = 0U;
    bool direct_mode_ = false;
};

template <typename T>
uint64_t write_success_values_to_file(
    BCWritableFile &file,
    const BCPositionLayerReader &position,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    const std::vector<T> &values,
    BCFileIOStats *stats = nullptr
) {
    if (!bc_success_dtype_matches_type<T>(dtype)) {
        throw std::invalid_argument("BC success streaming writer value type does not match dtype");
    }
    const uint64_t expected_values = bc_success_total_values_for(position, row_width);
    if (expected_values != static_cast<uint64_t>(values.size())) {
        throw std::invalid_argument("BC success streaming writer value count mismatch");
    }
    const uint32_t value_size = bc_success_dtype_value_size(dtype);
    if (expected_values > std::numeric_limits<uint64_t>::max() / value_size) {
        throw std::overflow_error("BC success streaming writer payload byte count overflow");
    }
    const uint64_t payload_bytes = expected_values * value_size;

    BCSuccessHeader header;
    header.dtype = static_cast<uint32_t>(dtype);
    header.row_width = row_width;
    header.family_count = position.header().family_count;
    header.descriptor_count = position.cell_count();
    header.payload_offset = kBCSuccessHeaderBytes;
    header.payload_bytes = payload_bytes;
    header.position_key_mode = position.header().key_mode;
    header.family_unit = position.header().family_unit;
    header.axis_base_coord = position.header().axis_base_coord;
    header.layer_sum = position.header().layer_sum;
    header.position_metadata_fingerprint = bc_success_position_fingerprint(position);

    std::vector<uint8_t> header_bytes;
    header_bytes.reserve(kBCSuccessHeaderBytes);
    bc_append_success_header(header_bytes, header);

    const uint64_t logical_size = bc_checked_add_u64(
        kBCSuccessHeaderBytes,
        payload_bytes,
        "BC success streaming writer logical size overflow"
    );
    file.prepare_full_overwrite(logical_size);
    BCSequentialSuccessWriteStager stager(file, stats);
    stager.append(header_bytes.data(), header_bytes.size());
    stager.append_values(values);
    stager.finish();
    return logical_size;
}

[[nodiscard]] inline std::vector<uint8_t> read_success_layer_from_file(
    const std::filesystem::path &path
) {
    return read_bytes_from_buffered_file(path);
}

struct BCSuccessLoadStats {
    uint64_t requested_extents = 0U;
    uint64_t coalesced_extents = 0U;
    uint64_t requested_bytes = 0U;
    uint64_t read_bytes = 0U;
    uint64_t backend_read_ops = 0U;
    uint64_t backend_read_bytes = 0U;
};

struct BCLoadedSuccessCell {
    CellId cid = 0U;
    uint32_t dtype = kBCSuccessDTypeUint32;
    uint32_t row_width = 0U;
    uint32_t success_rows = 0U;
    std::vector<uint8_t> raw_bytes;
    std::vector<uint32_t> values;

    [[nodiscard]] bool empty() const {
        return success_rows == 0U;
    }

    [[nodiscard]] BCSuccessDTypeMode dtype_mode() const {
        return bc_success_dtype_from_u32(dtype);
    }

    [[nodiscard]] uint32_t value_size() const {
        return bc_success_dtype_value_size(dtype_mode());
    }

    [[nodiscard]] uint32_t read_value(uint32_t row, uint32_t lane = 0U) const {
        if (dtype_mode() != BCSuccessDTypeMode::UInt32) {
            throw std::runtime_error("BC loaded success cell read_value requires uint32 dtype");
        }
        return read_value_typed<uint32_t>(row, lane);
    }

    template <typename T>
    [[nodiscard]] T read_value_typed(uint32_t row, uint32_t lane = 0U) const {
        if (!bc_success_dtype_matches_type<T>(dtype_mode())) {
            throw std::runtime_error("BC loaded success cell value type does not match dtype");
        }
        if (row >= success_rows) {
            throw std::out_of_range("BC loaded success cell row out of range");
        }
        if (lane >= row_width) {
            throw std::out_of_range("BC loaded success cell lane out of range");
        }
        const uint64_t index = static_cast<uint64_t>(row) * row_width + lane;
        const uint64_t offset = index * value_size();
        if (offset > raw_bytes.size() || value_size() > raw_bytes.size() - offset) {
            throw std::out_of_range("BC loaded success cell value index exceeds payload");
        }
        return bc_load_success_value_le<T>(raw_bytes.data() + static_cast<size_t>(offset));
    }
};

class BCSuccessStreamingReader {
public:
    BCSuccessStreamingReader() = default;

    template <class PositionReader>
    BCSuccessStreamingReader(
        std::unique_ptr<BCReadableFile> file,
        const PositionReader &position,
        uint32_t expected_row_width
    ) {
        open(std::move(file), position, expected_row_width);
    }

    template <class PositionReader>
    static BCSuccessStreamingReader open_buffered(
        const std::filesystem::path &path,
        const PositionReader &position,
        uint32_t expected_row_width
    ) {
        return BCSuccessStreamingReader(
            std::make_unique<BCBufferedFileReader>(path),
            position,
            expected_row_width
        );
    }

    template <class PositionReader>
    void open(
        std::unique_ptr<BCReadableFile> file,
        const PositionReader &position,
        uint32_t expected_row_width
    ) {
        if (!file) {
            throw std::invalid_argument("BC success streaming reader file is null");
        }
        if (expected_row_width == 0U) {
            throw std::invalid_argument("BC success streaming reader expected row_width must be non-zero");
        }
        file_ = std::move(file);
        file_size_ = file_->size();
        expected_row_width_ = expected_row_width;
        capture_position_metadata(position);
        if (file_size_ < kBCSuccessHeaderBytes) {
            throw std::runtime_error("BC success streaming file is smaller than header");
        }
        std::vector<uint8_t> header_bytes(kBCSuccessHeaderBytes);
        file_->read_at(0U, header_bytes.data(), header_bytes.size());
        header_ = bc_read_success_header(header_bytes);
        validate_header();
        build_cell_value_offsets();
    }

    [[nodiscard]] const BCSuccessHeader &header() const {
        return header_;
    }

    [[nodiscard]] uint32_t row_width() const {
        return header_.row_width;
    }

    [[nodiscard]] BCSuccessDTypeMode dtype_mode() const {
        return bc_success_dtype_from_u32(header_.dtype);
    }

    [[nodiscard]] uint32_t value_size() const {
        return bc_success_dtype_value_size(dtype_mode());
    }

    [[nodiscard]] uint32_t cell_count() const {
        return checked_u32_size(success_rows_.size(), "BC success streaming cell count exceeds uint32");
    }

    [[nodiscard]] uint64_t file_size() const {
        return file_size_;
    }

    [[nodiscard]] std::vector<uint8_t> read_all_bytes(BCFileIOStats *stats = nullptr) const {
        require_open();
        if (stats != nullptr) {
            *stats = {};
        }
        if (file_size_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC success streaming file exceeds addressable memory vector size");
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(file_size_));
        if (!bytes.empty()) {
            file_->read_many(
                std::vector<BCFileReadRequest>{
                    BCFileReadRequest{0U, bytes.data(), file_size_}
                },
                stats
            );
        }
        return bytes;
    }

    [[nodiscard]] BCLoadedSuccessCell load_cell(
        CellId cid,
        BCSuccessLoadStats *stats = nullptr
    ) const {
        const std::vector<BCLoadedSuccessCell> cells = load_cells(std::vector<CellId>{cid}, stats);
        if (cells.size() != 1U) {
            throw std::logic_error("BC success streaming load_cell internal result size mismatch");
        }
        return cells.front();
    }

    [[nodiscard]] std::vector<BCLoadedSuccessCell> load_cells(
        const std::vector<CellId> &cids,
        BCSuccessLoadStats *stats = nullptr
    ) const {
        require_open();
        if (stats != nullptr) {
            *stats = {};
        }

        std::vector<BCLoadedSuccessCell> cells;
        cells.reserve(cids.size());
        std::vector<ExtentRequest> requests;
        requests.reserve(cids.size());
        for (size_t index = 0U; index < cids.size(); ++index) {
            const CellId cid = cids[index];
            require_cell(cid);
            BCLoadedSuccessCell cell;
            cell.cid = cid;
            cell.dtype = header_.dtype;
            cell.row_width = header_.row_width;
            cell.success_rows = success_rows_[static_cast<size_t>(cid)];
            const uint64_t value_count =
                static_cast<uint64_t>(cell.success_rows) * header_.row_width;
            if (value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC success streaming cell value count exceeds size_t");
            }
            const uint64_t byte_count = checked_value_bytes(value_count);
            if (byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC success streaming cell byte count exceeds size_t");
            }
            cell.raw_bytes.assign(static_cast<size_t>(byte_count), 0U);
            if (dtype_mode() == BCSuccessDTypeMode::UInt32) {
                cell.values.assign(static_cast<size_t>(value_count), 0U);
            }
            cells.push_back(std::move(cell));
            if (value_count == 0U) {
                continue;
            }
            const uint64_t offset = value_byte_offset(cid);
            append_request(requests, index, offset, byte_count);
        }

        if (stats != nullptr) {
            stats->requested_extents = requests.size();
            for (const ExtentRequest &request : requests) {
                stats->requested_bytes = bc_checked_add_u64(
                    stats->requested_bytes,
                    request.bytes,
                    "BC success streaming requested bytes overflow"
                );
            }
        }

        std::vector<LoadedRange> ranges = read_coalesced_ranges(requests, stats);
        for (const ExtentRequest &request : requests) {
            const LoadedRange &range = find_loaded_range(ranges, request.offset, request.bytes);
            const uint64_t in_range_offset = request.offset - range.offset;
            if (in_range_offset > range.bytes.size() ||
                request.bytes > range.bytes.size() - in_range_offset) {
                throw std::logic_error("BC success streaming loaded range does not cover request");
            }
            const uint8_t *src = range.bytes.data() + static_cast<size_t>(in_range_offset);
            BCLoadedSuccessCell &cell = cells[request.cell_index];
            if (request.bytes != cell.raw_bytes.size()) {
                throw std::logic_error("BC success streaming loaded raw byte count mismatch");
            }
            std::copy(src, src + static_cast<size_t>(request.bytes), cell.raw_bytes.begin());
            if (dtype_mode() == BCSuccessDTypeMode::UInt32) {
                parse_u32_values(cell.raw_bytes.data(), request.bytes, cell.values);
            }
        }
        return cells;
    }

    [[nodiscard]] uint32_t read_value(CellId cid, uint32_t row, uint32_t lane = 0U) const {
        if (dtype_mode() != BCSuccessDTypeMode::UInt32) {
            throw std::runtime_error("BC success streaming read_value requires uint32 dtype");
        }
        return read_value_typed<uint32_t>(cid, row, lane);
    }

    template <typename T>
    [[nodiscard]] T read_value_typed(CellId cid, uint32_t row, uint32_t lane = 0U) const {
        require_open();
        if (!bc_success_dtype_matches_type<T>(dtype_mode())) {
            throw std::runtime_error("BC success streaming value type does not match dtype");
        }
        require_cell(cid);
        const uint32_t rows = success_rows_[static_cast<size_t>(cid)];
        if (row >= rows) {
            throw std::out_of_range("BC success streaming read_value row out of range");
        }
        if (lane >= header_.row_width) {
            throw std::out_of_range("BC success streaming read_value lane out of range");
        }
        const uint64_t value_index =
            static_cast<uint64_t>(row) * header_.row_width + lane;
        const uint64_t byte_offset = bc_checked_add_u64(
            value_byte_offset(cid),
            checked_value_bytes(value_index),
            "BC success streaming read_value byte offset overflow"
        );
        std::array<uint8_t, sizeof(T)> bytes = {};
        BCFileIOStats stats;
        file_->read_many(
            std::vector<BCFileReadRequest>{
                BCFileReadRequest{byte_offset, bytes.data(), bytes.size()}
            },
            &stats
        );
        (void)stats;
        return bc_load_success_value_le<T>(bytes.data());
    }

private:
    struct PositionMetadata {
        uint32_t key_mode = 0U;
        uint32_t family_unit = 0U;
        uint32_t axis_base_coord = 0U;
        uint32_t family_count = 0U;
        uint64_t layer_sum = 0U;
        uint64_t cell_count = 0U;
        uint64_t fingerprint = 0U;
    };

    struct ExtentRequest {
        uint64_t offset = 0U;
        uint64_t bytes = 0U;
        size_t cell_index = 0U;
    };

    struct LoadedRange {
        uint64_t offset = 0U;
        std::vector<uint8_t> bytes;
    };

    template <class PositionReader>
    void capture_position_metadata(const PositionReader &position) {
        const BCPositionHeader &p = position.header();
        position_ = PositionMetadata{
            p.key_mode,
            p.family_unit,
            p.axis_base_coord,
            p.family_count,
            p.layer_sum,
            position.cell_count(),
            bc_success_position_fingerprint_for(position)
        };
        if (position_->cell_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC success streaming position cell count exceeds size_t");
        }
        success_rows_.assign(static_cast<size_t>(position_->cell_count), 0U);
        for (CellId cid = 0; cid < position.cell_count(); ++cid) {
            success_rows_[static_cast<size_t>(cid)] = position.descriptor(cid).success_rows;
        }
    }

    void require_open() const {
        if (!file_ || !position_.has_value()) {
            throw std::logic_error("BC success streaming reader is not open");
        }
    }

    void require_cell(CellId cid) const {
        if (cid >= success_rows_.size()) {
            throw std::out_of_range("BC success streaming cell id out of range");
        }
    }

    void validate_header() const {
        require_open();
        const PositionMetadata &position = *position_;
        if (header_.magic != kBCSuccessMagic) {
            throw std::runtime_error("BC success streaming file magic mismatch");
        }
        if (header_.format_version != kBCSuccessFormatVersion) {
            throw std::runtime_error("BC success streaming file format version mismatch");
        }
        if (header_.header_bytes != kBCSuccessHeaderBytes) {
            throw std::runtime_error("BC success streaming file header size mismatch");
        }
        (void)bc_success_dtype_from_u32(header_.dtype);
        if (header_.row_width != expected_row_width_) {
            throw std::runtime_error("BC success streaming file row_width mismatch");
        }
        if (header_.family_count != position.family_count) {
            throw std::runtime_error("BC success streaming file family_count mismatch");
        }
        if (header_.descriptor_count != position.cell_count) {
            throw std::runtime_error("BC success streaming file descriptor_count mismatch");
        }
        if (header_.payload_offset != kBCSuccessHeaderBytes) {
            throw std::runtime_error("BC success streaming file payload offset mismatch");
        }
        if (header_.position_key_mode != position.key_mode ||
            header_.family_unit != position.family_unit ||
            header_.axis_base_coord != position.axis_base_coord ||
            header_.layer_sum != position.layer_sum ||
            header_.position_metadata_fingerprint != position.fingerprint) {
            throw std::runtime_error("BC success streaming file position metadata mismatch");
        }
        if (header_.reserved32 != 0U || header_.reserved64 != 0U) {
            throw std::runtime_error("BC success streaming file reserved field is non-zero");
        }
        const uint64_t expected_payload_bytes = expected_payload_bytes_from_success_rows(header_.row_width);
        if (header_.payload_bytes != expected_payload_bytes) {
            throw std::runtime_error("BC success streaming file payload byte size mismatch");
        }
        require_file_range(header_.payload_offset, header_.payload_bytes,
            "BC success streaming payload exceeds file");
        const uint64_t expected_file_size = bc_checked_add_u64(
            header_.payload_offset,
            header_.payload_bytes,
            "BC success streaming expected file size overflow"
        );
        if (expected_file_size != file_size_) {
            throw std::runtime_error("BC success streaming file has trailing or missing bytes");
        }
    }

    [[nodiscard]] uint64_t expected_payload_bytes_from_success_rows(uint32_t row_width) const {
        if (row_width == 0U) {
            throw std::invalid_argument("BC success streaming row_width must be non-zero");
        }
        uint64_t rows = 0U;
        for (uint32_t success_rows : success_rows_) {
            rows = bc_checked_add_u64(rows, success_rows, "BC success streaming total rows overflow");
        }
        if (rows > std::numeric_limits<uint64_t>::max() / row_width) {
            throw std::overflow_error("BC success streaming total value count overflow");
        }
        return checked_value_bytes(rows * row_width);
    }

    void build_cell_value_offsets() {
        cell_value_offsets_.assign(success_rows_.size() + 1U, 0U);
        uint64_t cursor = 0U;
        for (size_t cid = 0U; cid < success_rows_.size(); ++cid) {
            cell_value_offsets_[cid] = cursor;
            const uint64_t values =
                static_cast<uint64_t>(success_rows_[cid]) * header_.row_width;
            cursor = bc_checked_add_u64(cursor, values, "BC success streaming cell offset overflow");
        }
        cell_value_offsets_.back() = cursor;
        if (checked_value_bytes(cursor) != header_.payload_bytes) {
            throw std::runtime_error("BC success streaming derived cell offsets do not match payload bytes");
        }
    }

    [[nodiscard]] uint64_t value_byte_offset(CellId cid) const {
        require_cell(cid);
        const uint64_t value_offset = cell_value_offsets_[static_cast<size_t>(cid)];
        const uint64_t payload_delta = checked_value_bytes(value_offset);
        return bc_checked_add_u64(
            header_.payload_offset,
            payload_delta,
            "BC success streaming value byte offset overflow"
        );
    }

    [[nodiscard]] uint64_t checked_value_bytes(uint64_t value_count) const {
        const uint32_t value_size = bc_success_dtype_value_size(dtype_mode());
        if (value_count > std::numeric_limits<uint64_t>::max() / value_size) {
            throw std::overflow_error("BC success streaming value byte count overflow");
        }
        return value_count * value_size;
    }

    void require_file_range(uint64_t offset, uint64_t bytes, const char *label) const {
        const uint64_t end = bc_checked_add_u64(offset, bytes, label);
        if (end > file_size_) {
            throw std::out_of_range(label);
        }
    }

    void append_request(
        std::vector<ExtentRequest> &requests,
        size_t cell_index,
        uint64_t offset,
        uint64_t bytes
    ) const {
        if (bytes == 0U) {
            return;
        }
        require_file_range(offset, bytes, "BC success streaming request exceeds file");
        requests.push_back(ExtentRequest{offset, bytes, cell_index});
    }

    [[nodiscard]] std::vector<LoadedRange> read_coalesced_ranges(
        std::vector<ExtentRequest> requests,
        BCSuccessLoadStats *stats
    ) const {
        std::sort(
            requests.begin(),
            requests.end(),
            [](const ExtentRequest &lhs, const ExtentRequest &rhs) {
                if (lhs.offset != rhs.offset) {
                    return lhs.offset < rhs.offset;
                }
                return lhs.bytes < rhs.bytes;
            }
        );

        std::vector<BCFileExtent> extents;
        for (const ExtentRequest &request : requests) {
            const uint64_t end = bc_checked_add_u64(
                request.offset,
                request.bytes,
                "BC success streaming request end overflow"
            );
            if (!extents.empty()) {
                BCFileExtent &last = extents.back();
                const uint64_t last_end = bc_checked_add_u64(
                    last.offset,
                    last.bytes,
                    "BC success streaming coalesced extent end overflow"
                );
                if (request.offset <= last_end) {
                    if (end > last_end) {
                        last.bytes = end - last.offset;
                    }
                    continue;
                }
            }
            extents.push_back(BCFileExtent{request.offset, request.bytes});
        }

        std::vector<LoadedRange> ranges;
        ranges.reserve(extents.size());
        for (const BCFileExtent &extent : extents) {
            if (extent.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC success streaming coalesced extent exceeds size_t");
            }
            LoadedRange range;
            range.offset = extent.offset;
            range.bytes.assign(static_cast<size_t>(extent.bytes), 0U);
            if (stats != nullptr) {
                ++stats->coalesced_extents;
                stats->read_bytes = bc_checked_add_u64(
                    stats->read_bytes,
                    extent.bytes,
                    "BC success streaming read byte count overflow"
                );
            }
            ranges.push_back(std::move(range));
        }

        std::vector<BCFileReadRequest> read_requests;
        read_requests.reserve(ranges.size());
        for (LoadedRange &range : ranges) {
            read_requests.push_back(BCFileReadRequest{
                range.offset,
                range.bytes.data(),
                static_cast<uint64_t>(range.bytes.size())
            });
        }
        BCFileIOStats io_stats;
        file_->read_many(read_requests, &io_stats);
        if (stats != nullptr) {
            stats->backend_read_ops = io_stats.backend_io_count;
            stats->backend_read_bytes = io_stats.backend_bytes;
        }
        return ranges;
    }

    [[nodiscard]] const LoadedRange &find_loaded_range(
        const std::vector<LoadedRange> &ranges,
        uint64_t offset,
        uint64_t bytes
    ) const {
        const auto it = std::upper_bound(
            ranges.begin(),
            ranges.end(),
            offset,
            [](uint64_t value, const LoadedRange &range) {
                return value < range.offset;
            }
        );
        if (it == ranges.begin()) {
            throw std::logic_error("BC success streaming request is before loaded ranges");
        }
        const LoadedRange &range = *(it - 1);
        const uint64_t range_end = bc_checked_add_u64(
            range.offset,
            range.bytes.size(),
            "BC success streaming loaded range end overflow"
        );
        const uint64_t request_end = bc_checked_add_u64(
            offset,
            bytes,
            "BC success streaming request end overflow"
        );
        if (offset < range.offset || request_end > range_end) {
            throw std::logic_error("BC success streaming request is not covered by loaded range");
        }
        return range;
    }

    static void parse_u32_values(
        const uint8_t *data,
        uint64_t bytes,
        std::vector<uint32_t> &out
    ) {
        if ((bytes % sizeof(uint32_t)) != 0U) {
            throw std::runtime_error("BC success streaming loaded value bytes are not u32-aligned");
        }
        const uint64_t value_count = bytes / sizeof(uint32_t);
        if (value_count != out.size()) {
            throw std::runtime_error("BC success streaming loaded value count mismatch");
        }
        for (uint64_t i = 0U; i < value_count; ++i) {
            out[static_cast<size_t>(i)] =
                bc_load_u32_le(data + static_cast<size_t>(i * sizeof(uint32_t)));
        }
    }

    std::unique_ptr<BCReadableFile> file_;
    uint64_t file_size_ = 0U;
    uint32_t expected_row_width_ = 0U;
    BCSuccessHeader header_;
    std::optional<PositionMetadata> position_;
    std::vector<uint32_t> success_rows_;
    std::vector<uint64_t> cell_value_offsets_;
};

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
