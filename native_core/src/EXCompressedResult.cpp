#include "EXCompressedResult.h"

#include "EXFrozenLayer.h"
#include "EXPrefix36Runtime.h"
#include "NativeLzma.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace EXCompressedResult {

namespace {

namespace fs = std::filesystem;

constexpr char kPrefix36CompressedMagic[8] = {'E', 'X', '3', '6', 'C', 'Z', '1', '\0'};
constexpr char kPrefix36LayerMagic[8] = {'E', 'X', 'P', '3', '6', 'B', 'K', '\0'};
constexpr char kPrefix36LutMagic[8] = {'E', 'X', 'P', '3', '6', 'L', 'T', '\0'};
constexpr uint32_t kPrefix36LayerVersion = 5U;
constexpr uint32_t kPrefix36LutVersion = 2U;
constexpr uint16_t kPrefix36InvalidRank = 0xFFFFU;
constexpr double kPrefix36FixedScale = 4000000000.0;
constexpr double kPrefix36UInt64Scale = 1600000000000000000.0;

enum class Prefix36DTypeMode : uint32_t {
    UInt32 = 0,
    UInt64 = 1,
    Float32 = 2,
    Float64 = 3,
    OneMinusFloat32 = 4,
    OneMinusFloat64 = 5,
};

struct Prefix36CompressedHeader {
    char magic[8];
    uint32_t version = kPrefix36LayerVersion;
    uint32_t success_kind = 0;
    uint32_t dtype_mode = 0;
    uint32_t value_size = 0;
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved16 = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t bucket_count = 0;
    uint64_t success_value_count = 0;
    uint64_t live_board_count = 0;
    uint32_t bucket_block_buckets = 0;
    uint32_t success_block_values = 0;
    uint64_t bucket_block_count = 0;
    uint64_t success_block_count = 0;
    uint64_t bucket_dir_offset = 0;
    uint64_t success_dir_offset = 0;
    uint64_t data_offset = 0;
    uint64_t original_file_size = 0;
    uint64_t reserved0 = 0;
};

struct Prefix36LayerHeader {
    char magic[8];
    uint32_t version = kPrefix36LayerVersion;
    uint32_t success_kind = 0;
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t dtype_mode = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved16 = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t success_value_count = 0;
    uint64_t live_board_count = 0;
    uint64_t value_size = 0;
    uint64_t reserved1 = 0;
};

struct Prefix36LutHeader {
    char magic[8];
    uint32_t version = kPrefix36LutVersion;
    uint32_t reserved32 = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved16a = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    int8_t max_counts[16]{};
    uint32_t required_suffix24 = 0;
    uint8_t table_for_high[16]{};
    uint8_t packed_table0 = 0xFFU;
    uint8_t packed_table1 = 0xFFU;
    uint16_t reserved16 = 0;
    uint32_t rank_table_variant_count = 0;
    uint64_t valid_suffix_mask_count = 0;
    uint64_t rank_table_count = 0;
    uint64_t rank_table_values = 0;
    uint64_t packed_rank_pair_values = 0;
    uint64_t packed_meta_values = 0;
    uint64_t size_table_values = 0;
    uint64_t offset_table_values = 0;
    uint64_t unrank_array_values = 0;
    uint64_t high_base_values = 0;
    uint64_t valid_suffix_count = 0;
};

struct Prefix36BucketBlockEntry {
    uint64_t first_prefix36 = 0;
    uint64_t last_prefix36 = 0;
    uint32_t first_bucket_index = 0;
    uint32_t bucket_count = 0;
    uint64_t compressed_offset = 0;
    uint64_t compressed_size = 0;
    uint64_t raw_size = 0;
};

struct Prefix36SuccessBlockEntry {
    uint64_t first_value_index = 0;
    uint32_t value_count = 0;
    uint32_t value_size = 0;
    uint64_t compressed_offset = 0;
    uint64_t compressed_size = 0;
    uint64_t raw_size = 0;
};

struct Prefix36BucketBlockRawHeader {
    uint32_t bucket_count = 0;
    uint32_t small_bitmap_bytes = 0;
    uint32_t large_bitmap_words = 0;
    uint32_t reserved = 0;
};

struct Prefix36LutRuntime {
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    std::vector<uint32_t> size_table;
    std::vector<uint32_t> offset_table;
    std::vector<uint32_t> unrank_array;
    std::vector<uint16_t> high_base;
    std::vector<uint8_t> table_for_high;
    std::vector<std::vector<uint16_t>> rank_tables;
    std::vector<uint32_t> packed_rank_pair_table;
    uint8_t packed_table0 = 0xFFU;
    uint8_t packed_table1 = 0xFFU;
};

struct Prefix36LutFileLayout {
    uint64_t valid_suffix_masks_offset = 0;
    uint64_t rank_tables_offset = 0;
    uint64_t packed_rank_pair_offset = 0;
    uint64_t packed_meta_offset = 0;
    uint64_t size_table_offset = 0;
    uint64_t offset_table_offset = 0;
    uint64_t unrank_array_offset = 0;
    uint64_t high_base_offset = 0;
};

struct Prefix36LutPointIndex {
    Prefix36LutHeader header{};
    Prefix36LutFileLayout layout{};
};

double wall_time_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    return std::chrono::duration<double>(clock::now() - epoch).count();
}

uint64_t file_size_u64(const std::string &path) {
    return static_cast<uint64_t>(fs::file_size(path));
}

template <typename T>
void append_value(std::vector<uint8_t> &out, const T &value) {
    const auto *ptr = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), ptr, ptr + sizeof(T));
}

void append_bytes(std::vector<uint8_t> &out, const void *data, size_t bytes) {
    if (bytes == 0U) {
        return;
    }
    const auto *ptr = reinterpret_cast<const uint8_t *>(data);
    out.insert(out.end(), ptr, ptr + bytes);
}

template <typename T>
T load_unaligned(const uint8_t *ptr) {
    T value{};
    std::memcpy(&value, ptr, sizeof(T));
    return value;
}

void write_zero_bytes(std::fstream &out, uint64_t bytes) {
    std::vector<uint8_t> zero(1024 * 1024, 0U);
    while (bytes != 0U) {
        const uint64_t chunk = std::min<uint64_t>(bytes, zero.size());
        out.write(reinterpret_cast<const char *>(zero.data()), static_cast<std::streamsize>(chunk));
        bytes -= chunk;
    }
}

void write_at(std::fstream &out, uint64_t offset, const void *data, uint64_t bytes) {
    out.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error("failed to write EX compressed zbook");
    }
}

template <typename T>
T read_one_at(const std::string &path, uint64_t offset) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed to read file value: " + path);
    }
    return value;
}

template <typename T>
T read_one_from(std::ifstream &in, uint64_t offset, const std::string &path) {
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed to read file value: " + path);
    }
    return value;
}

std::vector<uint8_t> read_range_from(std::ifstream &in, uint64_t offset, uint64_t size, const std::string &path) {
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size != 0U) {
        in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("failed to read file range: " + path);
        }
    }
    return data;
}

std::vector<uint8_t> compress_block_or_throw(const uint8_t *data, size_t size, int level) {
    std::vector<uint8_t> compressed = compress_xz_block_native(data, size, level);
    if (compressed.empty() && size != 0U) {
        throw std::runtime_error("failed to compress EX compressed zbook block");
    }
    return compressed;
}

uint32_t compression_worker_count(uint64_t block_count) {
    if (block_count <= 1U) {
        return 1U;
    }
    uint32_t hw = std::thread::hardware_concurrency();
    if (hw == 0U) {
        hw = 4U;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(block_count, hw));
}

struct CompressedPrefix36BucketBlock {
    Prefix36BucketBlockEntry entry{};
    std::vector<uint8_t> compressed;
    uint64_t raw_size = 0;
};

struct CompressedPrefix36SuccessBlock {
    Prefix36SuccessBlockEntry entry{};
    std::vector<uint8_t> compressed;
    uint64_t raw_size = 0;
};

std::vector<uint8_t> decompress_block_or_throw_from(
    std::ifstream &in,
    const std::string &path,
    uint64_t offset,
    uint64_t compressed_size,
    uint64_t raw_size
) {
    std::vector<uint8_t> compressed = read_range_from(in, offset, compressed_size, path);
    std::vector<uint8_t> raw = decompress_xz_block_native(compressed.data(), compressed.size());
    if (raw.size() != raw_size) {
        throw std::runtime_error("EX compressed zbook block decompressed to unexpected size");
    }
    return raw;
}

SuccessRateKind storage_kind_for_prefix36_mode(Prefix36DTypeMode mode) {
    switch (mode) {
        case Prefix36DTypeMode::UInt64:
            return SuccessRateKind::UInt64;
        case Prefix36DTypeMode::Float32:
        case Prefix36DTypeMode::OneMinusFloat32:
            return SuccessRateKind::Float32;
        case Prefix36DTypeMode::Float64:
        case Prefix36DTypeMode::OneMinusFloat64:
            return SuccessRateKind::Float64;
        case Prefix36DTypeMode::UInt32:
        default:
            return SuccessRateKind::UInt32;
    }
}

uint32_t prefix36_value_size(Prefix36DTypeMode mode) {
    switch (mode) {
        case Prefix36DTypeMode::UInt64:
        case Prefix36DTypeMode::Float64:
        case Prefix36DTypeMode::OneMinusFloat64:
            return sizeof(uint64_t);
        case Prefix36DTypeMode::Float32:
        case Prefix36DTypeMode::OneMinusFloat32:
        case Prefix36DTypeMode::UInt32:
        default:
            return sizeof(uint32_t);
    }
}

uint32_t prefix36_fixed_from_unit(double value) {
    if (value <= 0.0) {
        return 0U;
    }
    if (value >= 1.0) {
        return 4000000000U;
    }
    return static_cast<uint32_t>(value * kPrefix36FixedScale);
}

uint32_t prefix36_fixed_from_raw_bits(uint64_t raw, Prefix36DTypeMode mode) {
    switch (mode) {
        case Prefix36DTypeMode::UInt64:
            return prefix36_fixed_from_unit(static_cast<double>(raw) / kPrefix36UInt64Scale);
        case Prefix36DTypeMode::Float32: {
            const uint32_t bits = static_cast<uint32_t>(raw);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return prefix36_fixed_from_unit(static_cast<double>(value));
        }
        case Prefix36DTypeMode::Float64: {
            double value = 0.0;
            std::memcpy(&value, &raw, sizeof(value));
            return prefix36_fixed_from_unit(value);
        }
        case Prefix36DTypeMode::OneMinusFloat32: {
            const uint32_t bits = static_cast<uint32_t>(raw);
            float value = -1.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return prefix36_fixed_from_unit(static_cast<double>(value) + 1.0);
        }
        case Prefix36DTypeMode::OneMinusFloat64: {
            double value = -1.0;
            std::memcpy(&value, &raw, sizeof(value));
            return prefix36_fixed_from_unit(value + 1.0);
        }
        case Prefix36DTypeMode::UInt32:
        default:
            return static_cast<uint32_t>(raw);
    }
}

uint64_t prefix36_raw_bits_from_fixed(uint32_t fixed, Prefix36DTypeMode mode) {
    const double unit = static_cast<double>(fixed) / kPrefix36FixedScale;
    switch (mode) {
        case Prefix36DTypeMode::UInt64:
            return static_cast<uint64_t>(unit * kPrefix36UInt64Scale);
        case Prefix36DTypeMode::Float32: {
            const float value = static_cast<float>(unit);
            uint32_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case Prefix36DTypeMode::Float64: {
            const double value = unit;
            uint64_t bits = 0ULL;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case Prefix36DTypeMode::OneMinusFloat32: {
            const float value = static_cast<float>(unit - 1.0);
            uint32_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case Prefix36DTypeMode::OneMinusFloat64: {
            const double value = unit - 1.0;
            uint64_t bits = 0ULL;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case Prefix36DTypeMode::UInt32:
        default:
            return fixed;
    }
}

double prefix36_numeric_from_fixed(uint32_t fixed, Prefix36DTypeMode mode) {
    const double unit = static_cast<double>(fixed) / kPrefix36FixedScale;
    switch (mode) {
        case Prefix36DTypeMode::OneMinusFloat32:
        case Prefix36DTypeMode::OneMinusFloat64:
            return unit - 1.0;
        case Prefix36DTypeMode::UInt32:
        case Prefix36DTypeMode::UInt64:
        case Prefix36DTypeMode::Float32:
        case Prefix36DTypeMode::Float64:
        default:
            return unit;
    }
}

uint32_t prefix36_tile_value(uint32_t tile) {
    return tile == 0U ? 0U : (1U << tile);
}

uint32_t prefix36_low24_sum(uint32_t low24) {
    uint32_t sum = 0U;
    for (uint32_t i = 0; i < 6U; ++i) {
        sum += prefix36_tile_value((low24 >> (i * 4U)) & 0xFU);
    }
    return sum;
}

uint32_t prefix36_suffix_group(uint32_t suffix28) {
    const uint32_t high = suffix28 >> 24U;
    return (prefix36_low24_sum(suffix28 & 0xFFFFFFU) + prefix36_tile_value(high)) >> 1U;
}

Prefix36DTypeMode prefix36_dtype_mode_from_u32(uint32_t value) {
    switch (static_cast<Prefix36DTypeMode>(value)) {
        case Prefix36DTypeMode::UInt32:
        case Prefix36DTypeMode::UInt64:
        case Prefix36DTypeMode::Float32:
        case Prefix36DTypeMode::Float64:
        case Prefix36DTypeMode::OneMinusFloat32:
        case Prefix36DTypeMode::OneMinusFloat64:
            return static_cast<Prefix36DTypeMode>(value);
    }
    throw std::runtime_error("unsupported prefix36 dtype mode");
}

void validate_prefix36_layer_header(const Prefix36LayerHeader &header) {
    if (std::memcmp(header.magic, kPrefix36LayerMagic, sizeof(kPrefix36LayerMagic)) != 0) {
        throw std::runtime_error("invalid prefix36 zbook magic");
    }
    if (header.version != kPrefix36LayerVersion) {
        throw std::runtime_error("unsupported prefix36 zbook version");
    }
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(header.dtype_mode);
    if (header.success_kind != static_cast<uint32_t>(storage_kind_for_prefix36_mode(mode)) ||
        header.value_size != prefix36_value_size(mode)) {
        throw std::runtime_error("invalid prefix36 zbook dtype header");
    }
}

Prefix36LayerHeader read_prefix36_layer_header(const std::string &path) {
    const Prefix36LayerHeader header = read_one_at<Prefix36LayerHeader>(path, 0);
    validate_prefix36_layer_header(header);
    return header;
}

struct Prefix36LayerLayout {
    uint64_t bucket_keys_offset = sizeof(Prefix36LayerHeader);
    uint64_t bitmap_offsets_offset = 0;
    uint64_t success_offsets_offset = 0;
    uint64_t small_bitmap_offset = 0;
    uint64_t large_bitmap_offset = 0;
    uint64_t success_values_offset = 0;
};

Prefix36LayerLayout prefix36_layer_layout(const Prefix36LayerHeader &header) {
    Prefix36LayerLayout layout;
    layout.bitmap_offsets_offset = layout.bucket_keys_offset + header.bucket_count * sizeof(uint64_t);
    layout.success_offsets_offset = layout.bitmap_offsets_offset + header.bucket_count * sizeof(uint32_t);
    layout.small_bitmap_offset = layout.success_offsets_offset + header.bucket_count * sizeof(uint32_t);
    layout.large_bitmap_offset = layout.small_bitmap_offset + header.small_bitmap_bytes;
    layout.success_values_offset = layout.large_bitmap_offset + header.large_bitmap_words * sizeof(uint64_t);
    return layout;
}

uint64_t prefix36_layer_view_logical_bytes(const Prefix36LayerView &layer) {
    return sizeof(Prefix36LayerHeader)
        + layer.bucket_count * sizeof(uint64_t)
        + layer.bucket_count * sizeof(uint32_t)
        + layer.bucket_count * sizeof(uint32_t)
        + layer.small_bitmap_byte_count
        + layer.large_bitmap_word_count * sizeof(uint64_t)
        + layer.success_value_count * layer.value_size;
}

void validate_prefix36_layer_view(const Prefix36LayerView &layer) {
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(layer.dtype_mode);
    if (layer.success_kind != storage_kind_for_prefix36_mode(mode) ||
        layer.value_size != prefix36_value_size(mode)) {
        throw std::runtime_error("invalid prefix36 layer view dtype");
    }
    if (layer.bucket_count != 0U &&
        (layer.bucket_keys == nullptr || layer.bitmap_offsets == nullptr || layer.success_offsets == nullptr)) {
        throw std::runtime_error("invalid prefix36 layer view bucket arrays");
    }
    if (layer.small_bitmap_byte_count != 0U && layer.small_bitmap_bytes == nullptr) {
        throw std::runtime_error("invalid prefix36 layer view small bitmap pool");
    }
    if (layer.large_bitmap_word_count != 0U && layer.large_bitmap_words == nullptr) {
        throw std::runtime_error("invalid prefix36 layer view large bitmap pool");
    }
    if (layer.success_value_count != 0U && layer.success_values == nullptr) {
        throw std::runtime_error("invalid prefix36 layer view success values");
    }
}

std::vector<uint8_t> prefix36_success_raw_from_fixed_view(
    const Prefix36LayerView &layer,
    uint64_t first,
    uint32_t count,
    Prefix36DTypeMode mode
) {
    const uint32_t value_size = prefix36_value_size(mode);
    std::vector<uint8_t> raw(static_cast<size_t>(count) * value_size);
    if (count == 0U) {
        return raw;
    }
    if (first + count > layer.success_value_count) {
        throw std::runtime_error("prefix36 layer view success range out of bounds");
    }
    if (mode == Prefix36DTypeMode::UInt32) {
        std::memcpy(
            raw.data(),
            layer.success_values + first,
            static_cast<size_t>(count) * sizeof(uint32_t)
        );
        return raw;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t bits = prefix36_raw_bits_from_fixed(layer.success_values[first + i], mode);
        std::memcpy(raw.data() + static_cast<size_t>(i) * value_size, &bits, value_size);
    }
    return raw;
}

Prefix36LutRuntime read_prefix36_lut_runtime(const std::string &zlut_path, bool load_unrank = false) {
    std::ifstream in(zlut_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prefix36 LUT: " + zlut_path);
    }
    Prefix36LutHeader header{};
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in ||
        std::memcmp(header.magic, kPrefix36LutMagic, sizeof(kPrefix36LutMagic)) != 0 ||
        header.version != kPrefix36LutVersion) {
        throw std::runtime_error("invalid prefix36 LUT file");
    }
    Prefix36LutRuntime lut;
    lut.physical_transform = header.physical_transform;
    lut.inverse_physical_transform = header.inverse_physical_transform;
    lut.logical_pattern_signature = header.logical_pattern_signature;
    lut.physical_pattern_signature = header.physical_pattern_signature;
    lut.table_for_high.assign(std::begin(header.table_for_high), std::end(header.table_for_high));
    lut.packed_table0 = header.packed_table0;
    lut.packed_table1 = header.packed_table1;
    if (header.valid_suffix_mask_count != 0U) {
        in.seekg(static_cast<std::streamoff>(header.valid_suffix_mask_count * sizeof(uint32_t)), std::ios::cur);
    }
    lut.rank_tables.resize(static_cast<size_t>(header.rank_table_count));
    for (auto &table : lut.rank_tables) {
        table.resize(ZMaskFrozen::kSuffixStateCount);
        if (!table.empty()) {
            in.read(reinterpret_cast<char *>(table.data()),
                    static_cast<std::streamsize>(table.size() * sizeof(uint16_t)));
        }
    }
    if (header.packed_rank_pair_values != 0U) {
        lut.packed_rank_pair_table.resize(static_cast<size_t>(header.packed_rank_pair_values));
        in.read(reinterpret_cast<char *>(lut.packed_rank_pair_table.data()),
                static_cast<std::streamsize>(lut.packed_rank_pair_table.size() * sizeof(uint32_t)));
    }
    if (header.packed_meta_values != 0U) {
        in.seekg(static_cast<std::streamoff>(header.packed_meta_values * sizeof(uint64_t)), std::ios::cur);
    }
    lut.size_table.resize(static_cast<size_t>(header.size_table_values));
    if (!lut.size_table.empty()) {
        in.read(reinterpret_cast<char *>(lut.size_table.data()),
                static_cast<std::streamsize>(lut.size_table.size() * sizeof(uint32_t)));
    }
    if (load_unrank) {
        lut.offset_table.resize(static_cast<size_t>(header.offset_table_values));
        if (!lut.offset_table.empty()) {
            in.read(reinterpret_cast<char *>(lut.offset_table.data()),
                    static_cast<std::streamsize>(lut.offset_table.size() * sizeof(uint32_t)));
        }
    } else if (header.offset_table_values != 0U) {
        in.seekg(static_cast<std::streamoff>(header.offset_table_values * sizeof(uint32_t)), std::ios::cur);
    }
    if (load_unrank) {
        lut.unrank_array.resize(static_cast<size_t>(header.unrank_array_values));
        if (!lut.unrank_array.empty()) {
            in.read(reinterpret_cast<char *>(lut.unrank_array.data()),
                    static_cast<std::streamsize>(lut.unrank_array.size() * sizeof(uint32_t)));
        }
    } else if (header.unrank_array_values != 0U) {
        in.seekg(static_cast<std::streamoff>(header.unrank_array_values * sizeof(uint32_t)), std::ios::cur);
    }
    lut.high_base.resize(static_cast<size_t>(header.high_base_values));
    if (!lut.high_base.empty()) {
        in.read(reinterpret_cast<char *>(lut.high_base.data()),
                static_cast<std::streamsize>(lut.high_base.size() * sizeof(uint16_t)));
    }
    if (!in) {
        throw std::runtime_error("truncated prefix36 LUT: " + zlut_path);
    }
    return lut;
}

struct Prefix36PreparedQuery {
    uint64_t prefix36 = 0;
    uint32_t group = 0;
    uint32_t valid_count = 0;
    uint32_t rank = 0;
    bool valid = false;
};

Prefix36PreparedQuery prepare_prefix36_query(const Prefix36LutRuntime &lut, uint64_t board) {
    Prefix36PreparedQuery query;
    const uint32_t suffix28 = static_cast<uint32_t>(board & 0x0FFFFFFFULL);
    const uint32_t high = suffix28 >> 24U;
    if (high >= lut.table_for_high.size()) {
        return query;
    }
    const uint8_t table_id = lut.table_for_high[high];
    if (table_id == 0xFFU) {
        return query;
    }
    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    uint16_t low_rank = kPrefix36InvalidRank;
    if (!lut.packed_rank_pair_table.empty() &&
        (table_id == lut.packed_table0 || table_id == lut.packed_table1)) {
        const uint32_t packed = lut.packed_rank_pair_table[low24];
        low_rank = table_id == lut.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
    } else if (table_id < lut.rank_tables.size()) {
        low_rank = lut.rank_tables[table_id][low24];
    }
    if (low_rank == kPrefix36InvalidRank) {
        return query;
    }
    const uint32_t group = prefix36_suffix_group(suffix28);
    if (group >= lut.size_table.size()) {
        return query;
    }
    const size_t base_index = static_cast<size_t>(high) * lut.size_table.size() + group;
    if (base_index >= lut.high_base.size()) {
        return query;
    }
    const uint32_t rank = static_cast<uint32_t>(lut.high_base[base_index]) + low_rank;
    const uint32_t valid_count = lut.size_table[group];
    if (valid_count == 0U || rank >= valid_count) {
        return query;
    }
    query.prefix36 = board >> 28U;
    query.group = group;
    query.valid_count = valid_count;
    query.rank = rank;
    query.valid = true;
    return query;
}

uint32_t board_sum(uint64_t board) {
    uint32_t sum = 0U;
    for (uint32_t i = 0; i < 16U; ++i) {
        sum += ZMaskFrozen::tile_value(static_cast<uint32_t>((board >> (i * 4U)) & 0xFULL));
    }
    return sum;
}

uint32_t popcount_u32(uint32_t value) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt(value));
#else
    return static_cast<uint32_t>(__builtin_popcount(value));
#endif
}

uint32_t popcount_u64(uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<uint32_t>(__popcnt64(value));
#elif defined(_MSC_VER)
    return popcount_u32(static_cast<uint32_t>(value)) + popcount_u32(static_cast<uint32_t>(value >> 32U));
#else
    return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
}

uint32_t dense_ordinal_small_raw(const uint8_t *bitmap, uint32_t rank) {
    const uint32_t byte_idx = rank >> 3U;
    uint32_t total = 0U;
    for (uint32_t i = 0; i < byte_idx; ++i) {
        total += popcount_u32(bitmap[i]);
    }
    const uint32_t bit_idx = rank & 7U;
    if (bit_idx != 0U) {
        const uint32_t mask = (1U << bit_idx) - 1U;
        total += popcount_u32(static_cast<uint32_t>(bitmap[byte_idx] & static_cast<uint8_t>(mask)));
    }
    return total;
}

uint32_t dense_ordinal_large_raw(const uint8_t *bitmap, uint32_t rank) {
    const uint32_t word_idx = rank >> 6U;
    uint32_t total = 0U;
    for (uint32_t i = 0; i < word_idx; ++i) {
        total += popcount_u64(load_unaligned<uint64_t>(bitmap + static_cast<size_t>(i) * sizeof(uint64_t)));
    }
    const uint32_t bit_idx = rank & 63U;
    if (bit_idx != 0U) {
        const uint64_t mask = (1ULL << bit_idx) - 1ULL;
        total += popcount_u64(load_unaligned<uint64_t>(bitmap + static_cast<size_t>(word_idx) * sizeof(uint64_t)) & mask);
    }
    return total;
}

bool test_small_raw(const uint8_t *bitmap, uint32_t rank) {
    return (bitmap[rank >> 3U] & static_cast<uint8_t>(1U << (rank & 7U))) != 0U;
}

bool test_large_raw(const uint8_t *bitmap, uint32_t rank) {
    const uint64_t word = load_unaligned<uint64_t>(bitmap + static_cast<size_t>(rank >> 6U) * sizeof(uint64_t));
    return (word & (1ULL << (rank & 63U))) != 0ULL;
}

struct Prefix36CompressedFileIndex {
    Prefix36CompressedHeader header;
    std::vector<Prefix36BucketBlockEntry> bucket_dir;
    std::vector<Prefix36SuccessBlockEntry> success_dir;
};

struct FileStamp {
    uint64_t size = 0;
    std::filesystem::file_time_type write_time{};
    bool valid = false;
};

template <typename T>
struct CachedFileEntry {
    FileStamp stamp;
    std::shared_ptr<const T> value;
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, CachedFileEntry<Prefix36CompressedFileIndex>> g_prefix36_compressed_index_cache;
std::unordered_map<std::string, CachedFileEntry<Prefix36LutPointIndex>> g_prefix36_lut_point_cache;

FileStamp file_stamp(const std::string &path) {
    FileStamp stamp;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return stamp;
    }
    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return stamp;
    }
    stamp.size = static_cast<uint64_t>(size);
    stamp.write_time = write_time;
    stamp.valid = true;
    return stamp;
}

bool same_stamp(const FileStamp &lhs, const FileStamp &rhs) {
    return lhs.valid && rhs.valid && lhs.size == rhs.size && lhs.write_time == rhs.write_time;
}

Prefix36CompressedFileIndex read_prefix36_compressed_index(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EX prefix36 compressed result: " + path);
    }
    Prefix36CompressedFileIndex index;
    in.read(reinterpret_cast<char *>(&index.header), sizeof(index.header));
    if (!in ||
        std::memcmp(index.header.magic, kPrefix36CompressedMagic, sizeof(kPrefix36CompressedMagic)) != 0 ||
        index.header.version != kPrefix36LayerVersion) {
        throw std::runtime_error("invalid EX prefix36 compressed result");
    }
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(index.header.dtype_mode);
    if (index.header.success_kind != static_cast<uint32_t>(storage_kind_for_prefix36_mode(mode)) ||
        index.header.value_size != prefix36_value_size(mode) ||
        index.header.bucket_block_buckets == 0U ||
        index.header.success_block_values == 0U) {
        throw std::runtime_error("invalid EX prefix36 compressed result header");
    }
    index.bucket_dir.resize(static_cast<size_t>(index.header.bucket_block_count));
    index.success_dir.resize(static_cast<size_t>(index.header.success_block_count));
    if (!index.bucket_dir.empty()) {
        in.seekg(static_cast<std::streamoff>(index.header.bucket_dir_offset), std::ios::beg);
        in.read(
            reinterpret_cast<char *>(index.bucket_dir.data()),
            static_cast<std::streamsize>(index.bucket_dir.size() * sizeof(Prefix36BucketBlockEntry))
        );
    }
    if (!index.success_dir.empty()) {
        in.seekg(static_cast<std::streamoff>(index.header.success_dir_offset), std::ios::beg);
        in.read(
            reinterpret_cast<char *>(index.success_dir.data()),
            static_cast<std::streamsize>(index.success_dir.size() * sizeof(Prefix36SuccessBlockEntry))
        );
    }
    if (!in) {
        throw std::runtime_error("failed to read EX prefix36 compressed directory");
    }
    return index;
}

std::shared_ptr<const Prefix36CompressedFileIndex> cached_prefix36_compressed_index(const std::string &path) {
    const FileStamp stamp = file_stamp(path);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_prefix36_compressed_index_cache.find(path);
        if (it != g_prefix36_compressed_index_cache.end() &&
            same_stamp(it->second.stamp, stamp) &&
            it->second.value) {
            return it->second.value;
        }
    }

    auto loaded = std::make_shared<Prefix36CompressedFileIndex>(read_prefix36_compressed_index(path));
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_prefix36_compressed_index_cache[path] = CachedFileEntry<Prefix36CompressedFileIndex>{stamp, loaded};
    }
    return loaded;
}

Prefix36LutFileLayout prefix36_lut_file_layout(const Prefix36LutHeader &header) {
    Prefix36LutFileLayout layout;
    layout.valid_suffix_masks_offset = sizeof(Prefix36LutHeader);
    layout.rank_tables_offset =
        layout.valid_suffix_masks_offset + header.valid_suffix_mask_count * sizeof(uint32_t);
    layout.packed_rank_pair_offset =
        layout.rank_tables_offset + header.rank_table_values * sizeof(uint16_t);
    layout.packed_meta_offset =
        layout.packed_rank_pair_offset + header.packed_rank_pair_values * sizeof(uint32_t);
    layout.size_table_offset =
        layout.packed_meta_offset + header.packed_meta_values * sizeof(uint64_t);
    layout.offset_table_offset =
        layout.size_table_offset + header.size_table_values * sizeof(uint32_t);
    layout.unrank_array_offset =
        layout.offset_table_offset + header.offset_table_values * sizeof(uint32_t);
    layout.high_base_offset =
        layout.unrank_array_offset + header.unrank_array_values * sizeof(uint32_t);
    return layout;
}

template <typename Header>
bool prefix36_physical_metadata_matches_lut(const Header &header, const Prefix36LutHeader &lut_header) {
    return header.physical_transform == lut_header.physical_transform &&
        header.inverse_physical_transform == lut_header.inverse_physical_transform &&
        header.logical_pattern_signature == lut_header.logical_pattern_signature &&
        header.physical_pattern_signature == lut_header.physical_pattern_signature;
}

bool prefix36_physical_metadata_matches_lut(const Prefix36LayerHeader &header, const Prefix36LutRuntime &lut) {
    return header.physical_transform == lut.physical_transform &&
        header.inverse_physical_transform == lut.inverse_physical_transform &&
        header.logical_pattern_signature == lut.logical_pattern_signature &&
        header.physical_pattern_signature == lut.physical_pattern_signature;
}

bool prefix36_physical_metadata_matches_lut(const Prefix36LayerView &layer, const Prefix36LutRuntime &lut) {
    return layer.physical_transform == lut.physical_transform &&
        layer.inverse_physical_transform == lut.inverse_physical_transform &&
        layer.logical_pattern_signature == lut.logical_pattern_signature &&
        layer.physical_pattern_signature == lut.physical_pattern_signature;
}

Prefix36LutPointIndex read_prefix36_lut_point_index(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prefix36 LUT: " + path);
    }
    Prefix36LutPointIndex index;
    in.read(reinterpret_cast<char *>(&index.header), sizeof(index.header));
    if (!in ||
        std::memcmp(index.header.magic, kPrefix36LutMagic, sizeof(kPrefix36LutMagic)) != 0 ||
        index.header.version != kPrefix36LutVersion ||
        index.header.size_table_values == 0U ||
        index.header.high_base_values == 0U) {
        throw std::runtime_error("invalid prefix36 LUT file");
    }
    index.layout = prefix36_lut_file_layout(index.header);
    return index;
}

std::shared_ptr<const Prefix36LutPointIndex> cached_prefix36_lut_point_index(const std::string &path) {
    const FileStamp stamp = file_stamp(path);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_prefix36_lut_point_cache.find(path);
        if (it != g_prefix36_lut_point_cache.end() &&
            same_stamp(it->second.stamp, stamp) &&
            it->second.value) {
            return it->second.value;
        }
    }

    auto loaded = std::make_shared<Prefix36LutPointIndex>(read_prefix36_lut_point_index(path));
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_prefix36_lut_point_cache[path] = CachedFileEntry<Prefix36LutPointIndex>{stamp, loaded};
    }
    return loaded;
}

class Prefix36LutPointReader {
public:
    explicit Prefix36LutPointReader(const std::string &path)
        : index_(cached_prefix36_lut_point_index(path)),
          in_(path, std::ios::binary),
          path_(path) {
        if (!in_) {
            throw std::runtime_error("failed to open prefix36 LUT: " + path);
        }
    }

    const Prefix36LutHeader &header() const {
        return index_->header;
    }

    Prefix36PreparedQuery prepare_query(uint64_t board) {
        Prefix36PreparedQuery query;
        const uint32_t suffix28 = static_cast<uint32_t>(board & 0x0FFFFFFFULL);
        const uint32_t high = suffix28 >> 24U;
        const uint8_t table_id = index_->header.table_for_high[high];
        if (table_id == 0xFFU) {
            return query;
        }

        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint16_t low_rank = low24_rank(table_id, low24);
        if (low_rank == kPrefix36InvalidRank) {
            return query;
        }

        const uint32_t group = prefix36_suffix_group(suffix28);
        uint32_t valid_count = 0U;
        if (!valid_count_for_group(group, valid_count) || valid_count == 0U) {
            return query;
        }
        const uint64_t high_base_index =
            static_cast<uint64_t>(high) * index_->header.size_table_values + group;
        if (high_base_index >= index_->header.high_base_values) {
            return query;
        }
        const uint16_t high_base = read_one_from<uint16_t>(
            in_,
            index_->layout.high_base_offset + high_base_index * sizeof(uint16_t),
            path_);
        const uint32_t rank = static_cast<uint32_t>(high_base) + low_rank;
        if (rank >= valid_count) {
            return query;
        }
        query.prefix36 = board >> 28U;
        query.group = group;
        query.valid_count = valid_count;
        query.rank = rank;
        query.valid = true;
        return query;
    }

    bool valid_count_for_group(uint32_t group, uint32_t &valid_count) {
        valid_count = 0U;
        if (group >= index_->header.size_table_values) {
            return false;
        }
        valid_count = read_one_from<uint32_t>(
            in_,
            index_->layout.size_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            path_);
        return true;
    }

    bool unrank_suffix(uint32_t group, uint32_t rank, uint32_t &suffix28) {
        suffix28 = 0U;
        uint32_t valid_count = 0U;
        if (!valid_count_for_group(group, valid_count) || rank >= valid_count ||
            group >= index_->header.offset_table_values) {
            return false;
        }
        const uint32_t unrank_offset = read_one_from<uint32_t>(
            in_,
            index_->layout.offset_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            path_);
        const uint64_t unrank_index = static_cast<uint64_t>(unrank_offset) + rank;
        if (unrank_index >= index_->header.unrank_array_values) {
            return false;
        }
        suffix28 = read_one_from<uint32_t>(
            in_,
            index_->layout.unrank_array_offset + unrank_index * sizeof(uint32_t),
            path_);
        return true;
    }

private:
    uint16_t low24_rank(uint8_t table_id, uint32_t low24) {
        const auto &header = index_->header;
        if (header.packed_rank_pair_values != 0U &&
            (table_id == header.packed_table0 || table_id == header.packed_table1)) {
            if (low24 >= header.packed_rank_pair_values) {
                return kPrefix36InvalidRank;
            }
            const uint32_t packed = read_one_from<uint32_t>(
                in_,
                index_->layout.packed_rank_pair_offset + static_cast<uint64_t>(low24) * sizeof(uint32_t),
                path_);
            return table_id == header.packed_table0
                ? static_cast<uint16_t>(packed & 0xFFFFU)
                : static_cast<uint16_t>(packed >> 16U);
        }
        if (table_id >= header.rank_table_count) {
            return kPrefix36InvalidRank;
        }
        const uint64_t rank_index =
            static_cast<uint64_t>(table_id) * ZMaskFrozen::kSuffixStateCount + low24;
        if (rank_index >= header.rank_table_values) {
            return kPrefix36InvalidRank;
        }
        return read_one_from<uint16_t>(
            in_,
            index_->layout.rank_tables_offset + rank_index * sizeof(uint16_t),
            path_);
    }

    std::shared_ptr<const Prefix36LutPointIndex> index_;
    std::ifstream in_;
    std::string path_;
};

const Prefix36BucketBlockEntry *find_prefix36_bucket_block(
    const std::vector<Prefix36BucketBlockEntry> &dir,
    uint64_t key
) {
    size_t low = 0;
    size_t high = dir.size();
    while (low < high) {
        const size_t mid = low + (high - low) / 2U;
        if (dir[mid].last_prefix36 < key) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low >= dir.size()) {
        return nullptr;
    }
    const Prefix36BucketBlockEntry &entry = dir[low];
    return entry.first_prefix36 <= key && key <= entry.last_prefix36 ? &entry : nullptr;
}

const Prefix36SuccessBlockEntry *find_prefix36_success_block(
    const std::vector<Prefix36SuccessBlockEntry> &dir,
    uint64_t index
) {
    size_t low = 0;
    size_t high = dir.size();
    while (low < high) {
        const size_t mid = low + (high - low) / 2U;
        const uint64_t end = dir[mid].first_value_index + dir[mid].value_count;
        if (end <= index) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low >= dir.size()) {
        return nullptr;
    }
    const Prefix36SuccessBlockEntry &entry = dir[low];
    return entry.first_value_index <= index && index < entry.first_value_index + entry.value_count ? &entry : nullptr;
}

uint32_t find_prefix36_bucket_in_raw_block(
    const uint8_t *keys_ptr,
    uint32_t count,
    uint64_t target_key
) {
    uint32_t low = 0U;
    uint32_t high = count;
    while (low < high) {
        const uint32_t mid = low + ((high - low) >> 1U);
        const uint64_t key = load_unaligned<uint64_t>(keys_ptr + static_cast<size_t>(mid) * sizeof(uint64_t));
        if (key < target_key) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low >= count) {
        return std::numeric_limits<uint32_t>::max();
    }
    const uint64_t key = load_unaligned<uint64_t>(keys_ptr + static_cast<size_t>(low) * sizeof(uint64_t));
    return key == target_key ? low : std::numeric_limits<uint32_t>::max();
}

bool is_prefix36_layer_file(const std::string &path) {
    constexpr char kPrefix36Magic[8] = {'E', 'X', 'P', '3', '6', 'B', 'K', '\0'};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    return in && std::memcmp(magic, kPrefix36Magic, sizeof(kPrefix36Magic)) == 0;
}

bool is_prefix36_compressed_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    return in && std::memcmp(magic, kPrefix36CompressedMagic, sizeof(kPrefix36CompressedMagic)) == 0;
}

CompressStats compress_prefix36_layer_file(
    const std::string &zbook_path,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets,
    uint32_t success_block_values,
    int compression_level
) {
    const double t0 = wall_time_seconds();
    if (bucket_block_buckets == 0U || success_block_values == 0U) {
        throw std::runtime_error("prefix36 compressed block sizes must be non-zero");
    }
    const Prefix36LayerHeader source = read_prefix36_layer_header(zbook_path);
    const Prefix36LayerLayout layout = prefix36_layer_layout(source);
    const Prefix36LutRuntime lut = read_prefix36_lut_runtime(zlut_path);
    if (!prefix36_physical_metadata_matches_lut(source, lut)) {
        throw std::runtime_error("EX prefix36 physical pattern metadata does not match LUT");
    }
    const uint64_t bucket_block_count = (source.bucket_count + bucket_block_buckets - 1ULL) / bucket_block_buckets;
    const uint64_t success_block_count = (source.success_value_count + success_block_values - 1ULL) / success_block_values;

    Prefix36CompressedHeader header{};
    std::memcpy(header.magic, kPrefix36CompressedMagic, sizeof(kPrefix36CompressedMagic));
    header.success_kind = source.success_kind;
    header.dtype_mode = source.dtype_mode;
    header.value_size = static_cast<uint32_t>(source.value_size);
    header.layer_sum = source.layer_sum;
    header.threshold_bits = source.threshold_bits;
    header.physical_transform = source.physical_transform;
    header.inverse_physical_transform = source.inverse_physical_transform;
    header.logical_pattern_signature = source.logical_pattern_signature;
    header.physical_pattern_signature = source.physical_pattern_signature;
    header.bucket_count = source.bucket_count;
    header.success_value_count = source.success_value_count;
    header.live_board_count = source.live_board_count;
    header.bucket_block_buckets = bucket_block_buckets;
    header.success_block_values = success_block_values;
    header.bucket_block_count = bucket_block_count;
    header.success_block_count = success_block_count;
    header.bucket_dir_offset = sizeof(Prefix36CompressedHeader);
    header.success_dir_offset = header.bucket_dir_offset + bucket_block_count * sizeof(Prefix36BucketBlockEntry);
    header.data_offset = header.success_dir_offset + success_block_count * sizeof(Prefix36SuccessBlockEntry);
    header.original_file_size = file_size_u64(zbook_path);

    std::vector<Prefix36BucketBlockEntry> bucket_dir(static_cast<size_t>(bucket_block_count));
    std::vector<Prefix36SuccessBlockEntry> success_dir(static_cast<size_t>(success_block_count));

    std::fstream out(output_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write EX prefix36 compressed result: " + output_path);
    }
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    write_zero_bytes(out, bucket_block_count * sizeof(Prefix36BucketBlockEntry) +
                          success_block_count * sizeof(Prefix36SuccessBlockEntry));

    uint64_t write_offset = header.data_offset;
    out.seekp(static_cast<std::streamoff>(write_offset), std::ios::beg);
    CompressStats stats;
    stats.original_bytes = header.original_file_size;
    stats.bucket_block_count = bucket_block_count;
    stats.success_block_count = success_block_count;

    const uint32_t bucket_workers = compression_worker_count(bucket_block_count);
    for (uint64_t batch_begin = 0; batch_begin < bucket_block_count; batch_begin += bucket_workers) {
        const uint64_t batch_end = std::min<uint64_t>(bucket_block_count, batch_begin + bucket_workers);
        std::vector<std::future<CompressedPrefix36BucketBlock>> futures;
        futures.reserve(static_cast<size_t>(batch_end - batch_begin));
        for (uint64_t block_idx = batch_begin; block_idx < batch_end; ++block_idx) {
            futures.emplace_back(std::async(std::launch::async, [&, block_idx]() {
                std::ifstream source_file(zbook_path, std::ios::binary);
                if (!source_file) {
                    throw std::runtime_error("failed to open prefix36 zbook: " + zbook_path);
                }
                const uint32_t begin = static_cast<uint32_t>(block_idx * bucket_block_buckets);
                const uint32_t end = static_cast<uint32_t>(
                    std::min<uint64_t>(source.bucket_count, begin + bucket_block_buckets));
                const uint32_t count = end - begin;
                std::vector<uint64_t> keys(count);
                std::vector<uint32_t> bitmap_offsets(count);
                std::vector<uint32_t> success_offsets(count);
                if (count != 0U) {
                    source_file.seekg(static_cast<std::streamoff>(
                        layout.bucket_keys_offset + static_cast<uint64_t>(begin) * sizeof(uint64_t)));
                    source_file.read(reinterpret_cast<char *>(keys.data()),
                                     static_cast<std::streamsize>(keys.size() * sizeof(uint64_t)));
                    source_file.seekg(static_cast<std::streamoff>(
                        layout.bitmap_offsets_offset + static_cast<uint64_t>(begin) * sizeof(uint32_t)));
                    source_file.read(reinterpret_cast<char *>(bitmap_offsets.data()),
                                     static_cast<std::streamsize>(bitmap_offsets.size() * sizeof(uint32_t)));
                    source_file.seekg(static_cast<std::streamoff>(
                        layout.success_offsets_offset + static_cast<uint64_t>(begin) * sizeof(uint32_t)));
                    source_file.read(reinterpret_cast<char *>(success_offsets.data()),
                                     static_cast<std::streamsize>(success_offsets.size() * sizeof(uint32_t)));
                    if (!source_file) {
                        throw std::runtime_error("failed to read prefix36 zbook metadata");
                    }
                }

                std::vector<uint32_t> local_bitmap_offsets(static_cast<size_t>(count) + 1U, 0U);
                std::vector<uint8_t> small_payload;
                std::vector<uint8_t> large_payload;
                for (uint32_t local = 0; local < count; ++local) {
                    const uint64_t key = keys[local];
                    const uint32_t group = static_cast<uint32_t>((key & ((1ULL << 18U) - 1ULL)) >> 1U);
                    const uint32_t valid_count = group < lut.size_table.size() ? lut.size_table[group] : 0U;
                    if (valid_count <= source.threshold_bits) {
                        local_bitmap_offsets[local] = static_cast<uint32_t>(small_payload.size());
                        const uint64_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
                        std::vector<uint8_t> bitmap = read_range_from(
                            source_file,
                            layout.small_bitmap_offset + bitmap_offsets[local],
                            bytes,
                            zbook_path
                        );
                        small_payload.insert(small_payload.end(), bitmap.begin(), bitmap.end());
                    } else {
                        local_bitmap_offsets[local] = static_cast<uint32_t>(large_payload.size() / sizeof(uint64_t));
                        const uint64_t bytes = ZMaskFrozen::words_for_bits(valid_count) * sizeof(uint64_t);
                        std::vector<uint8_t> bitmap = read_range_from(
                            source_file,
                            layout.large_bitmap_offset + static_cast<uint64_t>(bitmap_offsets[local]) * sizeof(uint64_t),
                            bytes,
                            zbook_path
                        );
                        large_payload.insert(large_payload.end(), bitmap.begin(), bitmap.end());
                    }
                }
                local_bitmap_offsets[count] = 0U;

                Prefix36BucketBlockRawHeader raw_header{};
                raw_header.bucket_count = count;
                raw_header.small_bitmap_bytes = static_cast<uint32_t>(small_payload.size());
                raw_header.large_bitmap_words = static_cast<uint32_t>(large_payload.size() / sizeof(uint64_t));
                std::vector<uint8_t> raw;
                raw.reserve(sizeof(raw_header)
                    + keys.size() * sizeof(uint64_t)
                    + success_offsets.size() * sizeof(uint32_t)
                    + local_bitmap_offsets.size() * sizeof(uint32_t)
                    + small_payload.size()
                    + large_payload.size());
                append_value(raw, raw_header);
                append_bytes(raw, keys.data(), keys.size() * sizeof(uint64_t));
                append_bytes(raw, success_offsets.data(), success_offsets.size() * sizeof(uint32_t));
                append_bytes(raw, local_bitmap_offsets.data(), local_bitmap_offsets.size() * sizeof(uint32_t));
                append_bytes(raw, small_payload.data(), small_payload.size());
                append_bytes(raw, large_payload.data(), large_payload.size());

                CompressedPrefix36BucketBlock result;
                result.compressed = compress_block_or_throw(raw.data(), raw.size(), compression_level);
                result.raw_size = raw.size();
                result.entry.first_prefix36 = count == 0U ? 0ULL : keys.front();
                result.entry.last_prefix36 = count == 0U ? 0ULL : keys.back();
                result.entry.first_bucket_index = begin;
                result.entry.bucket_count = count;
                result.entry.compressed_size = result.compressed.size();
                result.entry.raw_size = raw.size();
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedPrefix36BucketBlock result = futures[local].get();
            result.entry.compressed_offset = write_offset;
            bucket_dir[static_cast<size_t>(batch_begin) + local] = result.entry;
            out.write(reinterpret_cast<const char *>(result.compressed.data()),
                      static_cast<std::streamsize>(result.compressed.size()));
            write_offset += result.compressed.size();
            stats.bucket_raw_bytes += result.raw_size;
            stats.bucket_compressed_bytes += result.compressed.size();
        }
    }

    const uint32_t success_workers = compression_worker_count(success_block_count);
    for (uint64_t batch_begin = 0; batch_begin < success_block_count; batch_begin += success_workers) {
        const uint64_t batch_end = std::min<uint64_t>(success_block_count, batch_begin + success_workers);
        std::vector<std::future<CompressedPrefix36SuccessBlock>> futures;
        futures.reserve(static_cast<size_t>(batch_end - batch_begin));
        for (uint64_t block_idx = batch_begin; block_idx < batch_end; ++block_idx) {
            futures.emplace_back(std::async(std::launch::async, [&, block_idx]() {
                std::ifstream source_file(zbook_path, std::ios::binary);
                if (!source_file) {
                    throw std::runtime_error("failed to open prefix36 zbook: " + zbook_path);
                }
                const uint64_t first = block_idx * success_block_values;
                const uint32_t count = static_cast<uint32_t>(
                    std::min<uint64_t>(source.success_value_count - first, success_block_values));
                const uint64_t raw_size = static_cast<uint64_t>(count) * source.value_size;
                std::vector<uint8_t> raw = read_range_from(
                    source_file,
                    layout.success_values_offset + first * source.value_size,
                    raw_size,
                    zbook_path
                );
                CompressedPrefix36SuccessBlock result;
                result.compressed = compress_block_or_throw(raw.data(), raw.size(), compression_level);
                result.raw_size = raw.size();
                result.entry.first_value_index = first;
                result.entry.value_count = count;
                result.entry.value_size = static_cast<uint32_t>(source.value_size);
                result.entry.compressed_size = result.compressed.size();
                result.entry.raw_size = raw.size();
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedPrefix36SuccessBlock result = futures[local].get();
            result.entry.compressed_offset = write_offset;
            success_dir[static_cast<size_t>(batch_begin) + local] = result.entry;
            out.write(reinterpret_cast<const char *>(result.compressed.data()),
                      static_cast<std::streamsize>(result.compressed.size()));
            write_offset += result.compressed.size();
            stats.success_raw_bytes += result.raw_size;
            stats.success_compressed_bytes += result.compressed.size();
        }
    }

    write_at(out, 0, &header, sizeof(header));
    if (!bucket_dir.empty()) {
        write_at(out, header.bucket_dir_offset, bucket_dir.data(), bucket_dir.size() * sizeof(Prefix36BucketBlockEntry));
    }
    if (!success_dir.empty()) {
        write_at(out, header.success_dir_offset, success_dir.data(), success_dir.size() * sizeof(Prefix36SuccessBlockEntry));
    }
    out.close();
    stats.compressed_bytes = file_size_u64(output_path);
    stats.compress_seconds = wall_time_seconds() - t0;
    return stats;
}

CompressStats compress_prefix36_layer_view_impl(
    const Prefix36LayerView &layer,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets,
    uint32_t success_block_values,
    int compression_level
) {
    const double t0 = wall_time_seconds();
    if (bucket_block_buckets == 0U || success_block_values == 0U) {
        throw std::runtime_error("prefix36 compressed block sizes must be non-zero");
    }
    validate_prefix36_layer_view(layer);
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(layer.dtype_mode);
    const Prefix36LutRuntime lut = read_prefix36_lut_runtime(zlut_path);
    if (!prefix36_physical_metadata_matches_lut(layer, lut)) {
        throw std::runtime_error("EX prefix36 physical pattern metadata does not match LUT");
    }
    const uint64_t bucket_block_count = (layer.bucket_count + bucket_block_buckets - 1ULL) / bucket_block_buckets;
    const uint64_t success_block_count =
        (layer.success_value_count + success_block_values - 1ULL) / success_block_values;

    Prefix36CompressedHeader header{};
    std::memcpy(header.magic, kPrefix36CompressedMagic, sizeof(kPrefix36CompressedMagic));
    header.success_kind = static_cast<uint32_t>(layer.success_kind);
    header.dtype_mode = layer.dtype_mode;
    header.value_size = layer.value_size;
    header.layer_sum = layer.layer_sum;
    header.threshold_bits = layer.threshold_bits;
    header.physical_transform = layer.physical_transform;
    header.inverse_physical_transform = layer.inverse_physical_transform;
    header.logical_pattern_signature = layer.logical_pattern_signature;
    header.physical_pattern_signature = layer.physical_pattern_signature;
    header.bucket_count = layer.bucket_count;
    header.success_value_count = layer.success_value_count;
    header.live_board_count = layer.live_board_count;
    header.bucket_block_buckets = bucket_block_buckets;
    header.success_block_values = success_block_values;
    header.bucket_block_count = bucket_block_count;
    header.success_block_count = success_block_count;
    header.bucket_dir_offset = sizeof(Prefix36CompressedHeader);
    header.success_dir_offset = header.bucket_dir_offset + bucket_block_count * sizeof(Prefix36BucketBlockEntry);
    header.data_offset = header.success_dir_offset + success_block_count * sizeof(Prefix36SuccessBlockEntry);
    header.original_file_size = prefix36_layer_view_logical_bytes(layer);

    std::vector<Prefix36BucketBlockEntry> bucket_dir(static_cast<size_t>(bucket_block_count));
    std::vector<Prefix36SuccessBlockEntry> success_dir(static_cast<size_t>(success_block_count));

    std::fstream out(output_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write EX prefix36 compressed result: " + output_path);
    }
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    write_zero_bytes(out, bucket_block_count * sizeof(Prefix36BucketBlockEntry) +
                          success_block_count * sizeof(Prefix36SuccessBlockEntry));

    uint64_t write_offset = header.data_offset;
    out.seekp(static_cast<std::streamoff>(write_offset), std::ios::beg);
    CompressStats stats;
    stats.original_bytes = header.original_file_size;
    stats.bucket_block_count = bucket_block_count;
    stats.success_block_count = success_block_count;

    const uint32_t bucket_workers = compression_worker_count(bucket_block_count);
    for (uint64_t batch_begin = 0; batch_begin < bucket_block_count; batch_begin += bucket_workers) {
        const uint64_t batch_end = std::min<uint64_t>(bucket_block_count, batch_begin + bucket_workers);
        std::vector<std::future<CompressedPrefix36BucketBlock>> futures;
        futures.reserve(static_cast<size_t>(batch_end - batch_begin));
        for (uint64_t block_idx = batch_begin; block_idx < batch_end; ++block_idx) {
            futures.emplace_back(std::async(std::launch::async, [&, block_idx]() {
                const uint32_t begin = static_cast<uint32_t>(block_idx * bucket_block_buckets);
                const uint32_t end = static_cast<uint32_t>(
                    std::min<uint64_t>(layer.bucket_count, begin + bucket_block_buckets));
                const uint32_t count = end - begin;

                std::vector<uint64_t> keys(count);
                std::vector<uint32_t> success_offsets(count);
                std::vector<uint32_t> local_bitmap_offsets(static_cast<size_t>(count) + 1U, 0U);
                std::vector<uint8_t> small_payload;
                std::vector<uint8_t> large_payload;

                for (uint32_t local = 0; local < count; ++local) {
                    const uint32_t bucket_index = begin + local;
                    const uint64_t key = layer.bucket_keys[bucket_index];
                    keys[local] = key;
                    success_offsets[local] = layer.success_offsets[bucket_index];
                    const uint32_t group = static_cast<uint32_t>((key & ((1ULL << 18U) - 1ULL)) >> 1U);
                    const uint32_t valid_count = group < lut.size_table.size() ? lut.size_table[group] : 0U;
                    const uint32_t old_offset = layer.bitmap_offsets[bucket_index];
                    if (valid_count <= layer.threshold_bits) {
                        local_bitmap_offsets[local] = static_cast<uint32_t>(small_payload.size());
                        const uint64_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
                        if (static_cast<uint64_t>(old_offset) + bytes > layer.small_bitmap_byte_count) {
                            throw std::runtime_error("prefix36 layer view small bitmap offset out of bounds");
                        }
                        append_bytes(small_payload, layer.small_bitmap_bytes + old_offset, static_cast<size_t>(bytes));
                    } else {
                        local_bitmap_offsets[local] = static_cast<uint32_t>(large_payload.size() / sizeof(uint64_t));
                        const uint64_t words = ZMaskFrozen::words_for_bits(valid_count);
                        if (static_cast<uint64_t>(old_offset) + words > layer.large_bitmap_word_count) {
                            throw std::runtime_error("prefix36 layer view large bitmap offset out of bounds");
                        }
                        append_bytes(
                            large_payload,
                            reinterpret_cast<const uint8_t *>(layer.large_bitmap_words + old_offset),
                            static_cast<size_t>(words * sizeof(uint64_t))
                        );
                    }
                }
                local_bitmap_offsets[count] = 0U;

                Prefix36BucketBlockRawHeader raw_header{};
                raw_header.bucket_count = count;
                raw_header.small_bitmap_bytes = static_cast<uint32_t>(small_payload.size());
                raw_header.large_bitmap_words = static_cast<uint32_t>(large_payload.size() / sizeof(uint64_t));
                std::vector<uint8_t> raw;
                raw.reserve(sizeof(raw_header)
                    + keys.size() * sizeof(uint64_t)
                    + success_offsets.size() * sizeof(uint32_t)
                    + local_bitmap_offsets.size() * sizeof(uint32_t)
                    + small_payload.size()
                    + large_payload.size());
                append_value(raw, raw_header);
                append_bytes(raw, keys.data(), keys.size() * sizeof(uint64_t));
                append_bytes(raw, success_offsets.data(), success_offsets.size() * sizeof(uint32_t));
                append_bytes(raw, local_bitmap_offsets.data(), local_bitmap_offsets.size() * sizeof(uint32_t));
                append_bytes(raw, small_payload.data(), small_payload.size());
                append_bytes(raw, large_payload.data(), large_payload.size());

                CompressedPrefix36BucketBlock result;
                result.compressed = compress_block_or_throw(raw.data(), raw.size(), compression_level);
                result.raw_size = raw.size();
                result.entry.first_prefix36 = count == 0U ? 0ULL : keys.front();
                result.entry.last_prefix36 = count == 0U ? 0ULL : keys.back();
                result.entry.first_bucket_index = begin;
                result.entry.bucket_count = count;
                result.entry.compressed_size = result.compressed.size();
                result.entry.raw_size = raw.size();
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedPrefix36BucketBlock result = futures[local].get();
            result.entry.compressed_offset = write_offset;
            bucket_dir[static_cast<size_t>(batch_begin) + local] = result.entry;
            out.write(reinterpret_cast<const char *>(result.compressed.data()),
                      static_cast<std::streamsize>(result.compressed.size()));
            write_offset += result.compressed.size();
            stats.bucket_raw_bytes += result.raw_size;
            stats.bucket_compressed_bytes += result.compressed.size();
        }
    }

    const uint32_t success_workers = compression_worker_count(success_block_count);
    for (uint64_t batch_begin = 0; batch_begin < success_block_count; batch_begin += success_workers) {
        const uint64_t batch_end = std::min<uint64_t>(success_block_count, batch_begin + success_workers);
        std::vector<std::future<CompressedPrefix36SuccessBlock>> futures;
        futures.reserve(static_cast<size_t>(batch_end - batch_begin));
        for (uint64_t block_idx = batch_begin; block_idx < batch_end; ++block_idx) {
            futures.emplace_back(std::async(std::launch::async, [&, block_idx]() {
                const uint64_t first = block_idx * success_block_values;
                const uint32_t count = static_cast<uint32_t>(
                    std::min<uint64_t>(layer.success_value_count - first, success_block_values)
                );
                std::vector<uint8_t> raw = prefix36_success_raw_from_fixed_view(layer, first, count, mode);
                CompressedPrefix36SuccessBlock result;
                result.compressed = compress_block_or_throw(raw.data(), raw.size(), compression_level);
                result.raw_size = raw.size();
                result.entry.first_value_index = first;
                result.entry.value_count = count;
                result.entry.value_size = layer.value_size;
                result.entry.compressed_size = result.compressed.size();
                result.entry.raw_size = raw.size();
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedPrefix36SuccessBlock result = futures[local].get();
            result.entry.compressed_offset = write_offset;
            success_dir[static_cast<size_t>(batch_begin) + local] = result.entry;
            out.write(reinterpret_cast<const char *>(result.compressed.data()),
                      static_cast<std::streamsize>(result.compressed.size()));
            write_offset += result.compressed.size();
            stats.success_raw_bytes += result.raw_size;
            stats.success_compressed_bytes += result.compressed.size();
        }
    }

    write_at(out, 0, &header, sizeof(header));
    if (!bucket_dir.empty()) {
        write_at(out, header.bucket_dir_offset, bucket_dir.data(), bucket_dir.size() * sizeof(Prefix36BucketBlockEntry));
    }
    if (!success_dir.empty()) {
        write_at(out, header.success_dir_offset, success_dir.data(), success_dir.size() * sizeof(Prefix36SuccessBlockEntry));
    }
    out.close();
    stats.compressed_bytes = file_size_u64(output_path);
    stats.compress_seconds = wall_time_seconds() - t0;
    return stats;
}

ColdLookupResult lookup_prefix36_compressed_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t board
) {
    const auto index_ptr = cached_prefix36_compressed_index(compressed_path);
    const Prefix36CompressedFileIndex &index = *index_ptr;
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(index.header.dtype_mode);
    ColdLookupResult miss;
    miss.success_kind = storage_kind_for_prefix36_mode(mode);
    if (board_sum(board) != index.header.layer_sum ||
        index.header.bucket_count == 0U ||
        index.header.success_value_count == 0U) {
        return miss;
    }

    Prefix36LutPointReader lut(zlut_path);
    if (!prefix36_physical_metadata_matches_lut(index.header, lut.header())) {
        throw std::runtime_error("EX prefix36 physical pattern metadata does not match compressed layer");
    }
    const Prefix36PreparedQuery query = lut.prepare_query(board);
    if (!query.valid) {
        return miss;
    }
    const uint64_t target_key = (query.prefix36 << 18U) | (static_cast<uint64_t>(query.group) << 1U);
    const Prefix36BucketBlockEntry *bucket_entry =
        find_prefix36_bucket_block(index.bucket_dir, target_key);
    if (bucket_entry == nullptr) {
        return miss;
    }

    std::ifstream compressed_in(compressed_path, std::ios::binary);
    if (!compressed_in) {
        throw std::runtime_error("failed to open EX prefix36 compressed result: " + compressed_path);
    }

    std::vector<uint8_t> bucket_raw = decompress_block_or_throw_from(
        compressed_in,
        compressed_path,
        bucket_entry->compressed_offset,
        bucket_entry->compressed_size,
        bucket_entry->raw_size
    );
    if (bucket_raw.size() < sizeof(Prefix36BucketBlockRawHeader)) {
        throw std::runtime_error("EX prefix36 compressed bucket block is truncated");
    }
    const Prefix36BucketBlockRawHeader raw_header =
        load_unaligned<Prefix36BucketBlockRawHeader>(bucket_raw.data());
    if (raw_header.bucket_count != bucket_entry->bucket_count) {
        throw std::runtime_error("EX prefix36 compressed bucket block count mismatch");
    }
    const uint8_t *cursor = bucket_raw.data() + sizeof(Prefix36BucketBlockRawHeader);
    const uint8_t *end = bucket_raw.data() + bucket_raw.size();
    const size_t count = raw_header.bucket_count;
    const size_t keys_bytes = count * sizeof(uint64_t);
    const size_t success_offsets_bytes = count * sizeof(uint32_t);
    const size_t local_offsets_bytes = (count + 1U) * sizeof(uint32_t);
    if (cursor + keys_bytes + success_offsets_bytes + local_offsets_bytes > end) {
        throw std::runtime_error("EX prefix36 compressed bucket block metadata mismatch");
    }
    const uint8_t *keys_ptr = cursor;
    cursor += keys_bytes;
    const uint8_t *success_offsets_ptr = cursor;
    cursor += success_offsets_bytes;
    const uint8_t *local_bitmap_offsets_ptr = cursor;
    cursor += local_offsets_bytes;
    const uint8_t *small_payload = cursor;
    cursor += raw_header.small_bitmap_bytes;
    const uint8_t *large_payload = cursor;
    cursor += static_cast<size_t>(raw_header.large_bitmap_words) * sizeof(uint64_t);
    if (cursor != end) {
        throw std::runtime_error("EX prefix36 compressed bucket block payload size mismatch");
    }

    const uint32_t local_bucket = find_prefix36_bucket_in_raw_block(
        keys_ptr,
        raw_header.bucket_count,
        target_key
    );
    if (local_bucket == std::numeric_limits<uint32_t>::max()) {
        return miss;
    }
    const uint32_t success_offset =
        load_unaligned<uint32_t>(success_offsets_ptr + static_cast<size_t>(local_bucket) * sizeof(uint32_t));
    const uint32_t bitmap_offset =
        load_unaligned<uint32_t>(local_bitmap_offsets_ptr + static_cast<size_t>(local_bucket) * sizeof(uint32_t));

    uint64_t success_index = 0U;
    if (query.valid_count <= index.header.threshold_bits) {
        const uint64_t bytes = ZMaskFrozen::bytes_for_bits(query.valid_count);
        if (static_cast<uint64_t>(bitmap_offset) + bytes > raw_header.small_bitmap_bytes) {
            throw std::runtime_error("EX prefix36 compressed small bitmap offset mismatch");
        }
        const uint8_t *bitmap = small_payload + bitmap_offset;
        if (!test_small_raw(bitmap, query.rank)) {
            return miss;
        }
        success_index = static_cast<uint64_t>(success_offset) + dense_ordinal_small_raw(bitmap, query.rank);
    } else {
        const uint64_t words = ZMaskFrozen::words_for_bits(query.valid_count);
        if (static_cast<uint64_t>(bitmap_offset) + words > raw_header.large_bitmap_words) {
            throw std::runtime_error("EX prefix36 compressed large bitmap offset mismatch");
        }
        const uint8_t *bitmap = large_payload + static_cast<uint64_t>(bitmap_offset) * sizeof(uint64_t);
        if (!test_large_raw(bitmap, query.rank)) {
            return miss;
        }
        success_index = static_cast<uint64_t>(success_offset) + dense_ordinal_large_raw(bitmap, query.rank);
    }
    if (success_index >= index.header.success_value_count) {
        throw std::runtime_error("EX prefix36 compressed lookup produced out-of-range success index");
    }

    const Prefix36SuccessBlockEntry *success_entry =
        find_prefix36_success_block(index.success_dir, success_index);
    if (success_entry == nullptr || success_entry->value_size != index.header.value_size) {
        throw std::runtime_error("EX prefix36 compressed success block not found");
    }
    std::vector<uint8_t> success_raw = decompress_block_or_throw_from(
        compressed_in,
        compressed_path,
        success_entry->compressed_offset,
        success_entry->compressed_size,
        success_entry->raw_size
    );
    const uint64_t local_success = success_index - success_entry->first_value_index;
    const uint64_t value_offset = local_success * index.header.value_size;
    if (value_offset + index.header.value_size > success_raw.size()) {
        throw std::runtime_error("EX prefix36 compressed success block offset mismatch");
    }

    uint64_t raw_value = 0ULL;
    if (index.header.value_size == sizeof(uint32_t)) {
        raw_value = load_unaligned<uint32_t>(success_raw.data() + value_offset);
    } else {
        raw_value = load_unaligned<uint64_t>(success_raw.data() + value_offset);
    }
    const uint32_t fixed_value = prefix36_fixed_from_raw_bits(raw_value, mode);
    ColdLookupResult result;
    result.found = true;
    result.global_dense_index = success_index;
    result.raw_value_bits = prefix36_raw_bits_from_fixed(fixed_value, mode);
    result.numeric_value = prefix36_numeric_from_fixed(fixed_value, mode);
    result.success_kind = storage_kind_for_prefix36_mode(mode);
    result.bucket_block_raw_bytes = bucket_entry->raw_size;
    result.bucket_block_compressed_bytes = bucket_entry->compressed_size;
    result.success_block_raw_bytes = success_entry->raw_size;
    result.success_block_compressed_bytes = success_entry->compressed_size;
    return result;
}

bool load_prefix36_compressed_success_value(
    std::ifstream &compressed_in,
    const std::string &compressed_path,
    const Prefix36CompressedFileIndex &index,
    Prefix36DTypeMode mode,
    uint64_t success_index,
    uint64_t &raw_value_bits,
    double &numeric_value
) {
    raw_value_bits = 0ULL;
    numeric_value = 0.0;
    if (success_index >= index.header.success_value_count || index.header.success_block_count == 0U) {
        return true;
    }
    const Prefix36SuccessBlockEntry *entry = find_prefix36_success_block(index.success_dir, success_index);
    if (entry == nullptr || entry->value_size != index.header.value_size) {
        return false;
    }
    std::vector<uint8_t> raw = decompress_block_or_throw_from(
        compressed_in,
        compressed_path,
        entry->compressed_offset,
        entry->compressed_size,
        entry->raw_size
    );
    const uint64_t local = success_index - entry->first_value_index;
    const uint64_t value_offset = local * index.header.value_size;
    if (value_offset + index.header.value_size > raw.size()) {
        return false;
    }
    raw_value_bits = index.header.value_size == sizeof(uint32_t)
        ? load_unaligned<uint32_t>(raw.data() + value_offset)
        : load_unaligned<uint64_t>(raw.data() + value_offset);
    const uint32_t fixed_value = prefix36_fixed_from_raw_bits(raw_value_bits, mode);
    raw_value_bits = prefix36_raw_bits_from_fixed(fixed_value, mode);
    numeric_value = prefix36_numeric_from_fixed(fixed_value, mode);
    return true;
}

bool sample_prefix36_compressed_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value
) {
    const auto index_ptr = cached_prefix36_compressed_index(compressed_path);
    const Prefix36CompressedFileIndex &index = *index_ptr;
    if (index.header.bucket_block_count == 0U ||
        index.header.bucket_count == 0U ||
        index.header.success_value_count == 0U) {
        return false;
    }
    const Prefix36DTypeMode mode = prefix36_dtype_mode_from_u32(index.header.dtype_mode);
    Prefix36LutPointReader lut(zlut_path);
    if (!prefix36_physical_metadata_matches_lut(index.header, lut.header())) {
        return false;
    }
    std::ifstream compressed_in(compressed_path, std::ios::binary);
    if (!compressed_in) {
        return false;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> block_pick(0U, index.bucket_dir.size() - 1U);
    constexpr uint32_t kSampleAttempts = 128U;
    for (uint32_t attempt = 0U; attempt < kSampleAttempts; ++attempt) {
        const Prefix36BucketBlockEntry &bucket_entry = index.bucket_dir[block_pick(rng)];
        if (bucket_entry.bucket_count == 0U) {
            continue;
        }
        std::vector<uint8_t> bucket_raw = decompress_block_or_throw_from(
            compressed_in,
            compressed_path,
            bucket_entry.compressed_offset,
            bucket_entry.compressed_size,
            bucket_entry.raw_size
        );
        if (bucket_raw.size() < sizeof(Prefix36BucketBlockRawHeader)) {
            return false;
        }
        const Prefix36BucketBlockRawHeader raw_header =
            load_unaligned<Prefix36BucketBlockRawHeader>(bucket_raw.data());
        const uint8_t *cursor = bucket_raw.data() + sizeof(Prefix36BucketBlockRawHeader);
        const uint8_t *end = bucket_raw.data() + bucket_raw.size();
        const size_t count = raw_header.bucket_count;
        const size_t keys_bytes = count * sizeof(uint64_t);
        const size_t success_offsets_bytes = count * sizeof(uint32_t);
        const size_t local_offsets_bytes = (count + 1U) * sizeof(uint32_t);
        if (cursor + keys_bytes + success_offsets_bytes + local_offsets_bytes > end) {
            return false;
        }
        const uint8_t *keys_ptr = cursor;
        cursor += keys_bytes;
        const uint8_t *success_offsets_ptr = cursor;
        cursor += success_offsets_bytes;
        const uint8_t *local_bitmap_offsets_ptr = cursor;
        cursor += local_offsets_bytes;
        const uint8_t *small_payload = cursor;
        cursor += raw_header.small_bitmap_bytes;
        const uint8_t *large_payload = cursor;
        cursor += static_cast<size_t>(raw_header.large_bitmap_words) * sizeof(uint64_t);
        if (cursor != end) {
            return false;
        }
        if (raw_header.bucket_count == 0U) {
            continue;
        }

        std::uniform_int_distribution<uint32_t> local_pick(0U, raw_header.bucket_count - 1U);
        for (uint32_t local_attempt = 0; local_attempt < raw_header.bucket_count; ++local_attempt) {
            const uint32_t local = local_pick(rng);
            const uint64_t key = load_unaligned<uint64_t>(keys_ptr + static_cast<size_t>(local) * sizeof(uint64_t));
            const uint64_t prefix36 = key >> 18U;
            const uint32_t group = static_cast<uint32_t>((key & ((1ULL << 18U) - 1ULL)) >> 1U);
            uint32_t valid_count = 0U;
            if (!lut.valid_count_for_group(group, valid_count)) {
                continue;
            }
            const uint32_t success_offset =
                load_unaligned<uint32_t>(success_offsets_ptr + static_cast<size_t>(local) * sizeof(uint32_t));
            const uint32_t bitmap_offset =
                load_unaligned<uint32_t>(local_bitmap_offsets_ptr + static_cast<size_t>(local) * sizeof(uint32_t));
            if (valid_count <= index.header.threshold_bits) {
                const uint64_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
                if (static_cast<uint64_t>(bitmap_offset) + bytes > raw_header.small_bitmap_bytes) {
                    return false;
                }
                const uint8_t *bitmap = small_payload + bitmap_offset;
                uint32_t live_count = 0U;
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = bitmap[byte_idx];
                    if (byte_idx + 1U == bytes && (valid_count & 7U) != 0U) {
                        value = static_cast<uint8_t>(
                            value & static_cast<uint8_t>((1U << (valid_count & 7U)) - 1U)
                        );
                    }
                    live_count += popcount_u32(value);
                }
                if (live_count == 0U) {
                    continue;
                }
                std::uniform_int_distribution<uint32_t> ordinal_pick(0U, live_count - 1U);
                const uint32_t success_ordinal = ordinal_pick(rng);
                uint32_t ordinal = success_ordinal;
                uint32_t rank = std::numeric_limits<uint32_t>::max();
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = bitmap[byte_idx];
                    if (byte_idx + 1U == bytes && (valid_count & 7U) != 0U) {
                        value = static_cast<uint8_t>(
                            value & static_cast<uint8_t>((1U << (valid_count & 7U)) - 1U)
                        );
                    }
                    const uint32_t count = popcount_u32(value);
                    if (ordinal >= count) {
                        ordinal -= count;
                        continue;
                    }
                    while (value != 0U) {
                        const uint32_t bit = static_cast<uint32_t>(
#if defined(_MSC_VER)
                            _tzcnt_u32(value)
#else
                            __builtin_ctz(value)
#endif
                        );
                        const uint32_t candidate_rank = byte_idx * 8U + bit;
                        if (candidate_rank >= valid_count) {
                            break;
                        }
                        if (ordinal == 0U) {
                            rank = candidate_rank;
                            break;
                        }
                        --ordinal;
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                    if (rank != std::numeric_limits<uint32_t>::max()) {
                        break;
                    }
                }
                uint32_t suffix28 = 0U;
                if (rank == std::numeric_limits<uint32_t>::max() ||
                    !lut.unrank_suffix(group, rank, suffix28)) {
                    continue;
                }
                const uint64_t success_index = static_cast<uint64_t>(success_offset) + success_ordinal;
                if (success_index >= index.header.success_value_count) {
                    continue;
                }
                board = (prefix36 << 28U) | suffix28;
                if (load_prefix36_compressed_success_value(
                        compressed_in, compressed_path, index, mode, success_index, raw_value_bits, numeric_value)) {
                    return true;
                }
            } else {
                const uint64_t words = ZMaskFrozen::words_for_bits(valid_count);
                if (static_cast<uint64_t>(bitmap_offset) + words > raw_header.large_bitmap_words) {
                    return false;
                }
                const uint8_t *bitmap = large_payload + static_cast<uint64_t>(bitmap_offset) * sizeof(uint64_t);
                uint32_t live_count = 0U;
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value =
                        load_unaligned<uint64_t>(bitmap + static_cast<uint64_t>(word_idx) * sizeof(uint64_t));
                    if (word_idx + 1U == words && (valid_count & 63U) != 0U) {
                        value &= (1ULL << (valid_count & 63U)) - 1ULL;
                    }
                    live_count += popcount_u64(value);
                }
                if (live_count == 0U) {
                    continue;
                }
                std::uniform_int_distribution<uint32_t> ordinal_pick(0U, live_count - 1U);
                const uint32_t success_ordinal = ordinal_pick(rng);
                uint32_t ordinal = success_ordinal;
                uint32_t rank = std::numeric_limits<uint32_t>::max();
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value =
                        load_unaligned<uint64_t>(bitmap + static_cast<uint64_t>(word_idx) * sizeof(uint64_t));
                    if (word_idx + 1U == words && (valid_count & 63U) != 0U) {
                        value &= (1ULL << (valid_count & 63U)) - 1ULL;
                    }
                    const uint32_t count = popcount_u64(value);
                    if (ordinal >= count) {
                        ordinal -= count;
                        continue;
                    }
                    while (value != 0ULL) {
                        const uint32_t bit = static_cast<uint32_t>(
#if defined(_MSC_VER) && defined(_M_X64)
                            _tzcnt_u64(value)
#elif defined(_MSC_VER)
                            _tzcnt_u32(static_cast<uint32_t>(value))
#else
                            __builtin_ctzll(value)
#endif
                        );
                        const uint32_t candidate_rank = word_idx * 64U + bit;
                        if (candidate_rank >= valid_count) {
                            break;
                        }
                        if (ordinal == 0U) {
                            rank = candidate_rank;
                            break;
                        }
                        --ordinal;
                        value &= value - 1ULL;
                    }
                    if (rank != std::numeric_limits<uint32_t>::max()) {
                        break;
                    }
                }
                uint32_t suffix28 = 0U;
                if (rank == std::numeric_limits<uint32_t>::max() ||
                    !lut.unrank_suffix(group, rank, suffix28)) {
                    continue;
                }
                const uint64_t success_index = static_cast<uint64_t>(success_offset) + success_ordinal;
                if (success_index >= index.header.success_value_count) {
                    continue;
                }
                board = (prefix36 << 28U) | suffix28;
                if (load_prefix36_compressed_success_value(
                        compressed_in, compressed_path, index, mode, success_index, raw_value_bits, numeric_value)) {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

CompressStats compress_zbook_to_ex_result(
    const std::string &zbook_path,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets,
    uint32_t success_block_values,
    int compression_level
) {
    if (is_prefix36_layer_file(zbook_path)) {
        return compress_prefix36_layer_file(
            zbook_path,
            zlut_path,
            output_path,
            bucket_block_buckets,
            success_block_values,
            compression_level
        );
    }
    throw std::runtime_error("EX compression only supports current prefix36 zbook files; rebuild the input layer");
}

CompressStats compress_prefix36_layer_view_to_ex_result(
    const Prefix36LayerView &layer,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets,
    uint32_t success_block_values,
    int compression_level
) {
    return compress_prefix36_layer_view_impl(
        layer,
        zlut_path,
        output_path,
        bucket_block_buckets,
        success_block_values,
        compression_level
    );
}

ColdLookupResult lookup_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t board
) {
    if (is_prefix36_compressed_file(compressed_path)) {
        return lookup_prefix36_compressed_cold(compressed_path, zlut_path, board);
    }
    throw std::runtime_error("EX compressed cold lookup only supports current prefix36 compressed files");
}

ColdLookupResult lookup_zbook_cold(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t board
) {
    if (is_prefix36_layer_file(zbook_path)) {
        return EXPrefix36Runtime::lookup_zbook_cold(zbook_path, zlut_path, board);
    }
    throw std::runtime_error("EX zbook cold lookup only supports current prefix36 zbook files; rebuild the input layer");
}

bool sample_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value
) {
    if (is_prefix36_compressed_file(compressed_path)) {
        return sample_prefix36_compressed_cold(
            compressed_path,
            zlut_path,
            board,
            raw_value_bits,
            numeric_value
        );
    }
    return false;
}

} // namespace EXCompressedResult
