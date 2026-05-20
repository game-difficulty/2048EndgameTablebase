#include "EXADCompressedResult.h"

#include "EXADIO.h"
#include "EXADSolvedLayer.h"
#include "NativeLzma.h"

#include <algorithm>
#include <array>
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

namespace EXADCompressedResult {
namespace {

constexpr char kMagic[8] = {'E', 'X', 'A', 'D', 'C', 'Z', '1', '\0'};
constexpr char kExadLutMagic[8] = {'E', 'X', 'A', 'D', '7', 'L', 'U', 'T'};
constexpr uint32_t kVersion = 2;
constexpr uint32_t kSlotCount = 48;
constexpr uint32_t kBucketBlockHardCapBytes = 1024u * 1024u;
constexpr double kFixed32Scale = 4000000000.0;
constexpr double kFixed64Scale = 1600000000000000000.0;

struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t dtype_mode;
    uint32_t value_size;
    uint32_t original_board_sum;
    uint32_t threshold_bits;
    uint32_t slot_count;
    uint32_t bucket_block_raw_target_bytes;
    uint32_t bucket_block_raw_hard_cap_bytes;
    uint32_t success_block_values;
    uint64_t lut_signature;
    uint8_t physical_transform;
    uint8_t inverse_physical_transform;
    uint16_t reserved_transform;
    uint32_t reserved32;
    uint64_t logical_pattern_signature;
    uint64_t physical_pattern_signature;
    uint64_t live_board_count;
    uint64_t success_value_count;
    uint64_t bucket_block_count;
    uint64_t value_block_count;
    uint64_t slot_dir_offset;
    uint64_t bucket_dir_offset;
    uint64_t value_dir_offset;
    uint64_t data_offset;
    uint64_t original_file_size;
    uint64_t reserved0;
};

struct SlotDirEntry {
    int32_t ad_key;
    uint32_t row_width;
    uint64_t slot_row_base;
    uint64_t slot_value_base;
    uint64_t bucket_count;
    uint64_t live_board_count;
    uint64_t bucket_block_begin;
    uint64_t bucket_block_count;
    uint64_t reserved0;
};

struct BucketBlockDirEntry {
    uint32_t slot;
    uint32_t first_bucket_index;
    uint32_t bucket_count;
    uint32_t reserved0;
    uint64_t first_key;
    uint64_t last_key;
    uint64_t first_dense_offset;
    uint64_t compressed_offset;
    uint64_t compressed_size;
    uint64_t raw_size;
};

struct ValueBlockDirEntry {
    uint64_t first_value_index;
    uint32_t value_count;
    uint32_t value_size;
    uint64_t compressed_offset;
    uint64_t compressed_size;
    uint64_t raw_size;
};

struct BucketBlockRawHeader {
    uint32_t bucket_count;
    uint32_t small_bitmap_bytes;
    uint32_t large_bitmap_words;
    uint32_t reserved0;
    uint64_t first_dense_offset;
};

struct PendingBucketBlock {
    uint32_t slot = 0;
    uint32_t begin = 0;
    uint32_t count = 0;
    uint64_t raw_size_estimate = 0;
};

struct ExadLutFileHeader {
    char magic[8];
    uint32_t version = 2;
    uint32_t valid_mask_count = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved16 = 0;
    uint64_t config_signature = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t rank_table_count = 0;
    uint64_t rank_table_values = 0;
    uint64_t packed_rank_pair_values = 0;
    uint64_t size_table_values = 0;
    uint64_t offset_table_values = 0;
    uint64_t unrank_array_values = 0;
    uint64_t high_base_values = 0;
    uint64_t row16_sum_values = 0;
    uint8_t packed_table0 = 0xFFU;
    uint8_t packed_table1 = 0xFFU;
    uint8_t table_for_high[16]{};
    int8_t max_counts[16]{};
    uint32_t required_suffix24 = 0;
};

struct ExadLutFileLayout {
    uint64_t valid_suffix_masks_offset = 0;
    uint64_t rank_table_sizes_offset = 0;
    uint64_t rank_tables_offset = 0;
    uint64_t packed_rank_pair_offset = 0;
    uint64_t size_table_offset = 0;
    uint64_t offset_table_offset = 0;
    uint64_t unrank_array_offset = 0;
    uint64_t high_base_offset = 0;
};

struct ExadLutPointIndex {
    ExadLutFileHeader header{};
    ExadLutFileLayout layout{};
    std::vector<uint64_t> rank_table_sizes;
    std::vector<uint64_t> rank_table_offsets;
};

template <typename T>
void append_unaligned(std::vector<uint8_t>& out, const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T load_unaligned(const uint8_t* p) {
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

double wall_time_seconds() {
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();
    const auto now = Clock::now();
    return std::chrono::duration<double>(now - start).count();
}

uint64_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0ull : static_cast<uint64_t>(sz);
}

void read_exact(std::ifstream& in, void* dst, size_t bytes, const char* what) {
    if (bytes == 0) {
        return;
    }
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) {
        throw std::runtime_error(std::string("short read while reading ") + what);
    }
}

std::vector<uint8_t> read_range(const std::string& path, uint64_t offset, uint64_t size) {
    if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("range too large to read");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for reading: " + path);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("failed to seek file: " + path);
    }
    std::vector<uint8_t> out(static_cast<size_t>(size));
    read_exact(in, out.data(), out.size(), "range");
    return out;
}

void write_exact(std::fstream& out, const void* data, size_t bytes, const char* what) {
    if (bytes == 0) {
        return;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error(std::string("failed to write ") + what);
    }
}

void write_zero_bytes(std::fstream& out, uint64_t bytes) {
    std::vector<uint8_t> zero(1 << 20, 0);
    while (bytes > 0) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(bytes, zero.size()));
        write_exact(out, zero.data(), n, "zero padding");
        bytes -= n;
    }
}

void write_at(std::fstream& out, uint64_t offset, const void* data, size_t bytes, const char* what) {
    out.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!out) {
        throw std::runtime_error("failed to seek output file");
    }
    write_exact(out, data, bytes, what);
}

std::vector<uint8_t> compress_block_or_throw(const std::vector<uint8_t>& raw, int level) {
    auto compressed = compress_xz_block_native(raw.data(), raw.size(), level);
    if (compressed.empty() && !raw.empty()) {
        throw std::runtime_error("native LZMA compression failed");
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

struct CompressedBucketBlock {
    BucketBlockDirEntry dir{};
    std::vector<uint8_t> compressed;
    uint64_t raw_size = 0;
};

struct CompressedValueBlock {
    ValueBlockDirEntry dir{};
    std::vector<uint8_t> compressed;
    uint64_t raw_size = 0;
};

std::vector<uint8_t> decompress_block_or_throw(
    const uint8_t* compressed,
    size_t compressed_size,
    uint64_t expected_raw_size) {
    if (expected_raw_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("compressed block raw size too large");
    }
    auto raw = decompress_xz_block_native(compressed, compressed_size);
    if (raw.size() != static_cast<size_t>(expected_raw_size)) {
        throw std::runtime_error("native LZMA decompression size mismatch");
    }
    return raw;
}

SuccessRateKind success_kind_for_mode(EXAD::DTypeMode mode) {
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
        return SuccessRateKind::UInt32;
    case EXAD::DTypeMode::UInt64:
        return SuccessRateKind::UInt64;
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32:
        return SuccessRateKind::Float32;
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64:
        return SuccessRateKind::Float64;
    }
    return SuccessRateKind::UInt32;
}

template <typename T>
uint64_t raw_bits_for_value(T value) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return value;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return value;
    } else if constexpr (std::is_same_v<T, float>) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    } else {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
}

double numeric_from_raw_bits(uint64_t raw, EXAD::DTypeMode mode) {
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
        return static_cast<double>(static_cast<uint32_t>(raw)) / kFixed32Scale;
    case EXAD::DTypeMode::UInt64:
        return static_cast<double>(raw) / kFixed64Scale;
    case EXAD::DTypeMode::Float32: {
        uint32_t bits = static_cast<uint32_t>(raw);
        float v = 0.0f;
        std::memcpy(&v, &bits, sizeof(v));
        return static_cast<double>(v);
    }
    case EXAD::DTypeMode::Float64: {
        double v = 0.0;
        std::memcpy(&v, &raw, sizeof(v));
        return v;
    }
    case EXAD::DTypeMode::OneMinusFloat32: {
        uint32_t bits = static_cast<uint32_t>(raw);
        float v = 0.0f;
        std::memcpy(&v, &bits, sizeof(v));
        return static_cast<double>(v);
    }
    case EXAD::DTypeMode::OneMinusFloat64: {
        double v = 0.0;
        std::memcpy(&v, &raw, sizeof(v));
        return v;
    }
    }
    return 0.0;
}

size_t value_size_for_mode(EXAD::DTypeMode mode) {
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32:
        return 4;
    case EXAD::DTypeMode::UInt64:
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64:
        return 8;
    }
    return 0;
}

EXAD::DTypeMode read_exadbook_dtype_mode(const std::string& path) {
    EXAD::detail::SolvedFileHeader header{};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD solved layer: " + path);
    }
    read_exact(in, &header, sizeof(header), "EXAD solved header");
    if (std::memcmp(header.magic, EXAD::detail::kSolvedMagic, 8) != 0 ||
        header.version != EXAD::kSolvedFileVersion) {
        throw std::runtime_error("invalid EXAD solved layer header: " + path);
    }
    return static_cast<EXAD::DTypeMode>(header.dtype_mode);
}

template <typename Fn>
auto dispatch_dtype(EXAD::DTypeMode mode, Fn&& fn) {
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
        return fn.template operator()<uint32_t>();
    case EXAD::DTypeMode::UInt64:
        return fn.template operator()<uint64_t>();
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32:
        return fn.template operator()<float>();
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64:
        return fn.template operator()<double>();
    }
    throw std::runtime_error("unsupported EXAD dtype mode");
}

template <typename T>
uint64_t bitmap_bytes_for_bucket(
    const EXAD::Luts& luts,
    const EXAD::BucketEntry& bucket,
    uint32_t threshold_bits) {
    const uint32_t semantic_sum = EXAD::bucket_key_semantic_sum(bucket.key);
    const uint32_t group = EXAD::lut_group_index(semantic_sum);
    if (group >= luts.size_table.size()) {
        throw std::runtime_error("EXAD bucket semantic group is outside LUT");
    }
    const uint32_t valid_count = luts.size_table[group];
    if (valid_count <= threshold_bits) {
        return ZMaskFrozen::bytes_for_bits(valid_count);
    }
    return ZMaskFrozen::words_for_bits(valid_count) * sizeof(uint64_t);
}

template <typename T>
std::vector<PendingBucketBlock> build_bucket_blocks(
    const EXAD::SolvedLayer<T>& layer,
    const EXAD::Luts& luts,
    uint32_t target_raw_bytes,
    uint32_t hard_cap_bytes,
    std::array<SlotDirEntry, 48>& slots) {
    std::vector<PendingBucketBlock> blocks;
    const uint32_t target = std::max<uint32_t>(4096u, target_raw_bytes);
    const uint32_t hard_cap = std::max<uint32_t>(target, hard_cap_bytes);

    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const auto& set = layer.sets[slot];
        auto& dir = slots[slot];
        dir.ad_key = static_cast<int32_t>(slot) - 16;
        dir.row_width = layer.row_width[slot];
        dir.slot_row_base = layer.slot_row_base[slot];
        dir.slot_value_base = layer.slot_value_base[slot];
        dir.bucket_count = set.buckets.size();
        dir.live_board_count = set.live_board_count;
        dir.bucket_block_begin = blocks.size();
        dir.bucket_block_count = 0;
        dir.reserved0 = 0;

        uint32_t begin = 0;
        uint32_t count = 0;
        uint64_t raw_est = sizeof(BucketBlockRawHeader) + sizeof(uint32_t);
        for (uint32_t i = 0; i < set.buckets.size(); ++i) {
            const uint64_t bucket_bytes = 8ull + 4ull + 4ull +
                bitmap_bytes_for_bucket<T>(luts, set.buckets[i], layer.threshold_bits);
            if (count > 0 && raw_est + bucket_bytes > hard_cap) {
                blocks.push_back({slot, begin, count, raw_est});
                ++dir.bucket_block_count;
                begin = i;
                count = 0;
                raw_est = sizeof(BucketBlockRawHeader) + sizeof(uint32_t);
            }
            raw_est += bucket_bytes;
            ++count;
            if (count > 0 && raw_est >= target) {
                blocks.push_back({slot, begin, count, raw_est});
                ++dir.bucket_block_count;
                begin = i + 1;
                count = 0;
                raw_est = sizeof(BucketBlockRawHeader) + sizeof(uint32_t);
            }
        }
        if (count > 0) {
            blocks.push_back({slot, begin, count, raw_est});
            ++dir.bucket_block_count;
        }
    }
    return blocks;
}

template <typename T>
std::vector<uint8_t> build_bucket_block_raw(
    const EXAD::SolvedLayer<T>& layer,
    const EXAD::Luts& luts,
    const PendingBucketBlock& block) {
    const auto& set = layer.sets[block.slot];
    if (block.count == 0 || static_cast<uint64_t>(block.begin) + block.count > set.buckets.size()) {
        throw std::runtime_error("invalid EXAD bucket block range");
    }

    const uint64_t first_dense_offset = set.buckets[block.begin].dense_offset;
    std::vector<uint64_t> keys;
    std::vector<uint32_t> local_dense_offsets;
    std::vector<uint32_t> local_bitmap_offsets;
    keys.reserve(block.count);
    local_dense_offsets.reserve(block.count);
    local_bitmap_offsets.reserve(static_cast<size_t>(block.count) + 1);

    std::vector<uint8_t> small_payload;
    std::vector<uint64_t> large_payload_words;
    for (uint32_t i = 0; i < block.count; ++i) {
        const auto& bucket = set.buckets[block.begin + i];
        const uint32_t semantic_sum = EXAD::bucket_key_semantic_sum(bucket.key);
        const uint32_t group = EXAD::lut_group_index(semantic_sum);
        if (group >= luts.size_table.size()) {
            throw std::runtime_error("EXAD bucket semantic group is outside LUT");
        }
        const uint32_t valid_count = luts.size_table[group];
        keys.push_back(bucket.key);
        if (bucket.dense_offset < first_dense_offset ||
            bucket.dense_offset - first_dense_offset > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EXAD bucket dense offset delta is too large");
        }
        local_dense_offsets.push_back(static_cast<uint32_t>(bucket.dense_offset - first_dense_offset));
        if (valid_count <= layer.threshold_bits) {
            const uint64_t off = small_payload.size();
            if (off > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EXAD compressed small bitmap block is too large");
            }
            local_bitmap_offsets.push_back(static_cast<uint32_t>(off));
            const size_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
            if (bucket.bitmap_offset + bytes > set.small_bitmap_bytes.size()) {
                throw std::runtime_error("EXAD small bitmap offset outside source payload");
            }
            const auto* p = set.small_bitmap_bytes.data() + bucket.bitmap_offset;
            small_payload.insert(small_payload.end(), p, p + bytes);
        } else {
            const uint64_t off = large_payload_words.size();
            if (off > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EXAD compressed large bitmap block is too large");
            }
            local_bitmap_offsets.push_back(static_cast<uint32_t>(off));
            const size_t words = ZMaskFrozen::words_for_bits(valid_count);
            if (bucket.bitmap_offset + words > set.large_bitmap_words.size()) {
                throw std::runtime_error("EXAD large bitmap offset outside source payload");
            }
            const auto* p = set.large_bitmap_words.data() + bucket.bitmap_offset;
            large_payload_words.insert(large_payload_words.end(), p, p + words);
        }
    }
    local_bitmap_offsets.push_back(0);

    if (small_payload.size() > std::numeric_limits<uint32_t>::max() ||
        large_payload_words.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("EXAD compressed bucket block payload is too large");
    }

    BucketBlockRawHeader header{};
    header.bucket_count = block.count;
    header.small_bitmap_bytes = static_cast<uint32_t>(small_payload.size());
    header.large_bitmap_words = static_cast<uint32_t>(large_payload_words.size());
    header.first_dense_offset = first_dense_offset;

    std::vector<uint8_t> raw;
    raw.reserve(sizeof(header) +
                keys.size() * sizeof(uint64_t) +
                local_dense_offsets.size() * sizeof(uint32_t) +
                local_bitmap_offsets.size() * sizeof(uint32_t) +
                small_payload.size() +
                large_payload_words.size() * sizeof(uint64_t));
    append_unaligned(raw, header);
    for (const auto v : keys) {
        append_unaligned(raw, v);
    }
    for (const auto v : local_dense_offsets) {
        append_unaligned(raw, v);
    }
    for (const auto v : local_bitmap_offsets) {
        append_unaligned(raw, v);
    }
    raw.insert(raw.end(), small_payload.begin(), small_payload.end());
    const auto* large_bytes = reinterpret_cast<const uint8_t*>(large_payload_words.data());
    raw.insert(raw.end(), large_bytes, large_bytes + large_payload_words.size() * sizeof(uint64_t));
    return raw;
}

template <typename T>
CompressStats compress_layer_impl(
    const EXAD::SolvedLayer<T>& layer,
    const EXAD::Luts& luts,
    const std::string& source_label,
    uint64_t original_bytes,
    const std::string& output_path,
    EXAD::DTypeMode mode,
    uint32_t bucket_block_raw_target_bytes,
    uint32_t success_block_values,
    int compression_level) {
    const double t0 = wall_time_seconds();
    if (layer.lut_signature != 0 && luts.config_signature != 0 &&
        layer.lut_signature != luts.config_signature) {
        throw std::runtime_error("EXAD LUT signature does not match solved layer");
    }
    if (layer.physical_pattern_signature != luts.physical_pattern_signature ||
        layer.logical_pattern_signature != luts.logical_pattern_signature ||
        layer.physical_transform != luts.physical_transform ||
        layer.inverse_physical_transform != luts.inverse_physical_transform) {
        throw std::runtime_error("EXAD physical pattern metadata does not match solved layer");
    }

    std::array<SlotDirEntry, 48> slots{};
    auto bucket_blocks = build_bucket_blocks(layer, luts, bucket_block_raw_target_bytes,
                                             kBucketBlockHardCapBytes, slots);
    const uint64_t success_value_count = layer.success_values.size();
    const uint32_t value_block_values = std::max<uint32_t>(1, success_block_values);
    const uint64_t value_block_count =
        (success_value_count + value_block_values - 1) / value_block_values;

    FileHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(header.magic));
    header.version = kVersion;
    header.dtype_mode = static_cast<uint32_t>(mode);
    header.value_size = static_cast<uint32_t>(sizeof(T));
    header.original_board_sum = layer.original_board_sum;
    header.threshold_bits = layer.threshold_bits;
    header.slot_count = kSlotCount;
    header.bucket_block_raw_target_bytes = bucket_block_raw_target_bytes;
    header.bucket_block_raw_hard_cap_bytes = kBucketBlockHardCapBytes;
    header.success_block_values = value_block_values;
    header.lut_signature = layer.lut_signature;
    header.physical_transform = layer.physical_transform;
    header.inverse_physical_transform = layer.inverse_physical_transform;
    header.logical_pattern_signature = layer.logical_pattern_signature;
    header.physical_pattern_signature = layer.physical_pattern_signature;
    header.live_board_count = layer.live_board_count;
    header.success_value_count = success_value_count;
    header.bucket_block_count = bucket_blocks.size();
    header.value_block_count = value_block_count;
    header.slot_dir_offset = sizeof(FileHeader);
    header.bucket_dir_offset = header.slot_dir_offset + sizeof(SlotDirEntry) * kSlotCount;
    header.value_dir_offset = header.bucket_dir_offset + sizeof(BucketBlockDirEntry) * bucket_blocks.size();
    header.data_offset = header.value_dir_offset + sizeof(ValueBlockDirEntry) * value_block_count;
    header.original_file_size = original_bytes;

    std::vector<BucketBlockDirEntry> bucket_dirs(bucket_blocks.size());
    std::vector<ValueBlockDirEntry> value_dirs(value_block_count);

    std::filesystem::path out_path(output_path);
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::fstream out(output_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open EXAD compressed output: " + output_path);
    }
    write_exact(out, &header, sizeof(header), "EXAD compressed header");
    write_zero_bytes(out, sizeof(SlotDirEntry) * kSlotCount +
                          sizeof(BucketBlockDirEntry) * bucket_blocks.size() +
                          sizeof(ValueBlockDirEntry) * value_block_count);

    CompressStats stats;
    stats.source_path = source_label;
    stats.output_path = output_path;
    stats.original_bytes = header.original_file_size;
    stats.live_board_count = header.live_board_count;
    stats.success_value_count = header.success_value_count;
    stats.bucket_block_count = bucket_blocks.size();
    stats.value_block_count = value_block_count;

    uint64_t data_offset = header.data_offset;
    const uint32_t bucket_workers = compression_worker_count(bucket_blocks.size());
    for (size_t batch_begin = 0; batch_begin < bucket_blocks.size(); batch_begin += bucket_workers) {
        const size_t batch_end = std::min<size_t>(bucket_blocks.size(), batch_begin + bucket_workers);
        std::vector<std::future<CompressedBucketBlock>> futures;
        futures.reserve(batch_end - batch_begin);
        for (size_t i = batch_begin; i < batch_end; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&, i]() {
                const auto& block = bucket_blocks[i];
                const auto& set = layer.sets[block.slot];
                auto raw = build_bucket_block_raw(layer, luts, block);
                CompressedBucketBlock result;
                result.compressed = compress_block_or_throw(raw, compression_level);
                result.raw_size = raw.size();
                result.dir.slot = block.slot;
                result.dir.first_bucket_index = block.begin;
                result.dir.bucket_count = block.count;
                result.dir.first_key = set.buckets[block.begin].key;
                result.dir.last_key = set.buckets[block.begin + block.count - 1].key;
                result.dir.first_dense_offset = set.buckets[block.begin].dense_offset;
                result.dir.compressed_size = result.compressed.size();
                result.dir.raw_size = result.raw_size;
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedBucketBlock result = futures[local].get();
            result.dir.compressed_offset = data_offset;
            bucket_dirs[batch_begin + local] = result.dir;
            out.seekp(static_cast<std::streamoff>(data_offset), std::ios::beg);
            write_exact(out, result.compressed.data(), result.compressed.size(), "EXAD compressed bucket block");
            data_offset += result.compressed.size();
            stats.bucket_raw_bytes += result.raw_size;
            stats.bucket_compressed_bytes += result.compressed.size();
        }
    }

    const auto* values_bytes = reinterpret_cast<const uint8_t*>(layer.success_values.data());
    const uint32_t value_workers = compression_worker_count(value_block_count);
    for (uint64_t batch_begin = 0; batch_begin < value_block_count; batch_begin += value_workers) {
        const uint64_t batch_end = std::min<uint64_t>(value_block_count, batch_begin + value_workers);
        std::vector<std::future<CompressedValueBlock>> futures;
        futures.reserve(static_cast<size_t>(batch_end - batch_begin));
        for (uint64_t block = batch_begin; block < batch_end; ++block) {
            futures.emplace_back(std::async(std::launch::async, [&, block]() {
                const uint64_t first = block * value_block_values;
                const uint32_t count = static_cast<uint32_t>(
                    std::min<uint64_t>(value_block_values, success_value_count - first));
                const uint64_t raw_size = static_cast<uint64_t>(count) * sizeof(T);
                std::vector<uint8_t> raw(values_bytes + first * sizeof(T),
                                         values_bytes + first * sizeof(T) + raw_size);
                CompressedValueBlock result;
                result.compressed = compress_block_or_throw(raw, compression_level);
                result.raw_size = raw_size;
                result.dir.first_value_index = first;
                result.dir.value_count = count;
                result.dir.value_size = static_cast<uint32_t>(sizeof(T));
                result.dir.compressed_size = result.compressed.size();
                result.dir.raw_size = raw_size;
                return result;
            }));
        }
        for (size_t local = 0; local < futures.size(); ++local) {
            CompressedValueBlock result = futures[local].get();
            result.dir.compressed_offset = data_offset;
            value_dirs[static_cast<size_t>(batch_begin) + local] = result.dir;
            out.seekp(static_cast<std::streamoff>(data_offset), std::ios::beg);
            write_exact(out, result.compressed.data(), result.compressed.size(), "EXAD compressed value block");
            data_offset += result.compressed.size();
            stats.value_raw_bytes += result.raw_size;
            stats.value_compressed_bytes += result.compressed.size();
        }
    }

    out.flush();
    header.original_file_size = stats.original_bytes;
    write_at(out, 0, &header, sizeof(header), "EXAD compressed header");
    write_at(out, header.slot_dir_offset, slots.data(), sizeof(SlotDirEntry) * slots.size(), "EXAD slot dir");
    if (!bucket_dirs.empty()) {
        write_at(out, header.bucket_dir_offset, bucket_dirs.data(),
                 sizeof(BucketBlockDirEntry) * bucket_dirs.size(), "EXAD bucket dir");
    }
    if (!value_dirs.empty()) {
        write_at(out, header.value_dir_offset, value_dirs.data(),
                 sizeof(ValueBlockDirEntry) * value_dirs.size(), "EXAD value dir");
    }
    out.close();

    stats.compressed_bytes = file_size_or_zero(output_path);
    stats.compression_seconds = wall_time_seconds() - t0;
    return stats;
}

template <typename T>
CompressStats compress_impl(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    const std::string& output_path,
    EXAD::DTypeMode mode,
    uint32_t bucket_block_raw_target_bytes,
    uint32_t success_block_values,
    int compression_level) {
    const auto luts = EXAD::read_lut_file(exadlut_path);
    auto layer = EXAD::read_solved_layer_file<T>(exadbook_path, mode, {});
    return compress_layer_impl(
        layer,
        luts,
        exadbook_path,
        file_size_or_zero(exadbook_path),
        output_path,
        mode,
        bucket_block_raw_target_bytes,
        success_block_values,
        compression_level
    );
}

struct CompressedIndex {
    FileHeader header{};
    std::array<SlotDirEntry, 48> slots{};
    std::vector<BucketBlockDirEntry> bucket_dirs;
};

struct SolvedSlotLayout {
    EXAD::detail::SolvedSlotHeader header{};
    uint64_t bucket_entries_offset = 0;
    uint64_t small_bitmap_offset = 0;
    uint64_t large_bitmap_offset = 0;
};

struct SolvedFileIndex {
    EXAD::detail::SolvedFileHeader header{};
    std::array<SolvedSlotLayout, 48> slots{};
    uint64_t success_values_offset = 0;
    EXAD::DTypeMode mode = EXAD::DTypeMode::UInt32;
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
std::unordered_map<std::string, CachedFileEntry<ExadLutPointIndex>> g_exad_lut_point_cache;
std::unordered_map<std::string, CachedFileEntry<CompressedIndex>> g_compressed_index_cache;
std::unordered_map<std::string, CachedFileEntry<SolvedFileIndex>> g_solved_index_cache;

FileStamp file_stamp(const std::string& path) {
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

bool same_stamp(const FileStamp& lhs, const FileStamp& rhs) {
    return lhs.valid && rhs.valid && lhs.size == rhs.size && lhs.write_time == rhs.write_time;
}

CompressedIndex read_index(const std::string& path) {
    CompressedIndex index;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD compressed file: " + path);
    }
    read_exact(in, &index.header, sizeof(index.header), "EXAD compressed header");
    if (std::memcmp(index.header.magic, kMagic, sizeof(kMagic)) != 0 ||
        index.header.version != kVersion ||
        index.header.slot_count != kSlotCount) {
        throw std::runtime_error("invalid EXAD compressed file header: " + path);
    }
    index.bucket_dirs.resize(static_cast<size_t>(index.header.bucket_block_count));
    in.seekg(static_cast<std::streamoff>(index.header.slot_dir_offset), std::ios::beg);
    read_exact(in, index.slots.data(), sizeof(SlotDirEntry) * index.slots.size(), "EXAD slot dir");
    if (!index.bucket_dirs.empty()) {
        in.seekg(static_cast<std::streamoff>(index.header.bucket_dir_offset), std::ios::beg);
        read_exact(in, index.bucket_dirs.data(),
                   sizeof(BucketBlockDirEntry) * index.bucket_dirs.size(), "EXAD bucket dir");
    }
    return index;
}

template <typename T>
T read_one_from(std::ifstream& in, uint64_t offset, const char* what) {
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error(std::string("failed to seek while reading ") + what);
    }
    T value{};
    read_exact(in, &value, sizeof(value), what);
    return value;
}

std::vector<uint8_t> read_range_from(std::ifstream& in, uint64_t offset, uint64_t size, const char* what) {
    if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(std::string("range too large while reading ") + what);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error(std::string("failed to seek while reading ") + what);
    }
    std::vector<uint8_t> out(static_cast<size_t>(size));
    read_exact(in, out.data(), out.size(), what);
    return out;
}

SolvedFileIndex read_solved_file_index(std::ifstream& in, const std::string& path) {
    SolvedFileIndex index;
    in.seekg(0, std::ios::beg);
    read_exact(in, &index.header, sizeof(index.header), "EXAD solved header");
    if (std::memcmp(index.header.magic, EXAD::detail::kSolvedMagic, sizeof(index.header.magic)) != 0 ||
        index.header.version != EXAD::kSolvedFileVersion ||
        index.header.slot_count != kSlotCount) {
        throw std::runtime_error("invalid EXAD solved layer header: " + path);
    }
    index.mode = static_cast<EXAD::DTypeMode>(index.header.dtype_mode);
    const uint32_t expected_value_size = EXAD::dtype_value_size(index.mode);
    if (index.header.value_size != expected_value_size || expected_value_size == 0U) {
        throw std::runtime_error("EXAD solved layer value size mismatch: " + path);
    }
    std::array<EXAD::detail::SolvedSlotHeader, 48> slot_headers{};
    read_exact(in, slot_headers.data(), sizeof(slot_headers), "EXAD solved slot dir");

    uint64_t offset = sizeof(EXAD::detail::SolvedFileHeader) + sizeof(slot_headers);
    for (size_t slot = 0; slot < kSlotCount; ++slot) {
        auto& layout = index.slots[slot];
        layout.header = slot_headers[slot];
        layout.bucket_entries_offset = offset;
        offset += slot_headers[slot].bucket_count * sizeof(EXAD::BucketEntry);
        layout.small_bitmap_offset = offset;
        offset += slot_headers[slot].small_bitmap_bytes;
        layout.large_bitmap_offset = offset;
        offset += slot_headers[slot].large_bitmap_words * sizeof(uint64_t);
    }
    index.success_values_offset = offset;
    const uint64_t expected_size = offset +
        index.header.success_value_count * static_cast<uint64_t>(index.header.value_size);
    const uint64_t actual_size = file_size_or_zero(path);
    if (actual_size != 0 && actual_size < expected_size) {
        throw std::runtime_error("truncated EXAD solved layer: " + path);
    }
    return index;
}

ExadLutFileLayout exad_lut_file_layout(const ExadLutFileHeader& header) {
    ExadLutFileLayout layout;
    layout.valid_suffix_masks_offset = sizeof(ExadLutFileHeader);
    layout.rank_table_sizes_offset =
        layout.valid_suffix_masks_offset + static_cast<uint64_t>(header.valid_mask_count) * sizeof(uint32_t);
    layout.rank_tables_offset =
        layout.rank_table_sizes_offset + header.rank_table_count * sizeof(uint64_t);
    layout.packed_rank_pair_offset =
        layout.rank_tables_offset + header.rank_table_values * sizeof(uint16_t);
    layout.size_table_offset =
        layout.packed_rank_pair_offset + header.packed_rank_pair_values * sizeof(uint32_t);
    layout.offset_table_offset =
        layout.size_table_offset + header.size_table_values * sizeof(uint32_t);
    layout.unrank_array_offset =
        layout.offset_table_offset + header.offset_table_values * sizeof(uint32_t);
    layout.high_base_offset =
        layout.unrank_array_offset + header.unrank_array_values * sizeof(uint32_t);
    return layout;
}

uint32_t low24_sum_direct(uint32_t low24) {
    uint32_t sum = 0U;
    for (uint32_t cell = 0; cell < 6U; ++cell) {
        sum += EXAD::tile_value((low24 >> (cell * 4U)) & 0xFU);
    }
    return sum;
}

ExadLutPointIndex read_exad_lut_point_index(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD LUT: " + path);
    }
    ExadLutPointIndex index;
    read_exact(in, &index.header, sizeof(index.header), "EXAD LUT header");
    if (std::memcmp(index.header.magic, kExadLutMagic, sizeof(kExadLutMagic)) != 0 ||
        index.header.version != 2U) {
        throw std::runtime_error("invalid EXAD LUT file: " + path);
    }
    if (index.header.size_table_values == 0U ||
        index.header.offset_table_values == 0U ||
        index.header.unrank_array_values == 0U ||
        index.header.high_base_values == 0U) {
        throw std::runtime_error("invalid EXAD LUT table shape: " + path);
    }
    index.layout = exad_lut_file_layout(index.header);
    index.rank_table_sizes.resize(static_cast<size_t>(index.header.rank_table_count));
    if (!index.rank_table_sizes.empty()) {
        in.seekg(static_cast<std::streamoff>(index.layout.rank_table_sizes_offset), std::ios::beg);
        read_exact(
            in,
            index.rank_table_sizes.data(),
            index.rank_table_sizes.size() * sizeof(uint64_t),
            "EXAD LUT rank table sizes");
    }
    index.rank_table_offsets.resize(index.rank_table_sizes.size());
    uint64_t offset = index.layout.rank_tables_offset;
    uint64_t total_values = 0U;
    for (size_t i = 0; i < index.rank_table_sizes.size(); ++i) {
        index.rank_table_offsets[i] = offset;
        total_values += index.rank_table_sizes[i];
        offset += index.rank_table_sizes[i] * sizeof(uint16_t);
    }
    if (total_values != index.header.rank_table_values) {
        throw std::runtime_error("EXAD LUT rank table size mismatch: " + path);
    }
    return index;
}

std::shared_ptr<const ExadLutPointIndex> cached_exad_lut_point_index(const std::string& path) {
    const FileStamp stamp = file_stamp(path);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_exad_lut_point_cache.find(path);
        if (it != g_exad_lut_point_cache.end() && same_stamp(it->second.stamp, stamp) && it->second.value) {
            return it->second.value;
        }
    }

    auto loaded = std::make_shared<ExadLutPointIndex>(read_exad_lut_point_index(path));
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_exad_lut_point_cache[path] = CachedFileEntry<ExadLutPointIndex>{stamp, loaded};
    }
    return loaded;
}

class ExadLutPointReader {
public:
    explicit ExadLutPointReader(const std::string& path)
        : index_(cached_exad_lut_point_index(path)),
          in_(path, std::ios::binary) {
        if (!in_) {
            throw std::runtime_error("failed to open EXAD LUT: " + path);
        }
    }

    uint64_t signature() const {
        return index_->header.config_signature;
    }

    const ExadLutFileHeader& header() const {
        return index_->header;
    }

    bool valid_count_for_group(uint32_t group, uint32_t& valid_count) {
        valid_count = 0U;
        const auto& header = index_->header;
        if (group >= header.size_table_values) {
            return false;
        }
        valid_count = read_one_from<uint32_t>(
            in_,
            index_->layout.size_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            "EXAD LUT size table");
        return true;
    }

    bool valid_count_for_semantic_sum(uint32_t semantic_sum, uint32_t& valid_count) {
        valid_count = 0U;
        if ((semantic_sum & 1U) != 0U) {
            return false;
        }
        return valid_count_for_group(semantic_sum >> 1U, valid_count);
    }

    bool suffix28_query(
        uint32_t suffix28,
        uint32_t& group,
        uint32_t& rank,
        uint32_t& semantic_sum,
        uint32_t& valid_count) {
        group = 0U;
        rank = 0U;
        semantic_sum = 0U;
        valid_count = 0U;

        const auto& header = index_->header;
        const uint32_t high = suffix28 >> 24U;
        const uint8_t table_id = header.table_for_high[high];
        if (table_id == 0xFFU) {
            return false;
        }
        const uint32_t low24 = suffix28 & 0xFFFFFFU;
        const uint16_t low_rank = low24_rank(table_id, low24);
        if (low_rank == ZMaskFrozen::kInvalidRank) {
            return false;
        }

        const uint32_t sum = low24_sum_direct(low24) + EXAD::tile_value(high);
        if (sum > EXAD::kMaxSemanticSuffixSum || (sum & 1U) != 0U) {
            return false;
        }
        group = sum >> 1U;
        if (!valid_count_for_group(group, valid_count) || valid_count == 0U) {
            return false;
        }

        const uint64_t high_base_index =
            static_cast<uint64_t>(high) * header.size_table_values + group;
        if (high_base_index >= header.high_base_values) {
            return false;
        }
        const uint16_t high_base = read_one_from<uint16_t>(
            in_,
            index_->layout.high_base_offset + high_base_index * sizeof(uint16_t),
            "EXAD LUT high base");
        rank = static_cast<uint32_t>(high_base) + low_rank;
        if (rank >= valid_count) {
            return false;
        }
        semantic_sum = sum;
        return true;
    }

    bool reconstruct_board_from_key_rank(uint64_t key, uint32_t rank, uint64_t& board) {
        board = 0ULL;
        const uint32_t semantic_sum = EXAD::bucket_key_semantic_sum(key);
        uint32_t valid_count = 0U;
        if (!valid_count_for_semantic_sum(semantic_sum, valid_count) || rank >= valid_count) {
            return false;
        }
        const uint32_t group = semantic_sum >> 1U;
        if (group >= index_->header.offset_table_values) {
            return false;
        }
        const uint32_t unrank_offset = read_one_from<uint32_t>(
            in_,
            index_->layout.offset_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            "EXAD LUT offset table");
        const uint64_t unrank_index = static_cast<uint64_t>(unrank_offset) + rank;
        if (unrank_index >= index_->header.unrank_array_values) {
            return false;
        }
        const uint32_t suffix28 = read_one_from<uint32_t>(
            in_,
            index_->layout.unrank_array_offset + unrank_index * sizeof(uint32_t),
            "EXAD LUT unrank array");
        const uint64_t prefix36 = EXAD::bucket_key_prefix36(key);
        board = (prefix36 << EXAD::kSuffixBits) | static_cast<uint64_t>(suffix28);
        return true;
    }

private:
    uint16_t low24_rank(uint8_t table_id, uint32_t low24) {
        const auto& header = index_->header;
        if (header.packed_rank_pair_values != 0U &&
            (table_id == header.packed_table0 || table_id == header.packed_table1)) {
            if (low24 >= header.packed_rank_pair_values) {
                return ZMaskFrozen::kInvalidRank;
            }
            const uint32_t packed = read_one_from<uint32_t>(
                in_,
                index_->layout.packed_rank_pair_offset + static_cast<uint64_t>(low24) * sizeof(uint32_t),
                "EXAD LUT packed rank pair");
            return table_id == header.packed_table0
                ? static_cast<uint16_t>(packed & 0xFFFFU)
                : static_cast<uint16_t>(packed >> 16U);
        }
        if (table_id >= index_->rank_table_offsets.size() ||
            low24 >= index_->rank_table_sizes[table_id]) {
            return ZMaskFrozen::kInvalidRank;
        }
        return read_one_from<uint16_t>(
            in_,
            index_->rank_table_offsets[table_id] + static_cast<uint64_t>(low24) * sizeof(uint16_t),
            "EXAD LUT rank table");
    }

    std::shared_ptr<const ExadLutPointIndex> index_;
    std::ifstream in_;
};

template <typename Header>
bool physical_metadata_matches_lut(const Header& header, const ExadLutFileHeader& lut_header) {
    return header.physical_transform == lut_header.physical_transform &&
        header.inverse_physical_transform == lut_header.inverse_physical_transform &&
        header.logical_pattern_signature == lut_header.logical_pattern_signature &&
        header.physical_pattern_signature == lut_header.physical_pattern_signature;
}

std::shared_ptr<const CompressedIndex> cached_compressed_index(const std::string& path) {
    const FileStamp stamp = file_stamp(path);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_compressed_index_cache.find(path);
        if (it != g_compressed_index_cache.end() && same_stamp(it->second.stamp, stamp) && it->second.value) {
            return it->second.value;
        }
    }

    auto loaded = std::make_shared<CompressedIndex>(read_index(path));
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_compressed_index_cache[path] = CachedFileEntry<CompressedIndex>{stamp, loaded};
    }
    return loaded;
}

std::shared_ptr<const SolvedFileIndex> cached_solved_file_index(const std::string& path) {
    const FileStamp stamp = file_stamp(path);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_solved_index_cache.find(path);
        if (it != g_solved_index_cache.end() && same_stamp(it->second.stamp, stamp) && it->second.value) {
            return it->second.value;
        }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD solved layer: " + path);
    }
    auto loaded = std::make_shared<SolvedFileIndex>(read_solved_file_index(in, path));
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_solved_index_cache[path] = CachedFileEntry<SolvedFileIndex>{stamp, loaded};
    }
    return loaded;
}

int slot_index_from_ad_key(int ad_key) {
    const int slot = ad_key + 16;
    if (slot < 0 || slot >= static_cast<int>(kSlotCount)) {
        return -1;
    }
    return slot;
}

bool test_bit_raw(const uint8_t* bitmap, uint32_t rank) {
    return (bitmap[rank >> 3] & static_cast<uint8_t>(1u << (rank & 7))) != 0;
}

uint32_t popcount_u32(uint32_t value);
uint32_t popcount_u64(uint64_t value);

uint32_t dense_ordinal_small_raw(const uint8_t* bitmap, uint32_t rank) {
    uint32_t count = 0;
    const uint32_t full_bytes = rank >> 3;
    for (uint32_t i = 0; i < full_bytes; ++i) {
        count += popcount_u32(bitmap[i]);
    }
    const uint32_t rem = rank & 7;
    if (rem != 0) {
        const uint8_t mask = static_cast<uint8_t>((1u << rem) - 1u);
        count += popcount_u32(bitmap[full_bytes] & mask);
    }
    return count;
}

uint32_t dense_ordinal_large_raw(const uint8_t* bitmap, uint32_t rank) {
    uint32_t count = 0;
    const uint32_t word_idx = rank >> 6;
    const uint32_t bit_idx = rank & 63;
    for (uint32_t i = 0; i < word_idx; ++i) {
        count += popcount_u64(load_unaligned<uint64_t>(bitmap + static_cast<size_t>(i) * sizeof(uint64_t)));
    }
    if (bit_idx != 0) {
        const uint64_t mask = (uint64_t{1} << bit_idx) - 1ull;
        count += popcount_u64(
            load_unaligned<uint64_t>(bitmap + static_cast<size_t>(word_idx) * sizeof(uint64_t)) & mask);
    }
    return count;
}

bool read_value_block_dir_for_index(
    std::ifstream& in,
    const CompressedIndex& index,
    uint64_t value_index,
    ValueBlockDirEntry& entry) {
    if (index.header.success_block_values == 0U ||
        value_index >= index.header.success_value_count) {
        return false;
    }
    const uint64_t block_index = value_index / index.header.success_block_values;
    if (block_index >= index.header.value_block_count) {
        return false;
    }
    entry = read_one_from<ValueBlockDirEntry>(
        in,
        index.header.value_dir_offset + block_index * sizeof(ValueBlockDirEntry),
        "EXAD compressed value dir entry");
    return entry.first_value_index <= value_index &&
        value_index < entry.first_value_index + entry.value_count;
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
    return popcount_u32(static_cast<uint32_t>(value)) +
           popcount_u32(static_cast<uint32_t>(value >> 32U));
#else
    return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
}

ColdLookupResult cold_miss(EXAD::DTypeMode mode) {
    ColdLookupResult result;
    result.success_kind = success_kind_for_mode(mode);
    return result;
}

uint32_t trailing_zero_u32(uint32_t value) {
    uint32_t count = 0;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++count;
    }
    return count;
}

uint32_t trailing_zero_u64(uint64_t value) {
    uint32_t count = 0;
    while ((value & 1ULL) == 0ULL) {
        value >>= 1U;
        ++count;
    }
    return count;
}

bool rank_for_small_bitmap_ordinal(
    const uint8_t* bitmap,
    uint32_t valid_count,
    uint64_t ordinal,
    uint32_t& rank) {
    rank = std::numeric_limits<uint32_t>::max();
    if (valid_count == 0U) {
        return false;
    }
    const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
    for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
        uint8_t value = bitmap[byte_idx];
        if (byte_idx + 1U == bytes && (valid_count & 7U) != 0U) {
            value = static_cast<uint8_t>(value & static_cast<uint8_t>((1U << (valid_count & 7U)) - 1U));
        }
        const uint32_t count = popcount_u32(value);
        if (ordinal >= count) {
            ordinal -= count;
            continue;
        }
        while (value != 0U) {
            const uint32_t bit = trailing_zero_u32(value);
            const uint32_t candidate = byte_idx * 8U + bit;
            if (candidate >= valid_count) {
                break;
            }
            if (ordinal == 0U) {
                rank = candidate;
                return true;
            }
            --ordinal;
            value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
        }
    }
    return false;
}

bool rank_for_large_bitmap_ordinal(
    const uint8_t* bitmap,
    uint32_t valid_count,
    uint64_t ordinal,
    uint32_t& rank) {
    rank = std::numeric_limits<uint32_t>::max();
    if (valid_count == 0U) {
        return false;
    }
    const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
    for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
        uint64_t value = load_unaligned<uint64_t>(
            bitmap + static_cast<size_t>(word_idx) * sizeof(uint64_t));
        if (word_idx + 1U == words && (valid_count & 63U) != 0U) {
            value &= (1ULL << (valid_count & 63U)) - 1ULL;
        }
        const uint32_t count = popcount_u64(value);
        if (ordinal >= count) {
            ordinal -= count;
            continue;
        }
        while (value != 0ULL) {
            const uint32_t bit = trailing_zero_u64(value);
            const uint32_t candidate = word_idx * 64U + bit;
            if (candidate >= valid_count) {
                break;
            }
            if (ordinal == 0U) {
                rank = candidate;
                return true;
            }
            --ordinal;
            value &= value - 1ULL;
        }
    }
    return false;
}

bool reconstruct_board_from_key_rank(
    ExadLutPointReader& lut,
    uint64_t key,
    uint32_t rank,
    uint64_t& board) {
    return lut.reconstruct_board_from_key_rank(key, rank, board);
}

int select_solved_slot_by_live(
    const SolvedFileIndex& index,
    uint64_t target,
    uint64_t& local_row) {
    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const uint64_t count = index.slots[slot].header.live_board_count;
        if (target < count) {
            local_row = target;
            return static_cast<int>(slot);
        }
        target -= count;
    }
    return -1;
}

int select_compressed_slot_by_live(
    const CompressedIndex& index,
    uint64_t target,
    uint64_t& local_row) {
    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const uint64_t count = index.slots[slot].live_board_count;
        if (target < count) {
            local_row = target;
            return static_cast<int>(slot);
        }
        target -= count;
    }
    return -1;
}

bool read_solved_bucket_for_local_row(
    std::ifstream& in,
    const SolvedSlotLayout& slot,
    uint64_t local_row,
    EXAD::BucketEntry& bucket) {
    if (slot.header.bucket_count == 0U) {
        return false;
    }
    uint64_t left = 0;
    uint64_t right = slot.header.bucket_count;
    while (left < right) {
        const uint64_t mid = left + ((right - left) >> 1U);
        const auto probe = read_one_from<EXAD::BucketEntry>(
            in,
            slot.bucket_entries_offset + mid * sizeof(EXAD::BucketEntry),
            "EXAD solved bucket entry");
        if (static_cast<uint64_t>(probe.dense_offset) <= local_row) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    if (left == 0U) {
        return false;
    }
    bucket = read_one_from<EXAD::BucketEntry>(
        in,
        slot.bucket_entries_offset + (left - 1U) * sizeof(EXAD::BucketEntry),
        "EXAD solved bucket entry");
    return static_cast<uint64_t>(bucket.dense_offset) <= local_row;
}

bool sample_from_solved_bucket(
    std::ifstream& in,
    const SolvedFileIndex& index,
    const SolvedSlotLayout& slot,
    ExadLutPointReader& lut,
    uint64_t local_row,
    const EXAD::BucketEntry& bucket,
    uint64_t& board) {
    const uint32_t semantic_sum = EXAD::bucket_key_semantic_sum(bucket.key);
    uint32_t valid_count = 0U;
    if (!lut.valid_count_for_semantic_sum(semantic_sum, valid_count)) {
        return false;
    }
    const uint64_t ordinal = local_row - static_cast<uint64_t>(bucket.dense_offset);
    uint32_t rank = std::numeric_limits<uint32_t>::max();
    if (valid_count <= index.header.threshold_bits) {
        const size_t bitmap_bytes = ZMaskFrozen::bytes_for_bits(valid_count);
        if (static_cast<uint64_t>(bucket.bitmap_offset) + bitmap_bytes > slot.header.small_bitmap_bytes) {
            throw std::runtime_error("EXAD solved small bitmap offset outside file");
        }
        const auto bitmap = read_range_from(
            in,
            slot.small_bitmap_offset + bucket.bitmap_offset,
            bitmap_bytes,
            "EXAD solved small bitmap");
        if (!rank_for_small_bitmap_ordinal(bitmap.data(), valid_count, ordinal, rank)) {
            return false;
        }
    } else {
        const size_t bitmap_words = ZMaskFrozen::words_for_bits(valid_count);
        if (static_cast<uint64_t>(bucket.bitmap_offset) + bitmap_words > slot.header.large_bitmap_words) {
            throw std::runtime_error("EXAD solved large bitmap offset outside file");
        }
        const auto bitmap = read_range_from(
            in,
            slot.large_bitmap_offset + static_cast<uint64_t>(bucket.bitmap_offset) * sizeof(uint64_t),
            bitmap_words * sizeof(uint64_t),
            "EXAD solved large bitmap");
        if (!rank_for_large_bitmap_ordinal(bitmap.data(), valid_count, ordinal, rank)) {
            return false;
        }
    }
    return reconstruct_board_from_key_rank(lut, bucket.key, rank, board);
}

} // namespace

bool is_exad_compressed_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    return in && std::memcmp(magic, kMagic, sizeof(kMagic)) == 0;
}

CompressStats compress_exad_solved_layer_to_result(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    const std::string& output_path,
    uint32_t bucket_block_raw_target_bytes,
    uint32_t success_block_values,
    int compression_level) {
    const auto mode = read_exadbook_dtype_mode(exadbook_path);
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
        return compress_impl<uint32_t>(exadbook_path, exadlut_path, output_path, mode,
                                       bucket_block_raw_target_bytes, success_block_values,
                                       compression_level);
    case EXAD::DTypeMode::UInt64:
        return compress_impl<uint64_t>(exadbook_path, exadlut_path, output_path, mode,
                                       bucket_block_raw_target_bytes, success_block_values,
                                       compression_level);
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32:
        return compress_impl<float>(exadbook_path, exadlut_path, output_path, mode,
                                    bucket_block_raw_target_bytes, success_block_values,
                                    compression_level);
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64:
        return compress_impl<double>(exadbook_path, exadlut_path, output_path, mode,
                                     bucket_block_raw_target_bytes, success_block_values,
                                     compression_level);
    }
    throw std::runtime_error("unsupported EXAD dtype mode");
}

template <typename T>
CompressStats compress_exad_solved_layer_to_result_from_memory(
    const EXAD::SolvedLayer<T>& layer,
    const EXAD::Luts& luts,
    const std::string& source_label,
    const std::string& output_path,
    uint32_t bucket_block_raw_target_bytes,
    uint32_t success_block_values,
    int compression_level) {
    return compress_layer_impl(
        layer,
        luts,
        source_label,
        EXAD::solved_serialized_size(layer),
        output_path,
        layer.dtype_mode,
        bucket_block_raw_target_bytes,
        success_block_values,
        compression_level
    );
}

template CompressStats compress_exad_solved_layer_to_result_from_memory<uint32_t>(
    const EXAD::SolvedLayer<uint32_t>&,
    const EXAD::Luts&,
    const std::string&,
    const std::string&,
    uint32_t,
    uint32_t,
    int);
template CompressStats compress_exad_solved_layer_to_result_from_memory<uint64_t>(
    const EXAD::SolvedLayer<uint64_t>&,
    const EXAD::Luts&,
    const std::string&,
    const std::string&,
    uint32_t,
    uint32_t,
    int);
template CompressStats compress_exad_solved_layer_to_result_from_memory<float>(
    const EXAD::SolvedLayer<float>&,
    const EXAD::Luts&,
    const std::string&,
    const std::string&,
    uint32_t,
    uint32_t,
    int);
template CompressStats compress_exad_solved_layer_to_result_from_memory<double>(
    const EXAD::SolvedLayer<double>&,
    const EXAD::Luts&,
    const std::string&,
    const std::string&,
    uint32_t,
    uint32_t,
    int);

ColdLookupResult lookup_exad_cold(
    const std::string& compressed_path,
    const std::string& exadlut_path,
    int ad_key,
    uint64_t canonical_board,
    uint32_t column) {
    const auto index_ptr = cached_compressed_index(compressed_path);
    const auto& index = *index_ptr;
    const auto mode = static_cast<EXAD::DTypeMode>(index.header.dtype_mode);
    ExadLutPointReader lut(exadlut_path);
    if (index.header.lut_signature != 0 && lut.signature() != 0 &&
        index.header.lut_signature != lut.signature()) {
        throw std::runtime_error("EXAD LUT signature does not match compressed layer");
    }
    if (!physical_metadata_matches_lut(index.header, lut.header())) {
        throw std::runtime_error("EXAD physical pattern metadata does not match compressed layer");
    }

    ColdLookupResult result;
    result.success_kind = success_kind_for_mode(mode);
    const int slot_i = slot_index_from_ad_key(ad_key);
    if (slot_i < 0) {
        return result;
    }
    const auto& slot = index.slots[slot_i];
    if (slot.bucket_count == 0 || column >= slot.row_width) {
        return result;
    }

    uint32_t group = 0;
    uint32_t rank = 0;
    uint32_t semantic_sum = 0;
    uint32_t valid_count = 0;
    if (!lut.suffix28_query(
            static_cast<uint32_t>(canonical_board & EXAD::kSuffixMask),
            group,
            rank,
            semantic_sum,
            valid_count)) {
        return result;
    }
    (void)group;
    if (rank >= valid_count) {
        return result;
    }
    const uint64_t key = EXAD::pack_bucket_key(canonical_board >> EXAD::kSuffixBits, semantic_sum);

    const uint64_t block_begin = slot.bucket_block_begin;
    const uint64_t block_end = block_begin + slot.bucket_block_count;
    if (block_end > index.bucket_dirs.size()) {
        throw std::runtime_error("EXAD compressed slot dir points outside bucket dir");
    }
    auto first = index.bucket_dirs.begin() + static_cast<std::ptrdiff_t>(block_begin);
    auto last = index.bucket_dirs.begin() + static_cast<std::ptrdiff_t>(block_end);
    auto block_it = std::lower_bound(
        first, last, key,
        [](const BucketBlockDirEntry& dir, uint64_t k) {
            return dir.last_key < k;
        });
    if (block_it == last || key < block_it->first_key || key > block_it->last_key) {
        return result;
    }

    std::ifstream compressed_in(compressed_path, std::ios::binary);
    if (!compressed_in) {
        throw std::runtime_error("failed to open EXAD compressed file: " + compressed_path);
    }

    const auto compressed_bucket = read_range_from(
        compressed_in,
        block_it->compressed_offset,
        block_it->compressed_size,
        "EXAD compressed bucket block");
    auto raw_bucket = decompress_block_or_throw(compressed_bucket.data(), compressed_bucket.size(), block_it->raw_size);
    result.bucket_block_raw_bytes = block_it->raw_size;
    result.bucket_block_compressed_bytes = block_it->compressed_size;
    if (raw_bucket.size() < sizeof(BucketBlockRawHeader)) {
        throw std::runtime_error("EXAD compressed bucket raw block is truncated");
    }
    const auto raw_header = load_unaligned<BucketBlockRawHeader>(raw_bucket.data());
    if (raw_header.bucket_count != block_it->bucket_count) {
        throw std::runtime_error("EXAD compressed bucket count mismatch");
    }

    const size_t count = raw_header.bucket_count;
    const size_t keys_off = sizeof(BucketBlockRawHeader);
    const size_t dense_off = keys_off + count * sizeof(uint64_t);
    const size_t bitmap_offsets_off = dense_off + count * sizeof(uint32_t);
    const size_t small_off = bitmap_offsets_off + (count + 1) * sizeof(uint32_t);
    const size_t large_off = small_off + raw_header.small_bitmap_bytes;
    const size_t expected_size = large_off + static_cast<size_t>(raw_header.large_bitmap_words) * sizeof(uint64_t);
    if (raw_bucket.size() != expected_size) {
        throw std::runtime_error("EXAD compressed bucket raw layout mismatch");
    }
    const auto* keys = reinterpret_cast<const uint64_t*>(raw_bucket.data() + keys_off);
    auto key_it = std::lower_bound(keys, keys + count, key);
    if (key_it == keys + count || *key_it != key) {
        return result;
    }
    const size_t bucket_idx = static_cast<size_t>(key_it - keys);
    const auto* local_dense_offsets = reinterpret_cast<const uint32_t*>(raw_bucket.data() + dense_off);
    const auto* local_bitmap_offsets = reinterpret_cast<const uint32_t*>(raw_bucket.data() + bitmap_offsets_off);

    uint64_t ordinal = 0;
    if (valid_count <= index.header.threshold_bits) {
        const uint32_t bitmap_off = local_bitmap_offsets[bucket_idx];
        const size_t bitmap_bytes = ZMaskFrozen::bytes_for_bits(valid_count);
        if (static_cast<uint64_t>(bitmap_off) + bitmap_bytes > raw_header.small_bitmap_bytes) {
            throw std::runtime_error("EXAD compressed small bitmap offset outside block");
        }
        const uint8_t* bitmap = raw_bucket.data() + small_off + bitmap_off;
        if (!test_bit_raw(bitmap, rank)) {
            return result;
        }
        ordinal = dense_ordinal_small_raw(bitmap, rank);
    } else {
        const uint32_t bitmap_off = local_bitmap_offsets[bucket_idx];
        const size_t bitmap_words = ZMaskFrozen::words_for_bits(valid_count);
        if (static_cast<uint64_t>(bitmap_off) + bitmap_words > raw_header.large_bitmap_words) {
            throw std::runtime_error("EXAD compressed large bitmap offset outside block");
        }
        const auto* bitmap =
            raw_bucket.data() + large_off + static_cast<size_t>(bitmap_off) * sizeof(uint64_t);
        const uint32_t word_idx = rank >> 6;
        const uint32_t bit_idx = rank & 63;
        const uint64_t word =
            load_unaligned<uint64_t>(bitmap + static_cast<size_t>(word_idx) * sizeof(uint64_t));
        if ((word & (uint64_t{1} << bit_idx)) == 0) {
            return result;
        }
        ordinal = dense_ordinal_large_raw(bitmap, rank);
    }

    const uint64_t local_row =
        raw_header.first_dense_offset + local_dense_offsets[bucket_idx] + ordinal;
    const uint64_t value_index = slot.slot_value_base + local_row * slot.row_width + column;
    if (value_index >= index.header.success_value_count) {
        throw std::runtime_error("EXAD compressed value index outside success array");
    }
    ValueBlockDirEntry value_block{};
    if (!read_value_block_dir_for_index(compressed_in, index, value_index, value_block)) {
        throw std::runtime_error("EXAD compressed value block not found");
    }
    if (value_block.value_size != index.header.value_size ||
        value_block.value_size != value_size_for_mode(mode)) {
        throw std::runtime_error("EXAD compressed value size mismatch");
    }
    const auto compressed_value = read_range_from(
        compressed_in,
        value_block.compressed_offset,
        value_block.compressed_size,
        "EXAD compressed value block");
    auto raw_value_block = decompress_block_or_throw(compressed_value.data(), compressed_value.size(),
                                                     value_block.raw_size);
    result.value_block_raw_bytes = value_block.raw_size;
    result.value_block_compressed_bytes = value_block.compressed_size;
    const uint64_t local_value = value_index - value_block.first_value_index;
    const uint64_t byte_offset = local_value * value_block.value_size;
    if (byte_offset + value_block.value_size > raw_value_block.size()) {
        throw std::runtime_error("EXAD compressed value offset outside block");
    }
    switch (mode) {
    case EXAD::DTypeMode::UInt32: {
        const uint32_t v = load_unaligned<uint32_t>(raw_value_block.data() + byte_offset);
        result.raw_value_bits = v;
        break;
    }
    case EXAD::DTypeMode::UInt64: {
        result.raw_value_bits = load_unaligned<uint64_t>(raw_value_block.data() + byte_offset);
        break;
    }
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32: {
        const uint32_t v = load_unaligned<uint32_t>(raw_value_block.data() + byte_offset);
        result.raw_value_bits = v;
        break;
    }
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64: {
        result.raw_value_bits = load_unaligned<uint64_t>(raw_value_block.data() + byte_offset);
        break;
    }
    }
    result.found = true;
    result.value_index = value_index;
    result.local_row = local_row;
    result.numeric_value = numeric_from_raw_bits(result.raw_value_bits, mode);
    return result;
}

ColdLookupResult lookup_exadbook_cold(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    int ad_key,
    uint64_t canonical_board,
    uint32_t column) {
    std::ifstream in(exadbook_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD solved layer: " + exadbook_path);
    }
    const auto index_ptr = cached_solved_file_index(exadbook_path);
    const auto& index = *index_ptr;
    const auto mode = index.mode;
    ExadLutPointReader lut(exadlut_path);
    if (index.header.lut_signature != 0 && lut.signature() != 0 &&
        index.header.lut_signature != lut.signature()) {
        throw std::runtime_error("EXAD LUT signature does not match solved layer");
    }
    if (!physical_metadata_matches_lut(index.header, lut.header())) {
        throw std::runtime_error("EXAD physical pattern metadata does not match solved layer");
    }

    ColdLookupResult result;
    result.success_kind = success_kind_for_mode(mode);
    const int slot_i = slot_index_from_ad_key(ad_key);
    if (slot_i < 0) {
        return result;
    }
    const auto& slot = index.slots[slot_i];
    if (slot.header.bucket_count == 0 || column >= slot.header.row_width) {
        return result;
    }

    uint32_t group = 0;
    uint32_t rank = 0;
    uint32_t semantic_sum = 0;
    uint32_t valid_count = 0;
    if (!lut.suffix28_query(
            static_cast<uint32_t>(canonical_board & EXAD::kSuffixMask),
            group,
            rank,
            semantic_sum,
            valid_count)) {
        return result;
    }
    (void)group;
    if (rank >= valid_count) {
        return result;
    }
    const uint64_t key = EXAD::pack_bucket_key(canonical_board >> EXAD::kSuffixBits, semantic_sum);

    uint64_t left = 0;
    uint64_t right = slot.header.bucket_count;
    EXAD::BucketEntry bucket{};
    bool found_bucket = false;
    while (left < right) {
        const uint64_t mid = left + ((right - left) >> 1U);
        const uint64_t offset = slot.bucket_entries_offset + mid * sizeof(EXAD::BucketEntry);
        bucket = read_one_from<EXAD::BucketEntry>(in, offset, "EXAD solved bucket entry");
        if (bucket.key == key) {
            found_bucket = true;
            break;
        }
        if (bucket.key < key) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    if (!found_bucket) {
        return result;
    }

    uint64_t ordinal = 0;
    if (valid_count <= index.header.threshold_bits) {
        const uint32_t byte_idx = rank >> 3U;
        const uint64_t bytes_to_read = static_cast<uint64_t>(byte_idx) + 1U;
        const size_t total_bytes = ZMaskFrozen::bytes_for_bits(valid_count);
        if (bytes_to_read > total_bytes ||
            static_cast<uint64_t>(bucket.bitmap_offset) + bytes_to_read > slot.header.small_bitmap_bytes) {
            throw std::runtime_error("EXAD solved small bitmap offset outside file");
        }
        auto bitmap = read_range_from(
            in,
            slot.small_bitmap_offset + bucket.bitmap_offset,
            bytes_to_read,
            "EXAD solved small bitmap");
        if (!test_bit_raw(bitmap.data(), rank)) {
            return result;
        }
        ordinal = dense_ordinal_small_raw(bitmap.data(), rank);
    } else {
        const uint32_t word_idx = rank >> 6U;
        const uint64_t words_to_read = static_cast<uint64_t>(word_idx) + 1U;
        const size_t total_words = ZMaskFrozen::words_for_bits(valid_count);
        if (words_to_read > total_words ||
            static_cast<uint64_t>(bucket.bitmap_offset) + words_to_read > slot.header.large_bitmap_words) {
            throw std::runtime_error("EXAD solved large bitmap offset outside file");
        }
        auto bitmap_bytes = read_range_from(
            in,
            slot.large_bitmap_offset + static_cast<uint64_t>(bucket.bitmap_offset) * sizeof(uint64_t),
            words_to_read * sizeof(uint64_t),
            "EXAD solved large bitmap");
        const uint64_t word = load_unaligned<uint64_t>(
            bitmap_bytes.data() + static_cast<size_t>(word_idx) * sizeof(uint64_t));
        if ((word & (uint64_t{1} << (rank & 63U))) == 0) {
            return result;
        }
        ordinal = dense_ordinal_large_raw(bitmap_bytes.data(), rank);
    }

    const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
    const uint64_t value_index =
        slot.header.value_base + local_row * slot.header.row_width + column;
    if (value_index >= index.header.success_value_count) {
        throw std::runtime_error("EXAD solved value index outside success array");
    }
    const uint64_t value_offset =
        index.success_values_offset + value_index * static_cast<uint64_t>(index.header.value_size);
    auto value_bytes = read_range_from(in, value_offset, index.header.value_size, "EXAD solved value");
    switch (mode) {
    case EXAD::DTypeMode::UInt32:
    case EXAD::DTypeMode::Float32:
    case EXAD::DTypeMode::OneMinusFloat32:
        result.raw_value_bits = load_unaligned<uint32_t>(value_bytes.data());
        break;
    case EXAD::DTypeMode::UInt64:
    case EXAD::DTypeMode::Float64:
    case EXAD::DTypeMode::OneMinusFloat64:
        result.raw_value_bits = load_unaligned<uint64_t>(value_bytes.data());
        break;
    }
    result.found = true;
    result.local_row = local_row;
    result.value_index = value_index;
    result.numeric_value = numeric_from_raw_bits(result.raw_value_bits, mode);
    return result;
}

bool sample_exad_cold(
    const std::string& compressed_path,
    const std::string& exadlut_path,
    uint64_t& board) {
    try {
        const auto index_ptr = cached_compressed_index(compressed_path);
        const auto& index = *index_ptr;
        ExadLutPointReader lut(exadlut_path);
        if (index.header.lut_signature != 0 && lut.signature() != 0 &&
            index.header.lut_signature != lut.signature()) {
            return false;
        }
        if (!physical_metadata_matches_lut(index.header, lut.header())) {
            return false;
        }
        if (index.header.live_board_count == 0U || index.header.bucket_block_count == 0U) {
            return false;
        }
        std::ifstream compressed_in(compressed_path, std::ios::binary);
        if (!compressed_in) {
            return false;
        }

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> row_pick(0U, index.header.live_board_count - 1U);
        constexpr uint32_t kSampleAttempts = 128U;
        for (uint32_t attempt = 0; attempt < kSampleAttempts; ++attempt) {
            uint64_t local_row = 0;
            const int slot_i = select_compressed_slot_by_live(index, row_pick(rng), local_row);
            if (slot_i < 0) {
                continue;
            }
            const auto& slot = index.slots[static_cast<size_t>(slot_i)];
            const uint64_t block_begin = slot.bucket_block_begin;
            const uint64_t block_end = block_begin + slot.bucket_block_count;
            if (slot.bucket_count == 0U || block_end > index.bucket_dirs.size()) {
                continue;
            }

            auto first = index.bucket_dirs.begin() + static_cast<std::ptrdiff_t>(block_begin);
            auto last = index.bucket_dirs.begin() + static_cast<std::ptrdiff_t>(block_end);
            auto block_it = std::upper_bound(
                first, last, local_row,
                [](uint64_t row, const BucketBlockDirEntry& dir) {
                    return row < dir.first_dense_offset;
                });
            if (block_it == first) {
                continue;
            }
            --block_it;

            const auto compressed_bucket = read_range_from(
                compressed_in,
                block_it->compressed_offset,
                block_it->compressed_size,
                "EXAD compressed bucket block");
            auto raw_bucket =
                decompress_block_or_throw(compressed_bucket.data(), compressed_bucket.size(), block_it->raw_size);
            if (raw_bucket.size() < sizeof(BucketBlockRawHeader)) {
                throw std::runtime_error("EXAD compressed bucket raw block is truncated");
            }
            const auto raw_header = load_unaligned<BucketBlockRawHeader>(raw_bucket.data());
            if (raw_header.bucket_count != block_it->bucket_count ||
                local_row < raw_header.first_dense_offset) {
                continue;
            }
            const size_t count = raw_header.bucket_count;
            const size_t keys_off = sizeof(BucketBlockRawHeader);
            const size_t dense_off = keys_off + count * sizeof(uint64_t);
            const size_t bitmap_offsets_off = dense_off + count * sizeof(uint32_t);
            const size_t small_off = bitmap_offsets_off + (count + 1U) * sizeof(uint32_t);
            const size_t large_off = small_off + raw_header.small_bitmap_bytes;
            const size_t expected_size =
                large_off + static_cast<size_t>(raw_header.large_bitmap_words) * sizeof(uint64_t);
            if (raw_bucket.size() != expected_size) {
                throw std::runtime_error("EXAD compressed bucket raw layout mismatch");
            }

            const uint64_t block_local_row = local_row - raw_header.first_dense_offset;
            size_t left = 0;
            size_t right = count;
            while (left < right) {
                const size_t mid = left + ((right - left) >> 1U);
                const uint32_t dense = load_unaligned<uint32_t>(
                    raw_bucket.data() + dense_off + mid * sizeof(uint32_t));
                if (static_cast<uint64_t>(dense) <= block_local_row) {
                    left = mid + 1U;
                } else {
                    right = mid;
                }
            }
            if (left == 0U) {
                continue;
            }
            const size_t bucket_idx = left - 1U;
            const uint64_t key = load_unaligned<uint64_t>(
                raw_bucket.data() + keys_off + bucket_idx * sizeof(uint64_t));
            const uint32_t dense = load_unaligned<uint32_t>(
                raw_bucket.data() + dense_off + bucket_idx * sizeof(uint32_t));
            const uint64_t ordinal = block_local_row - static_cast<uint64_t>(dense);
            const uint32_t semantic_sum = EXAD::bucket_key_semantic_sum(key);
            uint32_t valid_count = 0U;
            if (!lut.valid_count_for_semantic_sum(semantic_sum, valid_count)) {
                continue;
            }
            const uint32_t bitmap_off = load_unaligned<uint32_t>(
                raw_bucket.data() + bitmap_offsets_off + bucket_idx * sizeof(uint32_t));
            uint32_t rank = std::numeric_limits<uint32_t>::max();
            if (valid_count <= index.header.threshold_bits) {
                const size_t bitmap_bytes = ZMaskFrozen::bytes_for_bits(valid_count);
                if (static_cast<uint64_t>(bitmap_off) + bitmap_bytes > raw_header.small_bitmap_bytes) {
                    throw std::runtime_error("EXAD compressed small bitmap offset outside block");
                }
                const uint8_t* bitmap = raw_bucket.data() + small_off + bitmap_off;
                if (!rank_for_small_bitmap_ordinal(bitmap, valid_count, ordinal, rank)) {
                    continue;
                }
            } else {
                const size_t bitmap_words = ZMaskFrozen::words_for_bits(valid_count);
                if (static_cast<uint64_t>(bitmap_off) + bitmap_words > raw_header.large_bitmap_words) {
                    throw std::runtime_error("EXAD compressed large bitmap offset outside block");
                }
                const uint8_t* bitmap =
                    raw_bucket.data() + large_off + static_cast<size_t>(bitmap_off) * sizeof(uint64_t);
                if (!rank_for_large_bitmap_ordinal(bitmap, valid_count, ordinal, rank)) {
                    continue;
                }
            }
            if (reconstruct_board_from_key_rank(lut, key, rank, board)) {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool sample_exadbook_cold(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    uint64_t& board) {
    try {
        std::ifstream in(exadbook_path, std::ios::binary);
        if (!in) {
            return false;
        }
        const auto index_ptr = cached_solved_file_index(exadbook_path);
        const auto& index = *index_ptr;
        ExadLutPointReader lut(exadlut_path);
        if (index.header.lut_signature != 0 && lut.signature() != 0 &&
            index.header.lut_signature != lut.signature()) {
            return false;
        }
        if (!physical_metadata_matches_lut(index.header, lut.header())) {
            return false;
        }
        if (index.header.live_board_count == 0U) {
            return false;
        }

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> row_pick(0U, index.header.live_board_count - 1U);
        constexpr uint32_t kSampleAttempts = 128U;
        for (uint32_t attempt = 0; attempt < kSampleAttempts; ++attempt) {
            uint64_t local_row = 0;
            const int slot_i = select_solved_slot_by_live(index, row_pick(rng), local_row);
            if (slot_i < 0) {
                continue;
            }
            const auto& slot = index.slots[static_cast<size_t>(slot_i)];
            EXAD::BucketEntry bucket{};
            if (!read_solved_bucket_for_local_row(in, slot, local_row, bucket)) {
                continue;
            }
            if (sample_from_solved_bucket(in, index, slot, lut, local_row, bucket, board)) {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}

} // namespace EXADCompressedResult
