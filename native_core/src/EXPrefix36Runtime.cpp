#include "EXPrefix36Runtime.h"

#include "EXFrozenLayer.h"
#include "EXIoConfig.h"
#include "EXPrefix40Layer.h"
#include "CompressionBridge.h"
#include "Formation.h"
#include "VBoardMover.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Shared prefix36/suffix28 implementation used by the EX production runtime.
#include "EXPrefix36Core.inl"

namespace EXPrefix36Runtime {

namespace {

namespace fs = std::filesystem;

constexpr char kLayerMagic[8] = {'E', 'X', 'P', '3', '6', 'B', 'K', '\0'};
constexpr char kLutMagic[8] = {'E', 'X', 'P', '3', '6', 'L', 'T', '\0'};
constexpr const char *kGeneratedLayerFileExtension = ".exgen";
constexpr const char *kLutFileExtension = ".zlut";
constexpr uint32_t kLayerVersion = 5U;
constexpr uint32_t kLutVersion = 2U;
constexpr double kDefaultReserveFactor = 3.0;
constexpr double kEarlyLayerReserveFactor = 8.0;
constexpr int kEarlyLayerReserveFactorSteps = 10;
constexpr double kLearnedReserveMinFactor = 1.18;
constexpr double kLearnedReserveQuantileGuard = 1.15;
constexpr double kLearnedReserveLastGuard = 1.20;
constexpr double kLearnedReserveRetryGuard = 1.25;
constexpr size_t kLearnedReserveHistoryWindow = 32;
constexpr double kFixedScale = 4000000000.0;
constexpr double kUInt64Scale = 1600000000000000000.0;
constexpr int kOptimalBranchOnlyStartStep = 21;

enum class DTypeMode : uint32_t {
    UInt32 = 0,
    UInt64 = 1,
    Float32 = 2,
    Float64 = 3,
    OneMinusFloat32 = 4,
    OneMinusFloat64 = 5,
};

struct LayerFileHeader {
    char magic[8];
    uint32_t version = kLayerVersion;
    uint32_t success_kind = static_cast<uint32_t>(SuccessRateKind::UInt32);
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t dtype_mode = static_cast<uint32_t>(DTypeMode::UInt32);
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
    uint64_t value_size = sizeof(uint32_t);
    uint64_t reserved1 = 0;
};

struct LutFileHeader {
    char magic[8];
    uint32_t version = kLutVersion;
    uint32_t reserved32 = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved16a = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    int8_t max_counts[16]{};
    uint32_t required_suffix24 = 0;
    uint8_t table_for_high[16]{};
    uint8_t packed_table0 = DenseLow24RankLut::kInvalidTable;
    uint8_t packed_table1 = DenseLow24RankLut::kInvalidTable;
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

struct LutFileLayout {
    uint64_t valid_suffix_masks_offset = 0;
    uint64_t rank_tables_offset = 0;
    uint64_t packed_rank_pair_offset = 0;
    uint64_t packed_meta_offset = 0;
    uint64_t size_table_offset = 0;
    uint64_t offset_table_offset = 0;
    uint64_t unrank_array_offset = 0;
    uint64_t high_base_offset = 0;
};

LutFileLayout lut_file_layout(const LutFileHeader &header) {
    LutFileLayout layout;
    layout.valid_suffix_masks_offset = sizeof(LutFileHeader);
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

struct LutBundle {
    ZMaskFrozen::TileLimitConfig config;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    ZMaskFrozen::ZMaskLuts row_luts;
    DenseLow24RankLut dense_lut;
};

bool physical_metadata_matches(const LayerFileHeader &layer, const LutFileHeader &lut) {
    return layer.physical_transform == lut.physical_transform &&
        layer.inverse_physical_transform == lut.inverse_physical_transform &&
        layer.logical_pattern_signature == lut.logical_pattern_signature &&
        layer.physical_pattern_signature == lut.physical_pattern_signature;
}

bool physical_metadata_matches(const LayerFileHeader &layer, const LutBundle &lut) {
    return layer.physical_transform == lut.physical_transform &&
        layer.inverse_physical_transform == lut.inverse_physical_transform &&
        layer.logical_pattern_signature == lut.logical_pattern_signature &&
        layer.physical_pattern_signature == lut.physical_pattern_signature;
}

std::string lut_file_path(const std::string &pathname) {
    return pathname + kLutFileExtension;
}

double now_seconds() {
#if defined(_OPENMP)
    return omp_get_wtime();
#else
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    return std::chrono::duration<double>(clock::now() - epoch).count();
#endif
}

template <typename T>
T read_one_at(std::ifstream &in, uint64_t offset, const std::string &path) {
    T value{};
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read EX prefix36 file at offset " +
            std::to_string(offset) + ": " + path);
    }
    return value;
}

std::vector<uint8_t> read_bytes_at(
    std::ifstream &in,
    uint64_t offset,
    uint64_t byte_count,
    const std::string &path
) {
    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (byte_count != 0U) {
        in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(byte_count));
    }
    if (!in) {
        throw std::runtime_error("failed to read EX prefix36 file range at offset " +
            std::to_string(offset) + ": " + path);
    }
    return bytes;
}

int thread_count_from_options(const RunOptions &options) {
#if defined(_OPENMP)
    return options.num_threads > 0 ? options.num_threads : std::max(1, omp_get_max_threads());
#else
    return options.num_threads > 0 ? options.num_threads : 1;
#endif
}

double throughput(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1e6 : 0.0;
}

DTypeMode dtype_mode_from_name(const std::string &name) {
    if (name == "uint64") {
        return DTypeMode::UInt64;
    }
    if (name == "float32") {
        return DTypeMode::Float32;
    }
    if (name == "float64") {
        return DTypeMode::Float64;
    }
    if (name == "1-float32") {
        return DTypeMode::OneMinusFloat32;
    }
    if (name == "1-float64") {
        return DTypeMode::OneMinusFloat64;
    }
    return DTypeMode::UInt32;
}

DTypeMode dtype_mode_from_header(const LayerFileHeader &header) {
    switch (static_cast<DTypeMode>(header.dtype_mode)) {
        case DTypeMode::UInt32:
        case DTypeMode::UInt64:
        case DTypeMode::Float32:
        case DTypeMode::Float64:
        case DTypeMode::OneMinusFloat32:
        case DTypeMode::OneMinusFloat64:
            return static_cast<DTypeMode>(header.dtype_mode);
    }
    throw std::runtime_error("unsupported EX prefix36 dtype mode");
}

SuccessRateKind storage_kind_for_dtype_mode(DTypeMode mode) {
    switch (mode) {
        case DTypeMode::UInt64:
            return SuccessRateKind::UInt64;
        case DTypeMode::Float32:
        case DTypeMode::OneMinusFloat32:
            return SuccessRateKind::Float32;
        case DTypeMode::Float64:
        case DTypeMode::OneMinusFloat64:
            return SuccessRateKind::Float64;
        case DTypeMode::UInt32:
        default:
            return SuccessRateKind::UInt32;
    }
}

uint64_t value_size_for_dtype_mode(DTypeMode mode) {
    switch (mode) {
        case DTypeMode::UInt64:
        case DTypeMode::Float64:
        case DTypeMode::OneMinusFloat64:
            return sizeof(uint64_t);
        case DTypeMode::Float32:
        case DTypeMode::OneMinusFloat32:
        case DTypeMode::UInt32:
        default:
            return sizeof(uint32_t);
    }
}

uint32_t fixed_from_unit(double value) {
    if (value <= 0.0) {
        return 0U;
    }
    if (value >= 1.0) {
        return max_scale_value<uint32_t>();
    }
    return static_cast<uint32_t>(value * kFixedScale);
}

uint64_t raw_bits_from_fixed(uint32_t fixed, DTypeMode mode) {
    const double unit = static_cast<double>(fixed) / kFixedScale;
    switch (mode) {
        case DTypeMode::UInt64:
            return static_cast<uint64_t>(unit * kUInt64Scale);
        case DTypeMode::Float32: {
            const float value = static_cast<float>(unit);
            uint32_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case DTypeMode::Float64: {
            const double value = unit;
            uint64_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case DTypeMode::OneMinusFloat32: {
            const float value = static_cast<float>(unit - 1.0);
            uint32_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case DTypeMode::OneMinusFloat64: {
            const double value = unit - 1.0;
            uint64_t bits = 0U;
            std::memcpy(&bits, &value, sizeof(value));
            return bits;
        }
        case DTypeMode::UInt32:
        default:
            return fixed;
    }
}

double numeric_from_fixed(uint32_t fixed, DTypeMode mode) {
    const double unit = static_cast<double>(fixed) / kFixedScale;
    switch (mode) {
        case DTypeMode::UInt64:
        case DTypeMode::Float32:
        case DTypeMode::Float64:
        case DTypeMode::UInt32:
            return unit;
        case DTypeMode::OneMinusFloat32:
        case DTypeMode::OneMinusFloat64:
            return unit - 1.0;
    }
    return unit;
}

uint32_t fixed_from_raw_bits(uint64_t raw, DTypeMode mode) {
    switch (mode) {
        case DTypeMode::UInt64:
            return fixed_from_unit(static_cast<double>(raw) / kUInt64Scale);
        case DTypeMode::Float32: {
            const uint32_t bits = static_cast<uint32_t>(raw);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return fixed_from_unit(static_cast<double>(value));
        }
        case DTypeMode::Float64: {
            double value = 0.0;
            std::memcpy(&value, &raw, sizeof(value));
            return fixed_from_unit(value);
        }
        case DTypeMode::OneMinusFloat32: {
            const uint32_t bits = static_cast<uint32_t>(raw);
            float value = -1.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return fixed_from_unit(static_cast<double>(value) + 1.0);
        }
        case DTypeMode::OneMinusFloat64: {
            double value = -1.0;
            std::memcpy(&value, &raw, sizeof(value));
            return fixed_from_unit(value + 1.0);
        }
        case DTypeMode::UInt32:
        default:
            return static_cast<uint32_t>(raw);
    }
}

uint64_t layer_metadata_bytes(const Prefix36Layer &layer) {
    return static_cast<uint64_t>(layer.bucket_keys.size()) * sizeof(uint64_t)
         + static_cast<uint64_t>(layer.bitmap_offsets.size()) * sizeof(uint32_t)
         + static_cast<uint64_t>(layer.success_offsets.size()) * sizeof(uint32_t)
         + static_cast<uint64_t>(layer.large_rank_offsets.size()) * sizeof(uint32_t)
         + static_cast<uint64_t>(layer.bucket_entries.size()) * sizeof(BucketEntry)
         + static_cast<uint64_t>(layer.small_bitmap_bytes.size())
         + static_cast<uint64_t>(layer.large_bitmap_words.size()) * sizeof(uint64_t)
         + static_cast<uint64_t>(layer.large_rank_bases.size()) * sizeof(uint16_t);
}

uint64_t layer_file_bytes(const Prefix36Layer &layer, DTypeMode mode) {
    return sizeof(LayerFileHeader)
         + static_cast<uint64_t>(layer.bucket_keys.size()) * sizeof(uint64_t)
         + static_cast<uint64_t>(layer.bitmap_offsets.size()) * sizeof(uint32_t)
         + static_cast<uint64_t>(layer.success_offsets.size()) * sizeof(uint32_t)
         + static_cast<uint64_t>(layer.small_bitmap_bytes.size())
         + static_cast<uint64_t>(layer.large_bitmap_words.size()) * sizeof(uint64_t)
         + static_cast<uint64_t>(layer.success_values.size()) * value_size_for_dtype_mode(mode);
}

uint64_t layer_file_bytes(const LayerFileHeader &header) {
    return sizeof(LayerFileHeader)
        + header.bucket_count * sizeof(uint64_t)
        + header.bucket_count * sizeof(uint32_t)
        + header.bucket_count * sizeof(uint32_t)
        + header.small_bitmap_bytes
        + header.large_bitmap_words * sizeof(uint64_t)
        + header.success_value_count * header.value_size;
}

struct LayerFileLayout {
    uint64_t bucket_keys_offset = 0;
    uint64_t bitmap_offsets_offset = 0;
    uint64_t success_offsets_offset = 0;
    uint64_t small_bitmap_offset = 0;
    uint64_t large_bitmap_offset = 0;
    uint64_t success_values_offset = 0;
};

LayerFileLayout layer_file_layout(const LayerFileHeader &header) {
    LayerFileLayout layout;
    layout.bucket_keys_offset = sizeof(LayerFileHeader);
    layout.bitmap_offsets_offset = layout.bucket_keys_offset + header.bucket_count * sizeof(uint64_t);
    layout.success_offsets_offset = layout.bitmap_offsets_offset + header.bucket_count * sizeof(uint32_t);
    layout.small_bitmap_offset = layout.success_offsets_offset + header.bucket_count * sizeof(uint32_t);
    layout.large_bitmap_offset = layout.small_bitmap_offset + header.small_bitmap_bytes;
    layout.success_values_offset = layout.large_bitmap_offset + header.large_bitmap_words * sizeof(uint64_t);
    return layout;
}

void validate_layer_header(const LayerFileHeader &header, const std::string &path) {
    if (std::memcmp(header.magic, kLayerMagic, sizeof(kLayerMagic)) != 0) {
        throw std::runtime_error("invalid EX prefix36 layer file magic, rebuild required: " + path);
    }
    if (header.version != kLayerVersion) {
        throw std::runtime_error("unsupported EX prefix36 layer version, rebuild required: " + path);
    }
    const DTypeMode mode = dtype_mode_from_header(header);
    if (header.success_kind != static_cast<uint32_t>(storage_kind_for_dtype_mode(mode)) ||
        header.value_size != value_size_for_dtype_mode(mode)) {
        throw std::runtime_error("invalid EX prefix36 dtype header, rebuild required: " + path);
    }
}

template <typename T>
void write_vector(std::ofstream &out, const std::vector<T> &values) {
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
    }
}

template <typename Writer, typename T>
void append_vector(Writer &out, const std::vector<T> &values) {
    if (!values.empty()) {
        out.append(values.data(), values.size() * sizeof(T));
    }
}

template <typename Writer>
void append_typed_success_values(
    Writer &out,
    const std::vector<uint32_t> &values,
    DTypeMode mode
) {
    if (values.empty()) {
        return;
    }
    if (mode == DTypeMode::UInt32) {
        out.append(values.data(), values.size() * sizeof(uint32_t));
        return;
    }

    constexpr size_t kConvertChunkValues = 1U << 20;
    if (value_size_for_dtype_mode(mode) == sizeof(uint32_t)) {
        std::vector<uint32_t> buffer(std::min(kConvertChunkValues, values.size()));
        for (size_t base = 0; base < values.size(); base += buffer.size()) {
            const size_t count = std::min(buffer.size(), values.size() - base);
            for (size_t i = 0; i < count; ++i) {
                buffer[i] = static_cast<uint32_t>(raw_bits_from_fixed(values[base + i], mode));
            }
            out.append(buffer.data(), count * sizeof(uint32_t));
        }
        return;
    }

    std::vector<uint64_t> buffer(std::min(kConvertChunkValues, values.size()));
    for (size_t base = 0; base < values.size(); base += buffer.size()) {
        const size_t count = std::min(buffer.size(), values.size() - base);
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = raw_bits_from_fixed(values[base + i], mode);
        }
        out.append(buffer.data(), count * sizeof(uint64_t));
    }
}

template <typename T>
void read_vector(std::ifstream &in, std::vector<T> &values, uint64_t count) {
    values.resize(static_cast<size_t>(count));
    if (!values.empty()) {
        in.read(
            reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
    }
}

template <typename Reader, typename T>
void read_vector(Reader &in, std::vector<T> &values, uint64_t count) {
    values.resize(static_cast<size_t>(count));
    if (!values.empty()) {
        in.read(values.data(), values.size() * sizeof(T));
    }
}

template <typename Reader>
void read_typed_success_values(
    Reader &in,
    std::vector<uint32_t> &values,
    uint64_t count,
    DTypeMode mode
) {
    values.resize(static_cast<size_t>(count));
    if (count == 0U) {
        return;
    }
    if (mode == DTypeMode::UInt32) {
        in.read(values.data(), values.size() * sizeof(uint32_t));
        return;
    }

    constexpr size_t kConvertChunkValues = 1U << 20;
    if (value_size_for_dtype_mode(mode) == sizeof(uint32_t)) {
        std::vector<uint32_t> buffer(std::min<uint64_t>(kConvertChunkValues, count));
        for (uint64_t base = 0; base < count; base += buffer.size()) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(buffer.size(), count - base));
            in.read(buffer.data(), chunk * sizeof(uint32_t));
            for (size_t i = 0; i < chunk; ++i) {
                values[static_cast<size_t>(base) + i] = fixed_from_raw_bits(buffer[i], mode);
            }
        }
    } else {
        std::vector<uint64_t> buffer(std::min<uint64_t>(kConvertChunkValues, count));
        for (uint64_t base = 0; base < count; base += buffer.size()) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(buffer.size(), count - base));
            in.read(buffer.data(), chunk * sizeof(uint64_t));
            for (size_t i = 0; i < chunk; ++i) {
                values[static_cast<size_t>(base) + i] = fixed_from_raw_bits(buffer[i], mode);
            }
        }
    }
}

uint32_t valid_suffix_count_for_key(
    const std::vector<uint32_t> &size_table,
    uint64_t key,
    const std::string &path
) {
    const uint32_t group = sum_index(key_remaining_sum(key));
    if (group >= size_table.size()) {
        throw std::runtime_error("EX prefix36 layer key has out-of-range remaining sum: " + path);
    }
    return size_table[group];
}

void rebuild_large_rank_metadata(
    Prefix36Layer &layer,
    const std::vector<uint32_t> &size_table,
    int requested_threads,
    const std::string &path
) {
    const uint64_t bucket_count = layer.bucket_keys.size();
    layer.large_rank_offsets.assign(static_cast<size_t>(bucket_count), 0U);
    if (bucket_count == 0U) {
        layer.large_rank_bases.clear();
        return;
    }

    int workers = 1;
#if defined(_OPENMP)
    workers = std::max(1, requested_threads);
#else
    (void)requested_threads;
#endif
    std::vector<uint64_t> rank_counts(static_cast<size_t>(workers), 0U);

    if (workers == 1) {
        for (uint64_t i = 0; i < bucket_count; ++i) {
            const uint32_t valid_count =
                valid_suffix_count_for_key(size_table, layer.bucket_keys[static_cast<size_t>(i)], path);
            if (valid_count > layer.threshold_bits) {
                rank_counts[0] += large_rank_bases_for_words(static_cast<uint32_t>(words_for_bits(valid_count)));
            }
        }
        const uint64_t total_rank_bases = rank_counts[0];
        if (total_rank_bases > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EX prefix36 large rank metadata exceeds uint32 offsets: " + path);
        }
        layer.large_rank_bases.assign(static_cast<size_t>(total_rank_bases), 0U);
        uint64_t rank_cursor = 0U;
        for (uint64_t i = 0; i < bucket_count; ++i) {
            const size_t index = static_cast<size_t>(i);
            const uint32_t valid_count = valid_suffix_count_for_key(size_table, layer.bucket_keys[index], path);
            if (valid_count <= layer.threshold_bits) {
                continue;
            }
            const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
            const uint32_t bitmap_offset = layer.bitmap_offsets[index];
            if (static_cast<uint64_t>(bitmap_offset) + words > layer.large_bitmap_words.size()) {
                throw std::runtime_error("EX prefix36 large bitmap offset out of range while rebuilding rank metadata: " + path);
            }
            layer.large_rank_offsets[index] = static_cast<uint32_t>(rank_cursor);
            fill_large_rank_bases(layer, bitmap_offset, static_cast<uint32_t>(rank_cursor), words);
            rank_cursor += large_rank_bases_for_words(words);
        }
        return;
    }

#if defined(_OPENMP)
    std::atomic<bool> invalid_key{false};
#pragma omp parallel num_threads(workers)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
#pragma omp single
        {
            workers = nthreads;
        }
        const uint64_t begin = bucket_count * static_cast<uint64_t>(tid) / static_cast<uint64_t>(nthreads);
        const uint64_t end = bucket_count * static_cast<uint64_t>(tid + 1) / static_cast<uint64_t>(nthreads);
        uint64_t local_count = 0U;
        for (uint64_t i = begin; i < end; ++i) {
            const uint32_t group = sum_index(key_remaining_sum(layer.bucket_keys[static_cast<size_t>(i)]));
            if (group >= size_table.size()) {
                invalid_key.store(true, std::memory_order_relaxed);
                continue;
            }
            const uint32_t valid_count = size_table[group];
            if (valid_count > layer.threshold_bits) {
                local_count += large_rank_bases_for_words(
                    static_cast<uint32_t>(words_for_bits(valid_count))
                );
            }
        }
        rank_counts[static_cast<size_t>(tid)] = local_count;
    }
    rank_counts.resize(static_cast<size_t>(workers));
    if (invalid_key.load(std::memory_order_relaxed)) {
        throw std::runtime_error("EX prefix36 layer key has out-of-range remaining sum: " + path);
    }
#else
    for (uint64_t i = 0; i < bucket_count; ++i) {
        const uint32_t valid_count =
            valid_suffix_count_for_key(size_table, layer.bucket_keys[static_cast<size_t>(i)], path);
        if (valid_count > layer.threshold_bits) {
            rank_counts[0] += large_rank_bases_for_words(static_cast<uint32_t>(words_for_bits(valid_count)));
        }
    }
#endif

    std::vector<uint64_t> rank_bases(static_cast<size_t>(workers) + 1U, 0U);
    for (int i = 0; i < workers; ++i) {
        rank_bases[static_cast<size_t>(i + 1)] =
            rank_bases[static_cast<size_t>(i)] + rank_counts[static_cast<size_t>(i)];
    }
    const uint64_t total_rank_bases = rank_bases.back();
    if (total_rank_bases > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("EX prefix36 large rank metadata exceeds uint32 offsets: " + path);
    }
    layer.large_rank_bases.assign(static_cast<size_t>(total_rank_bases), 0U);

#if defined(_OPENMP)
    std::atomic<bool> invalid_bitmap{false};
#pragma omp parallel num_threads(workers)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const uint64_t begin = bucket_count * static_cast<uint64_t>(tid) / static_cast<uint64_t>(nthreads);
        const uint64_t end = bucket_count * static_cast<uint64_t>(tid + 1) / static_cast<uint64_t>(nthreads);
        uint64_t rank_cursor = rank_bases[static_cast<size_t>(tid)];
        for (uint64_t i = begin; i < end; ++i) {
            const size_t index = static_cast<size_t>(i);
            const uint32_t valid_count = size_table[sum_index(key_remaining_sum(layer.bucket_keys[index]))];
            if (valid_count <= layer.threshold_bits) {
                continue;
            }
            const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
            const uint32_t bitmap_offset = layer.bitmap_offsets[index];
            if (static_cast<uint64_t>(bitmap_offset) + words > layer.large_bitmap_words.size()) {
                invalid_bitmap.store(true, std::memory_order_relaxed);
                continue;
            }
            const uint32_t rank_offset = static_cast<uint32_t>(rank_cursor);
            layer.large_rank_offsets[index] = rank_offset;
            fill_large_rank_bases(layer, bitmap_offset, rank_offset, words);
            rank_cursor += large_rank_bases_for_words(words);
        }
    }
    if (invalid_bitmap.load(std::memory_order_relaxed)) {
        throw std::runtime_error("EX prefix36 large bitmap offset out of range while rebuilding rank metadata: " + path);
    }
#else
    uint64_t rank_cursor = 0U;
    for (uint64_t i = 0; i < bucket_count; ++i) {
        const size_t index = static_cast<size_t>(i);
        const uint32_t valid_count = valid_suffix_count_for_key(size_table, layer.bucket_keys[index], path);
        if (valid_count > layer.threshold_bits) {
            fill_large_rank_bases(
                layer,
                layer.bitmap_offsets[index],
                layer.large_rank_offsets[index],
                static_cast<uint32_t>(words_for_bits(valid_count))
            );
        }
    }
#endif
}

ZMaskFrozen::ZMaskLuts make_row16_only_luts() {
    ZMaskFrozen::ZMaskLuts luts;
    luts.row16_sum.assign(1U << 16U, 0U);
    for (uint32_t row = 0; row < luts.row16_sum.size(); ++row) {
        uint32_t sum = 0U;
        for (uint32_t cell = 0; cell < 4U; ++cell) {
            sum += tile_value((row >> (cell * 4U)) & 0xFU);
        }
        luts.row16_sum[row] = sum;
    }
    return luts;
}

void write_lut_file(const std::string &path, const LutBundle &bundle) {
    LutFileHeader header{};
    std::memcpy(header.magic, kLutMagic, sizeof(kLutMagic));
    header.physical_transform = bundle.physical_transform;
    header.inverse_physical_transform = bundle.inverse_physical_transform;
    header.logical_pattern_signature = bundle.logical_pattern_signature;
    header.physical_pattern_signature = bundle.physical_pattern_signature;
    std::copy(bundle.config.max_counts.begin(), bundle.config.max_counts.end(), header.max_counts);
    header.required_suffix24 = bundle.config.required_suffix24 & 0xFFFFFFU;
    for (size_t i = 0; i < std::size(header.table_for_high); ++i) {
        header.table_for_high[i] = bundle.dense_lut.table_for_high[i];
    }
    header.packed_table0 = bundle.dense_lut.packed_table0;
    header.packed_table1 = bundle.dense_lut.packed_table1;
    header.rank_table_variant_count = bundle.dense_lut.rank_table_variant_count;
    header.valid_suffix_mask_count = bundle.config.valid_suffix_masks.size();
    header.rank_table_count = bundle.dense_lut.rank_tables.size();
    uint64_t rank_table_values = 0U;
    for (const auto &table : bundle.dense_lut.rank_tables) {
        rank_table_values += table.size();
    }
    header.rank_table_values = rank_table_values;
    header.packed_rank_pair_values = bundle.dense_lut.packed_rank_pair_table.size();
    header.packed_meta_values = bundle.dense_lut.packed_meta_table.size();
    header.size_table_values = bundle.dense_lut.size_table.size();
    header.offset_table_values = bundle.dense_lut.offset_table.size();
    header.unrank_array_values = bundle.dense_lut.unrank_array.size();
    header.high_base_values = bundle.dense_lut.high_base.size();
    header.valid_suffix_count = bundle.dense_lut.valid_suffix_count;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write EX prefix36 LUT: " + path);
    }
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    write_vector(out, bundle.config.valid_suffix_masks);
    for (const auto &table : bundle.dense_lut.rank_tables) {
        write_vector(out, table);
    }
    write_vector(out, bundle.dense_lut.packed_rank_pair_table);
    write_vector(out, bundle.dense_lut.packed_meta_table);
    write_vector(out, bundle.dense_lut.size_table);
    write_vector(out, bundle.dense_lut.offset_table);
    write_vector(out, bundle.dense_lut.unrank_array);
    write_vector(out, bundle.dense_lut.high_base);
    if (!out) {
        throw std::runtime_error("failed while writing EX prefix36 LUT: " + path);
    }
}

LutBundle read_lut_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read EX prefix36 LUT: " + path);
    }
    LutFileHeader header{};
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, kLutMagic, sizeof(kLutMagic)) != 0) {
        throw std::runtime_error("invalid EX prefix36 LUT file magic, rebuild required: " + path);
    }
    if (header.version != kLutVersion) {
        throw std::runtime_error("unsupported EX prefix36 LUT version, rebuild required: " + path);
    }
    LutBundle bundle;
    bundle.physical_transform = header.physical_transform;
    bundle.inverse_physical_transform = header.inverse_physical_transform;
    bundle.logical_pattern_signature = header.logical_pattern_signature;
    bundle.physical_pattern_signature = header.physical_pattern_signature;
    std::copy(std::begin(header.max_counts), std::end(header.max_counts), bundle.config.max_counts.begin());
    bundle.config.required_suffix24 = header.required_suffix24 & 0xFFFFFFU;
    read_vector(in, bundle.config.valid_suffix_masks, header.valid_suffix_mask_count);
    bundle.dense_lut.table_for_high.fill(DenseLow24RankLut::kInvalidTable);
    for (size_t i = 0; i < bundle.dense_lut.table_for_high.size(); ++i) {
        bundle.dense_lut.table_for_high[i] = header.table_for_high[i];
    }
    bundle.dense_lut.packed_table0 = header.packed_table0;
    bundle.dense_lut.packed_table1 = header.packed_table1;
    bundle.dense_lut.rank_table_variant_count = header.rank_table_variant_count;
    bundle.dense_lut.valid_suffix_count = header.valid_suffix_count;

    bundle.dense_lut.rank_tables.resize(static_cast<size_t>(header.rank_table_count));
    for (auto &table : bundle.dense_lut.rank_tables) {
        read_vector(in, table, ZMaskFrozen::kSuffixStateCount);
    }
    if (header.rank_table_values != header.rank_table_count * ZMaskFrozen::kSuffixStateCount) {
        throw std::runtime_error("invalid EX prefix36 LUT rank table shape: " + path);
    }
    read_vector(in, bundle.dense_lut.packed_rank_pair_table, header.packed_rank_pair_values);
    read_vector(in, bundle.dense_lut.packed_meta_table, header.packed_meta_values);
    read_vector(in, bundle.dense_lut.size_table, header.size_table_values);
    read_vector(in, bundle.dense_lut.offset_table, header.offset_table_values);
    read_vector(in, bundle.dense_lut.unrank_array, header.unrank_array_values);
    read_vector(in, bundle.dense_lut.high_base, header.high_base_values);
    if (!in) {
        throw std::runtime_error("truncated EX prefix36 LUT: " + path);
    }
    bundle.dense_lut.rank_table_bytes = header.packed_rank_pair_values * sizeof(uint32_t);
    for (const auto &table : bundle.dense_lut.rank_tables) {
        bundle.dense_lut.rank_table_bytes += static_cast<uint64_t>(table.size()) * sizeof(uint16_t);
    }
    bundle.dense_lut.packed_meta_bytes = header.packed_meta_values * sizeof(uint64_t);
    bundle.dense_lut.unrank_bytes = header.unrank_array_values * sizeof(uint32_t);
    bundle.row_luts = make_row16_only_luts();
    return bundle;
}

LutBundle make_lut_bundle(const ZMaskFrozen::TileLimitConfig &config) {
    LutBundle bundle;
    bundle.config = config;
    bundle.row_luts = make_row16_only_luts();
    bundle.dense_lut = build_dense_low24_rank_lut(config);
    return bundle;
}

std::string archive_entry_name_for_path(const std::string &path) {
    std::string name = fs::path(path).stem().string();
    if (name.empty()) {
        name = "data";
    }
    return name + ".bin";
}

LayerFileHeader make_layer_header(
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode
) {
    LayerFileHeader header{};
    std::memcpy(header.magic, kLayerMagic, sizeof(kLayerMagic));
    header.success_kind = static_cast<uint32_t>(storage_kind_for_dtype_mode(mode));
    header.dtype_mode = static_cast<uint32_t>(mode);
    header.layer_sum = layer.layer_sum;
    header.threshold_bits = layer.threshold_bits;
    header.physical_transform = spec.physical_transform;
    header.inverse_physical_transform = spec.inverse_physical_transform;
    header.logical_pattern_signature = spec.logical_pattern_signature;
    header.physical_pattern_signature = spec.physical_pattern_signature;
    header.bucket_count = layer.bucket_keys.size();
    header.small_bitmap_bytes = layer.small_bitmap_bytes.size();
    header.large_bitmap_words = layer.large_bitmap_words.size();
    header.success_value_count = layer.success_values.size();
    header.live_board_count = layer.live_board_count;
    header.value_size = value_size_for_dtype_mode(mode);
    return header;
}

template <typename Writer>
void append_layer_payload(
    Writer &out,
    const LayerFileHeader &header,
    const Prefix36Layer &layer,
    DTypeMode mode
) {
    out.append(&header, sizeof(header));
    if (!layer.bucket_keys.empty()) {
        append_vector(out, layer.bucket_keys);
        append_vector(out, layer.bitmap_offsets);
        append_vector(out, layer.success_offsets);
    }
    if (!layer.small_bitmap_bytes.empty()) {
        out.append(layer.small_bitmap_bytes.data(), layer.small_bitmap_bytes.size());
    }
    if (!layer.large_bitmap_words.empty()) {
        append_vector(out, layer.large_bitmap_words);
    }
    if (!layer.success_values.empty()) {
        append_typed_success_values(out, layer.success_values, mode);
    }
}

void write_layer_file(
    const std::string &path,
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode,
    FileIOUtils::DirectIoConfig io_config = {}
) {
    const LayerFileHeader header = make_layer_header(layer, spec, mode);
    FileIOUtils::DirectAppendWriter out(path, layer_file_bytes(layer, mode), io_config);
    append_layer_payload(out, header, layer, mode);
    out.close();
}

void write_layer_archive_file(
    const std::string &archive_path,
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode
) {
    const LayerFileHeader header = make_layer_header(layer, spec, mode);
    SevenZipArchiveWriter out(archive_path, archive_entry_name_for_path(archive_path), 1);
    append_layer_payload(out, header, layer, mode);
    out.close();
}

template <typename Reader>
Prefix36Layer read_layer_file_from_reader(
    Reader &in,
    const std::string &path,
    LayerFileHeader *out_header,
    const std::vector<uint32_t> *size_table,
    int rebuild_threads
) {
    LayerFileHeader header{};
    in.read(&header, sizeof(header));
    validate_layer_header(header, path);
    if (out_header != nullptr) {
        *out_header = header;
    }
    const DTypeMode mode = dtype_mode_from_header(header);
    Prefix36Layer layer;
    layer.layer_sum = header.layer_sum;
    layer.threshold_bits = header.threshold_bits;
    layer.live_board_count = header.live_board_count;
    layer.bucket_keys.resize(static_cast<size_t>(header.bucket_count));
    layer.bitmap_offsets.resize(static_cast<size_t>(header.bucket_count));
    layer.success_offsets.resize(static_cast<size_t>(header.bucket_count));
    layer.small_bitmap_bytes.resize(static_cast<size_t>(header.small_bitmap_bytes));
    layer.large_bitmap_words.resize(static_cast<size_t>(header.large_bitmap_words));
    if (!layer.bucket_keys.empty()) {
        read_vector(in, layer.bucket_keys, header.bucket_count);
        read_vector(in, layer.bitmap_offsets, header.bucket_count);
        read_vector(in, layer.success_offsets, header.bucket_count);
    }
    if (!layer.small_bitmap_bytes.empty()) {
        in.read(layer.small_bitmap_bytes.data(), layer.small_bitmap_bytes.size());
    }
    if (!layer.large_bitmap_words.empty()) {
        read_vector(in, layer.large_bitmap_words, header.large_bitmap_words);
    }
    if (header.success_value_count != 0U) {
        read_typed_success_values(in, layer.success_values, header.success_value_count, mode);
    }
    if (size_table == nullptr) {
        throw std::runtime_error("EX prefix36 layer requires zlut size table to rebuild rank metadata: " + path);
    }
    rebuild_large_rank_metadata(layer, *size_table, rebuild_threads, path);
    return layer;
}

bool has_archive_suffix(const std::string &path) {
    return path.size() >= 3U && path.compare(path.size() - 3U, 3U, ".7z") == 0;
}

Prefix36Layer read_layer_file(
    const std::string &path,
    LayerFileHeader *out_header = nullptr,
    FileIOUtils::DirectIoConfig io_config = {},
    const std::vector<uint32_t> *size_table = nullptr,
    int rebuild_threads = 1
) {
    std::string actual_path = path;
    if (!fs::exists(actual_path) && !has_archive_suffix(actual_path) && fs::exists(actual_path + ".7z")) {
        actual_path += ".7z";
    }
    if (has_archive_suffix(actual_path)) {
        SevenZipSequentialReader in(actual_path);
        Prefix36Layer layer = read_layer_file_from_reader(in, actual_path, out_header, size_table, rebuild_threads);
        in.close();
        return layer;
    }

    std::ifstream header_in(actual_path, std::ios::binary);
    if (!header_in) {
        throw std::runtime_error("failed to read EX prefix36 layer: " + actual_path);
    }
    LayerFileHeader header{};
    header_in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!header_in) {
        throw std::runtime_error("failed to read EX prefix36 layer header: " + actual_path);
    }
    validate_layer_header(header, actual_path);
    const uint64_t expected_bytes = layer_file_bytes(header);
    if (fs::exists(actual_path) && fs::file_size(actual_path) < expected_bytes) {
        throw std::runtime_error("truncated EX prefix36 layer: " + actual_path);
    }
    header_in.close();

    FileIOUtils::DirectSequentialReader in(actual_path, expected_bytes, io_config);
    Prefix36Layer layer = read_layer_file_from_reader(in, actual_path, out_header, size_table, rebuild_threads);
    in.close();
    return layer;
}

LutBundle load_or_build_prefix36_lut(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    const ZMaskFrozen::TileLimitConfig config =
        ZMaskFrozen::make_lut_tile_limit_config(
            options.target,
            arr_init,
            spec,
            options.is_free,
            options.is_variant
        );
    const std::string path = lut_file_path(options.pathname);
    if (fs::exists(path)) {
        try {
            LutBundle bundle = read_lut_file(path);
            if (ZMaskFrozen::tile_limit_configs_equal(bundle.config, config) &&
                bundle.physical_transform == spec.physical_transform &&
                bundle.inverse_physical_transform == spec.inverse_physical_transform &&
                bundle.logical_pattern_signature == spec.logical_pattern_signature &&
                bundle.physical_pattern_signature == spec.physical_pattern_signature) {
                return bundle;
            }
        } catch (...) {
        }
    }
    LutBundle bundle = make_lut_bundle(config);
    bundle.physical_transform = spec.physical_transform;
    bundle.inverse_physical_transform = spec.inverse_physical_transform;
    bundle.logical_pattern_signature = spec.logical_pattern_signature;
    bundle.physical_pattern_signature = spec.physical_pattern_signature;
    write_lut_file(path, bundle);
    return bundle;
}

LutBundle load_lut_for_reader(const std::string &path) {
    return read_lut_file(path);
}

struct ColdFileQuery {
    uint64_t target_key = 0;
    uint32_t rank = 0;
    uint32_t valid_count = 0;
    bool valid = false;
};

LutFileHeader read_lut_header_for_cold_lookup(std::ifstream &in, const std::string &path) {
    LutFileHeader header{};
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, kLutMagic, sizeof(kLutMagic)) != 0) {
        throw std::runtime_error("invalid EX prefix36 LUT file magic, rebuild required: " + path);
    }
    if (header.version != kLutVersion) {
        throw std::runtime_error("unsupported EX prefix36 LUT version, rebuild required: " + path);
    }
    if (header.size_table_values == 0U || header.high_base_values == 0U) {
        throw std::runtime_error("invalid EX prefix36 LUT table shape: " + path);
    }
    return header;
}

uint32_t board_tile_sum(uint64_t board) {
    uint32_t sum = 0U;
    for (uint32_t cell = 0; cell < 16U; ++cell) {
        sum += tile_value(static_cast<uint32_t>((board >> (cell * 4U)) & 0xFULL));
    }
    return sum;
}

ColdFileQuery prepare_cold_query_from_lut_file(
    const std::string &zlut_path,
    uint64_t board
) {
    std::ifstream in(zlut_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EX prefix36 LUT: " + zlut_path);
    }
    const LutFileHeader header = read_lut_header_for_cold_lookup(in, zlut_path);
    const LutFileLayout layout = lut_file_layout(header);

    ColdFileQuery query;
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    const uint32_t high = suffix28 >> 24U;
    const uint8_t table_id = header.table_for_high[high];
    if (table_id == DenseLow24RankLut::kInvalidTable) {
        return query;
    }

    const uint32_t low24 = suffix28 & 0xFFFFFFU;
    uint16_t low_rank = ZMaskFrozen::kInvalidRank;
    if (header.packed_rank_pair_values != 0U &&
        (table_id == header.packed_table0 || table_id == header.packed_table1)) {
        if (low24 >= header.packed_rank_pair_values) {
            return query;
        }
        const uint32_t packed = read_one_at<uint32_t>(
            in,
            layout.packed_rank_pair_offset + static_cast<uint64_t>(low24) * sizeof(uint32_t),
            zlut_path
        );
        low_rank = table_id == header.packed_table0
            ? static_cast<uint16_t>(packed & 0xFFFFU)
            : static_cast<uint16_t>(packed >> 16U);
    } else {
        if (table_id >= header.rank_table_count) {
            return query;
        }
        const uint64_t rank_index =
            static_cast<uint64_t>(table_id) * ZMaskFrozen::kSuffixStateCount + low24;
        if (rank_index >= header.rank_table_values) {
            return query;
        }
        low_rank = read_one_at<uint16_t>(
            in,
            layout.rank_tables_offset + rank_index * sizeof(uint16_t),
            zlut_path
        );
    }
    if (low_rank == ZMaskFrozen::kInvalidRank) {
        return query;
    }

    const uint32_t suffix_sum = low24_sum_fast(low24) + tile_value(high);
    if (suffix_sum > kMaxSuffix28Sum || (suffix_sum & 1U) != 0U) {
        return query;
    }
    const uint32_t group = suffix_sum >> 1U;
    if (group >= header.size_table_values) {
        return query;
    }
    const uint32_t valid_count = read_one_at<uint32_t>(
        in,
        layout.size_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
        zlut_path
    );
    const uint64_t high_base_index =
        static_cast<uint64_t>(high) * header.size_table_values + group;
    if (high_base_index >= header.high_base_values) {
        return query;
    }
    const uint16_t high_base = read_one_at<uint16_t>(
        in,
        layout.high_base_offset + high_base_index * sizeof(uint16_t),
        zlut_path
    );
    const uint32_t rank = static_cast<uint32_t>(high_base) + low_rank;
    if (valid_count == 0U || rank >= valid_count) {
        return query;
    }

    const uint64_t prefix36 = board >> kSuffixBits;
    query.target_key = pack_key(prefix36, suffix_sum);
    query.rank = rank;
    query.valid_count = valid_count;
    query.valid = true;
    return query;
}

Prefix40Baseline::Luts build_prefix_luts(
    const ZMaskFrozen::TileLimitConfig &config,
    const RunOptions &options
) {
    return Prefix40Baseline::build_luts(config, std::min(8, thread_count_from_options(options)));
}

bool all_layers_exist(const RunOptions &options) {
    if (options.steps <= 0) {
        return false;
    }
    for (int step = 0; step < options.steps; ++step) {
        if (!fs::exists(layer_file_path(options.pathname, step))) {
            return false;
        }
    }
    return true;
}

std::string optimal_layer_marker_path(const std::string &pathname) {
    return pathname + "ex_optlayer";
}

std::string optimal_complete_marker_path(const std::string &pathname) {
    return pathname + "ex_optimal_complete";
}

bool optimal_complete_marker_exists(const RunOptions &options) {
    return fs::exists(optimal_complete_marker_path(options.pathname));
}

int read_optimal_layer_marker(const RunOptions &options) {
    std::ifstream in(optimal_layer_marker_path(options.pathname));
    int step = kOptimalBranchOnlyStartStep - 1;
    if (in) {
        in >> step;
    }
    return step;
}

void write_optimal_layer_marker(const RunOptions &options, int step) {
    std::ofstream out(optimal_layer_marker_path(options.pathname), std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write EX optimal branch marker");
    }
    out << step;
}

void write_optimal_complete_marker(const RunOptions &options) {
    std::ofstream out(optimal_complete_marker_path(options.pathname), std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write EX optimal completion marker");
    }
    out << "complete\n";
}

std::string generated_layer_file_path(const std::string &pathname, int step) {
    return pathname + std::to_string(step) + kGeneratedLayerFileExtension;
}

std::string generated_layer_archive_path(const std::string &pathname, int step) {
    return generated_layer_file_path(pathname, step) + ".7z";
}

bool layer_input_exists(const std::string &pathname, int step) {
    return fs::exists(layer_file_path(pathname, step)) ||
           fs::exists(generated_layer_file_path(pathname, step)) ||
           fs::exists(generated_layer_archive_path(pathname, step));
}

bool all_layer_inputs_exist(const RunOptions &options) {
    if (options.steps <= 0) {
        return false;
    }
    for (int step = 0; step < options.steps; ++step) {
        if (!layer_input_exists(options.pathname, step)) {
            return false;
        }
    }
    return true;
}

bool optimal_resume_inputs_exist(const RunOptions &options, int last_done) {
    if (options.steps <= kOptimalBranchOnlyStartStep || last_done < kOptimalBranchOnlyStartStep) {
        return false;
    }
    const int first_needed = std::max(0, last_done - 1);
    for (int step = first_needed; step < options.steps; ++step) {
        if (!layer_input_exists(options.pathname, step)) {
            return false;
        }
    }
    return true;
}

std::string existing_layer_input_path(const std::string &pathname, int step) {
    const std::string final_path = layer_file_path(pathname, step);
    if (fs::exists(final_path)) {
        return final_path;
    }
    const std::string generated_path = generated_layer_file_path(pathname, step);
    if (fs::exists(generated_path)) {
        return generated_path;
    }
    const std::string generated_archive = generated_layer_archive_path(pathname, step);
    if (fs::exists(generated_archive)) {
        return generated_archive;
    }
    return final_path;
}

Prefix36Layer read_layer_input(
    const std::string &pathname,
    int step,
    FileIOUtils::DirectIoConfig io_config,
    const DenseLow24RankLut &dense_lut,
    int rebuild_threads,
    const LutBundle *expected_lut = nullptr
) {
    LayerFileHeader header{};
    Prefix36Layer layer = read_layer_file(
        existing_layer_input_path(pathname, step),
        &header,
        io_config,
        &dense_lut.size_table,
        rebuild_threads
    );
    if (expected_lut != nullptr && !physical_metadata_matches(header, *expected_lut)) {
        throw std::runtime_error(
            "EX prefix36 physical pattern metadata does not match LUT: " +
            existing_layer_input_path(pathname, step)
        );
    }
    return layer;
}

void remove_generated_layer_input(const std::string &pathname, int step) {
    std::error_code ec;
    fs::remove(generated_layer_file_path(pathname, step), ec);
    fs::remove(generated_layer_archive_path(pathname, step), ec);
}

void write_generated_layer_file(
    const RunOptions &options,
    int step,
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode,
    FileIOUtils::DirectIoConfig io_config
) {
    const std::string raw_path = generated_layer_file_path(options.pathname, step);
    const std::string archive_path = generated_layer_archive_path(options.pathname, step);
    std::error_code ec;
    if (options.compress_temp_files) {
        write_layer_archive_file(archive_path, layer, spec, mode);
        fs::remove(raw_path, ec);
    } else {
        write_layer_file(raw_path, layer, spec, mode, io_config);
        fs::remove(archive_path, ec);
    }
}

void promote_generated_layer_input(
    const std::string &pathname,
    int step,
    const PatternSpec &spec,
    DTypeMode mode,
    FileIOUtils::DirectIoConfig io_config,
    const LutBundle &lut,
    int rebuild_threads
) {
    const std::string final_path = layer_file_path(pathname, step);
    if (fs::exists(final_path)) {
        return;
    }
    std::error_code ec;
    const std::string generated_path = generated_layer_file_path(pathname, step);
    if (!fs::exists(generated_path)) {
        const std::string archive_path = generated_layer_archive_path(pathname, step);
        if (!fs::exists(archive_path)) {
            return;
        }
        LayerFileHeader header{};
        Prefix36Layer layer = read_layer_file(
            archive_path,
            &header,
            io_config,
            &lut.dense_lut.size_table,
            rebuild_threads
        );
        if (!physical_metadata_matches(header, lut)) {
            throw std::runtime_error("EX prefix36 physical pattern metadata does not match generated archive: " + archive_path);
        }
        write_layer_file(final_path, layer, spec, mode, io_config);
        fs::remove(archive_path, ec);
        return;
    }
    fs::rename(generated_path, final_path, ec);
    if (ec) {
        throw std::runtime_error("failed to promote EX generated layer to zbook: " + generated_path);
    }
}

std::string compressed_layer_file_path(const std::string &pathname, int step) {
    return pathname + std::to_string(step) + EXCompressedResult::kCompressedLayerFileExtension;
}

bool all_compressed_layers_exist(const RunOptions &options) {
    if (options.steps <= 0) {
        return false;
    }
    for (int step = 0; step < options.steps; ++step) {
        if (!fs::exists(compressed_layer_file_path(options.pathname, step))) {
            return false;
        }
    }
    return true;
}

bool compressed_results_are_complete_for_options(const RunOptions &options) {
    if (!all_compressed_layers_exist(options)) {
        return false;
    }
    return !options.optimal_branch_only || optimal_complete_marker_exists(options);
}

EXCompressedResult::Prefix36LayerView layer_compression_view(
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode
) {
    EXCompressedResult::Prefix36LayerView view;
    view.layer_sum = layer.layer_sum;
    view.threshold_bits = layer.threshold_bits;
    view.dtype_mode = static_cast<uint32_t>(mode);
    view.success_kind = storage_kind_for_dtype_mode(mode);
    view.value_size = static_cast<uint32_t>(value_size_for_dtype_mode(mode));
    view.physical_transform = spec.physical_transform;
    view.inverse_physical_transform = spec.inverse_physical_transform;
    view.logical_pattern_signature = spec.logical_pattern_signature;
    view.physical_pattern_signature = spec.physical_pattern_signature;
    view.live_board_count = layer.live_board_count;

    view.bucket_keys = layer.bucket_keys.empty() ? nullptr : layer.bucket_keys.data();
    view.bitmap_offsets = layer.bitmap_offsets.empty() ? nullptr : layer.bitmap_offsets.data();
    view.success_offsets = layer.success_offsets.empty() ? nullptr : layer.success_offsets.data();
    view.bucket_count = layer.bucket_keys.size();

    view.small_bitmap_bytes = layer.small_bitmap_bytes.empty() ? nullptr : layer.small_bitmap_bytes.data();
    view.small_bitmap_byte_count = layer.small_bitmap_bytes.size();
    view.large_bitmap_words = layer.large_bitmap_words.empty() ? nullptr : layer.large_bitmap_words.data();
    view.large_bitmap_word_count = layer.large_bitmap_words.size();

    view.success_values = layer.success_values.empty() ? nullptr : layer.success_values.data();
    view.success_value_count = layer.success_values.size();
    return view;
}

bool compressed_layer_is_fresh(const std::string &source_path, const std::string &output_path) {
    if (!fs::exists(source_path) || !fs::exists(output_path)) {
        return false;
    }
    std::error_code ec;
    const auto output_time = fs::last_write_time(output_path, ec);
    if (ec) {
        return false;
    }
    const auto source_time = fs::last_write_time(source_path, ec);
    return !ec && output_time >= source_time;
}

double compress_layer_result_from_memory(
    const RunOptions &options,
    int step,
    const Prefix36Layer &layer,
    const PatternSpec &spec,
    DTypeMode mode
) {
    if (!options.compress) {
        return 0.0;
    }
    const std::string zbook_path = layer_file_path(options.pathname, step);
    const std::string output_path = compressed_layer_file_path(options.pathname, step);
    if (compressed_layer_is_fresh(zbook_path, output_path)) {
        return 0.0;
    }
    const double t0 = now_seconds();
    const std::string temp_path = output_path + ".tmp";
    std::error_code ec;
    fs::remove(temp_path, ec);
    EXCompressedResult::compress_prefix36_layer_view_to_ex_result(
        layer_compression_view(layer, spec, mode),
        lut_file_path(options.pathname),
        temp_path,
        4096U,
        65536U,
        5
    );
    fs::remove(output_path, ec);
    fs::rename(temp_path, output_path, ec);
    if (ec) {
        fs::remove(temp_path, ec);
        throw std::runtime_error("failed to finalize EX compressed result: " + output_path);
    }
    return now_seconds() - t0;
}

double compress_layer_result_from_file(const RunOptions &options, int step) {
    if (!options.compress) {
        return 0.0;
    }
    const std::string zbook_path = layer_file_path(options.pathname, step);
    if (!fs::exists(zbook_path)) {
        return 0.0;
    }
    const std::string output_path = compressed_layer_file_path(options.pathname, step);
    if (compressed_layer_is_fresh(zbook_path, output_path)) {
        std::error_code ec;
        fs::remove(zbook_path, ec);
        return 0.0;
    }
    const double t0 = now_seconds();
    const std::string temp_path = output_path + ".tmp";
    std::error_code ec;
    fs::remove(temp_path, ec);
    EXCompressedResult::compress_zbook_to_ex_result(
        zbook_path,
        lut_file_path(options.pathname),
        temp_path,
        4096U,
        65536U,
        5
    );
    fs::remove(output_path, ec);
    fs::rename(temp_path, output_path, ec);
    if (ec) {
        fs::remove(temp_path, ec);
        throw std::runtime_error("failed to finalize EX compressed result: " + output_path);
    }
    fs::remove(zbook_path, ec);
    return now_seconds() - t0;
}

double compress_all_layer_results(const RunOptions &options) {
    if (!options.compress || options.steps <= 0) {
        return 0.0;
    }
    const double t0 = now_seconds();
    for (int step = 0; step < options.steps; ++step) {
        (void)compress_layer_result_from_file(options, step);
    }
    return now_seconds() - t0;
}

std::string generate_stats_path(const RunOptions &options) {
    return options.pathname + "zmask_generate_stats.csv";
}

std::string solve_stats_path(const RunOptions &options) {
    return options.pathname + "zmask_solve_stats.csv";
}

void reset_generate_stats(const RunOptions &options) {
    std::ofstream file(generate_stats_path(options), std::ios::trunc);
    file << "stage,step,layout,dtype_mode,canonical_batch_backend,direct_index_type,input_live,primary_live,secondary_live,"
            "bucket_count,input_bucket_count,primary_small_bitmap_bytes,primary_large_bitmap_words,"
            "secondary_bucket_count,secondary_small_bitmap_bytes,secondary_large_bitmap_words,"
            "metadata_bytes,bitmap_density,retry_count,total_seconds,throughput_mbps,compute_seconds,"
            "compute_throughput_mbps,prepare_seconds,work_seconds,finalize_seconds,cleanup_seconds,write_seconds,timestamp\n";
}

void append_generate_stats(
    const RunOptions &options,
    const std::string &stage,
    int step,
    uint64_t input_live,
    const Prefix36Layer *primary,
    const Prefix36Layer *secondary,
    uint32_t retry_count,
    double total_seconds,
    double compute_seconds,
    double prepare_seconds,
    double work_seconds,
    double finalize_seconds,
    double cleanup_seconds,
    double write_seconds
) {
    const uint64_t primary_live = primary ? primary->live_board_count : 0ULL;
    const uint64_t secondary_live = secondary ? secondary->live_board_count : 0ULL;
    const uint64_t denom_live = primary_live != 0ULL ? primary_live : input_live;
    const uint64_t bitmap_bits = primary
        ? static_cast<uint64_t>(primary->small_bitmap_bytes.size()) * 8ULL
            + static_cast<uint64_t>(primary->large_bitmap_words.size()) * 64ULL
        : 0ULL;
    const double density = primary && bitmap_bits != 0ULL
        ? static_cast<double>(primary->live_board_count) / static_cast<double>(bitmap_bits)
        : 0.0;
    std::ofstream file(generate_stats_path(options), std::ios::app);
    file << stage << ','
         << step << ','
         << kLayoutName << ','
         << options.success_rate_dtype << ','
         << CanonicalBatch::backend_name() << ','
         << kDirectIndexName << ','
         << input_live << ','
         << primary_live << ','
         << secondary_live << ','
         << (primary ? primary->bucket_keys.size() : 0ULL) << ','
         << 0 << ','
         << (primary ? primary->small_bitmap_bytes.size() : 0ULL) << ','
         << (primary ? primary->large_bitmap_words.size() : 0ULL) << ','
         << (secondary ? secondary->bucket_keys.size() : 0ULL) << ','
         << (secondary ? secondary->small_bitmap_bytes.size() : 0ULL) << ','
         << (secondary ? secondary->large_bitmap_words.size() : 0ULL) << ','
         << (primary ? layer_metadata_bytes(*primary) : 0ULL) << ','
         << std::setprecision(9) << density << ','
         << retry_count << ','
         << total_seconds << ','
         << throughput(denom_live, total_seconds) << ','
         << compute_seconds << ','
         << throughput(denom_live, compute_seconds) << ','
         << prepare_seconds << ','
         << work_seconds << ','
         << finalize_seconds << ','
         << cleanup_seconds << ','
         << write_seconds << ','
         << "\n";
}

void reset_solve_stats(const RunOptions &options) {
    std::ofstream file(solve_stats_path(options), std::ios::trunc);
    file << "stage,step,layout,dtype_mode,canonical_batch_backend,direct_index_type,input_live,output_live,"
            "future1_live,future2_live,future2_post_threshold_live,deletion_threshold,current_retained_ratio,"
            "future_threshold_retained_ratio,recalculate_seconds,"
            "freeze_zero_compact_seconds,future_index_seconds,future_compact_seconds,current_write_seconds,"
            "future_write_seconds,read_seconds,total_seconds,compute_seconds,active_throughput_mbps,compute_throughput_mbps,"
            "total_throughput_mbps,metadata_bytes,bitmap_density,timestamp\n";
}

void append_solve_stats(
    const RunOptions &options,
    const std::string &stage,
    int step,
    uint64_t input_live,
    uint64_t output_live,
    uint64_t future1_live,
    uint64_t future2_live,
    uint64_t future2_post_live,
    double deletion_threshold,
    double current_retained_ratio,
    double future_threshold_retained_ratio,
    double recalc_seconds,
    double compact_seconds,
    double future_index_seconds,
    double future_compact_seconds,
    double current_write_seconds,
    double future_write_seconds,
    double read_seconds,
    double total_seconds,
    double compute_seconds,
    const Prefix36Layer *layer
) {
    const uint64_t bitmap_bits = layer
        ? static_cast<uint64_t>(layer->small_bitmap_bytes.size()) * 8ULL
            + static_cast<uint64_t>(layer->large_bitmap_words.size()) * 64ULL
        : 0ULL;
    const double density = layer && bitmap_bits != 0ULL
        ? static_cast<double>(layer->live_board_count) / static_cast<double>(bitmap_bits)
        : 0.0;
    std::ofstream file(solve_stats_path(options), std::ios::app);
    file << stage << ','
         << step << ','
         << kLayoutName << ','
         << options.success_rate_dtype << ','
         << CanonicalBatch::backend_name() << ','
         << kDirectIndexName << ','
         << input_live << ','
         << output_live << ','
         << future1_live << ','
         << future2_live << ','
         << future2_post_live << ','
         << std::setprecision(9)
         << deletion_threshold << ','
         << current_retained_ratio << ','
         << future_threshold_retained_ratio << ','
         << recalc_seconds << ','
         << compact_seconds << ','
         << future_index_seconds << ','
         << future_compact_seconds << ','
         << current_write_seconds << ','
         << future_write_seconds << ','
         << read_seconds << ','
         << total_seconds << ','
         << compute_seconds << ','
         << throughput(input_live, recalc_seconds) << ','
         << throughput(input_live, compute_seconds) << ','
         << throughput(input_live, total_seconds) << ','
         << (layer ? layer_metadata_bytes(*layer) : 0ULL) << ','
         << density << ','
         << "\n";
}

struct SolveStepSummary {
    uint64_t input_live = 0;
    uint64_t output_live = 0;
    double recalc_seconds = 0.0;
    double compact_seconds = 0.0;
    double future_index_seconds = 0.0;
    double future_compact_seconds = 0.0;
    double current_write_seconds = 0.0;
    double future_write_seconds = 0.0;
    double read_seconds = 0.0;
    double total_seconds = 0.0;
    double compute_seconds = 0.0;
};

double transition_component_reserve_need(uint64_t current_size, uint64_t next_size, uint64_t padding) {
    if (next_size <= padding) {
        return 0.0;
    }
    if (current_size == 0U) {
        return kDefaultReserveFactor;
    }
    return static_cast<double>(next_size - padding) / static_cast<double>(current_size);
}

double transition_reserve_need(const Prefix36Layer &current, const Prefix36Layer &next) {
    double need = 0.0;
    need = std::max(
        need,
        transition_component_reserve_need(current.bucket_keys.size(), next.bucket_keys.size(), 4096ULL)
    );
    need = std::max(
        need,
        transition_component_reserve_need(
            current.small_bitmap_bytes.size(),
            next.small_bitmap_bytes.size(),
            64ULL * kPrefix36DynamicSmallChunkBytes
        )
    );
    need = std::max(
        need,
        transition_component_reserve_need(
            current.large_bitmap_words.size(),
            next.large_bitmap_words.size(),
            64ULL * kPrefix36DynamicLargeChunkWords
        )
    );
    return need;
}

double reserve_need_recent_quantile(const std::vector<double> &history) {
    const size_t begin = history.size() > kLearnedReserveHistoryWindow
        ? history.size() - kLearnedReserveHistoryWindow
        : 0U;
    std::vector<double> values(history.begin() + static_cast<std::ptrdiff_t>(begin), history.end());
    std::sort(values.begin(), values.end());
    const size_t index = ((values.size() - 1U) * 9U) / 10U;
    return values[index];
}

double regular_reserve_factor(const std::vector<double> &history, double retry_guard_factor) {
    if (history.empty()) {
        return kDefaultReserveFactor;
    }
    double factor = kLearnedReserveMinFactor;
    factor = std::max(factor, history.back() * kLearnedReserveLastGuard);
    factor = std::max(factor, reserve_need_recent_quantile(history) * kLearnedReserveQuantileGuard);
    factor = std::max(factor, retry_guard_factor);
    return std::min(kDefaultReserveFactor, std::max(kLearnedReserveMinFactor, factor));
}

double reserve_factor_for_step(
    int current_step,
    const std::vector<double> &reserve_need_history,
    double retry_guard_factor
) {
    if (current_step < kEarlyLayerReserveFactorSteps) {
        return kEarlyLayerReserveFactor;
    }
    return regular_reserve_factor(reserve_need_history, retry_guard_factor);
}

Prefix36DynamicState make_dynamic_for_current(
    uint32_t layer_sum,
    uint32_t threshold_bits,
    const Prefix36Layer &current,
    double reserve_factor
) {
    const double factor = std::max(0.25, reserve_factor);
    return make_prefix36_dynamic_state(
        layer_sum,
        threshold_bits,
        static_cast<uint64_t>(static_cast<double>(current.bucket_keys.size()) * factor) + 4096ULL,
        static_cast<uint64_t>(static_cast<double>(current.small_bitmap_bytes.size()) * factor)
            + 64ULL * kPrefix36DynamicSmallChunkBytes,
        static_cast<uint64_t>(static_cast<double>(current.large_bitmap_words.size()) * factor)
            + 64ULL * kPrefix36DynamicLargeChunkWords
    );
}

void insert_layer_into_dynamic(
    const Prefix36Layer &source,
    Prefix36DynamicState &target,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int num_threads
) {
#pragma omp parallel num_threads(num_threads)
    {
        std::array<Prefix36DynamicPending, kPrefix36DynamicInsertBufferSize> pending{};
        uint32_t pending_count = 0U;
        Prefix36DynamicThreadChunks chunks{};
        auto flush = [&]() {
            prefix36_dynamic_flush(pending.data(), pending_count, target, chunks);
            pending_count = 0U;
        };
#pragma omp for schedule(dynamic, 16)
        for (int64_t bucket_idx_signed = 0; bucket_idx_signed < static_cast<int64_t>(source.bucket_keys.size()); ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = source.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t group = sum_index(key_remaining_sum(key));
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            auto push_rank = [&](uint32_t rank) {
                const uint64_t board = (prefix36 << kSuffixBits) | dense_lut.unrank_array[unrank_offset + rank];
                prefix36_dynamic_push_board(target, dense_lut, z_luts, board, pending.data(), pending_count, chunks);
            };
            if (valid_count <= source.threshold_bits) {
                const uint32_t offset = source.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = source.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_rank(rank);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = source.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = source.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_rank(rank);
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
}

void build_terminal_success(Prefix36Layer &layer, const DenseLow24RankLut &dense_lut, const PatternSpec &spec, const RunOptions &options, int num_threads) {
    layer.success_values.assign(static_cast<size_t>(layer.live_board_count), 0U);
    const uint32_t max_scale = max_scale_value_for_dtype<uint32_t>(options.success_rate_dtype);
#pragma omp parallel for schedule(dynamic, 256) num_threads(num_threads)
    for (int64_t bucket_idx_signed = 0; bucket_idx_signed < static_cast<int64_t>(layer.bucket_keys.size()); ++bucket_idx_signed) {
        const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
        const uint64_t key = layer.bucket_keys[bucket_idx];
        const uint64_t prefix36 = key_prefix36(key);
        const uint32_t group = sum_index(key_remaining_sum(key));
        const uint32_t valid_count = dense_lut.size_table[group];
        const uint32_t unrank_offset = dense_lut.offset_table[group];
        const uint32_t success_base = layer.success_offsets[bucket_idx];
        uint32_t ordinal = 0U;
        auto set_success = [&](uint32_t rank) {
            const uint64_t board = (prefix36 << kSuffixBits) | dense_lut.unrank_array[unrank_offset + rank];
            layer.success_values[static_cast<size_t>(success_base + ordinal)] =
                is_success_by_shifts(board, options.target, spec.success_shifts) ? max_scale : 0U;
            ++ordinal;
        };
        if (valid_count <= layer.threshold_bits) {
            const uint32_t offset = layer.bitmap_offsets[bucket_idx];
            const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
            for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                uint8_t value = layer.small_bitmap_bytes[offset + byte_idx];
                while (value != 0U) {
                    const uint32_t bit = countr_zero_u32(value);
                    const uint32_t rank = byte_idx * 8U + bit;
                    if (rank >= valid_count) {
                        break;
                    }
                    set_success(rank);
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                }
            }
        } else {
            const uint32_t offset = layer.bitmap_offsets[bucket_idx];
            const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
            for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                uint64_t value = layer.large_bitmap_words[offset + word_idx];
                while (value != 0ULL) {
                    const uint32_t bit = countr_zero_u64(value);
                    const uint32_t rank = word_idx * 64U + bit;
                    if (rank >= valid_count) {
                        break;
                    }
                    set_success(rank);
                    value &= value - 1ULL;
                }
            }
        }
    }
}

struct CompactBucketPlan {
    uint32_t valid_count = 0;
    uint32_t live_count = 0;
    uint32_t kept_count = 0;
    uint32_t bitmap_units = 0;
    uint32_t old_bitmap_offset = 0;
    uint32_t old_success_offset = 0;
    uint32_t out_bucket_index = 0;
    uint32_t out_bitmap_offset = 0;
    uint32_t out_success_offset = 0;
    uint32_t out_large_rank_offset = 0;
    bool is_small = true;
};

uint32_t compact_count_bitmap_live(
    const Prefix36Layer &input,
    const CompactBucketPlan &plan
) {
    uint32_t live = 0U;
    if (plan.is_small) {
        for (uint32_t byte_idx = 0; byte_idx < plan.bitmap_units; ++byte_idx) {
            uint8_t value = input.small_bitmap_bytes[plan.old_bitmap_offset + byte_idx];
            if (byte_idx + 1U == plan.bitmap_units && (plan.valid_count & 7U) != 0U) {
                value = static_cast<uint8_t>(value & static_cast<uint8_t>((1U << (plan.valid_count & 7U)) - 1U));
            }
            live += popcount_u32(value);
        }
    } else {
        for (uint32_t word_idx = 0; word_idx < plan.bitmap_units; ++word_idx) {
            uint64_t value = input.large_bitmap_words[plan.old_bitmap_offset + word_idx];
            if (word_idx + 1U == plan.bitmap_units && (plan.valid_count & 63U) != 0U) {
                value &= ((1ULL << (plan.valid_count & 63U)) - 1ULL);
            }
            live += popcount_u32(static_cast<uint32_t>(value));
            live += popcount_u32(static_cast<uint32_t>(value >> 32U));
        }
    }
    return live;
}

uint32_t compact_count_success_kept(
    const std::vector<uint32_t> &success_values,
    uint32_t begin,
    uint32_t count,
    uint32_t threshold
) {
    uint32_t kept = 0U;
    const uint32_t *ptr = success_values.data() + begin;
    for (uint32_t i = 0; i < count; ++i) {
        kept += ptr[i] > threshold ? 1U : 0U;
    }
    return kept;
}

uint32_t compact_count_kept_by_bitmap(
    const Prefix36Layer &input,
    const CompactBucketPlan &plan,
    uint32_t threshold,
    uint32_t &live_count
) {
    uint32_t kept = 0U;
    uint32_t ordinal = 0U;
    auto count_rank = [&]() {
        if (static_cast<uint64_t>(plan.old_success_offset) + ordinal >= input.success_values.size()) {
            throw std::runtime_error("EX prefix36 compact success offset out of bounds");
        }
        kept += input.success_values[static_cast<size_t>(plan.old_success_offset + ordinal)] > threshold ? 1U : 0U;
        ++ordinal;
    };
    if (plan.is_small) {
        for (uint32_t byte_idx = 0; byte_idx < plan.bitmap_units; ++byte_idx) {
            uint8_t value = input.small_bitmap_bytes[plan.old_bitmap_offset + byte_idx];
            while (value != 0U) {
                const uint32_t bit = countr_zero_u32(value);
                const uint32_t rank = byte_idx * 8U + bit;
                if (rank >= plan.valid_count) {
                    break;
                }
                count_rank();
                value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
            }
        }
    } else {
        for (uint32_t word_idx = 0; word_idx < plan.bitmap_units; ++word_idx) {
            uint64_t value = input.large_bitmap_words[plan.old_bitmap_offset + word_idx];
            while (value != 0ULL) {
                const uint32_t bit = countr_zero_u64(value);
                const uint32_t rank = word_idx * 64U + bit;
                if (rank >= plan.valid_count) {
                    break;
                }
                count_rank();
                value &= value - 1ULL;
            }
        }
    }
    live_count = ordinal;
    return kept;
}

Prefix36Layer compact_layer(
    const Prefix36Layer &input,
    const DenseLow24RankLut &dense_lut,
    uint32_t threshold,
    int num_threads
) {
    Prefix36Layer out;
    out.layer_sum = input.layer_sum;
    out.threshold_bits = input.threshold_bits;

    const size_t bucket_count = input.bucket_keys.size();
    if (bucket_count == 0U || input.success_values.empty()) {
        return out;
    }

    std::vector<CompactBucketPlan> plans(bucket_count);
    const int thread_count = std::max(1, num_threads);

#pragma omp parallel for schedule(dynamic, 256) num_threads(thread_count)
    for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(bucket_count); ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        CompactBucketPlan plan;
        const uint32_t group = sum_index(key_remaining_sum(input.bucket_keys[i]));
        const uint32_t valid_count = dense_lut.size_table[group];
        plan.valid_count = valid_count;
        plan.is_small = valid_count <= input.threshold_bits;
        plan.bitmap_units = plan.is_small
            ? static_cast<uint32_t>(bytes_for_bits(valid_count))
            : static_cast<uint32_t>(words_for_bits(valid_count));
        plan.old_bitmap_offset = input.bitmap_offsets[i];
        plan.old_success_offset = input.success_offsets[i];
        const uint32_t bitmap_live = compact_count_bitmap_live(input, plan);
        const uint64_t next_success_offset = (i + 1U < bucket_count)
            ? static_cast<uint64_t>(input.success_offsets[i + 1U])
            : static_cast<uint64_t>(input.success_values.size());
        uint32_t live_count = bitmap_live;
        uint32_t kept = 0U;
        if (next_success_offset >= plan.old_success_offset &&
            next_success_offset <= input.success_values.size() &&
            next_success_offset - plan.old_success_offset == bitmap_live) {
            kept = compact_count_success_kept(input.success_values, plan.old_success_offset, bitmap_live, threshold);
        } else {
            kept = compact_count_kept_by_bitmap(input, plan, threshold, live_count);
        }
        plan.live_count = live_count;
        plan.kept_count = kept;
        plans[i] = plan;
    }

    uint64_t output_bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t large_rank_base_count = 0;
    uint64_t success_value_count = 0;
    for (size_t i = 0; i < bucket_count; ++i) {
        CompactBucketPlan &plan = plans[i];
        if (plan.kept_count == 0U) {
            continue;
        }
        if (output_bucket_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EX prefix36 compact bucket count exceeds uint32_t");
        }
        if (success_value_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EX prefix36 compact success offset exceeds uint32_t");
        }
        plan.out_bucket_index = static_cast<uint32_t>(output_bucket_count++);
        plan.out_success_offset = static_cast<uint32_t>(success_value_count);
        success_value_count += plan.kept_count;
        if (plan.is_small) {
            if (small_bitmap_bytes > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EX prefix36 compact small bitmap offset exceeds uint32_t");
            }
            plan.out_bitmap_offset = static_cast<uint32_t>(small_bitmap_bytes);
            small_bitmap_bytes += plan.bitmap_units;
        } else {
            if (large_bitmap_words > std::numeric_limits<uint32_t>::max() ||
                large_rank_base_count > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EX prefix36 compact large bitmap offset exceeds uint32_t");
            }
            plan.out_bitmap_offset = static_cast<uint32_t>(large_bitmap_words);
            plan.out_large_rank_offset = static_cast<uint32_t>(large_rank_base_count);
            large_bitmap_words += plan.bitmap_units;
            large_rank_base_count += large_rank_bases_for_words(plan.bitmap_units);
        }
    }
    if (success_value_count == 0U) {
        return out;
    }

    out.bucket_keys.resize(static_cast<size_t>(output_bucket_count));
    out.bitmap_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.success_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.large_rank_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.small_bitmap_bytes.assign(static_cast<size_t>(small_bitmap_bytes), 0U);
    out.large_bitmap_words.assign(static_cast<size_t>(large_bitmap_words), 0ULL);
    out.large_rank_bases.assign(static_cast<size_t>(large_rank_base_count), 0U);
    out.success_values.resize(static_cast<size_t>(success_value_count));

#pragma omp parallel for schedule(dynamic, 256) num_threads(thread_count)
    for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(bucket_count); ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        const CompactBucketPlan &plan = plans[i];
        if (plan.kept_count == 0U) {
            continue;
        }

        const uint32_t out_bucket = plan.out_bucket_index;
        out.bucket_keys[out_bucket] = input.bucket_keys[i];
        out.bitmap_offsets[out_bucket] = plan.out_bitmap_offset;
        out.success_offsets[out_bucket] = plan.out_success_offset;
        out.large_rank_offsets[out_bucket] = plan.is_small ? 0U : plan.out_large_rank_offset;

        if (plan.kept_count == plan.live_count) {
            std::copy_n(
                input.success_values.data() + plan.old_success_offset,
                plan.live_count,
                out.success_values.data() + plan.out_success_offset
            );
            if (plan.is_small) {
                std::copy_n(
                    input.small_bitmap_bytes.data() + plan.old_bitmap_offset,
                    plan.bitmap_units,
                    out.small_bitmap_bytes.data() + plan.out_bitmap_offset
                );
            } else {
                std::copy_n(
                    input.large_bitmap_words.data() + plan.old_bitmap_offset,
                    plan.bitmap_units,
                    out.large_bitmap_words.data() + plan.out_bitmap_offset
                );
                fill_large_rank_bases(
                    out,
                    plan.out_bitmap_offset,
                    plan.out_large_rank_offset,
                    plan.bitmap_units
                );
            }
            continue;
        }

        uint32_t ordinal = 0U;
        uint32_t kept = 0U;
        if (plan.is_small) {
            for (uint32_t byte_idx = 0; byte_idx < plan.bitmap_units; ++byte_idx) {
                uint8_t value = input.small_bitmap_bytes[plan.old_bitmap_offset + byte_idx];
                uint8_t out_value = 0U;
                while (value != 0U) {
                    const uint32_t bit = countr_zero_u32(value);
                    const uint32_t rank = byte_idx * 8U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    const uint32_t success =
                        input.success_values[static_cast<size_t>(plan.old_success_offset + ordinal)];
                    if (success > threshold) {
                        out_value = static_cast<uint8_t>(out_value | static_cast<uint8_t>(1U << bit));
                        out.success_values[static_cast<size_t>(plan.out_success_offset + kept)] = success;
                        ++kept;
                    }
                    ++ordinal;
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                }
                out.small_bitmap_bytes[plan.out_bitmap_offset + byte_idx] = out_value;
            }
        } else {
            for (uint32_t word_idx = 0; word_idx < plan.bitmap_units; ++word_idx) {
                uint64_t value = input.large_bitmap_words[plan.old_bitmap_offset + word_idx];
                uint64_t out_value = 0ULL;
                while (value != 0ULL) {
                    const uint32_t bit = countr_zero_u64(value);
                    const uint32_t rank = word_idx * 64U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    const uint32_t success =
                        input.success_values[static_cast<size_t>(plan.old_success_offset + ordinal)];
                    if (success > threshold) {
                        out_value |= (1ULL << bit);
                        out.success_values[static_cast<size_t>(plan.out_success_offset + kept)] = success;
                        ++kept;
                    }
                    ++ordinal;
                    value &= value - 1ULL;
                }
                out.large_bitmap_words[plan.out_bitmap_offset + word_idx] = out_value;
            }
            fill_large_rank_bases(
                out,
                plan.out_bitmap_offset,
                plan.out_large_rank_offset,
                plan.bitmap_units
            );
        }
        if (kept != plan.kept_count) {
            throw std::runtime_error("EX prefix36 compact kept count mismatch");
        }
    }
    out.live_board_count = success_value_count;
    return out;
}

bool keep_bit_is_set(const std::vector<uint64_t> &keep_bits, uint64_t index) {
    const uint64_t word = index >> 6U;
    if (word >= keep_bits.size()) {
        return false;
    }
    return (keep_bits[static_cast<size_t>(word)] & (1ULL << (index & 63U))) != 0ULL;
}

uint32_t count_keep_range(const std::vector<uint64_t> &keep_bits, uint32_t begin, uint32_t count) {
    if (count == 0U) {
        return 0U;
    }
    const uint64_t range_begin = begin;
    const uint64_t range_end = range_begin + count;
    const uint64_t first_word = range_begin >> 6U;
    const uint64_t last_word = (range_end - 1U) >> 6U;
    uint32_t kept = 0U;
    for (uint64_t word = first_word; word <= last_word && word < keep_bits.size(); ++word) {
        uint64_t value = keep_bits[static_cast<size_t>(word)];
        if (word == first_word && (range_begin & 63U) != 0U) {
            value &= ~((1ULL << (range_begin & 63U)) - 1ULL);
        }
        if (word == last_word && (range_end & 63U) != 0U) {
            value &= ((1ULL << (range_end & 63U)) - 1ULL);
        }
        kept += popcount_u64(value);
    }
    return kept;
}

Prefix36Layer compact_layer_by_keep_bits(
    const Prefix36Layer &input,
    const DenseLow24RankLut &dense_lut,
    const std::vector<uint64_t> &keep_bits,
    int num_threads
) {
    Prefix36Layer out;
    out.layer_sum = input.layer_sum;
    out.threshold_bits = input.threshold_bits;

    const size_t bucket_count = input.bucket_keys.size();
    if (bucket_count == 0U || input.success_values.empty()) {
        return out;
    }

    std::vector<CompactBucketPlan> plans(bucket_count);
    const int thread_count = std::max(1, num_threads);

#pragma omp parallel for schedule(dynamic, 256) num_threads(thread_count)
    for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(bucket_count); ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        CompactBucketPlan plan;
        const uint32_t group = sum_index(key_remaining_sum(input.bucket_keys[i]));
        const uint32_t valid_count = dense_lut.size_table[group];
        plan.valid_count = valid_count;
        plan.is_small = valid_count <= input.threshold_bits;
        plan.bitmap_units = plan.is_small
            ? static_cast<uint32_t>(bytes_for_bits(valid_count))
            : static_cast<uint32_t>(words_for_bits(valid_count));
        plan.old_bitmap_offset = input.bitmap_offsets[i];
        plan.old_success_offset = input.success_offsets[i];
        plan.live_count = compact_count_bitmap_live(input, plan);
        plan.kept_count = count_keep_range(keep_bits, plan.old_success_offset, plan.live_count);
        plans[i] = plan;
    }

    uint64_t output_bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t large_rank_base_count = 0;
    uint64_t success_value_count = 0;
    for (size_t i = 0; i < bucket_count; ++i) {
        CompactBucketPlan &plan = plans[i];
        if (plan.kept_count == 0U) {
            continue;
        }
        if (output_bucket_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EX prefix36 keep compact bucket count exceeds uint32_t");
        }
        if (success_value_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EX prefix36 keep compact success offset exceeds uint32_t");
        }
        plan.out_bucket_index = static_cast<uint32_t>(output_bucket_count++);
        plan.out_success_offset = static_cast<uint32_t>(success_value_count);
        success_value_count += plan.kept_count;
        if (plan.is_small) {
            if (small_bitmap_bytes > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EX prefix36 keep compact small bitmap offset exceeds uint32_t");
            }
            plan.out_bitmap_offset = static_cast<uint32_t>(small_bitmap_bytes);
            small_bitmap_bytes += plan.bitmap_units;
        } else {
            if (large_bitmap_words > std::numeric_limits<uint32_t>::max() ||
                large_rank_base_count > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("EX prefix36 keep compact large bitmap offset exceeds uint32_t");
            }
            plan.out_bitmap_offset = static_cast<uint32_t>(large_bitmap_words);
            plan.out_large_rank_offset = static_cast<uint32_t>(large_rank_base_count);
            large_bitmap_words += plan.bitmap_units;
            large_rank_base_count += large_rank_bases_for_words(plan.bitmap_units);
        }
    }
    if (success_value_count == 0U) {
        return out;
    }

    out.bucket_keys.resize(static_cast<size_t>(output_bucket_count));
    out.bitmap_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.success_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.large_rank_offsets.resize(static_cast<size_t>(output_bucket_count));
    out.small_bitmap_bytes.assign(static_cast<size_t>(small_bitmap_bytes), 0U);
    out.large_bitmap_words.assign(static_cast<size_t>(large_bitmap_words), 0ULL);
    out.large_rank_bases.assign(static_cast<size_t>(large_rank_base_count), 0U);
    out.success_values.resize(static_cast<size_t>(success_value_count));

#pragma omp parallel for schedule(dynamic, 256) num_threads(thread_count)
    for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(bucket_count); ++i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        const CompactBucketPlan &plan = plans[i];
        if (plan.kept_count == 0U) {
            continue;
        }

        const uint32_t out_bucket = plan.out_bucket_index;
        out.bucket_keys[out_bucket] = input.bucket_keys[i];
        out.bitmap_offsets[out_bucket] = plan.out_bitmap_offset;
        out.success_offsets[out_bucket] = plan.out_success_offset;
        out.large_rank_offsets[out_bucket] = plan.is_small ? 0U : plan.out_large_rank_offset;

        if (plan.kept_count == plan.live_count) {
            std::copy_n(
                input.success_values.data() + plan.old_success_offset,
                plan.live_count,
                out.success_values.data() + plan.out_success_offset
            );
            if (plan.is_small) {
                std::copy_n(
                    input.small_bitmap_bytes.data() + plan.old_bitmap_offset,
                    plan.bitmap_units,
                    out.small_bitmap_bytes.data() + plan.out_bitmap_offset
                );
            } else {
                std::copy_n(
                    input.large_bitmap_words.data() + plan.old_bitmap_offset,
                    plan.bitmap_units,
                    out.large_bitmap_words.data() + plan.out_bitmap_offset
                );
                fill_large_rank_bases(
                    out,
                    plan.out_bitmap_offset,
                    plan.out_large_rank_offset,
                    plan.bitmap_units
                );
            }
            continue;
        }

        uint32_t ordinal = 0U;
        uint32_t kept = 0U;
        if (plan.is_small) {
            for (uint32_t byte_idx = 0; byte_idx < plan.bitmap_units; ++byte_idx) {
                uint8_t value = input.small_bitmap_bytes[plan.old_bitmap_offset + byte_idx];
                uint8_t out_value = 0U;
                while (value != 0U) {
                    const uint32_t bit = countr_zero_u32(value);
                    const uint32_t rank = byte_idx * 8U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    const uint32_t success_index = plan.old_success_offset + ordinal;
                    if (keep_bit_is_set(keep_bits, success_index)) {
                        out_value = static_cast<uint8_t>(out_value | static_cast<uint8_t>(1U << bit));
                        out.success_values[static_cast<size_t>(plan.out_success_offset + kept)] =
                            input.success_values[static_cast<size_t>(success_index)];
                        ++kept;
                    }
                    ++ordinal;
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                }
                out.small_bitmap_bytes[plan.out_bitmap_offset + byte_idx] = out_value;
            }
        } else {
            for (uint32_t word_idx = 0; word_idx < plan.bitmap_units; ++word_idx) {
                uint64_t value = input.large_bitmap_words[plan.old_bitmap_offset + word_idx];
                uint64_t out_value = 0ULL;
                while (value != 0ULL) {
                    const uint32_t bit = countr_zero_u64(value);
                    const uint32_t rank = word_idx * 64U + bit;
                    if (rank >= plan.valid_count) {
                        break;
                    }
                    const uint32_t success_index = plan.old_success_offset + ordinal;
                    if (keep_bit_is_set(keep_bits, success_index)) {
                        out_value |= (1ULL << bit);
                        out.success_values[static_cast<size_t>(plan.out_success_offset + kept)] =
                            input.success_values[static_cast<size_t>(success_index)];
                        ++kept;
                    }
                    ++ordinal;
                    value &= value - 1ULL;
                }
                out.large_bitmap_words[plan.out_bitmap_offset + word_idx] = out_value;
            }
            fill_large_rank_bases(
                out,
                plan.out_bitmap_offset,
                plan.out_large_rank_offset,
                plan.bitmap_units
            );
        }
        if (kept != plan.kept_count) {
            throw std::runtime_error("EX prefix36 keep compact kept count mismatch");
        }
    }
    out.live_board_count = success_value_count;
    return out;
}

struct OptimalBranchStats {
    uint64_t source_live = 0;
    uint64_t candidates = 0;
    uint64_t found = 0;
    uint64_t marked = 0;
    double mark_seconds = 0.0;
};

struct OptimalMarkWorkspace {
    static constexpr size_t kMaxCells = static_cast<size_t>(kBatchSize) * 16U;
    static constexpr size_t kMaxCandidates = kMaxCells * 4U;
    std::array<uint32_t, kMaxCells> best_success{};
    std::array<uint32_t, kMaxCells> best_index{};
    std::array<uint64_t, kMaxCandidates> canonical_candidates{};
    std::array<uint16_t, kMaxCandidates> candidate_refs{};
    std::vector<PreparedQuery> queries;

    OptimalMarkWorkspace() {
        queries.reserve(kMaxCandidates);
    }
};

void mark_keep_index(std::vector<std::atomic<uint64_t>> &keep_words, uint32_t success_index) {
    const uint32_t word = success_index >> 6U;
    if (word >= keep_words.size()) {
        return;
    }
    keep_words[static_cast<size_t>(word)].fetch_or(
        1ULL << (success_index & 63U),
        std::memory_order_relaxed
    );
}

template <typename Mover>
OptimalBranchStats mark_optimal_batch(
    const uint64_t *boards,
    uint32_t board_count,
    const Prefix36Layer &target,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    const PatternSpec &spec,
    uint32_t spawn_exp,
    std::vector<std::atomic<uint64_t>> &keep_words,
    OptimalMarkWorkspace &workspace
) {
    OptimalBranchStats stats;
    stats.source_live = board_count;
    workspace.queries.clear();
    uint32_t cell_count = 0U;
    uint32_t canonical_count = 0U;

    auto flush_canonical = [&]() {
        if (canonical_count == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(
            workspace.canonical_candidates.data(),
            canonical_count,
            spec.symm_mode
        );
        for (uint32_t i = 0; i < canonical_count; ++i) {
            PreparedQuery query = prepare_query_dense_hot(
                dense_lut,
                z_luts,
                workspace.canonical_candidates[i],
                target.threshold_bits
            );
            query.ref = workspace.candidate_refs[i];
            workspace.queries.push_back(query);
        }
        canonical_count = 0U;
    };

    auto push_candidate = [&](uint64_t moved, uint16_t ref) {
        workspace.canonical_candidates[canonical_count] = moved;
        workspace.candidate_refs[canonical_count] = ref;
        ++canonical_count;
        ++stats.candidates;
        if (canonical_count == OptimalMarkWorkspace::kMaxCandidates) {
            flush_canonical();
        }
    };

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = zero_cell_mask16(board);
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>(cell_count++);
            workspace.best_success[ref] = 0U;
            workspace.best_index[ref] = std::numeric_limits<uint32_t>::max();
            const uint64_t spawned = board | (static_cast<uint64_t>(spawn_exp) << (4U * cell));
            const auto moves = Mover::move_all_dir(spawned);
            const uint64_t moved_boards[4] = {
                std::get<0>(moves), std::get<1>(moves),
                std::get<2>(moves), std::get<3>(moves)
            };
            for (uint64_t moved : moved_boards) {
                if (moved != spawned && is_pattern(moved, spec.pattern_masks)) {
                    push_candidate(moved, ref);
                }
            }
        }
    }
    flush_canonical();

    uint32_t success_indices[kBatchSize];
    uint8_t found_flags[kBatchSize];
    for (uint32_t base = 0; base < static_cast<uint32_t>(workspace.queries.size()); base += kBatchSize) {
        const uint32_t count = std::min<uint32_t>(kBatchSize, static_cast<uint32_t>(workspace.queries.size()) - base);
        lookup_prepared_batch_direct_entry(
            target,
            workspace.queries.data() + base,
            success_indices,
            found_flags,
            count
        );
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] != 0U) {
                __builtin_prefetch(&target.success_values[static_cast<size_t>(success_indices[i])], 0, 1);
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (found_flags[i] == 0U) {
                continue;
            }
            ++stats.found;
            const uint32_t success_index = success_indices[i];
            const uint32_t success = target.success_values[static_cast<size_t>(success_index)];
            const uint16_t ref = workspace.queries[base + i].ref;
            if (success > workspace.best_success[ref]) {
                workspace.best_success[ref] = success;
                workspace.best_index[ref] = success_index;
            }
        }
    }

    for (uint32_t ref = 0; ref < cell_count; ++ref) {
        const uint32_t success_index = workspace.best_index[ref];
        if (workspace.best_success[ref] != 0U &&
            success_index != std::numeric_limits<uint32_t>::max()) {
            mark_keep_index(keep_words, success_index);
            ++stats.marked;
        }
    }
    return stats;
}

template <typename Mover>
OptimalBranchStats mark_optimal_branches_from_source(
    const Prefix36Layer &source,
    const Prefix36Layer &target,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    const PatternSpec &spec,
    uint32_t spawn_exp,
    std::vector<std::atomic<uint64_t>> &keep_words,
    int num_threads
) {
    const double t0 = now_seconds();
    std::vector<OptimalBranchStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        OptimalBranchStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        uint32_t board_buffer_count = 0U;
        OptimalMarkWorkspace workspace;
        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            OptimalBranchStats batch = mark_optimal_batch<Mover>(
                board_buffer.data(),
                board_buffer_count,
                target,
                dense_lut,
                z_luts,
                spec,
                spawn_exp,
                keep_words,
                workspace
            );
            stats.source_live += batch.source_live;
            stats.candidates += batch.candidates;
            stats.found += batch.found;
            stats.marked += batch.marked;
            board_buffer_count = 0U;
        };
#pragma omp for schedule(dynamic, 16)
        for (int64_t bucket_idx_signed = 0; bucket_idx_signed < static_cast<int64_t>(source.bucket_keys.size()); ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = source.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t group = sum_index(key_remaining_sum(key));
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            auto push_rank = [&](uint32_t rank) {
                board_buffer[board_buffer_count++] =
                    (prefix36 << kSuffixBits) |
                    static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                if (board_buffer_count == kBatchSize) {
                    flush();
                }
            };
            if (valid_count <= source.threshold_bits) {
                const uint32_t offset = source.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = source.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_rank(rank);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = source.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = source.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_rank(rank);
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    OptimalBranchStats total;
    for (const OptimalBranchStats &stats : per_thread) {
        total.source_live += stats.source_live;
        total.candidates += stats.candidates;
        total.found += stats.found;
        total.marked += stats.marked;
    }
    total.mark_seconds = now_seconds() - t0;
    return total;
}

template <typename Mover>
void recalculate_batch_prefix36_write(
    const uint64_t *boards,
    const uint64_t *output_positions,
    uint32_t board_count,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    const PatternSpec &spec,
    double spawn_rate4,
    RecalcWorkspace &workspace,
    RecalcStats &stats,
    Prefix36Layer &current
) {
    workspace.queries1.clear();
    workspace.queries2.clear();
    const double gen_t0 = now_seconds();
    constexpr size_t kMaxCandidatesPerSide = static_cast<size_t>(kBatchSize) * 16U * 4U;
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates1{};
    std::array<uint64_t, kMaxCandidatesPerSide> canonical_candidates2{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs1{};
    std::array<uint16_t, kMaxCandidatesPerSide> candidate_refs2{};
    size_t canonical_count1 = 0U;
    size_t canonical_count2 = 0U;
    auto flush_canonical1 = [&]() {
        if (canonical_count1 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates1.data(), canonical_count1, spec.symm_mode);
        for (size_t i = 0; i < canonical_count1; ++i) {
            PreparedQuery query = prepare_query_dense_hot(dense_lut, z_luts, canonical_candidates1[i], future1.threshold_bits);
            query.ref = candidate_refs1[i];
            workspace.queries1.push_back(query);
        }
        canonical_count1 = 0U;
    };
    auto flush_canonical2 = [&]() {
        if (canonical_count2 == 0U) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(canonical_candidates2.data(), canonical_count2, spec.symm_mode);
        for (size_t i = 0; i < canonical_count2; ++i) {
            PreparedQuery query = prepare_query_dense_hot(dense_lut, z_luts, canonical_candidates2[i], future2.threshold_bits);
            query.ref = candidate_refs2[i];
            workspace.queries2.push_back(query);
        }
        canonical_count2 = 0U;
    };
    auto push1 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates1[canonical_count1] = moved;
        candidate_refs1[canonical_count1] = ref;
        if (++canonical_count1 == canonical_candidates1.size()) {
            flush_canonical1();
        }
    };
    auto push2 = [&](uint64_t moved, uint16_t ref) {
        canonical_candidates2[canonical_count2] = moved;
        candidate_refs2[canonical_count2] = ref;
        if (++canonical_count2 == canonical_candidates2.size()) {
            flush_canonical2();
        }
    };
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            workspace.best2[ref] = 0U;
            workspace.best4[ref] = 0U;
            const uint64_t spawn2 = board | (1ULL << (4U * cell));
            const auto moves2 = Mover::move_all_dir(spawn2);
            const uint64_t b2[4] = {std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)};
            for (uint64_t moved : b2) {
                if (moved != spawn2 && is_pattern(moved, spec.pattern_masks)) {
                    push1(moved, ref);
                }
            }
            const uint64_t spawn4 = board | (2ULL << (4U * cell));
            const auto moves4 = Mover::move_all_dir(spawn4);
            const uint64_t b4[4] = {std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)};
            for (uint64_t moved : b4) {
                if (moved != spawn4 && is_pattern(moved, spec.pattern_masks)) {
                    push2(moved, ref);
                }
            }
        }
    }
    flush_canonical1();
    flush_canonical2();
    const double gen_t1 = now_seconds();

    const double lookup1_t0 = now_seconds();
    const uint64_t found1 = lookup_reduce_query_vector_dense_direct_entry(future1, workspace.queries1, workspace.best2);
    const double lookup1_t1 = now_seconds();
    const double lookup2_t0 = now_seconds();
    const uint64_t found2 = lookup_reduce_query_vector_dense_direct_entry(future2, workspace.queries2, workspace.best4);
    const double lookup2_t1 = now_seconds();

    const double finalize_t0 = now_seconds();
    uint64_t checksum = 0U;
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        double success_probability = 0.0;
        uint32_t empty_count = 0U;
        uint32_t empty_mask = workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t cell = countr_zero_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + static_cast<size_t>(cell);
            success_probability += static_cast<double>(workspace.best2[best_index]) * (1.0 - spawn_rate4);
            success_probability += static_cast<double>(workspace.best4[best_index]) * spawn_rate4;
            ++empty_count;
        }
        const uint32_t value = empty_count > 0U
            ? static_cast<uint32_t>(success_probability / static_cast<double>(empty_count))
            : 0U;
        current.success_values[static_cast<size_t>(output_positions[board_slot])] = value;
        checksum += static_cast<uint64_t>(value) * (output_positions[board_slot] + 1ULL);
    }
    const double finalize_t1 = now_seconds();

    stats.generate_prepare_seconds += gen_t1 - gen_t0;
    stats.lookup1_seconds += lookup1_t1 - lookup1_t0;
    stats.lookup2_seconds += lookup2_t1 - lookup2_t0;
    stats.finalize_seconds += finalize_t1 - finalize_t0;
    stats.boards += board_count;
    stats.queries1 += workspace.queries1.size();
    stats.queries2 += workspace.queries2.size();
    stats.found1 += found1;
    stats.found2 += found2;
    stats.checksum += checksum;
}

template <typename Mover>
RecalcStats recalculate_current_layer(
    Prefix36Layer &current,
    const Prefix36Layer &future1,
    const Prefix36Layer &future2,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    const PatternSpec &spec,
    const RunOptions &options,
    bool do_check,
    int num_threads
) {
    current.success_values.assign(static_cast<size_t>(current.live_board_count), 0U);
    const uint32_t max_scale = max_scale_value_for_dtype<uint32_t>(options.success_rate_dtype);
    const double t0 = now_seconds();
    std::vector<RecalcStats> per_thread(static_cast<size_t>(num_threads));
#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        RecalcStats &stats = per_thread[static_cast<size_t>(tid)];
        std::array<uint64_t, kBatchSize> board_buffer{};
        std::array<uint64_t, kBatchSize> output_buffer{};
        uint32_t board_buffer_count = 0U;
        RecalcWorkspace workspace;
        auto flush = [&]() {
            if (board_buffer_count == 0U) {
                return;
            }
            recalculate_batch_prefix36_write<Mover>(
                board_buffer.data(),
                output_buffer.data(),
                board_buffer_count,
                future1,
                future2,
                dense_lut,
                z_luts,
                spec,
                options.spawn_rate4,
                workspace,
                stats,
                current
            );
            board_buffer_count = 0U;
        };
#pragma omp for schedule(dynamic, 16)
        for (int64_t bucket_idx_signed = 0; bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size()); ++bucket_idx_signed) {
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t group = sum_index(key_remaining_sum(key));
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            const uint32_t success_base = current.success_offsets[bucket_idx];
            uint32_t ordinal = 0U;
            auto push_or_mark_success = [&](uint32_t rank) {
                const uint64_t board =
                    (prefix36 << kSuffixBits) | static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank]);
                const uint32_t output_pos = success_base + ordinal;
                ++ordinal;
                if (do_check && is_success_by_shifts(board, options.target, spec.success_shifts)) {
                    current.success_values[static_cast<size_t>(output_pos)] = max_scale;
                    stats.boards += 1U;
                    stats.checksum += static_cast<uint64_t>(max_scale) * (static_cast<uint64_t>(output_pos) + 1ULL);
                    return;
                }
                board_buffer[board_buffer_count] = board;
                output_buffer[board_buffer_count] = output_pos;
                ++board_buffer_count;
                if (board_buffer_count == kBatchSize) {
                    flush();
                }
            };
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_or_mark_success(rank);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        push_or_mark_success(rank);
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush();
    }
    const double t1 = now_seconds();
    RecalcStats total;
    total.seconds = t1 - t0;
    for (const RecalcStats &stats : per_thread) {
        total.boards += stats.boards;
        total.generate_prepare_seconds += stats.generate_prepare_seconds;
        total.lookup1_seconds += stats.lookup1_seconds;
        total.lookup2_seconds += stats.lookup2_seconds;
        total.finalize_seconds += stats.finalize_seconds;
        total.queries1 += stats.queries1;
        total.queries2 += stats.queries2;
        total.found1 += stats.found1;
        total.found2 += stats.found2;
        total.checksum += stats.checksum;
    }
    total.mbps = throughput(total.boards, total.seconds);
    return total;
}

template <typename Mover>
void prefix36_dynamic_generate_into_production(
    const Prefix36Layer &current,
    Prefix36DynamicState &arr1_state,
    Prefix36DynamicState &arr2_state,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    const PatternSpec &spec,
    const RunOptions &options,
    int num_threads,
    bool do_check
) {
#pragma omp parallel num_threads(num_threads)
    {
        std::array<Prefix36DynamicPending, kPrefix36DynamicInsertBufferSize> pending1{};
        std::array<Prefix36DynamicPending, kPrefix36DynamicInsertBufferSize> pending2{};
        uint32_t pending_count1 = 0U;
        uint32_t pending_count2 = 0U;
        Prefix36DynamicThreadChunks chunks1{};
        Prefix36DynamicThreadChunks chunks2{};
        auto flush1 = [&]() {
            prefix36_dynamic_flush(pending1.data(), pending_count1, arr1_state, chunks1);
            pending_count1 = 0U;
        };
        auto flush2 = [&]() {
            prefix36_dynamic_flush(pending2.data(), pending_count2, arr2_state, chunks2);
            pending_count2 = 0U;
        };
        std::array<uint64_t, kPrefix36DynamicInsertBufferSize> canonical1{};
        std::array<uint64_t, kPrefix36DynamicInsertBufferSize> canonical2{};
        uint32_t canonical_count1 = 0U;
        uint32_t canonical_count2 = 0U;
        auto push_ready1 = [&](uint64_t canonical) {
            if (!is_pattern(canonical, spec.pattern_masks)) {
                return;
            }
            prefix36_dynamic_push_board(
                arr1_state, dense_lut, z_luts, canonical, pending1.data(), pending_count1, chunks1);
        };
        auto push_ready2 = [&](uint64_t canonical) {
            if (!is_pattern(canonical, spec.pattern_masks)) {
                return;
            }
            prefix36_dynamic_push_board(
                arr2_state, dense_lut, z_luts, canonical, pending2.data(), pending_count2, chunks2);
        };
        auto flush_canonical1 = [&]() {
            if (canonical_count1 == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(
                canonical1.data(), canonical_count1, spec.symm_mode);
            for (uint32_t i = 0; i < canonical_count1; ++i) {
                push_ready1(canonical1[i]);
            }
            canonical_count1 = 0U;
        };
        auto flush_canonical2 = [&]() {
            if (canonical_count2 == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(
                canonical2.data(), canonical_count2, spec.symm_mode);
            for (uint32_t i = 0; i < canonical_count2; ++i) {
                push_ready2(canonical2[i]);
            }
            canonical_count2 = 0U;
        };
        auto push_canonical1 = [&](uint64_t moved) {
            canonical1[canonical_count1++] = moved;
            if (canonical_count1 == canonical1.size()) {
                flush_canonical1();
            }
        };
        auto push_canonical2 = [&](uint64_t moved) {
            canonical2[canonical_count2++] = moved;
            if (canonical_count2 == canonical2.size()) {
                flush_canonical2();
            }
        };

#pragma omp for schedule(dynamic, 16)
        for (int64_t bucket_idx_signed = 0;
             bucket_idx_signed < static_cast<int64_t>(current.bucket_keys.size());
             ++bucket_idx_signed) {
            if (arr1_state.overflowed.load(std::memory_order_acquire) ||
                arr2_state.overflowed.load(std::memory_order_acquire)) {
                continue;
            }
            const uint32_t bucket_idx = static_cast<uint32_t>(bucket_idx_signed);
            const uint64_t key = current.bucket_keys[bucket_idx];
            const uint64_t prefix36 = key_prefix36(key);
            const uint32_t group = sum_index(key_remaining_sum(key));
            const uint32_t valid_count = dense_lut.size_table[group];
            const uint32_t unrank_offset = dense_lut.offset_table[group];
            auto handle_board = [&](uint64_t board) {
                if (do_check && is_success_by_shifts(board, options.target, spec.success_shifts)) {
                    return;
                }
                uint32_t empty_mask = zero_cell_mask16(board);
                while (empty_mask != 0U) {
                    const uint32_t cell = countr_zero_u32(empty_mask);
                    empty_mask &= empty_mask - 1U;
                    const uint64_t spawn2 = board | (1ULL << (4U * cell));
                    const auto moves2 = Mover::move_all_dir(spawn2);
                    const uint64_t boards2[4] = {
                        std::get<0>(moves2), std::get<1>(moves2),
                        std::get<2>(moves2), std::get<3>(moves2)
                    };
                    for (uint64_t moved : boards2) {
                        if (moved != spawn2) {
                            push_canonical1(moved);
                        }
                    }
                    const uint64_t spawn4 = board | (2ULL << (4U * cell));
                    const auto moves4 = Mover::move_all_dir(spawn4);
                    const uint64_t boards4[4] = {
                        std::get<0>(moves4), std::get<1>(moves4),
                        std::get<2>(moves4), std::get<3>(moves4)
                    };
                    for (uint64_t moved : boards4) {
                        if (moved != spawn4) {
                            push_canonical2(moved);
                        }
                    }
                }
            };
            if (valid_count <= current.threshold_bits) {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = current.small_bitmap_bytes[offset + byte_idx];
                    while (value != 0U) {
                        const uint32_t bit = countr_zero_u32(value);
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        handle_board(
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank])
                        );
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t offset = current.bitmap_offsets[bucket_idx];
                const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = current.large_bitmap_words[offset + word_idx];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero_u64(value);
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        handle_board(
                            (prefix36 << kSuffixBits) |
                            static_cast<uint64_t>(dense_lut.unrank_array[unrank_offset + rank])
                        );
                        value &= value - 1ULL;
                    }
                }
            }
        }
        flush_canonical1();
        flush_canonical2();
        flush1();
        flush2();
    }
}

void generate_forward_layers(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options,
    const LutBundle &lut
) {
    if (all_layer_inputs_exist(options)) {
        return;
    }
    reset_generate_stats(options);
    const int num_threads = thread_count_from_options(options);
    const DTypeMode mode = dtype_mode_from_name(options.success_rate_dtype);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    const double total_t0 = now_seconds();

    const double init_t0 = now_seconds();
    const Prefix40Baseline::Luts prefix_luts = build_prefix_luts(lut.config, options);
    Prefix40Baseline::Layer p40 = Prefix40Baseline::build_layer_from_sorted_boards(arr_init, prefix_luts, num_threads);
    Prefix36Layer current = build_prefix36_metadata_from_prefix40_single_bucket_parallel(
        p40,
        prefix_luts,
        lut.dense_lut,
        num_threads,
        64U
    );
    const double init_t1 = now_seconds();
    const double init_write_t0 = now_seconds();
    write_generated_layer_file(options, 0, current, spec, mode, io_config);
    const double init_write_t1 = now_seconds();
    append_generate_stats(
        options, "init", 0, arr_init.size(), &current, nullptr, 0,
        init_write_t1 - init_t0,
        init_t1 - init_t0,
        0.0,
        init_t1 - init_t0,
        0.0,
        0.0,
        init_write_t1 - init_write_t0
    );

    Prefix36Layer carry_layer;
    bool has_carry = false;
    uint64_t total_live = current.live_board_count;
    double total_prepare = 0.0;
    double total_work = init_t1 - init_t0;
    double total_finalize = 0.0;
    double total_cleanup = 0.0;
    double total_write = init_write_t1 - init_write_t0;
    std::vector<double> reserve_need_history;
    double retry_guard_factor = 0.0;
    const uint32_t progress_total = classic_build_progress_total(options);

    for (int current_step = 0; current_step <= options.steps - 3; ++current_step) {
        FormationProgress::update_build_progress(static_cast<uint32_t>(current_step + 1), progress_total);
        double factor = reserve_factor_for_step(current_step, reserve_need_history, retry_guard_factor);
        retry_guard_factor = 0.0;
        uint32_t retry_count = 0U;
        double prepare_seconds = 0.0;
        double work_seconds = 0.0;
        double finalize_seconds = 0.0;
        double cleanup_seconds = 0.0;
        Prefix36Layer next_layer;
        Prefix36Layer terminal_next2;
        const bool terminal = current_step == options.steps - 3;
        for (;;) {
            bool built = false;
            double cleanup_t0 = 0.0;
            {
                const double prepare_t0 = now_seconds();
                Prefix36DynamicState arr1 =
                    make_dynamic_for_current(current.layer_sum + 2U, current.threshold_bits, current, factor);
                Prefix36DynamicState arr2 =
                    make_dynamic_for_current(current.layer_sum + 4U, current.threshold_bits, current, factor);
                const double prepare_t1 = now_seconds();
                prepare_seconds += prepare_t1 - prepare_t0;
                const double seed_t0 = now_seconds();
                if (has_carry) {
                    insert_layer_into_dynamic(carry_layer, arr1, lut.dense_lut, lut.row_luts, num_threads);
                }
                const double work_t0 = now_seconds();
                const bool do_check = current_step > options.docheck_step;
                if (options.is_variant) {
                    prefix36_dynamic_generate_into_production<VBoardMover>(
                        current, arr1, arr2, lut.dense_lut, lut.row_luts, spec, options, num_threads, do_check);
                } else {
                    prefix36_dynamic_generate_into_production<BoardMover>(
                        current, arr1, arr2, lut.dense_lut, lut.row_luts, spec, options, num_threads, do_check);
                }
                const double work_t1 = now_seconds();
                work_seconds += (work_t1 - seed_t0);
                if (arr1.overflowed.load(std::memory_order_acquire) ||
                    arr2.overflowed.load(std::memory_order_acquire)) {
                    factor *= 2.0;
                    ++retry_count;
                    if (retry_count > 6U) {
                        throw std::runtime_error("EX prefix36 dynamic generation exceeded retry limit");
                    }
                    cleanup_t0 = now_seconds();
                } else {
                    const double finalize_t0 = now_seconds();
                    next_layer = finalize_prefix36_dynamic_state(arr1, lut.dense_lut, num_threads);
                    if (terminal) {
                        terminal_next2 = finalize_prefix36_dynamic_state(arr2, lut.dense_lut, num_threads);
                        build_terminal_success(next_layer, lut.dense_lut, spec, options, num_threads);
                        build_terminal_success(terminal_next2, lut.dense_lut, spec, options, num_threads);
                        next_layer = compact_layer(next_layer, lut.dense_lut, 0U, num_threads);
                        terminal_next2 = compact_layer(terminal_next2, lut.dense_lut, 0U, num_threads);
                        has_carry = false;
                    } else {
                        carry_layer = finalize_prefix36_dynamic_state(arr2, lut.dense_lut, num_threads);
                        has_carry = true;
                    }
                    const double finalize_t1 = now_seconds();
                    finalize_seconds += finalize_t1 - finalize_t0;
                    cleanup_t0 = finalize_t1;
                    built = true;
                }

            }
            cleanup_seconds += now_seconds() - cleanup_t0;
            if (!built) {
                continue;
            }
            break;
        }

        const double write_t0 = now_seconds();
        write_generated_layer_file(options, current_step + 1, next_layer, spec, mode, io_config);
        if (terminal) {
            write_generated_layer_file(options, current_step + 2, terminal_next2, spec, mode, io_config);
        }
        const double write_t1 = now_seconds();

        const double layer_compute_seconds = prepare_seconds + work_seconds + finalize_seconds + cleanup_seconds;
        const double layer_write_seconds = write_t1 - write_t0;
        append_generate_stats(
            options,
            terminal ? "forward_terminal" : "forward",
            current_step + 1,
            current.live_board_count,
            &next_layer,
            terminal ? &terminal_next2 : nullptr,
            retry_count,
            layer_compute_seconds + layer_write_seconds,
            layer_compute_seconds,
            prepare_seconds,
            work_seconds,
            finalize_seconds,
            cleanup_seconds,
            layer_write_seconds
        );
        total_live += next_layer.live_board_count + (terminal ? terminal_next2.live_board_count : 0ULL);
        total_prepare += prepare_seconds;
        total_work += work_seconds;
        total_finalize += finalize_seconds;
        total_cleanup += cleanup_seconds;
        total_write += layer_write_seconds;

        double observed_need = transition_reserve_need(current, next_layer);
        if (terminal) {
            observed_need = std::max(observed_need, transition_reserve_need(current, terminal_next2));
        } else if (has_carry) {
            observed_need = std::max(observed_need, transition_reserve_need(current, carry_layer));
        }
        if (observed_need > 0.0) {
            reserve_need_history.push_back(observed_need);
        }
        if (retry_count != 0U) {
            retry_guard_factor = std::min(kDefaultReserveFactor, factor * kLearnedReserveRetryGuard);
        }

        current = std::move(next_layer);
    }

    const double total_t1 = now_seconds();
    append_generate_stats(
        options, "_total", -1, total_live, nullptr, nullptr, 0,
        total_t1 - total_t0,
        total_prepare + total_work + total_finalize + total_cleanup,
        total_prepare,
        total_work,
        total_finalize,
        total_cleanup,
        total_write
    );
}

void ensure_direct_index_built(Prefix36Layer &layer, const std::vector<uint32_t> &size_table) {
    if (layer.live_board_count == 0U) {
        return;
    }
    if (layer.direct_entry_index.table_size != 0U && !layer.direct_entry_index.entries.empty()) {
        return;
    }
    build_direct_index(layer, size_table);
}

SolveStepSummary solve_loaded_step_impl(
    const PatternSpec &spec,
    const RunOptions &options,
    const DenseLow24RankLut &dense_lut,
    const ZMaskFrozen::ZMaskLuts &z_luts,
    int step,
    Prefix36Layer &current,
    Prefix36Layer &future1,
    Prefix36Layer &future2,
    double read_seconds,
    double layer_deletion_threshold
) {
    const int num_threads = thread_count_from_options(options);
    const DTypeMode mode = dtype_mode_from_name(options.success_rate_dtype);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    const double total_t0 = now_seconds();
    const double index_t0 = now_seconds();
    const bool futures_empty = future1.live_board_count == 0U && future2.live_board_count == 0U;
    if (!futures_empty) {
        ensure_direct_index_built(future1, dense_lut.size_table);
        ensure_direct_index_built(future2, dense_lut.size_table);
    }
    const double index_t1 = now_seconds();

    const uint64_t input_live = current.live_board_count;
    const double recalc_t0 = now_seconds();
    const bool do_check = step > options.docheck_step;
    const bool all_zero_output = futures_empty && !do_check;
    RecalcStats recalc;
    if (all_zero_output) {
        current.success_values.clear();
        recalc.boards = current.live_board_count;
    } else if (futures_empty) {
        current.success_values.assign(static_cast<size_t>(current.live_board_count), 0U);
        if (do_check) {
            build_terminal_success(current, dense_lut, spec, options, num_threads);
        }
        recalc.boards = current.live_board_count;
    } else {
        if (options.is_variant) {
            recalc = recalculate_current_layer<VBoardMover>(
                current,
                future1,
                future2,
                dense_lut,
                z_luts,
                spec,
                options,
                do_check,
                num_threads
            );
        } else {
            recalc = recalculate_current_layer<BoardMover>(
                current,
                future1,
                future2,
                dense_lut,
                z_luts,
                spec,
                options,
                do_check,
                num_threads
            );
        }
    }
    const double recalc_t1 = now_seconds();
    recalc.seconds = recalc_t1 - recalc_t0;
    recalc.mbps = throughput(recalc.boards, recalc.seconds);
    const double compact_t0 = now_seconds();
    if (all_zero_output) {
        Prefix36Layer empty;
        empty.layer_sum = current.layer_sum;
        empty.threshold_bits = current.threshold_bits;
        current = std::move(empty);
    } else {
        current = compact_layer(current, dense_lut, 0U, num_threads);
    }
    const double compact_t1 = now_seconds();
    const double write_t0 = now_seconds();
    write_layer_file(layer_file_path(options.pathname, step), current, spec, mode, io_config);
    remove_generated_layer_input(options.pathname, step);
    compress_layer_result_from_memory(options, step, current, spec, mode);
    const double write_t1 = now_seconds();

    double future_compact_seconds = 0.0;
    double future_write_seconds = 0.0;
    const uint64_t future2_pre_threshold_live = future2.live_board_count;
    uint64_t future2_post_live = future2.live_board_count;
    if (layer_deletion_threshold > 0.0) {
        const uint32_t threshold = static_cast<uint32_t>(
            layer_deletion_threshold *
            static_cast<double>(max_scale_value_for_dtype<uint32_t>(options.success_rate_dtype))
        );
        const double fc_t0 = now_seconds();
        future2 = compact_layer(future2, dense_lut, threshold, num_threads);
        const double fc_t1 = now_seconds();
        const double fw_t0 = now_seconds();
        write_layer_file(layer_file_path(options.pathname, step + 2), future2, spec, mode, io_config);
        remove_generated_layer_input(options.pathname, step + 2);
        compress_layer_result_from_memory(options, step + 2, future2, spec, mode);
        const double fw_t1 = now_seconds();
        future_compact_seconds = fc_t1 - fc_t0;
        future_write_seconds = fw_t1 - fw_t0;
        future2_post_live = future2.live_board_count;
    }
    const double current_retained_ratio =
        RuntimeControls::retention_ratio(current.live_board_count, input_live);
    const double future_threshold_retained_ratio =
        RuntimeControls::retention_ratio(future2_post_live, future2_pre_threshold_live);
    const double total_t1 = now_seconds();
    const double compute_seconds =
        (recalc_t1 - recalc_t0) + (compact_t1 - compact_t0) + (index_t1 - index_t0) + future_compact_seconds;
    append_solve_stats(
        options,
        "solve",
        step,
        input_live,
        current.live_board_count,
        future1.live_board_count,
        future2.live_board_count,
        future2_post_live,
        layer_deletion_threshold,
        current_retained_ratio,
        future_threshold_retained_ratio,
        recalc.seconds,
        compact_t1 - compact_t0,
        index_t1 - index_t0,
        future_compact_seconds,
        write_t1 - write_t0,
        future_write_seconds,
        read_seconds,
        (total_t1 - total_t0) + read_seconds,
        compute_seconds,
        &current
    );
    SolveStepSummary summary;
    summary.input_live = input_live;
    summary.output_live = current.live_board_count;
    summary.recalc_seconds = recalc.seconds;
    summary.compact_seconds = compact_t1 - compact_t0;
    summary.future_index_seconds = index_t1 - index_t0;
    summary.future_compact_seconds = future_compact_seconds;
    summary.current_write_seconds = write_t1 - write_t0;
    summary.future_write_seconds = future_write_seconds;
    summary.read_seconds = read_seconds;
    summary.total_seconds = (total_t1 - total_t0) + read_seconds;
    summary.compute_seconds = compute_seconds;
    return summary;
}

template <typename Mover>
SolveStepSummary keep_only_optimal_branches_prefix36_impl(
    const PatternSpec &spec,
    const RunOptions &options,
    const LutBundle &lut
) {
    SolveStepSummary total;
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_total = build_progress_total(options);
    if (options.steps <= kOptimalBranchOnlyStartStep) {
        for (int step = 0; step < options.steps; ++step) {
            FormationProgress::update_build_progress(
                solve_progress_total + static_cast<uint32_t>(step) + 1U,
                progress_total
            );
        }
        write_optimal_complete_marker(options);
        return total;
    }

    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    const int num_threads = thread_count_from_options(options);
    const DTypeMode mode = dtype_mode_from_name(options.success_rate_dtype);
    int last_done = std::max(kOptimalBranchOnlyStartStep - 1, read_optimal_layer_marker(options));
    if (last_done >= options.steps - 1) {
        if (options.compress) {
            for (int step = 0; step < options.steps; ++step) {
                const double seconds = compress_layer_result_from_file(options, step);
                total.current_write_seconds += seconds;
                total.total_seconds += seconds;
            }
        }
        write_optimal_complete_marker(options);
        return total;
    }

    Prefix36Layer prev2;
    Prefix36Layer prev1;
    int prev2_step = std::numeric_limits<int>::min();
    int prev1_step = std::numeric_limits<int>::min();

    for (int step = 0; step < options.steps; ++step) {
        FormationProgress::update_build_progress(
            solve_progress_total + static_cast<uint32_t>(step) + 1U,
            progress_total
        );
        if (step < kOptimalBranchOnlyStartStep || step <= last_done) {
            continue;
        }

        const double total_t0 = now_seconds();
        const double read_t0 = now_seconds();
        if (prev2_step != step - 2) {
            prev2 = read_layer_input(
                options.pathname,
                step - 2,
                io_config,
                lut.dense_lut,
                num_threads,
                &lut
            );
            prev2_step = step - 2;
        }
        if (prev1_step != step - 1) {
            prev1 = read_layer_input(
                options.pathname,
                step - 1,
                io_config,
                lut.dense_lut,
                num_threads,
                &lut
            );
            prev1_step = step - 1;
        }
        Prefix36Layer target = read_layer_input(
            options.pathname,
            step,
            io_config,
            lut.dense_lut,
            num_threads,
            &lut
        );
        const double read_seconds = now_seconds() - read_t0;
        const uint64_t target_live_before = target.live_board_count;

        const double index_t0 = now_seconds();
        ensure_direct_index_built(target, lut.dense_lut.size_table);
        const double index_seconds = now_seconds() - index_t0;

        std::vector<std::atomic<uint64_t>> keep_words(
            static_cast<size_t>((target.success_values.size() + 63U) >> 6U)
        );
        for (std::atomic<uint64_t> &word : keep_words) {
            word.store(0ULL, std::memory_order_relaxed);
        }

        const double mark_t0 = now_seconds();
        OptimalBranchStats from_prev2 = mark_optimal_branches_from_source<Mover>(
            prev2,
            target,
            lut.dense_lut,
            lut.row_luts,
            spec,
            2U,
            keep_words,
            num_threads
        );
        OptimalBranchStats from_prev1 = mark_optimal_branches_from_source<Mover>(
            prev1,
            target,
            lut.dense_lut,
            lut.row_luts,
            spec,
            1U,
            keep_words,
            num_threads
        );
        const double mark_seconds = now_seconds() - mark_t0;

        std::vector<uint64_t> keep_bits(keep_words.size(), 0ULL);
        for (size_t i = 0; i < keep_words.size(); ++i) {
            keep_bits[i] = keep_words[i].load(std::memory_order_relaxed);
        }

        const double compact_t0 = now_seconds();
        Prefix36Layer pruned = compact_layer_by_keep_bits(target, lut.dense_lut, keep_bits, num_threads);
        const double compact_seconds = now_seconds() - compact_t0;

        const double write_t0 = now_seconds();
        write_layer_file(layer_file_path(options.pathname, step), pruned, spec, mode, io_config);
        remove_generated_layer_input(options.pathname, step);
        const double write_seconds = now_seconds() - write_t0;
        write_optimal_layer_marker(options, step);

        double compress_seconds = 0.0;
        if (options.compress && step - 2 >= 0) {
            compress_seconds += compress_layer_result_from_memory(options, step - 2, prev2, spec, mode);
            std::error_code ec;
            fs::remove(layer_file_path(options.pathname, step - 2), ec);
        }

        const double total_seconds = now_seconds() - total_t0;
        const double compute_seconds = index_seconds + mark_seconds + compact_seconds;
        append_solve_stats(
            options,
            "optimal_branch",
            step,
            prev2.live_board_count + prev1.live_board_count,
            pruned.live_board_count,
            prev1.live_board_count,
            target_live_before,
            target_live_before,
            0.0,
            RuntimeControls::retention_ratio(pruned.live_board_count, target_live_before),
            0.0,
            mark_seconds,
            compact_seconds,
            index_seconds,
            0.0,
            write_seconds + compress_seconds,
            0.0,
            read_seconds,
            total_seconds,
            compute_seconds,
            &pruned
        );

        total.input_live += target_live_before;
        total.output_live += pruned.live_board_count;
        total.recalc_seconds += mark_seconds;
        total.compact_seconds += compact_seconds;
        total.future_index_seconds += index_seconds;
        total.current_write_seconds += write_seconds + compress_seconds;
        total.read_seconds += read_seconds;
        total.total_seconds += total_seconds;
        total.compute_seconds += compute_seconds;

        (void)from_prev2;
        (void)from_prev1;
        prev2 = std::move(prev1);
        prev2_step = step - 1;
        prev1 = std::move(pruned);
        prev1_step = step;
        last_done = step;
    }

    if (options.compress) {
        for (int step = 0; step < options.steps; ++step) {
            const double seconds = compress_layer_result_from_file(options, step);
            total.current_write_seconds += seconds;
            total.total_seconds += seconds;
        }
    }
    write_optimal_complete_marker(options);
    return total;
}

SolveStepSummary keep_only_optimal_branches_prefix36(
    const PatternSpec &spec,
    const RunOptions &options,
    const LutBundle &lut
) {
    return options.is_variant
        ? keep_only_optimal_branches_prefix36_impl<VBoardMover>(spec, options, lut)
        : keep_only_optimal_branches_prefix36_impl<BoardMover>(spec, options, lut);
}

SolveStepSummary solve_single_step_impl(
    const PatternSpec &spec,
    const RunOptions &options,
    const LutBundle &lut,
    int step
) {
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    const int num_threads = thread_count_from_options(options);
    const double read_t0 = now_seconds();
    Prefix36Layer future1 = read_layer_input(
        options.pathname, step + 1, io_config, lut.dense_lut, num_threads, &lut);
    Prefix36Layer future2 = read_layer_input(
        options.pathname, step + 2, io_config, lut.dense_lut, num_threads, &lut);
    Prefix36Layer current = read_layer_input(
        options.pathname, step, io_config, lut.dense_lut, num_threads, &lut);
    const double read_t1 = now_seconds();
    return solve_loaded_step_impl(
        spec,
        options,
        lut.dense_lut,
        lut.row_luts,
        step,
        current,
        future1,
        future2,
        read_t1 - read_t0,
        RuntimeControls::current_deletion_threshold(options)
    );
}

} // namespace

std::string layer_file_path(const std::string &pathname, int step) {
    return pathname + std::to_string(step) + kLayerFileExtension;
}

void run_pattern_build(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    if (compressed_results_are_complete_for_options(options)) {
        return;
    }
    if (options.optimal_branch_only && all_compressed_layers_exist(options)) {
        throw std::runtime_error(
            "EX optimal_branch_only requested, but compressed EX results lack ex_optimal_complete marker; rebuild the table"
        );
    }
    const LutBundle lut = load_or_build_prefix36_lut(arr_init, spec, options);
    generate_forward_layers(arr_init, spec, options, lut);
    run_pattern_solve(arr_init, spec, options);
}

void run_pattern_solve(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    if (compressed_results_are_complete_for_options(options)) {
        return;
    }
    if (options.optimal_branch_only && all_compressed_layers_exist(options)) {
        throw std::runtime_error(
            "EX optimal_branch_only requested, but compressed EX results lack ex_optimal_complete marker; rebuild the table"
        );
    }
    const LutBundle lut = load_or_build_prefix36_lut(arr_init, spec, options);

    if (options.optimal_branch_only && !optimal_complete_marker_exists(options)) {
        const int last_done = read_optimal_layer_marker(options);
        if (optimal_resume_inputs_exist(options, last_done)) {
            SolveStepSummary optimal_summary = keep_only_optimal_branches_prefix36(spec, options, lut);
            append_solve_stats(
                options,
                "_total",
                -1,
                optimal_summary.input_live,
                optimal_summary.output_live,
                0,
                0,
                0,
                RuntimeControls::current_deletion_threshold(options),
                RuntimeControls::retention_ratio(optimal_summary.output_live, optimal_summary.input_live),
                0.0,
                optimal_summary.recalc_seconds,
                optimal_summary.compact_seconds,
                optimal_summary.future_index_seconds,
                0.0,
                optimal_summary.current_write_seconds,
                0.0,
                optimal_summary.read_seconds,
                optimal_summary.total_seconds,
                optimal_summary.compute_seconds,
                nullptr
            );
            return;
        }
    }

    generate_forward_layers(arr_init, spec, options, lut);
    reset_solve_stats(options);
    if (options.optimal_branch_only) {
        std::error_code ec;
        fs::remove(optimal_layer_marker_path(options.pathname), ec);
        fs::remove(optimal_complete_marker_path(options.pathname), ec);
    }
    SolveStepSummary total;
    const int first_step = options.steps - 3;
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    const int num_threads = thread_count_from_options(options);
    const DTypeMode mode = dtype_mode_from_name(options.success_rate_dtype);
    promote_generated_layer_input(options.pathname, first_step + 1, spec, mode, io_config, lut, num_threads);
    promote_generated_layer_input(options.pathname, first_step + 2, spec, mode, io_config, lut, num_threads);
    const double initial_read_t0 = now_seconds();
    Prefix36Layer future1 = read_layer_input(
        options.pathname, first_step + 1, io_config, lut.dense_lut, num_threads, &lut);
    Prefix36Layer future2 = read_layer_input(
        options.pathname, first_step + 2, io_config, lut.dense_lut, num_threads, &lut);
    double carried_read_seconds = now_seconds() - initial_read_t0;
    double deletion_threshold_state = RuntimeControls::current_deletion_threshold(options);
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_base = build_progress_total(options);
    for (int step = first_step; step >= 0; --step) {
        FormationProgress::update_build_progress(
            solve_progress_base - static_cast<uint32_t>(step) - 2U,
            progress_total
        );
        const double read_t0 = now_seconds();
        Prefix36Layer current = read_layer_input(
            options.pathname, step, io_config, lut.dense_lut, num_threads, &lut);
        const double read_seconds = carried_read_seconds + (now_seconds() - read_t0);
        carried_read_seconds = 0.0;
        deletion_threshold_state = RuntimeControls::refresh_deletion_threshold(options, deletion_threshold_state);
        const SolveStepSummary step_summary = solve_loaded_step_impl(
            spec,
            options,
            lut.dense_lut,
            lut.row_luts,
            step,
            current,
            future1,
            future2,
            read_seconds,
            deletion_threshold_state
        );
        total.input_live += step_summary.input_live;
        total.output_live += step_summary.output_live;
        total.recalc_seconds += step_summary.recalc_seconds;
        total.compact_seconds += step_summary.compact_seconds;
        total.future_index_seconds += step_summary.future_index_seconds;
        total.future_compact_seconds += step_summary.future_compact_seconds;
        total.current_write_seconds += step_summary.current_write_seconds;
        total.future_write_seconds += step_summary.future_write_seconds;
        total.read_seconds += step_summary.read_seconds;
        total.total_seconds += step_summary.total_seconds;
        total.compute_seconds += step_summary.compute_seconds;
        future2 = std::move(future1);
        future1 = std::move(current);
    }
    if (options.optimal_branch_only) {
        SolveStepSummary optimal_summary = keep_only_optimal_branches_prefix36(spec, options, lut);
        total.input_live += optimal_summary.input_live;
        total.output_live += optimal_summary.output_live;
        total.recalc_seconds += optimal_summary.recalc_seconds;
        total.compact_seconds += optimal_summary.compact_seconds;
        total.future_index_seconds += optimal_summary.future_index_seconds;
        total.current_write_seconds += optimal_summary.current_write_seconds;
        total.read_seconds += optimal_summary.read_seconds;
        total.total_seconds += optimal_summary.total_seconds;
        total.compute_seconds += optimal_summary.compute_seconds;
    } else {
        const double compress_seconds = compress_all_layer_results(options);
        total.total_seconds += compress_seconds;
        total.current_write_seconds += compress_seconds;
    }
    append_solve_stats(
        options,
        "_total",
        -1,
        total.input_live,
        total.output_live,
        0,
        0,
        0,
        deletion_threshold_state,
        RuntimeControls::retention_ratio(total.output_live, total.input_live),
        0.0,
        total.recalc_seconds,
        total.compact_seconds,
        total.future_index_seconds,
        total.future_compact_seconds,
        total.current_write_seconds,
        total.future_write_seconds,
        total.read_seconds,
        total.total_seconds,
        total.compute_seconds,
        nullptr
    );
}

void run_pattern_solve_single_layer(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options,
    int step
) {
    const LutBundle lut = load_or_build_prefix36_lut(arr_init, spec, options);
    (void)solve_single_step_impl(spec, options, lut, step);
}

uint32_t find_bucket_in_layer_file(
    std::ifstream &in,
    const std::string &path,
    const LayerFileLayout &layout,
    uint64_t bucket_count,
    uint64_t target_key
) {
    uint64_t low = 0U;
    uint64_t high = bucket_count;
    while (low < high) {
        const uint64_t mid = low + ((high - low) >> 1U);
        const uint64_t key = read_one_at<uint64_t>(
            in,
            layout.bucket_keys_offset + mid * sizeof(uint64_t),
            path
        );
        if (key < target_key) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low >= bucket_count || low > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<uint32_t>::max();
    }
    const uint64_t key = read_one_at<uint64_t>(
        in,
        layout.bucket_keys_offset + low * sizeof(uint64_t),
        path
    );
    return key == target_key
        ? static_cast<uint32_t>(low)
        : std::numeric_limits<uint32_t>::max();
}

uint32_t find_bucket_by_success_index(
    std::ifstream &in,
    const std::string &path,
    const LayerFileLayout &layout,
    uint64_t bucket_count,
    uint64_t success_index
) {
    uint64_t low = 0U;
    uint64_t high = bucket_count;
    while (low < high) {
        const uint64_t mid = low + ((high - low) >> 1U);
        const uint32_t success_offset = read_one_at<uint32_t>(
            in,
            layout.success_offsets_offset + mid * sizeof(uint32_t),
            path
        );
        if (static_cast<uint64_t>(success_offset) <= success_index) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low == 0U || low > bucket_count || low > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(low - 1U);
}

bool small_bitmap_has_bit(const uint8_t *bitmap, uint32_t rank) {
    return (bitmap[rank >> 3U] & static_cast<uint8_t>(1U << (rank & 7U))) != 0U;
}

uint32_t dense_ordinal_small_raw(const uint8_t *bitmap, uint32_t rank) {
    const uint32_t byte_idx = rank >> 3U;
    const uint32_t bit_idx = rank & 7U;
    uint32_t total = 0U;
    for (uint32_t i = 0U; i < byte_idx; ++i) {
        total += popcount_u32(bitmap[i]);
    }
    if (bit_idx != 0U) {
        total += popcount_u32(bitmap[byte_idx] & static_cast<uint8_t>((1U << bit_idx) - 1U));
    }
    return total;
}

uint64_t read_success_raw_value(
    std::ifstream &in,
    const std::string &path,
    const LayerFileLayout &layout,
    uint64_t success_index,
    uint64_t value_size
) {
    const uint64_t offset = layout.success_values_offset + success_index * value_size;
    if (value_size == sizeof(uint32_t)) {
        return read_one_at<uint32_t>(in, offset, path);
    }
    if (value_size == sizeof(uint64_t)) {
        return read_one_at<uint64_t>(in, offset, path);
    }
    throw std::runtime_error("invalid EX prefix36 success value size: " + path);
}

EXCompressedResult::ColdLookupResult lookup_zbook_cold(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t board
) {
    std::ifstream in(zbook_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EX prefix36 zbook: " + zbook_path);
    }
    LayerFileHeader header{};
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in) {
        throw std::runtime_error("failed to read EX prefix36 zbook header: " + zbook_path);
    }
    validate_layer_header(header, zbook_path);
    const uint64_t expected_bytes = layer_file_bytes(header);
    if (fs::exists(zbook_path) && fs::file_size(zbook_path) < expected_bytes) {
        throw std::runtime_error("truncated EX prefix36 zbook: " + zbook_path);
    }
    const DTypeMode mode = dtype_mode_from_header(header);
    EXCompressedResult::ColdLookupResult result;
    result.success_kind = storage_kind_for_dtype_mode(mode);
    {
        std::ifstream lut_in(zlut_path, std::ios::binary);
        if (!lut_in) {
            throw std::runtime_error("failed to open EX prefix36 LUT: " + zlut_path);
        }
        const LutFileHeader lut_header = read_lut_header_for_cold_lookup(lut_in, zlut_path);
        if (!physical_metadata_matches(header, lut_header)) {
            throw std::runtime_error("EX prefix36 physical pattern metadata does not match zbook");
        }
    }
    if (board_tile_sum(board) != header.layer_sum ||
        header.bucket_count == 0U ||
        header.success_value_count == 0U) {
        return result;
    }

    const ColdFileQuery query = prepare_cold_query_from_lut_file(zlut_path, board);
    if (!query.valid) {
        return result;
    }
    const LayerFileLayout layout = layer_file_layout(header);
    const uint32_t bucket_index = find_bucket_in_layer_file(
        in,
        zbook_path,
        layout,
        header.bucket_count,
        query.target_key
    );
    if (bucket_index == std::numeric_limits<uint32_t>::max()) {
        return result;
    }

    const uint32_t bitmap_offset = read_one_at<uint32_t>(
        in,
        layout.bitmap_offsets_offset + static_cast<uint64_t>(bucket_index) * sizeof(uint32_t),
        zbook_path
    );
    const uint32_t success_offset = read_one_at<uint32_t>(
        in,
        layout.success_offsets_offset + static_cast<uint64_t>(bucket_index) * sizeof(uint32_t),
        zbook_path
    );

    uint64_t success_index = 0U;
    if (query.valid_count <= header.threshold_bits) {
        const uint64_t bitmap_bytes = bytes_for_bits(query.valid_count);
        if (static_cast<uint64_t>(bitmap_offset) + bitmap_bytes > header.small_bitmap_bytes) {
            throw std::runtime_error("EX prefix36 small bitmap offset out of range: " + zbook_path);
        }
        std::vector<uint8_t> bitmap = read_bytes_at(
            in,
            layout.small_bitmap_offset + bitmap_offset,
            bitmap_bytes,
            zbook_path
        );
        result.bucket_block_raw_bytes = bitmap_bytes;
        if (!small_bitmap_has_bit(bitmap.data(), query.rank)) {
            return result;
        }
        success_index = static_cast<uint64_t>(success_offset) +
            dense_ordinal_small_raw(bitmap.data(), query.rank);
    } else {
        const uint32_t word_idx = query.rank >> 6U;
        const uint32_t bit_idx = query.rank & 63U;
        const uint64_t bucket_words = words_for_bits(query.valid_count);
        if (static_cast<uint64_t>(bitmap_offset) + bucket_words > header.large_bitmap_words) {
            throw std::runtime_error("EX prefix36 large bitmap offset out of range: " + zbook_path);
        }
        constexpr uint32_t block_start = 0U;
        const uint32_t read_words = word_idx + 1U;
        constexpr uint32_t rank_base = 0U;
        std::vector<uint8_t> raw_words = read_bytes_at(
            in,
            layout.large_bitmap_offset +
                (static_cast<uint64_t>(bitmap_offset) + block_start) * sizeof(uint64_t),
            static_cast<uint64_t>(read_words) * sizeof(uint64_t),
            zbook_path
        );
        result.bucket_block_raw_bytes = static_cast<uint64_t>(read_words) * sizeof(uint64_t);
        uint32_t ordinal = rank_base;
        for (uint32_t i = 0U; i + 1U < read_words; ++i) {
            uint64_t value = 0ULL;
            std::memcpy(&value, raw_words.data() + static_cast<size_t>(i) * sizeof(uint64_t), sizeof(uint64_t));
            ordinal += popcount_u64(value);
        }
        uint64_t target_word = 0ULL;
        std::memcpy(
            &target_word,
            raw_words.data() + static_cast<size_t>(read_words - 1U) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
        if ((target_word & (1ULL << bit_idx)) == 0ULL) {
            return result;
        }
        if (bit_idx != 0U) {
            ordinal += popcount_u64(target_word & ((1ULL << bit_idx) - 1ULL));
        }
        success_index = static_cast<uint64_t>(success_offset) + ordinal;
    }
    if (success_index >= header.success_value_count) {
        throw std::runtime_error("EX prefix36 cold lookup produced out-of-range success index: " + zbook_path);
    }

    const uint64_t raw_value = read_success_raw_value(
        in,
        zbook_path,
        layout,
        success_index,
        header.value_size
    );
    const uint32_t fixed = fixed_from_raw_bits(raw_value, mode);
    result.found = true;
    result.global_dense_index = success_index;
    result.raw_value_bits = raw_bits_from_fixed(fixed, mode);
    result.numeric_value = numeric_from_fixed(fixed, mode);
    result.success_block_raw_bytes = header.value_size;
    return result;
}

bool sample_zbook_state(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value
) {
    std::ifstream layer_in(zbook_path, std::ios::binary);
    std::ifstream lut_in(zlut_path, std::ios::binary);
    if (!layer_in || !lut_in) {
        return false;
    }
    LayerFileHeader header{};
    layer_in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!layer_in) {
        return false;
    }
    try {
        validate_layer_header(header, zbook_path);
    } catch (...) {
        return false;
    }
    if (header.bucket_count == 0U || header.success_value_count == 0U) {
        return false;
    }
    const DTypeMode mode = dtype_mode_from_header(header);
    LutFileHeader lut_header{};
    try {
        lut_header = read_lut_header_for_cold_lookup(lut_in, zlut_path);
    } catch (...) {
        return false;
    }
    if (!physical_metadata_matches(header, lut_header)) {
        return false;
    }
    const LayerFileLayout layout = layer_file_layout(header);
    const LutFileLayout lut_layout = lut_file_layout(lut_header);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> success_pick(0U, header.success_value_count - 1U);
    constexpr uint32_t kSampleAttempts = 128U;
    for (uint32_t attempt = 0U; attempt < kSampleAttempts; ++attempt) {
        const uint64_t target_success_index = success_pick(rng);
        const uint32_t bucket_idx = find_bucket_by_success_index(
            layer_in,
            zbook_path,
            layout,
            header.bucket_count,
            target_success_index
        );
        if (bucket_idx == std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        const uint64_t key = read_one_at<uint64_t>(
            layer_in,
            layout.bucket_keys_offset + static_cast<uint64_t>(bucket_idx) * sizeof(uint64_t),
            zbook_path
        );
        const uint64_t prefix36 = key_prefix36(key);
        const uint32_t group = sum_index(key_remaining_sum(key));
        if (group >= lut_header.size_table_values || group >= lut_header.offset_table_values) {
            continue;
        }
        const uint32_t valid_count = read_one_at<uint32_t>(
            lut_in,
            lut_layout.size_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            zlut_path
        );
        const uint32_t unrank_offset = read_one_at<uint32_t>(
            lut_in,
            lut_layout.offset_table_offset + static_cast<uint64_t>(group) * sizeof(uint32_t),
            zlut_path
        );
        const uint32_t success_base = read_one_at<uint32_t>(
            layer_in,
            layout.success_offsets_offset + static_cast<uint64_t>(bucket_idx) * sizeof(uint32_t),
            zbook_path
        );
        if (target_success_index < static_cast<uint64_t>(success_base)) {
            continue;
        }
        const uint32_t bitmap_offset = read_one_at<uint32_t>(
            layer_in,
            layout.bitmap_offsets_offset + static_cast<uint64_t>(bucket_idx) * sizeof(uint32_t),
            zbook_path
        );
        uint64_t ordinal = target_success_index - static_cast<uint64_t>(success_base);
        uint32_t rank = std::numeric_limits<uint32_t>::max();
        if (valid_count <= header.threshold_bits) {
            const uint32_t bytes = static_cast<uint32_t>(bytes_for_bits(valid_count));
            for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                uint8_t value = read_one_at<uint8_t>(
                    layer_in,
                    layout.small_bitmap_offset + bitmap_offset + byte_idx,
                    zbook_path
                );
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
                    const uint32_t bit = countr_zero_u32(value);
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
        } else {
            const uint32_t words = static_cast<uint32_t>(words_for_bits(valid_count));
            for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                uint64_t value = read_one_at<uint64_t>(
                    layer_in,
                    layout.large_bitmap_offset +
                        (static_cast<uint64_t>(bitmap_offset) + word_idx) * sizeof(uint64_t),
                    zbook_path
                );
                if (word_idx + 1U == words && (valid_count & 63U) != 0U) {
                    value &= (1ULL << (valid_count & 63U)) - 1ULL;
                }
                const uint32_t count = popcount_u64(value);
                if (ordinal >= count) {
                    ordinal -= count;
                    continue;
                }
                while (value != 0ULL) {
                    const uint32_t bit = countr_zero_u64(value);
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
        }
        if (rank == std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(unrank_offset) + rank >= lut_header.unrank_array_values) {
            continue;
        }
        const uint32_t suffix28 = read_one_at<uint32_t>(
            lut_in,
            lut_layout.unrank_array_offset +
                (static_cast<uint64_t>(unrank_offset) + rank) * sizeof(uint32_t),
            zlut_path
        );
        const uint64_t raw = read_success_raw_value(
            layer_in,
            zbook_path,
            layout,
            target_success_index,
            header.value_size
        );
        const uint32_t fixed = fixed_from_raw_bits(raw, mode);
        board = (prefix36 << kSuffixBits) | suffix28;
        raw_value_bits = raw_bits_from_fixed(fixed, mode);
        numeric_value = numeric_from_fixed(fixed, mode);
        return true;
    }
    return false;
}

} // namespace EXPrefix36Runtime
