#include "EXADIO.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace EXAD {

namespace {

namespace fs = std::filesystem;

struct FileHeader {
    char magic[8];
    uint32_t version = 1;
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t slot_count = static_cast<uint32_t>(bucket_slot_count());
    uint64_t lut_signature = 0;
    uint64_t live_board_count = 0;
};

struct SlotHeader {
    uint64_t bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
};

struct LutFileHeader {
    char magic[8];
    uint32_t version = 1;
    uint32_t valid_mask_count = 0;
    uint64_t config_signature = 0;
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

constexpr char kMagic[8] = {'E', 'X', 'A', 'D', '7', 'T', 'M', 'P'};
constexpr char kLutMagic[8] = {'E', 'X', 'A', 'D', '7', 'L', 'U', 'T'};

FileHeader make_header(const Layer &layer) {
    FileHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.original_board_sum = layer.original_board_sum;
    header.threshold_bits = layer.threshold_bits;
    header.lut_signature = layer.lut_signature;
    header.live_board_count = layer.live_board_count;
    return header;
}

void validate_header(const FileHeader &header) {
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("invalid EXAD temp layer file magic");
    }
    if (header.version != 1U || header.slot_count != static_cast<uint32_t>(bucket_slot_count())) {
        throw std::runtime_error("unsupported EXAD temp layer version");
    }
}

uint64_t serialized_size(const Layer &layer) {
    uint64_t size = sizeof(FileHeader) + sizeof(SlotHeader) * bucket_slot_count();
    for (const BoardSet &set : layer.sets) {
        size += static_cast<uint64_t>(set.buckets.size()) * sizeof(BucketEntry);
        size += static_cast<uint64_t>(set.small_bitmap_bytes.size());
        size += static_cast<uint64_t>(set.large_bitmap_words.size()) * sizeof(uint64_t);
    }
    return size;
}

uint64_t file_size_or_throw(const std::string &path, const char *kind) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(std::string("failed to determine EXAD ") + kind + " file size: " + path);
    }
    return static_cast<uint64_t>(size);
}

void read_direct_exact(
    FileIOUtils::DirectSequentialReader &in,
    void *dst,
    size_t bytes,
    const std::string &
) {
    if (bytes != 0U) {
        in.read(dst, bytes);
    }
}

void read_slot_payload(
    FileIOUtils::DirectSequentialReader &in,
    BoardSet &set,
    const SlotHeader &slot,
    const std::string &path
) {
    set.buckets.resize(slot.bucket_count);
    set.small_bitmap_bytes.resize(slot.small_bitmap_bytes);
    set.large_bitmap_words.resize(slot.large_bitmap_words);
    set.live_board_count = slot.live_board_count;
    set.exact_bitmap_bits = slot.exact_bitmap_bits;
    set.aligned_bitmap_bits = slot.aligned_bitmap_bits;
    if (slot.bucket_count != 0U) {
        read_direct_exact(in, set.buckets.data(), set.buckets.size() * sizeof(BucketEntry), path);
    }
    if (!set.small_bitmap_bytes.empty()) {
        read_direct_exact(in, set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size(), path);
    }
    if (!set.large_bitmap_words.empty()) {
        read_direct_exact(in, set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t), path);
    }
}

} // namespace

std::string layer_file_path(const std::string &pathname, int step) {
    return pathname + std::to_string(step) + kLayerFileExtension;
}

std::string lut_file_path(const std::string &pathname) {
    return pathname + kLutFileExtension;
}

bool layer_file_exists(const std::string &path) {
    return fs::exists(path);
}

void remove_layer_file(const std::string &path) {
    std::error_code ec;
    fs::remove(path, ec);
}

class LayerSlotReader::Impl {
public:
    Impl(const std::string &path, FileIOUtils::DirectIoConfig config)
        : path_(path),
          in_(path, file_size_or_throw(path, "temp layer"), config) {
        FileHeader header{};
        read_direct_exact(in_, &header, sizeof(header), path_);
        validate_header(header);
        info_.original_board_sum = header.original_board_sum;
        info_.threshold_bits = header.threshold_bits;
        info_.lut_signature = header.lut_signature;
        info_.live_board_count = header.live_board_count;
        read_direct_exact(in_, slots_.data(), slots_.size() * sizeof(SlotHeader), path_);
    }

    const LayerFileInfo &info() const {
        return info_;
    }

    size_t next_slot() const {
        return next_slot_;
    }

    bool read_next(BoardSet &set) {
        if (next_slot_ >= slots_.size()) {
            return false;
        }
        set = BoardSet{};
        set.threshold_bits = info_.threshold_bits;
        read_slot_payload(in_, set, slots_[next_slot_], path_);
        ++next_slot_;
        return true;
    }

    void close() {
        in_.close();
    }

private:
    std::string path_;
    FileIOUtils::DirectSequentialReader in_;
    std::array<SlotHeader, bucket_slot_count()> slots_{};
    LayerFileInfo info_{};
    size_t next_slot_ = 0;
};

LayerSlotReader::LayerSlotReader(const std::string &path, FileIOUtils::DirectIoConfig config)
    : impl_(std::make_unique<Impl>(path, config)) {}

LayerSlotReader::~LayerSlotReader() = default;
LayerSlotReader::LayerSlotReader(LayerSlotReader &&) noexcept = default;
LayerSlotReader &LayerSlotReader::operator=(LayerSlotReader &&) noexcept = default;

const LayerFileInfo &LayerSlotReader::info() const {
    if (!impl_) {
        throw std::runtime_error("EXAD layer slot reader is not open");
    }
    return impl_->info();
}

size_t LayerSlotReader::next_slot() const {
    if (!impl_) {
        throw std::runtime_error("EXAD layer slot reader is not open");
    }
    return impl_->next_slot();
}

bool LayerSlotReader::read_next(BoardSet &set) {
    if (!impl_) {
        throw std::runtime_error("EXAD layer slot reader is not open");
    }
    return impl_->read_next(set);
}

void LayerSlotReader::close() {
    if (impl_) {
        impl_->close();
    }
}

void write_layer_file(
    const std::string &path,
    const Layer &layer,
    FileIOUtils::DirectIoConfig config
) {
    const FileHeader header = make_header(layer);
    std::array<SlotHeader, bucket_slot_count()> slots{};
    for (size_t i = 0; i < layer.sets.size(); ++i) {
        const BoardSet &set = layer.sets[i];
        slots[i].bucket_count = set.buckets.size();
        slots[i].small_bitmap_bytes = set.small_bitmap_bytes.size();
        slots[i].large_bitmap_words = set.large_bitmap_words.size();
        slots[i].live_board_count = set.live_board_count;
        slots[i].exact_bitmap_bits = set.exact_bitmap_bits;
        slots[i].aligned_bitmap_bits = set.aligned_bitmap_bits;
    }

    FileIOUtils::DirectAppendWriter out(path, serialized_size(layer), config);
    out.append(&header, sizeof(header));
    out.append(slots.data(), slots.size() * sizeof(SlotHeader));
    for (const BoardSet &set : layer.sets) {
        if (!set.buckets.empty()) {
            out.append(set.buckets.data(), set.buckets.size() * sizeof(BucketEntry));
        }
        if (!set.small_bitmap_bytes.empty()) {
            out.append(set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size());
        }
        if (!set.large_bitmap_words.empty()) {
            out.append(set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t));
        }
    }
    out.close();
}

Layer read_layer_file(
    const std::string &path,
    FileIOUtils::DirectIoConfig config
) {
    FileIOUtils::DirectSequentialReader in(path, file_size_or_throw(path, "temp layer"), config);
    FileHeader header{};
    read_direct_exact(in, &header, sizeof(header), path);
    validate_header(header);
    std::array<SlotHeader, bucket_slot_count()> slots{};
    read_direct_exact(in, slots.data(), slots.size() * sizeof(SlotHeader), path);

    Layer layer;
    layer.original_board_sum = header.original_board_sum;
    layer.threshold_bits = header.threshold_bits;
    layer.lut_signature = header.lut_signature;
    layer.live_board_count = header.live_board_count;
    for (size_t i = 0; i < layer.sets.size(); ++i) {
        layer.sets[i].threshold_bits = header.threshold_bits;
        read_slot_payload(in, layer.sets[i], slots[i], path);
    }
    in.close();
    return layer;
}

void write_lut_file(
    const std::string &path,
    const Luts &luts,
    FileIOUtils::DirectIoConfig config
) {
    LutFileHeader header{};
    std::memcpy(header.magic, kLutMagic, sizeof(kLutMagic));
    header.valid_mask_count = static_cast<uint32_t>(luts.config.valid_suffix_masks.size());
    header.config_signature = luts.config_signature;
    header.rank_table_count = luts.rank_tables.size();
    for (const auto &table : luts.rank_tables) {
        header.rank_table_values += table.size();
    }
    header.packed_rank_pair_values = luts.packed_rank_pair_table.size();
    header.size_table_values = luts.size_table.size();
    header.offset_table_values = luts.offset_table.size();
    header.unrank_array_values = luts.unrank_array.size();
    header.high_base_values = luts.high_base.size();
    header.row16_sum_values = luts.row16_sum.size();
    header.packed_table0 = luts.packed_table0;
    header.packed_table1 = luts.packed_table1;
    std::memcpy(header.table_for_high, luts.table_for_high.data(), luts.table_for_high.size());
    std::memcpy(header.max_counts, luts.config.max_counts.data(), luts.config.max_counts.size());
    header.required_suffix24 = luts.config.required_suffix24;

    uint64_t size = sizeof(LutFileHeader)
        + static_cast<uint64_t>(luts.config.valid_suffix_masks.size()) * sizeof(uint32_t)
        + header.rank_table_count * sizeof(uint64_t)
        + header.rank_table_values * sizeof(uint16_t)
        + header.packed_rank_pair_values * sizeof(uint32_t)
        + header.size_table_values * sizeof(uint32_t)
        + header.offset_table_values * sizeof(uint32_t)
        + header.unrank_array_values * sizeof(uint32_t)
        + header.high_base_values * sizeof(uint16_t)
        + header.row16_sum_values * sizeof(uint32_t);

    FileIOUtils::DirectAppendWriter out(path, size, config);
    out.append(&header, sizeof(header));
    if (!luts.config.valid_suffix_masks.empty()) {
        out.append(
            luts.config.valid_suffix_masks.data(),
            luts.config.valid_suffix_masks.size() * sizeof(uint32_t)
        );
    }
    for (const auto &table : luts.rank_tables) {
        const uint64_t table_size = table.size();
        out.append(&table_size, sizeof(table_size));
    }
    for (const auto &table : luts.rank_tables) {
        if (!table.empty()) {
            out.append(table.data(), table.size() * sizeof(uint16_t));
        }
    }
    if (!luts.packed_rank_pair_table.empty()) {
        out.append(luts.packed_rank_pair_table.data(), luts.packed_rank_pair_table.size() * sizeof(uint32_t));
    }
    if (!luts.size_table.empty()) {
        out.append(luts.size_table.data(), luts.size_table.size() * sizeof(uint32_t));
    }
    if (!luts.offset_table.empty()) {
        out.append(luts.offset_table.data(), luts.offset_table.size() * sizeof(uint32_t));
    }
    if (!luts.unrank_array.empty()) {
        out.append(luts.unrank_array.data(), luts.unrank_array.size() * sizeof(uint32_t));
    }
    if (!luts.high_base.empty()) {
        out.append(luts.high_base.data(), luts.high_base.size() * sizeof(uint16_t));
    }
    if (!luts.row16_sum.empty()) {
        out.append(luts.row16_sum.data(), luts.row16_sum.size() * sizeof(uint32_t));
    }
    out.close();
}

Luts read_lut_file(
    const std::string &path,
    FileIOUtils::DirectIoConfig config
) {
    FileIOUtils::DirectSequentialReader in(path, file_size_or_throw(path, "LUT"), config);
    LutFileHeader header{};
    read_direct_exact(in, &header, sizeof(header), path);
    if (std::memcmp(header.magic, kLutMagic, sizeof(kLutMagic)) != 0 || header.version != 1U) {
        throw std::runtime_error("invalid EXAD prefix36 LUT file");
    }
    Luts luts;
    luts.config_signature = header.config_signature;
    std::memcpy(luts.config.max_counts.data(), header.max_counts, luts.config.max_counts.size());
    luts.config.required_suffix24 = header.required_suffix24;
    luts.config.valid_suffix_masks.resize(header.valid_mask_count);
    if (!luts.config.valid_suffix_masks.empty()) {
        read_direct_exact(
            in,
            luts.config.valid_suffix_masks.data(),
            luts.config.valid_suffix_masks.size() * sizeof(uint32_t),
            path
        );
    }
    std::vector<uint64_t> table_sizes(static_cast<size_t>(header.rank_table_count), 0U);
    if (!table_sizes.empty()) {
        read_direct_exact(in, table_sizes.data(), table_sizes.size() * sizeof(uint64_t), path);
    }
    luts.rank_tables.resize(static_cast<size_t>(header.rank_table_count));
    for (size_t i = 0; i < luts.rank_tables.size(); ++i) {
        luts.rank_tables[i].resize(static_cast<size_t>(table_sizes[i]));
        if (!luts.rank_tables[i].empty()) {
            read_direct_exact(
                in,
                luts.rank_tables[i].data(),
                luts.rank_tables[i].size() * sizeof(uint16_t),
                path
            );
        }
    }
    luts.packed_rank_pair_table.resize(static_cast<size_t>(header.packed_rank_pair_values));
    if (!luts.packed_rank_pair_table.empty()) {
        read_direct_exact(
            in,
            luts.packed_rank_pair_table.data(),
            luts.packed_rank_pair_table.size() * sizeof(uint32_t),
            path
        );
    }
    luts.size_table.resize(static_cast<size_t>(header.size_table_values));
    luts.offset_table.resize(static_cast<size_t>(header.offset_table_values));
    luts.unrank_array.resize(static_cast<size_t>(header.unrank_array_values));
    luts.high_base.resize(static_cast<size_t>(header.high_base_values));
    luts.row16_sum.resize(static_cast<size_t>(header.row16_sum_values));
    if (!luts.size_table.empty()) {
        read_direct_exact(in, luts.size_table.data(), luts.size_table.size() * sizeof(uint32_t), path);
    }
    if (!luts.offset_table.empty()) {
        read_direct_exact(in, luts.offset_table.data(), luts.offset_table.size() * sizeof(uint32_t), path);
    }
    if (!luts.unrank_array.empty()) {
        read_direct_exact(in, luts.unrank_array.data(), luts.unrank_array.size() * sizeof(uint32_t), path);
    }
    if (!luts.high_base.empty()) {
        read_direct_exact(in, luts.high_base.data(), luts.high_base.size() * sizeof(uint16_t), path);
    }
    if (!luts.row16_sum.empty()) {
        read_direct_exact(in, luts.row16_sum.data(), luts.row16_sum.size() * sizeof(uint32_t), path);
    }
    std::memcpy(luts.table_for_high.data(), header.table_for_high, luts.table_for_high.size());
    luts.packed_table0 = header.packed_table0;
    luts.packed_table1 = header.packed_table1;
    luts.valid_suffix_count = 0;
    for (uint32_t count : luts.size_table) {
        luts.valid_suffix_count += count;
    }
    initialize_runtime_tables(luts);
    in.close();
    return luts;
}

} // namespace EXAD
