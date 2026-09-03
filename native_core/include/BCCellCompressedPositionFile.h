#pragma once

#include "BCDirectFileIO.h"
#include "BCFileIO.h"
#include "BCPositionFile.h"
#include "FileIOUtils.h"
#include "NativeLzma.h"
#include "PathUtils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace BC {

inline constexpr uint32_t kBCCellCompressedPositionMagic = 0x43504342U; // "BCPC" little-endian.
inline constexpr uint32_t kBCCellCompressedPositionVersion = 1U;
inline constexpr uint32_t kBCCellCompressedPositionCodecXzOrRaw = 1U;
inline constexpr uint32_t kBCCellCompressedPositionHeaderBytes = 32U;
inline constexpr uint32_t kBCCellCompressedPositionFooterBytes = 88U;
inline constexpr uint32_t kBCCellCompressedPositionIndexEntryBytes = 40U;
inline constexpr uint32_t kBCCellCompressedPositionCellFlagEmpty = 1U;
inline constexpr uint32_t kBCCellCompressedPositionCellFlagRaw = 2U;

struct BCCellCompressedPositionIndexEntry {
    uint64_t stored_offset = 0U;
    uint64_t stored_bytes = 0U;
    uint64_t raw_bytes = 0U;
    uint32_t flags = kBCCellCompressedPositionCellFlagEmpty;
    uint32_t reserved0 = 0U;
    uint64_t reserved1 = 0U;

    [[nodiscard]] bool empty() const {
        return (flags & kBCCellCompressedPositionCellFlagEmpty) != 0U;
    }

    [[nodiscard]] bool raw() const {
        return (flags & kBCCellCompressedPositionCellFlagRaw) != 0U;
    }
};

struct BCCellCompressedPositionFooter {
    uint32_t magic = kBCCellCompressedPositionMagic;
    uint32_t version = kBCCellCompressedPositionVersion;
    uint32_t footer_bytes = kBCCellCompressedPositionFooterBytes;
    uint32_t codec = kBCCellCompressedPositionCodecXzOrRaw;
    uint64_t raw_logical_size = 0U;
    uint64_t cell_count = 0U;
    uint64_t metadata_offset = 0U;
    uint64_t metadata_bytes = 0U;
    uint64_t index_offset = 0U;
    uint64_t index_bytes = 0U;
    uint64_t payload_bytes = 0U;
    uint64_t reserved0 = 0U;
    uint64_t reserved1 = 0U;
};

[[nodiscard]] inline std::filesystem::path bc_cell_compressed_position_path(
    const std::filesystem::path &position_path
) {
    return std::filesystem::path(position_path.string() + "c");
}

[[nodiscard]] inline bool bc_is_cell_compressed_position_path(
    const std::filesystem::path &path
) {
    const std::string name = path.filename().string();
    const std::string suffix = ".bcposc";
    return name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline void bc_append_cell_compressed_file_header(std::vector<uint8_t> &out) {
    bc_append_u32_le(out, kBCCellCompressedPositionMagic);
    bc_append_u32_le(out, kBCCellCompressedPositionVersion);
    bc_append_u32_le(out, kBCCellCompressedPositionHeaderBytes);
    bc_append_u32_le(out, kBCCellCompressedPositionCodecXzOrRaw);
    bc_append_u32_le(out, 0U);
    bc_append_u32_le(out, 0U);
    bc_append_u32_le(out, 0U);
    bc_append_u32_le(out, 0U);
    if (out.size() != kBCCellCompressedPositionHeaderBytes) {
        throw std::logic_error("BC cell-compressed position header serialized size mismatch");
    }
}

inline void bc_append_cell_compressed_index_entry(
    std::vector<uint8_t> &out,
    const BCCellCompressedPositionIndexEntry &entry
) {
    const size_t begin = out.size();
    bc_append_u64_le(out, entry.stored_offset);
    bc_append_u64_le(out, entry.stored_bytes);
    bc_append_u64_le(out, entry.raw_bytes);
    bc_append_u32_le(out, entry.flags);
    bc_append_u32_le(out, entry.reserved0);
    bc_append_u64_le(out, entry.reserved1);
    if (out.size() - begin != kBCCellCompressedPositionIndexEntryBytes) {
        throw std::logic_error("BC cell-compressed position index serialized size mismatch");
    }
}

[[nodiscard]] inline BCCellCompressedPositionIndexEntry bc_read_cell_compressed_index_entry(
    const uint8_t *data,
    size_t remaining
) {
    if (data == nullptr) {
        throw std::invalid_argument("BC cell-compressed position index pointer is null");
    }
    if (remaining < kBCCellCompressedPositionIndexEntryBytes) {
        throw std::out_of_range("BC cell-compressed position index entry is truncated");
    }
    BCCellCompressedPositionIndexEntry entry;
    entry.stored_offset = load_u64_le(data + 0U);
    entry.stored_bytes = load_u64_le(data + 8U);
    entry.raw_bytes = load_u64_le(data + 16U);
    entry.flags = bc_load_u32_le(data + 24U);
    entry.reserved0 = bc_load_u32_le(data + 28U);
    entry.reserved1 = load_u64_le(data + 32U);
    return entry;
}

inline void bc_append_cell_compressed_footer(
    std::vector<uint8_t> &out,
    const BCCellCompressedPositionFooter &footer
) {
    const size_t begin = out.size();
    bc_append_u32_le(out, footer.magic);
    bc_append_u32_le(out, footer.version);
    bc_append_u32_le(out, footer.footer_bytes);
    bc_append_u32_le(out, footer.codec);
    bc_append_u64_le(out, footer.raw_logical_size);
    bc_append_u64_le(out, footer.cell_count);
    bc_append_u64_le(out, footer.metadata_offset);
    bc_append_u64_le(out, footer.metadata_bytes);
    bc_append_u64_le(out, footer.index_offset);
    bc_append_u64_le(out, footer.index_bytes);
    bc_append_u64_le(out, footer.payload_bytes);
    bc_append_u64_le(out, footer.reserved0);
    bc_append_u64_le(out, footer.reserved1);
    if (out.size() - begin != kBCCellCompressedPositionFooterBytes) {
        throw std::logic_error("BC cell-compressed position footer serialized size mismatch");
    }
}

[[nodiscard]] inline BCCellCompressedPositionFooter bc_read_cell_compressed_footer(
    const std::vector<uint8_t> &bytes
) {
    if (bytes.size() != kBCCellCompressedPositionFooterBytes) {
        throw std::invalid_argument("BC cell-compressed position footer size mismatch");
    }
    const uint8_t *p = bytes.data();
    BCCellCompressedPositionFooter footer;
    footer.magic = bc_load_u32_le(p + 0U);
    footer.version = bc_load_u32_le(p + 4U);
    footer.footer_bytes = bc_load_u32_le(p + 8U);
    footer.codec = bc_load_u32_le(p + 12U);
    footer.raw_logical_size = load_u64_le(p + 16U);
    footer.cell_count = load_u64_le(p + 24U);
    footer.metadata_offset = load_u64_le(p + 32U);
    footer.metadata_bytes = load_u64_le(p + 40U);
    footer.index_offset = load_u64_le(p + 48U);
    footer.index_bytes = load_u64_le(p + 56U);
    footer.payload_bytes = load_u64_le(p + 64U);
    footer.reserved0 = load_u64_le(p + 72U);
    footer.reserved1 = load_u64_le(p + 80U);
    return footer;
}

[[nodiscard]] inline std::vector<uint8_t> bc_build_position_metadata_prefix(
    const BCFamilyTable &axis,
    const std::vector<BCPositionCellDescriptor> &descriptors,
    uint64_t bucket_bytes,
    uint64_t rank_payload_bytes
) {
    const uint64_t descriptor_count = descriptors.size();
    const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
    const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis.family_count());
    const uint64_t descriptor_offset = bc_checked_add_u64(
        kBCPositionHeaderBytes,
        axis_coord_bytes,
        "BC cell-compressed position descriptor table offset overflow"
    );
    const uint64_t bucket_offset = bc_checked_add_u64(
        descriptor_offset,
        descriptor_bytes,
        "BC cell-compressed position bucket stream offset overflow"
    );
    const uint64_t rank_offset = bc_checked_add_u64(
        bucket_offset,
        bucket_bytes,
        "BC cell-compressed position rank stream offset overflow"
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
    header.bucket_meta_bytes = bucket_bytes;
    header.rank_payload_offset = rank_offset;
    header.rank_payload_bytes = rank_payload_bytes;

    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(bucket_offset));
    bc_append_header(out, header);
    bc_append_axis_coord_table(out, axis);
    for (const BCPositionCellDescriptor &descriptor : descriptors) {
        bc_append_cell_descriptor(out, descriptor);
    }
    if (out.size() != bucket_offset) {
        throw std::logic_error("BC cell-compressed position metadata prefix size mismatch");
    }
    return out;
}

class BCCellCompressedPositionWriter {
public:
    BCCellCompressedPositionWriter() = default;

    explicit BCCellCompressedPositionWriter(const std::filesystem::path &final_path) {
        open(final_path);
    }

    ~BCCellCompressedPositionWriter() {
        try {
            close_without_publish();
        } catch (...) {
        }
    }

    BCCellCompressedPositionWriter(const BCCellCompressedPositionWriter &) = delete;
    BCCellCompressedPositionWriter &operator=(const BCCellCompressedPositionWriter &) = delete;

    void open(const std::filesystem::path &final_path) {
        if (opened_) {
            throw std::logic_error("BC cell-compressed position writer is already open");
        }
        final_path_ = NativePath::to_utf8_string(final_path);
        temp_path_ = final_path_ + ".tmp";
        std::error_code ec;
        NativePath::remove(temp_path_, ec);
        ec.clear();
        NativePath::remove(final_path_, ec);
        file_.open(NativePath::from_utf8(temp_path_), std::ios::binary | std::ios::trunc);
        if (!file_) {
            throw std::runtime_error("failed to open BC cell-compressed position temp file: " + temp_path_);
        }
        std::vector<uint8_t> header;
        bc_append_cell_compressed_file_header(header);
        write_bytes(header.data(), header.size());
        payload_begin_ = kBCCellCompressedPositionHeaderBytes;
        cursor_ = payload_begin_;
        opened_ = true;
    }

    void begin_cells(uint64_t cell_count) {
        if (!opened_) {
            throw std::logic_error("BC cell-compressed position writer is not open");
        }
        entries_.assign(static_cast<size_t>(cell_count), BCCellCompressedPositionIndexEntry{});
    }

    void write_empty_cell(CellId cid) {
        require_cell(cid);
        entries_[static_cast<size_t>(cid)] = BCCellCompressedPositionIndexEntry{};
    }

    void write_cell(
        CellId cid,
        const BCPositionCellDescriptor &descriptor,
        const std::vector<uint8_t> &bucket_bytes,
        const std::vector<uint8_t> &rank_payload
    ) {
        require_cell(cid);
        if (descriptor.empty()) {
            write_empty_cell(cid);
            return;
        }
        const uint64_t raw_bytes_u64 = bc_checked_add_u64(
            bucket_bytes.size(),
            rank_payload.size(),
            "BC cell-compressed position raw cell byte count overflow"
        );
        if (raw_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC cell-compressed position raw cell exceeds size_t");
        }
        std::vector<uint8_t> raw;
        raw.reserve(static_cast<size_t>(raw_bytes_u64));
        raw.insert(raw.end(), bucket_bytes.begin(), bucket_bytes.end());
        raw.insert(raw.end(), rank_payload.begin(), rank_payload.end());
        pending_cells_.push_back(PendingCell{
            cid,
            raw_bytes_u64,
            std::async(
                std::launch::async,
                [raw = std::move(raw)]() mutable {
                    EncodedCell encoded;
                    if (!raw.empty()) {
                        encoded.stored = compress_xz_block_native(raw.data(), raw.size(), 1);
                    }
                    if (encoded.stored.empty() || encoded.stored.size() >= raw.size()) {
                        encoded.stored = std::move(raw);
                        encoded.flags |= kBCCellCompressedPositionCellFlagRaw;
                    }
                    return encoded;
                })
        });
        if (pending_cells_.size() >= compression_parallelism()) {
            publish_front_cell();
        }
    }

    [[nodiscard]] uint64_t finish(
        const BCFamilyTable &axis,
        const std::vector<BCPositionCellDescriptor> &descriptors,
        uint64_t bucket_bytes,
        uint64_t rank_payload_bytes
    ) {
        if (!opened_) {
            throw std::logic_error("BC cell-compressed position writer is not open");
        }
        publish_pending_cells();
        if (entries_.size() != descriptors.size()) {
            throw std::logic_error("BC cell-compressed position index/descriptor count mismatch");
        }
        std::vector<uint8_t> metadata =
            bc_build_position_metadata_prefix(axis, descriptors, bucket_bytes, rank_payload_bytes);
        const BCPositionHeader raw_header = bc_read_header(metadata);
        const uint64_t raw_logical_size = bc_position_logical_size_from_header(raw_header);

        BCCellCompressedPositionFooter footer;
        footer.raw_logical_size = raw_logical_size;
        footer.cell_count = entries_.size();
        footer.metadata_offset = cursor_;
        footer.metadata_bytes = metadata.size();
        if (!metadata.empty()) {
            write_bytes(metadata.data(), metadata.size());
        }
        cursor_ = bc_checked_add_u64(
            cursor_,
            metadata.size(),
            "BC cell-compressed position metadata cursor overflow"
        );

        std::vector<uint8_t> index;
        index.reserve(entries_.size() * kBCCellCompressedPositionIndexEntryBytes);
        for (const BCCellCompressedPositionIndexEntry &entry : entries_) {
            bc_append_cell_compressed_index_entry(index, entry);
        }
        footer.index_offset = cursor_;
        footer.index_bytes = index.size();
        if (!index.empty()) {
            write_bytes(index.data(), index.size());
        }
        cursor_ = bc_checked_add_u64(
            cursor_,
            index.size(),
            "BC cell-compressed position index cursor overflow"
        );
        footer.payload_bytes = footer.metadata_offset - payload_begin_;

        std::vector<uint8_t> footer_bytes;
        footer_bytes.reserve(kBCCellCompressedPositionFooterBytes);
        bc_append_cell_compressed_footer(footer_bytes, footer);
        write_bytes(footer_bytes.data(), footer_bytes.size());
        cursor_ = bc_checked_add_u64(
            cursor_,
            footer_bytes.size(),
            "BC cell-compressed position footer cursor overflow"
        );
        file_.close();
        if (!file_) {
            throw std::runtime_error("failed to close BC cell-compressed position temp file: " + temp_path_);
        }
        std::error_code ec;
        NativePath::rename(temp_path_, final_path_, ec);
        if (ec) {
            NativePath::remove(temp_path_, ec);
            throw std::runtime_error("failed to publish BC cell-compressed position file: " + final_path_);
        }
        opened_ = false;
        return raw_logical_size;
    }

    void flush() {
        if (opened_) {
            publish_pending_cells();
            file_.flush();
        }
    }

    void close_without_publish() {
        if (!opened_) {
            return;
        }
        pending_cells_.clear();
        file_.close();
        std::error_code ec;
        NativePath::remove(temp_path_, ec);
        opened_ = false;
    }

private:
    struct EncodedCell {
        std::vector<uint8_t> stored;
        uint32_t flags = 0U;
    };

    struct PendingCell {
        CellId cid = 0U;
        uint64_t raw_bytes = 0U;
        std::future<EncodedCell> encoded;
    };

    [[nodiscard]] static size_t compression_parallelism() {
        const unsigned hardware = std::thread::hardware_concurrency();
        const size_t useful = hardware == 0U
            ? 4U
            : std::max<size_t>(2U, static_cast<size_t>(hardware) / 2U);
        return std::min<size_t>(8U, useful);
    }

    void publish_front_cell() {
        if (pending_cells_.empty()) {
            return;
        }
        PendingCell pending = std::move(pending_cells_.front());
        pending_cells_.pop_front();
        EncodedCell encoded = pending.encoded.get();

        BCCellCompressedPositionIndexEntry entry;
        entry.stored_offset = cursor_;
        entry.stored_bytes = static_cast<uint64_t>(encoded.stored.size());
        entry.raw_bytes = pending.raw_bytes;
        entry.flags = encoded.flags;
        entries_[static_cast<size_t>(pending.cid)] = entry;
        if (!encoded.stored.empty()) {
            write_bytes(encoded.stored.data(), encoded.stored.size());
        }
        cursor_ = bc_checked_add_u64(
            cursor_,
            encoded.stored.size(),
            "BC cell-compressed position writer cursor overflow"
        );
    }

    void publish_pending_cells() {
        while (!pending_cells_.empty()) {
            publish_front_cell();
        }
    }

    void require_cell(CellId cid) const {
        if (!opened_) {
            throw std::logic_error("BC cell-compressed position writer is not open");
        }
        if (cid >= entries_.size()) {
            throw std::out_of_range("BC cell-compressed position writer cell id out of range");
        }
    }

    void write_bytes(const void *data, size_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC cell-compressed position writer data pointer is null");
        }
        FileIOUtils::write_exact(file_, data, bytes, temp_path_);
    }

    std::ofstream file_;
    std::string final_path_;
    std::string temp_path_;
    std::vector<BCCellCompressedPositionIndexEntry> entries_;
    std::deque<PendingCell> pending_cells_;
    uint64_t payload_begin_ = 0U;
    uint64_t cursor_ = 0U;
    bool opened_ = false;
};

class BCCellCompressedPositionReadableFile final : public BCReadableFile {
public:
    explicit BCCellCompressedPositionReadableFile(
        const std::filesystem::path &path,
        bool preload_cells = false,
        bool direct_io = false,
        uint32_t direct_queue_depth = 16U,
        uint64_t direct_max_transfer_bytes = kBCDirectDefaultMaxTransferBytes,
        uint64_t cache_budget_bytes = 256ULL * 1024ULL * 1024ULL,
        uint32_t decode_threads = 4U
    )
        : path_(NativePath::to_utf8_string(path)),
          preload_cells_(preload_cells),
          direct_io_(direct_io),
          direct_queue_depth_(std::max<uint32_t>(1U, direct_queue_depth)),
          direct_max_transfer_bytes_(direct_max_transfer_bytes),
          cache_budget_bytes_(std::max<uint64_t>(1U, cache_budget_bytes)),
          decode_threads_(std::max<uint32_t>(1U, decode_threads)) {
        open();
    }

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC cell-compressed position read target is null");
        }
        if (offset > raw_logical_size_ || bytes > raw_logical_size_ - offset) {
            throw std::out_of_range("BC cell-compressed position virtual read exceeds raw size");
        }
        uint8_t *out = static_cast<uint8_t *>(data);
        uint64_t cursor = offset;
        uint64_t remaining = bytes;
        while (remaining > 0U) {
            if (cursor < metadata_.size()) {
                const uint64_t take = std::min<uint64_t>(
                    remaining,
                    static_cast<uint64_t>(metadata_.size()) - cursor
                );
                std::memcpy(
                    out,
                    metadata_.data() + static_cast<size_t>(cursor),
                    static_cast<size_t>(take)
                );
                out += take;
                cursor += take;
                remaining -= take;
                continue;
            }
            const uint64_t bucket_end = bc_checked_add_u64(
                raw_header_.bucket_meta_offset,
                raw_header_.bucket_meta_bytes,
                "BC cell-compressed position bucket end overflow"
            );
            if (cursor < bucket_end) {
                const uint64_t take = std::min<uint64_t>(remaining, bucket_end - cursor);
                read_data_stream(
                    raw_header_.bucket_meta_offset,
                    raw_header_.bucket_meta_bytes,
                    true,
                    cursor,
                    out,
                    take
                );
                out += take;
                cursor += take;
                remaining -= take;
                continue;
            }
            const uint64_t rank_end = bc_checked_add_u64(
                raw_header_.rank_payload_offset,
                raw_header_.rank_payload_bytes,
                "BC cell-compressed position rank end overflow"
            );
            if (cursor < rank_end) {
                const uint64_t take = std::min<uint64_t>(remaining, rank_end - cursor);
                read_data_stream(
                    raw_header_.rank_payload_offset,
                    raw_header_.rank_payload_bytes,
                    false,
                    cursor,
                    out,
                    take
                );
                out += take;
                cursor += take;
                remaining -= take;
                continue;
            }
            throw std::out_of_range("BC cell-compressed position virtual read hit unmapped range");
        }
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }
        if (preload_cells_ && !preloaded_) {
            preload_all_cells();
        }
        if (!preloaded_) {
            decode_requested_cells(requests, stats);
        }
        for (const BCFileReadRequest &request : requests) {
            read_at(request.offset, request.data, request.bytes);
            if (stats != nullptr) {
                if (stats->request_count == std::numeric_limits<uint64_t>::max() ||
                    stats->requested_bytes >
                        std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC cell-compressed read stats overflow");
                }
                ++stats->request_count;
                stats->requested_bytes += request.bytes;
            }
        }
        trim_cache();
    }

    [[nodiscard]] uint64_t size() const override {
        return raw_logical_size_;
    }

private:
    void open() {
        file_.open(NativePath::from_utf8(path_), std::ios::binary | std::ios::ate);
        if (!file_) {
            throw std::runtime_error("failed to open BC cell-compressed position file: " + path_);
        }
        const std::ifstream::pos_type end_pos = file_.tellg();
        if (end_pos < 0) {
            throw std::runtime_error("failed to size BC cell-compressed position file: " + path_);
        }
        physical_size_ = static_cast<uint64_t>(end_pos);
        if (physical_size_ < kBCCellCompressedPositionHeaderBytes + kBCCellCompressedPositionFooterBytes) {
            throw std::runtime_error("BC cell-compressed position file is too small: " + path_);
        }

        std::vector<uint8_t> header(kBCCellCompressedPositionHeaderBytes);
        read_physical(0U, header.data(), header.size());
        if (bc_load_u32_le(header.data()) != kBCCellCompressedPositionMagic ||
            bc_load_u32_le(header.data() + 4U) != kBCCellCompressedPositionVersion ||
            bc_load_u32_le(header.data() + 8U) != kBCCellCompressedPositionHeaderBytes) {
            throw std::runtime_error("BC cell-compressed position header mismatch: " + path_);
        }

        std::vector<uint8_t> footer_bytes(kBCCellCompressedPositionFooterBytes);
        read_physical(physical_size_ - kBCCellCompressedPositionFooterBytes,
                      footer_bytes.data(),
                      footer_bytes.size());
        footer_ = bc_read_cell_compressed_footer(footer_bytes);
        validate_footer();
        metadata_.resize(static_cast<size_t>(footer_.metadata_bytes));
        read_physical(footer_.metadata_offset, metadata_.data(), metadata_.size());
        raw_header_ = bc_read_header(metadata_);
        raw_logical_size_ = bc_position_logical_size_from_header(raw_header_);
        if (raw_logical_size_ != footer_.raw_logical_size) {
            throw std::runtime_error("BC cell-compressed raw logical size mismatch: " + path_);
        }
        read_descriptors();
        read_index();
        validate_index();
        if (direct_io_) {
            direct_prefix_size_ = bc_direct_align_down(physical_size_, 4096U);
            if (direct_prefix_size_ != 0U) {
                BCDirectFileIOOptions direct_options;
                direct_options.queue_depth = direct_queue_depth_;
                direct_options.max_transfer_bytes = direct_max_transfer_bytes_;
                direct_options.overlapped = direct_queue_depth_ > 1U;
                direct_options.logical_size = direct_prefix_size_;
                direct_options.physical_size = physical_size_;
                direct_reader_ = std::make_unique<BCDirectFileReader>(
                    NativePath::from_utf8(path_),
                    direct_options);
            }
        }
    }

    void validate_footer() const {
        if (footer_.magic != kBCCellCompressedPositionMagic ||
            footer_.version != kBCCellCompressedPositionVersion ||
            footer_.footer_bytes != kBCCellCompressedPositionFooterBytes ||
            footer_.codec != kBCCellCompressedPositionCodecXzOrRaw ||
            footer_.reserved0 != 0U ||
            footer_.reserved1 != 0U) {
            throw std::runtime_error("BC cell-compressed position footer mismatch: " + path_);
        }
        const uint64_t footer_offset = physical_size_ - kBCCellCompressedPositionFooterBytes;
        if (footer_.metadata_offset < kBCCellCompressedPositionHeaderBytes ||
            footer_.metadata_offset > footer_offset ||
            footer_.metadata_bytes > footer_offset - footer_.metadata_offset ||
            footer_.index_offset < kBCCellCompressedPositionHeaderBytes ||
            footer_.index_offset > footer_offset ||
            footer_.index_bytes > footer_offset - footer_.index_offset) {
            throw std::runtime_error("BC cell-compressed position footer range mismatch: " + path_);
        }
        if (footer_.index_bytes != footer_.cell_count * kBCCellCompressedPositionIndexEntryBytes) {
            throw std::runtime_error("BC cell-compressed position index size mismatch: " + path_);
        }
    }

    void read_descriptors() {
        const BCPositionHeader header = bc_read_header(metadata_);
        const uint64_t expected_metadata_bytes = header.bucket_meta_offset;
        if (footer_.metadata_bytes != expected_metadata_bytes) {
            throw std::runtime_error("BC cell-compressed position metadata size mismatch: " + path_);
        }
        if (header.descriptor_count != footer_.cell_count) {
            throw std::runtime_error("BC cell-compressed position descriptor count mismatch: " + path_);
        }
        descriptors_.clear();
        descriptors_.reserve(static_cast<size_t>(header.descriptor_count));
        for (uint64_t i = 0U; i < header.descriptor_count; ++i) {
            const size_t offset = static_cast<size_t>(
                header.descriptor_table_offset + i * kBCPositionCellDescriptorBytes
            );
            descriptors_.push_back(
                bc_read_cell_descriptor(metadata_.data() + offset, metadata_.size() - offset));
        }
    }

    void read_index() {
        std::vector<uint8_t> bytes(static_cast<size_t>(footer_.index_bytes));
        read_physical(footer_.index_offset, bytes.data(), bytes.size());
        entries_.clear();
        entries_.reserve(static_cast<size_t>(footer_.cell_count));
        for (uint64_t i = 0U; i < footer_.cell_count; ++i) {
            const size_t offset = static_cast<size_t>(i * kBCCellCompressedPositionIndexEntryBytes);
            entries_.push_back(
                bc_read_cell_compressed_index_entry(bytes.data() + offset, bytes.size() - offset));
        }
    }

    void validate_index() const {
        const uint64_t payload_end = bc_checked_add_u64(
            kBCCellCompressedPositionHeaderBytes,
            footer_.payload_bytes,
            "BC cell-compressed position payload end overflow"
        );
        if (payload_end != footer_.metadata_offset) {
            throw std::runtime_error("BC cell-compressed position payload size mismatch: " + path_);
        }
        for (CellId cid = 0U; cid < entries_.size(); ++cid) {
            const BCCellCompressedPositionIndexEntry &entry = entries_[static_cast<size_t>(cid)];
            const BCPositionCellDescriptor &desc = descriptors_[static_cast<size_t>(cid)];
            if (entry.reserved0 != 0U || entry.reserved1 != 0U ||
                (entry.flags & ~(kBCCellCompressedPositionCellFlagEmpty |
                                 kBCCellCompressedPositionCellFlagRaw)) != 0U) {
                throw std::runtime_error("BC cell-compressed position index flags mismatch: " + path_);
            }
            const uint64_t raw_bytes = bc_checked_add_u64(
                static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes,
                desc.rank_payload_bytes,
                "BC cell-compressed position descriptor raw byte overflow"
            );
            if (desc.empty()) {
                if (!entry.empty() || entry.stored_offset != 0U || entry.stored_bytes != 0U ||
                    entry.raw_bytes != 0U) {
                    throw std::runtime_error("BC cell-compressed position empty index mismatch: " + path_);
                }
                continue;
            }
            if (entry.empty() || entry.raw_bytes != raw_bytes ||
                entry.stored_offset < kBCCellCompressedPositionHeaderBytes ||
                entry.stored_offset > payload_end ||
                entry.stored_bytes > payload_end - entry.stored_offset) {
                throw std::runtime_error("BC cell-compressed position index range mismatch: " + path_);
            }
        }
    }

    void read_physical(uint64_t offset, void *data, size_t bytes) const {
        if (bytes == 0U) {
            return;
        }
        if (offset > physical_size_ || bytes > physical_size_ - offset) {
            throw std::out_of_range("BC cell-compressed physical read exceeds file: " + path_);
        }
        file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!file_) {
            throw std::runtime_error("BC cell-compressed seek failed: " + path_);
        }
        FileIOUtils::read_exact(file_, data, bytes, path_);
    }

    [[nodiscard]] bool direct_physical_range_eligible(
        uint64_t offset,
        uint64_t bytes
    ) const {
        if (!direct_reader_ || bytes == 0U || offset > direct_prefix_size_ ||
            bytes > direct_prefix_size_ - offset) {
            return false;
        }
        return bc_direct_align_up(offset + bytes, 4096U) <= direct_prefix_size_;
    }

    void add_requested_stream_cells(
        uint64_t stream_base,
        uint64_t stream_bytes,
        bool bucket_stream,
        uint64_t request_begin,
        uint64_t request_end,
        std::vector<uint8_t> &selected,
        std::vector<CellId> &cell_ids
    ) const {
        const uint64_t stream_end = bc_checked_add_u64(
            stream_base,
            stream_bytes,
            "BC cell-compressed stream end overflow");
        const uint64_t overlap_begin = std::max(request_begin, stream_base);
        const uint64_t overlap_end = std::min(request_end, stream_end);
        if (overlap_begin >= overlap_end) {
            return;
        }
        const uint64_t rel_begin = overlap_begin - stream_base;
        const uint64_t rel_end = overlap_end - stream_base;
        for (CellId cid = 0U; cid < descriptors_.size(); ++cid) {
            const BCPositionCellDescriptor &desc = descriptors_[static_cast<size_t>(cid)];
            if (desc.empty()) {
                continue;
            }
            const uint64_t cell_begin = bucket_stream
                ? desc.bucket_meta_offset
                : desc.rank_payload_offset;
            const uint64_t cell_bytes = bucket_stream
                ? static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes
                : desc.rank_payload_bytes;
            if (cell_begin + cell_bytes <= rel_begin || cell_begin >= rel_end ||
                selected[static_cast<size_t>(cid)] != 0U) {
                continue;
            }
            selected[static_cast<size_t>(cid)] = 1U;
            cell_ids.push_back(cid);
        }
    }

    [[nodiscard]] std::vector<uint8_t> decode_stored_payload(
        CellId cid,
        std::vector<uint8_t> stored
    ) const {
        const BCCellCompressedPositionIndexEntry &entry = entries_[static_cast<size_t>(cid)];
        if (entry.raw()) {
            if (stored.size() != entry.raw_bytes) {
                throw std::runtime_error("BC cell-compressed raw cell size mismatch");
            }
            return stored;
        }
        std::vector<uint8_t> decoded =
            decompress_xz_block_native(stored.data(), stored.size());
        if (decoded.size() != entry.raw_bytes) {
            throw std::runtime_error("BC cell-compressed decoded cell size mismatch");
        }
        return decoded;
    }

    void decode_requested_cells(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats
    ) const {
        std::vector<uint8_t> selected(entries_.size(), 0U);
        std::vector<CellId> requested_cells;
        requested_cells.reserve(requests.size());
        const uint64_t bucket_end = bc_checked_add_u64(
            raw_header_.bucket_meta_offset,
            raw_header_.bucket_meta_bytes,
            "BC cell-compressed bucket end overflow");
        const uint64_t rank_end = bc_checked_add_u64(
            raw_header_.rank_payload_offset,
            raw_header_.rank_payload_bytes,
            "BC cell-compressed rank end overflow");
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC cell-compressed read target is null");
            }
            if (request.offset > raw_logical_size_ ||
                request.bytes > raw_logical_size_ - request.offset) {
                throw std::out_of_range("BC cell-compressed virtual read exceeds raw size");
            }
            const uint64_t request_end = request.offset + request.bytes;
            if (request.offset < bucket_end && request_end > raw_header_.bucket_meta_offset) {
                add_requested_stream_cells(
                    raw_header_.bucket_meta_offset,
                    raw_header_.bucket_meta_bytes,
                    true,
                    request.offset,
                    request_end,
                    selected,
                    requested_cells);
            }
            if (request.offset < rank_end && request_end > raw_header_.rank_payload_offset) {
                add_requested_stream_cells(
                    raw_header_.rank_payload_offset,
                    raw_header_.rank_payload_bytes,
                    false,
                    request.offset,
                    request_end,
                    selected,
                    requested_cells);
            }
        }

        std::vector<CellId> missing;
        missing.reserve(requested_cells.size());
        for (CellId cid : requested_cells) {
            if (cache_.find(cid) == cache_.end()) {
                missing.push_back(cid);
            }
        }
        if (missing.empty()) {
            return;
        }

        std::vector<std::vector<uint8_t>> stored(missing.size());
        std::vector<BCFileReadRequest> direct_requests;
        std::vector<size_t> buffered_indices;
        direct_requests.reserve(missing.size());
        buffered_indices.reserve(missing.size());
        for (size_t i = 0U; i < missing.size(); ++i) {
            const BCCellCompressedPositionIndexEntry &entry =
                entries_[static_cast<size_t>(missing[i])];
            if (entry.stored_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC cell-compressed stored cell exceeds size_t");
            }
            stored[i].resize(static_cast<size_t>(entry.stored_bytes));
            if (direct_physical_range_eligible(entry.stored_offset, entry.stored_bytes)) {
                direct_requests.push_back(BCFileReadRequest{
                    entry.stored_offset,
                    stored[i].data(),
                    entry.stored_bytes});
            } else {
                buffered_indices.push_back(i);
            }
        }

        if (!direct_requests.empty()) {
            BCFileIOStats direct_stats;
            direct_reader_->read_many(direct_requests, &direct_stats);
            if (stats != nullptr) {
                stats->backend_io_count += direct_stats.backend_io_count;
                stats->backend_bytes += direct_stats.backend_bytes;
                stats->backend_seconds += direct_stats.backend_seconds;
            }
        }
        for (size_t i : buffered_indices) {
            const BCCellCompressedPositionIndexEntry &entry =
                entries_[static_cast<size_t>(missing[i])];
            const auto io_t0 = std::chrono::steady_clock::now();
            read_physical(entry.stored_offset, stored[i].data(), stored[i].size());
            if (stats != nullptr) {
                ++stats->backend_io_count;
                stats->backend_bytes += entry.stored_bytes;
                stats->backend_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - io_t0).count();
            }
        }

        std::vector<std::vector<uint8_t>> decoded(missing.size());
        std::exception_ptr decode_error;
#pragma omp parallel for schedule(dynamic, 1) num_threads(decode_threads_)
        for (int64_t i_signed = 0; i_signed < static_cast<int64_t>(missing.size()); ++i_signed) {
            try {
                const size_t i = static_cast<size_t>(i_signed);
                decoded[i] = decode_stored_payload(missing[i], std::move(stored[i]));
            } catch (...) {
#pragma omp critical(bc_cell_batch_decode_error)
                {
                    if (!decode_error) {
                        decode_error = std::current_exception();
                    }
                }
            }
        }
        if (decode_error) {
            std::rethrow_exception(decode_error);
        }
        for (size_t i = 0U; i < missing.size(); ++i) {
            cache_bytes_ = bc_checked_add_u64(
                cache_bytes_,
                decoded[i].size(),
                "BC cell-compressed cache byte count overflow");
            cache_order_.push_back(missing[i]);
            cache_.emplace(missing[i], std::move(decoded[i]));
        }
    }

    void trim_cache() const {
        while (cache_bytes_ > cache_budget_bytes_ && cache_order_.size() > 1U) {
            const CellId evict = cache_order_.front();
            cache_order_.pop_front();
            auto it = cache_.find(evict);
            if (it != cache_.end()) {
                cache_bytes_ -= it->second.size();
                cache_.erase(it);
            }
        }
    }

    void read_data_stream(
        uint64_t stream_base,
        uint64_t stream_bytes,
        bool bucket_stream,
        uint64_t virtual_offset,
        uint8_t *out,
        uint64_t bytes
    ) const {
        if (virtual_offset < stream_base || bytes > stream_base + stream_bytes - virtual_offset) {
            throw std::out_of_range("BC cell-compressed virtual stream read exceeds range");
        }
        const uint64_t rel_begin = virtual_offset - stream_base;
        const uint64_t rel_end = rel_begin + bytes;
        uint64_t written = 0U;
        for (CellId cid = 0U; cid < descriptors_.size() && written < bytes; ++cid) {
            const BCPositionCellDescriptor &desc = descriptors_[static_cast<size_t>(cid)];
            if (desc.empty()) {
                continue;
            }
            const uint64_t cell_begin = bucket_stream
                ? desc.bucket_meta_offset
                : desc.rank_payload_offset;
            const uint64_t cell_bytes = bucket_stream
                ? static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes
                : desc.rank_payload_bytes;
            const uint64_t cell_end = cell_begin + cell_bytes;
            if (cell_end <= rel_begin || cell_begin >= rel_end) {
                continue;
            }
            const uint64_t copy_begin = std::max<uint64_t>(rel_begin, cell_begin);
            const uint64_t copy_end = std::min<uint64_t>(rel_end, cell_end);
            const std::vector<uint8_t> &payload = cached_cell_payload(cid);
            const uint64_t bucket_bytes =
                static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes;
            const uint64_t payload_offset = (bucket_stream ? 0U : bucket_bytes) +
                (copy_begin - cell_begin);
            const uint64_t copy_bytes = copy_end - copy_begin;
            if (payload_offset > payload.size() || copy_bytes > payload.size() - payload_offset) {
                throw std::runtime_error("BC cell-compressed payload slice exceeds decoded cell");
            }
            std::memcpy(
                out + static_cast<size_t>(copy_begin - rel_begin),
                payload.data() + static_cast<size_t>(payload_offset),
                static_cast<size_t>(copy_bytes)
            );
            written += copy_bytes;
        }
        if (written != bytes) {
            throw std::runtime_error("BC cell-compressed virtual stream read underfilled");
        }
    }

    const std::vector<uint8_t> &cached_cell_payload(CellId cid) const {
        if (preloaded_) {
            return preloaded_payloads_[static_cast<size_t>(cid)];
        }
        const auto found = cache_.find(cid);
        if (found != cache_.end()) {
            return found->second;
        }
        std::vector<uint8_t> payload = decode_cell_payload(cid);
        cache_bytes_ = bc_checked_add_u64(
            cache_bytes_,
            payload.size(),
            "BC cell-compressed cache byte count overflow"
        );
        cache_order_.push_back(cid);
        auto inserted = cache_.emplace(cid, std::move(payload));
        trim_cache();
        return inserted.first->second;
    }

    void preload_all_cells() const {
        if (preloaded_) {
            return;
        }
        preloaded_payloads_.clear();
        preloaded_payloads_.resize(entries_.size());
        constexpr size_t kDecodeBatchCells = 64U;
        for (size_t batch_begin = 0U; batch_begin < entries_.size(); batch_begin += kDecodeBatchCells) {
            const size_t batch_end = std::min(entries_.size(), batch_begin + kDecodeBatchCells);
            std::vector<std::vector<uint8_t>> stored(batch_end - batch_begin);
            for (size_t cid = batch_begin; cid < batch_end; ++cid) {
                const BCCellCompressedPositionIndexEntry &entry = entries_[cid];
                if (entry.empty()) {
                    continue;
                }
                if (entry.stored_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                    throw std::overflow_error("BC cell-compressed stored cell exceeds size_t");
                }
                stored[cid - batch_begin].resize(static_cast<size_t>(entry.stored_bytes));
                read_physical(
                    entry.stored_offset,
                    stored[cid - batch_begin].data(),
                    stored[cid - batch_begin].size());
            }

            std::exception_ptr decode_error;
            #pragma omp parallel for schedule(dynamic, 1)
            for (int64_t cid_i = static_cast<int64_t>(batch_begin);
                 cid_i < static_cast<int64_t>(batch_end);
                 ++cid_i) {
                try {
                    const size_t cid = static_cast<size_t>(cid_i);
                    const BCCellCompressedPositionIndexEntry &entry = entries_[cid];
                    if (entry.empty()) {
                        continue;
                    }
                    const std::vector<uint8_t> &encoded = stored[cid - batch_begin];
                    std::vector<uint8_t> decoded;
                    if (entry.raw()) {
                        decoded = encoded;
                    } else {
                        decoded = decompress_xz_block_native(encoded.data(), encoded.size());
                    }
                    if (decoded.size() != entry.raw_bytes) {
                        throw std::runtime_error("BC cell-compressed decoded cell size mismatch");
                    }
                    preloaded_payloads_[cid] = std::move(decoded);
                } catch (...) {
                    #pragma omp critical(bc_cell_preload_error)
                    {
                        if (!decode_error) {
                            decode_error = std::current_exception();
                        }
                    }
                }
            }
            if (decode_error) {
                std::rethrow_exception(decode_error);
            }
        }
        cache_.clear();
        cache_order_.clear();
        cache_bytes_ = 0U;
        preloaded_ = true;
    }

    [[nodiscard]] std::vector<uint8_t> decode_cell_payload(CellId cid) const {
        const BCCellCompressedPositionIndexEntry &entry = entries_[static_cast<size_t>(cid)];
        if (entry.empty()) {
            return {};
        }
        if (entry.stored_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            entry.raw_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC cell-compressed cell payload exceeds size_t");
        }
        std::vector<uint8_t> stored(static_cast<size_t>(entry.stored_bytes));
        read_physical(entry.stored_offset, stored.data(), stored.size());
        return decode_stored_payload(cid, std::move(stored));
    }

    std::string path_;
    mutable std::ifstream file_;
    std::unique_ptr<BCDirectFileReader> direct_reader_;
    uint64_t physical_size_ = 0U;
    uint64_t direct_prefix_size_ = 0U;
    uint64_t raw_logical_size_ = 0U;
    BCCellCompressedPositionFooter footer_;
    BCPositionHeader raw_header_;
    std::vector<uint8_t> metadata_;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<BCCellCompressedPositionIndexEntry> entries_;
    mutable std::unordered_map<CellId, std::vector<uint8_t>> cache_;
    mutable std::deque<CellId> cache_order_;
    mutable uint64_t cache_bytes_ = 0U;
    bool preload_cells_ = false;
    bool direct_io_ = false;
    uint32_t direct_queue_depth_ = 1U;
    uint64_t direct_max_transfer_bytes_ = kBCDirectDefaultMaxTransferBytes;
    uint64_t cache_budget_bytes_ = 256ULL * 1024ULL * 1024ULL;
    int decode_threads_ = 1;
    mutable bool preloaded_ = false;
    mutable std::vector<std::vector<uint8_t>> preloaded_payloads_;
};

} // namespace BC
