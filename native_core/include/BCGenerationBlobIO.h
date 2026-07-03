#pragma once

#include "BCCellMutableBuilder.h"
#include "BCFileIO.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace BC {

struct BCDumpRef {
    uint64_t offset = 0U;
    // Logical compact dump bytes: metadata + live bitmap bytes. This excludes
    // direct-IO padding so stats can keep reporting useful dump payload size.
    uint64_t bytes = 0U;
    uint32_t dump_generation = 0U;
    uint32_t checksum = 0U;
    uint32_t metadata_bytes = 0U;
    uint32_t bitmap_words = 0U;
    // File backends may pad between metadata and bitmap so bitmap payloads are
    // 4KB aligned and can be read directly into a builder arena.
    uint32_t bitmap_offset_bytes = 0U;
    uint64_t physical_bytes = 0U;

    [[nodiscard]] bool valid() const {
        return bytes != 0U;
    }

    [[nodiscard]] bool has_compact_layout() const {
        return metadata_bytes != 0U;
    }

    [[nodiscard]] uint32_t bitmap_relative_offset() const {
        return bitmap_offset_bytes == 0U ? metadata_bytes : bitmap_offset_bytes;
    }

    [[nodiscard]] uint64_t bitmap_bytes() const {
        return static_cast<uint64_t>(bitmap_words) * sizeof(uint64_t);
    }

    [[nodiscard]] uint64_t physical_span_bytes() const {
        return physical_bytes == 0U ? bytes : physical_bytes;
    }
};

struct BCCellBuilderDumpBuffer {
    CellId cid = 0U;
    uint32_t dump_generation = 0U;
    std::vector<uint8_t> bytes;

    [[nodiscard]] BCCellBuilderDumpView view() const {
        if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC generation blob dump buffer exceeds uint32");
        }
        return BCCellBuilderDumpView{
            cid,
            dump_generation,
            bytes.data(),
            static_cast<uint32_t>(bytes.size())
        };
    }
};

struct BCRestoredCellBuilder {
    CellId cid = 0U;
    std::unique_ptr<BCCellMutableBuilder> builder;
};

struct BCGenerationBlobIOStats {
    uint64_t append_count = 0U;
    uint64_t read_count = 0U;
    uint64_t bytes_written = 0U;
    uint64_t bytes_read = 0U;
    uint64_t backend_write_ops = 0U;
    uint64_t backend_write_bytes = 0U;
    uint64_t requested_extents = 0U;
    uint64_t coalesced_extents = 0U;
    uint64_t requested_bytes = 0U;
    uint64_t read_bytes = 0U;
    uint64_t backend_read_ops = 0U;
    uint64_t backend_read_bytes = 0U;
    double backend_write_seconds = 0.0;
    double flush_seconds = 0.0;
    double backend_read_seconds = 0.0;
    double restore_seconds = 0.0;
};

[[nodiscard]] inline double bc_generation_blob_now_seconds() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline uint32_t bc_generation_blob_checksum32(const uint8_t *data, size_t size) {
    if (data == nullptr && size != 0U) {
        throw std::invalid_argument("BC generation blob checksum pointer is null");
    }
    uint32_t hash = 2166136261U;
    for (size_t i = 0U; i < size; ++i) {
        hash ^= data[i];
        hash *= 16777619U;
    }
    return hash;
}

[[nodiscard]] inline uint32_t bc_generation_blob_checksum32_continue(
    uint32_t hash,
    const uint8_t *data,
    size_t size
) {
    if (data == nullptr && size != 0U) {
        throw std::invalid_argument("BC generation blob checksum pointer is null");
    }
    for (size_t i = 0U; i < size; ++i) {
        hash ^= data[i];
        hash *= 16777619U;
    }
    return hash;
}

[[nodiscard]] inline uint64_t bc_generation_blob_checked_add_u64(
    uint64_t lhs,
    uint64_t rhs,
    const char *label
) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

[[nodiscard]] inline uint64_t bc_generation_blob_align_up_u64(
    uint64_t value,
    uint64_t alignment
) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC generation blob alignment must be a power of two");
    }
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
        throw std::overflow_error("BC generation blob align_up overflow");
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

struct BCGenerationBlobAlignedByteDeleter {
    void operator()(uint8_t *ptr) const noexcept {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        ::operator delete[](ptr, std::align_val_t{4096U});
#endif
    }
};

using BCGenerationBlobAlignedBytes = std::unique_ptr<uint8_t[], BCGenerationBlobAlignedByteDeleter>;

[[nodiscard]] inline BCGenerationBlobAlignedBytes bc_generation_blob_allocate_aligned_bytes(uint64_t bytes) {
    if (bytes == 0U) {
        return BCGenerationBlobAlignedBytes{};
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC generation blob aligned buffer exceeds size_t");
    }
#ifdef _WIN32
    void *ptr = _aligned_malloc(static_cast<size_t>(bytes), 4096U);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
#else
    void *ptr = ::operator new[](
        static_cast<size_t>(bytes),
        std::align_val_t{4096U}
    );
#endif
    return BCGenerationBlobAlignedBytes(static_cast<uint8_t *>(ptr));
}

[[nodiscard]] inline BCCellMutableBuilder::CompactDumpHeader
bc_generation_blob_parse_compact_layout(const BCCellBuilderDump &dump) {
    if (dump.bytes.size() < BCCellMutableBuilder::compact_dump_header_bytes()) {
        throw std::invalid_argument("BC generation blob compact dump is smaller than header");
    }
    return BCCellMutableBuilder::parse_compact_dump_header(
        dump.cid,
        dump.dump_generation,
        dump.bytes.data(),
        static_cast<uint32_t>(std::min<size_t>(
            dump.bytes.size(),
            BCCellMutableBuilder::compact_dump_header_bytes()
        ))
    );
}

class IBCGenerationBlobIO {
public:
    virtual ~IBCGenerationBlobIO() = default;
    [[nodiscard]] virtual BCDumpRef append_cell_dump(const BCCellBuilderDump &dump) = 0;
    [[nodiscard]] virtual BCDumpRef append_cell_dump_streamed(
        const BCCellMutableBuilder &builder,
        uint32_t dump_generation
    ) {
        return append_cell_dump(builder.dump(dump_generation));
    }
    [[nodiscard]] virtual std::vector<BCDumpRef> append_cell_dumps(
        const std::vector<BCCellBuilderDump> &dumps
    ) {
        std::vector<BCDumpRef> refs;
        refs.reserve(dumps.size());
        for (const BCCellBuilderDump &dump : dumps) {
            refs.push_back(append_cell_dump(dump));
        }
        return refs;
    }
    [[nodiscard]] virtual std::vector<BCCellBuilderDumpBuffer> read_many(
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats = nullptr
    ) const = 0;
    [[nodiscard]] virtual std::vector<BCRestoredCellBuilder> restore_many_builders(
        const BCLut &lut,
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats = nullptr
    ) const {
        const std::vector<BCCellBuilderDumpBuffer> buffers = read_many(refs, stats);
        std::vector<BCRestoredCellBuilder> out;
        out.reserve(buffers.size());
        for (const BCCellBuilderDumpBuffer &buffer : buffers) {
            out.push_back(BCRestoredCellBuilder{
                buffer.cid,
                BCCellMutableBuilder::restore(lut, buffer.cid, buffer.view())
            });
        }
        return out;
    }
    [[nodiscard]] virtual BCCellBuilderDumpView view(CellId cid, BCDumpRef ref) const = 0;
    [[nodiscard]] virtual BCGenerationBlobIOStats stats() const = 0;
    virtual void flush_pending_appends() const {}
};

// First-stage correctness blob backend. It preserves append/read semantics and
// latest-dump refs, but stores bytes in memory so unit tests do not need a temp
// file. Production FamilyChain should use BCFileGenerationBlobIO below so
// dumped mutable cells are not O(N) resident.
class BCGenerationBlobIO final : public IBCGenerationBlobIO {
public:
    [[nodiscard]] BCDumpRef append_cell_dump(const BCCellBuilderDump &dump) override {
        if (dump.bytes.empty()) {
            throw std::invalid_argument("BC generation blob cannot append empty dump");
        }
        const uint64_t offset = blob_.size();
        if (offset > std::numeric_limits<uint64_t>::max() - dump.bytes.size()) {
            throw std::overflow_error("BC generation blob offset overflow");
        }
        blob_.insert(blob_.end(), dump.bytes.begin(), dump.bytes.end());
        ++append_count_;
        bytes_written_ += dump.bytes.size();
        const BCCellMutableBuilder::CompactDumpHeader layout =
            bc_generation_blob_parse_compact_layout(dump);
        return BCDumpRef{
            offset,
            static_cast<uint64_t>(dump.bytes.size()),
            dump.dump_generation,
            bc_generation_blob_checksum32(dump.bytes.data(), dump.bytes.size()),
            layout.metadata_bytes,
            layout.bitmap_words,
            layout.metadata_bytes,
            static_cast<uint64_t>(dump.bytes.size())
        };
    }

    [[nodiscard]] BCDumpRef append_cell_dump_streamed(
        const BCCellMutableBuilder &builder,
        uint32_t dump_generation
    ) override {
        return append_cell_dump(builder.dump(dump_generation));
    }

    [[nodiscard]] BCCellBuilderDumpView view(CellId cid, BCDumpRef ref) const override {
        if (!ref.valid()) {
            throw std::invalid_argument("BC generation blob dump ref is invalid");
        }
        if (ref.offset > blob_.size() || ref.bytes > blob_.size() - ref.offset) {
            throw std::out_of_range("BC generation blob dump ref exceeds blob size");
        }
        const uint8_t *data = blob_.data() + static_cast<size_t>(ref.offset);
        if (bc_generation_blob_checksum32(data, static_cast<size_t>(ref.bytes)) != ref.checksum) {
            throw std::runtime_error("BC generation blob dump checksum mismatch");
        }
        ++read_count_;
        bytes_read_ += ref.bytes;
        if (ref.bytes > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC generation blob dump view exceeds uint32");
        }
        return BCCellBuilderDumpView{
            cid,
            ref.dump_generation,
            data,
            static_cast<uint32_t>(ref.bytes)
        };
    }

    [[nodiscard]] std::vector<BCCellBuilderDumpBuffer> read_many(
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }
        std::vector<BCCellBuilderDumpBuffer> out;
        out.reserve(refs.size());
        for (const auto &[cid, ref] : refs) {
            if (!ref.valid()) {
                throw std::invalid_argument("BC generation blob dump ref is invalid");
            }
            if (ref.offset > blob_.size() || ref.bytes > blob_.size() - ref.offset) {
                throw std::out_of_range("BC generation blob dump ref exceeds blob size");
            }
            if (ref.bytes > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC generation blob dump view exceeds uint32");
            }
            const uint8_t *data = blob_.data() + static_cast<size_t>(ref.offset);
            if (bc_generation_blob_checksum32(data, static_cast<size_t>(ref.bytes)) != ref.checksum) {
                throw std::runtime_error("BC generation blob dump checksum mismatch");
            }
            BCCellBuilderDumpBuffer buffer;
            buffer.cid = cid;
            buffer.dump_generation = ref.dump_generation;
            buffer.bytes.assign(data, data + static_cast<size_t>(ref.bytes));
            out.push_back(std::move(buffer));
            ++read_count_;
            bytes_read_ += ref.bytes;
            if (stats != nullptr) {
                ++stats->requested_extents;
                ++stats->coalesced_extents;
                ++stats->read_count;
                stats->requested_bytes += ref.bytes;
                stats->read_bytes += ref.bytes;
                stats->bytes_read += ref.bytes;
            }
        }
        return out;
    }

    [[nodiscard]] uint64_t bytes() const {
        return blob_.size();
    }

    [[nodiscard]] uint64_t append_count() const {
        return append_count_;
    }

    [[nodiscard]] uint64_t read_count() const {
        return read_count_;
    }

    [[nodiscard]] uint64_t bytes_written() const {
        return bytes_written_;
    }

    [[nodiscard]] uint64_t bytes_read() const {
        return bytes_read_;
    }

    [[nodiscard]] BCGenerationBlobIOStats stats() const override {
        BCGenerationBlobIOStats out;
        out.append_count = append_count_;
        out.read_count = read_count_;
        out.bytes_written = bytes_written_;
        out.bytes_read = bytes_read_;
        return out;
    }

private:
    std::vector<uint8_t> blob_;
    uint64_t append_count_ = 0U;
    mutable uint64_t read_count_ = 0U;
    uint64_t bytes_written_ = 0U;
    mutable uint64_t bytes_read_ = 0U;
};

// File-backed append-only generation blob. It is the production-direction
// backend for FamilyChain mutable dumps: only the latest BCDumpRef is kept in a
// cell header, while dump bytes live in BCWritableFile/BCReadableFile storage.
// view() reads a single dump into an internal scratch buffer; that view remains
// valid only until the next view()/read_many() call on this object.
class BCFileGenerationBlobIO final : public IBCGenerationBlobIO {
public:
    explicit BCFileGenerationBlobIO(
        BCWritableFile &writer,
        const BCReadableFile *reader = nullptr,
        uint64_t staging_bytes = 16ULL * 1024ULL * 1024ULL,
        bool verify_checksums = true
    ) : writer_(&writer),
        reader_(reader),
        staging_bytes_(staging_bytes),
        verify_checksums_(verify_checksums) {
        if (staging_bytes_ == 0U) {
            throw std::invalid_argument("BC file generation blob staging_bytes must be non-zero");
        }
        if (writer_->mode() == BCFileIOMode::Direct &&
            (staging_bytes_ & (kDirectBitmapAlignment - 1U)) != 0U) {
            throw std::invalid_argument("BC direct file generation blob staging_bytes must be 4KB aligned");
        }
        append_stage_ = bc_generation_blob_allocate_aligned_bytes(staging_bytes_);
        append_stage_file_offset_ = 0U;
    }

    void set_reader(const BCReadableFile &reader) {
        flush_pending_appends();
        reader_ = &reader;
    }

    [[nodiscard]] BCDumpRef append_cell_dump(const BCCellBuilderDump &dump) override {
        if (dump.bytes.empty()) {
            throw std::invalid_argument("BC file generation blob cannot append empty dump");
        }
        if (writer_ == nullptr) {
            throw std::logic_error("BC file generation blob writer is null");
        }
        const BCCellMutableBuilder::CompactDumpHeader layout =
            bc_generation_blob_parse_compact_layout(dump);
        const uint64_t bitmap_bytes = static_cast<uint64_t>(layout.bitmap_words) * sizeof(uint64_t);
        if (layout.total_bytes != dump.bytes.size()) {
            throw std::invalid_argument("BC file generation blob compact dump size mismatch");
        }
        const uint64_t offset = append_offset_;
        uint64_t bitmap_relative_offset = layout.metadata_bytes;
        uint64_t physical_bytes = dump.bytes.size();
        if (writer_->mode() == BCFileIOMode::Direct) {
            if (bitmap_bytes != 0U) {
                const uint64_t absolute_bitmap_offset =
                    bc_generation_blob_align_up_u64(offset + layout.metadata_bytes, kDirectBitmapAlignment);
                bitmap_relative_offset = absolute_bitmap_offset - offset;
                physical_bytes = bitmap_relative_offset +
                    bc_generation_blob_align_up_u64(bitmap_bytes, kDirectBitmapAlignment);
            } else {
                physical_bytes = bc_generation_blob_align_up_u64(layout.metadata_bytes, kDirectBitmapAlignment);
            }
            if (bitmap_relative_offset > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC file generation blob direct bitmap offset exceeds uint32");
            }
        }
        append_offset_ = bc_generation_blob_checked_add_u64(
            append_offset_,
            physical_bytes,
            "BC file generation blob append offset overflow"
        );
        ensure_append_file_capacity(append_offset_);
        append_stage_bytes(dump.bytes.data(), layout.metadata_bytes);
        if (bitmap_relative_offset > layout.metadata_bytes) {
            append_stage_zero(bitmap_relative_offset - layout.metadata_bytes);
        }
        if (bitmap_bytes != 0U) {
            append_stage_bytes(dump.bytes.data() + layout.metadata_bytes, bitmap_bytes);
        }
        if (physical_bytes > bitmap_relative_offset + bitmap_bytes) {
            append_stage_zero(physical_bytes - bitmap_relative_offset - bitmap_bytes);
        }
        ++stats_.append_count;
        stats_.bytes_written = bc_generation_blob_checked_add_u64(
            stats_.bytes_written,
            dump.bytes.size(),
            "BC file generation blob written byte stats overflow"
        );
        return BCDumpRef{
            offset,
            static_cast<uint64_t>(dump.bytes.size()),
            dump.dump_generation,
            verify_checksums_ ? bc_generation_blob_checksum32(dump.bytes.data(), dump.bytes.size()) : 0U,
            layout.metadata_bytes,
            layout.bitmap_words,
            static_cast<uint32_t>(bitmap_relative_offset),
            physical_bytes
        };
    }

    [[nodiscard]] BCDumpRef append_cell_dump_streamed(
        const BCCellMutableBuilder &builder,
        uint32_t dump_generation
    ) override {
        if (writer_ == nullptr) {
            throw std::logic_error("BC file generation blob writer is null");
        }
        const uint64_t offset = append_offset_;
        uint32_t checksum = 2166136261U;
        uint64_t logical_bytes_written = 0U;
        uint64_t bitmap_bytes_written = 0U;
        uint32_t metadata_bytes = 0U;
        uint32_t bitmap_words = 0U;
        uint32_t bitmap_relative_offset = 0U;
        uint64_t physical_bytes = 0U;
        bool metadata_emitted = false;

        auto emit_metadata = [&](const void *data, uint64_t bytes) {
            if (metadata_emitted) {
                throw std::logic_error("BC file generation blob streamed dump emitted metadata twice");
            }
            if (data == nullptr && bytes != 0U) {
                throw std::invalid_argument("BC file generation blob streamed metadata pointer is null");
            }
            metadata_emitted = true;
            metadata_bytes = checked_u32_blob(bytes, "BC file generation blob streamed metadata exceeds uint32");
            bitmap_relative_offset = metadata_bytes;
            if (writer_->mode() == BCFileIOMode::Direct) {
                const uint64_t absolute_bitmap_offset =
                    bc_generation_blob_align_up_u64(offset + metadata_bytes, kDirectBitmapAlignment);
                bitmap_relative_offset = checked_u32_blob(
                    absolute_bitmap_offset - offset,
                    "BC file generation blob streamed direct bitmap offset exceeds uint32"
                );
            }
            if (verify_checksums_) {
                checksum = bc_generation_blob_checksum32_continue(
                    checksum,
                    static_cast<const uint8_t *>(data),
                    static_cast<size_t>(bytes)
                );
            }
            append_stage_bytes(data, bytes);
            if (bitmap_relative_offset > metadata_bytes) {
                append_stage_zero(bitmap_relative_offset - metadata_bytes);
            }
            logical_bytes_written = bc_generation_blob_checked_add_u64(
                logical_bytes_written,
                bytes,
                "BC file generation blob streamed logical byte overflow"
            );
        };

        auto emit_bitmap = [&](const void *data, uint64_t bytes) {
            if (!metadata_emitted) {
                throw std::logic_error("BC file generation blob streamed dump emitted bitmap before metadata");
            }
            if (data == nullptr && bytes != 0U) {
                throw std::invalid_argument("BC file generation blob streamed bitmap pointer is null");
            }
            if (verify_checksums_) {
                checksum = bc_generation_blob_checksum32_continue(
                    checksum,
                    static_cast<const uint8_t *>(data),
                    static_cast<size_t>(bytes)
                );
            }
            append_stage_bytes(data, bytes);
            bitmap_bytes_written = bc_generation_blob_checked_add_u64(
                bitmap_bytes_written,
                bytes,
                "BC file generation blob streamed bitmap byte overflow"
            );
            logical_bytes_written = bc_generation_blob_checked_add_u64(
                logical_bytes_written,
                bytes,
                "BC file generation blob streamed logical byte overflow"
            );
        };

        const BCCellBuilderDumpStreamLayout layout =
            builder.dump_streamed_into(dump_generation, stream_dump_scratch_, emit_metadata, emit_bitmap);
        if (!metadata_emitted) {
            throw std::logic_error("BC file generation blob streamed dump emitted no metadata");
        }
        bitmap_words = layout.bitmap_words;
        const uint64_t bitmap_bytes = static_cast<uint64_t>(bitmap_words) * sizeof(uint64_t);
        if (layout.metadata_bytes != metadata_bytes ||
            bitmap_bytes_written != bitmap_bytes ||
            logical_bytes_written != layout.total_bytes) {
            throw std::logic_error("BC file generation blob streamed dump byte accounting mismatch");
        }
        physical_bytes = bitmap_relative_offset + bitmap_bytes;
        if (writer_->mode() == BCFileIOMode::Direct) {
            physical_bytes = bc_generation_blob_align_up_u64(physical_bytes, kDirectBitmapAlignment);
        }
        if (physical_bytes > bitmap_relative_offset + bitmap_bytes) {
            append_stage_zero(physical_bytes - bitmap_relative_offset - bitmap_bytes);
        }
        append_offset_ = bc_generation_blob_checked_add_u64(
            append_offset_,
            physical_bytes,
            "BC file generation blob streamed append offset overflow"
        );
        ensure_append_file_capacity(append_offset_);
        ++stats_.append_count;
        stats_.bytes_written = bc_generation_blob_checked_add_u64(
            stats_.bytes_written,
            layout.total_bytes,
            "BC file generation blob streamed written byte stats overflow"
        );
        return BCDumpRef{
            offset,
            layout.total_bytes,
            dump_generation,
            verify_checksums_ ? checksum : 0U,
            metadata_bytes,
            bitmap_words,
            bitmap_relative_offset,
            physical_bytes
        };
    }

    [[nodiscard]] std::vector<BCDumpRef> append_cell_dumps(
        const std::vector<BCCellBuilderDump> &dumps
    ) override {
        return IBCGenerationBlobIO::append_cell_dumps(dumps);
    }

    [[nodiscard]] BCCellBuilderDumpView view(CellId cid, BCDumpRef ref) const override {
        if (reader_ == nullptr) {
            throw std::logic_error("BC file generation blob reader is null");
        }
        if (!ref.valid()) {
            throw std::invalid_argument("BC file generation blob view saw invalid ref");
        }
        if (ref.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC file generation blob view dump exceeds size_t");
        }
        flush_pending_appends();
        scratch_.assign(static_cast<size_t>(ref.bytes), 0U);
        std::vector<BCFileReadRequest> reads;
        reads.reserve(2U);
        if (ref.has_compact_layout() &&
            (ref.bitmap_relative_offset() != ref.metadata_bytes ||
             ref.physical_span_bytes() != ref.bytes)) {
            const uint64_t bitmap_bytes = ref.bitmap_bytes();
            if (ref.metadata_bytes > ref.bytes ||
                ref.bytes - ref.metadata_bytes != bitmap_bytes ||
                ref.bitmap_relative_offset() < ref.metadata_bytes ||
                ref.physical_span_bytes() < ref.bitmap_relative_offset() + bitmap_bytes) {
                throw std::invalid_argument("BC file generation blob view split ref is invalid");
            }
            reads.push_back(BCFileReadRequest{
                ref.offset,
                scratch_.data(),
                ref.metadata_bytes
            });
            if (bitmap_bytes != 0U) {
                reads.push_back(BCFileReadRequest{
                    ref.offset + ref.bitmap_relative_offset(),
                    scratch_.data() + ref.metadata_bytes,
                    bitmap_bytes
                });
            }
        } else {
            reads.push_back(BCFileReadRequest{
                ref.offset,
                scratch_.data(),
                ref.bytes
            });
        }
        BCFileIOStats io_stats;
        const double read_begin = bc_generation_blob_now_seconds();
        reader_->read_many(reads, &io_stats);
        const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
        stats_.backend_read_seconds += read_seconds;
        ++stats_.read_count;
        stats_.bytes_read = bc_generation_blob_checked_add_u64(
            stats_.bytes_read,
            ref.bytes,
            "BC file generation blob view read byte stats overflow"
        );
        if (verify_checksums_ &&
            bc_generation_blob_checksum32(scratch_.data(), scratch_.size()) != ref.checksum) {
            throw std::runtime_error("BC file generation blob view checksum mismatch");
        }
        if (scratch_.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC file generation blob view exceeds uint32");
        }
        return BCCellBuilderDumpView{
            cid,
            ref.dump_generation,
            scratch_.data(),
            static_cast<uint32_t>(scratch_.size())
        };
    }

    [[nodiscard]] std::vector<BCCellBuilderDumpBuffer> read_many(
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats = nullptr
    ) const override {
        if (reader_ == nullptr) {
            throw std::logic_error("BC file generation blob reader is null");
        }
        flush_pending_appends();
        if (stats != nullptr) {
            *stats = {};
        }
        struct Request {
            CellId cid = 0U;
            BCDumpRef ref = {};
            size_t index = 0U;
        };
        std::vector<Request> requests;
        requests.reserve(refs.size());
        for (size_t i = 0U; i < refs.size(); ++i) {
            const auto &[cid, ref] = refs[i];
            if (!ref.valid()) {
                throw std::invalid_argument("BC file generation blob read_many saw invalid ref");
            }
            if (ref.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC file generation blob dump exceeds size_t");
            }
            requests.push_back(Request{cid, ref, i});
            accumulate(stats, &BCGenerationBlobIOStats::requested_extents, 1U);
            accumulate(stats, &BCGenerationBlobIOStats::requested_bytes, ref.bytes);
        }
        const bool has_split_layout = std::any_of(
            requests.begin(),
            requests.end(),
            [](const Request &request) {
                return request.ref.has_compact_layout() &&
                    (request.ref.bitmap_relative_offset() != request.ref.metadata_bytes ||
                     request.ref.physical_span_bytes() != request.ref.bytes);
            }
        );
        if (has_split_layout) {
            std::vector<BCCellBuilderDumpBuffer> out(refs.size());
            std::vector<BCFileReadRequest> io_requests;
            io_requests.reserve(requests.size() * 2U);
            for (const Request &request : requests) {
                if (!request.ref.has_compact_layout()) {
                    throw std::invalid_argument("BC file generation blob cannot mix whole-dump and split-layout refs");
                }
                const uint64_t bitmap_bytes = request.ref.bitmap_bytes();
                if (request.ref.metadata_bytes > request.ref.bytes ||
                    request.ref.bytes - request.ref.metadata_bytes != bitmap_bytes ||
                    request.ref.bitmap_relative_offset() < request.ref.metadata_bytes ||
                    request.ref.physical_span_bytes() < request.ref.bitmap_relative_offset() + bitmap_bytes) {
                    throw std::invalid_argument("BC file generation blob split ref is invalid");
                }
                BCCellBuilderDumpBuffer &buffer = out[request.index];
                buffer.cid = request.cid;
                buffer.dump_generation = request.ref.dump_generation;
                buffer.bytes.assign(static_cast<size_t>(request.ref.bytes), 0U);
                io_requests.push_back(BCFileReadRequest{
                    request.ref.offset,
                    buffer.bytes.data(),
                    request.ref.metadata_bytes
                });
                if (bitmap_bytes != 0U) {
                    io_requests.push_back(BCFileReadRequest{
                        request.ref.offset + request.ref.bitmap_relative_offset(),
                        buffer.bytes.data() + request.ref.metadata_bytes,
                        bitmap_bytes
                    });
                }
                accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, bitmap_bytes == 0U ? 1U : 2U);
                accumulate(stats, &BCGenerationBlobIOStats::read_bytes, request.ref.bytes);
            }
            BCFileIOStats io_stats;
            if (!io_requests.empty()) {
                const double read_begin = bc_generation_blob_now_seconds();
                reader_->read_many(io_requests, &io_stats);
                const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
                stats_.backend_read_seconds += read_seconds;
                if (stats != nullptr) {
                    stats->backend_read_seconds += read_seconds;
                }
            }
            if (stats != nullptr) {
                stats->backend_read_ops = io_stats.backend_io_count;
                stats->backend_read_bytes = io_stats.backend_bytes;
            }
            for (const Request &request : requests) {
                BCCellBuilderDumpBuffer &buffer = out[request.index];
                if (verify_checksums_ &&
                    bc_generation_blob_checksum32(buffer.bytes.data(), buffer.bytes.size()) != request.ref.checksum) {
                    throw std::runtime_error("BC file generation blob checksum mismatch");
                }
                ++stats_.read_count;
                stats_.bytes_read = bc_generation_blob_checked_add_u64(
                    stats_.bytes_read,
                    request.ref.bytes,
                    "BC file generation blob read byte stats overflow"
                );
                accumulate(stats, &BCGenerationBlobIOStats::read_count, 1U);
                accumulate(stats, &BCGenerationBlobIOStats::bytes_read, request.ref.bytes);
            }
            return out;
        }

        constexpr size_t kNoCoalesceDirectReadMaxRequests = 16U;
        if (requests.size() <= kNoCoalesceDirectReadMaxRequests) {
            std::vector<BCCellBuilderDumpBuffer> out(refs.size());
            std::vector<BCFileReadRequest> io_requests;
            io_requests.reserve(requests.size());
            for (const Request &request : requests) {
                BCCellBuilderDumpBuffer &buffer = out[request.index];
                buffer.cid = request.cid;
                buffer.dump_generation = request.ref.dump_generation;
                buffer.bytes.assign(static_cast<size_t>(request.ref.bytes), 0U);
                io_requests.push_back(BCFileReadRequest{
                    request.ref.offset,
                    buffer.bytes.data(),
                    request.ref.bytes
                });
                accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, 1U);
                accumulate(stats, &BCGenerationBlobIOStats::read_bytes, request.ref.bytes);
            }
            BCFileIOStats io_stats;
            if (!io_requests.empty()) {
                const double read_begin = bc_generation_blob_now_seconds();
                reader_->read_many(io_requests, &io_stats);
                const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
                stats_.backend_read_seconds += read_seconds;
                if (stats != nullptr) {
                    stats->backend_read_seconds += read_seconds;
                }
            }
            if (stats != nullptr) {
                stats->backend_read_ops = io_stats.backend_io_count;
                stats->backend_read_bytes = io_stats.backend_bytes;
            }
            for (const Request &request : requests) {
                BCCellBuilderDumpBuffer &buffer = out[request.index];
                if (verify_checksums_ &&
                    bc_generation_blob_checksum32(buffer.bytes.data(), buffer.bytes.size()) != request.ref.checksum) {
                    throw std::runtime_error("BC file generation blob checksum mismatch");
                }
                ++stats_.read_count;
                stats_.bytes_read = bc_generation_blob_checked_add_u64(
                    stats_.bytes_read,
                    request.ref.bytes,
                    "BC file generation blob read byte stats overflow"
                );
                accumulate(stats, &BCGenerationBlobIOStats::read_count, 1U);
                accumulate(stats, &BCGenerationBlobIOStats::bytes_read, request.ref.bytes);
            }
            return out;
        }
        std::vector<size_t> request_order(requests.size());
        std::iota(request_order.begin(), request_order.end(), size_t{0U});
        std::sort(request_order.begin(), request_order.end(), [&](size_t lhs, size_t rhs) {
            return requests[lhs].ref.offset < requests[rhs].ref.offset;
        });

        struct Range {
            uint64_t offset = 0U;
            uint64_t bytes = 0U;
            std::vector<uint8_t> data;
        };
        std::vector<Range> ranges;
        for (size_t request_index : request_order) {
            const Request &request = requests[request_index];
            const uint64_t end = bc_generation_blob_checked_add_u64(
                request.ref.offset,
                request.ref.bytes,
                "BC file generation blob request end overflow"
            );
            if (!ranges.empty()) {
                Range &last = ranges.back();
                const uint64_t last_end = bc_generation_blob_checked_add_u64(
                    last.offset,
                    last.bytes,
                    "BC file generation blob coalesced range end overflow"
                );
                if (request.ref.offset >= last.offset && request.ref.offset <= last_end) {
                    if (end > last_end) {
                        last.bytes = end - last.offset;
                    }
                    continue;
                }
            }
            ranges.push_back(Range{request.ref.offset, request.ref.bytes, {}});
        }
        std::vector<BCFileReadRequest> io_requests;
        io_requests.reserve(ranges.size());
        for (Range &range : ranges) {
            if (range.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC file generation blob coalesced range exceeds size_t");
            }
            range.data.assign(static_cast<size_t>(range.bytes), 0U);
            io_requests.push_back(BCFileReadRequest{
                range.offset,
                range.data.data(),
                range.bytes
            });
            accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, 1U);
            accumulate(stats, &BCGenerationBlobIOStats::read_bytes, range.bytes);
        }
        BCFileIOStats io_stats;
        const double read_begin = bc_generation_blob_now_seconds();
        reader_->read_many(io_requests, &io_stats);
        const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
        stats_.backend_read_seconds += read_seconds;
        if (stats != nullptr) {
            stats->backend_read_seconds += read_seconds;
        }
        if (stats != nullptr) {
            stats->backend_read_ops = io_stats.backend_io_count;
            stats->backend_read_bytes = io_stats.backend_bytes;
        }

        std::vector<BCCellBuilderDumpBuffer> out(refs.size());
        for (const Request &request : requests) {
            const Range &range = find_range(ranges, request.ref.offset, request.ref.bytes);
            const uint64_t in_range = request.ref.offset - range.offset;
            BCCellBuilderDumpBuffer buffer;
            buffer.cid = request.cid;
            buffer.dump_generation = request.ref.dump_generation;
            buffer.bytes.assign(
                range.data.begin() + static_cast<std::ptrdiff_t>(in_range),
                range.data.begin() + static_cast<std::ptrdiff_t>(in_range + request.ref.bytes)
            );
            if (verify_checksums_ &&
                bc_generation_blob_checksum32(buffer.bytes.data(), buffer.bytes.size()) != request.ref.checksum) {
                throw std::runtime_error("BC file generation blob checksum mismatch");
            }
            out[request.index] = std::move(buffer);
            ++stats_.read_count;
            stats_.bytes_read = bc_generation_blob_checked_add_u64(
                stats_.bytes_read,
                request.ref.bytes,
                "BC file generation blob read byte stats overflow"
            );
            accumulate(stats, &BCGenerationBlobIOStats::read_count, 1U);
            accumulate(stats, &BCGenerationBlobIOStats::bytes_read, request.ref.bytes);
        }
        return out;
    }

    [[nodiscard]] std::vector<BCRestoredCellBuilder> restore_many_builders(
        const BCLut &lut,
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats = nullptr
    ) const override {
        if (reader_ == nullptr) {
            throw std::logic_error("BC file generation blob reader is null");
        }
        if (stats != nullptr) {
            *stats = {};
        }
        auto staged = restore_many_builders_from_append_stage(lut, refs, stats);
        if (!staged.empty() || refs.empty()) {
            return staged;
        }
        flush_pending_appends();
        struct Request {
            CellId cid = 0U;
            BCDumpRef ref = {};
            size_t index = 0U;
            size_t range_index = 0U;
            BCCellMutableBuilder::CompactDumpHeader header = {};
            std::vector<uint8_t> metadata;
            std::unique_ptr<BCCellMutableBuilder> builder;
        };
        std::vector<Request> requests;
        requests.reserve(refs.size());
        for (size_t i = 0U; i < refs.size(); ++i) {
            const auto &[cid, ref] = refs[i];
            if (!ref.valid()) {
                throw std::invalid_argument("BC file generation blob restore_many saw invalid ref");
            }
            if (ref.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC file generation blob dump exceeds size_t");
            }
            requests.push_back(Request{cid, ref, i});
            accumulate(stats, &BCGenerationBlobIOStats::requested_extents, 1U);
            accumulate(stats, &BCGenerationBlobIOStats::requested_bytes, ref.bytes);
        }
        std::sort(requests.begin(), requests.end(), [](const Request &lhs, const Request &rhs) {
            return lhs.ref.offset < rhs.ref.offset;
        });

        if (requests.empty()) {
            return {};
        }

        if (reader_->mode() == BCFileIOMode::Direct) {
            constexpr uint64_t kDirectWholeDumpReadMaxBytes = 256ULL * 1024ULL;
            bool can_read_whole_dumps = true;
            for (const Request &request : requests) {
                const BCDumpRef &ref = request.ref;
                const uint64_t bitmap_bytes = ref.bitmap_bytes();
                if (!ref.has_compact_layout() ||
                    ref.metadata_bytes < BCCellMutableBuilder::compact_dump_header_bytes() ||
                    ref.metadata_bytes > ref.bytes ||
                    ref.bytes - ref.metadata_bytes != bitmap_bytes ||
                    ref.bitmap_relative_offset() < ref.metadata_bytes ||
                    ref.physical_span_bytes() < ref.bitmap_relative_offset() + bitmap_bytes ||
                    ref.physical_span_bytes() > kDirectWholeDumpReadMaxBytes ||
                    (ref.offset & (kDirectBitmapAlignment - 1U)) != 0U ||
                    (ref.physical_span_bytes() & (kDirectBitmapAlignment - 1U)) != 0U) {
                    can_read_whole_dumps = false;
                    break;
                }
            }
            if (can_read_whole_dumps) {
                struct DirectRange {
                    uint64_t offset = 0U;
                    uint64_t bytes = 0U;
                    BCCellAlignedBytePtr data;
                };
                std::vector<DirectRange> ranges;
                for (Request &request : requests) {
                    ranges.push_back(DirectRange{request.ref.offset, request.ref.physical_span_bytes(), {}});
                    request.range_index = ranges.size() - 1U;
                }
                std::vector<BCFileReadRequest> io_requests;
                io_requests.reserve(ranges.size());
                for (DirectRange &range : ranges) {
                    if ((range.offset & (kDirectBitmapAlignment - 1U)) != 0U ||
                        (range.bytes & (kDirectBitmapAlignment - 1U)) != 0U) {
                        throw std::logic_error("BC file generation blob direct whole restore range is unaligned");
                    }
                    range.data = bc_allocate_cell_aligned_bytes(range.bytes);
                    io_requests.push_back(BCFileReadRequest{
                        range.offset,
                        range.data.get(),
                        range.bytes
                    });
                }
                BCFileIOStats io_stats;
                const double read_begin = bc_generation_blob_now_seconds();
                reader_->read_many(io_requests, &io_stats);
                const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
                stats_.backend_read_seconds += read_seconds;
                if (stats != nullptr) {
                    stats->backend_read_seconds += read_seconds;
                }

                std::vector<BCRestoredCellBuilder> out(refs.size());
                std::vector<uint8_t> ok(requests.size(), 1U);
                std::vector<std::string> errors(requests.size());
                const double restore_begin = bc_generation_blob_now_seconds();
#pragma omp parallel for schedule(dynamic, 1)
                for (int64_t request_index = 0; request_index < static_cast<int64_t>(requests.size()); ++request_index) {
                    const Request &request = requests[static_cast<size_t>(request_index)];
                    try {
                        if (request.range_index >= ranges.size()) {
                            throw std::logic_error("BC file generation blob direct whole restore range index is invalid");
                        }
                        DirectRange &range = ranges[request.range_index];
                        const uint64_t in_range = request.ref.offset - range.offset;
                        if (in_range > range.bytes ||
                            request.ref.physical_span_bytes() > range.bytes - in_range) {
                            throw std::logic_error("BC file generation blob direct whole restore range does not cover dump");
                        }
                        const uint8_t *base = range.data.get() + static_cast<size_t>(in_range);
                        const BCCellMutableBuilder::CompactDumpHeader header =
                            BCCellMutableBuilder::parse_compact_dump_header(
                                request.cid,
                                request.ref.dump_generation,
                                base,
                                request.ref.metadata_bytes
                            );
                        if (header.total_bytes != request.ref.bytes ||
                            header.bitmap_words != request.ref.bitmap_words) {
                            throw std::invalid_argument("BC file generation blob whole direct dump metadata mismatch");
                        }
                        std::unique_ptr<BCCellMutableBuilder> builder =
                            BCCellMutableBuilder::restore_compact_metadata_for_direct_bitmap(
                                lut,
                                request.cid,
                                request.ref.dump_generation,
                                base,
                                request.ref.metadata_bytes,
                                true,
                                verify_checksums_
                            );
                        const uint64_t bitmap_bytes =
                            static_cast<uint64_t>(header.bitmap_words) * sizeof(uint64_t);
                        if (verify_checksums_) {
                            uint32_t hash = 2166136261U;
                            hash = bc_generation_blob_checksum32_continue(
                                hash,
                                base,
                                request.ref.metadata_bytes
                            );
                            if (bitmap_bytes != 0U) {
                                hash = bc_generation_blob_checksum32_continue(
                                    hash,
                                    base + request.ref.bitmap_relative_offset(),
                                    static_cast<size_t>(bitmap_bytes)
                                );
                            }
                            if (hash != request.ref.checksum) {
                                throw std::runtime_error("BC file generation blob checksum mismatch");
                            }
                        }
                        if (bitmap_bytes != 0U) {
                            std::memcpy(
                                builder->direct_restore_bitmap_bytes(header.bitmap_words),
                                base + request.ref.bitmap_relative_offset(),
                                static_cast<size_t>(bitmap_bytes)
                            );
                        }
                        builder->finish_direct_bitmap_restore(true);
                        out[request.index] = BCRestoredCellBuilder{
                            request.cid,
                            std::move(builder)
                        };
                    } catch (const std::exception &ex) {
                        ok[static_cast<size_t>(request_index)] = 0U;
                        errors[static_cast<size_t>(request_index)] = ex.what();
                    } catch (...) {
                        ok[static_cast<size_t>(request_index)] = 0U;
                        errors[static_cast<size_t>(request_index)] = "unknown BC whole direct dump restore error";
                    }
                }
                for (size_t i = 0U; i < requests.size(); ++i) {
                    if (ok[i] == 0U) {
                        throw std::runtime_error(errors[i]);
                    }
                }
                const double restore_seconds = bc_generation_blob_now_seconds() - restore_begin;
                stats_.restore_seconds += restore_seconds;
                if (stats != nullptr) {
                    stats->restore_seconds += restore_seconds;
                }

                uint64_t read_bytes_sum = 0U;
                for (const Request &request : requests) {
                    ++stats_.read_count;
                    stats_.bytes_read = bc_generation_blob_checked_add_u64(
                        stats_.bytes_read,
                        request.ref.bytes,
                        "BC file generation blob read byte stats overflow"
                    );
                    read_bytes_sum = bc_generation_blob_checked_add_u64(
                        read_bytes_sum,
                        request.ref.bytes,
                        "BC file generation blob read byte stats overflow"
                    );
                }
                accumulate(stats, &BCGenerationBlobIOStats::read_count, requests.size());
                accumulate(stats, &BCGenerationBlobIOStats::bytes_read, read_bytes_sum);
                accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, io_requests.size());
                accumulate(stats, &BCGenerationBlobIOStats::read_bytes, read_bytes_sum);
                if (stats != nullptr) {
                    stats->backend_read_ops = io_stats.backend_io_count;
                    stats->backend_read_bytes = io_stats.backend_bytes;
                }
                return out;
            }
        }

        const bool can_buffered_whole_dump_restore =
            reader_->mode() == BCFileIOMode::Buffered &&
            std::all_of(
                requests.begin(),
                requests.end(),
                [](const Request &request) {
                    const BCDumpRef &ref = request.ref;
                    return ref.has_compact_layout() &&
                           ref.bitmap_relative_offset() == ref.metadata_bytes &&
                           ref.physical_span_bytes() == ref.bytes;
                }
            );
        if (can_buffered_whole_dump_restore) {
            struct Range {
                uint64_t offset = 0U;
                uint64_t bytes = 0U;
                std::vector<uint8_t> data;
            };
            constexpr uint64_t kBufferedWholeRestoreMaxGapBytes = 64ULL * 1024ULL;
            constexpr uint64_t kBufferedWholeRestoreMaxRangeBytes = 4ULL * 1024ULL * 1024ULL;
            std::vector<Range> ranges;
            for (Request &request : requests) {
                const uint64_t end = bc_generation_blob_checked_add_u64(
                    request.ref.offset,
                    request.ref.bytes,
                    "BC file generation blob whole restore request end overflow"
                );
                if (ranges.empty()) {
                    ranges.push_back(Range{request.ref.offset, request.ref.bytes, {}});
                    request.range_index = 0U;
                    continue;
                }
                Range &last = ranges.back();
                const uint64_t last_end = bc_generation_blob_checked_add_u64(
                    last.offset,
                    last.bytes,
                    "BC file generation blob whole restore range end overflow"
                );
                const uint64_t gap = request.ref.offset > last_end ? request.ref.offset - last_end : 0U;
                if (request.ref.offset >= last.offset &&
                    (request.ref.offset <= last_end ||
                     (gap <= kBufferedWholeRestoreMaxGapBytes &&
                      end - last.offset <= kBufferedWholeRestoreMaxRangeBytes))) {
                    if (end > last_end) {
                        last.bytes = end - last.offset;
                    }
                    request.range_index = ranges.size() - 1U;
                    continue;
                }
                ranges.push_back(Range{request.ref.offset, request.ref.bytes, {}});
                request.range_index = ranges.size() - 1U;
            }

            std::vector<BCFileReadRequest> io_requests;
            io_requests.reserve(ranges.size());
            for (Range &range : ranges) {
                if (range.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                    throw std::overflow_error("BC file generation blob whole restore range exceeds size_t");
                }
                range.data.assign(static_cast<size_t>(range.bytes), 0U);
                io_requests.push_back(BCFileReadRequest{
                    range.offset,
                    range.data.data(),
                    range.bytes
                });
            }
            BCFileIOStats io_stats;
            if (!io_requests.empty()) {
                const double read_begin = bc_generation_blob_now_seconds();
                reader_->read_many(io_requests, &io_stats);
                const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
                stats_.backend_read_seconds += read_seconds;
                if (stats != nullptr) {
                    stats->backend_read_seconds += read_seconds;
                }
            }

            std::vector<BCRestoredCellBuilder> out(refs.size());
            std::vector<uint8_t> ok(requests.size(), 1U);
            std::vector<std::string> errors(requests.size());
            const double restore_begin = bc_generation_blob_now_seconds();
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t request_index = 0; request_index < static_cast<int64_t>(requests.size()); ++request_index) {
                const Request &request = requests[static_cast<size_t>(request_index)];
                try {
                    if (request.range_index >= ranges.size()) {
                        throw std::logic_error("BC file generation blob whole restore range index is invalid");
                    }
                    const Range &range = ranges[request.range_index];
                    const uint64_t in_range = request.ref.offset - range.offset;
                    if (in_range > range.data.size() ||
                        request.ref.bytes > range.data.size() - in_range) {
                        throw std::logic_error("BC file generation blob whole restore range does not cover dump");
                    }
                    const uint8_t *base = range.data.data() + static_cast<size_t>(in_range);
                    if (verify_checksums_ &&
                        bc_generation_blob_checksum32(base, static_cast<size_t>(request.ref.bytes)) != request.ref.checksum) {
                        throw std::runtime_error("BC file generation blob checksum mismatch");
                    }
                    std::unique_ptr<BCCellMutableBuilder> builder =
                        BCCellMutableBuilder::restore_compact_metadata_for_direct_bitmap(
                            lut,
                            request.cid,
                            request.ref.dump_generation,
                            base,
                            request.ref.metadata_bytes,
                            true,
                            verify_checksums_
                        );
                    const BCCellMutableBuilder::CompactDumpHeader header =
                        BCCellMutableBuilder::parse_compact_dump_header(
                            request.cid,
                            request.ref.dump_generation,
                            base,
                            request.ref.metadata_bytes
                        );
                    if (header.total_bytes != request.ref.bytes ||
                        header.metadata_bytes != request.ref.metadata_bytes ||
                        header.bitmap_words != request.ref.bitmap_words) {
                        throw std::invalid_argument("BC file generation blob whole restore metadata mismatch");
                    }
                    const uint64_t bitmap_bytes =
                        static_cast<uint64_t>(header.bitmap_words) * sizeof(uint64_t);
                    if (bitmap_bytes != 0U) {
                        std::memcpy(
                            builder->direct_restore_bitmap_bytes(header.bitmap_words),
                            base + header.metadata_bytes,
                            static_cast<size_t>(bitmap_bytes)
                        );
                    }
                    builder->finish_direct_bitmap_restore(true);
                    out[request.index] = BCRestoredCellBuilder{
                        request.cid,
                        std::move(builder)
                    };
                } catch (const std::exception &ex) {
                    ok[static_cast<size_t>(request_index)] = 0U;
                    errors[static_cast<size_t>(request_index)] = ex.what();
                } catch (...) {
                    ok[static_cast<size_t>(request_index)] = 0U;
                    errors[static_cast<size_t>(request_index)] = "unknown BC whole buffered dump restore error";
                }
            }
            for (size_t i = 0U; i < requests.size(); ++i) {
                if (ok[i] == 0U) {
                    throw std::runtime_error(errors[i]);
                }
            }
            const double restore_seconds = bc_generation_blob_now_seconds() - restore_begin;
            stats_.restore_seconds += restore_seconds;
            if (stats != nullptr) {
                stats->restore_seconds += restore_seconds;
            }

            uint64_t read_bytes_sum = 0U;
            for (const Request &request : requests) {
                ++stats_.read_count;
                stats_.bytes_read = bc_generation_blob_checked_add_u64(
                    stats_.bytes_read,
                    request.ref.bytes,
                    "BC file generation blob read byte stats overflow"
                );
                read_bytes_sum = bc_generation_blob_checked_add_u64(
                    read_bytes_sum,
                    request.ref.bytes,
                    "BC file generation blob read byte stats overflow"
                );
            }
            accumulate(stats, &BCGenerationBlobIOStats::read_count, requests.size());
            accumulate(stats, &BCGenerationBlobIOStats::bytes_read, read_bytes_sum);
            accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, ranges.size());
            uint64_t range_bytes = 0U;
            for (const Range &range : ranges) {
                range_bytes = bc_generation_blob_checked_add_u64(
                    range_bytes,
                    range.bytes,
                    "BC file generation blob whole restore range byte stats overflow"
                );
            }
            accumulate(stats, &BCGenerationBlobIOStats::read_bytes, range_bytes);
            if (stats != nullptr) {
                stats->backend_read_ops = io_stats.backend_io_count;
                stats->backend_read_bytes = io_stats.backend_bytes;
            }
            return out;
        }

        constexpr uint32_t kHeaderBytes = BCCellMutableBuilder::compact_dump_header_bytes();
        static_assert(kHeaderBytes == BCCellMutableBuilder::compact_dump_header_bytes());
        std::vector<std::array<uint8_t, kHeaderBytes>> headers(requests.size());
        std::vector<BCFileReadRequest> header_reads;
        header_reads.reserve(requests.size());
        for (size_t i = 0U; i < requests.size(); ++i) {
            if (requests[i].ref.bytes < kHeaderBytes) {
                throw std::invalid_argument("BC file generation blob dump is smaller than compact header");
            }
            if (requests[i].ref.has_compact_layout()) {
                if (requests[i].ref.metadata_bytes < kHeaderBytes ||
                    requests[i].ref.metadata_bytes > requests[i].ref.bytes) {
                    throw std::invalid_argument("BC file generation blob compact ref metadata size is invalid");
                }
                const uint64_t bitmap_bytes =
                    static_cast<uint64_t>(requests[i].ref.bitmap_words) * sizeof(uint64_t);
                if (bitmap_bytes != requests[i].ref.bytes - requests[i].ref.metadata_bytes) {
                    throw std::invalid_argument("BC file generation blob compact ref bitmap size is invalid");
                }
                if (requests[i].ref.bitmap_relative_offset() < requests[i].ref.metadata_bytes ||
                    requests[i].ref.physical_span_bytes() <
                        requests[i].ref.bitmap_relative_offset() + bitmap_bytes) {
                    throw std::invalid_argument("BC file generation blob compact ref physical layout is invalid");
                }
                continue;
            }
            header_reads.push_back(BCFileReadRequest{
                requests[i].ref.offset,
                headers[i].data(),
                kHeaderBytes
            });
        }
        BCFileIOStats header_io_stats;
        if (!header_reads.empty()) {
            const double read_begin = bc_generation_blob_now_seconds();
            reader_->read_many(header_reads, &header_io_stats);
            const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
            stats_.backend_read_seconds += read_seconds;
            if (stats != nullptr) {
                stats->backend_read_seconds += read_seconds;
            }
        }

        std::vector<BCFileReadRequest> metadata_reads;
        metadata_reads.reserve(requests.size());
        for (size_t i = 0U; i < requests.size(); ++i) {
            Request &request = requests[i];
            if (request.ref.has_compact_layout()) {
                request.metadata.assign(request.ref.metadata_bytes, 0U);
                metadata_reads.push_back(BCFileReadRequest{
                    request.ref.offset,
                    request.metadata.data(),
                    request.ref.metadata_bytes
                });
            } else {
                request.header = BCCellMutableBuilder::parse_compact_dump_header(
                    request.cid,
                    request.ref.dump_generation,
                    headers[i].data(),
                    kHeaderBytes
                );
                if (request.header.total_bytes != request.ref.bytes) {
                    throw std::invalid_argument("BC file generation blob compact dump size mismatch");
                }
                request.metadata.assign(request.header.metadata_bytes, 0U);
                std::memcpy(request.metadata.data(), headers[i].data(), kHeaderBytes);
                if (request.header.metadata_bytes == kHeaderBytes) {
                    continue;
                }
                metadata_reads.push_back(BCFileReadRequest{
                    request.ref.offset + kHeaderBytes,
                    request.metadata.data() + kHeaderBytes,
                    request.header.metadata_bytes - kHeaderBytes
                });
            }
        }
        BCFileIOStats metadata_io_stats;
        if (!metadata_reads.empty()) {
            const double read_begin = bc_generation_blob_now_seconds();
            reader_->read_many(metadata_reads, &metadata_io_stats);
            const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
            stats_.backend_read_seconds += read_seconds;
            if (stats != nullptr) {
                stats->backend_read_seconds += read_seconds;
            }
        }

        for (Request &request : requests) {
            request.header = BCCellMutableBuilder::parse_compact_dump_header(
                request.cid,
                request.ref.dump_generation,
                request.metadata.data(),
                static_cast<uint32_t>(request.metadata.size())
            );
            if (request.header.total_bytes != request.ref.bytes) {
                throw std::invalid_argument("BC file generation blob compact dump size mismatch");
            }
        }

        std::vector<uint8_t> ok(requests.size(), 1U);
        std::vector<std::string> errors(requests.size());
        const double metadata_restore_begin = bc_generation_blob_now_seconds();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t request_index = 0; request_index < static_cast<int64_t>(requests.size()); ++request_index) {
            Request &request = requests[static_cast<size_t>(request_index)];
            try {
                request.builder = BCCellMutableBuilder::restore_compact_metadata_for_direct_bitmap(
                    lut,
                    request.cid,
                    request.ref.dump_generation,
                    request.metadata.data(),
                    static_cast<uint32_t>(request.metadata.size()),
                    true,
                    verify_checksums_
                );
            } catch (const std::exception &ex) {
                ok[static_cast<size_t>(request_index)] = 0U;
                errors[static_cast<size_t>(request_index)] = ex.what();
            } catch (...) {
                ok[static_cast<size_t>(request_index)] = 0U;
                errors[static_cast<size_t>(request_index)] = "unknown BC compact metadata restore error";
            }
        }
        for (size_t i = 0U; i < requests.size(); ++i) {
            if (ok[i] == 0U) {
                throw std::runtime_error(errors[i]);
            }
        }
        const double metadata_restore_seconds = bc_generation_blob_now_seconds() - metadata_restore_begin;
        stats_.restore_seconds += metadata_restore_seconds;
        if (stats != nullptr) {
            stats->restore_seconds += metadata_restore_seconds;
        }

        std::vector<BCFileReadRequest> bitmap_reads;
        bitmap_reads.reserve(requests.size());
        for (Request &request : requests) {
            const uint64_t bitmap_bytes =
                static_cast<uint64_t>(request.header.bitmap_words) * sizeof(uint64_t);
            if (bitmap_bytes == 0U) {
                continue;
            }
            const uint64_t bitmap_file_offset = request.ref.offset + request.ref.bitmap_relative_offset();
            uint64_t read_bytes = bitmap_bytes;
            if (reader_->mode() == BCFileIOMode::Direct) {
                const uint64_t physical_bitmap_bytes =
                    request.ref.physical_span_bytes() - request.ref.bitmap_relative_offset();
                if ((bitmap_file_offset & 4095ULL) == 0U &&
                    physical_bitmap_bytes >= bitmap_bytes &&
                    (physical_bitmap_bytes & 4095ULL) == 0U) {
                    read_bytes = physical_bitmap_bytes;
                }
            }
            bitmap_reads.push_back(BCFileReadRequest{
                bitmap_file_offset,
                request.builder->direct_restore_bitmap_bytes(request.header.bitmap_words, read_bytes),
                read_bytes
            });
        }
        BCFileIOStats bitmap_io_stats;
        if (!bitmap_reads.empty()) {
            const double read_begin = bc_generation_blob_now_seconds();
            reader_->read_many(bitmap_reads, &bitmap_io_stats);
            const double read_seconds = bc_generation_blob_now_seconds() - read_begin;
            stats_.backend_read_seconds += read_seconds;
            if (stats != nullptr) {
                stats->backend_read_seconds += read_seconds;
            }
        }

        std::vector<BCRestoredCellBuilder> out(refs.size());
        const double bitmap_restore_begin = bc_generation_blob_now_seconds();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t request_index = 0; request_index < static_cast<int64_t>(requests.size()); ++request_index) {
            Request &request = requests[static_cast<size_t>(request_index)];
            try {
                if (verify_checksums_) {
                    uint32_t hash = 2166136261U;
                    hash = bc_generation_blob_checksum32_continue(
                        hash,
                        request.metadata.data(),
                        request.metadata.size()
                    );
                    const uint64_t bitmap_bytes =
                        static_cast<uint64_t>(request.header.bitmap_words) * sizeof(uint64_t);
                    if (bitmap_bytes != 0U) {
                        hash = bc_generation_blob_checksum32_continue(
                            hash,
                            request.builder->direct_restore_bitmap_bytes(request.header.bitmap_words),
                            static_cast<size_t>(bitmap_bytes)
                        );
                    }
                    if (hash != request.ref.checksum) {
                        throw std::runtime_error("BC file generation blob checksum mismatch");
                    }
                }
                request.builder->finish_direct_bitmap_restore(true);
                out[request.index] = BCRestoredCellBuilder{
                    request.cid,
                    std::move(request.builder)
                };
            } catch (const std::exception &ex) {
                ok[static_cast<size_t>(request_index)] = 0U;
                errors[static_cast<size_t>(request_index)] = ex.what();
            } catch (...) {
                ok[static_cast<size_t>(request_index)] = 0U;
                errors[static_cast<size_t>(request_index)] = "unknown BC compact bitmap restore error";
            }
        }
        for (size_t i = 0U; i < requests.size(); ++i) {
            if (ok[i] == 0U) {
                throw std::runtime_error(errors[i]);
            }
        }
        const double bitmap_restore_seconds = bc_generation_blob_now_seconds() - bitmap_restore_begin;
        stats_.restore_seconds += bitmap_restore_seconds;
        if (stats != nullptr) {
            stats->restore_seconds += bitmap_restore_seconds;
        }

        uint64_t read_bytes_sum = 0U;
        for (const Request &request : requests) {
            ++stats_.read_count;
            stats_.bytes_read = bc_generation_blob_checked_add_u64(
                stats_.bytes_read,
                request.ref.bytes,
                "BC file generation blob read byte stats overflow"
            );
            read_bytes_sum = bc_generation_blob_checked_add_u64(
                read_bytes_sum,
                request.ref.bytes,
                "BC file generation blob read byte stats overflow"
            );
        }
        accumulate(stats, &BCGenerationBlobIOStats::read_count, requests.size());
        accumulate(stats, &BCGenerationBlobIOStats::bytes_read, read_bytes_sum);
        accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, header_reads.size());
        accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, metadata_reads.size());
        accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, bitmap_reads.size());
        accumulate(stats, &BCGenerationBlobIOStats::read_bytes, read_bytes_sum);
        if (stats != nullptr) {
            stats->backend_read_ops =
                header_io_stats.backend_io_count +
                metadata_io_stats.backend_io_count +
                bitmap_io_stats.backend_io_count;
            stats->backend_read_bytes =
                header_io_stats.backend_bytes +
                metadata_io_stats.backend_bytes +
                bitmap_io_stats.backend_bytes;
        }
        return out;
    }

    [[nodiscard]] BCGenerationBlobIOStats stats() const override {
        return stats_;
    }

    [[nodiscard]] uint64_t append_offset() const {
        return append_offset_;
    }

    void flush_pending_appends() const override {
        const bool has_work = append_stage_size_ != 0U || writer_dirty_;
        const double flush_begin = has_work ? bc_generation_blob_now_seconds() : 0.0;
        flush_append_stage();
        if (writer_dirty_) {
            writer_->flush();
            writer_dirty_ = false;
        }
        if (has_work) {
            stats_.flush_seconds += bc_generation_blob_now_seconds() - flush_begin;
        }
    }

private:
    using StatMember = uint64_t BCGenerationBlobIOStats::*;

    static void accumulate(BCGenerationBlobIOStats *stats, StatMember member, uint64_t value) {
        if (stats == nullptr) {
            return;
        }
        if (stats->*member > std::numeric_limits<uint64_t>::max() - value) {
            throw std::overflow_error("BC file generation blob stats overflow");
        }
        stats->*member += value;
    }

    template <typename Range>
    [[nodiscard]] static const Range &find_range(
        const std::vector<Range> &ranges,
        uint64_t offset,
        uint64_t bytes
    ) {
        const uint64_t end = bc_generation_blob_checked_add_u64(
            offset,
            bytes,
            "BC file generation blob find range end overflow"
        );
        for (const Range &range : ranges) {
            const uint64_t range_end = bc_generation_blob_checked_add_u64(
                range.offset,
                range.bytes,
                "BC file generation blob range end overflow"
            );
            if (offset >= range.offset && end <= range_end) {
                return range;
            }
        }
        throw std::logic_error("BC file generation blob request is not covered by read range");
    }

    static constexpr uint64_t kDirectBitmapAlignment = 4096ULL;
    static constexpr uint64_t kDirectAppendReserveBytes = 256ULL * 1024ULL * 1024ULL;

    [[nodiscard]] static uint32_t checked_u32_blob(uint64_t value, const char *label) {
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(label);
        }
        return static_cast<uint32_t>(value);
    }

    [[nodiscard]] std::vector<BCRestoredCellBuilder> restore_many_builders_from_append_stage(
        const BCLut &lut,
        const std::vector<std::pair<CellId, BCDumpRef>> &refs,
        BCGenerationBlobIOStats *stats
    ) const {
        if (refs.empty()) {
            return {};
        }
        if (append_stage_size_ == 0U) {
            return {};
        }
        const uint64_t stage_begin = append_stage_file_offset_;
        const uint64_t stage_end = bc_generation_blob_checked_add_u64(
            append_stage_file_offset_,
            append_stage_size_,
            "BC file generation blob append stage end overflow"
        );
        for (const auto &[cid, ref] : refs) {
            (void)cid;
            if (!ref.valid() || !ref.has_compact_layout()) {
                return {};
            }
            const uint64_t ref_end = bc_generation_blob_checked_add_u64(
                ref.offset,
                ref.physical_span_bytes(),
                "BC file generation blob staged restore ref end overflow"
            );
            if (ref.offset < stage_begin || ref_end > stage_end) {
                return {};
            }
        }

        for (const auto &[cid, ref] : refs) {
            (void)cid;
            accumulate(stats, &BCGenerationBlobIOStats::requested_extents, 1U);
            accumulate(stats, &BCGenerationBlobIOStats::requested_bytes, ref.bytes);
        }

        const double restore_begin = bc_generation_blob_now_seconds();
        std::vector<BCRestoredCellBuilder> out(refs.size());
        std::vector<uint8_t> ok(refs.size(), 1U);
        std::vector<std::string> errors(refs.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t index = 0; index < static_cast<int64_t>(refs.size()); ++index) {
            const auto &[cid, ref] = refs[static_cast<size_t>(index)];
            try {
                const uint64_t in_stage = ref.offset - stage_begin;
                const uint8_t *base = append_stage_.get() + static_cast<size_t>(in_stage);
                const BCCellMutableBuilder::CompactDumpHeader header =
                    BCCellMutableBuilder::parse_compact_dump_header(
                        cid,
                        ref.dump_generation,
                        base,
                        ref.metadata_bytes
                    );
                if (header.total_bytes != ref.bytes ||
                    header.bitmap_words != ref.bitmap_words) {
                    throw std::invalid_argument("BC file generation blob staged dump metadata mismatch");
                }
                std::unique_ptr<BCCellMutableBuilder> builder =
                    BCCellMutableBuilder::restore_compact_metadata_for_direct_bitmap(
                        lut,
                        cid,
                        ref.dump_generation,
                        base,
                        ref.metadata_bytes,
                        true,
                        verify_checksums_
                    );
                const uint64_t bitmap_bytes =
                    static_cast<uint64_t>(header.bitmap_words) * sizeof(uint64_t);
                if (verify_checksums_) {
                    uint32_t hash = 2166136261U;
                    hash = bc_generation_blob_checksum32_continue(hash, base, ref.metadata_bytes);
                    if (bitmap_bytes != 0U) {
                        hash = bc_generation_blob_checksum32_continue(
                            hash,
                            base + ref.bitmap_relative_offset(),
                            static_cast<size_t>(bitmap_bytes)
                        );
                    }
                    if (hash != ref.checksum) {
                        throw std::runtime_error("BC file generation blob staged checksum mismatch");
                    }
                }
                if (bitmap_bytes != 0U) {
                    std::memcpy(
                        builder->direct_restore_bitmap_bytes(header.bitmap_words),
                        base + ref.bitmap_relative_offset(),
                        static_cast<size_t>(bitmap_bytes)
                    );
                }
                builder->finish_direct_bitmap_restore(true);
                out[static_cast<size_t>(index)] = BCRestoredCellBuilder{
                    cid,
                    std::move(builder)
                };
            } catch (const std::exception &ex) {
                ok[static_cast<size_t>(index)] = 0U;
                errors[static_cast<size_t>(index)] = ex.what();
            } catch (...) {
                ok[static_cast<size_t>(index)] = 0U;
                errors[static_cast<size_t>(index)] = "unknown BC staged dump restore error";
            }
        }
        for (size_t i = 0U; i < refs.size(); ++i) {
            if (ok[i] == 0U) {
                throw std::runtime_error(errors[i]);
            }
        }
        const double restore_seconds = bc_generation_blob_now_seconds() - restore_begin;
        stats_.restore_seconds += restore_seconds;
        if (stats != nullptr) {
            stats->restore_seconds += restore_seconds;
        }
        uint64_t read_bytes_sum = 0U;
        for (const auto &[cid, ref] : refs) {
            (void)cid;
            ++stats_.read_count;
            stats_.bytes_read = bc_generation_blob_checked_add_u64(
                stats_.bytes_read,
                ref.bytes,
                "BC file generation blob staged read byte stats overflow"
            );
            read_bytes_sum = bc_generation_blob_checked_add_u64(
                read_bytes_sum,
                ref.bytes,
                "BC file generation blob staged read byte stats overflow"
            );
        }
        accumulate(stats, &BCGenerationBlobIOStats::read_count, refs.size());
        accumulate(stats, &BCGenerationBlobIOStats::bytes_read, read_bytes_sum);
        accumulate(stats, &BCGenerationBlobIOStats::coalesced_extents, 0U);
        accumulate(stats, &BCGenerationBlobIOStats::read_bytes, read_bytes_sum);
        return out;
    }

    void ensure_append_file_capacity(uint64_t required_bytes) {
        if (writer_ == nullptr || writer_->mode() != BCFileIOMode::Direct) {
            return;
        }
        if (required_bytes <= append_reserved_bytes_) {
            return;
        }
        const uint64_t reserve_target = bc_generation_blob_align_up_u64(
            required_bytes,
            kDirectAppendReserveBytes
        );
        writer_->resize(reserve_target);
        append_reserved_bytes_ = reserve_target;
    }

    void append_stage_zero(uint64_t bytes) {
        static constexpr std::array<uint8_t, 4096U> kZeroPage = {};
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            const uint64_t take = std::min<uint64_t>(remaining, kZeroPage.size());
            append_stage_bytes(kZeroPage.data(), take);
            remaining -= take;
        }
    }

    void append_stage_bytes(const void *data, uint64_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC file generation blob append data pointer is null");
        }
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            const uint64_t available = staging_bytes_ - append_stage_size_;
            if (available == 0U) {
                flush_append_stage();
                continue;
            }
            const uint64_t take = std::min<uint64_t>(remaining, available);
            std::memcpy(
                append_stage_.get() + static_cast<size_t>(append_stage_size_),
                cursor,
                static_cast<size_t>(take)
            );
            append_stage_size_ += take;
            cursor += take;
            remaining -= take;
            if (append_stage_size_ == staging_bytes_) {
                flush_append_stage();
            }
        }
    }

    void flush_append_stage() const {
        if (append_stage_size_ == 0U) {
            return;
        }
        if (writer_ == nullptr) {
            throw std::logic_error("BC file generation blob writer is null");
        }
        const uint64_t bytes = append_stage_size_;
        BCFileIOStats io_stats;
        if (writer_->mode() == BCFileIOMode::Direct) {
            constexpr uint64_t kDirectWriteChunkBytes = 4ULL * 1024ULL * 1024ULL;
            if ((append_stage_file_offset_ & (kDirectBitmapAlignment - 1U)) != 0U ||
                (bytes & (kDirectBitmapAlignment - 1U)) != 0U) {
                throw std::logic_error("BC direct generation blob append stage is not 4KB aligned");
            }
            const uint64_t write_end = bc_generation_blob_checked_add_u64(
                append_stage_file_offset_,
                bytes,
                "BC direct generation blob append stage end overflow"
            );
            const_cast<BCFileGenerationBlobIO *>(this)->ensure_append_file_capacity(write_end);
            std::vector<BCFileWriteRequest> requests;
            requests.reserve(static_cast<size_t>((bytes + kDirectWriteChunkBytes - 1U) / kDirectWriteChunkBytes));
            uint64_t cursor = 0U;
            while (cursor < bytes) {
                const uint64_t take = std::min<uint64_t>(kDirectWriteChunkBytes, bytes - cursor);
                requests.push_back(BCFileWriteRequest{
                    append_stage_file_offset_ + cursor,
                    append_stage_.get() + static_cast<size_t>(cursor),
                    take
                });
                cursor += take;
            }
            const double write_begin = bc_generation_blob_now_seconds();
            writer_->write_many(requests, &io_stats);
            stats_.backend_write_seconds += bc_generation_blob_now_seconds() - write_begin;
        } else {
            const double write_begin = bc_generation_blob_now_seconds();
            writer_->write_at(append_stage_file_offset_, append_stage_.get(), bytes);
            stats_.backend_write_seconds += bc_generation_blob_now_seconds() - write_begin;
            io_stats.backend_io_count = 1U;
            io_stats.backend_bytes = bytes;
        }
        append_stage_file_offset_ = bc_generation_blob_checked_add_u64(
            append_stage_file_offset_,
            bytes,
            "BC file generation blob append stage offset overflow"
        );
        stats_.backend_write_ops = bc_generation_blob_checked_add_u64(
            stats_.backend_write_ops,
            io_stats.backend_io_count,
            "BC file generation blob backend write op stats overflow"
        );
        stats_.backend_write_bytes = bc_generation_blob_checked_add_u64(
            stats_.backend_write_bytes,
            io_stats.backend_bytes,
            "BC file generation blob backend written byte stats overflow"
        );
        writer_dirty_ = true;
        append_stage_size_ = 0U;
    }

    BCWritableFile *writer_ = nullptr;
    const BCReadableFile *reader_ = nullptr;
    uint64_t append_offset_ = 0U;
    uint64_t staging_bytes_ = 0U;
    bool verify_checksums_ = true;
    mutable uint64_t append_stage_file_offset_ = 0U;
    mutable BCGenerationBlobAlignedBytes append_stage_;
    mutable uint64_t append_stage_size_ = 0U;
    mutable bool writer_dirty_ = false;
    uint64_t append_reserved_bytes_ = 0U;
    mutable std::vector<uint8_t> scratch_;
    BCCellBuilderDumpStreamScratch stream_dump_scratch_;
    mutable BCGenerationBlobIOStats stats_;
};

} // namespace BC
