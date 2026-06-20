#pragma once

#include "BCCellCompressedPositionFile.h"
#include "BCFileIO.h"
#include "BCPositionFile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace BC {

template <typename T, std::size_t Alignment>
class BCAlignedAllocator {
public:
    using value_type = T;

    BCAlignedAllocator() noexcept = default;

    template <typename U>
    BCAlignedAllocator(const BCAlignedAllocator<U, Alignment> &) noexcept {}

    [[nodiscard]] T *allocate(std::size_t n) {
        if (n == 0U) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        const std::size_t bytes = n * sizeof(T);
#if defined(_WIN32)
        void *ptr = _aligned_malloc(bytes, Alignment);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(ptr);
#else
        void *ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(ptr);
#endif
    }

    void deallocate(T *p, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = BCAlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, std::size_t Alignment>
[[nodiscard]] bool operator==(
    const BCAlignedAllocator<T, Alignment> &,
    const BCAlignedAllocator<U, Alignment> &
) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
[[nodiscard]] bool operator!=(
    const BCAlignedAllocator<T, Alignment> &,
    const BCAlignedAllocator<U, Alignment> &
) noexcept {
    return false;
}

using BCFamilyPositionAlignedBuffer =
    std::vector<uint8_t, BCAlignedAllocator<uint8_t, 4096U>>;

struct BCFamilyPositionWriterOptions {
    uint64_t staging_bytes = 16ULL * 1024ULL * 1024ULL;
    bool backend_preserves_unaligned_positioned_writes = true;
    bool rank_first_direct_layout = false;
    uint64_t direct_alignment = 4096ULL;
};

struct BCFamilyPositionWriterStats {
    uint64_t finalized_cells = 0U;
    uint64_t empty_cells = 0U;
    uint64_t success_rows = 0U;
    uint64_t bucket_bytes = 0U;
    uint64_t rank_payload_bytes = 0U;
    uint64_t bucket_stage_flushes = 0U;
    uint64_t bucket_stage_write_bytes = 0U;
    uint64_t rank_stage_flushes = 0U;
    uint64_t rank_stage_write_bytes = 0U;
    uint64_t metadata_write_ops = 0U;
    uint64_t metadata_write_bytes = 0U;
    uint64_t rank_copy_chunks = 0U;
    uint64_t rank_copy_read_bytes = 0U;
    uint64_t rank_copy_write_bytes = 0U;
    uint64_t backend_read_ops = 0U;
    uint64_t backend_read_bytes = 0U;
    uint64_t backend_write_ops = 0U;
    uint64_t backend_write_bytes = 0U;
    double backend_read_seconds = 0.0;
    double backend_write_seconds = 0.0;
    uint64_t logical_size = 0U;
};

// FamilyChain streaming position writer. It is designed for the boundary
// finalize path: each cell is finalized once, written immediately, and then the
// caller can release the cell-local mutable builder. Only descriptor metadata
// and write flags stay resident. Rank payload is spooled because the final
// rank-payload stream offset is known only after all bucket metadata is written.
//
// Direct-no-buffering backends that do not preserve bytes inside aligned blocks
// must set backend_preserves_unaligned_positioned_writes=false. The legacy
// bucket-then-rank layout rejects that mode; the experimental rank-first layout
// writes aligned final-file regions and uses the spool for the smaller bucket
// stream.
class BCFamilyPositionWriter {
public:
    BCFamilyPositionWriter() = default;

    void begin_layer(
        BCWritableFile &final_file,
        BCWritableFile &rank_spool_file,
        const BCFamilyTable &axis,
        BCFamilyPositionWriterOptions options = {}
    ) {
        if (axis.family_count() == 0U) {
            throw std::invalid_argument("BC family position writer requires non-empty axis");
        }
        if (!options.backend_preserves_unaligned_positioned_writes &&
            !options.rank_first_direct_layout) {
            throw std::invalid_argument(
                "BC family position writer requires preserving positioned writes; "
                "use rank_first_direct_layout for direct no-buffering output"
            );
        }
        if (options.staging_bytes == 0U) {
            throw std::invalid_argument("BC family position writer staging_bytes must be non-zero");
        }
        if (options.rank_first_direct_layout &&
            (options.direct_alignment == 0U ||
             (options.direct_alignment & (options.direct_alignment - 1U)) != 0U)) {
            throw std::invalid_argument("BC family position writer direct alignment must be a power of two");
        }
        axis_ = axis;
        options_ = options;
        const BCCellMatrix matrix(axis_);
        final_file_ = &final_file;
        rank_spool_file_ = &rank_spool_file;
        compressed_writer_ = nullptr;
        descriptors_.assign(matrix.cell_count(), BCPositionCellDescriptor{});
        written_.assign(matrix.cell_count(), 0U);
        written_count_ = 0U;
        bucket_cursor_ = 0U;
        rank_cursor_ = 0U;
        bucket_stage_.clear();
        rank_stage_.clear();
        compressed_bucket_scratch_.clear();
        compressed_rank_scratch_.clear();
        header_scratch_.clear();
        axis_coord_table_scratch_.clear();
        descriptor_table_scratch_.clear();
        metadata_prefix_scratch_.clear();
        rank_copy_buffer_.clear();
        rank_first_rank_file_offset_ = options_.rank_first_direct_layout
            ? rank_first_rank_stream_file_offset()
            : 0U;
        bucket_stage_file_offset_ = options_.rank_first_direct_layout
            ? 0U
            : bucket_stream_file_offset();
        rank_stage_file_offset_ = options_.rank_first_direct_layout
            ? rank_first_rank_file_offset_
            : 0U;
        stats_ = {};
        begun_ = true;
        finished_ = false;
    }

    void begin_cell_compressed_layer(
        BCCellCompressedPositionWriter &compressed_writer,
        const BCFamilyTable &axis,
        BCFamilyPositionWriterOptions options = {}
    ) {
        if (axis.family_count() == 0U) {
            throw std::invalid_argument("BC family position writer requires non-empty axis");
        }
        if (options.staging_bytes == 0U) {
            throw std::invalid_argument("BC family position writer staging_bytes must be non-zero");
        }
        axis_ = axis;
        options_ = options;
        const BCCellMatrix matrix(axis_);
        final_file_ = nullptr;
        rank_spool_file_ = nullptr;
        compressed_writer_ = &compressed_writer;
        compressed_writer_->begin_cells(matrix.cell_count());
        descriptors_.assign(matrix.cell_count(), BCPositionCellDescriptor{});
        written_.assign(matrix.cell_count(), 0U);
        written_count_ = 0U;
        bucket_cursor_ = 0U;
        rank_cursor_ = 0U;
        bucket_stage_.clear();
        rank_stage_.clear();
        compressed_bucket_scratch_.clear();
        compressed_rank_scratch_.clear();
        header_scratch_.clear();
        axis_coord_table_scratch_.clear();
        descriptor_table_scratch_.clear();
        metadata_prefix_scratch_.clear();
        rank_copy_buffer_.clear();
        bucket_stage_file_offset_ = 0U;
        rank_stage_file_offset_ = 0U;
        rank_first_rank_file_offset_ = 0U;
        stats_ = {};
        begun_ = true;
        finished_ = false;
    }

    void write_finalized_cell(CellId cid, const FinalizedCellPayload &payload) {
        require_writable();
        require_cell_for_write(cid);
        if (payload.buckets.empty()) {
            if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
                throw std::invalid_argument("BC family position empty payload has non-empty data");
            }
            write_empty_cell(cid);
            return;
        }
        bc_validate_position_payload_for_file(payload);

        const uint64_t bucket_bytes =
            static_cast<uint64_t>(payload.buckets.size()) * kBCPositionBucketEntryBytes;
        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = static_cast<uint32_t>(payload.buckets.size());
        descriptor.success_rows = payload.success_rows;
        descriptor.bucket_meta_offset = bucket_cursor_;
        descriptor.rank_payload_offset = rank_cursor_;
        descriptor.rank_payload_bytes = payload.rank_payload.size();
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;

        std::vector<uint8_t> compressed_bucket_bytes;
        if (compressed_writer_ != nullptr) {
            compressed_bucket_bytes.reserve(static_cast<size_t>(bucket_bytes));
        }
        for (const BCBucketEntry &bucket : payload.buckets) {
            const std::array<uint8_t, kBCPositionBucketEntryBytes> bucket_bytes_le =
                serialize_bucket_entry(bucket);
            if (compressed_writer_ != nullptr) {
                compressed_bucket_bytes.insert(
                    compressed_bucket_bytes.end(),
                    bucket_bytes_le.begin(),
                    bucket_bytes_le.end());
            } else {
                append_bucket_bytes(bucket_bytes_le.data(), bucket_bytes_le.size());
            }
        }
        stats_.bucket_bytes = bc_checked_add_u64(
            stats_.bucket_bytes,
            bucket_bytes,
            "BC family position writer bucket byte stats overflow"
        );
        if (!payload.rank_payload.empty()) {
            if (compressed_writer_ == nullptr) {
                append_rank_spool_bytes(
                    payload.rank_payload.data(),
                    static_cast<uint64_t>(payload.rank_payload.size())
                );
            }
            stats_.rank_payload_bytes = bc_checked_add_u64(
                stats_.rank_payload_bytes,
                payload.rank_payload.size(),
                "BC family position writer rank byte stats overflow"
            );
        }

        descriptors_[static_cast<size_t>(cid)] = descriptor;
        if (compressed_writer_ != nullptr) {
            compressed_writer_->write_cell(
                cid,
                descriptor,
                compressed_bucket_bytes,
                payload.rank_payload);
            stats_.bucket_stage_write_bytes = bc_checked_add_u64(
                stats_.bucket_stage_write_bytes,
                bucket_bytes,
                "BC family position compressed bucket byte stats overflow"
            );
            stats_.rank_stage_write_bytes = bc_checked_add_u64(
                stats_.rank_stage_write_bytes,
                payload.rank_payload.size(),
                "BC family position compressed rank byte stats overflow"
            );
        }
        mark_written(cid);
        ++stats_.finalized_cells;
        stats_.success_rows = bc_checked_add_u64(
            stats_.success_rows,
            payload.success_rows,
            "BC family position writer success row stats overflow"
        );
        bucket_cursor_ = bc_checked_add_u64(
            bucket_cursor_,
            bucket_bytes,
            "BC family position bucket cursor overflow"
        );
        rank_cursor_ = bc_checked_add_u64(
            rank_cursor_,
            payload.rank_payload.size(),
            "BC family position rank cursor overflow"
        );
    }

    template <class EmitFn>
    void write_finalized_cell_streamed(
        CellId cid,
        uint32_t bucket_count,
        uint32_t success_rows,
        uint64_t rank_payload_bytes,
        EmitFn &&emit
    ) {
        require_writable();
        require_cell_for_write(cid);
        if (bucket_count == 0U) {
            if (success_rows != 0U || rank_payload_bytes != 0U) {
                throw std::invalid_argument("BC family position streamed empty payload has non-empty data");
            }
            write_empty_cell(cid);
            return;
        }
        if (rank_payload_bytes > std::numeric_limits<uint64_t>::max() - rank_cursor_) {
            throw std::overflow_error("BC family position streamed rank cursor overflow");
        }
        const uint64_t bucket_bytes =
            static_cast<uint64_t>(bucket_count) * kBCPositionBucketEntryBytes;
        BCPositionCellDescriptor descriptor;
        descriptor.bucket_count = bucket_count;
        descriptor.success_rows = success_rows;
        descriptor.bucket_meta_offset = bucket_cursor_;
        descriptor.rank_payload_offset = rank_cursor_;
        descriptor.rank_payload_bytes = rank_payload_bytes;
        descriptor.reserved0 = 0U;
        descriptor.flags_or_padding = 0U;

        uint32_t emitted_buckets = 0U;
        uint64_t emitted_rank_bytes = 0U;
        if (compressed_writer_ != nullptr) {
            compressed_bucket_scratch_.clear();
            compressed_rank_scratch_.clear();
            compressed_bucket_scratch_.reserve(static_cast<size_t>(bucket_bytes));
            if (rank_payload_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC family position streamed rank payload exceeds size_t");
            }
            compressed_rank_scratch_.reserve(static_cast<size_t>(rank_payload_bytes));
        }
        auto emit_bucket = [&](const BCBucketEntry &bucket) {
            const std::array<uint8_t, kBCPositionBucketEntryBytes> bucket_bytes_le =
                serialize_bucket_entry(bucket);
            if (compressed_writer_ != nullptr) {
                compressed_bucket_scratch_.insert(
                    compressed_bucket_scratch_.end(),
                    bucket_bytes_le.begin(),
                    bucket_bytes_le.end());
            } else {
                append_bucket_bytes(bucket_bytes_le.data(), bucket_bytes_le.size());
            }
            ++emitted_buckets;
        };
        auto emit_rank = [&](const void *data, uint64_t bytes) {
            if (bytes == 0U) {
                return;
            }
            if (data == nullptr) {
                throw std::invalid_argument("BC family position streamed rank chunk pointer is null");
            }
            if (compressed_writer_ != nullptr) {
                if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
                    compressed_rank_scratch_.size() >
                        std::numeric_limits<size_t>::max() - static_cast<size_t>(bytes)) {
                    throw std::overflow_error("BC family position compressed rank scratch overflow");
                }
                const uint8_t *chunk = static_cast<const uint8_t *>(data);
                compressed_rank_scratch_.insert(
                    compressed_rank_scratch_.end(),
                    chunk,
                    chunk + static_cast<size_t>(bytes));
            } else {
                append_rank_spool_bytes(data, bytes);
            }
            emitted_rank_bytes = bc_checked_add_u64(
                emitted_rank_bytes,
                bytes,
                "BC family position streamed emitted rank byte overflow"
            );
        };
        emit(emit_bucket, emit_rank);
        if (emitted_buckets != bucket_count) {
            throw std::logic_error("BC family position streamed bucket count mismatch");
        }
        if (emitted_rank_bytes != rank_payload_bytes) {
            throw std::logic_error("BC family position streamed rank byte count mismatch");
        }

        descriptors_[static_cast<size_t>(cid)] = descriptor;
        if (compressed_writer_ != nullptr) {
            compressed_writer_->write_cell(
                cid,
                descriptor,
                compressed_bucket_scratch_,
                compressed_rank_scratch_);
            stats_.bucket_stage_write_bytes = bc_checked_add_u64(
                stats_.bucket_stage_write_bytes,
                bucket_bytes,
                "BC family position compressed bucket byte stats overflow"
            );
            stats_.rank_stage_write_bytes = bc_checked_add_u64(
                stats_.rank_stage_write_bytes,
                rank_payload_bytes,
                "BC family position compressed rank byte stats overflow"
            );
        }
        mark_written(cid);
        ++stats_.finalized_cells;
        stats_.success_rows = bc_checked_add_u64(
            stats_.success_rows,
            success_rows,
            "BC family position writer streamed success row stats overflow"
        );
        stats_.bucket_bytes = bc_checked_add_u64(
            stats_.bucket_bytes,
            bucket_bytes,
            "BC family position writer streamed bucket byte stats overflow"
        );
        stats_.rank_payload_bytes = bc_checked_add_u64(
            stats_.rank_payload_bytes,
            rank_payload_bytes,
            "BC family position writer streamed rank byte stats overflow"
        );
        bucket_cursor_ = bc_checked_add_u64(
            bucket_cursor_,
            bucket_bytes,
            "BC family position streamed bucket cursor overflow"
        );
        rank_cursor_ = bc_checked_add_u64(
            rank_cursor_,
            rank_payload_bytes,
            "BC family position streamed rank cursor overflow"
        );
    }

    void write_empty_cell(CellId cid) {
        require_writable();
        require_cell_for_write(cid);
        BCPositionCellDescriptor descriptor;
        descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
        descriptors_[static_cast<size_t>(cid)] = descriptor;
        if (compressed_writer_ != nullptr) {
            compressed_writer_->write_empty_cell(cid);
        }
        mark_written(cid);
        ++stats_.empty_cells;
    }

    void flush_pending_streams_for_reader() {
        require_writable();
        if (compressed_writer_ != nullptr) {
            compressed_writer_->flush();
            return;
        }
        flush_bucket_stage();
        flush_rank_stage();
        rank_spool_file_->flush();
        if (!options_.rank_first_direct_layout) {
            final_file_->flush();
        }
    }

    [[nodiscard]] uint64_t finish_layer(const BCReadableFile &rank_spool_reader) {
        require_writable();
        if (written_count_ != descriptors_.size()) {
            throw std::logic_error("BC family position writer cannot finish with unwritten cells");
        }
        if (options_.rank_first_direct_layout) {
            return finish_layer_rank_first(rank_spool_reader);
        }

        const uint64_t descriptor_count = descriptors_.size();
        const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis_.family_count());
        const uint64_t descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC family position descriptor table offset overflow"
        );
        const uint64_t bucket_offset = bucket_stream_file_offset();
        const uint64_t rank_offset = bc_checked_add_u64(
            bucket_offset,
            bucket_cursor_,
            "BC family position rank stream offset overflow"
        );
        const uint64_t logical_size = bc_checked_add_u64(
            rank_offset,
            rank_cursor_,
            "BC family position logical file size overflow"
        );
        flush_bucket_stage();
        flush_rank_stage();
        rank_spool_file_->flush();

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
        header.bucket_meta_bytes = bucket_cursor_;
        header.rank_payload_offset = rank_offset;
        header.rank_payload_bytes = rank_cursor_;

        header_scratch_.clear();
        header_scratch_.reserve(kBCPositionHeaderBytes);
        bc_append_header(header_scratch_, header);

        axis_coord_table_scratch_.clear();
        axis_coord_table_scratch_.reserve(static_cast<size_t>(axis_coord_bytes));
        bc_append_axis_coord_table(axis_coord_table_scratch_, axis_);

        descriptor_table_scratch_.clear();
        descriptor_table_scratch_.reserve(static_cast<size_t>(descriptor_bytes));
        for (const BCPositionCellDescriptor &descriptor : descriptors_) {
            bc_append_cell_descriptor(descriptor_table_scratch_, descriptor);
        }

        final_file_->resize(logical_size);
        write_positioned_bytes(*final_file_, 0U, header_scratch_.data(), header_scratch_.size());
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            header_scratch_.size(),
            "BC family position metadata byte stats overflow"
        );
        write_positioned_bytes(
            *final_file_,
            kBCPositionHeaderBytes,
            axis_coord_table_scratch_.data(),
            static_cast<uint64_t>(axis_coord_table_scratch_.size())
        );
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            axis_coord_table_scratch_.size(),
            "BC family position metadata byte stats overflow"
        );
        write_positioned_bytes(
            *final_file_,
            descriptor_offset,
            descriptor_table_scratch_.data(),
            static_cast<uint64_t>(descriptor_table_scratch_.size())
        );
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            descriptor_table_scratch_.size(),
            "BC family position metadata byte stats overflow"
        );
        copy_rank_spool(rank_spool_reader, rank_offset);
        final_file_->flush();
        finished_ = true;
        stats_.logical_size = logical_size;
        return logical_size;
    }

    [[nodiscard]] uint64_t finish_cell_compressed_layer() {
        require_writable();
        if (compressed_writer_ == nullptr) {
            throw std::logic_error("BC family position writer is not in cell-compressed mode");
        }
        if (written_count_ != descriptors_.size()) {
            throw std::logic_error("BC family position writer cannot finish compressed layer with unwritten cells");
        }
        const uint64_t logical_size = compressed_writer_->finish(
            axis_,
            descriptors_,
            bucket_cursor_,
            rank_cursor_);
        finished_ = true;
        stats_.logical_size = logical_size;
        return logical_size;
    }

    [[nodiscard]] uint64_t finish_layer_to_sequential_file(
        BCWritableFile &output_file,
        const BCReadableFile &bucket_spool_reader,
        const BCReadableFile &rank_spool_reader
    ) {
        require_writable();
        if (written_count_ != descriptors_.size()) {
            throw std::logic_error("BC family position writer cannot finish archive with unwritten cells");
        }

        const uint64_t descriptor_count = descriptors_.size();
        const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis_.family_count());
        const uint64_t descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC family position archive descriptor table offset overflow"
        );
        const uint64_t bucket_offset = bc_checked_add_u64(
            descriptor_offset,
            descriptor_bytes,
            "BC family position archive bucket stream offset overflow"
        );
        const uint64_t rank_offset = bc_checked_add_u64(
            bucket_offset,
            bucket_cursor_,
            "BC family position archive rank stream offset overflow"
        );
        const uint64_t logical_size = bc_checked_add_u64(
            rank_offset,
            rank_cursor_,
            "BC family position archive logical file size overflow"
        );
        flush_bucket_stage();
        flush_rank_stage();
        rank_spool_file_->flush();
        final_file_->flush();

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
        header.bucket_meta_bytes = bucket_cursor_;
        header.rank_payload_offset = rank_offset;
        header.rank_payload_bytes = rank_cursor_;

        header_scratch_.clear();
        header_scratch_.reserve(kBCPositionHeaderBytes);
        bc_append_header(header_scratch_, header);

        axis_coord_table_scratch_.clear();
        axis_coord_table_scratch_.reserve(static_cast<size_t>(axis_coord_bytes));
        bc_append_axis_coord_table(axis_coord_table_scratch_, axis_);

        descriptor_table_scratch_.clear();
        descriptor_table_scratch_.reserve(static_cast<size_t>(descriptor_bytes));
        for (const BCPositionCellDescriptor &descriptor : descriptors_) {
            bc_append_cell_descriptor(descriptor_table_scratch_, descriptor);
        }

        output_file.prepare_full_overwrite(logical_size);
        write_positioned_bytes(output_file, 0U, header_scratch_.data(), header_scratch_.size());
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            header_scratch_.size(),
            "BC family position archive metadata byte stats overflow"
        );
        write_positioned_bytes(
            output_file,
            kBCPositionHeaderBytes,
            axis_coord_table_scratch_.data(),
            static_cast<uint64_t>(axis_coord_table_scratch_.size())
        );
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            axis_coord_table_scratch_.size(),
            "BC family position archive metadata byte stats overflow"
        );
        write_positioned_bytes(
            output_file,
            descriptor_offset,
            descriptor_table_scratch_.data(),
            static_cast<uint64_t>(descriptor_table_scratch_.size())
        );
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            descriptor_table_scratch_.size(),
            "BC family position archive metadata byte stats overflow"
        );
        copy_spool_stream_to(output_file, bucket_spool_reader, bucket_offset, bucket_cursor_);
        copy_spool_stream_to(output_file, rank_spool_reader, rank_offset, rank_cursor_);
        output_file.flush();
        finished_ = true;
        stats_.logical_size = logical_size;
        return logical_size;
    }

    [[nodiscard]] const std::vector<BCPositionCellDescriptor> &descriptors() const {
        return descriptors_;
    }

    [[nodiscard]] uint64_t bucket_meta_bytes() const {
        return bucket_cursor_;
    }

    [[nodiscard]] uint64_t rank_payload_bytes() const {
        return rank_cursor_;
    }

    [[nodiscard]] const BCFamilyPositionWriterStats &stats() const {
        return stats_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return axis_.allocated_bytes() +
            static_cast<uint64_t>(descriptors_.capacity()) * sizeof(BCPositionCellDescriptor) +
            static_cast<uint64_t>(written_.capacity()) * sizeof(uint8_t) +
            bucket_stage_.capacity() +
            rank_stage_.capacity() +
            compressed_bucket_scratch_.capacity() +
            compressed_rank_scratch_.capacity() +
            header_scratch_.capacity() +
            axis_coord_table_scratch_.capacity() +
            descriptor_table_scratch_.capacity() +
            metadata_prefix_scratch_.capacity() +
            rank_copy_buffer_.capacity();
    }

private:
    static constexpr uint64_t kCopyChunkBytes = 4ULL * 1024ULL * 1024ULL;

    [[nodiscard]] static double now_seconds() {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    [[nodiscard]] static uint64_t align_up(uint64_t value, uint64_t alignment) {
        if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
            throw std::invalid_argument("BC family position writer alignment must be a power of two");
        }
        if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
            throw std::overflow_error("BC family position writer align_up overflow");
        }
        return (value + alignment - 1U) & ~(alignment - 1U);
    }

    void record_backend_write_stats(const BCFileIOStats &io_stats, double seconds) {
        stats_.backend_write_ops = bc_checked_add_u64(
            stats_.backend_write_ops,
            io_stats.backend_io_count,
            "BC family position writer backend write op stats overflow"
        );
        stats_.backend_write_bytes = bc_checked_add_u64(
            stats_.backend_write_bytes,
            io_stats.backend_bytes,
            "BC family position writer backend write byte stats overflow"
        );
        stats_.backend_write_seconds += seconds;
    }

    void record_backend_read_stats(const BCFileIOStats &io_stats, double seconds) {
        stats_.backend_read_ops = bc_checked_add_u64(
            stats_.backend_read_ops,
            io_stats.backend_io_count,
            "BC family position writer backend read op stats overflow"
        );
        stats_.backend_read_bytes = bc_checked_add_u64(
            stats_.backend_read_bytes,
            io_stats.backend_bytes,
            "BC family position writer backend read byte stats overflow"
        );
        stats_.backend_read_seconds += seconds;
    }

    static void store_u32_le(uint8_t *out, uint32_t value) {
        out[0] = static_cast<uint8_t>(value & 0xFFU);
        out[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        out[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        out[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    }

    static void store_u64_le(uint8_t *out, uint64_t value) {
        for (uint32_t i = 0U; i < 8U; ++i) {
            out[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU);
        }
    }

    [[nodiscard]] static std::array<uint8_t, kBCPositionBucketEntryBytes>
    serialize_bucket_entry(const BCBucketEntry &entry) {
        std::array<uint8_t, kBCPositionBucketEntryBytes> out = {};
        store_u64_le(out.data() + 0U, entry.key);
        store_u32_le(out.data() + 8U, entry.rank_payload_offset);
        store_u32_le(out.data() + 12U, entry.success_row_offset);
        return out;
    }

    void require_writable() const {
        if (!begun_ ||
            (compressed_writer_ == nullptr && (final_file_ == nullptr || rank_spool_file_ == nullptr))) {
            throw std::logic_error("BC family position writer layer has not begun");
        }
        if (finished_) {
            throw std::logic_error("BC family position writer layer is already finished");
        }
    }

    void require_cell_for_write(CellId cid) const {
        if (cid >= descriptors_.size()) {
            throw std::out_of_range("BC family position writer cell id out of range");
        }
        if (written_[static_cast<size_t>(cid)] != 0U) {
            throw std::logic_error("BC family position writer cell already written");
        }
    }

    void mark_written(CellId cid) {
        written_[static_cast<size_t>(cid)] = 1U;
        ++written_count_;
    }

    [[nodiscard]] uint64_t bucket_stream_file_offset() const {
        return bc_checked_add_u64(
            bc_checked_add_u64(
                kBCPositionHeaderBytes,
                bc_axis_coord_table_bytes(axis_.family_count()),
                "BC family position axis coord table base overflow"
            ),
            static_cast<uint64_t>(descriptors_.size()) * kBCPositionCellDescriptorBytes,
            "BC family position bucket stream base overflow"
        );
    }

    [[nodiscard]] uint64_t metadata_end_offset() const {
        return bucket_stream_file_offset();
    }

    [[nodiscard]] uint64_t rank_first_rank_stream_file_offset() const {
        return align_up(metadata_end_offset(), options_.direct_alignment);
    }

    [[nodiscard]] uint64_t finish_layer_rank_first(const BCReadableFile &bucket_spool_reader) {
        const uint64_t descriptor_count = descriptors_.size();
        const uint64_t descriptor_bytes = descriptor_count * kBCPositionCellDescriptorBytes;
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(axis_.family_count());
        const uint64_t descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC family position rank-first descriptor table offset overflow"
        );
        const uint64_t rank_offset = rank_first_rank_file_offset_;
        const uint64_t rank_end = bc_checked_add_u64(
            rank_offset,
            rank_cursor_,
            "BC family position rank-first rank end overflow"
        );
        const uint64_t bucket_offset = align_up(rank_end, options_.direct_alignment);
        const uint64_t logical_size = bc_checked_add_u64(
            bucket_offset,
            bucket_cursor_,
            "BC family position rank-first logical file size overflow"
        );

        flush_bucket_stage();
        flush_rank_stage();
        rank_spool_file_->flush();

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
        header.bucket_meta_bytes = bucket_cursor_;
        header.rank_payload_offset = rank_offset;
        header.rank_payload_bytes = rank_cursor_;

        header_scratch_.clear();
        header_scratch_.reserve(kBCPositionHeaderBytes);
        bc_append_header(header_scratch_, header);

        axis_coord_table_scratch_.clear();
        axis_coord_table_scratch_.reserve(static_cast<size_t>(axis_coord_bytes));
        bc_append_axis_coord_table(axis_coord_table_scratch_, axis_);

        descriptor_table_scratch_.clear();
        descriptor_table_scratch_.reserve(static_cast<size_t>(descriptor_bytes));
        for (const BCPositionCellDescriptor &descriptor : descriptors_) {
            bc_append_cell_descriptor(descriptor_table_scratch_, descriptor);
        }

        if (rank_offset > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family position rank-first metadata prefix exceeds size_t");
        }
        metadata_prefix_scratch_.assign(static_cast<size_t>(rank_offset), 0U);
        std::memcpy(metadata_prefix_scratch_.data(), header_scratch_.data(), header_scratch_.size());
        if (!axis_coord_table_scratch_.empty()) {
            std::memcpy(
                metadata_prefix_scratch_.data() + kBCPositionHeaderBytes,
                axis_coord_table_scratch_.data(),
                axis_coord_table_scratch_.size()
            );
        }
        if (!descriptor_table_scratch_.empty()) {
            std::memcpy(
                metadata_prefix_scratch_.data() + static_cast<size_t>(descriptor_offset),
                descriptor_table_scratch_.data(),
                descriptor_table_scratch_.size()
            );
        }

        final_file_->resize(logical_size);
        write_positioned_bytes(
            *final_file_,
            0U,
            metadata_prefix_scratch_.data(),
            static_cast<uint64_t>(metadata_prefix_scratch_.size())
        );
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            header_scratch_.size() + axis_coord_table_scratch_.size() + descriptor_table_scratch_.size(),
            "BC family position rank-first metadata byte stats overflow"
        );
        copy_spool_stream(bucket_spool_reader, bucket_offset, bucket_cursor_);
        final_file_->flush();
        finished_ = true;
        stats_.logical_size = logical_size;
        return logical_size;
    }

    void append_bucket_bytes(const void *data, uint64_t bytes) {
        append_stream_bytes(
            options_.rank_first_direct_layout ? *rank_spool_file_ : *final_file_,
            bucket_stage_,
            bucket_stage_file_offset_,
            stats_.bucket_stage_flushes,
            stats_.bucket_stage_write_bytes,
            data,
            bytes
        );
    }

    void append_rank_spool_bytes(const void *data, uint64_t bytes) {
        append_stream_bytes(
            options_.rank_first_direct_layout ? *final_file_ : *rank_spool_file_,
            rank_stage_,
            rank_stage_file_offset_,
            stats_.rank_stage_flushes,
            stats_.rank_stage_write_bytes,
            data,
            bytes
        );
    }

    template <class StageBuffer>
    void append_stream_bytes(
        BCWritableFile &file,
        StageBuffer &stage,
        uint64_t &stage_file_offset,
        uint64_t &flush_count,
        uint64_t &write_bytes,
        const void *data,
        uint64_t bytes
    ) {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC family position writer append data pointer is null");
        }
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining > 0U) {
            const uint64_t available = options_.staging_bytes - stage.size();
            if (available == 0U) {
                flush_stream_stage(file, stage, stage_file_offset, flush_count, write_bytes);
                continue;
            }
            const uint64_t take = std::min<uint64_t>(remaining, available);
            const size_t old_size = stage.size();
            stage.resize(old_size + static_cast<size_t>(take));
            std::memcpy(stage.data() + old_size, cursor, static_cast<size_t>(take));
            cursor += take;
            remaining -= take;
            if (stage.size() == options_.staging_bytes) {
                flush_stream_stage(file, stage, stage_file_offset, flush_count, write_bytes);
            }
        }
    }

    void flush_bucket_stage() {
        flush_stream_stage(
            options_.rank_first_direct_layout ? *rank_spool_file_ : *final_file_,
            bucket_stage_,
            bucket_stage_file_offset_,
            stats_.bucket_stage_flushes,
            stats_.bucket_stage_write_bytes
        );
    }

    void flush_rank_stage() {
        flush_stream_stage(
            options_.rank_first_direct_layout ? *final_file_ : *rank_spool_file_,
            rank_stage_,
            rank_stage_file_offset_,
            stats_.rank_stage_flushes,
            stats_.rank_stage_write_bytes
        );
    }

    template <class StageBuffer>
    void flush_stream_stage(
        BCWritableFile &file,
        StageBuffer &stage,
        uint64_t &stage_file_offset,
        uint64_t &flush_count,
        uint64_t &write_bytes
    ) {
        if (stage.empty()) {
            return;
        }
        const uint64_t bytes = static_cast<uint64_t>(stage.size());
        BCFileIOStats io_stats;
        const double write_begin = now_seconds();
        file.write_many(
            std::vector<BCFileWriteRequest>{BCFileWriteRequest{stage_file_offset, stage.data(), bytes}},
            &io_stats
        );
        record_backend_write_stats(io_stats, now_seconds() - write_begin);
        ++flush_count;
        write_bytes = bc_checked_add_u64(
            write_bytes,
            bytes,
            "BC family position writer staging write byte stats overflow"
        );
        stage_file_offset = bc_checked_add_u64(
            stage_file_offset,
            bytes,
            "BC family position writer staging offset overflow"
        );
        stage.clear();
    }

    void write_positioned_bytes(
        BCWritableFile &file,
        uint64_t offset,
        const void *data,
        uint64_t bytes
    ) {
        if (bytes == 0U) {
            return;
        }
        BCFileIOStats io_stats;
        const double write_begin = now_seconds();
        file.write_many(
            std::vector<BCFileWriteRequest>{BCFileWriteRequest{offset, data, bytes}},
            &io_stats
        );
        record_backend_write_stats(io_stats, now_seconds() - write_begin);
    }

    void read_positioned_bytes(
        const BCReadableFile &file,
        uint64_t offset,
        void *data,
        uint64_t bytes
    ) {
        if (bytes == 0U) {
            return;
        }
        BCFileIOStats io_stats;
        const double read_begin = now_seconds();
        file.read_many(
            std::vector<BCFileReadRequest>{BCFileReadRequest{offset, data, bytes}},
            &io_stats
        );
        record_backend_read_stats(io_stats, now_seconds() - read_begin);
    }

    void copy_rank_spool(const BCReadableFile &rank_spool_reader, uint64_t rank_offset) {
        copy_spool_stream(rank_spool_reader, rank_offset, rank_cursor_);
    }

    void copy_spool_stream(
        const BCReadableFile &spool_reader,
        uint64_t target_offset,
        uint64_t bytes
    ) {
        copy_spool_stream_to(*final_file_, spool_reader, target_offset, bytes);
    }

    void copy_spool_stream_to(
        BCWritableFile &target_file,
        const BCReadableFile &spool_reader,
        uint64_t target_offset,
        uint64_t bytes
    ) {
        const uint64_t chunk_bytes = std::min<uint64_t>(kCopyChunkBytes, std::max<uint64_t>(bytes, 1U));
        if (chunk_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family position spool copy buffer exceeds size_t");
        }
        rank_copy_buffer_.resize(static_cast<size_t>(chunk_bytes));
        uint64_t cursor = 0U;
        while (cursor < bytes) {
            const uint64_t take = std::min<uint64_t>(bytes - cursor, rank_copy_buffer_.size());
            read_positioned_bytes(spool_reader, cursor, rank_copy_buffer_.data(), take);
            ++stats_.rank_copy_chunks;
            stats_.rank_copy_read_bytes = bc_checked_add_u64(
                stats_.rank_copy_read_bytes,
                take,
                "BC family position spool copy read stats overflow"
            );
            write_positioned_bytes(
                target_file,
                bc_checked_add_u64(target_offset, cursor, "BC family position spool copy offset overflow"),
                rank_copy_buffer_.data(),
                take
            );
            stats_.rank_copy_write_bytes = bc_checked_add_u64(
                stats_.rank_copy_write_bytes,
                take,
                "BC family position spool copy write stats overflow"
            );
            cursor += take;
        }
    }

    BCFamilyTable axis_;
    BCFamilyPositionWriterOptions options_;
    BCWritableFile *final_file_ = nullptr;
    BCWritableFile *rank_spool_file_ = nullptr;
    BCCellCompressedPositionWriter *compressed_writer_ = nullptr;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<uint8_t> written_;
    uint64_t written_count_ = 0U;
    uint64_t bucket_cursor_ = 0U;
    uint64_t rank_cursor_ = 0U;
    BCFamilyPositionAlignedBuffer bucket_stage_;
    BCFamilyPositionAlignedBuffer rank_stage_;
    std::vector<uint8_t> compressed_bucket_scratch_;
    std::vector<uint8_t> compressed_rank_scratch_;
    std::vector<uint8_t> header_scratch_;
    std::vector<uint8_t> axis_coord_table_scratch_;
    std::vector<uint8_t> descriptor_table_scratch_;
    std::vector<uint8_t> metadata_prefix_scratch_;
    BCFamilyPositionAlignedBuffer rank_copy_buffer_;
    uint64_t bucket_stage_file_offset_ = 0U;
    uint64_t rank_stage_file_offset_ = 0U;
    uint64_t rank_first_rank_file_offset_ = 0U;
    BCFamilyPositionWriterStats stats_;
    bool begun_ = false;
    bool finished_ = false;
};

} // namespace BC
