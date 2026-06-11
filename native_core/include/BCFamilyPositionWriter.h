#pragma once

#include "BCFileIO.h"
#include "BCPositionFile.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace BC {

struct BCFamilyPositionWriterOptions {
    uint64_t staging_bytes = 16ULL * 1024ULL * 1024ULL;
    bool backend_preserves_unaligned_positioned_writes = true;
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
    uint64_t logical_size = 0U;
};

// FamilyChain streaming position writer. It is designed for the boundary
// finalize path: each cell is finalized once, written immediately, and then the
// caller can release the cell-local mutable builder. Only descriptor metadata
// and write flags stay resident. Rank payload is spooled because the final
// rank-payload stream offset is known only after all bucket metadata is written.
//
// Direct-no-buffering backends that do not preserve bytes inside aligned blocks
// must set backend_preserves_unaligned_positioned_writes=false; this writer then
// rejects the backend until a planned aligned block writer is added.
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
        if (!options.backend_preserves_unaligned_positioned_writes) {
            throw std::invalid_argument(
                "BC family position writer requires preserving positioned writes; "
                "planned direct block writer is not implemented yet"
            );
        }
        if (options.staging_bytes == 0U) {
            throw std::invalid_argument("BC family position writer staging_bytes must be non-zero");
        }
        axis_ = axis;
        options_ = options;
        const BCCellMatrix matrix(axis_);
        final_file_ = &final_file;
        rank_spool_file_ = &rank_spool_file;
        descriptors_.assign(matrix.cell_count(), BCPositionCellDescriptor{});
        written_.assign(matrix.cell_count(), 0U);
        written_count_ = 0U;
        bucket_cursor_ = 0U;
        rank_cursor_ = 0U;
        bucket_stage_.clear();
        rank_stage_.clear();
        header_scratch_.clear();
        axis_coord_table_scratch_.clear();
        descriptor_table_scratch_.clear();
        rank_copy_buffer_.clear();
        bucket_stage_file_offset_ = bucket_stream_file_offset();
        rank_stage_file_offset_ = 0U;
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

        for (const BCBucketEntry &bucket : payload.buckets) {
            const std::array<uint8_t, kBCPositionBucketEntryBytes> bucket_bytes_le =
                serialize_bucket_entry(bucket);
            append_bucket_bytes(bucket_bytes_le.data(), bucket_bytes_le.size());
        }
        stats_.bucket_bytes = bc_checked_add_u64(
            stats_.bucket_bytes,
            bucket_bytes,
            "BC family position writer bucket byte stats overflow"
        );
        if (!payload.rank_payload.empty()) {
            append_rank_spool_bytes(
                payload.rank_payload.data(),
                static_cast<uint64_t>(payload.rank_payload.size())
            );
            stats_.rank_payload_bytes = bc_checked_add_u64(
                stats_.rank_payload_bytes,
                payload.rank_payload.size(),
                "BC family position writer rank byte stats overflow"
            );
        }

        descriptors_[static_cast<size_t>(cid)] = descriptor;
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
        auto emit_bucket = [&](const BCBucketEntry &bucket) {
            const std::array<uint8_t, kBCPositionBucketEntryBytes> bucket_bytes_le =
                serialize_bucket_entry(bucket);
            append_bucket_bytes(bucket_bytes_le.data(), bucket_bytes_le.size());
            ++emitted_buckets;
        };
        auto emit_rank = [&](const void *data, uint64_t bytes) {
            if (bytes == 0U) {
                return;
            }
            if (data == nullptr) {
                throw std::invalid_argument("BC family position streamed rank chunk pointer is null");
            }
            append_rank_spool_bytes(data, bytes);
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
        mark_written(cid);
        ++stats_.empty_cells;
    }

    void flush_pending_streams_for_reader() {
        require_writable();
        flush_bucket_stage();
        flush_rank_stage();
        rank_spool_file_->flush();
        final_file_->flush();
    }

    [[nodiscard]] uint64_t finish_layer(const BCReadableFile &rank_spool_reader) {
        require_writable();
        if (written_count_ != descriptors_.size()) {
            throw std::logic_error("BC family position writer cannot finish with unwritten cells");
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
        final_file_->write_at(0U, header_scratch_.data(), header_scratch_.size());
        ++stats_.metadata_write_ops;
        stats_.metadata_write_bytes = bc_checked_add_u64(
            stats_.metadata_write_bytes,
            header_scratch_.size(),
            "BC family position metadata byte stats overflow"
        );
        final_file_->write_at(
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
        final_file_->write_at(
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
            header_scratch_.capacity() +
            axis_coord_table_scratch_.capacity() +
            descriptor_table_scratch_.capacity() +
            rank_copy_buffer_.capacity();
    }

private:
    static constexpr uint64_t kCopyChunkBytes = 4ULL * 1024ULL * 1024ULL;

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
        if (!begun_ || final_file_ == nullptr || rank_spool_file_ == nullptr) {
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

    void append_bucket_bytes(const void *data, uint64_t bytes) {
        append_stream_bytes(
            *final_file_,
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
            *rank_spool_file_,
            rank_stage_,
            rank_stage_file_offset_,
            stats_.rank_stage_flushes,
            stats_.rank_stage_write_bytes,
            data,
            bytes
        );
    }

    void append_stream_bytes(
        BCWritableFile &file,
        std::vector<uint8_t> &stage,
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
            *final_file_,
            bucket_stage_,
            bucket_stage_file_offset_,
            stats_.bucket_stage_flushes,
            stats_.bucket_stage_write_bytes
        );
    }

    void flush_rank_stage() {
        flush_stream_stage(
            *rank_spool_file_,
            rank_stage_,
            rank_stage_file_offset_,
            stats_.rank_stage_flushes,
            stats_.rank_stage_write_bytes
        );
    }

    static void flush_stream_stage(
        BCWritableFile &file,
        std::vector<uint8_t> &stage,
        uint64_t &stage_file_offset,
        uint64_t &flush_count,
        uint64_t &write_bytes
    ) {
        if (stage.empty()) {
            return;
        }
        const uint64_t bytes = static_cast<uint64_t>(stage.size());
        file.write_at(stage_file_offset, stage.data(), bytes);
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

    void copy_rank_spool(const BCReadableFile &rank_spool_reader, uint64_t rank_offset) {
        const uint64_t chunk_bytes = std::min<uint64_t>(kCopyChunkBytes, std::max<uint64_t>(rank_cursor_, 1U));
        if (chunk_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC family position rank copy buffer exceeds size_t");
        }
        rank_copy_buffer_.resize(static_cast<size_t>(chunk_bytes));
        uint64_t cursor = 0U;
        while (cursor < rank_cursor_) {
            const uint64_t take = std::min<uint64_t>(rank_cursor_ - cursor, rank_copy_buffer_.size());
            rank_spool_reader.read_at(cursor, rank_copy_buffer_.data(), take);
            ++stats_.rank_copy_chunks;
            stats_.rank_copy_read_bytes = bc_checked_add_u64(
                stats_.rank_copy_read_bytes,
                take,
                "BC family position rank copy read stats overflow"
            );
            final_file_->write_at(
                bc_checked_add_u64(rank_offset, cursor, "BC family position rank copy offset overflow"),
                rank_copy_buffer_.data(),
                take
            );
            stats_.rank_copy_write_bytes = bc_checked_add_u64(
                stats_.rank_copy_write_bytes,
                take,
                "BC family position rank copy write stats overflow"
            );
            cursor += take;
        }
    }

    BCFamilyTable axis_;
    BCFamilyPositionWriterOptions options_;
    BCWritableFile *final_file_ = nullptr;
    BCWritableFile *rank_spool_file_ = nullptr;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<uint8_t> written_;
    uint64_t written_count_ = 0U;
    uint64_t bucket_cursor_ = 0U;
    uint64_t rank_cursor_ = 0U;
    std::vector<uint8_t> bucket_stage_;
    std::vector<uint8_t> rank_stage_;
    std::vector<uint8_t> header_scratch_;
    std::vector<uint8_t> axis_coord_table_scratch_;
    std::vector<uint8_t> descriptor_table_scratch_;
    std::vector<uint8_t> rank_copy_buffer_;
    uint64_t bucket_stage_file_offset_ = 0U;
    uint64_t rank_stage_file_offset_ = 0U;
    BCFamilyPositionWriterStats stats_;
    bool begun_ = false;
    bool finished_ = false;
};

} // namespace BC
