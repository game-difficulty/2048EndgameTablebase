#pragma once

#include "BCCellMatrix.h"
#include "BCFamilyPartitionPolicy.h"
#include "BCPositionCellLoader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace BC {

struct BCPositionFamilyRemapStats {
    uint64_t physical_cells_loaded = 0U;
    uint64_t physical_bytes_loaded = 0U;
    uint64_t remapped_bytes_kept = 0U;
    uint64_t discarded_bytes = 0U;
};

class BCPositionFamilyRemapReader {
public:
    BCPositionFamilyRemapReader(
        const BCPositionStreamingReader &source,
        BCFamilyTable logical_axis,
        const std::vector<LayerSum> &possible_8tile_sums
    ) : source_(&source),
        logical_axis_(std::move(logical_axis)),
        physical_matrix_(source.axis()),
        logical_matrix_(logical_axis_),
        physical_partition_(make_partition_for_axis(source.axis(), possible_8tile_sums)),
        logical_partition_(make_partition_for_axis(logical_axis_, possible_8tile_sums)),
        direct_(axes_equivalent(source.axis(), logical_axis_)) {}

    [[nodiscard]] const BCFamilyTable &axis() const {
        return direct_ ? source_->axis() : logical_axis_;
    }

    [[nodiscard]] bool direct() const {
        return direct_;
    }

    [[nodiscard]] uint32_t cell_count() const {
        return direct_ ? source_->cell_count() : logical_matrix_.cell_count();
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return source_->allocated_bytes() +
            logical_axis_.allocated_bytes() +
            partition_allocated_bytes(physical_partition_) +
            partition_allocated_bytes(logical_partition_) +
            candidate_old_cids_.capacity() * sizeof(CellId) +
            temp_buckets_.capacity() * sizeof(TempBucket);
    }

    [[nodiscard]] uint64_t file_size() const {
        return source_->file_size();
    }

    [[nodiscard]] const BCPositionFamilyRemapStats &stats() const {
        return stats_;
    }

    void reset_stats() const {
        stats_ = {};
    }

    [[nodiscard]] bool has_success_rows(const std::vector<CellId> &logical_cids) const {
        if (direct_) {
            for (CellId cid : logical_cids) {
                if (source_->descriptor(cid).success_rows != 0U) {
                    return true;
                }
            }
            return false;
        }
        for (CellId logical_cid : logical_cids) {
            collect_candidate_old_cids(logical_cid, candidate_old_cids_);
            for (CellId old_cid : candidate_old_cids_) {
                if (source_->descriptor(old_cid).success_rows != 0U) {
                    return true;
                }
            }
        }
        return false;
    }

    void load_cells_into(
        const std::vector<CellId> &logical_cids,
        std::vector<BCLoadedCell> &cells,
        BCCellLoadStats *stats = nullptr
    ) const {
        if (direct_) {
            source_->load_cells_into(logical_cids, cells, stats);
            return;
        }
        if (stats != nullptr) {
            *stats = {};
        }
        if (cells.size() < logical_cids.size()) {
            cells.resize(logical_cids.size());
        }
        if (cells.size() > logical_cids.size()) {
            cells.resize(logical_cids.size());
        }
        for (size_t i = 0U; i < logical_cids.size(); ++i) {
            load_logical_cell(logical_cids[i], cells[i], stats);
        }
    }

private:
    struct TempBucket {
        BCBucketEntry source_entry;
        std::vector<uint8_t> payload;
        uint32_t live_rows = 0U;
    };

    [[nodiscard]] static bool axes_equivalent(
        const BCFamilyTable &lhs,
        const BCFamilyTable &rhs
    ) {
        return lhs.layer_sum() == rhs.layer_sum() &&
            lhs.family_unit() == rhs.family_unit() &&
            lhs.coords() == rhs.coords();
    }

    [[nodiscard]] static bool axis_is_dense_modulo_axis(const BCFamilyTable &axis) {
        const std::vector<FamilyCoord> &coords = axis.coords();
        for (uint32_t i = 0U; i < coords.size(); ++i) {
            if (coords[i] != static_cast<FamilyCoord>(i)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static BCFamilyPartitionLayerMap make_partition_for_axis(
        const BCFamilyTable &axis,
        const std::vector<LayerSum> &possible_8tile_sums
    ) {
        if (axis_is_dense_modulo_axis(axis)) {
            return build_family_partition_layer_map(
                axis,
                possible_8tile_sums,
                BCFamilyPartitionPolicy::modulo(axis.family_count())
            );
        }
        return build_family_partition_layer_map_from_axis(axis);
    }

    [[nodiscard]] static uint64_t partition_allocated_bytes(
        const BCFamilyPartitionLayerMap &partition
    ) {
        uint64_t bytes = 0U;
        bytes += static_cast<uint64_t>(partition.exact_coords_by_family.capacity()) *
            sizeof(std::vector<FamilyCoord>);
        for (const std::vector<FamilyCoord> &coords : partition.exact_coords_by_family) {
            bytes += static_cast<uint64_t>(coords.capacity()) * sizeof(FamilyCoord);
        }
        bytes += static_cast<uint64_t>(partition.active_family.capacity()) * sizeof(uint8_t);
        bytes += static_cast<uint64_t>(partition.coord_to_family_lut.capacity()) * sizeof(FamilyId);
        return bytes;
    }

    [[nodiscard]] FamilyId logical_row(CellId cid) const {
        return logical_matrix_.row(cid);
    }

    [[nodiscard]] FamilyId logical_col(CellId cid) const {
        return logical_matrix_.col(cid);
    }

    void collect_candidate_old_cids(CellId logical_cid, std::vector<CellId> &out) const {
        out.clear();
        const FamilyId logical_row_id = logical_row(logical_cid);
        const FamilyId logical_col_id = logical_col(logical_cid);
        const std::vector<FamilyCoord> &row_coords =
            logical_partition_.exact_coords_by_family[logical_row_id];
        const std::vector<FamilyCoord> &col_coords =
            logical_partition_.exact_coords_by_family[logical_col_id];
        for (FamilyCoord row_coord : row_coords) {
            const FamilyId old_row = physical_partition_.try_coord_to_family_id(row_coord);
            if (old_row == BCFamilyTable::kInvalidFamilyId) {
                continue;
            }
            for (FamilyCoord col_coord : col_coords) {
                const FamilyId old_col = physical_partition_.try_coord_to_family_id(col_coord);
                if (old_col == BCFamilyTable::kInvalidFamilyId) {
                    continue;
                }
                out.push_back(physical_matrix_.cid(old_row, old_col));
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    [[nodiscard]] bool key_maps_to_logical_cell(
        const BCLut &lut,
        uint64_t key,
        CellId logical_cid
    ) const {
        const uint16_t nw = static_cast<uint16_t>((key >> 48U) & 0xFFFFU);
        const uint16_t ne = static_cast<uint16_t>((key >> 32U) & 0xFFFFU);
        const uint16_t sw = static_cast<uint16_t>((key >> 16U) & 0xFFFFU);
        const uint16_t se = static_cast<uint16_t>(key & 0xFFFFU);
        const BCWordDesc &nw_desc = lut.word_desc(nw);
        if (!nw_desc.valid) {
            throw std::runtime_error("BC remap source key has invalid NW word");
        }
        const uint64_t nw_sum = nw_desc.sum;
        const uint64_t ne_sum = lut.sum4_value(packed_sum_id(ne));
        const uint64_t sw_sum = lut.sum4_value(packed_sum_id(sw));
        const uint64_t se_sum = lut.sum4_value(packed_sum_id(se));
        const uint16_t family_unit = logical_axis_.family_unit();
        auto coord_from_pair = [family_unit](uint64_t lhs, uint64_t rhs, FamilyCoord &out) {
            const uint64_t min_sum = std::min<uint64_t>(lhs, rhs);
            if (family_unit == 0U || (min_sum % family_unit) != 0U) {
                return false;
            }
            const uint64_t coord = min_sum / family_unit;
            if (coord > std::numeric_limits<FamilyCoord>::max()) {
                return false;
            }
            out = static_cast<FamilyCoord>(coord);
            return true;
        };
        FamilyCoord row_coord = 0U;
        FamilyCoord col_coord = 0U;
        if (!coord_from_pair(nw_sum + ne_sum, sw_sum + se_sum, row_coord) ||
            !coord_from_pair(nw_sum + sw_sum, ne_sum + se_sum, col_coord)) {
            return false;
        }
        const FamilyId row_id = logical_partition_.try_coord_to_family_id(row_coord);
        const FamilyId col_id = logical_partition_.try_coord_to_family_id(col_coord);
        return row_id == logical_row(logical_cid) && col_id == logical_col(logical_cid);
    }

    [[nodiscard]] static uint32_t rank_payload_slice_end(
        const BCLut &lut,
        const BCRankPayloadView &payload,
        const BCBucketEntry &bucket
    ) {
        const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t end = static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(words_for_bits(bitmap_len)) * sizeof(uint64_t);
        if (end > payload.size || end > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("BC remap bucket payload exceeds source cell payload");
        }
        return static_cast<uint32_t>(end);
    }

    [[nodiscard]] static uint32_t bucket_live_rows(
        const BCLut &lut,
        const BCRankPayloadView &payload,
        const BCBucketEntry &bucket
    ) {
        const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint32_t word_count = words_for_bits(bitmap_len);
        uint32_t rows = 0U;
        for (uint32_t i = 0U; i < word_count; ++i) {
            uint64_t word = load_u64_le(payload.data + bitmap_offset + static_cast<size_t>(i) * sizeof(uint64_t));
            if (i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            rows += popcount64(word);
        }
        return rows;
    }

    void keep_bucket(
        const BCLoadedCell &old_cell,
        const BCBucketEntry &bucket,
        TempBucket &temp
    ) const {
        const BCRankPayloadView payload = old_cell.view().rank_payload;
        const uint32_t slice_begin = bucket.rank_payload_offset;
        const uint32_t slice_end = rank_payload_slice_end(source_->lut(), payload, bucket);
        if (slice_begin > slice_end || slice_end > payload.size) {
            throw std::runtime_error("BC remap invalid bucket payload slice");
        }
        temp.source_entry = bucket;
        temp.live_rows = bucket_live_rows(source_->lut(), payload, bucket);
        temp.payload.resize(static_cast<size_t>(slice_end - slice_begin));
        if (!temp.payload.empty()) {
            std::memcpy(
                temp.payload.data(),
                payload.data + slice_begin,
                temp.payload.size()
            );
        }
    }

    void append_temp_bucket(BCLoadedCell &out, const TempBucket &temp, uint32_t &success_cursor) const {
        const uint32_t aligned_offset = align_up_u32(
            checked_u32_size(out.rank_payload.size(), "BC remap rank payload size exceeds uint32"),
            8U
        );
        if (out.rank_payload.size() < aligned_offset) {
            out.rank_payload.resize(aligned_offset, 0U);
        }
        BCBucketEntry entry = temp.source_entry;
        entry.rank_payload_offset = aligned_offset;
        entry.success_row_offset = success_cursor;
        out.buckets.push_back(entry);
        out.rank_payload.insert(out.rank_payload.end(), temp.payload.begin(), temp.payload.end());
        success_cursor = checked_u32_add(success_cursor, temp.live_rows, "BC remap success rows overflow");
    }

    void add_load_stats(BCCellLoadStats &dst, const BCCellLoadStats &src) const {
        dst.requested_extents += src.requested_extents;
        dst.coalesced_extents += src.coalesced_extents;
        dst.requested_bytes += src.requested_bytes;
        dst.read_bytes += src.read_bytes;
        dst.backend_read_ops += src.backend_read_ops;
        dst.backend_read_bytes += src.backend_read_bytes;
    }

    void load_logical_cell(
        CellId logical_cid,
        BCLoadedCell &out,
        BCCellLoadStats *stats
    ) const {
        out.cid = logical_cid;
        out.success_rows = 0U;
        out.buckets.clear();
        out.rank_payload.clear();
        temp_buckets_.clear();
        collect_candidate_old_cids(logical_cid, candidate_old_cids_);
        for (CellId old_cid : candidate_old_cids_) {
            BCCellLoadStats one_stats;
            BCLoadedCell old_cell = source_->load_cell(old_cid, &one_stats);
            if (stats != nullptr) {
                add_load_stats(*stats, one_stats);
            }
            const uint64_t loaded_bytes =
                static_cast<uint64_t>(old_cell.buckets.capacity()) * sizeof(BCBucketEntry) +
                old_cell.rank_payload.capacity();
            stats_.physical_cells_loaded += 1U;
            stats_.physical_bytes_loaded += loaded_bytes;
            const BCLoadedCellView old_view = old_cell.view();
            for (uint32_t i = 0U; i < old_view.buckets.size; ++i) {
                const BCBucketEntry &bucket = old_view.buckets.data[i];
                if (!key_maps_to_logical_cell(source_->lut(), bucket.key, logical_cid)) {
                    continue;
                }
                temp_buckets_.push_back(TempBucket{});
                keep_bucket(old_cell, bucket, temp_buckets_.back());
            }
            old_cell.buckets.clear();
            old_cell.rank_payload.clear();
            old_cell.buckets.shrink_to_fit();
            old_cell.rank_payload.shrink_to_fit();
        }
        std::sort(
            temp_buckets_.begin(),
            temp_buckets_.end(),
            [](const TempBucket &lhs, const TempBucket &rhs) {
                return lhs.source_entry.key < rhs.source_entry.key;
            }
        );
        uint32_t success_cursor = 0U;
        for (const TempBucket &bucket : temp_buckets_) {
            append_temp_bucket(out, bucket, success_cursor);
        }
        out.success_rows = success_cursor;
        const uint64_t kept_bytes =
            static_cast<uint64_t>(out.buckets.capacity()) * sizeof(BCBucketEntry) +
            out.rank_payload.capacity();
        stats_.remapped_bytes_kept += kept_bytes;
        if (stats_.physical_bytes_loaded >= stats_.remapped_bytes_kept) {
            stats_.discarded_bytes = stats_.physical_bytes_loaded - stats_.remapped_bytes_kept;
        }
    }

    const BCPositionStreamingReader *source_ = nullptr;
    BCFamilyTable logical_axis_;
    BCCellMatrix physical_matrix_;
    BCCellMatrix logical_matrix_;
    BCFamilyPartitionLayerMap physical_partition_;
    BCFamilyPartitionLayerMap logical_partition_;
    bool direct_ = false;
    mutable BCPositionFamilyRemapStats stats_;
    mutable std::vector<CellId> candidate_old_cids_;
    mutable std::vector<TempBucket> temp_buckets_;
};

} // namespace BC
