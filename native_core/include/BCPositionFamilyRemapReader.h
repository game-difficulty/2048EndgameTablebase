#pragma once

#include "BCCellMatrix.h"
#include "BCFamilyPartitionPolicy.h"
#include "BCPositionCellLoader.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
            batch_old_cids_.capacity() * sizeof(CellId) +
            old_cells_.capacity() * sizeof(BCLoadedCell) +
            logical_output_index_.capacity() * sizeof(std::pair<CellId, uint32_t>) +
            tagged_temp_buckets_.capacity() * sizeof(TaggedTempBucket) +
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
        if (logical_cids.empty()) {
            cells.clear();
            return;
        }
        if (cells.size() < logical_cids.size()) {
            cells.resize(logical_cids.size());
        }
        if (cells.size() > logical_cids.size()) {
            cells.resize(logical_cids.size());
        }
        if (logical_cids.size() > 1U) {
            load_logical_cells_batch(logical_cids, cells, stats);
            return;
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

    struct TaggedTempBucket {
        uint32_t output_index = 0U;
        TempBucket bucket;
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

    [[nodiscard]] bool key_logical_cid(
        const BCLut &lut,
        uint64_t key,
        CellId &out
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
        if (row_id == BCFamilyTable::kInvalidFamilyId ||
            col_id == BCFamilyTable::kInvalidFamilyId) {
            return false;
        }
        out = logical_matrix_.cid(row_id, col_id);
        return true;
    }

    [[nodiscard]] bool key_maps_to_logical_cell(
        const BCLut &lut,
        uint64_t key,
        CellId logical_cid
    ) const {
        CellId key_cid = 0U;
        return key_logical_cid(lut, key, key_cid) && key_cid == logical_cid;
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

    [[nodiscard]] static uint64_t loaded_cell_allocated_bytes(const BCLoadedCell &cell) {
        return static_cast<uint64_t>(cell.buckets.capacity()) * sizeof(BCBucketEntry) +
            cell.rank_payload.capacity();
    }

    template <class Fn>
    void for_each_output_index(CellId logical_cid, Fn &&fn) const {
        const auto begin = std::lower_bound(
            logical_output_index_.begin(),
            logical_output_index_.end(),
            std::pair<CellId, uint32_t>{logical_cid, 0U},
            [](const auto &lhs, const auto &rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            }
        );
        for (auto it = begin;
             it != logical_output_index_.end() && it->first == logical_cid;
             ++it) {
            fn(it->second);
        }
    }

    void scan_old_cell_to_tagged(
        const BCLoadedCell &old_cell,
        std::vector<TaggedTempBucket> &out
    ) const {
        const BCLoadedCellView old_view = old_cell.view();
        for (uint32_t i = 0U; i < old_view.buckets.size; ++i) {
            const BCBucketEntry &bucket = old_view.buckets.data[i];
            CellId logical_cid = 0U;
            if (!key_logical_cid(source_->lut(), bucket.key, logical_cid)) {
                continue;
            }
            for_each_output_index(
                logical_cid,
                [&](uint32_t output_index) {
                    TaggedTempBucket tagged;
                    tagged.output_index = output_index;
                    keep_bucket(old_cell, bucket, tagged.bucket);
                    out.push_back(std::move(tagged));
                }
            );
        }
    }

    [[nodiscard]] static bool should_parallel_remap_scan(
        size_t old_cell_count,
        uint64_t bucket_count,
        uint64_t loaded_bytes,
        uint64_t source_file_size
    ) {
#if defined(_OPENMP)
        const char *bucket_threshold_env = std::getenv("BC_REMAP_PARALLEL_MIN_BUCKETS");
        const char *byte_threshold_env = std::getenv("BC_REMAP_PARALLEL_MIN_BYTES");
        const char *source_threshold_env = std::getenv("BC_REMAP_PARALLEL_MIN_SOURCE_BYTES");
        const bool has_explicit_threshold =
            (bucket_threshold_env != nullptr && bucket_threshold_env[0] != '\0') ||
            (byte_threshold_env != nullptr && byte_threshold_env[0] != '\0') ||
            (source_threshold_env != nullptr && source_threshold_env[0] != '\0');
        const uint64_t min_buckets =
            bucket_threshold_env == nullptr || bucket_threshold_env[0] == '\0'
                ? 0U
                : static_cast<uint64_t>(std::strtoull(bucket_threshold_env, nullptr, 10));
        const uint64_t min_bytes =
            byte_threshold_env != nullptr && byte_threshold_env[0] != '\0'
                ? static_cast<uint64_t>(std::strtoull(byte_threshold_env, nullptr, 10))
                : 0U;
        const uint64_t min_source_bytes =
            source_threshold_env != nullptr && source_threshold_env[0] != '\0'
                ? static_cast<uint64_t>(std::strtoull(source_threshold_env, nullptr, 10))
                : (has_explicit_threshold ? 0U : 64ULL * 1024ULL * 1024ULL);
        return old_cell_count >= 4U &&
            bucket_count >= min_buckets &&
            loaded_bytes >= min_bytes &&
            source_file_size >= min_source_bytes &&
            omp_get_max_threads() > 1;
#else
        (void)old_cell_count;
        (void)bucket_count;
        (void)loaded_bytes;
        (void)source_file_size;
        return false;
#endif
    }

    void scan_old_cells_to_tagged(uint64_t bucket_count, uint64_t loaded_bytes) const {
        tagged_temp_buckets_.clear();
        if (!should_parallel_remap_scan(
                old_cells_.size(),
                bucket_count,
                loaded_bytes,
                source_->file_size())) {
            for (const BCLoadedCell &old_cell : old_cells_) {
                scan_old_cell_to_tagged(old_cell, tagged_temp_buckets_);
            }
            return;
        }
#if defined(_OPENMP)
        const int thread_count = omp_get_max_threads();
        std::vector<std::vector<TaggedTempBucket>> per_thread(static_cast<size_t>(thread_count));
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<TaggedTempBucket> &local = per_thread[static_cast<size_t>(tid)];
#pragma omp for schedule(dynamic)
            for (long long i = 0; i < static_cast<long long>(old_cells_.size()); ++i) {
                scan_old_cell_to_tagged(old_cells_[static_cast<size_t>(i)], local);
            }
        }
        size_t total = 0U;
        for (const std::vector<TaggedTempBucket> &local : per_thread) {
            total += local.size();
        }
        tagged_temp_buckets_.reserve(total);
        for (std::vector<TaggedTempBucket> &local : per_thread) {
            for (TaggedTempBucket &tagged : local) {
                tagged_temp_buckets_.push_back(std::move(tagged));
            }
            local.clear();
        }
#endif
    }

    void reset_output_cells(
        const std::vector<CellId> &logical_cids,
        std::vector<BCLoadedCell> &cells
    ) const {
        for (size_t i = 0U; i < logical_cids.size(); ++i) {
            BCLoadedCell &cell = cells[i];
            cell.cid = logical_cids[i];
            cell.success_rows = 0U;
            cell.buckets.clear();
            cell.rank_payload.clear();
        }
    }

    void build_batch_old_cell_list(const std::vector<CellId> &logical_cids) const {
        batch_old_cids_.clear();
        logical_output_index_.clear();
        logical_output_index_.reserve(logical_cids.size());
        for (size_t i = 0U; i < logical_cids.size(); ++i) {
            const CellId logical_cid = logical_cids[i];
            logical_output_index_.push_back({
                logical_cid,
                checked_u32_size(i, "BC remap logical output index exceeds uint32")
            });
            collect_candidate_old_cids(logical_cid, candidate_old_cids_);
            batch_old_cids_.insert(
                batch_old_cids_.end(),
                candidate_old_cids_.begin(),
                candidate_old_cids_.end()
            );
        }
        std::sort(batch_old_cids_.begin(), batch_old_cids_.end());
        batch_old_cids_.erase(
            std::unique(batch_old_cids_.begin(), batch_old_cids_.end()),
            batch_old_cids_.end()
        );
        std::sort(
            logical_output_index_.begin(),
            logical_output_index_.end(),
            [](const auto &lhs, const auto &rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            }
        );
    }

    void load_logical_cells_batch(
        const std::vector<CellId> &logical_cids,
        std::vector<BCLoadedCell> &cells,
        BCCellLoadStats *stats
    ) const {
        reset_output_cells(logical_cids, cells);
        build_batch_old_cell_list(logical_cids);
        tagged_temp_buckets_.clear();
        if (batch_old_cids_.empty()) {
            return;
        }

        source_->load_cells_into(batch_old_cids_, old_cells_, stats);
        stats_.physical_cells_loaded += old_cells_.size();
        uint64_t old_bucket_count = 0U;
        uint64_t old_loaded_bytes = 0U;
        for (const BCLoadedCell &old_cell : old_cells_) {
            const uint64_t loaded_bytes = loaded_cell_allocated_bytes(old_cell);
            old_loaded_bytes += loaded_bytes;
            stats_.physical_bytes_loaded += loaded_bytes;
            old_bucket_count += static_cast<uint64_t>(old_cell.buckets.size());
        }
        scan_old_cells_to_tagged(old_bucket_count, old_loaded_bytes);
        std::sort(
            tagged_temp_buckets_.begin(),
            tagged_temp_buckets_.end(),
            [](const TaggedTempBucket &lhs, const TaggedTempBucket &rhs) {
                if (lhs.output_index != rhs.output_index) {
                    return lhs.output_index < rhs.output_index;
                }
                return lhs.bucket.source_entry.key < rhs.bucket.source_entry.key;
            }
        );

        size_t begin = 0U;
        while (begin < tagged_temp_buckets_.size()) {
            const uint32_t output_index = tagged_temp_buckets_[begin].output_index;
            if (output_index >= cells.size()) {
                throw std::out_of_range("BC remap tagged output index out of range");
            }
            uint32_t success_cursor = 0U;
            size_t end = begin;
            while (end < tagged_temp_buckets_.size() &&
                   tagged_temp_buckets_[end].output_index == output_index) {
                append_temp_bucket(cells[output_index], tagged_temp_buckets_[end].bucket, success_cursor);
                ++end;
            }
            cells[output_index].success_rows = success_cursor;
            begin = end;
        }
        for (const BCLoadedCell &cell : cells) {
            stats_.remapped_bytes_kept += loaded_cell_allocated_bytes(cell);
        }
        tagged_temp_buckets_.clear();
        old_cells_.clear();
        if (stats_.physical_bytes_loaded >= stats_.remapped_bytes_kept) {
            stats_.discarded_bytes = stats_.physical_bytes_loaded - stats_.remapped_bytes_kept;
        }
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
                loaded_cell_allocated_bytes(old_cell);
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
            loaded_cell_allocated_bytes(out);
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
    mutable std::vector<CellId> batch_old_cids_;
    mutable std::vector<BCLoadedCell> old_cells_;
    mutable std::vector<std::pair<CellId, uint32_t>> logical_output_index_;
    mutable std::vector<TaggedTempBucket> tagged_temp_buckets_;
    mutable std::vector<TempBucket> temp_buckets_;
};

} // namespace BC
