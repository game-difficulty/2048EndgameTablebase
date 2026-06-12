#pragma once

#include "BCFileIO.h"
#include "BCPositionFile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace BC {

struct BCCellLoadStats {
    uint64_t requested_extents = 0U;
    uint64_t coalesced_extents = 0U;
    uint64_t requested_bytes = 0U;
    uint64_t read_bytes = 0U;
    uint64_t backend_read_ops = 0U;
    uint64_t backend_read_bytes = 0U;
};

struct BCLoadedCellView {
    CellId cid = 0U;
    uint32_t success_rows = 0U;
    BCBucketEntryView buckets = {};
    BCRankPayloadView rank_payload = {};

    [[nodiscard]] bool empty() const {
        return buckets.size == 0U;
    }

    [[nodiscard]] BCLookupResult lookup(
        const BCLut &lut,
        uint64_t key,
        BucketRank rank
    ) const {
        return lookup_finalized_cell(lut, buckets, rank_payload, key, rank);
    }
};

struct BCLoadedCell {
    CellId cid = 0U;
    uint32_t success_rows = 0U;
    std::vector<BCBucketEntry> buckets;
    std::vector<uint8_t> rank_payload;

    [[nodiscard]] BCLoadedCellView view() const {
        return BCLoadedCellView{
            cid,
            success_rows,
            BCBucketEntryView{
                buckets.data(),
                checked_u32_size(buckets.size(), "BC loaded cell bucket count exceeds uint32")
            },
            BCRankPayloadView{
                rank_payload.data(),
                checked_u32_size(rank_payload.size(), "BC loaded cell rank payload exceeds uint32")
            }
        };
    }

    [[nodiscard]] BCLookupResult lookup(
        const BCLut &lut,
        uint64_t key,
        BucketRank rank
    ) const {
        return view().lookup(lut, key, rank);
    }
};

class BCPositionStreamingReader {
public:
    BCPositionStreamingReader() = default;

    BCPositionStreamingReader(std::unique_ptr<BCReadableFile> file, const BCLut &lut) {
        open(std::move(file), lut);
    }

    static BCPositionStreamingReader open_buffered(
        const std::filesystem::path &path,
        const BCLut &lut
    ) {
        return BCPositionStreamingReader(std::make_unique<BCBufferedFileReader>(path), lut);
    }

    void open(std::unique_ptr<BCReadableFile> file, const BCLut &lut) {
        if (!file) {
            throw std::invalid_argument("BC streaming position reader file is null");
        }
        file_ = std::move(file);
        lut_ = &lut;
        file_size_ = file_->size();
        if (file_size_ < kBCPositionHeaderBytes) {
            throw std::runtime_error("BC streaming position file is smaller than header");
        }

        std::vector<uint8_t> header_bytes(kBCPositionHeaderBytes);
        file_->read_at(0U, header_bytes.data(), header_bytes.size());
        header_ = bc_read_header(header_bytes);
        validate_header();
        read_axis();
        read_descriptors();
        validate_descriptors();
    }

    void set_validate_loaded_cells(bool enabled) {
        validate_loaded_cells_ = enabled;
    }

    [[nodiscard]] bool validate_loaded_cells() const {
        return validate_loaded_cells_;
    }

    [[nodiscard]] const BCPositionHeader &header() const {
        return header_;
    }

    [[nodiscard]] const BCFamilyTable &axis() const {
        return axis_;
    }

    [[nodiscard]] const BCLut &lut() const {
        if (lut_ == nullptr) {
            throw std::logic_error("BC streaming position reader is not open");
        }
        return *lut_;
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        return axis_.allocated_bytes() +
            static_cast<uint64_t>(descriptors_.capacity()) * sizeof(BCPositionCellDescriptor);
    }

    [[nodiscard]] uint64_t file_size() const {
        return file_size_;
    }

    [[nodiscard]] std::vector<uint8_t> read_all_bytes(BCFileIOStats *stats = nullptr) const {
        require_open();
        if (stats != nullptr) {
            *stats = {};
        }
        if (file_size_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC streaming position file exceeds addressable memory vector size");
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(file_size_));
        if (!bytes.empty()) {
            file_->read_many(
                std::vector<BCFileReadRequest>{
                    BCFileReadRequest{0U, bytes.data(), file_size_}
                },
                stats
            );
        }
        return bytes;
    }

    [[nodiscard]] uint32_t cell_count() const {
        return checked_u32_size(descriptors_.size(), "BC streaming position cell count exceeds uint32");
    }

    [[nodiscard]] const BCPositionCellDescriptor &descriptor(CellId cid) const {
        if (cid >= descriptors_.size()) {
            throw std::out_of_range("BC streaming position reader cell id out of range");
        }
        return descriptors_[static_cast<size_t>(cid)];
    }

    [[nodiscard]] std::vector<BCFileExtent> cell_extents(CellId cid) const {
        const BCPositionCellDescriptor &desc = descriptor(cid);
        if (desc.empty()) {
            return {};
        }
        const uint64_t bucket_bytes =
            static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes;
        std::vector<BCFileExtent> extents;
        extents.reserve(2U);
        extents.push_back(BCFileExtent{
            bc_checked_add_u64(
                header_.bucket_meta_offset,
                desc.bucket_meta_offset,
                "BC streaming position bucket file offset overflow"
            ),
            bucket_bytes
        });
        extents.push_back(BCFileExtent{
            bc_checked_add_u64(
                header_.rank_payload_offset,
                desc.rank_payload_offset,
                "BC streaming position rank file offset overflow"
            ),
            desc.rank_payload_bytes
        });
        return extents;
    }

    [[nodiscard]] BCLoadedCell load_cell(CellId cid, BCCellLoadStats *stats = nullptr) const {
        const std::vector<BCLoadedCell> cells = load_cells(std::vector<CellId>{cid}, stats);
        if (cells.size() != 1U) {
            throw std::logic_error("BC streaming position load_cell internal result size mismatch");
        }
        return cells.front();
    }

    [[nodiscard]] std::vector<BCLoadedCell> load_cells(
        const std::vector<CellId> &cids,
        BCCellLoadStats *stats = nullptr
    ) const {
        std::vector<BCLoadedCell> cells;
        load_cells_into(cids, cells, stats);
        return cells;
    }

    void load_cells_into(
        const std::vector<CellId> &cids,
        std::vector<BCLoadedCell> &cells,
        BCCellLoadStats *stats = nullptr
    ) const {
        require_open();
        if (stats != nullptr) {
            *stats = {};
        }
        if (cells.size() < cids.size()) {
            cells.resize(cids.size());
        }
        for (size_t i = 0U; i < cids.size(); ++i) {
            const CellId cid = cids[i];
            const BCPositionCellDescriptor &desc = descriptor(cid);
            BCLoadedCell &cell = cells[i];
            cell.cid = cid;
            cell.success_rows = desc.success_rows;
            cell.buckets.clear();
            cell.rank_payload.clear();
        }
        if (cells.size() > cids.size()) {
            cells.resize(cids.size());
        }

        std::vector<ExtentRequest> bucket_requests;
        std::vector<ExtentRequest> rank_requests;
        bucket_requests.reserve(cids.size());
        rank_requests.reserve(cids.size());
        for (size_t index = 0U; index < cids.size(); ++index) {
            const CellId cid = cids[index];
            const BCPositionCellDescriptor &desc = descriptor(cid);
            if (desc.empty()) {
                continue;
            }
            const uint64_t bucket_bytes =
                static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes;
            append_request(
                bucket_requests,
                index,
                ExtentKind::Bucket,
                bc_checked_add_u64(
                    header_.bucket_meta_offset,
                    desc.bucket_meta_offset,
                    "BC streaming position bucket request offset overflow"
                ),
                bucket_bytes
            );
            append_request(
                rank_requests,
                index,
                ExtentKind::RankPayload,
                bc_checked_add_u64(
                    header_.rank_payload_offset,
                    desc.rank_payload_offset,
                    "BC streaming position rank request offset overflow"
                ),
                desc.rank_payload_bytes
            );
        }
        std::vector<ExtentRequest> requests;
        requests.reserve(bucket_requests.size() + rank_requests.size());
        requests.insert(requests.end(), bucket_requests.begin(), bucket_requests.end());
        requests.insert(requests.end(), rank_requests.begin(), rank_requests.end());
        auto request_less = [](const ExtentRequest &lhs, const ExtentRequest &rhs) {
            if (lhs.offset != rhs.offset) {
                return lhs.offset < rhs.offset;
            }
            if (lhs.bytes != rhs.bytes) {
                return lhs.bytes < rhs.bytes;
            }
            if (lhs.cell_index != rhs.cell_index) {
                return lhs.cell_index < rhs.cell_index;
            }
            return static_cast<uint8_t>(lhs.kind) < static_cast<uint8_t>(rhs.kind);
        };
        bool monotonic = true;
        for (size_t i = 1U; i < requests.size(); ++i) {
            if (request_less(requests[i], requests[i - 1U])) {
                monotonic = false;
                break;
            }
        }
        if (!monotonic) {
            for (size_t i = 1U; i < requests.size(); ++i) {
                ExtentRequest current = requests[i];
                size_t j = i;
                while (j > 0U && request_less(current, requests[j - 1U])) {
                    requests[j] = requests[j - 1U];
                    --j;
                }
                requests[j] = current;
            }
        }

        if (stats != nullptr) {
            stats->requested_extents = requests.size();
            for (const ExtentRequest &request : requests) {
                stats->requested_bytes = bc_checked_add_u64(
                    stats->requested_bytes,
                    request.bytes,
                    "BC streaming position requested byte count overflow"
                );
            }
        }

        std::vector<LoadedRange> ranges = read_coalesced_ranges(requests, stats);
        size_t range_index = 0U;
        for (const ExtentRequest &request : requests) {
            if (request.cell_index >= cells.size()) {
                throw std::out_of_range("BC streaming position request cell index out of range");
            }
            const uint64_t request_end = bc_checked_add_u64(
                request.offset,
                request.bytes,
                "BC streaming position loaded request end overflow"
            );
            while (range_index < ranges.size()) {
                const uint64_t range_end = bc_checked_add_u64(
                    ranges[range_index].offset,
                    ranges[range_index].bytes.size(),
                    "BC streaming position loaded range end overflow"
                );
                if (range_end > request.offset) {
                    break;
                }
                ++range_index;
            }
            if (range_index >= ranges.size()) {
                throw std::logic_error("BC streaming position request is not covered by loaded range");
            }
            const LoadedRange &range = ranges[range_index];
            const uint64_t range_end = bc_checked_add_u64(
                range.offset,
                range.bytes.size(),
                "BC streaming position loaded range end overflow"
            );
            if (request.offset < range.offset || request_end > range_end) {
                throw std::logic_error("BC streaming position request is not covered by loaded range");
            }
            const uint64_t in_range_offset = request.offset - range.offset;
            if (in_range_offset > range.bytes.size() ||
                request.bytes > range.bytes.size() - in_range_offset) {
                throw std::logic_error("BC streaming position loaded range does not cover request");
            }
            const uint8_t *src = range.bytes.data() + static_cast<size_t>(in_range_offset);
            BCLoadedCell &cell = cells[request.cell_index];
            if (request.kind == ExtentKind::Bucket) {
                parse_bucket_entries(src, request.bytes, cell.buckets);
            } else {
                if (request.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                    throw std::overflow_error("BC streaming position rank payload exceeds size_t");
                }
                cell.rank_payload.assign(src, src + static_cast<size_t>(request.bytes));
            }
        }

        if (validate_loaded_cells_) {
            for (const BCLoadedCell &cell : cells) {
                validate_loaded_cell(cell);
            }
        }
    }

    [[nodiscard]] BCLookupResult cold_lookup(
        CellId cid,
        uint64_t key,
        BucketRank rank
    ) const {
        return load_cell(cid).lookup(lut(), key, rank);
    }

private:
    enum class ExtentKind : uint8_t {
        Bucket,
        RankPayload,
    };

    struct ExtentRequest {
        uint64_t offset = 0U;
        uint64_t bytes = 0U;
        size_t cell_index = 0U;
        ExtentKind kind = ExtentKind::Bucket;
    };

    struct LoadedRange {
        uint64_t offset = 0U;
        std::vector<uint8_t> bytes;
    };

    void require_open() const {
        if (!file_ || lut_ == nullptr) {
            throw std::logic_error("BC streaming position reader is not open");
        }
    }

    void validate_header() const {
        if (header_.magic != kBCPositionMagic) {
            throw std::runtime_error("BC streaming position file magic mismatch");
        }
        if (header_.format_version != kBCPositionFormatVersion) {
            throw std::runtime_error("BC streaming position file format version mismatch");
        }
        if (header_.header_bytes != kBCPositionHeaderBytes) {
            throw std::runtime_error("BC streaming position file header size mismatch");
        }
        if (header_.key_mode != kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256) {
            throw std::runtime_error("BC streaming position file key mode mismatch");
        }
        if (header_.rank_prefix_bits != kBCRankPrefixBits) {
            throw std::runtime_error("BC streaming position file rank prefix bits mismatch");
        }
        if (header_.rank_prefix_type != kBCPositionRankPrefixTypeUint16) {
            throw std::runtime_error("BC streaming position file rank prefix type mismatch");
        }
        if (header_.rank_payload_align != 8U) {
            throw std::runtime_error("BC streaming position file rank payload align mismatch");
        }
        if (header_.family_unit == 0U ||
            header_.family_unit > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("BC streaming position file invalid family_unit");
        }
        if (header_.axis_base_coord > std::numeric_limits<FamilyCoord>::max() ||
            header_.family_count == 0U ||
            header_.family_count > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("BC streaming position file invalid family axis");
        }
        if (header_.layer_sum > std::numeric_limits<LayerSum>::max()) {
            throw std::runtime_error("BC streaming position file layer_sum exceeds LayerSum");
        }
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(header_.family_count);
        if (header_.axis_coord_table_bytes != axis_coord_bytes) {
            throw std::runtime_error("BC streaming position axis coord table byte size mismatch");
        }

        if (header_.descriptor_count >
            std::numeric_limits<uint64_t>::max() / kBCPositionCellDescriptorBytes) {
            throw std::overflow_error("BC streaming position descriptor table byte size overflow");
        }
        const uint64_t expected_desc_bytes =
            header_.descriptor_count * kBCPositionCellDescriptorBytes;
        if (header_.descriptor_table_bytes != expected_desc_bytes) {
            throw std::runtime_error("BC streaming position descriptor table byte size mismatch");
        }
        if ((header_.bucket_meta_bytes % kBCPositionBucketEntryBytes) != 0U) {
            throw std::runtime_error("BC streaming position bucket metadata stream is not entry-aligned");
        }
        const uint64_t expected_descriptor_offset = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes,
            "BC streaming position expected descriptor offset overflow"
        );
        if (header_.descriptor_table_offset != expected_descriptor_offset) {
            throw std::runtime_error("BC streaming position descriptor table offset mismatch");
        }
        const uint64_t metadata_end = bc_checked_add_u64(
            header_.descriptor_table_offset,
            header_.descriptor_table_bytes,
            "BC streaming position metadata end overflow"
        );
        const uint64_t bucket_end = bc_checked_add_u64(
            header_.bucket_meta_offset,
            header_.bucket_meta_bytes,
            "BC streaming position bucket metadata end overflow"
        );
        const uint64_t rank_end = bc_checked_add_u64(
            header_.rank_payload_offset,
            header_.rank_payload_bytes,
            "BC streaming position rank payload end overflow"
        );
        if ((header_.bucket_meta_bytes != 0U && header_.bucket_meta_offset < metadata_end) ||
            (header_.rank_payload_bytes != 0U && header_.rank_payload_offset < metadata_end)) {
            throw std::runtime_error("BC streaming position data stream overlaps metadata");
        }
        if (header_.bucket_meta_bytes != 0U && header_.rank_payload_bytes != 0U &&
            header_.bucket_meta_offset < rank_end && header_.rank_payload_offset < bucket_end) {
            throw std::runtime_error("BC streaming position bucket and rank streams overlap");
        }
        const uint64_t data_end = std::max(bucket_end, rank_end);
        require_file_range(kBCPositionHeaderBytes, axis_coord_bytes,
            "BC streaming position axis coord table exceeds file");
        require_file_range(header_.descriptor_table_offset, header_.descriptor_table_bytes,
            "BC streaming position descriptor table exceeds file");
        require_file_range(header_.bucket_meta_offset, header_.bucket_meta_bytes,
            "BC streaming position bucket metadata exceeds file");
        require_file_range(header_.rank_payload_offset, header_.rank_payload_bytes,
            "BC streaming position rank payload exceeds file");
        if (data_end > file_size_) {
            throw std::runtime_error("BC streaming position file has trailing or missing bytes");
        }
        if (file_size_ - data_end >= 4096U) {
            throw std::runtime_error("BC streaming position file has excessive trailing padding");
        }
    }

    void read_axis() {
        const uint64_t axis_coord_bytes = bc_axis_coord_table_bytes(header_.family_count);
        if (axis_coord_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC streaming position axis coord table exceeds size_t");
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(axis_coord_bytes));
        if (!bytes.empty()) {
            file_->read_at(kBCPositionHeaderBytes, bytes.data(), axis_coord_bytes);
        }
        std::vector<FamilyCoord> coords;
        coords.reserve(static_cast<size_t>(header_.family_count));
        for (uint32_t i = 0U; i < header_.family_count; ++i) {
            coords.push_back(static_cast<FamilyCoord>(
                bc_load_u32_le(bytes.data() + i * sizeof(uint32_t))
            ));
        }
        axis_ = BCFamilyTable(
            static_cast<LayerSum>(header_.layer_sum),
            static_cast<uint16_t>(header_.family_unit),
            coords
        );
        if (axis_.axis_base_coord() != header_.axis_base_coord) {
            throw std::runtime_error("BC streaming position axis_base_coord does not match coord table");
        }
        const BCCellMatrix matrix(axis_);
        if (header_.descriptor_count != matrix.cell_count()) {
            throw std::runtime_error("BC streaming position descriptor count does not match dense cell matrix");
        }
    }

    void read_descriptors() {
        if (header_.descriptor_table_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC streaming position descriptor table exceeds size_t");
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(header_.descriptor_table_bytes));
        if (!bytes.empty()) {
            file_->read_at(header_.descriptor_table_offset, bytes.data(), header_.descriptor_table_bytes);
        }
        descriptors_.clear();
        descriptors_.reserve(static_cast<size_t>(header_.descriptor_count));
        for (uint64_t i = 0; i < header_.descriptor_count; ++i) {
            const size_t offset = static_cast<size_t>(i * kBCPositionCellDescriptorBytes);
            descriptors_.push_back(
                bc_read_cell_descriptor(
                    bytes.data() + offset,
                    bytes.size() - offset
                )
            );
        }
    }

    void validate_descriptors() const {
        for (const BCPositionCellDescriptor &desc : descriptors_) {
            if (desc.reserved0 != 0U) {
                throw std::runtime_error("BC streaming position descriptor reserved field is non-zero");
            }
            if ((desc.flags_or_padding & ~kBCPositionCellFlagEmpty) != 0U) {
                throw std::runtime_error("BC streaming position descriptor has unknown flags");
            }
            const uint64_t bucket_bytes =
                static_cast<uint64_t>(desc.bucket_count) * kBCPositionBucketEntryBytes;
            if ((desc.bucket_meta_offset % kBCPositionBucketEntryBytes) != 0U ||
                bc_checked_add_u64(desc.bucket_meta_offset, bucket_bytes,
                    "BC streaming position descriptor bucket range overflow") > header_.bucket_meta_bytes) {
                throw std::runtime_error("BC streaming position descriptor bucket range exceeds stream");
            }
            if (bc_checked_add_u64(desc.rank_payload_offset, desc.rank_payload_bytes,
                    "BC streaming position descriptor rank payload range overflow") > header_.rank_payload_bytes) {
                throw std::runtime_error("BC streaming position descriptor rank payload range exceeds stream");
            }
            if (desc.rank_payload_bytes > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("BC streaming position descriptor rank payload exceeds cell-local uint32");
            }
            if (desc.empty()) {
                if (desc.bucket_count != 0U || desc.success_rows != 0U || desc.rank_payload_bytes != 0U) {
                    throw std::runtime_error("BC streaming position empty descriptor carries non-empty data");
                }
                continue;
            }
            if (desc.bucket_count == 0U || desc.success_rows == 0U) {
                throw std::runtime_error("BC streaming position non-empty descriptor has no bucket or success rows");
            }
        }
    }

    void require_file_range(uint64_t offset, uint64_t bytes, const char *label) const {
        const uint64_t end = bc_checked_add_u64(offset, bytes, label);
        if (end > file_size_) {
            throw std::out_of_range(label);
        }
    }

    void append_request(
        std::vector<ExtentRequest> &requests,
        size_t cell_index,
        ExtentKind kind,
        uint64_t offset,
        uint64_t bytes
    ) const {
        if (bytes == 0U) {
            return;
        }
        require_file_range(offset, bytes, "BC streaming position load request exceeds file");
        requests.push_back(ExtentRequest{offset, bytes, cell_index, kind});
    }

    [[nodiscard]] std::vector<LoadedRange> read_coalesced_ranges(
        std::vector<ExtentRequest> requests,
        BCCellLoadStats *stats
    ) const {
        std::vector<BCFileExtent> extents;
        for (const ExtentRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            const uint64_t end = bc_checked_add_u64(
                request.offset,
                request.bytes,
                "BC streaming position request end overflow"
            );
            if (!extents.empty()) {
                BCFileExtent &last = extents.back();
                const uint64_t last_end = bc_checked_add_u64(
                    last.offset,
                    last.bytes,
                    "BC streaming position coalesced extent end overflow"
                );
                constexpr uint64_t kMaxCoalesceGapBytes = 64ULL * 1024ULL;
                constexpr uint64_t kMaxCoalescedExtentBytes = 4ULL * 1024ULL * 1024ULL;
                const uint64_t gap = request.offset > last_end ? request.offset - last_end : 0U;
                if (request.offset >= last.offset &&
                    (request.offset <= last_end ||
                     (gap <= kMaxCoalesceGapBytes &&
                      end - last.offset <= kMaxCoalescedExtentBytes))) {
                    if (end > last_end) {
                        last.bytes = end - last.offset;
                    }
                    continue;
                }
            }
            extents.push_back(BCFileExtent{request.offset, request.bytes});
        }

        std::vector<LoadedRange> ranges;
        ranges.reserve(extents.size());
        for (const BCFileExtent &extent : extents) {
            if (extent.bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC streaming position coalesced extent exceeds size_t");
            }
            LoadedRange range;
            range.offset = extent.offset;
            range.bytes.assign(static_cast<size_t>(extent.bytes), 0U);
            if (stats != nullptr) {
                ++stats->coalesced_extents;
                stats->read_bytes = bc_checked_add_u64(
                    stats->read_bytes,
                    extent.bytes,
                    "BC streaming position read byte count overflow"
                );
            }
            ranges.push_back(std::move(range));
        }

        std::vector<BCFileReadRequest> read_requests;
        read_requests.reserve(ranges.size());
        for (LoadedRange &range : ranges) {
            read_requests.push_back(BCFileReadRequest{
                range.offset,
                range.bytes.data(),
                static_cast<uint64_t>(range.bytes.size())
            });
        }
        BCFileIOStats io_stats;
        file_->read_many(read_requests, &io_stats);
        if (stats != nullptr) {
            stats->backend_read_ops = io_stats.backend_io_count;
            stats->backend_read_bytes = io_stats.backend_bytes;
        }
        return ranges;
    }

    [[nodiscard]] const LoadedRange &find_loaded_range(
        const std::vector<LoadedRange> &ranges,
        uint64_t offset,
        uint64_t bytes
    ) const {
        const uint64_t request_end = bc_checked_add_u64(
            offset,
            bytes,
            "BC streaming position request end overflow"
        );
        for (const LoadedRange &range : ranges) {
            const uint64_t range_end = bc_checked_add_u64(
                range.offset,
                range.bytes.size(),
                "BC streaming position loaded range end overflow"
            );
            if (offset >= range.offset && request_end <= range_end) {
                return range;
            }
        }
        throw std::logic_error("BC streaming position request is not covered by loaded range");
    }

    static void parse_bucket_entries(
        const uint8_t *data,
        uint64_t bytes,
        std::vector<BCBucketEntry> &out
    ) {
        if ((bytes % kBCPositionBucketEntryBytes) != 0U) {
            throw std::runtime_error("BC streaming loaded bucket bytes are not entry-aligned");
        }
        const uint64_t bucket_count = bytes / kBCPositionBucketEntryBytes;
        if (bucket_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC streaming loaded bucket count exceeds size_t");
        }
        out.clear();
        out.reserve(static_cast<size_t>(bucket_count));
        for (uint64_t i = 0U; i < bucket_count; ++i) {
            out.push_back(bc_read_bucket_entry(data + static_cast<size_t>(i * kBCPositionBucketEntryBytes)));
        }
    }

    void validate_loaded_cell(const BCLoadedCell &cell) const {
        const BCPositionCellDescriptor &desc = descriptor(cell.cid);
        if (desc.empty()) {
            if (!cell.buckets.empty() || !cell.rank_payload.empty() || cell.success_rows != 0U) {
                throw std::runtime_error("BC streaming loaded empty cell has data");
            }
            return;
        }
        if (cell.buckets.size() != desc.bucket_count) {
            throw std::runtime_error("BC streaming loaded cell bucket count mismatch");
        }
        if (cell.rank_payload.size() != desc.rank_payload_bytes) {
            throw std::runtime_error("BC streaming loaded cell rank payload size mismatch");
        }
        for (size_t i = 1U; i < cell.buckets.size(); ++i) {
            if (cell.buckets[i - 1U].key >= cell.buckets[i].key) {
                throw std::runtime_error("BC streaming loaded cell bucket keys are not strictly sorted");
            }
        }
        for (const BCBucketEntry &bucket : cell.buckets) {
            const uint32_t bitmap_len = bitmap_len_from_key(lut(), bucket.key);
            const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
            const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
            const uint64_t prefix_end =
                static_cast<uint64_t>(bucket.rank_payload_offset) +
                static_cast<uint64_t>(prefix_count) * sizeof(RankPrefix);
            const uint64_t bitmap_offset = bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
            const uint64_t bitmap_end =
                bitmap_offset + static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
            if (prefix_end > cell.rank_payload.size() || bitmap_end > cell.rank_payload.size()) {
                throw std::runtime_error("BC streaming loaded cell bucket payload range exceeds rank payload");
            }
            if (bucket.success_row_offset >= cell.success_rows) {
                throw std::runtime_error("BC streaming loaded cell bucket success offset exceeds rows");
            }
        }
    }

    std::unique_ptr<BCReadableFile> file_;
    const BCLut *lut_ = nullptr;
    uint64_t file_size_ = 0U;
    BCPositionHeader header_;
    BCFamilyTable axis_;
    std::vector<BCPositionCellDescriptor> descriptors_;
    bool validate_loaded_cells_ = true;
};

} // namespace BC
