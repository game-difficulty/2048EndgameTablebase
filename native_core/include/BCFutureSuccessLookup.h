#pragma once

#include "BCPositionCellLoader.h"
#include "BCPositionFile.h"
#include "BCSolveEdgeKernel.h"
#include "BCSuccessIO.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace BC {

template <typename StorageT>
class BCFutureSuccessLookupView {
public:
    static_assert(
        std::is_same_v<StorageT, uint32_t> || std::is_same_v<StorageT, uint64_t> ||
        std::is_same_v<StorageT, float> || std::is_same_v<StorageT, double>,
        "unsupported BC future success lookup value type"
    );

    BCFutureSuccessLookupView() = default;

    BCFutureSuccessLookupView(
        const BCLut &lut,
        const BCPositionLayerReader &position,
        const BCSuccessLayerReader &success
    ) {
        open(lut, position, success);
    }

    void open(
        const BCLut &lut,
        const BCPositionLayerReader &position,
        const BCSuccessLayerReader &success
    ) {
        if (!bc_success_dtype_matches_type<StorageT>(success.dtype_mode())) {
            throw std::invalid_argument("BC future success lookup dtype does not match storage type");
        }
        if (success.row_width() == 0U) {
            throw std::invalid_argument("BC future success lookup row_width must be non-zero");
        }
        lut_ = &lut;
        position_ = &position;
        row_width_ = success.row_width();
        owned_loaded_position_cells_.clear();
        owned_values_.clear();
        owned_flat_values_ = {};
        keep_rows_ = nullptr;
        keep_row_count_ = 0U;
        owned_values_.reserve(static_cast<size_t>(
            bc_success_total_values_for(position, row_width_)
        ));
        cells_.clear();
        cells_.resize(position.cell_count());

        for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
            const BCPositionCellDescriptor &desc = position.descriptor(cid);
            CellIndex &cell = cells_[static_cast<size_t>(cid)];
            cell.value_offset = owned_values_.size();
            std::vector<StorageT> values = success.template read_cell_typed<StorageT>(cid);
            const uint64_t expected_values =
                static_cast<uint64_t>(desc.success_rows) * row_width_;
            if (values.size() != expected_values) {
                throw std::runtime_error("BC future success lookup value count mismatch");
            }
            cell.value_count = values.size();
            owned_values_.insert(owned_values_.end(), values.begin(), values.end());
            if (desc.empty()) {
                if (cell.value_count != 0U) {
                    throw std::runtime_error("BC future empty cell has success values");
                }
                continue;
            }

            cell.rank_payload = position.rank_payload_for_cell(cid);
            const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
            const uint32_t capacity = direct_capacity_for_bucket_count(buckets.size);
            cell.entries.assign(capacity, DirectEntry{});
            cell.mask = capacity - 1U;
            for (uint32_t i = 0U; i < buckets.size; ++i) {
                const BCBucketEntry &bucket = buckets.data[i];
                uint32_t slot = static_cast<uint32_t>(mix_u64(bucket.key)) & cell.mask;
                while (!entry_empty(cell.entries[slot])) {
                    if (cell.entries[slot].key == bucket.key) {
                        throw std::runtime_error("BC future direct lookup saw duplicate bucket key");
                    }
                    slot = (slot + 1U) & cell.mask;
                }
                DirectEntry entry;
                entry.key = bucket.key;
                const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
                entry.bitmap_offset = bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
                entry.success_row_offset = bucket.success_row_offset;
                validate_entry_payload_range(cell, entry, bitmap_len);
                populate_entry_word_rank_bases(cell, entry, bitmap_len);
                cell.entries[slot] = entry;
            }
        }
        value_data_ = owned_values_.empty() ? nullptr : owned_values_.data();
        value_count_ = owned_values_.size();
    }

    void open_loaded(
        const BCLut &lut,
        uint32_t cell_count,
        std::vector<BCLoadedCell> position_cells,
        const std::vector<BCLoadedSuccessCell> &success_cells,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC future loaded lookup row_width must be non-zero");
        }
        if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
            throw std::invalid_argument("BC future loaded lookup dtype does not match storage type");
        }
        if (position_cells.size() != success_cells.size()) {
            throw std::invalid_argument("BC future loaded lookup cell count mismatch");
        }
        lut_ = &lut;
        position_ = nullptr;
        row_width_ = row_width;
        owned_values_.clear();
        owned_flat_values_ = {};
        value_data_ = nullptr;
        value_count_ = 0U;
        keep_rows_ = nullptr;
        keep_row_count_ = 0U;
        cells_.clear();
        cells_.resize(cell_count);
        owned_loaded_position_cells_ = std::move(position_cells);

        uint64_t total_expected_values = 0U;
        for (const BCLoadedSuccessCell &loaded_success : success_cells) {
            if (loaded_success.row_width != row_width ||
                !bc_success_dtype_matches_type<StorageT>(loaded_success.dtype_mode())) {
                throw std::invalid_argument("BC future loaded lookup success dtype/row_width mismatch");
            }
            if (loaded_success.success_rows >
                std::numeric_limits<uint64_t>::max() / row_width_) {
                throw std::overflow_error("BC future loaded lookup value count exceeds uint64");
            }
            total_expected_values = bc_checked_add_u64(
                total_expected_values,
                static_cast<uint64_t>(loaded_success.success_rows) * row_width_,
                "BC future loaded lookup total value count overflow"
            );
        }
        if (total_expected_values > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC future loaded lookup total value count exceeds size_t");
        }
        owned_values_.reserve(static_cast<size_t>(total_expected_values));

        for (size_t loaded_i = 0U; loaded_i < owned_loaded_position_cells_.size(); ++loaded_i) {
            BCLoadedCell &loaded_position = owned_loaded_position_cells_[loaded_i];
            const BCLoadedSuccessCell &loaded_success = success_cells[loaded_i];
            if (loaded_position.cid != loaded_success.cid) {
                throw std::invalid_argument("BC future loaded lookup cid mismatch");
            }
            if (loaded_position.cid >= cell_count) {
                throw std::out_of_range("BC future loaded lookup cid out of range");
            }
            if (loaded_position.success_rows != loaded_success.success_rows) {
                throw std::invalid_argument("BC future loaded lookup success row mismatch");
            }

            CellIndex &cell = cells_[static_cast<size_t>(loaded_position.cid)];
            if (!cell.entries.empty() || cell.value_count != 0U) {
                throw std::invalid_argument("BC future loaded lookup duplicate loaded cid");
            }
            cell.value_offset = owned_values_.size();
            const uint64_t expected_values =
                static_cast<uint64_t>(loaded_position.success_rows) * row_width_;
            if (expected_values > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC future loaded lookup value count exceeds size_t");
            }
            const uint64_t expected_bytes = expected_values * sizeof(StorageT);
            if (loaded_success.raw_bytes.size() != expected_bytes) {
                throw std::invalid_argument("BC future loaded lookup raw success byte mismatch");
            }
            cell.value_count = static_cast<size_t>(expected_values);
            if constexpr (std::is_same_v<StorageT, uint32_t>) {
                if (loaded_success.values.size() == cell.value_count) {
                    owned_values_.insert(
                        owned_values_.end(),
                        loaded_success.values.begin(),
                        loaded_success.values.end()
                    );
                } else {
                    for (uint64_t value_i = 0U; value_i < expected_values; ++value_i) {
                        owned_values_.push_back(
                            bc_load_success_value_le<StorageT>(
                                loaded_success.raw_bytes.data() +
                                static_cast<size_t>(value_i * sizeof(StorageT))
                            )
                        );
                    }
                }
            } else {
                for (uint64_t value_i = 0U; value_i < expected_values; ++value_i) {
                    owned_values_.push_back(
                        bc_load_success_value_le<StorageT>(
                            loaded_success.raw_bytes.data() +
                            static_cast<size_t>(value_i * sizeof(StorageT))
                        )
                    );
                }
            }

            const BCLoadedCellView view = loaded_position.view();
            if (view.empty()) {
                if (cell.value_count != 0U) {
                    throw std::runtime_error("BC future loaded empty cell has success values");
                }
                continue;
            }
            cell.rank_payload = view.rank_payload;
            const uint32_t capacity = direct_capacity_for_bucket_count(view.buckets.size);
            cell.entries.assign(capacity, DirectEntry{});
            cell.mask = capacity - 1U;
            for (uint32_t i = 0U; i < view.buckets.size; ++i) {
                const BCBucketEntry &bucket = view.buckets.data[i];
                uint32_t slot = static_cast<uint32_t>(mix_u64(bucket.key)) & cell.mask;
                while (!entry_empty(cell.entries[slot])) {
                    if (cell.entries[slot].key == bucket.key) {
                        throw std::runtime_error("BC future loaded direct lookup saw duplicate bucket key");
                    }
                    slot = (slot + 1U) & cell.mask;
                }
                DirectEntry entry;
                entry.key = bucket.key;
                const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
                entry.bitmap_offset = bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
                entry.success_row_offset = bucket.success_row_offset;
                validate_entry_payload_range(cell, entry, bitmap_len);
                populate_entry_word_rank_bases(cell, entry, bitmap_len);
                cell.entries[slot] = entry;
            }
        }
        value_data_ = owned_values_.empty() ? nullptr : owned_values_.data();
        value_count_ = owned_values_.size();
    }

    void open_loaded_flat_values(
        const BCLut &lut,
        uint32_t cell_count,
        std::vector<BCLoadedCell> position_cells,
        const std::vector<uint64_t> &cell_value_offsets,
        std::vector<StorageT> values,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        BCSuccessOwnedValues<StorageT> owned_values;
        owned_values.vector_values = std::move(values);
        open_loaded_flat_owned_values(
            lut,
            cell_count,
            std::move(position_cells),
            cell_value_offsets,
            std::move(owned_values),
            row_width,
            dtype
        );
    }

    void open_loaded_flat_owned_values(
        const BCLut &lut,
        uint32_t cell_count,
        std::vector<BCLoadedCell> position_cells,
        const std::vector<uint64_t> &cell_value_offsets,
        BCSuccessOwnedValues<StorageT> values,
        uint32_t row_width,
        BCSuccessDTypeMode dtype
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC future loaded flat lookup row_width must be non-zero");
        }
        if (!bc_success_dtype_matches_type<StorageT>(dtype)) {
            throw std::invalid_argument("BC future loaded flat lookup dtype does not match storage type");
        }
        if (cell_value_offsets.size() != static_cast<size_t>(cell_count) + 1U) {
            throw std::invalid_argument("BC future loaded flat lookup cell offset count mismatch");
        }
        if (cell_value_offsets.back() != static_cast<uint64_t>(values.size())) {
            throw std::invalid_argument("BC future loaded flat lookup value count mismatch");
        }
        lut_ = &lut;
        position_ = nullptr;
        row_width_ = row_width;
        owned_values_.clear();
        owned_flat_values_ = std::move(values);
        value_data_ = owned_flat_values_.empty() ? nullptr : owned_flat_values_.data();
        value_count_ = owned_flat_values_.size();
        keep_rows_ = nullptr;
        keep_row_count_ = 0U;
        cells_.clear();
        cells_.resize(cell_count);
        owned_loaded_position_cells_ = std::move(position_cells);

        for (BCLoadedCell &loaded_position : owned_loaded_position_cells_) {
            if (loaded_position.cid >= cell_count) {
                throw std::out_of_range("BC future loaded flat lookup cid out of range");
            }
            CellIndex &cell = cells_[static_cast<size_t>(loaded_position.cid)];
            if (!cell.entries.empty() || cell.value_count != 0U) {
                throw std::invalid_argument("BC future loaded flat lookup duplicate loaded cid");
            }
            cell.value_offset = static_cast<size_t>(cell_value_offsets[static_cast<size_t>(loaded_position.cid)]);
            const uint64_t expected_values =
                static_cast<uint64_t>(loaded_position.success_rows) * row_width_;
            if (expected_values > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC future loaded flat lookup value count exceeds size_t");
            }
            cell.value_count = static_cast<size_t>(expected_values);
            if (static_cast<uint64_t>(cell.value_offset) > value_count_ ||
                expected_values > value_count_ - static_cast<uint64_t>(cell.value_offset)) {
                throw std::out_of_range("BC future loaded flat lookup cell values exceed payload");
            }

            const BCLoadedCellView view = loaded_position.view();
            if (view.empty()) {
                if (cell.value_count != 0U) {
                    throw std::runtime_error("BC future loaded flat empty cell has success values");
                }
                continue;
            }
            cell.rank_payload = view.rank_payload;
            const uint32_t capacity = direct_capacity_for_bucket_count(view.buckets.size);
            cell.entries.assign(capacity, DirectEntry{});
            cell.mask = capacity - 1U;
            for (uint32_t i = 0U; i < view.buckets.size; ++i) {
                const BCBucketEntry &bucket = view.buckets.data[i];
                uint32_t slot = static_cast<uint32_t>(mix_u64(bucket.key)) & cell.mask;
                while (!entry_empty(cell.entries[slot])) {
                    if (cell.entries[slot].key == bucket.key) {
                        throw std::runtime_error("BC future loaded flat direct lookup saw duplicate bucket key");
                    }
                    slot = (slot + 1U) & cell.mask;
                }
                DirectEntry entry;
                entry.key = bucket.key;
                const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
                entry.bitmap_offset = bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
                entry.success_row_offset = bucket.success_row_offset;
                validate_entry_payload_range(cell, entry, bitmap_len);
                populate_entry_word_rank_bases(cell, entry, bitmap_len);
                cell.entries[slot] = entry;
            }
        }
    }

    void open_flat(
        const BCLut &lut,
        const BCPositionLayerReader &position,
        const StorageT *values,
        size_t value_count,
        uint32_t row_width,
        const uint8_t *keep_rows = nullptr,
        size_t keep_row_count = 0U
    ) {
        if (row_width == 0U) {
            throw std::invalid_argument("BC future success flat lookup row_width must be non-zero");
        }
        const uint64_t expected = bc_success_total_values_for(position, row_width);
        if (expected != value_count) {
            throw std::invalid_argument("BC future success flat lookup value count mismatch");
        }
        if (value_count != 0U && values == nullptr) {
            throw std::invalid_argument("BC future success flat lookup values pointer is null");
        }
        const size_t expected_rows = static_cast<size_t>(expected / row_width);
        if (keep_rows != nullptr && keep_row_count != expected_rows) {
            throw std::invalid_argument("BC future success flat lookup keep row count mismatch");
        }
        lut_ = &lut;
        position_ = &position;
        row_width_ = row_width;
        owned_loaded_position_cells_.clear();
        owned_values_.clear();
        owned_flat_values_ = {};
        value_data_ = values;
        value_count_ = value_count;
        keep_rows_ = keep_rows;
        keep_row_count_ = keep_rows == nullptr ? 0U : keep_row_count;
        cells_.clear();
        cells_.resize(position.cell_count());

        size_t value_cursor = 0U;
        for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
            const BCPositionCellDescriptor &desc = position.descriptor(cid);
            CellIndex &cell = cells_[static_cast<size_t>(cid)];
            const uint64_t expected_values =
                static_cast<uint64_t>(desc.success_rows) * row_width_;
            cell.value_offset = value_cursor;
            cell.value_count = static_cast<size_t>(expected_values);
            value_cursor += static_cast<size_t>(expected_values);
            if (desc.empty()) {
                if (cell.value_count != 0U) {
                    throw std::runtime_error("BC future flat empty cell has success values");
                }
                continue;
            }

            cell.rank_payload = position.rank_payload_for_cell(cid);
            const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
            const uint32_t capacity = direct_capacity_for_bucket_count(buckets.size);
            cell.entries.assign(capacity, DirectEntry{});
            cell.mask = capacity - 1U;
            for (uint32_t i = 0U; i < buckets.size; ++i) {
                const BCBucketEntry &bucket = buckets.data[i];
                uint32_t slot = static_cast<uint32_t>(mix_u64(bucket.key)) & cell.mask;
                while (!entry_empty(cell.entries[slot])) {
                    if (cell.entries[slot].key == bucket.key) {
                        throw std::runtime_error("BC future direct lookup saw duplicate bucket key");
                    }
                    slot = (slot + 1U) & cell.mask;
                }
                DirectEntry entry;
                entry.key = bucket.key;
                const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
                entry.bitmap_offset = bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
                entry.success_row_offset = bucket.success_row_offset;
                validate_entry_payload_range(cell, entry, bitmap_len);
                populate_entry_word_rank_bases(cell, entry, bitmap_len);
                cell.entries[slot] = entry;
            }
        }
    }

    void rebind_flat_values(
        const BCPositionLayerReader &position,
        const StorageT *values,
        size_t value_count,
        const uint8_t *keep_rows = nullptr,
        size_t keep_row_count = 0U
    ) {
        position_ = &position;
        value_data_ = values;
        value_count_ = value_count;
        keep_rows_ = keep_rows;
        keep_row_count_ = keep_rows == nullptr ? 0U : keep_row_count;
    }

    [[nodiscard]] uint32_t row_width() const {
        return row_width_;
    }

    [[nodiscard]] bool lookup(uint64_t canonical_board, StorageT &value_out, uint32_t lane = 0U) const {
        value_out = StorageT{};
        if (position_ == nullptr || lut_ == nullptr) {
            return false;
        }
        const BCBoardEncodedPosition encoded =
            encode_canonical_board_position(*lut_, position_->axis(), canonical_board);
        return lookup_encoded(encoded, lane, value_out);
    }

    [[nodiscard]] BCSolveLookupResult<StorageT> lookup(
        const BCSolvePreparedQuery &query,
        uint32_t lane
    ) const {
        StorageT value{};
        const bool found = lookup_query(query, lane, value);
        return BCSolveLookupResult<StorageT>{found, value};
    }

    [[nodiscard]] bool lookup_query(
        const BCSolvePreparedQuery &query,
        uint32_t lane,
        StorageT &value_out
    ) const {
        value_out = StorageT{};
        if (lut_ == nullptr) {
            return false;
        }
        BCBoardEncodedPosition encoded;
        encoded.cid = query.cid;
        encoded.key = query.key;
        encoded.rank = query.rank;
        encoded.bitmap_len = bitmap_len_from_trusted_key(*lut_, query.key);
        encoded.valid = true;
        return lookup_encoded(encoded, lane, value_out);
    }

    [[nodiscard]] uint64_t reduce_max_queries(
        const std::vector<BCSolvePreparedQuery> &queries,
        StorageT *best,
        size_t best_count,
        uint32_t lane,
        BCSolveEdgeStats *stats = nullptr,
        bool trusted_queries = false
    ) const {
        if (lane >= row_width_) {
            throw std::out_of_range("BC future batch lookup lane out of range");
        }
        if (best_count != 0U && best == nullptr) {
            throw std::invalid_argument("BC future batch lookup best pointer is null");
        }
        (void)stats;
        constexpr uint32_t kBatch = 512U;
        uint32_t value_indices[kBatch];
        for (uint32_t base = 0U; base < static_cast<uint32_t>(queries.size()); base += kBatch) {
            const uint32_t count = std::min<uint32_t>(kBatch, static_cast<uint32_t>(queries.size()) - base);
            if (trusted_queries) {
                if (keep_rows_ == nullptr) {
                    lookup_success_indices<true, false>(queries.data() + base, value_indices, count, lane);
                } else {
                    lookup_success_indices<true, true>(queries.data() + base, value_indices, count, lane);
                }
            } else {
                if (keep_rows_ == nullptr) {
                    lookup_success_indices<false, false>(queries.data() + base, value_indices, count, lane);
                } else {
                    lookup_success_indices<false, true>(queries.data() + base, value_indices, count, lane);
                }
            }
            for (uint32_t i = 0U; i < count; ++i) {
                if (value_indices[i] == kMissingValueIndex) {
                    continue;
                }
                const BCSolvePreparedQuery &query = queries[static_cast<size_t>(base) + i];
                if (query.ref >= best_count) {
                    continue;
                }
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(value_data_ + value_indices[i], 0, 1);
#endif
            }
            for (uint32_t i = 0U; i < count; ++i) {
                if (value_indices[i] == kMissingValueIndex) {
                    continue;
                }
                const BCSolvePreparedQuery &query = queries[static_cast<size_t>(base) + i];
                if (query.ref >= best_count) {
                    continue;
                }
                StorageT &slot = best[query.ref];
                const StorageT value = value_data_[value_indices[i]];
                if (value > slot) {
                    slot = value;
                }
            }
        }
        return 0U;
    }

    template <typename SumT>
    // Queries must be grouped by ref. Each group contributes
    // max(found values) - zero_value to sums[ref >> 4]; callers initialize
    // each sum with empty_count * zero_value when zero is not numeric 0.
    [[nodiscard]] uint64_t reduce_grouped_query_sums(
        const std::vector<BCSolvePreparedQuery> &queries,
        SumT *sums,
        size_t sum_count,
        uint32_t lane,
        StorageT zero_value,
        bool trusted_queries = false
    ) const {
        static_assert(
            std::is_arithmetic_v<SumT>,
            "BC future grouped query sums require arithmetic output"
        );
        if (lane >= row_width_) {
            throw std::out_of_range("BC future grouped batch lookup lane out of range");
        }
        if (sum_count != 0U && sums == nullptr) {
            throw std::invalid_argument("BC future grouped batch lookup sums pointer is null");
        }
        constexpr uint32_t kBatch = 512U;
        uint32_t value_indices[kBatch];
        uint32_t current_ref = std::numeric_limits<uint32_t>::max();
        StorageT current_best = zero_value;
        bool current_found = false;
        uint64_t found_count = 0U;
        const uint64_t ref_limit =
            std::min<uint64_t>(
                static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1ULL,
                static_cast<uint64_t>(sum_count) * static_cast<uint64_t>(kBCBoardCellCount)
            );

        auto flush_group = [&]() {
            if (!current_found) {
                return;
            }
            const size_t sum_index = static_cast<size_t>(current_ref >> 4U);
            if (sum_index < sum_count) {
                sums[sum_index] +=
                    static_cast<SumT>(current_best) - static_cast<SumT>(zero_value);
            }
        };

        for (uint32_t base = 0U; base < static_cast<uint32_t>(queries.size()); base += kBatch) {
            const uint32_t count = std::min<uint32_t>(kBatch, static_cast<uint32_t>(queries.size()) - base);
            if (trusted_queries) {
                if (keep_rows_ == nullptr) {
                    lookup_success_indices<true, false>(queries.data() + base, value_indices, count, lane);
                } else {
                    lookup_success_indices<true, true>(queries.data() + base, value_indices, count, lane);
                }
            } else {
                if (keep_rows_ == nullptr) {
                    lookup_success_indices<false, false>(queries.data() + base, value_indices, count, lane);
                } else {
                    lookup_success_indices<false, true>(queries.data() + base, value_indices, count, lane);
                }
            }
            for (uint32_t i = 0U; i < count; ++i) {
                if (value_indices[i] == kMissingValueIndex) {
                    continue;
                }
                const BCSolvePreparedQuery &query = queries[static_cast<size_t>(base) + i];
                if (static_cast<uint64_t>(query.ref) >= ref_limit) {
                    continue;
                }
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(value_data_ + value_indices[i], 0, 1);
#endif
            }
            for (uint32_t i = 0U; i < count; ++i) {
                if (value_indices[i] == kMissingValueIndex) {
                    continue;
                }
                const BCSolvePreparedQuery &query = queries[static_cast<size_t>(base) + i];
                if (static_cast<uint64_t>(query.ref) >= ref_limit) {
                    continue;
                }
                if (query.ref != current_ref) {
                    flush_group();
                    current_ref = query.ref;
                    current_best = zero_value;
                    current_found = false;
                }
                const StorageT value = value_data_[value_indices[i]];
                if (!current_found || value > current_best) {
                    current_best = value;
                    current_found = true;
                }
                ++found_count;
            }
        }
        flush_group();
        return found_count;
    }

private:
    static constexpr uint32_t kEmptyEntryOffset = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kMissingValueIndex = std::numeric_limits<uint32_t>::max();

    struct DirectEntry {
        uint64_t key = 0U;
        uint32_t bitmap_offset = kEmptyEntryOffset;
        uint32_t success_row_offset = 0U;
    };

    static_assert(sizeof(DirectEntry) == 16U, "BC future direct entry should stay compact");

    struct CellIndex {
        BCRankPayloadView rank_payload = {};
        size_t value_offset = 0U;
        size_t value_count = 0U;
        std::vector<DirectEntry> entries;
        std::vector<uint32_t> word_rank_bases;
        uint32_t mask = 0U;
    };

    [[nodiscard]] static uint64_t mix_u64(uint64_t value) {
        value ^= value >> 33U;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33U;
        value *= 0xc4ceb9fe1a85ec53ULL;
        value ^= value >> 33U;
        return value;
    }

    [[nodiscard]] static uint32_t next_power_of_two_u32(uint64_t value) {
        uint64_t cap = 1U;
        while (cap < value) {
            cap <<= 1U;
            if (cap > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::overflow_error("BC future direct lookup capacity exceeds uint32");
            }
        }
        return static_cast<uint32_t>(std::max<uint64_t>(cap, 1U));
    }

    [[nodiscard]] static uint32_t direct_capacity_for_bucket_count(uint32_t bucket_count) {
        const uint64_t required = std::max<uint64_t>(
            4ULL,
            (static_cast<uint64_t>(bucket_count) * 16ULL + 4ULL) / 5ULL
        );
        return next_power_of_two_u32(required);
    }

    [[nodiscard]] static bool entry_empty(const DirectEntry &entry) noexcept {
        return entry.bitmap_offset == kEmptyEntryOffset;
    }

    [[nodiscard]] static uint32_t aligned_prefix_bytes_for_bitmap_len(uint32_t bitmap_len) noexcept {
        const uint32_t prefix_count =
            (bitmap_len + kBCRankPrefixBits - 1U) / kBCRankPrefixBits;
        const uint32_t prefix_bytes =
            prefix_count * static_cast<uint32_t>(sizeof(RankPrefix));
        return (prefix_bytes + 7U) & ~7U;
    }

    [[nodiscard]] static uint32_t entry_prefix_offset_unchecked(
        const DirectEntry &entry,
        uint32_t bitmap_len
    ) noexcept {
        return entry.bitmap_offset - aligned_prefix_bytes_for_bitmap_len(bitmap_len);
    }

    static void validate_entry_payload_range(
        const CellIndex &cell,
        const DirectEntry &entry,
        BucketBitmapLen bitmap_len
    ) {
        if (bitmap_len == 0U || entry.bitmap_offset == kEmptyEntryOffset) {
            throw std::logic_error("BC future direct lookup saw invalid bucket metadata");
        }
        const uint32_t prefix_count = prefix_count_for_bits(bitmap_len);
        const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
        const uint32_t prefix_offset = entry_prefix_offset_unchecked(entry, bitmap_len);
        const uint32_t bitmap_offset = entry.bitmap_offset;
        const uint64_t prefix_end =
            static_cast<uint64_t>(prefix_offset) +
            static_cast<uint64_t>(prefix_count) * sizeof(RankPrefix);
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
        if (prefix_end > cell.rank_payload.size || bitmap_end > cell.rank_payload.size) {
            throw std::out_of_range("BC future direct lookup payload range is truncated");
        }
    }

    static void ensure_word_rank_base_capacity(CellIndex &cell) {
        const size_t words = (static_cast<size_t>(cell.rank_payload.size) + sizeof(uint64_t) - 1U) /
            sizeof(uint64_t);
        if (cell.word_rank_bases.size() < words) {
            cell.word_rank_bases.assign(words, 0U);
        }
    }

    static void populate_entry_word_rank_bases(
        CellIndex &cell,
        const DirectEntry &entry,
        BucketBitmapLen bitmap_len
    ) {
        const uint32_t bitmap_word_count = words_for_bits(bitmap_len);
        if (bitmap_word_count == 0U) {
            return;
        }
        ensure_word_rank_base_capacity(cell);
        const uint32_t prefix_offset = entry_prefix_offset_unchecked(entry, bitmap_len);
        const uint32_t bitmap_offset = entry.bitmap_offset;
        const uint8_t *bitmap_words = cell.rank_payload.data + bitmap_offset;
        const size_t rank_base_offset = static_cast<size_t>(bitmap_offset) / sizeof(uint64_t);
        for (uint32_t word_idx = 0U; word_idx < bitmap_word_count; ++word_idx) {
            const uint32_t block = word_idx / (kBCRankPrefixBits / kBCBitmapWordBits);
            const uint32_t word_in_block = word_idx & ((kBCRankPrefixBits / kBCBitmapWordBits) - 1U);
            uint32_t rank_before = load_u16_le(
                cell.rank_payload.data +
                static_cast<size_t>(prefix_offset) +
                static_cast<size_t>(block) * sizeof(RankPrefix)
            );
            for (uint32_t offset = 0U; offset < word_in_block; ++offset) {
                rank_before += popcount64(load_u64_le(
                    bitmap_words +
                    static_cast<size_t>(
                        word_idx - word_in_block + offset
                    ) * sizeof(uint64_t)
                ));
            }
            cell.word_rank_bases[rank_base_offset + word_idx] = rank_before;
        }
    }

    [[nodiscard]] const DirectEntry *find_entry(const CellIndex &cell, uint64_t key) const {
        if (cell.entries.empty()) {
            return nullptr;
        }
        uint32_t slot = static_cast<uint32_t>(mix_u64(key)) & cell.mask;
        while (true) {
            const DirectEntry &entry = cell.entries[slot];
            if (entry_empty(entry)) {
                return nullptr;
            }
            if (entry.key == key) {
                return &entry;
            }
            slot = (slot + 1U) & cell.mask;
        }
    }

    template <bool TrustedQueries, bool UseKeepRows>
    void lookup_success_indices(
        const BCSolvePreparedQuery *queries,
        uint32_t *value_indices,
        uint32_t count,
        uint32_t lane
    ) const {
        if (queries == nullptr || value_indices == nullptr) {
            throw std::invalid_argument("BC future batch lookup pointer is null");
        }
        uint32_t slots[512U];
        const DirectEntry *entries[512U];
        uint16_t retry_a[512U];
        uint16_t retry_b[512U];
        uint16_t *retry = retry_a;
        uint16_t *next_retry = retry_b;
        for (uint32_t i = 0U; i < count; ++i) {
            value_indices[i] = kMissingValueIndex;
            slots[i] = kMissingValueIndex;
            entries[i] = nullptr;
            if constexpr (!TrustedQueries) {
                if (queries[i].cid >= cells_.size()) {
                    continue;
                }
            }
            const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
            if (cell.entries.empty()) {
                continue;
            }
            slots[i] = static_cast<uint32_t>(mix_u64(queries[i].key)) & cell.mask;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&cell.entries[slots[i]], 0, 1);
#endif
        }

        uint32_t retry_count = 0U;
        for (uint32_t i = 0U; i < count; ++i) {
            if (slots[i] == kMissingValueIndex) {
                continue;
            }
            const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
            const DirectEntry &entry = cell.entries[slots[i]];
            if (entry_empty(entry)) {
                continue;
            }
            if (entry.key == queries[i].key) {
                entries[i] = &entry;
            } else {
                slots[i] = (slots[i] + 1U) & cell.mask;
                retry[retry_count++] = static_cast<uint16_t>(i);
            }
        }
        while (retry_count != 0U) {
            for (uint32_t r = 0U; r < retry_count; ++r) {
                const uint32_t i = retry[r];
                const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&cell.entries[slots[i]], 0, 1);
#endif
            }
            uint32_t next_retry_count = 0U;
            for (uint32_t r = 0U; r < retry_count; ++r) {
                const uint32_t i = retry[r];
                const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
                const DirectEntry &entry = cell.entries[slots[i]];
                if (entry_empty(entry)) {
                    continue;
                }
                if (entry.key == queries[i].key) {
                    entries[i] = &entry;
                } else {
                    slots[i] = (slots[i] + 1U) & cell.mask;
                    next_retry[next_retry_count++] = static_cast<uint16_t>(i);
                }
            }
            std::swap(retry, next_retry);
            retry_count = next_retry_count;
        }

        for (uint32_t i = 0U; i < count; ++i) {
            if (entries[i] == nullptr) {
                continue;
            }
            const DirectEntry &entry = *entries[i];
            if constexpr (!TrustedQueries) {
                const BucketBitmapLen bitmap_len = bitmap_len_from_trusted_key(*lut_, queries[i].key);
                if (queries[i].rank >= bitmap_len) {
                    entries[i] = nullptr;
                    continue;
                }
            }
            const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
            const uint32_t word_idx = static_cast<uint32_t>(queries[i].rank) >> 6U;
            const uint32_t bitmap_offset = entry.bitmap_offset;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(
                cell.rank_payload.data + bitmap_offset + static_cast<size_t>(word_idx) * sizeof(uint64_t),
                0,
                1
            );
#endif
        }

        for (uint32_t i = 0U; i < count; ++i) {
            if (entries[i] == nullptr) {
                continue;
            }
            const DirectEntry &entry = *entries[i];
            const CellIndex &cell = cells_[static_cast<size_t>(queries[i].cid)];
            const uint32_t rank = queries[i].rank;
            const uint32_t bit_in_word = rank & (kBCBitmapWordBits - 1U);
            const uint32_t target_word = rank >> 6U;
            const uint32_t bitmap_offset = entry.bitmap_offset;
            const uint8_t *bitmap_words = cell.rank_payload.data + bitmap_offset;
            const uint64_t target = load_u64_le(
                bitmap_words + static_cast<size_t>(target_word) * sizeof(uint64_t)
            );
            if (((target >> bit_in_word) & 1ULL) == 0ULL) {
                continue;
            }
            const size_t rank_base_index =
                static_cast<size_t>(bitmap_offset / sizeof(uint64_t)) + target_word;
            if (rank_base_index >= cell.word_rank_bases.size()) {
                throw std::out_of_range("BC future lookup word rank base is missing");
            }
            uint32_t rank_before = cell.word_rank_bases[rank_base_index];
            if (bit_in_word != 0U) {
                rank_before += popcount64(target & ((1ULL << bit_in_word) - 1ULL));
            }
            const uint64_t local_row =
                static_cast<uint64_t>(entry.success_row_offset) +
                static_cast<uint64_t>(rank_before);
            if constexpr (UseKeepRows) {
                const uint64_t global_row =
                    static_cast<uint64_t>(cell.value_offset / row_width_) + local_row;
                if (global_row >= keep_row_count_ ||
                    keep_rows_[static_cast<size_t>(global_row)] == 0U) {
                    continue;
                }
            }
            const uint64_t local_value_index = local_row * row_width_ + lane;
            if (local_value_index >= cell.value_count) {
                throw std::out_of_range("BC future batch lookup local row exceeds success values");
            }
            const uint64_t global_index = static_cast<uint64_t>(cell.value_offset) + local_value_index;
            if (global_index >= value_count_ ||
                global_index >= static_cast<uint64_t>(kMissingValueIndex)) {
                throw std::out_of_range("BC future batch lookup value index exceeds success values");
            }
            value_indices[i] = static_cast<uint32_t>(global_index);
        }
    }

    [[nodiscard]] bool lookup_encoded(
        const BCBoardEncodedPosition &encoded,
        uint32_t lane,
        StorageT &value_out
    ) const {
        value_out = StorageT{};
        if (!encoded.valid || lut_ == nullptr) {
            return false;
        }
        if (lane >= row_width_) {
            throw std::out_of_range("BC future direct lookup lane out of range");
        }
        if (encoded.cid >= cells_.size()) {
            return false;
        }
        const CellIndex &cell = cells_[static_cast<size_t>(encoded.cid)];
        const DirectEntry *entry = find_entry(cell, encoded.key);
        if (entry == nullptr || encoded.bitmap_len == 0U || encoded.rank >= encoded.bitmap_len) {
            return false;
        }

        const uint32_t prefix_offset = entry_prefix_offset_unchecked(*entry, encoded.bitmap_len);
        const uint32_t bitmap_offset = entry->bitmap_offset;
        const BCBitmapRankResult rank_result = bitmap_test_and_rank_le_bytes(
            cell.rank_payload.data + prefix_offset,
            prefix_count_for_bits(encoded.bitmap_len),
            cell.rank_payload.data + bitmap_offset,
            words_for_bits(encoded.bitmap_len),
            encoded.rank
        );
        if (!rank_result.found) {
            return false;
        }
        const uint64_t local_row =
            static_cast<uint64_t>(entry->success_row_offset) +
            static_cast<uint64_t>(rank_result.rank_before);
        const uint64_t global_row =
            static_cast<uint64_t>(cell.value_offset / row_width_) + local_row;
        if (keep_rows_ != nullptr) {
            if (global_row >= keep_row_count_ ||
                keep_rows_[static_cast<size_t>(global_row)] == 0U) {
                return false;
            }
        }
        const uint64_t value_index = local_row * row_width_ + lane;
        if (value_index >= cell.value_count) {
            throw std::out_of_range("BC future direct lookup local row exceeds success values");
        }
        const uint64_t global_index = static_cast<uint64_t>(cell.value_offset) + value_index;
        if (global_index >= value_count_) {
            throw std::out_of_range("BC future direct lookup value index exceeds success values");
        }
        value_out = value_data_[static_cast<size_t>(global_index)];
        return true;
    }

    const BCLut *lut_ = nullptr;
    const BCPositionLayerReader *position_ = nullptr;
    uint32_t row_width_ = 0U;
    std::vector<StorageT> owned_values_;
    BCSuccessOwnedValues<StorageT> owned_flat_values_;
    std::vector<BCLoadedCell> owned_loaded_position_cells_;
    const StorageT *value_data_ = nullptr;
    size_t value_count_ = 0U;
    const uint8_t *keep_rows_ = nullptr;
    size_t keep_row_count_ = 0U;
    std::vector<CellIndex> cells_;
};

} // namespace BC
