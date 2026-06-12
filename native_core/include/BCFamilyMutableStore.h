#pragma once

#include "BCGenerationBlobIO.h"
#include "BCFamilyGenerationScheduler.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace BC {

enum class BCMutableCellState : uint8_t {
    Empty,
    Resident,
    Dumped,
    Finalized,
};

struct BCMutableCellHeader {
    BCMutableCellState state = BCMutableCellState::Empty;
    BCDumpRef dump_ref = {};
    uint32_t dump_generation = 0U;
};

struct BCFamilyMutableStoreStats {
    uint64_t created_builders = 0U;
    uint64_t reloaded_builders = 0U;
    uint64_t dumped_builders = 0U;
    uint64_t finalized_cells = 0U;
    uint64_t kept_resident_cells = 0U;
    uint64_t builder_hash_grows = 0U;
    uint64_t builder_bitmap_grows = 0U;
    uint64_t builder_hash_replaced_bytes = 0U;
    uint64_t builder_bitmap_replaced_bytes = 0U;
    uint64_t dump_buffer_bytes_peak = 0U;
    uint64_t reload_dump_bytes_peak = 0U;
    uint64_t restore_builder_bytes_peak = 0U;
    uint64_t finalize_dump_bytes_peak = 0U;
    uint64_t static_metadata_bytes = 0U;
    uint64_t released_builder_bytes_total = 0U;
    uint64_t released_builder_bytes_peak = 0U;
    uint64_t release_batch_builder_bytes_peak = 0U;
};

struct BCCellMutableReserveHint {
    uint32_t buckets = 0U;
    uint32_t bitmap_words = 0U;
};

class BCFamilyMutableStore {
public:
    BCFamilyMutableStore(
        const BCLut &lut,
        const BCFamilyTable &axis,
        IBCGenerationBlobIO &blob
    ) : lut_(&lut),
        matrix_(axis),
        window_cell_set_(matrix_),
        blob_(&blob),
        headers_(matrix_.cell_count()),
        builders_(matrix_.cell_count()),
        builder_ptrs_(std::make_unique<std::atomic<BCCellMutableBuilder *>[]>(matrix_.cell_count())),
        lifecycle_locks_(std::make_unique<std::mutex[]>(matrix_.cell_count())),
        active_marks_(matrix_.cell_count(), 0U),
        keep_marks_(matrix_.cell_count(), 0U),
        resident_marks_(matrix_.cell_count(), 0U) {
        for (uint32_t cid = 0U; cid < matrix_.cell_count(); ++cid) {
            builder_ptrs_[cid].store(nullptr, std::memory_order_relaxed);
        }
        stats_.static_metadata_bytes = static_metadata_bytes();
    }

    [[nodiscard]] const BCCellMatrix &matrix() const {
        return matrix_;
    }

    [[nodiscard]] const BCMutableCellHeader &header(CellId cid) const {
        check_cid(cid);
        return headers_[cid];
    }

    [[nodiscard]] BCMutableCellState state(CellId cid) const {
        return header(cid).state;
    }

    void prepare_target_window_for_families(const FamilyIdList2 &target_families) {
        prepare_target_window_for_family_range(target_families);
    }

    void prepare_target_window_for_families(const FamilyIdList3 &target_families) {
        prepare_target_window_for_family_range(target_families);
    }

    void prepare_target_window_for_families(const std::vector<FamilyId> &target_families) {
        prepare_target_window_for_family_range(target_families);
    }

    void prepare_target_window_for_cached_cells(
        const FamilyIdList2 &target_families,
        const std::vector<CellId> &need_cells
    ) {
        prepare_target_window_for_cached_family_range(target_families, need_cells);
    }

    void prepare_target_window_for_cached_cells(
        const FamilyIdList3 &target_families,
        const std::vector<CellId> &need_cells
    ) {
        prepare_target_window_for_cached_family_range(target_families, need_cells);
    }

    void prepare_target_window_for_cached_cells(
        const std::vector<FamilyId> &target_families,
        const std::vector<CellId> &need_cells
    ) {
        prepare_target_window_for_cached_family_range(target_families, need_cells);
    }

    void prepare_target_window_for_cells_for_testing(const std::vector<CellId> &need_cells) {
        active_target_families_ = {};
        prepare_target_window_cells(need_cells);
    }

    [[nodiscard]] BCCellMutableBuilder &get_or_create(CellId cid) {
        check_cid(cid);
        if (active_marks_[cid] != active_epoch_) {
            throw std::logic_error("BC family mutable cell is outside active target window");
        }
        if (BCCellMutableBuilder *builder = builder_ptrs_[cid].load(std::memory_order_acquire)) {
            return *builder;
        }
        std::lock_guard<std::mutex> lock(lifecycle_locks_[cid]);
        if (BCCellMutableBuilder *builder = builder_ptrs_[cid].load(std::memory_order_acquire)) {
            return *builder;
        }
        BCMutableCellHeader &header = headers_[cid];
        if (header.state == BCMutableCellState::Finalized) {
            throw std::logic_error("BC family mutable store cannot insert into finalized cell");
        }
        if (header.state == BCMutableCellState::Dumped) {
            throw std::logic_error("BC family mutable dumped cell was not reloaded during prepare");
        }
        if (header.state == BCMutableCellState::Empty) {
            builders_[cid] = std::make_unique<BCCellMutableBuilder>(*lut_, cid);
            const BCCellMutableReserveHint reserve_hint = reserve_hint_for_cell(cid);
            if (reserve_hint.buckets != 0U || reserve_hint.bitmap_words != 0U) {
                builders_[cid]->reserve(reserve_hint.buckets, reserve_hint.bitmap_words);
            }
            builders_[cid]->mark_growth_baseline();
            header.state = BCMutableCellState::Resident;
            builder_ptrs_[cid].store(builders_[cid].get(), std::memory_order_release);
            mark_resident_and_count_created(cid);
        }
        if (!builders_[cid]) {
            throw std::logic_error("BC family mutable resident cell has no builder");
        }
        return *builders_[cid];
    }

    [[nodiscard]] BCInsertResult insert_encoded(CellId cid, const BCEncodedKeyRank &encoded) {
        BCCellMutableBuilder &builder = get_or_create(cid);
        return builder.insert_encoded(encoded);
    }

    void set_default_builder_reserve(uint32_t buckets, uint32_t bitmap_words) {
        default_reserve_buckets_ = buckets;
        default_reserve_bitmap_words_ = bitmap_words;
    }

    void set_builder_reserve_hints(std::vector<BCCellMutableReserveHint> hints) {
        if (!hints.empty() && hints.size() != headers_.size()) {
            throw std::invalid_argument("BC family mutable reserve hint count does not match cell count");
        }
        reserve_hints_ = std::move(hints);
    }

    void release_except(const std::vector<CellId> &keep_cells) {
        begin_keep_epoch();
        for (CellId cid : keep_cells) {
            check_cid(cid);
            keep_marks_[cid] = keep_epoch_;
        }
        release_survivors_scratch_.clear();
        release_cells_scratch_.clear();
        release_survivors_scratch_.reserve(resident_cells_.size());
        for (CellId cid : resident_cells_) {
            if (headers_[cid].state != BCMutableCellState::Resident) {
                resident_marks_[cid] = 0U;
                continue;
            }
            if (keep_marks_[cid] == keep_epoch_) {
                ++stats_.kept_resident_cells;
                release_survivors_scratch_.push_back(cid);
                continue;
            }
            release_cells_scratch_.push_back(cid);
        }
        release_resident_cells(release_cells_scratch_);
        resident_cells_.swap(release_survivors_scratch_);
    }

    void release_resident_cells(const std::vector<CellId> &release_cells) {
        last_release_batch_builder_bytes_ = 0U;
        if (release_cells.empty()) {
            return;
        }
        release_generations_scratch_.assign(release_cells.size(), 0U);
        release_dirty_positions_scratch_.clear();
        release_dirty_positions_scratch_.reserve(release_cells.size());
        for (size_t i = 0U; i < release_cells.size(); ++i) {
            const CellId cid = release_cells[i];
            BCMutableCellHeader &header = headers_[cid];
            if (!builders_[cid]) {
                throw std::logic_error("BC family mutable resident cell has no builder");
            }
            if (builders_[cid]->dirty()) {
                if (next_dump_generation_ == std::numeric_limits<uint32_t>::max()) {
                    throw std::overflow_error("BC family mutable dump generation overflow");
                }
                release_generations_scratch_[i] = ++next_dump_generation_;
                release_dirty_positions_scratch_.push_back(i);
            } else if (!header.dump_ref.valid()) {
                release_generations_scratch_[i] = 0U;
            }
        }
        release_dirty_refs_scratch_.assign(release_dirty_positions_scratch_.size(), BCDumpRef{});
        uint64_t release_batch_builder_bytes = 0U;
        for (size_t dirty_i = 0U; dirty_i < release_dirty_positions_scratch_.size(); ++dirty_i) {
            const size_t release_i = release_dirty_positions_scratch_[dirty_i];
            const CellId cid = release_cells[release_i];
            release_dirty_refs_scratch_[dirty_i] = blob_->append_cell_dump_streamed(
                *builders_[cid],
                release_generations_scratch_[release_i]
            );
            stats_.dump_buffer_bytes_peak = std::max<uint64_t>(
                stats_.dump_buffer_bytes_peak,
                release_dirty_refs_scratch_[dirty_i].bytes
            );
        }
        size_t dirty_cursor = 0U;
        for (size_t i = 0U; i < release_cells.size(); ++i) {
            const CellId cid = release_cells[i];
            BCMutableCellHeader &header = headers_[cid];
            const bool is_dirty =
                dirty_cursor < release_dirty_positions_scratch_.size() &&
                release_dirty_positions_scratch_[dirty_cursor] == i;
            if (is_dirty) {
                header.dump_ref = release_dirty_refs_scratch_[dirty_cursor];
                header.dump_generation = release_generations_scratch_[i];
                ++dirty_cursor;
                ++stats_.dumped_builders;
            } else if (!header.dump_ref.valid()) {
                accumulate_builder_growth_stats(*builders_[cid]);
                release_batch_builder_bytes += record_released_builder_bytes(*builders_[cid]);
                builder_ptrs_[cid].store(nullptr, std::memory_order_release);
                builders_[cid].reset();
                header.state = BCMutableCellState::Empty;
                resident_marks_[cid] = 0U;
                continue;
            }
            accumulate_builder_growth_stats(*builders_[cid]);
            release_batch_builder_bytes += record_released_builder_bytes(*builders_[cid]);
            builder_ptrs_[cid].store(nullptr, std::memory_order_release);
            builders_[cid].reset();
            header.state = BCMutableCellState::Dumped;
            resident_marks_[cid] = 0U;
        }
        stats_.release_batch_builder_bytes_peak = std::max<uint64_t>(
            stats_.release_batch_builder_bytes_peak,
            release_batch_builder_bytes
        );
        last_release_batch_builder_bytes_ = release_batch_builder_bytes;
    }

    [[nodiscard]] FinalizedCellPayload finalize_cell(CellId cid, BCCellFinalizeOptions options = {}) {
        BCCellMutableBuilder::FinalizeScratch scratch;
        finalize_cell_into(cid, scratch, options);
        return std::move(scratch.payload);
    }

    const FinalizedCellPayload &finalize_cell_into(
        CellId cid,
        BCCellMutableBuilder::FinalizeScratch &scratch,
        BCCellFinalizeOptions options = {}
    ) {
        check_cid(cid);
        BCMutableCellHeader &header = headers_[cid];
        if (header.state == BCMutableCellState::Finalized) {
            throw std::logic_error("BC family mutable cell finalized twice");
        }
        scratch.clear_payload_keep_capacity();
        if (header.state == BCMutableCellState::Dumped) {
            const BCCellBuilderDumpBuffer &dump_buffer =
                read_single_dump_for_finalize(cid, header.dump_ref);
            BCCellMutableBuilder::finalize_dump_into(
                *lut_,
                cid,
                dump_buffer.view(),
                scratch,
                options
            );
            header.state = BCMutableCellState::Finalized;
            ++stats_.finalized_cells;
            return scratch.payload;
        }
        if (header.state == BCMutableCellState::Resident) {
            if (!builders_[cid]) {
                throw std::logic_error("BC family mutable resident cell has no builder");
            }
            builders_[cid]->finalize_into(scratch, options);
            accumulate_builder_growth_stats(*builders_[cid]);
            (void)record_released_builder_bytes(*builders_[cid]);
            builder_ptrs_[cid].store(nullptr, std::memory_order_release);
            builders_[cid].reset();
            remove_resident_cell(cid);
        }
        header.state = BCMutableCellState::Finalized;
        ++stats_.finalized_cells;
        return scratch.payload;
    }

    template <class WriteCellFn>
    void finalize_cell_streamed_into(
        CellId cid,
        BCCellMutableBuilder::FinalizeScratch &scratch,
        WriteCellFn &&write_cell,
        BCCellFinalizeOptions options = {}
    ) {
        check_cid(cid);
        BCMutableCellHeader &header = headers_[cid];
        if (header.state == BCMutableCellState::Finalized) {
            throw std::logic_error("BC family mutable cell finalized twice");
        }
        scratch.clear_payload_keep_capacity();
        auto write_empty = [&]() {
            auto emit_empty = [](auto &&, auto &&) {};
            write_cell(0U, 0U, 0U, emit_empty);
        };
        if (header.state == BCMutableCellState::Dumped) {
            const BCCellBuilderDumpBuffer &dump_buffer =
                read_single_dump_for_finalize(cid, header.dump_ref);
            BCCellMutableBuilder::finalize_dump_streamed_into(
                *lut_,
                cid,
                dump_buffer.view(),
                scratch,
                std::forward<WriteCellFn>(write_cell),
                options
            );
            header.state = BCMutableCellState::Finalized;
            ++stats_.finalized_cells;
            return;
        }
        if (header.state == BCMutableCellState::Resident) {
            if (!builders_[cid]) {
                throw std::logic_error("BC family mutable resident cell has no builder");
            }
            builders_[cid]->finalize_streamed_into(
                scratch,
                std::forward<WriteCellFn>(write_cell),
                options
            );
            accumulate_builder_growth_stats(*builders_[cid]);
            (void)record_released_builder_bytes(*builders_[cid]);
            builder_ptrs_[cid].store(nullptr, std::memory_order_release);
            builders_[cid].reset();
            remove_resident_cell(cid);
        } else {
            write_empty();
        }
        header.state = BCMutableCellState::Finalized;
        ++stats_.finalized_cells;
    }

    [[nodiscard]] std::vector<FinalizedCellPayload> finalize_cells(
        const std::vector<CellId> &cells,
        BCCellFinalizeOptions options = {}
    ) {
        std::vector<FinalizedCellPayload> payloads(cells.size());
        std::vector<std::pair<CellId, BCDumpRef>> dumped_refs;
        std::vector<size_t> dumped_positions;
        dumped_refs.reserve(cells.size());
        dumped_positions.reserve(cells.size());

        for (size_t i = 0U; i < cells.size(); ++i) {
            const CellId cid = cells[i];
            check_cid(cid);
            BCMutableCellHeader &header = headers_[cid];
            if (header.state == BCMutableCellState::Finalized) {
                throw std::logic_error("BC family mutable cell finalized twice");
            }
            if (header.state == BCMutableCellState::Dumped) {
                dumped_refs.emplace_back(cid, header.dump_ref);
                dumped_positions.push_back(i);
            }
        }

        std::vector<BCCellBuilderDumpBuffer> dumped_buffers;
        if (!dumped_refs.empty()) {
            BCGenerationBlobIOStats read_stats;
            dumped_buffers = blob_->read_many(dumped_refs, &read_stats);
            add_blob_read_stats(read_stats);
            stats_.finalize_dump_bytes_peak = std::max<uint64_t>(
                stats_.finalize_dump_bytes_peak,
                dump_buffer_vector_bytes(dumped_buffers)
            );
            if (dumped_buffers.size() != dumped_refs.size()) {
                throw std::logic_error("BC family mutable batch finalize read size mismatch");
            }
        }

        size_t dumped_cursor = 0U;
        for (size_t i = 0U; i < cells.size(); ++i) {
            const CellId cid = cells[i];
            BCMutableCellHeader &header = headers_[cid];
            if (header.state == BCMutableCellState::Dumped) {
                if (dumped_cursor >= dumped_buffers.size() ||
                    dumped_positions[dumped_cursor] != i ||
                    dumped_buffers[dumped_cursor].cid != cid) {
                    throw std::logic_error("BC family mutable batch finalize dump order mismatch");
                }
                payloads[i] = BCCellMutableBuilder::finalize_dump(
                    *lut_,
                    cid,
                    dumped_buffers[dumped_cursor].view(),
                    options
                );
                ++dumped_cursor;
            } else if (header.state == BCMutableCellState::Resident) {
                if (!builders_[cid]) {
                    throw std::logic_error("BC family mutable resident cell has no builder");
                }
                payloads[i] = builders_[cid]->finalize(options);
                accumulate_builder_growth_stats(*builders_[cid]);
                builder_ptrs_[cid].store(nullptr, std::memory_order_release);
                builders_[cid].reset();
                remove_resident_cell(cid);
            }
            header.state = BCMutableCellState::Finalized;
            ++stats_.finalized_cells;
        }
        return payloads;
    }

    void finalize_cell_list_for_testing(const std::vector<CellId> &cells, BCCellFinalizeOptions options = {}) {
        for (CellId cid : cells) {
            (void)finalize_cell(cid, options);
        }
    }

    [[nodiscard]] uint32_t resident_cell_count() const {
        return static_cast<uint32_t>(resident_cells_.size());
    }

    [[nodiscard]] uint32_t active_cell_count() const {
        return active_cell_count_;
    }

    [[nodiscard]] const std::vector<CellId> &active_cells() const {
        return active_cells_;
    }

    [[nodiscard]] uint32_t cell_count() const {
        return matrix_.cell_count();
    }

    [[nodiscard]] bool is_active_cell(CellId cid) const {
        check_cid(cid);
        return active_marks_[cid] == active_epoch_;
    }

    [[nodiscard]] uint64_t active_builder_bytes() const {
        uint64_t bytes = 0U;
        for (CellId cid : resident_cells_) {
            if (BCCellMutableBuilder *builder = builder_ptrs_[cid].load(std::memory_order_acquire)) {
                bytes += builder->allocated_bytes();
            }
        }
        return bytes;
    }

    void validate_internal_for_testing() const {
        std::vector<uint8_t> resident_seen(headers_.size(), 0U);
        for (CellId cid : resident_cells_) {
            check_cid(cid);
            if (resident_seen[cid] != 0U) {
                throw std::logic_error("BC family mutable store validation saw duplicate resident cell");
            }
            resident_seen[cid] = 1U;
            if (resident_marks_[cid] == 0U) {
                throw std::logic_error("BC family mutable store validation saw unmarked resident cell");
            }
            if (headers_[cid].state != BCMutableCellState::Resident) {
                throw std::logic_error("BC family mutable store validation resident list state mismatch");
            }
            if (!builders_[cid] ||
                builder_ptrs_[cid].load(std::memory_order_acquire) != builders_[cid].get()) {
                throw std::logic_error("BC family mutable store validation resident builder pointer mismatch");
            }
            builders_[cid]->validate_internal_for_testing();
        }

        for (CellId cid = 0U; cid < headers_.size(); ++cid) {
            const BCMutableCellHeader &header = headers_[cid];
            const BCCellMutableBuilder *published =
                builder_ptrs_[cid].load(std::memory_order_acquire);
            switch (header.state) {
                case BCMutableCellState::Empty:
                    if (builders_[cid] || published != nullptr || resident_marks_[cid] != 0U) {
                        throw std::logic_error("BC family mutable store validation empty cell has resident data");
                    }
                    break;
                case BCMutableCellState::Resident:
                    if (resident_seen[cid] == 0U ||
                        !builders_[cid] ||
                        published != builders_[cid].get() ||
                        resident_marks_[cid] == 0U) {
                        throw std::logic_error("BC family mutable store validation resident cell missing data");
                    }
                    break;
                case BCMutableCellState::Dumped:
                    if (builders_[cid] || published != nullptr || resident_marks_[cid] != 0U) {
                        throw std::logic_error("BC family mutable store validation dumped cell has resident data");
                    }
                    if (!header.dump_ref.valid()) {
                        throw std::logic_error("BC family mutable store validation dumped cell has invalid ref");
                    }
                    break;
                case BCMutableCellState::Finalized:
                    if (builders_[cid] || published != nullptr || resident_marks_[cid] != 0U) {
                        throw std::logic_error("BC family mutable store validation finalized cell has resident data");
                    }
                    break;
            }
        }
    }

    [[nodiscard]] uint64_t allocated_bytes() const {
        const uint64_t cell_count = headers_.capacity();
        uint64_t bytes = sizeof(BCFamilyMutableStore);
        bytes = checked_add_u64(
            bytes,
            window_cell_set_.allocated_bytes(),
            "BC family mutable store cell set byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(headers_.capacity()) * sizeof(BCMutableCellHeader),
            "BC family mutable store header byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(builders_.capacity()) * sizeof(std::unique_ptr<BCCellMutableBuilder>),
            "BC family mutable store builder pointer vector byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            cell_count * sizeof(std::atomic<BCCellMutableBuilder *>),
            "BC family mutable store atomic pointer byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            cell_count * sizeof(std::mutex),
            "BC family mutable store lifecycle lock byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(active_marks_.capacity()) * sizeof(uint32_t),
            "BC family mutable store active mark byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(keep_marks_.capacity()) * sizeof(uint32_t),
            "BC family mutable store keep mark byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(resident_marks_.capacity()) * sizeof(uint8_t),
            "BC family mutable store resident mark byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(active_cells_.capacity()) * sizeof(CellId),
            "BC family mutable store active cell byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(resident_cells_.capacity()) * sizeof(CellId),
            "BC family mutable store resident cell byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(release_survivors_scratch_.capacity()) * sizeof(CellId),
            "BC family mutable store release survivor scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(release_cells_scratch_.capacity()) * sizeof(CellId),
            "BC family mutable store release cell scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(release_generations_scratch_.capacity()) * sizeof(uint32_t),
            "BC family mutable store generation scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(release_dirty_positions_scratch_.capacity()) * sizeof(size_t),
            "BC family mutable store dirty position scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(release_dirty_refs_scratch_.capacity()) * sizeof(BCDumpRef),
            "BC family mutable store dirty ref scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(reload_dumped_cells_scratch_.capacity()) * sizeof(CellId),
            "BC family mutable store reload cell scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(reload_refs_scratch_.capacity()) * sizeof(std::pair<CellId, BCDumpRef>),
            "BC family mutable store reload ref scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(reload_batch_refs_scratch_.capacity()) * sizeof(std::pair<CellId, BCDumpRef>),
            "BC family mutable store reload batch ref scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(finalize_single_ref_scratch_.capacity()) * sizeof(std::pair<CellId, BCDumpRef>),
            "BC family mutable store finalize ref scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            dump_buffer_vector_bytes(finalize_single_dump_scratch_),
            "BC family mutable store finalize dump scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(reload_restored_builders_scratch_.capacity()) * sizeof(BCRestoredCellBuilder),
            "BC family mutable store restored builder scratch byte overflow"
        );
        bytes = checked_add_u64(
            bytes,
            static_cast<uint64_t>(reserve_hints_.capacity()) * sizeof(BCCellMutableReserveHint),
            "BC family mutable store reserve hint byte overflow"
        );
        return bytes;
    }

    [[nodiscard]] const BCFamilyMutableStoreStats &stats() const {
        return stats_;
    }

    [[nodiscard]] uint64_t static_metadata_bytes() const {
        const uint64_t cells = headers_.size();
        return cells * (
            sizeof(BCMutableCellHeader) +
            sizeof(std::unique_ptr<BCCellMutableBuilder>) +
            sizeof(std::atomic<BCCellMutableBuilder *>) +
            sizeof(std::mutex) +
            sizeof(uint32_t) +
            sizeof(uint32_t) +
            sizeof(uint8_t) +
            sizeof(BCCellMutableReserveHint)
        );
    }

    void mark_builder_overflow() {
        builder_overflowed_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool builder_overflowed() const {
        return builder_overflowed_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const BCGenerationBlobIOStats &blob_read_stats() const {
        return blob_read_stats_;
    }

    [[nodiscard]] BCGenerationBlobIOStats blob_append_stats() const {
        return blob_->stats();
    }

    [[nodiscard]] const FamilyIdList3 &active_target_families() const {
        return active_target_families_;
    }

    [[nodiscard]] uint64_t last_release_batch_builder_bytes() const {
        return last_release_batch_builder_bytes_;
    }

    [[nodiscard]] const std::vector<CellId> &resident_cells_for_testing() const {
        return resident_cells_;
    }

private:
    void check_cid(CellId cid) const {
        if (cid >= headers_.size()) {
            throw std::out_of_range("BC family mutable store cell id out of range");
        }
    }

    void begin_active_epoch() {
        if (++active_epoch_ == 0U) {
            std::fill(active_marks_.begin(), active_marks_.end(), 0U);
            active_epoch_ = 1U;
        }
    }

    void begin_keep_epoch() {
        if (++keep_epoch_ == 0U) {
            std::fill(keep_marks_.begin(), keep_marks_.end(), 0U);
            keep_epoch_ = 1U;
        }
    }

    template <typename FamilyIdRange>
    void prepare_target_window_for_family_range(const FamilyIdRange &target_families) {
        if (target_families.size() > 3U) {
            throw std::invalid_argument("BC family mutable target window exceeds three target families");
        }
        active_target_families_ = {};
        for (FamilyId family : target_families) {
            active_target_families_.push_back(family);
        }
        window_cell_set_.begin_epoch();
        window_cell_set_.add_family_crosses(target_families);
        prepare_target_window_cells(window_cell_set_.cells());
    }

    template <typename FamilyIdRange>
    void prepare_target_window_for_cached_family_range(
        const FamilyIdRange &target_families,
        const std::vector<CellId> &need_cells
    ) {
        if (target_families.size() > 3U) {
            throw std::invalid_argument("BC family mutable target window exceeds three target families");
        }
        active_target_families_ = {};
        for (FamilyId family : target_families) {
            active_target_families_.push_back(family);
        }
        prepare_target_window_cells(need_cells);
    }

    void prepare_target_window_cells(const std::vector<CellId> &need_cells) {
        begin_active_epoch();
        active_cell_count_ = static_cast<uint32_t>(need_cells.size());
        active_cells_.assign(need_cells.begin(), need_cells.end());
        reload_dumped_cells_scratch_.clear();
        reload_dumped_cells_scratch_.reserve(need_cells.size());
        for (CellId cid : need_cells) {
            check_cid(cid);
            active_marks_[cid] = active_epoch_;
            BCMutableCellHeader &header = headers_[cid];
            if (header.state == BCMutableCellState::Finalized) {
                throw std::logic_error("BC family mutable store cannot prepare finalized cell");
            }
            if (header.state == BCMutableCellState::Dumped) {
                reload_dumped_cells_scratch_.push_back(cid);
            }
        }
        reload_dumped_cells(reload_dumped_cells_scratch_);
    }

    void reload_dumped_cells(const std::vector<CellId> &cids) {
        if (cids.empty()) {
            return;
        }
        reload_refs_scratch_.clear();
        reload_refs_scratch_.reserve(cids.size());
        for (CellId cid : cids) {
            BCMutableCellHeader &header = headers_[cid];
            if (header.state != BCMutableCellState::Dumped) {
                continue;
            }
            if (!header.dump_ref.valid()) {
                throw std::logic_error("BC family mutable dumped cell has no dump ref");
            }
            reload_refs_scratch_.push_back({cid, header.dump_ref});
        }
        constexpr size_t kReloadBatchCells = 64U;
        reload_batch_refs_scratch_.reserve(std::min(kReloadBatchCells, reload_refs_scratch_.size()));
        for (size_t batch_begin = 0U;
             batch_begin < reload_refs_scratch_.size();
             batch_begin += kReloadBatchCells) {
            const size_t batch_end = std::min(batch_begin + kReloadBatchCells, reload_refs_scratch_.size());
            reload_batch_refs_scratch_.assign(
                reload_refs_scratch_.begin() + static_cast<std::ptrdiff_t>(batch_begin),
                reload_refs_scratch_.begin() + static_cast<std::ptrdiff_t>(batch_end)
            );
            uint64_t requested_dump_bytes = 0U;
            for (const auto &ref : reload_batch_refs_scratch_) {
                requested_dump_bytes = checked_add_u64(
                    requested_dump_bytes,
                    ref.second.bytes,
                    "BC family mutable reload dump byte count overflow"
                );
            }
            stats_.reload_dump_bytes_peak = std::max<uint64_t>(
                stats_.reload_dump_bytes_peak,
                requested_dump_bytes
            );
            BCGenerationBlobIOStats read_stats;
            reload_restored_builders_scratch_ =
                blob_->restore_many_builders(*lut_, reload_batch_refs_scratch_, &read_stats);
            add_blob_read_stats(read_stats);
            stats_.restore_builder_bytes_peak = std::max<uint64_t>(
                stats_.restore_builder_bytes_peak,
                requested_dump_bytes
            );
            if (reload_restored_builders_scratch_.size() != reload_batch_refs_scratch_.size()) {
                throw std::logic_error("BC family mutable blob read_many size mismatch");
            }
            for (BCRestoredCellBuilder &restored : reload_restored_builders_scratch_) {
                const CellId cid = restored.cid;
                std::lock_guard<std::mutex> lock(lifecycle_locks_[cid]);
                BCMutableCellHeader &header = headers_[cid];
                if (header.state != BCMutableCellState::Dumped) {
                    continue;
                }
                if (!restored.builder) {
                    throw std::logic_error("BC family mutable blob restored null builder");
                }
                builders_[cid] = std::move(restored.builder);
                builders_[cid]->mark_growth_baseline();
                header.state = BCMutableCellState::Resident;
                builder_ptrs_[cid].store(builders_[cid].get(), std::memory_order_release);
                mark_resident_and_count_reloaded(cid);
            }
        }
    }

    void release_resident_cell(CellId cid) {
        BCMutableCellHeader &header = headers_[cid];
        if (!builders_[cid]) {
            throw std::logic_error("BC family mutable resident cell has no builder");
        }
        if (builders_[cid]->dirty()) {
            if (next_dump_generation_ == std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC family mutable dump generation overflow");
            }
            const uint32_t generation = ++next_dump_generation_;
            header.dump_ref = blob_->append_cell_dump_streamed(*builders_[cid], generation);
            header.dump_generation = generation;
            ++stats_.dumped_builders;
        } else if (!header.dump_ref.valid()) {
            accumulate_builder_growth_stats(*builders_[cid]);
            builder_ptrs_[cid].store(nullptr, std::memory_order_release);
                builders_[cid].reset();
                header.state = BCMutableCellState::Empty;
                resident_marks_[cid] = 0U;
                return;
        }
        accumulate_builder_growth_stats(*builders_[cid]);
        builder_ptrs_[cid].store(nullptr, std::memory_order_release);
        builders_[cid].reset();
        header.state = BCMutableCellState::Dumped;
        resident_marks_[cid] = 0U;
    }

    void add_blob_read_stats(const BCGenerationBlobIOStats &stats) {
        blob_read_stats_.append_count += stats.append_count;
        blob_read_stats_.read_count += stats.read_count;
        blob_read_stats_.bytes_read += stats.bytes_read;
        blob_read_stats_.requested_extents += stats.requested_extents;
        blob_read_stats_.coalesced_extents += stats.coalesced_extents;
        blob_read_stats_.requested_bytes += stats.requested_bytes;
        blob_read_stats_.read_bytes += stats.read_bytes;
        blob_read_stats_.backend_read_ops += stats.backend_read_ops;
        blob_read_stats_.backend_read_bytes += stats.backend_read_bytes;
        blob_read_stats_.backend_read_seconds += stats.backend_read_seconds;
        blob_read_stats_.restore_seconds += stats.restore_seconds;
    }

    void add_blob_read_stats_delta(
        const BCGenerationBlobIOStats &before,
        const BCGenerationBlobIOStats &after
    ) {
        BCGenerationBlobIOStats delta;
        delta.append_count = after.append_count >= before.append_count ? after.append_count - before.append_count : 0U;
        delta.read_count = after.read_count >= before.read_count ? after.read_count - before.read_count : 0U;
        delta.bytes_written = after.bytes_written >= before.bytes_written ? after.bytes_written - before.bytes_written : 0U;
        delta.bytes_read = after.bytes_read >= before.bytes_read ? after.bytes_read - before.bytes_read : 0U;
        delta.requested_extents = after.requested_extents >= before.requested_extents ? after.requested_extents - before.requested_extents : 0U;
        delta.coalesced_extents = after.coalesced_extents >= before.coalesced_extents ? after.coalesced_extents - before.coalesced_extents : 0U;
        delta.requested_bytes = after.requested_bytes >= before.requested_bytes ? after.requested_bytes - before.requested_bytes : 0U;
        delta.read_bytes = after.read_bytes >= before.read_bytes ? after.read_bytes - before.read_bytes : 0U;
        delta.backend_read_ops = after.backend_read_ops >= before.backend_read_ops ? after.backend_read_ops - before.backend_read_ops : 0U;
        delta.backend_read_bytes = after.backend_read_bytes >= before.backend_read_bytes ? after.backend_read_bytes - before.backend_read_bytes : 0U;
        delta.backend_write_ops = after.backend_write_ops >= before.backend_write_ops ? after.backend_write_ops - before.backend_write_ops : 0U;
        delta.backend_write_bytes = after.backend_write_bytes >= before.backend_write_bytes ? after.backend_write_bytes - before.backend_write_bytes : 0U;
        delta.backend_read_seconds = after.backend_read_seconds - before.backend_read_seconds;
        delta.restore_seconds = after.restore_seconds - before.restore_seconds;
        delta.backend_write_seconds = after.backend_write_seconds - before.backend_write_seconds;
        delta.flush_seconds = after.flush_seconds - before.flush_seconds;
        add_blob_read_stats(delta);
    }

    [[nodiscard]] const BCCellBuilderDumpBuffer &read_single_dump_for_finalize(
        CellId cid,
        BCDumpRef ref
    ) {
        finalize_single_ref_scratch_.clear();
        finalize_single_ref_scratch_.emplace_back(cid, ref);
        BCGenerationBlobIOStats read_stats;
        finalize_single_dump_scratch_ = blob_->read_many(finalize_single_ref_scratch_, &read_stats);
        add_blob_read_stats(read_stats);
        if (finalize_single_dump_scratch_.size() != 1U ||
            finalize_single_dump_scratch_.front().cid != cid) {
            throw std::logic_error("BC family mutable single finalize dump read mismatch");
        }
        stats_.finalize_dump_bytes_peak = std::max<uint64_t>(
            stats_.finalize_dump_bytes_peak,
            dump_buffer_vector_bytes(finalize_single_dump_scratch_)
        );
        return finalize_single_dump_scratch_.front();
    }

    void mark_resident_and_count_created(CellId cid) {
        std::lock_guard<std::mutex> lock(resident_cells_mutex_);
        mark_resident_unlocked(cid);
        ++stats_.created_builders;
    }

    void mark_resident_and_count_reloaded(CellId cid) {
        std::lock_guard<std::mutex> lock(resident_cells_mutex_);
        mark_resident_unlocked(cid);
        ++stats_.reloaded_builders;
    }

    void mark_resident_unlocked(CellId cid) {
        if (resident_marks_[cid] != 0U) {
            return;
        }
        resident_marks_[cid] = 1U;
        resident_cells_.push_back(cid);
    }

    void accumulate_builder_growth_stats(const BCCellMutableBuilder &builder) {
        stats_.builder_hash_grows += builder.runtime_hash_grow_count();
        stats_.builder_bitmap_grows += builder.runtime_bitmap_grow_count();
        stats_.builder_hash_replaced_bytes = checked_add_u64(
            stats_.builder_hash_replaced_bytes,
            builder.runtime_hash_replaced_bytes(),
            "BC family mutable hash replaced byte total overflow"
        );
        stats_.builder_bitmap_replaced_bytes = checked_add_u64(
            stats_.builder_bitmap_replaced_bytes,
            builder.runtime_bitmap_replaced_bytes(),
            "BC family mutable bitmap replaced byte total overflow"
        );
    }

    uint64_t record_released_builder_bytes(const BCCellMutableBuilder &builder) {
        const uint64_t bytes = builder.allocated_bytes();
        stats_.released_builder_bytes_total = checked_add_u64(
            stats_.released_builder_bytes_total,
            bytes,
            "BC family mutable released builder byte total overflow"
        );
        stats_.released_builder_bytes_peak = std::max<uint64_t>(
            stats_.released_builder_bytes_peak,
            bytes
        );
        return bytes;
    }

    [[nodiscard]] BCCellMutableReserveHint reserve_hint_for_cell(CellId cid) const {
        BCCellMutableReserveHint hint{
            default_reserve_buckets_,
            default_reserve_bitmap_words_
        };
        if (!reserve_hints_.empty()) {
            const BCCellMutableReserveHint &cell_hint = reserve_hints_[cid];
            hint.buckets = std::max(hint.buckets, cell_hint.buckets);
            hint.bitmap_words = std::max(hint.bitmap_words, cell_hint.bitmap_words);
        }
        return hint;
    }

    void unmark_resident(CellId cid) {
        std::lock_guard<std::mutex> lock(resident_cells_mutex_);
        resident_marks_[cid] = 0U;
    }

    void remove_resident_cell(CellId cid) {
        std::lock_guard<std::mutex> lock(resident_cells_mutex_);
        if (resident_marks_[cid] == 0U) {
            return;
        }
        resident_marks_[cid] = 0U;
        auto it = std::find(resident_cells_.begin(), resident_cells_.end(), cid);
        if (it != resident_cells_.end()) {
            *it = resident_cells_.back();
            resident_cells_.pop_back();
        }
    }

    [[nodiscard]] static uint64_t checked_add_u64(uint64_t lhs, uint64_t rhs, const char *label) {
        if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
            throw std::overflow_error(label);
        }
        return lhs + rhs;
    }

    [[nodiscard]] static uint64_t dump_buffer_vector_bytes(
        const std::vector<BCCellBuilderDumpBuffer> &buffers
    ) {
        uint64_t total = 0U;
        for (const BCCellBuilderDumpBuffer &buffer : buffers) {
            total = checked_add_u64(
                total,
                static_cast<uint64_t>(buffer.bytes.size()),
                "BC family mutable dump buffer byte count overflow"
            );
        }
        return total;
    }

    const BCLut *lut_ = nullptr;
    BCCellMatrix matrix_;
    BCCellSetBuilder window_cell_set_;
    IBCGenerationBlobIO *blob_ = nullptr;
    std::vector<BCMutableCellHeader> headers_;
    std::vector<std::unique_ptr<BCCellMutableBuilder>> builders_;
    std::unique_ptr<std::atomic<BCCellMutableBuilder *>[]> builder_ptrs_;
    std::unique_ptr<std::mutex[]> lifecycle_locks_;
    std::vector<uint32_t> active_marks_;
    std::vector<uint32_t> keep_marks_;
    std::vector<uint8_t> resident_marks_;
    std::vector<CellId> active_cells_;
    std::vector<CellId> resident_cells_;
    std::vector<CellId> release_survivors_scratch_;
    std::vector<CellId> release_cells_scratch_;
    std::vector<uint32_t> release_generations_scratch_;
    std::vector<size_t> release_dirty_positions_scratch_;
    std::vector<BCDumpRef> release_dirty_refs_scratch_;
    std::vector<CellId> reload_dumped_cells_scratch_;
    std::vector<std::pair<CellId, BCDumpRef>> reload_refs_scratch_;
    std::vector<std::pair<CellId, BCDumpRef>> reload_batch_refs_scratch_;
    std::vector<std::pair<CellId, BCDumpRef>> finalize_single_ref_scratch_;
    std::vector<BCCellBuilderDumpBuffer> finalize_single_dump_scratch_;
    std::vector<BCRestoredCellBuilder> reload_restored_builders_scratch_;
    std::mutex resident_cells_mutex_;
    FamilyIdList3 active_target_families_;
    uint32_t active_epoch_ = 1U;
    uint32_t keep_epoch_ = 1U;
    uint32_t active_cell_count_ = 0U;
    uint32_t default_reserve_buckets_ = 0U;
    uint32_t default_reserve_bitmap_words_ = 0U;
    std::vector<BCCellMutableReserveHint> reserve_hints_;
    uint32_t next_dump_generation_ = 0U;
    uint64_t last_release_batch_builder_bytes_ = 0U;
    std::atomic<bool> builder_overflowed_{false};
    BCFamilyMutableStoreStats stats_;
    BCGenerationBlobIOStats blob_read_stats_;
};

} // namespace BC
