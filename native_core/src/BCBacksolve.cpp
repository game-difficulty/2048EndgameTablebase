#include "BCBacksolve.h"

#include "BoardMover.h"
#include "CanonicalBatch.h"
#include "Formation.h"
#include "BCPositionScanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {
namespace {

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] int effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

[[nodiscard]] uint64_t mix_u64(uint64_t value) {
    value ^= value >> 33U;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33U;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33U;
    return value;
}

[[nodiscard]] uint32_t next_power_of_two_u32(uint64_t value) {
    uint64_t cap = 1U;
    while (cap < value) {
        cap <<= 1U;
        if (cap > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("BC backsolve direct index capacity exceeds uint32");
        }
    }
    return static_cast<uint32_t>(std::max<uint64_t>(cap, 1U));
}

[[nodiscard]] bool success_by_shifts(uint64_t board, const BCBacksolveOptions &options) {
    if (options.success_target_rank <= 0 ||
        options.success_shifts == nullptr ||
        options.success_shifts->empty()) {
        return false;
    }
    const uint64_t target = static_cast<uint64_t>(options.success_target_rank);
    for (uint8_t shift : *options.success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool pattern_accepts(uint64_t board, const BCBacksolveOptions &options) {
    return options.pattern_masks == nullptr ||
        options.pattern_masks->empty() ||
        is_pattern(board, *options.pattern_masks);
}

[[nodiscard]] uint32_t max_success_u32() {
    return max_scale_value<uint32_t>();
}

void validate_options(const BCBacksolveOptions &options) {
    if (options.canonical_batch_size == 0U) {
        throw std::invalid_argument("BC backsolve canonical_batch_size must be non-zero");
    }
    if (options.spawn_rate4 < 0.0 || options.spawn_rate4 > 1.0 || !std::isfinite(options.spawn_rate4)) {
        throw std::invalid_argument("BC backsolve spawn_rate4 must be finite and in [0, 1]");
    }
    if (options.dtype != BCSuccessDTypeMode::UInt32) {
        throw std::invalid_argument("BC backsolve v1 supports only UInt32 success dtype");
    }
}

void validate_layer_relationships(
    const BCPositionLayerReader &current,
    const BCPositionLayerReader &future2,
    const BCPositionLayerReader &future4
) {
    const uint64_t current_sum = current.header().layer_sum;
    if (future2.header().layer_sum != current_sum + 2U) {
        throw std::invalid_argument("BC backsolve future2 layer_sum must equal current + 2");
    }
    if (future4.header().layer_sum != current_sum + 4U) {
        throw std::invalid_argument("BC backsolve future4 layer_sum must equal current + 4");
    }
}

void accumulate_stats(BCBacksolveStats &dst, const BCBacksolveBatchStats &src) {
    dst.current_rows += src.rows;
    dst.queries2 += src.queries2;
    dst.queries4 += src.queries4;
    dst.found2 += src.found2;
    dst.found4 += src.found4;
    dst.terminal_success_rows += src.terminal_success_rows;
}

void accumulate_batch_stats(BCBacksolveBatchStats &dst, const BCBacksolveBatchStats &src) {
    dst.rows += src.rows;
    dst.queries2 += src.queries2;
    dst.queries4 += src.queries4;
    dst.found2 += src.found2;
    dst.found4 += src.found4;
    dst.terminal_success_rows += src.terminal_success_rows;
}

} // namespace

void BCFutureValueLayerView::open(
    const BCLut &lut,
    const BCPositionLayerReader &position,
    const BCSuccessLayerReader &success
) {
    if (success.row_width() != 1U) {
        throw std::invalid_argument("BC future success reader row_width must be 1");
    }
    if (success.dtype_mode() != BCSuccessDTypeMode::UInt32) {
        throw std::invalid_argument("BC future success reader dtype must be UInt32");
    }
    lut_ = &lut;
    position_ = &position;
    cells_.clear();
    cells_.resize(position.cell_count());

    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = position.descriptor(cid);
        CellIndex &cell = cells_[static_cast<size_t>(cid)];
        cell.values = success.read_cell(cid);
        if (cell.values.size() != desc.success_rows) {
            throw std::runtime_error("BC future success cell value count mismatch");
        }
        if (desc.empty()) {
            if (!cell.values.empty()) {
                throw std::runtime_error("BC future empty cell has success values");
            }
            continue;
        }

        cell.rank_payload = position.rank_payload_for_cell(cid);
        const BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
        const uint32_t capacity = next_power_of_two_u32(
            std::max<uint64_t>(2ULL, static_cast<uint64_t>(buckets.size) * 2ULL)
        );
        cell.entries.assign(capacity, DirectEntry{});
        cell.mask = capacity - 1U;
        for (uint32_t i = 0U; i < buckets.size; ++i) {
            const BCBucketEntry &bucket = buckets.data[i];
            uint32_t slot = static_cast<uint32_t>(mix_u64(bucket.key)) & cell.mask;
            while (cell.entries[slot].occupied) {
                if (cell.entries[slot].key == bucket.key) {
                    throw std::runtime_error("BC future direct index saw duplicate bucket key");
                }
                slot = (slot + 1U) & cell.mask;
            }
            DirectEntry entry;
            entry.key = bucket.key;
            entry.rank_payload_offset = bucket.rank_payload_offset;
            entry.success_row_offset = bucket.success_row_offset;
            entry.bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
            entry.occupied = true;
            cell.entries[slot] = entry;
        }
    }
}

const BCFutureValueLayerView::DirectEntry *BCFutureValueLayerView::find_entry(
    const CellIndex &cell,
    uint64_t key
) const {
    if (cell.entries.empty()) {
        return nullptr;
    }
    uint32_t slot = static_cast<uint32_t>(mix_u64(key)) & cell.mask;
    while (true) {
        const DirectEntry &entry = cell.entries[slot];
        if (!entry.occupied) {
            return nullptr;
        }
        if (entry.key == key) {
            return &entry;
        }
        slot = (slot + 1U) & cell.mask;
    }
}

bool BCFutureValueLayerView::lookup_encoded(
    const BCBoardEncodedPosition &encoded,
    uint32_t &value_out
) const {
    value_out = 0U;
    if (!encoded.valid || position_ == nullptr || lut_ == nullptr) {
        return false;
    }
    if (encoded.cid >= cells_.size()) {
        return false;
    }
    const CellIndex &cell = cells_[static_cast<size_t>(encoded.cid)];
    const DirectEntry *entry = find_entry(cell, encoded.key);
    if (entry == nullptr || encoded.rank >= entry->bitmap_len) {
        return false;
    }

    const uint32_t prefix_count = prefix_count_for_bits(entry->bitmap_len);
    const uint32_t bitmap_word_count = words_for_bits(entry->bitmap_len);
    const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
        entry->rank_payload_offset,
        entry->bitmap_len
    );
    const uint64_t prefix_end =
        static_cast<uint64_t>(entry->rank_payload_offset) +
        static_cast<uint64_t>(prefix_count) * sizeof(RankPrefix);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > cell.rank_payload.size || bitmap_end > cell.rank_payload.size) {
        throw std::out_of_range("BC future direct lookup payload range is truncated");
    }

    const BCBitmapRankResult rank_result = bitmap_test_and_rank_le_bytes(
        cell.rank_payload.data + entry->rank_payload_offset,
        prefix_count,
        cell.rank_payload.data + bitmap_offset,
        bitmap_word_count,
        encoded.rank
    );
    if (!rank_result.found) {
        return false;
    }
    const uint64_t local_row =
        static_cast<uint64_t>(entry->success_row_offset) +
        static_cast<uint64_t>(rank_result.rank_before);
    if (local_row >= cell.values.size()) {
        throw std::out_of_range("BC future direct lookup local row exceeds success values");
    }
    value_out = cell.values[static_cast<size_t>(local_row)];
    return true;
}

bool BCFutureValueLayerView::lookup(uint64_t canonical_board, uint32_t &value_out) const {
    value_out = 0U;
    if (position_ == nullptr || lut_ == nullptr) {
        return false;
    }
    const BCBoardEncodedPosition encoded =
        encode_canonical_board_position(*lut_, position_->axis(), canonical_board);
    return lookup_encoded(encoded, value_out);
}

BCBacksolveBatchStats backsolve_board_batch_uint32(
    const uint64_t *boards,
    const uint32_t *local_rows,
    uint32_t board_count,
    const BCFutureValueLayerView &future2,
    const BCFutureValueLayerView &future4,
    const BCBacksolveOptions &options,
    uint32_t *output_values
) {
    if (board_count == 0U) {
        return {};
    }
    if (boards == nullptr || local_rows == nullptr || output_values == nullptr) {
        throw std::invalid_argument("BC backsolve batch pointers must not be null");
    }

    BCBacksolveBatchStats stats;
    stats.rows = board_count;
    std::vector<uint64_t> candidates2;
    std::vector<uint64_t> candidates4;
    std::vector<uint32_t> refs2;
    std::vector<uint32_t> refs4;
    candidates2.reserve(static_cast<size_t>(board_count) * 16U);
    candidates4.reserve(static_cast<size_t>(board_count) * 16U);
    refs2.reserve(static_cast<size_t>(board_count) * 16U);
    refs4.reserve(static_cast<size_t>(board_count) * 16U);
    std::vector<uint32_t> best2(static_cast<size_t>(board_count) * 16U, 0U);
    std::vector<uint32_t> best4(static_cast<size_t>(board_count) * 16U, 0U);
    std::vector<uint16_t> empty_masks(board_count, 0U);
    std::vector<uint8_t> terminal(board_count, 0U);

    auto push_candidate = [](
        uint64_t moved,
        uint32_t ref,
        std::vector<uint64_t> &candidates,
        std::vector<uint32_t> &refs
    ) {
        candidates.push_back(moved);
        refs.push_back(ref);
    };

    for (uint32_t board_slot = 0U; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        if (success_by_shifts(board, options)) {
            output_values[local_rows[board_slot]] = max_success_u32();
            terminal[board_slot] = 1U;
            ++stats.terminal_success_rows;
            continue;
        }

        uint32_t empty_mask = bc_zero_cell_mask16(board);
        empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
#if defined(__GNUC__) || defined(__clang__)
            const uint32_t cell = static_cast<uint32_t>(__builtin_ctz(empty_mask));
#else
            uint32_t cell = 0U;
            while (((empty_mask >> cell) & 1U) == 0U) {
                ++cell;
            }
#endif
            empty_mask &= empty_mask - 1U;
            const uint32_t ref = board_slot * 16U + cell;

            const uint64_t spawn2 = set_board_tile_unchecked(board, cell, 1U);
            const auto moves2 = BoardMover::move_all_dir(spawn2);
            const uint64_t moved2[4] = {
                std::get<0>(moves2),
                std::get<1>(moves2),
                std::get<2>(moves2),
                std::get<3>(moves2)
            };
            for (uint64_t moved : moved2) {
                if (moved != spawn2 && pattern_accepts(moved, options)) {
                    push_candidate(moved, ref, candidates2, refs2);
                    ++stats.queries2;
                }
            }

            const uint64_t spawn4 = set_board_tile_unchecked(board, cell, 2U);
            const auto moves4 = BoardMover::move_all_dir(spawn4);
            const uint64_t moved4[4] = {
                std::get<0>(moves4),
                std::get<1>(moves4),
                std::get<2>(moves4),
                std::get<3>(moves4)
            };
            for (uint64_t moved : moved4) {
                if (moved != spawn4 && pattern_accepts(moved, options)) {
                    push_candidate(moved, ref, candidates4, refs4);
                    ++stats.queries4;
                }
            }
        }
    }

    CanonicalBatch::canonicalize_inplace(
        candidates2.data(),
        candidates2.size(),
        options.canonical_symm_mode
    );
    CanonicalBatch::canonicalize_inplace(
        candidates4.data(),
        candidates4.size(),
        options.canonical_symm_mode
    );

    for (size_t i = 0U; i < candidates2.size(); ++i) {
        uint32_t value = 0U;
        if (future2.lookup(candidates2[i], value)) {
            ++stats.found2;
            uint32_t &best = best2[refs2[i]];
            if (value > best) {
                best = value;
            }
        }
    }
    for (size_t i = 0U; i < candidates4.size(); ++i) {
        uint32_t value = 0U;
        if (future4.lookup(candidates4[i], value)) {
            ++stats.found4;
            uint32_t &best = best4[refs4[i]];
            if (value > best) {
                best = value;
            }
        }
    }

    for (uint32_t board_slot = 0U; board_slot < board_count; ++board_slot) {
        if (terminal[board_slot] != 0U) {
            continue;
        }
        double success_probability = 0.0;
        uint32_t empty_count = 0U;
        uint32_t empty_mask = empty_masks[board_slot];
        while (empty_mask != 0U) {
#if defined(__GNUC__) || defined(__clang__)
            const uint32_t cell = static_cast<uint32_t>(__builtin_ctz(empty_mask));
#else
            uint32_t cell = 0U;
            while (((empty_mask >> cell) & 1U) == 0U) {
                ++cell;
            }
#endif
            empty_mask &= empty_mask - 1U;
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + cell;
            success_probability += static_cast<double>(best2[best_index]) * (1.0 - options.spawn_rate4);
            success_probability += static_cast<double>(best4[best_index]) * options.spawn_rate4;
            ++empty_count;
        }
        output_values[local_rows[board_slot]] = empty_count == 0U
            ? 0U
            : static_cast<uint32_t>(success_probability / static_cast<double>(empty_count));
    }

    return stats;
}

BCBacksolveResult backsolve_resident_layer(
    const BCLut &lut,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCBacksolveOptions &options
) {
    validate_options(options);
    validate_layer_relationships(current, future2_position, future4_position);

    BCBacksolveResult result;
    const double index_t0 = now_seconds();
    BCFutureValueLayerView future2(lut, future2_position, future2_success);
    BCFutureValueLayerView future4(lut, future4_position, future4_success);
    const double index_t1 = now_seconds();
    result.stats.future_index_seconds = index_t1 - index_t0;

    std::vector<std::vector<uint32_t>> cell_values(current.cell_count());
    for (CellId cid = 0U; cid < current.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = current.descriptor(cid);
        cell_values[static_cast<size_t>(cid)].assign(desc.success_rows, 0U);
    }

    const int threads = effective_threads(options.num_threads);
    std::vector<BCBacksolveBatchStats> per_thread(static_cast<size_t>(threads));
    const double recalc_t0 = now_seconds();
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCBacksolveBatchStats &thread_stats = per_thread[static_cast<size_t>(tid)];
        std::vector<uint64_t> board_buffer;
        std::vector<uint32_t> row_buffer;
        board_buffer.reserve(options.canonical_batch_size);
        row_buffer.reserve(options.canonical_batch_size);

        auto flush = [&](std::vector<uint32_t> &values) {
            if (board_buffer.empty()) {
                return;
            }
            const BCBacksolveBatchStats batch = backsolve_board_batch_uint32(
                board_buffer.data(),
                row_buffer.data(),
                static_cast<uint32_t>(board_buffer.size()),
                future2,
                future4,
                options,
                values.data()
            );
            accumulate_batch_stats(thread_stats, batch);
            board_buffer.clear();
            row_buffer.clear();
        };

#pragma omp for schedule(dynamic, 1)
        for (int64_t cid_signed = 0; cid_signed < static_cast<int64_t>(current.cell_count()); ++cid_signed) {
            const CellId cid = static_cast<CellId>(cid_signed);
            const BCPositionCellDescriptor &desc = current.descriptor(cid);
            if (desc.success_rows == 0U || desc.empty()) {
                continue;
            }
            std::vector<uint32_t> &values = cell_values[static_cast<size_t>(cid)];
            board_buffer.clear();
            row_buffer.clear();
            BCPositionCellScanner(current, cid).for_each_board(
                [&](const BCScannedBoardEntry &entry) {
                    board_buffer.push_back(entry.board);
                    row_buffer.push_back(entry.local_success_row);
                    if (board_buffer.size() == options.canonical_batch_size) {
                        flush(values);
                    }
                }
            );
            flush(values);
        }
    }
    const double recalc_t1 = now_seconds();
    result.stats.recalc_seconds = recalc_t1 - recalc_t0;
    for (const BCBacksolveBatchStats &stats : per_thread) {
        accumulate_stats(result.stats, stats);
    }

    const double write_t0 = now_seconds();
    BCSuccessLayerWriter writer;
    writer.begin_layer(current, 1U, options.dtype);
    for (CellId cid = 0U; cid < current.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = current.descriptor(cid);
        if (desc.success_rows == 0U || desc.empty()) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, cell_values[static_cast<size_t>(cid)]);
    }
    result.success_bytes = writer.finish_layer();
    const double write_t1 = now_seconds();
    result.stats.write_seconds = write_t1 - write_t0;
    return result;
}

BCBacksolveStats backsolve_resident_layer_to_file(
    const std::filesystem::path &path,
    const BCLut &lut,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCBacksolveOptions &options
) {
    BCBacksolveResult result = backsolve_resident_layer(
        lut,
        current,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );
    const double write_t0 = now_seconds();
    write_success_layer_to_file(path, result.success_bytes);
    const double write_t1 = now_seconds();
    result.stats.write_seconds += write_t1 - write_t0;
    return result.stats;
}

} // namespace BC
