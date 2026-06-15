#pragma once

#include "BCDirectFileIO.h"
#include "BCLoadedCellScanner.h"
#include "BCResidentSolve.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace BC {

inline void bc_single_chunk_add_cell_load_stats(BCCellLoadStats &dst, const BCCellLoadStats &src) {
    dst.requested_extents += src.requested_extents;
    dst.coalesced_extents += src.coalesced_extents;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.read_bytes;
    dst.backend_read_ops += src.backend_read_ops;
    dst.backend_read_bytes += src.backend_read_bytes;
    dst.backend_read_seconds += src.backend_read_seconds;
}

inline void bc_single_chunk_add_file_read_stats(BCCellLoadStats &dst, const BCFileIOStats &src) {
    dst.requested_extents += src.request_count;
    dst.coalesced_extents += src.backend_io_count;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.requested_bytes;
    dst.backend_read_ops += src.backend_io_count;
    dst.backend_read_bytes += src.backend_bytes;
    dst.backend_read_seconds += src.backend_seconds;
}

inline void bc_single_chunk_add_success_load_stats(BCSuccessLoadStats &dst, const BCSuccessLoadStats &src) {
    dst.requested_extents += src.requested_extents;
    dst.coalesced_extents += src.coalesced_extents;
    dst.requested_bytes += src.requested_bytes;
    dst.read_bytes += src.read_bytes;
    dst.backend_read_ops += src.backend_read_ops;
    dst.backend_read_bytes += src.backend_read_bytes;
}

struct BCSingleChunkSolveStats {
    uint64_t current_chunks = 0U;
    uint64_t current_cells = 0U;
    uint64_t current_nonempty_cells = 0U;
    uint64_t current_empty_cells = 0U;
    uint64_t current_work_items = 0U;
    uint64_t current_rows = 0U;
    uint64_t current_boards = 0U;
    uint64_t output_values = 0U;
    uint64_t output_bytes = 0U;
    uint64_t partial_write_bytes = 0U;
    uint64_t partial_read_bytes = 0U;

    uint64_t future2_batch_loads = 0U;
    uint64_t future4_batch_loads = 0U;
    uint64_t future2_cells_loaded = 0U;
    uint64_t future4_cells_loaded = 0U;
    uint64_t future2_active_cells_max = 0U;
    uint64_t future4_active_cells_max = 0U;
    uint64_t future2_position_resident_bytes = 0U;
    uint64_t future4_position_resident_bytes = 0U;
    uint64_t future2_success_resident_bytes = 0U;
    uint64_t future4_success_resident_bytes = 0U;
    uint64_t future_resident_layers_max = 0U;
    uint64_t future_resident_bytes_max = 0U;

    uint64_t compact_input_rows = 0U;
    uint64_t compact_live_rows = 0U;
    uint64_t compact_zero_pruned_rows = 0U;
    uint64_t compact_live_cells = 0U;
    uint64_t compact_empty_cells = 0U;
    uint64_t current_board_cache_bytes = 0U;

    BCCellLoadStats current_position_load;
    BCCellLoadStats future2_position_load;
    BCCellLoadStats future4_position_load;
    BCSuccessLoadStats future2_success_load;
    BCSuccessLoadStats future4_success_load;
    BCFileIOStats output_position_write;
    BCFileIOStats output_success_write;
    BCFileIOStats partial_write_io;
    BCFileIOStats partial_read_io;
    BCSolveEdgeStats edge;

    double prepass_seconds = 0.0;
    double current_plan_seconds = 0.0;
    double current_position_read_seconds = 0.0;
    double future2_position_read_seconds = 0.0;
    double future4_position_read_seconds = 0.0;
    double future2_success_read_seconds = 0.0;
    double future4_success_read_seconds = 0.0;
    double future2_index_seconds = 0.0;
    double future4_index_seconds = 0.0;
    double future_cid_select_seconds = 0.0;
    double current_board_cache_seconds = 0.0;
    double raw_alloc_seconds = 0.0;
    double future_release_seconds = 0.0;
    double recalc_seconds = 0.0;
    double compact_seconds = 0.0;
    double partial_write_seconds = 0.0;
    double partial_read_seconds = 0.0;
    double temp_prepare_seconds = 0.0;
    double partial_cleanup_seconds = 0.0;
    double workspace_release_seconds = 0.0;
    double output_release_seconds = 0.0;
    double position_build_seconds = 0.0;
    double result_assembly_seconds = 0.0;
    double position_prepare_seconds = 0.0;
    double success_prepare_seconds = 0.0;
    double success_append_seconds = 0.0;
    double success_finish_seconds = 0.0;
    double success_header_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
};

using BCSingleChunkSolveTraceFn = void (*)(
    const char *event,
    const BCSingleChunkSolveStats &stats,
    void *user
);

template <typename StorageT>
struct BCSingleChunkSolveOptions {
    BCResidentSolveOptions<StorageT> solve;
    uint32_t current_chunk_cells = 64U;
    uint64_t current_chunk_max_rows = 1'000'000ULL;
    uint32_t current_chunk_rows = 128U;
    uint64_t current_chunk_max_bytes = 512ULL * 1024ULL * 1024ULL;
    bool restrict_future_cells_to_current_chunk = false;
    BCSingleChunkSolveTraceFn trace = nullptr;
    void *trace_user = nullptr;
};

template <typename StorageT>
struct BCSingleChunkSolveResult {
    BCResidentSolvedLayer<StorageT> layer;
    BCSingleChunkSolveStats stats;
};

struct BCSingleChunkSolveFileResult {
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    BCSingleChunkSolveStats stats;
};

template <typename StorageT>
struct BCSingleChunkFrontierLayer {
    BCPositionLayerReader position;
    BCSuccessOwnedValues<StorageT> success_values;
    BCFutureSuccessLookupView<StorageT> lookup;
    uint32_t row_width = 1U;
    BCSuccessDTypeMode dtype = bc_success_default_dtype_for_type<StorageT>();

    BCSingleChunkFrontierLayer() = default;
    BCSingleChunkFrontierLayer(const BCSingleChunkFrontierLayer &) = delete;
    BCSingleChunkFrontierLayer &operator=(const BCSingleChunkFrontierLayer &) = delete;

    BCSingleChunkFrontierLayer(BCSingleChunkFrontierLayer &&other) noexcept {
        *this = std::move(other);
    }

    BCSingleChunkFrontierLayer &operator=(BCSingleChunkFrontierLayer &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        position = std::move(other.position);
        success_values = std::move(other.success_values);
        lookup = std::move(other.lookup);
        row_width = other.row_width;
        dtype = other.dtype;
        refresh_lookup_view_after_move();
        return *this;
    }

    void open(
        std::vector<uint8_t> position_bytes,
        BCSuccessOwnedValues<StorageT> values,
        const BCLut &lut,
        uint32_t row_width_in = 1U,
        BCSuccessDTypeMode dtype_in = bc_success_default_dtype_for_type<StorageT>()
    ) {
        if (row_width_in == 0U) {
            throw std::invalid_argument("BC single chunk frontier layer row_width must be non-zero");
        }
        if (!bc_success_dtype_matches_type<StorageT>(dtype_in)) {
            throw std::invalid_argument("BC single chunk frontier layer dtype does not match storage type");
        }
        position.open(std::move(position_bytes), lut);
        row_width = row_width_in;
        dtype = dtype_in;
        success_values = std::move(values);
        lookup.open_flat(
            lut,
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size(),
            row_width
        );
    }

    [[nodiscard]] size_t success_value_count() const noexcept {
        return success_values.size();
    }

    [[nodiscard]] uint64_t success_resident_bytes(uint32_t value_size) const {
        if (success_values.file_backed) {
            return static_cast<uint64_t>(success_values.file_bytes.size());
        }
        return static_cast<uint64_t>(success_values.vector_values.capacity()) * value_size;
    }

private:
    void refresh_lookup_view_after_move() {
        if (position.cell_count() == 0U) {
            return;
        }
        lookup.rebind_flat_values(
            position,
            success_values.empty() ? nullptr : success_values.data(),
            success_values.size()
        );
    }
};

template <typename StorageT>
struct BCSingleChunkStrictFrontierFileResult {
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    BCSingleChunkSolveStats stats;
    BCSingleChunkFrontierLayer<StorageT> next_future4_layer;
};

template <typename StorageT>
struct BCSingleChunkCompactedBuild {
    std::vector<uint8_t> position_bytes;
    std::vector<StorageT> success_values;
    BCResidentCompactStats compact_stats;
    BCSingleChunkSolveStats stats;
};

struct BCSingleChunkLoadedWorkItem {
    size_t cell_index = 0U;
    uint32_t bucket_begin = 0U;
    uint32_t bucket_end = 0U;
    uint32_t word_begin = 0U;
    uint32_t word_end = 0U;
};

inline constexpr uint32_t kBCSingleChunkLargeBucketChunkWords = kBCResidentLargeBucketChunkWords;

template <typename StorageT>
// tmp4 stores the already weighted spawn-4 contribution. UInt32 accepts the
// extra rounding so the temporary payload stays one StorageT per current row.
using BCSingleChunkSum4T = StorageT;

template <typename StorageT>
[[nodiscard]] inline StorageT bc_single_chunk_scale_average_contribution(
    long double sum,
    uint32_t empty_count,
    double spawn_rate4,
    bool spawn4,
    StorageT zero_value
);

template <typename StorageT>
[[nodiscard]] inline StorageT bc_single_chunk_add_success_contribution(
    StorageT base,
    StorageT contribution
);

template <typename T>
class BCSingleChunkValueBuffer {
public:
    BCSingleChunkValueBuffer() = default;

    explicit BCSingleChunkValueBuffer(size_t size)
        : BCSingleChunkValueBuffer() {
        resize_uninitialized(size);
    }

    BCSingleChunkValueBuffer(const BCSingleChunkValueBuffer &) = delete;
    BCSingleChunkValueBuffer &operator=(const BCSingleChunkValueBuffer &) = delete;
    BCSingleChunkValueBuffer(BCSingleChunkValueBuffer &&other) noexcept
        : storage_(std::move(other.storage_)),
          values_(other.values_),
          size_(other.size_),
          capacity_(other.capacity_) {
        other.values_ = nullptr;
        other.size_ = 0U;
        other.capacity_ = 0U;
    }

    BCSingleChunkValueBuffer &operator=(BCSingleChunkValueBuffer &&other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            values_ = other.values_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.values_ = nullptr;
            other.size_ = 0U;
            other.capacity_ = 0U;
        }
        return *this;
    }

    [[nodiscard]] size_t size() const {
        return size_;
    }

    [[nodiscard]] size_t capacity() const {
        return capacity_;
    }

    [[nodiscard]] T *data() {
        return values_;
    }

    [[nodiscard]] const T *data() const {
        return values_;
    }

    [[nodiscard]] T &operator[](size_t index) {
        return values_[index];
    }

    [[nodiscard]] const T &operator[](size_t index) const {
        return values_[index];
    }

    void reset() {
        storage_.reset();
        values_ = nullptr;
        size_ = 0U;
        capacity_ = 0U;
    }

    void resize_uninitialized(size_t size) {
        if (size > capacity_) {
            const uint64_t data_bytes = checked_bytes_for(size);
            const uint64_t padded_bytes =
                data_bytes == 0U ? 0U : bc_direct_align_up(data_bytes, kAlignment);
            if (padded_bytes % sizeof(T) != 0U) {
                throw std::logic_error("BC single chunk value buffer padded byte count is not value-aligned");
            }
            storage_.reset(padded_bytes, kAlignment);
            values_ = reinterpret_cast<T *>(storage_.data());
            capacity_ = static_cast<size_t>(padded_bytes / sizeof(T));
        }
        size_ = size;
    }

    [[nodiscard]] uint64_t data_bytes() const {
        return checked_bytes_for(size_);
    }

    [[nodiscard]] uint64_t aligned_data_bytes(uint32_t alignment = kAlignment) const {
        const uint64_t bytes = data_bytes();
        return bytes == 0U ? 0U : bc_direct_align_up(bytes, alignment);
    }

    void zero_padding(uint32_t alignment = kAlignment) {
        const uint64_t bytes = data_bytes();
        const uint64_t padded = bytes == 0U ? 0U : bc_direct_align_up(bytes, alignment);
        if (padded <= bytes) {
            return;
        }
        if (padded > storage_.size()) {
            throw std::logic_error("BC single chunk value buffer padding exceeds storage");
        }
        std::memset(storage_.data() + static_cast<size_t>(bytes), 0, static_cast<size_t>(padded - bytes));
    }

private:
    static constexpr uint32_t kAlignment = 4096U;

    [[nodiscard]] static uint64_t checked_bytes_for(size_t size) {
        if (size > static_cast<size_t>(std::numeric_limits<uint64_t>::max() / sizeof(T))) {
            throw std::overflow_error("BC single chunk value buffer byte count overflow");
        }
        return static_cast<uint64_t>(size) * sizeof(T);
    }

    detail::BCAlignedBuffer storage_;
    T *values_ = nullptr;
    size_t size_ = 0U;
    size_t capacity_ = 0U;
};

template <typename StorageT>
struct BCSingleChunkSolveWorkspace {
    BCSingleChunkValueBuffer<StorageT> raw_values;
    BCSingleChunkValueBuffer<BCSingleChunkSum4T<StorageT>> sum4_values;
    std::vector<BCResidentBatchWorkspace<StorageT>> batch_workspaces;
};

[[nodiscard]] inline double bc_single_chunk_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <typename StorageT>
inline void bc_single_chunk_trace_stage(
    const BCSingleChunkSolveOptions<StorageT> &options,
    const char *event,
    const BCSingleChunkSolveStats &stats
) {
    if (options.trace != nullptr) {
        options.trace(event, stats, options.trace_user);
    }
}

template <typename StorageT>
void bc_single_chunk_validate_streaming_options(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    if (options.current_chunk_cells == 0U) {
        throw std::invalid_argument("BC single chunk solve current_chunk_cells must be non-zero");
    }
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC single chunk solve row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC single chunk solve dtype does not match storage type");
    }
    if (future2_success.row_width() != options.solve.row_width ||
        future4_success.row_width() != options.solve.row_width) {
        throw std::invalid_argument("BC single chunk solve future success row_width mismatch");
    }
    if (!bc_success_dtype_matches_type<StorageT>(future2_success.dtype_mode()) ||
        !bc_success_dtype_matches_type<StorageT>(future4_success.dtype_mode())) {
        throw std::invalid_argument("BC single chunk solve future success dtype mismatch");
    }
    bc_solve_validate_success_matches_position(future2_position, future2_success);
    bc_solve_validate_success_matches_position(future4_position, future4_success);
    if (current_position.cell_count() == 0U ||
        future2_position.cell_count() == 0U ||
        future4_position.cell_count() == 0U) {
        throw std::invalid_argument("BC single chunk solve positions must be open and non-empty");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.solve.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.solve.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC single chunk solve spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn4_tile_rank);
    if (future2_position.axis().layer_sum() != expected2 ||
        future4_position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC single chunk solve future layer_sum does not match spawn delta");
    }
    if (future2_position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC single chunk solve family_unit mismatch");
    }
}

template <typename StorageT>
void bc_single_chunk_validate_frontier_options(
    const BCPositionStreamingReader &current_position,
    const BCResidentSolvedLayer<StorageT> &future2_layer,
    const BCResidentSolvedLayer<StorageT> &future4_layer,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    if (options.current_chunk_cells == 0U) {
        throw std::invalid_argument("BC single chunk frontier current_chunk_cells must be non-zero");
    }
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC single chunk frontier row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC single chunk frontier dtype does not match storage type");
    }
    if (future2_layer.row_width != options.solve.row_width ||
        future4_layer.row_width != options.solve.row_width) {
        throw std::invalid_argument("BC single chunk frontier future row_width mismatch");
    }
    if (future2_layer.dtype != options.solve.dtype ||
        future4_layer.dtype != options.solve.dtype) {
        throw std::invalid_argument("BC single chunk frontier future dtype mismatch");
    }
    if (current_position.cell_count() == 0U ||
        future2_layer.position.cell_count() == 0U ||
        future4_layer.position.cell_count() == 0U) {
        throw std::invalid_argument("BC single chunk frontier positions must be open and non-empty");
    }
    const uint64_t future2_expected_values =
        bc_success_total_values_for(future2_layer.position, options.solve.row_width);
    const uint64_t future4_expected_values =
        bc_success_total_values_for(future4_layer.position, options.solve.row_width);
    if (future2_expected_values != future2_layer.success_values.size() ||
        future4_expected_values != future4_layer.success_values.size()) {
        throw std::invalid_argument("BC single chunk frontier future value count mismatch");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.solve.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.solve.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC single chunk frontier spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn4_tile_rank);
    if (future2_layer.position.axis().layer_sum() != expected2 ||
        future4_layer.position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC single chunk frontier future layer_sum does not match spawn delta");
    }
    if (future2_layer.position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_layer.position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC single chunk frontier family_unit mismatch");
    }
}

template <typename StorageT>
void bc_single_chunk_validate_strict_future4_frontier_options(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCSingleChunkFrontierLayer<StorageT> &future4_layer,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    if (options.current_chunk_cells == 0U) {
        throw std::invalid_argument("BC single chunk strict frontier current_chunk_cells must be non-zero");
    }
    if (options.solve.row_width == 0U) {
        throw std::invalid_argument("BC single chunk strict frontier row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(options.solve.dtype)) {
        throw std::invalid_argument("BC single chunk strict frontier dtype does not match storage type");
    }
    if (future2_success.row_width() != options.solve.row_width ||
        future4_layer.row_width != options.solve.row_width) {
        throw std::invalid_argument("BC single chunk strict frontier future row_width mismatch");
    }
    if (!bc_success_dtype_matches_type<StorageT>(future2_success.dtype_mode()) ||
        future4_layer.dtype != options.solve.dtype) {
        throw std::invalid_argument("BC single chunk strict frontier future dtype mismatch");
    }
    bc_solve_validate_success_matches_position(future2_position, future2_success);
    if (current_position.cell_count() == 0U ||
        future2_position.cell_count() == 0U ||
        future4_layer.position.cell_count() == 0U) {
        throw std::invalid_argument("BC single chunk strict frontier positions must be open and non-empty");
    }
    const uint64_t future4_expected_values =
        bc_success_total_values_for(future4_layer.position, options.solve.row_width);
    if (future4_expected_values != future4_layer.success_value_count()) {
        throw std::invalid_argument("BC single chunk strict frontier future4 value count mismatch");
    }
    const BCLut &lut = current_position.lut();
    if (!lut.is_legal_tile(options.solve.edge_options.spawn2_tile_rank) ||
        !lut.is_legal_tile(options.solve.edge_options.spawn4_tile_rank)) {
        throw std::invalid_argument("BC single chunk strict frontier spawn tile is outside the LUT alphabet");
    }
    const uint64_t expected2 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn2_tile_rank);
    const uint64_t expected4 =
        static_cast<uint64_t>(current_position.axis().layer_sum()) +
        lut.tile_sum_value(options.solve.edge_options.spawn4_tile_rank);
    if (future2_position.axis().layer_sum() != expected2 ||
        future4_layer.position.axis().layer_sum() != expected4) {
        throw std::invalid_argument("BC single chunk strict frontier future layer_sum does not match spawn delta");
    }
    if (future2_position.axis().family_unit() != current_position.axis().family_unit() ||
        future4_layer.position.axis().family_unit() != current_position.axis().family_unit()) {
        throw std::invalid_argument("BC single chunk strict frontier family_unit mismatch");
    }
}

template <typename StorageT>
[[nodiscard]] inline BCSingleChunkSolveOptions<StorageT>
bc_single_chunk_effective_recalc_options_for_current_layer(
    const BCPositionStreamingReader &current_position,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    BCSingleChunkSolveOptions<StorageT> effective = options;
    const int target_rank = effective.solve.edge_options.success_target_rank;
    if (target_rank > 0 && target_rank < 16) {
        const uint32_t target_sum =
            current_position.lut().tile_sum_value(static_cast<uint8_t>(target_rank));
        if (current_position.axis().layer_sum() < target_sum) {
            effective.solve.edge_options.success_target_rank = 0;
            effective.solve.edge_options.success_shifts = nullptr;
            effective.solve.edge_options.success_check_all_cells = false;
        }
    }
    return effective;
}

[[nodiscard]] inline std::vector<BCSingleChunkLoadedWorkItem> bc_single_chunk_build_loaded_work_items(
    const BCLut &lut,
    const std::vector<BCLoadedCell> &current_cells
) {
    std::vector<BCSingleChunkLoadedWorkItem> items;
    items.reserve(current_cells.size());
    for (size_t cell_i = 0U; cell_i < current_cells.size(); ++cell_i) {
        const BCLoadedCellView view = current_cells[cell_i].view();
        if (view.success_rows == 0U || view.empty()) {
            continue;
        }
        uint32_t small_begin = 0U;
        uint32_t small_end = 0U;
        auto flush_small = [&]() {
            if (small_begin == small_end) {
                return;
            }
            items.push_back(BCSingleChunkLoadedWorkItem{
                cell_i,
                small_begin,
                small_end,
                0U,
                0U
            });
            small_begin = small_end;
        };
        for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
            const BCBucketEntry &bucket = view.buckets.data[bucket_i];
            const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
            const uint32_t word_count = words_for_bits(bitmap_len);
            if (word_count >= kBCResidentLargeBucketMinWords) {
                flush_small();
                for (uint32_t word_begin = 0U; word_begin < word_count;) {
                    const uint32_t word_end = std::min<uint32_t>(
                        word_count,
                        word_begin + kBCSingleChunkLargeBucketChunkWords
                    );
                    items.push_back(BCSingleChunkLoadedWorkItem{
                        cell_i,
                        bucket_i,
                        bucket_i + 1U,
                        word_begin,
                        word_end
                    });
                    word_begin = word_end;
                }
                small_begin = bucket_i + 1U;
                small_end = small_begin;
            } else {
                if (small_begin == small_end) {
                    small_begin = bucket_i;
                }
                small_end = bucket_i + 1U;
            }
        }
        flush_small();
    }
    return items;
}

[[nodiscard]] inline std::vector<CellId> bc_single_chunk_next_current_cids(
    const BCPositionStreamingReader &current_position,
    CellId begin,
    uint32_t max_cells,
    uint64_t max_rows
) {
    const CellId cell_count = current_position.cell_count();
    std::vector<CellId> cids;
    cids.reserve(max_cells);
    uint64_t rows = 0U;
    for (CellId cid = begin; cid < cell_count; ++cid) {
        if (!cids.empty() && cids.size() >= max_cells) {
            break;
        }
        const uint32_t cell_rows = current_position.descriptor(cid).success_rows;
        if (!cids.empty() && max_rows != 0U && rows + cell_rows > max_rows) {
            break;
        }
        cids.push_back(cid);
        rows += cell_rows;
    }
    return cids;
}

[[nodiscard]] inline std::vector<CellId> bc_single_chunk_next_row_slab_cids(
    const BCPositionStreamingReader &current_position,
    FamilyId row_begin,
    uint32_t max_rows,
    uint64_t max_bytes,
    uint32_t row_width,
    uint32_t value_size
) {
    if (max_rows == 0U) {
        throw std::invalid_argument("BC single chunk row slab max_rows must be non-zero");
    }
    if (row_width == 0U || value_size == 0U) {
        throw std::invalid_argument("BC single chunk row slab value shape must be non-zero");
    }
    const BCCellMatrix matrix(current_position.axis());
    const FamilyId family_count = static_cast<FamilyId>(matrix.family_count());
    if (row_begin >= family_count) {
        return {};
    }
    std::vector<CellId> cids;
    const uint64_t reserve_rows =
        std::min<uint64_t>(max_rows, static_cast<uint64_t>(family_count - row_begin));
    if (reserve_rows > std::numeric_limits<size_t>::max() / family_count) {
        throw std::overflow_error("BC single chunk row slab cid reserve exceeds size_t");
    }
    cids.reserve(static_cast<size_t>(reserve_rows) * family_count);
    uint64_t bytes = 0U;
    uint32_t rows = 0U;
    for (FamilyId row = row_begin; row < family_count && rows < max_rows; ++row) {
        uint64_t row_values = 0U;
        for (FamilyId col = 0U; col < family_count; ++col) {
            const CellId cid = matrix.cid(row, col);
            row_values = bc_checked_add_u64(
                row_values,
                static_cast<uint64_t>(current_position.descriptor(cid).success_rows) * row_width,
                "BC single chunk row slab value count overflow"
            );
        }
        if (row_values > std::numeric_limits<uint64_t>::max() / value_size) {
            throw std::overflow_error("BC single chunk row slab byte count overflow");
        }
        const uint64_t row_bytes = row_values * value_size;
        if (!cids.empty() &&
            max_bytes != 0U &&
            (bytes > max_bytes || row_bytes > max_bytes - bytes)) {
            break;
        }
        for (FamilyId col = 0U; col < family_count; ++col) {
            cids.push_back(matrix.cid(row, col));
        }
        bytes = bc_checked_add_u64(bytes, row_bytes, "BC single chunk row slab bytes overflow");
        ++rows;
    }
    return cids;
}

[[nodiscard]] inline std::vector<CellId> bc_single_chunk_nonempty_cell_ids(
    const BCPositionStreamingReader &position
) {
    std::vector<CellId> cids;
    cids.reserve(position.cell_count());
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.success_rows != 0U && !desc.empty()) {
            cids.push_back(cid);
        }
    }
    return cids;
}

[[nodiscard]] inline std::vector<uint64_t> bc_single_chunk_success_value_offsets(
    const BCPositionStreamingReader &position,
    uint32_t row_width
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC single chunk success value offsets row_width must be non-zero");
    }
    std::vector<uint64_t> offsets(static_cast<size_t>(position.cell_count()) + 1U, 0U);
    uint64_t cursor = 0U;
    for (CellId cid = 0U; cid < position.cell_count(); ++cid) {
        offsets[static_cast<size_t>(cid)] = cursor;
        const uint64_t values =
            static_cast<uint64_t>(position.descriptor(cid).success_rows) * row_width;
        cursor = bc_checked_add_u64(
            cursor,
            values,
            "BC single chunk success value offset overflow"
        );
    }
    offsets.back() = cursor;
    return offsets;
}

inline void bc_single_chunk_mark_query_cids(
    const std::vector<BCSolvePreparedQuery> &queries,
    std::vector<uint8_t> &mark,
    std::vector<CellId> &out
) {
    for (const BCSolvePreparedQuery &query : queries) {
        if (!query.valid || query.cid >= mark.size()) {
            continue;
        }
        uint8_t &slot = mark[static_cast<size_t>(query.cid)];
        if (slot != 0U) {
            continue;
        }
        slot = 1U;
        out.push_back(query.cid);
    }
}

template <typename StorageT>
void bc_single_chunk_collect_future_cids(
    const BCLut &lut,
    const std::vector<BCLoadedCell> &current_cells,
    const BCFamilyTable &future2_axis,
    const BCFamilyTable &future4_axis,
    const BCSingleChunkSolveOptions<StorageT> &options,
    std::vector<CellId> &future2_cids,
    std::vector<CellId> &future4_cids
) {
    const uint64_t future2_cell_count =
        static_cast<uint64_t>(future2_axis.family_count()) * future2_axis.family_count();
    const uint64_t future4_cell_count =
        static_cast<uint64_t>(future4_axis.family_count()) * future4_axis.family_count();
    if (future2_cell_count > std::numeric_limits<size_t>::max() ||
        future4_cell_count > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("BC single chunk future cell count exceeds size_t");
    }
    std::vector<uint8_t> mark2(static_cast<size_t>(future2_cell_count), 0U);
    std::vector<uint8_t> mark4(static_cast<size_t>(future4_cell_count), 0U);
    BCSolveEdgeWorkspace<StorageT> workspace;
    for (const BCLoadedCell &cell : current_cells) {
        if (cell.success_rows == 0U || cell.buckets.empty()) {
            continue;
        }
        BCLoadedCellScanner(lut, cell.view()).for_each_board(
            [&](const BCScannedBoardEntry &entry) {
                (void)bc_solve_collect_board_queries<StorageT>(
                    lut,
                    future2_axis,
                    future4_axis,
                    entry.board,
                    options.solve.directions,
                    options.solve.filter2,
                    options.solve.filter4,
                    workspace,
                    options.solve.edge_options,
                    options.solve.word_sums,
                    nullptr
                );
                bc_single_chunk_mark_query_cids(workspace.queries2, mark2, future2_cids);
                bc_single_chunk_mark_query_cids(workspace.queries4, mark4, future4_cids);
            }
        );
    }
    std::sort(future2_cids.begin(), future2_cids.end());
    std::sort(future4_cids.begin(), future4_cids.end());
}

template <typename StorageT>
BCFutureSuccessLookupView<StorageT> bc_single_chunk_load_future_lookup(
    const BCPositionStreamingReader &position,
    const BCSuccessStreamingReader &success,
    const std::vector<CellId> &cids,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    BCCellLoadStats &position_stats,
    BCSuccessLoadStats &success_stats,
    double &position_read_seconds,
    double &success_read_seconds,
    double &index_seconds,
    uint64_t &resident_position_bytes,
    uint64_t &resident_success_bytes,
    bool use_full_success_payload = false,
    const BCSingleChunkSolveOptions<StorageT> *trace_options = nullptr,
    BCSingleChunkSolveStats *trace_stats = nullptr,
    const char *position_event = nullptr,
    const char *success_event = nullptr,
    const char *index_event = nullptr
) {
    BCCellLoadStats local_position_stats;
    const double position_t0 = bc_single_chunk_now_seconds();
    std::vector<BCLoadedCell> position_cells = position.load_cells(cids, &local_position_stats);
    position_read_seconds += bc_single_chunk_now_seconds() - position_t0;
    bc_single_chunk_add_cell_load_stats(position_stats, local_position_stats);
    if (trace_options != nullptr && trace_stats != nullptr && position_event != nullptr) {
        bc_single_chunk_trace_stage(*trace_options, position_event, *trace_stats);
    }

    resident_position_bytes = 0U;
    for (const BCLoadedCell &cell : position_cells) {
        resident_position_bytes +=
            static_cast<uint64_t>(cell.buckets.capacity()) * sizeof(BCBucketEntry) +
            static_cast<uint64_t>(cell.rank_payload.capacity());
    }

    if (use_full_success_payload) {
        BCSuccessLoadStats local_success_stats;
        const double success_t0 = bc_single_chunk_now_seconds();
        BCSuccessOwnedValues<StorageT> success_values =
            success.template read_all_values_typed_owned<StorageT>(&local_success_stats);
        success_read_seconds += bc_single_chunk_now_seconds() - success_t0;
        bc_single_chunk_add_success_load_stats(success_stats, local_success_stats);
        if (trace_options != nullptr && trace_stats != nullptr && success_event != nullptr) {
            bc_single_chunk_trace_stage(*trace_options, success_event, *trace_stats);
        }
        resident_success_bytes = success_values.file_backed ?
            static_cast<uint64_t>(success_values.file_bytes.size()) :
            static_cast<uint64_t>(success_values.vector_values.capacity()) * sizeof(StorageT);

        const double index_t0 = bc_single_chunk_now_seconds();
        BCFutureSuccessLookupView<StorageT> lookup;
        lookup.open_loaded_flat_owned_values(
            position.lut(),
            position.cell_count(),
            std::move(position_cells),
            bc_single_chunk_success_value_offsets(position, row_width),
            std::move(success_values),
            row_width,
            dtype
        );
        index_seconds += bc_single_chunk_now_seconds() - index_t0;
        if (trace_options != nullptr && trace_stats != nullptr && index_event != nullptr) {
            bc_single_chunk_trace_stage(*trace_options, index_event, *trace_stats);
        }
        return lookup;
    }

    BCSuccessLoadStats local_success_stats;
    const double success_t0 = bc_single_chunk_now_seconds();
    std::vector<BCLoadedSuccessCell> success_cells = success.load_cells(cids, &local_success_stats);
    success_read_seconds += bc_single_chunk_now_seconds() - success_t0;
    bc_single_chunk_add_success_load_stats(success_stats, local_success_stats);
    if (trace_options != nullptr && trace_stats != nullptr && success_event != nullptr) {
        bc_single_chunk_trace_stage(*trace_options, success_event, *trace_stats);
    }

    resident_success_bytes = 0U;
    for (const BCLoadedSuccessCell &cell : success_cells) {
        resident_success_bytes +=
            static_cast<uint64_t>(cell.raw_bytes.capacity()) +
            static_cast<uint64_t>(cell.values.capacity()) * sizeof(uint32_t);
    }

    const double index_t0 = bc_single_chunk_now_seconds();
    BCFutureSuccessLookupView<StorageT> lookup;
    lookup.open_loaded(
        position.lut(),
        position.cell_count(),
        std::move(position_cells),
        success_cells,
        row_width,
        dtype
    );
    index_seconds += bc_single_chunk_now_seconds() - index_t0;
    if (trace_options != nullptr && trace_stats != nullptr && index_event != nullptr) {
        bc_single_chunk_trace_stage(*trace_options, index_event, *trace_stats);
    }
    return lookup;
}

template <typename StorageT>
inline void bc_single_chunk_push_candidate(
    const BCLut &lut,
    const BCFamilyTable &axis,
    const BCSolveTargetFamilyFilter &filter,
    uint64_t spawned,
    uint64_t moved,
    uint16_t ref,
    BCDirectionMask move_axis,
    const BCResidentSolveOptions<StorageT> &options,
    std::vector<uint64_t> &boards,
    std::vector<uint16_t> &refs,
    uint64_t &unchanged_moves,
    uint64_t &prefilter_checks,
    uint64_t &prefilter_skips
) {
    if (moved == spawned) {
        ++unchanged_moves;
        return;
    }
    if (filter.enabled) {
        ++prefilter_checks;
        if (!bc_solve_physical_target_family_may_hit(
                lut,
                axis,
                filter,
                moved,
                move_axis,
                options.word_sums)) {
            ++prefilter_skips;
            return;
        }
    }
    boards.push_back(moved);
    refs.push_back(ref);
}

enum class BCSingleChunkSolvePhase : uint8_t {
    Spawn4,
    Spawn2,
};

template <typename StorageT>
void bc_single_chunk_solve_phase_batch(
    BCResidentBatchWorkspace<StorageT> &workspace,
    BCSingleChunkValueBuffer<StorageT> &raw_values,
    BCSingleChunkValueBuffer<BCSingleChunkSum4T<StorageT>> &sum4_values,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolvePhase phase,
    BCSingleChunkSolveStats &stats
) {
    const uint32_t count = workspace.count;
    if (count == 0U) {
        return;
    }
    ++stats.edge.batch_flushes;
    stats.edge.batch_source_boards += count;
    if (count < BCResidentBatchWorkspace<StorageT>::kBatchSize) {
        ++stats.edge.batch_tail_flushes;
    }
    workspace.canonical2_boards.clear();
    workspace.canonical2_refs.clear();
    workspace.queries2.clear();

    const bool phase2 = phase == BCSingleChunkSolvePhase::Spawn2;
    const uint8_t spawn_rank = phase2
        ? options.solve.edge_options.spawn2_tile_rank
        : options.solve.edge_options.spawn4_tile_rank;
    const BCSolveTargetFamilyFilter &filter = phase2 ? options.solve.filter2 : options.solve.filter4;
    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    uint64_t batch_terminal_success = 0U;
    uint64_t batch_empty_slots = 0U;
    uint64_t batch_spawned_boards = 0U;
    uint64_t batch_move_all_dir_calls = 0U;
    uint64_t batch_selective_move_calls = 0U;
    uint64_t batch_unchanged_moves = 0U;
    uint64_t batch_prefilter_checks = 0U;
    uint64_t batch_prefilter_skips = 0U;
    const bool fast_unfiltered_both =
        options.solve.directions == BCDirectionMask::Both && !filter.enabled;
    auto push_moved_unfiltered = [&](
        uint64_t spawned,
        uint64_t moved,
        uint16_t ref
    ) {
        if (moved == spawned) {
            ++batch_unchanged_moves;
            return;
        }
        workspace.canonical2_boards.push_back(moved);
        workspace.canonical2_refs.push_back(ref);
    };

    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = workspace.boards[board_slot];
        workspace.empty_counts[board_slot] = 0U;
        workspace.empty_masks[board_slot] = 0U;
        workspace.terminal[board_slot] = 0U;
        if (success_check_enabled &&
            bc_solve_is_success_board(board, options.solve.edge_options)) {
            workspace.terminal[board_slot] = 1U;
            if (phase2) {
                ++batch_terminal_success;
            }
            continue;
        }
        uint32_t empty_mask = bc_zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            ++workspace.empty_counts[board_slot];
            if (phase2) {
                ++batch_empty_slots;
                batch_spawned_boards += 2U;
            }
            const uint64_t spawned = set_board_tile_unchecked(board, cell, spawn_rank);
            if (fast_unfiltered_both) {
                ++batch_move_all_dir_calls;
                const auto moved_h = BoardMover::move_horizontal_pair(spawned);
                const auto moved_v = BoardMover::move_vertical_pair(spawned);
                push_moved_unfiltered(spawned, moved_h.first, ref);
                push_moved_unfiltered(spawned, moved_h.second, ref);
                push_moved_unfiltered(spawned, moved_v.first, ref);
                push_moved_unfiltered(spawned, moved_v.second, ref);
                continue;
            }
            if (options.solve.directions == BCDirectionMask::Both) {
                ++batch_move_all_dir_calls;
                const auto moved_h = BoardMover::move_horizontal_pair(spawned);
                const auto moved_v = BoardMover::move_vertical_pair(spawned);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_h.first, ref,
                    BCDirectionMask::Horizontal, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_h.second, ref,
                    BCDirectionMask::Horizontal, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_v.first, ref,
                    BCDirectionMask::Vertical, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_v.second, ref,
                    BCDirectionMask::Vertical, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
            } else {
                if (bc_has_horizontal(options.solve.directions)) {
                    batch_selective_move_calls += 2U;
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Horizontal, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Horizontal, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                }
                if (bc_has_vertical(options.solve.directions)) {
                    batch_selective_move_calls += 2U;
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Vertical, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Vertical, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                }
            }
        }
    }

    if (!workspace.canonical2_boards.empty()) {
        ++stats.edge.canonical_flushes;
        CanonicalBatch::canonicalize_inplace(
            workspace.canonical2_boards.data(),
            workspace.canonical2_boards.size(),
            options.solve.edge_options.canonical_symm_mode
        );
        stats.edge.canonicalized_candidates += workspace.canonical2_boards.size();
        const BCSolvePreparedQueryEncoder encoder(
            lut,
            future_axis,
            options.solve.edge_options.future_cell_modulus
        );
        for (size_t i = 0U; i < workspace.canonical2_boards.size(); ++i) {
            BCSolvePreparedQuery query;
            const bool encoded = encoder.encode(
                unpack_board_to_quadrants(workspace.canonical2_boards[i]),
                workspace.canonical2_refs[i],
                spawn_rank,
                BCDirectionMask::Both,
                query
            );
            if (!encoded) {
                ++stats.edge.encode_rejects;
                continue;
            }
            workspace.queries2.push_back(query);
            ++stats.edge.encoded_queries;
        }
        workspace.canonical2_boards.clear();
        workspace.canonical2_refs.clear();
    }
    if (phase2) {
        stats.edge.source_boards += count;
        stats.edge.terminal_success_boards += batch_terminal_success;
        stats.edge.empty_slots += batch_empty_slots;
        stats.edge.spawned_boards += batch_spawned_boards;
    }
    stats.edge.move_all_dir_calls += batch_move_all_dir_calls;
    stats.edge.selective_move_calls += batch_selective_move_calls;
    stats.edge.unchanged_moves += batch_unchanged_moves;
    stats.edge.prefilter_checks += batch_prefilter_checks;
    stats.edge.prefilter_skips += batch_prefilter_skips;

    for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
        std::fill_n(
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            options.solve.zero_value
        );
        (void)future_lookup.reduce_max_queries(
            workspace.queries2,
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            lane,
            &stats.edge,
            true
        );
        if (phase2) {
            stats.edge.finalized_boards += lane == 0U ? count : 0U;
        }
        if (phase == BCSingleChunkSolvePhase::Spawn4) {
            for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
                StorageT contribution = options.solve.zero_value;
                if (workspace.empty_counts[board_slot] != 0U) {
                    if constexpr (std::is_same_v<StorageT, uint32_t>) {
                        uint64_t integer_sum = 0U;
                        uint32_t mask = workspace.empty_masks[board_slot];
                        while (mask != 0U) {
                            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                            const size_t index =
                                static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                            integer_sum += static_cast<uint64_t>(workspace.best2[index]);
                        }
                        contribution = bc_single_chunk_scale_average_contribution<StorageT>(
                            static_cast<long double>(integer_sum),
                            workspace.empty_counts[board_slot],
                            options.solve.edge_options.spawn_rate4,
                            true,
                            options.solve.zero_value
                        );
                    } else {
                        long double sum = 0.0L;
                        uint32_t mask = workspace.empty_masks[board_slot];
                        while (mask != 0U) {
                            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                            const size_t index =
                                static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                            sum += static_cast<long double>(workspace.best2[index]);
                        }
                        contribution = bc_single_chunk_scale_average_contribution<StorageT>(
                            sum,
                            workspace.empty_counts[board_slot],
                            options.solve.edge_options.spawn_rate4,
                            true,
                            options.solve.zero_value
                        );
                    }
                }
                const uint64_t row_index = workspace.output_indices[board_slot];
                const uint64_t value_index =
                    row_index * static_cast<uint64_t>(options.solve.row_width) + lane;
                if (value_index >= sum4_values.size()) {
                    throw std::out_of_range("BC single chunk phase4 contribution index out of range");
                }
                sum4_values[static_cast<size_t>(value_index)] = contribution;
            }
            continue;
        }

        for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
            const uint64_t row_index = workspace.output_indices[board_slot];
            const uint64_t value_index =
                row_index * static_cast<uint64_t>(options.solve.row_width) + lane;
            if (value_index >= sum4_values.size()) {
                throw std::out_of_range("BC single chunk phase2 contribution index out of range");
            }
            StorageT value = options.solve.zero_value;
            if (workspace.terminal[board_slot] != 0U) {
                value = options.solve.terminal_value;
            } else if (workspace.empty_counts[board_slot] != 0U) {
                StorageT contribution = options.solve.zero_value;
                if constexpr (std::is_same_v<StorageT, uint32_t>) {
                    uint64_t integer_sum = 0U;
                    uint32_t mask = workspace.empty_masks[board_slot];
                    while (mask != 0U) {
                        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                        const size_t index =
                            static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                        integer_sum += static_cast<uint64_t>(workspace.best2[index]);
                    }
                    contribution = bc_single_chunk_scale_average_contribution<StorageT>(
                            static_cast<long double>(integer_sum),
                            workspace.empty_counts[board_slot],
                            options.solve.edge_options.spawn_rate4,
                            false,
                            options.solve.zero_value);
                } else {
                    long double sum2 = 0.0L;
                    uint32_t mask = workspace.empty_masks[board_slot];
                    while (mask != 0U) {
                        const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                        const size_t index =
                            static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                        sum2 += static_cast<long double>(workspace.best2[index]);
                    }
                    contribution = bc_single_chunk_scale_average_contribution<StorageT>(
                        sum2,
                        workspace.empty_counts[board_slot],
                        options.solve.edge_options.spawn_rate4,
                        false,
                        options.solve.zero_value
                    );
                }
                value = sum4_values[static_cast<size_t>(value_index)];
                if (contribution != options.solve.zero_value) {
                    value = bc_single_chunk_add_success_contribution<StorageT>(
                        value,
                        contribution
                    );
                }
            }
            if (value_index >= raw_values.size()) {
                throw std::out_of_range("BC single chunk raw output index out of range");
            }
            raw_values[static_cast<size_t>(value_index)] = value;
        }
    }
    workspace.clear_batch();
}

template <typename StorageT>
void bc_single_chunk_solve_phase_for_current_cells(
    const std::vector<BCLoadedCell> &current_cells,
    const std::vector<BCSingleChunkLoadedWorkItem> &work_items,
    const std::vector<uint64_t> &cell_offsets,
    BCSingleChunkValueBuffer<StorageT> &raw_values,
    BCSingleChunkValueBuffer<BCSingleChunkSum4T<StorageT>> &sum4_values,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolvePhase phase,
    BCSingleChunkSolveStats &stats,
    std::vector<BCResidentBatchWorkspace<StorageT>> *batch_workspaces = nullptr
) {
    if (cell_offsets.size() != current_cells.size() + 1U) {
        throw std::invalid_argument("BC single chunk current cell offsets mismatch");
    }
    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    std::vector<BCSingleChunkSolveStats> per_thread(static_cast<size_t>(threads));
    std::vector<BCResidentBatchWorkspace<StorageT>> local_workspaces;
    if (batch_workspaces == nullptr) {
        local_workspaces.resize(static_cast<size_t>(threads));
        batch_workspaces = &local_workspaces;
    } else if (batch_workspaces->size() < static_cast<size_t>(threads)) {
        batch_workspaces->resize(static_cast<size_t>(threads));
    }
    const double t0 = bc_single_chunk_now_seconds();
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCSingleChunkSolveStats &thread_stats = per_thread[static_cast<size_t>(tid)];
        BCResidentBatchWorkspace<StorageT> &workspace =
            (*batch_workspaces)[static_cast<size_t>(tid)];
        workspace.clear_batch();
        auto flush = [&]() {
            bc_single_chunk_solve_phase_batch<StorageT>(
                workspace,
                raw_values,
                sum4_values,
                lut,
                future_axis,
                future_lookup,
                options,
                phase,
                thread_stats
            );
        };
#pragma omp for schedule(dynamic, 1)
        for (int64_t item_i_signed = 0;
             item_i_signed < static_cast<int64_t>(work_items.size());
             ++item_i_signed) {
            const BCSingleChunkLoadedWorkItem &item =
                work_items[static_cast<size_t>(item_i_signed)];
            const size_t cell_i = item.cell_index;
            const BCLoadedCell &cell = current_cells[cell_i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                continue;
            }
            const uint64_t cell_base = cell_offsets[cell_i];
            BCLoadedCellScanner scanner(lut, cell.view());
            for (uint32_t bucket_i = item.bucket_begin; bucket_i < item.bucket_end; ++bucket_i) {
                const bool ranged_bucket =
                    item.bucket_end == item.bucket_begin + 1U && item.word_end != 0U;
                scanner.for_each_bucket_word_range_board(
                    bucket_i,
                    ranged_bucket ? item.word_begin : 0U,
                    ranged_bucket ? item.word_end : 0U,
                    [&](const BCScannedBoardEntry &entry) {
                        workspace.boards[workspace.count] = entry.board;
                        workspace.output_indices[workspace.count] = cell_base + entry.local_success_row;
                        ++workspace.count;
                        if (phase == BCSingleChunkSolvePhase::Spawn2) {
                            ++thread_stats.current_rows;
                            ++thread_stats.current_boards;
                        }
                        if (workspace.count == BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                            flush();
                        }
                    }
                );
            }
            flush();
        }
        flush();
    }
    stats.recalc_seconds += bc_single_chunk_now_seconds() - t0;
    for (const BCSingleChunkSolveStats &thread_stats : per_thread) {
        stats.current_rows += thread_stats.current_rows;
        stats.current_boards += thread_stats.current_boards;
        bc_resident_solve_accumulate_edge_stats(stats.edge, thread_stats.edge);
    }
}

template <typename StorageT>
[[nodiscard]] inline StorageT bc_single_chunk_cast_weighted_success(long double value) {
    if constexpr (std::is_integral_v<StorageT>) {
        if (value <= 0.0L) {
            return StorageT{};
        }
        const long double max_value =
            static_cast<long double>(std::numeric_limits<StorageT>::max());
        if (value >= max_value) {
            return std::numeric_limits<StorageT>::max();
        }
    }
    return static_cast<StorageT>(value);
}

template <typename StorageT>
[[nodiscard]] inline StorageT bc_single_chunk_scale_average_contribution(
    long double sum,
    uint32_t empty_count,
    double spawn_rate4,
    bool spawn4,
    StorageT zero_value
) {
    if (empty_count == 0U) {
        return zero_value;
    }
    if constexpr (std::is_same_v<StorageT, uint32_t>) {
        if (spawn_rate4 == 0.1) {
            const uint64_t integer_sum = static_cast<uint64_t>(sum);
            const uint64_t numerator = spawn4 ? integer_sum : 9ULL * integer_sum;
            const uint64_t denominator = 10ULL * static_cast<uint64_t>(empty_count);
            return static_cast<uint32_t>(numerator / denominator);
        }
    }
    const long double p4 = static_cast<long double>(spawn_rate4);
    const long double weight = spawn4 ? p4 : (1.0L - p4);
    return bc_single_chunk_cast_weighted_success<StorageT>(
        (sum * weight) / static_cast<long double>(empty_count)
    );
}

template <typename StorageT>
[[nodiscard]] inline StorageT bc_single_chunk_add_success_contribution(
    StorageT base,
    StorageT contribution
) {
    if constexpr (std::is_integral_v<StorageT>) {
        if (base > std::numeric_limits<StorageT>::max() - contribution) {
            return std::numeric_limits<StorageT>::max();
        }
    }
    return static_cast<StorageT>(base + contribution);
}

template <typename StorageT, BCSingleChunkSolvePhase Phase>
void bc_single_chunk_solve_weighted_phase_batch(
    BCResidentBatchWorkspace<StorageT> &workspace,
    BCSingleChunkValueBuffer<StorageT> &values,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveStats &stats
) {
    const uint32_t count = workspace.count;
    if (count == 0U) {
        return;
    }
    ++stats.edge.batch_flushes;
    stats.edge.batch_source_boards += count;
    if (count < BCResidentBatchWorkspace<StorageT>::kBatchSize) {
        ++stats.edge.batch_tail_flushes;
    }
    workspace.canonical2_boards.clear();
    workspace.canonical2_refs.clear();
    workspace.queries2.clear();

    constexpr bool phase2 = Phase == BCSingleChunkSolvePhase::Spawn2;
    const uint8_t spawn_rank = phase2
        ? options.solve.edge_options.spawn2_tile_rank
        : options.solve.edge_options.spawn4_tile_rank;
    const BCSolveTargetFamilyFilter &filter = phase2 ? options.solve.filter2 : options.solve.filter4;
    const bool success_check_enabled =
        bc_solve_success_check_enabled(options.solve.edge_options);
    uint64_t batch_terminal_success = 0U;
    uint64_t batch_empty_slots = 0U;
    uint64_t batch_spawned_boards = 0U;
    uint64_t batch_move_all_dir_calls = 0U;
    uint64_t batch_selective_move_calls = 0U;
    uint64_t batch_unchanged_moves = 0U;
    uint64_t batch_prefilter_checks = 0U;
    uint64_t batch_prefilter_skips = 0U;
    const bool fast_unfiltered_both =
        options.solve.directions == BCDirectionMask::Both && !filter.enabled;
    auto push_moved_unfiltered = [&](
        uint64_t spawned,
        uint64_t moved,
        uint16_t ref
    ) {
        if (moved == spawned) {
            ++batch_unchanged_moves;
            return;
        }
        workspace.canonical2_boards.push_back(moved);
        workspace.canonical2_refs.push_back(ref);
    };

    for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
        const uint64_t board = workspace.boards[board_slot];
        workspace.empty_counts[board_slot] = 0U;
        workspace.empty_masks[board_slot] = 0U;
        workspace.terminal[board_slot] = 0U;
        if (success_check_enabled &&
            bc_solve_is_success_board(board, options.solve.edge_options)) {
            workspace.terminal[board_slot] = 1U;
            if constexpr (phase2) {
                ++batch_terminal_success;
            }
            continue;
        }
        uint32_t empty_mask = bc_zero_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const uint32_t cell = bc_solve_pop_lowest_set_bit_index(empty_mask);
            const uint16_t ref = static_cast<uint16_t>((board_slot << 4U) | cell);
            ++workspace.empty_counts[board_slot];
            if constexpr (phase2) {
                ++batch_empty_slots;
                batch_spawned_boards += 2U;
            }
            const uint64_t spawned = set_board_tile_unchecked(board, cell, spawn_rank);
            if (fast_unfiltered_both) {
                ++batch_move_all_dir_calls;
                const auto moved_h = BoardMover::move_horizontal_pair(spawned);
                const auto moved_v = BoardMover::move_vertical_pair(spawned);
                push_moved_unfiltered(spawned, moved_h.first, ref);
                push_moved_unfiltered(spawned, moved_h.second, ref);
                push_moved_unfiltered(spawned, moved_v.first, ref);
                push_moved_unfiltered(spawned, moved_v.second, ref);
                continue;
            }
            if (options.solve.directions == BCDirectionMask::Both) {
                ++batch_move_all_dir_calls;
                const auto moved_h = BoardMover::move_horizontal_pair(spawned);
                const auto moved_v = BoardMover::move_vertical_pair(spawned);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_h.first, ref,
                    BCDirectionMask::Horizontal, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_h.second, ref,
                    BCDirectionMask::Horizontal, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_v.first, ref,
                    BCDirectionMask::Vertical, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                bc_single_chunk_push_candidate<StorageT>(
                    lut, future_axis, filter, spawned, moved_v.second, ref,
                    BCDirectionMask::Vertical, options.solve,
                    workspace.canonical2_boards, workspace.canonical2_refs,
                    batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
            } else {
                if (bc_has_horizontal(options.solve.directions)) {
                    batch_selective_move_calls += 2U;
                    const auto moved = BoardMover::move_horizontal_pair(spawned);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Horizontal, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Horizontal, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                }
                if (bc_has_vertical(options.solve.directions)) {
                    batch_selective_move_calls += 2U;
                    const auto moved = BoardMover::move_vertical_pair(spawned);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.first, ref,
                        BCDirectionMask::Vertical, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                    bc_single_chunk_push_candidate<StorageT>(
                        lut, future_axis, filter, spawned, moved.second, ref,
                        BCDirectionMask::Vertical, options.solve,
                        workspace.canonical2_boards, workspace.canonical2_refs,
                        batch_unchanged_moves, batch_prefilter_checks, batch_prefilter_skips);
                }
            }
        }
    }

    if (!workspace.canonical2_boards.empty()) {
        ++stats.edge.canonical_flushes;
        CanonicalBatch::canonicalize_inplace(
            workspace.canonical2_boards.data(),
            workspace.canonical2_boards.size(),
            options.solve.edge_options.canonical_symm_mode
        );
        stats.edge.canonicalized_candidates += workspace.canonical2_boards.size();
        const BCSolvePreparedQueryEncoder encoder(
            lut,
            future_axis,
            options.solve.edge_options.future_cell_modulus
        );
        for (size_t i = 0U; i < workspace.canonical2_boards.size(); ++i) {
            BCSolvePreparedQuery query;
            const bool encoded = encoder.encode(
                unpack_board_to_quadrants(workspace.canonical2_boards[i]),
                workspace.canonical2_refs[i],
                spawn_rank,
                BCDirectionMask::Both,
                query
            );
            if (!encoded) {
                ++stats.edge.encode_rejects;
                continue;
            }
            workspace.queries2.push_back(query);
            ++stats.edge.encoded_queries;
        }
        workspace.canonical2_boards.clear();
        workspace.canonical2_refs.clear();
    }
    if constexpr (phase2) {
        stats.edge.source_boards += count;
        stats.edge.terminal_success_boards += batch_terminal_success;
        stats.edge.empty_slots += batch_empty_slots;
        stats.edge.spawned_boards += batch_spawned_boards;
    }
    stats.edge.move_all_dir_calls += batch_move_all_dir_calls;
    stats.edge.selective_move_calls += batch_selective_move_calls;
    stats.edge.unchanged_moves += batch_unchanged_moves;
    stats.edge.prefilter_checks += batch_prefilter_checks;
    stats.edge.prefilter_skips += batch_prefilter_skips;

    for (uint32_t lane = 0U; lane < options.solve.row_width; ++lane) {
        std::fill_n(
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            options.solve.zero_value
        );
        (void)future_lookup.reduce_max_queries(
            workspace.queries2,
            workspace.best2.data(),
            static_cast<size_t>(count) * kBCBoardCellCount,
            lane,
            &stats.edge,
            true
        );
        if constexpr (phase2) {
            stats.edge.finalized_boards += lane == 0U ? count : 0U;
        }
        for (uint32_t board_slot = 0U; board_slot < count; ++board_slot) {
            const uint64_t row_index = workspace.output_indices[board_slot];
            const uint64_t value_index =
                row_index * static_cast<uint64_t>(options.solve.row_width) + lane;
            if (value_index >= values.size()) {
                throw std::out_of_range("BC single chunk weighted output index out of range");
            }
            if (workspace.terminal[board_slot] != 0U) {
                values[static_cast<size_t>(value_index)] =
                    phase2 ? options.solve.terminal_value : options.solve.zero_value;
                continue;
            }
            if (workspace.empty_counts[board_slot] == 0U) {
                values[static_cast<size_t>(value_index)] = options.solve.zero_value;
                continue;
            }
            StorageT contribution = options.solve.zero_value;
            if constexpr (std::is_same_v<StorageT, uint32_t>) {
                uint64_t integer_sum = 0U;
                uint32_t mask = workspace.empty_masks[board_slot];
                while (mask != 0U) {
                    const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                    const size_t index = static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                    integer_sum += static_cast<uint64_t>(workspace.best2[index]);
                }
                if (options.solve.edge_options.spawn_rate4 == 0.1) {
                    const uint64_t numerator = !phase2
                        ? integer_sum
                        : 9ULL * integer_sum;
                    const uint64_t denominator =
                        10ULL * static_cast<uint64_t>(workspace.empty_counts[board_slot]);
                    contribution = static_cast<uint32_t>(numerator / denominator);
                } else {
                    contribution = bc_single_chunk_scale_average_contribution<StorageT>(
                        static_cast<long double>(integer_sum),
                        workspace.empty_counts[board_slot],
                        options.solve.edge_options.spawn_rate4,
                        !phase2,
                        options.solve.zero_value
                    );
                }
            } else {
                long double sum = 0.0L;
                uint32_t mask = workspace.empty_masks[board_slot];
                while (mask != 0U) {
                    const uint32_t cell = bc_solve_pop_lowest_set_bit_index(mask);
                    const size_t index = static_cast<size_t>(board_slot) * kBCBoardCellCount + cell;
                    sum += static_cast<long double>(workspace.best2[index]);
                }
                contribution =
                    bc_single_chunk_scale_average_contribution<StorageT>(
                        sum,
                        workspace.empty_counts[board_slot],
                        options.solve.edge_options.spawn_rate4,
                        !phase2,
                        options.solve.zero_value
                    );
            }
            if constexpr (phase2) {
                if (contribution == options.solve.zero_value) {
                    continue;
                }
                values[static_cast<size_t>(value_index)] =
                    bc_single_chunk_add_success_contribution<StorageT>(
                        values[static_cast<size_t>(value_index)],
                        contribution
                    );
            } else {
                values[static_cast<size_t>(value_index)] = contribution;
            }
        }
    }
    workspace.clear_batch();
}

template <typename StorageT>
void bc_single_chunk_solve_weighted_phase_for_current_cells(
    const std::vector<BCLoadedCell> &current_cells,
    const std::vector<BCSingleChunkLoadedWorkItem> &work_items,
    const std::vector<uint64_t> &cell_offsets,
    BCSingleChunkValueBuffer<StorageT> &values,
    const BCLut &lut,
    const BCFamilyTable &future_axis,
    const BCFutureSuccessLookupView<StorageT> &future_lookup,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolvePhase phase,
    BCSingleChunkSolveStats &stats,
    std::vector<BCResidentBatchWorkspace<StorageT>> *batch_workspaces = nullptr
) {
    if (cell_offsets.size() != current_cells.size() + 1U) {
        throw std::invalid_argument("BC single chunk weighted current cell offsets mismatch");
    }
    const int threads = bc_resident_solve_effective_threads(options.solve.num_threads);
    std::vector<BCSingleChunkSolveStats> per_thread(static_cast<size_t>(threads));
    std::vector<BCResidentBatchWorkspace<StorageT>> local_workspaces;
    if (batch_workspaces == nullptr) {
        local_workspaces.resize(static_cast<size_t>(threads));
        batch_workspaces = &local_workspaces;
    } else if (batch_workspaces->size() < static_cast<size_t>(threads)) {
        batch_workspaces->resize(static_cast<size_t>(threads));
    }
    const double t0 = bc_single_chunk_now_seconds();
#pragma omp parallel num_threads(threads)
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        BCSingleChunkSolveStats &thread_stats = per_thread[static_cast<size_t>(tid)];
        BCResidentBatchWorkspace<StorageT> &workspace =
            (*batch_workspaces)[static_cast<size_t>(tid)];
        workspace.clear_batch();
        auto flush = [&]() {
            if (phase == BCSingleChunkSolvePhase::Spawn2) {
                bc_single_chunk_solve_weighted_phase_batch<
                    StorageT,
                    BCSingleChunkSolvePhase::Spawn2>(
                    workspace,
                    values,
                    lut,
                    future_axis,
                    future_lookup,
                    options,
                    thread_stats
                );
            } else {
                bc_single_chunk_solve_weighted_phase_batch<
                    StorageT,
                    BCSingleChunkSolvePhase::Spawn4>(
                    workspace,
                    values,
                    lut,
                    future_axis,
                    future_lookup,
                    options,
                    thread_stats
                );
            }
        };
#pragma omp for schedule(dynamic, 4)
        for (int64_t item_i_signed = 0;
             item_i_signed < static_cast<int64_t>(work_items.size());
             ++item_i_signed) {
            const BCSingleChunkLoadedWorkItem &item =
                work_items[static_cast<size_t>(item_i_signed)];
            const size_t cell_i = item.cell_index;
            const BCLoadedCell &cell = current_cells[cell_i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                continue;
            }
            const uint64_t cell_base = cell_offsets[cell_i];
            BCLoadedCellScanner scanner(lut, cell.view());
            for (uint32_t bucket_i = item.bucket_begin; bucket_i < item.bucket_end; ++bucket_i) {
                const bool ranged_bucket =
                    item.bucket_end == item.bucket_begin + 1U && item.word_end != 0U;
                scanner.for_each_bucket_word_range_board(
                    bucket_i,
                    ranged_bucket ? item.word_begin : 0U,
                    ranged_bucket ? item.word_end : 0U,
                    [&](const BCScannedBoardEntry &entry) {
                        workspace.boards[workspace.count] = entry.board;
                        workspace.output_indices[workspace.count] = cell_base + entry.local_success_row;
                        ++workspace.count;
                        if (phase == BCSingleChunkSolvePhase::Spawn2) {
                            ++thread_stats.current_rows;
                            ++thread_stats.current_boards;
                        }
                        if (workspace.count == BCResidentBatchWorkspace<StorageT>::kBatchSize) {
                            flush();
                        }
                    }
                );
            }
        }
        flush();
    }
    stats.recalc_seconds += bc_single_chunk_now_seconds() - t0;
    for (const BCSingleChunkSolveStats &thread_stats : per_thread) {
        stats.current_rows += thread_stats.current_rows;
        stats.current_boards += thread_stats.current_boards;
        bc_resident_solve_accumulate_edge_stats(stats.edge, thread_stats.edge);
    }
}

template <typename StorageT>
void bc_single_chunk_compact_loaded_cell(
    const BCLut &lut,
    const BCLoadedCell &cell,
    const BCSingleChunkValueBuffer<StorageT> &raw_values,
    uint64_t cell_value_offset,
    uint32_t row_width,
    StorageT zero_value,
    FinalizedCellPayload &payload,
    std::vector<StorageT> &cell_success_values,
    BCResidentCompactStats &stats
) {
    if (cell.success_rows == 0U || cell.buckets.empty()) {
        ++stats.empty_cells;
        return;
    }
    stats.input_rows += cell.success_rows;
    const BCLoadedCellView view = cell.view();
    uint64_t success_cursor = 0U;
    payload.buckets.reserve(view.buckets.size);
    payload.rank_payload.reserve(view.rank_payload.size);
    for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = view.buckets.data[bucket_i];
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (bitmap_end > view.rank_payload.size) {
            throw std::out_of_range("BC single chunk compact source bitmap exceeds rank payload");
        }
        std::vector<uint64_t> keep_bitmap(word_count, 0U);
        uint32_t bucket_seen = 0U;
        uint32_t bucket_kept = 0U;
        const uint8_t *bitmap_words = view.rank_payload.data + bitmap_offset;
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            uint64_t word = load_u64_le(bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t));
            if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            while (word != 0U) {
                const uint32_t bit = bc_resident_countr_zero64(word);
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (local_row >= cell.success_rows) {
                    throw std::out_of_range("BC single chunk compact source row exceeds descriptor");
                }
                const uint64_t row_base =
                    (cell_value_offset + local_row) * static_cast<uint64_t>(row_width);
                if (row_base + row_width > raw_values.size()) {
                    throw std::out_of_range("BC single chunk compact source row exceeds raw values");
                }
                bool keep_row = false;
                for (uint32_t lane = 0U; lane < row_width; ++lane) {
                    if (raw_values[static_cast<size_t>(row_base + lane)] != zero_value) {
                        keep_row = true;
                        break;
                    }
                }
                if (keep_row) {
                    keep_bitmap[word_i] |= (1ULL << bit);
                    for (uint32_t lane = 0U; lane < row_width; ++lane) {
                        cell_success_values.push_back(raw_values[static_cast<size_t>(row_base + lane)]);
                    }
                    ++bucket_kept;
                    ++stats.live_rows;
                } else {
                    ++stats.zero_pruned_rows;
                }
                ++bucket_seen;
                word &= word - 1ULL;
            }
        }
        if (bucket_kept == 0U) {
            continue;
        }
        const uint32_t payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t aligned_payload_offset = align_up_u32(payload_offset, 8U);
        bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_offset);
        const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        bc_resident_append_prefix256_le(payload.rank_payload, keep_bitmap.data(), word_count, bitmap_len);
        const uint32_t out_bitmap_offset = bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        bc_resident_append_padding(
            payload.rank_payload,
            out_bitmap_offset - static_cast<uint32_t>(payload.rank_payload.size())
        );
        bc_resident_append_bitmap_le(payload.rank_payload, keep_bitmap.data(), word_count);
        if (success_cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("BC single chunk compact success row offset exceeds uint32");
        }
        payload.buckets.push_back(BCBucketEntry{
            bucket.key,
            rank_payload_offset,
            static_cast<uint32_t>(success_cursor)
        });
        success_cursor += bucket_kept;
    }
    if (success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC single chunk compact success rows exceed uint32");
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    if (payload.success_rows == 0U) {
        payload.buckets.clear();
        payload.rank_payload.clear();
        ++stats.empty_cells;
    } else {
        ++stats.live_cells;
    }
}

template <typename StorageT>
void bc_single_chunk_compact_loaded_cell_in_place(
    const BCLut &lut,
    const BCLoadedCell &cell,
    BCSingleChunkValueBuffer<StorageT> &raw_values,
    uint64_t cell_value_offset,
    uint32_t row_width,
    StorageT zero_value,
    FinalizedCellPayload &payload,
    BCResidentCompactStats &stats
) {
    if (cell.success_rows == 0U || cell.buckets.empty()) {
        ++stats.empty_cells;
        return;
    }
    stats.input_rows += cell.success_rows;
    const BCLoadedCellView view = cell.view();
    payload.buckets.reserve(view.buckets.size);
    payload.rank_payload.reserve(view.rank_payload.size);
    uint64_t success_cursor = 0U;
    for (uint32_t bucket_i = 0U; bucket_i < view.buckets.size; ++bucket_i) {
        const BCBucketEntry &bucket = view.buckets.data[bucket_i];
        const uint32_t bitmap_len = bitmap_len_from_trusted_key(lut, bucket.key);
        const uint32_t word_count = words_for_bits(bitmap_len);
        const uint32_t bitmap_offset = bc_rank_payload_bitmap_offset(
            bucket.rank_payload_offset,
            bitmap_len
        );
        const uint64_t bitmap_end =
            static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (bitmap_end > view.rank_payload.size) {
            throw std::out_of_range(
                "BC single chunk in-place compact source bitmap exceeds rank payload"
            );
        }
        const uint64_t bucket_success_offset = success_cursor;
        const uint32_t payload_start = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t aligned_payload_offset = align_up_u32(payload_start, 8U);
        bc_resident_append_padding(payload.rank_payload, aligned_payload_offset - payload_start);
        const uint32_t rank_payload_offset = static_cast<uint32_t>(payload.rank_payload.size());
        const uint32_t out_bitmap_offset =
            bc_rank_payload_bitmap_offset(rank_payload_offset, bitmap_len);
        const uint64_t payload_end =
            static_cast<uint64_t>(out_bitmap_offset) +
            static_cast<uint64_t>(word_count) * sizeof(uint64_t);
        if (payload_end > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("BC single chunk compact rank payload exceeds uint32");
        }
        payload.rank_payload.resize(static_cast<size_t>(payload_end), 0U);

        uint32_t bucket_seen = 0U;
        uint32_t bucket_kept = 0U;
        uint32_t prefix_running = 0U;
        const uint8_t *bitmap_words = view.rank_payload.data + bitmap_offset;
        for (uint32_t word_i = 0U; word_i < word_count; ++word_i) {
            if ((word_i & 3U) == 0U) {
                if (prefix_running > std::numeric_limits<RankPrefix>::max()) {
                    throw std::logic_error("BC single chunk compact prefix exceeds uint16");
                }
                bc_resident_store_u16_le(
                    payload.rank_payload.data() +
                        static_cast<size_t>(rank_payload_offset) +
                        static_cast<size_t>(word_i / 4U) * sizeof(RankPrefix),
                    static_cast<RankPrefix>(prefix_running)
                );
            }
            uint64_t word = load_u64_le(
                bitmap_words + static_cast<size_t>(word_i) * sizeof(uint64_t)
            );
            if (word_i + 1U == word_count && (bitmap_len & 63U) != 0U) {
                word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
            }
            uint64_t keep_word = 0U;
            while (word != 0U) {
                const uint32_t bit = bc_resident_countr_zero64(word);
                const uint64_t local_row =
                    static_cast<uint64_t>(bucket.success_row_offset) + bucket_seen;
                if (local_row >= cell.success_rows) {
                    throw std::out_of_range(
                        "BC single chunk compact source row exceeds descriptor"
                    );
                }
                const uint64_t row_base =
                    (cell_value_offset + local_row) * static_cast<uint64_t>(row_width);
                if (row_base + row_width > raw_values.size()) {
                    throw std::out_of_range(
                        "BC single chunk compact source row exceeds raw values"
                    );
                }
                bool keep_row = false;
                for (uint32_t lane = 0U; lane < row_width; ++lane) {
                    if (raw_values[static_cast<size_t>(row_base + lane)] != zero_value) {
                        keep_row = true;
                        break;
                    }
                }
                if (keep_row) {
                    keep_word |= (1ULL << bit);
                    const uint64_t dst_base =
                        (cell_value_offset + success_cursor) *
                        static_cast<uint64_t>(row_width);
                    if (dst_base + row_width > raw_values.size()) {
                        throw std::out_of_range(
                            "BC single chunk compact destination exceeds raw values"
                        );
                    }
                    if (dst_base != row_base) {
                        for (uint32_t lane = 0U; lane < row_width; ++lane) {
                            raw_values[static_cast<size_t>(dst_base + lane)] =
                                raw_values[static_cast<size_t>(row_base + lane)];
                        }
                    }
                    ++success_cursor;
                    ++bucket_kept;
                    ++stats.live_rows;
                } else {
                    ++stats.zero_pruned_rows;
                }
                ++bucket_seen;
                word &= word - 1ULL;
            }
            bc_resident_store_u64_le(
                payload.rank_payload.data() +
                    static_cast<size_t>(out_bitmap_offset) +
                    static_cast<size_t>(word_i) * sizeof(uint64_t),
                keep_word
            );
            prefix_running += popcount64(keep_word);
        }
        if (bucket_kept == 0U) {
            payload.rank_payload.resize(payload_start);
            continue;
        }
        if (bucket_success_offset > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "BC single chunk compact success row offset exceeds uint32"
            );
        }
        payload.buckets.push_back(BCBucketEntry{
            bucket.key,
            rank_payload_offset,
            static_cast<uint32_t>(bucket_success_offset)
        });
    }
    if (success_cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("BC single chunk compact success rows exceed uint32");
    }
    payload.success_rows = static_cast<uint32_t>(success_cursor);
    if (payload.success_rows == 0U) {
        payload.buckets.clear();
        payload.rank_payload.clear();
        ++stats.empty_cells;
    } else {
        ++stats.live_cells;
    }
}

template <typename StorageT>
void bc_single_chunk_write_position_bytes(
    BCWritableFile &file,
    const std::vector<uint8_t> &bytes,
    BCFileIOStats *stats
) {
    file.prepare_full_overwrite(bytes.size());
    BCSequentialSuccessWriteStager stager(file, stats);
    if (!bytes.empty()) {
        stager.append(bytes.data(), bytes.size());
    }
    stager.finish();
}

inline void bc_single_chunk_write_exact(std::ostream &out, const void *data, size_t bytes, const char *label) {
    if (bytes == 0U) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument(std::string(label) + " data pointer is null");
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error(std::string(label) + " write failed");
    }
}

inline void bc_single_chunk_read_exact(std::istream &in, void *data, size_t bytes, const char *label) {
    if (bytes == 0U) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument(std::string(label) + " data pointer is null");
    }
    in.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    if (!in) {
        throw std::runtime_error(std::string(label) + " read failed");
    }
}

inline void bc_single_chunk_write_u32(std::ostream &out, uint32_t value, const char *label) {
    std::array<uint8_t, 4U> bytes{
        static_cast<uint8_t>(value & 0xFFU),
        static_cast<uint8_t>((value >> 8U) & 0xFFU),
        static_cast<uint8_t>((value >> 16U) & 0xFFU),
        static_cast<uint8_t>((value >> 24U) & 0xFFU)
    };
    bc_single_chunk_write_exact(out, bytes.data(), bytes.size(), label);
}

inline void bc_single_chunk_write_u64(std::ostream &out, uint64_t value, const char *label) {
    std::array<uint8_t, 8U> bytes{
        static_cast<uint8_t>(value & 0xFFU),
        static_cast<uint8_t>((value >> 8U) & 0xFFU),
        static_cast<uint8_t>((value >> 16U) & 0xFFU),
        static_cast<uint8_t>((value >> 24U) & 0xFFU),
        static_cast<uint8_t>((value >> 32U) & 0xFFU),
        static_cast<uint8_t>((value >> 40U) & 0xFFU),
        static_cast<uint8_t>((value >> 48U) & 0xFFU),
        static_cast<uint8_t>((value >> 56U) & 0xFFU)
    };
    bc_single_chunk_write_exact(out, bytes.data(), bytes.size(), label);
}

[[nodiscard]] inline uint32_t bc_single_chunk_read_u32(std::istream &in, const char *label) {
    std::array<uint8_t, 4U> bytes{};
    bc_single_chunk_read_exact(in, bytes.data(), bytes.size(), label);
    return bc_load_u32_le(bytes.data());
}

[[nodiscard]] inline uint64_t bc_single_chunk_read_u64(std::istream &in, const char *label) {
    std::array<uint8_t, 8U> bytes{};
    bc_single_chunk_read_exact(in, bytes.data(), bytes.size(), label);
    return load_u64_le(bytes.data());
}

inline void bc_single_chunk_skip_bytes(std::istream &in, uint64_t bytes, const char *label) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::overflow_error(std::string(label) + " skip exceeds streamoff");
    }
    in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!in) {
        throw std::runtime_error(std::string(label) + " skip failed");
    }
}

inline constexpr uint64_t kBCSingleChunkPartialMagic = 0x3150545241484342ULL; // "BCHARTP1".
inline constexpr uint32_t kBCSingleChunkTempVersion = 1U;

[[nodiscard]] inline std::filesystem::path bc_single_chunk_partial_path(
    const std::filesystem::path &dir,
    uint32_t chunk_index
) {
    return dir / ("partial4_" + std::to_string(chunk_index) + ".bcpart");
}

template <typename SumT>
void bc_single_chunk_write_partial4(
    const std::filesystem::path &path,
    const BCSingleChunkValueBuffer<SumT> &values
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("BC single chunk failed to open partial file: " + path.string());
    }
    bc_single_chunk_write_u64(out, kBCSingleChunkPartialMagic, "BC single chunk partial magic");
    bc_single_chunk_write_u32(out, kBCSingleChunkTempVersion, "BC single chunk partial version");
    bc_single_chunk_write_u32(out, static_cast<uint32_t>(sizeof(SumT)), "BC single chunk partial value size");
    bc_single_chunk_write_u64(out, static_cast<uint64_t>(values.size()), "BC single chunk partial value count");
    bc_single_chunk_write_exact(
        out,
        values.data(),
        values.size() * sizeof(SumT),
        "BC single chunk partial payload"
    );
}

template <typename SumT>
void bc_single_chunk_read_partial4(
    const std::filesystem::path &path,
    BCSingleChunkValueBuffer<SumT> &values,
    size_t expected_count
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("BC single chunk failed to open partial file: " + path.string());
    }
    const uint64_t magic = bc_single_chunk_read_u64(in, "BC single chunk partial magic");
    const uint32_t version = bc_single_chunk_read_u32(in, "BC single chunk partial version");
    const uint32_t value_size = bc_single_chunk_read_u32(in, "BC single chunk partial value size");
    const uint64_t value_count = bc_single_chunk_read_u64(in, "BC single chunk partial value count");
    if (magic != kBCSingleChunkPartialMagic ||
        version != kBCSingleChunkTempVersion ||
        value_size != sizeof(SumT) ||
        value_count != static_cast<uint64_t>(expected_count)) {
        throw std::runtime_error("BC single chunk partial header mismatch");
    }
    values.resize_uninitialized(expected_count);
    bc_single_chunk_read_exact(
        in,
        values.data(),
        expected_count * sizeof(SumT),
        "BC single chunk partial payload"
    );
}

inline constexpr uint64_t kBCSingleChunkPartialHeaderBytes = 24U;
inline constexpr uint64_t kBCSingleChunkTmp4PayloadOffset = 4096U;
inline constexpr uint64_t kBCSingleChunkDirectBlockBytes = 4096U;
inline constexpr uint64_t kBCSingleChunkDirectPayloadRequestBytes = 64ULL * 1024ULL * 1024ULL;

template <typename SumT>
[[nodiscard]] uint64_t bc_single_chunk_tmp4_file_bytes(
    const BCSingleChunkValueBuffer<SumT> &values,
    bool direct_io
) {
    const uint64_t payload_bytes = values.data_bytes();
    return bc_checked_add_u64(
        kBCSingleChunkTmp4PayloadOffset,
        direct_io ? values.aligned_data_bytes() : payload_bytes,
        "BC single chunk tmp4 byte count overflow"
    );
}

template <typename SumT>
void bc_single_chunk_write_tmp4_values(
    const std::filesystem::path &path,
    BCSingleChunkValueBuffer<SumT> &values,
    bool direct_io,
    BCFileIOStats *stats = nullptr
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::vector<uint8_t> header;
    header.reserve(static_cast<size_t>(kBCSingleChunkPartialHeaderBytes));
    bc_append_u64_le(header, kBCSingleChunkPartialMagic);
    bc_append_u32_le(header, kBCSingleChunkTempVersion);
    bc_append_u32_le(header, static_cast<uint32_t>(sizeof(SumT)));
    bc_append_u64_le(header, static_cast<uint64_t>(values.size()));
    if (header.size() != kBCSingleChunkPartialHeaderBytes) {
        throw std::logic_error("BC single chunk tmp4 header size mismatch");
    }
    const uint64_t payload_bytes = values.data_bytes();
    const uint64_t direct_payload_bytes =
        payload_bytes == 0U ? 0U : values.aligned_data_bytes();
    const uint64_t logical_size = bc_single_chunk_tmp4_file_bytes(values, direct_io);
    if (direct_io) {
        BCDirectFileIOOptions direct_options;
        direct_options.queue_depth = 16U;
        direct_options.overlapped = true;
        BCDirectFileWriter file(path, direct_options);
        file.prepare_full_overwrite(logical_size);
        detail::BCAlignedBuffer header_block(kBCSingleChunkTmp4PayloadOffset, direct_options.alignment);
        std::memset(header_block.data(), 0, header_block.size());
        std::memcpy(header_block.data(), header.data(), header.size());
        values.zero_padding(direct_options.alignment);
        std::vector<BCFileWriteRequest> requests;
        const uint64_t payload_request_count = direct_payload_bytes == 0U
            ? 0U
            : (direct_payload_bytes + kBCSingleChunkDirectPayloadRequestBytes - 1U) /
                kBCSingleChunkDirectPayloadRequestBytes;
        if (payload_request_count >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 1U)) {
            throw std::overflow_error("BC single chunk tmp4 write request count exceeds size_t");
        }
        requests.reserve(static_cast<size_t>(1U + payload_request_count));
        requests.push_back(BCFileWriteRequest{
            0U,
            header_block.data(),
            kBCSingleChunkTmp4PayloadOffset
        });
        const uint8_t *payload_data = reinterpret_cast<const uint8_t *>(values.data());
        for (uint64_t offset = 0U; offset < direct_payload_bytes;) {
            const uint64_t take = std::min<uint64_t>(
                kBCSingleChunkDirectPayloadRequestBytes,
                direct_payload_bytes - offset
            );
            requests.push_back(BCFileWriteRequest{
                kBCSingleChunkTmp4PayloadOffset + offset,
                payload_data + static_cast<std::ptrdiff_t>(offset),
                take
            });
            offset += take;
        }
        BCFileIOStats local_stats;
        file.write_many(requests, &local_stats);
        bc_success_accumulate_file_stats(stats, local_stats);
        return;
    }
    BCBufferedFileWriter file(path);
    file.prepare_full_overwrite(logical_size);
    BCFileIOStats local_stats;
    BCSequentialSuccessWriteStager stager(file, &local_stats);
    std::vector<uint8_t> header_block(static_cast<size_t>(kBCSingleChunkTmp4PayloadOffset), 0U);
    std::memcpy(header_block.data(), header.data(), header.size());
    stager.append(header_block.data(), header_block.size());
    stager.append(values.data(), payload_bytes);
    stager.finish();
    bc_success_accumulate_file_stats(stats, local_stats);
}

template <typename SumT>
void bc_single_chunk_read_tmp4_values(
    const std::filesystem::path &path,
    BCSingleChunkValueBuffer<SumT> &values,
    size_t expected_count,
    bool direct_io,
    BCFileIOStats *stats = nullptr
) {
    std::array<uint8_t, static_cast<size_t>(kBCSingleChunkPartialHeaderBytes)> header{};
    if (direct_io) {
        BCDirectFileIOOptions direct_options;
        direct_options.queue_depth = 16U;
        direct_options.overlapped = true;
        BCDirectFileReader file(path, direct_options);
        detail::BCAlignedBuffer header_block(kBCSingleChunkTmp4PayloadOffset, direct_options.alignment);
        BCFileIOStats header_stats;
        file.read_many(
            std::vector<BCFileReadRequest>{
                BCFileReadRequest{0U, header_block.data(), header_block.size()}
            },
            &header_stats
        );
        bc_success_accumulate_file_stats(stats, header_stats);
        std::memcpy(header.data(), header_block.data(), header.size());
        const uint64_t magic = load_u64_le(header.data());
        const uint32_t version = bc_load_u32_le(header.data() + 8U);
        const uint32_t value_size = bc_load_u32_le(header.data() + 12U);
        const uint64_t value_count = load_u64_le(header.data() + 16U);
        if (magic != kBCSingleChunkPartialMagic ||
            version != kBCSingleChunkTempVersion ||
            value_size != sizeof(SumT) ||
            value_count != static_cast<uint64_t>(expected_count)) {
            throw std::runtime_error("BC single chunk tmp4 header mismatch");
        }
        values.resize_uninitialized(expected_count);
        const uint64_t payload_bytes = static_cast<uint64_t>(expected_count) * sizeof(SumT);
        if (payload_bytes != 0U) {
            const uint64_t direct_payload_bytes = values.aligned_data_bytes();
            const uint64_t payload_request_count =
                (direct_payload_bytes + kBCSingleChunkDirectPayloadRequestBytes - 1U) /
                kBCSingleChunkDirectPayloadRequestBytes;
            if (payload_request_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC single chunk tmp4 read request count exceeds size_t");
            }
            std::vector<BCFileReadRequest> requests;
            requests.reserve(static_cast<size_t>(payload_request_count));
            uint8_t *payload_data = reinterpret_cast<uint8_t *>(values.data());
            for (uint64_t offset = 0U; offset < direct_payload_bytes;) {
                const uint64_t take = std::min<uint64_t>(
                    kBCSingleChunkDirectPayloadRequestBytes,
                    direct_payload_bytes - offset
                );
                requests.push_back(BCFileReadRequest{
                    kBCSingleChunkTmp4PayloadOffset + offset,
                    payload_data + static_cast<std::ptrdiff_t>(offset),
                    take
                });
                offset += take;
            }
            BCFileIOStats payload_stats;
            file.read_many(requests, &payload_stats);
            bc_success_accumulate_file_stats(stats, payload_stats);
        }
        return;
    } else {
        BCBufferedFileReader file(path);
        BCFileIOStats header_stats;
        file.read_many(
            std::vector<BCFileReadRequest>{
                BCFileReadRequest{0U, header.data(), header.size()}
            },
            &header_stats
        );
        bc_success_accumulate_file_stats(stats, header_stats);
        const uint64_t magic = load_u64_le(header.data());
        const uint32_t version = bc_load_u32_le(header.data() + 8U);
        const uint32_t value_size = bc_load_u32_le(header.data() + 12U);
        const uint64_t value_count = load_u64_le(header.data() + 16U);
        if (magic != kBCSingleChunkPartialMagic ||
            version != kBCSingleChunkTempVersion ||
            value_size != sizeof(SumT) ||
            value_count != static_cast<uint64_t>(expected_count)) {
            throw std::runtime_error("BC single chunk tmp4 header mismatch");
        }
        values.resize_uninitialized(expected_count);
        const uint64_t payload_bytes = static_cast<uint64_t>(expected_count) * sizeof(SumT);
        if (payload_bytes != 0U) {
            BCFileIOStats payload_stats;
            file.read_many(
                std::vector<BCFileReadRequest>{
                    BCFileReadRequest{kBCSingleChunkTmp4PayloadOffset, values.data(), payload_bytes}
                },
                &payload_stats
            );
            bc_success_accumulate_file_stats(stats, payload_stats);
        }
    }
}

[[nodiscard]] inline uint64_t bc_single_chunk_position_fingerprint(
    const BCFamilyTable &axis,
    const BCPositionHeader &header
) {
    uint64_t value = 0x9E3779B97F4A7C15ULL;
    auto mix = [&value](uint64_t x) {
        value ^= x + 0x9E3779B97F4A7C15ULL + (value << 6U) + (value >> 2U);
    };
    mix(header.key_mode);
    mix(header.family_unit);
    mix(header.axis_base_coord);
    mix(header.family_count);
    mix(header.axis_coord_table_bytes);
    mix(header.layer_sum);
    mix(header.descriptor_count);
    for (FamilyCoord coord : axis.coords()) {
        mix(coord);
    }
    return value;
}

inline void bc_single_chunk_write_file_extent(
    BCWritableFile &file,
    uint64_t offset,
    const void *data,
    uint64_t bytes,
    BCFileIOStats *stats
) {
    if (bytes == 0U) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("BC single chunk write extent pointer is null");
    }
    BCFileIOStats local;
    file.write_many(
        std::vector<BCFileWriteRequest>{BCFileWriteRequest{offset, data, bytes}},
        stats == nullptr ? nullptr : &local
    );
    bc_success_accumulate_file_stats(stats, local);
}

inline void bc_single_chunk_flush_for_stream_finish(BCWritableFile &file) {
    if (file.mode() != BCFileIOMode::Direct) {
        file.flush();
    }
}

template <typename StorageT>
class BCSingleChunkFinalFileStreamer {
public:
    BCSingleChunkFinalFileStreamer(
        const BCPositionStreamingReader &current_position,
        BCWritableFile &position_file,
        BCWritableFile &success_file,
        uint32_t row_width,
        BCSuccessDTypeMode dtype,
        BCSingleChunkSolveStats &stats
    )
        : current_position_(current_position),
          position_file_(position_file),
          success_file_(success_file),
          row_width_(row_width),
          dtype_(dtype),
          value_size_(bc_success_dtype_value_size(dtype)),
          stats_(stats) {
        if (!bc_success_dtype_matches_type<StorageT>(dtype_)) {
            throw std::invalid_argument("BC single chunk final streamer dtype mismatch");
        }
        if (row_width_ == 0U) {
            throw std::invalid_argument("BC single chunk final streamer row_width must be non-zero");
        }
        descriptors_.assign(current_position_.cell_count(), BCPositionCellDescriptor{});
        written_.assign(current_position_.cell_count(), 0U);
        for (BCPositionCellDescriptor &descriptor : descriptors_) {
            descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
        }

        const BCPositionHeader &source_header = current_position_.header();
        axis_coord_bytes_ = bc_axis_coord_table_bytes(current_position_.axis().family_count());
        descriptor_offset_ = bc_checked_add_u64(
            kBCPositionHeaderBytes,
            axis_coord_bytes_,
            "BC single chunk final descriptor offset overflow"
        );
        descriptor_bytes_ =
            static_cast<uint64_t>(descriptors_.size()) * kBCPositionCellDescriptorBytes;
        bucket_offset_ = bc_checked_add_u64(
            descriptor_offset_,
            descriptor_bytes_,
            "BC single chunk final bucket offset overflow"
        );
        bucket_offset_ = bc_direct_align_up(bucket_offset_, kBCSingleChunkDirectBlockBytes);
        bucket_capacity_bytes_ = source_header.bucket_meta_bytes;
        rank_offset_ = bc_checked_add_u64(
            bucket_offset_,
            bucket_capacity_bytes_,
            "BC single chunk final rank offset overflow"
        );
        rank_offset_ = bc_direct_align_up(rank_offset_, kBCSingleChunkDirectBlockBytes);
        rank_capacity_bytes_ = source_header.rank_payload_bytes;
        const uint64_t position_upper = bc_checked_add_u64(
            rank_offset_,
            rank_capacity_bytes_,
            "BC single chunk final position upper size overflow"
        );
        const uint64_t success_payload_upper =
            bc_success_expected_payload_bytes_for(current_position_, row_width_, dtype_);
        uint64_t success_upper = bc_checked_add_u64(
            kBCSuccessHeaderBytes,
            success_payload_upper,
            "BC single chunk final success upper size overflow"
        );
        success_upper = std::max<uint64_t>(success_upper, kBCSingleChunkDirectBlockBytes);

        double t0 = bc_single_chunk_now_seconds();
        position_file_.prepare_full_overwrite(position_upper);
        const double position_prepare = bc_single_chunk_now_seconds() - t0;
        stats_.position_prepare_seconds += position_prepare;
        stats_.position_write_seconds += position_prepare;
        t0 = bc_single_chunk_now_seconds();
        success_file_.prepare_full_overwrite(success_upper);
        const double success_prepare = bc_single_chunk_now_seconds() - t0;
        stats_.success_prepare_seconds += success_prepare;
        stats_.success_write_seconds += success_prepare;
        bucket_stager_ = std::make_unique<BCSequentialSuccessWriteStager>(
            position_file_,
            &stats_.output_position_write,
            bucket_offset_
        );
        rank_stager_ = std::make_unique<BCSequentialSuccessWriteStager>(
            position_file_,
            &stats_.output_position_write,
            rank_offset_
        );
        success_tail_stager_ = std::make_unique<BCSequentialSuccessWriteStager>(
            success_file_,
            &stats_.output_success_write,
            kBCSingleChunkDirectBlockBytes
        );
        success_first_block_.reset(kBCSingleChunkDirectBlockBytes, kBCSingleChunkDirectBlockBytes);
        std::memset(success_first_block_.data(), 0, success_first_block_.size());
        success_stream_cursor_ = kBCSuccessHeaderBytes;
    }

    void write_chunk(
        const std::vector<BCLoadedCell> &current_cells,
        const std::vector<FinalizedCellPayload> &payloads,
        const std::vector<StorageT> &compact_values
    ) {
        const double assembly_t0 = bc_single_chunk_now_seconds();
        if (current_cells.size() != payloads.size()) {
            throw std::invalid_argument("BC single chunk final streamer payload count mismatch");
        }
        std::vector<uint8_t> bucket_bytes;
        std::vector<uint8_t> rank_bytes;
        uint64_t expected_values = 0U;
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.cid >= descriptors_.size()) {
                throw std::out_of_range("BC single chunk final streamer cell id out of range");
            }
            if (written_[static_cast<size_t>(cell.cid)] != 0U) {
                throw std::runtime_error("BC single chunk final streamer duplicate cell");
            }
            written_[static_cast<size_t>(cell.cid)] = 1U;
            const FinalizedCellPayload &payload = payloads[i];
            if (payload.buckets.empty()) {
                if (payload.success_rows != 0U || !payload.rank_payload.empty()) {
                    throw std::invalid_argument("BC single chunk final streamer empty payload mismatch");
                }
                BCPositionCellDescriptor descriptor;
                descriptor.flags_or_padding = kBCPositionCellFlagEmpty;
                descriptors_[static_cast<size_t>(cell.cid)] = descriptor;
                continue;
            }
            if (payload.buckets.size() > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC single chunk final streamer bucket count exceeds uint32");
            }
            if (payload.rank_payload.size() > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("BC single chunk final streamer rank payload exceeds uint32");
            }
            const uint64_t local_bucket_offset =
                bc_checked_add_u64(
                    bucket_cursor_,
                    bucket_bytes.size(),
                    "BC single chunk final streamer bucket cursor overflow"
                );
            const uint64_t local_rank_offset =
                bc_checked_add_u64(
                    rank_cursor_,
                    rank_bytes.size(),
                    "BC single chunk final streamer rank cursor overflow"
                );
            BCPositionCellDescriptor descriptor;
            descriptor.bucket_count = static_cast<uint32_t>(payload.buckets.size());
            descriptor.success_rows = payload.success_rows;
            descriptor.bucket_meta_offset = local_bucket_offset;
            descriptor.rank_payload_offset = local_rank_offset;
            descriptor.rank_payload_bytes = payload.rank_payload.size();
            descriptor.reserved0 = 0U;
            descriptor.flags_or_padding = 0U;
            descriptors_[static_cast<size_t>(cell.cid)] = descriptor;
            for (const BCBucketEntry &bucket : payload.buckets) {
                bc_append_bucket_entry(bucket_bytes, bucket);
            }
            rank_bytes.insert(
                rank_bytes.end(),
                payload.rank_payload.begin(),
                payload.rank_payload.end()
            );
            expected_values = bc_checked_add_u64(
                expected_values,
                static_cast<uint64_t>(payload.success_rows) * row_width_,
                "BC single chunk final streamer value count overflow"
            );
        }
        if (expected_values != static_cast<uint64_t>(compact_values.size())) {
            throw std::runtime_error("BC single chunk final streamer compact value count mismatch");
        }
        stats_.result_assembly_seconds += bc_single_chunk_now_seconds() - assembly_t0;

        append_position_payload(*bucket_stager_, bucket_bytes);
        append_position_payload(*rank_stager_, rank_bytes);
        bucket_cursor_ = bc_checked_add_u64(
            bucket_cursor_,
            bucket_bytes.size(),
            "BC single chunk final streamer bucket cursor overflow"
        );
        rank_cursor_ = bc_checked_add_u64(
            rank_cursor_,
            rank_bytes.size(),
            "BC single chunk final streamer rank cursor overflow"
        );
        write_success_values(compact_values);
        success_value_cursor_ = bc_checked_add_u64(
            success_value_cursor_,
            static_cast<uint64_t>(compact_values.size()),
            "BC single chunk final streamer success cursor overflow"
        );
    }

    [[nodiscard]] BCSingleChunkSolveFileResult finish() {
        double assembly_t0 = bc_single_chunk_now_seconds();
        for (uint8_t written : written_) {
            if (written == 0U) {
                throw std::runtime_error("BC single chunk final streamer missing cell");
            }
        }
        if (bucket_cursor_ > bucket_capacity_bytes_) {
            throw std::overflow_error("BC single chunk final bucket payload exceeds reserved output range");
        }
        if (rank_cursor_ > rank_capacity_bytes_) {
            throw std::overflow_error("BC single chunk final rank payload exceeds reserved output range");
        }
        BCPositionHeader position_header;
        position_header.family_unit = current_position_.axis().family_unit();
        position_header.axis_base_coord = current_position_.axis().axis_base_coord();
        position_header.family_count = current_position_.axis().family_count();
        position_header.layer_sum = current_position_.axis().layer_sum();
        position_header.axis_coord_table_bytes = axis_coord_bytes_;
        position_header.descriptor_count = descriptors_.size();
        position_header.descriptor_table_offset = descriptor_offset_;
        position_header.descriptor_table_bytes = descriptor_bytes_;
        position_header.bucket_meta_offset = bucket_offset_;
        position_header.bucket_meta_bytes = bucket_cursor_;
        position_header.rank_payload_offset = rank_offset_;
        position_header.rank_payload_bytes = rank_cursor_;

        std::vector<uint8_t> position_metadata;
        if (bucket_offset_ > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC single chunk final position metadata exceeds size_t");
        }
        position_metadata.reserve(static_cast<size_t>(bucket_offset_));
        bc_append_header(position_metadata, position_header);
        bc_append_axis_coord_table(position_metadata, current_position_.axis());
        for (const BCPositionCellDescriptor &descriptor : descriptors_) {
            bc_append_cell_descriptor(position_metadata, descriptor);
        }
        if (position_metadata.size() > bucket_offset_) {
            throw std::logic_error("BC single chunk final position metadata exceeds bucket offset");
        }
        position_metadata.resize(static_cast<size_t>(bucket_offset_), 0U);

        const uint64_t position_logical_size = bc_checked_add_u64(
            rank_offset_,
            rank_cursor_,
            "BC single chunk final position logical size overflow"
        );
        stats_.result_assembly_seconds += bc_single_chunk_now_seconds() - assembly_t0;
        double t0 = bc_single_chunk_now_seconds();
        bucket_stager_->finish();
        rank_stager_->finish();
        stats_.position_write_seconds += bc_single_chunk_now_seconds() - t0;
        t0 = bc_single_chunk_now_seconds();
        bc_single_chunk_write_file_extent(
            position_file_,
            0U,
            position_metadata.data(),
            position_metadata.size(),
            &stats_.output_position_write
        );
        position_file_.resize(position_logical_size);
        bc_single_chunk_flush_for_stream_finish(position_file_);
        stats_.position_write_seconds += bc_single_chunk_now_seconds() - t0;

        const uint64_t payload_bytes = checked_value_bytes(success_value_cursor_);
        BCSuccessHeader success_header;
        success_header.dtype = static_cast<uint32_t>(dtype_);
        success_header.row_width = row_width_;
        success_header.family_count = position_header.family_count;
        success_header.descriptor_count = position_header.descriptor_count;
        success_header.payload_offset = kBCSuccessHeaderBytes;
        success_header.payload_bytes = payload_bytes;
        success_header.position_key_mode = position_header.key_mode;
        success_header.family_unit = position_header.family_unit;
        success_header.axis_base_coord = position_header.axis_base_coord;
        success_header.layer_sum = position_header.layer_sum;
        success_header.position_metadata_fingerprint =
            bc_single_chunk_position_fingerprint(current_position_.axis(), position_header);
        std::vector<uint8_t> success_header_bytes;
        assembly_t0 = bc_single_chunk_now_seconds();
        success_header_bytes.reserve(kBCSuccessHeaderBytes);
        bc_append_success_header(success_header_bytes, success_header);
        const uint64_t success_logical_size = bc_checked_add_u64(
            kBCSuccessHeaderBytes,
            payload_bytes,
            "BC single chunk final success logical size overflow"
        );
        if (success_stream_cursor_ != success_logical_size) {
            throw std::logic_error("BC single chunk final success stream cursor mismatch");
        }
        stats_.result_assembly_seconds += bc_single_chunk_now_seconds() - assembly_t0;
        t0 = bc_single_chunk_now_seconds();
        success_tail_stager_->finish();
        const double success_finish = bc_single_chunk_now_seconds() - t0;
        stats_.success_finish_seconds += success_finish;
        stats_.success_write_seconds += success_finish;
        t0 = bc_single_chunk_now_seconds();
        std::memcpy(success_first_block_.data(), success_header_bytes.data(), success_header_bytes.size());
        bc_single_chunk_write_file_extent(
            success_file_,
            0U,
            success_first_block_.data(),
            kBCSingleChunkDirectBlockBytes,
            &stats_.output_success_write
        );
        success_file_.resize(success_logical_size);
        bc_single_chunk_flush_for_stream_finish(success_file_);
        const double success_header_stage = bc_single_chunk_now_seconds() - t0;
        stats_.success_header_seconds += success_header_stage;
        stats_.success_write_seconds += success_header_stage;

        BCSingleChunkSolveFileResult result;
        result.position_bytes = position_logical_size;
        result.success_bytes = success_logical_size;
        result.stats.output_values = success_value_cursor_;
        result.stats.output_bytes = success_logical_size;
        return result;
    }

private:
    [[nodiscard]] uint64_t checked_value_bytes(uint64_t values) const {
        if (values > std::numeric_limits<uint64_t>::max() / value_size_) {
            throw std::overflow_error("BC single chunk final value byte count overflow");
        }
        return values * value_size_;
    }

    void append_position_payload(BCSequentialSuccessWriteStager &stager, const std::vector<uint8_t> &bytes) {
        if (bytes.empty()) {
            return;
        }
        const double t0 = bc_single_chunk_now_seconds();
        stager.append(bytes.data(), bytes.size());
        stats_.position_write_seconds += bc_single_chunk_now_seconds() - t0;
    }

    void write_success_values(const std::vector<StorageT> &values) {
        if (values.size() == 0U) {
            return;
        }
        const double t0 = bc_single_chunk_now_seconds();
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        append_success_bytes(values.data(), checked_value_bytes(values.size()));
#else
        std::vector<uint8_t> bytes;
        bytes.reserve(static_cast<size_t>(checked_value_bytes(values.size())));
        for (StorageT value : values) {
            bc_append_success_value_le(bytes, value);
        }
        append_success_bytes(bytes.data(), bytes.size());
#endif
        const double success_append = bc_single_chunk_now_seconds() - t0;
        stats_.success_append_seconds += success_append;
        stats_.success_write_seconds += success_append;
    }

    void append_success_bytes(const void *data, uint64_t bytes) {
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            if (success_stream_cursor_ < kBCSingleChunkDirectBlockBytes) {
                const uint64_t in_first = success_stream_cursor_;
                const uint64_t available = kBCSingleChunkDirectBlockBytes - in_first;
                const uint64_t take = std::min<uint64_t>(available, remaining);
                std::memcpy(
                    success_first_block_.data() + static_cast<size_t>(in_first),
                    cursor,
                    static_cast<size_t>(take)
                );
                success_stream_cursor_ += take;
                cursor += take;
                remaining -= take;
                continue;
            }
            success_tail_stager_->append(cursor, remaining);
            success_stream_cursor_ += remaining;
            remaining = 0U;
        }
    }

    const BCPositionStreamingReader &current_position_;
    BCWritableFile &position_file_;
    BCWritableFile &success_file_;
    uint32_t row_width_ = 0U;
    BCSuccessDTypeMode dtype_ = BCSuccessDTypeMode::UInt32;
    uint32_t value_size_ = 0U;
    BCSingleChunkSolveStats &stats_;
    std::vector<BCPositionCellDescriptor> descriptors_;
    std::vector<uint8_t> written_;
    uint64_t axis_coord_bytes_ = 0U;
    uint64_t descriptor_offset_ = 0U;
    uint64_t descriptor_bytes_ = 0U;
    uint64_t bucket_offset_ = 0U;
    uint64_t bucket_capacity_bytes_ = 0U;
    uint64_t rank_offset_ = 0U;
    uint64_t rank_capacity_bytes_ = 0U;
    uint64_t bucket_cursor_ = 0U;
    uint64_t rank_cursor_ = 0U;
    uint64_t success_value_cursor_ = 0U;
    std::unique_ptr<BCSequentialSuccessWriteStager> bucket_stager_;
    std::unique_ptr<BCSequentialSuccessWriteStager> rank_stager_;
    std::unique_ptr<BCSequentialSuccessWriteStager> success_tail_stager_;
    detail::BCAlignedBuffer success_first_block_;
    uint64_t success_stream_cursor_ = kBCSuccessHeaderBytes;
};

template <typename StorageT>
BCSingleChunkFrontierLayer<StorageT> bc_single_chunk_load_frontier_layer(
    const BCPositionStreamingReader &position,
    const BCSuccessStreamingReader &success,
    const BCLut &lut,
    uint32_t row_width,
    BCSuccessDTypeMode dtype,
    BCCellLoadStats *position_stats = nullptr,
    BCSuccessLoadStats *success_stats = nullptr,
    double *position_read_seconds = nullptr,
    double *success_read_seconds = nullptr,
    double *index_seconds = nullptr
) {
    if (row_width == 0U) {
        throw std::invalid_argument("BC single chunk frontier load row_width must be non-zero");
    }
    if (!bc_success_dtype_matches_type<StorageT>(dtype) ||
        !bc_success_dtype_matches_type<StorageT>(success.dtype_mode())) {
        throw std::invalid_argument("BC single chunk frontier load dtype mismatch");
    }
    if (success.row_width() != row_width) {
        throw std::invalid_argument("BC single chunk frontier load row_width mismatch");
    }
    bc_solve_validate_success_matches_position(position, success);
    BCSingleChunkFrontierLayer<StorageT> layer;
    BCFileIOStats position_io_stats;
    const double position_t0 = bc_single_chunk_now_seconds();
    std::vector<uint8_t> position_bytes = position.read_all_bytes(&position_io_stats);
    if (position_read_seconds != nullptr) {
        *position_read_seconds += bc_single_chunk_now_seconds() - position_t0;
    }
    if (position_stats != nullptr) {
        bc_single_chunk_add_file_read_stats(*position_stats, position_io_stats);
    }
    BCSuccessLoadStats local_success_stats;
    const double success_t0 = bc_single_chunk_now_seconds();
    BCSuccessOwnedValues<StorageT> values =
        success.template read_all_values_typed_owned<StorageT>(&local_success_stats);
    if (success_read_seconds != nullptr) {
        *success_read_seconds += bc_single_chunk_now_seconds() - success_t0;
    }
    if (success_stats != nullptr) {
        bc_single_chunk_add_success_load_stats(*success_stats, local_success_stats);
    }
    const double index_t0 = bc_single_chunk_now_seconds();
    layer.open(
        std::move(position_bytes),
        std::move(values),
        lut,
        row_width,
        dtype
    );
    if (index_seconds != nullptr) {
        *index_seconds += bc_single_chunk_now_seconds() - index_t0;
    }
    return layer;
}

template <typename StorageT>
BCSingleChunkCompactedBuild<StorageT> bc_single_chunk_solve_compacted_build(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    bc_single_chunk_validate_streaming_options(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );

    BCSingleChunkCompactedBuild<StorageT> build;
    const BCLut &lut = current_position.lut();
    const BCSingleChunkSolveOptions<StorageT> recalc_options =
        bc_single_chunk_effective_recalc_options_for_current_layer(
            current_position,
            options
        );
    const CellId cell_count = current_position.cell_count();
    std::vector<FinalizedCellPayload> payloads(cell_count);
    std::vector<StorageT> compact_values;
    BCSingleChunkSolveWorkspace<StorageT> local_workspace;
    BCSingleChunkSolveWorkspace<StorageT> &scratch =
        workspace == nullptr ? local_workspace : *workspace;

    for (CellId base = 0U; base < cell_count;) {
        const double current_select_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId> current_cids = bc_single_chunk_next_current_cids(
            current_position,
            base,
            options.current_chunk_cells,
            options.current_chunk_max_rows
        );
        build.stats.current_plan_seconds +=
            bc_single_chunk_now_seconds() - current_select_t0;
        if (current_cids.empty()) {
            break;
        }
        base = static_cast<CellId>(current_cids.back() + 1U);

        BCCellLoadStats current_load_stats;
        const double current_read_t0 = bc_single_chunk_now_seconds();
        std::vector<BCLoadedCell> current_cells =
            current_position.load_cells(current_cids, &current_load_stats);
        build.stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
        bc_single_chunk_add_cell_load_stats(build.stats.current_position_load, current_load_stats);
        ++build.stats.current_chunks;
        build.stats.current_cells += current_cells.size();

        const double current_layout_t0 = bc_single_chunk_now_seconds();
        std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                ++build.stats.current_empty_cells;
            } else {
                ++build.stats.current_nonempty_cells;
            }
            cell_offsets[i + 1U] = bc_checked_add_u64(
                cell_offsets[i],
                cell.success_rows,
                "BC single chunk current row offset overflow"
            );
        }
        std::vector<BCSingleChunkLoadedWorkItem> work_items =
            bc_single_chunk_build_loaded_work_items(lut, current_cells);
        build.stats.current_work_items += work_items.size();
        build.stats.current_plan_seconds +=
            bc_single_chunk_now_seconds() - current_layout_t0;
        bc_single_chunk_trace_stage(options, "current_loaded", build.stats);
        const uint64_t chunk_rows = cell_offsets.back();
        if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
            chunk_rows * options.solve.row_width >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC single chunk raw value count exceeds size_t");
        }
        const double raw_alloc_t0 = bc_single_chunk_now_seconds();
        scratch.raw_values.resize_uninitialized(
            static_cast<size_t>(chunk_rows * options.solve.row_width)
        );
        scratch.sum4_values.resize_uninitialized(scratch.raw_values.size());
        build.stats.raw_alloc_seconds += bc_single_chunk_now_seconds() - raw_alloc_t0;
        BCSingleChunkValueBuffer<StorageT> &raw_values = scratch.raw_values;
        BCSingleChunkValueBuffer<BCSingleChunkSum4T<StorageT>> &sum4_values = scratch.sum4_values;

        std::vector<CellId> future2_cids;
        std::vector<CellId> future4_cids;
        if (options.restrict_future_cells_to_current_chunk) {
            const double prepass_t0 = bc_single_chunk_now_seconds();
            bc_single_chunk_collect_future_cids<StorageT>(
                lut,
                current_cells,
                future2_position.axis(),
                future4_position.axis(),
                options,
                future2_cids,
                future4_cids
            );
            build.stats.prepass_seconds += bc_single_chunk_now_seconds() - prepass_t0;
        } else {
            const double cid_select_t0 = bc_single_chunk_now_seconds();
            future2_cids = bc_single_chunk_nonempty_cell_ids(future2_position);
            future4_cids = bc_single_chunk_nonempty_cell_ids(future4_position);
            build.stats.future_cid_select_seconds +=
                bc_single_chunk_now_seconds() - cid_select_t0;
        }

        double future_release_t0 = 0.0;
        {
            uint64_t position_bytes = 0U;
            uint64_t success_bytes = 0U;
            BCFutureSuccessLookupView<StorageT> future4_lookup =
                bc_single_chunk_load_future_lookup<StorageT>(
                    future4_position,
                    future4_success,
                    future4_cids,
                    options.solve.row_width,
                    options.solve.dtype,
                    build.stats.future4_position_load,
                    build.stats.future4_success_load,
                    build.stats.future4_position_read_seconds,
                    build.stats.future4_success_read_seconds,
                    build.stats.future4_index_seconds,
                    position_bytes,
                    success_bytes,
                    !options.restrict_future_cells_to_current_chunk,
                    &options,
                    &build.stats,
                    "future4_position_loaded",
                    "future4_success_loaded",
                    "future4_index_done"
                );
            ++build.stats.future4_batch_loads;
            build.stats.future4_cells_loaded += future4_cids.size();
            build.stats.future4_active_cells_max =
                std::max<uint64_t>(build.stats.future4_active_cells_max, future4_cids.size());
            build.stats.future4_position_resident_bytes =
                std::max(build.stats.future4_position_resident_bytes, position_bytes);
            build.stats.future4_success_resident_bytes =
                std::max(build.stats.future4_success_resident_bytes, success_bytes);
            build.stats.future_resident_layers_max =
                std::max<uint64_t>(build.stats.future_resident_layers_max, 1U);
            build.stats.future_resident_bytes_max = std::max<uint64_t>(
                build.stats.future_resident_bytes_max,
                position_bytes + success_bytes
            );
            bc_single_chunk_trace_stage(options, "future4_loaded", build.stats);
            bc_single_chunk_solve_phase_for_current_cells<StorageT>(
                current_cells,
                work_items,
                cell_offsets,
                raw_values,
                sum4_values,
                lut,
                future4_position.axis(),
                future4_lookup,
                options,
                BCSingleChunkSolvePhase::Spawn4,
                build.stats
            );
            bc_single_chunk_trace_stage(options, "spawn4_done", build.stats);
            future_release_t0 = bc_single_chunk_now_seconds();
        }
        build.stats.future_release_seconds +=
            bc_single_chunk_now_seconds() - future_release_t0;

        future_release_t0 = 0.0;
        {
            uint64_t position_bytes = 0U;
            uint64_t success_bytes = 0U;
            BCFutureSuccessLookupView<StorageT> future2_lookup =
                bc_single_chunk_load_future_lookup<StorageT>(
                    future2_position,
                    future2_success,
                    future2_cids,
                    options.solve.row_width,
                    options.solve.dtype,
                    build.stats.future2_position_load,
                    build.stats.future2_success_load,
                    build.stats.future2_position_read_seconds,
                    build.stats.future2_success_read_seconds,
                    build.stats.future2_index_seconds,
                    position_bytes,
                    success_bytes,
                    !options.restrict_future_cells_to_current_chunk,
                    &options,
                    &build.stats,
                    "future2_position_loaded",
                    "future2_success_loaded",
                    "future2_index_done"
                );
            ++build.stats.future2_batch_loads;
            build.stats.future2_cells_loaded += future2_cids.size();
            build.stats.future2_active_cells_max =
                std::max<uint64_t>(build.stats.future2_active_cells_max, future2_cids.size());
            build.stats.future2_position_resident_bytes =
                std::max(build.stats.future2_position_resident_bytes, position_bytes);
            build.stats.future2_success_resident_bytes =
                std::max(build.stats.future2_success_resident_bytes, success_bytes);
            build.stats.future_resident_layers_max =
                std::max<uint64_t>(build.stats.future_resident_layers_max, 1U);
            build.stats.future_resident_bytes_max = std::max<uint64_t>(
                build.stats.future_resident_bytes_max,
                position_bytes + success_bytes
            );
            bc_single_chunk_trace_stage(options, "future2_loaded", build.stats);
            bc_single_chunk_solve_phase_for_current_cells<StorageT>(
                current_cells,
                work_items,
                cell_offsets,
                raw_values,
                sum4_values,
                lut,
                future2_position.axis(),
                future2_lookup,
                options,
                BCSingleChunkSolvePhase::Spawn2,
                build.stats
            );
            bc_single_chunk_trace_stage(options, "spawn2_done", build.stats);
            future_release_t0 = bc_single_chunk_now_seconds();
        }
        build.stats.future_release_seconds +=
            bc_single_chunk_now_seconds() - future_release_t0;

        const double compact_t0 = bc_single_chunk_now_seconds();
        const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
        std::vector<BCResidentCompactStats> per_compact_thread(static_cast<size_t>(compact_threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
        for (int64_t i_signed = 0;
             i_signed < static_cast<int64_t>(current_cells.size());
             ++i_signed) {
#if defined(_OPENMP)
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            const size_t i = static_cast<size_t>(i_signed);
            const BCLoadedCell &cell = current_cells[i];
            bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
                lut,
                cell,
                raw_values,
                cell_offsets[i],
                options.solve.row_width,
                options.solve.zero_value,
                payloads[static_cast<size_t>(cell.cid)],
                per_compact_thread[static_cast<size_t>(tid)]
            );
        }
        for (const BCResidentCompactStats &stats : per_compact_thread) {
            build.compact_stats.input_rows += stats.input_rows;
            build.compact_stats.live_rows += stats.live_rows;
            build.compact_stats.zero_pruned_rows += stats.zero_pruned_rows;
            build.compact_stats.live_cells += stats.live_cells;
            build.compact_stats.empty_cells += stats.empty_cells;
        }
        uint64_t compact_chunk_values = 0U;
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[static_cast<size_t>(current_cells[i].cid)];
            compact_chunk_values = bc_checked_add_u64(
                compact_chunk_values,
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width,
                "BC single chunk compact chunk value count overflow"
            );
        }
        if (compact_chunk_values >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - compact_values.size())) {
            throw std::overflow_error("BC single chunk compact values exceed size_t");
        }
        compact_values.reserve(compact_values.size() + static_cast<size_t>(compact_chunk_values));
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[static_cast<size_t>(current_cells[i].cid)];
            const uint64_t value_count =
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width;
            const uint64_t src_offset = cell_offsets[i] * static_cast<uint64_t>(options.solve.row_width);
            if (src_offset + value_count > raw_values.size()) {
                throw std::out_of_range("BC single chunk compact output exceeds raw values");
            }
            const StorageT *begin = raw_values.data() + static_cast<std::ptrdiff_t>(src_offset);
            compact_values.insert(
                compact_values.end(),
                begin,
                begin + static_cast<std::ptrdiff_t>(value_count)
            );
        }
        build.stats.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
        const double release_t0 = bc_single_chunk_now_seconds();
        (void)release_t0;
        build.stats.workspace_release_seconds += bc_single_chunk_now_seconds() - release_t0;
        bc_single_chunk_trace_stage(options, "compact_done", build.stats);
    }

    const double position_build_t0 = bc_single_chunk_now_seconds();
    build.position_bytes =
        bc_resident_build_position_bytes_from_payloads(current_position.axis(), payloads);
    build.stats.position_build_seconds += bc_single_chunk_now_seconds() - position_build_t0;
    const double assembly_t0 = bc_single_chunk_now_seconds();
    build.success_values = std::move(compact_values);
    build.compact_stats.position_bytes = build.position_bytes.size();
    build.compact_stats.success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(build.success_values.size()) *
            bc_success_dtype_value_size(options.solve.dtype);
    build.compact_stats.compact_seconds = build.stats.compact_seconds;
    build.stats.compact_input_rows = build.compact_stats.input_rows;
    build.stats.compact_live_rows = build.compact_stats.live_rows;
    build.stats.compact_zero_pruned_rows = build.compact_stats.zero_pruned_rows;
    build.stats.compact_live_cells = build.compact_stats.live_cells;
    build.stats.compact_empty_cells = build.compact_stats.empty_cells;
    build.stats.output_values = build.success_values.size();
    build.stats.output_bytes = build.compact_stats.success_bytes;
    build.stats.result_assembly_seconds += bc_single_chunk_now_seconds() - assembly_t0;
    bc_single_chunk_trace_stage(options, "build_done", build.stats);
    return build;
}

template <typename StorageT>
BCSingleChunkCompactedBuild<StorageT> bc_single_chunk_solve_compacted_build_from_frontier(
    const BCPositionStreamingReader &current_position,
    const BCResidentSolvedLayer<StorageT> &future2_layer,
    const BCResidentSolvedLayer<StorageT> &future4_layer,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    bc_single_chunk_validate_frontier_options(
        current_position,
        future2_layer,
        future4_layer,
        options
    );

    BCSingleChunkCompactedBuild<StorageT> build;
    const BCLut &lut = current_position.lut();
    const CellId cell_count = current_position.cell_count();
    std::vector<FinalizedCellPayload> payloads(cell_count);
    std::vector<StorageT> compact_values;
    BCSingleChunkSolveWorkspace<StorageT> local_workspace;
    BCSingleChunkSolveWorkspace<StorageT> &scratch =
        workspace == nullptr ? local_workspace : *workspace;

    const uint64_t future2_position_bytes = future2_layer.position.bytes().size();
    const uint64_t future4_position_bytes = future4_layer.position.bytes().size();
    const uint64_t future2_success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(future2_layer.success_values.size()) *
            bc_success_dtype_value_size(options.solve.dtype);
    const uint64_t future4_success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(future4_layer.success_values.size()) *
            bc_success_dtype_value_size(options.solve.dtype);
    build.stats.future2_position_resident_bytes = future2_position_bytes;
    build.stats.future4_position_resident_bytes = future4_position_bytes;
    build.stats.future2_success_resident_bytes = future2_success_bytes;
    build.stats.future4_success_resident_bytes = future4_success_bytes;
    build.stats.future2_active_cells_max = future2_layer.position.cell_count();
    build.stats.future4_active_cells_max = future4_layer.position.cell_count();
    build.stats.future_resident_layers_max = 2U;
    build.stats.future_resident_bytes_max =
        future2_position_bytes + future2_success_bytes +
        future4_position_bytes + future4_success_bytes;

    for (CellId base = 0U; base < cell_count;) {
        const double current_select_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId> current_cids = bc_single_chunk_next_current_cids(
            current_position,
            base,
            options.current_chunk_cells,
            options.current_chunk_max_rows
        );
        build.stats.current_plan_seconds +=
            bc_single_chunk_now_seconds() - current_select_t0;
        if (current_cids.empty()) {
            break;
        }
        base = static_cast<CellId>(current_cids.back() + 1U);

        BCCellLoadStats current_load_stats;
        const double current_read_t0 = bc_single_chunk_now_seconds();
        std::vector<BCLoadedCell> current_cells =
            current_position.load_cells(current_cids, &current_load_stats);
        build.stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
        bc_single_chunk_add_cell_load_stats(build.stats.current_position_load, current_load_stats);
        ++build.stats.current_chunks;
        build.stats.current_cells += current_cells.size();

        const double current_layout_t0 = bc_single_chunk_now_seconds();
        std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                ++build.stats.current_empty_cells;
            } else {
                ++build.stats.current_nonempty_cells;
            }
            cell_offsets[i + 1U] = bc_checked_add_u64(
                cell_offsets[i],
                cell.success_rows,
                "BC single chunk frontier current row offset overflow"
            );
        }
        std::vector<BCSingleChunkLoadedWorkItem> work_items =
            bc_single_chunk_build_loaded_work_items(lut, current_cells);
        build.stats.current_work_items += work_items.size();
        build.stats.current_plan_seconds +=
            bc_single_chunk_now_seconds() - current_layout_t0;
        bc_single_chunk_trace_stage(options, "current_loaded", build.stats);
        const uint64_t chunk_rows = cell_offsets.back();
        if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
            chunk_rows * options.solve.row_width >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC single chunk frontier raw value count exceeds size_t");
        }
        const double raw_alloc_t0 = bc_single_chunk_now_seconds();
        scratch.raw_values.resize_uninitialized(
            static_cast<size_t>(chunk_rows * options.solve.row_width)
        );
        scratch.sum4_values.resize_uninitialized(scratch.raw_values.size());
        build.stats.raw_alloc_seconds += bc_single_chunk_now_seconds() - raw_alloc_t0;
        BCSingleChunkValueBuffer<StorageT> &raw_values = scratch.raw_values;
        BCSingleChunkValueBuffer<BCSingleChunkSum4T<StorageT>> &sum4_values = scratch.sum4_values;

        bc_single_chunk_trace_stage(options, "frontier4_ready", build.stats);
        bc_single_chunk_solve_phase_for_current_cells<StorageT>(
            current_cells,
            work_items,
            cell_offsets,
            raw_values,
            sum4_values,
            lut,
            future4_layer.position.axis(),
            future4_layer.lookup,
            options,
            BCSingleChunkSolvePhase::Spawn4,
            build.stats
        );
        bc_single_chunk_trace_stage(options, "spawn4_done", build.stats);

        bc_single_chunk_trace_stage(options, "frontier2_ready", build.stats);
        bc_single_chunk_solve_phase_for_current_cells<StorageT>(
            current_cells,
            work_items,
            cell_offsets,
            raw_values,
            sum4_values,
            lut,
            future2_layer.position.axis(),
            future2_layer.lookup,
            options,
            BCSingleChunkSolvePhase::Spawn2,
            build.stats
        );
        bc_single_chunk_trace_stage(options, "spawn2_done", build.stats);

        const double compact_t0 = bc_single_chunk_now_seconds();
        const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
        std::vector<BCResidentCompactStats> per_compact_thread(static_cast<size_t>(compact_threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
        for (int64_t i_signed = 0;
             i_signed < static_cast<int64_t>(current_cells.size());
             ++i_signed) {
#if defined(_OPENMP)
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            const size_t i = static_cast<size_t>(i_signed);
            const BCLoadedCell &cell = current_cells[i];
            bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
                lut,
                cell,
                raw_values,
                cell_offsets[i],
                options.solve.row_width,
                options.solve.zero_value,
                payloads[static_cast<size_t>(cell.cid)],
                per_compact_thread[static_cast<size_t>(tid)]
            );
        }
        for (const BCResidentCompactStats &stats : per_compact_thread) {
            build.compact_stats.input_rows += stats.input_rows;
            build.compact_stats.live_rows += stats.live_rows;
            build.compact_stats.zero_pruned_rows += stats.zero_pruned_rows;
            build.compact_stats.live_cells += stats.live_cells;
            build.compact_stats.empty_cells += stats.empty_cells;
        }
        uint64_t compact_chunk_values = 0U;
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[static_cast<size_t>(current_cells[i].cid)];
            compact_chunk_values = bc_checked_add_u64(
                compact_chunk_values,
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width,
                "BC single chunk frontier compact chunk value count overflow"
            );
        }
        if (compact_chunk_values >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - compact_values.size())) {
            throw std::overflow_error("BC single chunk frontier compact values exceed size_t");
        }
        compact_values.reserve(compact_values.size() + static_cast<size_t>(compact_chunk_values));
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[static_cast<size_t>(current_cells[i].cid)];
            const uint64_t value_count =
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width;
            const uint64_t src_offset = cell_offsets[i] * static_cast<uint64_t>(options.solve.row_width);
            if (src_offset + value_count > raw_values.size()) {
                throw std::out_of_range("BC single chunk frontier compact output exceeds raw values");
            }
            const StorageT *begin = raw_values.data() + static_cast<std::ptrdiff_t>(src_offset);
            compact_values.insert(
                compact_values.end(),
                begin,
                begin + static_cast<std::ptrdiff_t>(value_count)
            );
        }
        build.stats.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
        bc_single_chunk_trace_stage(options, "compact_done", build.stats);
    }

    const double position_build_t0 = bc_single_chunk_now_seconds();
    build.position_bytes =
        bc_resident_build_position_bytes_from_payloads(current_position.axis(), payloads);
    build.stats.position_build_seconds += bc_single_chunk_now_seconds() - position_build_t0;
    const double assembly_t0 = bc_single_chunk_now_seconds();
    build.success_values = std::move(compact_values);
    build.compact_stats.position_bytes = build.position_bytes.size();
    build.compact_stats.success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(build.success_values.size()) *
            bc_success_dtype_value_size(options.solve.dtype);
    build.compact_stats.compact_seconds = build.stats.compact_seconds;
    build.stats.compact_input_rows = build.compact_stats.input_rows;
    build.stats.compact_live_rows = build.compact_stats.live_rows;
    build.stats.compact_zero_pruned_rows = build.compact_stats.zero_pruned_rows;
    build.stats.compact_live_cells = build.compact_stats.live_cells;
    build.stats.compact_empty_cells = build.compact_stats.empty_cells;
    build.stats.output_values = build.success_values.size();
    build.stats.output_bytes = build.compact_stats.success_bytes;
    build.stats.result_assembly_seconds += bc_single_chunk_now_seconds() - assembly_t0;
    bc_single_chunk_trace_stage(options, "build_done", build.stats);
    return build;
}

template <typename StorageT>
BCSingleChunkSolveResult<StorageT> bc_single_chunk_solve_compacted_layer(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    BCSingleChunkCompactedBuild<StorageT> build =
        bc_single_chunk_solve_compacted_build<StorageT>(
            current_position,
            future2_position,
            future2_success,
            future4_position,
            future4_success,
            options,
            workspace
        );
    BCSingleChunkSolveResult<StorageT> result;
    result.stats = build.stats;
    result.layer.open(
        std::move(build.position_bytes),
        std::move(build.success_values),
        current_position.lut(),
        options.solve.row_width,
        options.solve.dtype
    );
    result.layer.compact_stats = build.compact_stats;
    return result;
}

template <typename StorageT>
BCSingleChunkSolveResult<StorageT> bc_single_chunk_solve_compacted_layer_from_frontier(
    const BCPositionStreamingReader &current_position,
    const BCResidentSolvedLayer<StorageT> &future2_layer,
    const BCResidentSolvedLayer<StorageT> &future4_layer,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    BCSingleChunkCompactedBuild<StorageT> build =
        bc_single_chunk_solve_compacted_build_from_frontier<StorageT>(
            current_position,
            future2_layer,
            future4_layer,
            options,
            workspace
        );
    BCSingleChunkSolveResult<StorageT> result;
    result.stats = build.stats;
    result.layer.open(
        std::move(build.position_bytes),
        std::move(build.success_values),
        current_position.lut(),
        options.solve.row_width,
        options.solve.dtype
    );
    result.layer.compact_stats = build.compact_stats;
    return result;
}

template <typename StorageT>
BCSingleChunkSolveFileResult bc_single_chunk_solve_strict_1x_to_files(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    BCWritableFile &position_file,
    BCWritableFile &success_file,
    const std::filesystem::path &temp_dir,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    bc_single_chunk_validate_streaming_options(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );
    if (options.restrict_future_cells_to_current_chunk) {
        throw std::invalid_argument(
            "BC strict 1+x single solve requires full future lookup, not per-chunk future filtering"
        );
    }
    BCSingleChunkSolveStats stats;
    const double temp_prepare_t0 = bc_single_chunk_now_seconds();
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_dir, cleanup_ec);
    std::filesystem::create_directories(temp_dir);
    stats.temp_prepare_seconds += bc_single_chunk_now_seconds() - temp_prepare_t0;

    BCSingleChunkSolveWorkspace<StorageT> local_workspace;
    BCSingleChunkSolveWorkspace<StorageT> &scratch =
        workspace == nullptr ? local_workspace : *workspace;
    const BCLut &lut = current_position.lut();
    const BCSingleChunkSolveOptions<StorageT> recalc_options =
        bc_single_chunk_effective_recalc_options_for_current_layer(
            current_position,
            options
        );
    const CellId cell_count = current_position.cell_count();
    const BCCellMatrix current_matrix(current_position.axis());
    const FamilyId family_count = static_cast<FamilyId>(current_matrix.family_count());
    const uint32_t value_size = bc_success_dtype_value_size(options.solve.dtype);
    const bool tmp_direct_io =
        position_file.mode() == BCFileIOMode::Direct ||
        success_file.mode() == BCFileIOMode::Direct;
    const std::vector<CellId> future4_cids = bc_single_chunk_nonempty_cell_ids(future4_position);
    const std::vector<CellId> future2_cids = bc_single_chunk_nonempty_cell_ids(future2_position);
    stats.future_cid_select_seconds = 0.0;

    uint32_t pass1_chunks = 0U;
    {
        uint64_t position_bytes = 0U;
        uint64_t success_bytes = 0U;
        BCFutureSuccessLookupView<StorageT> future4_lookup =
            bc_single_chunk_load_future_lookup<StorageT>(
                future4_position,
                future4_success,
                future4_cids,
                options.solve.row_width,
                options.solve.dtype,
                stats.future4_position_load,
                stats.future4_success_load,
                stats.future4_position_read_seconds,
                stats.future4_success_read_seconds,
                stats.future4_index_seconds,
                position_bytes,
                success_bytes,
                true,
                &options,
                &stats,
                "future4_position_loaded",
                "future4_success_loaded",
                "future4_index_done"
            );
        ++stats.future4_batch_loads;
        stats.future4_cells_loaded += future4_cids.size();
        stats.future4_active_cells_max = std::max<uint64_t>(
            stats.future4_active_cells_max,
            future4_cids.size()
        );
        stats.future4_position_resident_bytes = std::max(
            stats.future4_position_resident_bytes,
            position_bytes
        );
        stats.future4_success_resident_bytes = std::max(
            stats.future4_success_resident_bytes,
            success_bytes
        );
        stats.future_resident_layers_max = std::max<uint64_t>(stats.future_resident_layers_max, 1U);
        stats.future_resident_bytes_max = std::max<uint64_t>(
            stats.future_resident_bytes_max,
            position_bytes + success_bytes
        );

        for (FamilyId row_begin = 0U; row_begin < family_count;) {
            const double current_select_t0 = bc_single_chunk_now_seconds();
            std::vector<CellId> current_cids = bc_single_chunk_next_row_slab_cids(
                current_position,
                row_begin,
                options.current_chunk_rows,
                options.current_chunk_max_bytes,
                options.solve.row_width,
                value_size
            );
            stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_select_t0;
            if (current_cids.empty()) {
                break;
            }
            row_begin = static_cast<FamilyId>(current_matrix.row(current_cids.back()) + 1U);

            BCCellLoadStats current_load_stats;
            const double current_read_t0 = bc_single_chunk_now_seconds();
            std::vector<BCLoadedCell> current_cells =
                current_position.load_cells(current_cids, &current_load_stats);
            stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
            bc_single_chunk_add_cell_load_stats(stats.current_position_load, current_load_stats);

            const double current_layout_t0 = bc_single_chunk_now_seconds();
            std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
            for (size_t i = 0U; i < current_cells.size(); ++i) {
                cell_offsets[i + 1U] = bc_checked_add_u64(
                    cell_offsets[i],
                    current_cells[i].success_rows,
                    "BC strict single pass1 current row offset overflow"
                );
            }
            std::vector<BCSingleChunkLoadedWorkItem> work_items =
                bc_single_chunk_build_loaded_work_items(lut, current_cells);
            stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_layout_t0;

            const uint64_t chunk_rows = cell_offsets.back();
            if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
                chunk_rows * options.solve.row_width >
                    static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC strict single pass1 value count exceeds size_t");
            }
            scratch.raw_values.resize_uninitialized(0U);
            scratch.sum4_values.resize_uninitialized(
                static_cast<size_t>(chunk_rows * options.solve.row_width)
            );
            bc_single_chunk_solve_phase_for_current_cells<StorageT>(
                current_cells,
                work_items,
                cell_offsets,
                scratch.raw_values,
                scratch.sum4_values,
                lut,
                future4_position.axis(),
                future4_lookup,
                recalc_options,
                BCSingleChunkSolvePhase::Spawn4,
                stats,
                &scratch.batch_workspaces
            );
            const double partial_t0 = bc_single_chunk_now_seconds();
            const uint64_t tmp4_bytes =
                bc_single_chunk_tmp4_file_bytes(scratch.sum4_values, tmp_direct_io);
            bc_single_chunk_write_tmp4_values(
                bc_single_chunk_partial_path(temp_dir, pass1_chunks),
                scratch.sum4_values,
                tmp_direct_io,
                &stats.partial_write_io
            );
            stats.partial_write_seconds += bc_single_chunk_now_seconds() - partial_t0;
            stats.partial_write_bytes = bc_checked_add_u64(
                stats.partial_write_bytes,
                tmp4_bytes,
                "BC strict single tmp4 write byte stats overflow"
            );
            const double workspace_release_t0 = bc_single_chunk_now_seconds();
            std::vector<CellId>().swap(current_cids);
            std::vector<BCLoadedCell>().swap(current_cells);
            std::vector<uint64_t>().swap(cell_offsets);
            std::vector<BCSingleChunkLoadedWorkItem>().swap(work_items);
            stats.workspace_release_seconds += bc_single_chunk_now_seconds() - workspace_release_t0;
            ++pass1_chunks;
        }
        const double release_t0 = bc_single_chunk_now_seconds();
        (void)release_t0;
        stats.future_release_seconds += bc_single_chunk_now_seconds() - release_t0;
    }

    uint32_t pass2_chunks = 0U;
    {
        uint64_t position_bytes = 0U;
        uint64_t success_bytes = 0U;
        BCFutureSuccessLookupView<StorageT> future2_lookup =
            bc_single_chunk_load_future_lookup<StorageT>(
                future2_position,
                future2_success,
                future2_cids,
                options.solve.row_width,
                options.solve.dtype,
                stats.future2_position_load,
                stats.future2_success_load,
                stats.future2_position_read_seconds,
                stats.future2_success_read_seconds,
                stats.future2_index_seconds,
                position_bytes,
                success_bytes,
                true,
                &options,
                &stats,
                "future2_position_loaded",
                "future2_success_loaded",
                "future2_index_done"
            );
        ++stats.future2_batch_loads;
        stats.future2_cells_loaded += future2_cids.size();
        stats.future2_active_cells_max = std::max<uint64_t>(
            stats.future2_active_cells_max,
            future2_cids.size()
        );
        stats.future2_position_resident_bytes = std::max(
            stats.future2_position_resident_bytes,
            position_bytes
        );
        stats.future2_success_resident_bytes = std::max(
            stats.future2_success_resident_bytes,
            success_bytes
        );
        stats.future_resident_layers_max = std::max<uint64_t>(stats.future_resident_layers_max, 1U);
        stats.future_resident_bytes_max = std::max<uint64_t>(
            stats.future_resident_bytes_max,
            position_bytes + success_bytes
        );

        BCSingleChunkFinalFileStreamer<StorageT> output_streamer(
            current_position,
            position_file,
            success_file,
            options.solve.row_width,
            options.solve.dtype,
            stats
        );

        for (FamilyId row_begin = 0U; row_begin < family_count;) {
            const double current_select_t0 = bc_single_chunk_now_seconds();
            std::vector<CellId> current_cids = bc_single_chunk_next_row_slab_cids(
                current_position,
                row_begin,
                options.current_chunk_rows,
                options.current_chunk_max_bytes,
                options.solve.row_width,
                value_size
            );
            stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_select_t0;
            if (current_cids.empty()) {
                break;
            }
            row_begin = static_cast<FamilyId>(current_matrix.row(current_cids.back()) + 1U);

            BCCellLoadStats current_load_stats;
            const double current_read_t0 = bc_single_chunk_now_seconds();
            std::vector<BCLoadedCell> current_cells =
                current_position.load_cells(current_cids, &current_load_stats);
            stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
            bc_single_chunk_add_cell_load_stats(stats.current_position_load, current_load_stats);
            ++stats.current_chunks;
            stats.current_cells += current_cells.size();

            const double current_layout_t0 = bc_single_chunk_now_seconds();
            std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
            for (size_t i = 0U; i < current_cells.size(); ++i) {
                const BCLoadedCell &cell = current_cells[i];
                if (cell.success_rows == 0U || cell.buckets.empty()) {
                    ++stats.current_empty_cells;
                } else {
                    ++stats.current_nonempty_cells;
                }
                cell_offsets[i + 1U] = bc_checked_add_u64(
                    cell_offsets[i],
                    cell.success_rows,
                    "BC strict single pass2 current row offset overflow"
                );
            }
            std::vector<BCSingleChunkLoadedWorkItem> work_items =
                bc_single_chunk_build_loaded_work_items(lut, current_cells);
            stats.current_work_items += work_items.size();
            stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_layout_t0;

            const uint64_t chunk_rows = cell_offsets.back();
            if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
                chunk_rows * options.solve.row_width >
                    static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::overflow_error("BC strict single pass2 raw value count exceeds size_t");
            }
            const size_t value_count = static_cast<size_t>(chunk_rows * options.solve.row_width);
            const double partial_read_t0 = bc_single_chunk_now_seconds();
            bc_single_chunk_read_tmp4_values(
                bc_single_chunk_partial_path(temp_dir, pass2_chunks),
                scratch.sum4_values,
                value_count,
                tmp_direct_io,
                &stats.partial_read_io
            );
            stats.partial_read_seconds += bc_single_chunk_now_seconds() - partial_read_t0;
            stats.partial_read_bytes = bc_checked_add_u64(
                stats.partial_read_bytes,
                bc_single_chunk_tmp4_file_bytes(scratch.sum4_values, tmp_direct_io),
                "BC strict single tmp4 read byte stats overflow"
            );

            scratch.raw_values.resize_uninitialized(value_count);
            bc_single_chunk_solve_phase_for_current_cells<StorageT>(
                current_cells,
                work_items,
                cell_offsets,
                scratch.raw_values,
                scratch.sum4_values,
                lut,
                future2_position.axis(),
                future2_lookup,
                recalc_options,
                BCSingleChunkSolvePhase::Spawn2,
                stats,
                &scratch.batch_workspaces
            );

            const double compact_t0 = bc_single_chunk_now_seconds();
            const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
            std::vector<FinalizedCellPayload> payloads(current_cells.size());
            std::vector<BCResidentCompactStats> per_compact_thread(static_cast<size_t>(compact_threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
            for (int64_t i_signed = 0;
                 i_signed < static_cast<int64_t>(current_cells.size());
                 ++i_signed) {
#if defined(_OPENMP)
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                const size_t i = static_cast<size_t>(i_signed);
                const BCLoadedCell &cell = current_cells[i];
                bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
                    lut,
                    cell,
                    scratch.raw_values,
                    cell_offsets[i],
                    options.solve.row_width,
                    options.solve.zero_value,
                    payloads[i],
                    per_compact_thread[static_cast<size_t>(tid)]
                );
            }
            for (const BCResidentCompactStats &compact_stats : per_compact_thread) {
                stats.compact_input_rows += compact_stats.input_rows;
                stats.compact_live_rows += compact_stats.live_rows;
                stats.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
                stats.compact_live_cells += compact_stats.live_cells;
                stats.compact_empty_cells += compact_stats.empty_cells;
            }
            uint64_t compact_chunk_values = 0U;
            for (size_t i = 0U; i < current_cells.size(); ++i) {
                const FinalizedCellPayload &payload = payloads[i];
                compact_chunk_values = bc_checked_add_u64(
                    compact_chunk_values,
                    static_cast<uint64_t>(payload.success_rows) * options.solve.row_width,
                    "BC strict single compact chunk value count overflow"
                );
            }
            std::vector<StorageT> compact_values;
            compact_values.reserve(static_cast<size_t>(compact_chunk_values));
            for (size_t i = 0U; i < current_cells.size(); ++i) {
                const FinalizedCellPayload &payload = payloads[i];
                const uint64_t local_values =
                    static_cast<uint64_t>(payload.success_rows) * options.solve.row_width;
                const uint64_t src_offset = cell_offsets[i] * static_cast<uint64_t>(options.solve.row_width);
                if (src_offset + local_values > scratch.raw_values.size()) {
                    throw std::out_of_range("BC strict single compact values exceed raw buffer");
                }
                const StorageT *begin =
                    scratch.raw_values.data() + static_cast<std::ptrdiff_t>(src_offset);
                compact_values.insert(
                    compact_values.end(),
                    begin,
                    begin + static_cast<std::ptrdiff_t>(local_values)
                );
            }
            stats.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
            output_streamer.write_chunk(
                current_cells,
                payloads,
                compact_values
            );
            const double partial_cleanup_t0 = bc_single_chunk_now_seconds();
            std::error_code remove_partial_ec;
            std::filesystem::remove(bc_single_chunk_partial_path(temp_dir, pass2_chunks), remove_partial_ec);
            stats.partial_cleanup_seconds += bc_single_chunk_now_seconds() - partial_cleanup_t0;
            const double workspace_release_t0 = bc_single_chunk_now_seconds();
            std::vector<CellId>().swap(current_cids);
            std::vector<BCLoadedCell>().swap(current_cells);
            std::vector<uint64_t>().swap(cell_offsets);
            std::vector<BCSingleChunkLoadedWorkItem>().swap(work_items);
            std::vector<FinalizedCellPayload>().swap(payloads);
            std::vector<BCResidentCompactStats>().swap(per_compact_thread);
            std::vector<StorageT>().swap(compact_values);
            stats.workspace_release_seconds += bc_single_chunk_now_seconds() - workspace_release_t0;
            ++pass2_chunks;
        }
        if (pass1_chunks != pass2_chunks) {
            throw std::runtime_error("BC strict single pass chunk count mismatch");
        }
        BCSingleChunkSolveFileResult result = output_streamer.finish();
        const uint64_t result_position_bytes = result.position_bytes;
        const uint64_t result_success_bytes = result.success_bytes;
        const uint64_t output_values = result.stats.output_values;
        const uint64_t output_bytes = result.stats.output_bytes;
        const double release_t0 = bc_single_chunk_now_seconds();
        (void)release_t0;
        stats.future_release_seconds += bc_single_chunk_now_seconds() - release_t0;
        result.stats = stats;
        result.stats.output_values = output_values;
        result.stats.output_bytes = output_bytes;
        result.position_bytes = result_position_bytes;
        result.success_bytes = result_success_bytes;
        return result;
    }
    throw std::logic_error("BC strict single solve exited without pass2 result");
}

template <typename StorageT>
BCSingleChunkStrictFrontierFileResult<StorageT>
bc_single_chunk_solve_strict_1x_to_files_from_future4_frontier(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    BCSingleChunkFrontierLayer<StorageT> future4_layer,
    BCWritableFile &position_file,
    BCWritableFile &success_file,
    const std::filesystem::path &temp_dir,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    bc_single_chunk_validate_strict_future4_frontier_options(
        current_position,
        future2_position,
        future2_success,
        future4_layer,
        options
    );
    if (options.restrict_future_cells_to_current_chunk) {
        throw std::invalid_argument(
            "BC strict 1+x frontier solve requires full future lookup, not per-chunk future filtering"
        );
    }
    BCSingleChunkSolveStats stats;
    const double temp_prepare_t0 = bc_single_chunk_now_seconds();
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_dir, cleanup_ec);
    std::filesystem::create_directories(temp_dir);
    stats.temp_prepare_seconds += bc_single_chunk_now_seconds() - temp_prepare_t0;

    BCSingleChunkSolveWorkspace<StorageT> local_workspace;
    BCSingleChunkSolveWorkspace<StorageT> &scratch =
        workspace == nullptr ? local_workspace : *workspace;
    const BCLut &lut = current_position.lut();
    const BCSingleChunkSolveOptions<StorageT> recalc_options =
        bc_single_chunk_effective_recalc_options_for_current_layer(
            current_position,
            options
        );
    const BCCellMatrix current_matrix(current_position.axis());
    const FamilyId family_count = static_cast<FamilyId>(current_matrix.family_count());
    const uint32_t value_size = bc_success_dtype_value_size(options.solve.dtype);
    const bool tmp_direct_io =
        position_file.mode() == BCFileIOMode::Direct ||
        success_file.mode() == BCFileIOMode::Direct;

    const auto record_future4_resident = [&]() {
        const uint64_t position_bytes =
            static_cast<uint64_t>(future4_layer.position.bytes().size());
        const uint64_t success_bytes =
            kBCSuccessHeaderBytes +
            static_cast<uint64_t>(future4_layer.success_value_count()) *
                bc_success_dtype_value_size(options.solve.dtype);
        stats.future4_active_cells_max = std::max<uint64_t>(
            stats.future4_active_cells_max,
            future4_layer.position.cell_count()
        );
        stats.future4_position_resident_bytes = std::max(
            stats.future4_position_resident_bytes,
            position_bytes
        );
        stats.future4_success_resident_bytes = std::max(
            stats.future4_success_resident_bytes,
            success_bytes
        );
        stats.future_resident_layers_max = std::max<uint64_t>(stats.future_resident_layers_max, 1U);
        stats.future_resident_bytes_max = std::max<uint64_t>(
            stats.future_resident_bytes_max,
            position_bytes + success_bytes
        );
    };

    uint32_t pass1_chunks = 0U;
    record_future4_resident();
    for (FamilyId row_begin = 0U; row_begin < family_count;) {
        const double current_select_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId> current_cids = bc_single_chunk_next_row_slab_cids(
            current_position,
            row_begin,
            options.current_chunk_rows,
            options.current_chunk_max_bytes,
            options.solve.row_width,
            value_size
        );
        stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_select_t0;
        if (current_cids.empty()) {
            break;
        }
        row_begin = static_cast<FamilyId>(current_matrix.row(current_cids.back()) + 1U);

        BCCellLoadStats current_load_stats;
        const double current_read_t0 = bc_single_chunk_now_seconds();
        std::vector<BCLoadedCell> current_cells =
            current_position.load_cells(current_cids, &current_load_stats);
        stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
        bc_single_chunk_add_cell_load_stats(stats.current_position_load, current_load_stats);

        const double current_layout_t0 = bc_single_chunk_now_seconds();
        std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            cell_offsets[i + 1U] = bc_checked_add_u64(
                cell_offsets[i],
                current_cells[i].success_rows,
                "BC strict frontier pass1 current row offset overflow"
            );
        }
        std::vector<BCSingleChunkLoadedWorkItem> work_items =
            bc_single_chunk_build_loaded_work_items(lut, current_cells);
        stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_layout_t0;

        const uint64_t chunk_rows = cell_offsets.back();
        if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
            chunk_rows * options.solve.row_width >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC strict frontier pass1 value count exceeds size_t");
        }
        scratch.raw_values.resize_uninitialized(0U);
        scratch.sum4_values.resize_uninitialized(
            static_cast<size_t>(chunk_rows * options.solve.row_width)
        );
        bc_single_chunk_solve_phase_for_current_cells<StorageT>(
            current_cells,
            work_items,
            cell_offsets,
            scratch.raw_values,
            scratch.sum4_values,
            lut,
            future4_layer.position.axis(),
            future4_layer.lookup,
            recalc_options,
            BCSingleChunkSolvePhase::Spawn4,
            stats,
            &scratch.batch_workspaces
        );
        const double partial_t0 = bc_single_chunk_now_seconds();
        const uint64_t tmp4_bytes =
            bc_single_chunk_tmp4_file_bytes(scratch.sum4_values, tmp_direct_io);
        bc_single_chunk_write_tmp4_values(
            bc_single_chunk_partial_path(temp_dir, pass1_chunks),
            scratch.sum4_values,
            tmp_direct_io,
            &stats.partial_write_io
        );
        stats.partial_write_seconds += bc_single_chunk_now_seconds() - partial_t0;
        stats.partial_write_bytes = bc_checked_add_u64(
            stats.partial_write_bytes,
            tmp4_bytes,
            "BC strict frontier tmp4 write byte stats overflow"
        );
        const double workspace_release_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId>().swap(current_cids);
        std::vector<BCLoadedCell>().swap(current_cells);
        std::vector<uint64_t>().swap(cell_offsets);
        std::vector<BCSingleChunkLoadedWorkItem>().swap(work_items);
        stats.workspace_release_seconds += bc_single_chunk_now_seconds() - workspace_release_t0;
        ++pass1_chunks;
    }

    const double future4_release_t0 = bc_single_chunk_now_seconds();
    future4_layer = BCSingleChunkFrontierLayer<StorageT>();
    stats.future_release_seconds += bc_single_chunk_now_seconds() - future4_release_t0;

    BCSingleChunkFrontierLayer<StorageT> future2_layer =
        bc_single_chunk_load_frontier_layer<StorageT>(
            future2_position,
            future2_success,
            lut,
            options.solve.row_width,
            options.solve.dtype,
            &stats.future2_position_load,
            &stats.future2_success_load,
            &stats.future2_position_read_seconds,
            &stats.future2_success_read_seconds,
            &stats.future2_index_seconds
        );
    ++stats.future2_batch_loads;
    stats.future2_cells_loaded += future2_layer.position.cell_count();
    stats.future2_active_cells_max = std::max<uint64_t>(
        stats.future2_active_cells_max,
        future2_layer.position.cell_count()
    );
    const uint64_t future2_position_bytes =
        static_cast<uint64_t>(future2_layer.position.bytes().size());
    const uint64_t future2_success_bytes =
        kBCSuccessHeaderBytes +
        static_cast<uint64_t>(future2_layer.success_value_count()) *
            bc_success_dtype_value_size(options.solve.dtype);
    stats.future2_position_resident_bytes = std::max(
        stats.future2_position_resident_bytes,
        future2_position_bytes
    );
    stats.future2_success_resident_bytes = std::max(
        stats.future2_success_resident_bytes,
        future2_success_bytes
    );
    stats.future_resident_layers_max = std::max<uint64_t>(stats.future_resident_layers_max, 1U);
    stats.future_resident_bytes_max = std::max<uint64_t>(
        stats.future_resident_bytes_max,
        future2_position_bytes + future2_success_bytes
    );

    uint32_t pass2_chunks = 0U;
    BCSingleChunkFinalFileStreamer<StorageT> output_streamer(
        current_position,
        position_file,
        success_file,
        options.solve.row_width,
        options.solve.dtype,
        stats
    );

    for (FamilyId row_begin = 0U; row_begin < family_count;) {
        const double current_select_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId> current_cids = bc_single_chunk_next_row_slab_cids(
            current_position,
            row_begin,
            options.current_chunk_rows,
            options.current_chunk_max_bytes,
            options.solve.row_width,
            value_size
        );
        stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_select_t0;
        if (current_cids.empty()) {
            break;
        }
        row_begin = static_cast<FamilyId>(current_matrix.row(current_cids.back()) + 1U);

        BCCellLoadStats current_load_stats;
        const double current_read_t0 = bc_single_chunk_now_seconds();
        std::vector<BCLoadedCell> current_cells =
            current_position.load_cells(current_cids, &current_load_stats);
        stats.current_position_read_seconds += bc_single_chunk_now_seconds() - current_read_t0;
        bc_single_chunk_add_cell_load_stats(stats.current_position_load, current_load_stats);
        ++stats.current_chunks;
        stats.current_cells += current_cells.size();

        const double current_layout_t0 = bc_single_chunk_now_seconds();
        std::vector<uint64_t> cell_offsets(current_cells.size() + 1U, 0U);
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const BCLoadedCell &cell = current_cells[i];
            if (cell.success_rows == 0U || cell.buckets.empty()) {
                ++stats.current_empty_cells;
            } else {
                ++stats.current_nonempty_cells;
            }
            cell_offsets[i + 1U] = bc_checked_add_u64(
                cell_offsets[i],
                cell.success_rows,
                "BC strict frontier pass2 current row offset overflow"
            );
        }
        std::vector<BCSingleChunkLoadedWorkItem> work_items =
            bc_single_chunk_build_loaded_work_items(lut, current_cells);
        stats.current_work_items += work_items.size();
        stats.current_plan_seconds += bc_single_chunk_now_seconds() - current_layout_t0;

        const uint64_t chunk_rows = cell_offsets.back();
        if (chunk_rows > std::numeric_limits<uint64_t>::max() / options.solve.row_width ||
            chunk_rows * options.solve.row_width >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC strict frontier pass2 raw value count exceeds size_t");
        }
        const size_t value_count = static_cast<size_t>(chunk_rows * options.solve.row_width);
        const double partial_read_t0 = bc_single_chunk_now_seconds();
        bc_single_chunk_read_tmp4_values(
            bc_single_chunk_partial_path(temp_dir, pass2_chunks),
            scratch.sum4_values,
            value_count,
            tmp_direct_io,
            &stats.partial_read_io
        );
        stats.partial_read_seconds += bc_single_chunk_now_seconds() - partial_read_t0;
        stats.partial_read_bytes = bc_checked_add_u64(
            stats.partial_read_bytes,
            bc_single_chunk_tmp4_file_bytes(scratch.sum4_values, tmp_direct_io),
            "BC strict frontier tmp4 read byte stats overflow"
        );

        scratch.raw_values.resize_uninitialized(value_count);
        bc_single_chunk_solve_phase_for_current_cells<StorageT>(
            current_cells,
            work_items,
            cell_offsets,
            scratch.raw_values,
            scratch.sum4_values,
            lut,
            future2_layer.position.axis(),
            future2_layer.lookup,
            recalc_options,
            BCSingleChunkSolvePhase::Spawn2,
            stats,
            &scratch.batch_workspaces
        );

        const double compact_t0 = bc_single_chunk_now_seconds();
        const int compact_threads = bc_resident_solve_effective_threads(options.solve.num_threads);
        std::vector<FinalizedCellPayload> payloads(current_cells.size());
        std::vector<BCResidentCompactStats> per_compact_thread(static_cast<size_t>(compact_threads));
#pragma omp parallel for schedule(dynamic, kBCResidentCellDynamicChunk) num_threads(compact_threads)
        for (int64_t i_signed = 0;
             i_signed < static_cast<int64_t>(current_cells.size());
             ++i_signed) {
#if defined(_OPENMP)
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            const size_t i = static_cast<size_t>(i_signed);
            const BCLoadedCell &cell = current_cells[i];
            bc_single_chunk_compact_loaded_cell_in_place<StorageT>(
                lut,
                cell,
                scratch.raw_values,
                cell_offsets[i],
                options.solve.row_width,
                options.solve.zero_value,
                payloads[i],
                per_compact_thread[static_cast<size_t>(tid)]
            );
        }
        for (const BCResidentCompactStats &compact_stats : per_compact_thread) {
            stats.compact_input_rows += compact_stats.input_rows;
            stats.compact_live_rows += compact_stats.live_rows;
            stats.compact_zero_pruned_rows += compact_stats.zero_pruned_rows;
            stats.compact_live_cells += compact_stats.live_cells;
            stats.compact_empty_cells += compact_stats.empty_cells;
        }
        uint64_t compact_chunk_values = 0U;
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[i];
            compact_chunk_values = bc_checked_add_u64(
                compact_chunk_values,
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width,
                "BC strict frontier compact chunk value count overflow"
            );
        }
        std::vector<StorageT> compact_values;
        compact_values.reserve(static_cast<size_t>(compact_chunk_values));
        for (size_t i = 0U; i < current_cells.size(); ++i) {
            const FinalizedCellPayload &payload = payloads[i];
            const uint64_t local_values =
                static_cast<uint64_t>(payload.success_rows) * options.solve.row_width;
            const uint64_t src_offset = cell_offsets[i] * static_cast<uint64_t>(options.solve.row_width);
            if (src_offset + local_values > scratch.raw_values.size()) {
                throw std::out_of_range("BC strict frontier compact values exceed raw buffer");
            }
            const StorageT *begin =
                scratch.raw_values.data() + static_cast<std::ptrdiff_t>(src_offset);
            compact_values.insert(
                compact_values.end(),
                begin,
                begin + static_cast<std::ptrdiff_t>(local_values)
            );
        }
        stats.compact_seconds += bc_single_chunk_now_seconds() - compact_t0;
        output_streamer.write_chunk(
            current_cells,
            payloads,
            compact_values
        );
        const double partial_cleanup_t0 = bc_single_chunk_now_seconds();
        std::error_code remove_partial_ec;
        std::filesystem::remove(bc_single_chunk_partial_path(temp_dir, pass2_chunks), remove_partial_ec);
        stats.partial_cleanup_seconds += bc_single_chunk_now_seconds() - partial_cleanup_t0;
        const double workspace_release_t0 = bc_single_chunk_now_seconds();
        std::vector<CellId>().swap(current_cids);
        std::vector<BCLoadedCell>().swap(current_cells);
        std::vector<uint64_t>().swap(cell_offsets);
        std::vector<BCSingleChunkLoadedWorkItem>().swap(work_items);
        std::vector<FinalizedCellPayload>().swap(payloads);
        std::vector<BCResidentCompactStats>().swap(per_compact_thread);
        std::vector<StorageT>().swap(compact_values);
        stats.workspace_release_seconds += bc_single_chunk_now_seconds() - workspace_release_t0;
        ++pass2_chunks;
    }
    if (pass1_chunks != pass2_chunks) {
        throw std::runtime_error("BC strict frontier pass chunk count mismatch");
    }

    BCSingleChunkSolveFileResult file_result = output_streamer.finish();
    BCSingleChunkStrictFrontierFileResult<StorageT> result;
    result.position_bytes = file_result.position_bytes;
    result.success_bytes = file_result.success_bytes;
    result.stats = stats;
    result.stats.output_values = file_result.stats.output_values;
    result.stats.output_bytes = file_result.stats.output_bytes;
    result.next_future4_layer = std::move(future2_layer);
    return result;
}

template <typename StorageT>
BCSingleChunkSolveFileResult bc_single_chunk_solve_compacted_layer_to_files(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    BCWritableFile &position_file,
    BCWritableFile &success_file,
    const BCSingleChunkSolveOptions<StorageT> &options,
    BCSingleChunkSolveWorkspace<StorageT> *workspace = nullptr
) {
    BCSingleChunkCompactedBuild<StorageT> build =
        bc_single_chunk_solve_compacted_build<StorageT>(
            current_position,
            future2_position,
            future2_success,
            future4_position,
            future4_success,
            options,
            workspace
        );

    BCSingleChunkSolveFileResult result;
    result.stats = build.stats;
    const double position_t0 = bc_single_chunk_now_seconds();
    bc_single_chunk_write_position_bytes<StorageT>(
        position_file,
        build.position_bytes,
        &result.stats.output_position_write
    );
    result.stats.position_write_seconds = bc_single_chunk_now_seconds() - position_t0;
    result.position_bytes = build.position_bytes.size();

    BCPositionLayerReader compact_position(build.position_bytes, current_position.lut());
    const double success_t0 = bc_single_chunk_now_seconds();
    result.success_bytes = write_success_values_to_file<StorageT>(
        success_file,
        compact_position,
        options.solve.row_width,
        options.solve.dtype,
        build.success_values,
        &result.stats.output_success_write
    );
    result.stats.success_write_seconds = bc_single_chunk_now_seconds() - success_t0;
    result.stats.output_bytes = result.success_bytes;
    const double release_t0 = bc_single_chunk_now_seconds();
    std::vector<uint8_t>().swap(build.position_bytes);
    std::vector<StorageT>().swap(build.success_values);
    result.stats.output_release_seconds = bc_single_chunk_now_seconds() - release_t0;
    return result;
}

template <typename StorageT>
BCSingleChunkSolveResult<StorageT> bc_single_chunk_solve_success_layer(
    const BCPositionStreamingReader &current_position,
    const BCPositionStreamingReader &future2_position,
    const BCSuccessStreamingReader &future2_success,
    const BCPositionStreamingReader &future4_position,
    const BCSuccessStreamingReader &future4_success,
    const BCSingleChunkSolveOptions<StorageT> &options
) {
    return bc_single_chunk_solve_compacted_layer<StorageT>(
        current_position,
        future2_position,
        future2_success,
        future4_position,
        future4_success,
        options
    );
}

} // namespace BC
