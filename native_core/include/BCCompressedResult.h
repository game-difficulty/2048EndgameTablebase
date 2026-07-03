#pragma once

#include "BCBoardOps.h"
#include "BCPositionCellLoader.h"
#include "BCSuccessIO.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace BCCompressedResult {

inline constexpr const char *kCompressedLayerFileExtension = ".bccmp";

struct CompressOptions {
    uint32_t bucket_block_raw_target_bytes = 32U * 1024U;
    uint32_t bucket_block_raw_hard_cap_bytes = 128U * 1024U;
    uint32_t value_block_raw_target_bytes = 256U * 1024U;
    uint32_t value_block_raw_hard_cap_bytes = 1024U * 1024U;
    uint32_t compression_level = 1U;
    uint32_t worker_count = 0U;
};

struct CompressStats {
    uint64_t cells = 0U;
    uint64_t non_empty_cells = 0U;
    uint64_t bucket_blocks = 0U;
    uint64_t value_blocks = 0U;
    uint64_t success_values = 0U;
    uint64_t live_rows = 0U;
    uint64_t bucket_raw_bytes = 0U;
    uint64_t bucket_compressed_bytes = 0U;
    uint64_t value_raw_bytes = 0U;
    uint64_t value_compressed_bytes = 0U;
    uint64_t output_bytes = 0U;
    uint64_t original_position_bytes = 0U;
    uint64_t original_success_bytes = 0U;
    double read_seconds = 0.0;
    double write_seconds = 0.0;
    double compress_worker_seconds = 0.0;
    double total_seconds = 0.0;
};

struct ColdLookupResult {
    bool found = false;
    uint32_t dtype = BC::kBCSuccessDTypeUint32;
    uint32_t row_width = 0U;
    uint64_t raw_value_bits = 0U;
    double numeric_value = 0.0;
    BC::CellId cid = 0U;
    uint32_t local_success_row = 0U;
    uint64_t value_index = 0U;
    uint64_t bucket_block_raw_bytes = 0U;
    uint64_t bucket_block_compressed_bytes = 0U;
    uint64_t value_block_raw_bytes = 0U;
    uint64_t value_block_compressed_bytes = 0U;
};

CompressStats compress_exact_layer_to_result(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    const BC::BCLut &lut,
    const std::filesystem::path &output_path,
    const CompressOptions &options = {}
);

CompressStats compress_in_memory_layer_to_result(
    const BC::BCPositionLayerReader &position,
    const BC::BCSuccessLayerReader &success,
    const std::filesystem::path &output_path,
    const CompressOptions &options = {}
);

CompressStats compress_flat_success_layer_to_result(
    const BC::BCPositionLayerReader &position,
    const void *success_values,
    uint64_t success_value_count,
    uint32_t row_width,
    BC::BCSuccessDTypeMode dtype,
    const std::filesystem::path &output_path,
    const CompressOptions &options = {}
);

class PointReader {
public:
    PointReader() = default;
    PointReader(
        const std::filesystem::path &compressed_path,
        const BC::BCLut &lut
    );

    void open(
        const std::filesystem::path &compressed_path,
        const BC::BCLut &lut
    );

    [[nodiscard]] ColdLookupResult lookup(uint64_t board, uint32_t lane = 0U) const;
    [[nodiscard]] bool sample(
        uint64_t &board,
        uint64_t &raw_value_bits,
        double &numeric_value,
        uint32_t lane = 0U
    ) const;

    [[nodiscard]] uint32_t row_width() const;
    [[nodiscard]] BC::BCSuccessDTypeMode dtype_mode() const;
    [[nodiscard]] uint64_t layer_sum() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class StreamingBuilder {
public:
    StreamingBuilder() = default;
    StreamingBuilder(
        const BC::BCPositionStreamingReader &position,
        uint32_t row_width,
        BC::BCSuccessDTypeMode dtype,
        const std::filesystem::path &output_path,
        const CompressOptions &options = {}
    );
    ~StreamingBuilder();

    StreamingBuilder(const StreamingBuilder &) = delete;
    StreamingBuilder &operator=(const StreamingBuilder &) = delete;
    StreamingBuilder(StreamingBuilder &&) noexcept;
    StreamingBuilder &operator=(StreamingBuilder &&) noexcept;

    void open(
        const BC::BCPositionStreamingReader &position,
        uint32_t row_width,
        BC::BCSuccessDTypeMode dtype,
        const std::filesystem::path &output_path,
        const CompressOptions &options = {}
    );

    void write_cell(
        BC::CellId cid,
        const BC::FinalizedCellPayload &payload,
        const void *success_values,
        uint64_t success_value_count,
        std::shared_ptr<const void> owner = {}
    );

    [[nodiscard]] CompressStats finish();
    [[nodiscard]] bool is_open() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] ColdLookupResult lookup_cold(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut,
    uint64_t board,
    uint32_t lane = 0U
);

bool sample_cold(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value,
    uint32_t lane = 0U
);

} // namespace BCCompressedResult
