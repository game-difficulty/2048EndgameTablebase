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
    uint32_t value_block_raw_target_bytes = 4U * 1024U;
    uint32_t value_block_raw_hard_cap_bytes = 16U * 1024U;
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
