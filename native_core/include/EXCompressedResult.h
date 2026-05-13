#pragma once

#include "FormationRuntime.h"

#include <cstdint>
#include <string>

namespace EXCompressedResult {

constexpr const char *kCompressedLayerFileExtension = ".exzbook";

struct CompressStats {
    uint64_t original_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t bucket_block_count = 0;
    uint64_t success_block_count = 0;
    uint64_t bucket_raw_bytes = 0;
    uint64_t bucket_compressed_bytes = 0;
    uint64_t success_raw_bytes = 0;
    uint64_t success_compressed_bytes = 0;
    double compress_seconds = 0.0;

    [[nodiscard]] double ratio() const {
        return original_bytes == 0 ? 0.0 : static_cast<double>(compressed_bytes) / static_cast<double>(original_bytes);
    }

    [[nodiscard]] double save_ratio() const {
        return 1.0 - ratio();
    }
};

struct ColdLookupResult {
    bool found = false;
    uint64_t global_dense_index = 0;
    uint64_t raw_value_bits = 0;
    double numeric_value = 0.0;
    SuccessRateKind success_kind = SuccessRateKind::UInt32;
    uint64_t bucket_block_raw_bytes = 0;
    uint64_t success_block_raw_bytes = 0;
    uint64_t bucket_block_compressed_bytes = 0;
    uint64_t success_block_compressed_bytes = 0;
};

struct Prefix36LayerView {
    uint32_t layer_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t dtype_mode = 0;
    SuccessRateKind success_kind = SuccessRateKind::UInt32;
    uint32_t value_size = sizeof(uint32_t);
    uint64_t live_board_count = 0;

    const uint64_t *bucket_keys = nullptr;
    const uint32_t *bitmap_offsets = nullptr;
    const uint32_t *success_offsets = nullptr;
    uint64_t bucket_count = 0;

    const uint8_t *small_bitmap_bytes = nullptr;
    uint64_t small_bitmap_byte_count = 0;
    const uint64_t *large_bitmap_words = nullptr;
    uint64_t large_bitmap_word_count = 0;

    const uint32_t *success_values = nullptr;
    uint64_t success_value_count = 0;
};

CompressStats compress_zbook_to_ex_result(
    const std::string &zbook_path,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets = 4096,
    uint32_t success_block_values = 65536,
    int compression_level = 5
);

CompressStats compress_prefix36_layer_view_to_ex_result(
    const Prefix36LayerView &layer,
    const std::string &zlut_path,
    const std::string &output_path,
    uint32_t bucket_block_buckets = 4096,
    uint32_t success_block_values = 65536,
    int compression_level = 5
);

ColdLookupResult lookup_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t board
);

ColdLookupResult lookup_zbook_cold(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t board
);

bool sample_cold(
    const std::string &compressed_path,
    const std::string &zlut_path,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value
);

} // namespace EXCompressedResult
