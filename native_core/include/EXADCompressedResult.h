#pragma once

#include "FormationRuntime.h"

#include <cstdint>
#include <string>

namespace EXADCompressedResult {

inline constexpr const char* kCompressedLayerFileExtension = ".exadzbook";

struct CompressStats {
    std::string source_path;
    std::string output_path;
    uint64_t original_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t live_board_count = 0;
    uint64_t success_value_count = 0;
    uint64_t bucket_raw_bytes = 0;
    uint64_t bucket_compressed_bytes = 0;
    uint64_t value_raw_bytes = 0;
    uint64_t value_compressed_bytes = 0;
    uint64_t bucket_block_count = 0;
    uint64_t value_block_count = 0;
    double compression_seconds = 0.0;
};

struct ColdLookupResult {
    bool found = false;
    uint64_t value_index = 0;
    uint64_t local_row = 0;
    uint64_t raw_value_bits = 0;
    double numeric_value = 0.0;
    SuccessRateKind success_kind = SuccessRateKind::UInt32;
    uint64_t bucket_block_raw_bytes = 0;
    uint64_t value_block_raw_bytes = 0;
    uint64_t bucket_block_compressed_bytes = 0;
    uint64_t value_block_compressed_bytes = 0;
};

bool is_exad_compressed_file(const std::string& path);

CompressStats compress_exad_solved_layer_to_result(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    const std::string& output_path,
    uint32_t bucket_block_raw_target_bytes = 512u * 1024u,
    uint32_t success_block_values = 65536u,
    int compression_level = 5);

ColdLookupResult lookup_exad_cold(
    const std::string& compressed_path,
    const std::string& exadlut_path,
    int ad_key,
    uint64_t canonical_board,
    uint32_t column);

ColdLookupResult lookup_exadbook_cold(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    int ad_key,
    uint64_t canonical_board,
    uint32_t column);

bool sample_exad_cold(
    const std::string& compressed_path,
    const std::string& exadlut_path,
    uint64_t& board);

bool sample_exadbook_cold(
    const std::string& exadbook_path,
    const std::string& exadlut_path,
    uint64_t& board);

} // namespace EXADCompressedResult
