#pragma once

#include "FormationRuntime.h"
#include "EXCompressedResult.h"

#include <cstdint>
#include <string>
#include <vector>

namespace EXPrefix36Runtime {

constexpr const char *kLayoutName = "prefix36_suffix28";
constexpr const char *kDirectIndexName = "entry";
constexpr const char *kLayerFileExtension = ".zbook";

std::string layer_file_path(const std::string &pathname, int step);

void run_pattern_build(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
);

void run_pattern_solve(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
);

void run_pattern_solve_single_layer(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options,
    int step
);

EXCompressedResult::ColdLookupResult lookup_zbook_cold(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t board
);

bool sample_zbook_state(
    const std::string &zbook_path,
    const std::string &zlut_path,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value
);

} // namespace EXPrefix36Runtime
