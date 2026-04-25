#pragma once

#include <tuple>
#include <vector>

#include "FormationRuntime.h"

std::tuple<bool, PatternLayer, PatternLayer> run_pattern_generate_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
);

void run_pattern_solve_cpp(
    const PatternLayer &d1,
    const PatternLayer &d2,
    const PatternSpec &spec,
    const RunOptions &options
);

void run_pattern_build_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
);

void run_pattern_build_ad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
);

void run_pattern_solve_ad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    bool started_from_generate = false
);
