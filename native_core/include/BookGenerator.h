#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "BookGeneratorUtils.h"
#include "FormationRuntime.h"

namespace BookGenerator {

struct GenBoardsResult {
    size_t total_arr1 = 0;
    size_t total_arr2 = 0;
    std::vector<size_t> counts1;
    std::vector<size_t> counts2;
};

struct GenBoardsBigResult {
    std::vector<std::vector<uint64_t>> arr1s;
    std::vector<std::vector<uint64_t>> arr2s;
    std::vector<uint64_t> hashmap1;
    std::vector<uint64_t> hashmap2;
    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
};

template <typename Mover>
GenBoardsResult gen_boards(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    uint64_t *hashmap1, size_t hashmap1_size,
    uint64_t *hashmap2, size_t hashmap2_size,
    uint64_t *arr1, uint64_t *arr2,
    size_t capacity,
    int num_threads,
    bool do_check,
    bool isfree
);

template <typename Mover>
std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> gen_boards_simple(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    bool do_check,
    bool isfree
);

GenBoardsBigResult
gen_boards_big(
    const std::vector<uint64_t> &arr0,
    int target,
    const PatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    int num_threads,
    std::vector<std::vector<double>> &length_factors_list,
    double &length_factor_multiplier,
    bool do_check,
    bool isfree,
    bool is_variant
);

std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> generate_process(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
);

double predict_next_length_factor_quadratic(const std::vector<double> &length_factors);
std::vector<double> harmonic_mean_by_column(const std::vector<std::vector<double>> &matrix);
double get_system_memory_gb();
void update_hashmap_length(std::vector<uint64_t> &hashmap, size_t current_arr_size);

} // namespace BookGenerator
