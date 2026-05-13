#include "BookSolver.h"

#include "EXPrefix36Runtime.h"

void run_pattern_solve_zmask_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    EXPrefix36Runtime::run_pattern_solve(arr_init, spec, options);
}

void run_pattern_solve_zmask_single_layer_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options,
    int step
) {
    EXPrefix36Runtime::run_pattern_solve_single_layer(arr_init, spec, options, step);
}
