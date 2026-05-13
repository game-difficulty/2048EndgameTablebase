#include "BookSolver.h"

#include "EXPrefix36Runtime.h"
#include "Formation.h"

void run_pattern_build_zmask_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    FormationProgress::reset_build_progress(classic_build_progress_total(options));
    EXPrefix36Runtime::run_pattern_build(arr_init, spec, options);
}
