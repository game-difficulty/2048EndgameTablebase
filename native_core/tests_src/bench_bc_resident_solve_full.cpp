#include "bench_bc_solve_runner_wrapper.h"

int main(int argc, char **argv) {
    return BCSolveBench::run_solve_runner_bench(
        argc,
        argv,
        "bc_resident_solve_full_bench",
        BC::BCSolveRoute::Resident,
        false
    );
}
