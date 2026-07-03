#include "bench_bc_generation_runner_wrapper.h"

int main(int argc, char **argv) {
    return BCGenerationBench::run_generation_runner_bench(
        argc,
        argv,
        "bc_resident_generation_bench",
        BC::BCFamilyGenerationRoute::Resident,
        false
    );
}
