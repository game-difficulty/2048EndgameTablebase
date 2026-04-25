#pragma once
#ifndef __EMSCRIPTEN_OMP_STUB__
#define __EMSCRIPTEN_OMP_STUB__

static inline void omp_set_num_threads(int) {}
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline double omp_get_wtime() { return 0.0; } 

#endif
