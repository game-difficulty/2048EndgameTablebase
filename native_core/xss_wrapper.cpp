#include "x86simdsort.h"
#include <cstdint>

#if defined(_WIN32)
#define SORT_WRAPPER_EXPORT extern "C" __declspec(dllexport)
#else
#define SORT_WRAPPER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

SORT_WRAPPER_EXPORT void sort_uint64(uint64_t *arr, size_t arrsize, bool descending) {
        x86simdsort::qsort<uint64_t>(arr, arrsize, false, descending);
}

// This wrapper is linked into the canonical `bookgen_native` target.
// Build x86-simd-sort separately to produce `libx86simdsortcpp.a`, then link
// this translation unit through `cmake --build ... --target bookgen_native`.
