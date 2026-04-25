#ifndef X86_SIMD_SORT_STATIC_METHODS
#define X86_SIMD_SORT_STATIC_METHODS
#include <stdlib.h>
#include "xss-common-includes.h"

// Supported methods declared here for a quick reference:
namespace x86simdsortStatic {
template <typename T>
X86_SIMD_SORT_FINLINE void
qsort(T *arr, size_t size, bool hasnan = false, bool descending = false);
} // namespace x86simdsortStatic

#define XSS_METHODS(ISA) \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::qsort( \
            T *arr, size_t size, bool hasnan, bool descending) \
    { \
        ISA##_qsort(arr, size, hasnan, descending); \
    }

/*
 * qsort template functions.
 */
#include "xss-common-qsort.h"

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
/* 64-bit dtypes vector definitions on SKX */
#include "avx512-64bit-qsort.hpp"

XSS_METHODS(avx512)

#elif defined(__AVX2__)
/* 64-bit dtypes vector definitions on AVX2 */
#include "avx2-64bit-qsort.hpp"
XSS_METHODS(avx2)

#else
#error "x86simdsortStatic methods needs to be compiled with avx512/avx2 specific flags"
#endif // (__AVX512VL__ && __AVX512DQ__) || AVX2

#endif // X86_SIMD_SORT_STATIC_METHODS