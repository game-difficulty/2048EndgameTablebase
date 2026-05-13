#ifndef X86_SIMD_SORT_STATIC_METHODS
#define X86_SIMD_SORT_STATIC_METHODS

#include <stdlib.h>
#include <vector>

#include "xss-common-includes.h"

namespace x86simdsortStatic {
template <typename T>
X86_SIMD_SORT_FINLINE void
qsort(T *arr, size_t size, bool hasnan = false, bool descending = false);

template <typename T>
X86_SIMD_SORT_FINLINE std::vector<size_t> argsort(const T *arr,
                                                  size_t size,
                                                  bool hasnan = false,
                                                  bool descending = false);

template <typename T>
X86_SIMD_SORT_FINLINE void argsort(const T *arr,
                                   size_t *arg,
                                   size_t size,
                                   bool hasnan = false,
                                   bool descending = false);

} // namespace x86simdsortStatic

#define XSS_QSORT_AND_ARGSORT_METHODS(ISA) \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::qsort( \
            T *arr, size_t size, bool hasnan, bool descending) \
    { \
        ISA##_qsort(arr, size, hasnan, descending); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE void x86simdsortStatic::argsort(const T *arr, \
                                                          size_t *arg, \
                                                          size_t size, \
                                                          bool hasnan, \
                                                          bool descending) \
    { \
        ISA##_argsort(arr, arg, size, hasnan, descending); \
    } \
    template <typename T> \
    X86_SIMD_SORT_FINLINE std::vector<size_t> x86simdsortStatic::argsort( \
            const T *arr, size_t size, bool hasnan, bool descending) \
    { \
        std::vector<size_t> indices(size); \
        std::iota(indices.begin(), indices.end(), 0); \
        x86simdsortStatic::argsort( \
                arr, indices.data(), size, hasnan, descending); \
        return indices; \
    }

#include "xss-common-qsort.h"
#include "xss-common-argsort.h"

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"
#include "avx512-64bit-argsort.hpp"
XSS_QSORT_AND_ARGSORT_METHODS(avx512)

#elif defined(__AVX2__)
#include "avx2-32bit-half.hpp"
#include "avx2-32bit-qsort.hpp"
#include "avx2-64bit-qsort.hpp"
XSS_QSORT_AND_ARGSORT_METHODS(avx2)

#else
#error "x86simdsortStatic methods needs to be compiled with avx512/avx2 specific flags"
#endif

#endif
