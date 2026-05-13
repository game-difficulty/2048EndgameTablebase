// AVX2 specific routines:

#include "x86simdsort-static-incl.h"
#include "x86simdsort-internal.h"
#include "../src/xss-common-keyvaluesort.hpp"

#define DEFINE_SORT_AND_ARGSORT(type) \
    template <> \
    void qsort(type *arr, size_t arrsize, bool hasnan, bool descending) \
    { \
        x86simdsortStatic::qsort(arr, arrsize, hasnan, descending); \
    } \
    template <> \
    std::vector<size_t> argsort( \
            const type *arr, size_t arrsize, bool hasnan, bool descending) \
    { \
        return x86simdsortStatic::argsort(arr, arrsize, hasnan, descending); \
    }

namespace xss {
namespace avx2 {
    DEFINE_SORT_AND_ARGSORT(uint32_t)
    DEFINE_SORT_AND_ARGSORT(uint64_t)
} // namespace avx2
} // namespace xss

extern "C" void xss_avx2_keyvalue_sort_uint64_uint32(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
) {
    avx2_qsort_kv<uint64_t, uint32_t>(keys, values, static_cast<arrsize_t>(count), false, descending);
}
