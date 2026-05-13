// SKX specific routines:

#include "x86simdsort-static-incl.h"
#include "x86simdsort-internal.h"

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
namespace avx512 {
    DEFINE_SORT_AND_ARGSORT(uint32_t)
    DEFINE_SORT_AND_ARGSORT(uint64_t)
} // namespace avx512
} // namespace xss
