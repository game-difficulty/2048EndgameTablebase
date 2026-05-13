#ifndef XSS_INTERNAL_METHODS
#define XSS_INTERNAL_METHODS

#include "x86simdsort.h"
#include <stdint.h>
#include <vector>

#define DECLARE_SORT_AND_ARGSORT(name) \
    namespace name { \
    template <typename T> \
    XSS_HIDE_SYMBOL void qsort(T *arr, \
                               size_t arrsize, \
                               bool hasnan = false, \
                               bool descending = false); \
    template <typename T> \
    XSS_HIDE_SYMBOL std::vector<size_t> argsort(const T *arr, \
                                                size_t arrsize, \
                                                bool hasnan = false, \
                                                bool descending = false); \
    }

namespace xss {
DECLARE_SORT_AND_ARGSORT(avx512)
DECLARE_SORT_AND_ARGSORT(avx2)
DECLARE_SORT_AND_ARGSORT(scalar)
} // namespace xss

#endif
