#ifndef XSS_INTERNAL_METHODS
#define XSS_INTERNAL_METHODS
#include "x86simdsort.h"
#include <stdint.h>

/**
 * 极简版宏定义：只声明 qsort。
 * 删除了 keyvalue, qselect, partial_sort, argsort, argselect 所有的声明。
 * 删除了对 <vector> 的依赖。
 */
#define DECLARE_QSORT_ONLY(name) \
    namespace name { \
    template <typename T> \
    XSS_HIDE_SYMBOL void qsort(T *arr, \
                               size_t arrsize, \
                               bool hasnan = false, \
                               bool descending = false); \
    }

namespace xss {
/* * 仅为各个指令集命名空间声明 qsort 模板。
 * 即使某些命名空间（如 fp16_spr）现在是空的，保留声明也不会增加体积，
 * 只要不去实例化它们即可。
 */
DECLARE_QSORT_ONLY(avx512)
DECLARE_QSORT_ONLY(avx2)
DECLARE_QSORT_ONLY(scalar)
DECLARE_QSORT_ONLY(fp16_spr)
DECLARE_QSORT_ONLY(fp16_icl)
} // namespace xss
#endif
