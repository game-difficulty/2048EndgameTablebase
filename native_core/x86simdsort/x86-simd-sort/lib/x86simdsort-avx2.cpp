// AVX2 specific routines:

#include "x86simdsort-static-incl.h"
#include "x86simdsort-internal.h"

// 重新定义宏，仅保留 qsort 功能
#define DEFINE_QSORT_ONLY(type) \
    template <> \
    void qsort(type *arr, size_t arrsize, bool hasnan, bool descending) \
    { \
        x86simdsortStatic::qsort(arr, arrsize, hasnan, descending); \
    }

namespace xss {
namespace avx2 {
    /**
     * 只保留 uint64_t 的快速排序
     * 这样编译器就不会为 int32, float, double 以及所有的 Key-Value 组合生成 AVX2 代码
     */
    DEFINE_QSORT_ONLY(uint64_t)
} // namespace avx2
} // namespace xss