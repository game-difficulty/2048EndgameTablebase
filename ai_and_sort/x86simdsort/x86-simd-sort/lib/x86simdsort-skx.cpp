// SKX specific routines:

#include "x86simdsort-static-incl.h"
#include "x86simdsort-internal.h"

// 重新定义宏，强制只生成 qsort 的机器码
#define DEFINE_QSORT_ONLY(type) \
    template <> \
    void qsort(type *arr, size_t arrsize, bool hasnan, bool descending) \
    { \
        x86simdsortStatic::qsort(arr, arrsize, hasnan, descending); \
    }

namespace xss {
namespace avx512 {
    /**
     * 这里是针对你的 AMD 9950X 最关键的一行。
     * 它将只为 uint64_t 的快速排序生成 AVX-512 机器码。
     * 删除了：uint32_t, int32_t, float, int64_t, double 
     * 删除了：所有 Key-Value 组合代码（那部分代码占据了原库约 70% 的体积）
     */
    DEFINE_QSORT_ONLY(uint64_t)

} // namespace avx512
} // namespace xss