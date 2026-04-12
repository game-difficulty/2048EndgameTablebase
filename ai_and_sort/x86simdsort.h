#ifndef X86_SIMD_SORT
#define X86_SIMD_SORT
#include <stdint.h>
#include <cstddef>

// 保持符号导出/隐藏逻辑，这对于生成正确的 DLL/库 符号至关重要
#if defined(_MSC_VER)
#define XSS_EXPORT_SYMBOL __declspec(dllexport)
#define XSS_HIDE_SYMBOL
#else
#define XSS_EXPORT_SYMBOL __attribute__((visibility("default")))
#define XSS_HIDE_SYMBOL __attribute__((visibility("hidden")))
#endif

namespace x86simdsort {

template <typename T>
XSS_EXPORT_SYMBOL void
qsort(T *arr, size_t arrsize, bool hasnan = false, bool descending = false);

} // namespace x86simdsort
#endif