#ifndef XSS_SCALAR_H
#define XSS_SCALAR_H

#include "custom-compare.h"
#include <algorithm>
#include <functional>

namespace xss {
namespace utils {
    template <typename T>
    auto get_cmp_func(bool hasnan, bool reverse)
    {
        // 注意：这里我们不再手动指定返回类型，让编译器通过 auto 推导
        if (hasnan) {
            if (reverse) { return (std::function<bool(T, T)>)compare<T, std::greater<T>>(); }
            else { return (std::function<bool(T, T)>)compare<T, std::less<T>>(); }
        }
        else {
            if (reverse) { return (std::function<bool(T, T)>)std::greater<T>(); }
            else { return (std::function<bool(T, T)>)std::less<T>(); }
        }
    }
} // namespace utils

namespace scalar {
    template <typename T>
    void qsort(T *arr, size_t arrsize, bool hasnan, bool reversed)
    {
        std::sort(arr,
                  arr + arrsize,
                  xss::utils::get_cmp_func<T>(hasnan, reversed));
    }
} // namespace scalar
} // namespace xss
#endif