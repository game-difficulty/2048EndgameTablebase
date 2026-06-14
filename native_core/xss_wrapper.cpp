#include "x86simdsort.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "NativeDiagnostics.h"
#include "UniqueUtils.h"

extern "C" void xss_avx2_keyvalue_sort_uint64_uint32(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
);

#if defined(_WIN32)
#define SORT_WRAPPER_EXPORT extern "C" __declspec(dllexport)
#else
#define SORT_WRAPPER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

template <typename T>
void fallback_sort(T *arr, size_t arrsize, bool descending) {
    if (arr == nullptr || arrsize < 2) {
        return;
    }

    if (descending) {
        std::sort(arr, arr + arrsize, std::greater<T>());
        return;
    }
    std::sort(arr, arr + arrsize);
}

template <typename T>
void fallback_argsort(const T *arr, size_t arrsize, size_t *indices, bool descending) {
    if (indices == nullptr) {
        return;
    }

    std::iota(indices, indices + arrsize, size_t {0});
    if (arr == nullptr || arrsize < 2) {
        return;
    }

    if (descending) {
        std::sort(indices, indices + arrsize, [arr](size_t lhs, size_t rhs) {
            return arr[lhs] > arr[rhs];
        });
        return;
    }

    std::sort(indices, indices + arrsize, [arr](size_t lhs, size_t rhs) {
        return arr[lhs] < arr[rhs];
    });
}

template <typename T>
void simd_sort(T *arr, size_t arrsize, bool descending) {
    if (arr == nullptr || arrsize < 2) {
        return;
    }

    if (!UniqueUtils::cpu_has_avx2()) {
        fallback_sort(arr, arrsize, descending);
        return;
    }

    x86simdsort::qsort<T>(arr, arrsize, false, descending);
}

template <typename T>
void simd_argsort(const T *arr, size_t arrsize, size_t *indices, bool descending) {
    if (indices == nullptr) {
        return;
    }

    if (arr == nullptr || arrsize < 2 || !UniqueUtils::cpu_has_avx2()) {
        fallback_argsort(arr, arrsize, indices, descending);
        return;
    }

    std::vector<size_t> args = x86simdsort::argsort<T>(arr, arrsize, false, descending);
    std::memcpy(indices, args.data(), args.size() * sizeof(size_t));
}

void fallback_keyvalue_sort_uint64_uint32(uint64_t *keys, uint32_t *values, size_t count, bool descending) {
    if (keys == nullptr || values == nullptr || count < 2) {
        return;
    }
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), size_t {0});
    if (descending) {
        std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
            return keys[lhs] > keys[rhs];
        });
    } else {
        std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
            return keys[lhs] < keys[rhs];
        });
    }

    std::vector<uint64_t> sorted_keys(count);
    std::vector<uint32_t> sorted_values(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_keys[i] = keys[order[i]];
        sorted_values[i] = values[order[i]];
    }
    std::memcpy(keys, sorted_keys.data(), count * sizeof(uint64_t));
    std::memcpy(values, sorted_values.data(), count * sizeof(uint32_t));
}

void simd_keyvalue_sort_uint64_uint32(uint64_t *keys, uint32_t *values, size_t count, bool descending) {
    if (keys == nullptr || values == nullptr || count < 2) {
        return;
    }
    if (!UniqueUtils::cpu_has_avx2()) {
        fallback_keyvalue_sort_uint64_uint32(keys, values, count, descending);
        return;
    }
    xss_avx2_keyvalue_sort_uint64_uint32(keys, values, count, descending);
}

} // namespace

SORT_WRAPPER_EXPORT void sort_uint32(uint32_t *arr, size_t arrsize, bool descending) {
    NativeDiagnostics::install_crash_handler("bookgen_native");
    NativeDiagnostics::Scope scope(
        "bookgen_native.sort_uint32 len=" + std::to_string(arrsize) +
        " descending=" + (descending ? "1" : "0")
    );
    simd_sort(arr, arrsize, descending);
}

SORT_WRAPPER_EXPORT void sort_uint64(uint64_t *arr, size_t arrsize, bool descending) {
    NativeDiagnostics::install_crash_handler("bookgen_native");
    NativeDiagnostics::Scope scope(
        "bookgen_native.sort_uint64 len=" + std::to_string(arrsize) +
        " descending=" + (descending ? "1" : "0")
    );
    simd_sort(arr, arrsize, descending);
}

SORT_WRAPPER_EXPORT void argsort_uint32(const uint32_t *arr, size_t arrsize, size_t *indices, bool descending) {
    NativeDiagnostics::install_crash_handler("bookgen_native");
    NativeDiagnostics::Scope scope(
        "bookgen_native.argsort_uint32 len=" + std::to_string(arrsize) +
        " descending=" + (descending ? "1" : "0")
    );
    simd_argsort(arr, arrsize, indices, descending);
}

SORT_WRAPPER_EXPORT void argsort_uint64(const uint64_t *arr, size_t arrsize, size_t *indices, bool descending) {
    NativeDiagnostics::install_crash_handler("bookgen_native");
    NativeDiagnostics::Scope scope(
        "bookgen_native.argsort_uint64 len=" + std::to_string(arrsize) +
        " descending=" + (descending ? "1" : "0")
    );
    simd_argsort(arr, arrsize, indices, descending);
}

SORT_WRAPPER_EXPORT void keyvalue_sort_uint64_uint32(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
) {
    NativeDiagnostics::install_crash_handler("bookgen_native");
    NativeDiagnostics::Scope scope(
        "bookgen_native.keyvalue_sort_uint64_uint32 len=" + std::to_string(count) +
        " descending=" + (descending ? "1" : "0")
    );
    simd_keyvalue_sort_uint64_uint32(keys, values, count, descending);
}
