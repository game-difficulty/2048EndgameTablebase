#include <cstddef>
#include <cstdint>

#include "UniqueUtils.h"

#if defined(_WIN32)
#define UNIQUE_WRAPPER_EXPORT extern "C" __declspec(dllexport)
#else
#define UNIQUE_WRAPPER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

UNIQUE_WRAPPER_EXPORT size_t unique_sorted_u64_inplace(uint64_t *data, size_t size) {
    return UniqueUtils::unique_sorted_u64_inplace(data, size);
}

UNIQUE_WRAPPER_EXPORT int unique_sorted_u64_has_avx512() {
    return UniqueUtils::cpu_has_avx512() ? 1 : 0;
}
