#include "UniqueUtils.h"

#if defined(_WIN32)
#define BOOKGEN_NATIVE_EXPORT extern "C" __declspec(dllexport)
#else
#define BOOKGEN_NATIVE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

BOOKGEN_NATIVE_EXPORT int bookgen_native_has_avx512() {
    return UniqueUtils::cpu_has_avx512() ? 1 : 0;
}
