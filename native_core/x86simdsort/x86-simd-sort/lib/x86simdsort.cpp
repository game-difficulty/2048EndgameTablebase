#if defined(_MSC_VER)
#define XSS_ATTRIBUTE_CONSTRUCTOR
#else
#define XSS_ATTRIBUTE_CONSTRUCTOR __attribute__((constructor))
#endif

#include "x86simdsort.h"
#include "x86simdsort-internal.h"
#include "x86simdsort-scalar.h"
#include "x86simdsortcpuid.h"
#include <string>

static int check_cpu_feature_support(std::string_view cpufeature) {
    const char *disable_avx512 = std::getenv("XSS_DISABLE_AVX512");
    if ((cpufeature == "avx512_icl") && (!disable_avx512)) {
        return xss_cpu_supports("avx512f") && xss_cpu_supports("avx512vbmi2")
                && xss_cpu_supports("avx512bw") && xss_cpu_supports("avx512vl");
    }
    else if ((cpufeature == "avx512_skx") && (!disable_avx512)) {
        return xss_cpu_supports("avx512f") && xss_cpu_supports("avx512dq")
                && xss_cpu_supports("avx512vl");
    }
    else if (cpufeature == "avx2") {
        return xss_cpu_supports("avx2");
    }
    return 0;
}

static std::string_view find_preferred_cpu(std::initializer_list<std::string_view> cpulist) {
    for (auto cpu : cpulist) {
        if (check_cpu_feature_support(cpu)) {
            return cpu;
        }
    }
    return "scalar";
}

constexpr bool dispatch_requested(std::string_view cpurequested,
                                  std::initializer_list<std::string_view> cpulist) {
    for (auto cpu : cpulist) {
        if (cpu.find(cpurequested) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

namespace x86simdsort {

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define ISA_LIST(...) std::initializer_list<std::string_view> { __VA_ARGS__ }

#define PRE_DECLARE_QSORT(TYPE) \
    template <> XSS_EXPORT_SYMBOL void qsort<TYPE>(TYPE *, size_t, bool, bool);

#define PRE_DECLARE_ARGSORT(TYPE) \
    template <> XSS_EXPORT_SYMBOL std::vector<size_t> argsort<TYPE>(const TYPE *, size_t, bool, bool);

#define DEFINE_QSORT_DISPATCHER(TYPE, ISA) \
    static void (*internal_qsort##TYPE)(TYPE *, size_t, bool, bool) = nullptr; \
    static XSS_ATTRIBUTE_CONSTRUCTOR void CAT(resolve_qsort_, TYPE)(void) { \
        xss_cpu_init(); \
        std::string_view preferred_cpu = find_preferred_cpu(ISA); \
        internal_qsort##TYPE = &xss::scalar::qsort<TYPE>; \
        if constexpr (dispatch_requested("avx512", ISA)) { \
            if (preferred_cpu.find("avx512") != std::string_view::npos) { \
                internal_qsort##TYPE = &xss::avx512::qsort<TYPE>; \
                return; \
            } \
        } \
        if constexpr (dispatch_requested("avx2", ISA)) { \
            if (preferred_cpu.find("avx2") != std::string_view::npos) { \
                internal_qsort##TYPE = &xss::avx2::qsort<TYPE>; \
            } \
        } \
    }

#define DEFINE_ARGSORT_DISPATCHER(TYPE, ISA) \
    static std::vector<size_t> (*internal_argsort##TYPE)(const TYPE *, size_t, bool, bool) = nullptr; \
    static XSS_ATTRIBUTE_CONSTRUCTOR void CAT(resolve_argsort_, TYPE)(void) { \
        xss_cpu_init(); \
        std::string_view preferred_cpu = find_preferred_cpu(ISA); \
        internal_argsort##TYPE = &xss::scalar::argsort<TYPE>; \
        if constexpr (dispatch_requested("avx512", ISA)) { \
            if (preferred_cpu.find("avx512") != std::string_view::npos) { \
                internal_argsort##TYPE = &xss::avx512::argsort<TYPE>; \
                return; \
            } \
        } \
        if constexpr (dispatch_requested("avx2", ISA)) { \
            if (preferred_cpu.find("avx2") != std::string_view::npos) { \
                internal_argsort##TYPE = &xss::avx2::argsort<TYPE>; \
            } \
        } \
    }

#define IMPLEMENT_QSORT_SPECIALIZATION(TYPE) \
    template <> \
    void XSS_EXPORT_SYMBOL qsort<TYPE>(TYPE *arr, size_t arrsize, bool hasnan, bool descending) { \
        if (internal_qsort##TYPE == nullptr) { \
            CAT(resolve_qsort_, TYPE)(); \
        } \
        internal_qsort##TYPE(arr, arrsize, hasnan, descending); \
    }

#define IMPLEMENT_ARGSORT_SPECIALIZATION(TYPE) \
    template <> \
    std::vector<size_t> XSS_EXPORT_SYMBOL argsort<TYPE>(const TYPE *arr, size_t arrsize, bool hasnan, bool descending) { \
        if (internal_argsort##TYPE == nullptr) { \
            CAT(resolve_argsort_, TYPE)(); \
        } \
        return internal_argsort##TYPE(arr, arrsize, hasnan, descending); \
    }

PRE_DECLARE_QSORT(uint32_t)
PRE_DECLARE_QSORT(uint64_t)
PRE_DECLARE_ARGSORT(uint32_t)
PRE_DECLARE_ARGSORT(uint64_t)

DEFINE_QSORT_DISPATCHER(uint32_t, ISA_LIST("avx512_skx", "avx2"))
DEFINE_QSORT_DISPATCHER(uint64_t, ISA_LIST("avx512_skx", "avx2"))
DEFINE_ARGSORT_DISPATCHER(uint32_t, ISA_LIST("avx512_skx", "avx2"))
DEFINE_ARGSORT_DISPATCHER(uint64_t, ISA_LIST("avx512_skx", "avx2"))

IMPLEMENT_QSORT_SPECIALIZATION(uint32_t)
IMPLEMENT_QSORT_SPECIALIZATION(uint64_t)
IMPLEMENT_ARGSORT_SPECIALIZATION(uint32_t)
IMPLEMENT_ARGSORT_SPECIALIZATION(uint64_t)

} // namespace x86simdsort
