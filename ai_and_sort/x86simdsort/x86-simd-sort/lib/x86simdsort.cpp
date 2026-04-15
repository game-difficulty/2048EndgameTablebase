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

// --- 1. 基础 CPU 探测函数 (保持不变) ---
static int check_cpu_feature_support(std::string_view cpufeature) {
    const char *disable_avx512 = std::getenv("XSS_DISABLE_AVX512");
    if ((cpufeature == "avx512_icl") && (!disable_avx512))
        return xss_cpu_supports("avx512f") && xss_cpu_supports("avx512vbmi2")
                && xss_cpu_supports("avx512bw") && xss_cpu_supports("avx512vl");
    else if ((cpufeature == "avx512_skx") && (!disable_avx512))
        return xss_cpu_supports("avx512f") && xss_cpu_supports("avx512dq")
                && xss_cpu_supports("avx512vl");
    else if (cpufeature == "avx2")
        return xss_cpu_supports("avx2");
    return 0;
}

static std::string_view find_preferred_cpu(std::initializer_list<std::string_view> cpulist) {
    for (auto cpu : cpulist) { if (check_cpu_feature_support(cpu)) return cpu; }
    return "scalar";
}

constexpr bool dispatch_requested(std::string_view cpurequested, std::initializer_list<std::string_view> cpulist) {
    for (auto cpu : cpulist) { if (cpu.find(cpurequested) != std::string_view::npos) return true; }
    return false;
}

// --- 2. 核心宏重构 ---
namespace x86simdsort {

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

// 步骤 A: 显式特化声明 (告知编译器：别急着实例化，我要自己写实现)
#define PRE_DECLARE_SPECIALIZATION(TYPE) \
    template <> XSS_EXPORT_SYMBOL void qsort<TYPE>(TYPE *, size_t, bool, bool);

// 步骤 B: 定义内部函数指针和分发解析器
#define DEFINE_DISPATCHER(TYPE, ISA) \
    static void (*internal_qsort##TYPE)(TYPE *, size_t, bool, bool) = nullptr; \
    static XSS_ATTRIBUTE_CONSTRUCTOR void resolve_qsort_##TYPE(void) { \
        xss_cpu_init(); \
        std::string_view preferred_cpu = find_preferred_cpu(ISA); \
        internal_qsort##TYPE = &xss::scalar::qsort<TYPE>; \
        if constexpr (dispatch_requested("avx512", ISA)) { \
            if (preferred_cpu.find("avx512") != std::string_view::npos) { \
                internal_qsort##TYPE = &xss::avx512::qsort<TYPE>; \
                return; \
            } \
        } \
        if (preferred_cpu.find("avx2") != std::string_view::npos) { \
            internal_qsort##TYPE = &xss::avx2::qsort<TYPE>; \
        } \
    }

// 步骤 C: 真正的特化实现 (外部调用的入口)
#define IMPLEMENT_SPECIALIZATION(TYPE) \
    template <> \
    void XSS_EXPORT_SYMBOL qsort<TYPE>(TYPE *arr, size_t arrsize, bool hasnan, bool descending) { \
        if (!internal_qsort##TYPE) { resolve_qsort_##TYPE(); } \
        internal_qsort##TYPE(arr, arrsize, hasnan, descending); \
    }

// --- 3. 严格顺序执行 ---

// 1. 先进行特化预声明
PRE_DECLARE_SPECIALIZATION(uint64_t)

// 2. 定义分发逻辑
DEFINE_DISPATCHER(uint64_t, (std::initializer_list<std::string_view>{"avx512_skx", "avx2"}))

// 3. 最后写实现代码
IMPLEMENT_SPECIALIZATION(uint64_t)

} // namespace x86simdsort