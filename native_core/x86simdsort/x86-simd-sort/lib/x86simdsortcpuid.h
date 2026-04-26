#ifndef X86SIMDSORT_CPUID_H
#define X86SIMDSORT_CPUID_H

#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace xss_cpuid {

inline void init() {}

#ifdef _MSC_VER
inline unsigned long long read_xcr0() {
    return _xgetbv(0);
}

inline void cpuid(int leaf, int subleaf, int regs[4]) {
    __cpuidex(regs, leaf, subleaf);
}

inline int max_leaf() {
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 0, 0);
    return regs[0];
}
#else
inline unsigned long long read_xcr0() {
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}

inline void cpuid(int leaf, int subleaf, int regs[4]) {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    regs[0] = static_cast<int>(eax);
    regs[1] = static_cast<int>(ebx);
    regs[2] = static_cast<int>(ecx);
    regs[3] = static_cast<int>(edx);
}

inline int max_leaf() {
    return static_cast<int>(__get_cpuid_max(0, nullptr));
}
#endif

inline bool os_supports_avx(bool require_zmm) {
    int regs[4] = {0, 0, 0, 0};
    cpuid(1, 0, regs);
    if ((regs[2] & (1 << 27)) == 0) {
        return false;
    }

    const unsigned long long xcr0 = read_xcr0();
    constexpr unsigned long long xmm_ymm = 0x2ULL | 0x4ULL;
    constexpr unsigned long long zmm = 0x20ULL | 0x40ULL | 0x80ULL;
    return require_zmm ? ((xcr0 & (xmm_ymm | zmm)) == (xmm_ymm | zmm))
                       : ((xcr0 & xmm_ymm) == xmm_ymm);
}

inline bool has_feature(const char *feature) {
    if (feature == nullptr || max_leaf() < 7) {
        return false;
    }

    int regs[4] = {0, 0, 0, 0};
    cpuid(7, 0, regs);
    const int ebx = regs[1];
    const int ecx = regs[2];

    if (std::strcmp(feature, "avx2") == 0) {
        return os_supports_avx(false) && ((ebx & (1 << 5)) != 0);
    }
    if (std::strcmp(feature, "avx512f") == 0) {
        return os_supports_avx(true) && ((ebx & (1 << 16)) != 0);
    }
    if (std::strcmp(feature, "avx512dq") == 0) {
        return os_supports_avx(true) && ((ebx & (1 << 17)) != 0);
    }
    if (std::strcmp(feature, "avx512bw") == 0) {
        return os_supports_avx(true) && ((ebx & (1 << 30)) != 0);
    }
    if (std::strcmp(feature, "avx512vl") == 0) {
        return os_supports_avx(true) && ((ebx & (1 << 31)) != 0);
    }
    if (std::strcmp(feature, "avx512vbmi2") == 0) {
        return os_supports_avx(true) && ((ecx & (1 << 6)) != 0);
    }
    return false;
}

} // namespace xss_cpuid

#define xss_cpu_init() xss_cpuid::init()
#define xss_cpu_supports(feature) xss_cpuid::has_feature(feature)

#endif // X86SIMDSORT_CPUID_H
