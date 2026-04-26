#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace UniqueUtils {

inline size_t unique_sorted_u64_scalar(uint64_t *data, size_t size) {
    if (data == nullptr || size == 0) {
        return 0;
    }

    size_t out = 1;
    for (size_t i = 1; i < size; ++i) {
        if (data[i] != data[i - 1]) {
            data[out] = data[i];
            ++out;
        }
    }
    return out;
}

#if defined(__x86_64__) || defined(__i386)
inline unsigned long long read_xcr0() {
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}

inline bool cpu_has_avx512_mask_uncached(unsigned int required_ebx) {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    if ((ecx & bit_OSXSAVE) == 0) {
        return false;
    }

    const unsigned long long xcr0 = read_xcr0();
    constexpr unsigned long long kRequiredXcr0 =
        0x2ULL |  // XMM
        0x4ULL |  // YMM
        0x20ULL | // opmask
        0x40ULL | // ZMM_Hi256
        0x80ULL;  // Hi16_ZMM
    if ((xcr0 & kRequiredXcr0) != kRequiredXcr0) {
        return false;
    }

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & required_ebx) == required_ebx;
}
#elif defined(_M_X64) || defined(_M_IX86)
inline bool cpu_has_avx512_mask_uncached(int required_ebx) {
    int cpu_info[4] = {0, 0, 0, 0};
    __cpuidex(cpu_info, 0, 0);
    if (cpu_info[0] < 7) {
        return false;
    }

    int xsave_info[4] = {0, 0, 0, 0};
    __cpuidex(xsave_info, 1, 0);
    if ((xsave_info[2] & (1 << 27)) == 0) {
        return false;
    }

    const unsigned long long xcr0 = _xgetbv(0);
    const bool xmm_state = (xcr0 & 0x2u) != 0;
    const bool ymm_state = (xcr0 & 0x4u) != 0;
    const bool zmm_state = (xcr0 & 0xE0u) == 0xE0u;
    if (!(xmm_state && ymm_state && zmm_state)) {
        return false;
    }

    int ext_info[4] = {0, 0, 0, 0};
    __cpuidex(ext_info, 7, 0);
    return (ext_info[1] & required_ebx) == required_ebx;
}
#endif

inline bool cpu_has_avx512_uncached() {
#if defined(__x86_64__) || defined(__i386)
    return cpu_has_avx512_mask_uncached(1u << 16);
#elif defined(_M_X64) || defined(_M_IX86)
    return cpu_has_avx512_mask_uncached(1 << 16);
#else
    return false;
#endif
}

inline bool cpu_has_avx512() {
    static const bool cached = cpu_has_avx512_uncached();
    return cached;
}

inline bool cpu_has_avx512_dq_vl_uncached() {
#if defined(__x86_64__) || defined(__i386)
    constexpr unsigned int kRequired =
        (1u << 16) | // AVX512F
        (1u << 17) | // AVX512DQ
        (1u << 31);  // AVX512VL
    return cpu_has_avx512_mask_uncached(kRequired);
#elif defined(_M_X64) || defined(_M_IX86)
    constexpr int kRequired =
        (1 << 16) | // AVX512F
        (1 << 17) | // AVX512DQ
        (1 << 31);  // AVX512VL
    return cpu_has_avx512_mask_uncached(kRequired);
#else
    return false;
#endif
}

inline bool cpu_has_avx512_dq_vl() {
    static const bool cached = cpu_has_avx512_dq_vl_uncached();
    return cached;
}

inline bool cpu_has_avx512_dq_bw_vl_uncached() {
#if defined(__x86_64__) || defined(__i386)
    constexpr unsigned int kRequired =
        (1u << 16) | // AVX512F
        (1u << 17) | // AVX512DQ
        (1u << 30) | // AVX512BW
        (1u << 31);  // AVX512VL
    return cpu_has_avx512_mask_uncached(kRequired);
#elif defined(_M_X64) || defined(_M_IX86)
    constexpr int kRequired =
        (1 << 16) | // AVX512F
        (1 << 17) | // AVX512DQ
        (1 << 30) | // AVX512BW
        (1 << 31);  // AVX512VL
    return cpu_has_avx512_mask_uncached(kRequired);
#else
    return false;
#endif
}

inline bool cpu_has_avx512_dq_bw_vl() {
    static const bool cached = cpu_has_avx512_dq_bw_vl_uncached();
    return cached;
}

inline size_t popcount_mask(unsigned mask) {
#if defined(_MSC_VER)
    return static_cast<size_t>(__popcnt(mask));
#else
    return static_cast<size_t>(__builtin_popcount(mask));
#endif
}

#if defined(__GNUC__) || defined(__clang__)
// Compare each lane with its predecessor and let compress-store pack
// the surviving uint64 lanes contiguously without per-element branches.
__attribute__((target("avx512f")))
inline size_t unique_sorted_u64_avx512(uint64_t *data, size_t size) {
    if (data == nullptr || size == 0) {
        return 0;
    }

    size_t out = 1;
    size_t i = 1;

    for (; i + 8 <= size; i += 8) {
        const __m512i curr = _mm512_loadu_si512(reinterpret_cast<const void *>(data + i));
        const __m512i prev = _mm512_loadu_si512(reinterpret_cast<const void *>(data + i - 1));
        const __mmask8 keep_mask = _mm512_cmp_epi64_mask(curr, prev, _MM_CMPINT_NE);
        _mm512_mask_compressstoreu_epi64(static_cast<void *>(data + out), keep_mask, curr);
        out += popcount_mask(static_cast<unsigned>(keep_mask));
    }

    if (i < size) {
        const unsigned remaining = static_cast<unsigned>(size - i);
        const __mmask8 lane_mask = static_cast<__mmask8>((1u << remaining) - 1u);
        const __m512i curr = _mm512_maskz_loadu_epi64(lane_mask, data + i);
        const __m512i prev = _mm512_maskz_loadu_epi64(lane_mask, data + i - 1);
        const __mmask8 keep_mask =
            static_cast<__mmask8>(_mm512_cmp_epi64_mask(curr, prev, _MM_CMPINT_NE) & lane_mask);
        _mm512_mask_compressstoreu_epi64(static_cast<void *>(data + out), keep_mask, curr);
        out += popcount_mask(static_cast<unsigned>(keep_mask));
    }

    return out;
}
#endif

inline size_t unique_sorted_u64_inplace(uint64_t *data, size_t size) {
#if defined(__GNUC__) || defined(__clang__)
    if (cpu_has_avx512()) {
        return unique_sorted_u64_avx512(data, size);
    }
#endif
    return unique_sorted_u64_scalar(data, size);
}

} // namespace UniqueUtils
