#include "HybridSearch.h"
#include "UniqueUtils.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace HybridSearch {
namespace {

using ExactSearchFn = size_t (*)(const uint64_t *, size_t, uint64_t);
using LowerBoundFn = size_t (*)(const uint64_t *, size_t, uint64_t);

struct Dispatch {
    Mode mode = Mode::Scalar;
    ExactSearchFn exact = nullptr;
    LowerBoundFn lower_bound = nullptr;
};

inline unsigned countr_zero_u32(unsigned value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctz(value));
#endif
}

inline size_t exact_search_scalar(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (low < high) {
        const size_t mid = low + ((high - low) >> 1U);
        const uint64_t value = arr[mid];
        if (value < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    if (low < length && arr[low] == target) {
        return low;
    }
    return kNotFound;
}

inline size_t lower_bound_scalar(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (low < high) {
        const size_t mid = low + ((high - low) >> 1U);
        if (arr[mid] < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    return low;
}

inline bool cpu_has_avx2_uncached() {
    return UniqueUtils::cpu_has_avx2();
}

inline bool cpu_has_avx512_uncached() {
    return UniqueUtils::cpu_has_avx512();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), noinline))
#endif
size_t linear_lower_bound_avx2(const uint64_t *arr, size_t length, uint64_t target) {
    const __m128i sign = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
    const __m128i target_vec = _mm_xor_si128(_mm_set1_epi64x(static_cast<long long>(target)), sign);
    size_t i = 0;
    for (; i + 4 <= length; i += 4) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(arr + i));
        const __m128i lo = _mm_xor_si128(_mm256_castsi256_si128(values), sign);
        const __m128i hi = _mm_xor_si128(_mm256_extracti128_si256(values, 1), sign);
        const unsigned lt_mask_lo =
            static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpgt_epi64(target_vec, lo))));
        if (lt_mask_lo != 0x3U) {
            return i + static_cast<size_t>(countr_zero_u32((~lt_mask_lo) & 0x3U));
        }
        const unsigned lt_mask_hi =
            static_cast<unsigned>(_mm_movemask_pd(_mm_castsi128_pd(_mm_cmpgt_epi64(target_vec, hi))));
        if (lt_mask_hi != 0x3U) {
            return i + 2U + static_cast<size_t>(countr_zero_u32((~lt_mask_hi) & 0x3U));
        }
    }
    for (; i < length; ++i) {
        if (arr[i] >= target) {
            return i;
        }
    }
    return length;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f"), noinline))
#endif
size_t linear_lower_bound_avx512(const uint64_t *arr, size_t length, uint64_t target) {
    const __m512i sign = _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL));
    const __m512i target_vec =
        _mm512_xor_si512(_mm512_set1_epi64(static_cast<long long>(target)), sign);
    size_t i = 0;
    for (; i + 8 <= length; i += 8) {
        const __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(arr + i));
        const __m512i values_biased = _mm512_xor_si512(values, sign);
        const __mmask8 lt_mask = _mm512_cmpgt_epi64_mask(target_vec, values_biased);
        if (lt_mask != 0xFF) {
            return i + static_cast<size_t>(countr_zero_u32(static_cast<unsigned>((~lt_mask) & 0xFF)));
        }
    }
    if (i < length) {
        const __mmask8 active =
            static_cast<__mmask8>((1u << static_cast<unsigned>(length - i)) - 1u);
        const __m512i values =
            _mm512_maskz_loadu_epi64(active, reinterpret_cast<const void *>(arr + i));
        const __m512i values_biased = _mm512_xor_si512(values, sign);
        const __mmask8 lt_mask = _mm512_cmpgt_epi64_mask(target_vec, values_biased) & active;
        if (lt_mask != active) {
            return i + static_cast<size_t>(countr_zero_u32(static_cast<unsigned>((~lt_mask) & active)));
        }
    }
    return length;
}

template <size_t Tail>
size_t hybrid_exact_avx2(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (high - low > Tail) {
        const size_t mid = low + ((high - low) >> 1U);
        const uint64_t value = arr[mid];
        if (value < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    const size_t pos = low + linear_lower_bound_avx2(arr + low, high - low, target);
    if (pos < length && arr[pos] == target) {
        return pos;
    }
    return kNotFound;
}

template <size_t Tail>
size_t hybrid_exact_avx512(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (high - low > Tail) {
        const size_t mid = low + ((high - low) >> 1U);
        const uint64_t value = arr[mid];
        if (value < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    const size_t pos = low + linear_lower_bound_avx512(arr + low, high - low, target);
    if (pos < length && arr[pos] == target) {
        return pos;
    }
    return kNotFound;
}

template <size_t Tail>
size_t hybrid_lower_bound_avx2(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (high - low > Tail) {
        const size_t mid = low + ((high - low) >> 1U);
        if (arr[mid] < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    return low + linear_lower_bound_avx2(arr + low, high - low, target);
}

template <size_t Tail>
size_t hybrid_lower_bound_avx512(const uint64_t *arr, size_t length, uint64_t target) {
    size_t low = 0;
    size_t high = length;
    while (high - low > Tail) {
        const size_t mid = low + ((high - low) >> 1U);
        if (arr[mid] < target) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    return low + linear_lower_bound_avx512(arr + low, high - low, target);
}

Dispatch resolve_dispatch() {
    if (cpu_has_avx512_uncached()) {
        return Dispatch{
            Mode::AVX512,
            hybrid_exact_avx512<128>,
            hybrid_lower_bound_avx512<128>,
        };
    }
    if (cpu_has_avx2_uncached()) {
        return Dispatch{
            Mode::AVX2,
            hybrid_exact_avx2<64>,
            hybrid_lower_bound_avx2<64>,
        };
    }
    return Dispatch{
        Mode::Scalar,
        exact_search_scalar,
        lower_bound_scalar,
    };
}

const Dispatch &dispatch() {
    static const Dispatch cached = resolve_dispatch();
    return cached;
}

} // namespace

Mode mode() {
    static const Mode cached_mode = dispatch().mode;
    return cached_mode;
}

size_t exact_search(const uint64_t *arr, size_t length, uint64_t target) {
    static const ExactSearchFn fn = dispatch().exact;
    return fn(arr, length, target);
}

size_t lower_bound(const uint64_t *arr, size_t length, uint64_t target) {
    static const LowerBoundFn fn = dispatch().lower_bound;
    return fn(arr, length, target);
}

} // namespace HybridSearch
