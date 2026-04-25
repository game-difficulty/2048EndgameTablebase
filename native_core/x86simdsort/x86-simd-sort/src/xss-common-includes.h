#ifndef XSS_COMMON_INCLUDES
#define XSS_COMMON_INCLUDES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

#define X86_SIMD_SORT_MAX_UINT64 std::numeric_limits<uint64_t>::max()

#define PRAGMA(x) _Pragma(#x)
#define UNUSED(x) (void)(x)

/* Compiler specific macros specific */
#ifdef _MSC_VER
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static __forceinline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#elif defined(__CYGWIN__)
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static __attribute__((always_inline))
#define X86_SIMD_SORT_FINLINE static __attribute__((always_inline))
#elif defined(__GNUC__)
#define X86_SIMD_SORT_INLINE_ONLY inline
#define X86_SIMD_SORT_INLINE static inline
#define X86_SIMD_SORT_FINLINE static inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#else
#define X86_SIMD_SORT_INLINE_ONLY
#define X86_SIMD_SORT_INLINE static
#define X86_SIMD_SORT_FINLINE static
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#if defined(__INTEL_COMPILER) and !defined(__SANITIZE_ADDRESS__)
#define X86_SIMD_SORT_UNROLL_LOOP(num) PRAGMA(unroll(num))
#elif __GNUC__ >= 8 and !defined(__SANITIZE_ADDRESS__)
#define X86_SIMD_SORT_UNROLL_LOOP(num) PRAGMA(GCC unroll num)
#else
#define X86_SIMD_SORT_UNROLL_LOOP(num)
#endif

#define NETWORK_REVERSE_4LANES 0, 1, 2, 3
#define NETWORK_REVERSE_8LANES 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_REVERSE_16LANES \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define SHUFFLE_MASK(a, b, c, d) (a << 6) | (b << 4) | (c << 2) | d

#if defined(XSS_USE_OPENMP) && defined(_OPENMP)
#define XSS_COMPILE_OPENMP
#include <omp.h>

X86_SIMD_SORT_INLINE int xss_get_num_threads()
{
    return std::min(16, (int)omp_get_max_threads());
}
#endif

template <class... T>
constexpr bool always_false = false;

typedef size_t arrsize_t;

template <typename type>
struct zmm_vector;

template <typename type>
struct ymm_vector;

template <typename type>
struct avx2_vector;

template <typename type>
struct avx2_half_vector;

enum class simd_type : int { AVX2, AVX512 };

template <typename vtype, typename T = typename vtype::type_t>
X86_SIMD_SORT_INLINE bool comparison_func(const T &a, const T &b);

#endif // XSS_COMMON_INCLUDES
