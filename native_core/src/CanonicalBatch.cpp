#include "CanonicalBatch.h"

#include "Calculator.h"
#include "FormationRuntime.h"
#include "UniqueUtils.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace CanonicalBatch {
namespace {

using BatchImpl = void (*)(const uint64_t *, uint64_t *, size_t, int);

inline uint64_t scalar_by_mode(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal(board);
        case SymmMode::Min33:
            return Calculator::canonical_min33(board);
        case SymmMode::Min24:
            return Calculator::canonical_min24(board);
        case SymmMode::Min34:
            return Calculator::canonical_min34(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

void scalar_impl(const uint64_t *src, uint64_t *dst, size_t count, int symm_mode) {
    if (count == 0) {
        return;
    }
    if (static_cast<SymmMode>(symm_mode) == SymmMode::Identity) {
        if (src != dst) {
            std::memcpy(dst, src, count * sizeof(uint64_t));
        }
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        dst[i] = scalar_by_mode(src[i], symm_mode);
    }
}

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))

#define CANONICAL_AVX2 __attribute__((target("avx2")))
#define CANONICAL_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))

CANONICAL_AVX2 inline __m256i v256_set1(uint64_t value) {
    return _mm256_set1_epi64x(static_cast<long long>(value));
}

CANONICAL_AVX2 inline __m256i v256_or3(__m256i a, __m256i b, __m256i c) {
    return _mm256_or_si256(a, _mm256_or_si256(b, c));
}

CANONICAL_AVX2 inline __m256i v256_reverse_lr(__m256i board) {
    board = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xff00ff00ff00ff00ULL)), 8),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00ff00ff00ff00ffULL)), 8));
    return _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf0f0f0f0f0f0f0f0ULL)), 4),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0f0f0f0f0f0f0f0fULL)), 4));
}

CANONICAL_AVX2 inline __m256i v256_reverse_ud(__m256i board) {
    board = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xffffffff00000000ULL)), 32),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ffffffffULL)), 32));
    return _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xffff0000ffff0000ULL)), 16),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000ffff0000ffffULL)), 16));
}

CANONICAL_AVX2 inline __m256i v256_reverse_ul(__m256i board) {
    board = v256_or3(
        _mm256_and_si256(board, v256_set1(0xff00ff0000ff00ffULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00ff00ff00000000ULL)), 24),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ff00ff00ULL)), 24));
    return v256_or3(
        _mm256_and_si256(board, v256_set1(0xf0f00f0ff0f00f0fULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f0f00000f0f0000ULL)), 12),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000f0f00000f0f0ULL)), 12));
}

CANONICAL_AVX2 inline __m256i v256_reverse_ur(__m256i board) {
    board = v256_or3(
        _mm256_and_si256(board, v256_set1(0x0f0ff0f00f0ff0f0ULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf0f00000f0f00000ULL)), 20),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000f0f00000f0fULL)), 20));
    return v256_or3(
        _mm256_and_si256(board, v256_set1(0x00ff00ffff00ff00ULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xff00ff0000000000ULL)), 40),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000ff00ffULL)), 40));
}

CANONICAL_AVX2 inline __m256i v256_rotate_l(__m256i board) {
    board = v256_or3(
        _mm256_or_si256(
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xff00ff0000000000ULL)), 32),
            _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00ff00ff00000000ULL)), 8)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ff00ff00ULL)), 8),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000ff00ffULL)), 32));
    return v256_or3(
        _mm256_or_si256(
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf0f00000f0f00000ULL)), 16),
            _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0f0f00000f0f0000ULL)), 4)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0000f0f00000f0f0ULL)), 4),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000f0f00000f0fULL)), 16));
}

CANONICAL_AVX2 inline __m256i v256_rotate_r(__m256i board) {
    board = v256_or3(
        _mm256_or_si256(
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xff00ff0000000000ULL)), 8),
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00ff00ff00000000ULL)), 32)),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ff00ff00ULL)), 32),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000ff00ffULL)), 8));
    return v256_or3(
        _mm256_or_si256(
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf0f00000f0f00000ULL)), 4),
            _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f0f00000f0f0000ULL)), 16)),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000f0f00000f0f0ULL)), 16),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000f0f00000f0fULL)), 4));
}

CANONICAL_AVX2 inline __m256i v256_rotate180(__m256i board) {
    return v256_reverse_lr(v256_reverse_ud(board));
}

CANONICAL_AVX2 inline __m256i v256_reverse_ud34(__m256i board) {
    return v256_or3(
        _mm256_and_si256(board, v256_set1(0x0000ffff0000ffffULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xffff000000000000ULL)), 32),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ffff0000ULL)), 32));
}

CANONICAL_AVX2 inline __m256i v256_rotate18034(__m256i board) {
    const __m256i res = v256_rotate180(board);
    return _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(res, v256_set1(0xffff000000000000ULL)), 48),
        _mm256_slli_epi64(_mm256_and_si256(res, v256_set1(0x0000ffffffffffffULL)), 16));
}

CANONICAL_AVX2 inline __m256i v256_exchange_row02(__m256i board) {
    return v256_or3(
        _mm256_and_si256(board, v256_set1(0x0000ffff0000ffffULL)),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000ffff0000ULL)), 32),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xffff000000000000ULL)), 32));
}

CANONICAL_AVX2 inline __m256i v256_exchange_col02(__m256i board) {
    return v256_or3(
        _mm256_and_si256(board, v256_set1(0x0f0f0f0f0f0f0f0fULL)),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf000f000f000f000ULL)), 8),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00f000f000f000f0ULL)), 8));
}

CANONICAL_AVX2 inline __m256i v256_r90_33(__m256i board) {
    __m256i result = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf000000000000000ULL)), 32),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f00000000000000ULL)), 12));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00f0000000000000ULL)), 8));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0000f00000000000ULL)), 20));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000f000000000ULL)), 20));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00000000f0000000ULL)), 8));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000000f000000ULL)), 12));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000f00000ULL)), 32));
    return _mm256_or_si256(result, _mm256_and_si256(board, v256_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX2 inline __m256i v256_l90_33(__m256i board) {
    __m256i result = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf000000000000000ULL)), 8),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f00000000000000ULL)), 20));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00f0000000000000ULL)), 32));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000f00000000000ULL)), 12));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x000000f000000000ULL)), 12));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000f0000000ULL)), 32));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000000f000000ULL)), 20));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000f00000ULL)), 8));
    return _mm256_or_si256(result, _mm256_and_si256(board, v256_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX2 inline __m256i v256_r180_33(__m256i board) {
    __m256i result = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf000000000000000ULL)), 40),
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f00000000000000ULL)), 32));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00f0000000000000ULL)), 24));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0000f00000000000ULL)), 8));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000f000000000ULL)), 8));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000f0000000ULL)), 24));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000000f000000ULL)), 32));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000f00000ULL)), 40));
    return _mm256_or_si256(result, _mm256_and_si256(board, v256_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX2 inline __m256i v256_ul_33(__m256i board) {
    __m256i result = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f0000f000000000ULL)), 12),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000f0000f000000ULL)), 12));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x00f0000000000000ULL)), 24));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x00000000f0000000ULL)), 24));
    return _mm256_or_si256(result, _mm256_and_si256(board, v256_set1(0xf00f0f0f00ffffffULL)));
}

CANONICAL_AVX2 inline __m256i v256_ur_33(__m256i board) {
    __m256i result = _mm256_or_si256(
        _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0x0f00f00000000000ULL)), 20),
        _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x000000f00f000000ULL)), 20));
    result = _mm256_or_si256(result, _mm256_srli_epi64(_mm256_and_si256(board, v256_set1(0xf000000000000000ULL)), 40));
    result = _mm256_or_si256(result, _mm256_slli_epi64(_mm256_and_si256(board, v256_set1(0x0000000000f00000ULL)), 40));
    return _mm256_or_si256(result, _mm256_and_si256(board, v256_set1(0x00ff0f0ff00fffffULL)));
}

CANONICAL_AVX2 inline __m256i v256_cmp_select(__m256i current, __m256i candidate) {
    const __m256i sign = v256_set1(0x8000000000000000ULL);
    const __m256i cur_b = _mm256_xor_si256(current, sign);
    const __m256i cand_b = _mm256_xor_si256(candidate, sign);
    const __m256i lt = _mm256_cmpgt_epi64(cur_b, cand_b);
    return _mm256_blendv_epi8(current, candidate, lt);
}

CANONICAL_AVX2 inline __m256i v256_canonical(__m256i board, int symm_mode) {
    __m256i best = board;
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            best = v256_cmp_select(best, v256_reverse_lr(board));
            best = v256_cmp_select(best, v256_reverse_ud(board));
            best = v256_cmp_select(best, v256_reverse_ul(board));
            best = v256_cmp_select(best, v256_reverse_ur(board));
            best = v256_cmp_select(best, v256_rotate180(board));
            best = v256_cmp_select(best, v256_rotate_l(board));
            best = v256_cmp_select(best, v256_rotate_r(board));
            return best;
        case SymmMode::Diagonal:
            return v256_cmp_select(best, v256_reverse_ul(board));
        case SymmMode::Horizontal:
            return v256_cmp_select(best, v256_reverse_lr(board));
        case SymmMode::Min24:
            best = v256_cmp_select(best, v256_reverse_ud(board));
            best = v256_cmp_select(best, v256_reverse_lr(board));
            best = v256_cmp_select(best, v256_rotate180(board));
            return best;
        case SymmMode::Min34:
            best = v256_cmp_select(best, v256_reverse_lr(board));
            best = v256_cmp_select(best, v256_reverse_ud34(board));
            best = v256_cmp_select(best, v256_rotate18034(board));
            return best;
        case SymmMode::Min33:
            best = v256_cmp_select(best, v256_exchange_col02(board));
            best = v256_cmp_select(best, v256_exchange_row02(board));
            best = v256_cmp_select(best, v256_r90_33(board));
            best = v256_cmp_select(best, v256_l90_33(board));
            best = v256_cmp_select(best, v256_r180_33(board));
            best = v256_cmp_select(best, v256_ur_33(board));
            best = v256_cmp_select(best, v256_ul_33(board));
            return best;
        case SymmMode::Identity:
        default:
            return board;
    }
}

CANONICAL_AVX2 void avx2_impl(const uint64_t *src, uint64_t *dst, size_t count, int symm_mode) {
    if (static_cast<SymmMode>(symm_mode) == SymmMode::Identity) {
        if (src != dst) {
            std::memcpy(dst, src, count * sizeof(uint64_t));
        }
        return;
    }
    size_t i = 0;
    for (; i + 4U <= count; i += 4U) {
        const __m256i board = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        const __m256i canonical = v256_canonical(board, symm_mode);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), canonical);
    }
    for (; i < count; ++i) {
        dst[i] = scalar_by_mode(src[i], symm_mode);
    }
}

CANONICAL_AVX512 inline __m512i v512_set1(uint64_t value) {
    return _mm512_set1_epi64(static_cast<long long>(value));
}

CANONICAL_AVX512 inline __m512i v512_or3(__m512i a, __m512i b, __m512i c) {
    return _mm512_or_si512(a, _mm512_or_si512(b, c));
}

#define DEFINE_V512_FROM_V256_BODY(name, body) \
    CANONICAL_AVX512 inline __m512i name(__m512i board) body

DEFINE_V512_FROM_V256_BODY(v512_reverse_lr, {
    board = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xff00ff00ff00ff00ULL)), 8),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00ff00ff00ff00ffULL)), 8));
    return _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf0f0f0f0f0f0f0f0ULL)), 4),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0f0f0f0f0f0f0f0fULL)), 4));
})

DEFINE_V512_FROM_V256_BODY(v512_reverse_ud, {
    board = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xffffffff00000000ULL)), 32),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ffffffffULL)), 32));
    return _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xffff0000ffff0000ULL)), 16),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000ffff0000ffffULL)), 16));
})

DEFINE_V512_FROM_V256_BODY(v512_reverse_ul, {
    board = v512_or3(
        _mm512_and_si512(board, v512_set1(0xff00ff0000ff00ffULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00ff00ff00000000ULL)), 24),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ff00ff00ULL)), 24));
    return v512_or3(
        _mm512_and_si512(board, v512_set1(0xf0f00f0ff0f00f0fULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f0f00000f0f0000ULL)), 12),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000f0f00000f0f0ULL)), 12));
})

DEFINE_V512_FROM_V256_BODY(v512_reverse_ur, {
    board = v512_or3(
        _mm512_and_si512(board, v512_set1(0x0f0ff0f00f0ff0f0ULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf0f00000f0f00000ULL)), 20),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000f0f00000f0fULL)), 20));
    return v512_or3(
        _mm512_and_si512(board, v512_set1(0x00ff00ffff00ff00ULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xff00ff0000000000ULL)), 40),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000ff00ffULL)), 40));
})

DEFINE_V512_FROM_V256_BODY(v512_rotate_l, {
    board = v512_or3(
        _mm512_or_si512(
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xff00ff0000000000ULL)), 32),
            _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00ff00ff00000000ULL)), 8)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ff00ff00ULL)), 8),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000ff00ffULL)), 32));
    return v512_or3(
        _mm512_or_si512(
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf0f00000f0f00000ULL)), 16),
            _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0f0f00000f0f0000ULL)), 4)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0000f0f00000f0f0ULL)), 4),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000f0f00000f0fULL)), 16));
})

DEFINE_V512_FROM_V256_BODY(v512_rotate_r, {
    board = v512_or3(
        _mm512_or_si512(
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xff00ff0000000000ULL)), 8),
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00ff00ff00000000ULL)), 32)),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ff00ff00ULL)), 32),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000ff00ffULL)), 8));
    return v512_or3(
        _mm512_or_si512(
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf0f00000f0f00000ULL)), 4),
            _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f0f00000f0f0000ULL)), 16)),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000f0f00000f0f0ULL)), 16),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000f0f00000f0fULL)), 4));
})

CANONICAL_AVX512 inline __m512i v512_rotate180(__m512i board) {
    return v512_reverse_lr(v512_reverse_ud(board));
}

DEFINE_V512_FROM_V256_BODY(v512_reverse_ud34, {
    return v512_or3(
        _mm512_and_si512(board, v512_set1(0x0000ffff0000ffffULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xffff000000000000ULL)), 32),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ffff0000ULL)), 32));
})

CANONICAL_AVX512 inline __m512i v512_rotate18034(__m512i board) {
    const __m512i res = v512_rotate180(board);
    return _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(res, v512_set1(0xffff000000000000ULL)), 48),
        _mm512_slli_epi64(_mm512_and_si512(res, v512_set1(0x0000ffffffffffffULL)), 16));
}

DEFINE_V512_FROM_V256_BODY(v512_exchange_row02, {
    return v512_or3(
        _mm512_and_si512(board, v512_set1(0x0000ffff0000ffffULL)),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000ffff0000ULL)), 32),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xffff000000000000ULL)), 32));
})

DEFINE_V512_FROM_V256_BODY(v512_exchange_col02, {
    return v512_or3(
        _mm512_and_si512(board, v512_set1(0x0f0f0f0f0f0f0f0fULL)),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf000f000f000f000ULL)), 8),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00f000f000f000f0ULL)), 8));
})

CANONICAL_AVX512 inline __m512i v512_r90_33(__m512i board) {
    __m512i result = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf000000000000000ULL)), 32),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f00000000000000ULL)), 12));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00f0000000000000ULL)), 8));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0000f00000000000ULL)), 20));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000f000000000ULL)), 20));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00000000f0000000ULL)), 8));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000000f000000ULL)), 12));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000f00000ULL)), 32));
    return _mm512_or_si512(result, _mm512_and_si512(board, v512_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX512 inline __m512i v512_l90_33(__m512i board) {
    __m512i result = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf000000000000000ULL)), 8),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f00000000000000ULL)), 20));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00f0000000000000ULL)), 32));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000f00000000000ULL)), 12));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x000000f000000000ULL)), 12));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000f0000000ULL)), 32));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000000f000000ULL)), 20));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000f00000ULL)), 8));
    return _mm512_or_si512(result, _mm512_and_si512(board, v512_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX512 inline __m512i v512_r180_33(__m512i board) {
    __m512i result = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf000000000000000ULL)), 40),
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f00000000000000ULL)), 32));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00f0000000000000ULL)), 24));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0000f00000000000ULL)), 8));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000f000000000ULL)), 8));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000f0000000ULL)), 24));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000000f000000ULL)), 32));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000f00000ULL)), 40));
    return _mm512_or_si512(result, _mm512_and_si512(board, v512_set1(0x000f0f0f000fffffULL)));
}

CANONICAL_AVX512 inline __m512i v512_ul_33(__m512i board) {
    __m512i result = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f0000f000000000ULL)), 12),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000f0000f000000ULL)), 12));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x00f0000000000000ULL)), 24));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x00000000f0000000ULL)), 24));
    return _mm512_or_si512(result, _mm512_and_si512(board, v512_set1(0xf00f0f0f00ffffffULL)));
}

CANONICAL_AVX512 inline __m512i v512_ur_33(__m512i board) {
    __m512i result = _mm512_or_si512(
        _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0x0f00f00000000000ULL)), 20),
        _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x000000f00f000000ULL)), 20));
    result = _mm512_or_si512(result, _mm512_srli_epi64(_mm512_and_si512(board, v512_set1(0xf000000000000000ULL)), 40));
    result = _mm512_or_si512(result, _mm512_slli_epi64(_mm512_and_si512(board, v512_set1(0x0000000000f00000ULL)), 40));
    return _mm512_or_si512(result, _mm512_and_si512(board, v512_set1(0x00ff0f0ff00fffffULL)));
}

CANONICAL_AVX512 inline __m512i v512_cmp_select(__m512i current, __m512i candidate) {
    const __mmask8 mask = _mm512_cmp_epu64_mask(candidate, current, _MM_CMPINT_LT);
    return _mm512_mask_mov_epi64(current, mask, candidate);
}

CANONICAL_AVX512 inline __m512i v512_canonical(__m512i board, int symm_mode) {
    __m512i best = board;
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            best = v512_cmp_select(best, v512_reverse_lr(board));
            best = v512_cmp_select(best, v512_reverse_ud(board));
            best = v512_cmp_select(best, v512_reverse_ul(board));
            best = v512_cmp_select(best, v512_reverse_ur(board));
            best = v512_cmp_select(best, v512_rotate180(board));
            best = v512_cmp_select(best, v512_rotate_l(board));
            best = v512_cmp_select(best, v512_rotate_r(board));
            return best;
        case SymmMode::Diagonal:
            return v512_cmp_select(best, v512_reverse_ul(board));
        case SymmMode::Horizontal:
            return v512_cmp_select(best, v512_reverse_lr(board));
        case SymmMode::Min24:
            best = v512_cmp_select(best, v512_reverse_ud(board));
            best = v512_cmp_select(best, v512_reverse_lr(board));
            best = v512_cmp_select(best, v512_rotate180(board));
            return best;
        case SymmMode::Min34:
            best = v512_cmp_select(best, v512_reverse_lr(board));
            best = v512_cmp_select(best, v512_reverse_ud34(board));
            best = v512_cmp_select(best, v512_rotate18034(board));
            return best;
        case SymmMode::Min33:
            best = v512_cmp_select(best, v512_exchange_col02(board));
            best = v512_cmp_select(best, v512_exchange_row02(board));
            best = v512_cmp_select(best, v512_r90_33(board));
            best = v512_cmp_select(best, v512_l90_33(board));
            best = v512_cmp_select(best, v512_r180_33(board));
            best = v512_cmp_select(best, v512_ur_33(board));
            best = v512_cmp_select(best, v512_ul_33(board));
            return best;
        case SymmMode::Identity:
        default:
            return board;
    }
}

CANONICAL_AVX512 void avx512_impl(const uint64_t *src, uint64_t *dst, size_t count, int symm_mode) {
    if (static_cast<SymmMode>(symm_mode) == SymmMode::Identity) {
        if (src != dst) {
            std::memcpy(dst, src, count * sizeof(uint64_t));
        }
        return;
    }
    size_t i = 0;
    for (; i + 8U <= count; i += 8U) {
        const __m512i board = _mm512_loadu_si512(reinterpret_cast<const void *>(src + i));
        const __m512i canonical = v512_canonical(board, symm_mode);
        _mm512_storeu_si512(reinterpret_cast<void *>(dst + i), canonical);
    }
    if (i < count) {
        avx2_impl(src + i, dst + i, count - i, symm_mode);
    }
}

#undef DEFINE_V512_FROM_V256_BODY
#undef CANONICAL_AVX2
#undef CANONICAL_AVX512
#endif

BatchImpl select_impl() {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
    if (UniqueUtils::cpu_has_avx512_dq_bw_vl()) {
        return avx512_impl;
    }
    if (UniqueUtils::cpu_has_avx2()) {
        return avx2_impl;
    }
#endif
    return scalar_impl;
}

const char *select_backend_name() {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
    if (UniqueUtils::cpu_has_avx512_dq_bw_vl()) {
        return "avx512";
    }
    if (UniqueUtils::cpu_has_avx2()) {
        return "avx2";
    }
#endif
    return "scalar";
}

BatchImpl impl() {
    static const BatchImpl cached = select_impl();
    return cached;
}

} // namespace

void canonicalize_by_mode(
    const uint64_t *src,
    uint64_t *dst,
    size_t count,
    int symm_mode
) {
    if (src == nullptr || dst == nullptr || count == 0) {
        return;
    }
    impl()(src, dst, count, symm_mode);
}

void canonicalize_inplace(uint64_t *data, size_t count, int symm_mode) {
    canonicalize_by_mode(data, data, count, symm_mode);
}

const char *backend_name() {
    static const char *cached = select_backend_name();
    return cached;
}

} // namespace CanonicalBatch
