#ifndef AVX2_QSORT_64BIT
#define AVX2_QSORT_64BIT

#include "avx2-emu-funcs.hpp"

// Forward declarations for emulator functions
template <typename T> X86_SIMD_SORT_INLINE __m256i avx2_emu_max(__m256i x, __m256i y);
template <typename T> X86_SIMD_SORT_INLINE __m256i avx2_emu_min(__m256i x, __m256i y);
template <typename T> X86_SIMD_SORT_INLINE T avx2_emu_reduce_max64(__m256i x);
template <typename T> X86_SIMD_SORT_INLINE T avx2_emu_reduce_min64(__m256i x);
template <typename T> X86_SIMD_SORT_INLINE void avx2_emu_mask_compressstoreu64(void *base_addr, __m256i k, __m256i reg);
template <typename T> X86_SIMD_SORT_INLINE int32_t avx2_double_compressstore64(void *left_addr, void *right_addr, __m256i k, __m256i reg);

struct avx2_64bit_swizzle_ops;

template <>
struct avx2_vector<uint64_t> {
    using type_t = uint64_t;
    using reg_t = __m256i;
    using ymmi_t = __m256i;
    using opmask_t = __m256i;
    static const uint8_t numlanes = 4;
#ifdef XSS_MINIMAL_NETWORK_SORT
    static constexpr int network_sort_threshold = numlanes;
#else
    static constexpr int network_sort_threshold = 64;
#endif
    static constexpr int partition_unroll_factor = 8;
    static constexpr simd_type vec_type = simd_type::AVX2;

    using swizzle_ops = avx2_64bit_swizzle_ops;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT64;
    }
    static type_t type_min()
    {
        return 0;
    }
    static reg_t zmm_max()
    {
        return _mm256_set1_epi64x(type_max());
    }
    static reg_t zmm_min()
    {
        return _mm256_set1_epi64x(type_min());
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        auto allTrue = _mm256_set1_epi64x(0xFFFF'FFFF'FFFF'FFFF);
        return _mm256_xor_si256(x, allTrue);
    }
    static opmask_t get_partial_loadmask(uint64_t num_to_read)
    {
        auto mask = ((0x1ull << num_to_read) - 0x1ull);
        return convert_int_to_avx2_mask_64bit(mask);
    }
    static opmask_t convert_int_to_mask(uint64_t intMask)
    {
        return convert_int_to_avx2_mask_64bit(intMask);
    }
    static ymmi_t seti(int64_t v1, int64_t v2, int64_t v3, int64_t v4)
    {
        return _mm256_set_epi64x(v1, v2, v3, v4);
    }
    static reg_t set(type_t v1, type_t v2, type_t v3, type_t v4)
    {
        return _mm256_set_epi64x(v1, v2, v3, v4);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm256_mask_i64gather_epi64(
                src, (const long long int *)base, index, mask, scale);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m128i index, void const *base)
    {
        return _mm256_mask_i32gather_epi64(
                src, (const long long int *)base, index, mask, scale);
    }
    static reg_t i64gather(const type_t *arr, arrsize_t *ind)
    {
        return set(arr[ind[3]], arr[ind[2]], arr[ind[1]], arr[ind[0]]);
    }
    static opmask_t gt(reg_t x, reg_t y)
    {
        return _mm256_cmpgt_epi64(x, y);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        opmask_t equal = eq(x, y);
        opmask_t greater = _mm256_cmpgt_epi64(x, y);
        return _mm256_or_si256(equal, greater);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm256_cmpeq_epi64(x, y);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm256_loadu_si256((reg_t const *)mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return avx2_emu_max<type_t>(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return avx2_emu_mask_compressstoreu64<type_t>(mem, mask, x);
    }
    static int32_t double_compressstore(void *left_addr,
                                        void *right_addr,
                                        opmask_t k,
                                        reg_t reg)
    {
        return avx2_double_compressstore64<type_t>(
                left_addr, right_addr, k, reg);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        reg_t dst = _mm256_maskload_epi64((const long long int *)mem, mask);
        return mask_mov(x, mask, dst);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x),
                                                     _mm256_castsi256_pd(y),
                                                     _mm256_castsi256_pd(mask)));
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm256_maskstore_epi64((long long int *)mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return avx2_emu_min<type_t>(x, y);
    }
    template <int32_t idx>
    static reg_t permutexvar(reg_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    template <int32_t idx>
    static reg_t permutevar(reg_t ymm)
    {
        return _mm256_permute4x64_epi64(ymm, idx);
    }
    static reg_t reverse(reg_t ymm)
    {
        constexpr int32_t rev_index = SHUFFLE_MASK(0, 1, 2, 3);
        return permutexvar<rev_index>(ymm);
    }
    static type_t reducemax(reg_t v)
    {
        return avx2_emu_reduce_max64<type_t>(v);
    }
    static type_t reducemin(reg_t v)
    {
        return avx2_emu_reduce_min64<type_t>(v);
    }
    static reg_t set1(type_t v)
    {
        return _mm256_set1_epi64x(v);
    }
    template <uint8_t mask>
    static reg_t shuffle(reg_t ymm)
    {
        return _mm256_castpd_si256(
                _mm256_permute_pd(_mm256_castsi256_pd(ymm), mask));
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm256_storeu_si256((__m256i *)mem, x);
    }
    static reg_t sort_vec(reg_t x)
    {
        return sort_reg_4lanes<avx2_vector<type_t>>(x);
    }
    static reg_t cast_from(__m256i v)
    {
        return v;
    }
    static __m256i cast_to(reg_t v)
    {
        return v;
    }
    static bool all_false(opmask_t k)
    {
        return _mm256_movemask_pd(_mm256_castsi256_pd(k)) == 0;
    }
};

struct avx2_64bit_swizzle_ops {
    template <typename vtype, int scale>
    X86_SIMD_SORT_INLINE typename vtype::reg_t swap_n(typename vtype::reg_t reg)
    {
        __m256i v = vtype::cast_to(reg);

        if constexpr (scale == 2) {
            v = _mm256_permute4x64_epi64(v, 0b10110001);
        }
        else if constexpr (scale == 4) {
            v = _mm256_permute4x64_epi64(v, 0b01001110);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v);
    }

    template <typename vtype, int scale>
    X86_SIMD_SORT_INLINE typename vtype::reg_t
    reverse_n(typename vtype::reg_t reg)
    {
        __m256i v = vtype::cast_to(reg);

        if constexpr (scale == 2) { return swap_n<vtype, 2>(reg); }
        else if constexpr (scale == 4) {
            return vtype::reverse(reg);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v);
    }

    template <typename vtype, int scale>
    X86_SIMD_SORT_INLINE typename vtype::reg_t
    merge_n(typename vtype::reg_t reg, typename vtype::reg_t other)
    {
        __m256d v1 = _mm256_castsi256_pd(vtype::cast_to(reg));
        __m256d v2 = _mm256_castsi256_pd(vtype::cast_to(other));

        if constexpr (scale == 2) { v1 = _mm256_blend_pd(v1, v2, 0b0101); }
        else if constexpr (scale == 4) {
            v1 = _mm256_blend_pd(v1, v2, 0b0011);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(_mm256_castpd_si256(v1));
    }
};

// Emulators for intrinsics missing from AVX2 compared to AVX512
template <typename T>
X86_SIMD_SORT_INLINE T avx2_emu_reduce_max64(__m256i x)
{
    __m256i swap = _mm256_permute4x64_epi64(x, 0b10110001);
    __m256i m1 = avx2_emu_max<uint64_t>(x, swap);
    __m256i shuf = _mm256_permute4x64_epi64(m1, 0b01001110);
    __m256i m2 = avx2_emu_max<uint64_t>(m1, shuf);
    uint64_t arr[4];
    _mm256_storeu_si256((__m256i *)arr, m2);
    return (T)arr[0];
}

template <typename T>
X86_SIMD_SORT_INLINE T avx2_emu_reduce_min64(__m256i x)
{
    __m256i swap = _mm256_permute4x64_epi64(x, 0b10110001);
    __m256i m1 = avx2_emu_min<uint64_t>(x, swap);
    __m256i shuf = _mm256_permute4x64_epi64(m1, 0b01001110);
    __m256i m2 = avx2_emu_min<uint64_t>(m1, shuf);
    uint64_t arr[4];
    _mm256_storeu_si256((__m256i *)arr, m2);
    return (T)arr[0];
}

template <typename T>
X86_SIMD_SORT_INLINE void
avx2_emu_mask_compressstoreu64(void *base_addr, __m256i k, __m256i reg)
{
    using vtype = avx2_vector<uint64_t>;
    uint64_t *leftStore = (uint64_t *)base_addr;
    int32_t shortMask = convert_avx2_mask_to_int_64bit(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_perm[shortMask].data());
    const __m256i &left = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_left[shortMask].data());
    __m256i temp = _mm256_permutevar8x32_epi32(reg, perm);
    vtype::mask_storeu(leftStore, left, temp);
}

template <typename T>
X86_SIMD_SORT_INLINE int32_t
avx2_double_compressstore64(void *left_addr, void *right_addr, __m256i k, __m256i reg)
{
    using vtype = avx2_vector<uint64_t>;
    uint64_t *leftStore = (uint64_t *)left_addr;
    uint64_t *rightStore = (uint64_t *)right_addr;
    int32_t shortMask = convert_avx2_mask_to_int_64bit(k);
    const __m256i &perm = _mm256_loadu_si256(
            (const __m256i *)avx2_compressstore_lut64_perm[shortMask].data());
    __m256i temp = _mm256_permutevar8x32_epi32(reg, perm);
    vtype::storeu(leftStore, temp);
    vtype::storeu(rightStore, temp);
    return _mm_popcnt_u32(shortMask);
}

template <typename T>
X86_SIMD_SORT_INLINE __m256i avx2_emu_max(__m256i x, __m256i y)
{
    using vtype = avx2_vector<uint64_t>;
    __m256i nlt = vtype::gt(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(nlt)));
}

template <typename T>
X86_SIMD_SORT_INLINE __m256i avx2_emu_min(__m256i x, __m256i y)
{
    using vtype = avx2_vector<uint64_t>;
    __m256i nlt = vtype::gt(x, y);
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(x),
                                                _mm256_castsi256_pd(y),
                                                _mm256_castsi256_pd(nlt)));
}

DEFINE_METHODS(avx2, avx2_vector)

#endif // AVX2_QSORT_64BIT
