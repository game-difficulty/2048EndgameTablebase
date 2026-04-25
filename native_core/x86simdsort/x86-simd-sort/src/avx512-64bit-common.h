/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef AVX512_64BIT_COMMON
#define AVX512_64BIT_COMMON

template <typename vtype, typename reg_t>
X86_SIMD_SORT_INLINE reg_t sort_reg_8lanes(reg_t zmm);

struct avx512_64bit_swizzle_ops;

template <>
struct zmm_vector<uint64_t> {
    using type_t = uint64_t;
    using reg_t = __m512i;
    using regi_t = __m512i;
    using halfreg_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;
#ifdef XSS_MINIMAL_NETWORK_SORT
    static constexpr int network_sort_threshold = numlanes;
#else
    static constexpr int network_sort_threshold = 256;
#endif
    static constexpr int partition_unroll_factor = 8;
    static constexpr simd_type vec_type = simd_type::AVX512;

    using swizzle_ops = avx512_64bit_swizzle_ops;

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
        return _mm512_set1_epi64(type_max());
    }
    static reg_t zmm_min()
    {
        return _mm512_set1_epi64(type_min());
    }

    static regi_t
    seti(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    static reg_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m512i index, void const *base)
    {
        return _mm512_mask_i64gather_epi64(src, mask, index, base, scale);
    }
    template <int scale>
    static reg_t
    mask_i64gather(reg_t src, opmask_t mask, __m256i index, void const *base)
    {
        return _mm512_mask_i32gather_epi64(src, mask, index, base, scale);
    }
    static reg_t i64gather(const type_t *arr, arrsize_t *ind)
    {
        return set(arr[ind[7]],
                   arr[ind[6]],
                   arr[ind[5]],
                   arr[ind[4]],
                   arr[ind[3]],
                   arr[ind[2]],
                   arr[ind[1]],
                   arr[ind[0]]);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(reg_t x, reg_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_NLT);
    }
    static opmask_t get_partial_loadmask(uint64_t num_to_read)
    {
        return ((0x1ull << num_to_read) - 0x1ull);
    }
    static opmask_t eq(reg_t x, reg_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_EQ);
    }
    static reg_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static reg_t max(reg_t x, reg_t y)
    {
        return _mm512_max_epu64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static reg_t maskz_loadu(opmask_t mask, void const *mem)
    {
        return _mm512_maskz_loadu_epi64(mask, mem);
    }
    static reg_t mask_loadu(reg_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static reg_t mask_mov(reg_t x, opmask_t mask, reg_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, reg_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static reg_t min(reg_t x, reg_t y)
    {
        return _mm512_min_epu64(x, y);
    }
    static reg_t permutexvar(__m512i idx, reg_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(reg_t v)
    {
        return _mm512_reduce_max_epu64(v);
    }
    static type_t reducemin(reg_t v)
    {
        return _mm512_reduce_min_epu64(v);
    }
    static reg_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static reg_t shuffle(reg_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, reg_t x)
    {
        _mm512_storeu_si512(mem, x);
    }
    static int32_t double_compressstore(void *left_addr,
                                        void *right_addr,
                                        opmask_t k,
                                        reg_t reg)
    {
        int amount_ge_pivot = _mm_popcnt_u32((int)k);
        _mm512_mask_compressstoreu_epi64(left_addr, _knot_mask8(k), reg);
        _mm512_mask_compressstoreu_epi64(
                (uint64_t *)right_addr + 8 - amount_ge_pivot, k, reg);
        return amount_ge_pivot;
    }
    static reg_t reverse(reg_t zmm)
    {
        const regi_t rev_index = seti(NETWORK_REVERSE_8LANES);
        return permutexvar(rev_index, zmm);
    }
    static reg_t sort_vec(reg_t x)
    {
        return sort_reg_8lanes<zmm_vector<type_t>>(x);
    }
    static reg_t cast_from(__m512i v)
    {
        return v;
    }
    static __m512i cast_to(reg_t v)
    {
        return v;
    }
    static bool all_false(opmask_t k)
    {
        return k == 0;
    }
    static int double_compressstore(type_t *left_addr,
                                    type_t *right_addr,
                                    opmask_t k,
                                    reg_t reg)
    {
        return avx512_double_compressstore<zmm_vector<type_t>>(
                left_addr, right_addr, k, reg);
    }
};

struct avx512_64bit_swizzle_ops {
    template <typename vtype, int scale>
    X86_SIMD_SORT_INLINE typename vtype::reg_t swap_n(typename vtype::reg_t reg)
    {
        __m512i v = vtype::cast_to(reg);

        if constexpr (scale == 2) {
            v = _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0b01001110);
        }
        else if constexpr (scale == 4) {
            v = _mm512_shuffle_i64x2(v, v, 0b10110001);
        }
        else if constexpr (scale == 8) {
            v = _mm512_shuffle_i64x2(v, v, 0b01001110);
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
        __m512i v = vtype::cast_to(reg);

        if constexpr (scale == 2) { return swap_n<vtype, 2>(reg); }
        else if constexpr (scale == 4) {
            constexpr uint64_t mask = 0b00011011;
            v = _mm512_permutex_epi64(v, mask);
        }
        else if constexpr (scale == 8) {
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
        __m512i v1 = vtype::cast_to(reg);
        __m512i v2 = vtype::cast_to(other);

        if constexpr (scale == 2) {
            v1 = _mm512_mask_blend_epi64(0b01010101, v1, v2);
        }
        else if constexpr (scale == 4) {
            v1 = _mm512_mask_blend_epi64(0b00110011, v1, v2);
        }
        else if constexpr (scale == 8) {
            v1 = _mm512_mask_blend_epi64(0b00001111, v1, v2);
        }
        else {
            static_assert(scale == -1, "should not be reached");
        }

        return vtype::cast_from(v1);
    }
};

#endif
