#ifndef XSS_REG_NETWORKS
#define XSS_REG_NETWORKS

#include "xss-common-includes.h"

template <typename vtype, typename maskType>
typename vtype::opmask_t convert_int_to_mask(maskType mask);

template <typename vtype, typename reg_t, typename opmask_t>
X86_SIMD_SORT_INLINE reg_t cmp_merge(reg_t in1, reg_t in2, opmask_t mask);

template <typename vtype1,
          typename vtype2,
          typename reg_t1,
          typename reg_t2,
          typename opmask_t>
X86_SIMD_SORT_INLINE reg_t1 cmp_merge(reg_t1 in1,
                                      reg_t1 in2,
                                      reg_t2 &indexes1,
                                      reg_t2 indexes2,
                                      opmask_t mask);

// Single vector functions

/*
 * Assumes reg is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t sort_reg_4lanes(reg_t reg)
{
    using swizzle = typename vtype::swizzle_ops;

    const typename vtype::opmask_t oxA = convert_int_to_mask<vtype>(0xA);
    const typename vtype::opmask_t oxC = convert_int_to_mask<vtype>(0xC);

    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 2>(reg), oxA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 4>(reg), oxC);
    reg = cmp_merge<vtype>(reg, swizzle::template swap_n<vtype, 2>(reg), oxA);
    return reg;
}

/*
 * Assumes reg is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t sort_reg_8lanes(reg_t reg)
{
    using swizzle = typename vtype::swizzle_ops;

    const typename vtype::opmask_t oxAA = convert_int_to_mask<vtype>(0xAA);
    const typename vtype::opmask_t oxCC = convert_int_to_mask<vtype>(0xCC);
    const typename vtype::opmask_t oxF0 = convert_int_to_mask<vtype>(0xF0);

    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 2>(reg), oxAA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 4>(reg), oxCC);
    reg = cmp_merge<vtype>(reg, swizzle::template swap_n<vtype, 2>(reg), oxAA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 8>(reg), oxF0);
    reg = cmp_merge<vtype>(reg, swizzle::template swap_n<vtype, 4>(reg), oxCC);
    reg = cmp_merge<vtype>(reg, swizzle::template swap_n<vtype, 2>(reg), oxAA);
    return reg;
}

/*
 * Assumes reg is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t sort_reg_16lanes(reg_t reg)
{
    using swizzle = typename vtype::swizzle_ops;

    const typename vtype::opmask_t oxAAAA = convert_int_to_mask<vtype>(0xAAAA);
    const typename vtype::opmask_t oxCCCC = convert_int_to_mask<vtype>(0xCCCC);
    const typename vtype::opmask_t oxF0F0 = convert_int_to_mask<vtype>(0xF0F0);
    const typename vtype::opmask_t oxFF00 = convert_int_to_mask<vtype>(0xFF00);

    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 2>(reg), oxAAAA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 4>(reg), oxCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 8>(reg), oxF0F0);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 4>(reg), oxCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAA);
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 16>(reg), oxFF00);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 8>(reg), oxF0F0);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 4>(reg), oxCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAA);
    return reg;
}

/*
 * Assumes reg is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE reg_t sort_reg_32lanes(reg_t reg)
{
    using swizzle = typename vtype::swizzle_ops;

    const typename vtype::opmask_t oxAAAAAAAA
            = convert_int_to_mask<vtype>(0xAAAAAAAA);
    const typename vtype::opmask_t oxCCCCCCCC
            = convert_int_to_mask<vtype>(0xCCCCCCCC);
    const typename vtype::opmask_t oxF0F0F0F0
            = convert_int_to_mask<vtype>(0xF0F0F0F0);
    const typename vtype::opmask_t oxFF00FF00
            = convert_int_to_mask<vtype>(0xFF00FF00);
    const typename vtype::opmask_t oxFFFF0000
            = convert_int_to_mask<vtype>(0xFFFF0000);

    // Level 1
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 2>(reg), oxAAAAAAAA);
    // Level 2
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 4>(reg), oxCCCCCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAAAAAA);
    // Level 3
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 8>(reg), oxF0F0F0F0);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 4>(reg), oxCCCCCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAAAAAA);
    // Level 4
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 16>(reg), oxFF00FF00);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 8>(reg), oxF0F0F0F0);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 4>(reg), oxCCCCCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAAAAAA);
    // Level 5
    reg = cmp_merge<vtype>(
            reg, swizzle::template reverse_n<vtype, 32>(reg), oxFFFF0000);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 16>(reg), oxFF00FF00);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 8>(reg), oxF0F0F0F0);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 4>(reg), oxCCCCCCCC);
    reg = cmp_merge<vtype>(
            reg, swizzle::template swap_n<vtype, 2>(reg), oxAAAAAAAA);
    return reg;
}

#endif // XSS_REG_NETWORKS