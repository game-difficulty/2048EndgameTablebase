#ifndef XSS_COMMON_QSORT
#define XSS_COMMON_QSORT

#include "xss-pivot-selection.hpp"
#include "xss-network-qsort.hpp"
#include "xss-common-comparators.hpp"
#include "xss-reg-networks.hpp"


template <typename vtype,
          typename reg_t = typename vtype::reg_t,
          typename opmask_t = typename vtype::opmask_t>
X86_SIMD_SORT_INLINE reg_t cmp_merge(reg_t in1, reg_t in2, opmask_t mask)
{
    reg_t min = vtype::min(in2, in1);
    reg_t max = vtype::max(in2, in1);
    return vtype::mask_mov(min, mask, max); // 0 -> min, 1 -> max
}

template <typename vtype, typename type_t, typename reg_t>
int avx512_double_compressstore(type_t *left_addr,
                                type_t *right_addr,
                                typename vtype::opmask_t k,
                                reg_t reg)
{
    int amount_ge_pivot = _mm_popcnt_u32((int)k);

    vtype::mask_compressstoreu(left_addr, vtype::knot_opmask(k), reg);
    vtype::mask_compressstoreu(
            right_addr + vtype::numlanes - amount_ge_pivot, k, reg);

    return amount_ge_pivot;
}

// Generic function dispatches to AVX2 or AVX512 code
template <typename vtype,
          typename comparator,
          typename type_t,
          typename reg_t = typename vtype::reg_t>
X86_SIMD_SORT_INLINE arrsize_t partition_vec(type_t *l_store,
                                             type_t *r_store,
                                             const reg_t curr_vec,
                                             const reg_t pivot_vec,
                                             reg_t &smallest_vec,
                                             reg_t &biggest_vec)
{
    typename vtype::opmask_t right_mask
            = comparator::PartitionComparator(curr_vec, pivot_vec);

    int amount_ge_pivot = vtype::double_compressstore(
            l_store, r_store, right_mask, curr_vec);

    smallest_vec = vtype::min(curr_vec, smallest_vec);
    biggest_vec = vtype::max(curr_vec, biggest_vec);

    return amount_ge_pivot;
}

/*
 * Parition an array based on the pivot and returns the index of the
 * first element that is greater than or equal to the pivot.
 */
template <typename vtype, typename comparator, typename type_t>
X86_SIMD_SORT_INLINE arrsize_t partition(type_t *arr,
                                         arrsize_t left,
                                         arrsize_t right,
                                         type_t pivot,
                                         type_t *smallest,
                                         type_t *biggest)
{
    /* make array length divisible by vtype::numlanes , shortening the array */
    for (int32_t i = (right - left) % vtype::numlanes; i > 0; --i) {
        if (arr[left] < *smallest) *smallest = arr[left];
        if (arr[left] > *biggest) *biggest = arr[left];
        if (!comparator::STDSortComparator(arr[left], pivot)) {
            std::swap(arr[left], arr[--right]);
        }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than vtype::numlanes elements in the array */

    using reg_t = typename vtype::reg_t;
    reg_t pivot_vec = vtype::set1(pivot);
    reg_t min_vec = vtype::set1(*smallest);
    reg_t max_vec = vtype::set1(*biggest);

    if (right - left == vtype::numlanes) {
        reg_t vec = vtype::loadu(arr + left);
        arrsize_t unpartitioned = right - left - vtype::numlanes;
        arrsize_t l_store = left;

        arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                arr + l_store,
                arr + l_store + unpartitioned,
                vec,
                pivot_vec,
                min_vec,
                max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return l_store;
    }

    // first and last vtype::numlanes values are partitioned at the end
    reg_t vec_left = vtype::loadu(arr + left);
    reg_t vec_right = vtype::loadu(arr + (right - vtype::numlanes));
    // store points of the vectors
    arrsize_t unpartitioned = right - left - vtype::numlanes;
    arrsize_t l_store = left;
    // indices for loading the elements
    left += vtype::numlanes;
    right -= vtype::numlanes;
    while (right - left != 0) {
        reg_t curr_vec;
        if ((l_store + unpartitioned + vtype::numlanes) - right
            < left - l_store) {
            right -= vtype::numlanes;
            curr_vec = vtype::loadu(arr + right);
        }
        else {
            curr_vec = vtype::loadu(arr + left);
            left += vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                arr + l_store,
                arr + l_store + unpartitioned,
                curr_vec,
                pivot_vec,
                min_vec,
                max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    /* partition and save vec_left and vec_right */
    arrsize_t amount_ge_pivot
            = partition_vec<vtype, comparator>(arr + l_store,
                                               arr + l_store + unpartitioned,
                                               vec_left,
                                               pivot_vec,
                                               min_vec,
                                               max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    unpartitioned -= vtype::numlanes;

    amount_ge_pivot
            = partition_vec<vtype, comparator>(arr + l_store,
                                               arr + l_store + unpartitioned,
                                               vec_right,
                                               pivot_vec,
                                               min_vec,
                                               max_vec);
    l_store += (vtype::numlanes - amount_ge_pivot);
    unpartitioned -= vtype::numlanes;

    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype,
          typename comparator,
          int num_unroll,
          typename type_t = typename vtype::type_t>
X86_SIMD_SORT_INLINE arrsize_t partition_unrolled(type_t *arr,
                                                  arrsize_t left,
                                                  arrsize_t right,
                                                  type_t pivot,
                                                  type_t *smallest,
                                                  type_t *biggest)
{
    if constexpr (num_unroll == 0) {
        return partition<vtype, comparator>(
                arr, left, right, pivot, smallest, biggest);
    }

    /* Use regular partition for smaller arrays */
    if (right - left < 3 * num_unroll * vtype::numlanes) {
        return partition<vtype, comparator>(
                arr, left, right, pivot, smallest, biggest);
    }

    /* make array length divisible by vtype::numlanes, shortening the array */
    for (int32_t i = ((right - left) % (vtype::numlanes)); i > 0; --i) {
        if (arr[left] < *smallest) *smallest = arr[left];
        if (arr[left] > *biggest) *biggest = arr[left];
        if (!comparator::STDSortComparator(arr[left], pivot)) {
            std::swap(arr[left], arr[--right]);
        }
        else {
            ++left;
        }
    }

    arrsize_t unpartitioned = right - left - vtype::numlanes;
    arrsize_t l_store = left;

    using reg_t = typename vtype::reg_t;
    reg_t pivot_vec = vtype::set1(pivot);
    reg_t min_vec = vtype::set1(*smallest);
    reg_t max_vec = vtype::set1(*biggest);

    int vecsToPartition = ((right - left) / vtype::numlanes) % num_unroll;
    reg_t vec_align[num_unroll];
    for (int i = 0; i < vecsToPartition; i++) {
        vec_align[i] = vtype::loadu(arr + left + i * vtype::numlanes);
    }
    left += vecsToPartition * vtype::numlanes;

    reg_t vec_left[num_unroll], vec_right[num_unroll];
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        vec_left[ii] = vtype::loadu(arr + left + vtype::numlanes * ii);
        vec_right[ii] = vtype::loadu(
                arr + (right - vtype::numlanes * (num_unroll - ii)));
    }
    /* indices for loading the elements */
    left += num_unroll * vtype::numlanes;
    right -= num_unroll * vtype::numlanes;
    while (right - left != 0) {
        reg_t curr_vec[num_unroll];
        if ((l_store + unpartitioned + vtype::numlanes) - right
            < left - l_store) {
            right -= num_unroll * vtype::numlanes;
            X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + right + ii * vtype::numlanes);
#if !(defined(_MSC_VER) && defined(__clang__))
                _mm_prefetch((char *)(arr + right + ii * vtype::numlanes
                                       - num_unroll * vtype::numlanes),
                             _MM_HINT_T0);
#endif
            }
        }
        else {
            X86_SIMD_SORT_UNROLL_LOOP(8)
            for (int ii = 0; ii < num_unroll; ++ii) {
                curr_vec[ii] = vtype::loadu(arr + left + ii * vtype::numlanes);
#if !(defined(_MSC_VER) && defined(__clang__))
                _mm_prefetch((char *)(arr + left + ii * vtype::numlanes
                                       + num_unroll * vtype::numlanes),
                             _MM_HINT_T0);
#endif
            }
            left += num_unroll * vtype::numlanes;
        }
        X86_SIMD_SORT_UNROLL_LOOP(8)
        for (int ii = 0; ii < num_unroll; ++ii) {
            arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                    arr + l_store,
                    arr + l_store + unpartitioned,
                    curr_vec[ii],
                    pivot_vec,
                    min_vec,
                    max_vec);
            l_store += (vtype::numlanes - amount_ge_pivot);
            unpartitioned -= vtype::numlanes;
        }
    }

    /* partition and save vec_left[num_unroll] and vec_right[num_unroll] */
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                arr + l_store,
                arr + l_store + unpartitioned,
                vec_left[ii],
                pivot_vec,
                min_vec,
                max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < num_unroll; ++ii) {
        arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                arr + l_store,
                arr + l_store + unpartitioned,
                vec_right[ii],
                pivot_vec,
                min_vec,
                max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    /* partition and save vec_align[vecsToPartition] */
    X86_SIMD_SORT_UNROLL_LOOP(8)
    for (int ii = 0; ii < vecsToPartition; ++ii) {
        arrsize_t amount_ge_pivot = partition_vec<vtype, comparator>(
                arr + l_store,
                arr + l_store + unpartitioned,
                vec_align[ii],
                pivot_vec,
                min_vec,
                max_vec);
        l_store += (vtype::numlanes - amount_ge_pivot);
        unpartitioned -= vtype::numlanes;
    }

    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype, int maxN>
void sort_n(typename vtype::type_t *arr, int N);

template <typename vtype, typename comparator, typename type_t>
static void qsort_(type_t *arr,
                   arrsize_t left,
                   arrsize_t right,
                   arrsize_t max_iters,
                   arrsize_t task_threshold)
{
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1, comparator::STDSortComparator);
        return;
    }
    if (right + 1 - left <= vtype::network_sort_threshold) {
        sort_n<vtype, comparator, vtype::network_sort_threshold>(
                arr + left, (int32_t)(right + 1 - left));
        return;
    }

    auto pivot_result
            = get_pivot_smart<vtype, comparator, type_t>(arr, left, right);
    type_t pivot = pivot_result.pivot;

    if (pivot_result.result == pivot_result_t::Sorted) { return; }

    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();

    arrsize_t pivot_index = partition_unrolled<vtype,
                                               comparator,
                                               vtype::partition_unroll_factor>(
            arr, left, right + 1, pivot, &smallest, &biggest);

    if (pivot_result.result == pivot_result_t::Only2Values) { return; }

    type_t leftmostValue = comparator::leftmost(smallest, biggest);
    type_t rightmostValue = comparator::rightmost(smallest, biggest);

#ifdef XSS_COMPILE_OPENMP
    if (pivot != leftmostValue) {
        bool parallel_left = (pivot_index - left) > task_threshold;
        if (parallel_left) {
#pragma omp task
            qsort_<vtype, comparator>(
                    arr, left, pivot_index - 1, max_iters - 1, task_threshold);
        }
        else {
            qsort_<vtype, comparator>(
                    arr, left, pivot_index - 1, max_iters - 1, task_threshold);
        }
    }
    if (pivot != rightmostValue) {
        bool parallel_right = (right - pivot_index) > task_threshold;

        if (parallel_right) {
#pragma omp task
            qsort_<vtype, comparator>(
                    arr, pivot_index, right, max_iters - 1, task_threshold);
        }
        else {
            qsort_<vtype, comparator>(
                    arr, pivot_index, right, max_iters - 1, task_threshold);
        }
    }
#else
    UNUSED(task_threshold);

    if (pivot != leftmostValue)
        qsort_<vtype, comparator>(arr, left, pivot_index - 1, max_iters - 1, 0);
    if (pivot != rightmostValue)
        qsort_<vtype, comparator>(arr, pivot_index, right, max_iters - 1, 0);
#endif
}

// Quicksort routines:
template <typename vtype, typename T, bool descending = false>
X86_SIMD_SORT_INLINE void xss_qsort(T *arr, arrsize_t arrsize, bool hasnan)
{
    using comparator =
            typename std::conditional<descending,
                                      Comparator<vtype, true>,
                                      Comparator<vtype, false>>::type;

    if (arrsize > 1) {
        UNUSED(hasnan);
#ifdef XSS_COMPILE_OPENMP

        bool use_parallel = arrsize > 100000;

        if (use_parallel) {
            int thread_count = xss_get_num_threads();
            arrsize_t task_threshold
                    = std::max((arrsize_t)100000, arrsize / 100);

#pragma omp parallel num_threads(thread_count)
#pragma omp single
            qsort_<vtype, comparator, T>(arr,
                                         0,
                                         arrsize - 1,
                                         2 * (arrsize_t)log2(arrsize),
                                         task_threshold);
#pragma omp taskwait
        }
        else {
            qsort_<vtype, comparator, T>(arr,
                                         0,
                                         arrsize - 1,
                                         2 * (arrsize_t)log2(arrsize),
                                         std::numeric_limits<arrsize_t>::max());
        }
#else
        qsort_<vtype, comparator, T>(
                arr, 0, arrsize - 1, 2 * (arrsize_t)log2(arrsize), 0);
#endif
    }

#ifdef __MMX__
    _mm_empty();
#endif
}

#include <type_traits>

#define DEFINE_METHODS(ISA, VTYPE) \
    template <typename T> \
    X86_SIMD_SORT_INLINE_ONLY void ISA##_qsort(T *arr, \
                                          arrsize_t size, \
                                          bool hasnan = false, \
                                          bool descending = false) \
    { \
        static_assert(std::is_same_v<T, uint64_t>, "Only uint64_t is supported"); \
        if (descending) { xss_qsort<VTYPE<uint64_t>, uint64_t, true>((uint64_t*)arr, size, hasnan); } \
        else { \
            xss_qsort<VTYPE<uint64_t>, uint64_t, false>((uint64_t*)arr, size, hasnan); \
        } \
    }

#endif // XSS_COMMON_QSORT

