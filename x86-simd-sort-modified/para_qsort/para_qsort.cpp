#include <omp.h>
#include <array>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "x86simdsort.h"

int main() {
    return 0;
}

extern "C" void parallel_sort(uint64_t * arr, size_t size, const std::array<uint64_t, 7> &pivots, bool use_avx512 = false)
{
    auto partition_func = use_avx512
        ? x86simdsort::partition_uint64_array<uint64_t, true>
        : x86simdsort::partition_uint64_array<uint64_t, false>;

    // 初始化 segments 数组
    size_t segments[9] = {0, 0, 0, 0, 0, 0, 0, 0, size};

    // 第一步：使用 pivots[3] 分割数组
    segments[4] = partition_func(arr, 0, size, pivots[3]);

    // 第二步：并行使用 pivots[1] 和 pivots[5] 分割两部分
#pragma omp parallel sections
    {
#pragma omp section
        {
            segments[2] = partition_func(arr, 0, segments[4], pivots[1]);
        }
#pragma omp section
        {
            segments[6] = partition_func(arr, segments[4], size, pivots[5]);
        }
    }

    // 第三步：使用 pivots[0], pivots[2], pivots[4], 和 pivots[6] 分割四部分
#pragma omp parallel sections
    {
#pragma omp section
        {
            segments[1] = partition_func(arr, 0, segments[2], pivots[0]);
        }
#pragma omp section
        {
            segments[3] = partition_func(arr, segments[2], segments[4], pivots[2]);
        }
#pragma omp section
        {
            segments[5] = partition_func(arr, segments[4], segments[6], pivots[4]);
        }
#pragma omp section
        {
            segments[7] = partition_func(arr, segments[6], size, pivots[6]);
        }
    }

    // 使用并行计算对八个部分进行排序
#pragma omp parallel sections
    {
#pragma omp section
        {
            x86simdsort::qsort(arr, segments[1], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[1], segments[2] - segments[1], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[2], segments[3] - segments[2], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[3], segments[4] - segments[3], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[4], segments[5] - segments[4], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[5], segments[6] - segments[5], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[6], segments[7] - segments[6], false);
        }
#pragma omp section
        {
            x86simdsort::qsort(arr + segments[7], size - segments[7], false);
        }
    }
}

