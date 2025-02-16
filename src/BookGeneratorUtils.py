import ctypes
import os
from ctypes import c_uint64, c_size_t, POINTER
from typing import Callable, List
import traceback

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

import Config
from Config import SingletonConfig


PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[NDArray[np.uint64]], None]
ToFindFunc1 = Callable[[np.uint64], np.uint64]

logger = Config.logger
use_avx = SingletonConfig().use_avx


def initialize_sorting_library():
    global use_avx
    _dll = None
    try:
        current_dir = os.path.dirname(__file__)
        dll_path = os.path.join(current_dir, "para_qsort", "para_qsort.dll")
        if not os.path.exists(dll_path):
            dll_path = r"_internal/para_qsort.dll"
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"para_qsort.dll not found.")

        # 加载DLL
        _dll = ctypes.CDLL(dll_path)

        # 声明parallel_sort函数原型
        _dll.parallel_sort.argtypes = (POINTER(c_uint64), c_size_t, POINTER(c_uint64), ctypes.c_bool)
        _dll.parallel_sort.restype = None

        # # 测试排序操作
        # _arr = np.random.randint(0, 1 << 64, 1000, dtype=np.uint64)
        # _pivots = np.array([2 ** 61, 2 ** 62, 2 ** 61 * 3, 2 ** 63, 2 ** 61 * 5, 2 ** 62 * 3, 2 ** 61 * 7],
        #                    dtype=np.uint64)
        #
        # _arr_ptr = _arr.ctypes.data_as(POINTER(c_uint64))
        # _pivots_ptr = (c_uint64 * len(_pivots))(*_pivots)
        # _use_avx512 = True if use_avx == 2 else False
        #
        # # 调用DLL中的parallel_sort函数进行测试排序
        # _dll.parallel_sort(_arr_ptr, _arr.size, _pivots_ptr, _use_avx512)
        #
        # # 验证排序结果是否正确
        # if not np.all(_arr[:-1] <= _arr[1:]):
        #     raise ValueError("DLL sorting failed, results are not sorted correctly")
        # else:
        #     if _use_avx512:
        #         logger.info("Sorting success using AVX512.")
        #     else:
        #         logger.info("Sorting success using AVX2.")

    except Exception as e:
        # 如果加载DLL或排序出错，禁用AVX并记录日志
        logger.warning(f"Failed to load DLL or test sorting failed: {e}. Falling back to numpy sort.")
        logger.warning(traceback.format_exc())  # 输出完整的异常堆栈
        use_avx = False

    return _dll


# 在模块加载时执行初始化函数
dll = initialize_sorting_library()


def parallel_unique(aux: NDArray[np.uint64], n: int) -> NDArray[np.uint64]:
    if len(aux) < 128:
        return np.unique(aux)
    else:
        return _parallel_unique(aux, n)


@njit(parallel=True)
def _parallel_unique(aux: NDArray[np.uint64], n: int) -> NDArray[np.uint64]:
    # 获取分段的大小
    step = len(aux) // n
    c_list = np.zeros(n, dtype=np.int64)  # 存储每段的去重长度

    # 每个线程处理一段
    for i in prange(n):
        start = i * step
        end = (i + 1) * step if i != n - 1 else len(aux)

        # 这段的去重
        if i == 0:
            c = 1
            start_ = 1 + start
        else:
            c = 0
            start_ = start
        for j in range(start_, end):
            if aux[j] != aux[j - 1]:
                aux[start + c] = aux[j]
                c += 1
        c_list[i] = c  # 记录每段的去重元素数量

    # 计算结果数组的长度
    result_c = np.sum(c_list)  # 合并所有段的去重结果

    # 创建最终的去重数组
    result = np.empty(np.uint64(result_c), dtype=np.uint64)

    # 合并所有段的去重结果
    result_cumulative = 0
    for i in range(n):
        start = i * step
        result[result_cumulative:result_cumulative + c_list[i]] = aux[start:start + c_list[i]]
        result_cumulative += c_list[i]

    return result


@njit(nogil=True)
def largest_power_of_2(n):
    n = np.uint64(n)
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@njit(nogil=True)
def _merge_deduplicate_all(arrays: List[NDArray], length: int = 0) -> NDArray:
    if len(arrays) == 1:
        return arrays[0]
    if length == 0:
        for arr in arrays:
            length += len(arr)
    num_arrays = len(arrays)
    indices = np.zeros(num_arrays, dtype='uint32')  # 每个数组的当前索引
    merged_array = np.empty(length, dtype='uint64')  # 合并后且去重的数组
    last_added = np.uint64(0xffffffffffffffff)  # 上一个添加到 merged_array 的元素
    c = 0  # 已添加的元素数量
    # 继续循环，直到所有数组都被完全处理
    while True:
        current_min = np.uint64(0xffffffffffffffff)
        min_index = -1
        # 寻找当前可用元素中的最小值
        for i in range(num_arrays):
            if indices[i] < len(arrays[i]):  # 确保索引不超出数组长度
                if arrays[i][indices[i]] < current_min:
                    current_min = arrays[i][indices[i]]
                    min_index = i
        # 如果找不到最小值，说明所有数组都已处理完成
        if current_min == np.uint64(0xffffffffffffffff):
            break
        # 检查是否需要将当前最小值添加到结果数组中
        if current_min != last_added:
            merged_array[c] = current_min
            last_added = current_min
            c += 1
        # 移动选中数组的索引
        indices[min_index] += 1
    return merged_array[:c]


@njit(nogil=True)
def binary_search(arr, x):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


def merge_deduplicate_all(arrays: List[NDArray], pivots, n_threads: int | None = None) -> list[NDArray]:
    """
    多线程合并并去重多个已排序的数组
    """
    n_threads = os.cpu_count() if n_threads is None else n_threads
    num_arrays = len(arrays)

    # 在所有数组中找到每个枢轴的位置
    split_positions = np.zeros((num_arrays, n_threads + 1), dtype=np.int64)
    for a in range(num_arrays):
        for t in range(n_threads - 1):
            split_positions[a][t + 1] = binary_search(arrays[a], pivots[t])
        split_positions[a][n_threads] = len(arrays[a])
    return _merge_deduplicate_all_p(arrays, split_positions, n_threads)


@njit(parallel=True, nogil=True)
def _merge_deduplicate_all_p(arrays, split_positions, n_threads):
    num_arrays = len(arrays)
    res = [np.empty(0, dtype='uint64') for _ in range(n_threads)]
    # 并行归并每个分区
    for t in prange(n_threads):
        temp_arrays = [np.empty(0, dtype='uint64') for _ in range(num_arrays)]
        for a in range(num_arrays):
            s = split_positions[a][t]
            e = split_positions[a][t + 1]
            temp_arrays[a] = arrays[a][s:e]
        res[t] = _merge_deduplicate_all(temp_arrays)
    return res


@njit(parallel=True, nogil=True)
def concatenate(arrays):
    length = 0
    for array in arrays:
        length += len(array)
    res = np.empty(length, dtype='uint64')
    offset = 0
    for array in arrays:
        res[offset: len(array) + offset] = array
        offset += len(array)
    return res


def sort_array(arr: NDArray[np.uint64], pivots: NDArray[np.uint64] | None = None) -> None:
    # 小数组直接调用numpy sort
    if len(arr) < 1e6 or pivots is None or len(pivots) != 7 or not use_avx:
        arr.sort()
    else:
        # 转换numpy数组为ctypes数组
        arr_ptr = arr.ctypes.data_as(POINTER(c_uint64))
        pivots_ptr = (c_uint64 * 7)(*pivots)
        use_avx512 = True if use_avx == 2 else False
        # 调用DLL中的parallel_sort函数原地排序
        dll.parallel_sort(arr_ptr, arr.size, pivots_ptr, use_avx512)


@njit(nogil=True)
def hash_(board: np.uint64) -> np.uint64:
    return np.uint64((board ^ (board >> 27)) * 0x1A85EC53 + board >> 23)


@njit(nogil=True, parallel=True)
def is_sorted(arr: NDArray) -> bool:
    return np.all(arr[:-1] < arr[1:])


@njit(nogil=True, parallel=True)
def is_sorted2(arr: NDArray) -> bool:
    return np.all(arr[:-1] <= arr[1:])


def check_sorted(arr):
    try:
        if is_sorted(arr):
            return arr
        elif is_sorted2(arr):
            logger.warning('sorted but not deduplicated')
            return parallel_unique(arr, os.cpu_count())
        raise ValueError('not sorted')
    except ValueError:
        arr = np.unique(arr)
        return arr


@njit(nogil=True)
def merge_and_deduplicate(sorted_arr1: NDArray, sorted_arr2: NDArray) -> NDArray:
    # 结果数组的长度最多与两数组之和一样长
    unique_array = np.empty(len(sorted_arr1) + len(sorted_arr2), dtype=np.uint64)

    i, j, k = 0, 0, 0  # i, j 分别是两数组的索引，k 是结果数组的索引

    while i < len(sorted_arr1) and j < len(sorted_arr2):
        if sorted_arr1[i] < sorted_arr2[j]:
            if k == 0 or unique_array[k - 1] != sorted_arr1[i]:  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr1[i]
                k += 1
            i += 1
        elif sorted_arr1[i] > sorted_arr2[j]:
            if k == 0 or unique_array[k - 1] != sorted_arr2[j]:  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr2[j]
                k += 1
            j += 1
        else:  # sorted_arr1[i] == sorted_arr2[j]
            if k == 0 or unique_array[k - 1] != sorted_arr1[i]:  # 添加新元素，并确保不重复
                unique_array[k] = sorted_arr1[i]
                k += 1
            i += 1
            j += 1

    # 处理剩余的元素
    if i < len(sorted_arr1):
        unique_array[k:k + len(sorted_arr1) - i] = sorted_arr1[i:]
        k += len(sorted_arr1) - i
    if j < len(sorted_arr2):
        unique_array[k:k + len(sorted_arr2) - j] = sorted_arr2[j:]
        k += len(sorted_arr2) - j

    return unique_array[:k]  # 调整数组大小以匹配实际元素数


@njit(nogil=True, boundscheck=True)
def merge_inplace(arr: NDArray, segment_ends: NDArray, segment_starts: NDArray) -> NDArray:
    """
    将每一段有效数据依次合并到前一段末尾。
    如果读写区间产生重叠，则仅移动非重叠部分。

    参数:
    - arr: 需要合并的原始数组。
    - segment_ends: 每个数据段的结束索引。
    - n: 数组的总长度（可选，根据需要使用）。
    - segment_starts: 每个数据段的起始索引。

    返回:
    - 合并后的数组切片。
    """
    counts = np.uint64(segment_ends[0])
    num_segments = len(segment_starts)

    for i in range(1, num_segments):
        start = segment_starts[i]
        end = segment_ends[i]
        size = np.uint64(end - start)
        dest_start = counts
        dest_end = np.uint64(counts + size)

        # 检查读写区间是否重叠
        if start < dest_end:
            arr[dest_start: start] = arr[dest_end: end]
        else:
            arr[dest_start: dest_end] = arr[start: end]
        counts += size

    return arr[:counts]


def update_seg(seg: NDArray[float], percents: NDArray[float], learning_rate: float = 0.5) -> NDArray[float]:
    seg_length = seg[1:] - seg[:-1]
    if 0 not in percents:
        normalised = (seg_length / percents)
    else:
        normalised = (seg_length / (percents + 0.5 / len(percents)))
    seg_length_new = normalised / normalised.sum()
    seg[1:] = np.cumsum(seg_length_new * learning_rate + seg_length * (1 - learning_rate))
    seg[0], seg[-1] = 0, 1  # 避免浮点数精度导致的bug
    return seg
