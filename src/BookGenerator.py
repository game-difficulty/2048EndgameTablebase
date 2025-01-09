import ctypes
import gc
import os
import time
from ctypes import c_uint64, c_size_t, POINTER
from typing import Callable, Tuple, List
import traceback

import numpy as np
from numba import njit, prange
import psutil

from BoardMover import BoardMover
import Config
from Config import SingletonConfig
from LzmaCompressor import compress_with_7z

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.ndarray[np.uint64]], None]
ToFindFunc1 = Callable[[np.uint64], np.uint64]

logger = Config.logger
use_avx = SingletonConfig().use_avx


def initialize_sorting_library():
    global use_avx
    _dll = None
    try:
        current_dir = os.path.dirname(__file__)
        dll_path = os.path.join(current_dir, "psort", "para_qsort", "para_qsort.dll")
        # dll_path = r"_internal/para_qsort.dll"
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"para_qsort.dll not found.")

        # 加载DLL
        _dll = ctypes.CDLL(dll_path)

        # 声明parallel_sort函数原型
        _dll.parallel_sort.argtypes = (POINTER(c_uint64), c_size_t, POINTER(c_uint64), ctypes.c_bool)
        _dll.parallel_sort.restype = None

        # 测试排序操作
        _arr = np.random.randint(0, 1 << 64, 1000, dtype=np.uint64)
        _pivots = np.array([2 ** 61, 2 ** 62, 2 ** 61 * 3, 2 ** 63, 2 ** 61 * 5, 2 ** 62 * 3, 2 ** 61 * 7],
                           dtype=np.uint64)

        _arr_ptr = _arr.ctypes.data_as(POINTER(c_uint64))
        _pivots_ptr = (c_uint64 * len(_pivots))(*_pivots)
        _use_avx512 = True if use_avx == 2 else False

        # 调用DLL中的parallel_sort函数进行测试排序
        _dll.parallel_sort(_arr_ptr, _arr.size, _pivots_ptr, _use_avx512)

        # 验证排序结果是否正确
        if not np.all(_arr[:-1] <= _arr[1:]):
            raise ValueError("DLL sorting failed, results are not sorted correctly")
        else:
            if _use_avx512:
                logger.info("Sorting success using AVX512.")
            else:
                logger.info("Sorting success using AVX2.")

    except Exception as e:
        # 如果加载DLL或排序出错，禁用AVX并记录日志
        logger.warning(f"Failed to load DLL or test sorting failed: {e}. Falling back to numpy sort.")
        logger.warning(traceback.format_exc())  # 输出完整的异常堆栈
        use_avx = False

    return _dll


# 在模块加载时执行初始化函数
dll = initialize_sorting_library()


def parallel_unique(aux: np.ndarray[np.uint64], n: int) -> np.ndarray[np.uint64]:
    if len(aux) < 128:
        return np.unique(aux)
    else:
        return _parallel_unique(aux, n)


@njit(parallel=True)
def _parallel_unique(aux: np.ndarray[np.uint64], n: int) -> np.ndarray[np.uint64]:
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
    result = np.empty(result_c, dtype=np.uint64)

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
def _merge_deduplicate_all(arrays: List[np.ndarray], length: int = 0) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    if length == 0:
        for arr in arrays:
            length += len(arr)
    num_arrays = len(arrays)
    indices = np.zeros(num_arrays, dtype='uint32')  # 每个数组的当前索引
    merged_array = np.empty(length, dtype='uint64')  # 合并后且去重的数组
    last_added = None  # 上一个添加到 merged_array 的元素
    c = 0  # 已添加的元素数量
    # 继续循环，直到所有数组都被完全处理
    while True:
        current_min = None
        min_index = -1
        # 寻找当前可用元素中的最小值
        for i in range(num_arrays):
            if indices[i] < len(arrays[i]):  # 确保索引不超出数组长度
                if current_min is None or arrays[i][indices[i]] < current_min:
                    current_min = arrays[i][indices[i]]
                    min_index = i
        # 如果找不到最小值，说明所有数组都已处理完成
        if current_min is None:
            break
        # 检查是否需要将当前最小值添加到结果数组中
        if last_added is None or current_min != last_added:
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


def merge_deduplicate_all(arrays: List[np.ndarray], pivots, n_threads: int | None = None) -> list[np.ndarray]:
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


def sort_array(arr: np.ndarray[np.uint64], pivots: np.ndarray[np.uint64] | None = None) -> None:
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


def gen_boards_big(arr0: np.ndarray[np.uint64],
                   target: int,
                   position: int,
                   bm: BoardMover,
                   pattern_check_func: PatternCheckFunc,
                   success_check_func: SuccessCheckFunc,
                   to_find_func: ToFindFunc1,
                   seg_list: np.ndarray[float],
                   pivots_list: List[np.ndarray[np.uint64]],
                   hashmap1: np.ndarray[np.uint64],
                   hashmap2: np.ndarray[np.uint64],
                   n: int,
                   length_factors_list: List[List[float]],
                   length_factor_multiplier: float = 1.5,
                   do_check: bool = True,
                   isfree: bool = False,
                   ) -> Tuple[
    List[np.ndarray[np.uint64]], List[np.ndarray[np.uint64]], List[np.ndarray[np.uint64]], np.ndarray[float], List[
        List[float]], float, np.ndarray[np.uint64], np.ndarray[np.uint64], float, float, float]:
    """
    将arr0分段放入gen_boards生成排序去重后的局面，然后归并
    """
    arr1s: List[np.ndarray[np.uint64]] = []
    arr2s: List[np.ndarray[np.uint64]] = []
    actual_lengths2: np.ndarray[np.uint64] = np.empty(len(seg_list) - 1, dtype=np.uint64)
    actual_lengths1: np.ndarray[np.uint64] = np.empty(len(seg_list) - 1, dtype=np.uint64)
    t0 = time.time()

    gen_time = 0
    for seg_index in range(int(len(seg_list) // n)):
        t_ = time.time()

        start_index = int(seg_list[seg_index * n] * len(arr0))
        end_index = int(seg_list[(seg_index + 1) * n] * len(arr0))
        arr0t = arr0[start_index:end_index]

        seg = seg_list[(seg_index * n): ((seg_index + 1) * n + 1)]
        scaled_seg = (seg - seg[0]) / (seg[-1] - seg[0])

        length_factors = length_factors_list[seg_index]
        length_factor = predict_next_length_factor_quadratic(length_factors)
        length_factor *= 1.5 if isfree else 1.33
        length_factor *= length_factor_multiplier  # type: ignore

        arr1t, arr2t, percents2, percents1, hashmap1, hashmap2 = \
            gen_boards(arr0t, target, position, bm, pattern_check_func, success_check_func, to_find_func, scaled_seg,
                       hashmap1, hashmap2, n, length_factor, do_check, isfree)

        validate_length_and_balance(arr0t, arr2t, arr1t, seg, percents2, percents1, length_factor, True)

        actual_lengths2[(seg_index * n): ((seg_index + 1) * n)] = percents2 * len(arr2t)
        actual_lengths1[(seg_index * n): ((seg_index + 1) * n)] = percents1 * len(arr1t)
        length_factors_list[seg_index] = length_factors[1:] + [len(arr2t) / (1 + len(arr0t))]

        gen_time += time.time() - t_

        pivots = pivots_list[seg_index]
        sort_array(arr1t, pivots)
        sort_array(arr2t, pivots)
        pivots_list[seg_index] = arr2t[[len(arr2t) // 8 * i for i in range(1, 8)]] if len(arr2t) > 0 else \
            pivots_list[0].copy()

        arr1t, arr2t = parallel_unique(arr1t, n), parallel_unique(arr2t, n)
        arr1s.append(arr1t)
        arr2s.append(arr2t)

        del arr1t, arr2t, arr0t
        gc.collect()

    seg_list, length_factors_list, pivots_list, length_factor_multiplier \
        = update_parameters_big(actual_lengths2, actual_lengths1, seg_list, n, length_factors_list, pivots_list)

    t2 = time.time()
    gc.collect()

    return (arr1s, arr2s, pivots_list, seg_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2, t0,
            gen_time, t2)


def update_parameters_big(actual_lengths2, actual_lengths1, seg_list, n, length_factors_list, pivots_list):
    # 更新seg list
    percents2 = actual_lengths2 / actual_lengths2.sum()
    percents1 = actual_lengths1 / actual_lengths1.sum()
    logger.debug('Segmentation_pr ' + repr(np.round(seg_list, 5)))
    logger.debug('Segmentation_ac ' + repr(np.round(percents2, 5)))
    logger.debug('Segmentation_ac ' + repr(np.round(percents1, 5)))
    seg_list = update_seg(seg_list, percents2, 0.5)
    length_factor_multiplier = max(max(percents2) / np.mean(percents2), max(percents1) / np.mean(percents1))

    # 如果每次循环生成的数组过长则进行一次细分
    if np.mean(actual_lengths2) * n > 20971520 * (round(psutil.virtual_memory().total / (1024 ** 3),0) * 0.75):
        seg_list = split_seg(seg_list)
        length_factors_list = split_length_factor_list(length_factors_list)
        pivots_list = split_pivots_list(pivots_list)

    # 数组长度过短，则进行逆向细分操作
    if np.mean(actual_lengths2) * n < 524288 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75):
        seg_list = reverse_split_seg(seg_list)
        length_factors_list = reverse_split_length_factor_list(length_factors_list)
        pivots_list = reverse_split_pivots_list(pivots_list)

    return seg_list, length_factors_list, pivots_list, length_factor_multiplier


@njit(nogil=True, parallel=True)
def is_sorted(arr: np.ndarray) -> bool:
    return np.all(arr[:-1] < arr[1:])


@njit(nogil=True, parallel=True)
def is_sorted2(arr: np.ndarray) -> bool:
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


def initialize_parameters(n, pathname, isfree):
    """
    初始化分段间隔和长度乘数
    """
    seg_list_path = os.path.join(os.path.dirname(pathname), "seg_list.txt")
    length_factors_list_path = os.path.join(os.path.dirname(pathname), 'length_factors_list.txt')
    try:
        seg_list = np.loadtxt(seg_list_path, delimiter=',', dtype=np.float64)
        seg = extract_uniform_elements(n, seg_list)
    except FileNotFoundError:
        seg = np.array([round(1 / n * i, 4) for i in range(n + 1)], dtype=float)
        seg_list = seg

    percents = np.array([1 / n for _ in range(n)])
    # 预分配的储存局面的数组长度乘数
    length_factor_multiplier = 2.5

    try:
        # noinspection PyUnresolvedReferences
        length_factors_list = np.loadtxt(length_factors_list_path, delimiter=',', dtype=np.float64).tolist()
        if isinstance(length_factors_list[0], float):
            length_factors_list = [length_factors_list]
        length_factors = harmonic_mean_by_column(length_factors_list)
        length_factor = predict_next_length_factor_quadratic(length_factors) * 1.2
    except FileNotFoundError:
        length_factor = 3.2
        length_factors = [length_factor, length_factor, length_factor]
        length_factors_list = [length_factors]

    # 决定是否使用gen_boards_big处理
    segment_size = 5120000 if isfree else 8192000
    segment_size *= (round(psutil.virtual_memory().total / (1024 ** 3), 0) - 5)

    return seg, seg_list, seg_list_path, length_factor, length_factors, length_factors_list, length_factors_list_path, \
        percents, length_factor_multiplier, segment_size


def handle_restart(i, pathname, arr_init, started, d0, d1):
    """
    处理断点重连逻辑
    """
    if (os.path.exists(pathname + str(i + 1)) and os.path.exists(pathname + str(i))) \
            or (os.path.exists(pathname + str(i + 1) + '.book') and os.path.exists(pathname + str(i))) \
            or (os.path.exists(pathname + str(i + 1) + '.z') and os.path.exists(pathname + str(i))) \
            or os.path.exists(pathname + str(i) + '.book') \
            or os.path.exists(pathname + str(i) + '.z') \
            or os.path.exists(pathname + str(i) + '.book.7z') \
            or os.path.exists(pathname + str(i) + '.7z'):
        logger.debug(f"skipping step {i}")
        return False, None, None
    if i == 1:
        arr_init.tofile(pathname + str(i - 1))
        return True, arr_init, np.empty(0, dtype=np.uint64)
    elif not started:
        d0 = np.fromfile(pathname + str(i - 1), dtype=np.uint64)
        d1 = np.fromfile(pathname + str(i), dtype=np.uint64)
        return True, d0, d1

    return True, d0, d1


@njit(nogil=True, parallel=True)
def update_hashmap_length(hashmap: np.ndarray[np.uint64], arr: np.ndarray[np.uint64]) -> np.ndarray[np.uint64]:
    """根据当前layer大小调整哈希表大小"""
    length = max(largest_power_of_2(len(arr)), 1048576)
    if len(hashmap) < length:
        if len(hashmap) > 0:
            hashmap_new = np.empty(len(hashmap) * 2, dtype='uint64')
            hashmap_new[:len(hashmap)] = hashmap
            hashmap_new[len(hashmap):] = hashmap
            return hashmap_new
        else:
            return np.empty(length, dtype='uint64')
    else:
        return hashmap


def generate_process(
        arr_init: np.ndarray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc1,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        bm: BoardMover,
        isfree: bool
) -> Tuple[bool, np.ndarray[np.uint64], np.ndarray[np.uint64]]:
    started = False  # 是否进行了计算，如果是则需要进行final_steps处理最后一批局面
    d0, d1 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    pivots, pivots_list = None, None  # 用于快排的分割点
    hashmap1, hashmap2 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    n = max(4, min(32, os.cpu_count()))   # 并行线程数
    seg, seg_list, seg_list_path, length_factor, length_factors, length_factors_list, length_factors_list_path,\
        percents2, length_factor_multiplier, segment_size = initialize_parameters(n, pathname, isfree)
    percents1 = percents2.copy()

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        started, d0, d1 = handle_restart(i, pathname, arr_init, started, d0, d1)
        if not started:
            continue

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')
            pivots_list = [pivots] * int(len(seg_list) // n)

        # 生成新的棋盘状态
        if len(d0) < segment_size:
            t0 = time.time()

            if len(d0) < 10000 or arr_init[0] in (np.uint64(0xffff00000000ffff), np.uint64(0x000f000f000fffff)) or \
                    (3.2 in length_factors):
                # 数组较小(或2x4，3x3或断点重连)的应用简单方法
                d1t, d2 = gen_boards_simple(d0, target, position, bm, pattern_check_func, success_check_func,
                                            to_find_func, i > docheck_step, isfree)
            else:
                # 先预测预分配数组的长度乘数
                length_factor = predict_next_length_factor_quadratic(length_factors)
                length_factor *= 1.33 if isfree else 1.25
                length_factor *= length_factor_multiplier
                if len(hashmap1) == 0:
                    hashmap1, hashmap2 = update_hashmap_length(hashmap1, d0), update_hashmap_length(hashmap2, d0)  # 初始化

                d1t, d2, percents2, percents1, hashmap1, hashmap2 = \
                    gen_boards(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func, seg,
                               hashmap1, hashmap2, n, length_factor, i > docheck_step, isfree)

                validate_length_and_balance(d0, d2, d1t, seg, percents2, percents1, length_factor)

            t1 = time.time()
            # 排序
            sort_array(d1t, pivots)
            sort_array(d2, pivots)

            seg, seg_list, length_factors, length_factors_list, length_factor_multiplier, pivots, pivots_list\
                = update_parameters(d0, d2, seg, percents2, percents1, length_factors, seg_list_path, length_factors_list_path)

            t2 = time.time()

            # 去重
            d1t, d2 = parallel_unique(d1t, n), parallel_unique(d2, n)
            dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy()
            d1 = concatenate(merge_deduplicate_all([d1, d1t], dedup_pivots, n))
            # check_sorted(d1)

            t3 = time.time()
            log_performance(i, t0, t1, t2, t3, d1)

            d0, d1 = d1, d2
            del d1t, d2
            if len(hashmap1) > 0:
                hashmap1, hashmap2 = update_hashmap_length(hashmap1, d1), update_hashmap_length(hashmap2, d1)
        else:
            if len(hashmap1) == 0:
                hashmap_max_length = 20971520 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75)
                hashmap1, hashmap2 = (np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64),
                                      np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64))  # 初始化
            (d1s, d2s, pivots_list, seg_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2,
             t0, gen_time, t2) = \
                gen_boards_big(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                               seg_list, pivots_list, hashmap1, hashmap2, n, length_factors_list,
                               length_factor_multiplier, i > docheck_step, isfree)

            dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy()

            del d0
            d2 = merge_deduplicate_all(d2s, dedup_pivots, n)
            del d2s
            d2 = concatenate(d2)
            # check_sorted(d1)
            d1s.append(d1)
            d1 = merge_deduplicate_all(d1s, dedup_pivots, n)
            del d1s
            d1 = concatenate(d1)
            # check_sorted(d0)

            d0, d1 = d1, d2

            t3 = time.time()
            log_performance(i, t0, gen_time + t0, t2, t3, d0)

            np.savetxt(seg_list_path, seg_list, fmt='%.6f', delimiter=',')  # type: ignore
            np.savetxt(length_factors_list_path, length_factors_list, fmt='%.6f', delimiter=',')  # type: ignore
            seg = extract_uniform_elements(n, seg_list)
            length_factors = harmonic_mean_by_column(length_factors_list)

        d0.tofile(pathname + str(i))
        if SingletonConfig().config['compress_temp_files'] and i > 5:
            compress_with_7z(pathname + str(i - 2))
        hashmap1, hashmap2 = hashmap2, hashmap1

    return started, d0, d1


def validate_length_and_balance(d0, d2, d1t, seg, percents2, percents1, length_factor, isbig=False):
    if len(d0) > 2999999 and len(d2) > 2999999 and (
            max(percents2) / np.mean(percents2) >= length_factor / (len(d2) / len(d0)) or
            max(percents1) / np.mean(percents1) >= length_factor / (len(d1t) / len(d0))):
        logger.critical(
            f"length multiplier {length_factor:2f}, "
            f"actual multiplier {(len(d2) / len(d0)):2f}, {(len(d1t) / len(d0)):2f}"
            f"percents {np.round(percents2, 5)} {np.round(percents1, 5)}")
        raise IndexError("The length multiplier is not big enough. "
                         "This does not indicate an error in the program. "
                         "Please restart the program and continue running.")
    if not isbig:
        logger.debug('Segmentation_pr ' + repr(np.round(seg, 5)))
        logger.debug('Segmentation_ac ' + repr(np.round(percents2, 5)))
        logger.debug('Segmentation_ac ' + repr(np.round(percents1, 5)))
    if len(d0) > 0:
        logger.debug(
            f'length {len(d1t)}, Actual multiplier {round(len(d1t) / len(d0), 2)}, Using {round(length_factor, 2)}')
        logger.debug(
            f'length {len(d2)}, Actual multiplier {round(len(d2) / len(d0), 2)}, Using {round(length_factor, 2)}')


def update_parameters(d0, d2, seg, percents, percents1, length_factors, seg_list_path, length_factors_list_path):
    # 根据上一层实际分段情况调整下一层分段间隔
    seg = update_seg(seg, percents, 0.4)
    seg_list = seg
    np.savetxt(seg_list_path, seg_list, fmt='%.6f', delimiter=',')  # type: ignore

    # 更新数组长度乘数
    length_factors = length_factors[1:] + [max(len(d2) / (1 + len(d0)), 0.8)]
    length_factors_list = [length_factors]
    length_factor_multiplier = max(max(percents) / np.mean(percents), max(percents1) / np.mean(percents1))
    np.savetxt(length_factors_list_path, length_factors_list, fmt='%.6f', delimiter=',')  # type: ignore
    # 选取更准确的分割点使下一次快排更有效
    pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]].copy() if len(d2) > 0 else np.zeros(7, dtype='uint64')
    pivots_list = [pivots]
    return seg, seg_list, length_factors, length_factors_list, length_factor_multiplier, pivots, pivots_list


def log_performance(i, t0, t1, t2, t3, d1):
    if t3 > t0:
        logger.debug(f'step {i} generated: {round(len(d1) / (t3 - t0) / 1e6, 2)} mbps')
        logger.debug(f'generate/sort/deduplicate: {round((t1 - t0) / (t3 - t0), 2)}/'
                     f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}\n')


@njit(nogil=True)
def merge_and_deduplicate(sorted_arr1: np.ndarray, sorted_arr2: np.ndarray) -> np.ndarray:
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


@njit(nogil=True)
def hash_(board: np.uint64) -> np.uint64:
    return np.uint64((board ^ (board >> 27)) * 0x1A85EC53 + board >> 23)


@njit(nogil=True, parallel=True)
def gen_boards(arr0: np.ndarray[np.uint64],
               target: int,
               position: int,
               bm: BoardMover,
               pattern_check_func: PatternCheckFunc,
               success_check_func: SuccessCheckFunc,
               to_find_func: ToFindFunc1,
               seg: np.ndarray[float],
               hashmap1: np.ndarray[np.uint64],
               hashmap2: np.ndarray[np.uint64],
               n: int = 8,
               length_factor: float = 8,
               do_check: bool = True,
               isfree: bool = False
               ) -> \
        Tuple[np.ndarray[np.uint64], np.ndarray[np.uint64], np.ndarray[float], np.ndarray[float],
        np.ndarray[np.uint64], np.ndarray[np.uint64]]:
    """
    根据arr0中的面板，先生成数字，再移动，如果移动后仍是定式范围内且移动有效，则根据生成的数字（2,4）分别填入
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    min_length = 99999999 if isfree else 69999999
    length = max(min_length, int(len(arr0) * length_factor))
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    c1, c2 = np.empty(n, dtype=np.uint64), np.empty(n, dtype=np.uint64)
    hashmap1_length = len(hashmap1) - 1  # 要减一，这个长度用于计算哈希的时候取模
    hashmap2_length = len(hashmap2) - 1

    # seg = [0, 0.055, 0.13, 0.28, 0.42, 0.57, 0.72, 0.86, 1]
    # seg = [0, 0.13, 0.42, 0.72, 1]
    # seg = [0, 0.07, 0.21, 0.41, 0.61, 0.81, 1]

    starts = np.array([length // n * i for i in range(n)])

    for s in prange(n):
        start, end = int(seg[s] * len(arr0)), int(seg[s + 1] * len(arr0))
        c1t, c2t = length // n * s, length // n * s
        for b in range(start, end):
            t = arr0[b]
            if do_check and success_check_func(t, target, position):
                continue
            for i in range(16):  # 遍历每个位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):
                    t1 = t | (np.uint64(1) << np.uint64(4 * i))  # 填充数字2
                    for newt in bm.move_all_dir(t1):
                        if newt != t1 and pattern_check_func(newt):
                            newt = to_find_func(newt)
                            hashed_newt = (hash_(newt)) & hashmap1_length
                            if hashmap1[hashed_newt] != newt:
                                hashmap1[hashed_newt] = newt
                                arr1[c1t] = newt
                                c1t += 1

                    t1 = t | (np.uint64(2) << np.uint64(4 * i))  # 填充数字4
                    for newt in bm.move_all_dir(t1):
                        if newt != t1 and pattern_check_func(newt):
                            newt = to_find_func(newt)
                            hashed_newt = (hash_(newt)) & hashmap2_length
                            if hashmap2[hashed_newt] != newt:
                                hashmap2[hashed_newt] = newt
                                arr2[c2t] = newt
                                c2t += 1

        c1[s] = c1t
        c2[s] = c2t

    # 统计每个分段生成的新局面占比，用于平衡下一层的分组seg
    all_length = c2 - starts
    percents2 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n * i for i in range(n + 1)])

    all_length = c1 - starts
    percents1 = all_length / all_length.sum() if all_length.sum() > 0 else np.array([1 / n * i for i in range(n + 1)])

    arr1 = merge_inplace(arr1, c1, starts.copy())
    arr2 = merge_inplace(arr2, c2, starts.copy())

    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2, percents2, percents1, hashmap1, hashmap2


@njit(nogil=True, parallel=True)
def gen_boards_simple(arr0: np.ndarray[np.uint64],
                      target: int,
                      position: int,
                      bm: BoardMover,
                      pattern_check_func: PatternCheckFunc,
                      success_check_func: SuccessCheckFunc,
                      to_find_func: ToFindFunc1,
                      do_check: bool = True,
                      isfree: bool = False
                      ) -> Tuple[np.ndarray[np.uint64], np.ndarray[np.uint64]]:
    """
    根据arr0中的面板，先生成数字，再移动，如果移动后仍是定式范围内且移动有效，则根据生成的数字（2,4）分别填入
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    length = max(len(arr0) * 8, 499999999) if isfree else max(len(arr0) * 6, 199999999)
    arrs = [np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)]
    for p in prange(1, 3):
        arr = np.empty(length, dtype=np.uint64)
        ct = 0
        for t in arr0:
            # 如果当前棋盘状态已经符合成功条件，将其成功概率设为1
            if do_check and success_check_func(t, target, position):
                continue  # 由于已成功，无需进一步处理这个棋盘状态，继续下一个
            for i in range(16):  # 遍历每个位置
                # 检查第i位置是否为空，如果为空，进行填充操作
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):
                    # 分别用数字2和4填充当前空位，然后生成新的棋盘状态t1和t2
                    t1 = t | (np.uint64(p) << np.uint64(4 * i))  # 填充数字2（2的对数为1，即4位中的0001）
                    # 尝试所有四个方向上的移动
                    for newt in bm.move_all_dir(t1):
                        if newt != t1 and pattern_check_func(newt):
                            arr[ct] = to_find_func(newt)
                            ct += 1
        arrs[p - 1] = arr[:ct]
    # 返回包含可能的新棋盘状态的两个array
    return arrs[0], arrs[1]


@njit(nogil=True, parallel=True)
def merge_inplace(arr: np.ndarray, segment_ends: np.ndarray, segment_starts: np.ndarray) -> np.ndarray:
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
    counts = segment_ends[0]
    num_segments = len(segment_starts)

    for i in range(1, num_segments):
        start = segment_starts[i]
        end = segment_ends[i]
        size = end - start
        dest_start = counts
        dest_end = counts + size

        # 检查读写区间是否重叠
        if start < dest_end:
            arr[dest_start: start] = arr[dest_end: end]
        else:
            arr[dest_start: dest_end] = arr[start: end]
        counts += size

    return arr[:counts]


def predict_next_length_factor_quadratic(length_factors: List[float]) -> float:
    if np.allclose(length_factors, length_factors[-1], atol=0.1):
        return length_factors[-1]

    n = len(length_factors)
    x = np.arange(n)
    # 拟合一个二次多项式
    coefficients = np.polyfit(x, length_factors, 2)
    # 使用拟合的二次模型预测下一个数据点
    next_length_factor = np.polyval(coefficients, n)
    next_length_factor = min(max(next_length_factor, np.mean(length_factors)), length_factors[-1] * 2.5)  # type: ignore
    return next_length_factor


def update_seg(seg: np.ndarray[float], percents: np.ndarray[float], learning_rate: float = 0.5) -> np.ndarray[float]:
    seg_length = seg[1:] - seg[:-1]
    if 0 not in percents:
        normalised = (seg_length / percents)
    else:
        normalised = (seg_length / (percents + 0.5 / len(percents)))
    seg_length_new = normalised / normalised.sum()
    seg[1:] = np.cumsum(seg_length_new * learning_rate + seg_length * (1 - learning_rate))
    seg[0], seg[-1] = 0, 1  # 避免浮点数精度导致的bug
    return seg


def split_seg(seg: np.ndarray[float]) -> np.ndarray[float]:
    new_seg = np.empty(len(seg) * 2 - 1, dtype=float)
    new_seg[-1] = 1
    for i in range(len(seg) - 1):
        new_seg[i * 2] = seg[i]
    for i in range(len(seg) - 1):
        new_seg[i * 2 + 1] = (new_seg[i * 2] + new_seg[i * 2 + 2]) / 2
    return new_seg


def split_length_factor_list(length_factor_list: List[List[float]]) -> List[List[float]]:
    length_factor_list_new: List[List[float]] = [[i * 1.5 for i in length_factor_list[0]]]
    for i in length_factor_list:
        length_factor_list_new.append(i)
        length_factor_list_new.append(i)
    length_factor_list_new.pop()
    return length_factor_list_new


def split_pivots_list(pivots_list: List[np.ndarray[np.uint64]]) -> List[np.ndarray[np.uint64]]:
    pivots_list_new: List[np.ndarray[np.uint64]] = []
    for i in pivots_list:
        pivots_list_new.append(i)
        pivots_list_new.append(i)
    return pivots_list_new


def reverse_split_seg(seg: np.ndarray[float]) -> np.ndarray[float]:
    return seg[::2]


def reverse_split_length_factor_list(length_factor_list: List[List[float]]) -> List[List[float]]:
    length_factor_list_new = []
    for sublist in length_factor_list:
        length_factor_list_new.append(sublist[::2])
    return length_factor_list_new


def reverse_split_pivots_list(pivots_list: List[np.ndarray[np.uint64]]) -> List[np.ndarray[np.uint64]]:
    return pivots_list[::2]


def extract_uniform_elements(n: int, arr: np.ndarray) -> np.ndarray:
    k = (len(arr) - 1) // n  # 计算 k
    result = arr[::k]  # 每隔 k 个取一个，包括第一个和最后一个
    return result


def harmonic_mean_by_column(matrix: List[List[float]]) -> List[float]:
    num_columns = len(matrix[0])
    harmonic_means = []
    for col in range(num_columns):
        reciprocals = [1 / row[col] for row in matrix]
        harmonic_mean = len(matrix) / sum(reciprocals)
        harmonic_means.append(harmonic_mean)
    return harmonic_means
