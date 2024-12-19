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
        _pivots_ptr = (c_uint64 * 7)(*_pivots)
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


@njit(nogil=True)
def unique(aux: np.ndarray[np.uint64]) -> np.uint64:
    c = 0
    for i in range(0, len(aux)):
        if aux[i] != aux[i - 1]:
            aux[np.uint64(c)] = aux[i]
            c += 1
    return aux[:c].copy()


@njit(nogil=True)
def largest_power_of_2(n):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@njit(nogil=True, parallel=True)
def p_unique(arrs: List[np.ndarray[np.uint64]]
             ) -> List[np.ndarray[np.uint64]]:
    for i in prange(2):
        arrs[i] = unique(arrs[i])
    return arrs


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


def merge_deduplicate_all(arrays: List[np.ndarray], pivots_array, n_threads: int | None = None) -> list[np.ndarray]:
    """
    多线程合并并去重多个已排序的数组
    """
    n_threads = os.cpu_count() if n_threads is None else n_threads
    num_arrays = len(arrays)
    pivots = pivots_array[np.arange(1, n_threads) * len(pivots_array) // n_threads]

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
    res = [np.empty(0, dtype='uint64')] * n_threads
    # 并行归并每个分区
    for t in prange(n_threads):
        temp_arrays = [np.empty(0, dtype='uint64')] * num_arrays
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


def gen_boards_big(step: int,
                   arr0: np.ndarray[np.uint64],
                   target: int,
                   position: int,
                   bm: BoardMover,
                   pattern_check_func: PatternCheckFunc,
                   success_check_func: SuccessCheckFunc,
                   to_find_func: ToFindFunc,
                   d1: np.ndarray[np.uint64],
                   seg_list: np.ndarray[float],
                   pivots_list: List[np.ndarray[np.uint64]],
                   n: int,
                   length_factors_list: List[List[float]],
                   length_factor_multiplier: float = 1.5,
                   do_check: bool = True,
                   isfree: bool = False,
                   ) -> Tuple[
    np.ndarray[np.uint64], np.ndarray[np.uint64], List[np.ndarray[np.uint64]], np.ndarray[float], List[
        List[float]], float]:
    """
    将arr0分段放入gen_boards生成排序去重后的局面，然后归并
    """
    arr1s: List[np.ndarray[np.uint64]] = [d1]
    arr2s: List[np.ndarray[np.uint64]] = []
    actual_lengths: np.ndarray[np.uint64] = np.empty(len(seg_list) - 1, dtype=np.uint64)
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
        length_factor *= 2.4 if isfree else 2
        length_factor *= length_factor_multiplier  # type: ignore

        arr1t, arr2t, percents, percents1 = gen_boards(arr0t, target, position, bm, pattern_check_func,
                                                       success_check_func,
                                                       to_find_func, scaled_seg, n, length_factor, do_check, isfree)

        if len(arr0t) > 2999999 and len(arr2t) > 2999999 and \
                (max(percents) / np.mean(percents) >= length_factor / (len(arr2t) / len(arr0t)) or
                 max(percents1) / np.mean(percents1) >= length_factor / (len(arr1t) / len(arr0t))):
            logger.critical(
                f"length multiplier {length_factor:2f}, "
                f"actual multiplier {(len(arr2t) / len(arr0t)):2f}, {(len(arr1t) / len(arr0t)):2f}"
                f"percents {np.round(percents, 3)} {np.round(percents1, 3)}")
            raise IndexError("The length multiplier is not big enough. "
                             "This does not indicate an error in the program. "
                             "Please restart the program and continue running.")

        if len(arr0t) > 0:
            logger.debug(f'Actual multiplier {round(len(arr2t) / len(arr0t), 2)}, Using {round(length_factor, 2)}')

        actual_lengths[(seg_index * n): ((seg_index + 1) * n)] = percents * len(arr1t)
        length_factors_list[seg_index] = length_factors[1:] + [len(arr2t) / (1 + len(arr0t))]

        gen_time += time.time() - t_

        pivots = pivots_list[seg_index]
        sort_array(arr1t, pivots)
        sort_array(arr2t, pivots)
        pivots_list[seg_index] = arr2t[[len(arr2t) // 8 * i for i in range(1, 8)]] if len(arr2t) > 0 else \
            pivots_list[0].copy()

        arr1t, arr2t = p_unique([arr1t, arr2t])
        arr1s.append(arr1t.copy())  # 原地unique返回的是切片，不会释放数组后半部分占用的内存，这里需要复制
        arr2s.append(arr2t.copy())

        del arr1t, arr2t, arr0t
        gc.collect()

    # 更新seg list
    percents = actual_lengths / actual_lengths.sum()
    logger.debug('Segmentation_pr ' + repr(np.round(seg_list, 3)))
    logger.debug('Segmentation_ac ' + repr(np.round(percents, 3)))
    seg_list = update_seg(seg_list, percents, 0.5)
    # 如果每次循环生成的数组过长则进行一次细分
    if np.mean(actual_lengths) * n > 16777216 * round(psutil.virtual_memory().total / (1024**3),0):
        seg_list = split_seg(seg_list)
        length_factors_list = split_length_factor_list(length_factors_list)
        pivots_list = split_pivots_list(pivots_list)

    t2 = time.time()

    gc.collect()

    arr1 = merge_deduplicate_all(arr1s, arr0)
    del arr1s
    arr1 = concatenate(arr1)
    check_sorted(arr1)
    arr2 = merge_deduplicate_all(arr2s, arr0)
    del arr2s
    arr2 = concatenate(arr2)
    check_sorted(arr2)

    t3 = time.time()

    if t3 > t0:
        logger.debug(f'step {step} (big) generated: {round(len(arr1) / (t3 - t0) / 1e6, 2)} mbps')
        logger.debug(f'generate/sort/deduplicate: {round(gen_time / (t3 - t0), 2)}/'
                     f'{round((t2 - t0 - gen_time) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}\n')

    return arr1, arr2, pivots_list, seg_list, length_factors_list, max(percents) / np.mean(percents)


@njit(nogil=True, parallel=True)
def is_sorted(arr: np.ndarray) -> bool:
    return np.all(arr[:-1] < arr[1:])


def check_sorted(arr):
    if is_sorted(arr):
        return
    raise ValueError


def generate_process(
        arr_init: np.ndarray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        bm: BoardMover,
        isfree: bool
) -> Tuple[bool, np.ndarray[np.uint64], np.ndarray[np.uint64]]:
    # 是否进行了计算，如果是则需要进行final_steps处理最后一批局面
    started = False
    # n, n+2 的局面， 其中 d1 由 n-2 层生成
    d0, d1 = None, None

    # _list用于记录gen_boards_big中多个分段的seg\lengths\pivots信息
    # 用于快排的分割点
    pivots, pivots_list = None, None
    # 并行线程数
    n = max(4, min(32, os.cpu_count()))
    # 并行分段间隔
    seg_list_path = os.path.join(os.path.dirname(pathname), "seg_list.txt")
    length_factors_list_path = os.path.join(os.path.dirname(pathname), 'length_factors_list.txt')
    try:
        seg_list: np.ndarray[float] = np.loadtxt(seg_list_path, delimiter=',', dtype=np.float64)
        seg = extract_uniform_elements(n, seg_list)
    except FileNotFoundError:
        seg: np.ndarray = np.array([round(1 / n * i, 4) for i in range(n + 1)], dtype=float)
        seg_list: np.ndarray[float] = seg
    # 实际分段情况
    percents = np.array([1 / n for _ in range(n)])
    # 预分配的储存局面的数组长度乘数
    length_factor_multiplier = 1.5
    try:
        # noinspection PyUnresolvedReferences
        length_factors_list: List[List[float]] = \
            np.loadtxt(length_factors_list_path, delimiter=',', dtype=np.float64).tolist()
        if isinstance(length_factors_list[0], float):  # 如果length_factors_list_path是1*n二维数组，写入txt再读取会变成一维
            length_factors_list = [length_factors_list]  # type:ignore
        length_factors: List[float] = harmonic_mean_by_column(length_factors_list)
        length_factor: float = predict_next_length_factor_quadratic(length_factors) * 1.2
    except FileNotFoundError:
        length_factor = 3.2
        length_factors: List[float] = [length_factor, length_factor, length_factor]
        length_factors_list: List[List[float]] = [length_factors]

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        # 断点重连
        if (os.path.exists(pathname + str(i + 1)) and os.path.exists(pathname + str(i))) \
                or (os.path.exists(pathname + str(i + 1) + '.book') and os.path.exists(pathname + str(i))) \
                or (os.path.exists(pathname + str(i + 1) + '.z') and os.path.exists(pathname + str(i))) \
                or os.path.exists(pathname + str(i) + '.book') \
                or os.path.exists(pathname + str(i) + '.z'):
            logger.debug(f"skipping step {i}")
            continue
        if i == 1:
            arr_init.tofile(pathname + str(i - 1))
            d0, d1 = arr_init, np.empty(0, dtype=np.uint64)
            started = True
        elif not started:
            d0 = np.fromfile(pathname + str(i - 1), dtype=np.uint64)
            d1 = np.fromfile(pathname + str(i), dtype=np.uint64)
            started = True

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')
            pivots_list = [pivots] * int(len(seg_list) // n)

        # 决定是否使用gen_boards_big处理
        segment_size = 4194304 if isfree else 6553600
        segment_size *= round(psutil.virtual_memory().total / (1024**3),0)
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
                length_factor *= 2.25 if isfree else 1.8
                length_factor *= length_factor_multiplier

                d1t, d2, percents, percents1 = \
                    gen_boards(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func, seg,
                               n, length_factor, i > docheck_step, isfree)

                if len(d0) > 2999999 and len(d2) > 2999999 and (
                        max(percents) / np.mean(percents) >= length_factor / (len(d2) / len(d0)) or
                        max(percents1) / np.mean(percents1) >= length_factor / (len(d1t) / len(d0))):
                    logger.critical(
                        f"length multiplier {length_factor:2f}, "
                        f"actual multiplier {(len(d2) / len(d0)):2f}, {(len(d1t) / len(d0)):2f}"
                        f"percents {np.round(percents, 3)} {np.round(percents1, 3)}")
                    raise IndexError("The length multiplier is not big enough. "
                                     "This does not indicate an error in the program. "
                                     "Please restart the program and continue running.")
                # 根据上一层实际分段情况调整下一层分段间隔
                logger.debug('Segmentation_pr ' + repr(np.round(seg, 3)))
                logger.debug('Segmentation_ac ' + repr(np.round(percents, 3)))
                logger.debug('Segmentation_ac ' + repr(np.round(percents1, 3)))
                seg = update_seg(seg, percents, 0.4)
                seg_list = seg

            if len(d0) > 0:
                logger.debug(f'Actual multiplier {round(len(d2) / len(d0), 2)}, Using {round(length_factor, 2)}')
            # 更新数组长度乘数
            length_factors = length_factors[1:] + [max(len(d2) / (1 + len(d0)), 0.8)]
            length_factors_list = [length_factors]
            length_factor_multiplier = max(percents) / np.mean(percents)
            t1 = time.time()

            np.savetxt(seg_list_path, seg_list, fmt='%.3f', delimiter=',')  # type: ignore
            np.savetxt(length_factors_list_path, length_factors_list, fmt='%.3f', delimiter=',')  # type: ignore

            # 排序
            sort_array(d1t, pivots)
            sort_array(d2, pivots)
            # 选取更准确的分割点使下一次快排更有效
            pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]] if len(d2) > 0 else np.zeros(7, dtype='uint64')
            pivots_list = [pivots]
            t2 = time.time()

            # 去重
            d1t, d2 = p_unique([d1t, d2])
            d1 = merge_and_deduplicate(d1, d1t)
            check_sorted(d1)

            t3 = time.time()
            if t3 > t0:
                logger.debug(f'step {i} generated: {round(len(d1) / (t3 - t0) / 1e6, 2)} mbps')
                logger.debug(f'generate/sort/deduplicate: {round((t1 - t0) / (t3 - t0), 2)}/'
                             f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}\n')

            d0, d1 = d1, d2
            del d1t, d2
        else:
            d0, d1, pivots_list, seg_list, length_factors_list, length_factor_multiplier = \
                gen_boards_big(i, d0, target, position, bm, pattern_check_func, success_check_func, to_find_func, d1,
                               seg_list, pivots_list, n, length_factors_list, length_factor_multiplier,
                               i > docheck_step, isfree)
            np.savetxt(seg_list_path, seg_list, fmt='%.3f', delimiter=',')  # type: ignore
            np.savetxt(length_factors_list_path, length_factors_list, fmt='%.3f', delimiter=',')  # type: ignore
            seg = extract_uniform_elements(n, seg_list)
            length_factors = harmonic_mean_by_column(length_factors_list)

        d0.tofile(pathname + str(i))
        gc.collect()

    return started, d0, d1


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


@njit(nogil=True, parallel=True)
def gen_boards(arr0: np.ndarray[np.uint64],
               target: int,
               position: int,
               bm: BoardMover,
               pattern_check_func: PatternCheckFunc,
               success_check_func: SuccessCheckFunc,
               to_find_func: ToFindFunc,
               seg: np.ndarray[float],
               n: int = 8,
               length_factor: float = 8,
               do_check: bool = True,
               isfree: bool = False
               ) -> \
        Tuple[np.ndarray[np.uint64], np.ndarray[np.uint64], np.ndarray[float], np.ndarray[float]]:
    """
    根据arr0中的面板，先生成数字，再移动，如果移动后仍是定式范围内且移动有效，则根据生成的数字（2,4）分别填入
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    min_length = 99999999 if isfree else 69999999
    length = max(min_length, int(len(arr0) * length_factor))
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    c1, c2 = np.empty(n, dtype=np.uint64), np.empty(n, dtype=np.uint64)

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
                            arr1[c1t] = newt
                            c1t += 1
                    t1 = t | (np.uint64(2) << np.uint64(4 * i))  # 填充数字4
                    for newt in bm.move_all_dir(t1):
                        if newt != t1 and pattern_check_func(newt):
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
    # 求对称
    to_find_func(arr1)
    to_find_func(arr2)
    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2, percents2, percents1


@njit(nogil=True, parallel=True)
def gen_boards_simple(arr0: np.ndarray[np.uint64],
                      target: int,
                      position: int,
                      bm: BoardMover,
                      pattern_check_func: PatternCheckFunc,
                      success_check_func: SuccessCheckFunc,
                      to_find_func: ToFindFunc,
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
                            arr[ct] = newt
                            ct += 1
        arrs[p - 1] = arr[:ct]
    # 求对称
    to_find_func(arrs[0])
    to_find_func(arrs[1])
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


@njit(nogil=True)
def update_seg(seg: np.ndarray[float], percents: np.ndarray[float], learning_rate: float = 0.5) -> np.ndarray[float]:
    seg_length = seg[1:] - seg[:-1]
    normalised = (seg_length / percents)
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
