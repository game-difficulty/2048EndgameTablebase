import gc
import os
import time
from typing import Callable, Tuple, List

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import psutil

from BoardMover import BoardMover
from BookGeneratorUtils import sort_array, parallel_unique, concatenate, merge_deduplicate_all, largest_power_of_2, \
    hash_, merge_inplace
import Config
from Config import SingletonConfig
from LzmaCompressor import compress_with_7z

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]


logger = Config.logger


def gen_boards_big(arr0: NDArray[np.uint64],
                   target: int,
                   position: int,
                   bm: BoardMover,
                   pattern_check_func: PatternCheckFunc,
                   success_check_func: SuccessCheckFunc,
                   to_find_func: ToFindFunc,
                   pivots_list: List[NDArray[np.uint64]],
                   hashmap1: NDArray[np.uint64],
                   hashmap2: NDArray[np.uint64],
                   n: int,
                   length_factors_list: List[List[float]],
                   length_factor_multiplier: float = 1.5,
                   do_check: bool = True,
                   isfree: bool = False,
                   ) -> Tuple[
    List[NDArray[np.uint64]], List[NDArray[np.uint64]], List[NDArray[np.uint64]], List[
        List[float]], float, NDArray[np.uint64], NDArray[np.uint64], float, float, float]:
    """
    将arr0分段放入gen_boards生成排序去重后的局面，然后归并
    """
    segs_count = len(length_factors_list)
    arr1s: List[NDArray[np.uint64]] = []
    arr2s: List[NDArray[np.uint64]] = []
    actual_lengths1: NDArray[np.uint64] = np.empty(segs_count * n, dtype=np.uint64)
    actual_lengths2: NDArray[np.uint64] = np.empty(segs_count * n, dtype=np.uint64)
    seg_start_end = allocate_seg(length_factors_list, len(arr0))

    t0 = time.time()
    gen_time = 0
    for seg_index in range(segs_count):
        t_ = time.time()

        start_index = seg_start_end[seg_index]
        end_index = seg_start_end[seg_index + 1]
        arr0t = arr0[start_index:end_index]

        length_factors = length_factors_list[seg_index]
        length_factor = predict_next_length_factor_quadratic(length_factors)
        length_factor *= 1.25
        length_factor *= length_factor_multiplier  # type: ignore

        arr1t, arr2t, hashmap1, hashmap2, counts1, counts2 = \
            gen_boards(arr0t, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                       hashmap1, hashmap2, n, length_factor, do_check, isfree)

        validate_length_and_balance(arr0t, arr2t, arr1t, counts1, counts2, length_factor, True)

        actual_lengths2[(seg_index * n): ((seg_index + 1) * n)] = counts2
        actual_lengths1[(seg_index * n): ((seg_index + 1) * n)] = counts1
        length_factors_list[seg_index] = length_factors[1:] + [len(arr2t) / (1 + len(arr0t))]

        gen_time += time.time() - t_

        pivots = pivots_list[seg_index]
        sort_array(arr1t, pivots)
        sort_array(arr2t, pivots)
        pivots_list[seg_index] = arr2t[[len(arr2t) // 8 * i for i in range(1, 8)]] if len(arr2t) > 0 else \
            pivots_list[0].copy()

        arr1t = parallel_unique(arr1t, n)
        arr2t = parallel_unique(arr2t, n)
        arr1s.append(arr1t)
        arr2s.append(arr2t)

        del arr1t, arr2t, arr0t
        gc.collect()

    length_factors_list, pivots_list, length_factor_multiplier \
        = update_parameters_big(actual_lengths2, actual_lengths1, n, length_factors_list, pivots_list)

    t2 = time.time()
    gc.collect()

    return (arr1s, arr2s, pivots_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2, t0,
            gen_time, t2)


def update_parameters_big(actual_lengths2, actual_lengths1, n, length_factors_list, pivots_list):
    percents2 = actual_lengths2 / actual_lengths2.sum()
    percents1 = actual_lengths1 / actual_lengths1.sum()
    logger.debug('Segmentation_ac ' + repr(np.round(percents2, 5)))
    logger.debug('Segmentation_ac ' + repr(np.round(percents1, 5)))
    length_factor_multiplier = max(max(percents2) / np.mean(percents2), max(percents1) / np.mean(percents1))

    # 如果每次循环生成的数组过长则进行一次细分
    if np.mean(actual_lengths2) * n > 20971520 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75):
        length_factors_list = split_length_factor_list(length_factors_list)
        pivots_list = split_pivots_list(pivots_list)

    # 数组长度过短，则进行逆向细分操作
    if np.mean(actual_lengths2) * n < 524288 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75):
        length_factors_list = reverse_split_length_factor_list(length_factors_list)
        pivots_list = reverse_split_pivots_list(pivots_list)

    return length_factors_list, pivots_list, length_factor_multiplier


def initialize_parameters(n, pathname, isfree, default_length_factor = 3.2):
    """
    初始化长度乘数
    """
    length_factors_list_path = os.path.join(os.path.dirname(pathname), 'length_factors_list.txt')

    counts2 = np.ones(n, dtype=np.uint64)
    # 预分配的储存局面的数组长度乘数
    length_factor_multiplier = 2.0

    try:
        # noinspection PyUnresolvedReferences
        length_factors_list = np.loadtxt(length_factors_list_path, delimiter=',', dtype=np.float64).tolist()
        if isinstance(length_factors_list[0], float):
            length_factors_list = [length_factors_list]
        length_factors = harmonic_mean_by_column(length_factors_list)
        length_factor = predict_next_length_factor_quadratic(length_factors) * 1.2
    except FileNotFoundError:
        length_factor = default_length_factor
        length_factors = [length_factor, length_factor, length_factor]
        length_factors_list = [length_factors]

    # 决定是否使用gen_boards_big处理
    segment_size = 5120000 if isfree else 8192000
    segment_size *= (round(psutil.virtual_memory().total / (1024 ** 3), 0) - 5)

    return length_factor, length_factors, length_factors_list, length_factors_list_path, \
        counts2, length_factor_multiplier, segment_size


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
def update_hashmap_length(hashmap: NDArray[np.uint64], arr: NDArray[np.uint64]) -> NDArray[np.uint64]:
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
        arr_init: NDArray[np.uint64],
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
) -> Tuple[bool, NDArray[np.uint64], NDArray[np.uint64]]:
    started = False  # 是否进行了计算，如果是则需要进行final_steps处理最后一批局面
    d0, d1 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    pivots, pivots_list = None, None  # 用于快排的分割点
    hashmap1, hashmap2 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    n = max(4, min(32, os.cpu_count()))  # 并行线程数
    length_factor, length_factors, length_factors_list, length_factors_list_path, \
        counts2, length_factor_multiplier, segment_size = initialize_parameters(n, pathname, isfree)

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        started, d0, d1 = handle_restart(i, pathname, arr_init, started, d0, d1)
        if not started:
            continue

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')
            pivots_list = [pivots] * len(length_factors_list)

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
                length_factor *= 1.25
                length_factor *= length_factor_multiplier
                if len(hashmap1) == 0:
                    hashmap1, hashmap2 = update_hashmap_length(hashmap1, d0), update_hashmap_length(hashmap2, d0)  # 初始化

                d1t, d2, hashmap1, hashmap2, counts1, counts2 = \
                    gen_boards(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                               hashmap1, hashmap2, n, length_factor, i > docheck_step, isfree)

                validate_length_and_balance(d0, d2, d1t, counts1, counts2, length_factor, False)

            t1 = time.time()
            # 排序
            sort_array(d1t, pivots)
            sort_array(d2, pivots)

            length_factors, length_factors_list, pivots, pivots_list \
                = update_parameters(d0, d2, length_factors, length_factors_list_path)
            length_factor_multiplier = max(counts2) / np.mean(counts2)

            t2 = time.time()

            # 去重
            d1t = parallel_unique(d1t, n)
            d2 = parallel_unique(d2, n)
            dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy() if len(d0) > 0 else \
                (np.arange(1, n) * (1 << 50) // n).astype(np.uint64)
            d1 = merge_deduplicate_all([d1, d1t], dedup_pivots, n)
            del d1t
            d1 = concatenate(d1)
            # check_sorted(d1)

            t3 = time.time()
            log_performance(i, t0, t1, t2, t3, d1)

            d0, d1 = d1, d2
            del d2
            if len(hashmap1) > 0:
                hashmap1, hashmap2 = update_hashmap_length(hashmap1, d1), update_hashmap_length(hashmap2, d1)
        else:
            if len(hashmap1) == 0:
                hashmap_max_length = 20971520 * (round(psutil.virtual_memory().total / (1024 ** 3), 0) * 0.75)
                hashmap1, hashmap2 = (np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64),
                                      np.empty(largest_power_of_2(hashmap_max_length), dtype=np.uint64))  # 初始化
            (d1s, d2s, pivots_list, length_factors_list, length_factor_multiplier, hashmap1, hashmap2,
             t0, gen_time, t2) = \
                gen_boards_big(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                               pivots_list, hashmap1, hashmap2, n, length_factors_list,
                               length_factor_multiplier, i > docheck_step, isfree)

            dedup_pivots = d0[np.arange(1, n) * len(d0) // n].copy() if len(d0) > 0 else \
                (np.arange(1, n) * (1 << 50) // n).astype(np.uint64)

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

            np.savetxt(length_factors_list_path, length_factors_list, fmt='%.6f', delimiter=',')  # type: ignore
            length_factors = harmonic_mean_by_column(length_factors_list)

        d0.tofile(pathname + str(i))
        if SingletonConfig().config['compress_temp_files'] and i > 5:
            compress_with_7z(pathname + str(i - 2))
        hashmap1, hashmap2 = hashmap2, hashmap1

    return started, d0, d1


def validate_length_and_balance(d0, d2, d1t, counts1, counts2, length_factor, isbig):
    if len(d0) < 1999999 or len(d2) < 1999999:
        return

    length_needed = max(counts1.max(), counts2.max()) * len(counts1)
    length_factor_actual = length_needed / len(d0)
    length = max(69999999, int(len(d0) * length_factor))
    is_valid_length = length_needed <= length
    percents1 = counts1 / counts1.sum()
    percents2 = counts2 / counts2.sum()

    if not is_valid_length:
        logger.critical(
            f"length multiplier {length_factor:2f}, "
            f"need {length_factor_actual:2f}, \n"
            f"counts1 {(counts1 / 1e6):2f}, \n"
            f"counts2 {(counts2 / 1e6):2f}")
        raise IndexError("The length multiplier is not big enough. "
                         "This does not indicate an error in the program. "
                         "Please restart the program and continue running.")
    if not isbig:
        logger.debug(f'length {len(d1t)}, {len(d2)}, '
                     f'Using {round(length_factor, 2)}, '
                     f'Need {round(length_factor_actual, 2)}')
        logger.debug('Segmentation1_ac ' + repr(np.round(percents2, 5)))
        logger.debug('Segmentation2_ac ' + repr(np.round(percents1, 5)))
    if len(d0) > 0:
        logger.debug(
            f'length {len(d1t)}, {len(d2)}, '
            f'Using {round(length_factor, 2)}, '
            f'Need {round(length_factor_actual, 2)}')


def update_parameters(d0, d2, length_factors, length_factors_list_path):
    # 更新数组长度乘数
    length_factors = length_factors[1:] + [len(d2) / (len(d0) + 1)]
    length_factors_list = [length_factors]
    np.savetxt(length_factors_list_path, length_factors_list, fmt='%.6f', delimiter=',')  # type: ignore
    # 选取更准确的分割点使下一次快排更有效
    pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]].copy() if len(d2) > 0 else np.zeros(7, dtype='uint64')
    pivots_list = [pivots]
    return length_factors, length_factors_list, pivots, pivots_list


def log_performance(i, t0, t1, t2, t3, d1):
    if t3 > t0:
        logger.debug(f'step {i} generated: {round(len(d1) / (t3 - t0) / 1e6, 2)} mbps')
        logger.debug(f'generate/sort/deduplicate: {round((t1 - t0) / (t3 - t0), 2)}/'
                     f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}\n')


@njit(nogil=True, parallel=True)
def gen_boards(arr0: NDArray[np.uint64],
               target: int,
               position: int,
               bm: BoardMover,
               pattern_check_func: PatternCheckFunc,
               success_check_func: SuccessCheckFunc,
               to_find_func: ToFindFunc,
               hashmap1: NDArray[np.uint64],
               hashmap2: NDArray[np.uint64],
               n: int = 8,
               length_factor: float = 8,
               do_check: bool = True,
               isfree: bool = False
               ) -> \
        Tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64],
        NDArray[np.uint64], NDArray[np.uint64]]:
    """
    根据arr0中的面板，先生成数字，再移动，如果移动后仍是定式范围内且移动有效，则根据生成的数字（2,4）分别填入
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    min_length = 99999999 if isfree else 69999999
    length = max(min_length, int(len(arr0) * length_factor))
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    starts = np.array([length // n * i for i in range(n)], dtype=np.uint64)
    c1, c2 = starts.copy(),  starts.copy()
    hashmap1_length = len(hashmap1) - 1  # 要减一，这个长度用于计算哈希的时候取模
    hashmap2_length = len(hashmap2) - 1

    total_tasks = len(arr0)
    chunk_size = min(10 ** 6, total_tasks // (n * 5) + 1) * n
    # 向上取整
    chunks_count = (total_tasks + chunk_size - 1) // chunk_size

    for s in prange(n):
        c1t, c2t = length // n * s, length // n * s
        for chunk in range(chunks_count):
            chunk_start = chunk * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_tasks)

            thread_start = chunk_start + s * chunk_size // n
            thread_end = thread_start + chunk_size // n
            start = max(thread_start, chunk_start)
            end = min(thread_end, chunk_end)

            # 确保有效区间存在
            if not start < end:
                continue

            for b in range(start, end):
                t: np.uint64 = arr0[b]
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

    arr1 = merge_inplace(arr1, c1, starts.copy())
    arr2 = merge_inplace(arr2, c2, starts.copy())

    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2, hashmap1, hashmap2, c1-starts, c2-starts


@njit(nogil=True, parallel=True)
def gen_boards_simple(arr0: NDArray[np.uint64],
                      target: int,
                      position: int,
                      bm: BoardMover,
                      pattern_check_func: PatternCheckFunc,
                      success_check_func: SuccessCheckFunc,
                      to_find_func: ToFindFunc,
                      do_check: bool = True,
                      isfree: bool = False
                      ) -> Tuple[NDArray[np.uint64], NDArray[np.uint64]]:
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


def split_length_factor_list(length_factor_list: List[List[float]]) -> List[List[float]]:
    length_factor_list_new: List[List[float]] = [[i * 1.5 for i in length_factor_list[0]]]
    for i in length_factor_list:
        length_factor_list_new.append(i)
        length_factor_list_new.append(i)
    length_factor_list_new.pop()
    return length_factor_list_new


def split_pivots_list(pivots_list: List[NDArray[np.uint64]]) -> List[NDArray[np.uint64]]:
    pivots_list_new: List[NDArray[np.uint64]] = []
    for i in pivots_list:
        pivots_list_new.append(i)
        pivots_list_new.append(i)
    return pivots_list_new


def reverse_split_length_factor_list(length_factor_list: List[List[float]]) -> List[List[float]]:
    length_factor_list_new = []
    for sublist in length_factor_list:
        length_factor_list_new.append(sublist[::2])
    return length_factor_list_new


def reverse_split_pivots_list(pivots_list: List[NDArray[np.uint64]]) -> List[NDArray[np.uint64]]:
    return pivots_list[::2]


def harmonic_mean_by_column(matrix: List[List[float]]) -> List[float]:
    num_columns = len(matrix[0])
    harmonic_means = []
    for col in range(num_columns):
        reciprocals = [1 / row[col] for row in matrix]
        harmonic_mean = len(matrix) / sum(reciprocals)
        harmonic_means.append(harmonic_mean)
    return harmonic_means


def allocate_seg(length_factors_list, arr_length):
    if len(length_factors_list) == 1:
        return [0, arr_length]
    factors = np.array([_[-1] for _ in length_factors_list])
    factors = 1 / (factors + 0.2)
    factors /= np.sum(factors)
    factors = (np.cumsum(factors) * arr_length).astype(np.uint64)
    factors = [0] + list(factors)
    factors[-1] = arr_length
    return factors
