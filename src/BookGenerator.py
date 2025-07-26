import os
import time
from typing import Callable, Tuple, List

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
import psutil


from BookGeneratorUtils import sort_array, parallel_unique, concatenate, merge_deduplicate_all, largest_power_of_2, \
    hash_, merge_inplace
import Config
from BoardMover import move_all_dir
from Config import SingletonConfig

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]


logger = Config.logger


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
        length_factors = np.loadtxt(length_factors_list_path, delimiter=',', dtype=np.float64).tolist()
        length_factor = predict_next_length_factor_quadratic(length_factors) * 1.2
    except FileNotFoundError:
        length_factor = default_length_factor
        length_factors = [length_factor, length_factor, length_factor]

    # 决定是否使用gen_boards_big处理
    segment_size = 5120000 if isfree else 8192000
    segment_size *= (round(psutil.virtual_memory().total / (1024 ** 3), 0) - 5)

    return length_factor, length_factors, length_factors_list_path, \
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


@njit(nogil=True, parallel=True, cache=True)
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
        isfree: bool,
) -> Tuple[bool, NDArray[np.uint64], NDArray[np.uint64]]:

    started = False  # 是否进行了计算，如果是则需要进行final_steps处理最后一批局面
    d0, d1 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    pivots = None  # 用于快排的分割点
    hashmap1, hashmap2 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    n = max(4, min(32, os.cpu_count()))  # 并行线程数
    length_factor, length_factors, length_factors_list_path, \
        counts2, length_factor_multiplier, segment_size = initialize_parameters(n, pathname, isfree)

    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        started, d0, d1 = handle_restart(i, pathname, arr_init, started, d0, d1)
        if not started:
            continue

        if pivots is None:
            pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]] if len(d0) > 0 else np.zeros(7, dtype='uint64')

        # 生成新的棋盘状态
        t0 = time.time()

        # 先预测预分配数组的长度乘数
        length_factor = predict_next_length_factor_quadratic(length_factors)
        length_factor *= 1.25 if len(d0) > 1e8 else 1.5
        length_factor *= length_factor_multiplier
        if len(hashmap1) == 0:
            hashmap1, hashmap2 = update_hashmap_length(hashmap1, d0), update_hashmap_length(hashmap2, d0)  # 初始化

        d1t, d2, hashmap1, hashmap2, counts1, counts2 = \
            gen_boards(d0, target, position, pattern_check_func, success_check_func, to_find_func,
                       hashmap1, hashmap2, n, length_factor, i > docheck_step, isfree)

        validate_length_and_balance(d0, d2, d1t, counts1, counts2, length_factor, False)

        t1 = time.time()
        # 排序
        sort_array(d1t, pivots)
        sort_array(d2, pivots)

        length_factors, pivots = update_parameters(d0, d2, length_factors, length_factors_list_path)
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

        d0.tofile(pathname + str(i))

        hashmap1, hashmap2 = hashmap2, hashmap1

    if d1 is not None:
        d1.tofile(pathname + str(i + 1))
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
            f"counts1 {np.round(counts1 / 1e6, 2)}, \n"
            f"counts2 {np.round(counts2 / 1e6, 2)}")
        raise IndexError("The length multiplier is not big enough. "
                         "This does not indicate an error in the program. "
                         "Please restart the program and continue running.")
    if not isbig:
        logger.debug(f'length {len(d1t)}, {len(d2)}, '
                     f'Using {round(length_factor, 2)}, '
                     f'Need {round(length_factor_actual, 2)}')
        logger.debug('Segmentation1_ac ' + repr(np.round(percents2, 5)))
        logger.debug('Segmentation2_ac ' + repr(np.round(percents1, 5)))
    elif len(d0) > 0:
        logger.debug(
            f'length {len(d1t)}, {len(d2)}, '
            f'Using {round(length_factor, 2)}, '
            f'Need {round(length_factor_actual, 2)}')


def update_parameters(d0, d2, length_factors, length_factors_list_path):
    # 更新数组长度乘数
    length_factors = length_factors[1:] + [len(d2) / (len(d0) + 1)]

    np.savetxt(length_factors_list_path, length_factors, fmt='%.6f', delimiter=',')  # type: ignore
    # 选取更准确的分割点使下一次快排更有效
    pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]].copy() if len(d2) > 0 else np.zeros(7, dtype='uint64')

    return length_factors, pivots


def log_performance(i, t0, t1, t2, t3, d1):
    if t3 > t0:
        logger.debug(f'step {i} generated: {round(len(d1) / (t3 - t0) / 1e6, 2)} mbps')
        logger.debug(f'generate/sort/deduplicate: {round((t1 - t0) / (t3 - t0), 2)}/'
                     f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}\n')


@njit(nogil=True, parallel=True, cache=True)
def gen_boards(arr0: NDArray[np.uint64],
               target: int,
               position: int,
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
                        for newt in move_all_dir(t1):
                            if newt != t1 and pattern_check_func(newt):
                                newt = to_find_func(newt)
                                hashed_newt = (hash_(newt)) & hashmap1_length
                                if hashmap1[hashed_newt] != newt:
                                    hashmap1[hashed_newt] = newt
                                    arr1[c1t] = newt
                                    c1t += 1

                        t1 = t | (np.uint64(2) << np.uint64(4 * i))  # 填充数字4
                        for newt in move_all_dir(t1):
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

