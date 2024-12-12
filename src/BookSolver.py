import os
import time
from typing import Callable, Tuple

import numpy as np
from numba import njit, prange

from BoardMover import BoardMover
import Config
from Config import SingletonConfig
from TrieCompressor import trie_compress_progress


PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.ndarray[np.uint64]], None]
ToFindFunc1 = Callable[[np.uint64], np.uint64]

logger = Config.logger


def recalculate_process(
        d1: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
        d2: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc1,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        bm: BoardMover,
        spawn_rate4: float = 0.1
) -> None:
    started = False
    # 回算搜索时可能需要使用索引进行加速
    ind1: np.ndarray[np.uint32] | None = None

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        # 断点重连
        if os.path.exists(pathname + str(i) + '.book'):
            logger.debug(f"skipping step {i}")
            do_compress(pathname + str(i + 2) + '.book')
            continue
        elif os.path.exists(pathname + str(i) + '.z'):
            logger.debug(f"skipping step {i}")
            continue
        elif not started:
            started = True
            if i != steps - 3 or d1 is None or d2 is None:
                d1 = np.fromfile(pathname + str(i + 1) + '.book', dtype='uint64,uint32')
                d2 = np.fromfile(pathname + str(i + 2) + '.book', dtype='uint64,uint32')

        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        t0 = time.time()

        expanded_arr0 = expand(d0)
        del d0

        # 创建、更新查找索引
        if ind1 is not None:
            ind2 = ind1
        elif len(d2) < 100000:
            ind2 = None
        else:
            ind2 = create_index(d2)
        if len(d1) < 100000:
            ind1 = None
        else:
            ind1 = create_index(d1)
        t1 = time.time()

        # 回算
        d0 = recalculate(expanded_arr0, d1, d2, target, position, bm, pattern_check_func, success_check_func,
                         to_find_func, ind1, ind2, i > docheck_step, spawn_rate4)
        length = len(d0)
        t2 = time.time()
        d0 = remove_died(d0)  # 去除活不了的局面

        t3 = time.time()
        if t3 > t0:
            logger.debug(f'step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps')
            logger.debug(f'index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/'
                         f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}')

        d0.tofile(pathname + str(i) + '.book')
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        logger.debug(f'step {i} written\n')

        do_compress(pathname + str(i + 2) + '.book')  # 如果设置了压缩，则压缩i+2的book，其已经不需要再频繁查找
        if i > 0:
            d1, d2 = d0, d1


@njit(nogil=True, parallel=True)
def expand(arr: np.ndarray[np.uint64]) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    arr0 = np.empty(len(arr), dtype='uint64,uint32')
    for i in prange(len(arr)):
        arr0[i]['f0'] = arr[i]
    return arr0


@njit(nogil=True)
def remove_died(
        arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]
) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    count = 0
    # 原地移动不为0的元素
    for i in range(len(arr)):
        if arr[i]['f1'] != 0:
            arr[count] = arr[i]
            count += 1
    return arr[:count]


@njit(parallel=True, nogil=True)
def create_index(arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]
                 ) -> np.ndarray[np.uint32] | None:
    """
    根据uint64数据的前24位的分段位置创建一个索引，长度16777216+1
    """
    n = 16777217
    ind1: np.ndarray = np.full(n, 0xffffffff, dtype='uint32')
    header = arr[0][0] >> np.uint32(40)
    ind1[header] = 0

    for i in prange(1, len(arr)):
        header = arr[i][0] >> np.uint32(40)
        header_pre = arr[i - 1][0] >> np.uint32(40)
        if header != header_pre:
            ind1[header] = i
    # 向前填充
    num_threads = 8
    segment_size = (n + num_threads - 1) // num_threads

    # 记录每个段最后一个非零值
    last_values = np.empty(num_threads, dtype='uint32')

    # 每段并行处理
    for i in prange(num_threads):
        start = i * segment_size
        end = min(start + segment_size, n)

        # 初始化 last_value 为段末尾的第一个非零值
        last_value = len(arr)
        for j in range(end - 1, start - 1, -1):
            if ind1[j] != 0xffffffff:
                last_value = ind1[j]
            else:
                ind1[j] = last_value

        last_values[i] = last_value

    # 处理边界，确保每段的填充是正确的
    for i in prange(1, num_threads):
        if last_values[i] != len(arr):
            start = i * segment_size
            for j in range(start - 1, start - segment_size, -1):
                if ind1[j] == len(arr):
                    ind1[j] = last_values[i]
                else:
                    break

    ind1[0] = 0
    return ind1


@njit(nogil=True)
def binary_search_arr(arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
                      target: np.uint64, low: np.uint32 | None = None, high: np.uint32 | None = None) -> np.uint32:
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = np.uint32((low + high) // 2)
        mid_val = arr[mid][0]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return arr[mid][1]  # 找到匹配，返回对应的uint32值

    return 0  # 如果没有找到匹配项


@njit(nogil=True)
def search_arr(arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
               b: np.uint64, ind: np.ndarray[np.uint32] | None) -> np.uint32:
    """
    没有索引就直接二分查找，否则先从索引中确定一个更窄的范围再查找
    """
    if ind is None:
        return binary_search_arr(arr, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr(arr, b, low, high)


@njit(parallel=True, nogil=True)
def recalculate(
        arr0: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
        arr1: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
        arr2: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
        target: int,
        position: int,
        bm: BoardMover,
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc1,
        ind1: np.ndarray[np.uint32] | None,
        ind2: np.ndarray[np.uint32] | None,
        do_check: bool = False,
        spawn_rate4: float = 0.1
) -> np.ndarray[Tuple[np.uint64, np.uint32]]:
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
    ind1, ind2是预先计算的索引，用于加速二分查找过程
    """
    for start, end in (
            (0, len(arr0) // 10), (len(arr0) // 10, len(arr0) // 3), (len(arr0) // 3, len(arr0))):  # 缓解负载不均衡问题
        for k in prange(start, end):
            t = arr0[k][0]
            if do_check and success_check_func(t, target, position):
                arr0[k][1] = 4000000000
                continue
            # 初始化概率和权重
            success_probability = 0.0
            empty_slots = 0
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                    empty_slots += 1

                    # 对于每个空位置，尝试填充2和4
                    new_value, probability = 1, 1 - spawn_rate4
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = 0  # 记录有效移动后的面板成功概率中的最大值
                    for newt in bm.move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            optimal_success_rate = max(optimal_success_rate, search_arr(
                                arr1, to_find_func(newt), ind1))
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * probability

                    # 填4
                    new_value, probability = 2, spawn_rate4
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = 0  # 记录有效移动后的面板成功概率中的最大值
                    for newt in bm.move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            optimal_success_rate = max(optimal_success_rate, search_arr(
                                arr2, to_find_func(newt), ind2))
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * probability

            # t是进行一次有效移动后尚未生成新数字时的面板，因此不可能没有空位置
            arr0[k][1] = np.uint32(success_probability / empty_slots)
    return arr0


def do_compress(bookpath: str) -> None:
    if bookpath[-4:] != 'book':
        return
    if SingletonConfig().config.get('compress', False) and os.path.exists(bookpath):
        if os.path.getsize(bookpath) > 4194304:
            trie_compress_progress(*os.path.split(bookpath))
            if os.path.exists(bookpath):
                os.remove(bookpath)


def tofile_(data, path):
    data.tofile(path + '.book')
    if os.path.exists(path):
        os.remove(path)