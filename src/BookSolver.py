import os
import time
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from BoardMover import move_all_dir
import Config
from Config import SingletonConfig

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]

logger = Config.logger


def handle_restart(i, pathname, steps, d1, d2, rate1, rate2, started):
    """
    处理断点重连逻辑
    """
    if os.path.exists(pathname + str(i) + '.book'):
        logger.debug(f"skipping step {i}")

        return False, None, None, None, None

    elif not started:
        started = True
        if i != steps - 3 or d1 is None or d2 is None:
            d1 = np.fromfile(pathname + str(i + 1), dtype='uint64')
            d2 = np.fromfile(pathname + str(i + 2), dtype='uint64')
            rate1 = np.fromfile(pathname + str(i + 1) + '.book', dtype='uint32')
            rate2 = np.fromfile(pathname + str(i + 2) + '.book', dtype='uint32')
            points = SingletonConfig().config['data_points']
            rate1 = rate1.reshape((-1, points))
            rate2 = rate2.reshape((-1, points))
        return started, d1, d2, rate1, rate2

    return True, d1, d2, rate1, rate2


def recalculate_process(
        d1: NDArray,
        rate1: NDArray,
        d2: NDArray,
        rate2: NDArray,
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
) -> None:

    started = False
    # 回算搜索时可能需要使用索引进行加速
    ind1: NDArray[np.uint32] | None = None
    points = SingletonConfig().config['data_points']

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, d1, d2, rate1, rate2 = handle_restart(i, pathname, steps, d1, d2, rate1, rate2, started)
        if not started:
            continue

        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)

        t0 = time.time()

        rate0 = expand(d0, points)

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
        recalculate(d0, rate0, d1, rate1, d2, rate2, target, position, pattern_check_func, success_check_func,
                    to_find_func, ind1, ind2, i > docheck_step)
        length = len(d0)
        t2 = time.time()
        rate0, d0 = remove_died(rate0, d0)

        t3 = time.time()
        if t3 > t0:
            logger.debug(f'step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps')
            logger.debug(f'index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/'
                         f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}')

        rate0.tofile(pathname + str(i) + '.book')
        d0.tofile(pathname + str(i))

        logger.debug(f'step {i} written\n')

        if i > 0:
            d1, d2 = d0, d1
            rate1, rate2 = rate0, rate1


def expand(arr: NDArray[np.uint64], points):
    rates0 = np.empty((len(arr), points), dtype='uint32')
    return rates0


@njit(nogil=True, cache=True)
def remove_died(rate: NDArray, d0: NDArray):
    count = 0
    # 原地移动成功率高于阈值的元素
    for i in range(len(rate)):
        if np.any(rate[i]):
            rate[count] = rate[i]
            d0[count] = d0[i]
            count += 1
    return rate[:count], d0[:count]


@njit(parallel=True, nogil=True, cache=True)
def create_index(arr: NDArray[np.uint64]
                 ) -> NDArray[np.uint32] | None:
    """
    根据uint64数据的前24位的分段位置创建一个索引，长度16777216+1
    """
    n = 16777217
    ind1: NDArray = np.empty(n, dtype='uint32')
    header = arr[0] >> np.uint32(40)
    ind1[:header + 1] = 0

    for i in prange(1, len(arr)):
        header = arr[i] >> np.uint32(40)
        header_pre = arr[i - 1] >> np.uint32(40)
        if header != header_pre:
            ind1[header_pre + 1 : header + 1] = i

    header = arr[-1] >> np.uint32(40)
    ind1[header + 1:] = len(arr)
    return ind1


@njit(nogil=True, cache=True)
def binary_search_arr(arr: NDArray[np.uint64], rate: NDArray, optimal_success_rate,
                      target: np.uint64, low: np.uint32 | None = None, high: np.uint32 | None = None) -> np.uint32:
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = low + np.uint32((high - low) // 2)
        mid_val = arr[mid]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            arr_max(optimal_success_rate, rate[mid])  # 找到匹配
            return


@njit(nogil=True, inline='always')
def arr_max(arr1: NDArray, arr2: NDArray):
    # assert len(arr1) - len(arr2) == 0
    for i in range(len(arr1)):
        arr1[i] = max(arr1[i], arr2[i])


@njit(nogil=True, cache=True)
def search_arr(arr: NDArray,
               b: np.uint64, ind: NDArray[np.uint32] | None, rate: NDArray, optimal_success_rate) -> np.uint32:
    """
    没有索引就直接二分查找，否则先从索引中确定一个更窄的范围再查找
    """
    if ind is None:
        return binary_search_arr(arr, rate, optimal_success_rate, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr(arr, rate, optimal_success_rate, b, low, high)


@njit(parallel=True, nogil=True, cache=True)
def recalculate(
        d0, rate0, d1, rate1, d2, rate2,
        target: int,
        position: int,
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        ind1: NDArray[np.uint32] | None,
        ind2: NDArray[np.uint32] | None,
        do_check: bool = False,
) -> NDArray[[np.uint64, np.uint32]]:
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
    ind1, ind2是预先计算的索引，用于加速二分查找过程
    """
    points = len(rate0[0])
    sr4 = np.arange(0, 1, 1 / points, dtype=np.float32)
    sr2 = 1 - sr4

    chunk_count = max(min(1024, len(d0) // 1048576), 1)
    chunk_size = len(d0) // chunk_count
    for chunk in range(chunk_count):
        start, end = chunk_size * chunk, chunk_size * (chunk + 1)
        if chunk == chunk_count - 1:
            end = len(d0)
        for k in prange(start, end):
            t: np.uint64 = d0[k]
            if do_check and success_check_func(t, target, position):
                rate0[k] = 4000000000
                continue
            # 初始化概率和权重
            success_probability = np.zeros(points, dtype=np.float32)
            empty_slots = 0
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                    empty_slots += 1

                    # 对于每个空位置，尝试填充2和4
                    new_value = 1
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = np.zeros(points, dtype=np.float32)  # 记录有效移动后的面板成功概率中的最大值
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            search_arr(d1, to_find_func(newt), ind1, rate1, optimal_success_rate)
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * sr2

                    # 填4
                    new_value = 2
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = np.zeros(points, dtype=np.float32)  # 记录有效移动后的面板成功概率中的最大值
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            search_arr(d2, to_find_func(newt), ind2, rate2, optimal_success_rate)
                    # 对最佳移动下的成功概率加权平均
                    success_probability += optimal_success_rate * sr4

            # t是进行一次有效移动后尚未生成新数字时的面板，因此不可能没有空位置
            rate0[k] = (success_probability / empty_slots).astype(np.uint32)


def tofile_(data, path):
    data.tofile(path + '.book')
    if os.path.exists(path):
        os.remove(path)


@njit(nogil=True, cache=True)
def binary_search_arr2(arr: NDArray[[np.uint64, np.uint32]],
                       target: np.uint64, low: np.uint32 | None = None, high: np.uint32 | None = None
                       ) -> Tuple[np.uint32, np.uint64]:
    """相比binary_search_arr，还会返回索引"""
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
            return arr[mid][1], mid  # 找到匹配，返回对应的uint32值和索引位置

    return np.uint32(0), np.uint64(0)  # 如果没有找到匹配项


@njit(nogil=True, cache=True)
def search_arr2(arr: NDArray[[np.uint64, np.uint32]],
                b: np.uint64, ind: NDArray[np.uint32] | None) -> Tuple[np.uint32, np.uint64]:
    """
    相比search_arr，还会返回索引
    """
    if ind is None:
        return binary_search_arr2(arr, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr2(arr, b, low, high)



# if __name__ == "__main__":
#     from Calculator import is_free_pattern, min_all_symm
#     from BoardMover import BoardMover
#
#     keep_only_optimal_branches(is_free_pattern, min_all_symm, 536,
#                                r"D:\2048calculates\table\free10w_1024 - 副本\free10w_1024_")
