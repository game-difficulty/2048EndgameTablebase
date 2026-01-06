import os
import time
from datetime import datetime
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from BoardMover import move_all_dir
import Config
from Config import SingletonConfig
from TrieCompressor import trie_compress_progress
from LzmaCompressor import compress_with_7z, decompress_with_7z
from SignalHub import progress_signal

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]

logger = Config.logger


def handle_restart(i, pathname, steps, d1, d2, started):
    """
    处理断点重连逻辑
    """
    if os.path.exists(pathname + str(i) + '.book'):
        logger.debug(f"skipping step {i}")
        if not SingletonConfig().config.get('optimal_branch_only', False):
            do_compress(pathname + str(i + 2) + '.book')
        return False, None, None
    elif os.path.exists(pathname + str(i) + '.z') or \
            os.path.exists(pathname + str(i) + '.book.7z'):
        logger.debug(f"skipping step {i}")
        return False, None, None
    elif not started:
        started = True
        if i != steps - 3 or d1 is None or d2 is None:
            d1 = np.fromfile(pathname + str(i + 1) + '.book', dtype='uint64,uint32')
            d2 = np.fromfile(pathname + str(i + 2) + '.book', dtype='uint64,uint32')
        return started, d1, d2

    return True, d1, d2


def recalculate_process(
        d1: NDArray[[np.uint64, np.uint32]],
        d2: NDArray[[np.uint64, np.uint32]],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        spawn_rate4: float = 0.1,
        is_variant:bool = False,
) -> None:
    global move_all_dir
    if is_variant:
        from Variants.vBoardMover import move_all_dir
    else:
        from BoardMover import move_all_dir

    started = False
    # 回算搜索时可能需要使用索引进行加速
    ind1: NDArray[np.uint32] | None = None

    if not os.path.exists(pathname + 'stats.txt'):
        with open(pathname + 'stats.txt', 'a', encoding='utf-8') as file:
            file.write(','.join(['layer', 'length', 'max success rate', 'speed', 'deletion_threshold', 'time']) + '\n')

    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        started, d1, d2 = handle_restart(i, pathname, steps, d1, d2, started)
        if not started:
            continue

        progress_signal.progress_updated.emit(steps * 2 - i - 2, steps * 2)

        deletion_threshold = np.uint32(SingletonConfig().config.get('deletion_threshold', 0.0) * 4e9)
        if SingletonConfig().config.get('compress_temp_files', False):
            decompress_with_7z(pathname + str(i) + '.7z')
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
        d0 = recalculate(expanded_arr0, d1, d2, target, position, pattern_check_func, success_check_func,
                         to_find_func, ind1, ind2, i > docheck_step, spawn_rate4)
        length = len(d0)
        t2 = time.time()
        d0 = remove_died(d0).copy()  # 去除0成功率的局面

        t3 = time.time()
        if t3 > t0:
            logger.debug(f'step {i} recalculated: {round(length / (t3 - t0) / 1e6, 2)} mbps')
            logger.debug(f'index/solve/remove: {round((t1 - t0) / (t3 - t0), 2)}/'
                         f'{round((t2 - t1) / (t3 - t0), 2)}/{round((t3 - t2) / (t3 - t0), 2)}')

        if len(d0):
            with open(pathname + 'stats.txt', 'a', encoding='utf-8') as file:
                file.write(','.join([str(i), str(length), str(np.max(d0['f1']) / 4e9),
                                     f'{round(length / max((t3 - t0), 0.01) / 1e6, 2)} mbps',
                                     str(deletion_threshold / 4e9), str(datetime.now())]) + '\n')

        d0.tofile(pathname + str(i) + '.book')
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        logger.debug(f'step {i} written\n')

        if deletion_threshold > 0:
            remove_died(d2, deletion_threshold).tofile(pathname + str(i + 2) + '.book')  # 再写一次，把成功率低于阈值的局面去掉

        if (SingletonConfig().config.get('compress_temp_files', False) and
                SingletonConfig().config.get('optimal_branch_only', False)):
            compress_with_7z(pathname + str(i + 2) + '.book')
        elif (SingletonConfig().config.get('compress', False) and
              not SingletonConfig().config.get('optimal_branch_only', False)):
            do_compress(pathname + str(i + 2) + '.book')  # 如果设置了压缩，则压缩i+2的book，其已经不需要再频繁查找

        if i > 0:
            d1, d2 = d0, d1


@njit(nogil=True, parallel=True, cache=True)
def expand(arr: NDArray[np.uint64]) -> NDArray[[np.uint64, np.uint32]]:
    arr0 = np.empty(len(arr), dtype='uint64,uint32')
    for i in prange(len(arr)):
        arr0[i]['f0'] = arr[i]
    return arr0


@njit(nogil=True, cache=True)
def remove_died(
        arr: NDArray[[np.uint64, np.uint32]], deletion_threshold: np.uint32 = 0
) -> NDArray[[np.uint64, np.uint32]]:
    count = 0
    # 原地移动成功率高于阈值的元素
    for i in range(len(arr)):
        if arr[i]['f1'] > deletion_threshold:
            arr[count] = arr[i]
            count += 1
    return arr[:count]


@njit(parallel=True, nogil=True, cache=True)
def create_index(arr: NDArray[[np.uint64, np.uint32]]
                 ) -> NDArray[np.uint32] | None:
    """
    根据uint64数据的前24位的分段位置创建一个索引，长度16777216+1
    """
    n = 16777217
    ind1: NDArray = np.empty(n, dtype='uint32')
    header = arr[0][0] >> np.uint32(40)
    ind1[:header + 1] = 0

    for i in prange(1, len(arr)):
        header = arr[i][0] >> np.uint32(40)
        header_pre = arr[i - 1][0] >> np.uint32(40)
        if header != header_pre:
            ind1[header_pre + 1 : header + 1] = i

    header = arr[-1][0] >> np.uint32(40)
    ind1[header + 1:] = len(arr)
    return ind1


@njit(nogil=True, cache=True)
def binary_search_arr(arr: NDArray[[np.uint64, np.uint32]],
                      target: np.uint64, low: np.uint32 | None = None, high: np.uint32 | None = None) -> np.uint32:
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = low + np.uint32((high - low) // 2)
        mid_val = arr[mid][0]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return arr[mid][1]  # 找到匹配，返回对应的uint32值

    return np.uint32(0)  # 如果没有找到匹配项


@njit(nogil=True, cache=True)
def search_arr(arr: NDArray[[np.uint64, np.uint32]],
               b: np.uint64, ind: NDArray[np.uint32] | None) -> np.uint32:
    """
    没有索引就直接二分查找，否则先从索引中确定一个更窄的范围再查找
    """
    if ind is None:
        return binary_search_arr(arr, b)
    header = b >> np.uint32(40)
    low, high = ind[header], ind[header + 1] - 1
    return binary_search_arr(arr, b, low, high)


@njit(parallel=True, nogil=True, cache=True)
def recalculate(
        arr0: NDArray[[np.uint64, np.uint32]],
        arr1: NDArray[[np.uint64, np.uint32]],
        arr2: NDArray[[np.uint64, np.uint32]],
        target: int,
        position: int,
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        ind1: NDArray[np.uint32] | None,
        ind2: NDArray[np.uint32] | None,
        do_check: bool = False,
        spawn_rate4: float = 0.1
) -> NDArray[[np.uint64, np.uint32]]:
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
    ind1, ind2是预先计算的索引，用于加速二分查找过程
    """
    chunk_count = max(min(1024, len(arr0) // 1048576), 1)
    chunk_size = len(arr0) // chunk_count
    for chunk in range(chunk_count):
        start, end = chunk_size * chunk, chunk_size * (chunk + 1)
        if chunk == chunk_count - 1:
            end = len(arr0)
        for k in prange(start, end):
            t: np.uint64 = arr0[k][0]
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
                    for newt in move_all_dir(t_gen):
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
                    for newt in move_all_dir(t_gen):
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
    if bookpath[-7:] in ['_0.book', '_1.book', '_2.book']:
        return
    if SingletonConfig().config.get('compress', False) and os.path.exists(bookpath):
        if os.path.getsize(bookpath) > 2097152:
            trie_compress_progress(*os.path.split(bookpath))
            if os.path.exists(bookpath):
                os.remove(bookpath)


def tofile_(data, path):
    data.tofile(path + '.book')
    if os.path.exists(path):
        os.remove(path)


@njit(parallel=True, nogil=True, cache=True)
def find_optimal_branches(
        arr0: NDArray[[np.uint64, np.uint32]],
        arr1: NDArray[[np.uint64, np.uint32]],
        result: NDArray[bool],
        pattern_check_func: PatternCheckFunc,
        to_find_func: ToFindFunc,
        ind1: NDArray[np.uint32] | None,
        new_value: int,
) -> NDArray[bool]:
    for start, end in (
            (0, len(arr0) // 10), (len(arr0) // 10, len(arr0) // 3), (len(arr0) // 3, len(arr0))):  # 缓解负载不均衡问题
        for k in prange(start, end):
            t = arr0[k][0]
            for i in range(16):  # 遍历所有位置
                if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    optimal_success_rate = 0  # 记录有效移动后的面板成功概率中的最大值
                    optimal_pos_index = np.uint64(0)
                    for newt in move_all_dir(t_gen):
                        if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            success_rate, pos_index = search_arr2(arr1, to_find_func(newt), ind1)
                            if success_rate > optimal_success_rate:
                                optimal_success_rate = success_rate
                                optimal_pos_index = np.uint64(pos_index)
                    result[optimal_pos_index] = True
    return result


@njit(nogil=True, cache=True)
def binary_search_arr2(arr: NDArray[[np.uint64, np.uint32]],
                       target: np.uint64, low: np.uint32 | None = None, high: np.uint32 | None = None
                       ) -> Tuple[np.uint32, np.uint64]:
    """相比binary_search_arr，还会返回索引"""
    if low is None:
        low = 0
        high = len(arr) - 1

    while low <= high:
        mid = low + np.uint32((high - low) // 2)
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


def keep_only_optimal_branches(
    pattern_check_func: PatternCheckFunc,
    to_find_func: ToFindFunc,
    steps: int,
    pathname: str,
    ):
    d0, d1 = None, None
    started = False
    for i in range(0, steps):
        started, d0, d1 = handle_restart_opt_only(i, started, d0, d1, pathname)
        if SingletonConfig().config.get('compress_temp_files', False):
            decompress_with_7z(pathname + str(i) + '.book.7z')
        if i > 20 and started:
            d2 = np.fromfile(pathname + str(i) + '.book', dtype='uint64,uint32')
            ind = create_index(d2)
            is_in_optimal_branches_mask = np.zeros(len(d2), dtype='bool')
            is_in_optimal_branches_mask = find_optimal_branches(d0, d2, is_in_optimal_branches_mask,
                                                                pattern_check_func, to_find_func, ind, 2)
            is_in_optimal_branches_mask = find_optimal_branches(d1, d2, is_in_optimal_branches_mask,
                                                                pattern_check_func, to_find_func, ind, 1)
            d2 = d2[is_in_optimal_branches_mask].copy()
            d2.tofile(pathname + str(i) + '.book')
            d0, d1 = d1, d2
            del is_in_optimal_branches_mask, d2
            logger.debug(f'step {i} retains only the optimal branch\n')
        do_compress(pathname + str(i - 2) + '.book')

    do_compress(pathname + str(steps - 2) + '.book')
    do_compress(pathname + str(steps - 1) + '.book')
    if os.path.exists(pathname + 'optlayer'):
        os.remove(pathname + 'optlayer')


def handle_restart_opt_only(i, started, d0, d1, pathname):
    if started and d0 is not None:
        with open(pathname + 'optlayer', 'w') as f:
            f.write(str(i - 1))
        return True, d0, d1
    try:
        with open(pathname + 'optlayer', 'r') as f:
            current_layer = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        current_layer = i - 1
    if current_layer >= i:
        return False, None, None
    else:
        if (i > 20) & (d0 is None):
            try:
                d0 = np.fromfile(pathname + str(i - 2) + '.book', dtype='uint64,uint32')
                d1 = np.fromfile(pathname + str(i - 1) + '.book', dtype='uint64,uint32')
                return True, d0, d1
            except FileNotFoundError:
                pass
        return False, d0, d1


# if __name__ == "__main__":
#     from Calculator import is_free_pattern, min_all_symm
#     from BoardMover import BoardMover
#
#     keep_only_optimal_branches(is_free_pattern, min_all_symm, 536,
#                                r"D:\2048calculates\table\free10w_1024 - 副本\free10w_1024_")
