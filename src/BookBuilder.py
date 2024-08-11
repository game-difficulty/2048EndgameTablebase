import os
from itertools import combinations, permutations
import ctypes
from ctypes import c_uint64, c_size_t, POINTER
import gc
from typing import Callable, Tuple, List
# import time

import cpuinfo
import numpy as np
from numba import njit, prange

import Calculator
from BoardMover import BoardMover
from Config import SingletonConfig
from TrieCompressor import trie_compress_progress


PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.ndarray[np.uint64]], None]
ToFindFunc1 = Callable[[np.uint64], np.uint64]


# 加载DLL
dll = ctypes.CDLL(r"_internal/para_qsort.dll")

# 声明parallel_sort函数原型
dll.parallel_sort.argtypes = (POINTER(c_uint64), c_size_t, POINTER(c_uint64), ctypes.c_bool)
dll.parallel_sort.restype = None

# 获取当前cpu指令集信息
info = cpuinfo.get_cpu_info()
if 'avx512f' in info['flags'] and 'avx512vl' in info['flags'] and \
        ('avx512dq' in info['flags'] or ('avx512bw' in info['flags'] and 'avx512vbmi2' in info['flags'])):
    use_avx = 2
    # print("CPU info: AVX512 is supported")
elif 'avx2' in info['flags']:
    use_avx = 1
    # print("CPU info: AVX2 is supported; AVX512 is not supported")
else:
    use_avx = 0
    # print("CPU info: AVX512/AVX2 is not supported")


@njit(nogil=True)
def unique(aux):
    c = 1
    for i in range(1, len(aux)):
        if aux[i] != aux[i - 1]:
            aux[c] = aux[i]
            c += 1
    return aux[:c]


@njit(nogil=True, parallel=True)
def p_unique(arrs):
    for i in prange(2):
        arrs[i] = unique(arrs[i])
    return arrs


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
                   to_find_func: ToFindFunc,
                   d1: np.ndarray[np.uint64],
                   pivots_list: List[np.ndarray[np.uint64]] | None = None,
                   do_check: bool = True,
                   isfree: bool = False,
                   ) -> Tuple[np.ndarray[np.uint64], np.ndarray[np.uint64], List[np.ndarray[np.uint64]]]:
    """将arr0分段放入gen_boards生成排序去重后的局面，然后归并"""
    segment_size = 119999999
    start_index, seg_index = 0, 0
    arr1s: List[np.ndarray[np.uint64]] = [d1]
    arr2s: List[np.ndarray[np.uint64]] = []
    pivots_list_new: List[np.ndarray[np.uint64]] = []
    # t0 = time.time()
    while start_index < len(arr0):
        # tt0 = time.time()
        seg_length = {0: 39999999, 1: 49999999, 2: 69999999}.get(seg_index, segment_size)
        end_index = min(start_index + seg_length, len(arr0))
        arr0t = arr0[start_index:end_index].copy()
        arr1t, arr2t = gen_boards(arr0t, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                                  do_check, isfree)
        del arr0t
        # print(end_index - start_index, len(arr1t))
        # tt1 = time.time()
        pivots = pivots_list[seg_index] if pivots_list is not None and seg_index < len(pivots_list) else None
        sort_array(arr1t, pivots)
        sort_array(arr2t, pivots)
        pivots_list_new.append(arr2t[[len(arr2t) // 8 * i for i in range(1, 8)]])
        arr1t = unique(arr1t)
        arr2t = unique(arr2t)
        arr1s.append(arr1t)
        arr2s.append(arr2t)
        # print(round(time.time() - tt1, 3), round(tt1 - tt0, 3), len(arr1t), flush=True)
        start_index = end_index
        seg_index += 1

    # t1 = time.time()
    length = int(max(len(arr0) * 1.25, 319999999)) if isfree else int(max(len(arr0) * 1.2, 119999999))
    gc.collect()
    arr1 = merge_deduplicate_all(arr1s, length)
    del arr1s, arr1t
    # t2 = time.time()
    arr2 = merge_deduplicate_all(arr2s, length)
    del arr2s, arr2t
    # print(round(time.time() - t2, 3), round(t2 - t1, 3), round(t1 - t0, 3), flush=True)
    return arr1, arr2, pivots_list_new


@njit(nogil=True)
def merge_deduplicate_all(arrays: List[np.ndarray], length: int = 0) -> np.ndarray:
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
        do_check: bool = False,
        spawn_rate4: float = 0.1
) -> np.ndarray[Tuple[np.uint64, np.uint32]]:
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
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
                    for new_value, probability in [(1, 1 - spawn_rate4), (2, spawn_rate4)]:  # 填充2的概率为90%，填充4的概率为10%
                        subs_arr = arr1 if new_value == 1 else arr2
                        t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                        optimal_success_rate = 0  # 记录有效移动后的面板成功概率中的最大值
                        for newt in bm.move_all_dir(t_gen):
                            if newt != t_gen and pattern_check_func(newt):  # 只考虑有效的移动
                                # 获取移动后的面板成功概率
                                optimal_success_rate = max(optimal_success_rate, binary_search_arr(
                                    subs_arr, to_find_func(newt)))
                        # 对最佳移动下的成功概率加权平均
                        success_probability += optimal_success_rate * probability
            # t是进行一次有效移动后尚未生成新数字时的面板，因此不可能没有空位置
            arr0[k][1] = np.uint32(success_probability / empty_slots)
    return arr0


@njit(nogil=True)
def binary_search_arr(arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
                      target: np.uint64) -> np.uint32:
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = arr[mid][0]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return arr[mid][1]  # 找到匹配，返回对应的uint32值
    return 0  # 如果没有找到匹配项


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


def gen_lookup_table_big(
        arr_init: np.ndarray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        to_find_func1: ToFindFunc1,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        isfree: bool = False,
        spawn_rate4: float = 0.1
) -> None:
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """
    bm = BoardMover()
    started, d0, d1 = generate_process(arr_init, pattern_check_func, success_check_func, to_find_func, target, position,
                                       steps, pathname, docheck_step, bm, isfree)
    d0, d1 = final_steps(started, d0, d1, pathname, steps, success_check_func, target, position)
    recalculate_process(d0, d1, pattern_check_func, success_check_func, to_find_func1, target, position, steps,
                        pathname, docheck_step, bm, spawn_rate4)  # 这里的最后的两个book d0,d1就是回算的d1,d2


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
    started = False
    d0, d1 = None, None
    pivots, pivots_list = None, None
    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        # 断点重连
        if os.path.exists(pathname + str(i)) or os.path.exists(pathname + str(i) + '.book') or \
                os.path.exists(pathname + str(i) + '.z'):
            print(f"skipping step {i}", flush=True)
            continue
        if i == 1:
            arr_init.tofile(pathname + str(i - 1))
            d0, d1 = arr_init, np.empty(0, dtype=np.uint64)
            started = True
        elif not started:
            d0 = np.fromfile(pathname + str(i - 1), dtype=np.uint64)
            d1 = np.fromfile(pathname + str(i) + '.t', dtype=np.uint64)
            started = True

        segment_size = 67108864 if isfree else 104857600
        # 生成新的棋盘状态
        if len(d0) < segment_size:
            d1t, d2 = gen_boards(d0, target, position, bm, pattern_check_func, success_check_func, to_find_func,
                                 i > docheck_step, isfree)
            if pivots is None:
                pivots = d0[[len(d0) // 8 * i for i in range(1, 8)]]
            # 执行顺序：排序、获取分割点、去重
            sort_array(d1t, pivots)
            sort_array(d2, pivots)
            pivots = d2[[len(d2) // 8 * i for i in range(1, 8)]]
            d1t = unique(d1t)
            d2 = unique(d2)
            d1 = merge_and_deduplicate(d1, d1t)

            d0, d1 = d1, d2
            del d1t, d2
        else:
            d0, d1, pivots_list = gen_boards_big(d0, target, position, bm, pattern_check_func, success_check_func,
                                                 to_find_func, d1, pivots_list, i > docheck_step, isfree)
        # print(f"Processing step {i}",flush=True)
        d1.tofile(pathname + str(i + 1) + '.t')
        d0.tofile(pathname + str(i))
        if os.path.exists(pathname + str(i) + '.t'):
            os.remove(pathname + str(i) + '.t')
        gc.collect()
    return started, d0, d1


def final_steps(started: bool,
                d0: np.ndarray[np.uint64],
                d1: np.ndarray[np.uint64],
                pathname: str,
                steps: int,
                success_check_func: SuccessCheckFunc,
                target: int,
                position: int):
    if started:
        expanded_arr0 = np.empty(len(d0), dtype='uint64,uint32')
        expanded_arr0['f0'] = d0
        d0 = final_situation_process(expanded_arr0, success_check_func, target, position)
        d0.tofile(pathname + str(steps - 2) + '.book')
        expanded_arr0 = np.empty(len(d1), dtype='uint64,uint32')
        expanded_arr0['f0'] = d1
        d1 = final_situation_process(expanded_arr0, success_check_func, target, position)
        d1.tofile(pathname + str(steps - 1) + '.book')
    if os.path.exists(pathname + str(steps - 1) + '.t'):
        os.remove(pathname + str(steps - 1) + '.t')
    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


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
    # 从后向前更新ds中的array
    for i in range(steps - 3, -1, -1):
        # 断点重连
        if os.path.exists(pathname + str(i) + '.book'):
            print(f"skipping step {i}")
            do_compress(pathname + str(i + 2) + '.book')
            continue
        elif os.path.exists(pathname + str(i) + '.z'):
            print(f"skipping step {i}")
            continue
        elif not started:
            started = True
            if i != steps - 3:
                d1 = np.fromfile(pathname + str(i + 1) + '.book', dtype='uint64,uint32')
                d2 = np.fromfile(pathname + str(i + 2) + '.book', dtype='uint64,uint32')

        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)
        expanded_arr0 = np.empty(len(d0), dtype='uint64,uint32')
        expanded_arr0['f0'] = d0
        del d0
        # t0 = time.time()
        d0 = recalculate(expanded_arr0, d1, d2, target, position, bm, pattern_check_func, success_check_func,
                         to_find_func, i > docheck_step, spawn_rate4)
        # t1 = time.time()
        d0 = remove_died(d0)  # 去除活不了的局面
        # print(f"Updated step {i}", round(time.time() - t1, 3), round(t1 - t0, 3))
        d0.tofile(pathname + str(i) + '.book')
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        do_compress(pathname + str(i + 2) + '.book')  # 如果设置了压缩，则压缩i+2的book，其已经不需要再频繁查找
        if i > 0:
            d1, d2 = d0, d1


@njit(nogil=True, parallel=True)
def remove_died(d: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]
                ) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    d = d[d['f1'] != np.uint32(0)]
    return d


@njit(nogil=True, parallel=True)
def final_situation_process(expanded_arr0: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
                            success_check_func: SuccessCheckFunc, target: int, position: int
                            ) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    for i in prange(len(expanded_arr0)):
        if success_check_func(expanded_arr0[i][0], target, position):
            expanded_arr0[i][1] = 4000000000
        else:
            expanded_arr0[i][1] = 0
    expanded_arr0 = expanded_arr0[expanded_arr0['f1'] != 0]
    return expanded_arr0


def do_compress(bookpath: str) -> None:
    if bookpath[-4:] != 'book':
        return
    if SingletonConfig().config.get('compress', False) and os.path.exists(bookpath):
        if os.path.getsize(bookpath) > 4194304:
            trie_compress_progress(*os.path.split(bookpath))
            if os.path.exists(bookpath):
                os.remove(bookpath)


def generate_free_inits(target: int, t32ks: int, t2s: int) -> np.ndarray[np.uint64]:
    numbers = [target, ]
    generated = np.empty(86486400, dtype=np.uint64)
    c = 0
    for poss in combinations(range(16), t32ks):
        v = np.uint64(0)
        for p in poss:
            v |= np.uint64(15 << (p * 4))  # 填32k
        remain_pos = set(range(16)) - set(poss)
        for poss_2 in combinations(remain_pos, t2s):
            val = v
            for j in poss_2:
                val |= np.uint64(1 << (j * 4))  # 填2
            remain_pos2 = set(remain_pos) - set(poss_2)
            for poss_3 in permutations(remain_pos2, len(numbers)):
                poss_3: list
                value = val
                for k in range(len(poss_3)):
                    value |= np.uint64(numbers[k]) << np.uint64(poss_3[k] * 4)  # 全排列填numbers中数字
                generated[c] = value
                c += 1

    generated = np.unique(generated[:c])
    g2 = np.empty(10810800, dtype=np.uint64)
    c = 0
    bm = BoardMover()
    for b in generated:
        for nb in bm.move_all_dir(b):
            nb = np.uint64(nb)
            if nb == np.uint64(Calculator.min_all_symm(nb)):
                g2[c] = nb
                c += 1
    return np.unique(g2[:c])


def start_build(pattern: str, target: int, position: int, pathname: str) -> bool:
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    if pattern[:4] == 'free':
        if pattern[-1] != 'w':
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 5
            free_tiles = int(pattern[4:])
            arr_init = generate_free_inits(target, 15 - free_tiles, free_tiles)
            gen_lookup_table_big(arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.p_min_all_symm, Calculator.min_all_symm, target, 0, steps, pathname,
                                 docheck_step, isfree=True, spawn_rate4=spawn_rate4)
        else:
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 5
            free_tiles = int(pattern[4:-1])
            arr_init = generate_free_inits(0, 16 - free_tiles, free_tiles - 1)
            gen_lookup_table_big(arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.p_min_all_symm, Calculator.min_all_symm, target, 1, steps, pathname,
                                 docheck_step, isfree=True, spawn_rate4=spawn_rate4)
    else:
        steps = int(2 ** target / 2 + {'444': 96, '4431': 64, 'LL': 48, 'L3': 36, '4441': 48, '4432': 48, '442': 36,
                                       't': 36, }[pattern])
        docheck_step = int(2 ** target / 2) - 10
        inits = {
            '444': np.array([np.uint64(0x100000000000ffff), np.uint64(0x000000010000ffff)], dtype=np.uint64),
            '4431': np.array([np.uint64(0x10000000123f2fff), np.uint64(0x00000001123f2fff)], dtype=np.uint64),
            'LL': np.array([np.uint64(0x1000000023ff24ff), np.uint64(0x0000000123ff24ff)], dtype=np.uint64),
            'L3': np.array([np.uint64(0xfff2fff110000000), np.uint64(0xfff2fff100000001)], dtype=np.uint64),
            '4441': np.array([np.uint64(0x0000100012323fff), np.uint64(0x0001000012323fff)], dtype=np.uint64),
            '4432': np.array([np.uint64(0x00001000123f23ff), np.uint64(0x00010000123f23ff)], dtype=np.uint64),
            '442': np.array([np.uint64(0xffffff2110000000), np.uint64(0xffffff2100000001)], dtype=np.uint64),
            't': np.array([np.uint64(0xff2fff1f10000000), np.uint64(0xff2fff1f00000001)], dtype=np.uint64),
        }
        ini = inits[pattern]
        if ((pattern == 'LL') and (position == 0)) or (pattern == '4432'):
            to_find_func, to_find_func1 = Calculator.p_minUL, Calculator.minUL
        else:
            to_find_func, to_find_func1 = Calculator.p_re_self, Calculator.re_self
        isfree = True if pattern in ['4441', ] else False  # 4441太大了，4432暂时观望
        gen_lookup_table_big(ini, eval(f'Calculator.is_{pattern}_pattern'), eval(f'Calculator.is_{pattern}_success'),
                             to_find_func, to_find_func1, target, position, steps, pathname, docheck_step, isfree,
                             spawn_rate4)
    return True


# 并行版本
@njit(nogil=True, parallel=True)
def gen_boards(arr0: np.ndarray[np.uint64],
               target: int,
               position: int,
               bm: BoardMover,
               pattern_check_func: PatternCheckFunc,
               success_check_func: SuccessCheckFunc,
               to_find_func: ToFindFunc,
               do_check: bool = True,
               isfree: bool = False) -> Tuple[np.ndarray[np.uint64], np.ndarray[np.uint64]]:
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
def expand(arr: np.ndarray[np.uint64]) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    arr0 = np.empty(len(arr), dtype='uint64,uint32')
    for i in prange(len(arr)):
        arr0[i]['f0'] = arr[i]
    return arr0


@njit(nogil=True)
def remove_zeros_inplace(
        arr: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]
        ) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    count = 0
    # 原地移动不为0的元素
    for i in range(len(arr)):
        if arr[i]['f1'] != 0:
            arr[count] = arr[i]
            count += 1
    return arr[:count]


if __name__ == "__main__":
    pass
    # bm = BoardMover()
    #
    # from Calculator import is_LL_pattern, is_LL_success, p_minUL, minUL
    # import time
    #
    # # 生成阶段
    # arr = np.fromfile(r"D:\2048calculates\test\LL_1024_0_347", dtype='uint64')
    #
    # # 预热
    # arr1, arr2 = gen_boards(arr[:10], 10, 0, bm, is_LL_pattern, is_LL_success, p_minUL, False, False)
    # arr1, arr2 = p_unique([arr1, arr2])
    # del arr1, arr2
    # arr1, arr2 = gen_boards(arr[:20], 10, 0, bm, is_LL_pattern, is_LL_success, p_minUL, False, False)
    # arr1, arr2 = p_unique([arr1, arr2])
    # del arr1, arr2
    #
    # # 生成
    # t0 = time.time()
    # arr1, arr2 = gen_boards(arr, 10, 0, bm, is_LL_pattern, is_LL_success, p_minUL, False, False)
    # t1 = time.time()
    # print(t1 - t0, len(arr), len(arr1), len(arr2))
    #
    # # 排序
    # pivots = arr[[len(arr) // 8 * i for i in range(1, 8)]]
    # sort_array(arr1, pivots)
    # sort_array(arr2, pivots)
    # print(time.time() - t1)
    #
    # # 去重
    # arr1, arr2 = p_unique([arr1, arr2])
    # t3 = time.time()
    # print(t3 - t1, len(arr), len(arr1), len(arr2))
    # print(t3 - t0)

    # 回算阶段

    # arr0 = np.fromfile(r"D:\2048calculates\test\LL_1024_0_459", dtype='uint64')
    # arr1 = np.fromfile(r"D:\2048calculates\test\LL_1024_0_460.book", dtype='uint64,uint32')
    # arr2 = np.fromfile(r"D:\2048calculates\test\LL_1024_0_461.book", dtype='uint64,uint32')
    #
    # # 预热
    # arr_ = expand(arr0[:10])
    # arr_ = recalculate(arr_, arr1, arr2, 10, 0, bm, is_LL_pattern, is_LL_success, minUL)
    # remove_zeros_inplace(arr_)
    # arr_ = expand(arr0[:20])
    # arr_ = recalculate(arr_, arr1, arr2, 10, 0, bm, is_LL_pattern, is_LL_success, minUL)
    # remove_zeros_inplace(arr_)
    #
    # # 扩充
    # t0 = time.time()
    # arr0 = expand(arr0)
    # t1 = time.time()
    # print(t1 - t0, len(arr0), len(arr1), len(arr2))
    #
    # # 回算
    # arr0 = recalculate(arr0, arr1, arr2, 10, 0, bm, is_LL_pattern, is_LL_success, minUL)
    # t3 = time.time()
    # print(t3 - t1)
    #
    # # 清理
    # arr0 = remove_zeros_inplace(arr0)
    # t4 = time.time()
    # print(t4 - t3, len(arr0))
    # print(t4 - t0)
