import os
import time
import gc

import numpy as np
from numba import njit, prange

import Calculator
from BoardMover import BoardMover
from Config import SingletonConfig
from TrieCompressor import trie_compress_progress


@njit(nogil=True)
def gen_boards(arr0, bm, to_find_func):
    """
    根据arr0中的面板，先生成数字，再移动，如果移动后仍是定式范围内且移动有效，则根据生成的数字（2,4）分别填入
    """
    # 初始化两个arr，分别对应填充数字2和4后的棋盘状态
    length = max(len(arr0) * 8, 249999999)
    arr1 = np.empty(length, dtype=np.uint64)
    arr2 = np.empty(length, dtype=np.uint64)
    c1t, c2t = 0, 0
    for t in arr0:
        for i in range(16):  # 遍历每个位置
            # 检查第i位置是否为空，如果为空，进行填充操作
            if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):
                # 分别用数字2和4填充当前空位，然后生成新的棋盘状态t1和t2
                t1 = t | (np.uint64(1) << np.uint64(4 * i))  # 填充数字2（2的对数为1，即4位中的0001）
                t2 = t | (np.uint64(2) << np.uint64(4 * i))  # 填充数字4（4的对数为2，即4位中的0010）

                # 尝试所有四个方向上的移动
                for newt in bm.move_all_dir(t1):
                    if newt != t1:
                        arr1[c1t] = to_find_func(newt)
                        c1t += 1
                for newt in bm.move_all_dir(t2):
                    if newt != t2:
                        arr2[c2t] = to_find_func(newt)
                        c2t += 1

    arr1 = arr1[:c1t]
    arr2 = arr2[:c2t]
    # 返回包含可能的新棋盘状态的两个array
    return arr1, arr2


def gen_boards_big(arr0, bm, to_find_func, d1):
    """将arr0分段放入gen_boards生成排序去重后的局面，然后归并"""
    segment_size = 119999999
    start_index, seg_index = 0, 0
    arr1s = [d1]
    arr2s = []

    t0 = time.time()
    while start_index < len(arr0):
        tt0 = time.time()
        seg_length = {0:39999999, 1:49999999, 2:69999999}.get(seg_index, segment_size)
        end_index = min(start_index + seg_length, len(arr0))
        arr0t = arr0[start_index:end_index].copy()
        arr1t, arr2t = gen_boards(arr0t, bm, to_find_func)
        del arr0t
        print(end_index-start_index, len(arr1t))
        tt1 = time.time()
        arr1t = np.unique(arr1t)
        arr2t = np.unique(arr2t)
        arr1s.append(arr1t)
        arr2s.append(arr2t)
        print(round(time.time()-tt1,3),round(tt1-tt0,3),len(arr1t),flush=True)
        start_index = end_index
        seg_index += 1

    t1 = time.time()
    length = int(max(len(arr0) * 1.25, 199999999))
    gc.collect()
    arr1 = merge_deduplicate_all(arr1s, length)
    del arr1s, arr1t
    t2 = time.time()
    arr2 = merge_deduplicate_all(arr2s, length)
    del arr2s, arr2t
    print(round(time.time()-t2,3),round(t2-t1,3),round(t1-t0,3),flush=True)
    return arr1, arr2


@njit(nogil=True)
def merge_deduplicate_all(arrays, length):
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
def recalculate(arr0, arr1, arr2, bm, to_find_func):
    """
    根据已经填充好成功概率的array回算前一批面板的成功概率。
    对于arr0中的每个面板，考虑在每个空位填充数字2或4（90%概率为2，10%概率为4），
    然后对于每个可能的填充，执行所有有效的移动操作，并基于移动结果的成功概率来更新当前面板的成功概率。
    """
    for k in prange(len(arr0)):
        t = arr0[k][0]
        # 初始化概率和权重
        expected_score = 0
        empty_slots = 0
        for i in range(16):  # 遍历所有位置
            if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == np.uint64(0):  # 如果当前位置为空
                empty_slots += 1
                # 对于每个空位置，尝试填充2和4
                for new_value, probability in [(1, 0.9), (2, 0.1)]:  # 填充2的概率为90%，填充4的概率为10%
                    subs_arr = arr1 if new_value == 1 else arr2
                    t_gen = t | (np.uint64(new_value) << np.uint64(4 * i))
                    max_expected_score = 0  # 记录有效移动后的面板成功概率中的最大值
                    for newt in bm.move_all_dir(t_gen):
                        if newt != t_gen:  # 只考虑有效的移动
                            # 获取移动后的面板成功概率
                            max_expected_score = max(max_expected_score, binary_search_arr(
                                subs_arr, to_find_func(newt)))
                    if max_expected_score == 0:  # 游戏结束
                        max_expected_score = total_score(t_gen)
                    # 对最佳移动下的成功概率加权平均
                    expected_score += np.uint64(max_expected_score * probability)
        # t是进行一次有效移动后尚未生成新数字时的面板，因此不可能没有空位置
        arr0[k][1] = np.uint64(expected_score / empty_slots)
    return arr0


@njit(nogil=True)
def binary_search_arr(arr, target):
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
            return arr[mid][1]  # 找到匹配，返回对应的int32值
    print(f'{target} not found!')
    return 0  # 如果没有找到匹配项


@njit(nogil=True)
def merge_and_deduplicate(sorted_arr1, sorted_arr2):
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


def gen_lookup_table_big(arr_init, to_find_func, steps, pathname):
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """
    bm = BoardMover()
    started, d0, d1 = generate_process(arr_init, to_find_func, steps, pathname, bm)
    d0, d1 = final_steps(started, d0, d1, pathname, steps)
    recalculate_process(d0, d1, to_find_func, steps, pathname, bm)  # 这里的最后的两个book d0,d1就是回算的d1,d2


def generate_process(arr_init, to_find_func, steps, pathname, bm):
    started = False
    d0, d1 = None, None
    # 从前向后遍历，生成新的棋盘状态并保存到相应的array中
    for i in range(1, steps - 1):
        # 断点重连
        if os.path.exists(pathname + str(i)) or os.path.exists(pathname + str(i) + '.book') or \
                os.path.exists(pathname + str(i) + '.z'):
            print(f"skipping step {i}",flush=True)
            continue
        if i == 1:
            arr_init.tofile(pathname + str(i - 1))
            d0, d1 = arr_init, np.empty(0, dtype=np.uint64)
            started = True
        elif not started:
            d0 = np.fromfile(pathname + str(i - 1), dtype=np.uint64)
            d1 = np.fromfile(pathname + str(i) + '.t', dtype=np.uint64)
            started = True

        segment_size = 69999999
        # 生成新的棋盘状态
        if len(d0) < segment_size:
            d1t, d2 = gen_boards(d0, bm, to_find_func)
            d1t = np.unique(d1t)
            d2 = np.unique(d2)
            d1 = merge_and_deduplicate(d1, d1t)
            d0, d1 = d1, d2
            del d1t, d2
        else:
            d0, d1 = gen_boards_big(d0, bm, to_find_func, d1)
        # print(f"Processing step {i}",flush=True)
        d1.tofile(pathname + str(i + 1) + '.t')
        d0.tofile(pathname + str(i))
        if os.path.exists(pathname + str(i) + '.t'):
            os.remove(pathname + str(i) + '.t')
        gc.collect()
    return started, d0, d1


def final_steps(started, d0, d1, pathname, steps):
    if started:
        expanded_arr0 = np.empty(len(d0), dtype='uint64,uint64')
        expanded_arr0['f0'] = d0
        d0 = final_situation_process(expanded_arr0)
        d0.tofile(pathname + str(steps - 2) + '.book')
        expanded_arr0 = np.empty(len(d1), dtype='uint64,uint64')
        expanded_arr0['f0'] = d1
        d1 = final_situation_process(expanded_arr0)
        d1.tofile(pathname + str(steps - 1) + '.book')
    if os.path.exists(pathname + str(steps - 1) + '.t'):
        os.remove(pathname + str(steps - 1) + '.t')
    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


def recalculate_process(d1, d2, to_find_func, steps, pathname, bm):
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
                d1 = np.fromfile(pathname + str(i + 1) + '.book', dtype='uint64,uint64')
                d2 = np.fromfile(pathname + str(i + 2) + '.book', dtype='uint64,uint64')

        d0 = np.fromfile(pathname + str(i), dtype=np.uint64)
        expanded_arr0 = np.empty(len(d0), dtype='uint64,uint64')
        expanded_arr0['f0'] = d0
        del d0
        d0 = recalculate(expanded_arr0, d1, d2, bm, to_find_func)
        d0.tofile(pathname + str(i) + '.book')
        if os.path.exists(pathname + str(i)):
            os.remove(pathname + str(i))
        do_compress(pathname + str(i + 2) + '.book')  # 如果设置了压缩，则压缩i+2的book，其已经不需要再频繁查找
        if i > 0:
            d1, d2 = d0, d1
        # print(f"Updated step {i}")


@njit(nogil=True, parallel=True)
def final_situation_process(expanded_arr0):
    for i in prange(len(expanded_arr0)):
        expanded_arr0[i][1] = total_score(expanded_arr0[i][0])
    return expanded_arr0


@njit(nogil=True)
def total_score(b, n=16):
    score = 0
    for i in range(n):  # 遍历所有位置
        tile = (b >> np.uint64(4 * i)) & np.uint64(0xF)
        if tile == np.uint64(15):
            continue
        score += (tile - 1) * 2 ** tile
    return score * 2 ** 42


def do_compress(bookpath):
    if bookpath[-4:] != 'book':
        return
    if SingletonConfig().config.get('compress', False) and os.path.exists(bookpath):
        if os.path.getsize(bookpath) > 16000:
            trie_compress_progress(*os.path.split(bookpath))
            if os.path.exists(bookpath):
                os.remove(bookpath)


def start_build(pattern, pathname):
    inits = {
        '34': np.array([np.uint64(0x100000000000ffff), np.uint64(0x000000010000ffff)], dtype=np.uint64),
        '24': np.array([np.uint64(0xffff00001000ffff), np.uint64(0xffff00010000ffff)], dtype=np.uint64),
        '33': np.array([np.uint64(0x100f000f000fffff), np.uint64(0x000f000f001fffff)], dtype=np.uint64),
    }
    ini = inits[pattern]
    if pattern == '33':
        to_find_func = Calculator.min33
        steps = 1022
    elif pattern == '34':
        to_find_func = Calculator.min34
        steps = 8190
    elif pattern == '24':
        to_find_func = Calculator.min24
        steps = 510
    else:
        to_find_func = Calculator.re_self
        steps = 0
    gen_lookup_table_big(ini, to_find_func, steps, pathname)
    return True
