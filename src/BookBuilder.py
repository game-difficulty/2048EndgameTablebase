import os
from itertools import combinations, permutations
from typing import Callable

import numpy as np
from numba import njit, prange

import Calculator
from BoardMover import SingletonBoardMover, BoardMover
from Config import SingletonConfig
from BookSolver import recalculate_process, remove_died, expand
from BookGenerator import generate_process

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.ndarray[np.uint64]], None]
ToFindFunc1 = Callable[[np.uint64], np.uint64]


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
        bm: BoardMover,
        isfree: bool = False,
        spawn_rate4: float = 0.1,
) -> None:
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """

    started, d0, d1 = generate_process(arr_init, pattern_check_func, success_check_func, to_find_func, target, position,
                                       steps, pathname, docheck_step, bm, isfree)
    d0, d1 = final_steps(started, d0, d1, pathname, steps, success_check_func, target, position)
    recalculate_process(d0, d1, pattern_check_func, success_check_func, to_find_func1, target, position, steps,
                        pathname, docheck_step, bm, spawn_rate4)  # 这里的最后的两个book d0,d1就是回算的d1,d2


def final_steps(started: bool,
                d0: np.ndarray[np.uint64],
                d1: np.ndarray[np.uint64],
                pathname: str,
                steps: int,
                success_check_func: SuccessCheckFunc,
                target: int,
                position: int):
    if started:
        expanded_arr0 = expand(d0)
        d0 = final_situation_process(expanded_arr0, success_check_func, target, position)
        d0.tofile(pathname + str(steps - 2) + '.book')

        expanded_arr0 = expand(d1)
        d1 = final_situation_process(expanded_arr0, success_check_func, target, position)
        d1.tofile(pathname + str(steps - 1) + '.book')

    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


@njit(nogil=True, parallel=True)
def final_situation_process(expanded_arr0: np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])],
                            success_check_func: SuccessCheckFunc, target: int, position: int
                            ) -> np.ndarray[np.dtype([('f0', np.uint64), ('f1', np.uint32)])]:
    for i in prange(len(expanded_arr0)):
        if success_check_func(expanded_arr0[i][0], target, position):
            expanded_arr0[i][1] = 4000000000
        else:
            expanded_arr0[i][1] = 0
    expanded_arr0 = remove_died(expanded_arr0)
    return expanded_arr0


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
    bm = SingletonBoardMover(1)
    for b in generated:
        for nb in bm.move_all_dir(b):
            nb = np.uint64(nb)
            if nb == np.uint64(Calculator.min_all_symm(nb)):
                g2[c] = nb
                c += 1
    return np.unique(g2[:c])


def start_build(pattern: str, target: int, position: int, pathname: str) -> bool:
    bm = BoardMover()
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    if pattern[:4] == 'free':
        if pattern[-1] != 'w':
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 5
            free_tiles = int(pattern[4:])
            arr_init = generate_free_inits(target, 15 - free_tiles, free_tiles)
            gen_lookup_table_big(arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.p_min_all_symm, Calculator.min_all_symm, target, 0, steps, pathname,
                                 docheck_step, bm, isfree=True, spawn_rate4=spawn_rate4)
        else:
            # freew定式pos参数设为1，配合is_free_success中的设置
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 5
            free_tiles = int(pattern[4:-1])
            arr_init = generate_free_inits(0, 16 - free_tiles, free_tiles - 1)
            gen_lookup_table_big(arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.p_min_all_symm, Calculator.min_all_symm, target, 1, steps, pathname,
                                 docheck_step, bm, isfree=True, spawn_rate4=spawn_rate4)
    else:
        steps = int(2 ** target / 2 + {'444': 96, '4431': 64, 'LL': 48, 'L3': 36, '4441': 48, '4432': 48, '4442': 48,
                                       '442': 36, 't': 36, }[pattern])
        docheck_step = int(2 ** target / 2) - 16
        inits = {
            '444': np.array([np.uint64(0x100000000000ffff), np.uint64(0x000000010000ffff)], dtype=np.uint64),
            '4431': np.array([np.uint64(0x10000000123f2fff), np.uint64(0x00000001123f2fff)], dtype=np.uint64),
            'LL': np.array([np.uint64(0x1000000023ff24ff), np.uint64(0x0000000123ff24ff)], dtype=np.uint64),
            'L3': np.array([np.uint64(0x100000001fff2fff), np.uint64(0x000000011fff2fff)], dtype=np.uint64),
            '4441': np.array([np.uint64(0x0000100012323fff), np.uint64(0x0001000012323fff)], dtype=np.uint64),
            '4432': np.array([np.uint64(0x00001000123f23ff), np.uint64(0x00010000123f23ff)], dtype=np.uint64),
            '4442': np.array([np.uint64(0x00001000123424ff), np.uint64(0x00010000123424ff)], dtype=np.uint64),
            '442': np.array([np.uint64(0x1000000012ffffff), np.uint64(0x0000000112ffffff)], dtype=np.uint64),
            't': np.array([np.uint64(0x10000000f1fff2ff), np.uint64(0x00000001f1fff2ff)], dtype=np.uint64),
        }
        ini = inits[pattern]
        if ((pattern == 'LL') and (position == 0)) or (pattern == '4432'):
            to_find_func, to_find_func1 = Calculator.p_minUL, Calculator.minUL
        else:
            to_find_func, to_find_func1 = Calculator.p_re_self, Calculator.re_self
        isfree = True if pattern in ('4442', '4432', '4441') else False
        gen_lookup_table_big(ini, eval(f'Calculator.is_{pattern}_pattern'), eval(f'Calculator.is_{pattern}_success'),
                             to_find_func, to_find_func1, target, position, steps, pathname, docheck_step, bm, isfree,
                             spawn_rate4)
    return True


if __name__ == "__main__":
    pass
    # bm = BoardMover()
    #
    # from Calculator import is_4432_pattern, is_4432_success, p_minUL, minUL
    # import time
    #
    # # 生成阶段
    # arr = np.fromfile(r"C:\Users\ThinkPad\Desktop\新建文件夹\4432_128_46", dtype='uint64')
    #
    # # 预热
    # arr1, arr2 = gen_boards(arr[:10], 10, 0, bm, is_4432_pattern, is_4432_success, p_minUL, False, False)
    # arr1 = concatenate(arr1)
    # arr2 = concatenate(arr2)
    # arr1, arr2 = p_unique([arr1, arr2])
    # del arr1, arr2
    # arr1, arr2 = gen_boards(arr[:20], 10, 0, bm, is_4432_pattern, is_4432_success, p_minUL, False, False)
    # arr1 = concatenate(arr1)
    # arr2 = concatenate(arr2)
    # arr1, arr2 = p_unique([arr1, arr2])
    # del arr1, arr2
    #
    # # 生成
    # t0 = time.time()
    # arr1, arr2 = gen_boards(arr, 10, 0, bm, is_4432_pattern, is_4432_success, p_minUL, False, False)
    # arr1 = concatenate(arr1)
    # arr2 = concatenate(arr2)
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
