import os
from itertools import combinations, permutations
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from numba import types
from numba.typed import Dict

import Calculator
from BoardMover import move_all_dir
from Config import SingletonConfig, formation_info
from BookSolver import recalculate_process, remove_died, expand
from BookGenerator import generate_process


PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int, int], bool]
ToFindFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

KeyType = types.int8
ValueType1 = types.uint32[:, :]
BookDictType = Dict[KeyType, ValueType1]
ValueType2 = types.uint64[:]
IndexDictType = Dict[KeyType, ValueType2]


def gen_lookup_table_big(
        pattern: str,
        arr_init: NDArray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        to_find_func: ToFindFunc,
        target: int,
        position: int,
        steps: int,
        pathname: str,
        docheck_step: int,
        isfree: bool = False,
) -> None:
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """
    started, d0, d1 = generate_process(arr_init, pattern_check_func, success_check_func, to_find_func, target,
                                       position, steps, pathname, docheck_step, isfree)
    rate0, rate1 = final_steps(started, d0, d1, pathname, steps, success_check_func, target, position)
    recalculate_process(d0, rate0, d1, rate1, pattern_check_func, success_check_func, to_find_func, target, position,
                        steps, pathname, docheck_step)


def final_steps(started: bool,
                d0: NDArray[np.uint64],
                d1: NDArray[np.uint64],
                pathname: str,
                steps: int,
                success_check_func: SuccessCheckFunc,
                target: int,
                position: int) -> Tuple[NDArray, NDArray]:
    if started:
        points = SingletonConfig().config['data_points']
        rate0 = expand(d0, points)
        rate0, d0 = final_situation_process(d0, rate0, success_check_func, target, position)
        d0.tofile(pathname + str(steps - 2))
        rate0.tofile(pathname + str(steps - 2) + '.book')

        rate1 = expand(d1, points)
        rate1, d1 = final_situation_process(d1, rate1, success_check_func, target, position)
        d1.tofile(pathname + str(steps - 1))
        rate1.tofile(pathname + str(steps - 1) + '.book')
        return rate0, rate1

    return None, None


@njit(nogil=True, parallel=True, cache=True)
def final_situation_process(d0, rate0,
                            success_check_func: SuccessCheckFunc, target: int, position: int
                            ) -> NDArray[[np.uint64, np.uint32]]:
    for i in prange(len(d0)):
        if success_check_func(d0[i], target, position):
            rate0[i] = 4000000000
        else:
            rate0[i] = 0
    rate0, d0 = remove_died(rate0, d0)
    return rate0, d0


def generate_free_inits(target: int, t32ks: int, t2s: int) -> NDArray[np.uint64]:
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

    for b in generated:
        for nb in move_all_dir(b):
            nb = np.uint64(nb)
            if nb == np.uint64(Calculator.min_all_symm(nb)):
                g2[c] = nb
                c += 1
    return np.unique(g2[:c])


def start_build(pattern: str, target: int, position: int, pathname: str) -> bool:
    if pattern[:4] == 'free':
        if pattern[-1] != 'w':
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 20
            free_tiles = int(pattern[4:])
            arr_init = generate_free_inits(target, 15 - free_tiles, free_tiles)
            gen_lookup_table_big(pattern, arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.min_all_symm, target, 0, steps,
                                 pathname, docheck_step, isfree=True)
        else:
            # freew定式pos参数设为1，配合is_free_success中的设置
            steps = int(2 ** target / 2 + 24)
            docheck_step = int(2 ** target / 2) - 20
            free_tiles = int(pattern[4:-1])
            arr_init = generate_free_inits(0, 16 - free_tiles, free_tiles - 1)
            gen_lookup_table_big(pattern, arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.min_all_symm, target, 1, steps,
                                 pathname, docheck_step, isfree=True)
    else:
        steps = int(2 ** target / 2 + {'444': 96, '4431': 64, 'LL': 48, 'L3': 36, '4441': 48, '4432': 48, '4442': 48,
                                       '442': 36, '442t': 36, 't': 36, '4432f': 48, '4432ff': 48, 'L3t': 48, '4442f': 48,
                                       '4442ff': 48, "3433": 48, "3442": 48, "3432": 48, "2433": 48, "4441f": 48
                                       }[pattern])
        docheck_step = int(2 ** target / 2) - 16
        _, pattern_check_func, to_find_func, success_check_func, ini = formation_info[pattern]
        if pattern == 'LL' and position == 1:
            to_find_func = Calculator.re_self
        if pattern in ('4442', '4432', '4441', "3433", "3442", "3432", "4441f", '4432ff', '4432f', '4442f'):
            isfree = True
        else:
            isfree = False
        gen_lookup_table_big(pattern, ini, pattern_check_func, success_check_func, to_find_func,
                             target, position, steps, pathname, docheck_step, isfree)
    return True


if __name__ == "__main__":
    start_build('t',10,0,r"C:\Apps\2048endgameTablebase\tables\t1k-118\t_1024_")
    pass
