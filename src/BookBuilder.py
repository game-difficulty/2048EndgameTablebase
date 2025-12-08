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
from Config import SingletonConfig, formation_info, pattern_32k_tiles_map
from BookSolver import recalculate_process, remove_died, expand, keep_only_optimal_branches
from BookGenerator import generate_process
from BookGeneratorAD import generate_process_ad
from BoardMaskerAD import init_masker
from BookSolverAD import recalculate_process_ad
from BookSolverChunkedAD import recalculate_process_ad_c

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
        spawn_rate4: float = 0.1,
) -> None:
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """
    is_variant = pattern in ('2x4', '3x3', '3x4')
    # variant tables不支持进阶算法，用不到sym_func，随便返回一个
    sym_func = {Calculator.re_self: Calculator.re_self_pair,
                Calculator.minUL: Calculator.minUL_pair,
                Calculator.min_all_symm: Calculator.min_all_symm_pair}.get(to_find_func, Calculator.re_self_pair)
    save_config_to_txt(pathname + 'config.txt')
    if not SingletonConfig().config.get('advanced_algo', False):
        #  存在断点重连的情况下，以下d0, d1均可能为None
        started, d0, d1 = generate_process(arr_init, pattern_check_func, success_check_func, to_find_func, target,
                                           position, steps, pathname, docheck_step, isfree, is_variant)
        d0, d1 = final_steps(started, d0, d1, pathname, steps, success_check_func, target, position)
        recalculate_process(d0, d1, pattern_check_func, success_check_func, to_find_func, target, position, steps,
                            pathname, docheck_step, spawn_rate4, is_variant)  # 这里的最后的两个book d0,d1就是回算的d1,d2
    else:
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]
        permutation_dict, tiles_combinations_dict, param = init_masker(num_free_32k, target, pos_fixed_32k)
        started, d0, d1 = generate_process_ad(arr_init, pattern_check_func, to_find_func, sym_func,
                                              steps, pathname, isfree, tiles_combinations_dict, param)
        b1, b2, i1, i2 = final_steps_ad(started, pathname, steps)

        if SingletonConfig().config['chunked_solve']:
            recalculate_process_ad_c(arr_init, b1, b2, i1, i2, tiles_combinations_dict, permutation_dict, param,
                                     pattern_check_func, sym_func, steps, pathname, spawn_rate4)
        else:
            recalculate_process_ad(arr_init, b1, b2, i1, i2, tiles_combinations_dict, permutation_dict, param,
                                   pattern_check_func, sym_func, steps, pathname, spawn_rate4)
    if SingletonConfig().config['optimal_branch_only']:
        keep_only_optimal_branches(pattern_check_func, to_find_func, steps, pathname)


def save_config_to_txt(output_path):
    keys = ['compress', 'optimal_branch_only', 'compress_temp_files', 'advanced_algo',
            'deletion_threshold', '4_spawn_rate']
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for key in keys:
            f.write(f"{key}: {str(SingletonConfig().config.get(key, '?'))}\n")


def final_steps(started: bool,
                d0: NDArray[np.uint64],
                d1: NDArray[np.uint64],
                pathname: str,
                steps: int,
                success_check_func: SuccessCheckFunc,
                target: int,
                position: int) -> Tuple[NDArray, NDArray]:
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


def final_steps_ad(started: bool,
                   pathname: str,
                   steps: int) -> Tuple:
    if started:
        b0, i0 = Dict.empty(KeyType, ValueType1), Dict.empty(KeyType, ValueType2)
        b1, i1 = Dict.empty(KeyType, ValueType1), Dict.empty(KeyType, ValueType2)
        os.makedirs(pathname + str(steps - 1) + 'b', exist_ok=True)
        os.makedirs(pathname + str(steps - 2) + 'b', exist_ok=True)
        if os.path.exists(pathname + str(steps - 2)):
            os.remove(pathname + str(steps - 2))
        return b0, b1, i0, i1
    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return None, None, None, None


@njit(nogil=True, parallel=True, cache=True)
def final_situation_process(expanded_arr0: NDArray[[np.uint64, np.uint32]],
                            success_check_func: SuccessCheckFunc, target: int, position: int
                            ) -> NDArray[[np.uint64, np.uint32]]:
    for i in prange(len(expanded_arr0)):
        if success_check_func(expanded_arr0[i][0], target, position):
            expanded_arr0[i][1] = 4000000000
        else:
            expanded_arr0[i][1] = 0
    expanded_arr0 = remove_died(expanded_arr0)
    return expanded_arr0


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
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    if pattern[:4] == 'free':
        if pattern[-1] != 'w':
            steps = int(2 ** target / 2 + 36)
            docheck_step = int(2 ** target / 2) - 20
            free_tiles = int(pattern[4:])
            arr_init = generate_free_inits(target, 15 - free_tiles, free_tiles)
            gen_lookup_table_big(pattern, arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.min_all_symm, target, 0, steps,
                                 pathname, docheck_step, isfree=True, spawn_rate4=spawn_rate4)
        else:
            # freew定式pos参数设为1，配合is_free_success中的设置
            steps = int(2 ** target / 2 + 36)
            docheck_step = int(2 ** target / 2) - 20
            free_tiles = int(pattern[4:-1])
            arr_init = generate_free_inits(0, 16 - free_tiles, free_tiles - 1)
            gen_lookup_table_big(pattern, arr_init, Calculator.is_free_pattern, Calculator.is_free_success,
                                 Calculator.min_all_symm, target, 1, steps,
                                 pathname, docheck_step, isfree=True, spawn_rate4=spawn_rate4)
    else:
        steps = int(2 ** target / 2 + {'444': 96, '4431': 64, 'LL': 48, 'L3': 36, '4441': 48, '4432': 48, '4442': 48,
                                       '442': 36, '442t': 36, 't': 36, '4432f': 48, '4432ff': 48, 'L3t': 48, '4442f': 48,
                                       '4442ff': 48, "3433": 48, "3442": 48, "3432": 48, "2433": 48, "4441f": 48
                                       }[pattern])
        docheck_step = int(2 ** target / 2) - 16
        _, pattern_check_func, to_find_func, success_check_func, ini = formation_info[pattern]
        if pattern == 'LL' and position == 1:
            to_find_func = Calculator.re_self
        if pattern in ('4442', '4432', '4441', "3433", "3442", "3432", "4441f", '4432ff', '4432f', '4442f', '4442ff'):
            isfree = True
        else:
            isfree = False
        gen_lookup_table_big(pattern, ini, pattern_check_func, success_check_func, to_find_func,
                             target, position, steps, pathname, docheck_step, isfree, spawn_rate4)
    return True


# if __name__ == "__main__":
#     start_build('t',10,0,r"C:\Apps\2048endgameTablebase\tables\t1k-118\t_1024_")
#     pass
