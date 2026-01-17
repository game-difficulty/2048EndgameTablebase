import os
from itertools import combinations
from typing import Callable, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, from_dtype
from numba import types
from numba.typed import Dict

import Calculator
from BoardMover import move_all_dir, decode_board
from Config import SingletonConfig, formation_info, pattern_32k_tiles_map, DTYPE_CONFIG, category_info
from BookSolver import recalculate_process, remove_died, expand, keep_only_optimal_branches
from BookGenerator import generate_process
from BookGeneratorAD import generate_process_ad
from BoardMaskerAD import init_masker
from BookSolverAD import recalculate_process_ad
from BookSolverChunkedAD import recalculate_process_ad_c

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

ValidDType: TypeAlias = Union[type[np.uint32], type[np.uint64], type[np.float32], type[np.float64]]
SuccessRateDType: TypeAlias = Union[np.uint32, np.uint64, np.float32, np.float64]


KeyType = types.int8
ValueType1 = NDArray[NDArray[SuccessRateDType]]
BookDictType = Dict[KeyType, ValueType1]
ValueType2 = types.uint64[:]
IndexDictType = Dict[KeyType, ValueType2]


def gen_lookup_table_big(
        pattern: str,
        arr_init: NDArray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        success_check_func: SuccessCheckFunc,
        canonical_func: CanonicalFunc,
        target: int,
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
    is_variant = pattern in category_info.get('variant', [])
    # variant tables不支持进阶算法，用不到sym_func，随便返回一个

    sym_func = {Calculator.canonical_identity: Calculator.canonical_identity_pair,
                Calculator.canonical_diagonal: Calculator.canonical_diagonal_pair,
                Calculator.canonical_full: Calculator.canonical_full_pair,
                Calculator.canonical_horizontal: Calculator.canonical_horizontal_pair,
                }.get(canonical_func, Calculator.canonical_identity_pair)

    save_config_to_txt(pathname + 'config.txt')
    if not SingletonConfig().config.get('advanced_algo', False):
        #  存在断点重连的情况下，以下d0, d1均可能为None
        started, d0, d1 = generate_process(arr_init, pattern_check_func, success_check_func, canonical_func, target,
                                           steps, pathname, docheck_step, isfree, is_variant)
        d0, d1 = final_steps(started, d0, d1, pathname, steps, success_check_func, target)
        recalculate_process(d0, d1, pattern_check_func, success_check_func, canonical_func, target, steps,
                            pathname, docheck_step, spawn_rate4, is_variant)  # 这里的最后的两个book d0,d1就是回算的d1,d2
    else:
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]
        permutation_dict, tiles_combinations_dict, param = init_masker(num_free_32k, target, pos_fixed_32k)
        started, d0, d1 = generate_process_ad(arr_init, pattern_check_func, canonical_func, sym_func,
                                              steps, pathname, isfree, tiles_combinations_dict, param)
        b1, b2, i1, i2 = final_steps_ad(started, pathname, steps)

        if SingletonConfig().config['chunked_solve']:
            recalculate_process_ad_c(arr_init, b1, b2, i1, i2, tiles_combinations_dict, permutation_dict, param,
                                     pattern_check_func, sym_func, steps, pathname, spawn_rate4)
        else:
            recalculate_process_ad(arr_init, b1, b2, i1, i2, tiles_combinations_dict, permutation_dict, param,
                                   pattern_check_func, sym_func, steps, pathname, spawn_rate4)
    if SingletonConfig().config['optimal_branch_only']:
        keep_only_optimal_branches(pattern_check_func, canonical_func, steps, pathname)


def save_config_to_txt(output_path):
    keys = ['compress', 'optimal_branch_only', 'compress_temp_files', 'advanced_algo',
            'deletion_threshold', '4_spawn_rate', 'success_rate_dtype']
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
                ) -> Tuple[NDArray, NDArray]:
    if started:
        template, current_dtype, max_scale, zero_val = DTYPE_CONFIG[SingletonConfig().config.get(
            'success_rate_dtype', 'uint32')]

        arr0 = np.empty(len(d0), dtype=template.dtype)
        expanded_arr0 = expand(d0, arr0)
        d0 = final_situation_process(expanded_arr0, success_check_func, target, max_scale, zero_val)
        d0.tofile(pathname + str(steps - 2) + '.book')

        arr0 = np.empty(len(d1), dtype=template.dtype)
        expanded_arr0 = expand(d1, arr0)
        d1 = final_situation_process(expanded_arr0, success_check_func, target, max_scale, zero_val)
        d1.tofile(pathname + str(steps - 1) + '.book')

    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


def final_steps_ad(started: bool,
                   pathname: str,
                   steps: int) -> Tuple:
    if started:
        template, current_dtype, max_scale, zero_val = DTYPE_CONFIG[SingletonConfig().config.get(
            'success_rate_dtype', 'uint32')]
        b0, i0 = Dict.empty(KeyType, from_dtype(current_dtype)[:, :]), Dict.empty(KeyType, ValueType2)
        b1, i1 = Dict.empty(KeyType, from_dtype(current_dtype)[:, :]), Dict.empty(KeyType, ValueType2)
        os.makedirs(pathname + str(steps - 1) + 'b', exist_ok=True)
        os.makedirs(pathname + str(steps - 2) + 'b', exist_ok=True)
        if os.path.exists(pathname + str(steps - 2)):
            os.remove(pathname + str(steps - 2))
        return b0, b1, i0, i1
    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return None, None, None, None


@njit(nogil=True, parallel=True, cache=True)
def final_situation_process(expanded_arr0: NDArray[[np.uint64, SuccessRateDType]],
                            success_check_func: SuccessCheckFunc, target: int,
                            max_scale: int, zero_val: int,
                            ) -> NDArray[[np.uint64, SuccessRateDType]]:
    for i in prange(len(expanded_arr0)):
        if success_check_func(expanded_arr0[i][0], target):
            expanded_arr0[i][1] = max_scale
        else:
            expanded_arr0[i][1] = zero_val
    expanded_arr0 = remove_died(expanded_arr0, zero_val)
    return expanded_arr0


def generate_free_inits(t32ks: int, t2s: int) -> np.ndarray:
    """
    生成不包含 target 数字的初始合法局面。
    仅放置 t32ks 个 32k 块和 t2s 个数字 2。
    """
    max_estimated = 1000000
    generated = np.empty(max_estimated, dtype=np.uint64)
    c = 0

    # 放置 32k
    for poss in combinations(range(16), t32ks):
        v_base = np.uint64(0)
        for p in poss:
            v_base |= np.uint64(15 << (p * 4))
        remain_pos = set(range(16)) - set(poss)

        # 在剩余位置放置数字 2
        for poss_2 in combinations(remain_pos, t2s):
            if c >= max_estimated:
                break
            val = v_base
            for j in poss_2:
                val |= np.uint64(1 << (j * 4))

            generated[c] = val
            c += 1

    generated = np.unique(generated[:c])

    length = len(generated)
    g2 = np.empty(length, dtype=np.uint64)
    count_final = 0

    for b in generated:
        # 模拟四个方向的移动
        for nb in move_all_dir(b):
            nb = np.uint64(nb)
            if nb == np.uint64(Calculator.canonical_full(nb)):
                g2[count_final] = nb
                count_final += 1
        if count_final >= length - 3:
            break

    return np.unique(g2[:count_final])


def start_build(pattern: str, target: int, pathname: str) -> bool:
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    tile_sum, pattern_check_func, canonical_func, success_check_func, ini, extra_steps = formation_info[pattern]
    if pattern[:4] == 'free':
        steps = int(2 ** target / 2 + extra_steps)
        docheck_step = int(2 ** target / 2) - tile_sum % int(2 ** target) // 2

        ini_decoded = decode_board(ini[0])
        n32k = np.sum(ini_decoded==32768)
        extra_tile_sum = tile_sum - 32768 * n32k
        if extra_tile_sum <= (14 - n32k) * 2:
            arr_init = generate_free_inits(n32k, extra_tile_sum // 2)
        else:
            arr_init = ini

        gen_lookup_table_big(pattern, arr_init, pattern_check_func, success_check_func,
                             canonical_func, target, steps,
                             pathname, docheck_step, isfree=True, spawn_rate4=spawn_rate4)
    else:
        steps = int(2 ** target / 2 + extra_steps)
        docheck_step = int(2 ** target / 2) - tile_sum % int(2 ** target) // 2

        isfree = (len(pattern_32k_tiles_map[pattern][2]) < 4) and (tile_sum < 180000)
        gen_lookup_table_big(pattern, ini, pattern_check_func, success_check_func, canonical_func,
                             target, steps, pathname, docheck_step, isfree, spawn_rate4)
    return True


def v_start_build(pattern: str, target: int, pathname: str) -> bool:
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    tile_sum, pattern_check_func, canonical_func, success_check_func, ini, extra_steps = formation_info[pattern]

    steps = int(2 ** target / 2 + extra_steps)
    docheck_step = int(2 ** target / 2) - tile_sum % int(2 ** target) // 2

    isfree = True
    gen_lookup_table_big(pattern, ini, pattern_check_func, success_check_func, canonical_func, target,
                         steps, pathname, docheck_step, isfree, spawn_rate4)
    return True


# if __name__ == "__main__":
#     start_build('t',10,0,r"C:\Apps\2048endgameTablebase\tables\t1k-118\t_1024_")
#     pass
