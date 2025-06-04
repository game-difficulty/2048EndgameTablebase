import os
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import types
from numba.typed import Dict

from BoardMover import SingletonBoardMover, BoardMoverWithScore

from Config import SingletonConfig, formation_info
from BookSolver import recalculate_process, expand
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
        arr_init: NDArray[np.uint64],
        pattern_check_func: PatternCheckFunc,
        to_find_func: ToFindFunc,
        target: int,
        steps: int,
        pathname: str,
        bm: BoardMoverWithScore,
        pattern_encoded: np.uint64,
        spawn_rate4: float = 0.1,
) -> None:
    """
    传入包含所有初始局面的array，然后按照面板数字和依次生成下一阶段的所有局面。储存轮到系统生成数字时的面板。
    保障其中的每个arr储存的面板的数字和均相等
    """



    #  存在断点重连的情况下，以下d0, d1均可能为None
    started, d0, d1 = generate_process(arr_init, pattern_check_func, to_find_func,
                                       steps, pathname, bm, pattern_encoded, target)
    d0, d1 = final_steps(started, d0, d1, pathname, steps)

    recalculate_process(d0, d1, pattern_check_func, to_find_func, steps,
                        pathname, pattern_encoded, target, bm, spawn_rate4)  # 这里的最后的两个book d0,d1就是回算的d1,d2


def final_steps(started: bool,
                d0: NDArray[np.uint64],
                d1: NDArray[np.uint64],
                pathname: str,
                steps: int,) -> Tuple[NDArray, NDArray]:
    if started:
        d0 = expand(d0)
        d0['f1'] = np.uint32(4e9)
        d0.tofile(pathname + str(steps - 2) + '.book')

        d1 = expand(d1)
        d1['f1'] = np.uint32(4e9)
        d1.tofile(pathname + str(steps - 1) + '.book')

    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


def start_build(pattern: str, target: int, pathname: str, steps: int) -> bool:

    spawn_rate4 = SingletonConfig().config['4_spawn_rate']
    bm = SingletonBoardMover(2)
    _, pattern_check_func, to_find_func, base_pattern, ini = formation_info[pattern]
    pattern_encoded = get_pattern_encoded(bm, target, base_pattern)
    ini += pattern_encoded
    gen_lookup_table_big(ini, pattern_check_func, to_find_func,
                         target, steps, pathname, bm, pattern_encoded, spawn_rate4)
    return True


def get_pattern_encoded(bm: BoardMoverWithScore, target: int, base_pattern: np.uint64):
    base_pattern = bm.decode_board(base_pattern)
    pattern = base_pattern * np.uint32(2 ** target / 8)
    return np.uint64(bm.encode_board(pattern))