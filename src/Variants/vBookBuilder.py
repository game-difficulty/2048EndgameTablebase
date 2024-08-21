import numpy as np

from BoardMover import SingletonBoardMover
from Config import SingletonConfig
from BookBuilder import gen_lookup_table_big
from Variants import vCalculator


def v_start_build(pattern: str, target: int, position: int, pathname: str) -> bool:
    bm = SingletonBoardMover(3)
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']

    steps = int(2 ** target / 2 + {'2x4': 20, '3x3': 40, '3x4': 80, }[pattern])
    docheck_step = int(2 ** target / 2)
    inits = {
        '2x4': np.array([np.uint64(0xffff00000000ffff)], dtype=np.uint64),
        '3x3': np.array([np.uint64(0x000f000f000fffff)], dtype=np.uint64),
        '3x4': np.array([np.uint64(0x000000000000ffff)], dtype=np.uint64),
    }
    ini = inits[pattern]

    pattern_map = {
        '3x3': (vCalculator.p_min33, vCalculator.min33),
        '2x4': (vCalculator.p_min24, vCalculator.min24),
        '3x4': (vCalculator.p_min34, vCalculator.min34)
    }
    to_find_func, to_find_func1 = pattern_map.get(pattern)

    isfree = True
    gen_lookup_table_big(ini, vCalculator.is_variant_pattern, eval(f'vCalculator.is_{pattern}_success'),
                         to_find_func, to_find_func1, target, position, steps, pathname, docheck_step, bm, isfree,
                         spawn_rate4)
    return True
