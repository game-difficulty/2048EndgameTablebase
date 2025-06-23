from BoardMover import SingletonBoardMover
from BookBuilder import gen_lookup_table_big
from Config import SingletonConfig, formation_info
from Variants import vCalculator


def v_start_build(pattern: str, target: int, position: int, pathname: str) -> bool:
    bm = SingletonBoardMover(3)
    spawn_rate4 = SingletonConfig().config['4_spawn_rate']

    steps = int(2 ** target / 2 + {'2x4': 20, '3x3': 40, '3x4': 80, }[pattern])
    docheck_step = int(2 ** target / 2)
    _, pattern_check_func, to_find_func1, success_check_func, ini = formation_info[pattern]

    pattern_map = {
        '3x3': vCalculator.min33,
        '2x4': vCalculator.min24,
        '3x4': vCalculator.min34,
    }
    to_find_func = pattern_map.get(pattern)

    isfree = True
    gen_lookup_table_big(pattern, ini, pattern_check_func, success_check_func, to_find_func, target,
                         position, steps, pathname, docheck_step, bm, isfree, spawn_rate4)
    return True
