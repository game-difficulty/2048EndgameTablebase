import os
from itertools import combinations
from typing import Callable, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, from_dtype
from numba import types
from numba.typed import Dict

import egtb_core.Calculator as Calculator
from egtb_core.BoardMover import move_all_dir, decode_board
from Config import (
    SingletonConfig,
    formation_info,
    pattern_32k_tiles_map,
    DTYPE_CONFIG,
    category_info,
)
from egtb_core.BookSolver import (
    recalculate_process,
    remove_died,
    expand,
    keep_only_optimal_branches,
    _remove_died_numba,
    solver_record_dtype,
    write_solver_book,
)
from egtb_core.BookGenerator import generate_process
from egtb_core.BookGeneratorAD import generate_process_ad
from egtb_core.BoardMaskerAD import init_masker
from egtb_core.BookSolverAD import recalculate_process_ad
from egtb_core.BookSolverChunkedAD import recalculate_process_ad_c

try:
    from ai_and_sort import formation_core
except Exception:
    formation_core = None

PatternCheckFunc = Callable[[np.uint64], bool]
SuccessCheckFunc = Callable[[np.uint64, int], bool]
CanonicalFunc = Callable[[np.uint64], np.uint64]
SymFindFunc = Callable[[np.uint64], Tuple[np.uint64, int]]

ValidDType: TypeAlias = Union[
    type[np.uint32], type[np.uint64], type[np.float32], type[np.float64]
]
SuccessRateDType: TypeAlias = Union[np.uint32, np.uint64, np.float32, np.float64]


KeyType = types.int8
ValueType1 = NDArray[NDArray[SuccessRateDType]]  # type: ignore
BookDictType = Dict[KeyType, ValueType1]  # type: ignore
ValueType2 = types.uint64[:]
IndexDictType = Dict[KeyType, ValueType2]  # type: ignore


def _symm_mode_from_canonical_func(canonical_func: CanonicalFunc) -> int:
    mapping = {
        Calculator.canonical_identity: 0,
        Calculator.canonical_full: 1,
        Calculator.canonical_diagonal: 2,
        Calculator.canonical_horizontal: 3,
        Calculator.canonical_min33: 4,
        Calculator.canonical_min24: 5,
        Calculator.canonical_min34: 6,
    }
    return mapping.get(canonical_func, 0)


def _build_native_run_options(
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    isfree: bool,
    is_variant: bool,
    spawn_rate4: float,
):
    config = SingletonConfig().config
    options = formation_core.RunOptions()
    options.target = int(target)
    options.steps = int(steps)
    options.docheck_step = int(docheck_step)
    options.pathname = str(pathname)
    options.is_free = bool(isfree)
    options.is_variant = bool(is_variant)
    options.spawn_rate4 = float(spawn_rate4)
    options.success_rate_dtype = str(config.get("success_rate_dtype", "uint32"))
    options.deletion_threshold = float(config.get("deletion_threshold", 0.0))
    options.compress = bool(config.get("compress", False))
    options.compress_temp_files = bool(config.get("compress_temp_files", False))
    options.optimal_branch_only = bool(config.get("optimal_branch_only", False))
    options.chunked_solve = bool(config.get("chunked_solve", False))
    options.num_threads = int(max(4, min(32, os.cpu_count() or 2)))
    return options


def _build_native_pattern_spec(pattern: str, canonical_func: CanonicalFunc):
    pattern_masks, success_shifts = Calculator.PATTERN_DATA.get(pattern, ((), ()))
    pattern_spec = formation_core.PatternSpec()
    pattern_spec.name = pattern
    pattern_spec.pattern_masks = list(pattern_masks)
    pattern_spec.success_shifts = list(success_shifts)
    pattern_spec.symm_mode = _symm_mode_from_canonical_func(canonical_func)
    return pattern_spec


def _native_classic_enabled() -> bool:
    return formation_core is not None and hasattr(formation_core, "run_pattern_build")


def run_pattern_solve_ad_bridge(
    pattern: str,
    arr_init,
    symm_mode: int,
    num_free_32k: int,
    fixed_32k_shifts,
    target: int,
    steps: int,
    pathname: str,
    isfree: bool,
    spawn_rate4: float,
) -> None:
    arr_init = np.asarray(arr_init, dtype=np.uint64)
    _, pattern_check_func, canonical_func, _, _, _ = formation_info.get(
        pattern, [0, None, Calculator.canonical_identity, None, None, None]
    )
    if pattern_check_func is None:
        raise KeyError(f"Pattern {pattern} not found in formation_info")
    sym_func = {
        Calculator.canonical_identity: Calculator.canonical_identity_pair,
        Calculator.canonical_diagonal: Calculator.canonical_diagonal_pair,
        Calculator.canonical_full: Calculator.canonical_full_pair,
        Calculator.canonical_horizontal: Calculator.canonical_horizontal_pair,
    }.get(canonical_func, Calculator.canonical_identity_pair)

    permutation_dict, tiles_combinations_dict, param = init_masker(
        int(num_free_32k),
        int(target),
        np.asarray(fixed_32k_shifts, dtype=np.uint8),
    )
    b1, b2, i1, i2 = final_steps_ad(True, pathname, steps)

    if SingletonConfig().config["chunked_solve"]:
        recalculate_process_ad_c(
            arr_init,
            b1,
            b2,
            i1,
            i2,
            tiles_combinations_dict,
            permutation_dict,
            param,
            pattern_check_func,
            sym_func,
            steps,
            pathname,
            spawn_rate4,
        )
    else:
        recalculate_process_ad(
            arr_init,
            b1,
            b2,
            i1,
            i2,
            tiles_combinations_dict,
            permutation_dict,
            param,
            pattern_check_func,
            sym_func,
            steps,
            pathname,
            spawn_rate4,
        )


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
    is_variant = pattern in category_info.get("variant", [])
    # variant tables不支持进阶算法，用不到sym_func，随便返回一个

    sym_func = {
        Calculator.canonical_identity: Calculator.canonical_identity_pair,
        Calculator.canonical_diagonal: Calculator.canonical_diagonal_pair,
        Calculator.canonical_full: Calculator.canonical_full_pair,
        Calculator.canonical_horizontal: Calculator.canonical_horizontal_pair,
    }.get(canonical_func, Calculator.canonical_identity_pair)

    save_config_to_txt(pathname + "config.txt")
    native_classic_used = False
    if not SingletonConfig().config.get("advanced_algo", False):
        if _native_classic_enabled():
            pattern_spec = _build_native_pattern_spec(pattern, canonical_func)
            run_options = _build_native_run_options(
                target, steps, pathname, docheck_step, isfree, is_variant, spawn_rate4
            )
            formation_core.run_pattern_build(arr_init, pattern_spec, run_options)
            native_classic_used = True
        else:
            started, d0, d1 = generate_process(
                arr_init,
                pattern_check_func,
                success_check_func,
                canonical_func,
                target,
                steps,
                pathname,
                docheck_step,
                isfree,
                is_variant,
            )
            d0, d1 = final_steps(
                started, d0, d1, pathname, steps, success_check_func, target
            )
            recalculate_process(
                d0,
                d1,
                pattern_check_func,
                success_check_func,
                canonical_func,
                target,
                steps,
                pathname,
                docheck_step,
                spawn_rate4,
                is_variant,
            )  # 这里的最后的两个book d0,d1就是回算的d1,d2
    else:
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]
        if formation_core is not None and hasattr(
            formation_core, "run_pattern_build_ad"
        ):
            pattern_spec = formation_core.AdvancedPatternSpec()
            pattern_spec.name = pattern
            pattern_spec.pattern_masks = list(
                Calculator.PATTERN_DATA.get(pattern, ((), ()))[0]
            )
            pattern_spec.symm_mode = _symm_mode_from_canonical_func(canonical_func)
            pattern_spec.num_free_32k = int(num_free_32k)
            pattern_spec.fixed_32k_shifts = list(
                np.asarray(pos_fixed_32k, dtype=np.uint8)
            )
            pattern_spec.small_tile_sum_limit = SingletonConfig().config.get(
                "SmallTileSumLimit", 96
            )
            pattern_spec.target = int(target)

            run_options = _build_native_run_options(
                target, steps, pathname, docheck_step, isfree, is_variant, spawn_rate4
            )
            formation_core.run_pattern_build_ad(arr_init, pattern_spec, run_options)
        else:
            permutation_dict, tiles_combinations_dict, param = init_masker(
                num_free_32k, target, pos_fixed_32k
            )

            started, d0, d1 = generate_process_ad(
                arr_init,
                pattern_check_func,
                canonical_func,
                sym_func,
                steps,
                pathname,
                isfree,
                tiles_combinations_dict,
                param,
            )
            b1, b2, i1, i2 = final_steps_ad(started, pathname, steps)

            if SingletonConfig().config["chunked_solve"]:
                recalculate_process_ad_c(
                    arr_init,
                    b1,
                    b2,
                    i1,
                    i2,
                    tiles_combinations_dict,
                    permutation_dict,
                    param,
                    pattern_check_func,
                    sym_func,
                    steps,
                    pathname,
                    spawn_rate4,
                )
            else:
                recalculate_process_ad(
                    arr_init,
                    b1,
                    b2,
                    i1,
                    i2,
                    tiles_combinations_dict,
                    permutation_dict,
                    param,
                    pattern_check_func,
                    sym_func,
                    steps,
                    pathname,
                    spawn_rate4,
                )
    if SingletonConfig().config["optimal_branch_only"] and not native_classic_used:
        keep_only_optimal_branches(pattern_check_func, canonical_func, steps, pathname)


def save_config_to_txt(output_path):
    keys = [
        "compress",
        "optimal_branch_only",
        "compress_temp_files",
        "advanced_algo",
        "deletion_threshold",
        "4_spawn_rate",
        "success_rate_dtype",
    ]
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}: {str(SingletonConfig().config.get(key, '?'))}\n")


def final_steps(
    started: bool,
    d0: NDArray[np.uint64],
    d1: NDArray[np.uint64],
    pathname: str,
    steps: int,
    success_check_func: SuccessCheckFunc,
    target: int,
) -> Tuple[NDArray, NDArray]:
    if started:
        _, current_dtype, max_scale, zero_val = DTYPE_CONFIG[
            SingletonConfig().config.get("success_rate_dtype", "uint32")
        ]

        arr0 = np.empty(len(d0), dtype=solver_record_dtype(current_dtype))
        expanded_arr0 = expand(d0, arr0)
        d0 = final_situation_process(
            expanded_arr0, success_check_func, target, max_scale, zero_val
        )
        write_solver_book(pathname + str(steps - 2) + ".book", d0, current_dtype)

        arr0 = np.empty(len(d1), dtype=solver_record_dtype(current_dtype))
        expanded_arr0 = expand(d1, arr0)
        d1 = final_situation_process(
            expanded_arr0, success_check_func, target, max_scale, zero_val
        )
        write_solver_book(pathname + str(steps - 1) + ".book", d1, current_dtype)

    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return d0, d1


def final_steps_ad(started: bool, pathname: str, steps: int) -> Tuple:
    if started:
        template, current_dtype, max_scale, zero_val = DTYPE_CONFIG[
            SingletonConfig().config.get("success_rate_dtype", "uint32")
        ]
        b0, i0 = (
            Dict.empty(KeyType, from_dtype(current_dtype)[:, :]),
            Dict.empty(KeyType, ValueType2),
        )
        b1, i1 = (
            Dict.empty(KeyType, from_dtype(current_dtype)[:, :]),
            Dict.empty(KeyType, ValueType2),
        )
        os.makedirs(pathname + str(steps - 1) + "b", exist_ok=True)
        os.makedirs(pathname + str(steps - 2) + "b", exist_ok=True)
        if os.path.exists(pathname + str(steps - 2)):
            os.remove(pathname + str(steps - 2))
        return b0, b1, i0, i1
    if os.path.exists(pathname + str(steps - 2)):
        os.remove(pathname + str(steps - 2))
    return None, None, None, None


@njit(nogil=True, parallel=True, cache=True)
def final_situation_process(
    expanded_arr0: NDArray,
    success_check_func: SuccessCheckFunc,
    target: int,
    max_scale: int,
    zero_val: SuccessRateDType,
) -> NDArray:  # NDArray[[np.uint64, SuccessRateDType]]
    for i in prange(len(expanded_arr0)):
        if success_check_func(expanded_arr0[i][0], target):
            expanded_arr0[i][1] = max_scale
        else:
            expanded_arr0[i][1] = zero_val
    expanded_arr0 = _remove_died_numba(expanded_arr0, zero_val)
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
    spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
    (
        tile_sum,
        pattern_check_func,
        canonical_func,
        success_check_func,
        ini,
        extra_steps,
    ) = formation_info[pattern]
    tile_sum *= -1
    if pattern[:4] == "free":
        steps = int(2**target / 2 + extra_steps)
        docheck_step = int(2**target / 2) - tile_sum % int(2**target) // 2

        ini_decoded = decode_board(ini[0])
        n32k = np.sum(ini_decoded == 32768)
        extra_tile_sum = tile_sum - 32768 * n32k

        if extra_tile_sum <= (15 - n32k) * 2:
            arr_init = generate_free_inits(n32k, extra_tile_sum // 2)
        else:
            arr_init = ini

        gen_lookup_table_big(
            pattern,
            arr_init,
            pattern_check_func,
            success_check_func,
            canonical_func,
            target,
            steps,
            pathname,
            docheck_step,
            isfree=True,
            spawn_rate4=spawn_rate4,
        )
    else:
        steps = int(2**target / 2 + extra_steps)
        docheck_step = int(2**target / 2) - tile_sum % int(2**target) // 2

        isfree = (len(pattern_32k_tiles_map[pattern][2]) < 4) and (tile_sum < 180000)
        gen_lookup_table_big(
            pattern,
            ini,
            pattern_check_func,
            success_check_func,
            canonical_func,
            target,
            steps,
            pathname,
            docheck_step,
            isfree,
            spawn_rate4,
        )
    return True


def v_start_build(pattern: str, target: int, pathname: str) -> bool:
    spawn_rate4 = SingletonConfig().config["4_spawn_rate"]
    (
        tile_sum,
        pattern_check_func,
        canonical_func,
        success_check_func,
        ini,
        extra_steps,
    ) = formation_info[pattern]
    tile_sum *= -1
    steps = int(2**target / 2 + extra_steps)
    docheck_step = int(2**target / 2) - tile_sum % int(2**target) // 2

    isfree = True
    gen_lookup_table_big(
        pattern,
        ini,
        pattern_check_func,
        success_check_func,
        canonical_func,
        target,
        steps,
        pathname,
        docheck_step,
        isfree,
        spawn_rate4,
    )
    return True


if __name__ == "__main__":
    start_build("4421", 10, r"C:\2048_tables\test_old\4421_1024_")
    # start_build("free9", 9, r"C:\2048_tables\free9\free9_256_")
    pass
