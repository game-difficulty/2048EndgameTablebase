from __future__ import annotations

import os
from itertools import combinations

import numpy as np

from Config import (
    RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH,
    SingletonConfig,
    category_info,
    logger,
    pattern_32k_tiles_map,
    pattern_catalog,
    write_runtime_deletion_threshold_signal,
)
from engine_core import mover_runtime
from engine_core.EXPhysicalPattern import PhysicalPatternResolution, resolve_ex_physical_pattern

try:
    from native_core import formation_core
except Exception:
    formation_core = None


_SYMM_MODE_BY_NAME = {
    "identity": formation_core.SymmMode.Identity if formation_core else 0,
    "full": formation_core.SymmMode.Full if formation_core else 1,
    "diagonal": formation_core.SymmMode.Diagonal if formation_core else 2,
    "horizontal": formation_core.SymmMode.Horizontal if formation_core else 3,
    "min33": formation_core.SymmMode.Min33 if formation_core else 4,
    "min24": formation_core.SymmMode.Min24 if formation_core else 5,
    "min34": formation_core.SymmMode.Min34 if formation_core else 6,
    "min34top": formation_core.SymmMode.Min34Top if formation_core else 7,
}


def _require_native_build() -> None:
    if formation_core is None:
        raise RuntimeError("formation_core is unavailable")


def _symm_mode_value(name: str) -> int:
    mode = _SYMM_MODE_BY_NAME.get(name, 0)
    return int(mode.value if hasattr(mode, "value") else mode)


def _build_native_run_options(
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    is_free: bool,
    is_variant: bool,
    spawn_rate4: float,
):
    _require_native_build()
    config = SingletonConfig().config
    options = formation_core.RunOptions()
    options.target = int(target)
    options.steps = int(steps)
    options.docheck_step = int(docheck_step)
    options.pathname = str(pathname)
    options.is_free = bool(is_free)
    options.is_variant = bool(is_variant)
    options.spawn_rate4 = float(spawn_rate4)
    options.success_rate_dtype = str(config.get("success_rate_dtype", "uint32"))
    options.deletion_threshold = write_runtime_deletion_threshold_signal(
        config.get("deletion_threshold", 0.0)
    )
    if hasattr(options, "deletion_threshold_signal_path"):
        options.deletion_threshold_signal_path = RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH
    options.compress = bool(config.get("compress", False))
    options.compress_temp_files = bool(config.get("compress_temp_files", False))
    options.optimal_branch_only = bool(config.get("optimal_branch_only", False))
    options.chunked_solve = bool(config.get("chunked_solve", False))
    options.direct_io = bool(config.get("direct_io", True))
    options.direct_io_queue_depth = int(config.get("direct_io_queue_depth", 16))
    options.direct_io_chunk_mib = int(config.get("direct_io_chunk_mib", 8))
    options.num_threads = int(max(4, min(32, os.cpu_count() or 2)))
    return options


def _apply_physical_fields(pattern_spec, resolution: PhysicalPatternResolution | None) -> None:
    if resolution is None:
        return
    pattern_spec.pattern_masks = list(resolution.pattern_masks)
    pattern_spec.success_shifts = list(resolution.success_shifts)
    pattern_spec.symm_mode = _symm_mode_value(resolution.physical_canonical_mode)
    pattern_spec.physical_transform = int(resolution.transform_id)
    pattern_spec.inverse_physical_transform = int(resolution.inverse_transform_id)
    pattern_spec.logical_pattern_signature = int(resolution.logical_pattern_signature)
    pattern_spec.physical_pattern_signature = int(resolution.physical_pattern_signature)


def _build_native_pattern_spec(pattern: str, resolution: PhysicalPatternResolution | None = None):
    _require_native_build()
    meta = pattern_catalog.get(pattern)
    if meta is None:
        raise KeyError(f"Unknown pattern: {pattern}")

    pattern_spec = formation_core.PatternSpec()
    pattern_spec.name = pattern
    pattern_spec.pattern_masks = list(meta.get("pattern_masks", ()))
    pattern_spec.success_shifts = list(meta.get("success_shifts", ()))
    pattern_spec.symm_mode = _symm_mode_value(meta.get("canonical_mode", "identity"))
    _apply_physical_fields(pattern_spec, resolution)
    return pattern_spec


def _build_native_advanced_pattern_spec(
    pattern: str,
    target: int,
    resolution: PhysicalPatternResolution | None = None,
):
    _require_native_build()
    meta = pattern_catalog.get(pattern)
    if meta is None:
        raise KeyError(f"Unknown pattern: {pattern}")

    _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]

    pattern_spec = formation_core.AdvancedPatternSpec()
    pattern_spec.name = pattern
    pattern_spec.pattern_masks = list(meta.get("pattern_masks", ()))
    pattern_spec.success_shifts = list(meta.get("success_shifts", ()))
    pattern_spec.symm_mode = _symm_mode_value(meta.get("canonical_mode", "identity"))
    pattern_spec.num_free_32k = int(num_free_32k)
    pattern_spec.fixed_32k_shifts = list(np.asarray(pos_fixed_32k, dtype=np.uint8))
    pattern_spec.small_tile_sum_limit = int(
        SingletonConfig().config.get("SmallTileSumLimit", 96)
    )
    pattern_spec.target = int(target)
    _apply_physical_fields(pattern_spec, resolution)
    if resolution is not None:
        pattern_spec.fixed_32k_shifts = list(resolution.fixed_32k_shifts)
    return pattern_spec


def _resolve_build_meta(pattern: str):
    meta = pattern_catalog.get(pattern)
    if meta is None:
        raise KeyError(f"Unknown pattern: {pattern}")

    tile_sum = -int(meta.get("nums_adjust", 0))
    seed_boards = np.asarray(meta.get("seed_boards", ()), dtype=np.uint64)
    if seed_boards.size == 0:
        raise ValueError(f"Pattern {pattern} has no seed boards")
    extra_steps = int(meta.get("extra_steps", 36))
    return meta, tile_sum, seed_boards, extra_steps


def _steps_and_docheck(tile_sum: int, target: int, extra_steps: int) -> tuple[int, int]:
    target_tile = int(2**target)
    steps = int(target_tile / 2 + extra_steps)
    docheck_step = int(target_tile / 2) - tile_sum % target_tile // 2
    return steps, docheck_step


def _path_exists_any(paths) -> bool:
    return any(os.path.exists(path) for path in paths)


def _count_existing_steps(pathname: str, steps: int, suffixes) -> int:
    return sum(
        1
        for step in range(max(0, int(steps)))
        if _path_exists_any(f"{pathname}{step}{suffix}" for suffix in suffixes)
    )


def _all_existing_steps(pathname: str, steps: int, suffixes) -> bool:
    if steps <= 0:
        return False
    return all(
        _path_exists_any(f"{pathname}{step}{suffix}" for suffix in suffixes)
        for step in range(int(steps))
    )


def _read_int_marker(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return int(file.read().strip().split()[0])
    except Exception:
        return None


def estimate_build_progress(pattern: str, target: int, pathname: str) -> tuple[int, int]:
    config = SingletonConfig().config
    meta, tile_sum, _seed_boards, extra_steps = _resolve_build_meta(pattern)
    steps, _docheck_step = _steps_and_docheck(tile_sum, target, extra_steps)
    use_ex_algo = bool(config.get("zmask_algo", False))
    use_ad_algo = bool(config.get("advanced_algo", False))
    use_exad_algo = use_ex_algo and use_ad_algo
    optimal = bool(config.get("optimal_branch_only", False)) and not use_ad_algo
    total = steps * (3 if optimal else 2)
    if steps <= 0:
        return 0, 0

    current = 0
    if use_exad_algo:
        solved_count = _count_existing_steps(pathname, steps, (".exadbook", ".exadzbook"))
        if solved_count:
            current = steps + solved_count
        else:
            current = _count_existing_steps(pathname, steps, (".exadtmp", ".exadtmp.7z"))
    elif use_ex_algo:
        if optimal and os.path.exists(pathname + "ex_optimal_complete"):
            current = total
        elif not optimal and _all_existing_steps(pathname, steps, (".exzbook",)):
            current = total
        else:
            solved_count = _count_existing_steps(pathname, steps, (".zbook", ".exzbook"))
            if solved_count:
                current = steps + solved_count
            else:
                current = _count_existing_steps(pathname, steps, (".exgen", ".exgen.7z"))
            if optimal:
                opt_step = _read_int_marker(pathname + "ex_optlayer")
                if opt_step is not None:
                    current = max(current, 2 * steps + opt_step + 1)
                elif solved_count >= steps:
                    current = max(current, 2 * steps)
    elif use_ad_algo:
        solved_count = _count_existing_steps(pathname, steps, ("b", ".z", "b.7z"))
        if solved_count:
            current = steps + solved_count
        else:
            current = _count_existing_steps(pathname, steps, ("", ".7z"))
    else:
        solved_count = _count_existing_steps(pathname, steps, (".book", ".z", ".book.7z"))
        if solved_count:
            current = steps + solved_count
        else:
            current = _count_existing_steps(pathname, steps, ("", ".7z"))
        if optimal:
            opt_step = _read_int_marker(pathname + "optlayer")
            if opt_step is not None:
                current = max(current, 2 * steps + opt_step + 1)

    return max(0, min(int(current), int(total))), int(total)


def save_config_to_txt(output_path: str) -> None:
    keys = [
        "compress",
        "optimal_branch_only",
        "compress_temp_files",
        "advanced_algo",
        "zmask_algo",
        "direct_io",
        "direct_io_queue_depth",
        "direct_io_chunk_mib",
        "deletion_threshold",
        "4_spawn_rate",
        "success_rate_dtype",
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        for key in keys:
            file.write(f"{key}: {str(SingletonConfig().config.get(key, '?'))}\n")


def append_ex_physical_config(output_path: str, resolution: PhysicalPatternResolution | None) -> None:
    if resolution is None:
        return
    with open(output_path, "a", encoding="utf-8") as file:
        file.write(f"ex_physical_transform: {resolution.transform_id}\n")
        file.write(f"ex_inverse_physical_transform: {resolution.inverse_transform_id}\n")
        file.write(f"ex_physical_canonical_mode: {resolution.physical_canonical_mode}\n")
        file.write(f"ex_logical_pattern_signature: {resolution.logical_pattern_signature}\n")
        file.write(f"ex_physical_pattern_signature: {resolution.physical_pattern_signature}\n")
        file.write(f"ex_physical_transform_score: {resolution.score}\n")


def _collect_canonical_successors(boards: np.ndarray) -> np.ndarray:
    canonicalized: list[np.uint64] = []

    for board in boards:
        for moved in mover_runtime.std.move_all_dir(np.uint64(board)):
            moved = np.uint64(moved)
            if moved == mover_runtime.canonical_full(moved):
                canonicalized.append(moved)

    if not canonicalized:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.asarray(canonicalized, dtype=np.uint64))


def _is_reachable_free_init(board: np.uint64) -> bool:
    decoded = mover_runtime.decode_board(np.uint64(board))
    corners = (decoded[0, 0], decoded[0, 3], decoded[3, 0], decoded[3, 3])
    return sum(tile > 4 for tile in corners) < 4


def generate_free_inits(t32ks: int, t2s: int) -> np.ndarray:
    max_estimated = 1_000_000
    generated = np.empty(max_estimated, dtype=np.uint64)
    count = 0

    for positions_32k in combinations(range(16), t32ks):
        board_base = np.uint64(0)
        for pos in positions_32k:
            board_base |= np.uint64(15 << (pos * 4))
        remain_pos = set(range(16)) - set(positions_32k)

        for positions_2 in combinations(remain_pos, t2s):
            if count >= max_estimated:
                break
            board = board_base
            for pos in positions_2:
                board |= np.uint64(1 << (pos * 4))
            generated[count] = board
            count += 1

    generated = np.unique(generated[:count])
    canonicalized_a = _collect_canonical_successors(generated)
    canonicalized_b = _collect_canonical_successors(canonicalized_a)

    if canonicalized_b.size > 0:
        canonicalized = np.unique(np.concatenate((canonicalized_a, canonicalized_b)))
    else:
        canonicalized = canonicalized_a

    reachable = [
        board for board in canonicalized if _is_reachable_free_init(np.uint64(board))
    ]
    return np.asarray(reachable, dtype=np.uint64)


def _run_classic_build(
    pattern: str,
    arr_init: np.ndarray,
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    is_free: bool,
    is_variant: bool,
    spawn_rate4: float,
) -> None:
    pattern_spec = _build_native_pattern_spec(pattern)
    run_options = _build_native_run_options(
        target,
        steps,
        pathname,
        docheck_step,
        is_free,
        is_variant,
        spawn_rate4,
    )
    formation_core.run_pattern_build(
        np.asarray(arr_init, dtype=np.uint64), pattern_spec, run_options
    )


def _run_advanced_build(
    pattern: str,
    arr_init: np.ndarray,
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    is_free: bool,
    is_variant: bool,
    spawn_rate4: float,
) -> None:
    pattern_spec = _build_native_advanced_pattern_spec(pattern, target)
    run_options = _build_native_run_options(
        target,
        steps,
        pathname,
        docheck_step,
        is_free,
        is_variant,
        spawn_rate4,
    )
    formation_core.run_pattern_build_ad(
        np.asarray(arr_init, dtype=np.uint64), pattern_spec, run_options
    )


def _run_zmask_build(
    pattern: str,
    arr_init: np.ndarray,
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    is_free: bool,
    is_variant: bool,
    spawn_rate4: float,
    resolution: PhysicalPatternResolution | None = None,
) -> None:
    pattern_spec = _build_native_pattern_spec(pattern, resolution)
    run_options = _build_native_run_options(
        target,
        steps,
        pathname,
        docheck_step,
        is_free,
        is_variant,
        spawn_rate4,
    )
    run_options.chunked_solve = False
    formation_core.run_pattern_build_zmask(
        np.asarray(arr_init, dtype=np.uint64), pattern_spec, run_options
    )


def _run_exad_build(
    pattern: str,
    arr_init: np.ndarray,
    target: int,
    steps: int,
    pathname: str,
    docheck_step: int,
    is_free: bool,
    is_variant: bool,
    spawn_rate4: float,
    resolution: PhysicalPatternResolution | None = None,
) -> None:
    pattern_spec = _build_native_advanced_pattern_spec(pattern, target, resolution)
    run_options = _build_native_run_options(
        target,
        steps,
        pathname,
        docheck_step,
        is_free,
        is_variant,
        spawn_rate4,
    )
    formation_core.run_pattern_build_exad(
        np.asarray(arr_init, dtype=np.uint64), pattern_spec, run_options
    )


def _should_retry_build_resume(exc: Exception) -> bool:
    message = str(exc).strip()
    if not message:
        return False
    if message == "The length multiplier is not big enough. Please restart.":
        return True
    return message.startswith("length multiplier ")


def _run_with_single_resume_retry(build_label: str, build_fn) -> None:
    retried = False
    while True:
        try:
            build_fn()
            return
        except Exception as exc:
            if retried or not _should_retry_build_resume(exc):
                raise
            retried = True
            logger.warning(
                "%s interrupted by validate_length_and_balance (%s); "
                "re-entering build once via the same Python entrypoint.",
                build_label,
                exc,
            )


def start_build(pattern: str, target: int, pathname: str) -> bool:
    _require_native_build()
    config = SingletonConfig().config
    spawn_rate4 = float(config["4_spawn_rate"])
    meta, tile_sum, seed_boards, extra_steps = _resolve_build_meta(pattern)
    steps, docheck_step = _steps_and_docheck(tile_sum, target, extra_steps)
    is_variant = pattern in category_info.get("variant", [])
    use_ex_algo = bool(config.get("zmask_algo", False))
    use_ad_algo = bool(config.get("advanced_algo", False))
    use_exad_algo = use_ex_algo and use_ad_algo
    save_config_to_txt(pathname + "config.txt")

    if pattern.startswith("free"):
        decoded_seed = mover_runtime.decode_board(seed_boards[0])
        num_32k = int(np.sum(decoded_seed == 32768))
        extra_tile_sum = tile_sum - 32768 * num_32k
        if extra_tile_sum <= (15 - num_32k) * 2:
            arr_init = generate_free_inits(num_32k, extra_tile_sum // 2)
        else:
            arr_init = seed_boards
        is_free = True
    else:
        arr_init = seed_boards
        fixed_positions = pattern_32k_tiles_map[pattern][2]
        is_free = (len(fixed_positions) < 4) and (tile_sum < 180000)

    ex_resolution = None
    if use_ex_algo:
        ex_resolution = resolve_ex_physical_pattern(
            pattern,
            arr_init,
            target,
            int(config.get("SmallTileSumLimit", 96)),
            advanced=use_exad_algo,
        )
        arr_init = ex_resolution.initial_boards
        append_ex_physical_config(pathname + "config.txt", ex_resolution)

    if use_exad_algo:
        _run_with_single_resume_retry(
            f"EXAD build {pattern}_{2**target}",
            lambda: _run_exad_build(
                pattern,
                arr_init,
                target,
                steps,
                pathname,
                docheck_step,
                is_free,
                is_variant,
                spawn_rate4,
                ex_resolution,
            ),
        )
    elif use_ex_algo:
        _run_with_single_resume_retry(
            f"EX build {pattern}_{2**target}",
            lambda: _run_zmask_build(
                pattern,
                arr_init,
                target,
                steps,
                pathname,
                docheck_step,
                is_free,
                is_variant,
                spawn_rate4,
                ex_resolution,
            ),
        )
    elif use_ad_algo:
        _run_with_single_resume_retry(
            f"Advanced build {pattern}_{2**target}",
            lambda: _run_advanced_build(
                pattern,
                arr_init,
                target,
                steps,
                pathname,
                docheck_step,
                is_free,
                is_variant,
                spawn_rate4,
            ),
        )
    else:
        _run_with_single_resume_retry(
            f"Classic build {pattern}_{2**target}",
            lambda: _run_classic_build(
                pattern,
                arr_init,
                target,
                steps,
                pathname,
                docheck_step,
                is_free,
                is_variant,
                spawn_rate4,
            ),
        )
    return True


def v_start_build(pattern: str, target: int, pathname: str) -> bool:
    _require_native_build()
    config = SingletonConfig().config
    spawn_rate4 = float(config["4_spawn_rate"])
    meta, tile_sum, seed_boards, extra_steps = _resolve_build_meta(pattern)
    steps, docheck_step = _steps_and_docheck(tile_sum, target, extra_steps)
    use_ex_algo = bool(config.get("zmask_algo", False))
    use_ad_algo = bool(config.get("advanced_algo", False))
    use_exad_algo = use_ex_algo and use_ad_algo
    if use_ad_algo:
        raise ValueError("Variant patterns do not support advanced or EXAD algorithms; use classic or EX.")
    save_config_to_txt(pathname + "config.txt")
    ex_resolution = None
    seed_boards_for_build = seed_boards
    if use_ex_algo:
        ex_resolution = resolve_ex_physical_pattern(
            pattern,
            seed_boards,
            target,
            int(config.get("SmallTileSumLimit", 96)),
            advanced=use_exad_algo,
        )
        seed_boards_for_build = ex_resolution.initial_boards
        append_ex_physical_config(pathname + "config.txt", ex_resolution)
    if use_exad_algo:
        _run_with_single_resume_retry(
            f"Variant EXAD build {pattern}_{2**target}",
            lambda: _run_exad_build(
                pattern,
                seed_boards_for_build,
                target,
                steps,
                pathname,
                docheck_step,
                True,
                True,
                spawn_rate4,
                ex_resolution,
            ),
        )
    elif use_ex_algo:
        _run_with_single_resume_retry(
            f"Variant EX build {pattern}_{2**target}",
            lambda: _run_zmask_build(
                pattern,
                seed_boards_for_build,
                target,
                steps,
                pathname,
                docheck_step,
                True,
                True,
                spawn_rate4,
                ex_resolution,
            ),
        )
    elif use_ad_algo:
        _run_with_single_resume_retry(
            f"Variant advanced build {pattern}_{2**target}",
            lambda: _run_advanced_build(
                pattern,
                seed_boards_for_build,
                target,
                steps,
                pathname,
                docheck_step,
                True,
                True,
                spawn_rate4,
            ),
        )
    else:
        _run_with_single_resume_retry(
            f"Variant build {pattern}_{2**target}",
            lambda: _run_classic_build(
                pattern,
                seed_boards_for_build,
                target,
                steps,
                pathname,
                docheck_step,
                True,
                True,
                spawn_rate4,
            ),
        )
    return True


if __name__ == "__main__":
    # $env:PYTHONPATH = "."
    start_build('free10',9,r"C:\2048_tables\free10\free10_512_")
    pass
