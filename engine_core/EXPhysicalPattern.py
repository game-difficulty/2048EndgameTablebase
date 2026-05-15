from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from Config import pattern_32k_tiles_map, pattern_catalog
from engine_core import mover_runtime

try:
    from native_core import formation_core
except Exception:
    formation_core = None


_INVERSE_TRANSFORM = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 7,
    7: 6,
}

_ALLOWED_TRANSFORMS = {
    "full": tuple(range(8)),
    "identity": tuple(range(8)),
    "horizontal": (0, 1, 2, 5),
    "diagonal": (0, 3, 4, 5),
    "min24": (0,),
    "min33": (0,),
    "min34": (0,),
}


@dataclass(frozen=True)
class PhysicalPatternResolution:
    transform_id: int
    inverse_transform_id: int
    physical_canonical_mode: str
    pattern_masks: list[int]
    success_shifts: list[int]
    fixed_32k_shifts: list[int]
    initial_boards: np.ndarray
    logical_pattern_signature: int
    physical_pattern_signature: int
    score: tuple[int, ...]


def _require_native() -> None:
    if formation_core is None:
        raise RuntimeError("formation_core is unavailable")


def _apply_transform(value: int, transform_id: int) -> int:
    _require_native()
    return int(formation_core.apply_sym_like(int(value) & 0xFFFFFFFFFFFFFFFF, int(transform_id)))


def _transform_shift(shift: int, transform_id: int) -> int:
    shift = int(shift)
    if shift < 0 or shift > 60 or (shift % 4) != 0:
        raise ValueError(f"invalid 2048 board nibble shift: {shift}")
    transformed = _apply_transform(0xF << shift, transform_id)
    positions = [pos for pos in range(16) if ((transformed >> (4 * pos)) & 0xF) != 0]
    if len(positions) != 1 or ((transformed >> (4 * positions[0])) & 0xF) != 0xF:
        raise ValueError(f"shift transform did not produce one full nibble: {shift}")
    return positions[0] * 4


def _suffix_full_nibbles(mask: int) -> int:
    return sum(1 for pos in range(7) if ((int(mask) >> (4 * pos)) & 0xF) == 0xF)


def _suffix_shift_count(shifts: Iterable[int]) -> int:
    return sum(1 for shift in shifts if 0 <= int(shift) // 4 <= 6)


def _large_initial_suffix_count(boards: Sequence[int] | np.ndarray, target_exponent: int) -> int:
    count = 0
    for board in np.asarray(boards, dtype=np.uint64):
        value = int(board)
        for pos in range(7):
            tile = (value >> (4 * pos)) & 0xF
            if tile > int(target_exponent) and tile != 15:
                count += 1
    return count


def _variant_wall_suffix_count(boards: Sequence[int] | np.ndarray) -> int:
    count = 0
    for board in np.asarray(boards, dtype=np.uint64):
        value = int(board)
        for pos in range(7):
            if ((value >> (4 * pos)) & 0xF) == 0xF:
                count += 1
    return count


def _fnv1a64_update(value: int, item: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    item &= 0xFFFFFFFFFFFFFFFF
    for shift in range(0, 64, 8):
        value ^= (item >> shift) & 0xFF
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _signature(
    pattern: str,
    canonical_mode: str,
    pattern_masks: Sequence[int],
    success_shifts: Sequence[int],
    fixed_32k_shifts: Sequence[int],
    target_exponent: int,
    num_free_32k: int,
    small_tile_sum_limit: int,
    transform_id: int,
) -> int:
    h = 0xCBF29CE484222325
    for byte in pattern.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for byte in canonical_mode.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for item in (
        int(transform_id),
        int(target_exponent),
        int(num_free_32k),
        int(small_tile_sum_limit),
        len(pattern_masks),
    ):
        h = _fnv1a64_update(h, item)
    for mask in pattern_masks:
        h = _fnv1a64_update(h, int(mask))
    h = _fnv1a64_update(h, len(success_shifts))
    for shift in success_shifts:
        h = _fnv1a64_update(h, int(shift))
    h = _fnv1a64_update(h, len(fixed_32k_shifts))
    for shift in fixed_32k_shifts:
        h = _fnv1a64_update(h, int(shift))
    return h


def _canonicalize_boards(boards: np.ndarray, canonical_mode: str) -> np.ndarray:
    mode = canonical_mode.lower()
    canonicalizer = {
        "full": mover_runtime.canonical_full,
        "diagonal": mover_runtime.canonical_diagonal,
        "horizontal": mover_runtime.canonical_horizontal,
        "min33": mover_runtime.canonical_min33,
        "min24": mover_runtime.canonical_min24,
        "min34": mover_runtime.canonical_min34,
        "min34top": mover_runtime.canonical_min34top,
    }.get(mode)
    if canonicalizer is None:
        return np.unique(np.asarray(boards, dtype=np.uint64))
    return np.unique(np.asarray([canonicalizer(np.uint64(board)) for board in boards], dtype=np.uint64))


def resolve_ex_physical_pattern(
    pattern: str,
    initial_boards: Sequence[int] | np.ndarray,
    target_exponent: int,
    small_tile_sum_limit: int = 96,
    advanced: bool = False,
) -> PhysicalPatternResolution:
    meta = pattern_catalog.get(pattern)
    if meta is None:
        raise KeyError(f"Unknown pattern: {pattern}")

    canonical_mode = str(meta.get("canonical_mode", "identity")).lower()
    candidates = _ALLOWED_TRANSFORMS.get(canonical_mode, (0,))
    use_min34_top_variant = (
        not advanced
        and canonical_mode == "min34"
        and str(meta.get("category", "")).lower() == "variant"
    )
    if use_min34_top_variant:
        candidates = (0, 2)
    logical_masks = [int(mask) for mask in meta.get("pattern_masks", ())]
    logical_success = [int(shift) for shift in meta.get("success_shifts", ())]
    if advanced:
        _, num_free_32k, pos_fixed_32k = pattern_32k_tiles_map[pattern]
        logical_fixed = [int(x) for x in np.asarray(pos_fixed_32k, dtype=np.uint8)]
        num_free = int(num_free_32k)
    else:
        logical_fixed = []
        num_free = 0

    boards = np.asarray(initial_boards, dtype=np.uint64)
    best: tuple[tuple[int, ...], int, str, list[int], list[int], list[int], np.ndarray] | None = None
    for transform_id in candidates:
        physical_canonical_mode = canonical_mode
        if use_min34_top_variant and transform_id == 2:
            physical_canonical_mode = "min34top"
        masks = [_apply_transform(mask, transform_id) for mask in logical_masks]
        success = [_transform_shift(shift, transform_id) for shift in logical_success]
        fixed = [_transform_shift(shift, transform_id) for shift in logical_fixed]
        transformed_boards = np.asarray(
            [_apply_transform(int(board), transform_id) for board in boards],
            dtype=np.uint64,
        )
        transformed_boards = _canonicalize_boards(transformed_boards, physical_canonical_mode)
        per_mask_suffix = [_suffix_full_nibbles(mask) for mask in masks] or [0]
        variant_wall_suffix = (
            _variant_wall_suffix_count(transformed_boards)
            if meta.get("category") == "variant"
            else 0
        )
        score = (
            sum(per_mask_suffix),
            max(per_mask_suffix),
            _suffix_shift_count(fixed),
            variant_wall_suffix,
            _suffix_shift_count(success),
            _large_initial_suffix_count(transformed_boards, target_exponent),
            int(transform_id),
        )
        candidate = (score, int(transform_id), physical_canonical_mode, masks, success, fixed, transformed_boards)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError(f"no EX physical transform candidate for pattern {pattern}")

    score, transform_id, physical_canonical_mode, masks, success, fixed, transformed_boards = best
    inverse_id = _INVERSE_TRANSFORM[transform_id]
    logical_signature = _signature(
        pattern,
        canonical_mode,
        logical_masks,
        logical_success,
        logical_fixed,
        target_exponent,
        num_free,
        small_tile_sum_limit,
        0,
    )
    physical_signature = _signature(
        pattern,
        physical_canonical_mode,
        masks,
        success,
        fixed,
        target_exponent,
        num_free,
        small_tile_sum_limit,
        transform_id,
    )
    return PhysicalPatternResolution(
        transform_id=transform_id,
        inverse_transform_id=inverse_id,
        physical_canonical_mode=physical_canonical_mode,
        pattern_masks=masks,
        success_shifts=success,
        fixed_32k_shifts=fixed,
        initial_boards=transformed_boards,
        logical_pattern_signature=logical_signature,
        physical_pattern_signature=physical_signature,
        score=score,
    )
