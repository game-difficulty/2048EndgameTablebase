from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from egtb_core.Calculator import find_merge_positions, slide_distance
from egtb_core.VBoardMover import decode_board
from .session import u64


AnimationMetadata = dict[str, Any]


def _normalize_board_for_animation(
    board_2d: np.ndarray,
    use_variant: bool,
) -> np.ndarray:
    if not use_variant:
        return board_2d
    normalized = np.array(board_2d, copy=True)
    normalized[normalized == 32768] = -1
    return normalized


def compute_move_animation(
    board_encoded: int,
    direction_str: str,
    *,
    use_variant: bool = False,
    error_prefix: str | None = None,
    trace: bool = False,
) -> tuple[list[int], list[int]]:
    try:
        board_2d = decode_board(np.uint64(u64(board_encoded)))
        board_2d = _normalize_board_for_animation(board_2d, use_variant)
        distances = slide_distance(board_2d, direction_str).flatten().tolist()
        merges = find_merge_positions(board_2d, direction_str).flatten().tolist()
        return distances, merges
    except Exception as exc:
        if error_prefix:
            print(f"{error_prefix}: {exc}")
        if trace:
            traceback.print_exc()
        return [], []


def build_move_animation_metadata(
    direction_str: str,
    *,
    board_encoded: int | None = None,
    use_variant: bool = False,
    spawn_index: int | None = None,
    spawn_value: int | None = None,
    distances: list[int] | None = None,
    merges: list[int] | None = None,
    error_prefix: str | None = None,
    trace: bool = False,
) -> AnimationMetadata:
    if distances is None or merges is None:
        if board_encoded is None:
            distances = distances or []
            merges = merges or []
        else:
            distances, merges = compute_move_animation(
                board_encoded,
                direction_str,
                use_variant=use_variant,
                error_prefix=error_prefix,
                trace=trace,
            )

    metadata = {
        "direction": direction_str,
        "slide_distances": distances,
        "pop_positions": merges,
    }
    if spawn_index is not None and spawn_value is not None:
        metadata["appear_tile"] = {
            "index": int(spawn_index),
            "value": int(spawn_value),
        }
    return metadata
