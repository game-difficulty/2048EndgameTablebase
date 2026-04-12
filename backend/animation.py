from __future__ import annotations

import traceback
from typing import Any

from egtb_core.Calculator import find_merge_positions, slide_distance
from egtb_core.VBoardMover import decode_board


AnimationMetadata = dict[str, Any]


def compute_move_animation(
    board_encoded: int,
    direction_str: str,
    error_prefix: str | None = None,
    trace: bool = False,
) -> tuple[list[int], list[int]]:
    try:
        board_2d = decode_board(board_encoded)
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
