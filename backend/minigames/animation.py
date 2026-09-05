from __future__ import annotations

from typing import Any

import numpy as np


def _trace_line_animation(line: np.ndarray) -> tuple[list[int], np.ndarray, list[int]]:
    """Trace sources to targets using the minigame mover's merge rules."""
    values = [int(value) for value in line.tolist()]
    new_line: list[int] = []
    distances = np.zeros_like(line)
    pops = [0] * len(values)
    segment_start = 0

    while segment_start < len(values):
        if values[segment_start] == -1:
            new_line.append(-1)
            segment_start += 1
            continue

        segment_end = segment_start
        while segment_end < len(values) and values[segment_end] != -1:
            segment_end += 1

        sources = [
            (index, values[index])
            for index in range(segment_start, segment_end)
            if values[index] != 0
        ]
        target = segment_start
        source_index = 0
        while source_index < len(sources):
            value = sources[source_index][1]
            group_end = source_index + 1
            result_value = value

            if value == -3:
                while group_end < len(sources) and sources[group_end][1] == -3:
                    group_end += 1
            elif value >= 0 and group_end < len(sources) and sources[group_end][1] == value:
                group_end += 1
                result_value = value + 1

            for source, _ in sources[source_index:group_end]:
                distances[source] = source - target
            if group_end - source_index > 1:
                pops[target] = 1

            new_line.append(result_value)
            target += 1
            source_index = group_end

        new_line.extend([0] * (segment_end - target))
        segment_start = segment_end

    return new_line, distances, pops


def compute_minigame_move_animation(
    board: np.ndarray,
    direction: str,
) -> tuple[list[int], list[int]]:
    move_direction = str(direction or "").lower()
    distances = np.zeros_like(board)
    merges = np.zeros_like(board)

    rows, cols = board.shape
    axis_length = rows if move_direction in {"left", "right"} else cols
    for index in range(axis_length):
        if move_direction in {"left", "right"}:
            line = board[index, :]
        else:
            line = board[:, index]

        process_line = line[::-1] if move_direction in {"down", "right"} else line
        _, line_distances, line_merges = _trace_line_animation(process_line)

        if move_direction in {"down", "right"}:
            line_distances = line_distances[::-1]
            line_merges = line_merges[::-1]

        if move_direction in {"left", "right"}:
            distances[index, :] = line_distances
            merges[index, :] = line_merges
        else:
            distances[:, index] = line_distances
            merges[:, index] = line_merges

    return distances.flatten().astype(int).tolist(), merges.flatten().astype(int).tolist()


def build_minigame_move_animation_metadata(
    board_before: np.ndarray,
    direction: str,
    *,
    spawn_index: int | None = None,
    spawn_value: int | None = None,
) -> dict[str, Any]:
    slide_distances, pop_positions = compute_minigame_move_animation(board_before, direction)
    metadata: dict[str, Any] = {
        "direction": direction,
        "slide_distances": slide_distances,
        "pop_positions": pop_positions,
    }
    if spawn_index is not None and spawn_value is not None and spawn_index >= 0:
        metadata["appear_tile"] = {
            "index": int(spawn_index),
            "value": int(spawn_value),
        }
    return metadata
