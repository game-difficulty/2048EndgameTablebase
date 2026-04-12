from __future__ import annotations

from typing import Any

import numpy as np


def _simulate_line(line: np.ndarray) -> tuple[list[int], list[int]]:
    merged: list[int] = []
    new_line: list[int] = []

    segments: list[list[int]] = []
    current_segment: list[int] = []
    for value in line.tolist():
        if value == -1:
            if current_segment:
                segments.append(current_segment)
            segments.append([-1])
            current_segment = []
        else:
            current_segment.append(int(value))
    if current_segment:
        segments.append(current_segment)

    for segment in segments:
        if segment == [-1]:
            new_line.append(-1)
            merged.append(0)
            continue
        skip = False
        skip_special = False
        non_zero = [value for value in segment if value != 0]
        temp_merged: list[int] = []
        local_merge = [0] * len(segment)
        for idx in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if skip_special:
                if non_zero[idx] == -3:
                    continue
                skip_special = False
            if idx + 1 < len(non_zero) and non_zero[idx] == non_zero[idx + 1]:
                if non_zero[idx] >= 0:
                    temp_merged.append(non_zero[idx] + 1)
                    local_merge[len(temp_merged) - 1] = 1
                    skip = True
                    continue
                if non_zero[idx] == -3:
                    temp_merged.append(non_zero[idx])
                    local_merge[len(temp_merged) - 1] = 1
                    skip_special = True
                    continue
            temp_merged.append(non_zero[idx])

        temp_merged.extend([0] * (len(segment) - len(temp_merged)))
        new_line.extend(temp_merged[: len(segment)])
        merged.extend(local_merge[: len(segment)])

    new_line.extend([0] * (len(line) - len(new_line)))
    merged.extend([0] * (len(line) - len(merged)))
    return new_line[: len(line)], merged[: len(line)]


def _move_distance_line(line: np.ndarray) -> np.ndarray:
    moved_distance = 0
    last_tile = 0
    move_distance = np.zeros_like(line)

    for index, value in enumerate(line):
        current = int(value)
        if current == 0:
            moved_distance += 1
        elif current == -1:
            moved_distance = 0
            last_tile = 0
        elif current == -2:
            last_tile = 0
        elif last_tile == current and current >= 0:
            move_distance[index] = moved_distance + 1
            moved_distance += 1
            last_tile = 0
        else:
            move_distance[index] = moved_distance
            last_tile = current
    return move_distance


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
        line_distances = _move_distance_line(process_line)
        _, line_merges = _simulate_line(process_line)

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
    if spawn_index is not None and spawn_value is not None and spawn_index >= 0 and spawn_value > 0:
        metadata["appear_tile"] = {
            "index": int(spawn_index),
            "value": int(spawn_value),
        }
    return metadata
