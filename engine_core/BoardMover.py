from __future__ import annotations

from typing import Tuple

import numpy as np

from engine_core import mover_runtime
from engine_core.VBoardMover import decode_board, decode_row, encode_board, encode_row, reverse


def merge_line_with_score(
    line: np.typing.NDArray, reverse_line: bool = False
) -> Tuple[np.typing.NDArray, np.uint64]:
    values = [int(value) for value in np.asarray(line).reshape(-1).tolist()]
    if reverse_line:
        values.reverse()

    non_zero = [value for value in values if value != 0]
    merged = []
    score = 0
    index = 0
    while index < len(non_zero):
        if index + 1 < len(non_zero) and non_zero[index] == non_zero[index + 1] and non_zero[index] != 32768:
            merged_value = non_zero[index] * 2
            score += merged_value
            merged.append(merged_value)
            index += 2
        else:
            merged.append(non_zero[index])
            index += 1

    merged.extend([0] * (len(values) - len(merged)))
    if reverse_line:
        merged.reverse()
    return np.asarray(merged, dtype=np.int32), np.uint64(score)


movel = None
mover = None
moveu = None
moved = None
score = None


def calculate_all_moves() -> Tuple[
    np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray
]:
    global movel, mover, moveu, moved, score
    if movel is not None:
        return movel, mover, moveu, moved, score

    movel = np.empty(65536, dtype=np.uint64)
    mover = np.empty(65536, dtype=np.uint64)
    moveu = np.empty(65536, dtype=np.uint64)
    moved = np.empty(65536, dtype=np.uint64)
    score = np.empty(65536, dtype=np.uint64)

    for raw in range(16 ** 4):
        line = np.array(
            [2 ** value if value else 0 for value in [(raw // (16 ** j)) % 16 for j in range(4)]],
            dtype=np.int32,
        )
        original_line = encode_row(line)
        merged_left, _ = merge_line_with_score(line, False)
        merged_right, right_score = merge_line_with_score(line, True)
        movel[original_line] = encode_row(merged_left) ^ original_line
        mover[original_line] = encode_row(merged_right) ^ original_line
        score[original_line] = np.uint64(right_score)

    for raw in range(16 ** 4):
        moveu[raw] = reverse(movel[raw])
        moved[raw] = reverse(mover[raw])

    return movel, mover, moveu, moved, score


def move_left(board: np.uint64) -> np.uint64:
    return np.uint64(mover_runtime.std.move_left(board))


def move_right(board: np.uint64) -> np.uint64:
    return np.uint64(mover_runtime.std.move_right(board))


def move_up(board: np.uint64, board2: np.uint64 | None = None) -> np.uint64:
    return np.uint64(mover_runtime.std.move_up(board))


def move_down(board: np.uint64, board2: np.uint64 | None = None) -> np.uint64:
    return np.uint64(mover_runtime.std.move_down(board))


def move_board(board: np.uint64, direction: int) -> np.uint64:
    return np.uint64(mover_runtime.std.move_board(board, direction))


def move_board2(board: np.uint64, board2: np.uint64, direction: int) -> np.uint64:
    return np.uint64(mover_runtime.std.move_board(board, direction))


def move_all_dir(board: np.uint64) -> Tuple[np.uint64, np.uint64, np.uint64, np.uint64]:
    return tuple(np.uint64(value) for value in mover_runtime.std.move_all_dir(board))


def gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int]:
    state, count = mover_runtime.gen_new_num(t, p)
    return np.uint64(state), int(count)


def s_move_left(board: np.uint64) -> Tuple[np.uint64, np.uint32]:
    state, move_score = mover_runtime.std.s_move_left(board)
    return np.uint64(state), np.uint32(move_score)


def s_move_right(board: np.uint64) -> Tuple[np.uint64, np.uint32]:
    state, move_score = mover_runtime.std.s_move_right(board)
    return np.uint64(state), np.uint32(move_score)


def s_move_up(board: np.uint64, board2: np.uint64 | None = None) -> Tuple[np.uint64, np.uint32]:
    state, move_score = mover_runtime.std.s_move_up(board)
    return np.uint64(state), np.uint32(move_score)


def s_move_down(board: np.uint64, board2: np.uint64 | None = None) -> Tuple[np.uint64, np.uint32]:
    state, move_score = mover_runtime.std.s_move_down(board)
    return np.uint64(state), np.uint32(move_score)


def s_move_board(board: np.uint64, direction: int) -> Tuple[np.uint64, np.uint32]:
    state, move_score = mover_runtime.std.s_move_board(board, direction)
    return np.uint64(state), np.uint32(move_score)


def s_move_all_dir(board: np.uint64):
    return (
        s_move_left(board),
        s_move_right(board),
        s_move_up(board),
        s_move_down(board),
    )


def s_gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int, int, int]:
    state, count, pos, value = mover_runtime.s_gen_new_num(t, p)
    return np.uint64(state), int(count), int(pos), int(value)
