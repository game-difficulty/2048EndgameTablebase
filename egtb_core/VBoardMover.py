from __future__ import annotations

from typing import Tuple

import numpy as np

from egtb_core import mover_runtime


def _tile_to_log2(value: int) -> int:
    if value <= 0:
        return 0
    return int(value).bit_length() - 1


def reverse(board: np.uint64) -> np.uint64:
    return np.uint64(mover_runtime.reverse(board))


def encode_board(board: np.typing.NDArray) -> np.uint64:
    return np.uint64(mover_runtime.encode_board(board))


def decode_board(encoded_board: np.uint64) -> np.typing.NDArray:
    return np.asarray(mover_runtime.decode_board(encoded_board), dtype=np.int32)


def encode_row(row: np.typing.NDArray) -> np.uint64:
    encoded = np.uint64(0)
    values = np.asarray(row).reshape(-1).tolist()
    for index, num in enumerate(values[:4]):
        encoded |= np.uint64(_tile_to_log2(int(num))) << np.uint64(4 * (3 - index))
    return np.uint64(encoded)


def decode_row(encoded: np.uint64) -> np.typing.NDArray:
    row = np.empty(4, dtype=np.uint32)
    encoded = np.uint64(encoded)
    for index in range(4):
        value = int((encoded >> np.uint64(4 * (3 - index))) & np.uint64(0xF))
        row[index] = np.uint32(0 if value == 0 else (1 << value))
    return row


def merge_line_with_score(
    line: np.ndarray, reverse_line: bool = False
) -> Tuple[np.ndarray, np.uint32]:
    values = [int(value) for value in np.asarray(line).reshape(-1).tolist()]
    if reverse_line:
        values.reverse()

    segments = []
    current_segment = []
    for value in values:
        if value == 32768:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            segments.append([32768])
        else:
            current_segment.append(value)
    if current_segment:
        segments.append(current_segment)

    merged = []
    score = 0
    for segment in segments:
        if segment == [32768]:
            merged.append(32768)
            continue
        non_zero = [value for value in segment if value != 0]
        segment_result = []
        index = 0
        while index < len(non_zero):
            if (
                index + 1 < len(non_zero)
                and non_zero[index] == non_zero[index + 1]
                and non_zero[index] != 32768
            ):
                merged_value = non_zero[index] * 2
                score += merged_value
                segment_result.append(merged_value)
                index += 2
            else:
                segment_result.append(non_zero[index])
                index += 1
        segment_result.extend([0] * (len(segment) - len(segment_result)))
        merged.extend(segment_result)

    if reverse_line:
        merged.reverse()

    return np.asarray(merged, dtype=np.int32), np.uint32(score)


movel = None
mover = None
moveu = None
moved = None
score = None


def calculate_all_moves() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return np.uint64(mover_runtime.v.move_left(board))


def move_right(board: np.uint64) -> np.uint64:
    return np.uint64(mover_runtime.v.move_right(board))


def move_up(board: np.uint64, board2: np.uint64 | None = None) -> np.uint64:
    return np.uint64(mover_runtime.v.move_up(board))


def move_down(board: np.uint64, board2: np.uint64 | None = None) -> np.uint64:
    return np.uint64(mover_runtime.v.move_down(board))


def move_board(board: np.uint64, direction: int) -> np.uint64:
    return np.uint64(mover_runtime.v.move_board(board, direction))


def move_all_dir(board: np.uint64) -> Tuple[np.uint64, np.uint64, np.uint64, np.uint64]:
    return tuple(np.uint64(value) for value in mover_runtime.v.move_all_dir(board))


def gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int]:
    state, count = mover_runtime.gen_new_num(t, p)
    return np.uint64(state), int(count)


def s_move_left(board: np.uint64) -> Tuple[np.uint64, np.uint64]:
    state, move_score = mover_runtime.v.s_move_left(board)
    return np.uint64(state), np.uint64(move_score)


def s_move_right(board: np.uint64) -> Tuple[np.uint64, np.uint64]:
    state, move_score = mover_runtime.v.s_move_right(board)
    return np.uint64(state), np.uint64(move_score)


def s_move_up(board: np.uint64, board2: np.uint64 | None = None) -> Tuple[np.uint64, np.uint64]:
    state, move_score = mover_runtime.v.s_move_up(board)
    return np.uint64(state), np.uint64(move_score)


def s_move_down(board: np.uint64, board2: np.uint64 | None = None) -> Tuple[np.uint64, np.uint64]:
    state, move_score = mover_runtime.v.s_move_down(board)
    return np.uint64(state), np.uint64(move_score)


def s_move_board(board: np.uint64, direction: np.uint8) -> Tuple[np.uint64, np.uint64]:
    state, move_score = mover_runtime.v.s_move_board(board, int(direction))
    return np.uint64(state), np.uint64(move_score)


def s_gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int, int, int]:
    state, count, pos, value = mover_runtime.s_gen_new_num(t, p)
    return np.uint64(state), int(count), int(pos), int(value)
