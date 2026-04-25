from __future__ import annotations

from typing import List, Tuple

import numpy as np

from egtb_core import mover_runtime


PATTERN_DATA = {}


def create_pattern_func_source(name, masks):
    func_name = f"is_{name}_pattern"
    if not masks:
        body = "True"
    else:
        conditions = []
        for mask in masks:
            mask_value = f"np.uint64({mask})"
            conditions.append(f"(np.uint64(encoded_board) & {mask_value}) == {mask_value}")
        body = " or \\\n           ".join(conditions)
    source = f"def {func_name}(encoded_board):\n    return {body}"
    return func_name, source


def create_success_func_source(name, shifts):
    func_name = f"is_{name}_success"
    if not shifts:
        body = "True"
    else:
        conditions = []
        for shift in shifts:
            shift_value = f"np.uint64({shift})"
            conditions.append(
                f"((np.uint64(encoded_board) >> {shift_value}) & np.uint64(0xf)) == np.uint64(target)"
            )
        body = " or \\\n           ".join(conditions)
    source = f"def {func_name}(encoded_board, target):\n    return {body}"
    return func_name, source


def _make_pattern_func(func_name, masks):
    masks_u64 = tuple(np.uint64(mask) for mask in masks)

    if not masks_u64:
        def _func(encoded_board):
            return True
    else:
        def _func(encoded_board):
            board = np.uint64(encoded_board)
            return any((board & mask) == mask for mask in masks_u64)

    _func.__name__ = func_name
    return _func


def _make_success_func(func_name, shifts):
    shifts_u64 = tuple(np.uint64(shift) for shift in shifts)

    if not shifts_u64:
        def _func(encoded_board, target):
            return True
    else:
        def _func(encoded_board, target):
            board = np.uint64(encoded_board)
            target_u64 = np.uint64(target)
            return any(((board >> shift) & np.uint64(0xF)) == target_u64 for shift in shifts_u64)

    _func.__name__ = func_name
    return _func


def update_logic_functions():
    for name, (masks, shifts) in PATTERN_DATA.items():
        globals()[f"is_{name}_pattern"] = _make_pattern_func(f"is_{name}_pattern", masks)
        globals()[f"is_{name}_success"] = _make_success_func(f"is_{name}_success", shifts)


def is_success(board_encoded, target_stacked, mask=np.uint64(0xFFFFFFFFFFFFFFFF)):
    board_encoded = np.uint64(board_encoded) & np.uint64(mask)
    diff = board_encoded ^ np.uint64(target_stacked)
    mask_7 = np.uint64(0x7777777777777777)
    result = ~(((diff & mask_7) + mask_7) | diff | mask_7)
    return bool(result != 0)


def ReverseLR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFF00FF00FF00FF00)) >> np.uint64(8) | (
        board & np.uint64(0x00FF00FF00FF00FF)
    ) << np.uint64(8)
    board = (board & np.uint64(0xF0F0F0F0F0F0F0F0)) >> np.uint64(4) | (
        board & np.uint64(0x0F0F0F0F0F0F0F0F)
    ) << np.uint64(4)
    return np.uint64(board)


def ReverseUD(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFFFFFFFF00000000)) >> np.uint64(32) | (
        board & np.uint64(0x00000000FFFFFFFF)
    ) << np.uint64(32)
    board = (board & np.uint64(0xFFFF0000FFFF0000)) >> np.uint64(16) | (
        board & np.uint64(0x0000FFFF0000FFFF)
    ) << np.uint64(16)
    return np.uint64(board)


def ReverseUL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFF00FF0000FF00FF)) | (
        (board & np.uint64(0x00FF00FF00000000)) >> np.uint64(24)
    ) | ((board & np.uint64(0x00000000FF00FF00)) << np.uint64(24))
    board = (board & np.uint64(0xF0F00F0FF0F00F0F)) | (
        (board & np.uint64(0x0F0F00000F0F0000)) >> np.uint64(12)
    ) | ((board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(12))
    return np.uint64(board)


def ReverseUR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0x0F0FF0F00F0FF0F0)) | (
        (board & np.uint64(0xF0F00000F0F00000)) >> np.uint64(20)
    ) | ((board & np.uint64(0x00000F0F00000F0F)) << np.uint64(20))
    board = (board & np.uint64(0x00FF00FFFF00FF00)) | (
        (board & np.uint64(0xFF00FF0000000000)) >> np.uint64(40)
    ) | ((board & np.uint64(0x0000000000FF00FF)) << np.uint64(40))
    return np.uint64(board)


def Rotate180(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFFFFFFFF00000000)) >> np.uint64(32) | (
        board & np.uint64(0x00000000FFFFFFFF)
    ) << np.uint64(32)
    board = (board & np.uint64(0xFFFF0000FFFF0000)) >> np.uint64(16) | (
        board & np.uint64(0x0000FFFF0000FFFF)
    ) << np.uint64(16)
    board = (board & np.uint64(0xFF00FF00FF00FF00)) >> np.uint64(8) | (
        board & np.uint64(0x00FF00FF00FF00FF)
    ) << np.uint64(8)
    board = (board & np.uint64(0xF0F0F0F0F0F0F0F0)) >> np.uint64(4) | (
        board & np.uint64(0x0F0F0F0F0F0F0F0F)
    ) << np.uint64(4)
    return np.uint64(board)


def RotateL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFF00FF0000000000)) >> np.uint64(32) | (
        board & np.uint64(0x00FF00FF00000000)
    ) << np.uint64(8) | (
        (board & np.uint64(0x00000000FF00FF00)) >> np.uint64(8)
    ) | ((board & np.uint64(0x0000000000FF00FF)) << np.uint64(32))
    board = (board & np.uint64(0xF0F00000F0F00000)) >> np.uint64(16) | (
        board & np.uint64(0x0F0F00000F0F0000)
    ) << np.uint64(4) | (
        (board & np.uint64(0x0000F0F00000F0F0)) >> np.uint64(4)
    ) | ((board & np.uint64(0x00000F0F00000F0F)) << np.uint64(16))
    return np.uint64(board)


def RotateR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xFF00FF0000000000)) >> np.uint64(8) | (
        board & np.uint64(0x00FF00FF00000000)
    ) >> np.uint64(32) | (
        (board & np.uint64(0x00000000FF00FF00)) << np.uint64(32)
    ) | ((board & np.uint64(0x0000000000FF00FF)) << np.uint64(8))
    board = (board & np.uint64(0xF0F00000F0F00000)) >> np.uint64(4) | (
        board & np.uint64(0x0F0F00000F0F0000)
    ) >> np.uint64(16) | (
        (board & np.uint64(0x0000F0F00000F0F0)) << np.uint64(16)
    ) | ((board & np.uint64(0x00000F0F00000F0F)) << np.uint64(4))
    return np.uint64(board)


def canonical_full(board):
    return np.uint64(mover_runtime.canonical_full(board))


def canonical_diagonal(board):
    return np.uint64(mover_runtime.canonical_diagonal(board))


def canonical_horizontal(board):
    return np.uint64(mover_runtime.canonical_horizontal(board))


def canonical_identity(encoded_board):
    return np.uint64(mover_runtime.canonical_identity(encoded_board))


def canonical_diagonal_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return mover_runtime.canonical_diagonal_pair(bd1)


def canonical_horizontal_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return mover_runtime.canonical_horizontal_pair(bd1)


def canonical_full_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return mover_runtime.canonical_full_pair(bd1)


def canonical_identity_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return mover_runtime.canonical_identity_pair(bd1)


def exchange_row12(board):
    board = np.uint64(board)
    return (board & np.uint64(0xFFFF00000000FFFF)) | ((board & np.uint64(0x00000000FFFF0000)) << np.uint64(16)) | (
        (board & np.uint64(0x0000FFFF00000000)) >> np.uint64(16)
    )


def exchange_row02(board):
    board = np.uint64(board)
    return (board & np.uint64(0x0000FFFF0000FFFF)) | ((board & np.uint64(0x00000000FFFF0000)) << np.uint64(32)) | (
        (board & np.uint64(0xFFFF000000000000)) >> np.uint64(32)
    )


def exchange_col02(board):
    board = np.uint64(board)
    return (board & np.uint64(0x0F0F0F0F0F0F0F0F)) | ((board & np.uint64(0xF000F000F000F000)) >> np.uint64(8)) | (
        (board & np.uint64(0x00F000F000F000F0)) << np.uint64(8)
    )


def R90_33(board):
    board = np.uint64(board)
    return (
        ((board & np.uint64(0xF000000000000000)) >> np.uint64(32))
        | ((board & np.uint64(0x0F00000000000000)) >> np.uint64(12))
        | ((board & np.uint64(0x00F0000000000000)) << np.uint64(8))
        | ((board & np.uint64(0x0000F00000000000)) >> np.uint64(20))
        | ((board & np.uint64(0x000000F000000000)) << np.uint64(20))
        | ((board & np.uint64(0x0000000F00000000)) >> np.uint64(8))
        | ((board & np.uint64(0x000000000F000000)) << np.uint64(12))
        | ((board & np.uint64(0x0000000000F00000)) << np.uint64(32))
        | (board & np.uint64(0x000F0F0F000FFFFF))
    )


def L90_33(board):
    board = np.uint64(board)
    return (
        ((board & np.uint64(0xF000000000000000)) >> np.uint64(8))
        | ((board & np.uint64(0x0F00000000000000)) >> np.uint64(20))
        | ((board & np.uint64(0x00F0000000000000)) >> np.uint64(32))
        | ((board & np.uint64(0x0000F00000000000)) << np.uint64(12))
        | ((board & np.uint64(0x000000F000000000)) >> np.uint64(12))
        | ((board & np.uint64(0x0000000F00000000)) << np.uint64(32))
        | ((board & np.uint64(0x000000000F000000)) << np.uint64(20))
        | ((board & np.uint64(0x0000000000F00000)) << np.uint64(8))
        | (board & np.uint64(0x000F0F0F000FFFFF))
    )


def R180_33(board):
    board = np.uint64(board)
    return (
        ((board & np.uint64(0xF000000000000000)) >> np.uint64(40))
        | ((board & np.uint64(0x0F00000000000000)) >> np.uint64(32))
        | ((board & np.uint64(0x00F0000000000000)) >> np.uint64(24))
        | ((board & np.uint64(0x0000F00000000000)) >> np.uint64(8))
        | ((board & np.uint64(0x000000F000000000)) << np.uint64(8))
        | ((board & np.uint64(0x0000000F00000000)) << np.uint64(24))
        | ((board & np.uint64(0x000000000F000000)) << np.uint64(32))
        | ((board & np.uint64(0x0000000000F00000)) << np.uint64(40))
        | (board & np.uint64(0x000F0F0F000FFFFF))
    )


def UL_33(board):
    board = np.uint64(board)
    return (
        ((board & np.uint64(0x0F0000F000000000)) >> np.uint64(12))
        | ((board & np.uint64(0x0000F0000F000000)) << np.uint64(12))
        | ((board & np.uint64(0x00F0000000000000)) >> np.uint64(24))
        | ((board & np.uint64(0x00000000F0000000)) << np.uint64(24))
        | (board & np.uint64(0xF00F0F0F00FFFFFF))
    )


def UR_33(board):
    board = np.uint64(board)
    return (
        ((board & np.uint64(0x0F00F00000000000)) >> np.uint64(20))
        | ((board & np.uint64(0x000000F00F000000)) << np.uint64(20))
        | ((board & np.uint64(0xF000000000000000)) >> np.uint64(40))
        | ((board & np.uint64(0x0000000000F00000)) << np.uint64(40))
        | (board & np.uint64(0x00FF0F0FF00FFFFF))
    )


def Rotate18034(board):
    board = Rotate180(board)
    board = ((board & np.uint64(0xFFFF000000000000)) >> np.uint64(48)) | (
        (board & np.uint64(0x0000FFFFFFFFFFFF)) << np.uint64(16)
    )
    return np.uint64(board)


def ReverseUD34(board):
    board = np.uint64(board)
    board = (board & np.uint64(0x0000FFFF0000FFFF)) | ((board & np.uint64(0xFFFF000000000000)) >> np.uint64(32)) | (
        (board & np.uint64(0x00000000FFFF0000)) << np.uint64(32)
    )
    return np.uint64(board)


def canonical_min33(board):
    return np.uint64(mover_runtime.canonical_min33(board))


def canonical_min24(board):
    return np.uint64(mover_runtime.canonical_min24(board))


def canonical_min34(board):
    return np.uint64(mover_runtime.canonical_min34(board))


def simulate_move_and_merge(line: np.typing.NDArray) -> Tuple[List[int], List[int]]:
    non_zero = [int(value) for value in line if value != 0]
    merged = [0] * len(line)
    new_line = []
    skip = False

    for index in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if (
            index + 1 < len(non_zero)
            and non_zero[index] == non_zero[index + 1]
            and non_zero[index] != -1
            and non_zero[index] != 32768
        ):
            new_line.append(2 * non_zero[index])
            merged[len(new_line) - 1] = 1
            skip = True
        else:
            new_line.append(non_zero[index])

    new_line.extend([0] * (len(line) - len(new_line)))
    return new_line, merged


def find_merge_positions(current_board: np.typing.NDArray, move_direction: str) -> np.typing.NDArray:
    merge_positions = np.zeros_like(current_board)
    move_direction = move_direction.lower()
    rows, cols = current_board.shape

    for index in range(rows if move_direction in ["left", "right"] else cols):
        line = current_board[index, :] if move_direction in ["left", "right"] else current_board[:, index]
        line_to_process = line[::-1] if move_direction in ["down", "right"] else line
        _, merge_line = simulate_move_and_merge(line_to_process)
        if move_direction in ["right", "down"]:
            merge_line = merge_line[::-1]
        if move_direction in ["left", "right"]:
            merge_positions[index, :] = merge_line
        else:
            merge_positions[:, index] = merge_line

    return merge_positions


def _move_distance(line: np.typing.NDArray) -> np.typing.NDArray:
    moved_distance = 0
    last_tile = 0
    move_distance = np.zeros_like(line)

    for index, value in enumerate(line):
        if value == 0:
            moved_distance += 1
        elif value == -1:
            moved_distance = 0
            last_tile = 0
        elif value == -2:
            last_tile = 0
        elif last_tile == value and value != 32768:
            move_distance[index] = moved_distance + 1
            moved_distance += 1
            last_tile = 0
        else:
            move_distance[index] = moved_distance
            last_tile = value

    return move_distance


def slide_distance(current_board: np.typing.NDArray, move_direction: str) -> np.typing.NDArray:
    move_distance = np.zeros_like(current_board)
    move_direction = move_direction.lower()
    rows, cols = current_board.shape

    for index in range(rows if move_direction in ["left", "right"] else cols):
        line = current_board[index, :] if move_direction in ["left", "right"] else current_board[:, index]
        line_to_process = line[::-1] if move_direction in ["down", "right"] else line
        line_move_distance = _move_distance(line_to_process)
        if move_direction in ["right", "down"]:
            line_move_distance = line_move_distance[::-1]
        if move_direction in ["left", "right"]:
            move_distance[index, :] = line_move_distance
        else:
            move_distance[:, index] = line_move_distance

    return move_distance


def count_zeros(line: np.typing.NDArray) -> int:
    if len(line) > 0 and line[0] != 0:
        return 1
    if len(line) > 1 and line[1] != 0:
        return 2
    if len(line) > 2 and line[2] != 0:
        return 3
    return len(line)
