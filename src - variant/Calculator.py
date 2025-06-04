from typing import List, Tuple, Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit


ToFindFunc = Callable[[np.uint64], np.uint64]


@njit(nogil=True, inline='always')
def ReverseLR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(nogil=True, inline='always')
def ReverseUD(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
    return board


@njit(nogil=True, inline='always')
def ReverseUL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
                board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
                board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return board


@njit(nogil=True, inline='always')
def ReverseUR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0x0f0ff0f00f0ff0f0)) | (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(20) | (
                board & np.uint64(0x00000f0f00000f0f)) << np.uint64(20)
    board = (board & np.uint64(0x00ff00ffff00ff00)) | (board & np.uint64(0xff00ff0000000000)) >> np.uint64(40) | (
                board & np.uint64(0x0000000000ff00ff)) << np.uint64(40)
    return board


@njit(nogil=True, inline='always')
def Rotate180(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(nogil=True, inline='always')
def RotateL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff0000000000)) >> np.uint64(32) | (
                board & np.uint64(0x00ff00ff00000000)) << np.uint64(8) | \
            (board & np.uint64(0x00000000ff00ff00)) >> np.uint64(8) | (
                        board & np.uint64(0x0000000000ff00ff)) << np.uint64(32)
    board = (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(16) | (
                board & np.uint64(0x0f0f00000f0f0000)) << np.uint64(4) | \
            (board & np.uint64(0x0000f0f00000f0f0)) >> np.uint64(4) | (
                        board & np.uint64(0x00000f0f00000f0f)) << np.uint64(16)
    return board


@njit(nogil=True, inline='always')
def RotateR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff0000000000)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00000000)) >> np.uint64(32) | \
            (board & np.uint64(0x00000000ff00ff00)) << np.uint64(32) | (
                        board & np.uint64(0x0000000000ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(16) | \
            (board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(16) | (
                        board & np.uint64(0x00000f0f00000f0f)) << np.uint64(4)
    return board


@njit(nogil=True, inline='always')
def min_all_symm(board):
    board = np.uint64(board)
    return np.uint64(min(ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board),
                         Rotate180(board), RotateL(board), RotateR(board), board))


@njit(nogil=True, inline='always')
def minUL(bd):
    board = np.uint64(bd)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
            board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
            board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return min(bd, board)


@njit(nogil=True, inline='always')
def is_LL_pattern(encoded_board, pattern_encoded):
    return (np.uint64(encoded_board) & np.uint64(0xff00ff)) == np.uint64(pattern_encoded)


@njit(nogil=True, inline='always')
def is_4411_pattern(encoded_board, pattern_encoded):
    return (np.uint64(encoded_board) & np.uint64(0xfff0fff)) == np.uint64(pattern_encoded)


@njit(nogil=True, inline='always')
def re_self(encoded_board):
    return np.uint64(encoded_board)


def simulate_move_and_merge(line: np.typing.NDArray) -> Tuple[List[int], List[int]]:
    """模拟一行的移动和合并过程，返回新的行和合并发生的位置。"""
    # 移除所有的0，保留非0元素
    non_zero = [value for value in line if value != 0]
    merged = [0] * len(line)  # 合并标记
    new_line = []
    skip = False

    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != -1 and non_zero[i] != 32768:
            # 发生合并
            new_line.append(2 * non_zero[i])
            merged[len(new_line) - 1] = 1  # 标记合并发生的位置
            skip = True
        else:
            new_line.append(non_zero[i])

    # 用0填充剩余的空间
    new_line.extend([0] * (len(line) - len(new_line)))
    return new_line, merged


def find_merge_positions(current_board: np.typing.NDArray, move_direction: str) -> np.typing.NDArray:
    """找到当前棋盘上即将发生合并的位置。"""
    # 初始化合并位置数组
    merge_positions = np.zeros_like(current_board)
    move_direction = move_direction.lower()

    rows, cols = current_board.shape

    for i in range(rows if move_direction in ['left', 'right'] else cols):
        if move_direction in ['left', 'right']:
            line = current_board[i, :]
        else:
            line = current_board[:, i]
        line_to_process = line[::-1] if move_direction in ['down', 'right'] else line
        processed_line, merge_line = simulate_move_and_merge(line_to_process)
        if move_direction in ['right', 'down']:
            merge_line = merge_line[::-1]

        if move_direction in ['left', 'right']:
            merge_positions[i, :] = merge_line
        else:
            merge_positions[:, i] = merge_line

    return merge_positions