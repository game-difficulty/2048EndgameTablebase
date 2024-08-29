from typing import List, Tuple

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def ReverseLR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(nogil=True)
def ReverseUD(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
    return board


@njit(nogil=True)
def ReverseUL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
                board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
                board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return board


@njit(nogil=True)
def ReverseUR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0x0f0ff0f00f0ff0f0)) | (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(20) | (
                board & np.uint64(0x00000f0f00000f0f)) << np.uint64(20)
    board = (board & np.uint64(0x00ff00ffff00ff00)) | (board & np.uint64(0xff00ff0000000000)) >> np.uint64(40) | (
                board & np.uint64(0x0000000000ff00ff)) << np.uint64(40)
    return board


@njit(nogil=True)
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


@njit(nogil=True)
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


@njit(nogil=True)
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


@njit(nogil=True)
def min_all_symm(board):
    board = np.uint64(board)
    return np.uint64(min(ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board),
                         Rotate180(board), RotateL(board), RotateR(board), board))


@njit(nogil=True)
def minUL(bd):
    board = np.uint64(bd)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
            board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
            board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return min(bd, board)


@njit(nogil=True)
def is_44_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4294967295)) == np.uint64(4294967295)


@njit(nogil=True)
def is_44_success(encoded_board, target, position):
    return (np.uint64(encoded_board) >> np.uint64(44 - 4 * position) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_4431_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(987135)) == np.uint64(987135)


@njit(nogil=True)
def is_4431_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_444_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(65535)) == np.uint64(65535)


@njit(nogil=True)
def is_444_success(encoded_board, target, position):
    position = max(position, 1)
    return (np.uint64(encoded_board) >> np.uint64(28 - 4 * position) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xffffff)) == np.uint64(0xffffff) or \
        (np.uint64(encoded_board) & np.uint64(0xf0ff0fff)) == np.uint64(0xf0ff0fff) or \
        (np.uint64(encoded_board) & np.uint64(0xffff0ff)) == np.uint64(0xffff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xffff00ff)) == np.uint64(0xffff00ff)


@njit(nogil=True)
def is_442_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_t_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf0fff0ff)) == np.uint64(0xf0fff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f0ff00ff)) == np.uint64(0xf000f0ff00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f00000ff00ff)) == np.uint64(0xf000f00000ff00ff)


@njit(nogil=True)
def is_t_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_4442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(255)) == np.uint64(255)


@njit(nogil=True)
def is_4442_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(16) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_L3_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(268374015)) == np.uint64(268374015)


@njit(nogil=True)
def is_L3_success(encoded_board, target, position):
    if position > 0:
        return (np.uint64(encoded_board) >> np.uint64(40 - 4 * position) & np.uint64(0xf)) == np.uint64(target)
    else:
        return (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(40) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_LL_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(16711935)) == np.uint64(16711935)


@njit(nogil=True)
def is_LL_success(encoded_board, target, position):
    encoded_board = np.uint64(encoded_board)
    if position == 0:
        return (encoded_board >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target)
    else:
        return (encoded_board >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_free_pattern(_):
    return True


@njit(nogil=True)
def is_free_success(encoded_board, target, position):
    if position == 0:
        target += 1  # free要求合出更大一级的数字，freew定式pos参数为1
    encoded_board = np.uint64(encoded_board)
    for i in range(16):
        if (encoded_board >> np.uint64(4 * i)) & np.uint64(0xF) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_4441_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4095)) == np.uint64(4095)


@njit(nogil=True)
def is_4441_success(encoded_board, target, _):
    for pos in (12, 16, 20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_4432_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(983295)) == np.uint64(983295)


@njit(nogil=True)
def is_4432_success(encoded_board, target, _):
    for pos in (20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_movingLL_pattern(encoded_board):
    for pattern in (np.uint64(0xff00ff), np.uint64(0xff00ff0), np.uint64(0xff00ff00), np.uint64(0xff00ff0000),
                    np.uint64(0xff00ff00000), np.uint64(0xff00ff000000), np.uint64(0xff00ff00000000),
                    np.uint64(0xff00ff000000000), np.uint64(0xff00ff0000000000)):
        if (np.uint64(encoded_board) & pattern) == pattern:
            return True
        return False


@njit(nogil=True)
def is_movingLL_success(encoded_board, target, _):
    for pos in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_3433_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf0ff)) == np.uint64(0xf0ff)


@njit(nogil=True)
def is_3433_success(encoded_board, target, _):
    for pos in (8, 16, 20, 28):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_3442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xff00f)) == np.uint64(0xff00f)


@njit(nogil=True)
def is_3442_success(encoded_board, target, _):
    for pos in (4, 8, 28, 32):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_3432_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xff0ff)) == np.uint64(0xff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00f00ff)) == np.uint64(0xf00f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00000f00ff)) == np.uint64(0xf00000f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf0000000000f00ff)) == np.uint64(0xf0000000000f00ff)


@njit(nogil=True)
def is_3432_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_2433_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf000f0ff)) == np.uint64(0xf000f0ff)


@njit(nogil=True)
def is_2433_success(encoded_board, target, _):
    for pos in (8, 16, 20, 44):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def re_self(encoded_board):
    return np.uint64(encoded_board)


@njit(nogil=True, parallel=True)
def p_minUL(arr):
    for i in prange(len(arr)):
        arr[i] = minUL(arr[i])


@njit(nogil=True, parallel=True)
def p_min_all_symm(arr):
    for i in prange(len(arr)):
        arr[i] = min_all_symm(arr[i])


@njit(nogil=True)
def p_re_self(_):
    pass


def simulate_move_and_merge(line: np.ndarray) -> Tuple[List[int], List[int]]:
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


def find_merge_positions(current_board: np.ndarray, move_direction: str) -> np.ndarray:
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
