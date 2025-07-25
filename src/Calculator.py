from typing import List, Tuple, Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit


ToFindFunc = Callable[[np.uint64], np.uint64]


@njit(nogil=True, inline='always')
def ReverseLR(board):
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(nogil=True, inline='always')
def ReverseUD(board):
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
    return board


@njit(nogil=True, inline='always')
def ReverseUL(board):
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
                board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
                board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return board


@njit(nogil=True, inline='always')
def ReverseUR(board):
    board = (board & np.uint64(0x0f0ff0f00f0ff0f0)) | (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(20) | (
                board & np.uint64(0x00000f0f00000f0f)) << np.uint64(20)
    board = (board & np.uint64(0x00ff00ffff00ff00)) | (board & np.uint64(0xff00ff0000000000)) >> np.uint64(40) | (
                board & np.uint64(0x0000000000ff00ff)) << np.uint64(40)
    return board


@njit(nogil=True, inline='always')
def Rotate180(board):
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
def is_44_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4294967295)) == np.uint64(4294967295)


@njit(nogil=True, inline='always')
def is_44_success(encoded_board, target, position):
    return (np.uint64(encoded_board) >> np.uint64(44 - 4 * position) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_4431_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(987135)) == np.uint64(987135)


@njit(nogil=True, inline='always')
def is_4431_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_444_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(65535)) == np.uint64(65535)


@njit(nogil=True, inline='always')
def is_444_success(encoded_board, target, position):
    position = max(position, 1)
    return (np.uint64(encoded_board) >> np.uint64(28 - 4 * position) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xffffff)) == np.uint64(0xffffff)


@njit(nogil=True, inline='always')
def is_442_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_442t_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xffffff)) == np.uint64(0xffffff) or \
        (np.uint64(encoded_board) & np.uint64(0xf0ff0fff)) == np.uint64(0xf0ff0fff) or \
        (np.uint64(encoded_board) & np.uint64(0xffff0ff)) == np.uint64(0xffff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xffff00ff)) == np.uint64(0xffff00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00fffff0)) == np.uint64(0xf00fffff0) or \
        (np.uint64(encoded_board) & np.uint64(0xff0ff0ff0)) == np.uint64(0xff0ff0ff0) or \
        (np.uint64(encoded_board) & np.uint64(0xf00000ff0fff)) == np.uint64(0xf00000ff0fff) or \
        (np.uint64(encoded_board) & np.uint64(0xf0000fff00ff)) == np.uint64(0xf0000fff00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00f00ff0ff0)) == np.uint64(0xf00f00ff0ff0) or \
        (np.uint64(encoded_board) & np.uint64(0xf00f0fff00f0)) == np.uint64(0xf00f0fff00f0)


@njit(nogil=True, inline='always')
def is_442t_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_t_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf0fff0ff)) == np.uint64(0xf0fff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f0ff00ff)) == np.uint64(0xf000f0ff00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f00000ff00ff)) == np.uint64(0xf000f00000ff00ff)


@njit(nogil=True, inline='always')
def is_t_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_4442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(255)) == np.uint64(255)


@njit(nogil=True, inline='always')
def is_4442_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(16) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_4442f_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(255)) == np.uint64(255)


@njit(nogil=True, inline='always')
def is_4442f_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(12) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(16) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_4442ff_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(255)) == np.uint64(255)


@njit(nogil=True, inline='always')
def is_4442ff_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(12) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(16) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(48) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(52) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_L3t_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xfff0fff)) == np.uint64(0xfff0fff) or \
            (np.uint64(encoded_board) & np.uint64(0xf0fff0ff0)) == np.uint64(0xf0fff0ff0) or \
            (np.uint64(encoded_board) & np.uint64(0xf000f0ff00ff0)) == np.uint64(0xf000f0ff00ff0)


@njit(nogil=True, inline='always')
def is_L3t_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(40) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_L3_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(268374015)) == np.uint64(268374015)


@njit(nogil=True, inline='always')
def is_L3_success(encoded_board, target, position):
    if position > 0:
        return (np.uint64(encoded_board) >> np.uint64(40 - 4 * position) & np.uint64(0xf)) == np.uint64(target)
    else:
        return (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(40) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_LL_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(16711935)) == np.uint64(16711935)


@njit(nogil=True, inline='always')
def is_LL_success(encoded_board, target, position):
    encoded_board = np.uint64(encoded_board)
    if position == 0:
        return (encoded_board >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(36) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
            (encoded_board >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target)
    else:
        return (encoded_board >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_free_pattern(_):
    return True


@njit(nogil=True, inline='always')
def is_free_success(encoded_board, target, position):
    if position == 0:
        target += 1  # free要求合出更大一级的数字，freew定式pos参数为1
    encoded_board = np.uint64(encoded_board)
    for i in range(16):
        if (encoded_board >> np.uint64(4 * i)) & np.uint64(0xF) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_4441_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4095)) == np.uint64(4095)


@njit(nogil=True, inline='always')
def is_4441_success(encoded_board, target, _):
    for pos in (12, 16, 20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_4441f_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4095)) == np.uint64(4095)


@njit(nogil=True, inline='always')
def is_4441f_success(encoded_board, target, _):
    for pos in (12, 16, 20, 24):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_4432_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(983295)) == np.uint64(983295)


@njit(nogil=True, inline='always')
def is_4432_success(encoded_board, target, _):
    for pos in (20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_4432f_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(983295)) == np.uint64(983295)


@njit(nogil=True, inline='always')
def is_4432f_success(encoded_board, target, _):
    for pos in (20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_4432ff_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(983295)) == np.uint64(983295)


@njit(nogil=True, inline='always')
def is_4432ff_success(encoded_board, target, _):
    for pos in (20,):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_movingLL_pattern(encoded_board):
    for pattern in (np.uint64(0xff00ff), np.uint64(0xff00ff0), np.uint64(0xff00ff00), np.uint64(0xff00ff0000),
                    np.uint64(0xff00ff00000), np.uint64(0xff00ff000000), np.uint64(0xff00ff00000000),
                    np.uint64(0xff00ff000000000), np.uint64(0xff00ff0000000000)):
        if (np.uint64(encoded_board) & pattern) == pattern:
            return True
    return False


@njit(nogil=True, inline='always')
def is_movingLL_success(encoded_board, target, _):
    for pos in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_3433_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf0ff)) == np.uint64(0xf0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00000ff)) == np.uint64(0xf00000ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000000000ff)) == np.uint64(0xf000000000ff)


@njit(nogil=True, inline='always')
def is_3433_success(encoded_board, target, _):
    for pos in (8, 16, 20, 28):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_3442_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xff00f)) == np.uint64(0xff00f)


@njit(nogil=True, inline='always')
def is_3442_success(encoded_board, target, _):
    for pos in (4, 8, 28, 32):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def is_3432_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xff0ff)) == np.uint64(0xff0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf00f00ff)) == np.uint64(0xf00f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000000f00ff)) == np.uint64(0xf000000f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf0000000000f00ff)) == np.uint64(0xf0000000000f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0x0f000000000f00ff)) == np.uint64(0x0f000000000f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0x00f00000000f00ff)) == np.uint64(0x00f00000000f00ff) or \
        (np.uint64(encoded_board) & np.uint64(0x000f0000000f00ff)) == np.uint64(0x000f0000000f00ff)


@njit(nogil=True, inline='always')
def is_3432_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(20) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True, inline='always')
def is_2433_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(0xf000f0ff)) == np.uint64(0xf000f0ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f00000ff)) == np.uint64(0xf000f00000ff) or \
        (np.uint64(encoded_board) & np.uint64(0xf000f000000000ff)) == np.uint64(0xf000f000000000ff)


@njit(nogil=True, inline='always')
def is_2433_success(encoded_board, target, _):
    for pos in (8, 16, 20, 44):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True, inline='always')
def re_self(encoded_board):
    return np.uint64(encoded_board)


@njit(nogil=True, inline='always')
def minUL_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    board = ReverseUL(bd1)
    if bd1 <= board:
        return bd1, 0
    else:
        return board, 3


@njit(nogil=True)
def min_all_symm_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    min_value_bd1 = bd1
    best_symm = 0

    # 展开对称操作
    transformed_bd1 = ReverseLR(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 1

    transformed_bd1 = ReverseUD(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 2

    transformed_bd1 = ReverseUL(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 3

    transformed_bd1 = ReverseUR(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 4

    transformed_bd1 = Rotate180(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 5

    transformed_bd1 = RotateL(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 6

    transformed_bd1 = RotateR(bd1)
    if transformed_bd1 < min_value_bd1:
        min_value_bd1 = transformed_bd1
        best_symm = 7

    return min_value_bd1, best_symm


@njit(nogil=True, inline='always')
def re_self_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return bd1, 0


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


def _move_distance(line: np.typing.NDArray) -> np.typing.NDArray:
    moved_distance = 0
    last_tile = 0
    move_distance = np.zeros_like(line)

    for index, i in enumerate(line):
        if i == 0:
            moved_distance += 1
        elif i == -1:
            # minigame中代表无法移动、无法合并的格子
            moved_distance = 0
            last_tile = 0
        elif i == -2:
            last_tile = 0
        elif last_tile == i and i != 32768:
            move_distance[index] = moved_distance + 1
            moved_distance += 1
            last_tile = 0
        else:
            move_distance[index] = moved_distance
            last_tile = i

    return move_distance


def slide_distance(current_board: np.typing.NDArray, move_direction: str) -> np.typing.NDArray:
    """当棋盘移动时各个格子需要移动几格"""
    move_distance = np.zeros_like(current_board)
    move_direction = move_direction.lower()

    rows, cols = current_board.shape

    for i in range(rows if move_direction in ['left', 'right'] else cols):
        if move_direction in ['left', 'right']:
            line = current_board[i, :]
        else:
            line = current_board[:, i]
        line_to_process = line[::-1] if move_direction in ['down', 'right'] else line
        line_move_distance = _move_distance(line_to_process)
        if move_direction in ['right', 'down']:
            line_move_distance = line_move_distance[::-1]

        if move_direction in ['left', 'right']:
            move_distance[i, :] = line_move_distance
        else:
            move_distance[:, i] = line_move_distance

    return move_distance


def count_zeros(line: np.typing.NDArray) -> int:
    # 专用函数
    """ line中0的个数，line长度不超过3；中途遇见非零格子直接爆炸 """
    if len(line) > 0 and line[0] != 0:
        return 1
    if len(line) > 1 and line[1] != 0:
        return 2
    if len(line) > 2 and line[2] != 0:
        return 3
    return len(line)


if __name__ == '__main__':
    x,y=min_all_symm_pair(np.uint64(0x1a0b00237970df21))
    print(x,y)
