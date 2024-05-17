import numpy as np
from numba import njit, uint64, boolean


@njit(uint64(uint64))
def ReverseLR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(uint64(uint64))
def ReverseUD(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
    return board


@njit(uint64(uint64))
def ReverseUL(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
                board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
                board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return board


@njit(uint64(uint64))
def ReverseUR(board):
    board = np.uint64(board)
    board = (board & np.uint64(0x0f0ff0f00f0ff0f0)) | (board & np.uint64(0xf0f00000f0f00000)) >> np.uint64(20) | (
                board & np.uint64(0x00000f0f00000f0f)) << np.uint64(20)
    board = (board & np.uint64(0x00ff00ffff00ff00)) | (board & np.uint64(0xff00ff0000000000)) >> np.uint64(40) | (
                board & np.uint64(0x0000000000ff00ff)) << np.uint64(40)
    return board


@njit(uint64(uint64))
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


@njit(uint64(uint64))
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


@njit(uint64(uint64))
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


@njit(uint64(uint64))
def min_all_symm(board):
    board = np.uint64(board)
    return np.uint64(min(ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board),
                         Rotate180(board), RotateL(board), RotateR(board), board))


@njit(uint64(uint64))
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
    return (np.uint64(encoded_board) & np.uint64(16777215)) == np.uint64(16777215)


@njit(nogil=True)
def is_442_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_t_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(4043305215)) == np.uint64(4043305215) or \
        (np.uint64(encoded_board) & np.uint64(263886833910015)) == np.uint64(263886833910015) or \
        (np.uint64(encoded_board) & np.uint64(17294086451910082815)) == np.uint64(17294086451910082815)


@njit(nogil=True)
def is_t_success(encoded_board, target, _):
    return (np.uint64(encoded_board) >> np.uint64(8) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(32) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(44) & np.uint64(0xf)) == np.uint64(target) or \
        (np.uint64(encoded_board) >> np.uint64(24) & np.uint64(0xf)) == np.uint64(target)


@njit(nogil=True)
def is_L3_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(268374015)) == np.uint64(268374015)


@njit(nogil=True)
def is_L3_success(encoded_board, target, position):
    if position > 0:
        return (np.uint64(encoded_board) >> np.uint64(40 - 4 * position) & np.uint64(0xf)) == np.uint64(target)
    else:
        return (np.uint64(encoded_board) >> np.uint64(40) & np.uint64(0xf)) == np.uint64(target) or \
            (np.uint64(encoded_board) >> np.uint64(28) & np.uint64(0xf)) == np.uint64(target)


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
        target += 1  # free要求合出更大一级的数字
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
    for pos in (12, 16, 20, 24, 28):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_4432_pattern(encoded_board):
    return (np.uint64(encoded_board) & np.uint64(983295)) == np.uint64(983295)


@njit(nogil=True)
def is_4432_success(encoded_board, target, _):
    for pos in (8, 20, 32):
        if (np.uint64(encoded_board) >> np.uint64(pos) & np.uint64(0xf)) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def re_self(encoded_board):
    return np.uint64(encoded_board)


@njit(nogil=True)
def radix_sort(arr):
    temp = np.empty_like(arr)
    # 进行四轮排序，每轮对应一个16位的块
    for shift in range(0, 64, 16):
        # 计算每个元素在这一轮的桶编号
        counts = np.zeros(65537, dtype=np.int32)
        for i in arr:
            counts[((i >> np.uint64(shift)) & np.uint64(65535))+1] += 1
        # 计算累积和，用于确定元素在temp中的位置
        counts = np.cumsum(counts)
        # 分配元素到临时数组
        for i in arr:
            idx = i >> np.uint64(shift) & np.uint64(65535)
            temp[counts[idx]] = i
            counts[idx] += 1
        # 将排序后的数组复制回原数组，进行下一轮
        arr[:] = temp
    return unique_sorted(arr)


@njit(nogil=True)
def unique_sorted(arr):
    uniquer = np.empty(len(arr), dtype=boolean)
    uniquer[1:] = arr[1:] != arr[:-1]
    uniquer[0] = 1
    return arr[uniquer]


@njit(nogil=True)
def unique(arr):
    if len(arr) < 200000:
        return np.unique(arr)
    else:
        return radix_sort(arr)
