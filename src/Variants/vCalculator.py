import numpy as np
from numba import njit, prange
from Calculator import ReverseUD, ReverseLR, Rotate180


@njit(nogil=True)
def exchange_row12(board):
    return (board & np.uint64(0xffff00000000ffff)) | ((board & np.uint64(0x00000000ffff0000)) << np.uint64(16)) | (
                (board & np.uint64(0x0000ffff00000000)) >> np.uint64(16))


@njit(nogil=True)
def exchange_row02(board):
    return (board & np.uint64(0x0000ffff0000ffff)) | ((board & np.uint64(0x00000000ffff0000)) << np.uint64(32)) | (
                (board & np.uint64(0xffff000000000000)) >> np.uint64(32))


@njit(nogil=True)
def exchange_col02(board):
    return (board & np.uint64(0x0f0f0f0f0f0f0f0f)) | ((board & np.uint64(0xf000f000f000f000)) >> np.uint64(8)) | (
                (board & np.uint64(0x00f000f000f000f0)) << np.uint64(8))


@njit(nogil=True)
def R90_33(board):
    return ((board & np.uint64(0xf000000000000000)) >> np.uint64(32)) | (
                (board & np.uint64(0x0f00000000000000)) >> np.uint64(12)) | (
            (board & np.uint64(0x00f0000000000000)) << np.uint64(8)) | (
                (board & np.uint64(0x0000f00000000000)) >> np.uint64(20)) | (
            (board & np.uint64(0x000000f000000000)) << np.uint64(20)) | (
                (board & np.uint64(0x000000000f0000000)) >> np.uint64(8)) | (
            (board & np.uint64(0x000000000f000000)) << np.uint64(12)) | (
                (board & np.uint64(0x00000000000f00000)) << np.uint64(32)) | (
            board & np.uint64(0x000f0f0f000fffff))


@njit(nogil=True)
def L90_33(board):
    return ((board & np.uint64(0xf000000000000000)) >> np.uint64(8)) | (
                (board & np.uint64(0x0f00000000000000)) >> np.uint64(20)) | (
            (board & np.uint64(0x00f0000000000000)) >> np.uint64(32)) | (
                (board & np.uint64(0x0000f00000000000)) << np.uint64(12)) | (
            (board & np.uint64(0x000000f000000000)) >> np.uint64(12)) | (
                (board & np.uint64(0x000000000f0000000)) << np.uint64(32)) | (
            (board & np.uint64(0x000000000f000000)) << np.uint64(20)) | (
                (board & np.uint64(0x00000000000f00000)) << np.uint64(8)) | (
            board & np.uint64(0x000f0f0f000fffff))


@njit(nogil=True)
def R180_33(board):
    return ((board & np.uint64(0xf000000000000000)) >> np.uint64(40)) | (
                (board & np.uint64(0x0f00000000000000)) >> np.uint64(32)) | (
            (board & np.uint64(0x00f0000000000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x0000f00000000000)) >> np.uint64(8)) | (
            (board & np.uint64(0x000000f000000000)) << np.uint64(8)) | (
                (board & np.uint64(0x000000000f0000000)) << np.uint64(24)) | (
            (board & np.uint64(0x000000000f000000)) << np.uint64(32)) | (
                (board & np.uint64(0x00000000000f00000)) << np.uint64(40)) | (
            board & np.uint64(0x000f0f0f000fffff))


@njit(nogil=True)
def UL_33(board):
    return ((board & np.uint64(0x0f0000f000000000)) >> np.uint64(12)) | (
                (board & np.uint64(0x0000f0000f000000)) << np.uint64(12)) | (
            (board & np.uint64(0x00f0000000000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x00000000f0000000)) << np.uint64(24)) | (
            board & np.uint64(0xf00f0f0f00ffffff))


@njit(nogil=True)
def UR_33(board):
    return ((board & np.uint64(0x0f00f00000000000)) >> np.uint64(20)) | (
                (board & np.uint64(0x000000f00f000000)) << np.uint64(20)) | (
            (board & np.uint64(0xf000000000000000)) >> np.uint64(40)) | (
                (board & np.uint64(0x0000000000f00000)) << np.uint64(40)) | (
            board & np.uint64(0x00ff0f0ff00fffff))


@njit(nogil=True)
def Rotate18034(board):
    board = Rotate180(board)
    board = ((board & np.uint64(0xffff000000000000)) >> np.uint64(48)) | (
            (board & np.uint64(0x0000ffffffffffff)) << np.uint64(16))
    return board


@njit(nogil=True)
def ReverseUD34(board):
    board = (board & np.uint64(0x0000ffff0000ffff)) | ((board & np.uint64(0xffff000000000000)) >> np.uint64(32)) | (
            (board & np.uint64(0x00000000ffff0000)) << np.uint64(32))
    return board


@njit(nogil=True)
def min33(board):
    return np.uint64(min(exchange_col02(board), exchange_row02(board), R90_33(board), L90_33(board),
                         R180_33(board), UR_33(board), UL_33(board), board))


@njit(nogil=True)
def min24(board):
    return min(ReverseUD(board), ReverseLR(board), Rotate180(board), board)


@njit(nogil=True)
def min34(board):
    return min(ReverseLR(board), ReverseUD34(board), Rotate18034(board), board)


@njit(nogil=True)
def p_min24(arr):
    for i in prange(len(arr)):
        arr[i] = min24(arr[i])


@njit(nogil=True)
def p_min34(arr):
    for i in prange(len(arr)):
        arr[i] = min34(arr[i])


@njit(nogil=True)
def p_min33(arr):
    for i in prange(len(arr)):
        arr[i] = min33(arr[i])


@njit(nogil=True)
def is_variant_pattern(_):
    return True


@njit(nogil=True)
def is_3x4_success(encoded_board, target, _):
    for i in range(12):
        if (encoded_board >> np.uint64(4 * (i + 4))) & np.uint64(0xF) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_2x4_success(encoded_board, target, _):
    for i in range(8):
        if (encoded_board >> np.uint64(4 * (i + 4))) & np.uint64(0xF) == np.uint64(target):
            return True
    return False


@njit(nogil=True)
def is_3x3_success(encoded_board, target, _):
    for i in range(16):
        if (encoded_board >> np.uint64(4 * i)) & np.uint64(0xF) == np.uint64(target):
            return True
    return False
