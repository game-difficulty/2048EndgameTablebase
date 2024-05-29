import numpy as np
from numba import njit, uint64


@njit(uint64(uint64))
def ReverseUD(board):
    board = np.uint64(board)
    board = (board & np.uint64(0xffffffff00000000)) >> np.uint64(32) | (
                board & np.uint64(0x00000000ffffffff)) << np.uint64(32)
    board = (board & np.uint64(0xffff0000ffff0000)) >> np.uint64(16) | (
                board & np.uint64(0x0000ffff0000ffff)) << np.uint64(16)
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
def is_24_pattern(board):
    return (board & np.uint64(0xffff00000000ffff)) == np.uint64(0xffff00000000ffff)


@njit(uint64(uint64))
def is_34_pattern(board):
    return (board & np.uint64(0x000000000000ffff)) == np.uint64(0x000000000000ffff)


@njit(uint64(uint64))
def is_33_pattern(board):
    return (board & np.uint64(0x000f000f000fffff)) == np.uint64(0x000f000f000fffff)


@njit(uint64(uint64))
def ReverseLR(board):
    board = (board & np.uint64(0xff00ff00ff00ff00)) >> np.uint64(8) | (
                board & np.uint64(0x00ff00ff00ff00ff)) << np.uint64(8)
    board = (board & np.uint64(0xf0f0f0f0f0f0f0f0)) >> np.uint64(4) | (
                board & np.uint64(0x0f0f0f0f0f0f0f0f)) << np.uint64(4)
    return board


@njit(uint64(uint64))
def exchange_row12(board):
    return (board & np.uint64(0xffff00000000ffff)) | ((board & np.uint64(0x00000000ffff0000)) << np.uint64(16)) | (
                (board & np.uint64(0x0000ffff00000000)) >> np.uint64(16))


@njit(uint64(uint64))
def exchange_row02(board):
    return (board & np.uint64(0x0000ffff0000ffff)) | ((board & np.uint64(0x00000000ffff0000)) << np.uint64(32)) | (
                (board & np.uint64(0xffff000000000000)) >> np.uint64(32))


@njit(uint64(uint64))
def exchange_col02(board):
    return (board & np.uint64(0x0f0f0f0f0f0f0f0f)) | ((board & np.uint64(0xf000f000f000f000)) >> np.uint64(8)) | (
                (board & np.uint64(0x00f000f000f000f0)) << np.uint64(8))


@njit(uint64(uint64))
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


@njit(uint64(uint64))
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


@njit(uint64(uint64))
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


@njit(uint64(uint64))
def UL_33(board):
    return ((board & np.uint64(0x0f0000f000000000)) >> np.uint64(12)) | (
                (board & np.uint64(0x0000f0000f000000)) << np.uint64(12)) | (
            (board & np.uint64(0x00f0000000000000)) >> np.uint64(24)) | (
                (board & np.uint64(0x00000000f0000000)) << np.uint64(24)) | (
            board & np.uint64(0xf00f0f0f00ffffff))


@njit(uint64(uint64))
def UR_33(board):
    return ((board & np.uint64(0x0f00f00000000000)) >> np.uint64(20)) | (
                (board & np.uint64(0x000000f00f000000)) << np.uint64(20)) | (
            (board & np.uint64(0xf000000000000000)) >> np.uint64(40)) | (
                (board & np.uint64(0x0000000000f00000)) << np.uint64(40)) | (
            board & np.uint64(0x00ff0f0ff00fffff))


@njit(uint64(uint64))
def R180(board):
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
def R180_43(board):
    board = R180(board)
    return ((board & np.uint64(0xffff000000000000)) >> np.uint64(48)) | (
            (board & np.uint64(0x0000ffffffffffff)) << np.uint64(16))


@njit(uint64(uint64))
def min33(board):
    return np.uint64(min(exchange_col02(board), exchange_row02(board), R90_33(board), L90_33(board),
                         R180_33(board), UR_33(board), UL_33(board), board))


@njit(uint64(uint64))
def min24(board):
    board = np.uint64(board)
    return np.uint64(min(ReverseLR(board), exchange_row12(board), R180(board), board))


@njit(uint64(uint64))
def min34(board):
    board = np.uint64(board)
    return np.uint64(min(ReverseLR(board), exchange_row02(board), R180_43(board), board))


@njit(nogil=True)
def re_self(encoded_board):
    return np.uint64(encoded_board)
