from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import uint64, bool, njit

from BoardMover import encode_board, decode_board, encode_row, decode_row, reverse


"""
移动之后保持masked
定式计算专用，勿作他用
"""

@njit(nogil=True)
def merge_line(line: NDArray, reverse: bool = False) -> Tuple[NDArray, bool]:
    mask_new_tile = False
    if reverse:
        line = line[::-1]
    non_zero = [i for i in line if i != 0]  # 去掉所有的0
    merged = []
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != 32768:
            merged_value = 2 * non_zero[i]
            if merged_value >= 64:
                mask_new_tile = True
                merged_value = 32768
            merged.append(merged_value)
            skip = True
        else:
            merged.append(non_zero[i])

    # 补齐剩下的 0
    merged += [0] * (len(line) - len(merged))
    if reverse:
        merged = merged[::-1]
    return np.array(merged), mask_new_tile


@njit(nogil=True)
def calculate_all_moves() -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    # 初始化存储所有可能的行及其移动后结果差值的字典
    movel = np.empty(65536, dtype=np.uint16)
    mover = np.empty(65536, dtype=np.uint16)
    moveu = np.empty(65536, dtype=np.uint64)
    moved = np.empty(65536, dtype=np.uint64)
    mask_new_tiles = np.empty(65536, dtype=np.bool_)

    # 生成所有可能的行
    for i in range(16 ** 4):
        line = [(i // (16 ** j)) % 16 for j in range(4)]
        line = np.array([2 ** k if k else 0 for k in line])  # 把游戏中的数字转换成2的幂
        original_line = encode_row(line)  # 编码原始行为整数

        # 向左移动
        merged_linel, mask_new_tile = merge_line(line, False)
        movel[original_line] = np.uint16(encode_row(merged_linel) ^ original_line)
        mask_new_tiles[original_line] = mask_new_tile

        # 向右移动
        merged_liner, _ = merge_line(line, True)
        mover[original_line] = np.uint16(encode_row(merged_liner) ^ original_line)

    # 使用reverse函数计算向上和向下的移动差值
    for i in range(16 ** 4):
        moveu[i] = reverse(np.uint64(movel[i]))
        moved[i] = reverse(np.uint64(mover[i]))

    return movel, mover, moveu, moved, mask_new_tiles


movel, mover, moveu, moved, mask_new_tiles = calculate_all_moves()


@njit(nogil=True)
def m_move_left(board: np.uint64) -> np.uint64:
    board ^= movel[board & np.uint64(0xffff)]
    board ^= movel[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
    board ^= movel[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
    board ^= movel[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
    return board


@njit(nogil=True)
def m_move_left2(board: np.uint64) -> Tuple[np.uint64, bool]:
    mnt = False
    mnt |= mask_new_tiles[board & np.uint64(0xffff)]
    board ^= movel[board & np.uint64(0xffff)]
    mnt |= mask_new_tiles[board >> np.uint64(16) & np.uint64(0xffff)]
    board ^= movel[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
    mnt |= mask_new_tiles[board >> np.uint64(32) & np.uint64(0xffff)]
    board ^= movel[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
    mnt |= mask_new_tiles[board >> np.uint64(48) & np.uint64(0xffff)]
    board ^= movel[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
    return board, mnt


@njit(nogil=True)
def m_move_right(board: np.uint64) -> np.uint64:
    board ^= mover[board & np.uint64(0xffff)]
    board ^= mover[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
    board ^= mover[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
    board ^= mover[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
    return board


@njit(nogil=True)
def m_move_up(board: np.uint64, board2: np.uint64) -> np.uint64:
    board ^= moveu[board2 & np.uint64(0xffff)]
    board ^= moveu[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
    board ^= moveu[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
    board ^= moveu[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
    return board


@njit(nogil=True)
def m_move_up2(board: np.uint64, board2: np.uint64) -> Tuple[np.uint64, bool]:
    mnt = False
    mnt |= mask_new_tiles[board2 & np.uint64(0xffff)]
    board ^= moveu[board2 & np.uint64(0xffff)]
    mnt |= mask_new_tiles[board2 >> np.uint64(16) & np.uint64(0xffff)]
    board ^= moveu[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
    mnt |= mask_new_tiles[board2 >> np.uint64(32) & np.uint64(0xffff)]
    board ^= moveu[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
    mnt |= mask_new_tiles[board2 >> np.uint64(48) & np.uint64(0xffff)]
    board ^= moveu[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
    return board, mnt


@njit(nogil=True)
def m_move_down(board: np.uint64, board2: np.uint64) -> np.uint64:
    board ^= moved[board2 & np.uint64(0xffff)]
    board ^= moved[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
    board ^= moved[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
    board ^= moved[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
    return board


@njit(nogil=True)
def m_move_board(board: np.uint64, direction: int) -> np.uint64:
    if direction == 1:
        return m_move_left(board)
    elif direction == 2:
        return m_move_right(board)
    elif direction == 3:
        board2 = reverse(board)
        return m_move_up(board, board2)
    elif direction == 4:
        board2 = reverse(board)
        return m_move_down(board, board2)
    else:
        print(f'bad direction input:{direction}')
        return board


@njit(nogil=True)
def m_move_board2(board: np.uint64, board2: np.uint64, direction: int) -> np.uint64:
    if direction == 1:
        return m_move_left(board)
    elif direction == 2:
        return m_move_right(board)
    elif direction == 3:
        return m_move_up(board, board2)
    else:  # direction == 4:
        return m_move_down(board, board2)


@njit(nogil=True)
def m_move_all_dir(board: np.uint64
                   ) -> Tuple[Tuple[uint64, bool], Tuple[uint64, bool], Tuple[uint64, bool], Tuple[uint64, bool]]:
    board = np.uint64(board)
    board2 = reverse(board)
    md = m_move_down(board, board2)
    mr = m_move_right(board)
    ml, mnt_h = m_move_left2(board)
    mu, mnt_v = m_move_up2(board, board2)
    return (ml, mnt_h), (mr, mnt_h), (mu, mnt_v), (md, mnt_v)


@njit(nogil=True)
def m_move_all_dir2(board: np.uint64, board2: np.uint64
                    ) -> Tuple[Tuple[uint64, bool], Tuple[uint64, bool], Tuple[uint64, bool], Tuple[uint64, bool]]:
    md = m_move_down(board, board2)
    mr = m_move_right(board)
    ml, mnt_h = m_move_left2(board)
    mu, mnt_v = m_move_up2(board, board2)
    return (ml, mnt_h), (mr, mnt_h), (mu, mnt_v), (md, mnt_v)


if __name__ == "__main__":
    print(np.sum(mask_new_tiles))
    b = np.uint64(0x000101020aa37bcf)
    b_r = np.uint64(reverse(b))
    print(decode_board(b))
    print()
    for nt, _mnt in m_move_all_dir(b):
        print(decode_board(nt))
        print(_mnt)

    for nt, _mnt in m_move_all_dir2(b, b_r):
        print(decode_board(nt))
        print(_mnt)
