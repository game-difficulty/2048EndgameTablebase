from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import uint64, njit

from Variants.vBoardMover import (reverse, encode_board, encode_row, decode_board, decode_row)


@njit(cache=True)
def merge_line_with_score(line: np.typing.NDArray, reverse_line: bool = False) -> Tuple[np.typing.NDArray, np.uint64]:
    if reverse_line:
        line = line[::-1]
    non_zero = [i for i in line if i != 0]  # 去掉所有的0
    merged = []
    score = 0
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1] and non_zero[i] != 32768:
            merged_value = 2 * non_zero[i]
            score += merged_value
            merged.append(merged_value)
            skip = True
        else:
            merged.append(non_zero[i])

    # 补齐剩下的 0
    merged += [0] * (len(line) - len(merged))
    if reverse_line:
        merged = merged[::-1]
    return np.array(merged), score


@njit(cache=True)
def calculate_all_moves() -> Tuple[
    np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    # 初始化存储所有可能的行及其移动后结果差值的字典
    movel = np.empty(65536, dtype=np.uint64)
    mover = np.empty(65536, dtype=np.uint64)
    moveu = np.empty(65536, dtype=np.uint64)
    moved = np.empty(65536, dtype=np.uint64)
    score = np.empty(65536, dtype=np.uint64)
    # 生成所有可能的行
    for i in range(16 ** 4):
        line = [(i // (16 ** j)) % 16 for j in range(4)]
        line = np.array([2 ** k if k else 0 for k in line])  # 把游戏中的数字转换成2的幂
        original_line = encode_row(line)  # 编码原始行为整数

        # 向左移动
        merged_linel, s = merge_line_with_score(line, False)
        movel[original_line] = encode_row(merged_linel) ^ original_line

        # 向右移动
        merged_liner, s = merge_line_with_score(line, True)
        mover[original_line] = encode_row(merged_liner) ^ original_line

        score[original_line] = s
    # 使用reverse函数计算向上和向下的移动差值
    for i in range(16 ** 4):
        moveu[i] = reverse(movel[i])
        moved[i] = reverse(mover[i])

    return movel, mover, moveu, moved, score


movel, mover, moveu, moved, score = calculate_all_moves()
movel = movel.astype(np.uint16)
mover = mover.astype(np.uint16)


@njit(nogil=True, cache=True)
def move_left(board: np.uint64) -> np.uint64:
    board ^= movel[board & np.uint64(0xffff)]
    board ^= movel[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
    board ^= movel[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
    board ^= movel[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
    return board


@njit(nogil=True, cache=True)
def move_right(board: np.uint64) -> np.uint64:
    board ^= mover[board & np.uint64(0xffff)]
    board ^= mover[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
    board ^= mover[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
    board ^= mover[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
    return board


@njit(nogil=True, cache=True)
def move_up(board: np.uint64, board2: np.uint64) -> np.uint64:
    board ^= moveu[board2 & np.uint64(0xffff)]
    board ^= moveu[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
    board ^= moveu[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
    board ^= moveu[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
    return board


@njit(nogil=True, cache=True)
def move_down(board: np.uint64, board2: np.uint64) -> np.uint64:
    board ^= moved[board2 & np.uint64(0xffff)]
    board ^= moved[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
    board ^= moved[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
    board ^= moved[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
    return board


@njit(nogil=True, cache=True)
def move_board(board: np.uint64, direction: int) -> np.uint64:
    if direction == 1:
        return move_left(board)
    elif direction == 2:
        return move_right(board)
    elif direction == 3:
        board2 = reverse(board)
        return move_up(board, board2)
    elif direction == 4:
        board2 = reverse(board)
        return move_down(board, board2)
    else:
        print(f'bad direction input:{direction}')
        return board


@njit(nogil=True, cache=True)
def move_board2(board: np.uint64, board2: np.uint64, direction: int) -> np.uint64:
    if direction == 1:
        return move_left(board)
    elif direction == 2:
        return move_right(board)
    elif direction == 3:
        return move_up(board, board2)
    else:  # direction == 4:
        return move_down(board, board2)


@njit(nogil=True, cache=True)
def move_all_dir(board: np.uint64) -> Tuple[uint64, uint64, uint64, uint64]:
    board = np.uint64(board)
    board2 = reverse(board)
    return (
        move_left(board), move_right(board), move_up(board, board2), move_down(board, board2))


@njit(nogil=True, cache=True)
def gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int]:
    empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
    if not empty_slots:
        return t, 0  # 如果没有空位，返回原面板
    i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
    val = 2 if np.random.random() < p else 1  # 生成2或4，其中2的概率为0.9
    t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
    return t, len(empty_slots)


@njit(nogil=True, cache=True)
def s_move_left(board: np.uint64) -> Tuple[np.uint64, np.uint32]:
    total_score = np.uint64(0)
    for i in range(4):
        line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
        total_score += score[line]
        board ^= np.uint64(movel[line]) << np.uint64(16 * i)
    return board, total_score


@njit(nogil=True, cache=True)
def s_move_right(board: np.uint64) -> Tuple[np.uint64, np.uint32]:
    total_score = np.uint64(0)
    for i in range(4):
        line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
        total_score += score[line]
        board ^= np.uint64(mover[line]) << np.uint64(16 * i)
    return board, total_score


@njit(nogil=True, cache=True)
def s_move_up(board: np.uint64, board2: np.uint64) -> Tuple[np.uint64, np.uint32]:
    total_score = np.uint64(0)
    for i in range(4):
        line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
        total_score += score[line]
        board ^= np.uint64(moveu[line]) << np.uint64(4 * i)
    return board, total_score


@njit(nogil=True, cache=True)
def s_move_down(board: np.uint64, board2: np.uint64) -> Tuple[np.uint64, np.uint32]:
    total_score = np.uint64(0)
    for i in range(4):
        line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
        total_score += score[line]
        board ^= np.uint64(moved[line]) << np.uint64(4 * i)
    return board, total_score


@njit(nogil=True, cache=True)
def s_move_board(board: np.uint64, direction: int) -> Tuple[np.uint64, np.uint32]:
    if direction == 1:
        return s_move_left(board)
    elif direction == 2:
        return s_move_right(board)
    elif direction == 3:
        board2 = reverse(board)
        return s_move_up(board, board2)
    elif direction == 4:
        board2 = reverse(board)
        return s_move_down(board, board2)
    else:
        print(f'bad direction input:{direction}')
        return board, np.uint32(0)


@njit(nogil=True, cache=True)
def s_gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int, int, int]:
    empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
    if not empty_slots:
        return t, 0, 0, 0  # 如果没有空位，返回原面板
    i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
    val = 2 if np.random.random() < p else 1  # 生成2或4，其中2的概率为0.9
    t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
    return t, len(empty_slots), 15 - i, val


# from numba import njit, threading_layer
# @njit(parallel=True)
# def foo(a, b):
#     return a + b
# x = np.arange(10.)
# y = x.copy()
# foo(x, y)
#
# # demonstrate the threading layer chosen
# print(f"Threading layer chosen: {threading_layer()}")


if __name__ == "__main__":
    b = np.array([[32, 8, 0, 2],
                  [32, 32, 32, 32],
                  [64, 16, 4, 16],
                  [16384, 4096, 0, 4096]])
    r = move_all_dir(encode_board(b))
    print(b)
    for rb, d in zip(r, ('l', 'r', 'u', 'd')):
        print(d)
        print(decode_board(rb))

    eb = encode_board(b)
    for d in (1, 2, 3, 4):
        print(d)
        print(decode_board(s_move_board(eb, d)[0]))
