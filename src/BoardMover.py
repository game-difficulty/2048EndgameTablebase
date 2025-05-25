from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from numba import uint64, uint16, njit
from numba.experimental import jitclass

from Variants.vBoardMover import (VBoardMoverWithScore, VBoardMover,
                                  reverse, encode_board, encode_row, decode_board, decode_row)


@njit()
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


@njit()
def calculate_all_moves() -> Tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
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


_movel, _mover, _moveu, _moved, _score = calculate_all_moves()
_movel = _movel.astype(np.uint16)
_mover = _mover.astype(np.uint16)


spec = {
    'movel': uint16[:],  # 表示一个uint16类型的一维数组
    'mover': uint16[:],
    'moveu': uint64[:],
    'moved': uint64[:],
}


@jitclass(spec)
class BoardMover:
    def __init__(self):
        self.movel, self.mover, self.moveu, self.moved = _movel, _mover, _moveu, _moved
        print('BoardMover init')

    @staticmethod
    def encode_board(board: np.typing.NDArray) -> np.uint64:
        return encode_board(board)

    @staticmethod
    def decode_board(encoded_board: np.uint64) -> np.typing.NDArray:
        return decode_board(encoded_board)

    @staticmethod
    def encode_row(row: np.typing.NDArray) -> np.uint64:
        return encode_row(row)

    @staticmethod
    def decode_row(encoded: np.uint64) -> np.typing.NDArray:
        return decode_row(encoded)

    @staticmethod
    def reverse(board: np.uint64) -> np.uint64:
        return reverse(board)

    def move_left(self, board: np.uint64) -> np.uint64:
        board ^= self.movel[board & np.uint64(0xffff)]
        board ^= self.movel[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
        board ^= self.movel[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
        board ^= self.movel[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
        return board

    def move_right(self, board: np.uint64) -> np.uint64:
        board ^= self.mover[board & np.uint64(0xffff)]
        board ^= self.mover[board >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(16)
        board ^= self.mover[board >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(32)
        board ^= self.mover[board >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(48)
        return board

    def move_up(self, board: np.uint64, board2: np.uint64) -> np.uint64:
        board ^= self.moveu[board2 & np.uint64(0xffff)]
        board ^= self.moveu[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
        board ^= self.moveu[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
        board ^= self.moveu[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
        return board

    def move_down(self, board: np.uint64, board2: np.uint64) -> np.uint64:
        board ^= self.moved[board2 & np.uint64(0xffff)]
        board ^= self.moved[board2 >> np.uint64(16) & np.uint64(0xffff)] << np.uint64(4)
        board ^= self.moved[board2 >> np.uint64(32) & np.uint64(0xffff)] << np.uint64(8)
        board ^= self.moved[board2 >> np.uint64(48) & np.uint64(0xffff)] << np.uint64(12)
        return board

    def move_board(self, board: np.uint64, direction: int) -> np.uint64:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            board2 = self.reverse(board)
            return self.move_up(board, board2)
        elif direction == 4:
            board2 = self.reverse(board)
            return self.move_down(board, board2)
        else:
            print(f'bad direction input:{direction}')
            return board

    def move_board2(self, board: np.uint64, board2: np.uint64, direction: int) -> np.uint64:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            return self.move_up(board, board2)
        else:  # direction == 4:
            return self.move_down(board, board2)

    def move_all_dir(self, board: np.uint64) -> Tuple[uint64, uint64, uint64, uint64]:
        board = np.uint64(board)
        board2 = self.reverse(board)
        return (
            self.move_left(board), self.move_right(board), self.move_up(board, board2), self.move_down(board, board2))

    @staticmethod
    def gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int]:
        empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
        if not empty_slots:
            return t, 0  # 如果没有空位，返回原面板
        i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
        val = 2 if np.random.random() < p else 1  # 生成2或4，其中2的概率为0.9
        t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
        return t, len(empty_slots)


spec2 = {
    'movel': uint16[:],
    'mover': uint16[:],
    'moveu': uint64[:],
    'moved': uint64[:],
    'score': uint64[:],
}


@jitclass(spec2)
class BoardMoverWithScore:
    """额外统计得分，其他都一样"""
    def __init__(self):
        self.movel, self.mover, self.moveu, self.moved, self.score = _movel, _mover, _moveu, _moved, _score
        print('BoardMover init')

    @staticmethod
    def encode_board(board: np.typing.NDArray) -> np.uint64:
        return encode_board(board)

    @staticmethod
    def decode_board(encoded_board: np.uint64) -> np.typing.NDArray:
        return decode_board(encoded_board)

    @staticmethod
    def encode_row(row: np.typing.NDArray) -> np.uint64:
        return encode_row(row)

    @staticmethod
    def decode_row(encoded: np.uint64) -> np.typing.NDArray:
        return decode_row(encoded)

    @staticmethod
    def reverse(board: np.uint64) -> np.uint64:
        return reverse(board)

    def move_left(self, board: np.uint64) -> Tuple[np.uint64, np.uint32]:
        total_score = 0
        for i in range(4):
            line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.movel[line] << np.uint64(16 * i)
        return board, total_score

    def move_right(self, board: np.uint64) -> Tuple[np.uint64, np.uint32]:
        total_score = 0
        for i in range(4):
            line = (board >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.mover[line] << np.uint64(16 * i)
        return board, total_score

    def move_up(self, board: np.uint64, board2: np.uint64) -> Tuple[np.uint64, np.uint32]:
        total_score = 0
        for i in range(4):
            line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.moveu[line] << np.uint64(4 * i)
        return board, total_score

    def move_down(self, board: np.uint64, board2: np.uint64) -> Tuple[np.uint64, np.uint32]:
        total_score = 0
        for i in range(4):
            line = (board2 >> np.uint64(16 * i)) & np.uint64(0xFFFF)
            total_score += self.score[line]
            board ^= self.moved[line] << np.uint64(4 * i)
        return board, total_score

    def move_board(self, board: np.uint64, direction: int) -> Tuple[np.uint64, np.uint32]:
        if direction == 1:
            return self.move_left(board)
        elif direction == 2:
            return self.move_right(board)
        elif direction == 3:
            board2 = self.reverse(board)
            return self.move_up(board, board2)
        elif direction == 4:
            board2 = self.reverse(board)
            return self.move_down(board, board2)
        else:
            print(f'bad direction input:{direction}')
            return board, np.uint32(0)

    def move_all_dir(self, board: np.uint64) -> Tuple[np.uint64, np.uint64, np.uint64, np.uint64]:
        board = np.uint64(board)
        board2 = self.reverse(board)
        return (
            self.move_left(board)[0],  self.move_right(board)[0],
            self.move_up(board, board2)[0], self.move_down(board, board2)[0])

    @staticmethod
    def gen_new_num(t: np.uint64, p: float = 0.1) -> Tuple[np.uint64, int, int, int]:
        empty_slots = [i for i in range(16) if ((t >> np.uint64(4 * i)) & np.uint64(0xF)) == 0]  # 找到所有空位
        if not empty_slots:
            return t, 0, 0, 0  # 如果没有空位，返回原面板
        i = int(np.random.choice(np.array(empty_slots)))  # 随机选择一个空位
        val = 2 if np.random.random() < p else 1  # 生成2或4，其中2的概率为0.9
        t |= np.uint64(val) << np.uint64(4 * i)  # 在选中的位置放置新值
        return t, len(empty_slots), 15 - i, val


class SingletonBoardMover:
    _bm = None      # 对应 bm_type = 1
    _bmws = None    # 对应 bm_type = 2
    _vbm = None     # 对应 bm_type = 3
    _vbmws = None   # 对应 bm_type = 4

    def __new__(cls, bm_type, *args, **kwargs):
        if bm_type == 1:
            if cls._bm is None:
                cls._bm = BoardMover()
            return cls._bm
        elif bm_type == 2:
            if cls._bmws is None:
                cls._bmws = BoardMoverWithScore()
            return cls._bmws
        elif bm_type == 3:
            if cls._vbm is None:
                cls._vbm = VBoardMover()
            return cls._vbm
        elif bm_type == 4:
            if cls._vbmws is None:
                cls._vbmws = VBoardMoverWithScore()
            return cls._vbmws
        else:
            raise ValueError("Invalid board_mover_type. Expected 1, 2, 3, or 4.")


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
    bm = BoardMover()
    r = bm.move_all_dir(bm.encode_board(b))
    print(b)
    for rb, d in zip(r, ('d', 'r', 'l', 'u')):
        print(d)
        print(bm.decode_board(rb))

    b = np.array([[32, 8, 0, 2],
                  [32, 32, 32, 32],
                  [64, 16, 4, 16],
                  [16384, 4096, 0, 4096]])
    bm = BoardMoverWithScore()
    r = bm.move_all_dir(bm.encode_board(b))
    print(b)
    for rb, d in zip(r, ('d', 'r', 'l', 'u')):
        print(d)
        print(bm.decode_board(rb))
