from typing import List, Tuple

import numpy as np
from numba import njit


def create_pattern_func_source(name, masks):
    """生成 is_xx_pattern 函数的源代码字符串"""
    func_name = f"is_{name}_pattern"

    # 无掩码，直接返回True
    if not masks:
        body = "True"
    else:
        # 构建 (board & mask) == mask 逻辑链
        conditions = []
        for m in masks:
            m_val = f"np.uint64({m})"
            conditions.append(f"(np.uint64(encoded_board) & {m_val}) == {m_val}")
        body = " or \\\n           ".join(conditions)

    source = f"def {func_name}(encoded_board):\n    return {body}"
    return func_name, source


def create_success_func_source(name, shifts):
    """生成 is_xx_success 函数的源代码字符串"""
    func_name = f"is_{name}_success"

    # 构建 (board >> shift & 0xf) == target 逻辑链
    conditions = []
    for s in shifts:
        s_val = f"np.uint64({s})"
        conditions.append(f"((np.uint64(encoded_board) >> {s_val}) & np.uint64(0xf)) == np.uint64(target)")

    body = " or \\\n           ".join(conditions)
    source = f"def {func_name}(encoded_board, target):\n    return {body}"
    return func_name, source


def update_logic_functions():
    """
    根据当前的 PATTERN_DATA 重新生成函数并注入到本模块的全局空间
    """
    numba_dec = njit(nogil=True, inline='always')

    for name, (masks, shifts) in PATTERN_DATA.items():
        # 获取本模块的 globals()
        target_globals = globals()

        # 生成模式匹配函数
        p_name, p_code = create_pattern_func_source(name, masks)
        l_scope = {'np': np}
        exec(p_code, l_scope)
        target_globals[p_name] = numba_dec(l_scope[p_name])

        # 生成成功检测函数
        s_name, s_code = create_success_func_source(name, shifts)
        l_scope = {'np': np}
        exec(s_code, l_scope)
        target_globals[s_name] = numba_dec(l_scope[s_name])


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
def canonical_full(board):
    return np.uint64(min(ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board),
                         Rotate180(board), RotateL(board), RotateR(board), np.uint64(board)))


@njit(nogil=True, inline='always')
def canonical_diagonal(bd):
    board = np.uint64(bd)
    board = (board & np.uint64(0xff00ff0000ff00ff)) | (board & np.uint64(0x00ff00ff00000000)) >> np.uint64(24) | (
            board & np.uint64(0x00000000ff00ff00)) << np.uint64(24)
    board = (board & np.uint64(0xf0f00f0ff0f00f0f)) | (board & np.uint64(0x0f0f00000f0f0000)) >> np.uint64(12) | (
            board & np.uint64(0x0000f0f00000f0f0)) << np.uint64(12)
    return min(bd, board)


@njit(nogil=True, inline='always')
def canonical_horizontal(board):
    return np.uint64(min(ReverseLR(board), board))


@njit(nogil=True, inline='always')
def canonical_identity(encoded_board):
    return np.uint64(encoded_board)


@njit(nogil=True, inline='always')
def canonical_diagonal_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    board = ReverseUL(bd1)
    if bd1 <= board:
        return bd1, 0
    else:
        return board, 3


@njit(nogil=True, inline='always')
def canonical_horizontal_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    board = ReverseLR(bd1)
    if bd1 <= board:
        return bd1, 0
    else:
        return board, 1


@njit(nogil=True)
def canonical_full_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
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
def canonical_identity_pair(bd1: np.uint64) -> Tuple[np.uint64, int]:
    return bd1, 0


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
def canonical_min33(board):
    return np.uint64(min(exchange_col02(board), exchange_row02(board), R90_33(board), L90_33(board),
                         R180_33(board), UR_33(board), UL_33(board), board))


@njit(nogil=True)
def canonical_min24(board):
    return min(ReverseUD(board), ReverseLR(board), Rotate180(board), board)


@njit(nogil=True)
def canonical_min34(board):
    return min(ReverseLR(board), ReverseUD34(board), Rotate18034(board), board)


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
    x,y=canonical_full_pair(np.uint64(0x1a0b00237970df21))
    print(x,y)

    print(globals())
