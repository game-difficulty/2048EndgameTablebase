from __future__ import annotations

import numpy as np

from ai_and_sort import mover_core as _mover_core


class _MoverFacade:
    def __init__(self, impl):
        self._impl = impl

    def move_board(self, board, direction):
        return np.uint64(self._impl.move_board(np.uint64(board), int(direction)))

    def move_all_dir(self, board):
        return tuple(np.uint64(value) for value in self._impl.move_all_dir(np.uint64(board)))

    def s_move_board(self, board, direction):
        result = self._impl.s_move_board(np.uint64(board), int(direction))
        if hasattr(result, "board"):
            return np.uint64(result.board), int(result.score)
        return np.uint64(result[0]), int(result[1])

    def s_move_board_all(self, board):
        return self._impl.s_move_board_all(np.uint64(board))

    def move_left(self, board):
        return np.uint64(self._impl.move_left(np.uint64(board)))

    def move_right(self, board):
        return np.uint64(self._impl.move_right(np.uint64(board)))

    def move_up(self, board):
        return np.uint64(self._impl.move_up(np.uint64(board)))

    def move_down(self, board):
        return np.uint64(self._impl.move_down(np.uint64(board)))

    def s_move_left(self, board):
        result = self._impl.s_move_left(np.uint64(board))
        return np.uint64(result[0]), int(result[1])

    def s_move_right(self, board):
        result = self._impl.s_move_right(np.uint64(board))
        return np.uint64(result[0]), int(result[1])

    def s_move_up(self, board):
        result = self._impl.s_move_up(np.uint64(board))
        return np.uint64(result[0]), int(result[1])

    def s_move_down(self, board):
        result = self._impl.s_move_down(np.uint64(board))
        return np.uint64(result[0]), int(result[1])


def encode_board(board):
    return np.uint64(_mover_core.encode_board(np.asarray(board, dtype=np.int64).tolist()))


def decode_board(board):
    return np.asarray(_mover_core.decode_board(np.uint64(board)), dtype=np.int32)


def reverse(board):
    return np.uint64(_mover_core.reverse(np.uint64(board)))


def gen_new_num(board, p: float = 0.1):
    state, count = _mover_core.gen_new_num(np.uint64(board), float(p))
    return np.uint64(state), int(count)


def s_gen_new_num(board, p: float = 0.1):
    state, count, pos, value = _mover_core.s_gen_new_num(np.uint64(board), float(p))
    return np.uint64(state), int(count), int(pos), int(value)


def _canonical_uint64(func):
    def _wrapped(board):
        return np.uint64(func(np.uint64(board)))
    return _wrapped


def _canonical_pair(func):
    def _wrapped(board):
        key, symm = func(np.uint64(board))
        return np.uint64(key), int(symm)
    return _wrapped


canonical_identity = _canonical_uint64(_mover_core.canonical_identity)
canonical_diagonal = _canonical_uint64(_mover_core.canonical_diagonal)
canonical_full = _canonical_uint64(_mover_core.canonical_full)
canonical_horizontal = _canonical_uint64(_mover_core.canonical_horizontal)
canonical_min33 = _canonical_uint64(_mover_core.canonical_min33)
canonical_min24 = _canonical_uint64(_mover_core.canonical_min24)
canonical_min34 = _canonical_uint64(_mover_core.canonical_min34)

canonical_identity_pair = _canonical_pair(_mover_core.canonical_identity_pair)
canonical_diagonal_pair = _canonical_pair(_mover_core.canonical_diagonal_pair)
canonical_full_pair = _canonical_pair(_mover_core.canonical_full_pair)
canonical_horizontal_pair = _canonical_pair(_mover_core.canonical_horizontal_pair)

std = _MoverFacade(_mover_core.std)
v = _MoverFacade(_mover_core.v)