"""Deterministic streaming-route rules shared by the cloud and table worker."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.gamer_ranked.rules import simulate_move, random_spawn
from engine_core.VBoardMover import encode_board


def validate_options(options):
    fields = {'board_codes', 'rng_state', 'spawn_rate4', 'difficulty', 'random_only', 'steps'}
    if not isinstance(options, dict) or set(options) != fields:
        raise ValueError('Invalid route fields')
    for name, size, maximum in [('board_codes', 16, 31), ('rng_state', 4, 0xffffffff)]:
        values = options[name]
        if not isinstance(values, list) or len(values) != size or any(
                type(v) is not int or not 0 <= v <= maximum for v in values):
            raise ValueError('Invalid route state')
    if not any(options['rng_state']):
        raise ValueError('Invalid route RNG')
    for name, low, high in [('steps', 1, 1), ('difficulty', 0, 100)]:
        if type(options[name]) is not int or not low <= options[name] <= high:
            raise ValueError('Invalid route limit')
    if type(options['random_only']) is not bool:
        raise ValueError('Invalid random mode')
    rate = options['spawn_rate4']
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Invalid spawn rate')
    return options


def masked_board(values, count):
    board = np.asarray(values, dtype=np.int64).copy()
    board[np.argpartition(board, -count)[-count:]] = 32768
    return int(encode_board(board.reshape(4, 4)))


def predict_next(values, direction, rng, request, *, random_only=None):
    moved, _ = simulate_move(values, direction)
    if moved == values or 0 not in moved:
        return None
    if not (request.random_only if random_only is None else random_only):
        branch = rng.next_float()
        if request.difficulty >= 100 or (request.difficulty > 0 and branch < request.difficulty / 100):
            return None
    index, exp = random_spawn(moved, rng, request.spawn_rate4)
    moved[index] = 2 ** exp
    return moved




class GamerRouteCursor:
    """One deterministic route, advanced without re-reading its previous node."""
    def __init__(self, options, large_tiles):
        self.request = SimpleNamespace(**validate_options(options))
        self.values = [0 if code == 0 else 2 ** code for code in self.request.board_codes]
        self.rng = Xoshiro128StarStar(self.request.rng_state.copy())
        self.large_tiles = large_tiles
        self.index = 0

    @property
    def encoded(self):
        return masked_board(self.values, self.large_tiles)

    def node(self, results, dtype, legal_moves_mask=None):
        if legal_moves_mask is None:
            legal_moves_mask = getattr(results, "legal_moves_mask", None)
        metadata = {} if legal_moves_mask is None else {"legal_moves_mask": legal_moves_mask}
        return {**metadata,'board_codes': [0 if v == 0 else v.bit_length()-1 for v in self.values],
                'rng_state': self.rng.state.copy(), 'lookup_board': f'{self.encoded:016x}',
                'random_only': self.request.random_only and self.index == 0,
                'results': results, 'dtype': dtype}

    def advance(self, results, dtype):
        direction = next((k for k, v in results.items()
                          if isinstance(v, (int, float)) and math.isfinite(v)), None)
        if direction not in ('up', 'down', 'left', 'right'):
            return False
        if results[direction] + (1 if dtype.startswith('1-') else 0) <= 0:
            return False
        values = predict_next(self.values, direction, self.rng, self.request,
                              random_only=self.request.random_only and self.index == 0)
        if values is None:
            return False
        self.values = values
        self.index += 1
        return True
