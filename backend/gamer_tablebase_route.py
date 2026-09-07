"""Deterministic short-route rules shared by the cloud and table worker."""
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
    for name, low, high in [('steps', 1, 4), ('difficulty', 0, 100)]:
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
    if int(board.sum()) - count * 32768 < 24:
        board = np.array([32768,32768,32768,32768,0,32768,32768,0,
                          0,32768,32768,0,32768,32768,32768,32768])
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


def generate_route(options, large_tiles, lookup):
    request = SimpleNamespace(**validate_options(options))
    values = [0 if code == 0 else 2 ** code for code in request.board_codes]
    rng = Xoshiro128StarStar(request.rng_state.copy())
    nodes = []
    for index in range(request.steps):
        encoded = masked_board(values, large_tiles)
        results, dtype = lookup(encoded)
        nodes.append({'board_codes': [0 if v == 0 else v.bit_length()-1 for v in values],
                      'rng_state': rng.state.copy(), 'lookup_board': f'{encoded:016x}',
                      'random_only': request.random_only and index == 0,
                      'results': results, 'dtype': dtype})
        direction = next((k for k, v in results.items()
                          if isinstance(v, (int, float)) and math.isfinite(v)), None)
        if direction not in ('up', 'down', 'left', 'right'):
            break
        if results[direction] + (1 if dtype.startswith('1-') else 0) <= 0:
            break
        values = predict_next(values, direction, rng, request,
                              random_only=request.random_only and index == 0)
        if values is None:
            break
    return nodes
