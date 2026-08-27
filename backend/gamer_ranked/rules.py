from __future__ import annotations

import math
from typing import Iterable

from .prng import Xoshiro128StarStar


DIRECTIONS = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
}
SPAWN_RATE4 = 0.1


def line_positions(direction: str) -> list[list[int]]:
    if direction == "left":
        return [[row * 4 + column for column in range(4)] for row in range(4)]
    if direction == "right":
        return [[row * 4 + column for column in range(3, -1, -1)] for row in range(4)]
    if direction == "up":
        return [[row * 4 + column for row in range(4)] for column in range(4)]
    if direction == "down":
        return [[row * 4 + column for row in range(3, -1, -1)] for column in range(4)]
    return []


def simulate_line(values: Iterable[int]) -> tuple[list[int], int]:
    source = [int(value) for value in values if int(value) != 0]
    result = [0, 0, 0, 0]
    score_delta = 0
    read = 0
    write = 0
    while read < len(source):
        value = source[read]
        if read + 1 < len(source) and source[read + 1] == value:
            value *= 2
            score_delta += value
            read += 2
        else:
            read += 1
        result[write] = value
        write += 1
    return result, score_delta


def simulate_move(board: list[int], direction: str) -> tuple[list[int], int]:
    next_board = [0] * 16
    score_delta = 0
    for positions in line_positions(direction):
        line, line_score = simulate_line(board[index] for index in positions)
        score_delta += line_score
        for offset, board_index in enumerate(positions):
            next_board[board_index] = line[offset]
    return next_board, score_delta


def legal_moves(board: list[int]) -> list[str]:
    return [
        direction
        for direction in ("left", "right", "up", "down")
        if simulate_move(board, direction)[0] != board
    ]


def random_spawn(
    board: list[int],
    rng: Xoshiro128StarStar,
    spawn_rate4: float = SPAWN_RATE4,
) -> tuple[int, int]:
    empty = [index for index, value in enumerate(board) if value == 0]
    if not empty:
        raise ValueError("Cannot spawn on a full board.")
    rate4 = float(spawn_rate4)
    if not math.isfinite(rate4) or not 0.0 <= rate4 <= 1.0:
        raise ValueError("Invalid 4-spawn rate.")
    index = empty[rng.choose_index(len(empty))]
    exponent = 2 if rng.next_float() < rate4 else 1
    return index, exponent


def initial_board(
    seed_hex: str,
    spawn_rate4: float = SPAWN_RATE4,
) -> tuple[list[int], tuple[tuple[int, int], ...], Xoshiro128StarStar]:
    rng = Xoshiro128StarStar.from_seed_hex(seed_hex)
    board = [0] * 16
    initial_tiles: list[tuple[int, int]] = []
    for _ in range(2):
        index, exponent = random_spawn(board, rng, spawn_rate4)
        board[index] = 2**exponent
        initial_tiles.append((index, exponent - 1))
    return board, tuple(initial_tiles), rng


def board_codes(board: list[int]) -> list[int]:
    codes: list[int] = []
    for value in board:
        if value == 0:
            codes.append(0)
            continue
        exponent = int(math.log2(value))
        if 2**exponent != value or exponent < 1 or exponent > 31:
            raise ValueError("Ranked board contains an unsupported tile.")
        codes.append(exponent)
    return codes


def packed_native_board(board: list[int]) -> int:
    encoded = 0
    for index, value in enumerate(board[:16]):
        exponent = 0 if value <= 0 else min(15, int(math.log2(value)))
        encoded |= (exponent & 0xF) << ((15 - index) * 4)
    return encoded


def evil_spawn(board: list[int], *, depth: int = 5) -> tuple[int, int]:
    from native_core.ai_core import EvilGen

    encoded = packed_native_board(board)
    generator = EvilGen(encoded)
    result = generator.gen_new_num(depth)
    index = int(result[1])
    exponent = int(result[2])
    if index < 0 or index >= 16 or exponent not in (1, 2) or board[index] != 0:
        raise ValueError("EvilGen returned an invalid ranked spawn.")
    return index, exponent
