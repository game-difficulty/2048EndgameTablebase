from __future__ import annotations

import asyncio
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Config import DTYPE_CONFIG, category_info
from engine_core.BoardMover import s_move_board as classic_move_board
from engine_core.BookReader import BookReaderDispatcher
from engine_core.VBoardMover import (
    decode_board,
    encode_board,
    s_move_board as variant_move_board,
)

from backend.remote_workers.registry import remote_worker_registry
from backend.tablebase_catalog import (
    TABLE_EXTENSIONS,
    build_filepath_map_entry,
    resolve_tablebase,
)

from .route_codec import MAX_ROUTE_STEPS, RATE_SCALE, RouteStep, encode_changes, encode_route


TIE_ORDER = ("left", "right", "down", "up")
DIRECTION_QUERY_ORDER = ("up", "down", "left", "right")
MOVE_CODES = {"left": 1, "right": 2, "up": 3, "down": 4}
RANDOM_START_ATTEMPTS = 16


class BattleRouteGenerationError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = str(code or "ROUTE_GENERATION_FAILED")
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class GeneratedBattleRoute:
    route_blob: bytes
    step_count: int
    certainty_step: int | None
    termination_reason: str
    initial_board: int
    available_layers: int


def _available_layer_count(path: Path, full_pattern: str) -> int:
    suffixes = "|".join(re.escape(suffix) for suffix in TABLE_EXTENSIONS)
    pattern = re.compile(
        rf"^{re.escape(full_pattern)}_(\d+)(?:{suffixes}|b)$",
        re.IGNORECASE,
    )
    layers: set[int] = set()
    try:
        for item in path.iterdir():
            match = pattern.fullmatch(item.name)
            if match and (item.is_file() or item.is_dir()):
                layers.add(int(match.group(1)))
    except OSError as exc:
        raise BattleRouteGenerationError(
            "TABLE_PATH_UNREADABLE", "Table layer metadata is unavailable."
        ) from exc
    return len(layers)


def _normalize_rate(value: Any, dtype: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    _, _, _, zero_value = DTYPE_CONFIG.get(dtype, DTYPE_CONFIG["uint32"])
    if float(zero_value) < 0:
        numeric += abs(float(zero_value))
    return max(0.0, min(1.0, numeric))


def _contains_target(board: int, target: int) -> bool:
    if target < 2 or target & (target - 1):
        return False
    target_exponent = target.bit_length() - 1
    value = int(board)
    return any(((value >> (index * 4)) & 0xF) == target_exponent for index in range(16))


def _spawn(board: int, *, spawn_rate: float, rng: random.Random) -> tuple[int, int, int]:
    decoded = decode_board(np.uint64(board)).copy()
    empty = [
        index
        for index, value in enumerate(decoded.reshape(-1).tolist())
        if int(value) == 0
    ]
    if not empty:
        raise BattleRouteGenerationError("NO_SPAWN_CELL", "Moved board has no empty cell.")
    spawn_index = empty[rng.randrange(len(empty))]
    spawn_exponent = 2 if rng.random() < float(spawn_rate) else 1
    row, column = divmod(spawn_index, 4)
    decoded[row, column] = 1 << spawn_exponent
    return int(encode_board(decoded)), spawn_index, spawn_exponent


def _generate_once(
    reader: BookReaderDispatcher,
    *,
    pattern: str,
    target: int,
    full_pattern: str,
    initial_board: int,
    route_limit: int,
    spawn_rate: float,
    seed: int,
    available_layers: int,
) -> GeneratedBattleRoute:
    use_variant = pattern in category_info.get("variant", [])
    move_board = variant_move_board if use_variant else classic_move_board
    board = int(initial_board) & 0xFFFF_FFFF_FFFF_FFFF
    steps: list[RouteStep] = []
    rng = random.Random(seed)
    certainty_step: int | None = None
    termination_reason = "max_steps"

    if _contains_target(board, target):
        return GeneratedBattleRoute(
            encode_route(board, ()), 0, 0, "target_reached", board, available_layers
        )

    for step_index in range(route_limit):
        raw_results, dtype = reader.move_on_dic(
            decode_board(np.uint64(board)), pattern, str(target), full_pattern
        )
        rates_by_direction = {
            direction: _normalize_rate(raw_results.get(direction), str(dtype or "uint32"))
            for direction in DIRECTION_QUERY_ORDER
        }
        moved_boards: dict[str, int] = {}
        for direction, move_code in MOVE_CODES.items():
            moved, _score = move_board(np.uint64(board), move_code)
            moved_value = int(moved)
            if moved_value != board:
                moved_boards[direction] = moved_value
        if not moved_boards:
            termination_reason = "no_legal_move"
            break

        best_direction: str | None = None
        best_rate = -1.0
        for direction in TIE_ORDER:
            rate = rates_by_direction[direction]
            if direction in moved_boards and rate > best_rate:
                best_direction = direction
                best_rate = rate
        if best_direction is None or best_rate <= 0:
            termination_reason = "zero_success"
            break
        if certainty_step is None and best_rate >= 1.0:
            certainty_step = step_index

        next_board, spawn_index, spawn_exponent = _spawn(
            moved_boards[best_direction], spawn_rate=spawn_rate, rng=rng
        )
        encoded_rates = tuple(
            max(0, min(RATE_SCALE, round(rates_by_direction[name] * RATE_SCALE)))
            for name in DIRECTION_QUERY_ORDER
        )
        steps.append(
            RouteStep(
                encode_changes(best_direction, spawn_index, 1 << spawn_exponent),
                encoded_rates,
            )
        )
        board = next_board
        if _contains_target(board, target):
            termination_reason = "target_reached"
            break

    return GeneratedBattleRoute(
        encode_route(initial_board, steps),
        len(steps),
        certainty_step,
        termination_reason,
        int(initial_board),
        available_layers,
    )


def _generate_local(
    *,
    pattern: str,
    target: int,
    full_pattern: str,
    initial_board: int | None,
    max_steps: int | None,
    spawn_rate: float,
    seed_hex: str,
) -> GeneratedBattleRoute:
    path_list = build_filepath_map_entry(full_pattern, spawn_rate)
    if not path_list:
        raise BattleRouteGenerationError("TABLE_UNAVAILABLE", "Tablebase is unavailable.")
    table_path = Path(path_list[0][0])
    available_layers = _available_layer_count(table_path, full_pattern)
    if available_layers <= 0:
        raise BattleRouteGenerationError("LAYERS_UNAVAILABLE", "No table layers are available.")
    route_limit = min(MAX_ROUTE_STEPS, available_layers)
    if max_steps is not None:
        route_limit = min(route_limit, int(max_steps))
    minimum_steps = int(math.ceil(min(target // 2, available_layers) * 0.6))
    if minimum_steps > route_limit:
        raise BattleRouteGenerationError(
            "ROUTE_TOO_SHORT", "The selected maximum step count is below the minimum route length."
        )

    reader = BookReaderDispatcher()
    reader.dispatch(path_list, pattern, target)
    base_seed = int(seed_hex, 16)
    attempts = 1 if initial_board is not None else RANDOM_START_ATTEMPTS
    best: GeneratedBattleRoute | None = None
    for attempt in range(attempts):
        candidate = (
            int(initial_board)
            if initial_board is not None
            else int(reader.get_random_state(path_list, full_pattern))
        )
        generated = _generate_once(
            reader,
            pattern=pattern,
            target=target,
            full_pattern=full_pattern,
            initial_board=candidate,
            route_limit=route_limit,
            spawn_rate=spawn_rate,
            seed=(base_seed + attempt) & ((1 << 128) - 1),
            available_layers=available_layers,
        )
        if best is None or generated.step_count > best.step_count:
            best = generated
        if generated.step_count >= minimum_steps:
            return generated
    raise BattleRouteGenerationError(
        "ROUTE_TOO_SHORT",
        "The initial board could not produce a sufficiently long route.",
    )


async def generate_battle_route(
    *,
    pattern: str,
    target: int,
    full_pattern: str,
    initial_board: int | None,
    max_steps: int | None,
    seed_hex: str,
) -> GeneratedBattleRoute:
    entry = resolve_tablebase(full_pattern)
    if entry is None:
        raise BattleRouteGenerationError("TABLE_UNAVAILABLE", "Tablebase is unavailable.")
    spawn_rate = float(entry.get("spawn_rate", 0.1))
    if entry.get("_provider") == "remote":
        response = await remote_worker_registry.generate_battle_route(
            full_pattern=full_pattern,
            pattern=pattern,
            target=str(target),
            initial_board=(None if initial_board is None else f"{initial_board:016x}"),
            max_steps=max_steps,
            min_steps=0,
            spawn_rate=spawn_rate,
            seed_hex=seed_hex,
        )
        return GeneratedBattleRoute(
            route_blob=bytes(response["route_blob"]),
            step_count=int(response["step_count"]),
            certainty_step=(
                None if response.get("certainty_step") is None else int(response["certainty_step"])
            ),
            termination_reason=str(response["termination_reason"]),
            initial_board=int(str(response["initial_board"]), 16),
            available_layers=int(response["available_layers"]),
        )
    return await asyncio.to_thread(
        _generate_local,
        pattern=pattern,
        target=target,
        full_pattern=full_pattern,
        initial_board=initial_board,
        max_steps=max_steps,
        spawn_rate=spawn_rate,
        seed_hex=seed_hex,
    )
