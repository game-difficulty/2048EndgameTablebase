from __future__ import annotations

import asyncio
import json
import math
import secrets
import sqlite3
import struct
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
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

from backend.auth.db import auth_db
from backend.quota.config import operation_cost_units
from backend.quota.service import (
    TokenReservation,
    cancel_reservation,
    finalize_reservation,
    get_token_balance,
    load_token_reservation,
    reserve_operation_tokens,
)
from backend.remote_workers.registry import remote_worker_registry
from backend.tablebase_catalog import (
    build_filepath_map_entry,
    get_catalog_version,
    resolve_tablebase,
)
from backend.tablebase_query_service import (
    TablebaseLookupResult,
    TablebaseLookupSpec,
    TablebaseQueryOverloaded,
    tablebase_query_scheduler,
)

from ... import repository
from ...core.errors import BattleServiceError
from ...core.lifecycle import (
    assert_member,
    broadcast_room,
    iso,
    room_by_user,
    room_snapshot,
    utcnow,
)
from ...core.registry import register_battle_mode
from ...replay_records import append_replay_step, encode_replay_step
from ..goodness.runtime import _ready_players_for_start
from .mode import FreeGoodnessBattleMode
from .rules import (
    DIRECTIONS,
    MOVE_RISK_LIMIT,
    RISK_EPSILON,
    SPAWN_DRAWDOWN_LIMIT,
    SPAWN_RISK_LIMIT,
    SpawnRiskState,
    accumulate_goodness,
    best_direction,
    clamp_probability,
    decide_move,
    deterministic_spawn_choice,
    deterministic_ticket,
    evaluate_spawn,
)


ROUND_LIFETIME = timedelta(minutes=60)
WAITING_LIFETIME = timedelta(minutes=30)
ACK_GRACE_SECONDS = 5
CORRECTION_WINDOW_SECONDS = 15
MAX_RANDOM_BOARD_ATTEMPTS = 16
MAX_PREPARED_STATES = 2048
MAX_CERTAINTY_TAIL_STEPS = 4096
CERTAINTY_TAIL_FALLBACK_CODES = frozenset({
    "CERTAINTY_TAIL_UNAVAILABLE",
    "CERTAINTY_TAIL_TOO_LONG",
})
MOVE_CODES = {"left": 1, "right": 2, "up": 3, "down": 4}
MOVE_CODE_BITS = {"left": 0, "right": 1, "up": 2, "down": 3}


@dataclass(frozen=True, slots=True)
class PreparedSpawn:
    executed_direction: str
    moved_board: int
    next_board: int
    spawn_index: int
    spawn_value: int
    attempt_index: int
    next_results: dict[str, float]
    next_dtype: str
    next_best_success: float
    risk_multiplier: float
    risk_state: SpawnRiskState


@dataclass(frozen=True, slots=True)
class CertaintyTailStep:
    previous_board: int
    next_board: int
    direction: str
    spawn_index: int
    spawn_value: int
    rates: dict[str, float]

    def public_payload(self) -> dict[str, Any]:
        return {
            "previous_board_hex": f"{self.previous_board:016x}",
            "board_hex": f"{self.next_board:016x}",
            "executed_direction": self.direction,
            "spawn_index": self.spawn_index,
            "spawn_value": self.spawn_value,
        }


_free_mode = register_battle_mode(FreeGoodnessBattleMode(), replace=True)
_free_mode.bind_runtime(__import__(__name__, fromlist=["*"]))
_reader_cache: dict[tuple[str, str], BookReaderDispatcher] = {}
_reader_lock = asyncio.Lock()
_prepared: OrderedDict[
    tuple[str, int, int, int, str], dict[str, PreparedSpawn | None]
] = OrderedDict()
_prepared_complete: set[tuple[str, int, int, int, str]] = set()
_prepare_state_tasks: dict[
    tuple[str, int, int, int, str], asyncio.Task[dict[str, PreparedSpawn]]
] = {}
_cleanup_task: asyncio.Task | None = None


def _use_variant(pattern: str) -> bool:
    return str(pattern) in category_info.get("variant", [])


def _contains_target(board: int, target: int) -> bool:
    if target < 2 or target & (target - 1):
        return False
    exponent = target.bit_length() - 1
    value = int(board)
    return any(((value >> (index * 4)) & 0xF) == exponent for index in range(16))


def _normalize_results(result: TablebaseLookupResult) -> dict[str, float]:
    _, _, _, zero_value = DTYPE_CONFIG.get(result.dtype, DTYPE_CONFIG["uint32"])
    offset = abs(float(zero_value)) if float(zero_value) < 0 else 0.0
    normalized: dict[str, float] = {}
    for direction in DIRECTIONS:
        value = result.results.get(direction)
        try:
            numeric = float(value) + offset
        except (TypeError, ValueError):
            numeric = 0.0
        normalized[direction] = clamp_probability(numeric)
    return normalized


def _moved_boards(board: int, *, use_variant: bool) -> dict[str, int]:
    move = variant_move_board if use_variant else classic_move_board
    moved: dict[str, int] = {}
    for direction, code in MOVE_CODES.items():
        next_board, _score = move(np.uint64(board), code)
        encoded = int(next_board)
        if encoded != int(board):
            moved[direction] = encoded
    return moved


def _spawn_board(board: int, index: int, value: int) -> int:
    decoded = decode_board(np.uint64(board)).copy()
    flat = decoded.reshape(-1)
    if int(flat[index]) != 0:
        raise ValueError("spawn_cell_occupied")
    flat[index] = int(value)
    return int(encode_board(decoded))


def _empty_indices(board: int) -> list[int]:
    decoded = decode_board(np.uint64(board)).reshape(-1)
    return [index for index, value in enumerate(decoded.tolist()) if int(value) == 0]


def _is_supporter(user_id: int) -> bool:
    with auth_db() as db:
        row = db.execute(
            "SELECT tier FROM user_entitlements WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
    return row is not None and str(row["tier"] or "free") == "supporter"


def _room_supporter(room: Any) -> bool:
    try:
        if str(room["billing_policy"] or "user") == "platform":
            return False
        user_id = room["host_user_id"]
    except (KeyError, TypeError, IndexError):
        return False
    return user_id is not None and _is_supporter(int(user_id))


async def _book_reader(full_pattern: str, pattern: str, target: int):
    entry = resolve_tablebase(full_pattern)
    if entry is None:
        raise BattleServiceError("TABLE_UNAVAILABLE", "Tablebase is unavailable.", 409)
    if entry.get("_provider") == "remote":
        return None, "remote", entry
    reader_key = (str(full_pattern), str(entry.get("_absolute_path") or ""))
    async with _reader_lock:
        reader = _reader_cache.get(reader_key)
        if reader is None:
            path_list = build_filepath_map_entry(
                full_pattern, float(entry.get("spawn_rate", 0.1))
            )
            if not path_list:
                raise BattleServiceError("TABLE_UNAVAILABLE", "Tablebase is unavailable.", 409)
            reader = BookReaderDispatcher()
            await asyncio.to_thread(reader.dispatch, path_list, pattern, target)
            _reader_cache[reader_key] = reader
    return reader, "local", entry


async def _lookup(
    room: dict[str, Any],
    board: int,
    *,
    stream_key: str,
    supporter: bool,
    lane: str,
) -> TablebaseLookupResult:
    reader, provider, _entry = await _book_reader(
        str(room["full_pattern"]), str(room["pattern"]), int(room["target"])
    )
    spec = TablebaseLookupSpec(
        board_encoded=int(board),
        pattern=str(room["pattern"]),
        target=str(room["target"]),
        full_pattern=str(room["full_pattern"]),
        use_variant=_use_variant(str(room["pattern"])),
        book_reader=reader,
        provider_kind=provider,
        catalog_version=get_catalog_version(),
    )
    handle = await tablebase_query_scheduler.submit(
        spec,
        stream_key=stream_key,
        supporter=supporter,
        lane=lane,
        supersede=False,
        allow_overload=lane != "foreground",
    )
    try:
        return await handle.wait()
    except asyncio.CancelledError:
        handle.cancel()
        raise


async def _random_board(room: dict[str, Any]) -> int:
    reader, provider, entry = await _book_reader(
        str(room["full_pattern"]), str(room["pattern"]), int(room["target"])
    )
    if provider == "remote":
        response = await remote_worker_registry.random_state(
            full_pattern=str(room["full_pattern"]),
            pattern=str(room["pattern"]),
            target=str(room["target"]),
        )
        value = response.get("board") or response.get("board_hex")
        if not isinstance(value, (str, int)):
            raise BattleServiceError("RANDOM_BOARD_FAILED", "Could not select a starting board.", 503)
        return int(value, 16) if isinstance(value, str) else int(value)
    path_list = build_filepath_map_entry(
        str(room["full_pattern"]), float(entry.get("spawn_rate", 0.1))
    )
    return int(await asyncio.to_thread(reader.get_random_state, path_list, room["full_pattern"]))


def _reservation_payload(reservation: TokenReservation) -> dict[str, int]:
    return {
        "reservation_ledger_id": reservation.ledger_id,
        "reserved_bonus_units": reservation.reserved_bonus_units,
        "reserved_paid_units": reservation.reserved_paid_units,
        "token_cost_units": reservation.reserved_units,
    }


def _restore_reservation(round_row: sqlite3.Row, _room_row: sqlite3.Row) -> TokenReservation | None:
    return load_token_reservation(round_row["reservation_ledger_id"])


def _insert_round(
    *,
    room_id: str,
    round_number: int,
    seed_hex: str,
    initial_board: int,
    score_step_limit: int,
    ranking_min_steps: int,
    reservation: TokenReservation | None,
) -> str:
    round_id = str(uuid.uuid4())
    now = iso()
    mode_state = {
        "initial_board": f"{int(initial_board):016x}",
        "score_step_limit": int(score_step_limit),
        "ranking_min_steps": int(ranking_min_steps),
        "lookup_hit_steps": 0,
        "lookup_miss_steps": 0,
        "rules_version": _free_mode.version,
    }
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_id)
        db.execute(
            """
            INSERT INTO battle_rounds
            (round_id, room_id, round_number, status, token_reservation_id,
             token_cost_units, created_at, updated_at, reservation_ledger_id,
             reserved_bonus_units, reserved_paid_units, route_seed,
             reservation_status, artifact_kind, mode_state_json)
            VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                room_id,
                int(round_number),
                str(reservation.ledger_id) if reservation is not None else None,
                reservation.reserved_units if reservation is not None else 0,
                now,
                now,
                reservation.ledger_id if reservation is not None else None,
                reservation.reserved_bonus_units if reservation is not None else 0,
                reservation.reserved_paid_units if reservation is not None else 0,
                seed_hex,
                "reserved" if reservation is not None else "not_required",
                _free_mode.artifact_kind,
                json.dumps(mode_state, separators=(",", ":"), sort_keys=True),
            ),
        )
        db.execute(
            """
            UPDATE battle_rooms SET current_round_number = ?, status = 'waiting',
                initial_board = ?, generation_error = NULL, expires_at = ?,
                revision = revision + 1, updated_at = ? WHERE room_id = ?
            """,
            (
                int(round_number),
                f"{int(initial_board):016x}",
                repository.PERMANENT_ROOM_EXPIRES_AT
                if str(room["lifecycle_kind"] or "normal") == "permanent"
                else iso(utcnow() + WAITING_LIFETIME),
                now,
                room_id,
            ),
        )
    return round_id


def _reserve_round_budget(
    *,
    user_id: int,
    session_id: int | None,
    full_pattern: str,
    max_players: int,
    target: int,
) -> TokenReservation:
    quantity = int(max_players) * (int(target) // 2 + 10)
    reservation = reserve_operation_tokens(
        user_id=user_id,
        session_id=session_id,
        operation_key="tester_lookup_hit",
        full_pattern=full_pattern,
        quantity=quantity,
    )
    if reservation is None:
        raise BattleServiceError("TOKEN_CONFIGURATION_ERROR", "Battle token cost is not configured.", 500)
    return reservation


async def _select_initial_board(room: dict[str, Any], requested: str | None) -> tuple[int, dict[str, float], str]:
    attempts = 1 if requested else MAX_RANDOM_BOARD_ATTEMPTS
    supporter = _room_supporter(room)
    last_board = int(requested, 16) if requested else 0
    for attempt in range(attempts):
        board = int(requested, 16) if requested else await _random_board(room)
        last_board = board
        result = await _lookup(
            room,
            board,
            stream_key=f"battle-free-initial:{room['room_id']}:{attempt}",
            supporter=supporter,
            lane="foreground",
        )
        normalized = _normalize_results(result)
        legal = _moved_boards(board, use_variant=_use_variant(str(room["pattern"])))
        _best, rate = best_direction(normalized, legal)
        if legal and rate > 0:
            return board, normalized, result.dtype
    raise BattleServiceError(
        "INITIAL_BOARD_UNUSABLE",
        f"Could not find a playable starting board after {attempts} attempts.",
        409,
        extra={"board_hex": f"{last_board:016x}"},
    )


async def validate_lobby_initial_board(room: dict[str, Any], board: int) -> str:
    selected, _results, _dtype = await _select_initial_board(room, f"{int(board):016x}")
    return f"{selected:016x}"


async def create_room_for_mode(
    *,
    user_id: int,
    session_id: int | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if room_by_user(user_id) is not None:
        raise BattleServiceError("USER_ALREADY_IN_ROOM", "Leave the current room first.", 409)
    retry_after = repository.room_creation_retry_after(user_id)
    if retry_after > 0:
        raise BattleServiceError(
            "ROOM_CREATE_COOLDOWN",
            "A new room can only be created once every 30 seconds.",
            429,
            extra={"retry_after_seconds": retry_after},
        )
    try:
        settings = _free_mode.validate_settings(payload)
    except ValueError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc
    reservation = _reserve_round_budget(
        user_id=user_id,
        session_id=session_id,
        full_pattern=str(settings["full_pattern"]),
        max_players=int(settings["max_players"]),
        target=int(settings["target"]),
    )
    room = None
    try:
        public_settings = {key: value for key, value in settings.items() if key != "chat_roles"}
        room = repository.create_room(
            host_user_id=user_id,
            pattern=str(settings["pattern"]),
            target=int(settings["target"]),
            full_pattern=str(settings["full_pattern"]),
            visibility=str(settings["visibility"]),
            allow_spectators=bool(settings["allow_spectators"]),
            allow_guest_chat=bool(settings["allow_guest_chat"]),
            max_players=int(settings["max_players"]),
            initial_board=settings["initial_board"],
            max_steps=int(settings["score_step_limit"]),
            step_timeout_seconds=int(settings["step_timeout_seconds"]),
            mode_key=_free_mode.key,
            mode_version=_free_mode.version,
            chat_roles=settings["chat_roles"],
            settings=public_settings,
            status="preparing",
        )
        initial_board, _results, _dtype = await _select_initial_board(
            room, settings["initial_board"]
        )
        _insert_round(
            room_id=str(room["room_id"]),
            round_number=1,
            seed_hex=secrets.token_hex(32),
            initial_board=initial_board,
            score_step_limit=int(settings["score_step_limit"]),
            ranking_min_steps=int(settings["ranking_min_steps"]),
            reservation=reservation,
        )
    except Exception:
        cancel_reservation(reservation, reason="battle_free_room_not_created")
        if room is not None:
            try:
                repository.close_room(str(room["room_id"]), status="closed")
            except Exception:
                pass
        raise
    return {
        "room": room_snapshot(str(room["room_id"]), user_id=user_id),
        "token_balance": get_token_balance(user_id),
    }


def _identity_key(actor_key: str | int | None = None, user_id: int | None = None) -> str:
    if actor_key is None:
        if user_id is None:
            raise ValueError("actor_required")
        return f"u:{int(user_id)}"
    value = str(actor_key)
    return f"u:{value}" if value.isdigit() else value


def _candidate_key(round_id: str, actor_key: str | int, sequence: int, board: int):
    return (
        str(round_id),
        _identity_key(actor_key),
        int(sequence),
        int(board),
        get_catalog_version(),
    )


def _touch_prepared(key) -> dict[str, PreparedSpawn | None]:
    prepared = _prepared.setdefault(key, {})
    _prepared.move_to_end(key)
    while len(_prepared) > MAX_PREPARED_STATES:
        expired_key, _expired = _prepared.popitem(last=False)
        _prepared_complete.discard(expired_key)
        task = _prepare_state_tasks.pop(expired_key, None)
        if task is not None and not task.done():
            task.cancel()
    return prepared


def _store_prepared(key, prepared: dict[str, PreparedSpawn | None]) -> None:
    current = _touch_prepared(key)
    current.update(prepared)


def _store_prepared_direction(
    key,
    direction: str,
    candidate: PreparedSpawn | None,
) -> None:
    _touch_prepared(key)[str(direction)] = candidate


def _prepared_candidates(key) -> dict[str, PreparedSpawn]:
    prepared = _prepared.get(key) or {}
    if key in _prepared:
        _prepared.move_to_end(key)
    return {
        direction: candidate
        for direction, candidate in prepared.items()
        if isinstance(candidate, PreparedSpawn)
    }


def _candidate_choices(
    seed_hex: str,
    moved_board: int,
    spawn_rate: float,
):
    empty = _empty_indices(moved_board)
    possible_values = []
    if float(spawn_rate) < 1.0:
        possible_values.append(2)
    if float(spawn_rate) > 0.0:
        possible_values.append(4)
    remaining = {(index, value) for index in empty for value in possible_values}
    attempt = 0
    while remaining:
        index, value = deterministic_spawn_choice(
            seed_hex,
            moved_board,
            attempt,
            empty,
            spawn_rate=spawn_rate,
        )
        if (index, value) not in remaining:
            ticket = deterministic_ticket(seed_hex, moved_board, attempt)
            ordered = sorted(remaining)
            index, value = ordered[int.from_bytes(ticket[16:24], "big") % len(ordered)]
        remaining.remove((index, value))
        yield attempt, index, value
        attempt += 1


async def _generate_certainty_tail(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str | None = None,
    board: int,
    results: dict[str, float],
    seed_hex: str,
    sequence: int,
    risk_state: SpawnRiskState,
    user_id: int | None = None,
) -> tuple[list[CertaintyTailStep], int]:
    """Finish a proven position locally without adding scored Battle steps."""

    actor_key = _identity_key(actor_key, user_id)
    target = int(room["target"])
    use_variant = _use_variant(str(room["pattern"]))
    entry = resolve_tablebase(str(room["full_pattern"])) or {}
    spawn_rate = float(entry.get("spawn_rate", 0.1))
    supporter = _room_supporter(room)
    current_board = int(board)
    current_results = dict(results)
    current_risk = risk_state
    steps: list[CertaintyTailStep] = []
    limit = min(MAX_CERTAINTY_TAIL_STEPS, max(16, target // 2 + 16))

    for offset in range(limit):
        if _contains_target(current_board, target):
            return steps, current_board
        moved = _moved_boards(current_board, use_variant=use_variant)
        direction, best_success = best_direction(current_results, moved)
        if direction is None or best_success < 1.0 - RISK_EPSILON:
            raise BattleServiceError(
                "CERTAINTY_TAIL_UNAVAILABLE",
                "The guaranteed continuation could not be completed.",
                503,
            )
        moved_board = moved[direction]
        selected: tuple[int, int, int, dict[str, float], SpawnRiskState] | None = None
        for attempt, spawn_index, spawn_value in _candidate_choices(
            seed_hex,
            moved_board,
            spawn_rate,
        ):
            next_board = _spawn_board(moved_board, spawn_index, spawn_value)
            if _contains_target(next_board, target):
                selected = (
                    next_board,
                    spawn_index,
                    spawn_value,
                    {},
                    current_risk,
                )
                break
            lookup = await _lookup(
                room,
                next_board,
                stream_key=(
                    f"battle-free-certainty:{round_id}:{actor_key}:"
                    f"{int(sequence) + offset}:{attempt}"
                ),
                supporter=supporter,
                lane="foreground",
            )
            if not lookup.found:
                continue
            next_results = _normalize_results(lookup)
            legal = _moved_boards(next_board, use_variant=use_variant)
            _next_direction, next_best = best_direction(next_results, legal)
            accepted, _multiplier, next_risk = evaluate_spawn(
                executed_success=best_success,
                next_success=next_best,
                risk_state=current_risk,
            )
            if accepted:
                selected = (
                    next_board,
                    spawn_index,
                    spawn_value,
                    next_results,
                    next_risk,
                )
                break
        if selected is None:
            raise BattleServiceError(
                "CERTAINTY_TAIL_UNAVAILABLE",
                "The guaranteed continuation could not be completed.",
                503,
            )
        next_board, spawn_index, spawn_value, next_results, next_risk = selected
        steps.append(
            CertaintyTailStep(
                previous_board=current_board,
                next_board=next_board,
                direction=direction,
                spawn_index=spawn_index,
                spawn_value=spawn_value,
                rates=current_results,
            )
        )
        current_board = next_board
        current_results = next_results
        current_risk = next_risk

    raise BattleServiceError(
        "CERTAINTY_TAIL_TOO_LONG",
        "The guaranteed continuation exceeded its safety limit.",
        503,
    )


async def _generate_certainty_tail_best_effort(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str | None = None,
    board: int,
    results: dict[str, float],
    seed_hex: str,
    sequence: int,
    risk_state: SpawnRiskState,
    user_id: int | None = None,
) -> tuple[list[CertaintyTailStep], int, bool]:
    """Keep a proven result valid when its optional finish animation cannot be built."""

    try:
        steps, final_board = await _generate_certainty_tail(
            room=room,
            round_id=round_id,
            actor_key=actor_key,
            board=board,
            results=results,
            seed_hex=seed_hex,
            sequence=sequence,
            risk_state=risk_state,
            user_id=user_id,
        )
    except BattleServiceError as exc:
        if exc.code not in CERTAINTY_TAIL_FALLBACK_CODES:
            raise
        return [], int(board), False
    return steps, final_board, True


def _append_certainty_replay_steps(
    replay_blob: bytes,
    replay_move_count: int,
    steps: list[CertaintyTailStep],
) -> tuple[bytes, int]:
    payload = bytes(replay_blob or b"")
    count = int(replay_move_count)
    for step in steps:
        encoded = encode_replay_step(
            board=step.previous_board,
            selected_direction=step.direction,
            spawn_index=step.spawn_index,
            spawn_value=step.spawn_value,
            rates=step.rates,
        )
        payload, recorded = append_replay_step(payload, encoded)
        count += int(recorded)
    return payload, count


async def _prepare_direction(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str,
    sequence: int,
    direction: str,
    moved_board: int,
    executed_success: float,
    risk_state: SpawnRiskState,
    seed_hex: str,
    lane: str,
) -> PreparedSpawn | None:
    entry = resolve_tablebase(str(room["full_pattern"])) or {}
    spawn_rate = float(entry.get("spawn_rate", 0.1))
    supporter = _room_supporter(room)
    for attempt, spawn_index, spawn_value in _candidate_choices(
        seed_hex,
        moved_board,
        spawn_rate,
    ):
        candidate = await _evaluate_spawn_candidate(
            room=room,
            round_id=round_id,
            actor_key=actor_key,
            sequence=sequence,
            direction=direction,
            moved_board=moved_board,
            executed_success=executed_success,
            risk_state=risk_state,
            attempt=attempt,
            spawn_index=spawn_index,
            spawn_value=spawn_value,
            supporter=supporter,
            lane=lane,
        )
        if candidate is not None:
            return candidate
    return None


async def _evaluate_spawn_candidate(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str | None = None,
    sequence: int,
    direction: str,
    moved_board: int,
    executed_success: float,
    risk_state: SpawnRiskState,
    attempt: int,
    spawn_index: int,
    spawn_value: int,
    supporter: bool,
    lane: str,
    user_id: int | None = None,
) -> PreparedSpawn | None:
    actor_key = _identity_key(actor_key, user_id)
    next_board = _spawn_board(moved_board, spawn_index, spawn_value)
    lookup = await _lookup(
        room,
        next_board,
        stream_key=(
            f"battle-free:{round_id}:{actor_key}:{sequence}:{direction}:{attempt}"
        ),
        supporter=supporter,
        lane=lane,
    )
    if not lookup.found:
        return None
    next_results = _normalize_results(lookup)
    legal = _moved_boards(
        next_board,
        use_variant=_use_variant(str(room["pattern"])),
    )
    _next_direction, next_best = best_direction(next_results, legal)
    accepted, multiplier, next_risk = evaluate_spawn(
        executed_success=executed_success,
        next_success=next_best,
        risk_state=risk_state,
    )
    if not accepted:
        return None
    return PreparedSpawn(
        executed_direction=direction,
        moved_board=moved_board,
        next_board=next_board,
        spawn_index=spawn_index,
        spawn_value=spawn_value,
        attempt_index=attempt,
        next_results=next_results,
        next_dtype=lookup.dtype,
        next_best_success=next_best,
        risk_multiplier=multiplier,
        risk_state=next_risk,
    )


async def _prepare_directions(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str | None = None,
    sequence: int,
    direction_states: dict[str, tuple[int, float]],
    risk_state: SpawnRiskState,
    seed_hex: str,
    lane: str,
    prepared_key=None,
    user_id: int | None = None,
) -> dict[str, PreparedSpawn]:
    actor_key = _identity_key(actor_key, user_id)
    async def prepare_one(
        direction: str,
        moved_board: int,
        executed_success: float,
    ) -> tuple[str, PreparedSpawn | None, bool]:
        try:
            candidate = await _prepare_direction(
                room=room,
                round_id=round_id,
                actor_key=actor_key,
                sequence=sequence,
                direction=direction,
                moved_board=moved_board,
                executed_success=executed_success,
                risk_state=risk_state,
                seed_hex=seed_hex,
                lane=lane,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if lane == "foreground":
                raise
            return direction, None, False
        if prepared_key is not None:
            _store_prepared_direction(prepared_key, direction, candidate)
        return direction, candidate, True

    tasks = [
        asyncio.create_task(prepare_one(direction, moved_board, executed_success))
        for direction, (moved_board, executed_success) in direction_states.items()
    ]
    if not tasks:
        return {}
    prepared: dict[str, PreparedSpawn] = {}
    try:
        for completed in asyncio.as_completed(tasks):
            direction, candidate, _resolved = await completed
            if isinstance(candidate, PreparedSpawn):
                prepared[direction] = candidate
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return prepared


async def _prepare_state(
    *,
    room: dict[str, Any],
    round_id: str,
    actor_key: str | None = None,
    sequence: int,
    board: int,
    results: dict[str, float],
    risk_state: SpawnRiskState,
    seed_hex: str,
    lane: str = "prefetch",
    user_id: int | None = None,
) -> dict[str, PreparedSpawn]:
    actor_key = _identity_key(actor_key, user_id)
    key = _candidate_key(round_id, actor_key, sequence, board)
    if key in _prepared_complete:
        return _prepared_candidates(key)
    moved = _moved_boards(board, use_variant=_use_variant(str(room["pattern"])))
    optimal, best_success = best_direction(results, moved)
    if optimal is None or best_success <= 0:
        _store_prepared(key, {})
        _prepared_complete.add(key)
        return {}
    directions: list[str] = []
    for direction in sorted(
        moved,
        key=lambda item: (item != optimal, -clamp_probability(results.get(item))),
    ):
        decision = decide_move(
            selected_direction=direction,
            results=results,
            legal_directions=moved,
        )
        if direction == optimal or (decision and not decision.corrected):
            directions.append(direction)
    cached = _prepared.get(key) or {}
    missing_directions = [direction for direction in directions if direction not in cached]
    await _prepare_directions(
        room=room,
        round_id=round_id,
        actor_key=actor_key,
        sequence=sequence,
        direction_states={
            direction: (
                moved[direction],
                clamp_probability(results.get(direction)),
            )
            for direction in missing_directions
        },
        risk_state=risk_state,
        seed_hex=seed_hex,
        lane=lane,
        prepared_key=key,
    )
    current = _prepared.get(key) or {}
    if all(direction in current for direction in directions):
        _prepared_complete.add(key)
    return _prepared_candidates(key)


def _schedule_prepare_state(**kwargs) -> asyncio.Task[dict[str, PreparedSpawn]]:
    key = _candidate_key(
        kwargs["round_id"],
        kwargs["actor_key"],
        kwargs["sequence"],
        kwargs["board"],
    )
    existing = _prepare_state_tasks.get(key)
    if existing is not None and not existing.done():
        return existing
    task = asyncio.create_task(_prepare_state(**kwargs))
    _prepare_state_tasks[key] = task

    def completed(done: asyncio.Task) -> None:
        if _prepare_state_tasks.get(key) is done:
            _prepare_state_tasks.pop(key, None)
        if not done.cancelled():
            done.exception()

    task.add_done_callback(completed)
    return task


def _cancel_state_prefetch(key) -> None:
    task = _prepare_state_tasks.pop(key, None)
    if task is not None and not task.done():
        task.cancel()


def _discard_round_prefetch(round_id: str) -> None:
    normalized = str(round_id)
    for key in list(_prepare_state_tasks):
        if key[0] == normalized:
            _cancel_state_prefetch(key)
    for key in list(_prepared):
        if key[0] == normalized:
            _prepared.pop(key, None)
            _prepared_complete.discard(key)


def _load_round_context(db: sqlite3.Connection, room_ref: str, actor_key: str):
    room = repository._find_room(db, room_ref)
    member = db.execute(
        "SELECT * FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
        (room["room_id"], actor_key),
    ).fetchone()
    if member is None:
        raise BattleServiceError("ROOM_MEMBERSHIP_REQUIRED", "You are not in this room.", 403)
    round_row = db.execute(
        "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
        (room["room_id"],),
    ).fetchone()
    return room, member, round_row


async def _initialize_players(
    room: dict[str, Any], round_row: dict[str, Any], players
) -> bool:
    board = int(json.loads(round_row["mode_state_json"])["initial_board"], 16)
    query = await _lookup(
        room,
        board,
        stream_key=f"battle-free-round:{round_row['round_id']}:initial",
        supporter=_room_supporter(room),
        lane="foreground",
    )
    results = _normalize_results(query)
    legal = _moved_boards(board, use_variant=_use_variant(str(room["pattern"])))
    _optimal, best_success = best_direction(results, legal)
    if not legal or best_success <= 0:
        raise BattleServiceError("INITIAL_BOARD_UNUSABLE", "The starting board cannot continue.", 409)
    seed_hex = str(round_row["route_seed"])
    mode_state = json.loads(round_row["mode_state_json"] or "{}")
    step_limit = int(mode_state.get("score_step_limit") or int(room["target"]) // 2)
    ranking_min_steps = int(mode_state.get("ranking_min_steps") or step_limit)
    initial_finish = _finish_reason(
        board=board,
        target=int(room["target"]),
        step_index=0,
        step_limit=step_limit,
        next_best=best_success,
        use_variant=_use_variant(str(room["pattern"])),
    )
    initial_board = board
    certainty_steps: list[CertaintyTailStep] = []
    if initial_finish == "certainty":
        certainty_steps, board, tail_completed = await _generate_certainty_tail_best_effort(
            room=room,
            round_id=str(round_row["round_id"]),
            actor_key=str(room.get("host_actor_key") or "platform"),
            board=initial_board,
            results=results,
            seed_hex=seed_hex,
            sequence=0,
            risk_state=SpawnRiskState(),
        )
        initial_finish = "target_reached" if tail_completed else "certainty"
        results = {}
    if initial_finish is None:
        await asyncio.gather(*[
            _prepare_state(
                room=room,
                round_id=str(round_row["round_id"]),
                actor_key=str(player["actor_key"]),
                sequence=0,
                board=board,
                results=results,
                risk_state=SpawnRiskState(),
                seed_hex=seed_hex,
                lane="prefetch",
            )
            for player in players
        ])
    now = utcnow()
    deadline = (
        None
        if initial_finish
        else iso(now + timedelta(seconds=int(room["step_timeout_seconds"])))
    )
    now_text = iso(now)
    result_status = "completed" if initial_finish else "playing"
    state_status = "finished" if initial_finish else "input"
    initial_ranking_eligible = _ranking_eligible(
        result_status=result_status,
        finish_reason=initial_finish,
        progress=0,
        ranking_min_steps=ranking_min_steps,
    )
    certainty_payload = [step.public_payload() for step in certainty_steps]
    certainty_replay, certainty_replay_count = _append_certainty_replay_steps(
        b"", 0, certainty_steps
    )
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        fresh_round = db.execute(
            "SELECT status FROM battle_rounds WHERE round_id = ?", (round_row["round_id"],)
        ).fetchone()
        if fresh_round is None or fresh_round["status"] != "ready":
            raise BattleServiceError("ROUND_STATE_CHANGED", "Round state changed.", 409)
        for player in players:
            player_mode_data = {
                "finish_reason": initial_finish,
                "finish_class": _public_finish_class(initial_ranking_eligible),
                "ranking_min_steps": ranking_min_steps,
                "ranking_eligible": initial_ranking_eligible,
            }
            if certainty_payload:
                player_mode_data.update(
                    {
                        "auto_steps": certainty_payload,
                        "auto_playback_key": (
                            f"{round_row['round_id']}:{player['actor_key']}:0:certainty"
                        ),
                        "auto_start_board_hex": f"{initial_board:016x}",
                        "auto_final_board_hex": f"{board:016x}",
                    }
                )
            db.execute(
                """
                INSERT OR REPLACE INTO battle_player_results
                (result_id, round_id, actor_key, user_id, guest_id,
                 display_name_snapshot, status, route_index, last_sequence,
                 goodness_of_fit, primary_score, secondary_score, progress,
                 mode_data_json, choice_blob, finished_at, timeout_at,
                 created_at, updated_at, board_state)
                VALUES (
                  (SELECT result_id FROM battle_player_results WHERE round_id = ? AND actor_key = ?),
                  ?, ?, ?, ?, ?, ?, 0, 0, 1.0, 1.0, 0, 0, ?, X'', ?, ?, ?, ?, ?
                )
                """,
                (
                    round_row["round_id"], player["actor_key"],
                    round_row["round_id"], player["actor_key"],
                    player["user_id"], player["guest_id"],
                    player["display_name_snapshot"],
                    result_status, json.dumps(player_mode_data, separators=(",", ":")),
                    now_text if initial_finish else None,
                    deadline, now_text, now_text, f"{board:016x}",
                ),
            )
            if certainty_replay_count:
                db.execute(
                    "UPDATE battle_player_results SET replay_blob = ?, replay_move_count = ? WHERE round_id = ? AND actor_key = ?",
                    (
                        certainty_replay,
                        certainty_replay_count,
                        round_row["round_id"],
                        player["actor_key"],
                    ),
                )
            db.execute(
                """
                INSERT OR REPLACE INTO battle_free_player_states
                (round_id, actor_key, user_id, guest_id, board_state, step_index, sequence,
                 spawn_log_index, spawn_log_floor, rng_step,
                 state_status, finish_reason, timeout_at,
                 current_results_json, operation_blob, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, 0, 0.0, 0.0, 0, ?, ?, ?, ?, X'', ?, ?)
                """,
                (
                    round_row["round_id"], player["actor_key"],
                    player["user_id"], player["guest_id"], f"{board:016x}",
                    state_status, initial_finish, deadline,
                    json.dumps(results, separators=(",", ":")), now_text, now_text,
                ),
            )
        db.execute(
            "UPDATE battle_rounds SET status = ?, started_at = ?, ended_at = ?, expires_at = ?, updated_at = ? WHERE round_id = ?",
            (
                "completed" if initial_finish else "running",
                now_text,
                now_text if initial_finish else None,
                iso(now + (WAITING_LIFETIME if initial_finish else ROUND_LIFETIME)),
                now_text,
                round_row["round_id"],
            ),
        )
        db.execute(
            "UPDATE battle_rooms SET status = ?, expires_at = ?, revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (
                "waiting" if initial_finish else "running",
                iso(now + (WAITING_LIFETIME if initial_finish else ROUND_LIFETIME)),
                now_text,
                room["room_id"],
            ),
        )
        if initial_finish:
            db.execute(
                "UPDATE battle_members SET ready = 0, updated_at = ? WHERE room_id = ? AND status = 'active'",
                (now_text, room["room_id"]),
            )
    return initial_finish is not None


async def _new_round(
    room: dict[str, Any], *, user_id: int | None, session_id: int | None
) -> dict[str, Any]:
    reservation = None
    if str(room.get("billing_policy") or "user") != "platform":
        if user_id is None:
            raise BattleServiceError("AUTH_REQUIRED", "Authentication required.", 401)
        reservation = _reserve_round_budget(
            user_id=user_id,
            session_id=session_id,
            full_pattern=str(room["full_pattern"]),
            max_players=int(room["max_players"]),
            target=int(room["target"]),
        )
    try:
        settings = dict(room.get("settings") or {})
        requested = settings.get("initial_board")
        initial, _results, _dtype = await _select_initial_board(room, requested)
        round_id = _insert_round(
            room_id=str(room["room_id"]),
            round_number=int(room.get("current_round_number") or 0) + 1,
            seed_hex=secrets.token_hex(32),
            initial_board=initial,
            score_step_limit=int(settings.get("score_step_limit") or int(room["target"]) // 2),
            ranking_min_steps=int(
                settings.get("ranking_min_steps")
                or settings.get("score_step_limit")
                or int(room["target"]) // 2
            ),
            reservation=reservation,
        )
    except Exception:
        if reservation is not None:
            cancel_reservation(reservation, reason="battle_free_next_round_not_created")
        raise
    return repository.get_room(str(room["room_id"]))


async def start_room_for_mode(
    room_code: str,
    *,
    actor_key: str | None = None,
    user_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    actor_key = str(actor_key or (f"u:{user_id}" if user_id is not None else ""))
    room = repository.get_room(room_code)
    if str(room.get("host_actor_key") or f"u:{room.get('host_user_id')}") != str(actor_key):
        raise BattleServiceError("HOST_REQUIRED", "Only the host can start.", 403)
    if (room.get("round") or {}).get("status") == "completed":
        room = await _new_round(room, user_id=user_id, session_id=session_id)
    now_text = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room_row = repository._find_room(db, room_code)
        if room_row["status"] != "waiting":
            raise BattleServiceError("ROOM_NOT_READY", "Room is not ready to start.", 409)
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
            (room_row["room_id"],),
        ).fetchone()
        if round_row is None or round_row["status"] != "ready":
            raise BattleServiceError("ROUND_NOT_READY", "Round is not ready.", 409)
        players = _ready_players_for_start(db, room_row, now_text=now_text)
        room_data = dict(room_row)
        round_data = dict(round_row)
        player_data = [dict(player) for player in players]
    try:
        completed_immediately = await _initialize_players(
            room_data, round_data, player_data
        )
    except Exception:
        raise
    if completed_immediately:
        _settle_round(str(round_data["round_id"]))
    await broadcast_room(str(room_data["room_id"]))
    return room_snapshot(room_code, actor_key=actor_key)


async def ensure_permanent_room(definition) -> dict[str, Any]:
    settings = _free_mode.validate_settings(
        {
            "full_pattern": definition.full_pattern,
            "initial_board": definition.initial_board,
            "max_players": definition.max_players,
            "step_timeout_seconds": definition.step_timeout_seconds,
            "ranking_min_steps": None,
            "is_public": True,
            "allow_spectators": True,
            "allow_guest_chat": True,
            "chat_roles": ["host", "player", "spectator"],
        }
    )
    public_settings = {key: value for key, value in settings.items() if key != "chat_roles"}
    room = repository.create_permanent_room(
        template_key=definition.template_key,
        pattern=str(settings["pattern"]),
        target=int(settings["target"]),
        full_pattern=str(settings["full_pattern"]),
        mode_key=_free_mode.key,
        mode_version=_free_mode.version,
        initial_board=str(definition.initial_board),
        max_steps=int(settings["score_step_limit"]),
        step_timeout_seconds=int(settings["step_timeout_seconds"]),
        max_players=int(settings["max_players"]),
        settings=public_settings,
        status="preparing",
    )
    current = room.get("round") or {}
    if not current or current.get("status") in {"failed", "cancelled"}:
        initial, _results, _dtype = await _select_initial_board(
            room, str(definition.initial_board)
        )
        _insert_round(
            room_id=str(room["room_id"]),
            round_number=int(room.get("current_round_number") or 0) + 1,
            seed_hex=secrets.token_hex(32),
            initial_board=initial,
            score_step_limit=int(settings["score_step_limit"]),
            ranking_min_steps=int(settings["ranking_min_steps"]),
            reservation=None,
        )
    return repository.get_room(str(room["room_id"]))


def _ranking_eligible(
    *,
    result_status: str,
    finish_reason: str | None,
    progress: int,
    ranking_min_steps: int,
) -> bool:
    if result_status not in {"playing", "completed", "disconnected"}:
        return False
    if finish_reason in {"timed_out", "forfeit", "room_closed"}:
        return False
    return int(progress) >= max(1, int(ranking_min_steps))


def _public_finish_class(ranking_eligible: bool) -> str:
    return "completed" if ranking_eligible else "unranked"


def sanitize_snapshot_for_mode(
    payload: dict[str, Any], *, viewer_actor_key: str, viewer_user_id: int | None
) -> dict[str, Any]:
    round_payload = payload.get("round") or {}
    round_id = str(round_payload.get("round_id") or "")
    mode_state = dict(round_payload.get("mode_state") or {})
    step_limit = int(mode_state.get("score_step_limit") or payload.get("max_steps") or 0)
    ranking_min_steps = int(mode_state.get("ranking_min_steps") or step_limit)
    payload["route"] = {
        "initial_board": str(mode_state.get("initial_board") or payload.get("initial_board") or ""),
        "step_count": step_limit,
        "ranking_min_steps": ranking_min_steps,
        "termination_reason": "fixed_steps",
    }
    if not round_id:
        return payload
    with auth_db() as db:
        states = db.execute(
            "SELECT * FROM battle_free_player_states WHERE round_id = ?",
            (round_id,),
        ).fetchall()
    by_actor = {str(row["actor_key"]): dict(row) for row in states}
    viewer = by_actor.get(viewer_actor_key)
    viewer_member = next(
        (item for item in payload.get("members", []) if item["actor_key"] == viewer_actor_key),
        {},
    )
    reveal_all = viewer_member.get("role") == "spectator" or (
        viewer is not None and str(viewer.get("state_status")) == "finished"
    )
    for result in payload.get("results", []):
        state = by_actor.get(str(result["actor_key"]))
        if not state:
            continue
        mode_data = dict(result.get("mode_data") or {})
        progress = int(state["step_index"])
        ranking_eligible = _ranking_eligible(
            result_status=str(result.get("status") or ""),
            finish_reason=state.get("finish_reason"),
            progress=progress,
            ranking_min_steps=ranking_min_steps,
        )
        mode_data.update(
            {
                "finish_reason": state.get("finish_reason"),
                "finish_class": _public_finish_class(ranking_eligible),
                "state_status": state.get("state_status"),
                "score_step_limit": step_limit,
                "ranking_min_steps": ranking_min_steps,
                "ranking_eligible": ranking_eligible,
            }
        )
        if reveal_all or result["actor_key"] == viewer_actor_key:
            mode_data["board_hex"] = state["board_state"]
        else:
            mode_data.pop("last_step", None)
            mode_data.pop("correction", None)
        if (
            state.get("state_status") != "awaiting_ack"
            and str(result.get("status") or "") == "playing"
        ):
            mode_data.pop("correction", None)
        if result["actor_key"] != viewer_actor_key:
            mode_data.pop("auto_steps", None)
            mode_data.pop("auto_playback_key", None)
            mode_data.pop("auto_start_board_hex", None)
            mode_data.pop("auto_final_board_hex", None)
        result["mode_data"] = mode_data
        result["route_index"] = progress
        result["progress"] = progress
        result["timeout_at"] = state.get("timeout_at")
    return payload


def artifact_payload_for_mode(
    room_code: str, round_id: str, *, actor_key: str
) -> tuple[bytes, dict[str, Any]]:
    room = repository.get_room(room_code)
    assert_member(room, actor_key=actor_key)
    if str((room.get("round") or {}).get("round_id") or "") != str(round_id):
        raise BattleServiceError("ROUND_NOT_FOUND", "Round not found.", 404)
    mode_state = (room.get("round") or {}).get("mode_state") or {}
    blob = json.dumps(
        {
            "rules_version": _free_mode.version,
            "initial_board": mode_state.get("initial_board"),
            "score_step_limit": mode_state.get("score_step_limit"),
            "ranking_min_steps": mode_state.get("ranking_min_steps"),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return blob, {"artifact_kind": _free_mode.artifact_kind, "step_count": mode_state.get("score_step_limit", 0)}


def _append_operation(
    blob: bytes,
    *,
    selected: str,
    executed: str,
    corrected: bool,
    spawn_index: int,
    spawn_value: int,
    goodness: float,
) -> bytes:
    flags = (
        MOVE_CODE_BITS[selected]
        | (MOVE_CODE_BITS[executed] << 2)
        | ((1 if corrected else 0) << 4)
    )
    spawn = 0xFF if spawn_index < 0 else ((spawn_index & 0xF) | ((1 if spawn_value == 4 else 0) << 4))
    goodness_units = max(0, min(65535, round(float(goodness) * 65535)))
    return bytes(blob or b"") + struct.pack("<BBH", flags, spawn, goodness_units)


def _legacy_goodness_product(blob: bytes) -> float:
    """Recover Tester-style scoring for rounds created before rules version 2."""

    payload = bytes(blob or b"")
    if len(payload) % 4:
        raise BattleServiceError("ROUND_STATE_CHANGED", "Battle scoring data is invalid.", 409)
    goodness = 1.0
    for _flags, _spawn, goodness_units in struct.iter_unpack("<BBH", payload):
        goodness = accumulate_goodness(goodness, goodness_units / 65535.0)
    return goodness


def _settle_round(round_id: str, *, cancelled: bool = False, reason: str = "") -> None:
    with auth_db() as db:
        round_row = db.execute("SELECT * FROM battle_rounds WHERE round_id = ?", (round_id,)).fetchone()
        if round_row is None or str(round_row["reservation_status"] or "") != "reserved":
            return
        room_row = db.execute("SELECT * FROM battle_rooms WHERE room_id = ?", (round_row["room_id"],)).fetchone()
        if room_row is None:
            return
        reservation = _restore_reservation(round_row, room_row)
        try:
            mode_state = json.loads(round_row["mode_state_json"] or "{}")
        except (TypeError, ValueError):
            mode_state = {}
    if reservation is None:
        return
    if cancelled:
        cancel_reservation(
            reservation,
            reason=reason or "battle_free_cancelled",
            metadata={"room_id": room_row["room_id"], "round_id": round_id},
        )
        durable = "cancelled"
    else:
        hit_count = int(mode_state.get("lookup_hit_steps") or 0)
        miss_count = int(mode_state.get("lookup_miss_steps") or 0)
        actual_base = (
            hit_count * operation_cost_units("tester_lookup_hit")
            + miss_count * operation_cost_units("tester_lookup_miss")
        )
        finalize_reservation(
            reservation,
            actual_operation_key="battle_free_lookup_settlement",
            actual_base_units=actual_base,
            metadata={
                "room_id": room_row["room_id"],
                "round_id": round_id,
                "lookup_hit_steps": hit_count,
                "lookup_miss_steps": miss_count,
            },
        )
        durable = "finalized"
    with auth_db() as db:
        db.execute(
            "UPDATE battle_rounds SET reservation_status = ?, updated_at = ? WHERE round_id = ? AND reservation_status = 'reserved'",
            (durable, iso(), round_id),
        )


def _complete_round_if_done(db: sqlite3.Connection, room_id: str, round_id: str, now_text: str) -> bool:
    remaining = db.execute(
        "SELECT COUNT(*) AS count FROM battle_player_results WHERE round_id = ? AND status = 'playing'",
        (round_id,),
    ).fetchone()
    if int(remaining["count"] or 0) > 0:
        return False
    db.execute(
        "UPDATE battle_rounds SET status = 'completed', ended_at = ?, updated_at = ? WHERE round_id = ?",
        (now_text, now_text, round_id),
    )
    db.execute(
        "UPDATE battle_rooms SET status = 'waiting', expires_at = ?, revision = revision + 1, updated_at = ? WHERE room_id = ?",
        (
            repository.PERMANENT_ROOM_EXPIRES_AT
            if str(repository._find_room(db, room_id)["lifecycle_kind"] or "normal") == "permanent"
            else iso(utcnow() + WAITING_LIFETIME),
            now_text,
            room_id,
        ),
    )
    db.execute(
        "UPDATE battle_members SET ready = 0, updated_at = ? WHERE room_id = ? AND status = 'active'",
        (now_text, room_id),
    )
    from ...permanent.service import rotate_after_round_in_db

    rotate_after_round_in_db(db, room_id, now_text=now_text)
    return True


def _finish_reason(
    *,
    board: int,
    target: int,
    step_index: int,
    step_limit: int,
    next_best: float,
    use_variant: bool,
) -> str | None:
    if _contains_target(board, target):
        return "target_reached"
    if next_best >= 1.0:
        return "certainty"
    if step_index >= step_limit:
        return "step_limit"
    legal = _moved_boards(board, use_variant=use_variant)
    if not legal:
        return "no_legal_move"
    if next_best <= 0:
        return "zero_success"
    return None


def _resume_after_resolution_error(
    *,
    round_id: str,
    actor_key: str,
    request_id: str,
    original_timeout: str | None,
    resolution_started_at: str,
) -> None:
    now = utcnow()
    remaining = 0.0
    if original_timeout:
        try:
            remaining = max(
                0.0,
                (
                    datetime.fromisoformat(str(original_timeout))
                    - datetime.fromisoformat(str(resolution_started_at))
                ).total_seconds(),
            )
        except ValueError:
            remaining = 0.0
    deadline = iso(now + timedelta(seconds=remaining))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        restored = db.execute(
            """
            UPDATE battle_free_player_states
            SET state_status = 'input', resolution_request_id = NULL,
                resolution_started_at = NULL, timeout_at = ?, updated_at = ?
            WHERE round_id = ? AND actor_key = ? AND state_status = 'resolving'
              AND resolution_request_id = ?
            """,
            (deadline, iso(now), round_id, actor_key, request_id),
        )
        if restored.rowcount:
            db.execute(
                "UPDATE battle_player_results SET timeout_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ? AND status = 'playing'",
                (deadline, iso(now), round_id, actor_key),
            )


async def _select_prepared_spawn(
    *,
    room: dict[str, Any],
    round_data: dict[str, Any],
    state_data: dict[str, Any],
    actor_key: str,
    board: int,
    moved: dict[str, int],
    results: dict[str, float],
    decision,
) -> tuple[PreparedSpawn | None, str, str | None]:
    risk_state = SpawnRiskState(
        float(state_data["spawn_log_index"] or 0.0),
        float(state_data["spawn_log_floor"] or 0.0),
    )
    key = _candidate_key(
        round_data["round_id"], actor_key, int(state_data["sequence"]), board
    )
    prepared = _prepared.get(key) or {}
    if key in _prepared:
        _prepared.move_to_end(key)
    executed_direction = decision.executed_direction
    correction_reason = decision.correction_reason
    direction_ready = executed_direction in prepared
    candidate = prepared.get(executed_direction)
    if not direction_ready:
        candidate = await _prepare_direction(
            room=room,
            round_id=str(round_data["round_id"]),
            actor_key=actor_key,
            sequence=int(state_data["sequence"]),
            direction=executed_direction,
            moved_board=moved[executed_direction],
            executed_success=clamp_probability(results.get(executed_direction)),
            risk_state=risk_state,
            seed_hex=str(round_data["route_seed"]),
            lane="foreground",
        )
        _store_prepared_direction(key, executed_direction, candidate)
    if candidate is None and executed_direction != decision.best_direction:
        executed_direction = decision.best_direction
        correction_reason = "spawn_risk"
        direction_ready = executed_direction in prepared
        candidate = prepared.get(executed_direction)
        if not direction_ready:
            candidate = await _prepare_direction(
                room=room,
                round_id=str(round_data["round_id"]),
                actor_key=actor_key,
                sequence=int(state_data["sequence"]),
                direction=executed_direction,
                moved_board=moved[executed_direction],
                executed_success=clamp_probability(results.get(executed_direction)),
                risk_state=risk_state,
                seed_hex=str(round_data["route_seed"]),
                lane="foreground",
            )
            _store_prepared_direction(key, executed_direction, candidate)
    _cancel_state_prefetch(key)
    return candidate, executed_direction, correction_reason


async def _resolve_move(
    room_code: str, *, actor_key: str, payload: dict[str, Any]
) -> dict[str, Any]:
    direction = str(payload.get("direction") or "").lower()
    request_id = str(payload.get("request_id") or payload.get("resolution_request_id") or uuid.uuid4().hex)[:160]
    requested_round = str(payload.get("round_id") or "")
    requested_sequence = int(payload.get("sequence") or 0)
    now = utcnow()
    now_text = iso(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room_row, member, round_row = _load_round_context(db, room_code, actor_key)
        if member["role"] != "player" or round_row is None or round_row["status"] != "running":
            raise BattleServiceError("ROUND_NOT_RUNNING", "Round is not running.", 409)
        if requested_round and requested_round != str(round_row["round_id"]):
            raise BattleServiceError("ROUND_CONFLICT", "Round changed.", 409)
        state = db.execute(
            "SELECT * FROM battle_free_player_states WHERE round_id = ? AND actor_key = ?",
            (round_row["round_id"], actor_key),
        ).fetchone()
        result_row = db.execute(
            "SELECT * FROM battle_player_results WHERE round_id = ? AND actor_key = ?",
            (round_row["round_id"], actor_key),
        ).fetchone()
        if state is None or result_row is None or result_row["status"] != "playing":
            raise BattleServiceError("PLAYER_NOT_ACTIVE", "Player is not active.", 409)
        if state["state_status"] != "input":
            raise BattleServiceError("STEP_RESOLVING", "The previous step is still resolving.", 409)
        if requested_sequence != int(state["sequence"]) + 1:
            raise BattleServiceError("PROGRESS_CONFLICT", "Progress changed.", 409)
        if state["timeout_at"] and now > datetime.fromisoformat(str(state["timeout_at"])):
            raise BattleServiceError("STEP_TIMEOUT", "Step time expired.", 409)
        board = int(str(state["board_state"]), 16)
        moved = _moved_boards(board, use_variant=_use_variant(str(room_row["pattern"])))
        if direction not in moved:
            raise BattleServiceError("ILLEGAL_DIRECTION", "That direction does not move the board.", 409)
        results = json.loads(state["current_results_json"] or "{}")
        decision = decide_move(
            selected_direction=direction,
            results=results,
            legal_directions=moved,
        )
        if decision is None:
            raise BattleServiceError("POSITION_FINISHED", "This position cannot continue.", 409)
        db.execute(
            """
            UPDATE battle_free_player_states SET state_status = 'resolving', timeout_at = NULL,
                resolution_request_id = ?, resolution_started_at = ?, updated_at = ?
            WHERE round_id = ? AND actor_key = ? AND state_status = 'input'
            """,
            (request_id, now_text, now_text, round_row["round_id"], actor_key),
        )
        db.execute(
            "UPDATE battle_player_results SET timeout_at = NULL, updated_at = ? WHERE round_id = ? AND actor_key = ?",
            (now_text, round_row["round_id"], actor_key),
        )
        room = dict(room_row)
        round_data = dict(round_row)
        state_data = dict(state)
    risk_state = SpawnRiskState(
        float(state_data["spawn_log_index"] or 0.0),
        float(state_data["spawn_log_floor"] or 0.0),
    )
    try:
        candidate, executed_direction, correction_reason = await _select_prepared_spawn(
            room=room,
            round_data=round_data,
            state_data=state_data,
            actor_key=actor_key,
            board=board,
            moved=moved,
            results=results,
            decision=decision,
        )
    except asyncio.CancelledError:
        _resume_after_resolution_error(
            round_id=str(round_data["round_id"]),
            actor_key=actor_key,
            request_id=request_id,
            original_timeout=state_data.get("timeout_at"),
            resolution_started_at=now_text,
        )
        raise
    except Exception as exc:
        _resume_after_resolution_error(
            round_id=str(round_data["round_id"]),
            actor_key=actor_key,
            request_id=request_id,
            original_timeout=state_data.get("timeout_at"),
            resolution_started_at=now_text,
        )
        if isinstance(exc, TablebaseQueryOverloaded):
            raise BattleServiceError(
                "TABLEBASE_BUSY",
                "Tablebase service is busy. Please retry shortly.",
                503,
                extra=exc.payload,
            ) from exc
        raise
    risk_boundary = candidate is None
    if risk_boundary:
        next_board = moved[executed_direction]
        spawn_index = -1
        spawn_value = 0
        next_results: dict[str, float] = {}
        next_best = 0.0
        next_risk = risk_state
        correction_reason = correction_reason or "risk_boundary"
    else:
        next_board = candidate.next_board
        spawn_index = candidate.spawn_index
        spawn_value = candidate.spawn_value
        next_results = candidate.next_results
        next_best = candidate.next_best_success
        next_risk = candidate.risk_state
    next_step = int(state_data["step_index"]) + 1
    mode_state = json.loads(round_data["mode_state_json"] or "{}")
    step_limit = int(mode_state.get("score_step_limit") or int(room["target"]) // 2)
    ranking_min_steps = int(mode_state.get("ranking_min_steps") or step_limit)
    finish_reason = "risk_boundary" if risk_boundary else _finish_reason(
        board=next_board,
        target=int(room["target"]),
        step_index=next_step,
        step_limit=step_limit,
        next_best=next_best,
        use_variant=_use_variant(str(room["pattern"])),
    )
    certainty_steps: list[CertaintyTailStep] = []
    persisted_board = next_board
    persisted_results = next_results
    if finish_reason == "certainty":
        try:
            certainty_steps, persisted_board, tail_completed = await _generate_certainty_tail_best_effort(
                room=room,
                round_id=str(round_data["round_id"]),
                actor_key=actor_key,
                board=next_board,
                results=next_results,
                seed_hex=str(round_data["route_seed"]),
                sequence=next_step,
                risk_state=next_risk,
            )
            persisted_results = {}
            if tail_completed:
                finish_reason = "target_reached"
        except asyncio.CancelledError:
            _resume_after_resolution_error(
                round_id=str(round_data["round_id"]),
                actor_key=actor_key,
                request_id=request_id,
                original_timeout=state_data.get("timeout_at"),
                resolution_started_at=now_text,
            )
            raise
        except Exception as exc:
            _resume_after_resolution_error(
                round_id=str(round_data["round_id"]),
                actor_key=actor_key,
                request_id=request_id,
                original_timeout=state_data.get("timeout_at"),
                resolution_started_at=now_text,
            )
            if isinstance(exc, TablebaseQueryOverloaded):
                raise BattleServiceError(
                    "TABLEBASE_BUSY",
                    "Tablebase service is busy. Please retry shortly.",
                    503,
                    extra=exc.payload,
                ) from exc
            raise
    corrected = executed_direction != direction or bool(correction_reason and correction_reason != "risk_boundary")
    current_goodness = (
        float(result_row["goodness_of_fit"])
        if int(room.get("mode_version") or 1) >= 2
        else _legacy_goodness_product(bytes(state_data["operation_blob"] or b""))
    )
    cumulative_goodness = accumulate_goodness(current_goodness, decision.goodness)
    requires_ack = bool(corrected and not finish_reason)
    ack_deadline = (
        iso(utcnow() + timedelta(seconds=ACK_GRACE_SECONDS + CORRECTION_WINDOW_SECONDS))
        if requires_ack
        else None
    )
    next_timeout = (
        iso(utcnow() + timedelta(seconds=int(room["step_timeout_seconds"])))
        if not finish_reason and not requires_ack
        else None
    )
    operation_blob = _append_operation(
        bytes(state_data["operation_blob"] or b""),
        selected=direction,
        executed=executed_direction,
        corrected=corrected,
        spawn_index=spawn_index,
        spawn_value=spawn_value,
        goodness=decision.goodness,
    )
    replay_blob = bytes(result_row["replay_blob"] or b"")
    replay_move_count = int(result_row["replay_move_count"] or 0)
    if 0 <= spawn_index <= 15 and spawn_value in (2, 4):
        replay_step = encode_replay_step(
            board=board,
            selected_direction=direction,
            spawn_index=spawn_index,
            spawn_value=spawn_value,
            rates=results,
        )
        replay_blob, replay_recorded = append_replay_step(
            replay_blob, replay_step
        )
        replay_move_count += int(replay_recorded)
    replay_blob, replay_move_count = _append_certainty_replay_steps(
        replay_blob, replay_move_count, certainty_steps
    )
    certainty_payload = [step.public_payload() for step in certainty_steps]
    auto_playback_key = (
        f"{round_data['round_id']}:{actor_key}:{requested_sequence}:certainty"
        if certainty_payload
        else ""
    )
    round_completed = False
    try:
        with auth_db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT state_status, resolution_request_id FROM battle_free_player_states WHERE round_id = ? AND actor_key = ?",
                (round_data["round_id"], actor_key),
            ).fetchone()
            if current is None or current["state_status"] != "resolving" or current["resolution_request_id"] != request_id:
                raise BattleServiceError("PROGRESS_CONFLICT", "Progress changed while resolving.", 409)
            next_state_status = (
                "finished"
                if finish_reason
                else "awaiting_ack" if requires_ack else "input"
            )
            db.execute(
                """
                UPDATE battle_free_player_states SET board_state = ?, step_index = ?, sequence = ?,
                    spawn_log_index = ?, spawn_log_floor = ?, rng_step = ?,
                    state_status = ?, finish_reason = ?,
                    resolution_request_id = NULL, resolution_started_at = NULL,
                    ack_deadline_at = ?, timeout_at = ?,
                    current_results_json = ?, operation_blob = ?, updated_at = ?
                WHERE round_id = ? AND actor_key = ?
                """,
                (
                    f"{persisted_board:016x}", next_step, requested_sequence,
                    next_risk.log_index, next_risk.log_floor,
                    next_step, next_state_status, finish_reason,
                    ack_deadline, next_timeout,
                    json.dumps(persisted_results, separators=(",", ":")), operation_blob, iso(),
                    round_data["round_id"], actor_key,
                ),
            )
            next_status = "completed" if finish_reason else "playing"
            ranking_eligible = _ranking_eligible(
                result_status=next_status,
                finish_reason=finish_reason,
                progress=next_step,
                ranking_min_steps=ranking_min_steps,
            )
            mode_data = {
                "finish_reason": finish_reason,
                "finish_class": _public_finish_class(ranking_eligible),
                "ranking_min_steps": ranking_min_steps,
                "ranking_eligible": ranking_eligible,
                "corrected_steps": None,
                "last_step": {
                    "sequence": requested_sequence,
                    "previous_board_hex": f"{board:016x}",
                    "board_hex": f"{next_board:016x}",
                    "selected_direction": direction,
                    "executed_direction": executed_direction,
                    "corrected": corrected,
                    "correction_reason": correction_reason,
                    "step_goodness": decision.goodness,
                    "spawn_index": spawn_index,
                    "spawn_value": spawn_value,
                },
            }
            if corrected:
                mode_data["correction"] = {
                    "selected_direction": direction,
                    "standard_direction": executed_direction,
                    "goodness_drop": max(0.0, 1.0 - float(decision.goodness)),
                    "previous_board_hex": f"{board:016x}",
                    "visible_until": iso(
                        utcnow() + timedelta(seconds=CORRECTION_WINDOW_SECONDS)
                    ),
                }
            if certainty_payload:
                mode_data.update(
                    {
                        "auto_steps": certainty_payload,
                        "auto_playback_key": auto_playback_key,
                        "auto_start_board_hex": f"{next_board:016x}",
                        "auto_final_board_hex": f"{persisted_board:016x}",
                    }
                )
            db.execute(
                """
                UPDATE battle_player_results SET status = ?, route_index = ?, last_sequence = ?,
                    goodness_of_fit = ?, primary_score = ?, secondary_score = ?, progress = ?,
                    mode_data_json = ?, board_state = ?, choice_blob = ?, finished_at = ?,
                    replay_blob = ?, replay_move_count = ?, timeout_at = ?,
                    updated_at = ? WHERE round_id = ? AND actor_key = ?
                """,
                (
                    next_status, next_step, requested_sequence,
                    cumulative_goodness, cumulative_goodness, next_step, next_step,
                    json.dumps(mode_data, separators=(",", ":")), f"{persisted_board:016x}",
                    operation_blob, iso() if finish_reason else None,
                    replay_blob, replay_move_count, next_timeout, iso(),
                    round_data["round_id"], actor_key,
                ),
            )
            fresh_round = db.execute(
                "SELECT mode_state_json FROM battle_rounds WHERE round_id = ?",
                (round_data["round_id"],),
            ).fetchone()
            aggregate = json.loads(fresh_round["mode_state_json"] or "{}")
            aggregate["lookup_hit_steps"] = int(aggregate.get("lookup_hit_steps") or 0) + 1
            db.execute(
                "UPDATE battle_rounds SET mode_state_json = ?, updated_at = ? WHERE round_id = ?",
                (json.dumps(aggregate, separators=(",", ":"), sort_keys=True), iso(), round_data["round_id"]),
            )
            if finish_reason:
                round_completed = _complete_round_if_done(
                    db, str(room["room_id"]), str(round_data["round_id"]), iso()
                )
            else:
                db.execute(
                    "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
                    (iso(), room["room_id"]),
                )
    except Exception:
        _resume_after_resolution_error(
            round_id=str(round_data["round_id"]),
            actor_key=actor_key,
            request_id=request_id,
            original_timeout=state_data.get("timeout_at"),
            resolution_started_at=now_text,
        )
        raise
    if not finish_reason:
        _schedule_prepare_state(
            room=room,
            round_id=str(round_data["round_id"]),
            actor_key=actor_key,
            sequence=next_step,
            board=next_board,
            results=next_results,
            risk_state=next_risk,
            seed_hex=str(round_data["route_seed"]),
            lane="prefetch",
        )
    if round_completed:
        _settle_round(str(round_data["round_id"]))
    return {
        "round_id": str(round_data["round_id"]),
        "sequence": requested_sequence,
        "route_index": next_step,
        "board_hex": f"{next_board:016x}",
        "previous_board_hex": f"{board:016x}",
        "selected_direction": direction,
        "executed_direction": executed_direction,
        "best_direction": decision.best_direction,
        "corrected": corrected,
        "correction_reason": correction_reason,
        "step_goodness": decision.goodness,
        "goodness_of_fit": cumulative_goodness,
        "spawn_index": spawn_index,
        "spawn_value": spawn_value,
        "complete": bool(finish_reason),
        "finish_reason": finish_reason,
        "awaiting_ack": requires_ack,
        "timeout_at": next_timeout,
        "auto_steps": certainty_payload,
        "auto_playback_key": auto_playback_key or None,
        "auto_final_board_hex": (
            f"{persisted_board:016x}" if certainty_payload else None
        ),
    }


def _ack_step(room_code: str, *, actor_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    requested_round = str(payload.get("round_id") or "")
    sequence = int(payload.get("sequence") or 0)
    now = utcnow()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room, _member, round_row = _load_round_context(db, room_code, actor_key)
        if round_row is None or str(round_row["round_id"]) != requested_round:
            raise BattleServiceError("ROUND_CONFLICT", "Round changed.", 409)
        state = db.execute(
            "SELECT * FROM battle_free_player_states WHERE round_id = ? AND actor_key = ?",
            (requested_round, actor_key),
        ).fetchone()
        if state is None or int(state["sequence"]) != sequence:
            raise BattleServiceError("PROGRESS_CONFLICT", "Progress changed.", 409)
        if state["state_status"] == "input":
            return {"round_id": requested_round, "sequence": sequence, "timeout_at": state["timeout_at"]}
        if state["state_status"] != "awaiting_ack":
            return {"round_id": requested_round, "sequence": sequence, "complete": True}
        deadline = iso(now + timedelta(seconds=int(room["step_timeout_seconds"])))
        db.execute(
            "UPDATE battle_free_player_states SET state_status = 'input', ack_deadline_at = NULL, timeout_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ?",
            (deadline, iso(now), requested_round, actor_key),
        )
        db.execute(
            "UPDATE battle_player_results SET timeout_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ?",
            (deadline, iso(now), requested_round, actor_key),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (iso(now), room["room_id"]),
        )
    return {"round_id": requested_round, "sequence": sequence, "timeout_at": deadline}


async def handle_action_for_mode(
    room_code: str,
    *,
    actor_key: str | None = None,
    user_id: int | None = None,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    actor_key = _identity_key(actor_key, user_id)
    if action == "move":
        return await _resolve_move(room_code, actor_key=actor_key, payload=payload)
    if action == "step_ready_ack":
        return _ack_step(room_code, actor_key=actor_key, payload=payload)
    raise BattleServiceError("BATTLE_ACTION_UNSUPPORTED", "Unsupported Battle action.", 409)


def settle_unstarted_round_for_mode(room_id: str, *, reason: str) -> None:
    settlement = ""
    round_id = ""
    now_text = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
            (room_id,),
        ).fetchone()
        if round_row is None:
            return
        round_id = str(round_row["round_id"])
        if round_row["status"] in {"ready", "preparing"}:
            db.execute(
                "UPDATE battle_rounds SET status = 'cancelled', updated_at = ? WHERE round_id = ?",
                (now_text, round_id),
            )
            settlement = "cancel"
        elif round_row["status"] == "running":
            mode_data = json.dumps(
                {"finish_reason": "room_closed", "finish_class": "unranked"},
                separators=(",", ":"),
            )
            db.execute(
                "UPDATE battle_free_player_states SET state_status = 'finished', finish_reason = 'room_closed', resolution_request_id = NULL, resolution_started_at = NULL, ack_deadline_at = NULL, timeout_at = NULL, updated_at = ? WHERE round_id = ? AND state_status != 'finished'",
                (now_text, round_id),
            )
            db.execute(
                "UPDATE battle_player_results SET status = 'disqualified', mode_data_json = ?, timeout_at = NULL, finished_at = ?, updated_at = ? WHERE round_id = ? AND status IN ('playing', 'disconnected')",
                (mode_data, now_text, now_text, round_id),
            )
            db.execute(
                "UPDATE battle_rounds SET status = 'completed', ended_at = ?, updated_at = ? WHERE round_id = ?",
                (now_text, now_text, round_id),
            )
            settlement = "finalize"
        elif round_row["status"] == "completed":
            settlement = "finalize"
    _discard_round_prefetch(round_id)
    if settlement == "cancel":
        _settle_round(round_id, cancelled=True, reason=reason)
    elif settlement == "finalize":
        _settle_round(round_id)


def forfeit_round_for_mode(
    room_code: str,
    *,
    actor_key: str | None = None,
    user_id: int | None = None,
    round_id: str,
) -> dict[str, Any]:
    actor_key = _identity_key(actor_key, user_id)
    completed = False
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room, member, round_row = _load_round_context(db, room_code, actor_key)
        if member["role"] != "player" or round_row is None or str(round_row["round_id"]) != str(round_id):
            raise BattleServiceError("ROUND_CONFLICT", "Round changed.", 409)
        now = iso()
        db.execute(
            "UPDATE battle_free_player_states SET state_status = 'finished', finish_reason = 'forfeit', timeout_at = NULL, ack_deadline_at = NULL, updated_at = ? WHERE round_id = ? AND actor_key = ?",
            (now, round_id, actor_key),
        )
        db.execute(
            "UPDATE battle_player_results SET status = 'disqualified', mode_data_json = ?, timeout_at = NULL, finished_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ? AND status = 'playing'",
            (json.dumps({"finish_reason": "forfeit", "finish_class": "unranked"}), now, now, round_id, actor_key),
        )
        completed = _complete_round_if_done(db, str(room["room_id"]), round_id, now)
    if completed:
        _settle_round(round_id)
    return room_snapshot(room_code, actor_key=actor_key)


def _mark_timeouts() -> set[str]:
    now = utcnow()
    now_text = iso(now)
    changed: set[str] = set()
    completed_rounds: list[str] = []
    cancelled_rounds: list[str] = []
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        ack_rows = db.execute(
            """
            SELECT state.*, room.step_timeout_seconds, round.room_id
            FROM battle_free_player_states AS state
            JOIN battle_rounds AS round ON round.round_id = state.round_id
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE state.state_status = 'awaiting_ack' AND state.ack_deadline_at <= ?
            """,
            (now_text,),
        ).fetchall()
        for row in ack_rows:
            deadline = iso(now + timedelta(seconds=int(row["step_timeout_seconds"])))
            db.execute(
                "UPDATE battle_free_player_states SET state_status = 'input', ack_deadline_at = NULL, timeout_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ? AND state_status = 'awaiting_ack'",
                (deadline, now_text, row["round_id"], row["actor_key"]),
            )
            db.execute(
                "UPDATE battle_player_results SET timeout_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ? AND status = 'playing'",
                (deadline, now_text, row["round_id"], row["actor_key"]),
            )
            changed.add(str(row["room_id"]))
        timeout_rows = db.execute(
            """
            SELECT state.*, round.room_id FROM battle_free_player_states AS state
            JOIN battle_rounds AS round ON round.round_id = state.round_id
            WHERE state.state_status = 'input' AND state.timeout_at <= ?
            """,
            (now_text,),
        ).fetchall()
        touched_rounds: set[tuple[str, str]] = set()
        for row in timeout_rows:
            db.execute(
                "UPDATE battle_free_player_states SET state_status = 'finished', finish_reason = 'timed_out', timeout_at = NULL, updated_at = ? WHERE round_id = ? AND actor_key = ? AND state_status = 'input'",
                (now_text, row["round_id"], row["actor_key"]),
            )
            db.execute(
                "UPDATE battle_player_results SET status = 'timed_out', mode_data_json = ?, timeout_at = NULL, finished_at = ?, updated_at = ? WHERE round_id = ? AND actor_key = ? AND status = 'playing'",
                (json.dumps({"finish_reason": "timed_out", "finish_class": "unranked"}), now_text, now_text, row["round_id"], row["actor_key"]),
            )
            touched_rounds.add((str(row["room_id"]), str(row["round_id"])))
            changed.add(str(row["room_id"]))
        for room_id, round_id in touched_rounds:
            if _complete_round_if_done(db, room_id, round_id, now_text):
                completed_rounds.append(round_id)
        expired_rounds = db.execute(
            """
            SELECT round.round_id, round.room_id
            FROM battle_rounds AS round
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE room.mode_key = 'free_goodness' AND round.status = 'running'
              AND round.expires_at IS NOT NULL AND round.expires_at <= ?
            """,
            (now_text,),
        ).fetchall()
        for row in expired_rounds:
            db.execute(
                "UPDATE battle_free_player_states SET state_status = 'finished', finish_reason = 'timed_out', timeout_at = NULL, ack_deadline_at = NULL, updated_at = ? WHERE round_id = ? AND state_status != 'finished'",
                (now_text, row["round_id"]),
            )
            db.execute(
                "UPDATE battle_player_results SET status = 'timed_out', mode_data_json = ?, timeout_at = NULL, finished_at = ?, updated_at = ? WHERE round_id = ? AND status = 'playing'",
                (json.dumps({"finish_reason": "timed_out", "finish_class": "unranked"}), now_text, now_text, row["round_id"]),
            )
            if _complete_round_if_done(db, str(row["room_id"]), str(row["round_id"]), now_text):
                completed_rounds.append(str(row["round_id"]))
            changed.add(str(row["room_id"]))
        stale_rooms = db.execute(
            "SELECT * FROM battle_rooms WHERE mode_key = 'free_goodness' AND lifecycle_kind = 'normal' AND status IN ('preparing', 'waiting') AND expires_at <= ?",
            (now_text,),
        ).fetchall()
        for room in stale_rooms:
            round_row = db.execute(
                "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
                (room["room_id"],),
            ).fetchone()
            if round_row is not None and round_row["status"] in {"preparing", "ready"}:
                db.execute(
                    "UPDATE battle_rounds SET status = 'cancelled', updated_at = ? WHERE round_id = ?",
                    (now_text, round_row["round_id"]),
                )
                cancelled_rounds.append(str(round_row["round_id"]))
            db.execute(
                "UPDATE battle_rooms SET status = 'expired', closed_at = ?, updated_at = ?, revision = revision + 1 WHERE room_id = ?",
                (now_text, now_text, room["room_id"]),
            )
            db.execute(
                "UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ? WHERE room_id = ? AND status = 'active'",
                (now_text, now_text, room["room_id"]),
            )
            changed.add(str(room["room_id"]))
    for round_id in completed_rounds:
        _settle_round(round_id)
    for round_id in cancelled_rounds:
        _settle_round(round_id, cancelled=True, reason="battle_free_room_expired")
    return changed


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(1)
        for room_id in await asyncio.to_thread(_mark_timeouts):
            await broadcast_room(room_id)


def _migrate_legacy_goodness_scores() -> None:
    with auth_db() as db:
        rooms = db.execute(
            "SELECT room_id, settings_json FROM battle_rooms "
            "WHERE mode_key = 'free_goodness' AND mode_version < ?",
            (_free_mode.version,),
        ).fetchall()
        for room in rooms:
            states = db.execute(
                """
                SELECT state.round_id, state.actor_key, state.operation_blob
                FROM battle_free_player_states AS state
                JOIN battle_rounds AS round ON round.round_id = state.round_id
                WHERE round.room_id = ?
                """,
                (room["room_id"],),
            ).fetchall()
            try:
                scores = [
                    (
                        _legacy_goodness_product(bytes(state["operation_blob"] or b"")),
                        state["round_id"],
                        state["actor_key"],
                    )
                    for state in states
                ]
            except BattleServiceError:
                continue
            for score, round_id, actor_key in scores:
                db.execute(
                    """
                    UPDATE battle_player_results
                    SET goodness_of_fit = ?, primary_score = ?
                    WHERE round_id = ? AND actor_key = ?
                    """,
                    (score, score, round_id, actor_key),
                )
            try:
                settings = json.loads(room["settings_json"] or "{}")
            except (TypeError, ValueError):
                settings = {}
            settings["rules_version"] = _free_mode.version
            db.execute(
                "UPDATE battle_rooms SET mode_version = ?, settings_json = ? WHERE room_id = ?",
                (
                    _free_mode.version,
                    json.dumps(settings, separators=(",", ":"), sort_keys=True),
                    room["room_id"],
                ),
            )


async def startup() -> None:
    global _cleanup_task
    repository.init_battle_db()
    _migrate_legacy_goodness_scores()
    with auth_db() as db:
        pending = db.execute(
            """
            SELECT round.round_id, round.status AS round_status,
                   round.reservation_status, room.status AS room_status
            FROM battle_rounds AS round
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE room.mode_key = 'free_goodness'
              AND round.reservation_status = 'reserved'
            """
        ).fetchall()
        interrupted = db.execute(
            """
            SELECT state.round_id, state.actor_key, room.step_timeout_seconds
            FROM battle_free_player_states AS state
            JOIN battle_rounds AS round ON round.round_id = state.round_id
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE room.mode_key = 'free_goodness' AND round.status = 'running'
              AND state.state_status IN ('resolving', 'awaiting_ack')
            """
        ).fetchall()
        now = utcnow()
        now_text = iso(now)
        for state in interrupted:
            deadline = iso(
                now + timedelta(seconds=int(state["step_timeout_seconds"] or 90))
            )
            db.execute(
                """
                UPDATE battle_free_player_states
                SET state_status = 'input', resolution_request_id = NULL,
                    resolution_started_at = NULL, ack_deadline_at = NULL,
                    timeout_at = ?, updated_at = ?
                WHERE round_id = ? AND actor_key = ?
                  AND state_status IN ('resolving', 'awaiting_ack')
                """,
                (deadline, now_text, state["round_id"], state["actor_key"]),
            )
            db.execute(
                """
                UPDATE battle_player_results SET timeout_at = ?, updated_at = ?
                WHERE round_id = ? AND actor_key = ? AND status = 'playing'
                """,
                (deadline, now_text, state["round_id"], state["actor_key"]),
            )
    for row in pending:
        if row["round_status"] == "cancelled" or (
            row["room_status"] in {"closed", "expired"}
            and row["round_status"] in {"preparing", "ready"}
        ):
            _settle_round(
                str(row["round_id"]),
                cancelled=True,
                reason="battle_free_reservation_recovery",
            )
        elif row["round_status"] == "completed" or (
            row["round_status"] == "running"
            and row["room_status"] in {"closed", "expired"}
        ):
            _settle_round(str(row["round_id"]))
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


async def shutdown() -> None:
    global _cleanup_task
    tasks: list[asyncio.Task] = []
    if _cleanup_task is not None:
        tasks.append(_cleanup_task)
        _cleanup_task = None
    tasks.extend(_prepare_state_tasks.values())
    _prepare_state_tasks.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _prepared.clear()
    _prepared_complete.clear()
