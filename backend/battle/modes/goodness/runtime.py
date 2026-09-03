from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np

from Config import category_info
from engine_core.BoardMover import s_move_board as classic_move_board
from engine_core.VBoardMover import decode_board, encode_board, s_move_board as variant_move_board

from backend.auth.db import auth_db
from backend.quota.config import operation_cost_units, table_multiplier_units
from backend.quota.service import (
    TokenReservation,
    cancel_reservation,
    finalize_reservation,
    get_token_balance,
    reserve_operation_tokens,
)
from backend.tablebase_catalog import resolve_tablebase

from ... import repository
from .route_codec import (
    DIRECTION_CODES,
    DIRECTION_NAMES,
    decode_changes,
    decode_route,
    route_sha256,
)
from .route_generator import BattleRouteGenerationError, generate_battle_route
from .scoring import score_recorded_choice
from ...core.contracts import BattleModeError
from ...core.errors import BattleServiceError
from ...core.lifecycle import (
    assert_member as _assert_member,
    broadcast_room,
    current_room,
    iso,
    join_room,
    kick,
    leave_room,
    list_rooms,
    parse_iso,
    room_by_user as _room_by_user,
    room_snapshot,
    set_broadcast_callback,
    set_ready,
    set_role,
    utcnow,
)
from ...core.registry import get_battle_mode, register_battle_mode
from ...replay_records import append_replay_step, encode_replay_step
from .mode import GoodnessBattleMode


ROUND_LIFETIME = timedelta(minutes=60)
WAITING_LIFETIME = timedelta(minutes=30)
MAX_SPECTATORS = 64
VALID_STEP_TIMEOUTS = set(range(5, 601, 5))
UNSTARTED_ROOM_REFUND_PERCENT = 80
UNSTARTED_ROOM_CHARGE_PERCENT = 100 - UNSTARTED_ROOM_REFUND_PERCENT
CORRECTION_WINDOW_SECONDS = 15


def _ready_players_for_start(
    db: sqlite3.Connection,
    room: sqlite3.Row,
    *,
    now_text: str,
) -> list[sqlite3.Row]:
    players = db.execute(
        "SELECT * FROM battle_members WHERE room_id = ? AND status = 'active' AND role = 'player' ORDER BY seat_index",
        (room["room_id"],),
    ).fetchall()
    host = next(
        (player for player in players if int(player["user_id"]) == int(room["host_user_id"])),
        None,
    )
    if host is None or not bool(host["ready"]):
        raise BattleServiceError("HOST_NOT_READY", "The host must be ready to start.", 409)
    ready_players = [player for player in players if bool(player["ready"])]
    if len(ready_players) < 1:
        raise BattleServiceError(
            "NOT_ENOUGH_PLAYERS",
            "At least one ready player is required.",
            409,
            extra={"required_players": 1},
        )
    unready_players = [player for player in players if not bool(player["ready"])]
    if unready_players:
        member_ids = [int(player["member_id"]) for player in unready_players]
        placeholders = ",".join("?" for _member_id in member_ids)
        if bool(room["allow_spectators"]):
            db.execute(
                f"UPDATE battle_members SET role = 'spectator', seat_index = NULL, ready = 0, updated_at = ? WHERE member_id IN ({placeholders})",
                (now_text, *member_ids),
            )
        else:
            db.execute(
                f"UPDATE battle_members SET status = 'kicked', ready = 0, left_at = ?, updated_at = ? WHERE member_id IN ({placeholders})",
                (now_text, now_text, *member_ids),
            )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now_text, room["room_id"]),
        )
    return ready_players


_route_tasks: dict[str, asyncio.Task] = {}
_cleanup_task: asyncio.Task | None = None
_recovery_task: asyncio.Task | None = None
_goodness_mode = register_battle_mode(GoodnessBattleMode(), replace=True)
_broadcast = broadcast_room


def _reservation_payload(reservation: TokenReservation) -> dict[str, int]:
    return {
        "ledger_id": reservation.ledger_id,
        "reserved_bonus_units": reservation.reserved_bonus_units,
        "reserved_paid_units": reservation.reserved_paid_units,
        "reserved_units": reservation.reserved_units,
    }


def _restore_reservation(round_row: sqlite3.Row, room_row: sqlite3.Row) -> TokenReservation | None:
    ledger_id = round_row["reservation_ledger_id"]
    if ledger_id is None:
        return None
    multiplier = table_multiplier_units(str(room_row["full_pattern"]))
    return TokenReservation(
        ledger_id=int(ledger_id),
        user_id=int(room_row["host_user_id"]),
        session_id=None,
        operation_key="battle_route_generation",
        table_pattern=str(room_row["full_pattern"]),
        table_multiplier_units=multiplier,
        base_cost_units=operation_cost_units("battle_route_generation"),
        reserved_units=int(round_row["reserved_bonus_units"] or 0)
        + int(round_row["reserved_paid_units"] or 0),
        reserved_bonus_units=int(round_row["reserved_bonus_units"] or 0),
        reserved_paid_units=int(round_row["reserved_paid_units"] or 0),
    )


def _insert_round(
    *,
    room_id: str,
    round_number: int,
    seed_hex: str,
    reservation: TokenReservation,
    auto_start_after_generation: bool = False,
) -> str:
    round_id = str(uuid.uuid4())
    now = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        if auto_start_after_generation:
            room = repository._find_room(db, room_id)
            previous_round = db.execute(
                "SELECT status FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
                (room_id,),
            ).fetchone()
            if room["status"] != "waiting" or previous_round is None or previous_round["status"] != "completed":
                raise BattleServiceError("ROOM_NOT_READY", "The previous round is not complete.", 409)
            _ready_players_for_start(db, room, now_text=now)
        db.execute(
            """
            INSERT INTO battle_rounds
            (round_id, room_id, round_number, status, token_reservation_id,
             token_cost_units, created_at, updated_at, reservation_ledger_id,
             reserved_bonus_units, reserved_paid_units, route_seed,
             reservation_status, auto_start_after_generation, artifact_kind)
            VALUES (?, ?, ?, 'preparing', ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?)
            """,
            (
                round_id,
                room_id,
                int(round_number),
                str(reservation.ledger_id),
                reservation.reserved_units,
                now,
                now,
                reservation.ledger_id,
                reservation.reserved_bonus_units,
                reservation.reserved_paid_units,
                seed_hex,
                1 if auto_start_after_generation else 0,
                _goodness_mode.artifact_kind,
            ),
        )
        db.execute(
            """
            UPDATE battle_rooms
            SET current_round_number = ?, status = 'preparing', generation_error = NULL,
                revision = revision + 1, updated_at = ?
            WHERE room_id = ?
            """,
            (int(round_number), now, room_id),
        )
    return round_id


def _settle_round_reservation(
    round_id: str,
    reservation: TokenReservation | None,
    *,
    settlement: str,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    if reservation is None:
        return
    with auth_db() as db:
        row = db.execute(
            "SELECT reservation_status FROM battle_rounds WHERE round_id = ?",
            (round_id,),
        ).fetchone()
    if row is None or str(row["reservation_status"] or "reserved") != "reserved":
        return
    if settlement == "finalized":
        finalize_reservation(
            reservation,
            actual_operation_key="battle_route_generation",
            metadata=metadata,
        )
    elif settlement == "unstarted_refunded":
        actual_base_units = (
            reservation.base_cost_units * UNSTARTED_ROOM_CHARGE_PERCENT + 99
        ) // 100
        finalize_reservation(
            reservation,
            actual_operation_key="battle_route_generation",
            actual_base_units=actual_base_units,
            metadata={
                "refund_percent": UNSTARTED_ROOM_REFUND_PERCENT,
                "reason": reason or "battle_room_not_started",
                **(metadata or {}),
            },
        )
    elif settlement == "cancelled":
        cancel_reservation(
            reservation,
            reason=reason or "battle_route_cancelled",
            metadata=metadata,
        )
    else:
        raise ValueError("invalid battle reservation settlement")
    with auth_db() as db:
        settled = db.execute(
            """
            SELECT settlement.settlement_type, ledger.final_cost_units
            FROM token_reservation_settlements AS settlement
            LEFT JOIN token_ledger AS ledger ON ledger.id = settlement.settlement_ledger_id
            WHERE settlement.reservation_ledger_id = ?
            """,
            (reservation.ledger_id,),
        ).fetchone()
        if settled is None:
            return
        if settled["settlement_type"] == "cancel":
            durable_status = "cancelled"
        elif int(settled["final_cost_units"] or 0) < reservation.reserved_units:
            durable_status = "unstarted_refunded"
        else:
            durable_status = "finalized"
        db.execute(
            "UPDATE battle_rounds SET reservation_status = ?, updated_at = ? WHERE round_id = ? AND reservation_status = 'reserved'",
            (durable_status, iso(), round_id),
        )


async def create_room_for_mode(
    *,
    user_id: int,
    session_id: int | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if _room_by_user(user_id) is not None:
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
        settings = _goodness_mode.validate_settings(payload)
    except ValueError as exc:
        code = str(exc).upper()
        messages = {
            "TABLE_UNAVAILABLE": "The selected tablebase is unavailable.",
            "INVALID_BOARD": "Initial board must contain 16 hexadecimal digits.",
            "INVALID_MAX_STEPS": "Maximum steps must be between 1 and 9999.",
            "INVALID_STEP_TIMEOUT": "Unsupported step timeout.",
            "INVALID_MAX_PLAYERS": "Player count must be between 2 and 8.",
            "INVALID_CHAT_ROLES": "Unsupported chat role selection.",
        }
        raise BattleServiceError(code, messages.get(code, str(exc).replace("_", " ")), 409) from exc
    full_pattern = str(settings["full_pattern"])

    reservation = reserve_operation_tokens(
        user_id=user_id,
        session_id=session_id,
        operation_key="battle_route_generation",
        full_pattern=full_pattern,
    )
    if reservation is None:
        raise BattleServiceError("TOKEN_CONFIGURATION_ERROR", "Battle token cost is not configured.", 500)
    try:
        room = repository.create_room(
            host_user_id=user_id,
            pattern=str(settings["pattern"]),
            target=int(settings["target"]),
            full_pattern=full_pattern,
            visibility=str(settings["visibility"]),
            allow_spectators=bool(settings["allow_spectators"]),
            allow_guest_chat=bool(settings["allow_guest_chat"]),
            max_players=int(settings["max_players"]),
            initial_board=settings["initial_board"],
            max_steps=settings["max_steps"],
            step_timeout_seconds=int(settings["step_timeout_seconds"]),
            mode_key=_goodness_mode.key,
            mode_version=_goodness_mode.version,
            chat_roles=settings["chat_roles"],
            settings=_goodness_mode.public_settings({
                "settings": {key: value for key, value in settings.items() if key != "chat_roles"}
            }),
            status="preparing",
        )
        seed_hex = secrets.token_hex(16)
        round_id = _insert_round(
            room_id=str(room["room_id"]),
            round_number=1,
            seed_hex=seed_hex,
            reservation=reservation,
            auto_start_after_generation=False,
        )
    except Exception as exc:
        cancel_reservation(reservation, reason="battle_room_not_created")
        if (
            isinstance(exc, repository.BattleConflictError)
            and str(exc) == "room_create_cooldown"
        ):
            raise BattleServiceError(
                "ROOM_CREATE_COOLDOWN",
                "A new room can only be created once every 30 seconds.",
                429,
                extra={
                    "retry_after_seconds": max(
                        1, repository.room_creation_retry_after(user_id)
                    )
                },
            ) from exc
        raise
    _schedule_route(round_id, reservation=reservation, auto_start=False)
    snapshot = room_snapshot(str(room["room_id"]), user_id=user_id)
    return {"room": snapshot, "token_balance": get_token_balance(user_id)}


def _schedule_route(
    round_id: str,
    *,
    reservation: TokenReservation | None = None,
    auto_start: bool,
) -> None:
    previous = _route_tasks.get(round_id)
    if previous is not None and not previous.done():
        return
    task = asyncio.create_task(
        _prepare_route(round_id, reservation=reservation, auto_start=auto_start)
    )
    _route_tasks[round_id] = task
    task.add_done_callback(lambda _task, key=round_id: _route_tasks.pop(key, None))


async def _prepare_route(
    round_id: str,
    *,
    reservation: TokenReservation | None,
    auto_start: bool,
) -> None:
    with auth_db() as db:
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if round_row is None or round_row["status"] != "preparing":
            return
        room_row = db.execute(
            "SELECT * FROM battle_rooms WHERE room_id = ?", (round_row["room_id"],)
        ).fetchone()
        if room_row is None:
            return
        if reservation is None:
            reservation = _restore_reservation(round_row, room_row)
        room = dict(room_row)
        route_seed = str(round_row["route_seed"] or secrets.token_hex(16))
        auto_start = bool(round_row["auto_start_after_generation"])
    try:
        generated = await generate_battle_route(
            pattern=str(room["pattern"]),
            target=int(room["target"]),
            full_pattern=str(room["full_pattern"]),
            initial_board=(
                None if not room.get("initial_board") else int(str(room["initial_board"]), 16)
            ),
            max_steps=(None if room.get("max_steps") is None else int(room["max_steps"])),
            seed_hex=route_seed,
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, BattleRouteGenerationError) else "ROUTE_GENERATION_FAILED"
        now = iso()
        with auth_db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT status FROM battle_rounds WHERE round_id = ?", (round_id,)
            ).fetchone()
            if current is None or current["status"] != "preparing":
                return
            db.execute(
                "UPDATE battle_rounds SET status = 'failed', error_code = ?, updated_at = ? WHERE round_id = ?",
                (code, now, round_id),
            )
            db.execute(
                """
                UPDATE battle_rooms SET status = 'closed', generation_error = ?,
                    closed_at = ?, updated_at = ?, revision = revision + 1
                WHERE room_id = ?
                """,
                (code, now, now, room["room_id"]),
            )
            db.execute(
                "UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ? WHERE room_id = ? AND status = 'active'",
                (now, now, room["room_id"]),
            )
        _settle_round_reservation(
            round_id,
            reservation,
            settlement="cancelled",
            reason="battle_route_generation_failed",
            metadata={"room_id": room["room_id"], "round_id": round_id, "code": code},
        )
        await _broadcast(str(room["room_id"]))
        return

    now = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute(
            "SELECT status FROM battle_rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if current is None or current["status"] != "preparing":
            return
        db.execute(
            """
            INSERT OR REPLACE INTO battle_routes
            (route_id, round_id, full_pattern, target, initial_board, step_count,
             certainty_step, termination_reason, route_hash, route_blob, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                round_id,
                room["full_pattern"],
                room["target"],
                f"{generated.initial_board:016x}",
                generated.step_count,
                generated.certainty_step,
                generated.termination_reason,
                route_sha256(generated.route_blob),
                generated.route_blob,
                now,
            ),
        )
        db.execute(
            """
            UPDATE battle_rounds
            SET status = 'ready', updated_at = ?, error_code = NULL,
                artifact_kind = ?, artifact_hash = ?
            WHERE round_id = ?
            """,
            (
                now,
                _goodness_mode.artifact_kind,
                route_sha256(generated.route_blob),
                round_id,
            ),
        )
        db.execute(
            """
            UPDATE battle_rooms
            SET status = 'waiting', generation_error = NULL, expires_at = ?,
                revision = revision + 1, updated_at = ?
            WHERE room_id = ?
            """,
            (iso(utcnow() + WAITING_LIFETIME), now, room["room_id"]),
        )
    if auto_start:
        try:
            _start_ready_round(str(room["room_id"]), host_user_id=int(room["host_user_id"]))
        except BattleServiceError:
            pass
    await _broadcast(str(room["room_id"]))


def settle_unstarted_round_for_mode(room_id: str, *, reason: str) -> None:
    reservation: TokenReservation | None = None
    round_id = ""
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
            (room_id,),
        ).fetchone()
        room_row = db.execute(
            "SELECT * FROM battle_rooms WHERE room_id = ?",
            (room_id,),
        ).fetchone()
        if (
            round_row is None
            or room_row is None
            or round_row["status"] not in {"preparing", "ready"}
        ):
            return
        round_id = str(round_row["round_id"])
        reservation = _restore_reservation(round_row, room_row)
        db.execute(
            "UPDATE battle_rounds SET status = 'cancelled', error_code = ?, updated_at = ? WHERE round_id = ? AND status IN ('preparing', 'ready')",
            (reason[:120], iso(), round_id),
        )
    _settle_round_reservation(
        round_id,
        reservation,
        settlement="unstarted_refunded",
        reason=reason,
        metadata={"room_id": room_id, "round_id": round_id},
    )


def _start_ready_round(room_ref: str, *, host_user_id: int) -> dict[str, Any]:
    now = utcnow()
    now_text = iso(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_ref)
        if int(room["host_user_id"]) != int(host_user_id):
            raise BattleServiceError("HOST_REQUIRED", "Only the host can start.", 403)
        if room["status"] != "waiting":
            raise BattleServiceError("ROOM_NOT_READY", "Room is not ready to start.", 409)
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
            (room["room_id"],),
        ).fetchone()
        if round_row is None or round_row["status"] != "ready":
            raise BattleServiceError("ROUTE_NOT_READY", "Battle route is not ready.", 409)
        route = db.execute(
            "SELECT * FROM battle_routes WHERE round_id = ?", (round_row["round_id"],)
        ).fetchone()
        if route is None:
            raise BattleServiceError("ROUTE_NOT_READY", "Battle route is not ready.", 409)
        players = _ready_players_for_start(db, room, now_text=now_text)
        reservation = _restore_reservation(round_row, room)
        deadline = iso(now + timedelta(seconds=int(room["step_timeout_seconds"])))
        for player in players:
            initial_status = (
                "completed"
                if route["certainty_step"] is not None and int(route["certainty_step"]) == 0
                else "playing"
            )
            route_index = int(route["step_count"]) if initial_status == "completed" else 0
            db.execute(
                """
                INSERT OR REPLACE INTO battle_player_results
                (result_id, round_id, actor_key, user_id, guest_id,
                 display_name_snapshot, status, route_index, last_sequence,
                 goodness_of_fit, choice_blob, finished_at, timeout_at, created_at, updated_at,
                 board_state)
                VALUES (
                  (SELECT result_id FROM battle_player_results WHERE round_id = ? AND actor_key = ?),
                  ?, ?, ?, ?, ?, ?, ?, 0, 1.0, X'', ?, ?, ?, ?, ?
                )
                """,
                (
                    round_row["round_id"],
                    player["actor_key"],
                    round_row["round_id"],
                    player["actor_key"],
                    player["user_id"],
                    player["guest_id"],
                    player["display_name_snapshot"],
                    initial_status,
                    route_index,
                    now_text if initial_status == "completed" else None,
                    None if initial_status == "completed" else deadline,
                    now_text,
                    now_text,
                    str(route["initial_board"]),
                ),
            )
        all_completed = all(
            route["certainty_step"] is not None and int(route["certainty_step"]) == 0
            for _player in players
        )
        db.execute(
            "UPDATE battle_rounds SET status = 'running', started_at = ?, expires_at = ?, updated_at = ? WHERE round_id = ?",
            (now_text, iso(now + ROUND_LIFETIME), now_text, round_row["round_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET status = 'running', expires_at = ?, revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (iso(now + ROUND_LIFETIME), now_text, room["room_id"]),
        )
        if all_completed:
            _complete_round_if_done(
                db,
                room_id=str(room["room_id"]),
                round_id=str(round_row["round_id"]),
                now_text=now_text,
            )
    try:
        _settle_round_reservation(
            str(round_row["round_id"]),
            reservation,
            settlement="finalized",
            metadata={
                "room_id": room["room_id"],
                "round_id": round_row["round_id"],
                "started": True,
            },
        )
    except Exception:
        # The balance is already reserved; startup recovery can finish the audit settlement.
        pass
    return repository.get_room(str(room["room_id"]))


async def start_room_for_mode(
    room_code: str,
    *,
    user_id: int,
    session_id: int | None,
) -> dict[str, Any]:
    room = repository.get_room(room_code)
    if int(room["host_user_id"]) != int(user_id):
        raise BattleServiceError("HOST_REQUIRED", "Only the host can start.", 403)
    current_round = room.get("round") or {}
    if current_round.get("status") == "completed":
        reservation = reserve_operation_tokens(
            user_id=user_id,
            session_id=session_id,
            operation_key="battle_route_generation",
            full_pattern=str(room["full_pattern"]),
        )
        if reservation is None:
            raise BattleServiceError("TOKEN_CONFIGURATION_ERROR", "Battle token cost is not configured.", 500)
        round_number = int(room.get("current_round_number") or 0) + 1
        try:
            round_id = _insert_round(
                room_id=str(room["room_id"]),
                round_number=round_number,
                seed_hex=secrets.token_hex(16),
                reservation=reservation,
                auto_start_after_generation=True,
            )
        except Exception:
            cancel_reservation(reservation, reason="battle_next_round_not_created")
            raise
        _schedule_route(round_id, reservation=reservation, auto_start=True)
        snapshot = room_snapshot(room_code, user_id=user_id)
        await _broadcast(str(room["room_id"]))
        return snapshot
    _start_ready_round(room_code, host_user_id=user_id)
    await _broadcast(str(room["room_id"]))
    return room_snapshot(room_code, user_id=user_id)


def artifact_payload_for_mode(
    room_code: str, round_id: str, *, actor_key: str
) -> tuple[bytes, dict[str, Any]]:
    room = repository.get_room(room_code)
    _assert_member(room, actor_key=actor_key)
    with auth_db() as db:
        row = db.execute(
            """
            SELECT route.* FROM battle_routes AS route
            JOIN battle_rounds AS round ON round.round_id = route.round_id
            WHERE route.round_id = ? AND round.room_id = ?
            """,
            (str(round_id), room["room_id"]),
        ).fetchone()
    if row is None:
        raise BattleServiceError("ROUTE_NOT_FOUND", "Battle route is not available.", 404)
    return bytes(row["route_blob"]), dict(row)


def _apply_route_step(board: int, step, *, use_variant: bool) -> int:
    move_board = variant_move_board if use_variant else classic_move_board
    decoded = decode_changes(step.changes)
    moved, _score = move_board(np.uint64(board), MOVE_CODE_BY_DIRECTION[decoded.direction])
    if int(moved) == int(board):
        raise BattleServiceError("ROUTE_CORRUPT", "Stored battle route is invalid.", 500)
    array = decode_board(np.uint64(moved)).copy()
    row, column = divmod(decoded.spawn_index, 4)
    if int(array[row, column]) != 0:
        raise BattleServiceError("ROUTE_CORRUPT", "Stored battle route is invalid.", 500)
    array[row, column] = decoded.spawn_value
    return int(encode_board(array))


def _board_at(route_blob: bytes, index: int, *, use_variant: bool) -> int:
    route = decode_route(route_blob)
    board = int(route.initial_board)
    for step in route.steps[: int(index)]:
        board = _apply_route_step(board, step, use_variant=use_variant)
    return board


MOVE_CODE_BY_DIRECTION = {"left": 1, "right": 2, "up": 3, "down": 4}


def _record_choice_goodness(
    room_code: str,
    *,
    actor_key: str,
    round_id: str,
    sequence: int,
    route_index: int,
    direction: str,
) -> dict[str, Any]:
    direction = str(direction or "").lower()
    if direction not in DIRECTION_CODES:
        raise BattleServiceError("INVALID_DIRECTION", "Invalid direction.")
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_code)
        if room["status"] != "running":
            raise BattleServiceError("ROUND_NOT_RUNNING", "Round is not running.", 409)
        round_row = db.execute(
            "SELECT * FROM battle_rounds WHERE round_id = ? AND room_id = ?",
            (str(round_id), room["room_id"]),
        ).fetchone()
        result = db.execute(
            "SELECT * FROM battle_player_results WHERE round_id = ? AND actor_key = ?",
            (str(round_id), actor_key),
        ).fetchone()
        route_row = db.execute(
            "SELECT * FROM battle_routes WHERE round_id = ?", (str(round_id),)
        ).fetchone()
        if round_row is None or result is None or route_row is None:
            raise BattleServiceError("ROUND_NOT_FOUND", "Round state is unavailable.", 404)
        if result["status"] != "playing":
            raise BattleServiceError("PLAYER_FINISHED", "This player has already finished.", 409)
        expected_sequence = int(result["last_sequence"]) + 1
        expected_index = int(result["route_index"])
        if int(sequence) != expected_sequence or int(route_index) != expected_index:
            raise BattleServiceError("PROGRESS_CONFLICT", "Battle progress is out of date.", 409)
        route = decode_route(bytes(route_row["route_blob"]))
        if expected_index >= len(route.steps):
            raise BattleServiceError("ROUTE_COMPLETE", "Battle route is complete.", 409)
        use_variant = str(room["pattern"]) in category_info.get("variant", [])
        board = (
            int(str(result["board_state"]), 16)
            if result["board_state"]
            else _board_at(bytes(route_row["route_blob"]), expected_index, use_variant=use_variant)
        )
        move_fn = variant_move_board if use_variant else classic_move_board
        selected_board, _score = move_fn(
            np.uint64(board), MOVE_CODE_BY_DIRECTION[direction]
        )
        if int(selected_board) == int(board):
            raise BattleServiceError("ILLEGAL_DIRECTION", "Illegal directions do not advance the route.", 409)
        step = route.steps[expected_index]
        decoded_step = decode_changes(step.changes)
        next_board = _apply_route_step(board, step, use_variant=use_variant)
        score = score_recorded_choice(
            step.rates,
            direction,
            current_goodness=float(result["goodness_of_fit"]),
        )
        standard_direction = decoded_step.direction
        wrong = direction != standard_direction
        next_index = expected_index + 1
        complete = next_index >= len(route.steps) or (
            route_row["certainty_step"] is not None
            and next_index >= int(route_row["certainty_step"])
        )
        status = "completed" if complete else "playing"
        stored_index = len(route.steps) if complete else next_index
        timeout_at = (
            None
            if complete
            else iso(
                utcnow()
                + timedelta(
                    seconds=(
                        int(room["step_timeout_seconds"])
                        + (CORRECTION_WINDOW_SECONDS if wrong else 0)
                    )
                )
            )
        )
        choices = bytes(result["choice_blob"] or b"") + bytes([DIRECTION_CODES[direction]])
        replay_step = encode_replay_step(
            board=board,
            selected_direction=direction,
            spawn_index=decoded_step.spawn_index,
            spawn_value=decoded_step.spawn_value,
            rates=dict(zip(DIRECTION_NAMES, step.rates)),
            rates_already_scaled=True,
        )
        replay_blob, replay_recorded = append_replay_step(
            result["replay_blob"], replay_step
        )
        replay_move_count = int(result["replay_move_count"] or 0) + int(
            replay_recorded
        )
        now = iso()
        db.execute(
            """
            UPDATE battle_player_results
            SET status = ?, route_index = ?, last_sequence = ?, goodness_of_fit = ?,
                choice_blob = ?, finished_at = ?, timeout_at = ?, updated_at = ?,
                board_state = ?, progress = ?, primary_score = ?,
                replay_blob = ?, replay_move_count = ?
            WHERE result_id = ?
            """,
            (
                status,
                stored_index,
                expected_sequence,
                score.goodness_of_fit,
                choices,
                now if complete else None,
                timeout_at,
                now,
                f"{next_board:016x}",
                stored_index,
                score.goodness_of_fit,
                replay_blob,
                replay_move_count,
                result["result_id"],
            ),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now, room["room_id"]),
        )
        _complete_round_if_done(db, room_id=str(room["room_id"]), round_id=str(round_id), now_text=now)
    return {
        "round_id": str(round_id),
        "sequence": expected_sequence,
        "route_index": stored_index,
        "goodness_of_fit": score.goodness_of_fit,
        "step_ratio": score.step_ratio,
        "goodness_drop": score.goodness_drop,
        "selected_direction": direction,
        "standard_direction": standard_direction,
        "wrong": wrong,
        "correction_seconds": CORRECTION_WINDOW_SECONDS if wrong else 0,
        "complete": complete,
    }


def _complete_correction_goodness(
    room_code: str,
    *,
    actor_key: str,
    round_id: str,
    sequence: int,
    route_index: int,
) -> dict[str, Any]:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_code)
        if room["status"] != "running":
            raise BattleServiceError("ROUND_NOT_RUNNING", "Round is not running.", 409)
        result = db.execute(
            "SELECT * FROM battle_player_results WHERE round_id = ? AND actor_key = ?",
            (str(round_id), actor_key),
        ).fetchone()
        route_row = db.execute(
            "SELECT * FROM battle_routes WHERE round_id = ?", (str(round_id),)
        ).fetchone()
        if result is None or route_row is None:
            raise BattleServiceError("ROUND_NOT_FOUND", "Round state is unavailable.", 404)
        if result["status"] != "playing":
            raise BattleServiceError("PLAYER_FINISHED", "This player has already finished.", 409)
        current_index = int(result["route_index"])
        current_sequence = int(result["last_sequence"])
        if int(sequence) != current_sequence or int(route_index) != current_index:
            raise BattleServiceError("PROGRESS_CONFLICT", "Battle progress is out of date.", 409)
        choices = bytes(result["choice_blob"] or b"")
        route = decode_route(bytes(route_row["route_blob"]))
        if current_index <= 0 or current_index > len(route.steps) or not choices:
            raise BattleServiceError("NO_CORRECTION_PENDING", "No correction is pending.", 409)
        standard_direction = decode_changes(route.steps[current_index - 1].changes).direction
        if int(choices[-1]) == int(DIRECTION_CODES[standard_direction]):
            raise BattleServiceError("NO_CORRECTION_PENDING", "No correction is pending.", 409)

        current_deadline = parse_iso(result["timeout_at"])
        if current_deadline is None:
            raise BattleServiceError("NO_CORRECTION_PENDING", "No correction is pending.", 409)
        resumed_deadline = utcnow() + timedelta(seconds=int(room["step_timeout_seconds"]))
        timeout_at = iso(min(current_deadline, resumed_deadline))
        now = iso()
        db.execute(
            "UPDATE battle_player_results SET timeout_at = ?, updated_at = ? WHERE result_id = ?",
            (timeout_at, now, result["result_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now, room["room_id"]),
        )
    return {
        "kind": "correction_complete",
        "round_id": str(round_id),
        "sequence": current_sequence,
        "route_index": current_index,
        "goodness_of_fit": float(result["goodness_of_fit"]),
        "timeout_at": timeout_at,
        "complete": False,
    }


def handle_action_for_mode(
    room_code: str,
    *,
    actor_key: str,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized_action = str(action or "").lower()
    if normalized_action == "move":
        return _record_choice_goodness(
            room_code,
            actor_key=actor_key,
            round_id=str(payload.get("round_id") or ""),
            sequence=int(payload.get("sequence")),
            route_index=int(payload.get("route_index")),
            direction=str(payload.get("direction") or ""),
        )
    if normalized_action == "correction_complete":
        return _complete_correction_goodness(
            room_code,
            actor_key=actor_key,
            round_id=str(payload.get("round_id") or ""),
            sequence=int(payload.get("sequence")),
            route_index=int(payload.get("route_index")),
        )
    raise BattleServiceError("MODE_ACTION_UNSUPPORTED", "Unsupported Battle action.", 400)


def _complete_round_if_done(
    db: sqlite3.Connection,
    *,
    room_id: str,
    round_id: str,
    now_text: str,
) -> bool:
    remaining = db.execute(
        "SELECT COUNT(*) AS count FROM battle_player_results WHERE round_id = ? AND status IN ('playing', 'disconnected')",
        (round_id,),
    ).fetchone()
    if int(remaining["count"] or 0) > 0:
        return False
    db.execute(
        "UPDATE battle_rounds SET status = 'completed', ended_at = ?, updated_at = ? WHERE round_id = ?",
        (now_text, now_text, round_id),
    )
    db.execute(
        "UPDATE battle_members SET ready = 0, updated_at = ? WHERE room_id = ? AND status = 'active'",
        (now_text, room_id),
    )
    db.execute(
        """
        UPDATE battle_rooms SET status = 'waiting', expires_at = ?,
            revision = revision + 1, updated_at = ? WHERE room_id = ?
        """,
        (iso(utcnow() + WAITING_LIFETIME), now_text, room_id),
    )
    return True


def forfeit_round_for_mode(
    room_code: str,
    *,
    actor_key: str,
    round_id: str,
) -> dict[str, Any]:
    normalized_round_id = str(round_id or "")
    if not normalized_round_id:
        raise BattleServiceError("ROUND_NOT_FOUND", "Round state is unavailable.", 404)

    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_code)
        if room["status"] != "running":
            raise BattleServiceError("ROUND_NOT_RUNNING", "Round is not running.", 409)
        member = db.execute(
            "SELECT * FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (room["room_id"], actor_key),
        ).fetchone()
        if member is None or member["role"] != "player":
            raise BattleServiceError("PLAYER_NOT_ACTIVE", "You are not an active player.", 409)
        result = db.execute(
            "SELECT * FROM battle_player_results WHERE round_id = ? AND actor_key = ?",
            (normalized_round_id, actor_key),
        ).fetchone()
        if result is None:
            raise BattleServiceError("ROUND_NOT_FOUND", "Round state is unavailable.", 404)

        try:
            mode_data = json.loads(result["mode_data_json"] or "{}")
        except (TypeError, ValueError):
            mode_data = {}
        if result["status"] == "disqualified" and mode_data.get("finish_reason") == "forfeit":
            return room_snapshot(room_code, actor_key=actor_key)
        if result["status"] not in {"playing", "disconnected"}:
            raise BattleServiceError("PLAYER_FINISHED", "This player has already finished.", 409)

        now = iso()
        mode_data["finish_reason"] = "forfeit"
        db.execute(
            """
            UPDATE battle_player_results
            SET status = 'disqualified', mode_data_json = ?, finished_at = ?,
                timeout_at = NULL, updated_at = ?
            WHERE result_id = ?
            """,
            (json.dumps(mode_data, separators=(",", ":")), now, now, result["result_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now, room["room_id"]),
        )
        _complete_round_if_done(
            db,
            room_id=str(room["room_id"]),
            round_id=normalized_round_id,
            now_text=now,
        )
    return room_snapshot(room_code, actor_key=actor_key)


def mark_timeouts() -> set[str]:
    changed_rooms: set[str] = set()
    unstarted_reservations: list[tuple[str, TokenReservation | None]] = []
    now = utcnow()
    now_text = iso(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        expired = db.execute(
            """
            SELECT result.result_id, result.round_id, round.room_id
            FROM battle_player_results AS result
            JOIN battle_rounds AS round ON round.round_id = result.round_id
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE result.status = 'playing' AND result.timeout_at IS NOT NULL
              AND result.timeout_at <= ?
              AND room.mode_key = 'goodness'
            """,
            (now_text,),
        ).fetchall()
        for row in expired:
            db.execute(
                "UPDATE battle_player_results SET status = 'timed_out', finished_at = ?, timeout_at = NULL, updated_at = ? WHERE result_id = ?",
                (now_text, now_text, row["result_id"]),
            )
            changed_rooms.add(str(row["room_id"]))
            db.execute(
                "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
                (now_text, row["room_id"]),
            )
        for room_id in tuple(changed_rooms):
            round_id = next(str(row["round_id"]) for row in expired if str(row["room_id"]) == room_id)
            _complete_round_if_done(db, room_id=room_id, round_id=round_id, now_text=now_text)

        expired_rounds = db.execute(
            """
            SELECT round.round_id, round.room_id
            FROM battle_rounds AS round
            JOIN battle_rooms AS room ON room.room_id = round.room_id
            WHERE round.status = 'running' AND round.expires_at IS NOT NULL
              AND round.expires_at <= ?
              AND room.mode_key = 'goodness'
            """,
            (now_text,),
        ).fetchall()
        for row in expired_rounds:
            db.execute(
                """
                UPDATE battle_player_results
                SET status = 'timed_out', finished_at = ?, timeout_at = NULL, updated_at = ?
                WHERE round_id = ? AND status IN ('playing', 'disconnected')
                """,
                (now_text, now_text, row["round_id"]),
            )
            _complete_round_if_done(
                db,
                room_id=str(row["room_id"]),
                round_id=str(row["round_id"]),
                now_text=now_text,
            )
            changed_rooms.add(str(row["room_id"]))

        stale_rooms = db.execute(
            "SELECT * FROM battle_rooms WHERE mode_key = 'goodness' AND status IN ('preparing', 'waiting') AND expires_at <= ?",
            (now_text,),
        ).fetchall()
        for row in stale_rooms:
            room_id = str(row["room_id"])
            round_row = db.execute(
                "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
                (room_id,),
            ).fetchone()
            if (
                round_row is not None
                and round_row["status"] in {"preparing", "ready"}
                and str(round_row["reservation_status"] or "reserved") == "reserved"
            ):
                db.execute(
                    "UPDATE battle_rounds SET status = 'cancelled', error_code = 'ROOM_EXPIRED', updated_at = ? WHERE round_id = ?",
                    (now_text, round_row["round_id"]),
                )
                unstarted_reservations.append(
                    (str(round_row["round_id"]), _restore_reservation(round_row, row))
                )
            db.execute(
                "UPDATE battle_rooms SET status = 'expired', closed_at = ?, updated_at = ?, revision = revision + 1 WHERE room_id = ?",
                (now_text, now_text, room_id),
            )
            db.execute(
                "UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ? WHERE room_id = ? AND status = 'active'",
                (now_text, now_text, room_id),
            )
            changed_rooms.add(room_id)
    for round_id, reservation in unstarted_reservations:
        _settle_round_reservation(
            round_id,
            reservation,
            settlement="unstarted_refunded",
            reason="battle_room_expired",
        )
    return changed_rooms


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
        for room_id in await asyncio.to_thread(mark_timeouts):
            await _broadcast(room_id)


async def startup() -> None:
    global _cleanup_task, _recovery_task
    repository.init_battle_db()
    async def recover() -> None:
        await asyncio.sleep(15.0)
        with auth_db() as db:
            pending = db.execute(
                """
                SELECT round.*, room.*,
                       round.status AS round_status,
                       room.status AS room_status
                FROM battle_rounds AS round
                JOIN battle_rooms AS room ON room.room_id = round.room_id
                WHERE round.reservation_status = 'reserved'
                  AND room.mode_key = 'goodness'
                ORDER BY round.created_at
                """
            ).fetchall()
        for row in pending:
            status = str(row["round_status"])
            room_status = str(row["room_status"])
            round_id = str(row["round_id"])
            if status == "preparing":
                if room_status in {"closed", "expired"}:
                    _settle_round_reservation(
                        round_id,
                        _restore_reservation(row, row),
                        settlement="unstarted_refunded",
                        reason="battle_reservation_recovery",
                    )
                    continue
                _schedule_route(
                    round_id,
                    reservation=None,
                    auto_start=bool(row["auto_start_after_generation"]),
                )
                continue
            reservation = _restore_reservation(row, row)
            if status == "ready":
                if room_status in {"closed", "expired"}:
                    _settle_round_reservation(
                        round_id,
                        reservation,
                        settlement="unstarted_refunded",
                        reason="battle_reservation_recovery",
                    )
                elif bool(row["auto_start_after_generation"]):
                    try:
                        _start_ready_round(str(row["room_id"]), host_user_id=int(row["host_user_id"]))
                    except BattleServiceError:
                        pass
            elif status in {"running", "completed"}:
                _settle_round_reservation(
                    round_id,
                    reservation,
                    settlement="finalized",
                    metadata={"room_id": row["room_id"], "round_id": round_id, "recovered": True},
                )
            elif status == "failed":
                _settle_round_reservation(
                    round_id,
                    reservation,
                    settlement="cancelled",
                    reason="battle_reservation_recovery",
                )
            elif status == "cancelled":
                _settle_round_reservation(
                    round_id,
                    reservation,
                    settlement="unstarted_refunded",
                    reason="battle_reservation_recovery",
                )

    _recovery_task = asyncio.create_task(recover())
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


async def shutdown() -> None:
    global _cleanup_task, _recovery_task
    if _recovery_task is not None:
        _recovery_task.cancel()
        await asyncio.gather(_recovery_task, return_exceptions=True)
        _recovery_task = None
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        await asyncio.gather(_cleanup_task, return_exceptions=True)
        _cleanup_task = None
    tasks = list(_route_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _route_tasks.clear()


_goodness_mode.bind_runtime(sys.modules[__name__])
