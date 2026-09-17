from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from backend.auth.db import auth_db

from .. import repository
from ..actors import BattleActor, coerce_actor, user_actor
from .contracts import BattleModeError
from .errors import BattleServiceError
from .hosting import is_permanent_room, is_room_host
from .registry import get_battle_mode


_broadcast_callback: Callable[[str], Awaitable[None]] | None = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utcnow()).astimezone(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def set_broadcast_callback(callback: Callable[[str], Awaitable[None]] | None) -> None:
    global _broadcast_callback
    _broadcast_callback = callback


async def broadcast_room(room_id: str) -> None:
    if _broadcast_callback is not None:
        await _broadcast_callback(str(room_id))


def room_by_actor(
    actor: Any | None = None, *, user_id: int | None = None
) -> dict[str, Any] | None:
    identity = coerce_actor(actor, user_id=user_id)
    with auth_db() as db:
        row = db.execute(
            """
            SELECT r.room_id FROM battle_members AS m
            JOIN battle_rooms AS r ON r.room_id = m.room_id
            WHERE m.actor_key = ? AND m.status = 'active'
              AND r.status IN ('preparing', 'waiting', 'running')
            LIMIT 1
            """,
            (identity.actor_key,),
        ).fetchone()
    return repository.get_room(str(row["room_id"])) if row is not None else None


def room_by_user(user_id: int) -> dict[str, Any] | None:
    return room_by_actor(user_id=user_id)


def assert_member(
    room: dict[str, Any],
    actor: Any | None = None,
    *,
    actor_key: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    identity = coerce_actor(actor or actor_key, user_id=user_id)
    member = next(
        (item for item in room.get("members", []) if item["actor_key"] == identity.actor_key),
        None,
    )
    if member is None:
        raise BattleServiceError("ROOM_MEMBERSHIP_REQUIRED", "You are not in this room.", 403)
    return member


def sanitize_snapshot(
    room: dict[str, Any],
    *,
    actor: Any | None = None,
    viewer_user_id: int | None = None,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=viewer_user_id)
    payload = dict(room)
    member = assert_member(payload, identity)
    mode = None
    try:
        mode = get_battle_mode(str(payload.get("mode_key") or "goodness"))
        payload["mode_settings"] = mode.public_settings(payload)
    except BattleModeError:
        payload["mode_settings"] = dict(payload.get("settings") or {})
    payload["viewer"] = {
        "actor_key": identity.actor_key,
        "actor_kind": identity.kind,
        "is_guest": identity.is_guest,
        "user_id": identity.user_id,
        "guest_id": identity.guest_id,
        "role": member["role"],
        "is_host": is_room_host(payload, identity.actor_key),
    }
    round_payload = dict(payload.get("round") or {})
    for field in (
        "token_reservation_id",
        "reservation_ledger_id",
        "reserved_bonus_units",
        "reserved_paid_units",
        "route_seed",
        "artifact_blob",
    ):
        round_payload.pop(field, None)
    payload["round"] = round_payload or None
    for item in payload.get("members", []):
        item.pop("member_id", None)
    return (
        mode.sanitize_snapshot(
            payload,
            viewer_actor_key=identity.actor_key,
            viewer_user_id=identity.user_id,
        )
        if mode is not None
        else payload
    )


def room_snapshot(
    room_ref: str,
    *,
    actor: Any | None = None,
    actor_key: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    try:
        room = repository.get_room(room_ref)
    except repository.BattleNotFoundError as exc:
        raise BattleServiceError("ROOM_NOT_FOUND", "Room not found.", 404) from exc
    return sanitize_snapshot(room, actor=actor or actor_key, viewer_user_id=user_id)


def current_room(
    *, actor: Any | None = None, user_id: int | None = None
) -> dict[str, Any] | None:
    identity = coerce_actor(actor, user_id=user_id)
    room = room_by_actor(identity)
    return None if room is None else sanitize_snapshot(room, actor=identity)


def list_rooms(*, user_id: int | None = None) -> list[dict[str, Any]]:
    del user_id
    return repository.list_public_rooms(limit=50)


def join_room(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    role: str | None,
    ip_address: str = "",
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    try:
        from ..permanent.service import prepare_vacant_room_for_join

        prepare_vacant_room_for_join(room_code)
        repository.join_room(
            room_code,
            actor=identity,
            preferred_role=role,
            ip_address=ip_address,
        )
        from ..permanent.service import claim_host_if_vacant, note_activity

        claim_host_if_vacant(room_code, identity)
        note_activity(room_code, identity)
        return room_snapshot(room_code, actor=identity)
    except repository.BattleRateLimitError as exc:
        raise BattleServiceError(
            exc.code,
            "Too many Battle rooms have been joined. Try again later.",
            429,
            extra={"retry_after_seconds": exc.retry_after_seconds},
        ) from exc
    except repository.BattleRepositoryError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc


def leave_room(
    room_code: str, *, actor: Any | None = None, user_id: int | None = None
) -> None:
    identity = coerce_actor(actor, user_id=user_id)
    room = repository.get_room(room_code)
    assert_member(room, identity)
    if is_permanent_room(room):
        from ..permanent.service import leave_permanent_room

        leave_permanent_room(room_code, identity)
        return
    if is_room_host(room, identity.actor_key):
        try:
            mode = get_battle_mode(str(room.get("mode_key") or "goodness"))
        except BattleModeError as exc:
            raise BattleServiceError(
                "BATTLE_MODE_UNAVAILABLE", "This Battle mode is unavailable.", 409
            ) from exc
        mode.settle_unstarted_round(
            str(room["room_id"]), reason="battle_host_closed_room"
        )
        repository.close_room(room_code, host_user_id=int(identity.user_id))
        return
    now = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ? WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (now, now, room["room_id"], identity.actor_key),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now, room["room_id"]),
        )


def set_ready(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    ready: bool,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    try:
        repository.set_member_ready(room_code, actor=identity, ready=ready)
        from ..permanent.service import note_activity

        note_activity(room_code, identity)
    except repository.BattleRepositoryError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc
    return room_snapshot(room_code, actor=identity)


def set_role(
    room_code: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    role: str,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    if role not in {"player", "spectator"}:
        raise BattleServiceError("INVALID_ROLE", "Role must be player or spectator.")
    now = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, room_code)
        if room["status"] not in {"preparing", "waiting"}:
            raise BattleServiceError("ROOM_ALREADY_STARTED", "Role cannot change after start.", 409)
        member = db.execute(
            "SELECT * FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if member is None:
            raise BattleServiceError("MEMBER_NOT_FOUND", "Room member not found.", 404)
        if role == "spectator" and str(room["host_actor_key"] or f"u:{room['host_user_id']}") == identity.actor_key:
            raise BattleServiceError(
                "HOST_CANNOT_SPECTATE",
                "The room host must remain in a player seat.",
                409,
            )
        if role == "spectator":
            if not bool(room["allow_spectators"]):
                raise BattleServiceError("SPECTATORS_DISABLED", "Spectating is disabled.", 409)
            seat = None
        else:
            seat = repository._next_seat(db, str(room["room_id"]), int(room["max_players"]))
            if member["role"] == "player":
                seat = int(member["seat_index"])
            if seat is None:
                raise BattleServiceError("PLAYER_SLOTS_FULL", "Player slots are full.", 409)
        if member["role"] != role:
            db.execute(
                "UPDATE battle_members SET role = ?, seat_index = ?, ready = 0, updated_at = ? WHERE member_id = ?",
                (role, seat, now, member["member_id"]),
            )
            db.execute(
                "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
                (now, room["room_id"]),
            )
    from ..permanent.service import claim_host_if_vacant, note_activity

    claim_host_if_vacant(room_code, identity)
    note_activity(room_code, identity)
    return room_snapshot(room_code, actor=identity)


def kick(
    room_code: str,
    *,
    actor: Any,
    target_actor_key: str | None = None,
    target_user_id: int | None = None,
) -> dict[str, Any]:
    identity = coerce_actor(actor)
    try:
        repository.kick_member(
            room_code,
            host_actor_key=identity.actor_key,
            target_actor_key=target_actor_key,
            target_user_id=target_user_id,
        )
    except repository.BattleRepositoryError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc
    from ..permanent.service import note_activity

    note_activity(room_code, identity)
    return room_snapshot(room_code, actor=identity)


async def update_room_settings(
    room_code: str,
    *,
    actor: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    identity = coerce_actor(actor)
    room = repository.get_room(room_code)
    if not is_room_host(room, identity.actor_key):
        raise BattleServiceError("HOST_REQUIRED", "Only the current host can edit settings.", 403)
    if str(room.get("status") or "") not in {"preparing", "waiting"}:
        raise BattleServiceError("ROOM_ALREADY_STARTED", "Settings cannot change during a round.", 409)
    try:
        expected_revision = int(payload.get("expected_revision"))
    except (TypeError, ValueError) as exc:
        raise BattleServiceError("REVISION_REQUIRED", "Room revision is required.", 409) from exc
    mode = get_battle_mode(str(room.get("mode_key") or "goodness"))
    if room.get("lifecycle_kind") == "permanent" and room.get("mode_key") == "goodness" and room.get("status") != "waiting":
        raise BattleServiceError("ROOM_ALREADY_STARTED", "Settings cannot change during route generation.", 409)
    try:
        normalized = await mode.normalize_lobby_settings_patch(room, payload)
    except ValueError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc

    now = iso()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        current = repository._find_room(db, room_code)
        if current["lifecycle_kind"] == "permanent" and current["mode_key"] == "goodness" and current["status"] != "waiting":
            raise BattleServiceError("ROOM_ALREADY_STARTED", "Settings cannot change during route generation.", 409)
        current_host = str(current["host_actor_key"] or f"u:{current['host_user_id']}")
        if current_host != identity.actor_key:
            raise BattleServiceError("HOST_REQUIRED", "Only the current host can edit settings.", 403)
        if str(current["status"]) not in {"preparing", "waiting"}:
            raise BattleServiceError("ROOM_ALREADY_STARTED", "Settings cannot change during a round.", 409)
        if int(current["settings_revision"] or 0) != expected_revision:
            raise BattleServiceError(
                "ROOM_REVISION_CONFLICT",
                "Room settings changed. Try again.",
                409,
            )
        try:
            settings = json.loads(current["settings_json"] or "{}")
        except (TypeError, ValueError):
            settings = {}
        settings.update(normalized)
        initial_board = normalized.get("initial_board", current["initial_board"])
        max_steps = (
            int(normalized["score_step_limit"])
            if "score_step_limit" in normalized
            else current["max_steps"]
        )
        db.execute(
            """
            UPDATE battle_rooms
            SET settings_json = ?, settings_revision = settings_revision + 1,
                initial_board = ?, max_steps = ?, step_timeout_seconds = ?,
                revision = revision + 1, updated_at = ?
            WHERE room_id = ?
            """,
            (
                json.dumps(settings, separators=(",", ":"), sort_keys=True),
                initial_board,
                max_steps,
                int(normalized["step_timeout_seconds"]),
                now,
                current["room_id"],
            ),
        )
        if str(current["mode_key"]) == "free_goodness":
            latest = db.execute(
                "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
                (current["room_id"],),
            ).fetchone()
            if latest is not None and str(latest["status"]) == "ready":
                try:
                    mode_state = json.loads(latest["mode_state_json"] or "{}")
                except (TypeError, ValueError):
                    mode_state = {}
                mode_state.update(
                    {
                        "initial_board": str(initial_board),
                        "score_step_limit": int(normalized["score_step_limit"]),
                        "ranking_min_steps": int(normalized["ranking_min_steps"]),
                        "lookup_hit_steps": 0,
                        "lookup_miss_steps": 0,
                    }
                )
                db.execute(
                    """
                    UPDATE battle_rounds
                    SET mode_state_json = ?, route_seed = ?, updated_at = ?
                    WHERE round_id = ?
                    """,
                    (
                        json.dumps(mode_state, separators=(",", ":"), sort_keys=True),
                        secrets.token_hex(32),
                        now,
                        latest["round_id"],
                    ),
                )
    from ..permanent.service import note_activity

    note_activity(room_code, identity)
    return room_snapshot(room_code, actor=identity)
