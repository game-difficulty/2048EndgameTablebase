from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from backend.auth.db import auth_db

from .. import repository
from ..actors import BattleActor, coerce_actor, user_actor
from .contracts import BattleModeError
from .errors import BattleServiceError
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
        "is_host": identity.is_user
        and int(payload["host_user_id"]) == int(identity.user_id),
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
        repository.join_room(
            room_code,
            actor=identity,
            preferred_role=role,
            ip_address=ip_address,
        )
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
    if identity.is_user and int(room["host_user_id"]) == int(identity.user_id):
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
        if role == "spectator" and identity.is_user and int(room["host_user_id"]) == int(identity.user_id):
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
        db.execute(
            "UPDATE battle_members SET role = ?, seat_index = ?, ready = 0, updated_at = ? WHERE member_id = ?",
            (role, seat, now, member["member_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (now, room["room_id"]),
        )
    return room_snapshot(room_code, actor=identity)


def kick(
    room_code: str,
    *,
    host_user_id: int,
    target_actor_key: str | None = None,
    target_user_id: int | None = None,
) -> dict[str, Any]:
    try:
        repository.kick_member(
            room_code,
            host_user_id=host_user_id,
            target_actor_key=target_actor_key,
            target_user_id=target_user_id,
        )
    except repository.BattleRepositoryError as exc:
        raise BattleServiceError(str(exc).upper(), str(exc).replace("_", " "), 409) from exc
    return room_snapshot(room_code, user_id=host_user_id)
