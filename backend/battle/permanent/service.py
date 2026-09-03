from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.db import auth_db

from .. import repository
from ..actors import BattleActor, coerce_actor
from ..core.errors import BattleServiceError
from ..core.hosting import is_permanent_room, is_room_host
from ..core.registry import get_battle_mode
from .definitions import HostPolicy, PermanentRoomDefinition, load_definitions


logger = logging.getLogger(__name__)

_policy = HostPolicy()
_sweep_task: asyncio.Task | None = None
_reconcile_task: asyncio.Task | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utcnow()).astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def init_permanent_db() -> None:
    with auth_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS battle_permanent_room_state (
              room_id TEXT PRIMARY KEY,
              template_key TEXT NOT NULL UNIQUE,
              definition_version INTEGER NOT NULL DEFAULT 1,
              default_settings_json TEXT NOT NULL,
              host_generation INTEGER NOT NULL DEFAULT 0,
              host_assigned_at TEXT,
              host_last_action_at TEXT,
              host_last_seen_at TEXT,
              host_idle_expires_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS battle_permanent_presence (
              room_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              PRIMARY KEY(room_id, actor_key),
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_battle_permanent_presence_seen
              ON battle_permanent_presence(last_seen_at);
            """
        )


def register_definition(
    room: dict[str, Any], definition: PermanentRoomDefinition
) -> None:
    settings = dict(room.get("settings") or {})
    settings.update(
        {
            "full_pattern": str(definition.full_pattern),
            "initial_board": str(definition.initial_board),
            "step_timeout_seconds": int(definition.step_timeout_seconds),
        }
    )
    if str(definition.mode_key) == "free_goodness":
        default_steps = max(1, int(room.get("target") or 0) // 2)
        settings.update(
            {
                "score_step_limit": default_steps,
                "ranking_min_steps": default_steps,
            }
        )
    defaults = {
        "settings": settings,
        "initial_board": str(definition.initial_board),
        "max_steps": (
            max(1, int(room.get("target") or 0) // 2)
            if str(definition.mode_key) == "free_goodness"
            else None
        ),
        "step_timeout_seconds": int(definition.step_timeout_seconds),
    }
    now = _iso()
    with auth_db() as db:
        db.execute(
            """
            INSERT INTO battle_permanent_room_state
              (room_id, template_key, definition_version, default_settings_json,
               created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_id) DO UPDATE SET
              template_key = excluded.template_key,
              definition_version = excluded.definition_version,
              default_settings_json = excluded.default_settings_json,
              updated_at = excluded.updated_at
            """,
            (
                str(room["room_id"]),
                str(definition.template_key),
                int(definition.version),
                json.dumps(defaults, separators=(",", ":"), sort_keys=True),
                now,
                now,
            ),
        )


def _active_players(db, room_id: str) -> list[Any]:
    return db.execute(
        """
        SELECT * FROM battle_members
        WHERE room_id = ? AND status = 'active' AND role = 'player'
        ORDER BY COALESCE(seat_index, 999), joined_at, member_id
        """,
        (str(room_id),),
    ).fetchall()


def _set_host_in_db(
    db,
    room: Any,
    actor_key: str | None,
    *,
    now: datetime | None = None,
) -> str | None:
    current = now or _utcnow()
    now_text = _iso(current)
    member = None
    normalized = str(actor_key or "") or None
    if normalized:
        member = db.execute(
            """
            SELECT * FROM battle_members
            WHERE room_id = ? AND actor_key = ? AND status = 'active' AND role = 'player'
            """,
            (room["room_id"], normalized),
        ).fetchone()
        if member is None:
            normalized = None
    db.execute(
        """
        UPDATE battle_rooms
        SET host_actor_key = ?, host_user_id = ?, revision = revision + 1,
            updated_at = ?
        WHERE room_id = ?
        """,
        (
            normalized,
            None if member is None else member["user_id"],
            now_text,
            room["room_id"],
        ),
    )
    db.execute(
        """
        UPDATE battle_permanent_room_state
        SET host_generation = host_generation + 1,
            host_assigned_at = ?, host_last_action_at = ?, host_last_seen_at = ?,
            host_idle_expires_at = ?, updated_at = ?
        WHERE room_id = ?
        """,
        (
            now_text if normalized else None,
            now_text if normalized else None,
            now_text if normalized else None,
            _iso(current + timedelta(seconds=_policy.idle_timeout_seconds))
            if normalized
            else None,
            now_text,
            room["room_id"],
        ),
    )
    return normalized


def _next_host_key(
    db,
    room: Any,
    *,
    after_actor_key: str | None = None,
) -> str | None:
    players = _active_players(db, str(room["room_id"]))
    if not players:
        return None
    after = str(after_actor_key or "")
    if not after:
        return str(players[0]["actor_key"])
    for index, player in enumerate(players):
        if str(player["actor_key"]) == after:
            return str(players[(index + 1) % len(players)]["actor_key"])
    return str(players[0]["actor_key"])


def _reset_empty_room_in_db(db, room: Any, *, now_text: str) -> None:
    players = _active_players(db, str(room["room_id"]))
    if players:
        return
    state = db.execute(
        "SELECT default_settings_json FROM battle_permanent_room_state WHERE room_id = ?",
        (room["room_id"],),
    ).fetchone()
    if state is None:
        return
    try:
        defaults = json.loads(state["default_settings_json"] or "{}")
    except (TypeError, ValueError):
        defaults = {}
    db.execute(
        """
        UPDATE battle_rooms
        SET settings_json = ?, settings_revision = settings_revision + 1,
            initial_board = ?, max_steps = ?, step_timeout_seconds = ?,
            visibility = 'public', allow_spectators = 1, allow_guest_chat = 1,
            chat_roles_json = '["host","player","spectator"]',
            revision = revision + 1, updated_at = ?
        WHERE room_id = ?
        """,
        (
            json.dumps(defaults.get("settings") or {}, separators=(",", ":"), sort_keys=True),
            defaults.get("initial_board"),
            defaults.get("max_steps"),
            int(defaults.get("step_timeout_seconds") or 90),
            now_text,
            room["room_id"],
        ),
    )
    if str(room["mode_key"] or "") == "free_goodness":
        latest = db.execute(
            "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
            (room["room_id"],),
        ).fetchone()
        if latest is not None and str(latest["status"] or "") == "ready":
            try:
                mode_state = json.loads(latest["mode_state_json"] or "{}")
            except (TypeError, ValueError):
                mode_state = {}
            settings = dict(defaults.get("settings") or {})
            mode_state.update(
                {
                    "initial_board": str(defaults.get("initial_board") or ""),
                    "score_step_limit": int(
                        settings.get("score_step_limit")
                        or defaults.get("max_steps")
                        or 1
                    ),
                    "ranking_min_steps": int(
                        settings.get("ranking_min_steps")
                        or settings.get("score_step_limit")
                        or defaults.get("max_steps")
                        or 1
                    ),
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
                    now_text,
                    latest["round_id"],
                ),
            )


def claim_host_if_vacant(room_ref: str, actor: Any) -> bool:
    identity = coerce_actor(actor)
    now = _utcnow()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, str(room_ref))
        if str(room["lifecycle_kind"] or "normal") != "permanent":
            return False
        if room["host_actor_key"]:
            return False
        member = db.execute(
            """
            SELECT 1 FROM battle_members
            WHERE room_id = ? AND actor_key = ? AND status = 'active' AND role = 'player'
            """,
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if member is None:
            return False
        _set_host_in_db(db, room, identity.actor_key, now=now)
        return True


def note_presence(room_ref: str, actor: Any) -> None:
    identity = coerce_actor(actor)
    now = _utcnow()
    now_text = _iso(now)
    with auth_db() as db:
        try:
            room = repository._find_room(db, str(room_ref))
        except repository.BattleNotFoundError:
            return
        if str(room["lifecycle_kind"] or "normal") != "permanent":
            return
        member = db.execute(
            "SELECT 1 FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if member is None:
            return
        db.execute(
            """
            INSERT INTO battle_permanent_presence(room_id, actor_key, last_seen_at)
            VALUES (?, ?, ?)
            ON CONFLICT(room_id, actor_key) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (room["room_id"], identity.actor_key, now_text),
        )
        if str(room["host_actor_key"] or "") == identity.actor_key:
            db.execute(
                """
                UPDATE battle_permanent_room_state
                SET host_last_seen_at = ?, updated_at = ? WHERE room_id = ?
                """,
                (now_text, now_text, room["room_id"]),
            )


def note_activity(room_ref: str, actor: Any) -> None:
    identity = coerce_actor(actor)
    now = _utcnow()
    now_text = _iso(now)
    with auth_db() as db:
        try:
            room = repository._find_room(db, str(room_ref))
        except repository.BattleNotFoundError:
            return
        if str(room["lifecycle_kind"] or "normal") != "permanent":
            return
        note_member = db.execute(
            "SELECT 1 FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if note_member is None:
            return
        db.execute(
            """
            INSERT INTO battle_permanent_presence(room_id, actor_key, last_seen_at)
            VALUES (?, ?, ?)
            ON CONFLICT(room_id, actor_key) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (room["room_id"], identity.actor_key, now_text),
        )
        if str(room["host_actor_key"] or "") == identity.actor_key:
            db.execute(
                """
                UPDATE battle_permanent_room_state
                SET host_last_action_at = ?, host_last_seen_at = ?,
                    host_idle_expires_at = ?, updated_at = ?
                WHERE room_id = ?
                """,
                (
                    now_text,
                    now_text,
                    _iso(now + timedelta(seconds=_policy.idle_timeout_seconds)),
                    now_text,
                    room["room_id"],
                ),
            )


def renew_host(room_ref: str, actor: Any) -> dict[str, Any]:
    identity = coerce_actor(actor)
    room = repository.get_room(str(room_ref))
    if not is_permanent_room(room) or not is_room_host(room, identity.actor_key):
        raise BattleServiceError("HOST_REQUIRED", "Only the current host can continue hosting.", 403)
    note_activity(str(room["room_id"]), identity)
    return repository.get_room(str(room["room_id"]))


def leave_permanent_room(room_ref: str, actor: Any) -> bool:
    identity = coerce_actor(actor)
    now = _utcnow()
    now_text = _iso(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = repository._find_room(db, str(room_ref))
        if str(room["lifecycle_kind"] or "normal") != "permanent":
            return False
        member = db.execute(
            "SELECT * FROM battle_members WHERE room_id = ? AND actor_key = ? AND status = 'active'",
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if member is None:
            raise BattleServiceError("ROOM_MEMBERSHIP_REQUIRED", "You are not in this room.", 403)
        was_host = str(room["host_actor_key"] or "") == identity.actor_key
        db.execute(
            """
            UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ?
            WHERE member_id = ?
            """,
            (now_text, now_text, member["member_id"]),
        )
        db.execute(
            "DELETE FROM battle_permanent_presence WHERE room_id = ? AND actor_key = ?",
            (room["room_id"], identity.actor_key),
        )
        if was_host:
            _set_host_in_db(
                db,
                room,
                _next_host_key(db, room, after_actor_key=identity.actor_key),
                now=now,
            )
        else:
            db.execute(
                "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
                (now_text, room["room_id"]),
            )
        _reset_empty_room_in_db(db, room, now_text=now_text)
        return True


def rotate_after_round_in_db(db, room_id: str, *, now_text: str) -> bool:
    room = repository._find_room(db, str(room_id))
    if str(room["lifecycle_kind"] or "normal") != "permanent":
        return False
    old = str(room["host_actor_key"] or "") or None
    _set_host_in_db(
        db,
        room,
        _next_host_key(db, room, after_actor_key=old),
        now=_parse_iso(now_text) or _utcnow(),
    )
    _reset_empty_room_in_db(db, room, now_text=now_text)
    return True


def _sweep_once() -> set[str]:
    now = _utcnow()
    now_text = _iso(now)
    offline_cutoff = now - timedelta(seconds=_policy.offline_grace_seconds)
    member_cutoff = now - timedelta(seconds=_policy.member_offline_grace_seconds)
    changed: set[str] = set()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        rooms = db.execute(
            """
            SELECT room.*, state.host_last_seen_at, state.host_idle_expires_at
            FROM battle_rooms AS room
            JOIN battle_permanent_room_state AS state ON state.room_id = room.room_id
            WHERE room.lifecycle_kind = 'permanent'
              AND room.status IN ('preparing', 'waiting')
            """
        ).fetchall()
        for room in rooms:
            room_id = str(room["room_id"])
            host_key = str(room["host_actor_key"] or "")
            host_seen = _parse_iso(room["host_last_seen_at"])
            idle_at = _parse_iso(room["host_idle_expires_at"])
            host_expired = bool(
                host_key
                and (
                    (host_seen is not None and host_seen <= offline_cutoff)
                    or (idle_at is not None and idle_at <= now)
                )
            )
            if host_expired:
                db.execute(
                    """
                    UPDATE battle_members
                    SET status = 'kicked', ready = 0, left_at = ?, updated_at = ?
                    WHERE room_id = ? AND actor_key = ? AND status = 'active'
                    """,
                    (now_text, now_text, room_id, host_key),
                )
                db.execute(
                    "DELETE FROM battle_permanent_presence WHERE room_id = ? AND actor_key = ?",
                    (room_id, host_key),
                )
                _set_host_in_db(
                    db,
                    room,
                    _next_host_key(db, room, after_actor_key=host_key),
                    now=now,
                )
                changed.add(room_id)

            stale = db.execute(
                """
                SELECT member.member_id, member.actor_key
                FROM battle_members AS member
                LEFT JOIN battle_permanent_presence AS presence
                  ON presence.room_id = member.room_id AND presence.actor_key = member.actor_key
                WHERE member.room_id = ? AND member.status = 'active'
                  AND member.actor_key != COALESCE((SELECT host_actor_key FROM battle_rooms WHERE room_id = ?), '')
                  AND (presence.last_seen_at IS NULL OR presence.last_seen_at <= ?)
                """,
                (room_id, room_id, _iso(member_cutoff)),
            ).fetchall()
            for member in stale:
                db.execute(
                    """
                    UPDATE battle_members SET status = 'left', ready = 0, left_at = ?, updated_at = ?
                    WHERE member_id = ?
                    """,
                    (now_text, now_text, member["member_id"]),
                )
                changed.add(room_id)
            if room_id in changed:
                refreshed = repository._find_room(db, room_id)
                _reset_empty_room_in_db(db, refreshed, now_text=now_text)
    return changed


async def _sweep_loop() -> None:
    from ..core.lifecycle import broadcast_room

    while True:
        await asyncio.sleep(_policy.sweep_interval_seconds)
        for room_id in await asyncio.to_thread(_sweep_once):
            await broadcast_room(room_id)


async def _reconcile(definitions: tuple[PermanentRoomDefinition, ...]) -> None:
    for definition in definitions:
        try:
            room = await get_battle_mode(definition.mode_key).ensure_permanent_room(
                definition
            )
            register_definition(room, definition)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Permanent Battle room unavailable template=%s error=%s",
                definition.template_key,
                type(exc).__name__,
            )


async def startup() -> None:
    global _policy, _sweep_task, _reconcile_task
    init_permanent_db()
    _policy, definitions = load_definitions()
    if _reconcile_task is None or _reconcile_task.done():
        _reconcile_task = asyncio.create_task(_reconcile(definitions))
    if _sweep_task is None or _sweep_task.done():
        _sweep_task = asyncio.create_task(_sweep_loop())


async def shutdown() -> None:
    global _sweep_task, _reconcile_task
    if _reconcile_task is not None:
        _reconcile_task.cancel()
        await asyncio.gather(_reconcile_task, return_exceptions=True)
        _reconcile_task = None
    if _sweep_task is not None:
        _sweep_task.cancel()
        await asyncio.gather(_sweep_task, return_exceptions=True)
        _sweep_task = None
