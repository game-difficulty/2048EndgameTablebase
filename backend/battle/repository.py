from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.db import auth_db
from backend.auth.guest_service import hash_ip_bucket

from .actors import BattleActor, coerce_actor, user_actor
from .core.chat_policy import decode_chat_roles, normalize_chat_roles


WAITING_ROOM_LIFETIME = timedelta(minutes=30)
ROOM_CREATION_COOLDOWN_SECONDS = 30
ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ROOM_STATUSES = {"preparing", "waiting", "running", "closed", "expired"}
ROOM_VISIBILITIES = {"public", "private"}
ACTIVE_ROOM_STATUSES = {"preparing", "waiting", "running"}
GUEST_JOIN_WINDOW = timedelta(hours=1)
DEFAULT_GUEST_JOIN_LIMIT = 10
DEFAULT_GUEST_JOIN_IP_LIMIT = 30


class BattleRepositoryError(RuntimeError):
    pass


class BattleNotFoundError(BattleRepositoryError):
    pass


class BattleConflictError(BattleRepositoryError):
    pass


class BattlePermissionError(BattleRepositoryError):
    pass


class BattleRateLimitError(BattleRepositoryError):
    def __init__(self, code: str, *, retry_after_seconds: int):
        self.code = str(code)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__(self.code.lower())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _room_code() -> str:
    return "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(6))


def _normalize_room_code(value: str) -> str:
    return str(value or "").strip().upper()


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in ("allow_spectators", "allow_guest_chat", "ready"):
        if field in payload:
            payload[field] = bool(payload[field])
    return payload


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})")}


def _env_limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _guest_join_retry_after(db: sqlite3.Connection, where: str, value: str, now: datetime) -> int:
    row = db.execute(
        f"SELECT MIN(created_at) AS oldest FROM battle_guest_join_events WHERE {where} = ? AND created_at >= ?",
        (value, _iso(now - GUEST_JOIN_WINDOW)),
    ).fetchone()
    if row is None or not row["oldest"]:
        return int(GUEST_JOIN_WINDOW.total_seconds())
    try:
        oldest = datetime.fromisoformat(str(row["oldest"]))
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        return max(1, math.ceil((oldest + GUEST_JOIN_WINDOW - now).total_seconds()))
    except ValueError:
        return int(GUEST_JOIN_WINDOW.total_seconds())


def _reserve_guest_join(
    db: sqlite3.Connection,
    *,
    identity: BattleActor,
    room_id: str,
    ip_address: str,
    now: datetime,
) -> None:
    """Count a guest's first successful join attempt for a room exactly once."""
    if not identity.is_guest:
        return
    guest_id = str(identity.guest_id)
    existing = db.execute(
        "SELECT 1 FROM battle_guest_join_events WHERE guest_id = ? AND room_id = ?",
        (guest_id, room_id),
    ).fetchone()
    if existing is not None:
        return

    cutoff = _iso(now - GUEST_JOIN_WINDOW)
    guest_count = int(
        db.execute(
            "SELECT COUNT(*) AS count FROM battle_guest_join_events WHERE guest_id = ? AND created_at >= ?",
            (guest_id, cutoff),
        ).fetchone()["count"]
    )
    if guest_count >= _env_limit("BATTLE_GUEST_JOIN_LIMIT_PER_HOUR", DEFAULT_GUEST_JOIN_LIMIT):
        raise BattleRateLimitError(
            "GUEST_JOIN_RATE_LIMITED",
            retry_after_seconds=_guest_join_retry_after(db, "guest_id", guest_id, now),
        )

    ip_hash = hash_ip_bucket(ip_address)
    ip_count = int(
        db.execute(
            "SELECT COUNT(*) AS count FROM battle_guest_join_events WHERE ip_hash = ? AND created_at >= ?",
            (ip_hash, cutoff),
        ).fetchone()["count"]
    )
    if ip_count >= _env_limit("BATTLE_GUEST_JOIN_IP_LIMIT_PER_HOUR", DEFAULT_GUEST_JOIN_IP_LIMIT):
        raise BattleRateLimitError(
            "GUEST_NETWORK_JOIN_RATE_LIMITED",
            retry_after_seconds=_guest_join_retry_after(db, "ip_hash", ip_hash, now),
        )
    db.execute(
        "DELETE FROM battle_guest_join_events WHERE created_at < ?",
        (_iso(now - timedelta(days=1)),),
    )
    db.execute(
        "INSERT INTO battle_guest_join_events (guest_id, room_id, ip_hash, created_at) VALUES (?, ?, ?, ?)",
        (guest_id, room_id, ip_hash, _iso(now)),
    )


def _migrate_actor_identity_tables(db: sqlite3.Connection) -> None:
    """Rebuild legacy user-only Battle tables without losing active rooms."""
    migration_scripts: list[str] = []
    if "actor_key" not in _table_columns(db, "battle_members"):
        migration_scripts.append(
            """
            ALTER TABLE battle_members RENAME TO battle_members_actor_legacy;
            CREATE TABLE battle_members (
              member_id INTEGER PRIMARY KEY AUTOINCREMENT,
              room_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              role TEXT NOT NULL,
              seat_index INTEGER,
              ready INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'active',
              joined_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              left_at TEXT,
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(room_id, actor_key),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL)),
              CHECK(role IN ('player', 'spectator')),
              CHECK(status IN ('active', 'left', 'kicked')),
              CHECK(ready IN (0, 1)),
              CHECK((role = 'player' AND seat_index IS NOT NULL AND seat_index >= 0)
                    OR (role = 'spectator' AND seat_index IS NULL))
            );
            INSERT INTO battle_members
              (member_id, room_id, actor_key, user_id, guest_id,
               display_name_snapshot, role, seat_index, ready, status,
               joined_at, updated_at, left_at)
            SELECT legacy.member_id, legacy.room_id, 'u:' || legacy.user_id,
                   legacy.user_id, NULL, COALESCE(users.display_name, ''),
                   legacy.role, legacy.seat_index, legacy.ready, legacy.status,
                   legacy.joined_at, legacy.updated_at, legacy.left_at
            FROM battle_members_actor_legacy AS legacy
            LEFT JOIN users ON users.id = legacy.user_id;
            DROP TABLE battle_members_actor_legacy;
            """
        )
    if "actor_key" not in _table_columns(db, "battle_player_results"):
        migration_scripts.append(
            """
            ALTER TABLE battle_player_results
              RENAME TO battle_player_results_actor_legacy;
            CREATE TABLE battle_player_results (
              result_id INTEGER PRIMARY KEY AUTOINCREMENT,
              round_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'playing',
              route_index INTEGER NOT NULL DEFAULT 0,
              last_sequence INTEGER NOT NULL DEFAULT 0,
              goodness_of_fit REAL NOT NULL DEFAULT 1.0,
              primary_score REAL, secondary_score REAL,
              progress INTEGER NOT NULL DEFAULT 0,
              mode_data_json TEXT NOT NULL DEFAULT '{}',
              choice_blob BLOB, board_state TEXT,
              replay_blob BLOB NOT NULL DEFAULT X'',
              replay_move_count INTEGER NOT NULL DEFAULT 0,
              finished_at TEXT, timeout_at TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              FOREIGN KEY(round_id) REFERENCES battle_rounds(round_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(round_id, actor_key),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL)),
              CHECK(status IN ('playing', 'completed', 'timed_out', 'disconnected', 'disqualified')),
              CHECK(route_index >= 0), CHECK(last_sequence >= 0),
              CHECK(replay_move_count >= 0),
              CHECK(goodness_of_fit >= 0.0 AND goodness_of_fit <= 1.0)
            );
            INSERT INTO battle_player_results
              (result_id, round_id, actor_key, user_id, guest_id,
               display_name_snapshot, status, route_index, last_sequence,
               goodness_of_fit, primary_score, secondary_score, progress,
               mode_data_json, choice_blob, board_state, replay_blob,
               replay_move_count, finished_at, timeout_at, created_at, updated_at)
            SELECT legacy.result_id, legacy.round_id, 'u:' || legacy.user_id,
                   legacy.user_id, NULL, COALESCE(users.display_name, ''),
                   legacy.status, legacy.route_index, legacy.last_sequence,
                   legacy.goodness_of_fit, legacy.primary_score,
                   legacy.secondary_score, legacy.progress,
                   legacy.mode_data_json, legacy.choice_blob,
                   legacy.board_state, legacy.replay_blob,
                   legacy.replay_move_count, legacy.finished_at,
                   legacy.timeout_at, legacy.created_at, legacy.updated_at
            FROM battle_player_results_actor_legacy AS legacy
            LEFT JOIN users ON users.id = legacy.user_id;
            DROP TABLE battle_player_results_actor_legacy;
            """
        )
    if "actor_key" not in _table_columns(db, "battle_chat_messages"):
        migration_scripts.append(
            """
            ALTER TABLE battle_chat_messages
              RENAME TO battle_chat_messages_actor_legacy;
            CREATE TABLE battle_chat_messages (
              message_id INTEGER PRIMARY KEY AUTOINCREMENT,
              room_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              request_id TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(room_id, actor_key, request_id),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL))
            );
            INSERT INTO battle_chat_messages
              (message_id, room_id, actor_key, user_id, guest_id,
               display_name_snapshot, request_id, content, created_at)
            SELECT legacy.message_id, legacy.room_id, 'u:' || legacy.user_id,
                   legacy.user_id, NULL, COALESCE(users.display_name, ''),
                   legacy.request_id, legacy.content, legacy.created_at
            FROM battle_chat_messages_actor_legacy AS legacy
            LEFT JOIN users ON users.id = legacy.user_id;
            DROP TABLE battle_chat_messages_actor_legacy;
            """
        )
    if "actor_key" not in _table_columns(db, "battle_free_player_states"):
        migration_scripts.append(
            """
            ALTER TABLE battle_free_player_states
              RENAME TO battle_free_player_states_actor_legacy;
            CREATE TABLE battle_free_player_states (
              round_id TEXT NOT NULL, actor_key TEXT NOT NULL,
              user_id INTEGER, guest_id TEXT,
              board_state TEXT NOT NULL, step_index INTEGER NOT NULL DEFAULT 0,
              sequence INTEGER NOT NULL DEFAULT 0,
              spawn_log_index REAL NOT NULL DEFAULT 0.0,
              spawn_log_floor REAL NOT NULL DEFAULT 0.0,
              rng_step INTEGER NOT NULL DEFAULT 0,
              state_status TEXT NOT NULL DEFAULT 'input', finish_reason TEXT,
              resolution_request_id TEXT, resolution_started_at TEXT,
              ack_deadline_at TEXT, timeout_at TEXT,
              current_results_json TEXT NOT NULL DEFAULT '{}',
              operation_blob BLOB NOT NULL DEFAULT X'',
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              PRIMARY KEY(round_id, actor_key),
              FOREIGN KEY(round_id) REFERENCES battle_rounds(round_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL))
            );
            INSERT INTO battle_free_player_states
              (round_id, actor_key, user_id, guest_id, board_state, step_index,
               sequence, spawn_log_index, spawn_log_floor, rng_step,
               state_status, finish_reason, resolution_request_id,
               resolution_started_at, ack_deadline_at, timeout_at,
               current_results_json, operation_blob, created_at, updated_at)
            SELECT legacy.round_id, 'u:' || legacy.user_id, legacy.user_id, NULL,
                   legacy.board_state, legacy.step_index, legacy.sequence,
                   legacy.spawn_log_index, legacy.spawn_log_floor, legacy.rng_step,
                   legacy.state_status, legacy.finish_reason,
                   legacy.resolution_request_id, legacy.resolution_started_at,
                   legacy.ack_deadline_at, legacy.timeout_at,
                   legacy.current_results_json, legacy.operation_blob,
                   legacy.created_at, legacy.updated_at
            FROM battle_free_player_states_actor_legacy AS legacy;
            DROP TABLE battle_free_player_states_actor_legacy;
            """
        )
    if migration_scripts:
        db.executescript(
            "BEGIN IMMEDIATE;\n"
            + "\n".join(migration_scripts)
            + "\nCOMMIT;"
        )
    db.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_battle_members_active_actor
          ON battle_members(actor_key) WHERE status = 'active';
        CREATE UNIQUE INDEX IF NOT EXISTS uq_battle_members_active_player_seat
          ON battle_members(room_id, seat_index)
          WHERE status = 'active' AND role = 'player';
        CREATE INDEX IF NOT EXISTS ix_battle_members_room_active
          ON battle_members(room_id, status, role, seat_index);
        CREATE INDEX IF NOT EXISTS ix_battle_chat_room
          ON battle_chat_messages(room_id, message_id DESC);
        CREATE INDEX IF NOT EXISTS ix_battle_chat_rate
          ON battle_chat_messages(room_id, actor_key, created_at DESC);
        CREATE INDEX IF NOT EXISTS ix_battle_free_states_timeout
          ON battle_free_player_states(state_status, timeout_at);
        CREATE INDEX IF NOT EXISTS ix_battle_guest_joins_guest_time
          ON battle_guest_join_events(guest_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS ix_battle_guest_joins_ip_time
          ON battle_guest_join_events(ip_hash, created_at DESC);
        """
    )


def init_battle_db() -> None:
    """Create Battle V1 persistence tables in the shared auth database."""
    with auth_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS battle_rooms (
              room_id TEXT PRIMARY KEY,
              room_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
              host_user_id INTEGER NOT NULL,
              status TEXT NOT NULL DEFAULT 'preparing',
              visibility TEXT NOT NULL DEFAULT 'public',
              allow_spectators INTEGER NOT NULL DEFAULT 1,
              allow_guest_chat INTEGER NOT NULL DEFAULT 0,
              max_players INTEGER NOT NULL DEFAULT 2,
              mode_key TEXT NOT NULL DEFAULT 'goodness',
              mode_version INTEGER NOT NULL DEFAULT 1,
              settings_json TEXT NOT NULL DEFAULT '{}',
              chat_roles_json TEXT NOT NULL DEFAULT '["host","player","spectator"]',
              pattern TEXT NOT NULL,
              target INTEGER NOT NULL,
              full_pattern TEXT NOT NULL,
              initial_board TEXT,
              max_steps INTEGER,
              step_timeout_seconds INTEGER NOT NULL DEFAULT 90,
              current_round_number INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              closed_at TEXT,
              FOREIGN KEY(host_user_id) REFERENCES users(id),
              CHECK(status IN ('preparing', 'waiting', 'running', 'closed', 'expired')),
              CHECK(visibility IN ('public', 'private')),
              CHECK(allow_spectators IN (0, 1)),
              CHECK(max_players BETWEEN 2 AND 8),
              CHECK(target > 0),
              CHECK(max_steps IS NULL OR max_steps > 0),
              CHECK(step_timeout_seconds > 0)
            );

            CREATE TABLE IF NOT EXISTS battle_members (
              member_id INTEGER PRIMARY KEY AUTOINCREMENT,
              room_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              role TEXT NOT NULL,
              seat_index INTEGER,
              ready INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'active',
              joined_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              left_at TEXT,
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(room_id, actor_key),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL)),
              CHECK(role IN ('player', 'spectator')),
              CHECK(status IN ('active', 'left', 'kicked')),
              CHECK(ready IN (0, 1)),
              CHECK(
                (role = 'player' AND seat_index IS NOT NULL AND seat_index >= 0)
                OR (role = 'spectator' AND seat_index IS NULL)
              )
            );

            CREATE TABLE IF NOT EXISTS battle_rounds (
              round_id TEXT PRIMARY KEY,
              room_id TEXT NOT NULL,
              round_number INTEGER NOT NULL,
              status TEXT NOT NULL DEFAULT 'preparing',
              token_reservation_id TEXT,
              token_cost_units INTEGER NOT NULL DEFAULT 0,
              started_at TEXT,
              ended_at TEXT,
              expires_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              artifact_kind TEXT,
              artifact_blob BLOB,
              artifact_hash TEXT,
              mode_state_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE,
              UNIQUE(room_id, round_number),
              CHECK(status IN ('preparing', 'ready', 'running', 'completed', 'failed', 'cancelled')),
              CHECK(round_number > 0),
              CHECK(token_cost_units >= 0)
            );

            CREATE TABLE IF NOT EXISTS battle_routes (
              route_id TEXT PRIMARY KEY,
              round_id TEXT NOT NULL UNIQUE,
              full_pattern TEXT NOT NULL,
              target INTEGER NOT NULL,
              initial_board TEXT NOT NULL,
              step_count INTEGER NOT NULL,
              certainty_step INTEGER,
              termination_reason TEXT NOT NULL,
              route_hash TEXT NOT NULL,
              route_blob BLOB NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(round_id) REFERENCES battle_rounds(round_id) ON DELETE CASCADE,
              CHECK(step_count >= 0),
              CHECK(certainty_step IS NULL OR certainty_step >= 0)
            );

            CREATE TABLE IF NOT EXISTS battle_player_results (
              result_id INTEGER PRIMARY KEY AUTOINCREMENT,
              round_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'playing',
              route_index INTEGER NOT NULL DEFAULT 0,
              last_sequence INTEGER NOT NULL DEFAULT 0,
              goodness_of_fit REAL NOT NULL DEFAULT 1.0,
              primary_score REAL,
              secondary_score REAL,
              progress INTEGER NOT NULL DEFAULT 0,
              mode_data_json TEXT NOT NULL DEFAULT '{}',
              choice_blob BLOB,
              replay_blob BLOB NOT NULL DEFAULT X'',
              replay_move_count INTEGER NOT NULL DEFAULT 0,
              finished_at TEXT,
              timeout_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(round_id) REFERENCES battle_rounds(round_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(round_id, actor_key),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL)),
              CHECK(status IN ('playing', 'completed', 'timed_out', 'disconnected', 'disqualified')),
              CHECK(route_index >= 0),
              CHECK(last_sequence >= 0),
              CHECK(replay_move_count >= 0),
              CHECK(goodness_of_fit >= 0.0 AND goodness_of_fit <= 1.0)
            );

            CREATE INDEX IF NOT EXISTS ix_battle_rooms_public_lobby
              ON battle_rooms(visibility, status, created_at DESC);

            CREATE INDEX IF NOT EXISTS ix_battle_rounds_room
              ON battle_rounds(room_id, round_number DESC);

            CREATE INDEX IF NOT EXISTS ix_battle_routes_hash
              ON battle_routes(route_hash);

            CREATE TABLE IF NOT EXISTS battle_request_ids (
              user_id INTEGER NOT NULL,
              request_id TEXT NOT NULL,
              operation TEXT NOT NULL,
              response_json TEXT,
              created_at TEXT NOT NULL,
              PRIMARY KEY(user_id, request_id),
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS battle_guest_join_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              guest_id TEXT NOT NULL,
              room_id TEXT NOT NULL,
              ip_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(guest_id, room_id),
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS battle_chat_messages (
              message_id INTEGER PRIMARY KEY AUTOINCREMENT,
              room_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              display_name_snapshot TEXT NOT NULL DEFAULT '',
              request_id TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(room_id) REFERENCES battle_rooms(room_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              UNIQUE(room_id, actor_key, request_id),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL))
            );

            CREATE TABLE IF NOT EXISTS battle_free_player_states (
              round_id TEXT NOT NULL,
              actor_key TEXT NOT NULL,
              user_id INTEGER,
              guest_id TEXT,
              board_state TEXT NOT NULL,
              step_index INTEGER NOT NULL DEFAULT 0,
              sequence INTEGER NOT NULL DEFAULT 0,
              spawn_log_index REAL NOT NULL DEFAULT 0.0,
              spawn_log_floor REAL NOT NULL DEFAULT 0.0,
              rng_step INTEGER NOT NULL DEFAULT 0,
              state_status TEXT NOT NULL DEFAULT 'input',
              finish_reason TEXT,
              resolution_request_id TEXT,
              resolution_started_at TEXT,
              ack_deadline_at TEXT,
              timeout_at TEXT,
              current_results_json TEXT NOT NULL DEFAULT '{}',
              operation_blob BLOB NOT NULL DEFAULT X'',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (round_id, actor_key),
              FOREIGN KEY(round_id) REFERENCES battle_rounds(round_id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id),
              CHECK((user_id IS NOT NULL AND guest_id IS NULL) OR
                    (user_id IS NULL AND guest_id IS NOT NULL))
            );

            """
        )
        room_columns = {row["name"] for row in db.execute("PRAGMA table_info(battle_rooms)")}
        round_columns = {row["name"] for row in db.execute("PRAGMA table_info(battle_rounds)")}
        for name, declaration in {
            "revision": "INTEGER NOT NULL DEFAULT 1",
            "generation_error": "TEXT",
            "mode_key": "TEXT NOT NULL DEFAULT 'goodness'",
            "mode_version": "INTEGER NOT NULL DEFAULT 1",
            "settings_json": "TEXT NOT NULL DEFAULT '{}'",
            "chat_roles_json": "TEXT NOT NULL DEFAULT '[\"host\",\"player\",\"spectator\"]'",
            "allow_guest_chat": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in room_columns:
                db.execute(f"ALTER TABLE battle_rooms ADD COLUMN {name} {declaration}")
        for name, declaration in {
            "reservation_ledger_id": "INTEGER",
            "reserved_bonus_units": "INTEGER NOT NULL DEFAULT 0",
            "reserved_paid_units": "INTEGER NOT NULL DEFAULT 0",
            "route_seed": "TEXT",
            "error_code": "TEXT",
            "reservation_status": "TEXT NOT NULL DEFAULT 'reserved'",
            "auto_start_after_generation": "INTEGER NOT NULL DEFAULT 0",
            "artifact_kind": "TEXT",
            "artifact_blob": "BLOB",
            "artifact_hash": "TEXT",
            "mode_state_json": "TEXT NOT NULL DEFAULT '{}'",
        }.items():
            if name not in round_columns:
                db.execute(f"ALTER TABLE battle_rounds ADD COLUMN {name} {declaration}")
        result_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(battle_player_results)")
        }
        for name, declaration in {
            "board_state": "TEXT",
            "primary_score": "REAL",
            "secondary_score": "REAL",
            "progress": "INTEGER NOT NULL DEFAULT 0",
            "mode_data_json": "TEXT NOT NULL DEFAULT '{}'",
            "replay_blob": "BLOB NOT NULL DEFAULT X''",
            "replay_move_count": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in result_columns:
                db.execute(f"ALTER TABLE battle_player_results ADD COLUMN {name} {declaration}")
        _migrate_actor_identity_tables(db)
        db.execute(
            "CREATE INDEX IF NOT EXISTS ix_battle_rooms_mode_lobby "
            "ON battle_rooms(mode_key, visibility, status, created_at DESC)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS ix_battle_rooms_host_created "
            "ON battle_rooms(host_user_id, created_at DESC)"
        )


def _active_user(db: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    row = db.execute(
        "SELECT id FROM users WHERE id = ? AND status = 'active'", (int(user_id),)
    ).fetchone()
    if row is None:
        raise BattleNotFoundError("user_not_found")
    return row


def _find_room(db: sqlite3.Connection, room_ref: str) -> sqlite3.Row:
    value = str(room_ref or "").strip()
    row = db.execute(
        """
        SELECT * FROM battle_rooms
        WHERE room_id = ? OR room_code = ? COLLATE NOCASE
        """,
        (value, _normalize_room_code(value)),
    ).fetchone()
    if row is None:
        raise BattleNotFoundError("room_not_found")
    return row


def _active_membership(db: sqlite3.Connection, actor: BattleActor) -> sqlite3.Row | None:
    return db.execute(
        "SELECT * FROM battle_members WHERE actor_key = ? AND status = 'active'",
        (actor.actor_key,),
    ).fetchone()


def _room_creation_retry_after(
    db: sqlite3.Connection,
    user_id: int,
    *,
    now: datetime,
) -> int:
    row = db.execute(
        "SELECT created_at FROM battle_rooms WHERE host_user_id = ? ORDER BY created_at DESC LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        return 0
    try:
        created_at = datetime.fromisoformat(str(row["created_at"]))
    except (TypeError, ValueError):
        return 0
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    elapsed = (now.astimezone(timezone.utc) - created_at.astimezone(timezone.utc)).total_seconds()
    return max(0, int(math.ceil(ROOM_CREATION_COOLDOWN_SECONDS - elapsed)))


def room_creation_retry_after(user_id: int, *, now: datetime | None = None) -> int:
    with auth_db() as db:
        return _room_creation_retry_after(
            db,
            int(user_id),
            now=(now or _utc_now()),
        )


def _next_seat(db: sqlite3.Connection, room_id: str, max_players: int) -> int | None:
    occupied = {
        int(row["seat_index"])
        for row in db.execute(
            """
            SELECT seat_index FROM battle_members
            WHERE room_id = ? AND status = 'active' AND role = 'player'
            """,
            (room_id,),
        ).fetchall()
    }
    return next((seat for seat in range(max_players) if seat not in occupied), None)


def _room_payload(db: sqlite3.Connection, room: sqlite3.Row) -> dict[str, Any]:
    payload = _row_dict(room) or {}
    raw_settings = payload.pop("settings_json", None)
    raw_chat_roles = payload.pop("chat_roles_json", None)
    try:
        payload["settings"] = json.loads(raw_settings or "{}")
    except (TypeError, ValueError):
        payload["settings"] = {}
    payload["chat_roles"] = decode_chat_roles(raw_chat_roles)
    members = db.execute(
        """
        SELECT m.*, COALESCE(u.display_name, m.display_name_snapshot) AS display_name,
               p.avatar_key,
               COALESCE(e.tier, 'free') AS entitlement_tier
        FROM battle_members AS m
        LEFT JOIN users AS u ON u.id = m.user_id
        LEFT JOIN user_profiles AS p ON p.user_id = m.user_id
        LEFT JOIN user_entitlements AS e ON e.user_id = m.user_id
        WHERE m.room_id = ? AND m.status = 'active'
        ORDER BY CASE m.role WHEN 'player' THEN 0 ELSE 1 END, m.seat_index, m.joined_at
        """,
        (room["room_id"],),
    ).fetchall()
    payload["members"] = []
    for member in members:
        item = _row_dict(member) or {}
        item["avatar_url"] = (
            f"/media/avatars/{item['avatar_key']}" if item.get("avatar_key") else None
        )
        item.pop("avatar_key", None)
        item["actor_kind"] = "guest" if item.get("guest_id") else "user"
        item["is_guest"] = bool(item.get("guest_id"))
        payload["members"].append(item)
    payload["player_count"] = sum(member["role"] == "player" for member in members)
    payload["spectator_count"] = sum(member["role"] == "spectator" for member in members)
    round_row = db.execute(
        "SELECT * FROM battle_rounds WHERE room_id = ? ORDER BY round_number DESC LIMIT 1",
        (room["room_id"],),
    ).fetchone()
    payload["round"] = _row_dict(round_row)
    if round_row is not None:
        raw_mode_state = payload["round"].pop("mode_state_json", None)
        try:
            payload["round"]["mode_state"] = json.loads(raw_mode_state or "{}")
        except (TypeError, ValueError):
            payload["round"]["mode_state"] = {}
        payload["round"].pop("artifact_blob", None)
        route = db.execute(
            "SELECT route_id, round_id, initial_board, step_count, certainty_step, termination_reason, route_hash FROM battle_routes WHERE round_id = ?",
            (round_row["round_id"],),
        ).fetchone()
        payload["route"] = _row_dict(route)
        results = db.execute(
            """
            SELECT r.*, COALESCE(u.display_name, r.display_name_snapshot) AS display_name,
                   p.avatar_key
            FROM battle_player_results AS r
            LEFT JOIN users AS u ON u.id = r.user_id
            LEFT JOIN user_profiles AS p ON p.user_id = r.user_id
            WHERE r.round_id = ? ORDER BY r.goodness_of_fit DESC, r.finished_at, r.actor_key
            """,
            (round_row["round_id"],),
        ).fetchall()
        payload["results"] = []
        for row in results:
            item = _row_dict(row) or {}
            item["avatar_url"] = (
                f"/media/avatars/{item['avatar_key']}" if item.get("avatar_key") else None
            )
            item.pop("avatar_key", None)
            item["actor_kind"] = "guest" if item.get("guest_id") else "user"
            item["is_guest"] = bool(item.get("guest_id"))
            item.pop("choice_blob", None)
            item.pop("replay_blob", None)
            item.pop("board_state", None)
            raw_mode_data = item.pop("mode_data_json", None)
            try:
                item["mode_data"] = json.loads(raw_mode_data or "{}")
            except (TypeError, ValueError):
                item["mode_data"] = {}
            payload["results"].append(item)
    else:
        payload["route"] = None
        payload["results"] = []
    return payload


def create_room(
    *,
    host_user_id: int,
    pattern: str,
    target: int,
    full_pattern: str | None = None,
    room_code: str | None = None,
    visibility: str = "public",
    allow_spectators: bool = True,
    allow_guest_chat: bool = False,
    max_players: int = 2,
    initial_board: str | None = None,
    max_steps: int | None = None,
    step_timeout_seconds: int = 90,
    mode_key: str = "goodness",
    mode_version: int = 1,
    chat_roles: list[str] | tuple[str, ...] | None = None,
    settings: dict[str, Any] | None = None,
    status: str = "preparing",
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    pattern = str(pattern or "").strip()
    if not pattern:
        raise ValueError("invalid_pattern")
    target = int(target)
    max_players = int(max_players)
    step_timeout_seconds = int(step_timeout_seconds)
    mode_key = str(mode_key or "").strip().lower()
    mode_version = int(mode_version)
    normalized_chat_roles = normalize_chat_roles(chat_roles)
    if target <= 0:
        raise ValueError("invalid_target")
    if not 2 <= max_players <= 8:
        raise ValueError("invalid_max_players")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("invalid_max_steps")
    if step_timeout_seconds <= 0:
        raise ValueError("invalid_step_timeout")
    if not mode_key or len(mode_key) > 64:
        raise ValueError("invalid_mode_key")
    if mode_version <= 0:
        raise ValueError("invalid_mode_version")
    if visibility not in ROOM_VISIBILITIES:
        raise ValueError("invalid_visibility")
    if status not in ROOM_STATUSES:
        raise ValueError("invalid_room_status")

    now = _utc_now()
    room_id = str(uuid.uuid4())
    requested_code = _normalize_room_code(room_code) if room_code else None
    if requested_code and (len(requested_code) != 6 or any(c not in ROOM_CODE_ALPHABET for c in requested_code)):
        raise ValueError("invalid_room_code")

    try:
        with auth_db() as db:
            db.execute("BEGIN IMMEDIATE")
            _active_user(db, host_user_id)
            host_actor = user_actor(host_user_id)
            if _active_membership(db, host_actor) is not None:
                raise BattleConflictError("user_already_in_room")
            if _room_creation_retry_after(db, host_user_id, now=now) > 0:
                raise BattleConflictError("room_create_cooldown")

            selected_code = requested_code
            for attempt in range(16):
                selected_code = selected_code or _room_code()
                try:
                    db.execute(
                        """
                        INSERT INTO battle_rooms
                        (room_id, room_code, host_user_id, status, visibility,
                         allow_spectators, allow_guest_chat, max_players, mode_key, mode_version,
                         settings_json, chat_roles_json, pattern, target, full_pattern,
                         initial_board, max_steps, step_timeout_seconds,
                         created_at, updated_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            room_id,
                            selected_code,
                            int(host_user_id),
                            status,
                            visibility,
                            1 if allow_spectators else 0,
                            1 if allow_guest_chat else 0,
                            max_players,
                            mode_key,
                            mode_version,
                            json.dumps(settings or {}, separators=(",", ":"), sort_keys=True),
                            json.dumps(normalized_chat_roles, separators=(",", ":")),
                            pattern,
                            target,
                            str(full_pattern or f"{pattern}_{target}"),
                            str(initial_board) if initial_board is not None else None,
                            int(max_steps) if max_steps is not None else None,
                            step_timeout_seconds,
                            _iso(now),
                            _iso(now),
                            _iso(expires_at or now + WAITING_ROOM_LIFETIME),
                        ),
                    )
                    break
                except sqlite3.IntegrityError as exc:
                    if requested_code or "battle_rooms.room_code" not in str(exc):
                        raise
                    selected_code = None
            else:
                raise BattleConflictError("room_code_generation_failed")

            db.execute(
                """
                INSERT INTO battle_members
                (room_id, actor_key, user_id, guest_id, display_name_snapshot,
                 role, seat_index, ready, status, joined_at, updated_at)
                SELECT ?, ?, ?, NULL, COALESCE(display_name, ''),
                       'player', 0, 0, 'active', ?, ?
                FROM users WHERE id = ?
                """,
                (
                    room_id,
                    host_actor.actor_key,
                    int(host_user_id),
                    _iso(now),
                    _iso(now),
                    int(host_user_id),
                ),
            )
            return _room_payload(db, _find_room(db, room_id))
    except sqlite3.IntegrityError as exc:
        raise BattleConflictError("room_create_conflict") from exc


def get_room(room_ref: str) -> dict[str, Any]:
    with auth_db() as db:
        return _room_payload(db, _find_room(db, room_ref))


def room_unavailable_reason(
    room_ref: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
) -> str | None:
    """Return a close reason only when room access is definitively gone."""
    with auth_db() as db:
        try:
            room = _find_room(db, room_ref)
        except BattleNotFoundError:
            return "ROOM_NOT_FOUND"
        status = str(room["status"] or "")
        if status not in ACTIVE_ROOM_STATUSES:
            if room["generation_error"]:
                return str(room["generation_error"])
            return "ROOM_EXPIRED" if status == "expired" else "ROOM_CLOSED"
        identity = coerce_actor(actor, user_id=user_id)
        member = db.execute(
            "SELECT status FROM battle_members WHERE room_id = ? AND actor_key = ?",
            (str(room["room_id"]), identity.actor_key),
        ).fetchone()
        if member is None:
            return "ROOM_MEMBERSHIP_REQUIRED"
        member_status = str(member["status"] or "")
        if member_status == "active":
            return None
        if member_status == "kicked":
            return "KICKED_FROM_ROOM"
        return "ROOM_MEMBERSHIP_REQUIRED"


def list_public_rooms(*, limit: int = 50, now: datetime | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    current = _iso(now or _utc_now())
    with auth_db() as db:
        rows = db.execute(
            """
            SELECT r.*, host.display_name AS host_display_name,
                   profile.avatar_key AS host_avatar_key,
                   SUM(CASE WHEN m.status = 'active' AND m.role = 'player' THEN 1 ELSE 0 END) AS player_count,
                   SUM(CASE WHEN m.status = 'active' AND m.role = 'spectator' THEN 1 ELSE 0 END) AS spectator_count
            FROM battle_rooms AS r
            JOIN users AS host ON host.id = r.host_user_id
            LEFT JOIN user_profiles AS profile ON profile.user_id = r.host_user_id
            LEFT JOIN battle_members AS m ON m.room_id = r.room_id
            WHERE r.visibility = 'public'
              AND (
                r.status IN ('preparing', 'waiting')
                OR (r.status = 'running' AND r.allow_spectators = 1)
              )
              AND r.expires_at > ?
            GROUP BY r.room_id
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (current, limit),
        ).fetchall()
        payloads = []
        for row in rows:
            payload = _row_dict(row) or {}
            payload["player_count"] = int(payload.get("player_count") or 0)
            payload["spectator_count"] = int(payload.get("spectator_count") or 0)
            payload["host"] = {
                "display_name": str(payload.pop("host_display_name", "") or ""),
                "avatar_url": (
                    f"/media/avatars/{payload['host_avatar_key']}"
                    if payload.get("host_avatar_key")
                    else None
                ),
            }
            payload.pop("host_avatar_key", None)
            payloads.append(payload)
        return payloads


def join_room(
    room_ref: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    preferred_role: str | None = None,
    ip_address: str = "",
) -> dict[str, Any]:
    if preferred_role not in (None, "player", "spectator"):
        raise ValueError("invalid_member_role")
    identity = coerce_actor(actor, user_id=user_id)
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        if identity.is_user:
            _active_user(db, int(identity.user_id))
        room = _find_room(db, room_ref)
        existing_active = _active_membership(db, identity)
        if existing_active is not None:
            if existing_active["room_id"] == room["room_id"]:
                return _row_dict(existing_active) or {}
            raise BattleConflictError("user_already_in_room")
        if room["status"] not in ACTIVE_ROOM_STATUSES:
            raise BattleConflictError("room_not_joinable")
        if datetime.fromisoformat(str(room["expires_at"])) <= now and room["status"] != "running":
            raise BattleConflictError("room_expired")

        seat = _next_seat(db, str(room["room_id"]), int(room["max_players"]))
        if room["status"] == "running":
            role = "spectator"
        elif preferred_role == "spectator":
            role = "spectator"
        elif seat is not None:
            role = "player"
        else:
            role = "spectator"
        if role == "spectator" and not bool(room["allow_spectators"]):
            raise BattleConflictError("spectators_disabled")
        if role == "player" and seat is None:
            raise BattleConflictError("player_slots_full")

        existing = db.execute(
            "SELECT member_id, status FROM battle_members WHERE room_id = ? AND actor_key = ?",
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if existing is not None and existing["status"] == "kicked":
            raise BattleConflictError("kicked_from_room")
        _reserve_guest_join(
            db,
            identity=identity,
            room_id=str(room["room_id"]),
            ip_address=ip_address,
            now=now,
        )
        if existing is None:
            cursor = db.execute(
                """
                INSERT INTO battle_members
                (room_id, actor_key, user_id, guest_id, display_name_snapshot,
                 role, seat_index, ready, status, joined_at, updated_at, left_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, NULL)
                """,
                (
                    room["room_id"],
                    identity.actor_key,
                    identity.user_id,
                    identity.guest_id,
                    identity.display_name,
                    role,
                    seat if role == "player" else None,
                    _iso(now),
                    _iso(now),
                ),
            )
            member_id = int(cursor.lastrowid)
        else:
            member_id = int(existing["member_id"])
            db.execute(
                """
                UPDATE battle_members
                SET role = ?, seat_index = ?, ready = 0, status = 'active',
                    display_name_snapshot = ?, joined_at = ?, updated_at = ?, left_at = NULL
                WHERE member_id = ?
                """,
                (
                    role,
                    seat if role == "player" else None,
                    identity.display_name,
                    _iso(now),
                    _iso(now),
                    member_id,
                ),
            )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (_iso(now), room["room_id"]),
        )
        member = db.execute(
            "SELECT * FROM battle_members WHERE member_id = ?", (member_id,)
        ).fetchone()
        return _row_dict(member) or {}


def set_member_ready(
    room_ref: str,
    *,
    actor: Any | None = None,
    user_id: int | None = None,
    ready: bool,
) -> dict[str, Any]:
    identity = coerce_actor(actor, user_id=user_id)
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = _find_room(db, room_ref)
        if room["status"] not in {"preparing", "waiting"}:
            raise BattleConflictError("room_not_waiting")
        member = db.execute(
            """
            SELECT * FROM battle_members
            WHERE room_id = ? AND actor_key = ? AND status = 'active'
            """,
            (room["room_id"], identity.actor_key),
        ).fetchone()
        if member is None:
            raise BattleNotFoundError("member_not_found")
        if member["role"] != "player":
            raise BattleConflictError("spectator_cannot_ready")
        db.execute(
            "UPDATE battle_members SET ready = ?, updated_at = ? WHERE member_id = ?",
            (1 if ready else 0, _iso(now), member["member_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (_iso(now), room["room_id"]),
        )
        updated = db.execute(
            "SELECT * FROM battle_members WHERE member_id = ?", (member["member_id"],)
        ).fetchone()
        return _row_dict(updated) or {}


def kick_member(
    room_ref: str,
    *,
    host_user_id: int,
    target_actor_key: str | None = None,
    target_user_id: int | None = None,
) -> dict[str, Any]:
    target_key = str(target_actor_key or "") or user_actor(int(target_user_id)).actor_key
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = _find_room(db, room_ref)
        if int(room["host_user_id"]) != int(host_user_id):
            raise BattlePermissionError("host_required")
        if target_key == user_actor(host_user_id).actor_key:
            raise BattleConflictError("host_cannot_be_kicked")
        if room["status"] not in {"preparing", "waiting"}:
            raise BattleConflictError("room_already_started")
        member = db.execute(
            """
            SELECT * FROM battle_members
            WHERE room_id = ? AND actor_key = ? AND status = 'active'
            """,
            (room["room_id"], target_key),
        ).fetchone()
        if member is None:
            raise BattleNotFoundError("member_not_found")
        db.execute(
            """
            UPDATE battle_members
            SET status = 'kicked', ready = 0, updated_at = ?, left_at = ?
            WHERE member_id = ?
            """,
            (_iso(now), _iso(now), member["member_id"]),
        )
        db.execute(
            "UPDATE battle_rooms SET revision = revision + 1, updated_at = ? WHERE room_id = ?",
            (_iso(now), room["room_id"]),
        )
        updated = db.execute(
            "SELECT * FROM battle_members WHERE member_id = ?", (member["member_id"],)
        ).fetchone()
        return _row_dict(updated) or {}


def close_room(
    room_ref: str,
    *,
    host_user_id: int | None = None,
    status: str = "closed",
) -> dict[str, Any]:
    if status not in {"closed", "expired"}:
        raise ValueError("invalid_close_status")
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        room = _find_room(db, room_ref)
        if host_user_id is not None and int(room["host_user_id"]) != int(host_user_id):
            raise BattlePermissionError("host_required")
        if room["status"] in {"closed", "expired"}:
            return _room_payload(db, room)
        db.execute(
            """
            UPDATE battle_rooms
            SET status = ?, updated_at = ?, closed_at = ?, revision = revision + 1
            WHERE room_id = ?
            """,
            (status, _iso(now), _iso(now), room["room_id"]),
        )
        db.execute(
            """
            UPDATE battle_members
            SET status = 'left', ready = 0, updated_at = ?, left_at = ?
            WHERE room_id = ? AND status = 'active'
            """,
            (_iso(now), _iso(now), room["room_id"]),
        )
        return _room_payload(db, _find_room(db, str(room["room_id"])))
