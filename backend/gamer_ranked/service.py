from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import secrets
import sqlite3
import uuid
from typing import Any

from backend.auth.db import auth_db
from backend.replay_2048next import RANKED_RULES_VERSION, REPLAY_PREFIX

from .validator import (
    RankedValidationError,
    ValidatedGame,
    classify_ranked_candidate,
    validate_ranked_game,
)


RUN_LIFETIME = timedelta(days=30)
MAX_RECORD_BYTES = 500 * 1024
MAX_PENDING_GLOBAL = 32
MAX_DAILY_SUBMISSIONS_USER = 5
MAX_DAILY_SUBMISSIONS_IP = 20
DEFAULT_REPLAY_RETENTION_PER_BOARD = 100
REPLAY_RETENTION_ENV = "GAMER_REPLAY_RETENTION_PER_BOARD"
RANKED_BOARD_KEYS = ("gamer_high_score", "gamer_adversarial")
WEEKLY_BOARD_KEYS = {
    "gamer_high_score": "gamer_high_score_weekly",
    "gamer_adversarial": "gamer_adversarial_weekly",
}
RANKING_TIMEZONE = timezone(timedelta(hours=8), name="UTC+08:00")
MIN_RANKED_SPAWN_RATE4_MILLIS = 100
MAX_RANKED_SPAWN_RATE4_MILLIS = 800
LEASE_LIFETIME = timedelta(seconds=60)
CREATE_RUN_WINDOW = timedelta(seconds=60)
MAX_RUN_CREATIONS_USER = 30
MAX_RUN_CREATIONS_IP = 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _week_start_iso(value: datetime | str | None = None) -> str:
    if isinstance(value, str):
        current = datetime.fromisoformat(value)
    else:
        current = value or _utc_now()
    local = current.astimezone(RANKING_TIMEZONE)
    monday = (local - timedelta(days=local.weekday())).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return monday.isoformat()


def _seed_hex() -> str:
    while True:
        words = [secrets.randbits(32) for _ in range(4)]
        if all(words):
            return "".join(f"{word:08x}" for word in words)


def _spawn_rate4_millis(value: float) -> int:
    rate4 = float(value)
    if (
        not math.isfinite(rate4)
        or rate4 < MIN_RANKED_SPAWN_RATE4_MILLIS / 1000.0
        or rate4 > MAX_RANKED_SPAWN_RATE4_MILLIS / 1000.0
    ):
        raise ValueError("spawn_rate_out_of_range")
    millis = round(rate4 * 1000)
    if not MIN_RANKED_SPAWN_RATE4_MILLIS <= millis <= MAX_RANKED_SPAWN_RATE4_MILLIS:
        raise ValueError("spawn_rate_out_of_range")
    return millis


def _lease_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _lease_matches(row: sqlite3.Row | dict[str, Any], token: str) -> bool:
    expected = str(row["lease_token_hash"] or "")
    supplied = _lease_hash(token)
    return bool(expected) and secrets.compare_digest(expected, supplied)


def _new_lease(now: datetime, token: str | None = None) -> tuple[str, str, str]:
    token = str(token or secrets.token_urlsafe(32))
    if len(token) < 16 or len(token) > 256:
        raise ValueError("invalid_lease_token")
    return token, _lease_hash(token), _iso(now + LEASE_LIFETIME)


def _lease_expired(row: sqlite3.Row | dict[str, Any], now: datetime) -> bool:
    raw_expiry = row["lease_expires_at"]
    if not raw_expiry:
        return True
    try:
        return datetime.fromisoformat(str(raw_expiry)) <= now
    except ValueError:
        return True


def _public_run(
    row: sqlite3.Row | dict[str, Any],
    *,
    lease_token: str | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": str(row["run_id"]),
        "status": str(row["status"]),
        "rules_version": int(row["rules_version"]),
        "spawn_rate4": int(row["spawn_rate4_millis"]) / 1000.0,
        "seed_hex": str(row["seed_hex"]),
        "started_at": row["started_at"],
        "expires_at": row["expires_at"],
        "submitted_at": row["submitted_at"],
        "completed_at": row["completed_at"],
        "board_key": row["board_key"],
        "new_personal_best": bool(row["new_personal_best"]),
        "error_code": row["error_code"],
        "lease_expires_at": row["lease_expires_at"],
    }
    if lease_token:
        payload["lease_token"] = lease_token
    return payload


def create_ranked_run(
    *,
    user_id: int,
    request_id: str,
    ip_address: str,
    spawn_rate4: float = 0.1,
    lease_token: str,
    replace_run_id: str | None = None,
    replace_lease_token: str | None = None,
) -> dict[str, Any]:
    normalized_request = str(request_id or "").strip()
    if not normalized_request or len(normalized_request) > 128:
        raise ValueError("invalid_request_id")
    spawn_rate4_millis = _spawn_rate4_millis(spawn_rate4)
    now = _utc_now()
    expires = now + RUN_LIFETIME
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE user_id = ? AND request_id = ?",
            (user_id, normalized_request),
        ).fetchone()
        if existing is not None:
            if existing["status"] != "active":
                return _public_run(existing)
            if not _lease_matches(existing, lease_token):
                raise PermissionError("lease_mismatch")
            lease_token, lease_hash, lease_expires_at = _new_lease(now, lease_token)
            db.execute(
                """
                UPDATE gamer_ranked_runs
                SET lease_token_hash = ?, lease_expires_at = ?, lease_last_seen_at = ?
                WHERE run_id = ? AND status = 'active'
                """,
                (lease_hash, lease_expires_at, _iso(now), existing["run_id"]),
            )
            existing = db.execute(
                "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (existing["run_id"],)
            ).fetchone()
            return _public_run(existing, lease_token=lease_token)
        # Retries above are free; only newly issued seeds count toward the limit.
        cutoff = _iso(now - CREATE_RUN_WINDOW)
        for column, value, limit in (
            ("user_id", user_id, MAX_RUN_CREATIONS_USER),
            ("start_ip", ip_address, MAX_RUN_CREATIONS_IP),
        ):
            count = db.execute(
                f"SELECT COUNT(*) FROM gamer_ranked_runs WHERE {column} = ? AND started_at > ?",
                (value, cutoff),
            ).fetchone()[0]
            if count >= limit:
                raise RuntimeError("run_creation_rate_limit")
        if replace_run_id:
            active = db.execute(
                "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (replace_run_id,),
            ).fetchone()
            if active is not None:
                if int(active["user_id"]) != user_id:
                    raise PermissionError("run_owner_mismatch")
                if active["status"] == "active":
                    if not _lease_matches(active, str(replace_lease_token or "")):
                        raise PermissionError("lease_mismatch")
                    db.execute(
                        """
                        UPDATE gamer_ranked_runs
                        SET status = 'expired', completed_at = ?, error_code = 'replaced',
                            lease_token_hash = NULL, lease_expires_at = NULL
                        WHERE run_id = ? AND status = 'active'
                        """,
                        (_iso(now), active["run_id"]),
                    )
        run_id = str(uuid.uuid4())
        seed_hex = _seed_hex()
        lease_token, lease_hash, lease_expires_at = _new_lease(now, lease_token)
        db.execute(
            """
            INSERT INTO gamer_ranked_runs
            (run_id, user_id, request_id, seed_hex, rules_version,
             spawn_rate4_millis, status, started_at, expires_at, start_ip)
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                run_id,
                user_id,
                normalized_request,
                seed_hex,
                RANKED_RULES_VERSION,
                spawn_rate4_millis,
                _iso(now),
                _iso(expires),
                ip_address,
            ),
        )
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET lease_token_hash = ?, lease_expires_at = ?, lease_last_seen_at = ?
            WHERE run_id = ?
            """,
            (lease_hash, lease_expires_at, _iso(now), run_id),
        )
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _public_run(row, lease_token=lease_token)


def heartbeat_ranked_run(*, run_id: str, user_id: int, lease_token: str) -> dict[str, Any]:
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != user_id:
            raise PermissionError("run_owner_mismatch")
        if row["status"] != "active":
            raise RuntimeError("run_not_active")
        if not _lease_matches(row, lease_token):
            raise PermissionError("lease_mismatch")
        lease_expires_at = _iso(now + LEASE_LIFETIME)
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET lease_expires_at = ?, lease_last_seen_at = ?
            WHERE run_id = ? AND status = 'active'
            """,
            (lease_expires_at, _iso(now), run_id),
        )
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _public_run(row)


def abandon_ranked_run(*, run_id: str, user_id: int, lease_token: str) -> dict[str, Any]:
    now = _utc_now()
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != user_id:
            raise PermissionError("run_owner_mismatch")
        if row["status"] != "active":
            return _public_run(row)
        if not _lease_matches(row, lease_token):
            raise PermissionError("lease_mismatch")
        if _lease_expired(row, now):
            db.execute(
                """
                UPDATE gamer_ranked_runs
                SET status = 'expired', completed_at = ?, error_code = 'lease_expired',
                    lease_token_hash = NULL, lease_expires_at = NULL
                WHERE run_id = ? AND status = 'active'
                """,
                (_iso(now), run_id),
            )
            row = db.execute(
                "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return _public_run(row)
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'expired', completed_at = ?, error_code = 'abandoned',
                lease_token_hash = NULL, lease_expires_at = NULL
            WHERE run_id = ? AND status = 'active'
            """,
            (_iso(now), run_id),
        )
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _public_run(row)


def _decoded_record_size(record_encoding: str) -> int:
    text = str(record_encoding or "").strip()
    if not text.startswith(REPLAY_PREFIX):
        raise ValueError("invalid_record")
    encoded = "".join(text[len(REPLAY_PREFIX):].split())
    try:
        return len(base64.b64decode(encoded, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid_record") from exc


def _top_100_cutoff(
    db: sqlite3.Connection,
    *,
    board_key: str,
    week_start: str | None = None,
) -> int | None:
    source_table = "gamer_weekly_high_scores" if week_start is not None else "gamer_high_scores"
    period_clause = "AND scores.week_start = ?" if week_start is not None else ""
    params: list[Any] = [board_key]
    if week_start is not None:
        params.append(week_start)
    row = db.execute(
        f"""
        SELECT scores.score
        FROM {source_table} scores
        JOIN users u ON u.id = scores.user_id
        WHERE scores.board_key = ?
          {period_clause}
          AND u.status = 'active'
          AND TRIM(COALESCE(u.display_name, '')) <> ''
        ORDER BY scores.score DESC, scores.achieved_at ASC, scores.user_id ASC
        LIMIT 1 OFFSET 99
        """,
        params,
    ).fetchone()
    return None if row is None else int(row["score"])


def _qualifies_for_priority_submission(
    db: sqlite3.Connection,
    *,
    user_id: int,
    board_key: str,
    score: int,
    submitted_at: datetime,
) -> bool:
    personal_best = db.execute(
        "SELECT score FROM gamer_high_scores WHERE user_id = ? AND board_key = ?",
        (user_id, board_key),
    ).fetchone()
    if personal_best is None or score > int(personal_best["score"]):
        return True

    all_time_cutoff = _top_100_cutoff(db, board_key=board_key)
    if all_time_cutoff is None or score > all_time_cutoff:
        return True

    weekly_cutoff = _top_100_cutoff(
        db,
        board_key=board_key,
        week_start=_week_start_iso(submitted_at),
    )
    return weekly_cutoff is None or score > weekly_cutoff


def submit_ranked_run(
    *,
    run_id: str,
    user_id: int,
    score: int,
    final_board_codes: list[int],
    record_encoding: str,
    ip_address: str,
    lease_token: str,
) -> dict[str, Any]:
    if not isinstance(score, int) or score < 0 or score > 2**63 - 1:
        raise ValueError("invalid_score")
    if (
        len(final_board_codes) != 16
        or any(not isinstance(value, int) or value < 0 or value > 31 for value in final_board_codes)
    ):
        raise ValueError("invalid_final_board")
    record_size = _decoded_record_size(record_encoding)
    if record_size > MAX_RECORD_BYTES:
        raise ValueError("record_too_large")

    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        # Timestamp acceptance under the same lock as weekly settlement.
        now = _utc_now()
        cutoff = _iso(now - timedelta(days=1))
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if int(row["user_id"]) != user_id:
            raise PermissionError("run_owner_mismatch")
        if row["status"] != "active":
            return _public_run(row)
        if not _lease_matches(row, lease_token):
            raise PermissionError("lease_mismatch")
        if datetime.fromisoformat(str(row["expires_at"])) <= now:
            db.execute(
                "UPDATE gamer_ranked_runs SET status = 'expired', completed_at = ? WHERE run_id = ?",
                (_iso(now), run_id),
            )
            row = db.execute("SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)).fetchone()
            return _public_run(row)
        try:
            candidate_board_key = classify_ranked_candidate(
                seed_hex=str(row["seed_hex"]),
                rules_version=int(row["rules_version"]),
                record_encoding=record_encoding,
            )
        except RankedValidationError:
            candidate_board_key = ""
        priority_submission = bool(candidate_board_key) and _qualifies_for_priority_submission(
            db,
            user_id=user_id,
            board_key=candidate_board_key,
            score=score,
            submitted_at=now,
        )
        user_pending = db.execute(
            """
            SELECT COUNT(*) AS count FROM gamer_ranked_runs
            WHERE user_id = ? AND status IN ('pending', 'validating')
            """,
            (user_id,),
        ).fetchone()["count"]
        if int(user_pending) >= 1:
            raise RuntimeError("user_pending_limit")
        pending_total = db.execute(
            "SELECT COUNT(*) AS count FROM gamer_ranked_runs WHERE status IN ('pending', 'validating')"
        ).fetchone()["count"]
        if int(pending_total) >= MAX_PENDING_GLOBAL:
            raise RuntimeError("queue_full")
        user_daily = db.execute(
            "SELECT COUNT(*) AS count FROM gamer_ranked_runs WHERE user_id = ? AND submitted_at >= ?",
            (user_id, cutoff),
        ).fetchone()["count"]
        ip_daily = db.execute(
            "SELECT COUNT(*) AS count FROM gamer_ranked_runs WHERE submit_ip = ? AND submitted_at >= ?",
            (ip_address, cutoff),
        ).fetchone()["count"]
        if not priority_submission:
            if int(user_daily) >= MAX_DAILY_SUBMISSIONS_USER:
                raise RuntimeError("user_daily_limit")
            if ip_address and int(ip_daily) >= MAX_DAILY_SUBMISSIONS_IP:
                raise RuntimeError("ip_daily_limit")
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'pending', submitted_at = ?, pending_record = ?,
                claimed_score = ?, claimed_final_board = ?, submit_ip = ?, error_code = NULL,
                lease_token_hash = NULL, lease_expires_at = NULL
            WHERE run_id = ? AND status = 'active'
            """,
            (
                _iso(now),
                record_encoding.strip(),
                score,
                json.dumps(final_board_codes, separators=(",", ":")),
                ip_address,
                run_id,
            ),
        )
        row = db.execute("SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _public_run(row)


def get_ranked_run(*, run_id: str, user_id: int) -> dict[str, Any]:
    now = _utc_now()
    with auth_db() as db:
        row = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE run_id = ? AND user_id = ?",
            (run_id, user_id),
        ).fetchone()
        if row is None:
            raise LookupError("run_not_found")
        if row["status"] == "active" and datetime.fromisoformat(str(row["expires_at"])) <= now:
            db.execute(
                "UPDATE gamer_ranked_runs SET status = 'expired', completed_at = ? WHERE run_id = ?",
                (_iso(now), run_id),
            )
            row = db.execute("SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _public_run(row)


def prepare_validation_queue() -> None:
    with auth_db() as db:
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'pending', validation_started_at = NULL
            WHERE status = 'validating'
            """
        )


def cleanup_stale_ranked_runs() -> int:
    prune_ranked_replays(refresh_leaderboards=True)
    now = _utc_now()
    cutoff = _iso(now - timedelta(days=30))
    with auth_db() as db:
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'expired', completed_at = ?, pending_record = NULL
            WHERE status = 'active' AND expires_at <= ?
            """,
            (_iso(now), _iso(now)),
        )
        return db.execute(
            """
            DELETE FROM gamer_ranked_runs
            WHERE run_id NOT IN (SELECT run_id FROM gamer_high_scores)
              AND run_id NOT IN (SELECT run_id FROM gamer_weekly_high_scores)
              AND (
                (completed_at IS NOT NULL AND completed_at < ?)
                OR (status = 'active' AND started_at < ?)
              )
            """,
            (cutoff, cutoff),
        ).rowcount


def _configured_replay_retention() -> int:
    try:
        configured = int(os.environ.get(REPLAY_RETENTION_ENV, DEFAULT_REPLAY_RETENTION_PER_BOARD))
    except (TypeError, ValueError):
        configured = DEFAULT_REPLAY_RETENTION_PER_BOARD
    return max(100, min(configured, 1000))


def prune_ranked_replays(
    *,
    per_board_limit: int | None = None,
    refresh_leaderboards: bool = False,
) -> int:
    limit = _configured_replay_retention() if per_board_limit is None else max(1, int(per_board_limit))
    removed = 0
    changed_boards: set[str] = set()
    current_week = _week_start_iso()
    previous_week = (
        datetime.fromisoformat(current_week) - timedelta(days=7)
    ).isoformat()
    with auth_db() as db:
        for board_key in RANKED_BOARD_KEYS:
            stale = db.execute(
                """
                SELECT user_id
                FROM gamer_high_scores
                WHERE board_key = ?
                ORDER BY score DESC, achieved_at ASC, user_id ASC
                LIMIT -1 OFFSET ?
                """,
                (board_key, limit),
            ).fetchall()
            if not stale:
                continue
            user_ids = [int(row["user_id"]) for row in stale]
            placeholders = ",".join("?" for _ in user_ids)
            removed += db.execute(
                f"DELETE FROM gamer_high_scores WHERE board_key = ? AND user_id IN ({placeholders})",
                (board_key, *user_ids),
            ).rowcount
            changed_boards.add(board_key)

        removed += db.execute(
            """DELETE FROM gamer_weekly_high_scores WHERE week_start < ?
              AND (week_start < (SELECT value FROM token_reward_state WHERE key='weekly_start')
                   OR week_start IN (SELECT week_start FROM token_weekly_settlements))""",
            (previous_week,),
        ).rowcount
        periods = db.execute(
            """
            SELECT DISTINCT board_key, week_start
            FROM gamer_weekly_high_scores
            """
        ).fetchall()
        for period in periods:
            stale = db.execute(
                """
                SELECT rowid
                FROM gamer_weekly_high_scores
                WHERE board_key = ? AND week_start = ?
                ORDER BY score DESC, achieved_at ASC, user_id ASC
                LIMIT -1 OFFSET ?
                """,
                (period["board_key"], period["week_start"], limit),
            ).fetchall()
            if not stale:
                continue
            row_ids = [int(row["rowid"]) for row in stale]
            placeholders = ",".join("?" for _ in row_ids)
            removed += db.execute(
                f"DELETE FROM gamer_weekly_high_scores WHERE rowid IN ({placeholders})",
                row_ids,
            ).rowcount
            if str(period["week_start"]) == current_week:
                changed_boards.add(WEEKLY_BOARD_KEYS[str(period["board_key"])])
    if refresh_leaderboards and changed_boards:
        from backend.leaderboards.service import refresh_leaderboard

        for board_key in sorted(changed_boards):
            refresh_leaderboard(board_key, force=True)
    return removed


def _claim_pending() -> dict[str, Any] | None:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            """
            SELECT * FROM gamer_ranked_runs
            WHERE status = 'pending'
            ORDER BY submitted_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        now = _iso(_utc_now())
        changed = db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'validating', validation_started_at = ?
            WHERE run_id = ? AND status = 'pending'
            """,
            (now, row["run_id"]),
        ).rowcount
        if changed != 1:
            return None
        return dict(row)


def _finish_rejected(run_id: str, code: str) -> None:
    now = _iso(_utc_now())
    with auth_db() as db:
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'rejected', error_code = ?, completed_at = ?, pending_record = NULL
            WHERE run_id = ?
            """,
            (str(code)[:64], now, run_id),
        )


def _store_validated(run: dict[str, Any], game: ValidatedGame) -> bool:
    now = _iso(_utc_now())
    replay_id = str(uuid.uuid4())
    final_board = json.dumps(game.final_board_codes, separators=(",", ":"))
    achieved_at = str(run["submitted_at"] or now)
    week_start = _week_start_iso(achieved_at)
    improved_all_time = False
    improved_weekly = False
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT score FROM gamer_high_scores WHERE user_id = ? AND board_key = ?",
            (int(run["user_id"]), game.board_key),
        ).fetchone()
        if existing is None or game.score > int(existing["score"]):
            db.execute(
                """
                INSERT INTO gamer_high_scores
                (user_id, board_key, score, max_tile, move_count, used_ai,
                 final_board, record_blob, replay_id, run_id, achieved_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, board_key) DO UPDATE SET
                  score = excluded.score,
                  max_tile = excluded.max_tile,
                  move_count = excluded.move_count,
                  used_ai = excluded.used_ai,
                  final_board = excluded.final_board,
                  record_blob = excluded.record_blob,
                  replay_id = excluded.replay_id,
                  run_id = excluded.run_id,
                  achieved_at = excluded.achieved_at,
                  updated_at = excluded.updated_at
                """,
                (
                    int(run["user_id"]), game.board_key, game.score, game.max_tile,
                    game.move_count, int(game.used_ai), final_board,
                    str(run["pending_record"]), replay_id, str(run["run_id"]),
                    achieved_at, now,
                ),
            )
            improved_all_time = True

        weekly_existing = db.execute(
            """
            SELECT score
            FROM gamer_weekly_high_scores
            WHERE user_id = ? AND board_key = ? AND week_start = ?
            """,
            (int(run["user_id"]), game.board_key, week_start),
        ).fetchone()
        if weekly_existing is None or game.score > int(weekly_existing["score"]):
            db.execute(
                """
                INSERT INTO gamer_weekly_high_scores
                (user_id, board_key, week_start, score, max_tile, move_count, used_ai,
                 final_board, record_blob, replay_id, run_id, achieved_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, board_key, week_start) DO UPDATE SET
                  score = excluded.score,
                  max_tile = excluded.max_tile,
                  move_count = excluded.move_count,
                  used_ai = excluded.used_ai,
                  final_board = excluded.final_board,
                  record_blob = excluded.record_blob,
                  replay_id = excluded.replay_id,
                  run_id = excluded.run_id,
                  achieved_at = excluded.achieved_at,
                  updated_at = excluded.updated_at
                """,
                (
                    int(run["user_id"]), game.board_key, week_start, game.score,
                    game.max_tile, game.move_count, int(game.used_ai), final_board,
                    str(run["pending_record"]), replay_id, str(run["run_id"]),
                    achieved_at, now,
                ),
            )
            improved_weekly = True

        improved = improved_all_time or improved_weekly
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = ?, completed_at = ?, pending_record = NULL,
                board_key = ?, new_personal_best = ?, error_code = NULL
            WHERE run_id = ?
            """,
            (
                "verified" if improved else "no_improvement",
                now,
                game.board_key,
                int(improved_all_time),
                str(run["run_id"]),
            ),
        )
    if improved:
        from backend.leaderboards.service import refresh_leaderboard

        prune_ranked_replays()
        if improved_all_time:
            refresh_leaderboard(game.board_key, force=True)
        if improved_weekly:
            refresh_leaderboard(WEEKLY_BOARD_KEYS[game.board_key], force=True)
    return improved


def process_one_pending_run() -> bool:
    run = _claim_pending()
    if run is None:
        return False
    try:
        game = validate_ranked_game(
            seed_hex=str(run["seed_hex"]),
            rules_version=int(run["rules_version"]),
            record_encoding=str(run["pending_record"] or ""),
            claimed_score=int(run["claimed_score"]),
            claimed_final_board=json.loads(str(run["claimed_final_board"])),
            spawn_rate4=int(run["spawn_rate4_millis"]) / 1000.0,
        )
        _store_validated(run, game)
    except RankedValidationError as exc:
        _finish_rejected(str(run["run_id"]), exc.code)
    except Exception:
        _finish_rejected(str(run["run_id"]), "validation_failed")
    return True


def public_replay(replay_id: str) -> dict[str, Any]:
    with auth_db() as db:
        row = db.execute(
            """
            SELECT ghs.record_blob, ghs.score, ghs.max_tile, ghs.move_count,
                   ghs.used_ai, ghs.board_key, u.display_name
            FROM gamer_high_scores ghs
            JOIN users u ON u.id = ghs.user_id
            WHERE ghs.replay_id = ? AND u.status = 'active'
            UNION ALL
            SELECT gws.record_blob, gws.score, gws.max_tile, gws.move_count,
                   gws.used_ai, gws.board_key, u.display_name
            FROM gamer_weekly_high_scores gws
            JOIN users u ON u.id = gws.user_id
            WHERE gws.replay_id = ? AND u.status = 'active'
            LIMIT 1
            """,
            (replay_id, replay_id),
        ).fetchone()
    if row is None:
        raise LookupError("replay_not_found")
    return dict(row)
