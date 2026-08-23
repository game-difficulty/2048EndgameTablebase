from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
import json
import os
import secrets
import sqlite3
import uuid
from typing import Any

from backend.auth.db import auth_db
from backend.replay_2048next import RANKED_RULES_VERSION, REPLAY_PREFIX

from .validator import RankedValidationError, ValidatedGame, validate_ranked_game


RUN_LIFETIME = timedelta(days=30)
MAX_RECORD_BYTES = 500 * 1024
MAX_PENDING_GLOBAL = 32
MAX_DAILY_SUBMISSIONS_USER = 5
MAX_DAILY_SUBMISSIONS_IP = 20
MAX_ACTIVE_RUNS_USER = 20
DEFAULT_REPLAY_RETENTION_PER_BOARD = 100
REPLAY_RETENTION_ENV = "GAMER_REPLAY_RETENTION_PER_BOARD"
RANKED_BOARD_KEYS = ("gamer_high_score", "gamer_adversarial")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _seed_hex() -> str:
    while True:
        words = [secrets.randbits(32) for _ in range(4)]
        if all(words):
            return "".join(f"{word:08x}" for word in words)


def _public_run(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"]),
        "status": str(row["status"]),
        "rules_version": int(row["rules_version"]),
        "seed_hex": str(row["seed_hex"]),
        "started_at": row["started_at"],
        "expires_at": row["expires_at"],
        "submitted_at": row["submitted_at"],
        "completed_at": row["completed_at"],
        "board_key": row["board_key"],
        "new_personal_best": bool(row["new_personal_best"]),
        "error_code": row["error_code"],
    }


def create_ranked_run(*, user_id: int, request_id: str, ip_address: str) -> dict[str, Any]:
    normalized_request = str(request_id or "").strip()
    if not normalized_request or len(normalized_request) > 128:
        raise ValueError("invalid_request_id")
    now = _utc_now()
    expires = now + RUN_LIFETIME
    with auth_db() as db:
        existing = db.execute(
            "SELECT * FROM gamer_ranked_runs WHERE user_id = ? AND request_id = ?",
            (user_id, normalized_request),
        ).fetchone()
        if existing is not None:
            return _public_run(existing)
        active_rows = db.execute(
            """
            SELECT run_id FROM gamer_ranked_runs
            WHERE user_id = ? AND status = 'active'
            ORDER BY started_at ASC
            """,
            (user_id,),
        ).fetchall()
        if len(active_rows) >= MAX_ACTIVE_RUNS_USER:
            stale_ids = [row["run_id"] for row in active_rows[: len(active_rows) - MAX_ACTIVE_RUNS_USER + 1]]
            placeholders = ",".join("?" for _ in stale_ids)
            db.execute(
                f"UPDATE gamer_ranked_runs SET status = 'expired', completed_at = ? WHERE run_id IN ({placeholders})",
                (_iso(now), *stale_ids),
            )
        run_id = str(uuid.uuid4())
        seed_hex = _seed_hex()
        db.execute(
            """
            INSERT INTO gamer_ranked_runs
            (run_id, user_id, request_id, seed_hex, rules_version, status,
             started_at, expires_at, start_ip)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                run_id,
                user_id,
                normalized_request,
                seed_hex,
                RANKED_RULES_VERSION,
                _iso(now),
                _iso(expires),
                ip_address,
            ),
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


def submit_ranked_run(
    *,
    run_id: str,
    user_id: int,
    score: int,
    final_board_codes: list[int],
    record_encoding: str,
    ip_address: str,
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

    now = _utc_now()
    cutoff = _iso(now - timedelta(days=1))
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
        if datetime.fromisoformat(str(row["expires_at"])) <= now:
            db.execute(
                "UPDATE gamer_ranked_runs SET status = 'expired', completed_at = ? WHERE run_id = ?",
                (_iso(now), run_id),
            )
            row = db.execute("SELECT * FROM gamer_ranked_runs WHERE run_id = ?", (run_id,)).fetchone()
            return _public_run(row)
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
        if int(user_daily) >= MAX_DAILY_SUBMISSIONS_USER:
            raise RuntimeError("user_daily_limit")
        if ip_address and int(ip_daily) >= MAX_DAILY_SUBMISSIONS_IP:
            raise RuntimeError("ip_daily_limit")
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'pending', submitted_at = ?, pending_record = ?,
                claimed_score = ?, claimed_final_board = ?, submit_ip = ?, error_code = NULL
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
    changed_boards: list[str] = []
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
            changed_boards.append(board_key)
    if refresh_leaderboards and changed_boards:
        from backend.leaderboards.service import refresh_leaderboard

        for board_key in changed_boards:
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
    improved = False
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
                    str(run["submitted_at"] or now), now,
                ),
            )
            improved = True
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
                int(improved),
                str(run["run_id"]),
            ),
        )
    if improved:
        from backend.leaderboards.service import refresh_leaderboard

        prune_ranked_replays()
        refresh_leaderboard(game.board_key, force=True)
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
            """,
            (replay_id,),
        ).fetchone()
    if row is None:
        raise LookupError("replay_not_found")
    return dict(row)
