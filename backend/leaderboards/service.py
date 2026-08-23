from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
import threading
from typing import Any

from backend.auth.db import auth_db


SUPPORTERS_BOARD = "supporters"
TOKEN_LIFETIME_BOARD = "token_lifetime"
TOKEN_LAST_WEEK_BOARD = "token_last_week"
LEADERBOARD_LIMIT = 100
DEFAULT_ADMIN_IDENTITIES = ("user0", "assweeass@163.com")
_REFRESH_LOCK = threading.Lock()


@dataclass(frozen=True)
class BoardDefinition:
    key: str
    cadence: str
    score_visible: bool
    unit: str | None
    retained_snapshots: int


BOARD_DEFINITIONS: tuple[BoardDefinition, ...] = (
    BoardDefinition(SUPPORTERS_BOARD, "daily", False, None, 31),
    BoardDefinition(TOKEN_LIFETIME_BOARD, "daily", True, "token", 31),
    BoardDefinition(TOKEN_LAST_WEEK_BOARD, "weekly", True, "token", 104),
)
BOARD_BY_KEY = {board.key: board for board in BOARD_DEFINITIONS}


def _leaderboard_timezone() -> timezone:
    return timezone(timedelta(hours=8), name="UTC+08:00")


def _admin_identities() -> tuple[str, ...]:
    raw = str(os.getenv("ADMIN_ALLOWED_IDENTITIES") or "")
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    return values or DEFAULT_ADMIN_IDENTITIES


def _local_midnight(value: datetime) -> datetime:
    local = value.astimezone(_leaderboard_timezone())
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _period_for(board_key: str, now: datetime | None = None) -> tuple[str | None, str]:
    current = now or datetime.now(timezone.utc)
    today = _local_midnight(current)
    if board_key == TOKEN_LAST_WEEK_BOARD:
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        return last_monday.isoformat(), this_monday.isoformat()
    return None, today.isoformat()


def _utc_iso(local_iso: str) -> str:
    return datetime.fromisoformat(local_iso).astimezone(timezone.utc).isoformat()


def _supporter_rows(db) -> list[dict[str, Any]]:
    identities = _admin_identities()
    placeholders = ",".join("?" for _ in identities)
    query = f"""
        SELECT
          u.id AS user_id,
          TRIM(u.display_name) AS display_name,
          COALESCE(ta.paid_balance_units, 0) AS score_units,
          1 AS is_supporter
        FROM users u
        JOIN user_entitlements ue ON ue.user_id = u.id AND ue.tier = 'supporter'
        LEFT JOIN token_accounts ta ON ta.user_id = u.id
        WHERE u.status = 'active'
          AND TRIM(COALESCE(u.display_name, '')) <> ''
          AND LOWER(COALESCE(u.role, 'user')) <> 'admin'
          AND LOWER(u.email) NOT IN ({placeholders})
          AND LOWER(TRIM(u.display_name)) NOT IN ({placeholders})
        ORDER BY score_units DESC, u.id ASC
        LIMIT ?
    """
    params = (*identities, *identities, LEADERBOARD_LIMIT)
    return [dict(row) for row in db.execute(query, params).fetchall()]


def _token_rows(
    db,
    *,
    period_start: str | None,
    period_end: str,
) -> list[dict[str, Any]]:
    clauses = [
        "tl.event_type IN ('finalize', 'consume')",
        "tl.final_cost_units > 0",
        "tl.created_at < ?",
        "u.status = 'active'",
        "TRIM(COALESCE(u.display_name, '')) <> ''",
    ]
    params: list[Any] = [_utc_iso(period_end)]
    if period_start:
        clauses.append("tl.created_at >= ?")
        params.append(_utc_iso(period_start))
    params.append(LEADERBOARD_LIMIT)
    query = f"""
        SELECT
          u.id AS user_id,
          TRIM(u.display_name) AS display_name,
          SUM(tl.final_cost_units) AS score_units,
          CASE WHEN ue.tier = 'supporter' THEN 1 ELSE 0 END AS is_supporter
        FROM token_ledger tl
        JOIN users u ON u.id = tl.user_id
        LEFT JOIN user_entitlements ue ON ue.user_id = u.id
        WHERE {' AND '.join(clauses)}
        GROUP BY u.id, u.display_name, ue.tier
        ORDER BY score_units DESC, u.id ASC
        LIMIT ?
    """
    return [dict(row) for row in db.execute(query, params).fetchall()]


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "rank": index} for index, row in enumerate(rows, start=1)]


def _build_rows(db, board_key: str, period_start: str | None, period_end: str) -> list[dict[str, Any]]:
    if board_key == SUPPORTERS_BOARD:
        return _rank_rows(_supporter_rows(db))
    return _rank_rows(
        _token_rows(db, period_start=period_start, period_end=period_end)
    )


def _prune_snapshots(db, definition: BoardDefinition) -> None:
    stale = db.execute(
        """
        SELECT id FROM leaderboard_snapshots
        WHERE board_key = ?
        ORDER BY generated_at DESC
        LIMIT -1 OFFSET ?
        """,
        (definition.key, definition.retained_snapshots),
    ).fetchall()
    if not stale:
        return
    ids = [int(row["id"]) for row in stale]
    placeholders = ",".join("?" for _ in ids)
    db.execute(f"DELETE FROM leaderboard_snapshots WHERE id IN ({placeholders})", ids)


def refresh_due_leaderboards(*, force: bool = False, now: datetime | None = None) -> dict[str, int]:
    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    refreshed: dict[str, int] = {}
    with _REFRESH_LOCK:
        with auth_db() as db:
            for definition in BOARD_DEFINITIONS:
                period_start, period_end = _period_for(definition.key, now)
                existing = db.execute(
                    "SELECT id FROM leaderboard_snapshots WHERE board_key = ? AND period_end = ?",
                    (definition.key, period_end),
                ).fetchone()
                if existing is not None and not force:
                    continue
                if existing is not None:
                    db.execute("DELETE FROM leaderboard_snapshots WHERE id = ?", (int(existing["id"]),))
                rows = _build_rows(db, definition.key, period_start, period_end)
                cursor = db.execute(
                    """
                    INSERT INTO leaderboard_snapshots
                    (board_key, period_start, period_end, generated_at, entry_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (definition.key, period_start, period_end, generated_at, len(rows)),
                )
                snapshot_id = int(cursor.lastrowid)
                db.executemany(
                    """
                    INSERT INTO leaderboard_entries
                    (snapshot_id, user_id, rank, score_units, display_name, is_supporter)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            snapshot_id,
                            int(row["user_id"]),
                            int(row["rank"]),
                            int(row["score_units"]),
                            str(row["display_name"]),
                            int(row.get("is_supporter") or 0),
                        )
                        for row in rows
                    ],
                )
                _prune_snapshots(db, definition)
                refreshed[definition.key] = len(rows)
    return refreshed


def leaderboard_catalog() -> list[dict[str, Any]]:
    return [
        {
            "key": definition.key,
            "cadence": definition.cadence,
            "score_visible": definition.score_visible,
            "unit": definition.unit,
        }
        for definition in BOARD_DEFINITIONS
    ]


def leaderboard_payload(board_key: str, *, limit: int = LEADERBOARD_LIMIT) -> dict[str, Any]:
    definition = BOARD_BY_KEY.get(str(board_key or ""))
    if definition is None:
        raise KeyError(board_key)
    refresh_due_leaderboards()
    with auth_db() as db:
        snapshot = db.execute(
            """
            SELECT id, board_key, period_start, period_end, generated_at, entry_count
            FROM leaderboard_snapshots
            WHERE board_key = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (definition.key,),
        ).fetchone()
        if snapshot is None:
            raise RuntimeError("Leaderboard snapshot is unavailable.")
        rows = db.execute(
            """
            SELECT
              leaderboard_entries.user_id,
              leaderboard_entries.rank,
              leaderboard_entries.score_units,
              leaderboard_entries.display_name,
              leaderboard_entries.is_supporter,
              user_profiles.avatar_key
            FROM leaderboard_entries
            LEFT JOIN user_profiles ON user_profiles.user_id = leaderboard_entries.user_id
            WHERE leaderboard_entries.snapshot_id = ?
            ORDER BY leaderboard_entries.rank ASC,
                     leaderboard_entries.score_units DESC,
                     leaderboard_entries.user_id ASC
            LIMIT ?
            """,
            (int(snapshot["id"]), max(1, min(int(limit), LEADERBOARD_LIMIT))),
        ).fetchall()

    entries = []
    for row in rows:
        entry = {
            "entry_key": hashlib.sha256(
                f"{snapshot['id']}:{row['user_id']}:leaderboard".encode("utf-8")
            ).hexdigest()[:16],
            "rank": int(row["rank"]),
            "display_name": str(row["display_name"]),
            "is_supporter": bool(row["is_supporter"]),
            "avatar_url": (
                f"/media/avatars/{row['avatar_key']}" if row["avatar_key"] else None
            ),
        }
        if definition.score_visible:
            entry["score"] = round(int(row["score_units"]) / 1000, 3)
        entries.append(entry)
    return {
        "key": definition.key,
        "cadence": definition.cadence,
        "score_visible": definition.score_visible,
        "unit": definition.unit,
        "period": {
            "start": snapshot["period_start"],
            "end": snapshot["period_end"],
        },
        "generated_at": snapshot["generated_at"],
        "entry_count": int(snapshot["entry_count"]),
        "entries": entries,
    }
