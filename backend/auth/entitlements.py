from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .db import auth_db


DEFAULT_TIER = "free"
SUPPORTER_TIER = "supporter"
VALID_TIERS = {DEFAULT_TIER, SUPPORTER_TIER}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tier(value: str | None) -> str:
    tier = str(value or "").strip().lower()
    return tier if tier in VALID_TIERS else DEFAULT_TIER


class _ExistingConnection:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        return self.db

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@contextmanager
def _maybe_connection(db: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
    if db is None:
        with auth_db() as connection:
            yield connection
        return
    with _ExistingConnection(db) as connection:
        yield connection


def _row_value(row: sqlite3.Row | dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        if isinstance(row, sqlite3.Row) and key not in row.keys():
            return default
        return row[key]
    except (KeyError, IndexError):
        return default


def public_entitlements(row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any]:
    tier = normalize_tier(_row_value(row, "tier", DEFAULT_TIER))
    return {
        "tier": tier,
        "is_supporter": tier == SUPPORTER_TIER,
        "supporter_since": _row_value(row, "supporter_since"),
        "supporter_until": _row_value(row, "supporter_until"),
        "show_supporter_badge": bool(_row_value(row, "show_supporter_badge", 1)),
        "can_upload_avatar": bool(_row_value(row, "can_upload_avatar", 0)),
    }


def ensure_user_entitlements(
    db: sqlite3.Connection,
    user_id: int,
    *,
    tier: str = DEFAULT_TIER,
) -> sqlite3.Row:
    now = iso_now()
    normalized_tier = normalize_tier(tier)
    db.execute(
        """
        INSERT OR IGNORE INTO user_entitlements
        (user_id, tier, show_supporter_badge, can_upload_avatar, created_at, updated_at)
        VALUES (?, ?, 1, 0, ?, ?)
        """,
        (int(user_id), normalized_tier, now, now),
    )
    return db.execute(
        "SELECT * FROM user_entitlements WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()


def get_user_entitlements(
    user_id: int,
    *,
    db: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = ensure_user_entitlements(connection, int(user_id))
        return public_entitlements(row)


def mark_user_supporter(
    db: sqlite3.Connection,
    user_id: int,
    *,
    notes: str = "",
) -> dict[str, Any]:
    now = iso_now()
    row = ensure_user_entitlements(db, int(user_id))
    supporter_since = row["supporter_since"] or now
    db.execute(
        """
        UPDATE user_entitlements
        SET tier = 'supporter',
            supporter_since = ?,
            show_supporter_badge = 1,
            notes = COALESCE(NULLIF(?, ''), notes),
            updated_at = ?
        WHERE user_id = ?
        """,
        (supporter_since, str(notes or "").strip()[:240], now, int(user_id)),
    )
    updated = db.execute(
        "SELECT * FROM user_entitlements WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()
    return public_entitlements(updated)
