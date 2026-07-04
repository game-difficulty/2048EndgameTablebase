from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.db import auth_db

from .config import (
    TOKEN_UNIT,
    apply_multiplier,
    operation_cost_units,
    table_multiplier_units,
    token_to_units,
)
from .errors import InsufficientTokens


WEEKLY_GRANT_UNITS = token_to_units(1000)
BONUS_BALANCE_CAP_UNITS = token_to_units(2000)
WEEKLY_GRANT_INTERVAL = timedelta(days=7)


@dataclass(frozen=True)
class TokenReservation:
    ledger_id: int
    user_id: int
    session_id: int | None
    operation_key: str
    table_pattern: str
    table_multiplier_units: int
    base_cost_units: int
    reserved_units: int
    reserved_bonus_units: int
    reserved_paid_units: int


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _maybe_connection(db: sqlite3.Connection | None):
    return auth_db() if db is None else _ExistingConnection(db)


class _ExistingConnection:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        return self.db

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _ensure_token_account(db: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    now = iso()
    db.execute(
        """
        INSERT OR IGNORE INTO token_accounts
        (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at)
        VALUES (?, 0, 0, ?, ?)
        """,
        (int(user_id), now, now),
    )
    return db.execute(
        "SELECT * FROM token_accounts WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()


def _balance_units(row: sqlite3.Row) -> int:
    return int(row["bonus_balance_units"]) + int(row["paid_balance_units"])


def _public_balance(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "bonus": int(row["bonus_balance_units"]) / TOKEN_UNIT,
        "paid": int(row["paid_balance_units"]) / TOKEN_UNIT,
        "total": _balance_units(row) / TOKEN_UNIT,
        "last_weekly_grant_at": row["last_weekly_grant_at"],
    }


def get_token_balance(user_id: int, *, db: sqlite3.Connection | None = None) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = _ensure_token_account(connection, int(user_id))
        return _public_balance(row)


def grant_weekly_tokens_if_due(user_id: int, *, db: sqlite3.Connection | None = None) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = _ensure_token_account(connection, int(user_id))
        now_dt = utcnow()
        last_grant = parse_iso(row["last_weekly_grant_at"])
        if last_grant is not None and now_dt - last_grant < WEEKLY_GRANT_INTERVAL:
            return _public_balance(row)

        before = _balance_units(row)
        current_bonus = int(row["bonus_balance_units"])
        grant_units = max(0, min(WEEKLY_GRANT_UNITS, BONUS_BALANCE_CAP_UNITS - current_bonus))
        if grant_units > 0:
            connection.execute(
                """
                UPDATE token_accounts
                SET bonus_balance_units = bonus_balance_units + ?,
                    last_weekly_grant_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (grant_units, iso(now_dt), iso(now_dt), int(user_id)),
            )
        else:
            connection.execute(
                """
                UPDATE token_accounts
                SET last_weekly_grant_at = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (iso(now_dt), iso(now_dt), int(user_id)),
            )
        after_row = _ensure_token_account(connection, int(user_id))
        _insert_ledger(
            connection,
            user_id=int(user_id),
            session_id=None,
            event_type="weekly_grant",
            operation_key="weekly_grant",
            table_pattern="",
            table_multiplier_units=1000,
            base_cost_units=-grant_units,
            final_cost_units=-grant_units,
            bonus_delta_units=grant_units,
            paid_delta_units=0,
            balance_before_units=before,
            balance_after_units=_balance_units(after_row),
            metadata={"cap_units": BONUS_BALANCE_CAP_UNITS},
        )
        return _public_balance(after_row)


def _insert_ledger(
    db: sqlite3.Connection,
    *,
    user_id: int,
    session_id: int | None,
    event_type: str,
    operation_key: str,
    table_pattern: str,
    table_multiplier_units: int,
    base_cost_units: int,
    final_cost_units: int,
    bonus_delta_units: int,
    paid_delta_units: int,
    balance_before_units: int,
    balance_after_units: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    cursor = db.execute(
        """
        INSERT INTO token_ledger
        (user_id, session_id, event_type, operation_key, table_pattern,
         table_multiplier_units, base_cost_units, final_cost_units,
         bonus_delta_units, paid_delta_units, balance_before_units,
         balance_after_units, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            session_id,
            event_type,
            operation_key,
            table_pattern,
            int(table_multiplier_units),
            int(base_cost_units),
            int(final_cost_units),
            int(bonus_delta_units),
            int(paid_delta_units),
            int(balance_before_units),
            int(balance_after_units),
            json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
            iso(),
        ),
    )
    return int(cursor.lastrowid)


def _subtract_from_account(
    db: sqlite3.Connection,
    *,
    user_id: int,
    required_units: int,
) -> tuple[sqlite3.Row, sqlite3.Row, int, int]:
    row = _ensure_token_account(db, int(user_id))
    required = max(0, int(required_units))
    balance = _balance_units(row)
    if balance < required:
        raise InsufficientTokens(required_units=required, balance_units=balance)

    bonus_before = int(row["bonus_balance_units"])
    paid_before = int(row["paid_balance_units"])
    bonus_spent = min(bonus_before, required)
    paid_spent = required - bonus_spent
    if paid_spent > paid_before:
        raise InsufficientTokens(required_units=required, balance_units=balance)

    db.execute(
        """
        UPDATE token_accounts
        SET bonus_balance_units = bonus_balance_units - ?,
            paid_balance_units = paid_balance_units - ?,
            updated_at = ?
        WHERE user_id = ?
        """,
        (bonus_spent, paid_spent, iso(), int(user_id)),
    )
    after = _ensure_token_account(db, int(user_id))
    return row, after, bonus_spent, paid_spent


def reserve_operation_tokens(
    *,
    user_id: int | None,
    session_id: int | None,
    operation_key: str,
    full_pattern: str | None = "",
) -> TokenReservation | None:
    if user_id is None:
        return None
    base_units = operation_cost_units(operation_key)
    multiplier_units = table_multiplier_units(full_pattern)
    reserve_units = apply_multiplier(base_units, multiplier_units)
    if reserve_units <= 0:
        return None
    with auth_db() as db:
        before, after, bonus_spent, paid_spent = _subtract_from_account(
            db,
            user_id=int(user_id),
            required_units=reserve_units,
        )
        ledger_id = _insert_ledger(
            db,
            user_id=int(user_id),
            session_id=session_id,
            event_type="reserve",
            operation_key=operation_key,
            table_pattern=str(full_pattern or ""),
            table_multiplier_units=multiplier_units,
            base_cost_units=base_units,
            final_cost_units=reserve_units,
            bonus_delta_units=-bonus_spent,
            paid_delta_units=-paid_spent,
            balance_before_units=_balance_units(before),
            balance_after_units=_balance_units(after),
            metadata={"reserved_bonus_units": bonus_spent, "reserved_paid_units": paid_spent},
        )
    return TokenReservation(
        ledger_id=ledger_id,
        user_id=int(user_id),
        session_id=session_id,
        operation_key=operation_key,
        table_pattern=str(full_pattern or ""),
        table_multiplier_units=multiplier_units,
        base_cost_units=base_units,
        reserved_units=reserve_units,
        reserved_bonus_units=bonus_spent,
        reserved_paid_units=paid_spent,
    )


def finalize_reservation(reservation: TokenReservation | None, *, actual_operation_key: str, metadata: dict[str, Any] | None = None) -> None:
    if reservation is None:
        return
    actual_base_units = operation_cost_units(actual_operation_key)
    actual_units = min(
        reservation.reserved_units,
        apply_multiplier(actual_base_units, reservation.table_multiplier_units),
    )
    actual_bonus = min(reservation.reserved_bonus_units, actual_units)
    actual_paid = min(reservation.reserved_paid_units, actual_units - actual_bonus)
    refund_bonus = reservation.reserved_bonus_units - actual_bonus
    refund_paid = reservation.reserved_paid_units - actual_paid
    with auth_db() as db:
        before = _ensure_token_account(db, reservation.user_id)
        if refund_bonus or refund_paid:
            db.execute(
                """
                UPDATE token_accounts
                SET bonus_balance_units = bonus_balance_units + ?,
                    paid_balance_units = paid_balance_units + ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (refund_bonus, refund_paid, iso(), reservation.user_id),
            )
        after = _ensure_token_account(db, reservation.user_id)
        _insert_ledger(
            db,
            user_id=reservation.user_id,
            session_id=reservation.session_id,
            event_type="finalize",
            operation_key=actual_operation_key,
            table_pattern=reservation.table_pattern,
            table_multiplier_units=reservation.table_multiplier_units,
            base_cost_units=actual_base_units,
            final_cost_units=actual_units,
            bonus_delta_units=refund_bonus,
            paid_delta_units=refund_paid,
            balance_before_units=_balance_units(before),
            balance_after_units=_balance_units(after),
            metadata={
                "reservation_id": reservation.ledger_id,
                "reserved_units": reservation.reserved_units,
                **(metadata or {}),
            },
        )


def consume_operation_tokens(
    *,
    user_id: int | None,
    session_id: int | None,
    operation_key: str,
    full_pattern: str | None = "",
    multiplier_override_units: int | None = None,
    quantity: int = 1,
    metadata: dict[str, Any] | None = None,
) -> None:
    if user_id is None:
        return
    base_units = operation_cost_units(operation_key) * max(1, int(quantity))
    multiplier_units = (
        int(multiplier_override_units)
        if multiplier_override_units is not None
        else table_multiplier_units(full_pattern)
    )
    cost_units = apply_multiplier(base_units, multiplier_units)
    if cost_units <= 0:
        return
    with auth_db() as db:
        before, after, bonus_spent, paid_spent = _subtract_from_account(
            db,
            user_id=int(user_id),
            required_units=cost_units,
        )
        _insert_ledger(
            db,
            user_id=int(user_id),
            session_id=session_id,
            event_type="consume",
            operation_key=operation_key,
            table_pattern=str(full_pattern or ""),
            table_multiplier_units=multiplier_units,
            base_cost_units=base_units,
            final_cost_units=cost_units,
            bonus_delta_units=-bonus_spent,
            paid_delta_units=-paid_spent,
            balance_before_units=_balance_units(before),
            balance_after_units=_balance_units(after),
            metadata=metadata,
        )


def has_numeric_result(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    for value in result.values():
        if isinstance(value, (int, float)) and value == value:
            return True
    return False
