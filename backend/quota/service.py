from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.auth.db import auth_db
from backend.auth.entitlements import DEFAULT_TIER, SUPPORTER_TIER, mark_user_supporter

from .config import (
    TOKEN_UNIT,
    apply_multiplier,
    operation_cost_units,
    table_multiplier_units,
    token_to_units,
)
from .errors import InsufficientTokens


INVITED_WEEKLY_GRANT_UNITS = token_to_units(4096)
PUBLIC_WEEKLY_GRANT_UNITS = token_to_units(512)
SUPPORTER_WEEKLY_GRANT_UNITS = token_to_units(32768)
WEEKLY_GRANT_INTERVAL = timedelta(days=7)
MAX_ADMIN_TOKEN_ADJUSTMENT = 100_000_000


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


def _current_entitlement_tier(db: sqlite3.Connection, user_id: int) -> str:
    row = db.execute(
        "SELECT COALESCE(tier, ?) AS tier FROM user_entitlements WHERE user_id = ?",
        (DEFAULT_TIER, int(user_id)),
    ).fetchone()
    return str(row["tier"] if row else DEFAULT_TIER).strip().lower() or DEFAULT_TIER


def _parse_admin_token_amount(value: Any, *, mode: str) -> tuple[int, float]:
    try:
        token_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Token amount must be numeric.") from exc
    if not math.isfinite(token_value):
        raise ValueError("Token amount must be finite.")
    if token_value < 0:
        raise ValueError("Token amount must not be negative.")
    if mode == "add_paid" and token_value <= 0:
        raise ValueError("Token amount must be greater than zero.")
    if token_value > MAX_ADMIN_TOKEN_ADJUSTMENT:
        raise ValueError("Token amount is too large.")
    units = token_to_units(token_value)
    if units < 0:
        raise ValueError("Token amount must not be negative.")
    return units, units / TOKEN_UNIT


def adjust_paid_tokens_for_admin(
    *,
    target_user_id: int,
    admin_user: dict[str, Any],
    mode: str,
    tokens: Any,
    reason: str = "",
    payment_amount_cny: Any = None,
    payment_channel: str = "",
    set_supporter: bool = False,
) -> dict[str, Any]:
    normalized_mode = str(mode or "").strip()
    if normalized_mode not in {"add_paid", "set_paid"}:
        raise ValueError("Invalid token adjustment mode.")
    reason_value = str(reason or "").strip()
    if not reason_value:
        raise ValueError("Reason is required.")
    if len(reason_value) > 240:
        raise ValueError("Reason is too long.")

    token_units, token_value = _parse_admin_token_amount(tokens, mode=normalized_mode)
    admin_user_id = int(admin_user.get("id") or 0)
    admin_email = str(admin_user.get("email") or "")
    admin_display_name = str(admin_user.get("display_name") or "")
    payment_channel_value = str(payment_channel or "").strip()[:40]
    payment_amount_value = None
    if payment_amount_cny not in (None, ""):
        try:
            payment_amount_value = round(float(payment_amount_cny), 2)
        except (TypeError, ValueError) as exc:
            raise ValueError("Payment amount must be numeric.") from exc
        if not math.isfinite(payment_amount_value) or payment_amount_value < 0:
            raise ValueError("Payment amount must not be negative.")

    with auth_db() as db:
        target_user = db.execute(
            "SELECT id, email, display_name, status FROM users WHERE id = ?",
            (int(target_user_id),),
        ).fetchone()
        if target_user is None:
            raise FileNotFoundError("User not found.")

        was_supporter = _current_entitlement_tier(db, int(target_user_id)) == SUPPORTER_TIER
        before = _ensure_token_account(db, int(target_user_id))
        before_paid_units = int(before["paid_balance_units"])
        before_total_units = _balance_units(before)

        if normalized_mode == "add_paid":
            new_paid_units = before_paid_units + token_units
            event_type = "admin_topup"
        else:
            new_paid_units = token_units
            event_type = "admin_set_paid_balance"

        if new_paid_units < 0:
            raise ValueError("Paid token balance must not be negative.")
        paid_delta_units = new_paid_units - before_paid_units
        db.execute(
            """
            UPDATE token_accounts
            SET paid_balance_units = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (new_paid_units, iso(), int(target_user_id)),
        )
        after_paid_adjustment = _ensure_token_account(db, int(target_user_id))
        after_paid_total_units = _balance_units(after_paid_adjustment)
        entitlements = None
        if bool(set_supporter):
            entitlements = mark_user_supporter(
                db,
                int(target_user_id),
                notes=reason_value,
            )
        ledger_id = _insert_ledger(
            db,
            user_id=int(target_user_id),
            session_id=None,
            event_type=event_type,
            operation_key=normalized_mode,
            table_pattern="",
            table_multiplier_units=1000,
            base_cost_units=-paid_delta_units if paid_delta_units > 0 else 0,
            final_cost_units=-paid_delta_units if paid_delta_units > 0 else 0,
            bonus_delta_units=0,
            paid_delta_units=paid_delta_units,
            balance_before_units=before_total_units,
            balance_after_units=after_paid_total_units,
            metadata={
                "admin_user_id": admin_user_id,
                "admin_email": admin_email,
                "admin_display_name": admin_display_name,
                "target_user_id": int(target_user_id),
                "target_email": target_user["email"],
                "target_display_name": target_user["display_name"] or "",
                "mode": normalized_mode,
                "tokens": token_value,
                "reason": reason_value,
                "payment_amount_cny": payment_amount_value,
                "payment_channel": payment_channel_value,
                "set_supporter": bool(set_supporter),
                "before_paid_tokens": before_paid_units / TOKEN_UNIT,
                "after_paid_tokens": new_paid_units / TOKEN_UNIT,
                "before_total_tokens": before_total_units / TOKEN_UNIT,
                "after_total_tokens": after_paid_total_units / TOKEN_UNIT,
            },
        )
        bonus_reset = None
        if bool(set_supporter) and not was_supporter:
            bonus_reset = reset_bonus_to_weekly_cap(
                int(target_user_id),
                db=db,
                event_type="supporter_bonus_cap_reset",
                reason=reason_value,
                metadata={
                    "admin_user_id": admin_user_id,
                    "admin_email": admin_email,
                    "admin_display_name": admin_display_name,
                    "target_user_id": int(target_user_id),
                    "target_email": target_user["email"],
                    "target_display_name": target_user["display_name"] or "",
                    "source": "admin_set_supporter",
                },
            )
        final_row = _ensure_token_account(db, int(target_user_id))
        return {
            "ledger_id": ledger_id,
            "token_balance": _public_balance(final_row),
            "paid_delta": paid_delta_units / TOKEN_UNIT,
            "before_paid": before_paid_units / TOKEN_UNIT,
            "after_paid": new_paid_units / TOKEN_UNIT,
            "entitlements": entitlements,
            "bonus_reset": bonus_reset,
        }


def _weekly_grant_for_user(db: sqlite3.Connection, user_id: int) -> tuple[int, str]:
    row = db.execute(
        """
        SELECT
          users.registered_with_invite,
          COALESCE(user_entitlements.tier, 'free') AS entitlement_tier
        FROM users
        LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
        WHERE users.id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if row is not None and str(row["entitlement_tier"] or "").strip().lower() == SUPPORTER_TIER:
        return SUPPORTER_WEEKLY_GRANT_UNITS, SUPPORTER_TIER
    if row is None or int(row["registered_with_invite"] or 0):
        return INVITED_WEEKLY_GRANT_UNITS, "invite"
    return PUBLIC_WEEKLY_GRANT_UNITS, "public"


def grant_weekly_tokens_if_due(user_id: int, *, db: sqlite3.Connection | None = None) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = _ensure_token_account(connection, int(user_id))
        weekly_grant_units, grant_tier = _weekly_grant_for_user(connection, int(user_id))
        bonus_balance_cap_units = weekly_grant_units
        now_dt = utcnow()
        last_grant = parse_iso(row["last_weekly_grant_at"])
        if last_grant is not None and now_dt - last_grant < WEEKLY_GRANT_INTERVAL:
            return _public_balance(row)

        before = _balance_units(row)
        current_bonus = int(row["bonus_balance_units"])
        grant_units = max(0, min(weekly_grant_units, bonus_balance_cap_units - current_bonus))
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
            metadata={
                "cap_units": bonus_balance_cap_units,
                "grant_tier": grant_tier,
                "weekly_grant_units": weekly_grant_units,
            },
        )
        return _public_balance(after_row)


def reset_bonus_to_weekly_cap(
    user_id: int,
    *,
    db: sqlite3.Connection | None = None,
    event_type: str = "admin_bonus_cap_reset",
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = _ensure_token_account(connection, int(user_id))
        cap_units, grant_tier = _weekly_grant_for_user(connection, int(user_id))
        before_total_units = _balance_units(row)
        before_bonus_units = int(row["bonus_balance_units"])
        now = iso()
        connection.execute(
            """
            UPDATE token_accounts
            SET bonus_balance_units = ?,
                last_weekly_grant_at = ?,
                updated_at = ?
            WHERE user_id = ?
            """,
            (cap_units, now, now, int(user_id)),
        )
        after_row = _ensure_token_account(connection, int(user_id))
        bonus_delta_units = cap_units - before_bonus_units
        ledger_id = None
        if bonus_delta_units != 0:
            ledger_id = _insert_ledger(
                connection,
                user_id=int(user_id),
                session_id=None,
                event_type=event_type,
                operation_key="bonus_cap_reset",
                table_pattern="",
                table_multiplier_units=1000,
                base_cost_units=-bonus_delta_units if bonus_delta_units > 0 else 0,
                final_cost_units=-bonus_delta_units if bonus_delta_units > 0 else 0,
                bonus_delta_units=bonus_delta_units,
                paid_delta_units=0,
                balance_before_units=before_total_units,
                balance_after_units=_balance_units(after_row),
                metadata={
                    **(metadata or {}),
                    "reason": str(reason or "").strip()[:240],
                    "grant_tier": grant_tier,
                    "cap_units": cap_units,
                    "before_bonus_tokens": before_bonus_units / TOKEN_UNIT,
                    "after_bonus_tokens": cap_units / TOKEN_UNIT,
                },
            )
        return {
            "user_id": int(user_id),
            "grant_tier": grant_tier,
            "cap_tokens": cap_units / TOKEN_UNIT,
            "before_bonus": before_bonus_units / TOKEN_UNIT,
            "after_bonus": cap_units / TOKEN_UNIT,
            "bonus_delta": bonus_delta_units / TOKEN_UNIT,
            "changed": bonus_delta_units != 0,
            "ledger_id": ledger_id,
            "token_balance": _public_balance(after_row),
        }


def reset_all_bonus_to_weekly_caps(*, reason: str = "Reset weekly free token balances to caps.") -> dict[str, Any]:
    with auth_db() as db:
        user_rows = db.execute("SELECT id FROM users ORDER BY id").fetchall()
        summary: dict[str, Any] = {
            "total_users": len(user_rows),
            "changed_users": 0,
            "total_bonus_delta": 0,
            "by_tier": {},
        }
        for user_row in user_rows:
            result = reset_bonus_to_weekly_cap(
                int(user_row["id"]),
                db=db,
                event_type="admin_bulk_bonus_cap_reset",
                reason=reason,
                metadata={"source": "admin_bulk_reset"},
            )
            tier = result["grant_tier"]
            tier_summary = summary["by_tier"].setdefault(
                tier,
                {"users": 0, "changed_users": 0, "bonus_delta": 0, "cap_tokens": result["cap_tokens"]},
            )
            tier_summary["users"] += 1
            if result["changed"]:
                summary["changed_users"] += 1
                tier_summary["changed_users"] += 1
            summary["total_bonus_delta"] += result["bonus_delta"]
            tier_summary["bonus_delta"] += result["bonus_delta"]
        return summary


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
