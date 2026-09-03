from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import string
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import auth_db
from .security import hash_token, new_token


GUEST_COOKIE_NAME = "tb_guest"
GUEST_TOKEN_HEADER = "x-guest-token"
GUEST_SESSION_DAYS = 30
DEFAULT_QUERY_ALLOWANCE = 5
DEFAULT_IP_QUERY_LIMIT = 15
DEFAULT_IP_WINDOW_DAYS = 30
DEFAULT_ISSUE_LIMIT_PER_DAY = 5


class GuestLimitError(RuntimeError):
    def __init__(self, code: str, message: str, *, remaining: int = 0):
        super().__init__(message)
        self.code = code
        self.remaining = max(0, int(remaining))


@dataclass(frozen=True)
class GuestQueryReservation:
    event_id: int
    guest_id: str
    request_id: str
    remaining: int
    already_finalized: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value or "")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _env_int(name: str, fallback: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(fallback))))
    except ValueError:
        return fallback


def guest_query_allowance_limit() -> int:
    return _env_int("GUEST_QUERY_ALLOWANCE", DEFAULT_QUERY_ALLOWANCE)


def guest_ip_query_limit() -> int:
    return _env_int("GUEST_IP_QUERY_LIMIT", DEFAULT_IP_QUERY_LIMIT)


def guest_ip_window_days() -> int:
    return _env_int("GUEST_IP_WINDOW_DAYS", DEFAULT_IP_WINDOW_DAYS)


def guest_session_days() -> int:
    return _env_int("GUEST_SESSION_DAYS", GUEST_SESSION_DAYS)


def cleanup_expired_guest_sessions(*, now: datetime | None = None) -> int:
    cutoff = _iso(now or _now())
    with auth_db() as db:
        cursor = db.execute(
            "DELETE FROM guest_sessions WHERE expires_at <= ?",
            (cutoff,),
        )
        return max(0, int(cursor.rowcount or 0))


def hash_ip_bucket(ip_address: str) -> str:
    secret = (
        os.getenv("GUEST_IP_HASH_SECRET")
        or os.getenv("REMOTE_TABLEBASE_WORKER_SECRET")
        or os.getenv("AUTH_SECRET")
        or "2048tables-development-guest-ip"
    ).encode("utf-8")
    normalized = str(ip_address or "unknown").strip().lower().encode("utf-8")
    return hmac.new(secret, normalized, hashlib.sha256).hexdigest()


def _guest_payload(row, *, db=None) -> dict[str, Any]:
    payload = {
        "guest_id": str(row["guest_id"]),
        "display_name": str(row["display_name"]),
        "expires_at": str(row["expires_at"]),
    }
    if db is not None:
        payload["query_allowance"] = guest_query_allowance(str(row["guest_id"]), db=db)
    return payload


def guest_query_allowance(guest_id: str, *, db=None) -> dict[str, int]:
    owns_db = db is None
    context = auth_db() if owns_db else None
    connection = context.__enter__() if context is not None else db
    try:
        used = int(
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM guest_query_events
                WHERE guest_id = ? AND status IN ('reserved', 'consumed')
                """,
                (str(guest_id),),
            ).fetchone()["count"]
        )
        limit = guest_query_allowance_limit()
        return {"remaining": max(0, limit - used), "limit": limit, "used": used}
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def issue_guest_session(*, ip_address: str = "") -> dict[str, Any]:
    now = _now()
    ip_hash = hash_ip_bucket(ip_address)
    cutoff = _iso(now - timedelta(days=1))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM guest_sessions WHERE expires_at <= ?", (_iso(now),))
        issued = int(
            db.execute(
                """
                SELECT COUNT(*) AS count FROM guest_sessions
                WHERE created_ip_hash = ? AND created_at >= ?
                """,
                (ip_hash, cutoff),
            ).fetchone()["count"]
        )
        if issued >= _env_int("GUEST_SESSION_ISSUE_LIMIT_PER_DAY", DEFAULT_ISSUE_LIMIT_PER_DAY):
            raise GuestLimitError(
                "GUEST_SESSION_RATE_LIMITED",
                "Too many guest sessions have been created from this network. Try again later.",
            )
        guest_id = str(uuid.uuid4())
        token = new_token()
        alphabet = string.ascii_uppercase + string.digits
        suffix = "".join(secrets.choice(alphabet) for _ in range(4))
        display_name = f"Guest-{suffix}"
        expires_at = now + timedelta(days=guest_session_days())
        db.execute(
            """
            INSERT INTO guest_sessions
            (guest_id, token_hash, display_name, created_ip_hash, last_ip_hash,
             created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guest_id,
                hash_token(token),
                display_name,
                ip_hash,
                ip_hash,
                _iso(now),
                _iso(now),
                _iso(expires_at),
            ),
        )
        row = db.execute(
            "SELECT * FROM guest_sessions WHERE guest_id = ?", (guest_id,)
        ).fetchone()
        guest = _guest_payload(row, db=db)
    return {"guest": guest, "token": token, "expires_at": _iso(expires_at)}


def authenticate_guest_token(
    token: str | None, *, ip_address: str = "", touch: bool = True
) -> dict[str, Any] | None:
    if not token:
        return None
    now = _now()
    with auth_db() as db:
        row = db.execute(
            "SELECT * FROM guest_sessions WHERE token_hash = ?", (hash_token(token),)
        ).fetchone()
        if row is None or row["revoked_at"]:
            return None
        expires_at = _parse_iso(row["expires_at"])
        if expires_at is None or expires_at <= now:
            return None
        last_seen = _parse_iso(row["last_seen_at"])
        if touch and (last_seen is None or last_seen <= now - timedelta(minutes=5)):
            db.execute(
                """
                UPDATE guest_sessions SET last_seen_at = ?, last_ip_hash = ?
                WHERE guest_id = ?
                """,
                (_iso(now), hash_ip_bucket(ip_address), row["guest_id"]),
            )
        return _guest_payload(row, db=db)


def revoke_guest_session(token: str | None) -> None:
    if not token:
        return
    with auth_db() as db:
        db.execute(
            "UPDATE guest_sessions SET revoked_at = ? WHERE token_hash = ?",
            (_iso(), hash_token(token)),
        )


def reserve_guest_query(
    *, guest_id: str, request_id: str, full_pattern: str, ip_address: str
) -> GuestQueryReservation:
    normalized_request_id = str(request_id or "").strip()[:160]
    if not normalized_request_id:
        raise GuestLimitError("GUEST_REQUEST_ID_REQUIRED", "A request id is required.")
    now = _now()
    ip_hash = hash_ip_bucket(ip_address)
    cutoff = _iso(now - timedelta(days=guest_ip_window_days()))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            """
            SELECT * FROM guest_query_events
            WHERE guest_id = ? AND request_id = ?
            """,
            (str(guest_id), normalized_request_id),
        ).fetchone()
        allowance = guest_query_allowance(str(guest_id), db=db)
        if existing is not None and str(existing["status"]) != "cancelled":
            return GuestQueryReservation(
                event_id=int(existing["id"]),
                guest_id=str(guest_id),
                request_id=normalized_request_id,
                remaining=allowance["remaining"],
                already_finalized=str(existing["status"]) == "consumed",
            )
        if existing is not None:
            db.execute("DELETE FROM guest_query_events WHERE id = ?", (existing["id"],))
        if allowance["remaining"] <= 0:
            raise GuestLimitError(
                "GUEST_QUERY_ALLOWANCE_EXHAUSTED",
                "Guest trial queries have been used up. Sign in or register to continue.",
            )
        network_used = int(
            db.execute(
                """
                SELECT COUNT(*) AS count FROM guest_query_events
                WHERE ip_bucket_hash = ? AND created_at >= ?
                  AND status IN ('reserved', 'consumed')
                """,
                (ip_hash, cutoff),
            ).fetchone()["count"]
        )
        if network_used >= guest_ip_query_limit():
            raise GuestLimitError(
                "GUEST_NETWORK_QUERY_LIMIT_REACHED",
                "The guest trial limit for this network has been reached. Sign in to continue.",
                remaining=allowance["remaining"],
            )
        cursor = db.execute(
            """
            INSERT INTO guest_query_events
            (guest_id, request_id, full_pattern, ip_bucket_hash, status, created_at)
            VALUES (?, ?, ?, ?, 'reserved', ?)
            """,
            (str(guest_id), normalized_request_id, str(full_pattern), ip_hash, _iso(now)),
        )
        return GuestQueryReservation(
            event_id=int(cursor.lastrowid),
            guest_id=str(guest_id),
            request_id=normalized_request_id,
            remaining=max(0, allowance["remaining"] - 1),
        )


def finalize_guest_query(
    reservation: GuestQueryReservation, *, consume: bool
) -> dict[str, int]:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT status FROM guest_query_events WHERE id = ? AND guest_id = ?",
            (reservation.event_id, reservation.guest_id),
        ).fetchone()
        if row is not None and row["status"] == "reserved":
            db.execute(
                """
                UPDATE guest_query_events SET status = ?, finalized_at = ? WHERE id = ?
                """,
                ("consumed" if consume else "cancelled", _iso(), reservation.event_id),
            )
        return guest_query_allowance(reservation.guest_id, db=db)
