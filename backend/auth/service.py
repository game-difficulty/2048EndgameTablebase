from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import auth_db
from .mailer import send_verification_email
from .security import constant_time_equal, hash_password, hash_token, new_token, verify_password
from backend.quota.service import get_token_balance, grant_weekly_tokens_if_due


SESSION_COOKIE_NAME = "tb_session"
DEFAULT_SESSION_DAYS = 14
EMAIL_CODE_MINUTES = 10
DEFAULT_QUOTA_KEYS = (
    "trainer_query",
    "tester_move",
    "analysis_job",
    "upload_analysis",
    "upload_replay",
    "replay_load",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def public_user(
    row: sqlite3.Row | dict[str, Any],
    *,
    db: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    user_id = int(row["id"])
    return {
        "id": user_id,
        "email": row["email"],
        "display_name": row["display_name"] or "",
        "role": row["role"],
        "status": row["status"],
        "email_verified": bool(row["email_verified_at"]),
        "token_balance": get_token_balance(user_id, db=db),
    }


def session_days() -> int:
    try:
        return max(1, int(os.getenv("AUTH_SESSION_DAYS", str(DEFAULT_SESSION_DAYS))))
    except ValueError:
        return DEFAULT_SESSION_DAYS


def _invite_row(db: sqlite3.Connection, invite_code: str) -> sqlite3.Row | None:
    return db.execute(
        "SELECT * FROM invite_codes WHERE code_hash = ?",
        (hash_token(invite_code.strip()),),
    ).fetchone()


def _validate_invite(
    db: sqlite3.Connection,
    invite_code: str,
    email: str,
) -> sqlite3.Row:
    row = _invite_row(db, invite_code)
    if row is None:
        raise ValueError("Invalid invite code.")
    if row["disabled_at"]:
        raise ValueError("Invite code is disabled.")
    expires_at = parse_iso(row["expires_at"])
    if expires_at is not None and expires_at <= utcnow():
        raise ValueError("Invite code has expired.")
    if int(row["used_count"]) >= int(row["max_uses"]):
        raise ValueError("Invite code has already been used.")
    allowed_email = normalize_email(row["allowed_email"] or "")
    if allowed_email and allowed_email != email:
        raise ValueError("Invite code is not valid for this email.")
    allowed_domain = str(row["allowed_domain"] or "").strip().lower().lstrip("@")
    if allowed_domain and not email.endswith(f"@{allowed_domain}"):
        raise ValueError("Invite code is not valid for this email domain.")
    return row


def create_default_quotas(db: sqlite3.Connection, user_id: int) -> None:
    now = iso()
    for quota_key in DEFAULT_QUOTA_KEYS:
        db.execute(
            """
            INSERT OR IGNORE INTO user_quotas
            (user_id, quota_key, period, limit_value, used_value, created_at, updated_at)
            VALUES (?, ?, 'lifetime', 0, 0, ?, ?)
            """,
            (user_id, quota_key, now, now),
        )


def create_session(
    db: sqlite3.Connection,
    user_id: int,
    *,
    user_agent: str = "",
    ip_address: str = "",
) -> tuple[str, int, str]:
    token = new_token()
    now = utcnow()
    expires_at = now + timedelta(days=session_days())
    cursor = db.execute(
        """
        INSERT INTO sessions
        (user_id, session_token_hash, user_agent, ip_address, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, hash_token(token), user_agent, ip_address, iso(now), iso(expires_at)),
    )
    refresh_token = new_token()
    db.execute(
        """
        INSERT INTO refresh_tokens
        (user_id, token_hash, session_id, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, hash_token(refresh_token), cursor.lastrowid, iso(now), iso(expires_at)),
    )
    return token, int(cursor.lastrowid), iso(expires_at)


def send_register_email_code(
    *,
    email: str,
    invite_code: str,
    ip_address: str = "",
) -> dict[str, Any]:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise ValueError("Invalid email.")
    if not str(invite_code or "").strip():
        raise ValueError("Invite code is required.")

    code = f"{random.SystemRandom().randint(0, 999999):06d}"
    with auth_db() as db:
        invite = _validate_invite(db, invite_code, normalized)
        db.execute(
            """
            INSERT INTO email_verification_codes
            (email, purpose, code_hash, invite_code_id, created_at, expires_at, ip_address)
            VALUES (?, 'register', ?, ?, ?, ?, ?)
            """,
            (
                normalized,
                hash_token(code),
                invite["id"],
                iso(),
                iso(utcnow() + timedelta(minutes=EMAIL_CODE_MINUTES)),
                ip_address,
            ),
        )

    sent = send_verification_email(normalized, code)
    if not sent and os.getenv("AUTH_ALLOW_DEV_EMAIL_CODES", "0") != "1":
        raise RuntimeError("Email service is not configured.")

    payload = {"sent": sent, "expires_in": EMAIL_CODE_MINUTES * 60}
    if not sent:
        payload["dev_code"] = code
    return payload


def _consume_email_code(
    db: sqlite3.Connection,
    *,
    email: str,
    code: str,
) -> None:
    row = db.execute(
        """
        SELECT * FROM email_verification_codes
        WHERE email = ? AND purpose = 'register' AND consumed_at IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (email,),
    ).fetchone()
    if row is None:
        raise ValueError("Verification code not found.")
    if parse_iso(row["expires_at"]) <= utcnow():
        raise ValueError("Verification code has expired.")
    if int(row["attempts"]) >= int(row["max_attempts"]):
        raise ValueError("Too many verification attempts.")
    db.execute(
        "UPDATE email_verification_codes SET attempts = attempts + 1 WHERE id = ?",
        (row["id"],),
    )
    if not constant_time_equal(hash_token(code.strip()), row["code_hash"]):
        raise ValueError("Invalid verification code.")
    db.execute(
        "UPDATE email_verification_codes SET consumed_at = ? WHERE id = ?",
        (iso(), row["id"]),
    )


def register_user(
    *,
    email: str,
    password: str,
    invite_code: str,
    verification_code: str,
    display_name: str = "",
    user_agent: str = "",
    ip_address: str = "",
) -> dict[str, Any]:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise ValueError("Invalid email.")
    if len(password or "") < 8:
        raise ValueError("Password must contain at least 8 characters.")

    with auth_db() as db:
        invite = _validate_invite(db, invite_code, normalized)
        _consume_email_code(db, email=normalized, code=verification_code)
        now = iso()
        try:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_verified_at, password_hash, display_name, role, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'user', 'active', ?, ?)
                """,
                (
                    normalized,
                    now,
                    hash_password(password),
                    str(display_name or "").strip()[:80] or None,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Email is already registered.") from exc
        user_id = int(cursor.lastrowid)
        db.execute(
            "UPDATE invite_codes SET used_count = used_count + 1 WHERE id = ?",
            (invite["id"],),
        )
        create_default_quotas(db, user_id)
        token, session_id, expires_at = create_session(
            db,
            user_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        grant_weekly_tokens_if_due(user_id, db=db)
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return {
            "token": token,
            "session_id": session_id,
            "expires_at": expires_at,
            "user": public_user(user, db=db),
        }


def login_user(
    *,
    email: str,
    password: str,
    user_agent: str = "",
    ip_address: str = "",
) -> dict[str, Any]:
    normalized = normalize_email(email)
    with auth_db() as db:
        user = db.execute("SELECT * FROM users WHERE email = ?", (normalized,)).fetchone()
        if user is None or not verify_password(password or "", user["password_hash"]):
            raise ValueError("Invalid email or password.")
        if user["status"] != "active":
            raise ValueError("Account is not active.")
        token, session_id, expires_at = create_session(
            db,
            int(user["id"]),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        db.execute(
            "UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?",
            (iso(), iso(), user["id"]),
        )
        grant_weekly_tokens_if_due(int(user["id"]), db=db)
        fresh_user = db.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return {
            "token": token,
            "session_id": session_id,
            "expires_at": expires_at,
            "user": public_user(fresh_user, db=db),
        }


def authenticate_session_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    with auth_db() as db:
        row = db.execute(
            """
            SELECT
              sessions.id AS session_id,
              sessions.expires_at,
              sessions.revoked_at,
              users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.session_token_hash = ?
            """,
            (hash_token(token),),
        ).fetchone()
        if row is None or row["revoked_at"] or row["status"] != "active":
            return None
        expires_at = parse_iso(row["expires_at"])
        if expires_at is None or expires_at <= utcnow():
            return None
        user = public_user(row, db=db)
        user["session_id"] = int(row["session_id"])
        return user


def revoke_session(token: str | None) -> None:
    if not token:
        return
    with auth_db() as db:
        now = iso()
        db.execute(
            "UPDATE sessions SET revoked_at = ? WHERE session_token_hash = ?",
            (now, hash_token(token)),
        )
        db.execute(
            """
            UPDATE refresh_tokens
            SET revoked_at = ?
            WHERE session_id IN (
              SELECT id FROM sessions WHERE session_token_hash = ?
            )
            """,
            (now, hash_token(token)),
        )


def record_usage(
    *,
    user_id: int,
    session_id: int | None,
    event_type: str,
    quota_key: str,
    cost: int = 0,
    metadata: dict[str, Any] | None = None,
    ip_address: str = "",
) -> None:
    now = iso()
    with auth_db() as db:
        db.execute(
            """
            INSERT INTO usage_events
            (user_id, session_id, event_type, quota_key, cost, metadata_json, ip_address, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                event_type,
                quota_key,
                max(0, int(cost)),
                json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
                ip_address,
                now,
            ),
        )
        if cost > 0:
            create_default_quotas(db, user_id)
            db.execute(
                """
                UPDATE user_quotas
                SET used_value = used_value + ?, updated_at = ?
                WHERE user_id = ? AND quota_key = ? AND period = 'lifetime'
                """,
                (int(cost), now, user_id, quota_key),
            )
