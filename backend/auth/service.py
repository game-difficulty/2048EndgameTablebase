from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import auth_db
from .entitlements import get_user_entitlements
from .mailer import send_verification_email
from .security import constant_time_equal, hash_password, hash_token, new_token, verify_password
from backend.quota.service import get_token_balance, grant_weekly_tokens_if_due


SESSION_COOKIE_NAME = "tb_session"
BROWSER_COOKIE_NAME = "tb_browser"
DEFAULT_SESSION_DAYS = 14
EMAIL_CODE_MINUTES = 10
EMAIL_CODE_BROWSER_COOLDOWN_SECONDS = 5 * 60
EMAIL_CODE_RATE_WINDOW_MINUTES = 60
EMAIL_CODE_MAX_PER_EMAIL = 5
EMAIL_CODE_MAX_PER_IP = 20
ACCOUNT_DEACTIVATE_CONFIRM_TEXT = "DELETE"
DEFAULT_REGISTRATION_EMAIL_DOMAINS = (
    "qq.com",
    "foxmail.com",
    "163.com",
    "126.com",
    "yeah.net",
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
)
PLUS_ALIAS_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
}
DEFAULT_QUOTA_KEYS = (
    "trainer_query",
    "tester_move",
    "analysis_job",
    "upload_analysis",
    "upload_replay",
    "replay_load",
)


class EmailCodeCooldownError(ValueError):
    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__("Please wait before requesting another verification code.")


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


def split_email(email: str) -> tuple[str, str]:
    normalized = normalize_email(email)
    local, separator, domain = normalized.rpartition("@")
    if not separator or not local or not domain or "@" in local:
        raise ValueError("Invalid email.")
    return local, domain


def registration_email_domains() -> tuple[str, ...]:
    raw = os.getenv("AUTH_REGISTRATION_EMAIL_DOMAINS", "").strip()
    if not raw:
        return DEFAULT_REGISTRATION_EMAIL_DOMAINS
    domains = tuple(
        item.strip().lower().lstrip("@")
        for item in raw.split(",")
        if item.strip()
    )
    return domains or DEFAULT_REGISTRATION_EMAIL_DOMAINS


def canonical_email_identity(email: str) -> str:
    local, domain = split_email(email)
    if domain == "googlemail.com":
        domain = "gmail.com"
    if domain == "gmail.com":
        local = local.split("+", 1)[0].replace(".", "")
    elif domain in PLUS_ALIAS_DOMAINS:
        local = local.split("+", 1)[0]
    if not local:
        raise ValueError("Invalid email.")
    return f"{local}@{domain}"


def validate_registration_email(email: str) -> tuple[str, str]:
    normalized = normalize_email(email)
    split_email(normalized)
    domain = normalized.rsplit("@", 1)[1]
    if domain not in set(registration_email_domains()):
        raise ValueError("Email domain is not currently supported for registration.")
    return normalized, canonical_email_identity(normalized)


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
        "entitlements": get_user_entitlements(user_id, db=db),
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


def _validate_password(password: str) -> None:
    if len(password or "") < 8:
        raise ValueError("Password must contain at least 8 characters.")


def _validate_display_name(display_name: str) -> str:
    value = str(display_name or "").strip()
    if not value:
        raise ValueError("Username is required.")
    if len(value) > 80:
        raise ValueError("Username must contain at most 80 characters.")
    return value


def _check_email_code_rate_limit(
    db: sqlite3.Connection,
    *,
    email: str,
    purpose: str,
    ip_address: str = "",
) -> None:
    cutoff = iso(utcnow() - timedelta(minutes=EMAIL_CODE_RATE_WINDOW_MINUTES))
    email_count = db.execute(
        """
        SELECT COUNT(*) AS count
        FROM email_verification_codes
        WHERE email = ? AND purpose = ? AND created_at >= ?
        """,
        (email, purpose, cutoff),
    ).fetchone()["count"]
    if int(email_count) >= EMAIL_CODE_MAX_PER_EMAIL:
        raise ValueError("Too many verification emails. Please try again later.")

    if ip_address:
        ip_count = db.execute(
            """
            SELECT COUNT(*) AS count
            FROM email_verification_codes
            WHERE ip_address = ? AND purpose = ? AND created_at >= ?
            """,
            (ip_address, purpose, cutoff),
        ).fetchone()["count"]
        if int(ip_count) >= EMAIL_CODE_MAX_PER_IP:
            raise ValueError("Too many verification emails. Please try again later.")


def _reserve_browser_email_code_cooldown(
    db: sqlite3.Connection,
    *,
    browser_id: str = "",
    purpose: str,
) -> None:
    browser_token_hash = hash_token(str(browser_id or "").strip())
    if not str(browser_id or "").strip():
        return

    now = utcnow()
    now_text = iso(now)
    cutoff_text = iso(now - timedelta(seconds=EMAIL_CODE_BROWSER_COOLDOWN_SECONDS))
    try:
        db.execute(
            """
            INSERT INTO auth_browser_cooldowns
            (browser_token_hash, purpose, last_sent_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (browser_token_hash, purpose, now_text, now_text),
        )
        return
    except sqlite3.IntegrityError:
        pass

    cursor = db.execute(
        """
        UPDATE auth_browser_cooldowns
        SET last_sent_at = ?, updated_at = ?
        WHERE browser_token_hash = ? AND purpose = ? AND last_sent_at <= ?
        """,
        (now_text, now_text, browser_token_hash, purpose, cutoff_text),
    )
    if cursor.rowcount == 1:
        return

    row = db.execute(
        """
        SELECT last_sent_at
        FROM auth_browser_cooldowns
        WHERE browser_token_hash = ? AND purpose = ?
        """,
        (browser_token_hash, purpose),
    ).fetchone()
    last_sent_at = parse_iso(row["last_sent_at"] if row else None)
    if last_sent_at is None:
        retry_after = EMAIL_CODE_BROWSER_COOLDOWN_SECONDS
    else:
        elapsed = max(0, int((now - last_sent_at).total_seconds()))
        retry_after = EMAIL_CODE_BROWSER_COOLDOWN_SECONDS - elapsed
    raise EmailCodeCooldownError(retry_after)


def _create_email_code(
    db: sqlite3.Connection,
    *,
    email: str,
    purpose: str,
    ip_address: str = "",
    invite_code_id: int | None = None,
    browser_id: str = "",
) -> str:
    _check_email_code_rate_limit(
        db,
        email=email,
        purpose=purpose,
        ip_address=ip_address,
    )
    _reserve_browser_email_code_cooldown(
        db,
        browser_id=browser_id,
        purpose=purpose,
    )
    code = f"{random.SystemRandom().randint(0, 999999):06d}"
    db.execute(
        """
        INSERT INTO email_verification_codes
        (email, purpose, code_hash, invite_code_id, created_at, expires_at, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            email,
            purpose,
            hash_token(code),
            invite_code_id,
            iso(),
            iso(utcnow() + timedelta(minutes=EMAIL_CODE_MINUTES)),
            ip_address,
        ),
    )
    return code


def _send_code_or_raise(email: str, code: str, *, purpose: str) -> dict[str, Any]:
    try:
        sent = send_verification_email(email, code, purpose=purpose)
    except Exception as exc:
        if os.getenv("AUTH_ALLOW_DEV_EMAIL_CODES", "0") != "1":
            raise RuntimeError("Email service is not available.") from exc
        sent = False

    if not sent and os.getenv("AUTH_ALLOW_DEV_EMAIL_CODES", "0") != "1":
        raise RuntimeError("Email service is not configured.")

    payload = {"sent": sent, "expires_in": EMAIL_CODE_MINUTES * 60}
    if not sent:
        payload["dev_code"] = code
    return payload


def send_register_email_code(
    *,
    email: str,
    invite_code: str,
    ip_address: str = "",
    browser_id: str = "",
) -> dict[str, Any]:
    normalized, email_identity = validate_registration_email(email)

    with auth_db() as db:
        invite_code_value = str(invite_code or "").strip()
        invite = _validate_invite(db, invite_code_value, normalized) if invite_code_value else None
        existing_user = db.execute(
            "SELECT id FROM users WHERE email = ? OR email_identity = ?",
            (normalized, email_identity),
        ).fetchone()
        if existing_user is not None:
            raise ValueError("Email is already registered.")
        code = _create_email_code(
            db,
            email=normalized,
            purpose="register",
            invite_code_id=int(invite["id"]) if invite is not None else None,
            ip_address=ip_address,
            browser_id=browser_id,
        )

    return _send_code_or_raise(normalized, code, purpose="register")


def request_password_reset_code(
    *,
    email: str,
    ip_address: str = "",
    browser_id: str = "",
) -> dict[str, Any]:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise ValueError("Invalid email.")

    generic_payload: dict[str, Any] = {
        "sent": True,
        "expires_in": EMAIL_CODE_MINUTES * 60,
    }
    with auth_db() as db:
        user = db.execute(
            "SELECT id, status FROM users WHERE email = ?",
            (normalized,),
        ).fetchone()
        if user is None or user["status"] != "active":
            return generic_payload
        code = _create_email_code(
            db,
            email=normalized,
            purpose="password_reset",
            ip_address=ip_address,
            browser_id=browser_id,
        )

    return _send_code_or_raise(normalized, code, purpose="password_reset")


def request_account_deactivation_code(
    *,
    user_id: int,
    ip_address: str = "",
    browser_id: str = "",
) -> dict[str, Any]:
    with auth_db() as db:
        user = db.execute("SELECT email, status FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None or user["status"] != "active":
            raise ValueError("Account is not active.")
        email = normalize_email(user["email"])
        code = _create_email_code(
            db,
            email=email,
            purpose="account_deactivate",
            ip_address=ip_address,
            browser_id=browser_id,
        )

    return _send_code_or_raise(email, code, purpose="account_deactivate")


def _consume_email_code(
    db: sqlite3.Connection,
    *,
    email: str,
    code: str,
    purpose: str,
) -> None:
    row = db.execute(
        """
        SELECT * FROM email_verification_codes
        WHERE email = ? AND purpose = ? AND consumed_at IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (email, purpose),
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
    cursor = db.execute(
        """
        UPDATE email_verification_codes
        SET consumed_at = ?
        WHERE id = ? AND consumed_at IS NULL
        """,
        (iso(), row["id"]),
    )
    if cursor.rowcount != 1:
        raise ValueError("Verification code has already been used.")


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
    normalized, email_identity = validate_registration_email(email)
    _validate_password(password)
    display_name_value = _validate_display_name(display_name)

    with auth_db() as db:
        invite_code_value = str(invite_code or "").strip()
        invite = _validate_invite(db, invite_code_value, normalized) if invite_code_value else None
        existing_user = db.execute(
            "SELECT id FROM users WHERE email = ? OR email_identity = ?",
            (normalized, email_identity),
        ).fetchone()
        if existing_user is not None:
            raise ValueError("Email is already registered.")
        _consume_email_code(
            db,
            email=normalized,
            code=verification_code,
            purpose="register",
        )
        now = iso()
        try:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, email_verified_at, password_hash, display_name,
                 registered_with_invite, invite_code_id, role, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'user', 'active', ?, ?)
                """,
                (
                    normalized,
                    email_identity,
                    now,
                    hash_password(password),
                    display_name_value,
                    1 if invite is not None else 0,
                    int(invite["id"]) if invite is not None else None,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Email is already registered.") from exc
        user_id = int(cursor.lastrowid)
        if invite is not None:
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


def _revoke_user_sessions(
    db: sqlite3.Connection,
    user_id: int,
    *,
    except_session_id: int | None = None,
) -> None:
    now = iso()
    if except_session_id is None:
        db.execute(
            "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
            (now, user_id),
        )
        db.execute(
            "UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
            (now, user_id),
        )
        return

    db.execute(
        """
        UPDATE sessions
        SET revoked_at = ?
        WHERE user_id = ? AND id != ? AND revoked_at IS NULL
        """,
        (now, user_id, except_session_id),
    )
    db.execute(
        """
        UPDATE refresh_tokens
        SET revoked_at = ?
        WHERE user_id = ? AND (session_id IS NULL OR session_id != ?) AND revoked_at IS NULL
        """,
        (now, user_id, except_session_id),
    )


def reset_password(
    *,
    email: str,
    verification_code: str,
    new_password: str,
    user_agent: str = "",
    ip_address: str = "",
) -> dict[str, Any]:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise ValueError("Invalid email.")
    _validate_password(new_password)

    with auth_db() as db:
        user = db.execute("SELECT * FROM users WHERE email = ?", (normalized,)).fetchone()
        if user is None or user["status"] != "active":
            raise ValueError("Invalid verification code.")
        _consume_email_code(
            db,
            email=normalized,
            code=verification_code,
            purpose="password_reset",
        )
        now = iso()
        user_id = int(user["id"])
        _revoke_user_sessions(db, user_id)
        db.execute(
            """
            UPDATE users
            SET password_hash = ?, password_changed_at = ?, last_login_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (hash_password(new_password), now, now, now, user_id),
        )
        token, session_id, expires_at = create_session(
            db,
            user_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        grant_weekly_tokens_if_due(user_id, db=db)
        fresh_user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return {
            "token": token,
            "session_id": session_id,
            "expires_at": expires_at,
            "user": public_user(fresh_user, db=db),
        }


def change_password(
    *,
    user_id: int,
    session_id: int | None,
    current_password: str,
    new_password: str,
) -> dict[str, Any]:
    _validate_password(new_password)
    with auth_db() as db:
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None or user["status"] != "active":
            raise ValueError("Account is not active.")
        if not verify_password(current_password or "", user["password_hash"]):
            raise ValueError("Invalid current password.")
        now = iso()
        db.execute(
            """
            UPDATE users
            SET password_hash = ?, password_changed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (hash_password(new_password), now, now, user_id),
        )
        _revoke_user_sessions(db, user_id, except_session_id=session_id)
        fresh_user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return {"user": public_user(fresh_user, db=db)}


def deactivate_account(
    *,
    user_id: int,
    password: str,
    confirm: str,
    verification_code: str,
) -> None:
    if str(confirm or "").strip() != ACCOUNT_DEACTIVATE_CONFIRM_TEXT:
        raise ValueError("Confirmation text is incorrect.")
    with auth_db() as db:
        user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None or user["status"] != "active":
            raise ValueError("Account is not active.")
        if not verify_password(password or "", user["password_hash"]):
            raise ValueError("Invalid password.")
        _consume_email_code(
            db,
            email=normalize_email(user["email"]),
            code=verification_code,
            purpose="account_deactivate",
        )
        now = iso()
        db.execute(
            """
            UPDATE users
            SET status = 'disabled', deactivated_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, now, user_id),
        )
        _revoke_user_sessions(db, user_id)


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
