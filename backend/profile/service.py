from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Any, Iterator

from backend.auth.db import auth_db

from .storage import (
    ProcessedAvatar,
    delete_avatar_file,
    save_processed_avatar,
)
from .validation import canonical_display_name_key, validate_display_name


PROFILE_CHANGE_COOLDOWN = timedelta(days=30)


class ProfileCooldownError(ValueError):
    def __init__(self, field: str, available_at: str):
        super().__init__(f"{field} cannot be changed yet.")
        self.field = field
        self.available_at = available_at


class DisplayNameTakenError(ValueError):
    pass


class AvatarUploadDisabledError(ValueError):
    pass


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


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utcnow()).astimezone(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ensure_user_profile(db: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    user = db.execute(
        "SELECT id, created_at FROM users WHERE id = ?",
        (int(user_id),),
    ).fetchone()
    if user is None:
        raise ValueError("User not found.")
    now = iso()
    db.execute(
        """
        INSERT OR IGNORE INTO user_profiles
        (user_id, display_name_changed_at, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (int(user_id), user["created_at"], now, now),
    )
    return db.execute(
        "SELECT * FROM user_profiles WHERE user_id = ?",
        (int(user_id),),
    ).fetchone()


def _available_at(changed_at: str | None) -> datetime | None:
    changed = parse_iso(changed_at)
    return changed + PROFILE_CHANGE_COOLDOWN if changed is not None else None


def _cooldown_state(changed_at: str | None, now: datetime) -> tuple[bool, str | None]:
    available = _available_at(changed_at)
    if available is None or available <= now:
        return True, None
    return False, iso(available)


def public_profile(
    user_id: int,
    *,
    db: sqlite3.Connection | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    with _maybe_connection(db) as connection:
        row = ensure_user_profile(connection, int(user_id))
        entitlement = connection.execute(
            "SELECT can_upload_avatar FROM user_entitlements WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        upload_enabled = entitlement is None or bool(entitlement["can_upload_avatar"])
        current_time = now or utcnow()
        can_change_name, name_available_at = _cooldown_state(
            row["display_name_changed_at"], current_time
        )
        can_change_avatar, avatar_available_at = _cooldown_state(
            row["avatar_changed_at"], current_time
        )
        avatar_url = (
            f"/media/avatars/{row['avatar_key']}" if row["avatar_key"] else None
        )
        return {
            "avatar_url": avatar_url,
            "can_upload_avatar": upload_enabled,
            "can_change_avatar": upload_enabled and can_change_avatar,
            "avatar_change_available_at": avatar_available_at,
            "can_change_display_name": can_change_name,
            "display_name_change_available_at": name_available_at,
        }


def _assert_cooldown(field: str, changed_at: str | None, now: datetime) -> None:
    available = _available_at(changed_at)
    if available is not None and available > now:
        raise ProfileCooldownError(field, iso(available))


def _insert_change_event(
    db: sqlite3.Connection,
    *,
    user_id: int,
    change_type: str,
    old_value: str | None,
    new_value: str | None,
    ip_address: str,
    user_agent: str,
    created_at: str,
) -> None:
    db.execute(
        """
        INSERT INTO user_profile_change_events
        (user_id, change_type, old_value, new_value, ip_address, user_agent, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            change_type,
            old_value,
            new_value,
            str(ip_address or "")[:96],
            str(user_agent or "")[:320],
            created_at,
        ),
    )


def update_display_name(
    user_id: int,
    display_name: str,
    *,
    ip_address: str = "",
    user_agent: str = "",
) -> bool:
    normalized, name_key = validate_display_name(display_name)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        user = db.execute(
            "SELECT * FROM users WHERE id = ? AND status = 'active'",
            (int(user_id),),
        ).fetchone()
        if user is None:
            raise ValueError("User not found.")
        profile = ensure_user_profile(db, int(user_id))
        current_name = str(user["display_name"] or "")
        if normalized == current_name:
            return False

        now = utcnow()
        _assert_cooldown("display_name", profile["display_name_changed_at"], now)
        existing = db.execute(
            "SELECT id FROM users WHERE display_name_key = ? AND id != ?",
            (name_key, int(user_id)),
        ).fetchone()
        if existing is not None:
            raise DisplayNameTakenError("Username is already in use.")

        changed_at = iso(now)
        try:
            db.execute(
                """
                UPDATE users
                SET display_name = ?, display_name_key = ?, updated_at = ?
                WHERE id = ?
                """,
                (normalized, name_key, changed_at, int(user_id)),
            )
        except sqlite3.IntegrityError as exc:
            raise DisplayNameTakenError("Username is already in use.") from exc
        db.execute(
            """
            UPDATE user_profiles
            SET display_name_changed_at = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (changed_at, changed_at, int(user_id)),
        )
        db.execute(
            "UPDATE leaderboard_entries SET display_name = ? WHERE user_id = ?",
            (normalized, int(user_id)),
        )
        _insert_change_event(
            db,
            user_id=int(user_id),
            change_type="display_name",
            old_value=current_name,
            new_value=normalized,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=changed_at,
        )
        return True


def assert_avatar_change_allowed(user_id: int) -> None:
    with auth_db() as db:
        user = db.execute(
            "SELECT id FROM users WHERE id = ? AND status = 'active'",
            (int(user_id),),
        ).fetchone()
        if user is None:
            raise ValueError("User not found.")
        entitlement = db.execute(
            "SELECT can_upload_avatar FROM user_entitlements WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if entitlement is not None and not bool(entitlement["can_upload_avatar"]):
            raise AvatarUploadDisabledError("Avatar uploads are disabled for this account.")


def update_avatar(
    user_id: int,
    avatar: ProcessedAvatar,
    *,
    ip_address: str = "",
    user_agent: str = "",
) -> bool:
    new_key = save_processed_avatar(int(user_id), avatar)
    old_key: str | None = None
    changed = False
    try:
        with auth_db() as db:
            db.execute("BEGIN IMMEDIATE")
            user = db.execute(
                "SELECT id FROM users WHERE id = ? AND status = 'active'",
                (int(user_id),),
            ).fetchone()
            if user is None:
                raise ValueError("User not found.")
            entitlement = db.execute(
                "SELECT can_upload_avatar FROM user_entitlements WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
            if entitlement is not None and not bool(entitlement["can_upload_avatar"]):
                raise AvatarUploadDisabledError("Avatar uploads are disabled for this account.")
            profile = ensure_user_profile(db, int(user_id))
            old_key = profile["avatar_key"]
            if profile["avatar_sha256"] == avatar.sha256:
                return False

            now = utcnow()
            _assert_cooldown("avatar", profile["avatar_changed_at"], now)
            changed_at = iso(now)
            db.execute(
                """
                UPDATE user_profiles
                SET avatar_key = ?, avatar_sha256 = ?, avatar_changed_at = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (new_key, avatar.sha256, changed_at, changed_at, int(user_id)),
            )
            _insert_change_event(
                db,
                user_id=int(user_id),
                change_type="avatar",
                old_value=old_key,
                new_value=new_key,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=changed_at,
            )
            changed = True
    except Exception:
        if not _avatar_key_is_referenced(new_key):
            delete_avatar_file(new_key)
        raise

    if changed and old_key and old_key != new_key:
        delete_avatar_file(old_key)
    return changed


def delete_avatar(
    user_id: int,
    *,
    ip_address: str = "",
    user_agent: str = "",
) -> bool:
    old_key: str | None = None
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        user = db.execute(
            "SELECT id FROM users WHERE id = ? AND status = 'active'",
            (int(user_id),),
        ).fetchone()
        if user is None:
            raise ValueError("User not found.")
        entitlement = db.execute(
            "SELECT can_upload_avatar FROM user_entitlements WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if entitlement is not None and not bool(entitlement["can_upload_avatar"]):
            raise AvatarUploadDisabledError("Avatar uploads are disabled for this account.")
        profile = ensure_user_profile(db, int(user_id))
        old_key = profile["avatar_key"]
        if not old_key:
            return False
        now = utcnow()
        _assert_cooldown("avatar", profile["avatar_changed_at"], now)
        changed_at = iso(now)
        db.execute(
            """
            UPDATE user_profiles
            SET avatar_key = NULL, avatar_sha256 = NULL,
                avatar_changed_at = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (changed_at, changed_at, int(user_id)),
        )
        _insert_change_event(
            db,
            user_id=int(user_id),
            change_type="avatar_remove",
            old_value=old_key,
            new_value=None,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=changed_at,
        )
    delete_avatar_file(old_key)
    return True


def _avatar_key_is_referenced(key: str) -> bool:
    with auth_db() as db:
        return db.execute(
            "SELECT 1 FROM user_profiles WHERE avatar_key = ? LIMIT 1",
            (key,),
        ).fetchone() is not None
