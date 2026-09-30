from __future__ import annotations

import secrets

from .db import auth_db
from .security import hash_password
from .service import (
    _revoke_user_sessions,
    _validate_password,
    create_default_quotas,
    grant_weekly_tokens_if_due,
    iso,
)


ACCOUNT_DOMAIN = "accounts.invalid"
ACCOUNT_COUNT = 6


def provision_accounts() -> list[dict]:
    """Create the fixed set once; existing accounts are never overwritten."""
    created = []
    with auth_db() as db:
        for number in range(1, ACCOUNT_COUNT + 1):
            email = f"site-test-{number:02d}@{ACCOUNT_DOMAIN}"
            existing = db.execute(
                "SELECT id,managed_test_account FROM users WHERE email=?", (email,)
            ).fetchone()
            if existing is not None:
                if not existing["managed_test_account"]:
                    raise ValueError(f"Account already exists without managed marker: {email}")
                continue
            name = f"SiteTest{number:02d}"
            if db.execute("SELECT 1 FROM users WHERE display_name_key=?", (name.lower(),)).fetchone():
                raise ValueError(f"Display name is already in use: {name}")
            password = secrets.token_urlsafe(24)
            now = iso()
            cursor = db.execute(
                """INSERT INTO users
                (email,email_identity,password_hash,display_name,display_name_key,
                 registered_with_invite,managed_test_account,role,status,created_at,updated_at)
                VALUES(?,?,?,?,?,0,1,'user','active',?,?)""",
                (email, email, hash_password(password), name, name.lower(), now, now),
            )
            user_id = int(cursor.lastrowid)
            create_default_quotas(db, user_id)
            grant_weekly_tokens_if_due(user_id, db=db)
            db.execute(
                "INSERT INTO managed_test_account_audit(user_id,action,created_at) VALUES(?,'provision',?)",
                (user_id, now),
            )
            created.append({"id": user_id, "email": email, "password": password})
    return created


def reset_managed_password(*, user_id: int, operator_id: int, new_password: str) -> None:
    _validate_password(new_password)
    with auth_db() as db:
        user = db.execute(
            "SELECT id FROM users WHERE id=? AND managed_test_account=1 AND role='user'",
            (user_id,),
        ).fetchone()
        if user is None:
            raise FileNotFoundError("Managed test account not found.")
        now = iso()
        db.execute(
            "UPDATE users SET password_hash=?,password_changed_at=?,updated_at=? WHERE id=?",
            (hash_password(new_password), now, now, user_id),
        )
        _revoke_user_sessions(db, user_id)
        db.execute(
            """INSERT INTO managed_test_account_audit(user_id,operator_id,action,created_at)
            VALUES(?,?,'password_reset',?)""",
            (user_id, operator_id, now),
        )
