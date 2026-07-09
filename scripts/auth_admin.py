from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.auth.db import auth_db, init_auth_db
from backend.auth.security import hash_password, hash_token, new_token
from backend.auth.service import canonical_email_identity, create_default_quotas, iso, normalize_email, utcnow


def create_admin(args: argparse.Namespace) -> None:
    email = normalize_email(args.email)
    email_identity = canonical_email_identity(email)
    password_hash = hash_password(args.password)
    with auth_db() as db:
        cursor = db.execute(
            """
            INSERT INTO users (email, email_identity, password_hash, display_name, role, status, email_verified_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'admin', 'active', ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
              email_identity = COALESCE(users.email_identity, excluded.email_identity),
              password_hash = excluded.password_hash,
              display_name = excluded.display_name,
              role = 'admin',
              status = 'active',
              email_verified_at = COALESCE(users.email_verified_at, excluded.email_verified_at),
              updated_at = excluded.updated_at
            RETURNING id
            """,
            (
                email,
                email_identity,
                password_hash,
                args.display_name or "Admin",
                iso(),
                iso(),
                iso(),
            ),
        )
        user_id = int(cursor.fetchone()["id"])
        create_default_quotas(db, user_id)
    print(f"admin_user_id={user_id}")
    print(f"email={email}")


def create_invite(args: argparse.Namespace) -> None:
    code = args.code or new_token(18)
    code_hash = hash_token(code)
    expires_at = (
        iso(utcnow() + timedelta(days=args.expires_days))
        if args.expires_days > 0
        else None
    )
    created_by_user_id = None
    if args.created_by_email:
        email = normalize_email(args.created_by_email)
        with auth_db() as db:
            row = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            raise SystemExit(f"creator user not found: {email}")
        created_by_user_id = int(row["id"])

    with auth_db() as db:
        db.execute(
            """
            INSERT INTO invite_codes
            (code_hash, label, allowed_domain, allowed_email, max_uses, used_count, expires_at, created_by_user_id, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                code_hash,
                args.label or "",
                args.email_domain.strip().lower().lstrip("@") if args.email_domain else None,
                normalize_email(args.email) if args.email else None,
                max(1, args.uses),
                expires_at,
                created_by_user_id,
                iso(),
            ),
        )
    print(f"invite_code={code}")
    print(f"uses={args.uses}")
    if expires_at:
        print(f"expires_at={expires_at}")


def list_users(_args: argparse.Namespace) -> None:
    with auth_db() as db:
        rows = db.execute(
            """
            SELECT id, email, display_name, role, status, created_at, last_login_at
            FROM users
            ORDER BY id
            """
        ).fetchall()
    if not rows:
        print("no users")
        return
    print("id\temail\trole\tstatus\tcreated_at\tlast_login_at\tdisplay_name")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["id"]),
                    row["email"],
                    row["role"],
                    row["status"],
                    row["created_at"],
                    row["last_login_at"] or "",
                    row["display_name"] or "",
                ]
            )
        )


def list_invites(_args: argparse.Namespace) -> None:
    with auth_db() as db:
        rows = db.execute(
            """
            SELECT id, label, allowed_domain, allowed_email, max_uses, used_count, disabled_at, expires_at, created_at
            FROM invite_codes
            ORDER BY id DESC
            """
        ).fetchall()
    if not rows:
        print("no invites")
        return
    print("id\tlabel\tallowed_domain\tallowed_email\tuses\tstatus\texpires_at\tcreated_at")
    for row in rows:
        uses = f"{row['used_count']}/{row['max_uses']}"
        status = "disabled" if row["disabled_at"] else "active"
        print(
            "\t".join(
                [
                    str(row["id"]),
                    row["label"] or "",
                    row["allowed_domain"] or "",
                    row["allowed_email"] or "",
                    uses,
                    status,
                    row["expires_at"] or "",
                    row["created_at"],
                ]
            )
        )


def disable_user(args: argparse.Namespace) -> None:
    email = normalize_email(args.email)
    with auth_db() as db:
        now = iso()
        cursor = db.execute(
            """
            UPDATE users
            SET status = 'disabled', deactivated_at = COALESCE(deactivated_at, ?), updated_at = ?
            WHERE email = ?
            """,
            (now, now, email),
        )
        if cursor.rowcount < 1:
            raise SystemExit(f"user not found: {email}")
        db.execute(
            """
            UPDATE sessions
            SET revoked_at = ?
            WHERE user_id IN (SELECT id FROM users WHERE email = ?) AND revoked_at IS NULL
            """,
            (now, email),
        )
        db.execute(
            """
            UPDATE refresh_tokens
            SET revoked_at = ?
            WHERE user_id IN (SELECT id FROM users WHERE email = ?) AND revoked_at IS NULL
            """,
            (now, email),
        )
    print(f"disabled={email}")


def enable_user(args: argparse.Namespace) -> None:
    email = normalize_email(args.email)
    with auth_db() as db:
        now = iso()
        cursor = db.execute(
            """
            UPDATE users
            SET status = 'active', deactivated_at = NULL, updated_at = ?
            WHERE email = ?
            """,
            (now, email),
        )
        if cursor.rowcount < 1:
            raise SystemExit(f"user not found: {email}")
    print(f"enabled={email}")


def set_password(args: argparse.Namespace) -> None:
    email = normalize_email(args.email)
    with auth_db() as db:
        now = iso()
        cursor = db.execute(
            """
            UPDATE users
            SET password_hash = ?, password_changed_at = ?, updated_at = ?
            WHERE email = ?
            """,
            (hash_password(args.password), now, now, email),
        )
        if cursor.rowcount < 1:
            raise SystemExit(f"user not found: {email}")
        db.execute(
            """
            UPDATE sessions
            SET revoked_at = ?
            WHERE user_id IN (SELECT id FROM users WHERE email = ?) AND revoked_at IS NULL
            """,
            (now, email),
        )
        db.execute(
            """
            UPDATE refresh_tokens
            SET revoked_at = ?
            WHERE user_id IN (SELECT id FROM users WHERE email = ?) AND revoked_at IS NULL
            """,
            (now, email),
        )
    print(f"password_updated={email}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2048 cloud auth administration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    admin_parser = subparsers.add_parser("create-admin", help="Create or update an admin user")
    admin_parser.add_argument("--email", required=True)
    admin_parser.add_argument("--password", required=True)
    admin_parser.add_argument("--display-name", default="Admin")
    admin_parser.set_defaults(func=create_admin)

    invite_parser = subparsers.add_parser("create-invite", help="Create an invite code")
    invite_parser.add_argument("--code", default="")
    invite_parser.add_argument("--uses", type=int, default=1)
    invite_parser.add_argument("--expires-days", type=int, default=30)
    invite_parser.add_argument("--label", default="")
    invite_parser.add_argument("--email", default="")
    invite_parser.add_argument("--email-domain", default="")
    invite_parser.add_argument("--created-by-email", default="")
    invite_parser.set_defaults(func=create_invite)

    users_parser = subparsers.add_parser("list-users", help="List users")
    users_parser.set_defaults(func=list_users)

    invites_parser = subparsers.add_parser("list-invites", help="List invite metadata")
    invites_parser.set_defaults(func=list_invites)

    disable_parser = subparsers.add_parser("disable-user", help="Disable a user and revoke sessions")
    disable_parser.add_argument("--email", required=True)
    disable_parser.set_defaults(func=disable_user)

    enable_parser = subparsers.add_parser("enable-user", help="Enable a disabled user")
    enable_parser.add_argument("--email", required=True)
    enable_parser.set_defaults(func=enable_user)

    password_parser = subparsers.add_parser("set-password", help="Set user password and revoke sessions")
    password_parser.add_argument("--email", required=True)
    password_parser.add_argument("--password", required=True)
    password_parser.set_defaults(func=set_password)
    return parser


def main() -> None:
    init_auth_db()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
