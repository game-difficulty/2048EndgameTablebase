from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.auth.db import auth_db, init_auth_db
from backend.auth.managed_test_accounts import provision_accounts


def ensure_managed_schema() -> None:
    with auth_db() as db:
        columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
        if "managed_test_account" not in columns:
            db.execute("ALTER TABLE users ADD COLUMN managed_test_account INTEGER NOT NULL DEFAULT 0")
        db.execute("""CREATE TABLE IF NOT EXISTS managed_test_account_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            operator_id INTEGER REFERENCES users(id),
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision six ordinary admin-managed test accounts")
    parser.add_argument("--credentials-file", required=True, type=Path)
    args = parser.parse_args()
    if args.credentials_file.exists():
        parser.error("credentials file already exists")
    init_auth_db()
    ensure_managed_schema()
    created = provision_accounts()
    if created:
        descriptor = os.open(args.credentials_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(created, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(f"Created {len(created)} accounts; {6 - len(created)} already existed.")


if __name__ == "__main__":
    main()
