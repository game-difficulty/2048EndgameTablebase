from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

from backend.auth.db import auth_db, init_auth_db
from backend.auth.entitlements import SUPPORTER_TIER
from backend.quota.service import grant_weekly_tokens_if_due


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WeeklyTokenGrantTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_auth_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "auth.sqlite3")
        init_auth_db()

    def tearDown(self) -> None:
        if self._old_auth_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_auth_db
        self._tmpdir.cleanup()

    def _create_user(self, email: str, *, invited: bool, supporter: bool = False) -> int:
        now = _iso_now()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, ?, 'user', 'active', ?, ?)
                """,
                (email, email, email.split("@", 1)[0], 1 if invited else 0, now, now),
            )
            user_id = int(cursor.lastrowid)
            if supporter:
                db.execute(
                    """
                    INSERT INTO user_entitlements
                    (user_id, tier, supporter_since, show_supporter_badge,
                     can_upload_avatar, created_at, updated_at)
                    VALUES (?, ?, ?, 1, 0, ?, ?)
                    """,
                    (user_id, SUPPORTER_TIER, now, now, now),
                )
            return user_id

    def test_public_user_gets_512_weekly_bonus_tokens(self) -> None:
        user_id = self._create_user("public@example.com", invited=False)

        balance = grant_weekly_tokens_if_due(user_id)

        self.assertEqual(balance["bonus"], 512)
        self.assertEqual(balance["paid"], 0)
        self.assertEqual(balance["total"], 512)

    def test_invited_user_gets_4096_weekly_bonus_tokens(self) -> None:
        user_id = self._create_user("invited@example.com", invited=True)

        balance = grant_weekly_tokens_if_due(user_id)

        self.assertEqual(balance["bonus"], 4096)
        self.assertEqual(balance["paid"], 0)
        self.assertEqual(balance["total"], 4096)

    def test_supporter_gets_32768_weekly_bonus_tokens_even_when_invited(self) -> None:
        user_id = self._create_user("supporter@example.com", invited=True, supporter=True)

        balance = grant_weekly_tokens_if_due(user_id)

        self.assertEqual(balance["bonus"], 32768)
        self.assertEqual(balance["paid"], 0)
        self.assertEqual(balance["total"], 32768)


if __name__ == "__main__":
    unittest.main()
