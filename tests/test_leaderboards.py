from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import unittest

from backend.auth.db import auth_db, init_auth_db
from backend.leaderboards.service import (
    SUPPORTERS_BOARD,
    TOKEN_LAST_WEEK_BOARD,
    TOKEN_LIFETIME_BOARD,
    _period_for,
    leaderboard_payload,
    refresh_due_leaderboards,
)


class LeaderboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("CLOUD_AUTH_DB")
        self.old_admins = os.environ.get("ADMIN_ALLOWED_IDENTITIES")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        os.environ["ADMIN_ALLOWED_IDENTITIES"] = "user0,admin@example.com"
        init_auth_db()

    def tearDown(self) -> None:
        if self.old_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.old_db
        if self.old_admins is None:
            os.environ.pop("ADMIN_ALLOWED_IDENTITIES", None)
        else:
            os.environ["ADMIN_ALLOWED_IDENTITIES"] = self.old_admins
        self.tempdir.cleanup()

    def _add_user(
        self,
        db,
        *,
        email: str,
        name: str,
        role: str = "user",
        status: str = "active",
        supporter: bool = False,
        paid_units: int = 0,
    ) -> int:
        cursor = db.execute(
            """
            INSERT INTO users
            (email, email_identity, password_hash, display_name, role, status, created_at, updated_at)
            VALUES (?, ?, 'hash', ?, ?, ?, '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00')
            """,
            (email, email, name, role, status),
        )
        user_id = int(cursor.lastrowid)
        db.execute(
            """
            INSERT INTO token_accounts
            (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at)
            VALUES (?, 0, ?, '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00')
            """,
            (user_id, paid_units),
        )
        db.execute(
            """
            INSERT INTO user_entitlements
            (user_id, tier, created_at, updated_at)
            VALUES (?, ?, '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00')
            """,
            (user_id, "supporter" if supporter else "free"),
        )
        return user_id

    def _add_ledger(self, db, user_id: int, event_type: str, units: int, created_at: str) -> None:
        db.execute(
            """
            INSERT INTO token_ledger
            (user_id, event_type, operation_key, table_pattern, table_multiplier_units,
             base_cost_units, final_cost_units, bonus_delta_units, paid_delta_units,
             balance_before_units, balance_after_units, metadata_json, created_at)
            VALUES (?, ?, 'test', 'L3_128', 1000, ?, ?, 0, 0, 0, 0, '{}', ?)
            """,
            (user_id, event_type, units, units, created_at),
        )

    def test_supporters_hide_scores_and_token_boards_use_actual_costs(self) -> None:
        now = datetime.now(timezone.utc)
        week_start, _week_end = _period_for(TOKEN_LAST_WEEK_BOARD, now)
        event_time = datetime.fromisoformat(str(week_start)).astimezone(timezone.utc)
        event_times = [
            (event_time + timedelta(hours=10)).isoformat(),
            (event_time + timedelta(hours=11)).isoformat(),
            (event_time + timedelta(hours=12)).isoformat(),
            (event_time + timedelta(hours=13)).isoformat(),
        ]
        with auth_db() as db:
            admin = self._add_user(
                db,
                email="admin@example.com",
                name="user0",
                role="admin",
                supporter=True,
                paid_units=9_000_000,
            )
            alice = self._add_user(
                db,
                email="alice@example.com",
                name="Alice",
                supporter=True,
                paid_units=2_000_000,
            )
            bob = self._add_user(
                db,
                email="bob@example.com",
                name="Bob",
                supporter=True,
                paid_units=1_000_000,
            )
            self._add_user(
                db,
                email="disabled@example.com",
                name="Disabled",
                status="disabled",
                supporter=True,
                paid_units=8_000_000,
            )
            self._add_ledger(db, alice, "reserve", 1000, event_times[0])
            self._add_ledger(db, alice, "finalize", 200, event_times[1])
            self._add_ledger(db, bob, "consume", 300, event_times[2])
            self._add_ledger(db, admin, "consume", 500, event_times[3])

        refresh_due_leaderboards(force=True, now=now)

        supporters = leaderboard_payload(SUPPORTERS_BOARD)
        self.assertEqual([entry["display_name"] for entry in supporters["entries"]], ["Alice", "Bob"])
        self.assertTrue(all("score" not in entry for entry in supporters["entries"]))

        lifetime = leaderboard_payload(TOKEN_LIFETIME_BOARD)
        self.assertEqual(
            [(entry["display_name"], entry["score"]) for entry in lifetime["entries"]],
            [("user0", 0.5), ("Bob", 0.3), ("Alice", 0.2)],
        )
        weekly = leaderboard_payload(TOKEN_LAST_WEEK_BOARD)
        self.assertEqual(
            [(entry["display_name"], entry["score"]) for entry in weekly["entries"]],
            [("user0", 0.5), ("Bob", 0.3), ("Alice", 0.2)],
        )


if __name__ == "__main__":
    unittest.main()
