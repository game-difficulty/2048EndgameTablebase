from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from starlette.requests import Request

from backend.auth.db import auth_db, init_auth_db
from backend.auth.dependencies import current_actor_from_request
from backend.auth.guest_service import (
    GuestLimitError,
    authenticate_guest_token,
    cleanup_expired_guest_sessions,
    finalize_guest_query,
    guest_query_allowance,
    issue_guest_session,
    reserve_guest_query,
)
from backend.auth.principal import ActorRef
from backend.auth.service import create_session


class GuestSessionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_env = {
            key: os.environ.get(key)
            for key in (
                "CLOUD_AUTH_DB",
                "GUEST_QUERY_ALLOWANCE",
                "GUEST_IP_QUERY_LIMIT",
                "GUEST_SESSION_ISSUE_LIMIT_PER_DAY",
                "GUEST_IP_HASH_SECRET",
            )
        }
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        os.environ["GUEST_QUERY_ALLOWANCE"] = "5"
        os.environ["GUEST_IP_QUERY_LIMIT"] = "15"
        os.environ["GUEST_SESSION_ISSUE_LIMIT_PER_DAY"] = "5"
        os.environ["GUEST_IP_HASH_SECRET"] = "test-only-secret"
        init_auth_db()

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tempdir.cleanup()

    def test_guest_is_separate_from_users_and_token_accounts(self):
        result = issue_guest_session(ip_address="203.0.113.10")
        guest = authenticate_guest_token(result["token"], ip_address="203.0.113.10")
        self.assertIsNotNone(guest)
        self.assertEqual(guest["query_allowance"]["remaining"], 5)
        actor = ActorRef.from_guest(guest)
        self.assertTrue(actor.is_guest)
        self.assertTrue(actor.actor_key.startswith("g:"))
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM token_accounts").fetchone()[0], 0)

    def test_query_reservation_is_idempotent_and_miss_consumes(self):
        result = issue_guest_session(ip_address="203.0.113.11")
        guest_id = result["guest"]["guest_id"]
        first = reserve_guest_query(
            guest_id=guest_id,
            request_id="request-1",
            full_pattern="L3_128",
            ip_address="203.0.113.11",
        )
        duplicate = reserve_guest_query(
            guest_id=guest_id,
            request_id="request-1",
            full_pattern="L3_128",
            ip_address="203.0.113.11",
        )
        self.assertEqual(first.event_id, duplicate.event_id)
        self.assertEqual(guest_query_allowance(guest_id)["remaining"], 4)
        balance = finalize_guest_query(first, consume=True)
        self.assertEqual(balance["remaining"], 4)

    def test_expired_guest_cleanup_cascades_query_events(self):
        result = issue_guest_session(ip_address="203.0.113.14")
        guest_id = result["guest"]["guest_id"]
        reservation = reserve_guest_query(
            guest_id=guest_id,
            request_id="expired-query",
            full_pattern="L3_128",
            ip_address="203.0.113.14",
        )
        finalize_guest_query(reservation, consume=True)
        with auth_db() as db:
            db.execute(
                "UPDATE guest_sessions SET expires_at = ? WHERE guest_id = ?",
                ("2020-01-01T00:00:00+00:00", guest_id),
            )
        self.assertEqual(cleanup_expired_guest_sessions(), 1)
        with auth_db() as db:
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM guest_query_events WHERE guest_id = ?", (guest_id,)
                ).fetchone()
            )

    def test_registered_session_takes_priority_over_guest_credentials(self):
        guest_result = issue_guest_session(ip_address="203.0.113.15")
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES ('user@example.com', 'user@example.com', 'hash', 'User',
                        0, 'user', 'active', ?, ?)
                """,
                (now, now),
            )
            user_token, _session_id, _expires_at = create_session(
                db, int(cursor.lastrowid), user_agent="test", ip_address="203.0.113.15"
            )
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [
                    (b"authorization", f"Bearer {user_token}".encode("ascii")),
                    (b"x-guest-token", guest_result["token"].encode("ascii")),
                ],
                "client": ("203.0.113.15", 1234),
            }
        )
        actor = current_actor_from_request(request)
        self.assertIsNotNone(actor)
        self.assertTrue(actor.is_user)
        self.assertEqual(actor.actor_key, f"u:{actor.user_id}")

    def test_server_failure_refunds_and_same_request_can_retry(self):
        result = issue_guest_session(ip_address="203.0.113.12")
        guest_id = result["guest"]["guest_id"]
        first = reserve_guest_query(
            guest_id=guest_id,
            request_id="retry-me",
            full_pattern="L3_128",
            ip_address="203.0.113.12",
        )
        self.assertEqual(finalize_guest_query(first, consume=False)["remaining"], 5)
        retried = reserve_guest_query(
            guest_id=guest_id,
            request_id="retry-me",
            full_pattern="L3_128",
            ip_address="203.0.113.12",
        )
        self.assertNotEqual(first.event_id, retried.event_id)
        self.assertEqual(finalize_guest_query(retried, consume=True)["remaining"], 4)

    def test_guest_and_network_limits_are_atomic(self):
        os.environ["GUEST_QUERY_ALLOWANCE"] = "2"
        os.environ["GUEST_IP_QUERY_LIMIT"] = "2"
        first_guest = issue_guest_session(ip_address="203.0.113.13")["guest"]["guest_id"]
        second_guest = issue_guest_session(ip_address="203.0.113.13")["guest"]["guest_id"]
        for index in range(2):
            reservation = reserve_guest_query(
                guest_id=first_guest,
                request_id=f"first-{index}",
                full_pattern="L3_128",
                ip_address="203.0.113.13",
            )
            finalize_guest_query(reservation, consume=True)
        with self.assertRaises(GuestLimitError) as guest_error:
            reserve_guest_query(
                guest_id=first_guest,
                request_id="over-guest",
                full_pattern="L3_128",
                ip_address="203.0.113.13",
            )
        self.assertEqual(guest_error.exception.code, "GUEST_QUERY_ALLOWANCE_EXHAUSTED")
        with self.assertRaises(GuestLimitError) as network_error:
            reserve_guest_query(
                guest_id=second_guest,
                request_id="over-network",
                full_pattern="L3_128",
                ip_address="203.0.113.13",
            )
        self.assertEqual(network_error.exception.code, "GUEST_NETWORK_QUERY_LIMIT_REACHED")


if __name__ == "__main__":
    unittest.main()
