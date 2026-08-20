import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth.db import init_auth_db
from backend.auth.dependencies import require_user
from backend.replay_routes import router as replay_router
from backend.quota.service import (
    consume_operation_tokens_once,
    get_token_balance,
)
from backend.auth.db import auth_db
from backend.tester import LATEST_TESTER_REPLAY_BY_SCOPE
from engine_core.replay_utils import REPLAY_DTYPE


class ReplayQuotaIdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_db = os.environ.get("CLOUD_AUTH_DB")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        init_auth_db()
        with auth_db() as db:
            now = "2026-08-18T00:00:00+00:00"
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("replay@example.com", "replay@example.com", "test", "Replay", now, now),
            )
            self.user_id = int(cursor.lastrowid)
            session = db.execute(
                """
                INSERT INTO sessions
                (user_id, session_token_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.user_id, "replay-test-session", now, "2027-08-18T00:00:00+00:00"),
            )
            self.session_id = int(session.lastrowid)
            db.execute(
                """
                INSERT INTO token_accounts
                (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?)
                """,
                (self.user_id, 10_000, now, now),
            )

    def tearDown(self):
        if self.previous_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.previous_db
        LATEST_TESTER_REPLAY_BY_SCOPE.clear()
        self.tempdir.cleanup()

    def test_same_request_only_consumes_once(self):
        first = consume_operation_tokens_once(
            request_id="replay-request-1",
            user_id=self.user_id,
            session_id=None,
            operation_key="replay_load",
        )
        second = consume_operation_tokens_once(
            request_id="replay-request-1",
            user_id=self.user_id,
            session_id=None,
            operation_key="replay_load",
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(get_token_balance(self.user_id)["total"], 7)
        with auth_db() as db:
            count = db.execute(
                "SELECT COUNT(*) AS count FROM token_ledger WHERE operation_key = 'replay_load'"
            ).fetchone()["count"]
        self.assertEqual(count, 1)

    def test_http_routes_charge_once_and_return_latest_binary(self):
        app = FastAPI()
        app.include_router(replay_router)
        user = {
            "id": self.user_id,
            "session_id": self.session_id,
            "email": "replay@example.com",
        }
        app.dependency_overrides[require_user] = lambda: user
        client = TestClient(app)

        request_id = str(uuid4())
        first = client.post(
            "/api/replay/load-authorization",
            json={"request_id": request_id, "filename": "sample.rpl", "size": 50},
        )
        repeated = client.post(
            "/api/replay/load-authorization",
            json={"request_id": request_id, "filename": "sample.rpl", "size": 50},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["token_balance"]["total"], 7)

        record = np.zeros(1, dtype=REPLAY_DTYPE)
        record[0]["f0"] = np.uint64(0x0210030129AB4CDE)
        record[0]["f1"] = np.uint8(32)
        LATEST_TESTER_REPLAY_BY_SCOPE[f"session:{self.session_id}"] = {
            "record": record,
            "pattern": "L3_256",
            "source": "Tester session",
            "use_variant": False,
            "terminal_board": np.uint64(0x1021003129AB4CDE),
            "goodness_of_fit": 0.98765,
        }
        latest = client.post(
            "/api/replay/latest",
            json={"request_id": str(uuid4())},
        )
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(len(latest.content), REPLAY_DTYPE.itemsize * 2)
        self.assertEqual(latest.headers["x-replay-pattern"], "L3_256")
        self.assertIn(
            'filename="L3_256_0.9877.rpl"',
            latest.headers["content-disposition"],
        )
        self.assertEqual(latest.headers["x-token-total"], "4.0")

        LATEST_TESTER_REPLAY_BY_SCOPE.clear()
        missing = client.post(
            "/api/replay/latest",
            json={"request_id": str(uuid4())},
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(get_token_balance(self.user_id)["total"], 4)

    def test_oversized_latest_replay_is_rejected_without_charge(self):
        app = FastAPI()
        app.include_router(replay_router)
        user = {
            "id": self.user_id,
            "session_id": self.session_id,
            "email": "replay@example.com",
        }
        app.dependency_overrides[require_user] = lambda: user
        client = TestClient(app)

        oversized_moves = (500 * 1024) // REPLAY_DTYPE.itemsize + 1
        record = np.zeros(oversized_moves, dtype=REPLAY_DTYPE)
        LATEST_TESTER_REPLAY_BY_SCOPE[f"user:{self.user_id}"] = {
            "record": record,
            "pattern": "L3_256",
            "source": "Tester session",
            "use_variant": False,
            "terminal_board": None,
        }
        before = get_token_balance(self.user_id)["total"]

        response = client.post(
            "/api/replay/latest",
            json={"request_id": str(uuid4())},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "REPLAY_TOO_LARGE")
        self.assertEqual(get_token_balance(self.user_id)["total"], before)


if __name__ == "__main__":
    unittest.main()
