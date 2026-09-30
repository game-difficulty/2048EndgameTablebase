from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.admin.routes import router as admin_router
from backend.auth.db import auth_db, init_auth_db
from backend.auth.managed_test_accounts import provision_accounts
from backend.auth.service import create_session, iso, login_user, request_password_reset_code


class ManagedTestAccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CLOUD_AUTH_DB": self.temp.name + "/auth.sqlite3",
            "ADMIN_ALLOWED_IDENTITIES": "site admin",
        })
        self.env.start()
        init_auth_db()
        with auth_db() as db:
            now = iso()
            db.execute(
                """INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(1,'admin@test.invalid','!','Site Admin',?,?)""", (now, now)
            )
            db.execute(
                """INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(2,'ordinary@test.invalid','!','Ordinary',?,?)""", (now, now)
            )
            self.admin_token = create_session(db, 1)[0]
            self.user_token = create_session(db, 2)[0]

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def client(self, token):
        app = FastAPI()
        app.include_router(admin_router)
        client = TestClient(app)
        client.headers["Authorization"] = "Bearer " + token
        return client

    def test_provision_is_idempotent_and_uses_ordinary_account_defaults(self):
        created = provision_accounts()
        self.assertEqual(len(created), 6)
        self.assertEqual(provision_accounts(), [])
        with auth_db() as db:
            rows = db.execute(
                """SELECT id,email,email_verified_at,registered_with_invite,
                managed_test_account,role,status FROM users WHERE managed_test_account=1 ORDER BY id"""
            ).fetchall()
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["role"] == "user" and row["status"] == "active" for row in rows))
            self.assertTrue(all(row["email_verified_at"] is None and not row["registered_with_invite"] for row in rows))
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM user_quotas WHERE user_id=?", (rows[0]["id"],)
            ).fetchone()[0], 6)
        logged_in = login_user(email=created[0]["email"], password=created[0]["password"])
        self.assertEqual(logged_in["user"]["id"], created[0]["id"])
        with patch("backend.auth.service.send_verification_email") as send:
            self.assertTrue(request_password_reset_code(email=created[0]["email"])["sent"])
        send.assert_not_called()

    def test_admin_reset_revokes_sessions_and_rejects_other_accounts(self):
        created = provision_accounts()
        account = created[0]
        old_token = login_user(email=account["email"], password=account["password"])["token"]
        with self.client(self.user_token) as client:
            self.assertEqual(client.post(
                f"/api/admin/users/{account['id']}/managed-password",
                json={"new_password": "new-password-123"},
            ).status_code, 403)
        with self.client(self.admin_token) as client:
            self.assertEqual(client.post(
                "/api/admin/users/2/managed-password",
                json={"new_password": "new-password-123"},
            ).status_code, 404)
            self.assertEqual(client.post(
                f"/api/admin/users/{account['id']}/managed-password",
                json={"new_password": "short"},
            ).status_code, 422)
            response = client.post(
                f"/api/admin/users/{account['id']}/managed-password",
                json={"new_password": "new-password-123"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), {"ok": True})
        with auth_db() as db:
            self.assertIsNotNone(db.execute(
                "SELECT revoked_at FROM sessions WHERE session_token_hash=(SELECT session_token_hash FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT 1)",
                (account["id"],),
            ).fetchone()[0])
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM refresh_tokens WHERE user_id=? AND revoked_at IS NULL", (account["id"],)
            ).fetchone()[0], 0)
            audit = db.execute(
                "SELECT operator_id,action FROM managed_test_account_audit WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (account["id"],),
            ).fetchone()
            self.assertEqual(tuple(audit), (1, "password_reset"))
        with self.assertRaises(ValueError):
            login_user(email=account["email"], password=account["password"])
        self.assertEqual(login_user(email=account["email"], password="new-password-123")["user"]["id"], account["id"])
        self.assertTrue(old_token)


if __name__ == "__main__":
    unittest.main()
