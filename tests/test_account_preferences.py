import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import create_session, iso
from backend.profile.routes import router


class AccountPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"CLOUD_AUTH_DB": self.temp.name + "/auth.db"})
        self.environment.start()
        init_auth_db()
        with auth_db() as db:
            for user_id in (1, 2):
                db.execute(
                    "INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (user_id, f"{user_id}@test.invalid", "!disabled", f"Player {user_id}", iso(), iso()),
                )
            self.tokens = [create_session(db, user_id)[0] for user_id in (1, 2)]
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.url = "/api/profile/preferences"

    def tearDown(self):
        self.client.close()
        self.environment.stop()
        self.temp.cleanup()

    def headers(self, index=0):
        return {"Authorization": "Bearer " + self.tokens[index]}

    def test_account_isolation_and_partial_updates(self):
        self.assertEqual(self.client.get(self.url, headers=self.headers()).json(),
                         {"preferences": {}, "revision": 0})
        first = self.client.patch(self.url, headers=self.headers(), json={
            "preferences": {"language": "zh", "theme": "Default", "showSpeed": True}
        })
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["revision"], 1)
        second = self.client.patch(self.url, headers=self.headers(), json={
            "preferences": {"dark_mode": True}
        }).json()
        self.assertEqual(second["preferences"], {
            "language": "zh", "theme": "Default", "showSpeed": True, "dark_mode": True,
        })
        self.assertEqual(self.client.get(self.url, headers=self.headers(1)).json(),
                         {"preferences": {}, "revision": 0})

    def test_first_login_migration_only_fills_missing_fields(self):
        self.client.patch(self.url, headers=self.headers(), json={"preferences": {"language": "en"}})
        result = self.client.patch(self.url, headers=self.headers(), json={
            "preferences": {"language": "zh", "dark_mode": True}, "only_if_missing": True,
        }).json()
        self.assertEqual(result["preferences"], {"language": "en", "dark_mode": True})
        self.assertEqual(result["revision"], 2)

    def test_rejects_unknown_or_invalid_settings(self):
        invalid = (
            {"demo_speed": 1}, {"swipeSensitivity": 200}, {"language": "xx"},
            {"theme": "unknown"}, {"dark_mode": 1}, {"font_size_factor": 151},
            {"custom_colors": ["#ffffff"]},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                response = self.client.patch(self.url, headers=self.headers(), json={"preferences": changes})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get(self.url, headers=self.headers()).json()["preferences"], {})

    def test_requires_account(self):
        self.assertEqual(self.client.get(self.url).status_code, 401)
        self.assertEqual(self.client.patch(self.url, json={"preferences": {"language": "en"}}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
