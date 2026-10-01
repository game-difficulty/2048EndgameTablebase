import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import create_session, iso
from backend.profile.routes import router


def theme_payload(background="#123456"):
    style = {
        "--tile-color": "#ffffff",
        "--tile-background": background,
        "--tile-shadow-color": "#00000000",
        "--tile-outline-color": "#ffffff22",
    }
    return {mode: {str(2 ** exponent): dict(style) for exponent in range(1, 17)} for mode in ("light", "dark")}


class SavedThemeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"CLOUD_AUTH_DB": self.temp.name + "/auth.db"})
        self.environment.start()
        init_auth_db()
        with auth_db() as db:
            for user_id in (1, 2):
                db.execute(
                    "INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (user_id, f"{user_id}@test.invalid", "!disabled", f"Player {user_id}", iso(), iso()),
                )
            self.tokens = [create_session(db, user_id)[0] for user_id in (1, 2)]
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.environment.stop()
        self.temp.cleanup()

    def headers(self, index=0):
        return {"Authorization": "Bearer " + self.tokens[index]}

    def test_crud_is_account_owned_and_includes_65k(self):
        created = self.client.post("/api/profile/themes", headers=self.headers(), json={"name": "Verse", "theme": theme_payload()})
        self.assertEqual(created.status_code, 201)
        theme_id = created.json()["id"]
        self.assertIn("65536", created.json()["theme"]["light"])
        self.assertEqual(self.client.get(f"/api/profile/themes/{theme_id}", headers=self.headers(1)).status_code, 404)
        self.assertEqual(len(self.client.get("/api/profile/themes", headers=self.headers()).json()["themes"]), 1)
        updated = self.client.put(f"/api/profile/themes/{theme_id}", headers=self.headers(), json={"name": "Verse 2", "theme": theme_payload("#654321")})
        self.assertEqual(updated.json()["theme"]["dark"]["65536"]["--tile-background"], "#654321")
        self.assertEqual(self.client.patch("/api/profile/preferences", headers=self.headers(), json={"preferences": {"saved_theme_id": theme_id}}).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/profile/themes/{theme_id}", headers=self.headers()).status_code, 204)
        self.assertEqual(self.client.get("/api/profile/preferences", headers=self.headers()).json()["preferences"]["saved_theme_id"], 0)

    def test_invalid_theme_reference_has_specific_error_without_accepting_other_invalid_settings(self):
        response = self.client.patch("/api/profile/preferences", headers=self.headers(),
                                     json={"preferences": {"saved_theme_id": 999, "dark_mode": True}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "invalid_saved_theme")
        response = self.client.patch("/api/profile/preferences", headers=self.headers(),
                                     json={"preferences": {"saved_theme_id": 999, "dark_mode": "invalid"}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "invalid_preferences")
        self.assertEqual(self.client.get("/api/profile/preferences", headers=self.headers()).json()["preferences"], {})

    def test_rejects_131k_and_duplicate_names(self):
        payload = theme_payload()
        payload["light"]["131072"] = dict(payload["light"]["65536"])
        self.assertEqual(self.client.post("/api/profile/themes", headers=self.headers(), json={"name": "Bad", "theme": payload}).status_code, 400)
        good = theme_payload()
        self.assertEqual(self.client.post("/api/profile/themes", headers=self.headers(), json={"name": "Same", "theme": good}).status_code, 201)
        self.assertEqual(self.client.post("/api/profile/themes", headers=self.headers(), json={"name": "same", "theme": good}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
