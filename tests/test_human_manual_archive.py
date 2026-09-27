import gzip
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.admin.routes import router as admin_router
from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import create_session, iso
from backend.human_play import manual_archive, service
from backend.human_play.routes import router as human_router
from backend.human_play.store import database, init_db


FIXTURE = (Path(__file__).resolve().parents[1] / "frontend" / "tests" / "fixtures" /
           "verse-replay" / "Blueawa_3x4_2026-09-20_71356.vrs")


class ManualArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CLOUD_AUTH_DB": self.temp.name + "/auth.db",
            "HUMAN_PLAY_DB": self.temp.name + "/human.db",
            "ADMIN_ALLOWED_IDENTITIES": "player one",
        })
        self.env.start()
        init_auth_db(); init_db()
        with auth_db() as db:
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(1,'player@test.invalid','!','Player One',?,?)""", (iso(), iso()))
            self.token = create_session(db, 1)[0]
        self.replay = FIXTURE.read_bytes()

    def tearDown(self):
        self.env.stop(); self.temp.cleanup()

    def test_validated_application_enters_all_time_facts_but_not_rolling_board(self):
        application = manual_archive.submit(
            1, "3x4", time.time() - 3600, 71356, "record.vrs", self.replay)
        self.assertEqual((application["status"], application["moves"]), ("pending", 3244))
        self.assertEqual(service.history(1, 1)["entries"], [])

        approved = manual_archive.decide(application["id"], 99, True, "Replay reviewed")
        self.assertEqual(approved["status"], "approved")
        history = service.history(1, 1)["entries"]
        self.assertEqual((len(history), history[0]["score"], history[0]["source"]),
                         (1, 71356, "manual"))
        self.assertEqual(service.personal_bests(1)["bests"]["3x4"], 71356)
        self.assertTrue(gzip.decompress(service.replay(approved["run_id"], 1)).startswith(b"RPL1"))
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM rolling_candidates").fetchone()[0], 0)
            self.assertEqual(db.execute("""SELECT count(*) FROM human_archive_application_audit
                WHERE application_id=?""", (application["id"],)).fetchone()[0], 2)

        revoked = manual_archive.revoke(application["id"], 99, "Approval withdrawn")
        self.assertEqual(revoked["status"], "revoked")
        self.assertEqual(service.history(1, 1)["entries"], [])
        self.assertEqual(service.personal_bests(1)["bests"].get("3x4", 0), 0)

    def test_score_mismatch_is_rejected_before_application_is_created(self):
        with self.assertRaisesRegex(service.RunError, "replay_score_mismatch"):
            manual_archive.submit(1, "3x4", time.time(), 71357, "record.vrs", self.replay)
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM human_archive_applications").fetchone()[0], 0)

    def test_utf8_saved_verse_text_replay_is_accepted(self):
        utf8_replay = self.replay.decode("latin-1").encode("utf-8")
        application = manual_archive.submit(
            1, "3x4", time.time(), 71356, "record.txt", utf8_replay)
        self.assertEqual((application["status"], application["moves"]), ("pending", 3244))

    def test_rejection_is_audited_and_allows_a_new_application(self):
        first = manual_archive.submit(1, "3x4", time.time(), 71356, "one.vrs", self.replay)
        rejected = manual_archive.decide(first["id"], 99, False, "Insufficient evidence")
        self.assertEqual(rejected["status"], "rejected")
        second = manual_archive.submit(1, "3x4", time.time() - 1, 71356, "two.vrs", self.replay)
        self.assertNotEqual(second["id"], first["id"])

    def test_player_upload_and_admin_decision_api(self):
        app = FastAPI(); app.include_router(human_router); app.include_router(admin_router)
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + self.token
            query = (f"variant=3x4&ended_at={time.time()-60}&score=71356"
                     "&filename=record.vrs")
            submitted = client.post("/api/human/me/archive-applications?" + query,
                                    content=self.replay,
                                    headers={"Content-Type": "application/octet-stream"})
            self.assertEqual(submitted.status_code, 201)
            application_id = submitted.json()["application"]["id"]
            self.assertEqual(client.get("/api/human/me/archive-applications").status_code, 200)
            listed = client.get("/api/admin/archive-applications")
            self.assertEqual(listed.json()["applications"][0]["id"], application_id)
            approved = client.post(f"/api/admin/archive-applications/{application_id}/decision",
                json={"approved": True, "note": "Replay and claimed date reviewed"})
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(approved.json()["application"]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
