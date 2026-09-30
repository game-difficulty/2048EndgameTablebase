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

    def submit(self, *args, **kwargs):
        kwargs.setdefault('started_at', time.time() - 86400)
        return manual_archive.submit(*args, **kwargs)

    def test_start_time_is_required_and_persisted_through_approval(self):
        end = time.time() - 60
        for invalid in [None, float('nan'), float('inf'), end + 1, 0]:
            with self.assertRaisesRegex(service.RunError, 'invalid_started_at'):
                self.submit(1, '3x4', end, 71356, 'record.vrs', self.replay, started_at=invalid)
        start = end - 3600
        application = self.submit(1, '3x4', end, 71356, 'record.vrs', self.replay, started_at=start)
        self.assertEqual(application['started_at'], start)
        approved = manual_archive.decide(application['id'], 99, True, 'Checked')
        with database() as db:
            row = db.execute('SELECT first_move_at,created,ended FROM human_runs WHERE id=?', (approved['run_id'],)).fetchone()
            self.assertEqual(tuple(row), (start, start, end))

    def test_duration_boundary_rejects_shorter_and_accepts_equal(self):
        result = manual_archive.inspect_upload(self.replay, '3x4', 71356)
        end = 1790000000.0
        start = end - result['elapsed'] / 1000
        with self.assertRaisesRegex(service.RunError, 'archive_duration_too_short'):
            self.submit(1, '3x4', end, 71356, 'record.vrs', self.replay, started_at=start + 0.001)
        application = self.submit(1, '3x4', end, 71356, 'record.vrs', self.replay, started_at=start)
        self.assertEqual(manual_archive.decide(application['id'], 99, True, 'Checked')['status'], 'approved')

    def test_approval_recomputes_duration_instead_of_trusting_cached_summary(self):
        application = self.submit(1, '3x4', time.time(), 71356, 'record.vrs', self.replay)
        with database() as db:
            db.execute("UPDATE human_archive_applications SET claimed_started_at=claimed_ended_at, timing_summary_json=? WHERE id=?",
                       ('{"elapsed_ms":0}', application['id']))
        with self.assertRaisesRegex(service.RunError, 'archive_duration_too_short'):
            manual_archive.decide(application['id'], 99, True, 'Checked')
        with database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM human_runs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT status FROM human_archive_applications WHERE id=?', (application['id'],)).fetchone()[0], 'pending')

    def test_unknown_move_times_are_excluded_from_duration_bound(self):
        from backend.human_play import verse_replay
        normalized = manual_archive.inspect_upload(self.replay, '3x4', 71356)['normalized']
        variant, board, moves = verse_replay._rpl1(normalized)
        moves = [(*move[:3], 1234 if i == 0 else verse_replay.UNKNOWN_TIMING_MS) for i, move in enumerate(moves)]
        replay = verse_replay.encode_rpl1(variant, board, moves)
        end = 1790000000.0
        application = self.submit(1, variant, end, 71356, 'record.vrs', replay, started_at=end-1.234)
        self.assertEqual(application['timing'], {'elapsed_ms':1234, 'timed_moves':1})
        self.assertEqual(manual_archive.decide(application['id'], 99, True, 'Checked')['status'], 'approved')

    def test_legacy_application_without_start_time_cannot_bypass_review(self):
        application = self.submit(1, '3x4', time.time(), 71356, 'record.vrs', self.replay)
        with database() as db:
            db.execute('UPDATE human_archive_applications SET claimed_started_at=NULL WHERE id=?', (application['id'],))
        with self.assertRaisesRegex(service.RunError, 'invalid_started_at'):
            manual_archive.decide(application['id'], 99, True, 'Checked')

    def tearDown(self):
        self.env.stop(); self.temp.cleanup()

    def test_validated_application_enters_all_time_facts_but_not_rolling_board(self):
        application = self.submit(
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
            self.submit(1, "3x4", time.time(), 71357, "record.vrs", self.replay)
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM human_archive_applications").fetchone()[0], 0)

    def test_utf8_saved_verse_text_replay_is_accepted(self):
        utf8_replay = self.replay.decode("latin-1").encode("utf-8")
        application = self.submit(
            1, "3x4", time.time(), 71356, "record.txt", utf8_replay)
        self.assertEqual((application["status"], application["moves"]), ("pending", 3244))

    def test_rejection_is_audited_and_allows_a_new_application(self):
        first = self.submit(1, "3x4", time.time(), 71356, "one.vrs", self.replay)
        rejected = manual_archive.decide(first["id"], 99, False, "Insufficient evidence")
        self.assertEqual(rejected["status"], "rejected")
        second = self.submit(1, "3x4", time.time() - 1, 71356, "two.vrs", self.replay)
        self.assertNotEqual(second["id"], first["id"])

    def test_player_upload_and_admin_decision_api(self):
        app = FastAPI(); app.include_router(human_router); app.include_router(admin_router)
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + self.token
            query = (f"variant=3x4&started_at={time.time()-86400}&ended_at={time.time()-60}&score=71356"
                     "&filename=record.vrs")
            missing = client.post(f"/api/human/me/archive-applications?variant=3x4&ended_at={time.time()}&score=71356",
                content=self.replay, headers={"Content-Type": "application/octet-stream"})
            self.assertEqual(missing.status_code, 422)
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
