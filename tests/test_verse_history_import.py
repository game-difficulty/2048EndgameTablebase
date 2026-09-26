import gzip
import json
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
from backend.auth.service import authenticate_session_token, create_session, iso
from backend.human_play import admin, engine, rating, service, verse_history, verse_replay
from backend.human_play.routes import router as human_router
from backend.human_play.store import database, init_db
from backend import rolling_leaderboards


def varint(value):
    result = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        result.append(byte | (128 if value else 0))
        if not value:
            return result


def segment(variant, score, game_id, played_ms):
    return segment_records(variant, [(game_id, played_ms, score)], played_ms + 60_000)


def segment_records(variant, entries, collected_ms):
    rows, cols = engine.VARIANTS[variant]
    board = [0] * (rows * cols)
    board[0] = 11
    packed = sum(value << (i * 5) for i, value in enumerate(board))
    output = bytearray(b"VHS2")
    output.extend((verse_history.VARIANTS.index(variant), 1))
    for value in (collected_ms, len(entries), len(entries), len(entries), max((r[2] for r in entries), default=0)):
        output.extend(varint(value))
    prior = 0
    for index, (game_id, played_ms, score) in enumerate(entries):
        for value in (played_ms if index == 0 else played_ms - prior, game_id, score):
            output.extend(varint(value))
        output.extend(packed.to_bytes((len(board) * 5 + 7) // 8, "little"))
        prior = played_ms
    return bytes(output)


class VerseHistoryImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CLOUD_AUTH_DB": self.temp.name + "/auth.db",
            "HUMAN_PLAY_DB": self.temp.name + "/human.db",
            "HUMAN_VERSE_ARCHIVE_ROOT": self.temp.name + "/archive",
            "ADMIN_ALLOWED_IDENTITIES": "player one",
        })
        self.env.start()
        init_auth_db()
        init_db()
        with auth_db() as db:
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES (1,'player@test.invalid','!disabled','Player One',?,?)""", (iso(), iso()))
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES (2,'other@test.invalid','!disabled','Player Two',?,?)""", (iso(), iso()))
            self.token = create_session(db, 1)[0]
        folder = Path(self.temp.name) / "archive" / "Verse_Player"
        folder.mkdir(parents=True)
        played = int(time.time() * 1000) - 100_000
        for index, variant in enumerate(verse_history.VARIANTS):
            (folder / f"{variant}.vhs").write_bytes(segment(variant, 5000 + index, 100 + index, played))

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_claim_imports_into_native_run_table_and_ranks_without_replay(self):
        claim = verse_history.request_claim(1, "Verse_Player")
        self.assertEqual(verse_history.request_claim(1, "verse_player")["id"], claim["id"])
        self.assertEqual(service.leaderboard("4x4")["entries"], [])
        verse_history.decide_claim(claim["id"], 99, True, "Live account control verified")
        with patch.object(verse_history, "_fetch") as fetch:
            verse_history.process_approved()
        fetch.assert_called_once_with("Verse_Player", incremental=True)
        current = verse_history.own_claim(1)
        self.assertEqual(current["status"], "complete")
        self.assertEqual(sum(current["counts"].values()), 4)
        entries = service.leaderboard("4x4")["entries"]
        self.assertEqual((len(entries), entries[0]["score"], entries[0]["source"]), (1, 5000, "verse"))
        self.assertEqual(service.personal_bests(1)["bests"]["4x4"], 5000)
        history = service.history(1, 1)["entries"]
        self.assertEqual(len(history), 4)
        self.assertTrue(all(row["source"] == "verse" and not row["has_replay"] for row in history))
        self.assertEqual({len(row["board"]) for row in history}, {8, 9, 12, 16})
        best = service.best_ten(1, 1, "4x4")
        self.assertEqual(len(best["entries"][0]["board"]), 16)
        self.assertAlmostEqual(best['rating'], rating.single_rating('4x4', best['entries'][0]['board']))
        self.assertEqual((best['rating_games'], best['pb_rank'], best['ra_rank']), (1, 1, 1))
        stats = service.player_statistics(1, '4x4')
        self.assertEqual(set(stats['summaries']), set(verse_history.VARIANTS))
        self.assertEqual(stats['summaries']['4x4']['game_count'], 1)
        self.assertEqual(stats['summaries']['4x4']['rate_32k']['covered_games'], 0)
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM human_run_statistics").fetchone()[0], 4)
        run_id = history[0]["id"]
        with self.assertRaisesRegex(service.RunError, "run_not_found"):
            service.status(1, f"verse:{claim['id']}", run_id)
        with self.assertRaisesRegex(service.RunError, "replay_not_found"):
            service.replay(run_id, 1)
        with self.assertRaisesRegex(service.RunError, "native_run_required"):
            admin.inspect(run_id)
        with patch.object(verse_history, "_fetch"):
            verse_history.process_approved()
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM human_runs WHERE source='verse'").fetchone()[0], 4)
        verse_history.revoke_claim(claim["id"], 99, "Claim ownership was revoked")
        self.assertEqual(service.leaderboard("4x4")["entries"], [])
        self.assertEqual(service.history(1, 1)["entries"], [])
        self.assertIsNone(service.best_ten(1, 1, '4x4')['rating'])

    def test_incomplete_segment_cannot_be_published(self):
        path = Path(self.temp.name) / "archive" / "Verse_Player" / "4x4.vhs"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            verse_history.snapshot("Verse_Player")

    def test_refresh_is_applied_before_all_four_modes_become_visible(self):
        claim = verse_history.request_claim(1, "Verse_Player")
        verse_history.decide_claim(claim["id"], 99, True, "Live account control verified")
        path = Path(self.temp.name) / "archive" / "Verse_Player" / "4x4.vhs"
        original = verse_history.decode_segment(path, "4x4")["records"][0]

        def refresh(username, *, incremental=False):
            self.assertEqual((username, incremental), ("Verse_Player", True))
            second = (200, original[1] + 1000, 9000)
            path.write_bytes(segment_records("4x4", [original[:3], second], second[1] + 60000))

        with patch.object(verse_history, "_fetch", side_effect=refresh):
            verse_history.process_approved()
        self.assertEqual(verse_history.own_claim(1)["counts"]["4x4"], 2)
        self.assertEqual(service.personal_bests(1)["bests"]["4x4"], 9000)
        with database() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM human_runs WHERE source='verse' AND visible=1").fetchone()[0], 5)

    def test_active_bulk_collector_defers_instead_of_failing_claim(self):
        claim = verse_history.request_claim(1, "Verse_Player")
        verse_history.decide_claim(claim["id"], 99, True, "Live account control verified")
        with patch.object(verse_history, "_bulk_collector_running", return_value=True), \
                patch.object(verse_history, "_retry_after_collector") as retry:
            verse_history.process_approved()
        self.assertEqual(verse_history.own_claim(1)["status"], "approved")
        retry.assert_called_once()
        self.assertEqual(service.leaderboard("4x4")["entries"], [])

    def test_refresh_uses_canonical_cache_directory_without_live_requests(self):
        with patch.object(verse_history, "_bulk_collector_running", return_value=False), \
                patch.object(verse_history.shutil, "which", return_value="node"), \
                patch.object(verse_history.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = '{"variants":[]}'
            self.assertEqual(verse_history._fetch("verse_player", incremental=True), {"variants": []})
        arguments = run.call_args.args[0]
        self.assertTrue(arguments[1].endswith("refresh-verse-history-api.mjs"))
        self.assertEqual(arguments[arguments.index("--username") + 1], "Verse_Player")

    def test_remote_removal_is_retained_in_claim_audit(self):
        claim = verse_history.request_claim(1, "Verse_Player")
        verse_history.decide_claim(claim["id"], 99, True, "Live account control verified")
        report = {"variants": [{"file": "/cache/Verse_Player/4x4.vhs", "count": 5,
            "remoteTotal": 1, "remoteRemoved": 4,
            "remoteRemovedIds": [11, 12, 13, 14]}]}
        with patch.object(verse_history, "_fetch", return_value=report):
            verse_history.process_approved()
        with database() as db:
            audit = db.execute("""SELECT note FROM human_external_audit
                WHERE claim_id=? AND action='remote_removed'""", (claim["id"],)).fetchone()
        self.assertIsNotNone(audit)
        note = json.loads(audit["note"])
        self.assertEqual(note["variants"][0]["remote_removed"], 4)
        self.assertEqual(note["variants"][0]["remote_removed_ids"], [11, 12, 13, 14])

    def test_claim_and_admin_review_api(self):
        app = FastAPI()
        app.include_router(human_router)
        app.include_router(admin_router)
        with TestClient(app) as client, patch.object(verse_history, "start_worker"):
            client.headers["Authorization"] = "Bearer " + self.token
            self.assertIsNone(client.get("/api/human/me/verse-claim").json()["claim"])
            requested = client.post("/api/human/me/verse-claim", json={"username": "Verse_Player"})
            self.assertEqual(requested.status_code, 200)
            claim = requested.json()["claim"]
            self.assertEqual(claim["counts"]["4x4"], 1)
            listed = client.get("/api/admin/verse-claims")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["claims"][0]["id"], claim["id"])
            self.assertEqual(client.get("/api/admin/verse-claims?user_id=2").json()["claims"], [])
            self.assertEqual(client.get("/api/admin/verse-claims?user_id=1").json()["claims"][0]["id"], claim["id"])
            pending_users = client.get("/api/admin/overview?tier=pending").json()
            self.assertEqual([user["id"] for user in pending_users["users"]], [1])
            self.assertTrue(pending_users["users"][0]["pending_approval"])
            all_users = client.get("/api/admin/overview").json()["users"]
            self.assertTrue(next(user for user in all_users if user["id"] == 1)["pending_approval"])
            self.assertFalse(next(user for user in all_users if user["id"] == 2)["pending_approval"])
            approved = client.post(f"/api/admin/verse-claims/{claim['id']}/decision",
                json={"approved": True})
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(client.get("/api/admin/overview?tier=pending").json()["users"], [])
        with patch.object(verse_history, "_fetch"):
            verse_history.process_approved()
        self.assertEqual(verse_history.own_claim(1)["status"], "complete")
        with self.assertRaisesRegex(service.RunError, "verse_account_claimed"):
            verse_history.request_claim(2, "verse_player")

    def test_rejected_claim_can_be_replaced_with_a_different_username(self):
        first = verse_history.request_claim(1, "Verse_Player")
        rejected = verse_history.decide_claim(first["id"], 99, False,
                                               "Account ownership did not match")
        self.assertEqual(rejected["status"], "rejected")
        self.assertIsNone(verse_history.own_claim(1))

        replacement = verse_history.request_claim(1, "Another_Player")
        self.assertNotEqual(replacement["id"], first["id"])
        self.assertEqual((replacement["username"], replacement["status"]),
                         ("Another_Player", "pending"))
        self.assertEqual(verse_history.own_claim(1)["id"], replacement["id"])

    def test_admin_can_disable_and_reenable_another_account(self):
        now = time.time()
        with auth_db() as db:
            target_token = create_session(db, 2)[0]
            rolling_leaderboards.add(
                db, board_key="gamer_high_score_weekly", run_id="target-run",
                user_id=2, score=12345, achieved_at=now - 5, eligible_at=now - 5,
                now=now,
            )
            self.assertEqual(len(rolling_leaderboards.entries(db, "gamer_high_score_weekly", now=now)), 1)

        app = FastAPI()
        app.include_router(admin_router)
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + self.token
            disabled = client.post("/api/admin/users/2/status", json={"status": "disabled"})
            self.assertEqual(disabled.status_code, 200)
            self.assertEqual(disabled.json()["user"]["status"], "disabled")
            self.assertEqual(client.post("/api/admin/users/1/status", json={"status": "disabled"}).status_code, 400)
            enabled = client.post("/api/admin/users/2/status", json={"status": "active"})
            self.assertEqual(enabled.status_code, 200)
            self.assertEqual(enabled.json()["user"]["status"], "active")

        with auth_db() as db:
            revoked = db.execute("SELECT revoked_at FROM sessions WHERE user_id=2 LIMIT 1").fetchone()
            self.assertIsNotNone(revoked["revoked_at"])
            self.assertEqual(len(rolling_leaderboards.entries(db, "gamer_high_score_weekly", now=now + 1)), 1)
        self.assertIsNone(authenticate_session_token(target_token))

    def test_owner_can_attach_matching_verse_replay_without_changing_qualification(self):
        claim = verse_history.request_claim(1, "Verse_Player")
        verse_history.decide_claim(claim["id"], 99, True, "Live account control verified")
        with patch.object(verse_history, "_fetch"):
            verse_history.process_approved()
        with database() as db:
            row = db.execute("""SELECT id,state FROM human_runs
                WHERE source='verse' AND variant='3x4'""").fetchone()
            state = json.loads(row["state"])
            state["score"] = 71356
            state["board"] = [2, 128, 64, 2, 512, 256, 16, 4, 4096, 2048, 32, 8]
            db.execute("UPDATE human_runs SET state=? WHERE id=?", (json.dumps(state), row["id"]))
        replay = (Path(__file__).resolve().parents[1] / "frontend" / "tests" /
                  "fixtures" / "verse-replay" / "Blueawa_3x4_2026-09-20_71356.vrs").read_bytes()
        result = verse_replay.attach(row["id"], 1, replay)
        self.assertEqual(result["moves"], 3244)
        archive = service.replay(row["id"], 2)
        self.assertEqual(gzip.decompress(archive)[:4], b"RPL1")
        self.assertTrue(verse_replay.archived_to_analysis_text(archive, "3x4", 3244).startswith(verse_replay.PREFIX))
        self.assertEqual(service.personal_bests(1)["bests"]["3x4"], 71356)
        self.assertEqual(verse_replay.attach(row["id"], 1, replay), result)
        app = FastAPI()
        app.include_router(human_router)
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + self.token
            received = client.get(f"/api/human/replays/{row['id']}")
            self.assertEqual(received.status_code, 200)
            self.assertTrue(received.content.startswith(b"RPL1"))
            self.assertEqual(client.post(f"/api/human/verse-replays/{row['id']}", content=replay).status_code, 200)
        with self.assertRaisesRegex(service.RunError, "replay_run_not_found"):
            verse_replay.attach(row["id"], 2, replay)
        with self.assertRaisesRegex(ValueError, "replay_result_mismatch"):
            verse_replay.validate(replay, "3x4", 71357, state["board"])


if __name__ == "__main__":
    unittest.main()
