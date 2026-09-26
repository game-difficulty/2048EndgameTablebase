import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.admin.routes import router as admin_router
from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import create_session, iso
from backend.human_play.store import database, init_db


class AdminApprovalTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CLOUD_AUTH_DB": self.temp.name + "/auth.db",
            "HUMAN_PLAY_DB": self.temp.name + "/human.db",
            "ADMIN_ALLOWED_IDENTITIES": "site admin",
        })
        self.env.start()
        init_auth_db()
        init_db()
        with auth_db() as db:
            now = iso()
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(1,'admin@test.invalid','!','Site Admin',?,?)""", (now, now))
            db.execute("""INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at)
                VALUES(2,'aurora@test.invalid','!','Aurora',?,?)""", (now, now))
            self.token = create_session(db, 1)[0]
        now = time.time()
        with database() as db:
            db.execute("""INSERT INTO human_external_claims
                (id,user_id,provider,username,username_key,status,requested,updated,counts)
                VALUES(11,2,'verse','Verse_Aurora','verse_aurora','pending',?,?,?)""",
                (now - 30, now - 10, json.dumps({"4x4": 18})))
            db.execute("""INSERT INTO human_archive_applications
                (id,user_id,variant,claimed_ended_at,claimed_score,status,replay_crc,replay_size,
                 moves,final_board_json,is_game_over,timing_summary_json,warning_flags_json,
                 original_filename,requested_at,updated_at)
                VALUES(12,2,'4x4',830000,830440,'rejected',1,100,29304,?,1,'{}','[]',
                       'aurora.rpl',?,?)""",
                (json.dumps([0] * 16), now - 60, now - 20))

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def client(self):
        app = FastAPI()
        app.include_router(admin_router)
        client = TestClient(app)
        client.headers["Authorization"] = "Bearer " + self.token
        return client

    def test_unified_queue_is_paged_without_reading_replay_payloads(self):
        with self.client() as client:
            response = client.get("/api/admin/approval-transactions?page_size=1")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["page"], {"page": 1, "page_size": 1, "total": 2, "page_count": 2})
        item = payload["transactions"][0]
        self.assertEqual((item["kind"], item["transaction_id"], item["user"]["display_name"]),
                         ("verse", 11, "Aurora"))
        self.assertEqual(item["actions"], ["approve", "reject"])
        self.assertNotIn("archive", item)

    def test_filters_and_account_identity_search_apply_before_paging(self):
        with self.client() as client:
            by_user = client.get("/api/admin/approval-transactions?q=aurora@test.invalid").json()
            rejected = client.get(
                "/api/admin/approval-transactions?kind=archive&stage=rejected").json()
        self.assertEqual({item["transaction_id"] for item in by_user["transactions"]}, {11, 12})
        self.assertEqual([item["transaction_id"] for item in rejected["transactions"]], [12])
        self.assertEqual(rejected["transactions"][0]["actions"], [])


if __name__ == "__main__":
    unittest.main()
