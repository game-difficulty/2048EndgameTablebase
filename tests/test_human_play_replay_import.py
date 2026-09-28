import base64
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from backend.human_play.store import database, init_db
from tools.human_play_replay_import import (
    OLD_VERSE_ALPHABET,
    apply_plan,
    build_plan,
    inspect_2048next,
    inspect_old_verse,
)


def _next_replay() -> str:
    payload = bytearray(b"RPL1")
    payload.extend((0x44, 0, 2))
    payload.extend((5, 7))
    payload.extend((131, 2, 4))
    payload.extend(b"pow2")
    payload.extend((13, 10))  # right, spawn 2 at cell 3
    payload.extend((66, 20))  # down, spawn 4 at cell 0; then undo it
    payload.extend((128, 0))
    payload.extend((63, 30))  # left, spawn 2 at cell 15
    payload.append(132)
    payload.extend((zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little"))
    return "REPLAY_v1RPL_B64_" + base64.b64encode(payload).decode("ascii")


class HumanPlayReplayImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CLOUD_AUTH_DB": self.temp.name + "/auth.db",
            "HUMAN_PLAY_DB": self.temp.name + "/human.db",
        })
        self.env.start()
        init_db()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_2048next_is_folded_into_canonical_rpl1(self):
        result = inspect_2048next(_next_replay().encode("ascii"))
        self.assertEqual(result["variant"], "4x4")
        self.assertEqual(result["moves"], 2)
        self.assertEqual(result["elapsed"], 40)
        self.assertTrue(result["normalized"].startswith(b"RPL1"))

    def test_old_verse_replay_is_normalized(self):
        def event(direction, cell, four=False):
            row, column = divmod(cell, 4)
            code = (direction << 5) | (int(four) << 4) | (column << 2) | row
            return OLD_VERSE_ALPHABET[code]

        replay = "replay_" + "".join((
            event(0, 5), event(0, 7), event(1, 3), event(3, 15),
        ))
        result = inspect_old_verse(replay.encode("utf-8"), "4x4")
        self.assertEqual(result["moves"], 2)
        self.assertEqual(result["elapsed"], 0)
        self.assertEqual(result["timed_moves"], 0)

    def test_plan_matches_existing_fact_and_apply_attaches_without_new_run(self):
        result = inspect_2048next(_next_replay().encode("ascii"))
        run_id = "imported-run"
        state = {"score": result["score"], "board": result["board"], "seq": None,
                 "elapsed": None, "nodes": {}, "hash": None}
        with database() as db:
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,status,eligibility,
                 reason,created,ended,writer,epoch,permit_until,monitored,state,archive,
                 display_threshold,visible,has_replay,source)
                VALUES(?,7,'verse:1','4x4','1','',0,'sealed','eligible','imported',
                       1767312000,1767355200,'',0,0,0,?,NULL,0,1,0,'verse')""",
                (run_id, json.dumps(state)))
        root = Path(self.temp.name) / "bundle"
        root.mkdir()
        replay_path = root / f"XLB_4x4_32k_#1_2026-01-02_{result['score']}_2048next.txt"
        replay_path.write_text(_next_replay(), encoding="ascii")

        plan = build_plan(root, user_id=7, timezone_name="Asia/Shanghai")
        self.assertEqual([(item.status, item.run_id) for item in plan], [("matched", run_id)])
        outcome = apply_plan(plan, user_id=7, input_root=root)
        self.assertEqual(outcome["counts"], {"attached": 1})
        with database() as db:
            row = db.execute("SELECT archive,has_replay,state FROM human_runs WHERE id=?",
                             (run_id,)).fetchone()
            audit = db.execute("SELECT status FROM human_bulk_replay_import_items").fetchone()
            total = db.execute("SELECT count(*) FROM human_runs").fetchone()[0]
        self.assertEqual(total, 1)
        self.assertEqual(audit["status"], "attached")
        self.assertEqual(gzip.decompress(row["archive"]), result["normalized"])
        self.assertEqual(row["has_replay"], 1)
        self.assertEqual(json.loads(row["state"])["seq"], 2)

    def test_pku_is_skipped_without_parsing(self):
        root = Path(self.temp.name) / "bundle"
        root.mkdir()
        (root / "XLB_4x4_32k_#174_2026-09-21_1313636_pku.txt").write_text(
            "already imported", encoding="utf-8")
        plan = build_plan(root, user_id=7, timezone_name="Asia/Shanghai")
        self.assertEqual(plan[0].status, "skipped_pku")


if __name__ == "__main__":
    unittest.main()
