from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tools.tablebase_worker.config import load_worker_config
from tools.tablebase_worker.reader_pool import ReaderPool


class FakeReader:
    def __init__(self, table, tracker):
        self.table = table
        self.tracker = tracker

    def lookup(self, board, *, use_variant, board_is_lookup):
        with self.tracker["lock"]:
            self.tracker["active"] += 1
            self.tracker["max_active"] = max(
                self.tracker["max_active"], self.tracker["active"]
            )
            self.tracker["calls"].append((self.table.table_id, board, board_is_lookup))
        time.sleep(0.02)
        with self.tracker["lock"]:
            self.tracker["active"] -= 1
        return {"left": board / 100}, "uint32"

    def random_state(self):
        return 0x1234


def config_fixture(root: Path):
    data = {
        "server_url": "wss://example.test/ws/tablebase-worker",
        "worker_id": "worker",
        "auth_token_env": "TOKEN",
        "heartbeat_seconds": 10,
        "resource_groups": {"shared": {"concurrency": 1}},
        "tables": [
            {
                "table_id": "free11_512",
                "path": str(root / "a"),
                "resource_group": "shared",
                "concurrency": 1,
            },
            {
                "table_id": "free11_1024",
                "path": str(root / "b"),
                "resource_group": "shared",
                "concurrency": 1,
            },
        ],
    }
    path = root / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_worker_config(path, environ={"TOKEN": "secret"})


class ReaderPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_refresh_does_not_initialize_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            ready = {"free11_512": True, "free11_1024": False}
            created = []

            def checker(table):
                is_ready = ready[table.table_id]
                return is_ready, None if is_ready else "TABLE_FILES_MISSING"

            pool = ReaderPool(
                config_fixture(Path(temp)),
                reader_factory=lambda table: created.append(table.table_id),
                path_checker=checker,
            )
            ready["free11_512"] = False
            ready["free11_1024"] = True
            self.assertEqual(
                pool.refresh_readiness(),
                [
                    {"full_pattern": "free11_1024", "ready": True},
                    {"full_pattern": "free11_512", "ready": False},
                ],
            )
            self.assertEqual(created, [])
            await pool.close()

    async def test_reader_is_lazy_and_resource_group_serializes_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            tracker = {"active": 0, "max_active": 0, "calls": [], "lock": threading.Lock()}
            created = []

            def factory(table):
                created.append(table.table_id)
                return FakeReader(table, tracker)

            pool = ReaderPool(
                config_fixture(Path(temp)),
                reader_factory=factory,
                path_checker=lambda table: (True, None),
            )
            self.assertEqual(created, [])
            first, second = await __import__("asyncio").gather(
                pool.lookup(
                    "free11_512", 1, use_variant=False, board_is_lookup=False
                ),
                pool.lookup(
                    "free11_1024", 2, use_variant=False, board_is_lookup=True
                ),
            )
            self.assertEqual(first[0]["left"], 0.01)
            self.assertEqual(second[0]["left"], 0.02)
            self.assertEqual(tracker["max_active"], 1)
            self.assertEqual(set(created), {"free11_512", "free11_1024"})
            self.assertIn(("free11_1024", 2, True), tracker["calls"])
            await pool.close()

    async def test_random_state_uses_fake_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            tracker = {"active": 0, "max_active": 0, "calls": [], "lock": threading.Lock()}
            pool = ReaderPool(
                config_fixture(Path(temp)),
                reader_factory=lambda table: FakeReader(table, tracker),
                path_checker=lambda table: (True, None),
            )
            self.assertEqual(await pool.random_state("free11_512"), 0x1234)
            await pool.close()


if __name__ == "__main__":
    unittest.main()
