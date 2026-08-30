from __future__ import annotations

import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tools.tablebase_worker.config import load_worker_config
from tools.tablebase_worker.reader_pool import (
    BattleRouteGenerationError,
    ReaderPool,
    _board_contains_target,
)


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


class RouteReader:
    def lookup(self, board, *, use_variant, board_is_lookup):
        return {
            "up": 0.6,
            "down": 0.9,
            "left": 0.8,
            "right": 0.7,
        }, "float64"

    def random_state(self):
        return 0x1000000000000000


class ZeroRouteReader(RouteReader):
    def lookup(self, board, *, use_variant, board_is_lookup):
        return {direction: 0 for direction in ("up", "down", "left", "right")}, "uint32"


class CertainRouteReader(RouteReader):
    def lookup(self, board, *, use_variant, board_is_lookup):
        return {
            "up": 0.6,
            "down": 1.0,
            "left": 0.8,
            "right": 0.7,
        }, "float64"


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
    async def test_target_detection_does_not_treat_pattern_f_as_small_target(self):
        self.assertFalse(_board_contains_target(0xF000000000000000, 128))
        self.assertTrue(_board_contains_target(0x7000000000000000, 128))

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

    async def test_battle_route_is_deterministic_and_uses_trainer_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table_path = root / "a"
            table_path.mkdir()
            for layer in range(4):
                (table_path / f"free11_512_{layer}.zbook").touch()
            pool = ReaderPool(
                config_fixture(root),
                reader_factory=lambda table: RouteReader(),
                path_checker=lambda table: (True, None),
            )
            kwargs = {
                "initial_board": 0x1000000000000000,
                "max_steps": 2,
                "min_steps": 2,
                "spawn_rate": 0.1,
                "seed_hex": "0123456789abcdef0123456789abcdef",
            }
            first = await pool.generate_battle_route("free11_512", **kwargs)
            second = await pool.generate_battle_route("free11_512", **kwargs)
            self.assertEqual(first.route_blob, second.route_blob)
            self.assertEqual(first.step_count, 2)
            self.assertEqual(first.termination_reason, "max_steps")
            self.assertEqual(first.available_layers, 4)
            self.assertEqual(len(first.route_blob), 3 * 17)
            header = struct.unpack_from("<B4I", first.route_blob)
            self.assertEqual(header, (0, 0, 0, 0, 0x1000))
            first_step = struct.unpack_from("<B4I", first.route_blob, 17)
            self.assertLess(first_step[0], 128)
            self.assertEqual(first_step[0] & 0b11, 1)
            self.assertEqual(
                first_step[1:],
                (2400000000, 3600000000, 3200000000, 2800000000),
            )
            await pool.close()

    async def test_certainty_step_does_not_stop_route_before_target_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table_path = root / "a"
            table_path.mkdir()
            for layer in range(4):
                (table_path / f"free11_512_{layer}.zbook").touch()
            pool = ReaderPool(
                config_fixture(root),
                reader_factory=lambda table: CertainRouteReader(),
                path_checker=lambda table: (True, None),
            )
            result = await pool.generate_battle_route(
                "free11_512",
                initial_board=0x1000000000000000,
                max_steps=2,
                min_steps=2,
                spawn_rate=0.1,
                seed_hex="2" * 32,
            )
            self.assertEqual(result.certainty_step, 0)
            self.assertEqual(result.step_count, 2)
            self.assertEqual(result.termination_reason, "max_steps")
            await pool.close()

    async def test_explicit_battle_route_rejects_short_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table_path = root / "a"
            table_path.mkdir()
            for layer in range(4):
                (table_path / f"free11_512_{layer}.zbook").touch()
            pool = ReaderPool(
                config_fixture(root),
                reader_factory=lambda table: ZeroRouteReader(),
                path_checker=lambda table: (True, None),
            )
            with self.assertRaisesRegex(BattleRouteGenerationError, "shorter"):
                await pool.generate_battle_route(
                    "free11_512",
                    initial_board=0x1000000000000000,
                    max_steps=2,
                    min_steps=1,
                    spawn_rate=0.1,
                    seed_hex="0" * 32,
                )
            await pool.close()


if __name__ == "__main__":
    unittest.main()
