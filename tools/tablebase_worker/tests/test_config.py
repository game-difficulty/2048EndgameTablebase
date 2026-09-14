from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.tablebase_worker.config import (
    WorkerConfigError,
    load_worker_config,
    table_path_status,
)
from tools.tablebase_worker.reader_pool import _count_available_layers


def write_config(root: Path, **overrides) -> Path:
    table_a = root / "table-a"
    table_b = root / "table-b"
    table_a.mkdir()
    table_b.mkdir()
    (table_a / "free11_512_0.exadbook").write_bytes(b"fixture")
    shard = table_b / "4442f_2048_0b"
    shard.mkdir()
    (shard / "1.b").write_bytes(b"fixture")
    data = {
        "server_url": "wss://example.test/ws/tablebase-worker",
        "worker_id": "test-worker",
        "auth_token_env": "TEST_WORKER_TOKEN",
        "heartbeat_seconds": 10,
        "resource_groups": {"disk-a": {"concurrency": 1}},
        "tables": [
            {
                "table_id": "free11_512",
                "pattern": "free11",
                "target": "512",
                "path": str(table_a),
                "dtype": "uint32",
                "resource_group": "disk-a",
                "concurrency": 1,
            },
            {
                "table_id": "4442f_2048",
                "path": str(table_b),
                "resource_group": "disk-a",
                "concurrency": 1,
            },
        ],
    }
    data.update(overrides)
    path = root / "worker.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class WorkerConfigTests(unittest.TestCase):
    def test_multiple_paths_are_preserved_checked_and_layers_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_config(root)
            data = json.loads(path.read_text(encoding="utf-8"))
            table = data["tables"][0]
            first = table.pop("path")
            second = root / "second-part"
            second.mkdir()
            (second / "free11_512_0.bccmp").touch()
            (second / "free11_512_20.bccmp").touch()
            table["paths"] = [first, str(second)]
            path.write_text(json.dumps(data), encoding="utf-8")
            config = load_worker_config(path, require_auth=False).tables["free11_512"]
            self.assertEqual(config.path_list, [(first, "uint32"), (str(second), "uint32")])
            self.assertEqual(table_path_status(config), (True, None))
            self.assertEqual(_count_available_layers(config), 2)
            second.rename(root / "offline")
            self.assertEqual(table_path_status(config), (False, "TABLE_PATH_MISSING"))

    def test_rejects_ambiguous_empty_or_duplicate_paths(self):
        for paths in ([], "not-an-array", ["same", "same"]):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as temp:
                path = write_config(Path(temp))
                data = json.loads(path.read_text(encoding="utf-8"))
                data["tables"][0].pop("path")
                data["tables"][0]["paths"] = paths
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(WorkerConfigError):
                    load_worker_config(path, require_auth=False)
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["tables"][0]["paths"] = [data["tables"][0]["path"]]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(WorkerConfigError):
                load_worker_config(path, require_auth=False)

    def test_loads_allowlist_without_touching_real_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp))
            config = load_worker_config(
                path, environ={"TEST_WORKER_TOKEN": "secret"}
            )
            self.assertEqual(config.heartbeat_seconds, 10)
            self.assertEqual(set(config.tables), {"free11_512", "4442f_2048"})
            self.assertEqual(config.tables["free11_512"].concurrency, 1)
            self.assertEqual(config.resource_groups["disk-a"].concurrency, 1)
            self.assertEqual(config.auth_token, "secret")

    def test_requires_wss(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp), server_url="ws://example.test/worker")
            with self.assertRaisesRegex(WorkerConfigError, "wss"):
                load_worker_config(path, environ={"TEST_WORKER_TOKEN": "secret"})

    def test_requires_fixed_ten_second_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp), heartbeat_seconds=5)
            with self.assertRaisesRegex(WorkerConfigError, "must be 10"):
                load_worker_config(path, environ={"TEST_WORKER_TOKEN": "secret"})

    def test_rejects_unknown_config_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp), remote_path="C:/not-allowed")
            with self.assertRaisesRegex(WorkerConfigError, "Unknown"):
                load_worker_config(path, environ={"TEST_WORKER_TOKEN": "secret"})

    def test_rejects_duplicate_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_config(root)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["tables"].append(dict(data["tables"][0]))
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(WorkerConfigError, "Duplicate"):
                load_worker_config(path, environ={"TEST_WORKER_TOKEN": "secret"})

    def test_requires_auth_environment_variable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp))
            with self.assertRaisesRegex(WorkerConfigError, "TEST_WORKER_TOKEN"):
                load_worker_config(path, environ={})

    def test_table_file_detection_supports_exad_files_and_ad_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_config(Path(temp))
            config = load_worker_config(
                path, environ={"TEST_WORKER_TOKEN": "secret"}
            )
            self.assertEqual(table_path_status(config.tables["free11_512"]), (True, None))
            self.assertEqual(table_path_status(config.tables["4442f_2048"]), (True, None))

    def test_missing_table_path_is_reported_without_reader_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_config(root)
            config = load_worker_config(
                path, environ={"TEST_WORKER_TOKEN": "secret"}
            )
            config.tables["free11_512"].path.rename(root / "moved")
            self.assertEqual(
                table_path_status(config.tables["free11_512"]),
                (False, "TABLE_PATH_MISSING"),
            )

    def test_empty_shard_directory_is_not_enough_for_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_config(root)
            config = load_worker_config(
                path, environ={"TEST_WORKER_TOKEN": "secret"}
            )
            shard_file = config.tables["4442f_2048"].path / "4442f_2048_0b" / "1.b"
            shard_file.unlink()
            self.assertEqual(
                table_path_status(config.tables["4442f_2048"]),
                (False, "TABLE_FILES_MISSING"),
            )


if __name__ == "__main__":
    unittest.main()
