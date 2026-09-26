from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.remote_workers import analysis_proxy, internal_bridge
from backend.tablebase_catalog import _iter_remote_entries


class PlayTablebaseBridgeTests(unittest.TestCase):
    def test_bridge_requires_secret_and_validates_table(self):
        app = FastAPI()
        app.include_router(internal_bridge.router)
        with patch.dict(os.environ, {"REMOTE_TABLEBASE_WORKER_SECRET": "test-secret"}), \
             patch.object(internal_bridge.remote_worker_registry, "online_tables", return_value=frozenset({"free10_512"})), \
             patch.object(internal_bridge.remote_worker_registry, "lookup_batch", new_callable=AsyncMock) as lookup:
            lookup.return_value = {"items": [{"results": {"up": 1}, "dtype": "uint32"}]}
            client = TestClient(app)
            self.assertEqual(client.get("/internal/tablebases/online").status_code, 403)
            headers = {"X-Internal-Tablebase-Secret": "test-secret"}
            online = client.get("/internal/tablebases/online", headers=headers)
            self.assertEqual(online.status_code, 200)
            self.assertEqual(online.json()["tables"], ["free10_512"])
            payload = {"full_pattern": "free10_512", "pattern": "free10", "target": "512",
                       "boards": ["0000000000000001"], "use_variant": False}
            self.assertEqual(client.post("/internal/tablebases/lookup-batch", headers=headers,
                                         json={**payload, "pattern": "other"}).status_code, 400)
            response = client.post("/internal/tablebases/lookup-batch", headers=headers, json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["items"][0]["results"], {"up": 1})
            self.assertEqual(lookup.await_count, 1)

    def test_play_catalog_only_exposes_online_proxy_tables(self):
        with patch.dict(os.environ, {"REMOTE_TABLEBASE_PROXY_BASE_URL": "http://127.0.0.1:8000"}), \
             patch("backend.tablebase_catalog.proxy_online_tables", return_value={"free10_512"}):
            entries = _iter_remote_entries(online_only=True)
        self.assertEqual([entry["_full_pattern"] for entry in entries], ["free10_512"])

    def test_analysis_reader_uses_proxy_for_batch_and_single_lookup(self):
        reader = analysis_proxy.RemoteAnalysisBookReader(
            full_pattern="free10_512", pattern="free10", target="512", use_variant=False)
        with patch.dict(os.environ, {"REMOTE_TABLEBASE_PROXY_BASE_URL": "http://127.0.0.1:8000"}), \
             patch.object(analysis_proxy, "proxy_lookup_batch", return_value={
                 "items": [{"results": {"up": 3}, "dtype": "uint32"}]}) as proxy:
            reader.preload([1])
            self.assertEqual(reader.move_on_dic(1, "free10", "512", "free10_512"), ({"up": 3}, "uint32"))
            self.assertEqual(reader.move_on_dic(2, "free10", "512", "free10_512"), ({"up": 3}, "uint32"))
            self.assertEqual(proxy.call_count, 2)


if __name__ == "__main__":
    unittest.main()
