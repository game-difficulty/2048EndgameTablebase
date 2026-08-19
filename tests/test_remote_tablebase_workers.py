from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from backend.remote_workers.config import configured_workers, load_remote_manifest
from backend.remote_workers.errors import RemoteTablebaseOffline
from backend.remote_workers.registry import RemoteWorkerRegistry
from backend.handlers import tablebase_query as query_handler
from backend.session import GameSession
from backend.tablebase_query_service import TablebaseLookupSpec
from backend.tablebase_catalog import get_available_tablebases


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed: list[int] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        self.closed.append(code)


class RemoteWorkerRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        load_remote_manifest.cache_clear()
        self.registry = RemoteWorkerRegistry()
        await self.registry.start()
        self.secret_patch = patch.dict(
            os.environ, {"REMOTE_TABLEBASE_WORKER_SECRET": "test-secret"}
        )
        self.secret_patch.start()

    async def asyncTearDown(self) -> None:
        self.secret_patch.stop()
        await self.registry.close()

    async def _connect(self, tables=None):
        websocket = FakeWebSocket()
        worker = await self.registry._accept_hello(
            websocket,
            {
                "type": "HELLO",
                "protocol_version": 1,
                "worker_id": "home-main",
                "auth_token": "test-secret",
                "tables": tables
                or [{"full_pattern": "free11_512", "ready": True}],
            },
        )
        return websocket, worker

    async def test_lookup_round_trip_uses_allowlisted_table(self):
        websocket, worker = await self._connect()
        task = asyncio.create_task(
            self.registry.lookup(
                full_pattern="free11_512",
                pattern="free11",
                target="512",
                board="0123456789abcdef",
                use_variant=False,
            )
        )
        await asyncio.sleep(0)
        request = websocket.sent[-1]
        self.assertEqual(request["type"], "LOOKUP")
        self.assertNotIn("path", request)
        await self.registry._handle_message(
            worker,
            {
                "type": "LOOKUP_RESULT",
                "request_id": request["request_id"],
                "results": {"left": 0.75},
                "dtype": "uint32",
            },
        )
        response = await task
        self.assertEqual(response["results"]["left"], 0.75)

    async def test_heartbeat_only_changes_epoch_for_capability_change(self):
        _websocket, worker = await self._connect()
        epoch = self.registry.availability_epoch
        await self.registry._handle_message(worker, {"type": "HEARTBEAT"})
        self.assertEqual(self.registry.availability_epoch, epoch)
        await self.registry._handle_message(
            worker,
            {"type": "HEARTBEAT", "tables": []},
        )
        self.assertEqual(self.registry.availability_epoch, epoch + 1)
        self.assertFalse(self.registry.is_table_online("free11_512"))

    async def test_unknown_or_offline_table_is_rejected(self):
        await self._connect()
        with self.assertRaises(RemoteTablebaseOffline):
            await self.registry.lookup(
                full_pattern="not_allowed_1",
                pattern="not_allowed",
                target="1",
                board="0123456789abcdef",
                use_variant=False,
            )


class RemoteCatalogTests(unittest.TestCase):
    def test_online_remote_tables_are_safe_public_catalog_entries(self):
        with patch(
            "backend.tablebase_catalog.remote_worker_registry.online_tables",
            return_value=frozenset({"free11_512"}),
        ):
            entries = [
                item
                for item in get_available_tablebases()
                if item["full_pattern"] == "free11_512"
            ]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["pattern"], "free11")
        self.assertNotIn("path", entries[0])
        self.assertNotIn("worker_id", entries[0])

    def test_manifest_assigns_each_table_once(self):
        workers = configured_workers()
        all_tables = [
            table
            for worker in workers.values()
            for table in worker["tables"]
        ]
        self.assertEqual(len(all_tables), len(set(all_tables)))


class RemoteQuotaFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_query_refunds_reservation(self):
        class OfflineHandle:
            generation = 1
            is_current = True

            async def wait(self):
                raise RemoteTablebaseOffline()

        websocket = FakeWebSocket()
        session = GameSession("trainer_remote_offline")
        session.user_id = 17
        spec = TablebaseLookupSpec(
            board_encoded=1,
            pattern="free11",
            target="512",
            full_pattern="free11_512",
            use_variant=False,
            book_reader=None,
            provider_kind="remote",
            catalog_version="remote-test",
        )
        reservation = object()
        with (
            patch.object(query_handler, "cancel_reservation") as cancel,
            patch.object(query_handler, "finalize_reservation") as finalize,
            patch.object(query_handler, "get_token_balance", return_value={"total": 3}),
        ):
            await query_handler._finish_query(
                session,
                websocket,
                page="trainer",
                query_id="offline-query",
                stream_key="17:trainer:trainer",
                catalog_version="remote-test",
                spec=spec,
                handle=OfflineHandle(),
                reservation=reservation,
                supporter=False,
            )
        cancel.assert_called_once()
        finalize.assert_not_called()
        self.assertEqual(
            websocket.sent[-1]["data"]["code"], "REMOTE_TABLEBASE_OFFLINE"
        )


if __name__ == "__main__":
    unittest.main()
