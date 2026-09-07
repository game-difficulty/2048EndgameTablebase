from __future__ import annotations

import asyncio
import base64
import os
import unittest
from unittest.mock import patch

from backend.remote_workers.config import configured_workers, load_remote_manifest
from backend.remote_workers.errors import (
    RemoteTablebaseOffline,
    RemoteTablebaseProtocolError,
)
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
    async def test_gamer_route_capability_and_single_round_trip(self):
        from backend.gamer_tablebase_route import generate_route
        from Config import pattern_32k_tiles_map
        websocket, worker = await self._connect(capabilities=['gamer_route_v1'])
        self.assertTrue(self.registry.supports_gamer_route('free11_512'))
        options = dict(board_codes=[1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0],
            rng_state=[1,2,3,4], steps=4, difficulty=0, spawn_rate4=.1, random_only=False)
        task = asyncio.create_task(self.registry.generate_gamer_route(full_pattern='free11_512',
            pattern='free11', target='512', options=options))
        await asyncio.sleep(0)
        request = websocket.sent[-1]
        self.assertEqual(request['type'], 'GENERATE_GAMER_ROUTE')
        nodes = generate_route(options, pattern_32k_tiles_map['free11'][0], lambda board: ({'left':.9}, 'float64'))
        await self.registry._handle_message(worker, dict(type='GAMER_ROUTE_RESULT',
            request_id=request['request_id'], items=nodes))
        self.assertEqual(await task, nodes)

    async def test_old_worker_has_no_gamer_route_capability(self):
        await self._connect()
        self.assertFalse(self.registry.supports_gamer_route('free11_512'))

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

    async def _connect(self, tables=None, capabilities=None):
        websocket = FakeWebSocket()
        hello = {
            "type": "HELLO",
            "protocol_version": 1,
            "worker_id": "home-main",
            "auth_token": "test-secret",
            "tables": tables
            or [{"full_pattern": "free11_512", "ready": True}],
        }
        if capabilities is not None:
            hello["capabilities"] = capabilities
        worker = await self.registry._accept_hello(
            websocket,
            hello,
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

    async def test_availability_listener_receives_connect_and_disconnect(self):
        epochs: list[int] = []
        self.registry.add_availability_listener(epochs.append)
        _websocket, worker = await self._connect()
        self.assertEqual(epochs, [1])
        await self.registry._remove_worker(worker, RemoteTablebaseOffline())
        self.assertEqual(epochs, [1, 2])
        self.registry.remove_availability_listener(epochs.append)

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

    async def test_old_worker_remains_online_but_cannot_generate_routes(self):
        await self._connect()
        self.assertTrue(self.registry.is_table_online("free11_512"))
        with self.assertRaises(RemoteTablebaseProtocolError):
            await self.registry.generate_battle_route(
                full_pattern="free11_512",
                pattern="free11",
                target="512",
                initial_board=None,
                max_steps=None,
                min_steps=64,
                spawn_rate=0.1,
                seed_hex="1" * 32,
            )

    async def test_battle_route_round_trip_requires_advertised_capability(self):
        websocket, worker = await self._connect(capabilities=["battle_route_v1"])
        task = asyncio.create_task(
            self.registry.generate_battle_route(
                full_pattern="free11_512",
                pattern="free11",
                target="512",
                initial_board="0000000000001234",
                max_steps=1,
                min_steps=1,
                spawn_rate=0.1,
                seed_hex="0123456789abcdef0123456789abcdef",
            )
        )
        await asyncio.sleep(0)
        request = websocket.sent[-1]
        self.assertEqual(request["type"], "GENERATE_BATTLE_ROUTE")
        self.assertNotIn("path", request)
        self.assertEqual(request["initial_board"], "0000000000001234")
        route_blob = b"\x00" * 34
        await self.registry._handle_message(
            worker,
            {
                "type": "BATTLE_ROUTE_RESULT",
                "request_id": request["request_id"],
                "route_blob_base64": base64.b64encode(route_blob).decode("ascii"),
                "step_count": 1,
                "certainty_step": 0,
                "termination_reason": "target_reached",
                "initial_board": "0000000000001234",
                "available_layers": 256,
            },
        )
        response = await task
        self.assertEqual(response["step_count"], 1)
        self.assertEqual(response["available_layers"], 256)


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

    def test_manifest_exposes_the_expected_home_worker_tables(self):
        tables = configured_workers()["home-main"]["tables"]
        self.assertEqual(
            set(tables),
            {
                "free11_512",
                "free11_1024",
                "free10_128",
                "free10_256",
                "free10_512",
                "4421_1024",
                "4421_2048",
                "2432t_2048",
                "4431_1024",
                "444_1024",
                "444_2048",
                "LL_1024",
            },
        )
        self.assertNotIn("4442f_2048", tables)


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
