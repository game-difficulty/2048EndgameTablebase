from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from tools.tablebase_worker.client import WorkerClient
from tools.tablebase_worker.protocol import Request


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send(self, payload):
        self.messages.append(json.loads(payload))

    async def close(self):
        pass


class FakeReaderPool:
    async def lookup(self, table_id, board, *, use_variant, board_is_lookup):
        return {"left": 0.75, "right": None}, "uint32"

    async def lookup_batch(self, table_id, boards, *, use_variant, board_is_lookup):
        return [({"left": board / 10}, "uint32") for board in boards]

    async def random_state(self, table_id):
        return 0x1234


class ClientProtocolTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        return WorkerClient(object(), FakeReaderPool())

    async def test_lookup_result_is_flat(self):
        websocket = FakeWebSocket()
        await self.client()._execute_request(
            websocket,
            Request(
                message_type="LOOKUP",
                request_id="lookup-1",
                full_pattern="free11_512",
                pattern="free11",
                target="512",
                boards=(1,),
            ),
        )
        self.assertEqual(
            websocket.messages,
            [
                {
                    "type": "LOOKUP_RESULT",
                    "request_id": "lookup-1",
                    "results": {"left": 0.75, "right": None},
                    "dtype": "uint32",
                    "board": "0000000000000001",
                }
            ],
        )

    async def test_batch_items_preserve_board_order(self):
        websocket = FakeWebSocket()
        await self.client()._execute_request(
            websocket,
            Request(
                message_type="LOOKUP_BATCH",
                request_id="batch-1",
                full_pattern="free11_512",
                pattern="free11",
                target="512",
                boards=(1, 2),
            ),
        )
        response = websocket.messages[0]
        self.assertEqual(response["type"], "LOOKUP_BATCH_RESULT")
        self.assertEqual(
            [item["board"] for item in response["items"]],
            ["0000000000000001", "0000000000000002"],
        )
        self.assertNotIn("data", response)

    async def test_random_state_result_is_flat(self):
        websocket = FakeWebSocket()
        await self.client()._execute_request(
            websocket,
            Request(
                message_type="RANDOM_STATE",
                request_id="random-1",
                full_pattern="free11_512",
                pattern="free11",
                target="512",
            ),
        )
        self.assertEqual(
            websocket.messages[0],
            {
                "type": "RANDOM_STATE_RESULT",
                "request_id": "random-1",
                "board": "0000000000001234",
            },
        )

    async def test_established_connection_failure_returns_connection_duration(self):
        client = WorkerClient(
            SimpleNamespace(
                worker_id="home-main",
                auth_token="secret",
                hello_timeout_seconds=5,
            ),
            SimpleNamespace(hello_tables=lambda: []),
        )
        websocket = FakeWebSocket()
        websocket.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "HELLO_ACK",
                    "protocol_version": 1,
                    "worker_id": "home-main",
                    "tables": [],
                }
            )
        )
        client._connect = AsyncMock(return_value=websocket)
        client._receive_loop = AsyncMock(side_effect=ConnectionResetError())
        client._heartbeat_loop = AsyncMock()
        connected_seconds = await client._run_connection()
        self.assertGreaterEqual(connected_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
