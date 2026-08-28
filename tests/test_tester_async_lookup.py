import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.actions import Action
from backend.handlers.tester import handle_tester_action
from backend.session import GameSession, np_u64, safe_hex
from backend.tester import _tester_reset_record
from engine_core.VBoardMover import encode_board


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class TesterQueryProtocolTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self):
        session = GameSession("tester_query_protocol_test")
        session.tester_pattern = ["L3", "256"]
        session.tester_full_pattern = "L3_256"
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.tester_table_found = True
        session.tester_status = "Tablebase loaded"
        session.tester_ready = True
        return session

    async def test_set_board_returns_state_and_waits_for_explicit_query_action(self):
        session = self.make_session()
        websocket = RecordingWebSocket()
        with patch(
            "backend.handlers.tester._tester_prepare_selection",
            return_value=(True, [("unused", "float64")]),
        ):
            await handle_tester_action(
                Action.TESTER_SET_BOARD,
                {"hex_str": "0000000000000011"},
                session,
                websocket,
            )

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_STATE")
        self.assertTrue(websocket.messages[0]["data"]["lookup_pending"])
        self.assertIsNone(session.tester_lookup_task)
        self.assertEqual(session.tester_results, {})

    async def test_client_local_selection_returns_seed_without_mutating_session_board(self):
        session = self.make_session()
        session.board_encoded = np_u64(0x1234)
        websocket = RecordingWebSocket()
        random_board = np_u64(0x2200)

        with (
            patch(
                "backend.handlers.tester._tester_prepare_selection",
                return_value=(True, [("unused", "float64")]),
            ) as prepare,
            patch(
                "backend.handlers.tester._tester_get_random_state",
                new_callable=AsyncMock,
                return_value=random_board,
            ),
            patch(
                "backend.handlers.tester._tester_random_rotate",
                return_value=random_board,
            ),
        ):
            await handle_tester_action(
                Action.TESTER_SELECT_PATTERN,
                {
                    "pattern": "L3",
                    "target": "256",
                    "request_id": "local-seed-1",
                    "client_revision": 17,
                    "client_local_board": True,
                },
                session,
                websocket,
            )

        prepare.assert_called_once_with(
            session,
            "L3",
            "256",
            reset_board=False,
        )
        self.assertEqual(int(session.board_encoded), 0x1234)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_BOARD_SEED")
        self.assertEqual(websocket.messages[0]["data"]["hex_str"], safe_hex(random_board))
        self.assertEqual(websocket.messages[0]["data"]["load_request_id"], "local-seed-1")
        self.assertEqual(websocket.messages[0]["data"]["client_revision"], 17)

    async def test_client_reconnect_attaches_tablebase_without_requesting_a_new_board(self):
        session = self.make_session()
        session.board_encoded = np_u64(0x4321)
        websocket = RecordingWebSocket()

        with (
            patch(
                "backend.handlers.tester._tester_prepare_selection",
                return_value=(True, [("unused", "float64")]),
            ) as prepare,
            patch(
                "backend.handlers.tester._tester_get_random_state",
                new_callable=AsyncMock,
            ) as random_state,
        ):
            await handle_tester_action(
                Action.TESTER_SELECT_PATTERN,
                {
                    "pattern": "L3",
                    "target": "256",
                    "request_id": "reattach-1",
                    "client_local_board": True,
                    "preserve_client_board": True,
                },
                session,
                websocket,
            )

        prepare.assert_called_once_with(
            session,
            "L3",
            "256",
            reset_board=False,
        )
        random_state.assert_not_awaited()
        self.assertEqual(int(session.board_encoded), 0x4321)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_TABLEBASE_READY")
        self.assertEqual(websocket.messages[0]["data"]["request_id"], "reattach-1")

    async def test_move_uses_authorized_result_and_starts_requested_query(self):
        session = self.make_session()
        session.user_id = 1
        websocket = RecordingWebSocket()
        session.board_encoded = np_u64(0x11)
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.tester_logs = []
        _tester_reset_record(session)
        moved_array = np.zeros((4, 4), dtype=np.int32)
        moved_array[0, 0] = 2
        moved_board = np_u64(encode_board(moved_array))
        next_array = moved_array.copy()
        next_array[0, 1] = 2
        next_board = np_u64(encode_board(next_array))
        session.tester_results = {
            "left": 0.9,
            "right": 0.8,
            "down": 0.7,
            "up": 0.6,
        }
        session.tester_result_dtype = "float64"
        session.tester_best_move = "left"
        session.tester_results_board = np_u64(session.board_encoded)

        with (
            patch("backend.handlers.tester.r_move_board", return_value=(moved_board, 4)),
            patch(
                "backend.handlers.tablebase_query.handle_tablebase_query_action",
                new_callable=AsyncMock,
                return_value=True,
            ) as query,
        ):
            await handle_tester_action(
                Action.TESTER_MOVE,
                {
                    "dir": "left",
                    "from_board_hex": safe_hex(session.board_encoded),
                    "board_hex": safe_hex(next_board),
                    "spawn_index": 1,
                    "spawn_value": 2,
                    "query_id": "tester-move-query",
                },
                session,
                websocket,
            )

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_MOVE_ACCEPTED")
        self.assertEqual(websocket.messages[0]["data"]["board_hex"], safe_hex(next_board))
        self.assertTrue(websocket.messages[0]["data"]["lookup_pending"])
        self.assertEqual(session.tester_step_count, 1)
        self.assertEqual(session.tester_results, {})
        self.assertIsNone(session.tester_lookup_task)
        query.assert_awaited_once()
        query_payload = query.await_args.args[1]
        self.assertEqual(query_payload["query_id"], "tester-move-query")
        self.assertEqual(query_payload["board_hex"], safe_hex(next_board))

    async def test_move_without_authoritative_result_resyncs(self):
        session = self.make_session()
        websocket = RecordingWebSocket()
        session.board_encoded = np_u64(0x11)
        await handle_tester_action(
            Action.TESTER_MOVE,
            {
                "dir": "left",
                "from_board_hex": "0000000000000011",
                "board_hex": "0000000000000021",
                "spawn_index": 1,
                "spawn_value": 2,
            },
            session,
            websocket,
        )

        self.assertEqual(websocket.messages[0]["action"], "TESTER_STATE")
        self.assertEqual(int(session.board_encoded), 0x11)
        self.assertEqual(session.tester_step_count, 0)

    async def test_prefetched_move_waits_for_in_process_query_result(self):
        session = self.make_session()
        session.user_id = 1
        websocket = RecordingWebSocket()
        session.board_encoded = np_u64(0x11)
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.tester_logs = []
        _tester_reset_record(session)

        moved_array = np.zeros((4, 4), dtype=np.int32)
        moved_array[0, 0] = 2
        moved_board = np_u64(encode_board(moved_array))
        next_array = moved_array.copy()
        next_array[0, 1] = 2
        next_board = np_u64(encode_board(next_array))

        async def complete_current_query():
            await asyncio.sleep(0)
            session.tester_results = {
                "left": 0.9,
                "right": 0.8,
                "down": 0.7,
                "up": 0.6,
            }
            session.tester_result_dtype = "float64"
            session.tester_best_move = "left"
            session.tester_results_board = np_u64(session.board_encoded)

        session.tester_query_task = asyncio.create_task(
            complete_current_query()
        )

        with (
            patch("backend.handlers.tester.r_move_board", return_value=(moved_board, 4)),
            patch(
                "backend.handlers.tablebase_query.handle_tablebase_query_action",
                new_callable=AsyncMock,
                return_value=True,
            ) as query,
        ):
            await handle_tester_action(
                Action.TESTER_MOVE,
                {
                    "dir": "left",
                    "from_board_hex": safe_hex(session.board_encoded),
                    "board_hex": safe_hex(next_board),
                    "spawn_index": 1,
                    "spawn_value": 2,
                    "query_id": "pipelined-tester-query",
                },
                session,
                websocket,
            )

        self.assertEqual(websocket.messages[0]["action"], "TESTER_MOVE_ACCEPTED")
        self.assertEqual(websocket.messages[0]["data"]["board_hex"], safe_hex(next_board))
        self.assertEqual(session.tester_step_count, 1)
        query.assert_awaited_once()

    async def test_move_rejects_invalid_client_spawn_and_resyncs(self):
        session = self.make_session()
        session.user_id = 1
        websocket = RecordingWebSocket()
        session.board_encoded = np_u64(0x11)
        moved_array = np.zeros((4, 4), dtype=np.int32)
        moved_array[0, 0] = 2
        moved_board = np_u64(encode_board(moved_array))
        session.tester_results = {"left": 0.9, "right": 0.8}
        session.tester_result_dtype = "float64"
        session.tester_best_move = "left"
        session.tester_results_board = np_u64(session.board_encoded)

        with (
            patch("backend.handlers.tester.r_move_board", return_value=(moved_board, 4)),
            patch(
                "backend.tester.get_token_balance",
                return_value={"bonus": 100, "paid": 0, "total": 100},
            ),
            patch(
                "backend.handlers.tablebase_query.handle_tablebase_query_action",
                new_callable=AsyncMock,
                return_value=True,
            ) as query,
        ):
            await handle_tester_action(
                Action.TESTER_MOVE,
                {
                    "dir": "left",
                    "from_board_hex": safe_hex(session.board_encoded),
                    "board_hex": safe_hex(moved_board),
                    "spawn_index": 0,
                    "spawn_value": 2,
                    "query_id": "forged-tester-query",
                },
                session,
                websocket,
            )

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_STATE")
        self.assertEqual(websocket.messages[0]["data"]["hex_str"], "0000000000000011")
        self.assertEqual(int(session.board_encoded), 0x11)
        self.assertEqual(session.tester_step_count, 0)
        query.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
