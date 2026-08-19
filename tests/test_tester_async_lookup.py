import unittest
from unittest.mock import patch

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

    async def test_move_uses_authorized_session_result_and_does_not_query_inline(self):
        session = self.make_session()
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

        with patch("backend.handlers.tester.r_move_board", return_value=(moved_board, 4)):
            await handle_tester_action(
                Action.TESTER_MOVE,
                {
                    "dir": "left",
                    "from_board_hex": safe_hex(session.board_encoded),
                    "board_hex": safe_hex(next_board),
                    "spawn_index": 1,
                    "spawn_value": 2,
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

    async def test_move_rejects_invalid_client_spawn_and_resyncs(self):
        session = self.make_session()
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
        ):
            await handle_tester_action(
                Action.TESTER_MOVE,
                {
                    "dir": "left",
                    "from_board_hex": safe_hex(session.board_encoded),
                    "board_hex": safe_hex(moved_board),
                    "spawn_index": 0,
                    "spawn_value": 2,
                },
                session,
                websocket,
            )

        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0]["action"], "TESTER_STATE")
        self.assertEqual(websocket.messages[0]["data"]["hex_str"], "0000000000000011")
        self.assertEqual(int(session.board_encoded), 0x11)
        self.assertEqual(session.tester_step_count, 0)


if __name__ == "__main__":
    unittest.main()
