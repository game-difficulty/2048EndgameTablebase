import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.actions import Action, Message
from backend.handlers.trainer import handle_trainer_action
from backend.remote_workers.errors import RemoteTablebaseOffline
from backend.session import GameSession, np_u64
from engine_core.VBoardMover import encode_board


class RecordingManager:
    def __init__(self):
        self.states = []

    async def send_state(self, websocket, metadata=None):
        del websocket
        self.states.append(metadata or {})


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


def encoded(values):
    return np_u64(encode_board(np.asarray(values, dtype=np.int32).reshape(4, 4)))


class TrainerOptimisticMoveTests(unittest.IsolatedAsyncioTestCase):
    async def test_optimistic_board_edit_uses_lightweight_ack(self):
        session = GameSession("trainer_optimistic_board_edit")
        session.board_encoded = encoded([0] * 16)
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        target = encoded([2, 4, 8, 16, *([0] * 12)])
        manager = RecordingManager()
        websocket = RecordingWebSocket()

        handled = await handle_trainer_action(
            Action.SET_BOARD,
            {
                "hex_str": f"{int(target):016x}",
                "client_optimistic": True,
                "edit_source": "palette",
            },
            session,
            websocket,
            manager,
        )

        self.assertTrue(handled)
        self.assertEqual(session.board_encoded, target)
        self.assertEqual(manager.states, [])
        self.assertEqual(
            websocket.messages,
            [
                {
                    "action": Message.TRAINER_BOARD_SYNCED,
                    "data": {
                        "board_hex": f"{int(target):016x}",
                        "edit_source": "palette",
                    },
                }
            ],
        )

    async def test_random_spawn_submission_is_validated_and_accepted(self):
        session = GameSession("trainer_optimistic_random")
        session.user_id = 1
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.board_encoded = encoded([0, 2, 2, 0, *([0] * 12)])
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.spawn_mode = 0
        target = encoded([4, 2, 0, 0, *([0] * 12)])
        manager = RecordingManager()

        with patch(
            "backend.handlers.tablebase_query.handle_tablebase_query_action",
            new_callable=AsyncMock,
            return_value=True,
        ) as query:
            handled = await handle_trainer_action(
                Action.TRAINER_MOVE,
                {
                    "dir": "left",
                    "client_optimistic": True,
                    "from_board_hex": f"{int(session.board_encoded):016x}",
                    "board_hex": f"{int(target):016x}",
                    "spawn_index": 1,
                    "spawn_value": 2,
                    "query_id": "trainer-move-query",
                },
                session,
                object(),
                manager,
            )

        self.assertTrue(handled)
        self.assertEqual(session.board_encoded, target)
        self.assertEqual(len(session.history), 2)
        self.assertEqual(manager.states, [{}])
        query.assert_awaited_once()
        query_payload = query.await_args.args[1]
        self.assertEqual(query_payload["query_id"], "trainer-move-query")
        self.assertEqual(query_payload["board_hex"], f"{int(target):016x}")

    async def test_forged_random_board_does_not_mutate_session(self):
        session = GameSession("trainer_optimistic_rejected")
        session.user_id = 1
        original = encoded([0, 2, 2, 0, *([0] * 12)])
        session.board_encoded = original
        session.score = 17
        session.history = [(session.board_encoded, session.score)]
        session.move_history = [None]
        session.spawn_mode = 0
        manager = RecordingManager()

        with patch(
            "backend.handlers.tablebase_query.handle_tablebase_query_action",
            new_callable=AsyncMock,
            return_value=True,
        ) as query:
            handled = await handle_trainer_action(
                Action.TRAINER_MOVE,
                {
                    "dir": "left",
                    "client_optimistic": True,
                    "from_board_hex": f"{int(original):016x}",
                    "board_hex": "ffffffffffffffff",
                    "spawn_index": 1,
                    "spawn_value": 2,
                    "query_id": "forged-query",
                },
                session,
                object(),
                manager,
            )

        self.assertTrue(handled)
        self.assertEqual(session.board_encoded, original)
        self.assertEqual(session.score, 17)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(manager.states, [{}])
        query.assert_not_awaited()

    async def test_manual_mode_accepts_move_without_spawn(self):
        session = GameSession("trainer_optimistic_manual")
        session.board_encoded = encoded([0, 2, 2, 0, *([0] * 12)])
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.spawn_mode = 3
        target = encoded([4, 0, 0, 0, *([0] * 12)])
        manager = RecordingManager()

        await handle_trainer_action(
            Action.TRAINER_MOVE,
            {
                "dir": "left",
                "client_optimistic": True,
                "from_board_hex": f"{int(session.board_encoded):016x}",
                "board_hex": f"{int(target):016x}",
                "spawn_index": -1,
                "spawn_value": 0,
            },
            session,
            object(),
            manager,
        )

        self.assertEqual(session.board_encoded, target)
        self.assertEqual(session.moved, 1)
        self.assertEqual(manager.states, [{}])

    async def test_server_selected_spawn_queries_the_final_board(self):
        session = GameSession("trainer_optimistic_worst_spawn")
        session.user_id = 1
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.board_encoded = encoded([0, 2, 2, 0, *([0] * 12)])
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.spawn_mode = 2
        moved = encoded([4, 0, 0, 0, *([0] * 12)])
        target = encoded([4, 2, 0, 0, *([0] * 12)])
        manager = RecordingManager()

        with (
            patch(
                "backend.handlers.trainer.compute_spawns_async",
                new_callable=AsyncMock,
                return_value={(1, 1): 0.1},
            ),
            patch(
                "backend.handlers.tablebase_query.handle_tablebase_query_action",
                new_callable=AsyncMock,
                return_value=True,
            ) as query,
        ):
            await handle_trainer_action(
                Action.TRAINER_MOVE,
                {
                    "dir": "left",
                    "client_optimistic": True,
                    "from_board_hex": f"{int(session.board_encoded):016x}",
                    "board_hex": f"{int(moved):016x}",
                    "query_id": "trainer-worst-query",
                },
                session,
                object(),
                manager,
            )

        self.assertEqual(session.board_encoded, target)
        self.assertEqual(manager.states, [{"appear_tile": {"index": 1, "value": 2}}])
        query.assert_awaited_once()
        query_payload = query.await_args.args[1]
        self.assertEqual(query_payload["query_id"], "trainer-worst-query")
        self.assertEqual(query_payload["board_hex"], f"{int(target):016x}")

    async def test_server_spawn_failure_resyncs_without_starting_query(self):
        session = GameSession("trainer_optimistic_spawn_failure")
        session.user_id = 1
        session.current_pattern = "free11_512"
        session.pattern_settings = ["free11", "512"]
        original = encoded([0, 2, 2, 0, *([0] * 12)])
        moved = encoded([4, 0, 0, 0, *([0] * 12)])
        session.board_encoded = original
        session.history = [(session.board_encoded, 0)]
        session.move_history = [None]
        session.spawn_mode = 2
        manager = RecordingManager()
        websocket = RecordingWebSocket()

        with (
            patch(
                "backend.handlers.trainer.compute_spawns_async",
                new_callable=AsyncMock,
                side_effect=RemoteTablebaseOffline(),
            ),
            patch(
                "backend.handlers.tablebase_query.handle_tablebase_query_action",
                new_callable=AsyncMock,
                return_value=True,
            ) as query,
        ):
            await handle_trainer_action(
                Action.TRAINER_MOVE,
                {
                    "dir": "left",
                    "client_optimistic": True,
                    "from_board_hex": f"{int(original):016x}",
                    "board_hex": f"{int(moved):016x}",
                    "query_id": "trainer-offline-query",
                },
                session,
                websocket,
                manager,
            )

        self.assertEqual(session.board_encoded, original)
        self.assertEqual(manager.states, [{}])
        query.assert_not_awaited()
        self.assertEqual(websocket.messages[0]["data"]["query_id"], "trainer-offline-query")
        self.assertEqual(websocket.messages[0]["data"]["board_hex"], f"{int(original):016x}")

    async def test_manual_spawn_submission_is_validated_and_accepted(self):
        session = GameSession("trainer_optimistic_manual_spawn")
        session.user_id = 1
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        original = encoded([4, 0, 0, 0, *([0] * 12)])
        session.board_encoded = original
        session.history = [(session.board_encoded, 0)]
        session.move_history = ["left"]
        session.spawn_mode = 3
        session.moved = 1
        target = encoded([4, 2, 0, 0, *([0] * 12)])
        manager = RecordingManager()

        with patch(
            "backend.handlers.tablebase_query.handle_tablebase_query_action",
            new_callable=AsyncMock,
            return_value=True,
        ) as query:
            await handle_trainer_action(
                Action.TRAINER_MANUAL_SPAWN,
                {
                    "row": 0,
                    "col": 1,
                    "val": 2,
                    "client_optimistic": True,
                    "from_board_hex": f"{int(original):016x}",
                    "board_hex": f"{int(target):016x}",
                    "query_id": "trainer-manual-query",
                },
                session,
                object(),
                manager,
            )

        self.assertEqual(session.board_encoded, target)
        self.assertEqual(session.moved, 0)
        self.assertEqual(manager.states, [{}])
        query.assert_awaited_once()
        self.assertEqual(
            query.await_args.args[1]["query_id"],
            "trainer-manual-query",
        )


if __name__ == "__main__":
    unittest.main()
