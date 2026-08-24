import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from backend.actions import Action
from backend.handlers import tablebase_query as query_handler
from backend.quota.errors import InsufficientTokens
from backend.session import GameSession, np_u64
from backend.tablebase_query_service import TablebaseQueryScheduler
from backend.tablebase_query_service import TablebaseQuerySuperseded
from engine_core.VBoardMover import decode_board, encode_board


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class ConstantReader:
    def __init__(self):
        self.calls = []

    def move_on_dic(self, board, pattern, target, full_pattern):
        del pattern, target, full_pattern
        encoded = int(encode_board(np.asarray(board, dtype=np.int32)))
        self.calls.append(encoded)
        return {
            "left": 0.9,
            "right": 0.8,
            "down": 0.7,
            "up": 0.6,
        }, "float64"


class TablebaseQueryHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_superseded_query_refunds_its_reservation(self):
        class SupersededHandle:
            generation = 1
            is_current = False

            async def wait(self):
                raise TablebaseQuerySuperseded()

        session = GameSession("trainer_superseded_refund_test")
        session.user_id = 1
        websocket = RecordingWebSocket()
        spec = query_handler.TablebaseLookupSpec(
            board_encoded=0x11,
            pattern="L3",
            target="256",
            full_pattern="L3_256",
            use_variant=False,
            book_reader=ConstantReader(),
            catalog_version="catalog-test",
        )
        reservation = object()

        with (
            patch.object(query_handler, "cancel_reservation") as cancel,
            patch.object(query_handler, "finalize_reservation") as finalize,
        ):
            await query_handler._finish_query(
                session,
                websocket,
                page="trainer",
                query_id="superseded-query",
                stream_key="1:trainer:trainer",
                catalog_version="catalog-test",
                spec=spec,
                handle=SupersededHandle(),
                reservation=reservation,
                supporter=False,
            )

        cancel.assert_called_once_with(
            reservation,
            reason="superseded_before_result",
            metadata={"page": "trainer", "query_id": "superseded-query"},
        )
        finalize.assert_not_called()
        self.assertEqual(websocket.messages, [])

    def test_prefetch_includes_twos_then_fours_when_at_most_four_cells_are_empty(self):
        board = np.array(
            [
                [2, 2, 4, 8],
                [16, 32, 64, 128],
                [256, 512, 1024, 2048],
                [4096, 8192, 0, 0],
            ],
            dtype=np.int32,
        )
        board_encoded = np_u64(encode_board(board))
        moved_board, _ = query_handler.r_move_board(board_encoded, 1)
        moved = decode_board(np_u64(moved_board))
        empty_count = int(np.count_nonzero(moved == 0))

        children = query_handler._prefetch_boards(
            board_encoded,
            best_move="left",
            use_variant=False,
        )

        self.assertLessEqual(empty_count, 4)
        self.assertEqual(len(children), empty_count * 2)
        spawned_values = []
        for child_encoded in children:
            child = decode_board(np_u64(child_encoded))
            delta = child - moved
            spawned_values.append(int(delta[delta > 0][0]))
        self.assertEqual(spawned_values[:empty_count], [2] * empty_count)
        self.assertEqual(spawned_values[empty_count:], [4] * empty_count)

    def test_deterministic_prefetch_matches_frontend_rng_contract(self):
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, :2] = 2
        child = query_handler._deterministic_prefetch_child(
            np_u64(encode_board(board)),
            direction="left",
            use_variant=False,
            rng_context=query_handler._PrefetchRngContext(
                state=(1, 2, 3, 4),
                turn=7,
                spawn_rate_4=0.1,
            ),
        )

        self.assertIsNotNone(child)
        child_board, child_rng = child
        self.assertEqual(
            decode_board(np_u64(child_board))[0].tolist(),
            [4, 4, 0, 0],
        )
        self.assertEqual(child_rng.state, (12295, 1029, 1029, 25165824))
        self.assertEqual(child_rng.turn, 8)

    def test_deterministic_prefetch_prioritizes_best_move(self):
        result = query_handler.TablebaseLookupResult(
            board_encoded=0x11,
            full_pattern="L3_256",
            results={"left": 0.7, "right": 0.9, "down": 0.8, "up": 0.6},
            dtype="float64",
            best_move="right",
        )

        self.assertEqual(
            query_handler._direction_order(result),
            ["right", "left", "down", "up"],
        )

    async def test_tester_prefetch_streams_eight_complete_board_results(self):
        scheduler = TablebaseQueryScheduler(worker_count=4)
        session = GameSession("tester_deterministic_prefetch_test")
        session.user_id = 1
        session.auth_session_id = 2
        session.tester_full_pattern = "L3_256"
        session.tester_pattern = ["L3", "256"]
        session.tester_table_found = True
        session.tester_tablebase_provider_kind = "local"
        session.book_reader = ConstantReader()
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, :2] = 2
        session.board_encoded = np_u64(encode_board(board))
        websocket = RecordingWebSocket()

        try:
            with (
                patch.object(query_handler, "tablebase_query_scheduler", scheduler),
                patch.object(query_handler, "get_catalog_version", return_value="catalog-test"),
                patch.object(query_handler, "reserve_operation_tokens", return_value=object()),
                patch.object(query_handler, "finalize_reservation"),
                patch.object(query_handler, "get_token_balance", return_value={"total": 100}),
            ):
                await query_handler.handle_tablebase_query_action(
                    Action.TABLEBASE_QUERY,
                    {
                        "page": "tester",
                        "query_id": "query-deterministic",
                        "full_pattern": "L3_256",
                        "board_hex": f"{int(session.board_encoded):016x}",
                        "prefetch_rng": {
                            "version": 1,
                            "state": [1, 2, 3, 4],
                            "turn": 0,
                            "spawn_rate_4": 0.1,
                        },
                    },
                    session,
                    websocket,
                )
                while query_handler._TABLEBASE_QUERY_TASKS:
                    await asyncio.gather(
                        *list(query_handler._TABLEBASE_QUERY_TASKS),
                        return_exceptions=True,
                    )

            self.assertEqual(websocket.messages[0]["action"], "TABLEBASE_QUERY_RESULT")
            prefetch_messages = [
                message for message in websocket.messages
                if message["action"] == "TABLEBASE_PREFETCH"
            ]
            self.assertEqual(len(prefetch_messages), 8)
            self.assertTrue(
                all(len(message["data"]["entries"]) == 1 for message in prefetch_messages)
            )
            self.assertEqual(len(session.book_reader.calls), 9)
            depths = [message["data"]["entries"][0]["depth"] for message in prefetch_messages]
            self.assertEqual(depths.count(1), 3)
            self.assertGreater(max(depths), 1)
        finally:
            await scheduler.close()

    async def test_trainer_prefetch_uses_the_deterministic_rng_contract(self):
        scheduler = TablebaseQueryScheduler(worker_count=4)
        session = GameSession("trainer_deterministic_prefetch_test")
        session.user_id = 1
        session.auth_session_id = 2
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.use_variant = False
        session.book_reader = ConstantReader()
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, :2] = 2
        session.board_encoded = np_u64(encode_board(board))
        websocket = RecordingWebSocket()

        try:
            with (
                patch.object(query_handler, "tablebase_query_scheduler", scheduler),
                patch.object(query_handler, "get_catalog_version", return_value="catalog-test"),
                patch.object(query_handler, "reserve_operation_tokens", return_value=object()),
                patch.object(query_handler, "finalize_reservation"),
                patch.object(query_handler, "get_token_balance", return_value={"total": 100}),
            ):
                await query_handler.handle_tablebase_query_action(
                    Action.TABLEBASE_QUERY,
                    {
                        "page": "trainer",
                        "query_id": "trainer-deterministic",
                        "full_pattern": "L3_256",
                        "board_hex": f"{int(session.board_encoded):016x}",
                        "prefetch_rng": {
                            "version": 1,
                            "state": [1, 2, 3, 4],
                            "turn": 0,
                            "spawn_rate_4": 0.1,
                        },
                    },
                    session,
                    websocket,
                )
                while query_handler._TABLEBASE_QUERY_TASKS:
                    await asyncio.gather(
                        *list(query_handler._TABLEBASE_QUERY_TASKS),
                        return_exceptions=True,
                    )

            prefetch_messages = [
                message for message in websocket.messages
                if message["action"] == "TABLEBASE_PREFETCH"
            ]
            self.assertEqual(len(prefetch_messages), 8)
            depths = [message["data"]["entries"][0]["depth"] for message in prefetch_messages]
            self.assertEqual(depths.count(1), 3)
            self.assertGreater(max(depths), 1)
        finally:
            await scheduler.close()

    async def test_current_result_is_sent_before_eight_two_tile_prefetches(self):
        scheduler = TablebaseQueryScheduler(worker_count=4)
        session = GameSession("trainer_handler_test")
        session.user_id = 1
        session.auth_session_id = 2
        session.user_entitlement_tier = "supporter"
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.use_variant = False
        session.book_reader = ConstantReader()
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 2
        board[0, 1] = 2
        session.board_encoded = np_u64(encode_board(board))
        websocket = RecordingWebSocket()

        try:
            with (
                patch.object(query_handler, "tablebase_query_scheduler", scheduler),
                patch.object(query_handler, "get_catalog_version", return_value="catalog-test"),
                patch.object(query_handler, "reserve_operation_tokens", return_value=object()),
                patch.object(query_handler, "finalize_reservation") as finalize,
                patch.object(
                    query_handler,
                    "get_token_balance",
                    return_value={"bonus": 100, "paid": 0, "total": 100},
                ),
            ):
                handled = await query_handler.handle_tablebase_query_action(
                    Action.TABLEBASE_QUERY,
                    {
                        "page": "trainer",
                        "query_id": "query-1",
                        "full_pattern": "L3_256",
                        "board_hex": f"{int(session.board_encoded):016x}",
                    },
                    session,
                    websocket,
                )
                self.assertTrue(handled)
                while query_handler._TABLEBASE_QUERY_TASKS:
                    await asyncio.gather(
                        *list(query_handler._TABLEBASE_QUERY_TASKS),
                        return_exceptions=True,
                    )

            self.assertEqual(websocket.messages[0]["action"], "TABLEBASE_QUERY_RESULT")
            self.assertEqual(websocket.messages[1]["action"], "TABLEBASE_PREFETCH")
            entries = websocket.messages[1]["data"]["entries"]
            self.assertEqual(len(entries), 8)
            self.assertEqual(len(session.book_reader.calls), 9)
            finalize.assert_called_once()
            for entry in entries:
                child = decode_board(np_u64(int(entry["board_hex"], 16)))
                self.assertEqual(int(np.count_nonzero(child == 2)), 1)
                self.assertEqual(int(np.count_nonzero(child == 4)), 1)
        finally:
            await scheduler.close()

    async def test_cache_hit_is_still_charged_without_repeating_native_query(self):
        scheduler = TablebaseQueryScheduler(worker_count=4)
        session = GameSession("trainer_cache_charge_test")
        session.user_id = 1
        session.auth_session_id = 2
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.book_reader = ConstantReader()
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 2
        board[0, 1] = 2
        session.board_encoded = np_u64(encode_board(board))
        websocket = RecordingWebSocket()

        try:
            with (
                patch.object(query_handler, "tablebase_query_scheduler", scheduler),
                patch.object(query_handler, "get_catalog_version", return_value="catalog-test"),
                patch.object(
                    query_handler,
                    "reserve_operation_tokens",
                    side_effect=[object(), object()],
                ) as reserve,
                patch.object(query_handler, "finalize_reservation") as finalize,
                patch.object(query_handler, "get_token_balance", return_value={"total": 100}),
            ):
                for query_id in ("query-1", "query-2"):
                    await query_handler.handle_tablebase_query_action(
                        Action.TABLEBASE_QUERY,
                        {
                            "page": "trainer",
                            "query_id": query_id,
                            "full_pattern": "L3_256",
                            "board_hex": f"{int(session.board_encoded):016x}",
                        },
                        session,
                        websocket,
                    )
                    while query_handler._TABLEBASE_QUERY_TASKS:
                        await asyncio.gather(
                            *list(query_handler._TABLEBASE_QUERY_TASKS),
                            return_exceptions=True,
                        )

            self.assertEqual(reserve.call_count, 2)
            self.assertEqual(finalize.call_count, 2)
            self.assertEqual(len(session.book_reader.calls), 9)
        finally:
            await scheduler.close()

    async def test_insufficient_tokens_cancel_before_native_query_runs(self):
        scheduler = TablebaseQueryScheduler(worker_count=1)
        session = GameSession("trainer_insufficient_test")
        session.user_id = 1
        session.auth_session_id = 2
        session.current_pattern = "L3_256"
        session.pattern_settings = ["L3", "256"]
        session.book_reader = ConstantReader()
        session.board_encoded = np_u64(encode_board(np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)))
        websocket = RecordingWebSocket()

        try:
            with (
                patch.object(query_handler, "tablebase_query_scheduler", scheduler),
                patch.object(query_handler, "get_catalog_version", return_value="catalog-test"),
                patch.object(
                    query_handler,
                    "reserve_operation_tokens",
                    side_effect=InsufficientTokens(required_units=1000, balance_units=0),
                ),
            ):
                with self.assertRaises(InsufficientTokens):
                    await query_handler.handle_tablebase_query_action(
                        Action.TABLEBASE_QUERY,
                        {
                            "page": "trainer",
                            "query_id": "query-insufficient",
                            "full_pattern": "L3_256",
                            "board_hex": f"{int(session.board_encoded):016x}",
                        },
                        session,
                        websocket,
                    )
                await asyncio.sleep(0.02)
            self.assertEqual(session.book_reader.calls, [])
            self.assertEqual(websocket.messages, [])
        finally:
            await scheduler.close()


if __name__ == "__main__":
    unittest.main()
