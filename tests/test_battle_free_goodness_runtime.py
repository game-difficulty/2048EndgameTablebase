from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import numpy as np

from backend.auth.db import auth_db, init_auth_db
from backend.battle import repository
from backend.battle.core.errors import BattleServiceError
from backend.battle.modes.free_goodness import runtime
from backend.quota.service import get_token_balance
from backend.tablebase_query_service import TablebaseLookupResult
from engine_core.VBoardMover import encode_board


class FreeGoodnessRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_legacy_operation_stream_recovers_cumulative_product(self) -> None:
        blob = runtime._append_operation(
            b"", selected="left", executed="left", corrected=False,
            spawn_index=3, spawn_value=2, goodness=0.8,
        )
        blob = runtime._append_operation(
            blob, selected="right", executed="right", corrected=False,
            spawn_index=4, spawn_value=4, goodness=0.5,
        )
        self.assertAlmostEqual(runtime._legacy_goodness_product(blob), 0.4, places=4)

    async def asyncSetUp(self) -> None:
        self._old_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmp.name, "free-battle.sqlite3")
        init_auth_db()
        repository.init_battle_db()
        now = datetime.now(timezone.utc).isoformat()
        with auth_db() as db:
            ids = []
            for index in range(2):
                cursor = db.execute(
                    """
                    INSERT INTO users
                    (email, email_identity, password_hash, display_name,
                     registered_with_invite, role, status, created_at, updated_at)
                    VALUES (?, ?, 'hash', ?, 0, 'user', 'active', ?, ?)
                    """,
                    (
                        f"free-battle-{index}@example.com",
                        f"free-battle-{index}@example.com",
                        f"player-{index}",
                        now,
                        now,
                    ),
                )
                user_id = int(cursor.lastrowid)
                ids.append(user_id)
                db.execute(
                    "INSERT INTO token_accounts (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at) VALUES (?, 1000000000, 0, ?, ?)",
                    (user_id, now, now),
                )
        self.host_id, self.player_id = ids
        self.board = int(encode_board(np.array([
            [2, 2, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)))
        self.entry = {
            "pattern": "L3",
            "target": "128",
            "spawn_rate": 0.1,
            "_provider": "local",
        }
        self.lookup = TablebaseLookupResult(
            board_encoded=self.board,
            full_pattern="L3_128",
            results={"left": 0.9, "right": 0.89, "down": 0.7, "up": 0.6},
            dtype="uint32",
            best_move="left",
        )
        runtime._prepared.clear()
        runtime._prepared_complete.clear()

    async def asyncTearDown(self) -> None:
        tasks = list(runtime._prepare_state_tasks.values())
        runtime._prepare_state_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        runtime._prepared.clear()
        runtime._prepared_complete.clear()
        if self._old_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_db
        self._tmp.cleanup()

    def test_ranking_min_steps_defaults_and_validates_against_target(self) -> None:
        with patch(
            "backend.battle.modes.free_goodness.mode.resolve_tablebase",
            return_value=self.entry,
        ):
            defaults = runtime._free_mode.validate_settings({
                "full_pattern": "L3_128",
                "max_players": 2,
                "step_timeout_seconds": 90,
            })
            selected = runtime._free_mode.validate_settings({
                "full_pattern": "L3_128",
                "max_players": 2,
                "step_timeout_seconds": 90,
                "ranking_min_steps": 32,
            })
            public = runtime._free_mode.public_settings({
                "settings": {
                    **defaults,
                    "move_risk_min_absolute_increase": 0.002,
                },
            })
            self.assertEqual(defaults["ranking_min_steps"], 64)
            self.assertEqual(selected["ranking_min_steps"], 32)
            self.assertNotIn("move_risk_min_absolute_increase", defaults)
            self.assertNotIn("move_risk_min_absolute_increase", public)
            with self.assertRaisesRegex(ValueError, "invalid_ranking_min_steps"):
                runtime._free_mode.validate_settings({
                    "full_pattern": "L3_128",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                    "ranking_min_steps": 65,
                })

    def test_ranking_eligibility_requires_threshold_and_valid_finish(self) -> None:
        self.assertTrue(runtime._ranking_eligible(
            result_status="playing",
            finish_reason=None,
            progress=12,
            ranking_min_steps=12,
        ))
        self.assertTrue(runtime._ranking_eligible(
            result_status="completed",
            finish_reason="no_legal_move",
            progress=20,
            ranking_min_steps=12,
        ))
        self.assertFalse(runtime._ranking_eligible(
            result_status="completed",
            finish_reason="no_legal_move",
            progress=11,
            ranking_min_steps=12,
        ))
        self.assertFalse(runtime._ranking_eligible(
            result_status="timed_out",
            finish_reason="timed_out",
            progress=20,
            ranking_min_steps=12,
        ))

    async def test_certainty_tail_stops_after_the_target_merge_without_terminal_lookup(self) -> None:
        near_target = int(encode_board(np.array([
            [64, 64, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)))
        with (
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock()) as lookup,
        ):
            steps, final_board = await runtime._generate_certainty_tail(
                room={
                    "room_id": "room",
                    "host_user_id": self.host_id,
                    "full_pattern": "L3_128",
                    "pattern": "L3",
                    "target": 128,
                },
                round_id="round",
                user_id=self.host_id,
                board=near_target,
                results={"left": 1.0, "right": 1.0, "down": 0.0, "up": 0.0},
                seed_hex="11" * 32,
                sequence=3,
                risk_state=runtime.SpawnRiskState(),
            )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].direction, "left")
        self.assertTrue(runtime._contains_target(final_board, 128))
        lookup.assert_not_awaited()

    async def test_create_start_and_normal_moves_use_independent_state(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "ranking_min_steps": 12,
                    "step_timeout_seconds": 90,
                    "is_public": True,
                    "allow_spectators": True,
                    "chat_roles": ["host", "player", "spectator"],
                },
            )
            room = created["room"]
            self.assertEqual(room["mode_settings"]["ranking_min_steps"], 12)
            repository.join_room(room["room_code"], user_id=self.player_id)
            repository.set_member_ready(room["room_code"], user_id=self.host_id, ready=True)
            repository.set_member_ready(room["room_code"], user_id=self.player_id, ready=True)
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            self.assertEqual(started["round"]["status"], "running")
            self.assertEqual(len(started["results"]), 2)
            own = next(item for item in started["results"] if item["user_id"] == self.host_id)
            self.assertEqual(own["mode_data"]["board_hex"], f"{self.board:016x}")
            self.assertEqual(own["mode_data"]["ranking_min_steps"], 12)
            self.assertFalse(own["mode_data"]["ranking_eligible"])

            accepted = await runtime.handle_action_for_mode(
                room["room_code"],
                user_id=self.host_id,
                action="move",
                payload={
                    "round_id": started["round"]["round_id"],
                    "sequence": 1,
                    "direction": "left",
                },
            )
            self.assertEqual(accepted["sequence"], 1)
            self.assertEqual(accepted["selected_direction"], "left")
            self.assertFalse(accepted["corrected"])
            self.assertFalse(accepted["awaiting_ack"])
            self.assertTrue(accepted["timeout_at"])
            with auth_db() as db:
                awaiting = db.execute(
                    "SELECT state_status, timeout_at FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
                replay = db.execute(
                    "SELECT replay_blob, replay_move_count FROM battle_player_results WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
            self.assertEqual(awaiting["state_status"], "input")
            self.assertTrue(awaiting["timeout_at"])
            self.assertEqual(int(replay["replay_move_count"]), 1)
            self.assertEqual(len(bytes(replay["replay_blob"])), 25)
            host_view = runtime.room_snapshot(room["room_code"], user_id=self.host_id)
            player_view = runtime.room_snapshot(room["room_code"], user_id=self.player_id)
            host_result = next(
                item for item in host_view["results"] if item["user_id"] == self.host_id
            )
            hidden_result = next(
                item for item in player_view["results"] if item["user_id"] == self.host_id
            )
            self.assertEqual(host_result["mode_data"]["last_step"]["sequence"], 1)
            self.assertNotIn("last_step", hidden_result["mode_data"])

            with auth_db() as db:
                state = db.execute(
                    "SELECT * FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
                round_row = db.execute(
                    "SELECT mode_state_json FROM battle_rounds WHERE round_id = ?",
                    (started["round"]["round_id"],),
                ).fetchone()
            self.assertEqual(state["state_status"], "input")
            self.assertEqual(state["step_index"], 1)
            self.assertEqual(len(bytes(state["operation_blob"])), 4)
            self.assertIn('"lookup_hit_steps":1', round_row["mode_state_json"])

            host_view = runtime.room_snapshot(room["room_code"], user_id=self.host_id)
            host_result = next(
                item for item in host_view["results"] if item["user_id"] == self.host_id
            )
            self.assertFalse(host_result["mode_data"]["ranking_eligible"])

            player_move = await runtime.handle_action_for_mode(
                room["room_code"],
                user_id=self.player_id,
                action="move",
                payload={
                    "round_id": started["round"]["round_id"],
                    "sequence": 1,
                    "direction": "right",
                },
            )
            self.assertEqual(player_move["executed_direction"], "right")
            self.assertFalse(player_move["corrected"])
            with auth_db() as db:
                boards = db.execute(
                    "SELECT user_id, board_state FROM battle_free_player_states WHERE round_id = ? ORDER BY user_id",
                    (started["round"]["round_id"],),
                ).fetchall()
            self.assertEqual(len(boards), 2)
            self.assertNotEqual(boards[0]["board_state"], boards[1]["board_state"])

            runtime.forfeit_round_for_mode(
                room["room_code"], user_id=self.host_id, round_id=started["round"]["round_id"]
            )
            runtime.forfeit_round_for_mode(
                room["room_code"], user_id=self.player_id, round_id=started["round"]["round_id"]
            )
            balance = get_token_balance(self.host_id)
            self.assertEqual(balance["total"], 999_998)
            with auth_db() as db:
                settlements = db.execute(
                    "SELECT COUNT(*) AS count FROM token_ledger WHERE user_id = ? AND event_type = 'finalize' AND operation_key = 'battle_free_lookup_settlement'",
                    (self.host_id,),
                ).fetchone()
            self.assertEqual(int(settlements["count"]), 1)

    async def test_corrected_move_still_waits_for_ack_before_next_input(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            room = created["room"]
            repository.set_member_ready(
                room["room_code"], user_id=self.host_id, ready=True
            )
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            accepted = await runtime.handle_action_for_mode(
                room["room_code"],
                user_id=self.host_id,
                action="move",
                payload={
                    "round_id": started["round"]["round_id"],
                    "sequence": 1,
                    "direction": "down",
                },
            )
            self.assertTrue(accepted["corrected"])
            self.assertTrue(accepted["awaiting_ack"])
            self.assertIsNone(accepted["timeout_at"])
            with auth_db() as db:
                waiting = db.execute(
                    "SELECT state_status, timeout_at FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
            self.assertEqual(waiting["state_status"], "awaiting_ack")
            self.assertIsNone(waiting["timeout_at"])

            acknowledged = await runtime.handle_action_for_mode(
                room["room_code"],
                user_id=self.host_id,
                action="step_ready_ack",
                payload={
                    "round_id": started["round"]["round_id"],
                    "sequence": 1,
                },
            )
            self.assertTrue(acknowledged["timeout_at"])
            runtime.forfeit_round_for_mode(
                room["room_code"],
                user_id=self.host_id,
                round_id=started["round"]["round_id"],
            )

    async def test_ready_host_can_start_a_solo_free_round(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "ranking_min_steps": 12,
                    "step_timeout_seconds": 90,
                    "is_public": True,
                    "allow_spectators": True,
                    "chat_roles": ["host", "player", "spectator"],
                },
            )
            room = created["room"]
            repository.set_member_ready(
                room["room_code"], user_id=self.host_id, ready=True
            )

            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            self.assertEqual(started["round"]["status"], "running")
            self.assertEqual(len(started["results"]), 1)
            self.assertEqual(int(started["results"][0]["user_id"]), self.host_id)
            with auth_db() as db:
                states = db.execute(
                    "SELECT user_id FROM battle_free_player_states WHERE round_id = ?",
                    (started["round"]["round_id"],),
                ).fetchall()
            self.assertEqual([int(state["user_id"]) for state in states], [self.host_id])

    async def test_explicit_unplayable_board_is_rejected_and_reservation_refunded(self) -> None:
        unavailable = TablebaseLookupResult(
            board_encoded=self.board,
            full_pattern="L3_128",
            results={"left": 0.0, "right": 0.0, "down": 0.0, "up": 0.0},
            dtype="uint32",
            best_move="left",
        )
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=unavailable)),
        ):
            with self.assertRaises(BattleServiceError) as raised:
                await runtime.create_room_for_mode(
                    user_id=self.host_id,
                    session_id=None,
                    payload={
                        "full_pattern": "L3_128",
                        "initial_board": f"{self.board:016x}",
                        "max_players": 2,
                        "step_timeout_seconds": 90,
                    },
                )
        self.assertEqual(raised.exception.code, "INITIAL_BOARD_UNUSABLE")
        self.assertEqual(get_token_balance(self.host_id)["total"], 1_000_000)
        with auth_db() as db:
            active = db.execute(
                "SELECT COUNT(*) AS count FROM battle_members WHERE user_id = ? AND status = 'active'",
                (self.host_id,),
            ).fetchone()
        self.assertEqual(int(active["count"]), 0)

    async def test_initial_certainty_completes_without_charging_a_step(self) -> None:
        certain = TablebaseLookupResult(
            board_encoded=self.board,
            full_pattern="L3_128",
            results={"left": 1.0, "right": 0.9, "down": 0.8, "up": 0.7},
            dtype="uint32",
            best_move="left",
        )
        target_board = 7
        certainty_tail = [runtime.CertaintyTailStep(
            previous_board=self.board,
            next_board=target_board,
            direction="left",
            spawn_index=3,
            spawn_value=2,
            rates={"left": 1.0, "right": 0.9, "up": 0.7, "down": 0.8},
        )]
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=certain)),
            patch.object(
                runtime,
                "_generate_certainty_tail",
                new=AsyncMock(return_value=(certainty_tail, target_board)),
            ),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            repository.join_room(created["room"]["room_code"], user_id=self.player_id)
            repository.set_member_ready(
                created["room"]["room_code"], user_id=self.host_id, ready=True
            )
            repository.set_member_ready(
                created["room"]["room_code"], user_id=self.player_id, ready=True
            )
            started = await runtime.start_room_for_mode(
                created["room"]["room_code"], user_id=self.host_id, session_id=None
            )
        self.assertEqual(started["status"], "waiting")
        self.assertEqual(started["round"]["status"], "completed")
        self.assertTrue(all(item["status"] == "completed" for item in started["results"]))
        self.assertTrue(
            all(item["mode_data"]["finish_reason"] == "target_reached" for item in started["results"])
        )
        self.assertTrue(all(item["route_index"] == 0 for item in started["results"]))
        self.assertEqual(started["results"][0]["mode_data"]["board_hex"], f"{target_board:016x}")
        self.assertEqual(len(started["results"][0]["mode_data"]["auto_steps"]), 1)
        self.assertEqual(get_token_balance(self.host_id)["total"], 1_000_000)

    async def test_move_certainty_plays_to_target_without_charging_tail(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            room = created["room"]
            repository.set_member_ready(room["room_code"], user_id=self.host_id, ready=True)
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            moved_board = runtime._moved_boards(self.board, use_variant=False)["left"]
            spawn_index = runtime._empty_indices(moved_board)[0]
            certain_board = runtime._spawn_board(moved_board, spawn_index, 2)
            certain_results = {"left": 1.0, "right": 0.9, "down": 0.8, "up": 0.7}
            candidate = runtime.PreparedSpawn(
                executed_direction="left",
                moved_board=moved_board,
                next_board=certain_board,
                spawn_index=spawn_index,
                spawn_value=2,
                attempt_index=0,
                next_results=certain_results,
                next_dtype="uint32",
                next_best_success=1.0,
                risk_multiplier=1.0,
                risk_state=runtime.SpawnRiskState(),
            )
            target_board = 7
            tail = [runtime.CertaintyTailStep(
                previous_board=certain_board,
                next_board=target_board,
                direction="left",
                spawn_index=3,
                spawn_value=2,
                rates=certain_results,
            )]
            with (
                patch.object(
                    runtime,
                    "_select_prepared_spawn",
                    new=AsyncMock(return_value=(candidate, "left", None)),
                ),
                patch.object(
                    runtime,
                    "_generate_certainty_tail",
                    new=AsyncMock(return_value=(tail, target_board)),
                ),
            ):
                accepted = await runtime.handle_action_for_mode(
                    room["room_code"],
                    user_id=self.host_id,
                    action="move",
                    payload={
                        "round_id": started["round"]["round_id"],
                        "sequence": 1,
                        "direction": "left",
                    },
                )

        self.assertTrue(accepted["complete"])
        self.assertEqual(accepted["finish_reason"], "target_reached")
        self.assertEqual(accepted["board_hex"], f"{certain_board:016x}")
        self.assertEqual(accepted["auto_final_board_hex"], f"{target_board:016x}")
        self.assertEqual(len(accepted["auto_steps"]), 1)
        with auth_db() as db:
            state = db.execute(
                "SELECT board_state, step_index, operation_blob FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                (started["round"]["round_id"], self.host_id),
            ).fetchone()
            result = db.execute(
                "SELECT route_index, replay_move_count FROM battle_player_results WHERE round_id = ? AND user_id = ?",
                (started["round"]["round_id"], self.host_id),
            ).fetchone()
            round_row = db.execute(
                "SELECT mode_state_json FROM battle_rounds WHERE round_id = ?",
                (started["round"]["round_id"],),
            ).fetchone()
        self.assertEqual(state["board_state"], f"{target_board:016x}")
        self.assertEqual(int(state["step_index"]), 1)
        self.assertEqual(len(bytes(state["operation_blob"])), 4)
        self.assertEqual(int(result["route_index"]), 1)
        self.assertEqual(int(result["replay_move_count"]), 2)
        self.assertIn('"lookup_hit_steps":1', round_row["mode_state_json"])
        self.assertEqual(get_token_balance(self.host_id)["total"], 999_999)

    async def test_prefetch_stops_each_direction_after_first_accepted_candidate(self) -> None:
        calls = []

        async def fake_candidate(**kwargs):
            calls.append((kwargs["direction"], kwargs["attempt"]))
            if kwargs["direction"] == "left" and kwargs["attempt"] == 0:
                return runtime.PreparedSpawn(
                    executed_direction="left",
                    moved_board=self.board,
                    next_board=self.board,
                    spawn_index=0,
                    spawn_value=2,
                    attempt_index=0,
                    next_results={"left": 0.9},
                    next_dtype="uint32",
                    next_best_success=0.9,
                    risk_multiplier=1.0,
                    risk_state=runtime.SpawnRiskState(),
                )
            return None

        choices = [(0, 0, 2), (1, 1, 2), (2, 2, 4)]
        with (
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(
                runtime,
                "_candidate_choices",
                side_effect=lambda *_args, **_kwargs: iter(choices),
            ),
            patch.object(runtime, "_evaluate_spawn_candidate", new=fake_candidate),
        ):
            prepared = await runtime._prepare_directions(
                room={
                    "full_pattern": "L3_128",
                    "pattern": "L3",
                    "target": 128,
                    "host_user_id": self.host_id,
                },
                round_id="round",
                user_id=self.host_id,
                sequence=0,
                direction_states={
                    "left": (self.board, 0.9),
                    "right": (self.board, 0.8),
                    "down": (self.board, 0.7),
                    "up": (self.board, 0.6),
                },
                risk_state=runtime.SpawnRiskState(),
                seed_hex="11" * 32,
                lane="prefetch",
            )
        self.assertIn("left", prepared)
        self.assertEqual(
            [attempt for direction, attempt in calls if direction == "left"],
            [0],
        )
        for direction in ("right", "down", "up"):
            self.assertEqual(
                [attempt for item_direction, attempt in calls if item_direction == direction],
                [0, 1, 2],
            )

    async def test_prefetch_publishes_each_direction_before_the_batch_finishes(self) -> None:
        release_others = asyncio.Event()
        left_candidate = runtime.PreparedSpawn(
            executed_direction="left",
            moved_board=self.board,
            next_board=self.board,
            spawn_index=0,
            spawn_value=2,
            attempt_index=0,
            next_results={"left": 0.9},
            next_dtype="uint32",
            next_best_success=0.9,
            risk_multiplier=1.0,
            risk_state=runtime.SpawnRiskState(),
        )

        async def fake_direction(**kwargs):
            if kwargs["direction"] == "left":
                return left_candidate
            await release_others.wait()
            return None

        key = runtime._candidate_key("round", self.host_id, 0, self.board)
        with patch.object(runtime, "_prepare_direction", new=fake_direction):
            task = asyncio.create_task(runtime._prepare_directions(
                room={
                    "full_pattern": "L3_128",
                    "pattern": "L3",
                    "target": 128,
                    "host_user_id": self.host_id,
                },
                round_id="round",
                user_id=self.host_id,
                sequence=0,
                direction_states={
                    "left": (self.board, 0.9),
                    "right": (self.board, 0.8),
                },
                risk_state=runtime.SpawnRiskState(),
                seed_hex="11" * 32,
                lane="prefetch",
                prepared_key=key,
            ))
            for _attempt in range(10):
                if "left" in runtime._prepared.get(key, {}):
                    break
                await asyncio.sleep(0)
            self.assertIs(runtime._prepared[key]["left"], left_candidate)
            self.assertFalse(task.done())
            release_others.set()
            await asyncio.wait_for(task, timeout=1)

    async def test_prefetch_includes_direction_below_absolute_move_risk_floor(self) -> None:
        captured = {}

        async def capture_directions(**kwargs):
            captured.update(kwargs["direction_states"])
            return {}

        with patch.object(runtime, "_prepare_directions", new=capture_directions):
            await runtime._prepare_state(
                room={
                    "full_pattern": "L3_128",
                    "pattern": "L3",
                    "target": 128,
                    "host_user_id": self.host_id,
                },
                round_id="risk-floor-round",
                user_id=self.host_id,
                sequence=0,
                board=self.board,
                results={
                    "left": 0.999,
                    "right": 0.9988,
                    "down": 0.5,
                    "up": 0.0,
                },
                risk_state=runtime.SpawnRiskState(),
                seed_hex="22" * 32,
            )

        self.assertIn("left", captured)
        self.assertIn("right", captured)

    async def test_host_closing_running_room_settles_consumed_steps(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            room = created["room"]
            repository.join_room(room["room_code"], user_id=self.player_id)
            repository.set_member_ready(room["room_code"], user_id=self.host_id, ready=True)
            repository.set_member_ready(room["room_code"], user_id=self.player_id, ready=True)
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            await runtime.handle_action_for_mode(
                room["room_code"],
                user_id=self.host_id,
                action="move",
                payload={
                    "round_id": started["round"]["round_id"],
                    "sequence": 1,
                    "direction": "left",
                },
            )
        runtime.settle_unstarted_round_for_mode(
            room["room_id"], reason="battle_host_closed_room"
        )
        repository.close_room(room["room_code"], host_user_id=self.host_id)
        self.assertEqual(get_token_balance(self.host_id)["total"], 999_999)
        with auth_db() as db:
            round_row = db.execute(
                "SELECT status, reservation_status FROM battle_rounds WHERE round_id = ?",
                (started["round"]["round_id"],),
            ).fetchone()
            result_rows = db.execute(
                "SELECT status FROM battle_player_results WHERE round_id = ?",
                (started["round"]["round_id"],),
            ).fetchall()
        self.assertEqual(round_row["status"], "completed")
        self.assertEqual(round_row["reservation_status"], "finalized")
        self.assertTrue(all(row["status"] == "disqualified" for row in result_rows))

    async def test_startup_keeps_active_round_budget_reserved(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            room = created["room"]
            repository.join_room(room["room_code"], user_id=self.player_id)
            repository.set_member_ready(room["room_code"], user_id=self.host_id, ready=True)
            repository.set_member_ready(room["room_code"], user_id=self.player_id, ready=True)
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )
            await runtime.startup()

        with auth_db() as db:
            round_row = db.execute(
                "SELECT status, reservation_status FROM battle_rounds WHERE round_id = ?",
                (started["round"]["round_id"],),
            ).fetchone()
        self.assertEqual(round_row["status"], "running")
        self.assertEqual(round_row["reservation_status"], "reserved")

        runtime.settle_unstarted_round_for_mode(
            room["room_id"], reason="battle_test_cleanup"
        )
        repository.close_room(room["room_code"], host_user_id=self.host_id)
        await runtime.shutdown()

    async def test_step_ready_does_not_wait_for_next_step_prefetch(self) -> None:
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=self.lookup)),
        ):
            created = await runtime.create_room_for_mode(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "initial_board": f"{self.board:016x}",
                    "max_players": 2,
                    "step_timeout_seconds": 90,
                },
            )
            room = created["room"]
            repository.join_room(room["room_code"], user_id=self.player_id)
            repository.set_member_ready(room["room_code"], user_id=self.host_id, ready=True)
            repository.set_member_ready(room["room_code"], user_id=self.player_id, ready=True)
            started = await runtime.start_room_for_mode(
                room["room_code"], user_id=self.host_id, session_id=None
            )

            prefetch_started = asyncio.Event()
            release_prefetch = asyncio.Event()

            async def delayed_prefetch(**_kwargs):
                prefetch_started.set()
                await release_prefetch.wait()
                return {}

            with patch.object(runtime, "_prepare_state", new=delayed_prefetch):
                accepted = await runtime.handle_action_for_mode(
                    room["room_code"],
                    user_id=self.host_id,
                    action="move",
                    payload={
                        "round_id": started["round"]["round_id"],
                        "sequence": 1,
                        "direction": "left",
                    },
                )
                await asyncio.wait_for(prefetch_started.wait(), timeout=1)
                with auth_db() as db:
                    state = db.execute(
                        "SELECT state_status, timeout_at FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                        (started["round"]["round_id"], self.host_id),
                    ).fetchone()
                self.assertEqual(state["state_status"], "input")
                self.assertTrue(state["timeout_at"])
                release_prefetch.set()
                tasks = list(runtime._prepare_state_tasks.values())
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)

        self.assertFalse(accepted["awaiting_ack"])
        runtime.settle_unstarted_round_for_mode(
            room["room_id"], reason="battle_test_cleanup"
        )
        repository.close_room(room["room_code"], host_user_id=self.host_id)


if __name__ == "__main__":
    unittest.main()
