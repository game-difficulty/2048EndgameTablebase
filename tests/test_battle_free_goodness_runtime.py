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

    async def asyncTearDown(self) -> None:
        runtime._prepared.clear()
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
            self.assertEqual(defaults["ranking_min_steps"], 64)
            self.assertEqual(selected["ranking_min_steps"], 32)
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

    async def test_create_start_move_and_ack_use_independent_state(self) -> None:
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
            self.assertTrue(accepted["awaiting_ack"])
            with auth_db() as db:
                awaiting = db.execute(
                    "SELECT state_status, timeout_at FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
                replay = db.execute(
                    "SELECT replay_blob, replay_move_count FROM battle_player_results WHERE round_id = ? AND user_id = ?",
                    (started["round"]["round_id"], self.host_id),
                ).fetchone()
            self.assertEqual(awaiting["state_status"], "awaiting_ack")
            self.assertIsNone(awaiting["timeout_at"])
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
        with (
            patch("backend.battle.modes.free_goodness.mode.resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "resolve_tablebase", return_value=self.entry),
            patch.object(runtime, "_lookup", new=AsyncMock(return_value=certain)),
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
            all(item["mode_data"]["finish_reason"] == "certainty" for item in started["results"])
        )
        self.assertEqual(get_token_balance(self.host_id)["total"], 1_000_000)

    async def test_exact_prefetch_uses_four_first_candidates_then_eight_item_wave(self) -> None:
        calls = []
        first_wave_ready = asyncio.Event()
        second_wave_ready = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()

        async def fake_candidate(**kwargs):
            calls.append((kwargs["direction"], kwargs["attempt"]))
            if len(calls) <= 4:
                if len(calls) == 4:
                    first_wave_ready.set()
                await release_first.wait()
            else:
                if len(calls) == 12:
                    second_wave_ready.set()
                await release_second.wait()
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
                    "down": (self.board, 0.7),
                    "up": (self.board, 0.6),
                },
                risk_state=runtime.SpawnRiskState(),
                seed_hex="11" * 32,
                lane="prefetch",
            ))
            await asyncio.wait_for(first_wave_ready.wait(), timeout=1)
            self.assertEqual(len(calls), 4)
            self.assertEqual({direction for direction, _attempt in calls}, {
                "left", "right", "down", "up",
            })
            release_first.set()
            await asyncio.wait_for(second_wave_ready.wait(), timeout=1)
            self.assertEqual(len(calls), 12)
            release_second.set()
            prepared = await asyncio.wait_for(task, timeout=1)
        self.assertEqual(prepared, {})

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

    async def test_step_ready_waits_for_next_step_prefetch(self) -> None:
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
                move_task = asyncio.create_task(runtime.handle_action_for_mode(
                    room["room_code"],
                    user_id=self.host_id,
                    action="move",
                    payload={
                        "round_id": started["round"]["round_id"],
                        "sequence": 1,
                        "direction": "left",
                    },
                ))
                await asyncio.wait_for(prefetch_started.wait(), timeout=1)
                self.assertFalse(move_task.done())
                with auth_db() as db:
                    state = db.execute(
                        "SELECT state_status, timeout_at FROM battle_free_player_states WHERE round_id = ? AND user_id = ?",
                        (started["round"]["round_id"], self.host_id),
                    ).fetchone()
                self.assertEqual(state["state_status"], "resolving")
                self.assertIsNone(state["timeout_at"])
                release_prefetch.set()
                accepted = await asyncio.wait_for(move_task, timeout=1)

        self.assertTrue(accepted["awaiting_ack"])
        runtime.settle_unstarted_round_for_mode(
            room["room_id"], reason="battle_test_cleanup"
        )
        repository.close_room(room["room_code"], host_user_id=self.host_id)


if __name__ == "__main__":
    unittest.main()
