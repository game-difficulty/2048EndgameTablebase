from __future__ import annotations

import asyncio
import os
import struct
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

if "cpuinfo" not in sys.modules:
    sys.modules["cpuinfo"] = types.SimpleNamespace(get_cpu_info=lambda: {"flags": []})

from backend.auth.db import auth_db, init_auth_db
from backend.battle import repository, service
from backend.battle.modes.goodness import runtime as goodness_runtime
from backend.battle.route_codec import RATE_SCALE, RouteStep, encode_changes, encode_route
from backend.battle.route_generator import GeneratedBattleRoute
from backend.quota.service import (
    cancel_reservation,
    finalize_reservation,
    get_token_balance,
    reserve_operation_tokens,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BattleServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_auth_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "battle-service.sqlite3")
        init_auth_db()
        repository.init_battle_db()
        self.host_id = self._create_user("battle-host@example.com", 1_000_000)
        self.player_id = self._create_user("battle-player@example.com", 1_000_000)
        self.spectator_id = self._create_user("battle-spectator@example.com", 1_000_000)

    async def asyncTearDown(self) -> None:
        tasks = list(service._route_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        service._route_tasks.clear()
        if self._old_auth_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_auth_db
        self._tmpdir.cleanup()

    def _create_user(self, email: str, paid_units: int) -> int:
        now = _now()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name,
                 registered_with_invite, role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, 0, 'user', 'active', ?, ?)
                """,
                (email, email, email.split("@", 1)[0], now, now),
            )
            user_id = int(cursor.lastrowid)
            db.execute(
                """
                INSERT INTO token_accounts
                (user_id, bonus_balance_units, paid_balance_units, created_at, updated_at)
                VALUES (?, 0, ?, ?, ?)
                """,
                (user_id, int(paid_units), now, now),
            )
        return user_id

    def _expire_creation_cooldown(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(seconds=31)).isoformat()
        with auth_db() as db:
            db.execute(
                "UPDATE battle_rooms SET created_at = ? WHERE host_user_id = ?",
                (old, self.host_id),
            )

    @staticmethod
    def _generated_route(step_count: int = 1) -> GeneratedBattleRoute:
        initial_board = 0x11
        rates = (RATE_SCALE // 4, RATE_SCALE // 2, RATE_SCALE, RATE_SCALE * 3 // 4)
        blob = encode_route(
            initial_board,
            [RouteStep(encode_changes("left", 0, 2), rates)] * int(step_count),
        )
        return GeneratedBattleRoute(
            route_blob=blob,
            step_count=int(step_count),
            certainty_step=None,
            termination_reason="max_steps",
            initial_board=initial_board,
            available_layers=8,
        )

    async def _create_ready_room(
        self,
        *,
        step_timeout_seconds: int = 90,
        max_players: int = 2,
        allow_spectators: bool = True,
        generated_route: GeneratedBattleRoute | None = None,
    ) -> dict:
        with (
            patch.object(
                goodness_runtime,
                "resolve_tablebase",
                return_value={"pattern": "L3", "target": 128, "full_pattern": "L3_128"},
            ),
            patch.object(
                goodness_runtime,
                "generate_battle_route",
                new=AsyncMock(return_value=generated_route or self._generated_route()),
            ),
        ):
            created = await service.create_room(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "max_players": max_players,
                    "step_timeout_seconds": step_timeout_seconds,
                    "is_public": True,
                    "allow_spectators": allow_spectators,
                },
            )
            round_id = str(created["room"]["round"]["round_id"])
            await service._route_tasks[round_id]
        return repository.get_room(created["room"]["room_id"])

    async def test_step_timeout_accepts_five_second_increments_at_boundaries(self) -> None:
        room = await self._create_ready_room(step_timeout_seconds=5)
        self.assertEqual(room["step_timeout_seconds"], 5)
        service.leave_room(room["room_code"], user_id=self.host_id)
        self._expire_creation_cooldown()

        room = await self._create_ready_room(step_timeout_seconds=600)
        self.assertEqual(room["step_timeout_seconds"], 600)
        service.leave_room(room["room_code"], user_id=self.host_id)

        self.assertNotIn(6, service.VALID_STEP_TIMEOUTS)
        self.assertNotIn(605, service.VALID_STEP_TIMEOUTS)

    async def test_host_cannot_enter_spectator_seat(self) -> None:
        room = await self._create_ready_room(allow_spectators=True)
        with self.assertRaises(service.BattleServiceError) as raised:
            service.set_role(
                room["room_code"], user_id=self.host_id, role="spectator"
            )
        self.assertEqual(raised.exception.code, "HOST_CANNOT_SPECTATE")

        host_view = repository.get_room(room["room_code"])
        host_member = next(
            member
            for member in host_view["members"]
            if int(member["user_id"]) == self.host_id
        )
        self.assertEqual(host_member["role"], "player")

        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        switched = service.set_role(
            room["room_code"], user_id=self.player_id, role="spectator"
        )
        player_member = next(
            member
            for member in switched["members"]
            if int(member["user_id"]) == self.player_id
        )
        self.assertEqual(player_member["role"], "spectator")
        service.leave_room(room["room_code"], user_id=self.host_id)

    async def test_wrong_move_correction_does_not_consume_next_step_timeout(self) -> None:
        room = await self._create_ready_room(
            step_timeout_seconds=30,
            generated_route=self._generated_route(step_count=2),
        )
        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        round_id = str(started["round"]["round_id"])

        choice = service.record_choice(
            room["room_code"],
            user_id=self.host_id,
            round_id=round_id,
            sequence=1,
            route_index=0,
            direction="right",
        )
        self.assertTrue(choice["wrong"])
        self.assertEqual(choice["correction_seconds"], 15)
        self.assertFalse(choice["complete"])
        before = repository.get_room(room["room_code"])
        before_result = next(
            result for result in before["results"] if int(result["user_id"]) == self.host_id
        )
        extended_deadline = datetime.fromisoformat(before_result["timeout_at"])
        self.assertGreaterEqual(
            (extended_deadline - datetime.now(timezone.utc)).total_seconds(),
            44,
        )

        resumed = service.handle_mode_action(
            room["room_code"],
            user_id=self.host_id,
            action="correction_complete",
            payload={
                "round_id": round_id,
                "sequence": 1,
                "route_index": 1,
            },
        )
        resumed_deadline = datetime.fromisoformat(resumed["timeout_at"])
        remaining = (resumed_deadline - datetime.now(timezone.utc)).total_seconds()
        self.assertGreaterEqual(remaining, 29)
        self.assertLessEqual(remaining, 30)

        repeated = service.handle_mode_action(
            room["room_code"],
            user_id=self.host_id,
            action="correction_complete",
            payload={
                "round_id": round_id,
                "sequence": 1,
                "route_index": 1,
            },
        )
        self.assertEqual(repeated["timeout_at"], resumed["timeout_at"])

    async def test_forfeit_ends_only_the_players_round_and_keeps_membership(self) -> None:
        room = await self._create_ready_room(
            generated_route=self._generated_route(step_count=2),
        )
        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        round_id = str(started["round"]["round_id"])

        first = service.forfeit_round(
            room["room_code"], user_id=self.host_id, round_id=round_id
        )
        host_result = next(
            item for item in first["results"] if int(item["user_id"]) == self.host_id
        )
        self.assertEqual(first["status"], "running")
        self.assertEqual(host_result["status"], "disqualified")
        self.assertEqual(host_result["mode_data"]["finish_reason"], "forfeit")
        self.assertTrue(any(int(item["user_id"]) == self.host_id for item in first["members"]))

        repeated = service.forfeit_round(
            room["room_code"], user_id=self.host_id, round_id=round_id
        )
        self.assertEqual(repeated["revision"], first["revision"])

        completed = service.forfeit_round(
            room["room_code"], user_id=self.player_id, round_id=round_id
        )
        self.assertEqual(completed["status"], "waiting")
        self.assertEqual(completed["round"]["status"], "completed")

    async def test_full_round_returns_to_same_lobby_and_resets_ready(self) -> None:
        room = await self._create_ready_room()
        self.assertEqual(room["status"], "waiting")
        self.assertEqual(get_token_balance(self.host_id)["total"], 900)

        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        self.assertEqual(started["status"], "running")
        player_snapshot = service.room_snapshot(
            room["room_code"], user_id=self.player_id
        )
        self.assertEqual(player_snapshot["status"], "running")
        self.assertEqual(player_snapshot["viewer"]["role"], "player")
        self.assertIsNotNone(player_snapshot["route"])
        self.assertIsNotNone(
            next(
                result
                for result in player_snapshot["results"]
                if int(result["user_id"]) == self.player_id
            )
        )
        self.assertEqual(get_token_balance(self.host_id)["total"], 900)
        with auth_db() as db:
            reservation_status = db.execute(
                "SELECT reservation_status FROM battle_rounds WHERE round_id = ?",
                (started["round"]["round_id"],),
            ).fetchone()["reservation_status"]
        self.assertEqual(reservation_status, "finalized")
        started_revision = int(started["revision"])

        round_id = str(started["round"]["round_id"])
        host_choice = service.record_choice(
            room["room_code"],
            user_id=self.host_id,
            round_id=round_id,
            sequence=1,
            route_index=0,
            direction="left",
        )
        progressed = service.room_snapshot(room["room_code"], user_id=self.host_id)
        self.assertGreater(int(progressed["revision"]), started_revision)
        self.assertTrue(all("board_state" not in result for result in progressed["results"]))
        with auth_db() as db:
            stored = db.execute(
                "SELECT board_state FROM battle_player_results WHERE round_id = ? AND user_id = ?",
                (round_id, self.host_id),
            ).fetchone()
        self.assertIsNotNone(stored["board_state"])

        player_choice = service.record_choice(
            room["room_code"],
            user_id=self.player_id,
            round_id=round_id,
            sequence=1,
            route_index=0,
            direction="right",
        )
        self.assertTrue(host_choice["complete"])
        self.assertTrue(player_choice["complete"])
        self.assertLess(player_choice["goodness_of_fit"], host_choice["goodness_of_fit"])

        host_replay, replay_metadata = service.player_replay_payload(
            room["room_code"], round_id, user_id=self.host_id
        )
        self.assertEqual(len(host_replay), 50)
        board, change, left, right, up, down = struct.unpack(
            "<QB4I", host_replay[:25]
        )
        self.assertEqual(board, 0x11)
        self.assertEqual((change >> 5) & 0b11, 0)
        self.assertEqual(
            (left, right, up, down),
            (4_000_000_000, 3_000_000_000, 1_000_000_000, 2_000_000_000),
        )
        self.assertEqual(replay_metadata["full_pattern"], "L3_128")
        self.assertEqual(replay_metadata["move_count"], 1)
        with self.assertRaises(service.BattleServiceError) as replay_error:
            service.player_replay_payload(
                room["room_code"], round_id, user_id=self.player_id + 10_000
            )
        self.assertEqual(replay_error.exception.code, "BATTLE_REPLAY_NOT_FOUND")

        finished = repository.get_room(room["room_code"])
        self.assertEqual(finished["status"], "waiting")
        self.assertEqual(finished["round"]["status"], "completed")
        self.assertTrue(all(not member["ready"] for member in finished["members"]))
        self.assertEqual(finished["room_code"], room["room_code"])

    async def test_two_ready_players_can_start_and_unready_players_become_spectators(self) -> None:
        room = await self._create_ready_room(max_players=8)
        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        extra_ids = [
            self._create_user(f"battle-extra-{index}@example.com", 1_000_000)
            for index in range(2)
        ]
        for user_id in extra_ids:
            service.join_room(room["room_code"], user_id=user_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        self.assertEqual(started["status"], "running")
        self.assertEqual(len(started["results"]), 2)
        roles = {int(member["user_id"]): member["role"] for member in started["members"]}
        self.assertEqual(roles[self.host_id], "player")
        self.assertEqual(roles[self.player_id], "player")
        self.assertTrue(all(roles[user_id] == "spectator" for user_id in extra_ids))

    async def test_ready_host_can_start_and_finish_a_solo_round(self) -> None:
        room = await self._create_ready_room(max_players=2)
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)

        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        self.assertEqual(started["status"], "running")
        self.assertEqual(len(started["results"]), 1)
        self.assertEqual(int(started["results"][0]["user_id"]), self.host_id)

        completed = service.record_choice(
            room["room_code"],
            user_id=self.host_id,
            round_id=str(started["round"]["round_id"]),
            sequence=1,
            route_index=0,
            direction="left",
        )
        self.assertTrue(completed["complete"])
        finished = repository.get_room(room["room_code"])
        self.assertEqual(finished["status"], "waiting")
        self.assertEqual(finished["round"]["status"], "completed")

    async def test_unready_players_are_kicked_when_spectating_is_disabled(self) -> None:
        room = await self._create_ready_room(max_players=4, allow_spectators=False)
        extra_id = self._create_user("battle-unready@example.com", 1_000_000)
        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        service.join_room(room["room_code"], user_id=extra_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)

        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        self.assertEqual(len(started["results"]), 2)
        self.assertNotIn(extra_id, {int(member["user_id"]) for member in started["members"]})
        with auth_db() as db:
            status = db.execute(
                "SELECT status FROM battle_members WHERE room_id = ? AND user_id = ?",
                (room["room_id"], extra_id),
            ).fetchone()["status"]
        self.assertEqual(status, "kicked")

    async def test_host_must_be_one_of_the_ready_players(self) -> None:
        room = await self._create_ready_room(max_players=8)
        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        extra_id = self._create_user("battle-ready-extra@example.com", 1_000_000)
        service.join_room(room["room_code"], user_id=extra_id, role="player")
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        service.set_ready(room["room_code"], user_id=extra_id, ready=True)
        with self.assertRaisesRegex(service.BattleServiceError, "host must be ready") as rejected:
            await service.start_room(
                room["room_code"], user_id=self.host_id, session_id=None
            )
        self.assertEqual(rejected.exception.code, "HOST_NOT_READY")

    async def test_host_leaving_during_generation_refunds_reservation_once(self) -> None:
        gate = asyncio.Event()

        async def delayed_generation(**_kwargs):
            await gate.wait()
            return self._generated_route()

        with (
            patch.object(
                goodness_runtime,
                "resolve_tablebase",
                return_value={"pattern": "L3", "target": 128, "full_pattern": "L3_128"},
            ),
            patch.object(
                goodness_runtime,
                "generate_battle_route",
                side_effect=delayed_generation,
            ),
        ):
            created = await service.create_room(
                user_id=self.host_id,
                session_id=None,
                payload={"full_pattern": "L3_128", "max_players": 2},
            )
            self.assertEqual(get_token_balance(self.host_id)["total"], 900)
            service.leave_room(created["room"]["room_code"], user_id=self.host_id)
            gate.set()
            await asyncio.gather(*list(service._route_tasks.values()), return_exceptions=True)

        self.assertEqual(get_token_balance(self.host_id)["total"], 980)
        with auth_db() as db:
            settlements = db.execute(
                "SELECT settlement_type FROM token_reservation_settlements"
            ).fetchall()
            final_cost = db.execute(
                "SELECT final_cost_units FROM token_ledger WHERE event_type = 'finalize' ORDER BY id DESC LIMIT 1"
            ).fetchone()["final_cost_units"]
        self.assertEqual([row["settlement_type"] for row in settlements], ["finalize"])
        self.assertEqual(final_cost, 20_000)

    async def test_creation_cooldown_rejects_before_a_second_reservation(self) -> None:
        room = await self._create_ready_room()
        service.leave_room(room["room_code"], user_id=self.host_id)
        with auth_db() as db:
            reserves_before = db.execute(
                "SELECT COUNT(*) AS count FROM token_ledger WHERE event_type = 'reserve'"
            ).fetchone()["count"]

        with self.assertRaises(service.BattleServiceError) as raised:
            await service.create_room(
                user_id=self.host_id,
                session_id=None,
                payload={"full_pattern": "L3_128", "max_players": 2},
            )

        self.assertEqual(raised.exception.code, "ROOM_CREATE_COOLDOWN")
        self.assertEqual(raised.exception.status_code, 429)
        with auth_db() as db:
            reserves_after = db.execute(
                "SELECT COUNT(*) AS count FROM token_ledger WHERE event_type = 'reserve'"
            ).fetchone()["count"]
        self.assertEqual(reserves_after, reserves_before)
        self.assertEqual(get_token_balance(self.host_id)["total"], 980)

    async def test_waiting_room_expiry_refunds_eighty_percent(self) -> None:
        room = await self._create_ready_room()
        round_id = str(room["round"]["round_id"])
        with auth_db() as db:
            self.assertEqual(
                db.execute(
                    "SELECT reservation_status FROM battle_rounds WHERE round_id = ?",
                    (round_id,),
                ).fetchone()["reservation_status"],
                "reserved",
            )
            db.execute(
                "UPDATE battle_rooms SET expires_at = ? WHERE room_id = ?",
                ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), room["room_id"]),
            )

        changed = goodness_runtime.mark_timeouts()

        self.assertIn(room["room_id"], changed)
        self.assertEqual(repository.get_room(room["room_id"])["status"], "expired")
        self.assertEqual(get_token_balance(self.host_id)["total"], 980)
        with auth_db() as db:
            status = db.execute(
                "SELECT reservation_status FROM battle_rounds WHERE round_id = ?",
                (round_id,),
            ).fetchone()["reservation_status"]
        self.assertEqual(status, "unstarted_refunded")

    async def test_route_generation_failure_still_refunds_the_full_reservation(self) -> None:
        with (
            patch.object(
                goodness_runtime,
                "resolve_tablebase",
                return_value={"pattern": "L3", "target": 128, "full_pattern": "L3_128"},
            ),
            patch.object(
                goodness_runtime,
                "generate_battle_route",
                new=AsyncMock(side_effect=RuntimeError("generation failed")),
            ),
        ):
            created = await service.create_room(
                user_id=self.host_id,
                session_id=None,
                payload={"full_pattern": "L3_128", "max_players": 2},
            )
            round_id = str(created["room"]["round"]["round_id"])
            await service._route_tasks[round_id]

        self.assertEqual(get_token_balance(self.host_id)["total"], 1000)
        with auth_db() as db:
            settlement = db.execute(
                "SELECT settlement_type FROM token_reservation_settlements"
            ).fetchone()["settlement_type"]
        self.assertEqual(settlement, "cancel")

    async def test_reservation_settlement_is_globally_idempotent(self) -> None:
        reservation = reserve_operation_tokens(
            user_id=self.host_id,
            session_id=None,
            operation_key="battle_route_generation",
            full_pattern="L3_128",
        )
        self.assertIsNotNone(reservation)
        self.assertEqual(get_token_balance(self.host_id)["total"], 900)

        cancel_reservation(reservation, reason="test")
        cancel_reservation(reservation, reason="test-again")
        finalize_reservation(
            reservation,
            actual_operation_key="battle_route_generation",
        )
        self.assertEqual(get_token_balance(self.host_id)["total"], 1000)
        with auth_db() as db:
            settlements = db.execute(
                "SELECT settlement_type FROM token_reservation_settlements WHERE reservation_ledger_id = ?",
                (reservation.ledger_id,),
            ).fetchall()
        self.assertEqual([row["settlement_type"] for row in settlements], ["cancel"])


if __name__ == "__main__":
    unittest.main()
