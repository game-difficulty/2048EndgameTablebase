from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
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

    @staticmethod
    def _generated_route() -> GeneratedBattleRoute:
        initial_board = 0x11
        rates = (RATE_SCALE // 4, RATE_SCALE // 2, RATE_SCALE, RATE_SCALE * 3 // 4)
        blob = encode_route(
            initial_board,
            [RouteStep(encode_changes("left", 0, 2), rates)],
        )
        return GeneratedBattleRoute(
            route_blob=blob,
            step_count=1,
            certainty_step=None,
            termination_reason="max_steps",
            initial_board=initial_board,
            available_layers=8,
        )

    async def _create_ready_room(self, *, step_timeout_seconds: int = 30) -> dict:
        with (
            patch.object(
                goodness_runtime,
                "resolve_tablebase",
                return_value={"pattern": "L3", "target": 128, "full_pattern": "L3_128"},
            ),
            patch.object(
                goodness_runtime,
                "generate_battle_route",
                new=AsyncMock(return_value=self._generated_route()),
            ),
        ):
            created = await service.create_room(
                user_id=self.host_id,
                session_id=None,
                payload={
                    "full_pattern": "L3_128",
                    "max_players": 2,
                    "step_timeout_seconds": step_timeout_seconds,
                    "is_public": True,
                },
            )
            round_id = str(created["room"]["round"]["round_id"])
            await service._route_tasks[round_id]
        return repository.get_room(created["room"]["room_id"])

    async def test_step_timeout_accepts_five_second_increments_at_boundaries(self) -> None:
        room = await self._create_ready_room(step_timeout_seconds=5)
        self.assertEqual(room["step_timeout_seconds"], 5)
        service.leave_room(room["room_code"], user_id=self.host_id)

        room = await self._create_ready_room(step_timeout_seconds=120)
        self.assertEqual(room["step_timeout_seconds"], 120)
        service.leave_room(room["room_code"], user_id=self.host_id)

        self.assertNotIn(6, service.VALID_STEP_TIMEOUTS)

    async def test_full_round_returns_to_same_lobby_and_resets_ready(self) -> None:
        room = await self._create_ready_room()
        self.assertEqual(room["status"], "waiting")
        self.assertEqual(get_token_balance(self.host_id)["total"], 995)

        service.join_room(room["room_code"], user_id=self.player_id, role="player")
        service.set_ready(room["room_code"], user_id=self.host_id, ready=True)
        service.set_ready(room["room_code"], user_id=self.player_id, ready=True)
        started = await service.start_room(
            room["room_code"], user_id=self.host_id, session_id=None
        )
        self.assertEqual(started["status"], "running")
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

        finished = repository.get_room(room["room_code"])
        self.assertEqual(finished["status"], "waiting")
        self.assertEqual(finished["round"]["status"], "completed")
        self.assertTrue(all(not member["ready"] for member in finished["members"]))
        self.assertEqual(finished["room_code"], room["room_code"])

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
            self.assertEqual(get_token_balance(self.host_id)["total"], 995)
            service.leave_room(created["room"]["room_code"], user_id=self.host_id)
            gate.set()
            await asyncio.gather(*list(service._route_tasks.values()), return_exceptions=True)

        self.assertEqual(get_token_balance(self.host_id)["total"], 1000)
        with auth_db() as db:
            settlements = db.execute(
                "SELECT settlement_type FROM token_reservation_settlements"
            ).fetchall()
        self.assertEqual([row["settlement_type"] for row in settlements], ["cancel"])

    async def test_reservation_settlement_is_globally_idempotent(self) -> None:
        reservation = reserve_operation_tokens(
            user_id=self.host_id,
            session_id=None,
            operation_key="battle_route_generation",
            full_pattern="L3_128",
        )
        self.assertIsNotNone(reservation)
        self.assertEqual(get_token_balance(self.host_id)["total"], 995)

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
