from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from backend.auth.db import auth_db, init_auth_db
from backend.battle import repository
from backend.battle import service as battle_service
from backend.battle.actors import BattleActor
from backend.battle.core import lifecycle
from backend.battle.core.errors import BattleServiceError
from backend.battle.modes.goodness import runtime as goodness_runtime
from backend.battle.modes.free_goodness import runtime as free_goodness_runtime
from backend.battle.permanent import service as permanent_service
from backend.battle.permanent.definitions import PermanentRoomDefinition, load_definitions
from backend.battle.permanent.service import (
    init_permanent_db,
    register_definition,
)


class PermanentBattleRoomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {"CLOUD_AUTH_DB": os.path.join(self.tempdir.name, "auth.sqlite3")},
        )
        self.env.start()
        init_auth_db()
        repository.init_battle_db()
        init_permanent_db()
        with auth_db() as db:
            now = "2026-09-03T00:00:00+00:00"
            cursor = db.execute(
                """
                INSERT INTO users
                  (email, email_identity, password_hash, display_name, display_name_key,
                   status, created_at, updated_at)
                VALUES ('host@example.com', 'host@example.com', 'x', 'Host', 'host',
                        'active', ?, ?)
                """,
                (now, now),
            )
            self.user_id = int(cursor.lastrowid)
        self.definition = PermanentRoomDefinition(
            template_key="goodness:test",
            mode_key="goodness",
            full_pattern="L3_256",
            initial_board="011112221fff3fff",
            max_players=8,
            step_timeout_seconds=90,
            version=1,
        )
        self.room = repository.create_permanent_room(
            template_key=self.definition.template_key,
            pattern="L3",
            target=256,
            full_pattern="L3_256",
            mode_key="goodness",
            mode_version=1,
            initial_board=self.definition.initial_board,
            max_steps=None,
            step_timeout_seconds=90,
            max_players=8,
            settings={
                "full_pattern": "L3_256",
                "initial_board": self.definition.initial_board,
                "step_timeout_seconds": 90,
            },
            status="waiting",
        )
        register_definition(self.room, self.definition)

    async def asyncTearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    @staticmethod
    def _guest() -> BattleActor:
        return BattleActor(
            kind="guest",
            actor_key="g:permanent-test",
            user_id=None,
            guest_id="permanent-test",
            display_name="Guest",
        )

    def test_first_player_claims_host_and_host_leave_rotates(self) -> None:
        guest = self._guest()
        guest_room = lifecycle.join_room(
            self.room["room_code"], actor=guest, role="player", ip_address="203.0.113.1"
        )
        self.assertTrue(guest_room["viewer"]["is_host"])
        self.assertEqual(guest_room["host_actor_key"], guest.actor_key)
        self.assertIsNotNone(guest_room["host_idle_expires_at"])

        lifecycle.join_room(
            self.room["room_code"], user_id=self.user_id, role="player"
        )
        lifecycle.leave_room(self.room["room_code"], actor=guest)
        rotated = lifecycle.room_snapshot(self.room["room_code"], user_id=self.user_id)
        self.assertEqual(rotated["host_actor_key"], f"u:{self.user_id}")
        self.assertTrue(rotated["viewer"]["is_host"])

    def test_idle_host_is_kicked_and_next_player_takes_over(self) -> None:
        guest = self._guest()
        lifecycle.join_room(
            self.room["room_code"], actor=guest, role="player", ip_address="203.0.113.1"
        )
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        with auth_db() as db:
            db.execute(
                """
                UPDATE battle_permanent_room_state
                SET host_last_seen_at = '2020-01-01T00:00:00+00:00',
                    host_idle_expires_at = '2020-01-01T00:00:00+00:00'
                WHERE room_id = ?
                """,
                (self.room["room_id"],),
            )

        changed = permanent_service._sweep_once()

        self.assertIn(self.room["room_id"], changed)
        rotated = lifecycle.room_snapshot(self.room["room_code"], user_id=self.user_id)
        self.assertEqual(rotated["host_actor_key"], f"u:{self.user_id}")
        with auth_db() as db:
            kicked = db.execute(
                "SELECT status FROM battle_members WHERE room_id = ? AND actor_key = ?",
                (self.room["room_id"], guest.actor_key),
            ).fetchone()
        self.assertEqual(kicked["status"], "left")

    def test_kicked_member_can_rejoin_after_host_changes(self) -> None:
        guest = self._guest()
        lifecycle.join_room(
            self.room["room_code"], actor=guest, role="player", ip_address="203.0.113.1"
        )
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        lifecycle.kick(
            self.room["room_code"],
            actor=guest,
            target_actor_key=f"u:{self.user_id}",
        )

        with self.assertRaises(BattleServiceError):
            lifecycle.join_room(
                self.room["room_code"], user_id=self.user_id, role="player"
            )

        lifecycle.leave_room(self.room["room_code"], actor=guest)
        rejoined = lifecycle.join_room(
            self.room["room_code"], user_id=self.user_id, role="player"
        )
        self.assertTrue(rejoined["viewer"]["is_host"])

    async def test_kicked_member_can_rejoin_after_next_round_starts(self) -> None:
        guest = self._guest()
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        lifecycle.join_room(
            self.room["room_code"], actor=guest, role="player", ip_address="203.0.113.1"
        )
        lifecycle.kick(
            self.room["room_code"],
            actor=f"u:{self.user_id}",
            target_actor_key=guest.actor_key,
        )
        fake_mode = Mock()
        fake_mode.start_room = AsyncMock(return_value={"status": "running"})

        with patch.object(battle_service, "_mode_for_room", return_value=fake_mode):
            await battle_service.start_room(
                self.room["room_code"], user_id=self.user_id
            )

        rejoined = lifecycle.join_room(
            self.room["room_code"],
            actor=guest,
            role="player",
            ip_address="203.0.113.1",
        )
        self.assertEqual(rejoined["viewer"]["actor_key"], guest.actor_key)

    async def test_settings_use_revision_and_empty_room_stays_open(self) -> None:
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        before = lifecycle.room_snapshot(self.room["room_code"], user_id=self.user_id)
        updated = await lifecycle.update_room_settings(
            self.room["room_code"],
            actor=f"u:{self.user_id}",
            payload={
                "expected_revision": before["settings_revision"],
                "step_timeout_seconds": 120,
            },
        )
        self.assertEqual(updated["step_timeout_seconds"], 120)
        with self.assertRaises(Exception):
            await lifecycle.update_room_settings(
                self.room["room_code"],
                actor=f"u:{self.user_id}",
                payload={
                    "expected_revision": before["settings_revision"],
                    "step_timeout_seconds": 125,
                },
            )

        lifecycle.leave_room(self.room["room_code"], user_id=self.user_id)
        empty = repository.get_room(self.room["room_code"])
        self.assertEqual(empty["status"], "waiting")
        self.assertIsNone(empty["host_actor_key"])
        self.assertEqual(empty["step_timeout_seconds"], 90)

    def test_platform_round_has_no_token_reservation(self) -> None:
        round_id = goodness_runtime._insert_round(
            room_id=self.room["room_id"],
            round_number=1,
            seed_hex="00" * 16,
            reservation=None,
        )
        with auth_db() as db:
            row = db.execute(
                "SELECT * FROM battle_rounds WHERE round_id = ?", (round_id,)
            ).fetchone()
        self.assertEqual(row["reservation_status"], "not_required")
        self.assertIsNone(row["reservation_ledger_id"])
        self.assertEqual(row["token_cost_units"], 0)

    def test_solo_round_blocks_restart_and_handoffs_until_cooldown_resumes(self) -> None:
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        actor_key = f"u:{self.user_id}"
        now = permanent_service._iso()
        with auth_db() as db:
            db.execute(
                """
                INSERT INTO battle_rounds
                  (round_id, room_id, round_number, status, created_at, updated_at)
                VALUES ('solo-round', ?, 1, 'completed', ?, ?)
                """,
                (self.room["room_id"], now, now),
            )
            db.execute(
                """
                INSERT INTO battle_player_results
                  (round_id, actor_key, user_id, display_name_snapshot, status,
                   created_at, updated_at)
                VALUES ('solo-round', ?, ?, 'Host', 'completed', ?, ?)
                """,
                (actor_key, self.user_id, now, now),
            )
            permanent_service.rotate_after_round_in_db(
                db, self.room["room_id"], now_text=now
            )

        with self.assertRaises(BattleServiceError) as blocked:
            permanent_service.assert_start_allowed(self.room["room_code"], actor_key)
        self.assertEqual(blocked.exception.code, "PERMANENT_SOLO_COOLDOWN")

        guest = self._guest()
        joined = lifecycle.join_room(
            self.room["room_code"], actor=guest, role="player", ip_address="203.0.113.1"
        )
        self.assertEqual(joined["host_actor_key"], guest.actor_key)
        with auth_db() as db:
            paused = db.execute(
                "SELECT * FROM battle_permanent_room_state WHERE room_id = ?",
                (self.room["room_id"],),
            ).fetchone()
        self.assertEqual(paused["solo_cooldown_actor_key"], actor_key)
        self.assertIsNone(paused["solo_cooldown_expires_at"])
        self.assertGreater(float(paused["solo_cooldown_remaining_seconds"]), 0)

        lifecycle.leave_room(self.room["room_code"], actor=guest)
        resumed = lifecycle.room_snapshot(self.room["room_code"], user_id=self.user_id)
        self.assertEqual(resumed["host_actor_key"], actor_key)
        with auth_db() as db:
            active = db.execute(
                "SELECT * FROM battle_permanent_room_state WHERE room_id = ?",
                (self.room["room_id"],),
            ).fetchone()
        self.assertIsNotNone(active["solo_cooldown_expires_at"])
        with self.assertRaises(BattleServiceError):
            permanent_service.assert_start_allowed(self.room["room_code"], actor_key)

    def test_expired_solo_cooldown_is_cleared_on_start_check(self) -> None:
        lifecycle.join_room(self.room["room_code"], user_id=self.user_id, role="player")
        actor_key = f"u:{self.user_id}"
        with auth_db() as db:
            db.execute(
                """
                UPDATE battle_permanent_room_state
                SET solo_cooldown_actor_key = ?,
                    solo_cooldown_remaining_seconds = 60,
                    solo_cooldown_expires_at = '2020-01-01T00:00:00+00:00'
                WHERE room_id = ?
                """,
                (actor_key, self.room["room_id"]),
            )

        permanent_service.assert_start_allowed(self.room["room_code"], actor_key)

        with auth_db() as db:
            state = db.execute(
                "SELECT * FROM battle_permanent_room_state WHERE room_id = ?",
                (self.room["room_id"],),
            ).fetchone()
        self.assertIsNone(state["solo_cooldown_actor_key"])
        self.assertEqual(float(state["solo_cooldown_remaining_seconds"]), 0)

    def test_empty_free_room_restores_template_and_ready_round(self) -> None:
        definition = PermanentRoomDefinition(
            template_key="free_goodness:test",
            mode_key="free_goodness",
            full_pattern="L3_256",
            initial_board="011112221fff3fff",
            max_players=8,
            step_timeout_seconds=90,
            version=1,
        )
        changed_board = "0000000000000011"
        room = repository.create_permanent_room(
            template_key=definition.template_key,
            pattern="L3",
            target=256,
            full_pattern="L3_256",
            mode_key="free_goodness",
            mode_version=2,
            initial_board=changed_board,
            max_steps=24,
            step_timeout_seconds=120,
            max_players=8,
            settings={
                "full_pattern": "L3_256",
                "initial_board": changed_board,
                "score_step_limit": 24,
                "ranking_min_steps": 12,
                "step_timeout_seconds": 120,
            },
            status="waiting",
        )
        round_id = free_goodness_runtime._insert_round(
            room_id=room["room_id"],
            round_number=1,
            seed_hex="11" * 32,
            initial_board=int(changed_board, 16),
            score_step_limit=24,
            ranking_min_steps=12,
            reservation=None,
        )

        # A restart must preserve template defaults, not capture a host's edits.
        register_definition(repository.get_room(room["room_id"]), definition)
        lifecycle.join_room(room["room_code"], user_id=self.user_id, role="player")
        lifecycle.leave_room(room["room_code"], user_id=self.user_id)

        reset = repository.get_room(room["room_id"])
        self.assertEqual(reset["initial_board"], definition.initial_board)
        self.assertEqual(reset["max_steps"], 128)
        self.assertEqual(reset["step_timeout_seconds"], 90)
        self.assertEqual(reset["settings"]["score_step_limit"], 128)
        self.assertEqual(reset["settings"]["ranking_min_steps"], 128)
        with auth_db() as db:
            latest = db.execute(
                "SELECT * FROM battle_rounds WHERE round_id = ?", (round_id,)
            ).fetchone()
        mode_state = json.loads(latest["mode_state_json"])
        self.assertEqual(mode_state["initial_board"], definition.initial_board)
        self.assertEqual(mode_state["score_step_limit"], 128)
        self.assertEqual(mode_state["ranking_min_steps"], 128)
        self.assertNotEqual(latest["route_seed"], "11" * 32)


class BattleRoomLifecycleMigrationTests(unittest.TestCase):
    def test_production_config_defines_both_modes_for_each_template(self) -> None:
        _policy, definitions = load_definitions()
        self.assertEqual(len(definitions), 14)
        grouped = {(item.mode_key, item.full_pattern) for item in definitions}
        expected = {
            "L3_256",
            "L3_512",
            "442t_256",
            "442t_512",
            "3x3_1024",
            "2x4_512",
            "3x4free9_512",
        }
        self.assertEqual(
            {pattern for mode, pattern in grouped if mode == "goodness"}, expected
        )
        self.assertEqual(
            {pattern for mode, pattern in grouped if mode == "free_goodness"}, expected
        )

    def test_legacy_room_schema_is_migrated_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(
            os.environ,
            {"CLOUD_AUTH_DB": os.path.join(tempdir, "auth.sqlite3")},
        ):
            init_auth_db()
            with auth_db() as db:
                now = "2026-09-03T00:00:00+00:00"
                cursor = db.execute(
                    """
                    INSERT INTO users
                      (email, email_identity, password_hash, display_name,
                       display_name_key, status, created_at, updated_at)
                    VALUES ('legacy@example.com', 'legacy@example.com', 'x',
                            'Legacy', 'legacy', 'active', ?, ?)
                    """,
                    (now, now),
                )
                user_id = int(cursor.lastrowid)
                db.executescript(
                    """
                    CREATE TABLE battle_rooms (
                      room_id TEXT PRIMARY KEY,
                      room_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
                      host_user_id INTEGER NOT NULL,
                      status TEXT NOT NULL DEFAULT 'preparing',
                      visibility TEXT NOT NULL DEFAULT 'public',
                      allow_spectators INTEGER NOT NULL DEFAULT 1,
                      max_players INTEGER NOT NULL DEFAULT 2,
                      pattern TEXT NOT NULL,
                      target INTEGER NOT NULL,
                      full_pattern TEXT NOT NULL,
                      initial_board TEXT,
                      max_steps INTEGER,
                      step_timeout_seconds INTEGER NOT NULL DEFAULT 90,
                      current_round_number INTEGER NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      expires_at TEXT NOT NULL,
                      closed_at TEXT,
                      FOREIGN KEY(host_user_id) REFERENCES users(id)
                    );
                    """
                )
                db.execute(
                    """
                    INSERT INTO battle_rooms
                      (room_id, room_code, host_user_id, status, visibility,
                       allow_spectators, max_players, pattern, target,
                       full_pattern, step_timeout_seconds, created_at,
                       updated_at, expires_at)
                    VALUES ('legacy-room', 'LEGACY', ?, 'waiting', 'public',
                            1, 2, 'L3', 256, 'L3_256', 90, ?, ?, ?)
                    """,
                    (user_id, now, now, "2026-09-03T00:30:00+00:00"),
                )

            repository.init_battle_db()

            migrated = repository.get_room("LEGACY")
            self.assertEqual(migrated["host_actor_key"], f"u:{user_id}")
            self.assertEqual(migrated["lifecycle_kind"], "normal")
            self.assertEqual(migrated["billing_policy"], "user")
            self.assertEqual(migrated["settings_revision"], 1)
            with auth_db() as db:
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
