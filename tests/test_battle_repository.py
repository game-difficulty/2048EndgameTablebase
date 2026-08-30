from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from backend.auth.db import auth_db, init_auth_db
from backend.battle.repository import (
    BattleConflictError,
    BattlePermissionError,
    close_room,
    create_room,
    get_room,
    init_battle_db,
    join_room,
    kick_member,
    list_public_rooms,
    room_unavailable_reason,
    set_member_ready,
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BattleRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_auth_db = os.environ.get("CLOUD_AUTH_DB")
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["CLOUD_AUTH_DB"] = os.path.join(self._tmpdir.name, "battle.sqlite3")
        init_auth_db()
        init_battle_db()
        self.host_id = self._create_user("host@example.com")
        self.player_id = self._create_user("player@example.com")
        self.third_id = self._create_user("third@example.com")
        self.fourth_id = self._create_user("fourth@example.com")

    def tearDown(self) -> None:
        if self._old_auth_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self._old_auth_db
        self._tmpdir.cleanup()

    def _create_user(self, email: str) -> int:
        now = _iso_now()
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
            return int(cursor.lastrowid)

    def _create_room(self, **overrides):
        values = {
            "host_user_id": self.host_id,
            "pattern": "442t",
            "target": 512,
            "room_code": "ABC234",
            "visibility": "public",
            "max_players": 2,
            "status": "waiting",
        }
        values.update(overrides)
        return create_room(**values)

    def _expire_creation_cooldown(self, user_id: int | None = None) -> None:
        old = (datetime.now(timezone.utc) - timedelta(seconds=31)).isoformat()
        with auth_db() as db:
            db.execute(
                "UPDATE battle_rooms SET created_at = ? WHERE host_user_id = ?",
                (old, int(user_id or self.host_id)),
            )

    def test_init_creates_all_tables_and_partial_unique_index(self) -> None:
        init_battle_db()
        with auth_db() as db:
            tables = {
                row["name"]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'battle_%'"
                ).fetchall()
            }
            index = db.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'index' AND name = 'uq_battle_members_active_user'
                """
            ).fetchone()
        self.assertEqual(
            tables,
            {
                "battle_rooms",
                "battle_members",
                "battle_rounds",
                "battle_routes",
                "battle_player_results",
                "battle_request_ids",
                "battle_chat_messages",
            },
        )
        self.assertIn("WHERE status = 'active'", index["sql"])

    def test_room_unavailable_reason_only_reports_authoritative_loss(self) -> None:
        room = self._create_room()
        self.assertIsNone(
            room_unavailable_reason(room["room_id"], user_id=self.host_id)
        )
        join_room(room["room_id"], user_id=self.player_id)
        kick_member(room["room_id"], host_user_id=self.host_id, target_user_id=self.player_id)
        self.assertEqual(
            room_unavailable_reason(room["room_id"], user_id=self.player_id),
            "KICKED_FROM_ROOM",
        )
        close_room(room["room_id"], host_user_id=self.host_id)
        self.assertEqual(
            room_unavailable_reason(room["room_id"], user_id=self.host_id),
            "ROOM_CLOSED",
        )

    def test_create_room_is_atomic_and_adds_host_as_first_player(self) -> None:
        room = self._create_room()

        self.assertEqual(room["room_code"], "ABC234")
        self.assertEqual(room["player_count"], 1)
        self.assertEqual(room["spectator_count"], 0)
        self.assertEqual(room["members"][0]["user_id"], self.host_id)
        self.assertEqual(room["members"][0]["seat_index"], 0)
        self.assertFalse(room["members"][0]["ready"])
        self.assertEqual(room["chat_roles"], ["host", "player", "spectator"])
        self.assertEqual(get_room("abc234")["room_id"], room["room_id"])

    def test_create_room_persists_selected_chat_roles(self) -> None:
        room = self._create_room(chat_roles=["spectator", "host"])
        self.assertEqual(room["chat_roles"], ["host", "spectator"])
        self.assertEqual(get_room("ABC234")["chat_roles"], ["host", "spectator"])

    def test_create_room_rolls_back_when_host_is_already_active(self) -> None:
        self._create_room()

        with self.assertRaisesRegex(BattleConflictError, "user_already_in_room"):
            self._create_room(room_code="DEF234")

        with auth_db() as db:
            count = db.execute("SELECT COUNT(*) AS n FROM battle_rooms").fetchone()["n"]
        self.assertEqual(count, 1)

    def test_room_creation_is_limited_to_once_per_thirty_seconds(self) -> None:
        first = self._create_room()
        close_room(first["room_id"], host_user_id=self.host_id)

        with self.assertRaisesRegex(BattleConflictError, "room_create_cooldown"):
            self._create_room(room_code="DEF234")

        self._expire_creation_cooldown()
        second = self._create_room(room_code="DEF234")
        self.assertEqual(second["room_code"], "DEF234")

    def test_public_list_includes_spectatable_running_rooms(self) -> None:
        visible = self._create_room()
        self._create_room(
            host_user_id=self.player_id,
            room_code="ABC236",
            visibility="private",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        running = self._create_room(
            host_user_id=self.third_id,
            room_code="ABC237",
            status="running",
        )
        self._create_room(
            host_user_id=self.fourth_id,
            room_code="ABC238",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

        rooms = list_public_rooms()

        self.assertEqual(
            {room["room_id"] for room in rooms},
            {visible["room_id"], running["room_id"]},
        )
        self.assertTrue(all(room["player_count"] == 1 for room in rooms))

    def test_join_assigns_player_seat_then_falls_back_to_spectator(self) -> None:
        room = self._create_room(allow_spectators=True)

        player = join_room(room["room_id"], user_id=self.player_id)
        spectator = join_room(room["room_id"], user_id=self.third_id)

        self.assertEqual((player["role"], player["seat_index"]), ("player", 1))
        self.assertEqual((spectator["role"], spectator["seat_index"]), ("spectator", None))
        self.assertEqual(get_room(room["room_id"])["player_count"], 2)
        self.assertEqual(get_room(room["room_id"])["spectator_count"], 1)

    def test_joining_running_room_is_always_spectator(self) -> None:
        room = self._create_room(
            allow_spectators=True,
            max_players=8,
            status="running",
        )

        member = join_room(
            room["room_id"],
            user_id=self.player_id,
            preferred_role="player",
        )

        self.assertEqual((member["role"], member["seat_index"]), ("spectator", None))

    def test_active_membership_index_blocks_joining_a_second_room(self) -> None:
        first = self._create_room()
        join_room(first["room_id"], user_id=self.player_id)
        close_room(first["room_id"], host_user_id=self.host_id)
        self._expire_creation_cooldown()
        second = self._create_room(room_code="DEF234")

        # Re-open a conflicting active membership directly to exercise the DB invariant.
        now = _iso_now()
        with auth_db() as db:
            db.execute(
                """
                UPDATE battle_members
                SET status = 'active', left_at = NULL, updated_at = ?
                WHERE room_id = ? AND user_id = ?
                """,
                (now, first["room_id"], self.player_id),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute(
                    """
                    INSERT INTO battle_members
                    (room_id, user_id, role, seat_index, ready, status,
                     joined_at, updated_at)
                    VALUES (?, ?, 'player', 1, 0, 'active', ?, ?)
                    """,
                    (second["room_id"], self.player_id, now, now),
                )
        with self.assertRaisesRegex(BattleConflictError, "user_already_in_room"):
            join_room(second["room_id"], user_id=self.player_id)

    def test_ready_kick_and_rejoin_lifecycle(self) -> None:
        room = self._create_room(max_players=3)
        join_room(room["room_id"], user_id=self.player_id)

        ready = set_member_ready(room["room_id"], user_id=self.player_id, ready=True)
        self.assertTrue(ready["ready"])
        with self.assertRaises(BattlePermissionError):
            kick_member(
                room["room_id"],
                host_user_id=self.third_id,
                target_user_id=self.player_id,
            )

        kicked = kick_member(
            room["room_id"],
            host_user_id=self.host_id,
            target_user_id=self.player_id,
        )
        self.assertEqual(kicked["status"], "kicked")
        self.assertFalse(kicked["ready"])

        with self.assertRaisesRegex(BattleConflictError, "kicked_from_room"):
            join_room(room["room_id"], user_id=self.player_id)

    def test_spectator_cannot_ready_and_disabled_spectators_are_rejected(self) -> None:
        room = self._create_room(allow_spectators=False)
        join_room(room["room_id"], user_id=self.player_id)
        with self.assertRaisesRegex(BattleConflictError, "spectators_disabled"):
            join_room(room["room_id"], user_id=self.third_id)

        close_room(room["room_id"], host_user_id=self.host_id)
        self._expire_creation_cooldown()
        room = self._create_room(room_code="DEF234", allow_spectators=True)
        spectator = join_room(
            room["room_id"], user_id=self.third_id, preferred_role="spectator"
        )
        self.assertEqual(spectator["role"], "spectator")
        with self.assertRaisesRegex(BattleConflictError, "spectator_cannot_ready"):
            set_member_ready(room["room_id"], user_id=self.third_id, ready=True)

    def test_close_room_releases_all_active_memberships(self) -> None:
        room = self._create_room(max_players=3)
        join_room(room["room_id"], user_id=self.player_id)

        closed = close_room(room["room_id"], host_user_id=self.host_id)

        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["members"], [])
        self.assertEqual(list_public_rooms(), [])
        replacement = create_room(
            host_user_id=self.player_id,
            pattern="L3",
            target=256,
            room_code="XYZ234",
            status="waiting",
        )
        self.assertEqual(replacement["host_user_id"], self.player_id)


if __name__ == "__main__":
    unittest.main()
